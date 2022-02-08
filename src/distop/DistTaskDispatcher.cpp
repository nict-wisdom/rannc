//
// Created by Masahiro Tanaka on 2022/02/04.
//

#include "DistTaskDispatcher.h"
#include <comm/ObjectComm.h>
#include <comm/SComm.h>
#include <pybind11/pybind11.h>

namespace rannc {

DistTaskDispatcher::DistTaskDispatcher()
    : nccl_(NCCLWrapper::get()),
      dpl_(DistributedParamLocator::get()),
      scomm_(SComm::get()),
      ocomm_(ObjectComm::get()) {}

void DistTaskDispatcher::start(const std::shared_ptr<GraphProfiler>& sg_prof) {
  sg_prof_ = sg_prof;

  TagMap& tag_map = TagMap::get();
  comm_tag_ = tag_map.getRankSetTag(mpi::getAllRanks());
  nccl_.createCommunicator(comm_tag_, mpi::getAllRanks());

  if (mpi::getRank() != 0) {
    int task_type_buf;

    bool running = true;
    while (running) {
      MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

      const auto task_type = static_cast<DistTaskType>(task_type_buf);
      switch (task_type) {
        case DistTaskType::GET_PARAM: {
          logger->trace("Received GET_PARAM");
          long param_id;
          MPI_Bcast(&param_id, 1, MPI_LONG, 0, MPI_COMM_WORLD);
          logger->trace("Received pid={}", param_id);
          dpl_.load(dpl_.pidToLocal(param_id));
          break;
        }
        case DistTaskType::PROFILE: {
          logger->trace("Received PROFILE");
          std::vector<int> target_ranks_vec;
          target_ranks_vec = ocomm_.bcast(target_ranks_vec);
          std::unordered_set<int> target_ranks = vectorToSet(target_ranks_vec);

          if (!contains(target_ranks, mpi::getRank())) {
            break;
          }

          int comm_tag = tag_map.getRankSetTag(target_ranks);
          MPI_Comm comm = scomm_.getCommunicator(comm_tag, target_ranks);
          ProfilingInput prof_input;
          prof_input = ocomm_.bcast(prof_input, 0, comm);

          int tag = tag_map.getRankSetTag(target_ranks);
          int src_tag = tag_map.getRankSetTag({0});
          RouteDP bcast_route(
              IValueLocation("PROF_INPUTS"), {0}, target_ranks_vec, tag,
              src_tag, RouteTypeDP::BROADCAST);
          createRouteCommunicator({bcast_route});

          IValueMap input_vals, constants;
          input_vals = scomm_.bcastIValueMap(input_vals, bcast_route);
          constants = scomm_.bcastIValueMap(constants, bcast_route);
          nccl_.syncWithErrorCheck();

          sg_prof_->updateConstants(constants);

          {
            pybind11::gil_scoped_release no_gil;
            sg_prof_->profile(
                prof_input.ir_graphs, input_vals, prof_input.iteration,
                prof_input.replica_num, prof_input.checkpointing);
          }

          for (const auto& it : constants) {
            sg_prof_->removeConstant(it.first);
          }

        } break;
        case DistTaskType::STOP:
          logger->trace("Received STOP");
          running = false;
          break;
      }
    }
  }
}

at::Tensor DistTaskDispatcher::getParam(long param_id) {
  int task_type_buf = static_cast<int>(DistTaskType::GET_PARAM);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&param_id, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  return dpl_.load(dpl_.pidToLocal(param_id));
}

ProfilingResult DistTaskDispatcher::profile(
    const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& ir_graphs,
    const IValueMap& input_vals, const IValueMap& constants, int iteration,
    size_t replica_num, bool checkpointing,
    const std::unordered_set<int>& target_ranks) {
  int task_type_buf = static_cast<int>(DistTaskType::PROFILE);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> target_ranks_vec = setToVector(target_ranks);
  ocomm_.bcast(target_ranks_vec);

  std::unordered_map<IValueLocation, IRType, IValueLocationHash> types;
  for (const auto& it : input_vals) {
    types[it.first] = toIRType(it.second);
  }

  TagMap& tag_map = TagMap::get();
  int comm_tag = tag_map.getRankSetTag(target_ranks);
  MPI_Comm comm = scomm_.getCommunicator(comm_tag, target_ranks);

  ProfilingInput prof_input{
      ir_graphs, types, iteration, replica_num, checkpointing};
  ocomm_.bcast(prof_input, 0, comm);

  int tag = tag_map.getRankSetTag(target_ranks);
  int src_tag = tag_map.getRankSetTag({0});
  RouteDP bcast_route(
      IValueLocation("PROF_INPUTS"), {0}, target_ranks_vec, tag, src_tag,
      RouteTypeDP::BROADCAST);
  createRouteCommunicator({bcast_route});

  scomm_.bcastIValueMap(input_vals, bcast_route);
  scomm_.bcastIValueMap(constants, bcast_route);
  nccl_.syncWithErrorCheck();

  sg_prof_->updateConstants(constants);
  ProfilingResult ret =
      sg_prof_->profile(ir_graphs, iteration, replica_num, checkpointing);
  for (const auto& it : constants) {
    sg_prof_->removeConstant(it.first);
  }

  return ret;
}

void DistTaskDispatcher::stop() {
  sg_prof_.reset();
  if (mpi::getRank() == 0) {
    int task_type_buf = static_cast<int>(DistTaskType::STOP);
    MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

}; // namespace rannc
