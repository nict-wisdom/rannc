//
// Created by Masahiro Tanaka on 2022/02/04.
//

#include "DistTaskDispatcher.h"
#include <comm/ObjectComm.h>
#include <comm/SComm.h>

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
          spdlog::info(
              "Bcasting target_ranks ranks={}", join_as_str(target_ranks_vec));
          target_ranks_vec = ocomm_.bcast(target_ranks_vec);
          spdlog::info(
              "Bcasted target_ranks ranks={}", join_as_str(target_ranks_vec));
          std::unordered_set<int> target_ranks = vectorToSet(target_ranks_vec);

          if (!contains(target_ranks, mpi::getRank())) {
            break;
          }

          int comm_tag = tag_map.getRankSetTag(target_ranks);
          MPI_Comm comm = scomm_.getCommunicator(comm_tag, target_ranks);
          ProfilingInput prof_input;
          spdlog::info(
              "Bcasting prof_input ranks={}", join_as_str(target_ranks_vec));
          prof_input = ocomm_.bcast(prof_input, 0, comm);
          spdlog::info(
              "Bcasted prof_input ranks={}", join_as_str(target_ranks_vec));
          //          spdlog::info("DUMMY Profiling {}",
          //          prof_input.ir_graph->getName()); prof_util_->profile(
          //              prof_input.ir_graph, prof_input.iteration,
          //              prof_input.replica_num, prof_input.checkpointing);

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
    int iteration, size_t replica_num, bool checkpointing,
    const std::unordered_set<int>& target_ranks) {
  int task_type_buf = static_cast<int>(DistTaskType::PROFILE);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> target_ranks_vec = setToVector(target_ranks);
  spdlog::info("Bcasting target ranks {}", join_as_str(target_ranks_vec));
  ocomm_.bcast(target_ranks_vec);
  spdlog::info("Bcasted target ranks {}", join_as_str(target_ranks_vec));

  TagMap& tag_map = TagMap::get();
  int comm_tag = tag_map.getRankSetTag(target_ranks);
  MPI_Comm comm = scomm_.getCommunicator(comm_tag, target_ranks);

  ProfilingInput prof_input{ir_graphs, iteration, replica_num, checkpointing};
  spdlog::info("Bcasting prof_inputs ranks={}", join_as_str(target_ranks_vec));
  ocomm_.bcast(prof_input, 0, comm);
  spdlog::info("Bcasted prof_inputs ranks={}", join_as_str(target_ranks_vec));

  //  spdlog::info("Profiling {}", prof_input.ir_graph->getName());
  return sg_prof_->profile(ir_graphs, iteration, replica_num, checkpointing);
}

void DistTaskDispatcher::stop() {
  sg_prof_.reset();
  if (mpi::getRank() == 0) {
    int task_type_buf = static_cast<int>(DistTaskType::STOP);
    MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

}; // namespace rannc
