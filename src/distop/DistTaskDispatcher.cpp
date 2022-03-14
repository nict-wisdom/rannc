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
      ocomm_(ObjectComm::get()),
      param_cache_(0) {}

void DistTaskDispatcher::start(
    const std::shared_ptr<GraphProfiler>& sg_prof, size_t cache_size) {
  sg_prof_ = sg_prof;
  param_cache_ = ParamCache(cache_size);

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
          long param_id; // dummy
          getParamWithCache(param_id);
          break;
        }
        case DistTaskType::PROFILE: {
          logger->trace("Received PROFILE");
          pybind11::gil_scoped_release no_gil;
          runProfiling(ProfilingInput{}, IValueMap{});
        } break;
        case DistTaskType::STOP:
          logger->trace("Received STOP");
          running = false;
          break;
      }
    }
  }
}

at::Tensor DistTaskDispatcher::getParamWithCache(long param_id) {
  MPI_Bcast(&param_id, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  long local_pid = dpl_.pidToLocal(param_id);

  if (!param_cache_.exists(local_pid)) {
    param_cache_.put(local_pid, dpl_.load(local_pid));
  }
  return param_cache_.get(local_pid);
}

at::Tensor DistTaskDispatcher::getParam(long param_id) {
  int task_type_buf = static_cast<int>(DistTaskType::GET_PARAM);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return getParamWithCache(param_id);
}

ProfilingResult DistTaskDispatcher::runProfiling(
    const ProfilingInput& input, IValueMap input_vals) {
  ProfilingInput prof_input = input;
  prof_input = ocomm_.bcast(prof_input);

  assert(prof_input.ir_graphs.size() == 1);
  const std::shared_ptr<IRGraph>& graph = prof_input.ir_graphs.begin()->second;
  assert(contains(prof_input.part_info, graph->getName()));
  const auto& part_info = prof_input.part_info.at(graph->getName());
  std::unordered_set<int> ranks = vectorToSet(part_info.ranks);
  if (!contains(ranks, mpi::getRank())) {
    return ProfilingResult{};
  }

  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);

  int src_tag = tag_map.getRankSetTag({0});
  RouteDP bcast_route(
      IValueLocation("PROF_INPUTS"), {0}, part_info.ranks, tag, src_tag,
      RouteTypeDP::BROADCAST);
  createRouteCommunicator({bcast_route});
  nccl_.syncWithErrorCheck();

  input_vals = scomm_.bcastIValueMap(input_vals, bcast_route);

  IValueMap constants;
  for (const auto& it : part_info.rank_values) {
    constants[it.first] = it.second;
  }
  for (const auto& it : part_info.dim_values) {
    constants[it.first] = it.second;
  }
  scomm_.bcastIValueMap(constants, bcast_route);

  sg_prof_->updateConstants(constants);
  ProfilingResult ret = sg_prof_->profile(prof_input, input_vals);

  for (const auto& it : constants) {
    sg_prof_->removeConstant(it.first);
  }
  return ret;
}

ProfilingResult DistTaskDispatcher::profile(
    const ProfilingInput& input, IValueMap input_vals) {
  int task_type_buf = static_cast<int>(DistTaskType::PROFILE);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return runProfiling(input, input_vals);
}

void DistTaskDispatcher::stop() {
  sg_prof_.reset();
  if (mpi::getRank() == 0) {
    int task_type_buf = static_cast<int>(DistTaskType::STOP);
    MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

} // namespace rannc
