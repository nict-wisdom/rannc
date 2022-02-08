//
// Created by Masahiro Tanaka on 2022/02/04.
//

#ifndef PYRANNC_DISTTASKDISPATCHER_H
#define PYRANNC_DISTTASKDISPATCHER_H

#include <comm/NCCLWrapper.h>
#include <comm/ObjectComm.h>
#include <comm/SComm.h>
#include <comp/DistributedParamLocator.h>
#include <comp/GraphProfiler.h>
#include <graph/ProfilerUtil.h>

namespace rannc {

enum class DistTaskType { STOP, GET_PARAM, PROFILE };

class DistTaskDispatcher {
 public:
  void start(const std::shared_ptr<GraphProfiler>& sg_prof);
  void stop();

  at::Tensor getParam(long param_id);
  ProfilingResult profile(
      const std::unordered_map<std::string, std::shared_ptr<IRGraph>>&
          ir_graphs,
      const IValueMap& input_vals, const TensorPartioningGraphInfo& part_info,
      int iteration, size_t replica_num, bool checkpointing,
      const std::unordered_set<int>& target_ranks);

  static DistTaskDispatcher& get() {
    static DistTaskDispatcher instance;
    return instance;
  }

 private:
  DistTaskDispatcher();
  ProfilingResult runProfiling(
      std::unordered_map<std::string, std::shared_ptr<IRGraph>> ir_graphs,
      IValueMap input_vals, TensorPartioningGraphInfo part_info, int iteration,
      size_t replica_num, bool checkpointing,
      std::unordered_set<int> target_ranks);

  NCCLWrapper& nccl_;
  DistributedParamLocator& dpl_;
  ObjectComm& ocomm_;
  SComm& scomm_;

  std::shared_ptr<GraphProfiler> sg_prof_;
  int comm_tag_;

  const std::shared_ptr<spdlog::logger> logger =
      getLogger("DistTaskDispatcher");
};

} // namespace rannc

#endif // PYRANNC_DISTTASKDISPATCHER_H
