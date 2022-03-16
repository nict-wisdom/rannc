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
#include <comp/GraphValueCache.h>
#include <graph/ProfilerUtil.h>

namespace rannc {

enum class DistTaskType { STOP, GET_PARAM, PROFILE, CLEAR_CACHE };

class DistTaskDispatcher {
 public:
  void start(const std::shared_ptr<GraphProfiler>& sg_prof, size_t cache_size);
  void stop();

  at::Tensor getParam(long param_id);
  ProfilingResult profile(const ProfilingInput& input, IValueMap input_vals);
  void clearCache();

  static DistTaskDispatcher& get() {
    static DistTaskDispatcher instance;
    return instance;
  }

 private:
  DistTaskDispatcher();
  ProfilingResult runProfiling(
      const ProfilingInput& input, IValueMap input_vals);
  at::Tensor getParamWithCache(long param_id);

  NCCLWrapper& nccl_;
  DistributedParamLocator& dpl_;
  ObjectComm& ocomm_;
  SComm& scomm_;

  std::shared_ptr<GraphProfiler> sg_prof_;
  int comm_tag_;
  ParamCache param_cache_;

  const std::shared_ptr<spdlog::logger> logger =
      getLogger("DistTaskDispatcher");
};

} // namespace rannc

#endif // PYRANNC_DISTTASKDISPATCHER_H
