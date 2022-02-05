//
// Created by Masahiro Tanaka on 2022/02/04.
//

#ifndef PYRANNC_DISTTASKDISPATCHER_H
#define PYRANNC_DISTTASKDISPATCHER_H

#include <comm/NCCLWrapper.h>
#include <comp/DistributedParamLocator.h>

namespace rannc {

enum class DistTaskType { STOP, GET_PARAM, PROFILE };

class DistTaskDispatcher {
 public:
  void start();
  void stop();
  at::Tensor getParam(long param_id);

  static DistTaskDispatcher& get() {
    static DistTaskDispatcher instance;
    return instance;
  }

 private:
  DistTaskDispatcher();
  NCCLWrapper& nccl_;
  DistributedParamLocator& dpl_;
  int comm_tag_;

  const std::shared_ptr<spdlog::logger> logger =
      getLogger("DistTaskDispatcher");
};

} // namespace rannc

#endif // PYRANNC_DISTTASKDISPATCHER_H
