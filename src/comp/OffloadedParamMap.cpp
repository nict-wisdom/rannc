//
// Created by Masahiro Tanaka on 2021/10/26.
//

#include "OffloadedParamMap.h"
#include "Common.h"

namespace rannc {

void OffloadedParamMap::registerParam(
    const std::string& name, const at::Tensor& param) {
  param_map_[name] = param;
}

at::Tensor OffloadedParamMap::getParam(const std::string& name) {
  assert(contains(param_map_, name));
}

} // namespace rannc