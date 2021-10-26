//
// Created by Masahiro Tanaka on 2021/09/03.
//

#include "CustomOps.h"

#include <torch/torch.h>

#include "comp/OffloadedParamMap.h"
#include "graph/ir.h"
#include "torch/TorchUtil.h"

namespace rannc {

at::Tensor displayValueHook(const at::Tensor& tensor, const std::string& name) {
  spdlog::info(
      "{} {} {} value={}", name, toString(toIRType(tensor)),
      tensor.device().str(), tensorToString(tensor));
  return tensor;
}

at::Tensor offloadingPreHook(
    const at::Tensor& tensor, const std::string& name) {
  OffloadedParamMap& param_map = OffloadedParamMap::get();

  at::Tensor param = param_map.getParam(name);
  toCUDAInPlace(param);
  return tensor;
}

at::Tensor offloadingPostHook(
    const at::Tensor& tensor, const std::string& name) {
  OffloadedParamMap& param_map = OffloadedParamMap::get();

  at::Tensor param = param_map.getParam(name);
  toCPUInPlace(param);
  return tensor;
}

TORCH_LIBRARY(rannc, m) {
  m.def("displayValueHook", displayValueHook);
  m.def("offloadingPreHook", offloadingPreHook);
  m.def("offloadingPostHook", offloadingPostHook);
}
} // namespace rannc
