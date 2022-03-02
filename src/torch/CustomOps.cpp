//
// Created by Masahiro Tanaka on 2021/09/03.
//

#include "CustomOps.h"

#include <comp/Backward.h>
#include <torch/torch.h>

#include "comp/OffloadedParamMap.h"
#include "distop/DistMatmul.h"
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
  return OffloadingHookFunction::apply(tensor, name, true);
}

at::Tensor offloadingPostHook(
    const at::Tensor& tensor, const std::string& name) {
  return OffloadingHookFunction::apply(tensor, name, false);
}

at::Tensor matmulDist(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const std::vector<int64_t>& dist_ranks) {
  return DistLinearFunction::apply(input, weight, bias, dist_ranks);
}

TORCH_LIBRARY(rannc, m) {
  m.def("displayValueHook", displayValueHook);
  m.def("offloadingPreHook", offloadingPreHook);
  m.def("offloadingPostHook", offloadingPostHook);

  m.def(
      TORCH_SELECTIVE_SCHEMA(
          "rannc::linear_dist(Tensor input, Tensor weight, Tensor? bias, int[] dist_ranks) -> Tensor"),
      matmulDist);
}
} // namespace rannc
