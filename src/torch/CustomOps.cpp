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

torch::jit::IValue displayValueHook(
    const torch::jit::IValue& val, const std::string& name) {
  if (val.isTensor()) {
    spdlog::info(
        "{} {} {} value={}", name, toString(toIRType(val)),
        val.toTensor().device().str(), tensorToString(val.toTensor()));
  }
  return val;
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

at::Tensor gather(
    const at::Tensor& input, int64_t dim,
    const std::vector<int64_t>& dist_ranks) {
  return GatherFunction::apply(input, dim, dist_ranks);
}

TORCH_LIBRARY(rannc, m) {
  m.def("displayValueHook", displayValueHook);
  m.def("offloadingPreHook", offloadingPreHook);
  m.def("offloadingPostHook", offloadingPostHook);

  m.def(
      TORCH_SELECTIVE_SCHEMA(
          "rannc::linear_dist(Tensor input, Tensor weight, Tensor? bias, int[] dist_ranks) -> Tensor"),
      matmulDist);

  m.def(
      TORCH_SELECTIVE_SCHEMA(
          "rannc::gather(Tensor input, int dim_idx, int[] dist_ranks) -> Tensor"),
      gather);
}
} // namespace rannc
