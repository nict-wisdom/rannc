//
// Created by Masahiro Tanaka on 2021/09/03.
//

#include "CustomOps.h"

#include <torch/torch.h>

#include "graph/ir.h"
#include "torch/TorchUtil.h"

namespace rannc {

at::Tensor displayValueHook(const at::Tensor& tensor, const std::string& name) {
  spdlog::info(
      "{} {} {} value={}", name, toString(toIRType(tensor)),
      tensor.device().str(), tensorToString(tensor));
  return tensor;
}

TORCH_LIBRARY(rannc, m) {
  m.def("displayValueHook", displayValueHook);
}
} // namespace rannc
