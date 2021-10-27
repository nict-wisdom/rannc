//
// Created by Masahiro Tanaka on 2021/09/03.
//

#ifndef PYRANNC_CUSTOMOPS_H
#define PYRANNC_CUSTOMOPS_H

#include <torch/torch.h>

namespace rannc {

// Inherit from Function
class OffloadingPostHookFunction
    : public torch::autograd::Function<OffloadingPostHookFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor input) {
    return input;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    return grad_outputs;
  }
};

at::Tensor displayValueHook(const at::Tensor& tensor, const std::string& name);

at::Tensor offloadingPreHook(const at::Tensor& tensor, const std::string& name);
at::Tensor offloadingPostHook(
    const at::Tensor& tensor, const std::string& name);

} // namespace rannc

#endif // PYRANNC_CUSTOMOPS_H
