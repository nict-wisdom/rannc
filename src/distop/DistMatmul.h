//
// Created by Masahiro Tanaka on 2021/05/11.
//

#ifndef TPTESTS_DISTMATMUL_H
#define TPTESTS_DISTMATMUL_H

#include <comp/TimeCounter.h>
#include <torch/torch.h>
#include <torch/TorchUtil.h>

namespace rannc {

class DistMatmul {
 public:
  at::Tensor run(
      const at::Tensor& x, const at::Tensor& y,
      const std::unordered_set<int>& ranks);

 private:
  at::Tensor out_buf_;
};

class DistLinearFunction
    : public torch::autograd::Function<DistLinearFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor input,
      torch::Tensor weight, c10::optional<torch::Tensor> bias,
      std::vector<int64_t> dist_ranks);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};
} // namespace rannc

#endif // TPTESTS_DISTMATMUL_H
