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
  at::Tensor runRRR_AG(
      const at::Tensor& x, const at::Tensor& y,
      const std::unordered_set<int>& ranks);
  at::Tensor runRCR_AG(
      const at::Tensor& x, const at::Tensor& y,
      const std::unordered_set<int>& ranks);
  at::Tensor runCRC(
      const at::Tensor& x, const at::Tensor& y,
      const std::unordered_set<int>& ranks);

 private:
  at::Tensor run_AG(
      const at::Tensor& x, const at::Tensor& y,
      const std::unordered_set<int>& ranks, bool part_y_column);

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

 private:
  static const std::shared_ptr<spdlog::logger> logger;
};

class GatherFunction : public torch::autograd::Function<GatherFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor input,
      int64_t dim_idx, std::vector<int64_t> dist_ranks);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);

 private:
  static const std::shared_ptr<spdlog::logger> logger;
};
} // namespace rannc

#endif // TPTESTS_DISTMATMUL_H
