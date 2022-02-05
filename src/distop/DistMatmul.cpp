//
// Created by Masahiro Tanaka on 2021/04/23.
//

#include "DistMatmul.h"

#include <torch/torch.h>

#include <comm/MPIUtil.h>
#include <iostream>

#include "comm/NCCLWrapper.h"
#include "comp/TimeCounter.h"
#include "cuda/CudaUtil.h"
#include "torch/TorchUtil.h"

namespace rannc {

at::TensorOptions makeTensorOptions(at::ScalarType dtype, bool requires_grad) {
  at::TensorOptions options;
  options = options.dtype(dtype)
                .device(c10::Device(c10::DeviceType::CUDA))
                .requires_grad(requires_grad);
  return options;
}

at::Tensor DistMatmul::run(const at::Tensor& x, const at::Tensor& y) {
  NCCLWrapper& nccl = NCCLWrapper::get();

  /*
   * We assume both x and y are partitioned along 0-th dim
   */
  int np = mpi::getSize();
  int my_rank = mpi::getRank();

  nccl.createCommunicator(100, mpi::getAllRanks());

  std::vector<int64_t> x_dim = getTensorDim(x);
  assert(x_dim.size() > 1);
  std::vector<int64_t> y_dim = getTensorDim(y);
  assert(y_dim.size() == 2);

  if (out_buf_.defined()) {
    out_buf_ = out_buf_.detach();
    out_buf_.zero_();
  } else {
    std::vector<int64_t> out_dims = x_dim;
    out_dims[x_dim.size() - 1] = y_dim.at(1);
    out_buf_ = torch::zeros(
                   out_dims, makeTensorOptions(x.dtype().toScalarType(), true))
                   .clone();
  }

  for (int i = 0; i <= np; i++) {
    if (i < np) {
      nccl.bcast(100, {y}, {i});
      nccl.syncWithErrorCheck();
    }

    int target_idx;
    if (i == 0) {
      target_idx = my_rank;
    } else {
      target_idx = i - 1;
    }

    if (i != my_rank) {
      int64_t x_split_dim = x_dim.size() - 1;

      int64_t step = x_dim.at(x_split_dim) / np;

      const auto x_slice =
          x.slice(x_split_dim, step * target_idx, step * (target_idx + 1));

      //      spdlog::info("x.shape={} x_slice.shape={} y.shape={} step={}
      //      target_idx={}",
      //                   join_as_str(getTensorDim(x)),
      //                   join_as_str(getTensorDim(x_slice)),
      //                   join_as_str(getTensorDim(y)),
      //                   step, target_idx);

      at::Tensor z = torch::matmul(x_slice, y);
      //      spdlog::info("matmul done: x.shape={} out_buf_.shape={}",
      //                   join_as_str(getTensorDim(z)),
      //                   join_as_str(getTensorDim(out_buf_)));
      out_buf_ += z;
      nccl.syncWithErrorCheck();
    }
  }

  return out_buf_;
}

torch::Tensor DistLinearFunction::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor input,
    torch::Tensor weight, c10::optional<torch::Tensor> bias) {
  ctx->saved_data["input"] = input;
  ctx->saved_data["weight"] = weight;
  ctx->saved_data["bias"] = bias;

  spdlog::info(
      "input.size={} weight.size={} bias.size={}",
      join_as_str(getTensorDim(input)), join_as_str(getTensorDim(weight)),
      join_as_str(getTensorDim(*bias)));

  at::Tensor out = torch::matmul(input, weight.t());
  if (bias) {
    out += *bias;
  }

  spdlog::info("DistLinearFunction forward");

  return out;
}

torch::autograd::tensor_list DistLinearFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  spdlog::info(
      "DistLinearFunction backward. #grad_outputs={}", grad_outputs.size());

  assert(grad_outputs.size() == 1);

  spdlog::info(
      "weight.size={} og.size={}",
      join_as_str(getTensorDim(ctx->saved_data["weight"].toTensor())),
      join_as_str(getTensorDim(grad_outputs.at(0))));
  at::Tensor d_input =
      torch::matmul(grad_outputs.at(0), ctx->saved_data["weight"].toTensor());

  spdlog::info(
      "og.size={} input.size={} ",
      join_as_str(getTensorDim(grad_outputs.at(0))),
      join_as_str(getTensorDim(ctx->saved_data["input"].toTensor())));
  at::Tensor d_weight = torch::matmul(
      grad_outputs.at(0).t(), ctx->saved_data["input"].toTensor());

  at::Tensor d_bias;

  if (ctx->saved_data["bias"].isTensor()) {
    d_bias = grad_outputs.at(0);
  }
  return {d_input, d_weight, d_bias};
}

} // namespace rannc