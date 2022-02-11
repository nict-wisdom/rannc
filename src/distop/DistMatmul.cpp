//
// Created by Masahiro Tanaka on 2021/04/23.
//

#include "DistMatmul.h"

#include <comm/MPIUtil.h>
#include <comm/SComm.h>

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

at::Tensor DistMatmul::run(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  NCCLWrapper& nccl = NCCLWrapper::get();
  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);
  nccl.createCommunicator(tag, ranks);

  /*
   * We assume both x and y are partitioned along 0-th dim
   */
  int np = ranks.size();
  int my_rank = mpi::getRank();
  assert(contains(ranks, my_rank));

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
      nccl.bcast(tag, {y}, {i});
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

at::Tensor DistMatmul::run_AG(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks, bool part_y_column) {
  torch::NoGradGuard no_grad;

  NCCLWrapper& nccl = NCCLWrapper::get();
  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);
  nccl.createCommunicator(tag, ranks);

  /*
   * We assume both x and y are partitioned along 0-th dim
   */
  int np = ranks.size();
  int my_rank = mpi::getRank();
  assert(contains(ranks, my_rank));

  std::vector<int64_t> x_dim = getTensorDim(x);
  assert(x_dim.size() > 1);
  std::vector<int64_t> y_dim = getTensorDim(y);
  assert(y_dim.size() == 2);

  at::Tensor ret;
  if (part_y_column) {
    std::vector<int64_t> gathered_y_buf_dim = {y_dim.at(1) * np, y_dim.at(0)};
    at::Tensor gathered_y = torch::zeros(
        gathered_y_buf_dim, makeTensorOptions(x.dtype().toScalarType(), false));
    nccl.allgather(tag, {y.t().contiguous()}, {gathered_y});
    ret = torch::matmul(x, gathered_y.t()).contiguous();
  } else {
    std::vector<int64_t> gathered_y_dim = {y_dim.at(0) * np, y_dim.at(1)};
    at::Tensor gathered_y = torch::zeros(
        gathered_y_dim, makeTensorOptions(x.dtype().toScalarType(), false));
    nccl.allgather(tag, {y}, {gathered_y});
    ret = torch::matmul(x, gathered_y);
  }
  nccl.syncWithErrorCheck();
  return ret;
}

at::Tensor DistMatmul::runRR_AG(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  return run_AG(x, y, ranks, false);
}

at::Tensor DistMatmul::runRC_AG(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  return run_AG(x, y, ranks, true);
}

at::Tensor DistMatmul::runCR(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  torch::NoGradGuard no_grad;

  NCCLWrapper& nccl = NCCLWrapper::get();
  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);
  nccl.createCommunicator(tag, ranks);

  /*
   * We assume both x and y are partitioned along 0-th dim
   */
  int np = ranks.size();
  int my_rank = mpi::getRank();
  assert(contains(ranks, my_rank));

  std::vector<int64_t> x_dim = getTensorDim(x);
  assert(x_dim.size() > 1);
  std::vector<int64_t> y_dim = getTensorDim(y);
  assert(y_dim.size() == 2);

  at::Tensor z = torch::matmul(x, y);
  at::Tensor out_buf = torch::zeros(
      {x_dim.at(0) / np, y_dim.at(1)},
      makeTensorOptions(x.dtype().toScalarType(), true));

  int64_t step = x_dim.at(0) / np;
  for (int i = 0; i < np; i++) {
    const auto z_slice = z.slice(0, step * i, step * (i + 1));
    nccl.reduce(tag, {z_slice}, {getLocalRank(ranks, i)});
    if (getLocalRank(ranks, mpi::getRank()) == i) {
      out_buf.copy_(z_slice);
    }
  }
  nccl.syncWithErrorCheck();
  return out_buf;
}

torch::Tensor DistLinearFunction::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor input,
    torch::Tensor weight, c10::optional<torch::Tensor> bias,
    std::vector<int64_t> dist_ranks) {
  ctx->saved_data["input"] = input;
  ctx->saved_data["weight"] = weight;
  ctx->saved_data["bias"] = bias;
  ctx->saved_data["ranks"] = dist_ranks;

  spdlog::info(
      "input.size={} weight.size={} bias.size={} dist_ranks={}",
      join_as_str(getTensorDim(input)), join_as_str(getTensorDim(weight)),
      join_as_str(getTensorDim(*bias)), join_as_str(dist_ranks));

  std::unordered_set<int> ranks;
  for (const int64_t r : dist_ranks) {
    ranks.insert(r);
  }

  DistMatmul matmul;
  at::Tensor out =
      matmul.runRR_AG(input.contiguous(), weight.t().contiguous(), ranks);
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

  std::unordered_set<int> ranks;
  for (int r : ctx->saved_data["ranks"].toIntList()) {
    ranks.insert(r);
  }

  spdlog::info(
      "backward weight.size={} og.size={} ranks={}",
      join_as_str(getTensorDim(ctx->saved_data["weight"].toTensor())),
      join_as_str(getTensorDim(grad_outputs.at(0))), join_as_str(ranks));

  DistMatmul matmul1;
  //  at::Tensor d_input =
  //      torch::matmul(grad_outputs.at(0),
  //      ctx->saved_data["weight"].toTensor());
  at::Tensor d_input = matmul1.runRC_AG(
      grad_outputs.at(0).contiguous(),
      ctx->saved_data["weight"].toTensor().contiguous(), ranks);

  spdlog::info(
      "backward og.size={} input.size={} ",
      join_as_str(getTensorDim(grad_outputs.at(0))),
      join_as_str(getTensorDim(ctx->saved_data["input"].toTensor())));
  //  at::Tensor d_weight = torch::matmul(
  //      grad_outputs.at(0).t(), ctx->saved_data["input"].toTensor());
  DistMatmul matmul2;
  at::Tensor d_weight = matmul2.run(
      grad_outputs.at(0).t().contiguous(),
      ctx->saved_data["input"].toTensor().contiguous(), ranks);

  at::Tensor d_bias;

  if (ctx->saved_data["bias"].isTensor()) {
    d_bias = grad_outputs.at(0);
  }
  return {d_input, d_weight, d_bias, at::Tensor()};
}

} // namespace rannc
