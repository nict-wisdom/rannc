//
// Created by Masahiro Tanaka on 2021/04/23.
//

#include "DistMatmul.h"

#include <comm/MPIUtil.h>
#include <comm/SComm.h>
#include <comp/EventRecorder.h>

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

const std::shared_ptr<spdlog::logger> DistLinearFunction::logger =
    getLogger("DistLinearFunction");

at::Tensor DistMatmul::run(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  TraceEvent evt(getFuncKey("DistMatmul", "run", "no_id", 0, false));

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
      TraceEvent evt_bcast(
          getFuncKey("DistMatmul", "bcast", "no_id", i, false));

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
      TraceEvent evt_bcast(
          getFuncKey("DistMatmul", "matmul", "no_id", i, false));

      int64_t x_split_dim = x_dim.size() - 1;

      int64_t step = x_dim.at(x_split_dim) / np;

      const auto x_slice =
          x.slice(x_split_dim, step * target_idx, step * (target_idx + 1));
      at::Tensor z = torch::matmul(x_slice, y);
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
    TraceEvent evt(getFuncKey("DistMatmul", "run_AG", "part_y=true", 0, false));

    std::vector<int64_t> gathered_y_buf_dim = {y_dim.at(1) * np, y_dim.at(0)};
    at::Tensor gathered_y = torch::zeros(
        gathered_y_buf_dim, makeTensorOptions(x.dtype().toScalarType(), false));
    {
      TraceEvent evt_allgather(
          getFuncKey("DistMatmul", "run_AG", "allgather", 0, false));
      nccl.allgather(tag, {y.t().contiguous()}, {gathered_y});
    }
    {
      TraceEvent evt_matmul(
          getFuncKey("DistMatmul", "run_AG", "matmul", 0, false));
      ret = torch::matmul(x, gathered_y.t()).contiguous();
    }
  } else {
    TraceEvent evt(
        getFuncKey("DistMatmul", "run_AG", "part_y=false", 0, false));

    std::vector<int64_t> gathered_y_dim = {y_dim.at(0) * np, y_dim.at(1)};
    at::Tensor gathered_y = torch::zeros(
        gathered_y_dim, makeTensorOptions(x.dtype().toScalarType(), false));
    {
      TraceEvent evt_allgather(
          getFuncKey("DistMatmul", "run_AG", "allgather", 0, false));
      nccl.allgather(tag, {y}, {gathered_y});
    }
    { ret = torch::matmul(x, gathered_y); }
    nccl.syncWithErrorCheck();
  }
  return ret;
}

at::Tensor DistMatmul::runRRR_AG(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  return run_AG(x, y, ranks, false);
}

at::Tensor DistMatmul::runRCR_AG(
    const at::Tensor& x, const at::Tensor& y,
    const std::unordered_set<int>& ranks) {
  return run_AG(x, y, ranks, true);
}

at::Tensor DistMatmul::runCRC(
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

  TraceEvent evt(getFuncKey("DistMatmul", "runCRC", "part_y=true", 0, false));

  at::Tensor z;
  {
    TraceEvent evt(getFuncKey("DistMatmul", "runCRC", "matmul", 0, false));
    z = torch::matmul(x, y);
  }

  at::Tensor out_buf = torch::zeros(
      {x_dim.at(0), y_dim.at(1) / np},
      makeTensorOptions(x.dtype().toScalarType(), true));

  int64_t step = y_dim.at(1) / np;
  for (int i = 0; i < np; i++) {
    const auto z_slice = z.slice(1, step * i, step * (i + 1)).contiguous();
    {
      TraceEvent evt(getFuncKey("DistMatmul", "runCRC", "reduce", i, false));
      nccl.reduce(tag, {z_slice}, {i});
    }
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

  logger->trace("DistLinearFunction forward starting");

  if (bias) {
    logger->trace(
        "input.size={} weight.size={} bias.size={} dist_ranks={}",
        join_as_str(getTensorDim(input)), join_as_str(getTensorDim(weight)),
        join_as_str(getTensorDim(*bias)), join_as_str(dist_ranks));
  } else {
    logger->trace(
        "input.size={} weight.size={} bias=none dist_ranks={}",
        join_as_str(getTensorDim(input)), join_as_str(getTensorDim(weight)),
        join_as_str(dist_ranks));
  }

  std::unordered_set<int> ranks;
  for (const int64_t r : dist_ranks) {
    ranks.insert(r);
  }

  DistMatmul matmul;
  at::Tensor out =
      matmul.runRRR_AG(input.contiguous(), weight.t().contiguous(), ranks);
  if (bias) {
    out += *bias;
  }

  logger->trace("DistLinearFunction forward finished");

  return out;
}

torch::autograd::tensor_list DistLinearFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  logger->trace(
      "DistLinearFunction backward starting. #grad_outputs={}",
      grad_outputs.size());

  assert(grad_outputs.size() == 1);

  std::unordered_set<int> ranks;
  for (int r : ctx->saved_data["ranks"].toIntList()) {
    ranks.insert(r);
  }

  auto og = grad_outputs.at(0);

  logger->trace(
      "backward weight.size={} og.size={} ranks={}",
      join_as_str(getTensorDim(ctx->saved_data["weight"].toTensor())),
      join_as_str(getTensorDim(og)), join_as_str(ranks));

  DistMatmul matmul1;
  at::Tensor d_input = matmul1.runRCR_AG(
      og.contiguous(), ctx->saved_data["weight"].toTensor().contiguous(),
      ranks);

  logger->trace(
      "backward og.size={} input.size={} ", join_as_str(getTensorDim(og)),
      join_as_str(getTensorDim(ctx->saved_data["input"].toTensor())));

  DistMatmul matmul2;

  const auto input = ctx->saved_data["input"].toTensor();
  at::Tensor d_weight = matmul2.runCRC(
      og.reshape({-1, og.size(-1)}).t().contiguous(),
      input.reshape({-1, input.size(-1)}).contiguous(), ranks);

  at::Tensor d_bias;

  if (ctx->saved_data["bias"].isTensor()) {
    d_bias = grad_outputs.at(0);
  }

  logger->trace(
      "DistLinearFunction backward finished. #grad_outputs={}",
      grad_outputs.size());

  return {d_input, d_weight, d_bias, at::Tensor()};
}

torch::Tensor GatherFunction::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor input, int64_t dim_idx,
    std::vector<int64_t> dist_ranks) {
  NCCLWrapper& nccl = NCCLWrapper::get();
  TagMap& tag_map = TagMap::get();

  TraceEvent evt(getFuncKey("GatherFunction", "forward", "no_id", 0, false));

  ctx->saved_data["dim_idx"] = dim_idx;
  ctx->saved_data["ranks"] = dist_ranks;

  std::unordered_set<int> rank_set;
  for (int r : dist_ranks) {
    rank_set.insert(r);
  }
  int tag = tag_map.getRankSetTag(rank_set);
  nccl.createCommunicator(tag, rank_set);

  std::vector<int64_t> gathered_dim;
  gathered_dim.push_back(dist_ranks.size());
  const auto local_dim = getTensorDim(input);
  for (size_t d : getTensorDim(input)) {
    gathered_dim.push_back(d);
  }
  at::Tensor gathered_y = torch::zeros(
      gathered_dim, makeTensorOptions(input.dtype().toScalarType(), true));

  nccl.allgather(tag, {input.contiguous()}, {gathered_y});

  std::vector<at::Tensor> gathered_tensors;
  for (int64_t idx = 0; idx < dist_ranks.size(); idx++) {
    at::indexing::TensorIndex t_index{{idx}};
    at::Tensor part = gathered_y.index(t_index);
    gathered_tensors.push_back(part);
  }
  at::Tensor concat = torch::cat(gathered_tensors, dim_idx);
  concat = concat.set_requires_grad(true);
  return concat;
}

torch::autograd::tensor_list GatherFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  TraceEvent evt(getFuncKey("GatherFunction", "backward", "no_id", 0, false));

  std::unordered_set<int> rank_set;
  for (int r : ctx->saved_data["ranks"].toIntList()) {
    rank_set.insert(r);
  }
  NCCLWrapper& nccl = NCCLWrapper::get();
  TagMap& tag_map = TagMap::get();

  int tag = tag_map.getRankSetTag(rank_set);
  nccl.createCommunicator(tag, rank_set);

  const auto& out_grad = grad_outputs.at(0);

  std::vector<int64_t> split_dim = getTensorDim(out_grad);
  int64_t dim_idx = ctx->saved_data["dim_idx"].toInt();
  split_dim[dim_idx] /= rank_set.size();

  at::Tensor scattered_y = torch::zeros(
      split_dim, makeTensorOptions(out_grad.dtype().toScalarType(), true));

  auto split_out_grad = out_grad.split(split_dim.at(dim_idx), dim_idx);
  for (size_t i = 0; i < split_out_grad.size(); i++) {
    split_out_grad[i] = split_out_grad[i].contiguous();
  }
  for (int r : rank_set) {
    nccl.reduce(tag, {split_out_grad[r]}, {r});
  }
  return {split_out_grad[mpi::getRank()], at::Tensor(), at::Tensor()};
}

} // namespace rannc
