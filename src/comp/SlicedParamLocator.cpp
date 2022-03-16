//
// Created by Masahiro Tanaka on 2022/02/28.
//

#include "SlicedParamLocator.h"
#include <comm/NCCLWrapper.h>
#include "comm/SComm.h"
#include "torch/TorchUtil.h"

namespace rannc {

at::Tensor sliceParam(
    const at::Tensor& param, const std::unordered_set<int>& ranks, int my_rank,
    size_t dim_idx) {
  assert(contains(ranks, my_rank));

  if (param.size(dim_idx) % ranks.size() != 0) {
    std::stringstream ss;
    ss << "Dimension " << dim_idx
       << " is not divisible. shape=" << join_as_str(getTensorDim(param));
    throw std::runtime_error(ss.str());
  }

  size_t segment_size = param.size(dim_idx) / ranks.size();
  int local_rank = getLocalRank(ranks, my_rank);

  torch::NoGradGuard no_grad;
  return param.slice(
      dim_idx, segment_size * local_rank, segment_size * (local_rank + 1));
}

at::Tensor SlicedParamLocator::registerParam(
    long pid, at::Tensor orig_param, size_t dim,
    const std::unordered_set<int>& ranks) {
  at::NoGradGuard no_grad;

  const IRType orig_type = toIRType(orig_param);
  const auto sliced = sliceParam(orig_param, ranks, mpi::getRank(), dim);
  orig_param.set_data(sliced);

  sliced_param_[pid] = orig_param;
  slice_info_[pid] = {ranks, orig_type, dim};

  return orig_param;
}

bool SlicedParamLocator::registered(long pid) const {
  return contains(sliced_param_, pid);
}

at::Tensor SlicedParamLocator::gather(long param_id, at::Tensor src) const {
  assert(registered(param_id));

  torch::NoGradGuard no_grad;

  const auto& slice_info = slice_info_.at(param_id);
  const auto buf_type = IRType::createTensorType(
      toTensorElemType(src.scalar_type()), slice_info.type.getTensorDim(),
      false);
  at::Tensor buf = createBufTensor(buf_type);
  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(slice_info.ranks);
  NCCLWrapper& nccl = NCCLWrapper::get();
  nccl.createCommunicator(tag, slice_info.ranks);

  src = src.contiguous();
  nccl.allgather(tag, {src}, {buf});
  nccl.syncWithErrorCheck();

  std::vector<at::Tensor> recv_bufs;
  size_t seg_size = src.numel();
  for (size_t i = 0; i < slice_info.ranks.size(); i++) {
    const auto segment = torch::flatten(buf).narrow(0, i * seg_size, seg_size);
    recv_bufs.push_back(segment.view(src.sizes()));
  }
  return torch::cat(recv_bufs, slice_info.dim)
      .view(slice_info.type.getTensorDim())
      .clone()
      .detach();
}

at::Tensor SlicedParamLocator::gather(long param_id, bool grad) const {
  assert(registered(param_id));

  torch::NoGradGuard no_grad;

  const auto& param = sliced_param_.at(param_id);
  at::Tensor src = grad ? param.grad() : param;
  if (!src.defined()) {
    src = torch::zeros_like(param);
  }

  return gather(param_id, src);
}
} // namespace rannc