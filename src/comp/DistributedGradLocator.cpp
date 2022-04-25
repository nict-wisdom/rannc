//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include "DistributedGradLocator.h"
#include <ConfiguredTorch.h>

namespace rannc {

void DistributedGradLocator::registerGrad(
    long pid, const at::Tensor& param, const std::unordered_set<int>& ranks) {
  doRegister(pid, param, ranks);
  params_[pid] = param;
  buf_aligned_[pid] = false;
}

at::Tensor DistributedGradLocator::getLocalParamSegment(long pid) {
  if (!contains(local_param_segments_, pid)) {
    // Keep local segment to set grad to it.
    auto& param = params_.at(pid);
    local_param_segments_[pid] = getSegment(pid, my_indices_.at(pid), false)
                                     .detach()
                                     .set_requires_grad(param.requires_grad());
  }
  return local_param_segments_.at(pid);
}

void DistributedGradLocator::setGradToLocalParamSegment(long pid) {
  auto local_param_segment = getLocalParamSegment(pid);
  if (local_param_segment.requires_grad()) {
    getMutableGradRef(local_param_segment) =
        getSegment(pid, my_indices_.at(pid), true);
  }
}

at::Tensor DistributedGradLocator::getSegment(long pid, int index, bool grad) {
  checkIndices(pid, index);

  auto& param = params_.at(pid);
  size_t offset = offsets_.at(pid).at(index);
  size_t src_size = src_sizes_.at(pid).at(index);

  auto& ten = grad ? param.grad() : param;
  if (grad) {
    assert(param.grad().defined());
  }
  return ten.view(-1).narrow(0, offset, src_size);
}

void DistributedGradLocator::alignBuffer(long pid) {
  const auto& param = params_.at(pid);
  const auto& ranks = ranks_.at(pid);

  int64_t aligned_size = calcAlignedNumElems(param, ranks.size());

  const auto buf_type = IRType::createTensorType(
      toTensorElemType(param.scalar_type()), {aligned_size}, false);
  at::Tensor buf = createBufTensor(buf_type);
  buf.narrow(0, 0, param.numel()).copy_(param.flatten());
  buf_tensors_[pid] = buf;
  param.set_data(buf.narrow(0, 0, param.numel()).view(getTensorDim(param)));
  buf_aligned_[pid] = true;
}

at::Tensor DistributedGradLocator::getBuffer(long pid, bool grad) const {
  assert(contains(buf_aligned_, pid) && buf_aligned_.at(pid));

  assert(contains(buf_tensors_, pid));
  auto& param = buf_tensors_.at(pid);
  return grad ? param.grad() : param;
}

at::Tensor DistributedGradLocator::getBufferSegment(
    long pid, int index, bool grad) const {
  checkIndices(pid, index);
  assert(contains(buf_aligned_, pid) && buf_aligned_.at(pid));

  auto& param = buf_tensors_.at(pid);
  const auto seg_size = segment_sizes_.at(pid);

  auto& ten = grad ? param.grad() : param;
  if (grad) {
    assert(param.grad().defined());
  }
  return ten.view(-1).narrow(0, seg_size * index, seg_size);
}

void DistributedGradLocator::checkIndices(long pid, int index) const {
  assert(contains(params_, pid));

  auto& param = params_.at(pid);
  assert(contains(offsets_, pid));
  assert(offsets_.at(pid).size() > index);
  size_t offset = offsets_.at(pid).at(index);

  assert(contains(src_sizes_, pid));
  assert(src_sizes_.at(pid).size() > index);
  size_t src_size = src_sizes_.at(pid).at(index);

  assert(param.numel() >= offset + src_size);
}
} // namespace rannc