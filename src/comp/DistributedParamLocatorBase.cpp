//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include "DistributedParamLocatorBase.h"
#include <comm/ObjectComm.h>
#include <comm/SComm.h>

#include <Config.h>
#include <mpi.h>

namespace rannc {

void DistributedParamLocatorBase::doRegister(
    long pid, const at::Tensor& param, const std::unordered_set<int>& ranks) {
  torch::NoGradGuard no_grad;

  int64_t aligned_size = alignSize(param, ranks.size());
  int64_t segment_size = aligned_size / ranks.size();
  int64_t offset = 0;
  for (size_t i = 0; i < ranks.size(); i++) {
    offsets_[pid].push_back(offset);
    int64_t src_size =
        std::min((int64_t)(param.numel() - offset), segment_size);
    src_sizes_[pid].push_back(src_size);
    offset += src_size;
  }

  segment_sizes_[pid] = segment_size;
  ranks_[pid] = ranks;

  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);
  SComm& scomm = SComm::get();
  MPI_Comm communicator = scomm.getCommunicator(tag, ranks);

  ObjectComm& ocomm = ObjectComm::get();
  long global_id = pid;
  global_id = ocomm.bcast(global_id, 0, communicator);
  global_id_to_local_[global_id] = pid;

  ir_types_[pid] = toIRType(param);
  my_indices_[pid] = getLocalRank(ranks, mpi::getRank());

  MPI_Barrier(communicator);
}

void DistributedParamLocatorBase::remove(long pid) {
  global_id_to_local_.erase(pid);
  offsets_.erase(pid);
  src_sizes_.erase(pid);
  ranks_.erase(pid);
  segment_sizes_.erase(pid);
  ir_types_.erase(pid);
  my_indices_.erase(pid);
  buf_tensors_.erase(pid);
}

size_t DistributedParamLocatorBase::getSegmentNum(long pid) {
  assert(contains(ranks_, pid));
  return ranks_.at(pid).size();
}

std::pair<int64_t, int64_t> DistributedParamLocatorBase::getSegmentRange(
    long pid, int index) {
  assert(contains(offsets_, pid));
  assert(contains(src_sizes_, pid));
  assert(offsets_.at(pid).size() > index);
  assert(src_sizes_.at(pid).size() > index);

  int64_t offset = offsets_.at(pid).at(index);
  int64_t src_size = src_sizes_.at(pid).at(index);
  return std::pair<int64_t, int64_t>(offset, offset + src_size);
}

std::pair<int64_t, int64_t> DistributedParamLocatorBase::getSegmentRange(
    long pid) {
  assert(contains(my_indices_, pid));
  return getSegmentRange(pid, my_indices_.at(pid));
}

size_t DistributedParamLocatorBase::getOwner(long pid, int index) {
  assert(contains(ranks_, pid));

  auto ranks_buf = setToVector(ranks_.at(pid));
  assert(ranks_buf.size() > index);
  std::sort(ranks_buf.begin(), ranks_buf.end());

  return ranks_buf.at(index);
}

bool DistributedParamLocatorBase::registered(long pid) {
  return contains(ranks_, pid);
}

at::Tensor DistributedParamLocatorBase::gather(
    const at::Tensor& tensor_part, long pid) {
  torch::NoGradGuard no_grad;

  assert(contains(ranks_, pid));
  assert(contains(ir_types_, pid));

  const IRType& ir_type = ir_types_.at(pid);
  const auto& ranks = ranks_.at(pid);

  bool use_mpi =
      config::Config::get().getVal<bool>(config::USE_MPI_TO_GATHER_DIST_PARAMS);

  at::TensorOptions options;
  options = options.dtype(tensor_part.dtype()).requires_grad(false);

  if (use_mpi) {
    options = options.device(c10::Device(c10::DeviceType::CPU));
  } else {
    options = options.device(c10::Device(c10::DeviceType::CUDA));
  }

  at::Tensor buf =
      torch::zeros({(int64_t)(segment_sizes_.at(pid) * ranks.size())}, options);

  TagMap& tag_map = TagMap::get();
  int tag = tag_map.getRankSetTag(ranks);
  nccl_.createCommunicator(tag, ranks);

  // An error occurs when tensor_part's requires_grad is true.
  bool requires_grad = tensor_part.requires_grad();
  tensor_part.set_requires_grad(false);
  at::Tensor sendbuf =
      torch::zeros({(int64_t)(segment_sizes_.at(pid))}, options);
  auto sendbuf_view = sendbuf.narrow(0, 0, tensor_part.numel());
  sendbuf_view.copy_(tensor_part);
  // Recover the flag
  tensor_part.set_requires_grad(requires_grad);

  if (use_mpi) {
    SComm& scomm = SComm::get();
    MPI_Comm communicator = scomm.getCommunicator(tag, ranks);
    MPI_Allgather(
        sendbuf.data_ptr(), sendbuf.numel(),
        scalarTypeToMPIDatatype(sendbuf.scalar_type()), buf.data_ptr(),
        sendbuf.numel(), scalarTypeToMPIDatatype(buf.scalar_type()),
        communicator);
  } else {
    nccl_.allgather(tag, {sendbuf}, {buf});
    nccl_.syncWithErrorCheck();
  }

  return buf.narrow(0, 0, productDim(ir_type.getTensorDim()))
      .view(ir_type.getTensorDim())
      .cpu()
      .detach()
      .set_requires_grad(requires_grad);
}

long DistributedParamLocatorBase::pidToLocal(long global_pid) const {
  assert(contains(global_id_to_local_, global_pid));
  return global_id_to_local_.at(global_pid);
}

void DistributedParamLocatorBase::clear() {
  offsets_.clear();
  src_sizes_.clear();
  global_id_to_local_.clear();
  ir_types_.clear();
  segment_sizes_.clear();
  ranks_.clear();
  my_indices_.clear();
  buf_tensors_.clear();
}
} // namespace rannc