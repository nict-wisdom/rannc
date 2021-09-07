//
// Created by Masahiro Tanaka on 2021/05/13.
//

#include "DistributedParamLocator.h"

#include "comm/MPIUtil.h"
#include "comm/NCCLWrapper.h"
#include "comm/ObjectComm.h"
#include "comm/SComm.h"

namespace rannc {

at::Tensor DistributedParamLocator::store(long pid, const at::Tensor& param) {
  const auto ranks = mpi::getAllRanks();
  doRegister(pid, param, ranks);

  assert(offsets_.at(pid).size() == ranks.size());
  assert(src_sizes_.at(pid).size() == ranks.size());

  int local_rank = getLocalRank(ranks, mpi::getRank());
  int64_t offset = offsets_.at(pid).at(local_rank);
  int64_t src_size = src_sizes_.at(pid).at(local_rank);

  at::TensorOptions options;
  options = options.dtype(param.dtype())
                .device(param.device())
                .requires_grad(param.requires_grad());
  at::Tensor part_tensor = torch::zeros({src_size}, options);

  if (src_size > 0) {
    torch::NoGradGuard no_grad;
    auto src_buf = torch::flatten(param).narrow(0, offset, src_size);
    part_tensor.copy_(src_buf);
  }
  param_parts_[pid] = part_tensor;
  return param_parts_[pid];
}

at::Tensor DistributedParamLocator::load(long pid) {
  assert(contains(param_parts_, pid));
  auto param_part = param_parts_.at(pid);
  return gather(param_part, pid);
}

at::Tensor DistributedParamLocator::getSegment(long pid) {
  assert(contains(param_parts_, pid));
  return param_parts_.at(pid);
}

void DistributedParamLocator::set(long pid, const at::Tensor& src) {
  const auto ranks = mpi::getAllRanks();
  int local_rank = getLocalRank(ranks, mpi::getRank());

  assert(contains(src_sizes_, pid));
  int64_t src_size = src_sizes_.at(pid).at(local_rank);

  if (src_size > 0) {
    torch::NoGradGuard no_grad;

    int64_t offset = offsets_.at(pid).at(local_rank);
    auto src_buf = torch::flatten(src).narrow(0, offset, src_size);
    assert(contains(param_parts_, pid));
    auto param_part = param_parts_.at(pid);
    param_part.copy_(src_buf);
  }
}

void DistributedParamLocator::setScalarType(
    long pid, const c10::ScalarType& stype) {
  assert(contains(param_parts_, pid));
  auto param_part = param_parts_.at(pid);

  torch::NoGradGuard no_grad;
  param_part.set_requires_grad(false);
  param_parts_[pid] = param_part.to(stype);

  assert(contains(ir_types_, pid));
  auto ir_type = ir_types_.at(pid);
  ir_types_[pid] = IRType::createTensorType(
      toTensorElemType(stype), ir_type.getTensorDim(), ir_type.requiresGrad());
  param_parts_[pid].set_requires_grad(ir_type.requiresGrad());
}

void DistributedParamLocator::fetchStart() {
  TagMap& tag_map = TagMap::get();
  comm_tag_ = tag_map.getRankSetTag(mpi::getAllRanks());
  nccl_.createCommunicator(comm_tag_, mpi::getAllRanks());

  if (mpi::getRank() != 0) {
    long global_pid;
    MPI_Bcast(&global_pid, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    while (global_pid != 0) {
      assert(contains(global_id_to_local_, global_pid));
      load(global_id_to_local_.at(global_pid));
      MPI_Bcast(&global_pid, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    }
  }
}

at::Tensor DistributedParamLocator::fetch(long pid) {
  MPI_Bcast(&pid, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  return load(pid);
}

void DistributedParamLocator::fetchEnd() {
  if (mpi::getRank() == 0) {
    long pid = 0;
    MPI_Bcast(&pid, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  }
}

void DistributedParamLocator::remove(long pid) {
  DistributedParamLocatorBase::remove(pid);
  param_parts_.erase(pid);
}

void DistributedParamLocator::clear() {
  DistributedParamLocatorBase::clear();
  param_parts_.clear();
}
} // namespace rannc