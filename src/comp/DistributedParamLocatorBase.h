//
// Created by Masahiro Tanaka on 2021/05/19.
//

#ifndef PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H
#define PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H

#include <comm/NCCLWrapper.h>
#include <torch/torch.h>

#include "graph/ir.h"

namespace rannc {

class DistributedParamLocatorBase {
 public:
  virtual void remove(long pid);
  size_t getSegmentNum(long pid);
  size_t getOwner(long pid, int index);
  bool registered(long pid);
  std::pair<int64_t, int64_t> getSegmentRange(long pid, int index);
  std::pair<int64_t, int64_t> getSegmentRange(long pid);
  at::Tensor gather(const at::Tensor& tensor_part, long pid);
  long pidToLocal(long global_pid) const;
  virtual void clear();

 protected:
  NCCLWrapper& nccl_;
  int comm_tag_;

  std::unordered_map<long, std::vector<int64_t>> offsets_;
  std::unordered_map<long, std::vector<int64_t>> src_sizes_;
  std::unordered_map<long, long> global_id_to_local_;
  std::unordered_map<long, IRType> ir_types_;
  std::unordered_map<long, int64_t> segment_sizes_;
  std::unordered_map<long, std::unordered_set<int>> ranks_;
  std::unordered_map<long, int> my_indices_;
  std::unordered_map<long, at::Tensor> buf_tensors_;

  DistributedParamLocatorBase() : nccl_(NCCLWrapper::get()){};
  ~DistributedParamLocatorBase() = default;

  void doRegister(
      long pid, const at::Tensor& param, const std::unordered_set<int>& ranks);
  int64_t alignSize(const at::Tensor& ten, size_t split_num);

  static const int FETCH_TAG;
};

} // namespace rannc

#endif // PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H
