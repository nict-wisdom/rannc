//
// Created by Masahiro Tanaka on 2022/02/28.
//

#ifndef PYRANNC_SLICEDPARAMLOCATOR_H
#define PYRANNC_SLICEDPARAMLOCATOR_H

#include <graph/ir.h>

namespace rannc {

struct SliceInfo {
  std::unordered_set<int> ranks;
  IRType type;
  size_t dim;
};

class SlicedParamLocator {
 public:
  SlicedParamLocator() = default;

  at::Tensor registerParam(
      long pid, at::Tensor orig_param, size_t dim,
      const std::unordered_set<int>& ranks);
  bool registered(long pid) const;
  at::Tensor gather(long pid, bool grad) const;
  // for amp master param
  at::Tensor gather(long pid, at::Tensor src) const;

 private:
  std::unordered_map<long, at::Tensor> sliced_param_;
  std::unordered_map<long, SliceInfo> slice_info_;
};

} // namespace rannc

#endif // PYRANNC_SLICEDPARAMLOCATOR_H
