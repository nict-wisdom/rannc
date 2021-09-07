//
// Created by Masahiro Tanaka on 2021/05/19.
//

#ifndef PYRANNC_DISTRIBUTEDGRADLOCATOR_H
#define PYRANNC_DISTRIBUTEDGRADLOCATOR_H

#include <torch/torch.h>
#include "DistributedParamLocatorBase.h"

namespace rannc {

class DistributedGradLocator : public DistributedParamLocatorBase {
 public:
  void registerGrad(
      long pid, const at::Tensor& param, const std::unordered_set<int>& ranks);
  at::Tensor getSegment(long pid, int index, bool grad);
  at::Tensor getLocalParamSegment(long pid);
  void setGradToLocalParamSegment(long pid);

 private:
  void checkIndices(long pid, int index);

  std::unordered_map<long, at::Tensor> params_;
  std::unordered_map<long, at::Tensor> local_param_segments_;
};
} // namespace rannc

#endif // PYRANNC_DISTRIBUTEDGRADLOCATOR_H
