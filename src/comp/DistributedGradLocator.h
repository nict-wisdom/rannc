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
        void registerGrad(long pid, const at::Tensor& param, const std::unordered_set<int>& ranks);
        at::Tensor getParamSegment(long pid, int index);
        at::Tensor getGradSegment(long pid, int index);

    private:
        at::Tensor getSegment(long pid, int index, bool grad);
        std::unordered_map<long, at::Tensor> params_;
        std::unordered_map<long, at::Tensor> grad_buffers_;
        std::unordered_map<long, at::Tensor> stashed_buffers_;
    };
}

#endif //PYRANNC_DISTRIBUTEDGRADLOCATOR_H
