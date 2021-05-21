//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include <ConfiguredTorch.h>
#include "DistributedGradLocator.h"

namespace rannc {

    int DistributedGradLocator::registerGrad(long pid, const at::Tensor &param, const std::unordered_set<int> &ranks) {
        int owner = doRegister(pid, param, ranks);
        params_[pid] = param;
        grad_buffers_[pid] = torch::zeros_like(param).cuda();
        return owner;
    }

    void DistributedGradLocator::setGrad(long pid) {
        assert(contains(params_, pid));
        assert(contains(grad_buffers_, pid));

        auto &param = params_.at(pid);
        stashed_buffers_[pid] = param.grad();
        getMutableGradRef(param) = grad_buffers_.at(pid);
        param.grad().zero_();
    }

    at::Tensor DistributedGradLocator::getGradBuffer(long pid) {
        assert(contains(grad_buffers_, pid));
        return grad_buffers_.at(pid);
    }

    void DistributedGradLocator::unstashGrad(long pid) {
        assert(contains(params_, pid));
        assert(contains(stashed_buffers_, pid));

        auto &param = params_.at(pid);
        getMutableGradRef(param) = stashed_buffers_.at(pid);
    }
}