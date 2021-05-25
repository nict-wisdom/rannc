//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include <ConfiguredTorch.h>
#include "DistributedGradLocator.h"

namespace rannc {

    void DistributedGradLocator::registerGrad(long pid, const at::Tensor &param, const std::unordered_set<int> &ranks) {
        doRegister(pid, param, ranks);
        params_[pid] = param;
        grad_buffers_[pid] = torch::zeros_like(param).cuda();
    }

    void DistributedGradLocator::stashGrad(long pid) {
        assert(contains(params_, pid));
        assert(contains(grad_buffers_, pid));

        auto &param = params_.at(pid);
        if (param.grad().defined()) {
            stashed_buffers_[pid] = param.grad();
        }
        grad_buffers_.at(pid).zero_();
        getMutableGradRef(param) = grad_buffers_.at(pid);
    }

    at::Tensor DistributedGradLocator::getSegment(long pid, int index) {
        assert(contains(params_, pid));

        auto &param = params_.at(pid);
        assert(param.grad().defined());

        assert(contains(offsets_, pid));
        assert(offsets_.at(pid).size() > index);
        size_t offset = offsets_.at(pid).at(index);

        assert(contains(src_sizes_, pid));
        assert(src_sizes_.at(pid).size() > index);
        size_t src_size = offsets_.at(pid).at(index);

        auto &grad = param.grad();
        assert(grad.numel() > offset + src_size);

        return grad.flatten().slice(0, offset, offset+src_size);
    }

    void DistributedGradLocator::unstashGrad(long pid) {
        assert(contains(params_, pid));

        if (!contains(stashed_buffers_, pid)) {
            return;
        }

        auto &param = params_.at(pid);
        if (stashed_buffers_.at(pid).defined()) {
            getMutableGradRef(param) += stashed_buffers_.at(pid);
        }
    }
}