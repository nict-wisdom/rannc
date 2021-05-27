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

    at::Tensor DistributedGradLocator::getSegment(long pid, int index, bool grad) {
        assert(contains(params_, pid));

        auto &param = params_.at(pid);

        assert(contains(offsets_, pid));
        assert(offsets_.at(pid).size() > index);
        size_t offset = offsets_.at(pid).at(index);

        assert(contains(src_sizes_, pid));
        assert(src_sizes_.at(pid).size() > index);
        size_t src_size = src_sizes_.at(pid).at(index);

        auto &ten = grad ? param.grad() : param;

        if (grad) {
            assert(param.grad().defined());
        }

        assert(ten.numel() >= offset + src_size);
        return ten.flatten().slice(0, offset, offset+src_size);
    }
}