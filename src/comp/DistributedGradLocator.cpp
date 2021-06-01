//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include <ConfiguredTorch.h>
#include "DistributedGradLocator.h"

namespace rannc {

    void DistributedGradLocator::registerGrad(long pid, const at::Tensor &param, const std::unordered_set<int> &ranks) {
        doRegister(pid, param, ranks);
        params_[pid] = param;
    }

    at::Tensor DistributedGradLocator::getLocalParamSegment(long pid) {
        if (!contains(local_param_segments_, pid)) {
            // Keep local segment to set grad to it.
            local_param_segments_[pid] = getSegment(pid, my_indices_.at(pid), false).detach();
        }
        return local_param_segments_.at(pid);
    }

    void DistributedGradLocator::setGradToLocalParamSegment(long pid) {
        assert(contains(local_param_segments_, pid));
        getMutableGradRef(local_param_segments_.at(pid)) = getSegment(pid, my_indices_.at(pid), true);
    }

    at::Tensor DistributedGradLocator::getSegment(long pid, int index, bool grad) {
        checkIndices(pid, index);

        auto &param = params_.at(pid);
        size_t offset = offsets_.at(pid).at(index);
        size_t src_size = src_sizes_.at(pid).at(index);

        auto &ten = grad ? param.grad() : param;
        if (grad) {
            assert(param.grad().defined());
        }
        return ten.flatten().slice(0, offset, offset+src_size);
    }

    void DistributedGradLocator::checkIndices(long pid, int index) {
        assert(contains(params_, pid));

        auto &param = params_.at(pid);
        assert(contains(offsets_, pid));
        assert(offsets_.at(pid).size() > index);
        size_t offset = offsets_.at(pid).at(index);

        assert(contains(src_sizes_, pid));
        assert(src_sizes_.at(pid).size() > index);
        size_t src_size = src_sizes_.at(pid).at(index);

        assert(param.numel() >= offset + src_size);
    }
}