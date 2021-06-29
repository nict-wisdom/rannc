//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include <comm/ObjectComm.h>
#include <comm/SComm.h>
#include "DistributedParamLocatorBase.h"

namespace rannc {

    void DistributedParamLocatorBase::doRegister(long pid, const at::Tensor& param, const std::unordered_set<int>& ranks) {

        int64_t segment_size = ceil(param.numel() / (double) ranks.size());
        int64_t offset = 0;
        for (size_t i=0; i<ranks.size(); i++) {
            offsets_[pid].push_back(offset);
            int64_t src_size = std::min((int64_t) (param.numel() - offset), segment_size);
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
        ir_types_.erase(pid);
        segment_sizes_.erase(pid);
        ir_types_.erase(pid);
        my_indices_.erase(pid);
    }

    size_t DistributedParamLocatorBase::getSegmentNum(long pid) {
        assert(contains(ranks_, pid));
        return ranks_.at(pid).size();
    }

    std::pair<int64_t, int64_t> DistributedParamLocatorBase::getSegmentRange(long pid, int index) {
        assert(contains(offsets_, pid));
        assert(contains(src_sizes_, pid));
        assert(offsets_.at(pid).size() > index);
        assert(src_sizes_.at(pid).size() > index);

        int64_t offset = offsets_.at(pid).at(index);
        int64_t src_size = src_sizes_.at(pid).at(index);
        return std::pair<int64_t, int64_t>(offset, offset + src_size);
    }

    std::pair<int64_t, int64_t> DistributedParamLocatorBase::getSegmentRange(long pid) {
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

    at::Tensor DistributedParamLocatorBase::gather(const at::Tensor& tensor_part, long pid) {
        assert(contains(segment_sizes_, pid));
        assert(contains(ranks_, pid));
        assert(contains(ir_types_, pid));

        const IRType& ir_type = ir_types_.at(pid);
        const auto& ranks = ranks_.at(pid);

        at::TensorOptions options;
        options = options.dtype(tensor_part.dtype())
                .requires_grad(tensor_part.requires_grad())
                .device(c10::Device(c10::DeviceType::CUDA));
        at::Tensor buf = torch::zeros({(int64_t)(segment_sizes_.at(pid)*ranks.size())}, options);

        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(ranks);
        nccl_.createCommunicator(tag, ranks);

        at::NoGradGuard no_grad;

        const auto sendbuf = tensor_part.set_requires_grad(false).cuda();
        nccl_.allgather(tag, {sendbuf}, {buf});

        return buf.slice(0, 0, productDim(ir_type.getTensorDim()))
                .view(ir_type.getTensorDim())
                .cpu().detach();
    }
}