//
// Created by Masahiro Tanaka on 2021/05/13.
//

#ifndef PYRANNC_DISTRIBUTEDPARAMLOCATOR_H
#define PYRANNC_DISTRIBUTEDPARAMLOCATOR_H

#include <torch/torch.h>
#include <comm/NCCLWrapper.h>

#include "graph/ir.h"

namespace rannc {

    class DistributedParamLocator {
    public:
        DistributedParamLocator(const DistributedParamLocator&) = delete;
        DistributedParamLocator& operator=(const DistributedParamLocator&) = delete;
        DistributedParamLocator(DistributedParamLocator&&) = delete;
        DistributedParamLocator& operator=(DistributedParamLocator&&) = delete;

        static DistributedParamLocator& get() {
            static DistributedParamLocator instance;
            return instance;
        }

        int store(long pid, const at::Tensor& param);
        at::Tensor load(long pid);
        void disable(long pid);

        int getOwner(long pid);

        void fetchStart();
        at::Tensor fetch(long pid);
        void fetchEnd();

    private:
        DistributedParamLocator() : nccl_(NCCLWrapper::get()) {};
        ~DistributedParamLocator() = default;

        NCCLWrapper& nccl_;
        int comm_tag_;

        std::unordered_map<long, at::Tensor> params_;
        std::unordered_map<int, int64_t> sizes_;
        std::unordered_map<long, int> owners_;
        std::unordered_map<long, long> global_id_to_local_;

        std::unordered_map<long, IRType> ir_types_;

        static const int FETCH_TAG;
    };
}

#endif //PYRANNC_DISTRIBUTEDPARAMLOCATOR_H
