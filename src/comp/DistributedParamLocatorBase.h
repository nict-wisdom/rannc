//
// Created by Masahiro Tanaka on 2021/05/19.
//

#ifndef PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H
#define PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H

#include <torch/torch.h>
#include <comm/NCCLWrapper.h>

#include "graph/ir.h"

namespace rannc {

    class DistributedParamLocatorBase {
    public:
        int getOwner(long pid);

    protected:
        NCCLWrapper& nccl_;
        int comm_tag_;

        std::unordered_map<int, int64_t> sizes_;
        std::unordered_map<long, int> owners_;
        std::unordered_map<long, long> global_id_to_local_;
        std::unordered_map<long, IRType> ir_types_;
        std::unordered_map<long, at::Tensor> params_;

        DistributedParamLocatorBase() : nccl_(NCCLWrapper::get()) {};
        ~DistributedParamLocatorBase() = default;

        int doRegister(long pid, const at::Tensor& param, const std::unordered_set<int>& ranks);

        static const int FETCH_TAG;
    };

}

#endif //PYRANNC_DISTRIBUTEDPARAMLOCATORBASE_H
