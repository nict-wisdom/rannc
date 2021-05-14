//
// Created by Masahiro Tanaka on 2021/05/13.
//

#ifndef PYRANNC_ZEROPARAMLOCATOR_H
#define PYRANNC_ZEROPARAMLOCATOR_H

#include <torch/torch.h>
#include <comm/NCCLWrapper.h>

#include "graph/ir.h"

namespace rannc {

    class ZeroParamLocator {
    public:
        ZeroParamLocator(const ZeroParamLocator&) = delete;
        ZeroParamLocator& operator=(const ZeroParamLocator&) = delete;
        ZeroParamLocator(ZeroParamLocator&&) = delete;
        ZeroParamLocator& operator=(ZeroParamLocator&&) = delete;

        static ZeroParamLocator& get() {
            static ZeroParamLocator instance;
            return instance;
        }

        int store(long pid, const at::Tensor& param);
        at::Tensor load(long pid);

        void fetchStart();
        at::Tensor fetch(long pid);
        void fetchEnd();

    private:
        ZeroParamLocator() : nccl_(NCCLWrapper::get()) {};
        ~ZeroParamLocator() = default;

        NCCLWrapper& nccl_;

        std::unordered_map<long, at::Tensor> params_;
        std::unordered_map<int, int64_t> sizes_;
        std::unordered_map<long, int> owners_;
        std::unordered_map<long, long> global_id_to_local_;

        std::unordered_map<long, IRType> ir_types_;
        std::unordered_map<long, c10::DeviceType> device_types_;

        static const int FETCH_TAG;
    };
}

#endif //PYRANNC_ZEROPARAMLOCATOR_H
