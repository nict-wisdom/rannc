//
// Created by Masahiro Tanaka on 2021/05/13.
//

#ifndef PYRANNC_ZEROPARAMLOCATOR_H
#define PYRANNC_ZEROPARAMLOCATOR_H

#include <torch/torch.h>

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

    private:
        ZeroParamLocator() = default;
        ~ZeroParamLocator() = default;

        std::unordered_map<long, at::Tensor> params_;
        std::unordered_map<int, int64_t> sizes_;
        std::unordered_map<long, int> owners_;
    };
}

#endif //PYRANNC_ZEROPARAMLOCATOR_H
