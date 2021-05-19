//
// Created by Masahiro Tanaka on 2021/05/13.
//

#ifndef PYRANNC_DISTRIBUTEDPARAMLOCATOR_H
#define PYRANNC_DISTRIBUTEDPARAMLOCATOR_H

#include <torch/torch.h>
#include <comm/NCCLWrapper.h>

#include "graph/ir.h"
#include "DistributedParamLocatorBase.h"

namespace rannc {

    class DistributedParamLocator : public DistributedParamLocatorBase {
    public:
        int store(long pid, const at::Tensor& param);
        at::Tensor load(long pid);
        void disable(long pid);

        void fetchStart();
        at::Tensor fetch(long pid);
        void fetchEnd();

        static DistributedParamLocator& get() {
            static DistributedParamLocator instance;
            return instance;
        }

    private:
        std::unordered_map<long, at::Tensor> params_;
    };
}

#endif //PYRANNC_DISTRIBUTEDPARAMLOCATOR_H
