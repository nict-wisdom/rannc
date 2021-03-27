//
// Created by Masahiro Tanaka on 2019-03-15.
//

#ifndef PYRANNC_FAIRWEIGHTDECOMPOSER_H
#define PYRANNC_FAIRWEIGHTDECOMPOSER_H


#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {
    class IRGraph;

    class FairWeightDecomposer {
    public:
        FairWeightDecomposer(std::shared_ptr<GraphProfiler> sg_prof, int worker_num, int64_t batch_size, size_t dev_mem)
            :sg_prof_(std::move(sg_prof)), worker_num_(worker_num), batch_size_(batch_size), dev_mem_(dev_mem) {}

        Deployment decompose(const std::shared_ptr<IRGraph>& irGraph);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;
        int worker_num_;
        int64_t batch_size_;
        size_t dev_mem_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };
}

#endif //PYRANNC_FAIRWEIGHTDECOMPOSER_H
