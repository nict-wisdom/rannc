//
// Created by Masahiro Tanaka on 2019-07-01.
//

#ifndef PYRANNC_MANUALDECOMPOSER_H
#define PYRANNC_MANUALDECOMPOSER_H

#include "Decomposition.h"

namespace rannc {
    class IRGraph;

    class ManualDecomposer {
    public:
        Deployment decompose(const std::shared_ptr<IRGraph>& irGraph, int n_partition, int64_t batch_size);

    private:
        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };

}
#endif //PYRANNC_MANUALDECOMPOSER_H
