//
// Created by Masahiro Tanaka on 2019-05-30.
//

#ifndef PYRANNC_NODEPROFILER_H
#define PYRANNC_NODEPROFILER_H

#include <graph/ir.h>
#include <torch/TorchUtil.h>
#include <ostream>
#include "GraphProfiler.h"

namespace rannc {

    class NodeProfiler {
    public:
        NodeProfiler(std::shared_ptr<GraphProfiler> sg_prof) :
                sg_prof_(std::move(sg_prof)) {}

        ProfilingResult profile(const std::shared_ptr<IRGraph>& ir_graph, int iteration);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("NodeProfiler");
    };

}

#endif //PYRANNC_NODEPROFILER_H
