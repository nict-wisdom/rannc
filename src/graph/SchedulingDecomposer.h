//
// Created by Masahiro Tanaka on 2020/02/04.
//

#ifndef PYRANNC_SCHEDULINGDECOMPOSER_H
#define PYRANNC_SCHEDULINGDECOMPOSER_H

#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {

    class SchedulingDecomposer {
    public:
        SchedulingDecomposer(std::shared_ptr<GraphProfiler> sg_prof, int worker_num, int64_t batch_size,
                             std::unordered_map<std::string, GraphProfile> node_profiles, size_t dev_mem)
        :sg_prof_(std::move(sg_prof)), worker_num_(worker_num), batch_size_(batch_size), dev_mem_(dev_mem),
         node_profiles_(std::move(node_profiles)) {}

        Deployment decompose(const std::shared_ptr<IRGraph>& irGraph);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;
        int worker_num_;
        int64_t batch_size_;
        size_t dev_mem_;
        std::unordered_map<std::string, GraphProfile> node_profiles_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };


}

#endif //PYRANNC_SCHEDULINGDECOMPOSER_H
