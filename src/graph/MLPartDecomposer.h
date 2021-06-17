//
// Created by Masahiro Tanaka on 2020/02/24.
//

#ifndef PYRANNC_MLPARTDECOMPOSER_H
#define PYRANNC_MLPARTDECOMPOSER_H

#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {

    class MLPartDecomposer {
    public:
        MLPartDecomposer(std::shared_ptr<GraphProfiler> sg_prof, int worker_num, int64_t batch_size,
                         std::unordered_map<std::string, GraphProfile> node_profiles, size_t dev_mem,
                         bool use_amp_master_params, bool enable_zero)
        :sg_prof_(std::move(sg_prof)), worker_num_(worker_num), batch_size_(batch_size), dev_mem_(dev_mem),
         node_profiles_(std::move(node_profiles)),
         use_amp_master_params_(use_amp_master_params),
         enable_zero_(enable_zero) {}

        Deployment decompose(const std::shared_ptr<IRGraph>& irGraph);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;
        int worker_num_;
        int64_t batch_size_;
        size_t dev_mem_;
        std::unordered_map<std::string, GraphProfile> node_profiles_;
        bool use_amp_master_params_;
        bool enable_zero_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };

}

#endif //PYRANNC_MLPARTDECOMPOSER_H
