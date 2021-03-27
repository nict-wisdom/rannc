//
// Created by Masahiro Tanaka on 2020/02/06.
//

#ifndef PYRANNC_METADECOMPOSER_H
#define PYRANNC_METADECOMPOSER_H

#include "Decomposition.h"

namespace rannc {

    class GraphProfiler;
    class MetaDecomposer {
    public:
        MetaDecomposer(std::shared_ptr<GraphProfiler> sg_prof, int worker_num, int64_t batch_size,
                       std::unordered_map<std::string, GraphProfile> node_profiles, size_t dev_mem,
                       bool use_amp_master_params)
          :sg_prof_(std::move(sg_prof)), worker_num_(worker_num), batch_size_(batch_size), dev_mem_(dev_mem),
           node_profiles_(std::move(node_profiles)),
           use_amp_master_params_(use_amp_master_params) {}

        Deployment decompose(const std::string& name, const std::shared_ptr<IRGraph>& ir_graph);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;
        int worker_num_;
        int64_t batch_size_;
        size_t dev_mem_;
        std::unordered_map<std::string, GraphProfile> node_profiles_;
        bool use_amp_master_params_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };
}

#endif //PYRANNC_METADECOMPOSER_H
