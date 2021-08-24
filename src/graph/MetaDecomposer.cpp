//
// Created by Masahiro Tanaka on 2020/02/06.
//

#include "MetaDecomposer.h"
#include "FairWeightDecomposer.h"
#include "ProfiledWeightDecomposer.h"
#include "SchedulingDecomposer.h"
#include "MLPartDecomposer.h"

namespace rannc {
    enum class DecomposerType {
        FAIR_WEIGHT,
        PROFILED_WEIGHT,
        SCHEDULING,
        ML_PART
    };

    struct DecomposerInfo {
        std::string name;
        DecomposerType type;
    };

    DecomposerInfo decomp_table[] = {
        {"fair_weight", DecomposerType::FAIR_WEIGHT},
        {"profiled_weight", DecomposerType::PROFILED_WEIGHT},
        {"scheduling", DecomposerType::SCHEDULING},
        {"ml_part", DecomposerType::ML_PART}
    };

    Deployment MetaDecomposer::decompose(const std::string& name, const std::shared_ptr<IRGraph>& ir_graph) {
        std::unordered_map<std::string, DecomposerType> type_map;
        for (const auto& it: decomp_table) {
            type_map[it.name] = it.type;
        }

        if (!contains(type_map, name)) {
            throw std::invalid_argument("Unknown decomposer: " + name);
        }

        logger->info("Decomposer: {}", name);

        Deployment deployment;
        switch (type_map.at(name)) {
            case DecomposerType::FAIR_WEIGHT: {
                FairWeightDecomposer decomposer(sg_prof_, worker_num_, batch_size_, dev_mem_);
                deployment = decomposer.decompose(ir_graph);
                break;
            }
            case DecomposerType::PROFILED_WEIGHT: {
                ProfiledWeightDecomposer decomposer(sg_prof_, worker_num_, batch_size_, dev_mem_);
                deployment = decomposer.decompose(ir_graph);
                break;
            }
            case DecomposerType::SCHEDULING: {
                SchedulingDecomposer decomposer(sg_prof_, worker_num_, batch_size_, node_profiles_, dev_mem_);
                deployment = decomposer.decompose(ir_graph);
                break;
            }
            case DecomposerType::ML_PART: {
                MLPartDecomposer decomposer(sg_prof_, worker_num_, batch_size_, node_profiles_, dev_mem_,
                                            use_amp_master_params_, enable_zero_);
                deployment = decomposer.decompose(ir_graph);
                break;
            }
        }
        return deployment;
    }
}