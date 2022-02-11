//
// Created by Masahiro Tanaka on 2020/02/06.
//

#include "MetaDecomposer.h"
#include "FairWeightDecomposer.h"
#include "MLPartDecomposer.h"

namespace rannc {
enum class DecomposerType { FAIR_WEIGHT, ML_PART };

struct DecomposerInfo {
  std::string name;
  DecomposerType type;
};

DecomposerInfo decomp_table[] = {
    {"fair_weight", DecomposerType::FAIR_WEIGHT},
    {"ml_part", DecomposerType::ML_PART}};

Deployment MetaDecomposer::decompose(
    const std::string& name, const std::shared_ptr<IRGraph>& ir_graph) {
  std::unordered_map<std::string, DecomposerType> type_map;
  for (const auto& it : decomp_table) {
    type_map[it.name] = it.type;
  }

  if (!contains(type_map, name)) {
    throw std::invalid_argument("Unknown decomposer: " + name);
  }

  logger->info("Decomposer: {}", name);

  Deployment deployment;
  switch (type_map.at(name)) {
    case DecomposerType::FAIR_WEIGHT: {
      FairWeightDecomposer decomposer(
          sg_prof_, conf_.dev_num, conf_.batch_size, conf_.dev_mem);
      deployment = decomposer.decompose(ir_graph);
      break;
    }
    case DecomposerType::ML_PART: {
      MLPartDecomposer decomposer(sg_prof_, conf_);
      deployment = decomposer.decompose(ir_graph);
      break;
    }
  }
  return deployment;
}
} // namespace rannc