//
// Created by Masahiro Tanaka on 2022/02/02.
//

#include "PartitionTensor.h"

namespace rannc {
std::vector<DistOp> dist_ops = {
    {"aten::linear", "rannc::linear_dist", {{1, 0}}}};

std::unordered_map<std::string, std::string> getDistOpNameMap() {
  std::unordered_map<std::string, std::string> name_map;
  for (const auto& op : dist_ops) {
    name_map[op.original_name] = op.dist_name;
  }
  return name_map;
}

std::unordered_map<std::string, std::pair<size_t, size_t>> getDistParams(
    const std::shared_ptr<IRGraph>& g) {
  std::unordered_map<std::string, DistOp> dist_op_map;
  for (const auto& op : dist_ops) {
    dist_op_map[op.dist_name] = op;
  }

  std::unordered_map<std::string, std::pair<size_t, size_t>> ret;

  for (const auto& node : g->getNodes()) {
    if (contains(dist_op_map, node.getName())) {
      const auto& dist_op = dist_op_map.at(node.getName());
      for (const auto& part_dims : dist_op.partition_dim) {
        assert(part_dims.first < node.getInputNames().size());
        ret[node.getInputNames().at(part_dims.first)] = part_dims;
      }
    }
  }

  return ret;
}
} // namespace rannc