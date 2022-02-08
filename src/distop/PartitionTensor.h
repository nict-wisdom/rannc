//
// Created by Masahiro Tanaka on 2022/02/02.
//

#ifndef PYRANNC_PARTITIONTENSOR_H
#define PYRANNC_PARTITIONTENSOR_H

#include <graph/ir.h>

namespace rannc {

struct DistOp {
  std::string original_name;
  std::string dist_name;

  // (arg index, partitioning dim)
  std::vector<std::pair<size_t, size_t>> partition_dim;
};

std::unordered_map<std::string, std::string> getDistOpNameMap();

std::unordered_map<std::string, std::pair<size_t, size_t>> getDistParams(
    const std::shared_ptr<IRGraph>& g);

struct TensorPartioningGraphInfo {
  std::shared_ptr<IRGraph> graph;
  // param name -> (arg index, dim index)
  std::unordered_map<std::string, std::pair<size_t, size_t>> param_partitions;
  // value name -> rank
  std::unordered_map<std::string, int> rank_values;

  MSGPACK_DEFINE(graph, param_partitions, rank_values);
};

TensorPartioningGraphInfo replaceWithDistOp(
    const std::shared_ptr<IRGraph>& g, const std::vector<int>& ranks);
} // namespace rannc

#endif // PYRANNC_PARTITIONTENSOR_H
