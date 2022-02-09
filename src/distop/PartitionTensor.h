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
  std::vector<int> ranks;
  // param name -> (arg index, dim index)
  std::unordered_map<std::string, std::pair<size_t, size_t>> param_partitions;
  // value name -> rank
  std::unordered_map<std::string, int> rank_values;
  // node id -> [val names]
  std::unordered_map<std::string, std::vector<std::string>> rank_value_names;

  MSGPACK_DEFINE(graph, ranks, param_partitions, rank_values, rank_value_names);
};

TensorPartioningGraphInfo replaceWithDistOp(
    const std::shared_ptr<IRGraph>& g, const std::vector<int>& ranks);

at::Tensor sliceParam(
    const std::string& name, const at::Tensor& param,
    const TensorPartioningGraphInfo& part_info, int my_rank);
} // namespace rannc

#endif // PYRANNC_PARTITIONTENSOR_H
