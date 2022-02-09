//
// Created by Masahiro Tanaka on 2022/02/02.
//

#include "PartitionTensor.h"
#include <torch/TorchUtil.h>

namespace rannc {
std::vector<DistOp> dist_ops = {
    {"aten::linear", "rannc::linear_dist", {{1, 1}}}};

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

TensorPartioningGraphInfo replaceWithDistOp(
    const std::shared_ptr<IRGraph>& g, const std::vector<int>& ranks) {
  std::vector<IRNode> new_nodes;
  std::unordered_map<std::string, IRValue> new_values;
  std::unordered_map<std::string, int> dist_ranks;
  std::unordered_map<std::string, std::vector<std::string>> rank_value_names;

  const std::unordered_map<std::string, IRValue>& vals = g->getValues();
  for (const auto& in_name : g->getInputNames()) {
    assert(contains(vals, in_name));
    new_values[in_name] = vals.at(in_name);
  }

  const auto name_map = getDistOpNameMap();
  int ex_rank_arg_idx = 0;
  int ex_rank_list_arg_idx = 0;
  for (const auto& n : g->getNodes()) {
    if (contains(name_map, n.getName())) {
      std::vector<std::string> ranks_val_node_names;
      for (const int r : ranks) {
        std::stringstream ss;
        ss << "_ex_rank_" << ex_rank_arg_idx;
        std::string val_name = ss.str();
        IRNode const_node("prim::Constant", {}, {val_name});
        new_nodes.push_back(const_node);

        IRValue val(val_name, IRType::createScalarType(IRScalarType::INT));
        new_values[val_name] = val;
        ranks_val_node_names.push_back(val_name);

        dist_ranks[val_name] = r;
        ex_rank_arg_idx++;
      }
      std::stringstream ss;
      ss << "_ex_rank_list_" << ex_rank_list_arg_idx++;
      std::string int_list_val_name = ss.str();
      IRNode int_list_node(
          "prim::ListConstruct", ranks_val_node_names, {int_list_val_name});
      new_nodes.push_back(int_list_node);

      IRValue int_list_val(
          int_list_val_name, IRType::createListType(IRListType::INT));
      new_values[int_list_val_name] = int_list_val;

      std::vector<std::string> input_names = n.getInputNames();
      input_names.push_back(int_list_val_name);

      IRNode new_node{
          name_map.at(n.getName()), input_names, n.getOutputNames()};
      new_nodes.push_back(new_node);

      rank_value_names[new_node.getId()] = ranks_val_node_names;
    } else {
      new_nodes.emplace_back(
          n.getName(), n.getInputNames(), n.getOutputNames());
    }
    for (const auto& out_name : n.getOutputNames()) {
      new_values[out_name] = vals.at(out_name);
    }
  }

  std::shared_ptr<IRGraph> ret_graph = std::make_shared<IRGraph>(
      g->getName(), new_nodes, new_values, g->getInputNames(),
      g->getOutputNames());

  std::unordered_map<std::string, std::pair<size_t, size_t>> param_part =
      getDistParams(ret_graph);

  return TensorPartioningGraphInfo{
      ret_graph, ranks, param_part, dist_ranks, rank_value_names};
}

at::Tensor sliceParam(
    const std::string& name, const at::Tensor& param,
    const TensorPartioningGraphInfo& part_info, int my_rank) {
  assert(contains(part_info.param_partitions, name));

  const auto ranks = vectorToSet(part_info.ranks);
  assert(contains(ranks, my_rank));

  const auto& partition = part_info.param_partitions.at(name);
  size_t arg_idx = partition.first;
  size_t dim_idx = partition.second;

  if (param.size(dim_idx) % part_info.ranks.size() != 0) {
    std::stringstream ss;
    ss << "Dimension " << dim_idx << " of " << name
       << " is not divisible. shape=" << join_as_str(getTensorDim(param));
    throw std::runtime_error(ss.str());
  }

  size_t segment_size = param.size(dim_idx) / part_info.ranks.size();
  int local_rank = getLocalRank(vectorToSet(part_info.ranks), my_rank);

  torch::NoGradGuard no_grad;
  return param.slice(
      dim_idx, segment_size * local_rank, segment_size * (local_rank + 1));
}

} // namespace rannc