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

std::pair<std::shared_ptr<IRGraph>, std::unordered_map<std::string, int>>
replaceWithDistOp(
    const std::shared_ptr<IRGraph>& g, const std::vector<int>& ranks) {
  std::vector<IRNode> new_nodes;
  std::unordered_map<std::string, IRValue> new_values;
  std::unordered_map<std::string, int> dist_ranks;

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

      new_nodes.emplace_back(
          name_map.at(n.getName()), input_names, n.getOutputNames());
    } else {
      new_nodes.emplace_back(
          n.getName(), n.getInputNames(), n.getOutputNames());
    }
    for (const auto& out_name : n.getOutputNames()) {
      new_values[out_name] = vals.at(out_name);
    }
  }

  const auto ret_graph = std::make_shared<IRGraph>(
      g->getName(), new_nodes, new_values, g->getInputNames(),
      g->getOutputNames());

  return {
      std::make_shared<IRGraph>(
          g->getName(), new_nodes, new_values, g->getInputNames(),
          g->getOutputNames()),
      dist_ranks};
}
} // namespace rannc