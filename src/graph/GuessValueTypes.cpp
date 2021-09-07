//
// Created by Masahiro Tanaka on 2020/01/31.
//

#include "GuessValueTypes.h"
#include "Decomposition.h"

namespace rannc {

enum class ValueType { BATCH, PARAM, CONST, LOSS, UNKNOWN };

std::string toString(ValueType type) {
  switch (type) {
    case ValueType::BATCH:
      return "BATCH";
    case ValueType::PARAM:
      return "PARAM";
    case ValueType::CONST:
      return "CONST";
    case ValueType::LOSS:
      return "LOSS";
    case ValueType::UNKNOWN:
      return "UNKNOWN";
  }
  throw std::invalid_argument("Failed to convert ValueType to string.");
}

std::vector<ValueType> toValueTypes(
    const std::vector<Vertex>& vertices,
    std::unordered_map<std::string, ValueType>& types, const BGraph& bg) {
  std::vector<ValueType> ret;
  ret.reserve(vertices.size());
  for (const auto& v : vertices) {
    assert(contains(types, bg[v].id));
    ret.push_back(types.at(bg[v].id));
  }
  return ret;
}

std::vector<IRType> toIRTypes(
    const std::vector<Vertex>& vertices, const BGraph& bg) {
  std::vector<IRType> ret;
  ret.reserve(vertices.size());
  for (const auto& v : vertices) {
    ret.push_back(bg[v].value.getType());
  }
  return ret;
}

bool matchLossRule(
    const std::vector<ValueType>& src_types,
    const std::vector<IRType>& tgt_ir_types) {
  if (tgt_ir_types.size() != 1) {
    return false;
  }
  const auto& tgt_ir_type = tgt_ir_types.front();
  const IRBaseType tgt_base_type = tgt_ir_type.getBaseType();
  if (tgt_base_type != IRBaseType::TENSOR) {
    return false;
  }
  const auto dim = tgt_ir_type.getTensorDim();
  if (!dim.empty()) { // must be a scalar
    return false;
  }

  // count batch
  int batch_count = 0;
  for (const auto& t : src_types) {
    if (t == ValueType::BATCH) {
      batch_count++;
    }
  }

  // count loss
  int loss_count = 0;
  for (const auto& t : src_types) {
    if (t == ValueType::LOSS) {
      loss_count++;
    }
  }

  if (loss_count > 0 && batch_count == 0) {
    return true;
  }

  // the number of batch values must be exactly two
  return batch_count == 2;
}

bool matchBatchRule(
    const std::vector<ValueType>& src_types,
    const std::vector<IRType>& tgt_ir_types) {
  // count batch
  int batch_count = 0;
  for (const auto& t : src_types) {
    if (t == ValueType::BATCH) {
      batch_count++;
    }
  }
  if (batch_count == 0) {
    return false;
  }

  int out_batch_count = 0;
  for (const auto& ir_type : tgt_ir_types) {
    if (isTensorOrTensorList(ir_type)) {
      out_batch_count++;
    }
  }

  return out_batch_count > 0;
}

bool matchParamRule(const std::vector<ValueType>& src_types) {
  // count
  int param_count = 0;
  for (const auto& t : src_types) {
    if (t == ValueType::PARAM) {
      param_count++;
    }
  }
  return param_count == 1;
}

bool matchConstRule(const std::vector<ValueType>& src_types) {
  // count
  int const_count = 0;
  for (const auto& t : src_types) {
    if (t == ValueType::CONST) {
      const_count++;
    }
  }
  // the number of batch values must be exactly two
  return const_count >= 0;
}

std::shared_ptr<IRGraph> guessValueTypes(const std::shared_ptr<IRGraph>& g) {
  const auto& bg = toBGL(g);

  std::unordered_map<std::string, ValueType> types;
  std::unordered_map<std::string, IRValue> values = g->getValues();
  std::unordered_map<std::string, IRNode> node_map;
  for (const auto& v : all_nodes_topo<Vertex, BGraph>(bg)) {
    if (bg[v].type == NODE) {
      node_map[bg[v].id] = bg[v].node;
    }
  }

  for (const auto& in : graph_regular_input_nodes<Vertex, BGraph>(bg)) {
    IRValue& in_val = values.at(bg[in].name);
    const IRType& type = in_val.getType();
    assert(type.getBaseType() == IRBaseType::TENSOR);
    in_val.setBatch(true);

    types[bg[in].id] = ValueType::BATCH;
  }

  for (const auto& pin : graph_param_input_nodes<Vertex, BGraph>(bg)) {
    IRValue& param_val = values.at(bg[pin].name);
    types[bg[pin].id] = ValueType::PARAM;
    param_val.setParam(true);
  }

  for (const auto& in : graph_input_nodes<Vertex, BGraph>(bg)) {
    if (!contains(types, bg[in].id)) {
      types[bg[in].id] = ValueType::CONST;
    }
  }

  std::vector<Vertex> topo_vertices, rev_topo_vertices;
  boost::topological_sort(bg, std::back_inserter(rev_topo_vertices));
  boost::for_each(
      rev_topo_vertices | boost::adaptors::reversed,
      [&topo_vertices](Vertex v) { topo_vertices.push_back(v); });

  for (auto& v : topo_vertices) {
    const auto& vi = bg[v];
    if (contains(types, vi.id)) {
      continue;
    }

    const auto src_types = toValueTypes(source_nodes(v, bg), types, bg);
    const auto tgt_ir_types = toIRTypes(target_nodes(v, bg), bg);

    if (bg[v].type == VALUE) {
      assert(src_types.size() == 1);
      const auto& src_type = src_types.front();
      types[bg[v].id] = src_type;
      switch (src_type) {
        case ValueType::BATCH: {
          IRValue& ir_val = values.at(bg[v].name);
          ir_val.setBatch(true);
          break;
        }
        case ValueType::PARAM:
          break;
        case ValueType::CONST:
          break;
        case ValueType::LOSS: {
          IRValue& ir_val = values.at(bg[v].name);
          ir_val.setLoss(true);
          break;
        }
        case ValueType::UNKNOWN:
          break;
      }
    } else {
      if (matchLossRule(src_types, tgt_ir_types)) {
        //                    spdlog::info("match {} LOSS", bg[v].name);
        types[bg[v].id] = ValueType::LOSS;
        IRNode& ir_node = node_map.at(bg[v].id);
        ir_node.setCriterion(true);
      } else if (matchBatchRule(src_types, tgt_ir_types)) {
        //                    spdlog::info("match {} BATCH", bg[v].name);
        types[bg[v].id] = ValueType::BATCH;
        assert(contains(node_map, bg[v].id));
        IRNode& ir_node = node_map.at(bg[v].id);
        ir_node.setBatch(true);
      } else if (matchParamRule(src_types)) {
        //                    spdlog::info("match {} PARAM", bg[v].name);
        types[bg[v].id] = ValueType::PARAM;
      } else if (matchConstRule(src_types)) {
        //                    spdlog::info("match {} CONST", bg[v].name);
        types[bg[v].id] = ValueType::CONST;
      } else {
        std::vector<std::string> sources;
        for (const auto& s : source_nodes(v, bg)) {
          sources.push_back(bg[s].name);
        }
        std::vector<std::string> targets;
        for (const auto& t : target_nodes(v, bg)) {
          targets.push_back(bg[t].name);
        }
        std::stringstream ss;
        ss << "Failed to determine value type: " << bg[v].name
           << " src=" << join_as_str(sources)
           << " tgt=" << join_as_str(targets);
        throw std::runtime_error(ss.str());
      }
    }
  }

  std::vector<IRNode> nodes;
  for (const auto& v : all_nodes_topo<Vertex, BGraph>(bg)) {
    if (bg[v].type == NODE) {
      assert(contains(node_map, bg[v].id));
      nodes.push_back(node_map.at(bg[v].id));
    }
  }

  const auto typed_graph = std::make_shared<IRGraph>(
      g->getName(), nodes, values, g->getInputNames(), g->getOutputNames());
  //        std::stringstream ss;
  //        ss << "typed graph: " << *typed_graph;
  //        spdlog::info(ss.str());
  //
  //        const auto& typed_bg = toBGL(typed_graph);
  //        std::ofstream file("typed_graph.dot");
  //        boost::write_graphviz(file, typed_bg,
  //        vertex_rank_label_writer<BGraph>(typed_bg));

  return typed_graph;
}
} // namespace rannc