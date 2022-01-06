//
// Created by Masahiro Tanaka on 2021/12/27.
//

#include "CPG.h"

namespace {
std::vector<rannc::CPGVar> toCPGVars(
    const rannc::IRGraph& ir_g, const std::vector<std::string>& v_names) {
  std::vector<rannc::CPGVar> vars;
  for (const auto& in : v_names) {
    const rannc::IRValue& v = ir_g.getValue(in);
    vars.emplace_back(v, ir_g.getName());
  }
  return vars;
}

} // namespace

namespace rannc {

CPGVar::CPGVar(IRValue value, std::string op_graph)
    : value_(std::move(value)), op_graph_(std::move(op_graph)) {
  const IRType& type = value_.getType();
  assert(type.getBaseType() == IRBaseType::TENSOR);

  int start_dim = value.isBatch() ? 1 : 0;
  for (int i = start_dim; i < type.getTensorDim().size(); i++) {
    nodes_.emplace_back(value_.getName(), i, op_graph_);
  }
}

std::vector<std::vector<CPGNode>> doCartesianProductDim(
    std::vector<CPGVar> rest_vars, const std::vector<CPGNode>& combination) {
  if (rest_vars.empty()) {
    return {combination};
  }

  CPGVar last_var = rest_vars.at(rest_vars.size() - 1);
  rest_vars.pop_back();

  std::vector<std::vector<CPGNode>> results;
  for (const auto& n : last_var.getNodes()) {
    std::vector<CPGNode> new_comb = combination;
    new_comb.emplace_back(
        last_var.getValue().getName(), n.split_dim_, last_var.getOpGraph());
    for (const auto& c : doCartesianProductDim(rest_vars, new_comb)) {
      results.push_back(c);
    }
  }
  return results;
}

std::vector<std::vector<CPGNode>> cartesianProductDim(
    const std::vector<CPGVar>& vars) {
  return doCartesianProductDim(vars, {});
}

OpCPG generateOpCPG(const IRGraph& ir_g) {
  IRNode op_node;
  for (const auto& node : ir_g.getNodes()) {
    //    spdlog::info("node {} batch={}", node.getName(), node.isBatch());
    if (node.isBatch()) {
      op_node = node;
    }
  }
  spdlog::info("target op {}", op_node.getName());

  std::vector<CPGVar> in_vars = toCPGVars(ir_g, ir_g.getInputNames());
  std::vector<CPGVar> out_vars = toCPGVars(ir_g, ir_g.getOutputNames());

  std::unordered_set<std::string> in_names, out_names;
  for (const auto& v : in_vars) {
    in_names.insert(v.getValue().getName());
  }
  for (const auto& v : out_vars) {
    out_names.insert(v.getValue().getName());
  }

  assert(out_vars.size() == 1);
  std::vector<CPGVar> target_vals = in_vars;
  target_vals.push_back(out_vars.front());
  const std::vector<std::vector<CPGNode>> comb =
      cartesianProductDim(target_vals);

  std::vector<OpHyEdge> edges;
  for (const auto& it : comb) {
    std::stringstream ss;
    ss << "[";
    for (const CPGNode& n : it) {
      ss << "(" << n.name_ << "," << n.split_dim_ << "," << n.op_graph_ << ")";
    }
    ss << "]";
    spdlog::info("comb {}", ss.str());

    std::vector<CPGNode> in_nodes;
    std::vector<CPGNode> out_nodes;

    for (const CPGNode& n : it) {
      if (contains(in_names, n.name_)) {
        in_nodes.push_back(n);
      } else if (contains(out_names, n.name_)) {
        out_nodes.push_back(n);
      } else {
        throw std::runtime_error("name not found");
      }
    }
    edges.emplace_back(in_nodes, out_nodes);
  }

  return {ir_g.getName(), in_vars, out_vars, edges};
}

CPG createCPG(const std::vector<OpCPG>& op_cpgs) {
  // value_name -> op_graph
  std::unordered_map<std::string, std::string> val_out_graphs;
  for (const auto& op_cpg : op_cpgs) {
    for (const auto& v : op_cpg.getOutVars()) {
      val_out_graphs[v.getValue().getName()] = op_cpg.getName();
    }
  }

  // value_name -> (src_op_graph, tgt_op_graph)
  std::unordered_map<std::string, std::pair<std::string, std::string>> io_pairs;
  for (const auto& op_cpg : op_cpgs) {
    for (const auto& v : op_cpg.getInVars()) {
      const auto& val_name = v.getValue().getName();
      if (contains(val_out_graphs, val_name)) {
        io_pairs[val_name] = {val_out_graphs.at(val_name), v.getOpGraph()};
      }
    }
  }

  //  std::vector<CPGNode> out_nodes;
  //
  //  std::vector<CPGNode> nodes;
  //  for (const auto& op_cpg: op_cpgs) {
  //    for (const auto& v: op_cpg.getInVars()) {
  //      nodes.push_back(v);
  //    }
  //    for (const auto& v: op_cpg.getOutVars()) {
  //      nodes.push_back(v);
  //    }
  //  }

  std::vector<CPGNode> out_nodes;

  return CPG();
}

IRGraph testLoadGraph(const std::string& file) {
  std::ifstream input(file, std::ios::in | std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Failed to open file: " + file);
  }

  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return deserialize<IRGraph>(buffer);
}

void testCPG() {
  std::vector<std::string> files;
  files.emplace_back("/tmp/graph_mod_y9757acx1eve7nmo_p0");
  files.emplace_back("/tmp/graph_mod_y9757acx1eve7nmo_p1");
  files.emplace_back("/tmp/graph_mod_y9757acx1eve7nmo_p2");
  files.emplace_back("/tmp/graph_mod_y9757acx1eve7nmo_p3");

  std::vector<IRGraph> graphs;
  for (const auto& f : files) {
    graphs.push_back(testLoadGraph(f));
  }

  for (const auto& g : graphs) {
    spdlog::info("testCPG {}", toString(g));
    OpCPG opg = generateOpCPG(g);
    spdlog::info("g={}", toString(opg));
  }
}
} // namespace rannc