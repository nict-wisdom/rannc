//
// Created by Masahiro Tanaka on 2021/12/27.
//

#ifndef PYRANNC_CPG_H
#define PYRANNC_CPG_H

#include <ostream>
#include "graph/ir.h"

namespace rannc {

struct CPGNode {
  CPGNode(std::string name, int split_dim, std::string op_graph)
      : name_(std::move(name)),
        split_dim_(split_dim),
        op_graph_(std::move(op_graph)) {}

  std::string name_;
  int split_dim_;
  std::string op_graph_;

  friend std::ostream& operator<<(std::ostream& os, const CPGNode& node) {
    os << "{name_=" << node.name_ << ",split_dim_=" << node.split_dim_
       << ",op_graph=" << node.op_graph_ << "}";
    return os;
  }
};

class CPGVar {
 public:
  explicit CPGVar(IRValue value, std::string op_graph);

  const std::string& getOpGraph() const {
    return op_graph_;
  }

  const IRValue& getValue() const {
    return value_;
  }

  const std::vector<CPGNode>& getNodes() const {
    return nodes_;
  }

  friend std::ostream& operator<<(std::ostream& os, const CPGVar& var) {
    os << "nodes:" << var.nodes_;
    return os;
  }

 private:
  IRValue value_;
  std::vector<CPGNode> nodes_;
  std::string op_graph_;
};

struct OpHyEdge {
  OpHyEdge(
      const std::vector<CPGNode>& in_nodes,
      const std::vector<CPGNode>& out_nodes)
      : in_nodes(in_nodes), out_nodes(out_nodes) {}

  std::vector<CPGNode> in_nodes;
  std::vector<CPGNode> out_nodes;

  friend std::ostream& operator<<(std::ostream& os, const OpHyEdge& edge) {
    os << "in_nodes:" << edge.in_nodes << " out_nodes:" << edge.out_nodes;
    return os;
  }
};

class OpCPG {
 public:
  OpCPG(
      std::string name, std::vector<CPGVar> in_vars,
      std::vector<CPGVar> out_vars, std::vector<OpHyEdge> edges)
      : name_(std::move(name)),
        in_vars_(std::move(in_vars)),
        out_vars_(std::move(out_vars)),
        edges_(std::move(edges)) {}

  const std::string& getName() const {
    return name_;
  }
  const std::vector<CPGVar>& getInVars() const {
    return in_vars_;
  }
  const std::vector<CPGVar>& getOutVars() const {
    return out_vars_;
  }
  const std::vector<OpHyEdge>& getEdges() const {
    return edges_;
  }

  friend std::ostream& operator<<(std::ostream& os, const OpCPG& cpg) {
    os << "in_vars_: " << cpg.in_vars_ << " out_vars_: " << cpg.out_vars_
       << " edges_: " << cpg.edges_;
    return os;
  }

 private:
  std::string name_;
  std::vector<CPGVar> in_vars_;
  std::vector<CPGVar> out_vars_;
  std::vector<OpHyEdge> edges_;
};

class CPG {};

void testCPG();

} // namespace rannc

#endif // PYRANNC_CPG_H
