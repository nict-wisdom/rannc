//
// Created by Masahiro Tanaka on 2020/04/11.
//

#ifndef PYRANNC_MLGRAPH_H
#define PYRANNC_MLGRAPH_H

#include <comp/GraphProfiler.h>
#include "Decomposition.h"
#include "ir.h"

namespace rannc {

struct MLBaseEdge {
  std::string src_id;
  std::string tgt_id;
  IRValue value;

  friend std::ostream& operator<<(std::ostream& os, const MLBaseEdge& edge) {
    os << "src_node: " << edge.src_id << " tgt_node: " << edge.tgt_id
       << " value: " << edge.value.getName();
    return os;
  }

  MSGPACK_DEFINE(src_id, tgt_id, value);
};

struct MLEdge {
  std::string src_id;
  std::string tgt_id;
  std::vector<MLEdge> sub_edges; // level-zero edges has no subedges
  std::vector<MLBaseEdge> base_edges;

  friend std::ostream& operator<<(std::ostream& os, const MLEdge& edge) {
    os << "src_node: " << edge.src_id << " tgt_node: " << edge.tgt_id
       << " sub_edges: [" << edge.sub_edges << "] base_edges: ["
       << edge.base_edges << "]";
    return os;
  }

  MSGPACK_DEFINE(src_id, tgt_id, sub_edges, base_edges);
};

struct MLGraph;

struct MLNode {
  std::string id;
  std::shared_ptr<IRGraph> graph;
  std::vector<MLNode> sub_nodes;
  std::vector<MLEdge> sub_edges;

  bool is_atomic() const {
    return sub_nodes.empty();
  }

  int getLevel() const {
    if (sub_nodes.empty()) {
      return 0;
    }
    return sub_nodes.front().getLevel() + 1;
  }

  int getSize() const {
    if (sub_nodes.empty()) {
      return 1;
    }
    int sum = 0;
    for (const auto& sn : sub_nodes) {
      sum += sn.getSize();
    }
    return sum;
  }

  MSGPACK_DEFINE(id, graph, sub_nodes, sub_edges);
};

struct MLGraph {
  std::vector<MLNode> nodes;
  std::vector<MLEdge> edges;

  int getLevel() const {
    if (nodes.empty()) {
      return -1;
    }
    return nodes.front().getLevel();
  }

  int getSize() const {
    if (nodes.empty()) {
      return 0;
    }
    int sum = 0;
    for (const auto& n : nodes) {
      sum += n.getSize();
    }
    return sum;
  }

  MSGPACK_DEFINE(nodes, edges);
};

using MLEdgeKey = std::pair<std::string, std::string>;
using MLEdgeKeyHash = StringPairHash;
using MLEdgeMap = std::unordered_map<MLEdgeKey, MLEdge, MLEdgeKeyHash>;
using MLEdgeKeySet = std::unordered_set<MLEdgeKey, MLEdgeKeyHash>;
using MLEdgeVectorMap =
    std::unordered_map<MLEdgeKey, std::vector<MLEdge>, MLEdgeKeyHash>;

using GraphMergeKey = std::pair<int, int>;
using GraphMergeKeyHash = IntPairHash;
using GraphMergeCache = std::unordered_map<
    GraphMergeKey, std::shared_ptr<MLNode>, GraphMergeKeyHash>;

std::string dumpMLGraph(const MLGraph& g);
std::string dumpMLNode(const MLNode& n, int indent = 0);
std::string dumpMLEdge(const std::vector<MLEdge>& edges, int indent = 0);

typedef boost::adjacency_list<
    boost::listS, boost::vecS, boost::bidirectionalS, MLNode, MLEdge>
    MLBGraph;
typedef boost::graph_traits<MLBGraph>::vertex_descriptor MLVertex;

MLGraph convert(const Partition& partition);
bool isConvex(const MLVertex& n1, const MLVertex& n2, MLBGraph& graph);
MLNode liftUp(const MLNode& n);
std::vector<IRValue> getCutValues(const MLNode& n1, const MLNode& n2);
std::vector<MLEdge> mergeEdgesNoCopy(
    std::vector<MLEdge>&& edges,
    const std::unordered_map<std::string, std::string>& name_map);
std::vector<MLEdge> mergeEdges(
    std::vector<MLEdge> edges,
    const std::unordered_map<std::string, std::string>& name_map);

std::unordered_set<std::string> getRequiredInputs(
    const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2,
    const std::vector<std::shared_ptr<IRGraph>>& all_graphs);
std::unordered_set<std::string> getRequiredInputs(
    const MLNode& n1, const MLNode& n2, const std::vector<MLNode>& all_nodes);
size_t calcEdgeSize(const MLEdge& e);
size_t sumEdgeSizes(const std::vector<MLEdge>& edges);

size_t countRefTgtInSubgraph(const MLNode& node, const MLNode& sub_node);
size_t countRefSrcInSubgraph(const MLNode& node, const MLNode& sub_node);
size_t countRefInEdgesSrc(const MLNode& node, const std::vector<MLEdge>& edges);
size_t countRefInEdgesTgt(const MLNode& node, const std::vector<MLEdge>& edges);
bool containsSubedge(const MLEdge& edge, const MLEdge& sub_edge);
bool containsSubgraph(const MLNode& n, const std::string& subgraph_id);
std::unordered_set<std::string> getPreservedOutputs(
    const std::string& src_node_id, const std::string& moved_graph_id,
    const std::vector<MLNode>& all_nodes);

bool verify(const MLNode& n);
bool verify(const MLGraph& g);
MLGraph toLowerLevel(const MLGraph& ml_graph);
MLGraph sortMLGraph(const MLGraph& graph);
MLNode merge(
    const MLNode& n1, const MLNode& n2, const std::vector<MLNode>& all_nodes,
    const std::vector<MLEdge>& edges);
MLNode merge(
    const MLNode& n1, const MLNode& n2,
    const std::vector<std::shared_ptr<MLNode>>& all_nodes,
    const std::vector<MLEdge>& edges);
std::shared_ptr<IRGraph> merge(
    const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2,
    const std::unordered_set<std::string>& required_inputs);

MLBGraph toBGL(const MLGraph& ml_graph);
MLBGraph toBGL(const MLNode& node);
MLBGraph toBGL(
    const std::vector<MLNode>& nodes, const std::vector<MLEdge>& edges);

struct MLGraphNodeLabelWriter {
  MLGraphNodeLabelWriter(const MLBGraph& g) : graph_(g) {}

  void operator()(std::ostream& out, const MLVertex& vertex) const {
    write(out, vertex);
  }

 private:
  void write(std::ostream& out, const MLVertex& vertex) const {
    const auto& n = graph_[vertex];
    std::vector<std::string> sub_node_ids;
    for (const auto& sn : n.sub_nodes) {
      sub_node_ids.push_back(sn.id);
    }

    std::vector<std::string> sub_edges_str;
    for (const auto& se : n.sub_edges) {
      sub_edges_str.push_back(se.src_id + "->" + se.tgt_id);
    }

    std::stringstream ss;
    ss << n.id << std::endl
       << "size=" << n.getSize() << std::endl
       << "subnodes=" << join_as_str(sub_node_ids) << std::endl
       << "subedges=" << join_as_str(sub_edges_str);

    out << "[label=\"" << ss.str() << "\"]";
  }
  const MLBGraph& graph_;
};

struct MLGraphEdgeLabelWriter {
  MLGraphEdgeLabelWriter(const MLBGraph& g) : graph_(g) {}

  void operator()(
      std::ostream& out,
      const boost::graph_traits<MLBGraph>::edge_descriptor& edge) const {
    write(out, edge);
  }

 private:
  void write(
      std::ostream& out,
      const boost::graph_traits<MLBGraph>::edge_descriptor& edge) const {
    std::vector<std::string> edges_str;
    MLEdge e = graph_[edge];
    for (const auto& se : e.sub_edges) {
      edges_str.push_back(se.src_id + "->" + se.tgt_id);
    }
    out << "[label=\"" << join_as_str(edges_str) << "\"]";
  }
  const MLBGraph& graph_;
};
} // namespace rannc

#endif // PYRANNC_MLGRAPH_H
