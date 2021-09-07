//
// Created by Masahiro Tanaka on 2020/04/11.
//

#include "MLGraph.h"

namespace rannc {
MLGraph convert(const Partition& partition) {
  std::vector<MLNode> nodes;
  std::vector<MLEdge> edges;

  std::unordered_map<std::string, std::string> id_map;

  for (const auto& it : partition.subgraphs) {
    const auto& sg = it.second;
    MLNode n;
    n.id = "ML_" + it.first;
    n.graph = std::make_shared<IRGraph>(n.id, *sg);
    n.sub_nodes = {};
    n.sub_edges = {};
    nodes.push_back(n);

    id_map[it.first] = n.id;
  }

  for (const auto& con : partition.connections) {
    // Graph inputs/outputs are ignored
    if (contains(id_map, con.src) && contains(id_map, con.dest)) {
      MLEdge e;
      e.src_id = id_map.at(con.src);
      e.tgt_id = id_map.at(con.dest);

      MLBaseEdge be{con.src, con.dest, partition.graph->getValue(con.value)};
      e.base_edges = {be};
      edges.push_back(e);
    }
  }

  return MLGraph{nodes, edges};
}

std::vector<MLNode> adj_ml_nodes(
    const MLNode& n, const MLGraph& graph,
    const std::vector<std::string>& ids) {
  std::unordered_map<std::string, MLNode> node_map;
  for (const auto& gn : graph.nodes) {
    node_map[gn.id] = gn;
  }

  std::vector<MLNode> ret;
  for (const auto& id : ids) {
    if (!contains(node_map, id)) {
      std::stringstream ss;
      ss << "Graph does not have node with id " << id
         << ". graph: " << dumpMLGraph(graph);
      throw std::invalid_argument(ss.str());
    }

    ret.push_back(node_map.at(id));
  }
  return ret;
}

std::vector<MLNode> src_ml_nodes(const MLNode& n, const MLGraph& graph) {
  std::vector<std::string> src_ids;
  for (const auto& e : graph.edges) {
    if (e.tgt_id == n.id) {
      src_ids.push_back(e.src_id);
    }
  }
  return adj_ml_nodes(n, graph, src_ids);
}

std::vector<MLNode> tgt_ml_nodes(const MLNode& n, const MLGraph& graph) {
  std::vector<std::string> target_ids;
  for (const auto& e : graph.edges) {
    if (e.src_id == n.id) {
      target_ids.push_back(e.tgt_id);
    }
  }
  return adj_ml_nodes(n, graph, target_ids);
}

std::vector<MLEdge> ml_edge(
    const MLNode& src, const MLNode& tgt, const std::vector<MLEdge>& edges) {
  for (const auto& e : edges) {
    if (e.src_id == src.id && e.tgt_id == tgt.id) {
      return {e};
    }
  }
  return {};
}

std::unordered_set<std::string> getRequiredInputs(
    const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2,
    const std::vector<std::shared_ptr<IRGraph>>& all_graphs) {
  std::unordered_set<std::string> required_inputs;
  for (const auto& g : all_graphs) {
    if (g->getName() != g1->getName() && g->getName() != g2->getName()) {
      for (const auto& in : g->getInputNames()) {
        required_inputs.insert(in);
      }
    }
  }
  return required_inputs;
}

std::unordered_set<std::string> getRequiredInputs(
    const MLNode& n1, const MLNode& n2, const std::vector<MLNode>& all_nodes) {
  std::vector<std::shared_ptr<IRGraph>> all_graphs;
  all_graphs.reserve(all_nodes.size());

  for (const auto& n : all_nodes) {
    all_graphs.push_back(n.graph);
  }
  return getRequiredInputs(n1.graph, n2.graph, all_graphs);
}

std::unordered_set<std::string> getPreservedOutputs(
    const std::string& src_node_id, const std::string& moved_graph_id,
    const std::vector<MLNode>& all_nodes) {
  std::unordered_set<std::string> preserved_outputs;
  for (const auto& n : all_nodes) {
    if (n.id != src_node_id) {
      for (const auto& sn : n.sub_nodes) {
        if (sn.graph->getName() != moved_graph_id) {
          const auto& g = sn.graph;
          for (const auto& in : g->getInputNames()) {
            preserved_outputs.insert(in);
          }
        }
      }
    }
  }
  return preserved_outputs;
}

size_t countRefTgtInSubgraph(const MLNode& node, const MLNode& sub_node) {
  MLGraph g{node.sub_nodes, node.sub_edges};
  return tgt_ml_nodes(sub_node, g).size();
}
size_t countRefSrcInSubgraph(const MLNode& node, const MLNode& sub_node) {
  MLGraph g{node.sub_nodes, node.sub_edges};
  return src_ml_nodes(sub_node, g).size();
}

size_t countRefInEdgesSrc(
    const MLNode& node, const std::vector<MLEdge>& edges) {
  size_t count = 0;
  for (const auto& e : edges) {
    if (e.src_id == node.id) {
      count++;
    }
  }
  return count;
}

size_t countRefInEdgesTgt(
    const MLNode& node, const std::vector<MLEdge>& edges) {
  size_t count = 0;
  for (const auto& e : edges) {
    if (e.tgt_id == node.id) {
      count++;
    }
  }
  return count;
}

std::vector<IRValue> getCutValues(
    const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2) {
  std::vector<std::string> cut_names;

  for (const auto& o1 : g1->getOutputNames()) {
    if (contains(g2->getInputNames(), o1)) {
      cut_names.push_back(o1);
    }
  }
  for (const auto& o2 : g2->getOutputNames()) {
    if (contains(g1->getInputNames(), o2)) {
      cut_names.push_back(o2);
    }
  }
  //        spdlog::info("Cuts between {} and {}: {}", g1->getName(),
  //        g2->getName(), join_as_str(cut_names));

  std::vector<IRValue> cut_values;
  for (const auto& n : cut_names) {
    cut_values.push_back(g1->getValue(n));
  }
  return cut_values;
}

std::vector<IRValue> getCutValues(const MLNode& n1, const MLNode& n2) {
  return getCutValues(n1.graph, n2.graph);
}

size_t calcEdgeSize(const MLEdge& e) {
  size_t sum = 0;
  if (e.sub_edges.empty()) {
    for (const auto& be : e.base_edges) {
      sum += be.value.getSizeInByte();
    }
    return sum;
  }
  for (const auto& se : e.sub_edges) {
    sum += calcEdgeSize(se);
  }
  return sum;
}

size_t sumEdgeSizes(const std::vector<MLEdge>& edges) {
  size_t sum = 0;
  for (const auto& e : edges) {
    sum += calcEdgeSize(e);
  }
  return sum;
}

bool outputAvailable(
    const IRNode& n, const std::unordered_set<std::string>& avail_vals) {
  bool available = false;
  for (const auto& o : n.getOutputNames()) {
    if (contains(avail_vals, o)) {
      available = true;
    }
  }
  return available;
}

std::shared_ptr<IRGraph> merge(
    const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2,
    const std::unordered_set<std::string>& required_inputs) {
  std::vector<std::string> outputs;

  for (const auto& o1 : g1->getOutputNames()) {
    if (!contains(g2->getInputNames(), o1) || contains(required_inputs, o1)) {
      outputs.push_back(o1);
    }
  }
  for (const auto& o2 : g2->getOutputNames()) {
    outputs.push_back(o2);
  }

  std::vector<std::string> non_param_inputs;
  for (const auto& i1 : (getNonParamInputNames(g1))) {
    non_param_inputs.push_back(i1);
  }
  for (const auto& i2 : (getNonParamInputNames(g2))) {
    if (!contains(g1->getOutputNames(), i2) &&
        !contains(non_param_inputs, i2)) {
      non_param_inputs.push_back(i2);
    }
  }

  std::vector<std::string> param_inputs;
  for (const auto& i1 : (getParamInputNames(g1))) {
    if (!contains(param_inputs, i1)) {
      param_inputs.push_back(i1);
    }
  }
  for (const auto& i2 : (getParamInputNames(g2))) {
    if (!contains(param_inputs, i2)) {
      param_inputs.push_back(i2);
    }
  }
  std::vector<std::string> all_inputs = addAll(non_param_inputs, param_inputs);

  std::vector<IRNode> nodes;
  std::unordered_set<std::string> node_out_names =
      vectorToSet(non_param_inputs);
  for (const auto& n : g1->getNodes()) {
    if (!outputAvailable(n, node_out_names)) {
      for (const auto& o : n.getOutputNames()) {
        node_out_names.insert(o);
      }
      nodes.push_back(n);
    }
  }
  for (const auto& n : g2->getNodes()) {
    if (!outputAvailable(n, node_out_names)) {
      for (const auto& o : n.getOutputNames()) {
        node_out_names.insert(o);
      }
      nodes.push_back(n);
    }
  }

  std::unordered_map<std::string, IRValue> merged_values;
  std::unordered_set<std::string> value_names;
  for (const auto& v : g1->getValues()) {
    merged_values[v.first] = v.second;
    value_names.insert(v.first);
  }
  for (const auto& v : g2->getValues()) {
    if (!contains(value_names, v.first)) {
      merged_values[v.first] = v.second;
      value_names.insert(v.first);
    }
  }
  return std::make_shared<IRGraph>(
      generateName("ML_"), nodes, merged_values, all_inputs, outputs);
}

bool ensureOutputsExist(const std::shared_ptr<IRGraph>& g) {
  const auto& values = g->getValues();

  for (const auto& node : g->getNodes()) {
    for (const auto& out : node.getOutputNames()) {
      if (!contains(values, out)) {
        spdlog::error(
            "{} does not have output of {}: {}", g->getName(), node.getName(),
            out);
        return false;
      }
      if (contains(g->getInputNames(), out)) {
        spdlog::error(
            "{} has {} as input, but the value is an output of {}",
            g->getName(), out, node.getName());
        return false;
      }
    }
  }
  return true;
}

// This assumes the direction n1 -> n2
MLNode merge(
    const MLNode& n1, const MLNode& n2,
    const std::unordered_set<std::string>& required_inputs,
    const std::vector<MLEdge>& edges) {
  MLNode n;
  n.graph = merge(n1.graph, n2.graph, required_inputs);
  n.graph = removeUnusedNodes(n.graph);

  //        if (!ensureOutputsExist(n.graph)
  //            || !verifyNoDuplicatedOutputs(n.graph)
  //            || !verifyNodeInputs(n.graph)
  //            || !noUnusedValue(n.graph, true)) {
  //            std::stringstream ss;
  //            ss << "Verification failed on graph merge result."
  //               << " n1=" << toString(*n1.graph)
  //               << " n2=" << toString(*n2.graph)
  //               << " required_inputs=" << join_as_str(required_inputs)
  //               << " merged=" << toString(*n.graph);
  //            throw std::runtime_error(ss.str());
  //        }

  n.id = n.graph->getName();
  n.sub_nodes = {n1, n2};
  n.sub_edges = ml_edge(n1, n2, edges);
  return n;
}

// This assumes the direction n1 -> n2
MLNode merge(
    const MLNode& n1, const MLNode& n2, const std::vector<MLNode>& all_nodes,
    const std::vector<MLEdge>& edges) {
  return merge(n1, n2, getRequiredInputs(n1, n2, all_nodes), edges);
}

MLNode merge(
    const MLNode& n1, const MLNode& n2,
    const std::vector<std::shared_ptr<MLNode>>& all_nodes,
    const std::vector<MLEdge>& edges) {
  std::vector<std::shared_ptr<IRGraph>> graphs;
  graphs.reserve(all_nodes.size());
  for (const auto& n : all_nodes) {
    graphs.push_back(n->graph);
  }

  return merge(n1, n2, getRequiredInputs(n1.graph, n2.graph, graphs), edges);
}

// lift n up to the next level without merging
MLNode liftUp(const MLNode& n) {
  MLNode n_up;
  n_up.graph = std::make_shared<IRGraph>(generateName("ML_"), *n.graph);
  n_up.id = n_up.graph->getName();
  n_up.sub_nodes = {n};

  return n_up;
}

std::vector<MLEdge> mergeEdgesNoCopy(
    std::vector<MLEdge>&& edges,
    const std::unordered_map<std::string, std::string>& name_map) {
  MLEdgeVectorMap unmerged_edges;
  for (auto& e : edges) {
    assert(contains(name_map, e.src_id));
    assert(contains(name_map, e.tgt_id));

    const std::string& new_src = name_map.at(e.src_id);
    const std::string& new_tgt = name_map.at(e.tgt_id);
    if (new_src == new_tgt) {
      continue;
    }

    MLEdgeKey key{new_src, new_tgt};
    unmerged_edges[key].push_back(std::move(e));
  }

  std::vector<MLEdge> merged_edges;
  merged_edges.reserve(unmerged_edges.size());
  for (auto& it : unmerged_edges) {
    const auto& key = it.first;
    merged_edges.push_back(
        MLEdge{key.first, key.second, std::move(it.second), {}});
  }

  return merged_edges;
}

std::vector<MLEdge> mergeEdges(
    std::vector<MLEdge> edges,
    const std::unordered_map<std::string, std::string>& name_map) {
  return mergeEdgesNoCopy(std::move(edges), name_map);
}

// topologically sort ml graph
MLGraph sortMLGraph(const MLGraph& graph) {
  std::vector<MLNode> nodes;

  const auto bg = toBGL(graph);
  for (const auto& v : all_nodes_topo<MLVertex, MLBGraph>(bg)) {
    nodes.push_back(bg[v]);
  }
  return MLGraph{nodes, graph.edges};
}

std::string dumpMLEdge(const std::vector<MLEdge>& edges, int indent) {
  std::stringstream ss;
  if (!edges.empty()) {
    for (int i = 0; i < indent * 2; i++) {
      ss << " ";
    }
    std::vector<std::string> edges_str;
    for (const auto& se : edges) {
      edges_str.push_back(se.src_id + "->" + se.tgt_id);
    }
    ss << join_as_str(edges_str) << std::endl;
  }
  return ss.str();
}

std::string dumpMLNode(const MLNode& n, int indent) {
  std::stringstream ss;
  for (int i = 0; i < indent * 2; i++) {
    ss << " ";
  }
  ss << "MLNode: " << n.id << " #sub_nodes=" << n.sub_nodes.size()
     << " #sub_edges=" << n.sub_edges.size() << " size=" << n.getSize()
     << " L=" << n.getLevel() << std::endl;
  for (const auto& sn : n.sub_nodes) {
    ss << dumpMLNode(sn, indent + 1);
  }
  ss << dumpMLEdge(n.sub_edges, indent + 1);
  return ss.str();
}

std::string dumpMLGraph(const MLGraph& g) {
  std::stringstream ss;
  ss << "MLGraph: "
     << " #nodes=" << g.nodes.size() << " #edges=" << g.edges.size()
     << " size=" << g.getSize() << " L=" << g.getLevel() << std::endl;

  for (const auto& n : g.nodes) {
    ss << dumpMLNode(n, 1);
  }
  ss << dumpMLEdge(g.edges, 1);

  return ss.str();
}

bool verify(const MLNode& n) {
  std::unordered_set<std::string> ids;
  ids.reserve(n.sub_nodes.size());
  for (const auto& sn : n.sub_nodes) {
    ids.insert(sn.id);
  }
  for (const auto& se : n.sub_edges) {
    if (!contains(ids, se.src_id) || !contains(ids, se.tgt_id)) {
      spdlog::info(
          "@verify nodes={} subedges={}->{}", join_as_str(ids), se.src_id,
          se.tgt_id);
      return false;
    }
  }
  return true;
}

bool verify(const MLGraph& g) {
  for (const auto& n : g.nodes) {
    if (!verify(n)) {
      spdlog::info("Verification failed. node={}", dumpMLNode(n));
      return false;
    }
  }

  std::unordered_map<std::string, std::unordered_set<std::string>> subnode_ids;
  for (const auto& n : g.nodes) {
    for (const auto& sn : n.sub_nodes) {
      subnode_ids[n.id].insert(sn.id);
    }
  }

  for (const auto& e : g.edges) {
    if (!contains(subnode_ids, e.src_id)) {
      spdlog::info(
          "graph does not contains edge src {}. edge: {}->{}", e.src_id,
          e.src_id, e.tgt_id);
      return false;
    }
    if (!contains(subnode_ids, e.tgt_id)) {
      spdlog::info(
          "graph does not contains edge tgt {}. edge: {}->{}", e.tgt_id,
          e.src_id, e.tgt_id);
      return false;
    }

    assert(contains(subnode_ids, e.tgt_id));
    const auto& src_ids = subnode_ids.at(e.src_id);
    const auto& tgt_ids = subnode_ids.at(e.tgt_id);

    for (const auto& se : e.sub_edges) {
      if (!contains(src_ids, se.src_id)) {
        spdlog::info(
            "Edge src verification failed. {} not found. edge: {}->{} subedge: {}->{}",
            se.src_id, e.src_id, e.tgt_id, se.src_id, se.tgt_id);
        return false;
      }
      if (!contains(tgt_ids, se.tgt_id)) {
        spdlog::info(
            "Edge tgt verification failed. {} not found. edge: {}->{} subedge: {}->{}",
            se.tgt_id, e.src_id, e.tgt_id, se.src_id, se.tgt_id);
        return false;
      }
    }
  }
  return true;
}

MLNode doToLowerLevel(const MLNode& ml_node) {
  std::vector<MLNode> new_sub_nodes;
  std::vector<MLEdge> new_sub_edges;

  assert(ml_node.getLevel() > 0);

  for (const auto& sn : ml_node.sub_nodes) {
    for (const auto& ssn : sn.sub_nodes) {
      new_sub_nodes.push_back(ssn);
    }
    for (const auto& sse : sn.sub_edges) {
      new_sub_edges.push_back(sse);
    }
  }

  for (const auto& se : ml_node.sub_edges) {
    for (const auto& sse : se.sub_edges) {
      new_sub_edges.push_back(sse);
    }
  }

  return MLNode{ml_node.id, ml_node.graph, new_sub_nodes, new_sub_edges};
}

MLGraph toLowerLevel(const MLGraph& ml_graph) {
  assert(ml_graph.getLevel() > 0);

  std::vector<MLNode> new_nodes;
  for (const auto& n : ml_graph.nodes) {
    new_nodes.push_back(doToLowerLevel(n));
  }

  std::vector<MLEdge> new_edges;
  for (const auto& e : ml_graph.edges) { // e @ level L
    std::vector<MLEdge> new_sub_edges;
    std::vector<MLBaseEdge> new_base_edges;
    for (const auto& se : e.sub_edges) { // se @ level L-1
      for (const auto& sse : se.sub_edges) { // sse @ level L-2
        new_sub_edges.push_back(sse);
      }
      for (const auto& bse : se.base_edges) {
        new_base_edges.push_back(bse);
      }
    }
    MLEdge new_edge{e.src_id, e.tgt_id, new_sub_edges, new_base_edges};
    new_edges.push_back(new_edge);
  }

  return MLGraph{new_nodes, new_edges};
}

bool isConvex(const MLVertex& n1, const MLVertex& n2, MLBGraph& graph) {
  boost::remove_edge(n1, n2, graph);
  bool ret = is_reachable(n1, n2, graph);
  boost::add_edge(n1, n2, graph);

  return !ret;
}

std::unordered_set<std::string> getMLNodeIds(const std::vector<MLNode>& nodes) {
  std::unordered_set<std::string> ret;
  ret.reserve(nodes.size());
  for (const auto& n : nodes) {
    ret.insert(n.id);
  }
  return ret;
}

bool containsSubgraph(const MLNode& n, const std::string& subgraph_id) {
  return contains(getMLNodeIds(n.sub_nodes), subgraph_id);
}

bool containsSubedge(const MLEdge& edge, const MLEdge& sub_edge) {
  MLEdgeKeySet sub_edge_set;
  for (const auto& se : edge.sub_edges) {
    MLEdgeKey k{se.src_id, se.tgt_id};
    sub_edge_set.insert(k);
  }

  MLEdgeKey k{sub_edge.src_id, sub_edge.tgt_id};
  return contains(sub_edge_set, k);
}

MLBGraph toBGL(
    const std::vector<MLNode>& nodes, const std::vector<MLEdge>& edges) {
  MLBGraph bg;
  std::unordered_map<std::string, MLVertex> node_map;

  for (const auto& n : nodes) {
    MLVertex v = boost::add_vertex(bg);
    bg[v] = n;
    node_map[n.id] = v;
  }

  for (const auto& e : edges) {
    assert(contains(node_map, e.src_id));
    assert(contains(node_map, e.tgt_id));

    boost::add_edge(node_map.at(e.src_id), node_map.at(e.tgt_id), e, bg);
  }

  return bg;
}

MLBGraph toBGL(const MLNode& node) {
  return toBGL(node.sub_nodes, node.sub_edges);
}

MLBGraph toBGL(const MLGraph& ml_graph) {
  const std::vector<MLNode>& nodes = ml_graph.nodes;
  const std::vector<MLEdge>& edges = ml_graph.edges;

  return toBGL(nodes, edges);
}

MLGraph fromBGL(const MLBGraph& bg) {
  std::vector<MLNode> nodes;
  nodes.reserve(num_vertices(bg));
  for (const auto& v : all_nodes_topo<MLVertex, MLBGraph>(bg)) {
    ;
    nodes.push_back(bg[v]);
  }

  std::vector<MLEdge> edges;
  edges.reserve(num_edges(bg));

  auto edge_range = boost::edges(bg);
  for (auto first = edge_range.first, last = edge_range.second; first != last;
       ++first) {
    const auto& e = *first;
    edges.push_back(bg[e]);
  }

  return MLGraph{nodes, edges};
}
} // namespace rannc