//
// Created by Masahiro Tanaka on 2019-03-05.
//

#ifndef PYRANNC_DECOMPOSITION_H
#define PYRANNC_DECOMPOSITION_H

#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/algorithm/for_each.hpp>

#include <msgpack.hpp>
#include <ostream>

#include <comm/MPIUtil.h>
#include <comm/SComm.h>
#include <graph/ir.h>

namespace rannc {
class IRGraph;
class IRValue;
class IRNode;

enum VertexType { VALUE, NODE };

struct VertexInfo {
  std::string name;
  IRValue value;
  IRNode node;
  VertexType type;
  bool is_param;
  bool is_criterion;
  bool is_loss;
  bool is_input;
  bool is_orig_input;
  bool is_output;
  bool is_orig_output;
  std::string id;
  std::unordered_set<int> ranks;
  long long fwd_time;
  long long bwd_time;
};

struct EdgeInfo {};

struct GraphInfo {
  std::string id;
};

typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS, VertexInfo, EdgeInfo,
    GraphInfo>
    BGraph;
typedef boost::graph_traits<BGraph>::vertex_descriptor Vertex;
typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, VertexInfo, EdgeInfo,
    GraphInfo>
    UndirectedBGraph;

struct GraphConnection {
  std::string value;
  std::string src;
  std::string dest;
  std::string id;
  std::string targetId;

  friend std::ostream& operator<<(
      std::ostream& os, const GraphConnection& connection) {
    os << "value: " << connection.value << " src: " << connection.src
       << " dest: " << connection.dest << " id: " << connection.id
       << " targetId: " << connection.targetId;
    return os;
  }
};
extern const std::string MASTER_NAME;

struct BDecomposition {
  std::unordered_map<std::string, BGraph> graphs;
  std::vector<GraphConnection> connections;
  std::vector<std::string> order;
};

struct Partition {
  std::string id;
  std::shared_ptr<IRGraph> graph;
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> subgraphs;
  std::vector<GraphConnection> connections;
  std::vector<std::string> order;
};

struct GraphConnectionDP {
  GraphConnectionDP() = default;
  GraphConnectionDP(const GraphConnection& con) {
    value = con.value;
    src_graphs = {con.src};
    dest_graphs = {con.dest};
  }

  std::string value;
  std::vector<std::string> src_graphs;
  std::vector<std::string> dest_graphs;

  friend std::ostream& operator<<(
      std::ostream& os, const GraphConnectionDP& connection) {
    os << "value: " << connection.value
       << " src: " << join_as_str(connection.src_graphs)
       << " dest: " << join_as_str(connection.dest_graphs);
    return os;
  }
};

struct PartitionDP {
  PartitionDP() = default;
  PartitionDP(const Partition& p) {
    id = p.id;
    graph = p.graph;
    subgraphs = p.subgraphs;
    connections = p.connections;
  }

  std::string id;
  std::shared_ptr<IRGraph> graph;
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> subgraphs;
  std::unordered_map<std::string, int> replica_nums;
  std::vector<GraphConnection> connections;
};

enum class RVertexType { GRAPH, ROUTE };
struct RVertexInfo {
  std::string name;
  RVertexType type;
  RouteDP route;
};
typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS, RVertexInfo>
    RGraph;
typedef boost::graph_traits<RGraph>::vertex_descriptor RVertex;

struct GraphRoutes {
  std::vector<RouteDP> fwd_send_routes;
  std::vector<RouteDP> fwd_recv_routes;
  std::vector<RouteDP> bwd_send_routes;
  std::vector<RouteDP> bwd_recv_routes;

  MSGPACK_DEFINE(
      fwd_send_routes, fwd_recv_routes, bwd_send_routes, bwd_recv_routes);
};

struct Deployment {
  std::string id;
  std::shared_ptr<IRGraph> graph;
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> subgraphs;
  std::unordered_map<std::string, std::unordered_set<int>> allocation;
  std::vector<RouteDP> fwd_routes;
  std::vector<RouteDP> fwd_in_routes;
  std::vector<RouteDP> fwd_out_routes;
  std::vector<std::string> fwd_graph_order;
  std::vector<RouteDP> bwd_routes;
  std::vector<RouteDP> bwd_in_routes;
  std::vector<RouteDP> bwd_out_routes;
  std::vector<std::string> bwd_graph_order;
  int pipeline_num;
  bool checkpointing;
  bool offload_params;

  friend std::ostream& operator<<(
      std::ostream& os, const Deployment& deployment);

  MSGPACK_DEFINE(
      id, graph, subgraphs, allocation, fwd_routes, fwd_in_routes,
      fwd_out_routes, fwd_graph_order, bwd_routes, bwd_in_routes,
      bwd_out_routes, bwd_graph_order, pipeline_num, checkpointing,
      offload_params);
};

void verifyDeployment(const Deployment& deployment);

Partition createPartition(
    const std::shared_ptr<IRGraph>& ir_graph,
    const std::vector<std::shared_ptr<IRGraph>>& subgraphs);

template <typename Vertex, typename Graph>
std::vector<Vertex> all_nodes(const Graph& g) {
  std::vector<Vertex> nodes;

  auto vertex_range = boost::vertices(g);
  for (auto first = vertex_range.first, last = vertex_range.second;
       first != last; ++first) {
    nodes.push_back(*first);
  }
  return nodes;
}

template <typename Vertex, typename Graph>
std::vector<Vertex> all_nodes_topo(const Graph& g) {
  std::vector<Vertex> rev_topo_vertices;
  boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

  std::vector<Vertex> sorted;
  boost::for_each(
      rev_topo_vertices | boost::adaptors::reversed,
      [&sorted](Vertex v) { sorted.push_back(v); });
  return sorted;
}

template <typename Vertex, typename Graph>
std::vector<Vertex> source_nodes(const Vertex& v, const Graph& g) {
  typedef
      typename boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
  in_edge_iterator ei, edge_end;

  std::vector<Vertex> sources;
  for (boost::tie(ei, edge_end) = in_edges(v, g); ei != edge_end; ++ei) {
    sources.push_back(source(*ei, g));
  }
  return sources;
}

template <typename Vertex, typename Graph>
std::vector<Vertex> graph_input_nodes(const Graph& g) {
  typename boost::graph_traits<Graph>::vertex_iterator i, end;
  std::vector<Vertex> inputs;
  for (boost::tie(i, end) = vertices(g); i != end; ++i) {
    auto sources = source_nodes(*i, g);
    if (sources.empty()) {
      inputs.push_back(*i);
    }
  }
  return inputs;
}

template <typename Vertex, typename Graph>
std::vector<Vertex> graph_input_values(
    const Graph& g, std::function<bool(Vertex)> condition) {
  std::vector<Vertex> input_nodes = graph_input_nodes<Vertex, Graph>(g);
  std::vector<Vertex> val_inputs;
  std::copy_if(
      input_nodes.begin(), input_nodes.end(), std::back_inserter(val_inputs),
      condition);
  return val_inputs;
}

template <typename Vertex, typename Graph>
std::vector<Vertex> graph_param_input_nodes(const Graph& g) {
  return graph_input_values<Vertex, Graph>(
      g, [&g](Vertex v) { return g[v].type == VALUE && g[v].is_param; });
}

template <typename Vertex, typename Graph>
std::vector<Vertex> graph_regular_input_nodes(const Graph& g) {
  return graph_input_values<Vertex, Graph>(
      g, [&g](Vertex v) { return g[v].type == VALUE && !g[v].is_param; });
}

template <typename Vertex, typename Graph>
std::vector<Vertex> target_nodes(const Vertex& v, const Graph& g) {
  typedef
      typename boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
  typedef typename boost::adjacency_iterator_generator<
      Graph, Vertex, out_edge_iterator>::type adjacency_iterator;

  std::pair<adjacency_iterator, adjacency_iterator> iter_pair =
      boost::adjacent_vertices(v, g);
  std::vector<Vertex> adjacents;
  for (auto first = iter_pair.first, last = iter_pair.second; first != last;
       ++first) {
    adjacents.push_back(*first);
  }
  return adjacents;
}

template <typename Vertex, typename Graph>
std::unordered_set<Vertex> graph_output_nodes(const Graph& g) {
  typename boost::graph_traits<Graph>::vertex_iterator i, end;
  std::unordered_set<Vertex> outputs;
  for (boost::tie(i, end) = vertices(g); i != end; ++i) {
    auto sources = target_nodes(*i, g);
    if (sources.empty()) {
      outputs.emplace(*i);
    }
  }
  return outputs;
}

template <typename Vertex, typename Graph>
std::vector<std::string> adjacent_node_names(const Vertex& v, const Graph& g) {
  std::vector<std::string> names;
  for (const auto& adj : target_nodes(v, g)) {
    names.push_back(g[adj].name);
  }
  return names;
}

template <typename Vertex, typename Graph>
Vertex find_value_node(const std::string& name, const Graph& g) {
  auto vertex_range = vertices(g);
  for (auto first = vertex_range.first, last = vertex_range.second;
       first != last; ++first) {
    Vertex v = *first;
    if (g[v].type == VALUE && g[v].name == name) {
      return v;
    }
  }
  throw std::invalid_argument("Value node not found: " + name);
}

template <typename Vertex, typename Graph>
bool is_reachable(Vertex v_start, Vertex v_goal, const Graph& g) {
  std::vector<boost::default_color_type> color(
      boost::num_vertices(g), boost::white_color);
  return boost::is_reachable(v_start, v_goal, g, color.data());
}

template <typename Vertex, typename Graph>
void print_is_reachable(Vertex v_start, Vertex v_goal, const Graph& g) {
  std::vector<boost::default_color_type> color(
      boost::num_vertices(g), boost::white_color);
  if (boost::is_reachable(v_start, v_goal, g, color.data())) {
    std::cout << "REACHABLE: " << g[v_start].name << " -> " << g[v_goal].name
              << std::endl;
  } else {
    std::cout << "UNREACHABLE: " << g[v_start].name << " -> " << g[v_goal].name
              << std::endl;
  }
}

template <typename Vertex, typename Graph>
Vertex addSubgraphVertex(
    const Graph& parent, const Vertex& pv, Graph& subgraph,
    std::unordered_map<Vertex, Vertex>& vmap) {
  Vertex sv = boost::add_vertex(subgraph);
  subgraph[sv] = parent[pv];
  vmap[pv] = sv;
  return sv;
}

template <typename Vertex, typename Graph>
bool hasValue(const Graph& g, const std::string& name) {
  const auto nodes = all_nodes<Vertex, Graph>(g);
  for (const auto& v : nodes) {
    if (g[v].type == VALUE && g[v].name == name) {
      return true;
    }
  }
  return false;
}

template <class Graph>
struct vertex_rank_label_writer {
  vertex_rank_label_writer(const Graph& g) : graph_(g) {}
  template <class Vertex>
  void operator()(std::ostream& out, const Vertex& vertex) const {
    write(out, vertex);
  }

 private:
  template <class Vertex>
  void write(std::ostream& out, const Vertex& vertex) const {
    std::string color;
    if (graph_[vertex].ranks.empty()) {
      color = "ghostwhite";
    } else {
      switch (*graph_[vertex].ranks.begin() % 6) {
        case 0:
          color = "ghostwhite";
          break;
        case 1:
          color = "antiquewhite";
          break;
        case 2:
          color = "azure";
          break;
        case 3:
          color = "gold1";
          break;
        case 4:
          color = "lightpink";
          break;
        case 5:
          color = "olivedrab1";
          break;
      }
    }

    std::string shape = "ellipse";
    if (graph_[vertex].type == NODE) {
      shape = "box";
    }

    if (graph_[vertex].type == NODE) {
      std::stringstream label_ss;
      label_ss << graph_[vertex].name << "\n " << graph_[vertex].id
               << "\n fwd=" << graph_[vertex].fwd_time
               << "\n bwd=" << graph_[vertex].bwd_time;

      out << "[label=\"" << label_ss.str() << "\", shape=\"" << shape
          << "\", id=\"" << graph_[vertex].id << "\", rank=\""
          << join_as_str(graph_[vertex].ranks) << "\", style=\""
          << "filled"
          << "\", fillcolor=\"" << color << "\"]";
    } else {
      std::stringstream label_ss;
      label_ss << graph_[vertex].name;
      auto ir_val = graph_[vertex].value;
      if (ir_val.isBatch()) {
        label_ss << "(B)";
      } else if (ir_val.isParam()) {
        label_ss << "(P)";
      } else if (ir_val.isLoss()) {
        label_ss << "(L)";
      }

      label_ss << "\n" << toString(ir_val.getType());

      out << "[label=\"" << label_ss.str() << "\", shape=\"" << shape
          << "\", id=\"" << graph_[vertex].id << "\", rank=\""
          << join_as_str(graph_[vertex].ranks) << "\", style=\""
          << "filled"
          << "\", fillcolor=\"" << color << "\"]";
    }
  }
  const Graph& graph_;
};

template <class Graph>
inline vertex_rank_label_writer<Graph> make_rank_label_writer(const Graph& g) {
  return vertex_rank_label_writer<Graph>(g);
}

BDecomposition createSubGraphs(BGraph& g);

std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_set<int>>>
getParamRanks(const Deployment& deployment);
std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_set<int>>>
getConstantRanks(const Deployment& deployment);
std::vector<RouteDP> filterRecvRoutes(
    const std::vector<RouteDP>& routes, int rank);
std::vector<RouteDP> filterSendRoutes(
    const std::vector<RouteDP>& routes, int rank);
std::string getSourceGraphName(const RouteDP& route, int rank);
std::string getDestGraphName(const RouteDP& route, int rank);
std::unordered_map<std::string, GraphRoutes> getRoutesByGraph(
    const Deployment& deployment,
    const std::vector<std::shared_ptr<IRGraph>>& subgraphs, int rank);

template <typename Graph>
std::shared_ptr<IRGraph> fromBGL(const Graph& b_graph) {
  std::shared_ptr<IRGraph> irGraph = std::make_shared<IRGraph>();

  typename boost::graph_traits<Graph>::vertex_iterator i, end;

  std::vector<IRNode> nodes;
  std::unordered_map<std::string, IRValue> values;

  std::vector<std::string> non_param_input_names;
  std::vector<std::string> param_input_names;
  std::vector<std::string> output_names;

  for (const Vertex& v : all_nodes_topo<Vertex, Graph>(b_graph)) {
    const VertexInfo& vi = b_graph[v];

    if (vi.type == NODE) {
      nodes.push_back(vi.node);
    } else if (vi.type == VALUE) {
      values[vi.name] = vi.value;

      if (vi.is_input) {
        if (vi.is_param) {
          param_input_names.push_back(vi.name);
        } else {
          non_param_input_names.push_back(vi.name);
        }
      }
      if (vi.is_output) {
        output_names.push_back(vi.name);
      }
    } else {
      throw std::runtime_error("Unexpected type");
    }
  }

  std::vector<std::string> input_names;
  for (const auto& in : non_param_input_names) {
    input_names.push_back(in);
  }
  for (const auto& in : param_input_names) {
    input_names.push_back(in);
  }

  return std::make_shared<IRGraph>(
      b_graph[boost::graph_bundle].id, nodes, values, input_names,
      output_names);
}

BGraph toBGL(const std::shared_ptr<IRGraph>& ir_graph);

//    void mergeNodesFromInput(BGraph &g);
//    void fixConstantRank(BGraph &g);
Partition createPartition(const BGraph& g);
std::shared_ptr<IRGraph> scaleGraph(
    const std::shared_ptr<IRGraph>& graph, int num, int64_t batch_size);
PartitionDP replicate(
    const PartitionDP& partition,
    const std::unordered_map<std::string, int>& repl_nums, int pipeline_num,
    int64_t batch_size);
Deployment createDeployment(
    const PartitionDP& partition,
    const std::unordered_map<std::string, std::unordered_set<int>>& allocation,
    int np);

std::vector<IRValue> searchGraphValuesByName(
    const std::shared_ptr<IRGraph>& ir_graph, const std::string& val_name);
std::vector<IRValue> searchGraphValuesByName(
    const std::vector<std::shared_ptr<IRGraph>>& ir_graphs,
    const std::string& val_name);

std::unordered_map<std::string, std::unordered_set<int>> searchAllocation(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem);
std::unordered_map<std::string, std::unordered_set<int>> searchAllocationFlat(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem);
std::unordered_map<std::string, std::unordered_set<int>> searchAllocationSimple(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem);
std::unordered_map<std::string, std::unordered_set<int>>
searchAllocationFitToDevice(const PartitionDP& partition);

size_t getSizeInByte(const std::shared_ptr<IRGraph>& ir_graph);
std::string toString(const GraphRoutes& routes);

void fixNonBatchRanks(BGraph& g);
std::vector<size_t> splitByValueSizes(const BGraph& g, int n_partition);
void setRanksOnGraph(BGraph& g, const std::vector<size_t>& split);
BGraph copyGraphWithBatch(const BGraph& g);

typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS, std::shared_ptr<IRGraph>>
    HGraph;
typedef boost::graph_traits<HGraph>::vertex_descriptor HVertex;

struct AllocSolution {
  std::vector<std::shared_ptr<IRGraph>> graphs;
  std::unordered_map<std::string, int> repl_nums;
  int pipeline_num;
  bool checkpointing;
  std::vector<size_t> boundaries;
  std::vector<size_t> dev_nums;

  MSGPACK_DEFINE(
      graphs, repl_nums, pipeline_num, checkpointing, boundaries, dev_nums);
};

struct PartitioningConf {
  int dev_num;
  size_t batch_size;
  size_t dev_mem;
  int opt_param_factor;
  bool use_amp_master_params;
  bool enable_zero;
  bool offload_params;
  bool force_dist_matmul;
  int min_pipeline_num;
  int max_pipeline_num;
  int min_partition_num;
  int max_partition_num;
  int cfg_pipeline_num;
  size_t cfg_stage_num;

  MSGPACK_DEFINE(
      dev_num, batch_size, dev_mem, opt_param_factor, use_amp_master_params,
      enable_zero, offload_params, force_dist_matmul, min_pipeline_num,
      max_pipeline_num, min_partition_num, max_partition_num, cfg_pipeline_num,
      cfg_stage_num);
};

PartitioningConf makePartitioningConf(
    int dev_num, size_t batch_size, size_t dev_mem, bool use_amp_master_params,
    bool enable_zero);

} // namespace rannc

#endif // PYRANNC_DECOMPOSITION_H
