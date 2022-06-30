//
// Created by Masahiro Tanaka on 2019-03-05.
//
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/strong_components.hpp>

#include <Common.h>
#include <Config.h>
#include <graph/ir.h>

#include "comp/BatchSizeCalculator.h"
#include "Decomposition.h"

namespace {
const std::string LOGGER_NAME = "Decomposition";
}

namespace rannc {
const std::string MASTER_NAME = "MASTER";

typedef boost::filtered_graph<
    BGraph, std::function<bool(BGraph::edge_descriptor)>,
    std::function<bool(BGraph::vertex_descriptor)>>
    ComponentGraph;

void setProperty(VertexInfo& v, const IRValue& value) {
  v.name = value.getName();
  v.value = value;
  v.type = VALUE;
  v.is_param = value.isParam();
  v.id = "v_" + value.getName();
  v.ranks = {};
  v.is_input = false;
  v.is_orig_input = false;
  v.is_output = false;
  v.is_orig_output = false;
  v.is_loss = value.isLoss();
}

void setProperty(VertexInfo& v, const IRNode& node) {
  v.name = node.getName();
  v.node = node;
  v.type = NODE;
  v.id = node.getId();
  v.ranks = {};
  v.is_input = false;
  v.is_orig_input = false;
  v.is_output = false;
  v.is_orig_output = false;
  v.is_criterion = node.isCriterion();
  v.fwd_time = 0;
  v.bwd_time = 0;
}

/**
 * Convert an IR graph to a BGL graph.
 *
 * @param ir_graph IR graph to convert
 * @return BGL graph
 */
BGraph toBGL(const std::shared_ptr<IRGraph>& ir_graph) {
  BGraph b_graph;

  b_graph[boost::graph_bundle].id = ir_graph->getName();
  b_graph[boost::graph_bundle].batch_size = ir_graph->getBatchSize();

  const auto logger = getLogger(LOGGER_NAME);
  //        logger->trace("Converting graph:");

  std::unordered_map<std::string, Vertex> values;
  for (const std::string& input_name : ir_graph->getInputNames()) {
    const auto& value = ir_graph->getValue(input_name);
    Vertex v = boost::add_vertex(b_graph);
    setProperty(b_graph[v], value);
    values[input_name] = v;

    b_graph[v].is_input = true;
    b_graph[v].is_orig_input = true;

    //            logger->trace("   Value name={} id={} param={} input={}
    //            output={}", b_graph[v].name, b_graph[v].id,
    //                          b_graph[v].is_param, b_graph[v].is_input,
    //                          b_graph[v].is_output);
  }

  // assume that nodes are topologically sorted
  for (const auto& in_node : ir_graph->getNodes()) {
    Vertex v = boost::add_vertex(b_graph);
    setProperty(b_graph[v], in_node);

    //            logger->trace("Node name={} id={}", b_graph[v].name,
    //            b_graph[v].id);

    for (const auto& input_name : in_node.getInputNames()) {
      Vertex in_v = values[input_name];
      boost::add_edge(in_v, v, b_graph);
      //                std::cout << "  Created edge: " << (*bGraph)[in_v].name
      //                << " -> " << (*bGraph)[v].name << std::endl;
    }

    for (const auto& output_name : in_node.getOutputNames()) {
      const auto& value = ir_graph->getValue(output_name);
      Vertex out_v = boost::add_vertex(b_graph);
      setProperty(b_graph[out_v], value);
      values[output_name] = out_v;
      boost::add_edge(v, out_v, b_graph);

      if (contains(ir_graph->getOutputNames(), output_name)) {
        b_graph[out_v].is_output = true;
        b_graph[out_v].is_orig_output = true;
      }

      //                logger->trace("   Value name={} id={} param={} input={}
      //                output={}", b_graph[out_v].name,
      //                              b_graph[out_v].id,
      //                              b_graph[out_v].is_param,
      //                              b_graph[out_v].is_input,
      //                              b_graph[out_v].is_output);

      //                std::cout << "  Created edge: " << (*bGraph)[v].name <<
      //                " -> " << (*bGraph)[out_v].name << std::endl;
    }
  }

  return b_graph;
}

bool containsVertex(
    const BGraph& g, const std::vector<Vertex>& vertices, const Vertex tgt) {
  for (const Vertex& v : vertices) {
    if (g[v].id == g[tgt].id) {
      return true;
    }
  }
  return false;
}

bool containsVertexById(const BGraph& g, const std::string& id) {
  for (const Vertex& v : all_nodes<Vertex, BGraph>(g)) {
    if (g[v].id == id) {
      return true;
    }
  }
  return false;
}

Vertex findVertexById(const BGraph& g, const std::string& id) {
  for (const Vertex& v : all_nodes<Vertex, BGraph>(g)) {
    if (g[v].id == id) {
      return v;
    }
  }
  throw std::invalid_argument("Vertex with id '" + id + "' is not found.");
}

BGraph createSubgraph(
    const BGraph& g, const std::vector<Vertex>& sub_vertices) {
  std::vector<Vertex> topo_vertices, rev_topo_vertices;
  boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

  BGraph subgraph;
  std::unordered_map<std::string, Vertex> new_nodes;
  for (const Vertex& v : rev_topo_vertices) {
    if (containsVertex(g, sub_vertices, v)) {
      Vertex sv = boost::add_vertex(subgraph);
      subgraph[sv] = g[v];
      new_nodes[g[v].id] = sv;

      for (const auto& out : target_nodes(v, g)) {
        if (contains(new_nodes, g[out].id)) {
          boost::add_edge(sv, new_nodes[g[out].id], subgraph);
        }
      }
    }
  }
  return subgraph;
}

BGraph createSubgraphByRank(const BGraph& g, int rank) {
  std::vector<Vertex> sub_vertices;

  for (const auto& v : all_nodes<Vertex, BGraph>(g)) {
    if (contains(g[v].ranks, rank)) {
      sub_vertices.push_back(v);
    }
  }
  return createSubgraph(g, sub_vertices);
}

std::unordered_map<int, BGraph> createSubgraphsByRank(const BGraph& g) {
  std::unordered_map<int, BGraph> subgraphs;
  std::unordered_set<int> ranks;

  for (const auto& v : all_nodes<Vertex, BGraph>(g)) {
    for (int r : g[v].ranks) {
      ranks.insert(r);
    }
  }
  for (int r : ranks) {
    subgraphs[r] = createSubgraphByRank(g, r);
  }

  return subgraphs;
}

int getRank(std::unordered_map<int, BGraph>& subgraphs, const std::string& id) {
  for (auto& sg : subgraphs) {
    if (containsVertexById(sg.second, id))
      return sg.first;
  }
  throw std::invalid_argument("Node not found in a subgraph: " + id);
}

std::string createSubgraphId(const std::string& orig_name, int index) {
  std::stringstream ss;
  ss << orig_name << "_p" << index;
  return ss.str();
}

bool containsConnection(
    const std::vector<GraphConnection>& connections,
    const GraphConnection& con) {
  bool found = false;
  for (const auto& c : connections) {
    if (c.src == con.src && c.dest == con.dest && c.value == con.value) {
      return true;
    }
  }
  return found;
}

bool hasSameValueConnection(
    const std::vector<GraphConnection>& connections,
    const std::string& value_name, const std::string& dest_sg) {
  for (const auto& c : connections) {
    if (c.value == value_name && c.dest == dest_sg) {
      return true;
    }
  }
  return false;
}

BDecomposition createSubGraphs(const BGraph& g) {
  std::unordered_map<int, std::string> subgraph_id_map;
  std::unordered_map<std::string, BGraph> subgraphs;
  std::vector<std::string> order;

  auto subgraphs_by_rank = createSubgraphsByRank(g);
  for (auto idx : keys(subgraphs_by_rank)) {
    auto& sg = subgraphs_by_rank.at(idx);
    const auto sg_id = createSubgraphId(g[boost::graph_bundle].id, idx);
    sg[boost::graph_bundle].id = sg_id;
    sg[boost::graph_bundle].batch_size = g[boost::graph_bundle].batch_size;

    subgraph_id_map[idx] = sg_id;
    subgraphs[sg_id] = sg;

    order.push_back(sg_id);
  }

  typedef boost::graph_traits<BGraph>::edge_iterator edge_iterator;
  edge_iterator ei, ei_end;

  // Add inter-graph connections
  std::vector<GraphConnection> connections;
  std::vector<GraphConnection> param_connections;

  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
    Vertex src = boost::source(*ei, g);
    Vertex tgt = boost::target(*ei, g);

    for (int tgt_rank : g[tgt].ranks) {
      if (!contains(g[src].ranks, tgt_rank)) {
        assert(!g[src].ranks.empty());
        int src_rank = *(g[src].ranks.begin());

        //        spdlog::info(
        //            "Checking graph edge: {}@rank{}->{}@rank{}", g[src].name,
        //            src_rank, g[tgt].name, tgt_rank);
        const auto& src_graph_id = subgraph_id_map[src_rank];
        const auto& tgt_graph_id = subgraph_id_map[tgt_rank];

        GraphConnection con{
            g[src].name, src_graph_id, tgt_graph_id, g[src].id, g[tgt].id};
        if (g[src].is_param) {
          if (!containsConnection(param_connections, con)) {
            param_connections.emplace_back(con);
            //            spdlog::info(
            //                "Found shared param connection: {} {}->{}",
            //                g[src].name, src_graph_id, tgt_graph_id);
          }
        } else {
          if (!containsConnection(connections, con)) {
            connections.emplace_back(con);
            //            spdlog::info(
            //                "Created graph connection: {} {}->{}",
            //                g[src].name, src_graph_id, tgt_graph_id);
          }
        }
      }
    }
  }

  // Add vertices for receive buffer
  for (const auto& con : addAll(connections, param_connections)) {
    BGraph& src_graph = subgraphs[con.src];
    BGraph& tgt_graph = subgraphs[con.dest];
    Vertex src = findVertexById(g, con.id);

    Vertex tgt_src = boost::add_vertex(tgt_graph);
    tgt_graph[tgt_src] = g[src];
    Vertex tgt_tgt = findVertexById(tgt_graph, con.targetId);
    boost::add_edge(tgt_src, tgt_tgt, tgt_graph);

    if (!g[src].is_param) {
      Vertex v_src = findVertexById(src_graph, con.id);
      src_graph[v_src].is_output = true;
      Vertex v_tgt = findVertexById(tgt_graph, con.id);
      tgt_graph[v_tgt].is_input = true;
    }
  }

  // Add connections from/to master
  const auto logger = getLogger(LOGGER_NAME);
  for (const auto& it : subgraphs) {
    const auto& sg_id = it.first;
    const BGraph& subgraph = it.second;

    for (const Vertex& v : all_nodes<Vertex, BGraph>(subgraph)) {
      const VertexInfo& vi = subgraph[v];
      if (vi.type != VALUE)
        continue;

      if (vi.is_orig_input && !vi.is_param) {
        if (!hasSameValueConnection(connections, vi.name, sg_id)) {
          GraphConnection con{vi.name, MASTER_NAME, sg_id, "NA", vi.id};
          connections.push_back(con);
          //          spdlog::info("Created graph connection: {} MASTER->{}",
          //          vi.name, sg_id);
        }
      }
      if (vi.is_orig_output) {
        GraphConnection con{vi.name, sg_id, MASTER_NAME, vi.id, "NA"};
        connections.push_back(con);
        //
        //        spdlog::info("Created graph connection: {} {}->MASTER",
        //        vi.name, sg_id);
      }
    }
  }

  return BDecomposition{subgraphs, connections, order};
}

std::vector<IRValue> searchGraphValuesByName(
    const std::shared_ptr<IRGraph>& ir_graph, const std::string& val_name) {
  std::vector<IRValue> found;
  for (const auto& v : ir_graph->getValues()) {
    if (v.first == val_name) {
      found.push_back(v.second);
    }
  }
  return found;
}

std::vector<IRValue> searchGraphValuesByName(
    const std::vector<std::shared_ptr<IRGraph>>& ir_graphs,
    const std::string& val_name) {
  std::vector<IRValue> found;

  for (const auto& g : ir_graphs) {
    for (const auto& v : searchGraphValuesByName(g, val_name)) {
      found.push_back(v);
    }
  }
  return found;
}

RouteTypeDP chooseRouteType(bool is_batch, bool is_loss, bool fwd) {
  if (is_batch) {
    return RouteTypeDP::REDIST;
  } else if (is_loss) {
    if (fwd) {
      return RouteTypeDP::BROADCAST;
    } else {
      return RouteTypeDP::WEIGHTED_REDUCE;
    }
  }
  std::stringstream ss;
  ss << "Failed to choose route type: is_batch=" << is_batch
     << " is_loss=" << is_loss << " fwd=" << fwd;
  throw std::invalid_argument(ss.str());
}

std::vector<int> getGraphRanks(
    const std::string& graph_id,
    const std::unordered_map<std::string, std::unordered_set<int>>& allocation,
    int np) {
  if (graph_id == MASTER_NAME) {
    std::unordered_set<int> ranks;
    for (int i = 0; i < np; i++) {
      ranks.insert(i);
    }
    return setToVector(ranks);
  }

  assert(contains(allocation, graph_id));
  return setToVector(allocation.at(graph_id));
}

bool isGraphInput(const GraphConnectionDP& con) {
  return con.src_graphs.size() == 1 && *con.src_graphs.begin() == MASTER_NAME;
}

bool isGraphOutput(const GraphConnectionDP& con) {
  return con.dest_graphs.size() == 1 && *con.dest_graphs.begin() == MASTER_NAME;
}

RGraph toRGraph(
    const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& subgraphs,
    const std::vector<RouteDP>& routes) {
  RGraph rg;
  std::unordered_set<std::string> graph_ids;
  std::unordered_map<std::string, RVertex> v_map;

  for (const auto& it : subgraphs) {
    const auto& id = it.first;

    const auto v = boost::add_vertex(rg);
    v_map[id] = v;
    rg[v].name = id;
    rg[v].type = RVertexType::GRAPH;
    graph_ids.insert(id);
  }

  for (const auto& r : routes) {
    const auto& src_id = r.source_graph;
    const auto& tgt_id = r.dest_graph;

    if (!contains(graph_ids, src_id)) {
      const auto v = boost::add_vertex(rg);
      v_map[src_id] = v;
      rg[v].name = src_id;
      rg[v].type = RVertexType::GRAPH;
      graph_ids.insert(src_id);
    }

    if (!contains(graph_ids, tgt_id)) {
      const auto v = boost::add_vertex(rg);
      v_map[tgt_id] = v;
      rg[v].name = tgt_id;
      rg[v].type = RVertexType::GRAPH;
      graph_ids.insert(tgt_id);
    }

    const auto e = boost::add_vertex(rg);
    rg[e].name = r.ir_value.getName();
    rg[e].type = RVertexType::ROUTE;
    rg[e].route = r;

    boost::add_edge(v_map.at(src_id), e, rg);
    boost::add_edge(e, v_map.at(tgt_id), rg);
  }

  return rg;
}

std::vector<std::string> sortGraphs(
    const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& subgraphs,
    const std::vector<RouteDP>& routes) {
  RGraph rg = toRGraph(subgraphs, routes);

  std::vector<std::string> sorted_graph_id;
  for (const auto& v : all_nodes_topo<RVertex, RGraph>(rg)) {
    if (rg[v].type == RVertexType::GRAPH) {
      sorted_graph_id.push_back(rg[v].name);
    }
  }

  return sorted_graph_id;
}

Deployment createDeployment(
    const PartitionDP& partition,
    const std::unordered_map<std::string, std::unordered_set<int>>& allocation,
    int np) {
  const auto logger = getLogger(LOGGER_NAME);
  logger->trace(
      "createDeployment starting. id={} #connections={}", partition.id,
      partition.connections.size());

  Deployment deployment;

  deployment.id = partition.id;
  deployment.graph = partition.graph;
  deployment.subgraphs = partition.subgraphs;
  deployment.allocation = allocation;

  std::vector<RouteDP> fwd_routes, fwd_in_routes, fwd_out_routes;
  std::vector<RouteDP> bwd_routes, bwd_in_routes, bwd_out_routes;

  TagMap& tag_map = TagMap::get();
  for (const auto& con : partition.connections) {
    logger->trace("createDeployment converting connection: {}", toString(con));

    std::vector<IRValue> ir_values;
    for (const auto& sg : values(partition.subgraphs)) {
      for (const auto& v : searchGraphValuesByName(sg, con.value)) {
        ir_values.push_back(v);
      }
    }
    assert(!ir_values.empty());
    const IRValue& val = ir_values.at(0);

    RouteDP fwd_route;
    fwd_route.location = con.value;
    fwd_route.sources = getGraphRanks(con.src, allocation, np);
    fwd_route.source_graph = con.src;
    fwd_route.dests = getGraphRanks(con.dest, allocation, np);
    fwd_route.dest_graph = con.dest;
    fwd_route.type = chooseRouteType(val.isBatch(), val.isLoss(), true);
    fwd_route.ir_value = val;

    int tag = tag_map.getRouteTag(fwd_route);
    fwd_route.tag = tag;
    fwd_route.source_tag = tag_map.getRouteSourceTag(fwd_route);

    if (isGraphInput(con)) {
      fwd_in_routes.push_back(fwd_route);
    } else if (isGraphOutput(con)) {
      fwd_out_routes.push_back(fwd_route);
    } else {
      fwd_routes.push_back(fwd_route);
    }

    logger->trace(
        "Created route for forwarding: {} (batch={} loss={} con={})",
        toString(fwd_route), val.isBatch(), val.isLoss(), toString(con));

    if (passedForBackward(val.getType())) {
      RouteDP bwd_route;
      bwd_route.location = con.value;
      bwd_route.sources = getGraphRanks(con.dest, allocation, np);
      bwd_route.source_graph = con.dest;
      bwd_route.dests = getGraphRanks(con.src, allocation, np);
      bwd_route.dest_graph = con.src;
      bwd_route.tag = tag;
      bwd_route.source_tag = tag_map.getRouteSourceTag(bwd_route);
      bwd_route.type = chooseRouteType(val.isBatch(), val.isLoss(), false);
      bwd_route.ir_value = val;

      if (isGraphInput(con)) {
        bwd_out_routes.push_back(bwd_route);
      } else if (isGraphOutput(con)) {
        bwd_in_routes.push_back(bwd_route);
      } else {
        bwd_routes.push_back(bwd_route);
      }

      logger->trace("Created route for backwarding: {}", toString(bwd_route));
    }
  }

  // Order values for communication
  // Hopefully this ordering allows blocking communication among graphs
  deployment.fwd_routes = fwd_routes;
  deployment.fwd_in_routes = fwd_in_routes;
  deployment.fwd_out_routes = fwd_out_routes;
  deployment.fwd_graph_order = sortGraphs(deployment.subgraphs, fwd_routes);
  deployment.bwd_routes = bwd_routes;
  deployment.bwd_in_routes = bwd_in_routes;
  deployment.bwd_out_routes = bwd_out_routes;
  deployment.bwd_graph_order = sortGraphs(deployment.subgraphs, bwd_routes);
  deployment.pipeline_num =
      config::Config::get().getVal<int>(config::PIPELINE_NUM);

  for (const auto& r : deployment.fwd_in_routes) {
    logger->debug("fwd_in_routes: {}", toString(r));
  }
  for (const auto& r : deployment.fwd_routes) {
    logger->debug("fwd_routes: {}", toString(r));
  }
  for (const auto& r : deployment.fwd_out_routes) {
    logger->debug("fwd_out_routes: {}", toString(r));
  }
  logger->debug("fwd_graph_order: {}", join_as_str(deployment.fwd_graph_order));
  for (const auto& r : deployment.bwd_in_routes) {
    logger->debug("bwd_in_routes: {}", toString(r));
  }
  for (const auto& r : deployment.bwd_routes) {
    logger->debug("bwd_routes: {}", toString(r));
  }
  for (const auto& r : deployment.bwd_out_routes) {
    logger->debug("bwd_out_routes: {}", toString(r));
  }
  logger->debug("bwd_graph_order: {}", join_as_str(deployment.bwd_graph_order));

  logger->trace("createDeployment finished");

  return deployment;
}

template <typename F>
std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_set<int>>>
getGraphValueRanks(const Deployment& deployment, F f) {
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::unordered_set<int>>>
      results;

  for (const auto& it : deployment.subgraphs) {
    const auto& sg_name = it.first;
    const auto& subgraph = it.second;

    for (const auto& val : f(subgraph)) {
      auto& ranks = results[sg_name][val.getName()];
      for (const int r : deployment.allocation.at(sg_name)) {
        ranks.insert(r);
      }
    }
  }
  return results;
}

std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_set<int>>>
getParamRanks(const Deployment& deployment) {
  return getGraphValueRanks(deployment, graphParamValues);
}

std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_set<int>>>
getConstantRanks(const Deployment& deployment) {
  return getGraphValueRanks(deployment, graphConstantValues);
}

std::vector<RouteDP> filterRoutes(
    const std::vector<RouteDP>& routes, int rank,
    std::function<std::vector<int>(const RouteDP& r)>& f) {
  std::vector<RouteDP> results;
  std::copy_if(
      routes.begin(), routes.end(), std::back_inserter(results),
      [rank, &f](const RouteDP& r) { return contains(f(r), rank); });
  return results;
}

std::vector<RouteDP> filterRecvRoutes(
    const std::vector<RouteDP>& routes, int rank) {
  std::function<std::vector<int>(const RouteDP& r)> f = [](const RouteDP& r) {
    return r.dests;
  };
  return filterRoutes(routes, rank, f);
}

std::vector<RouteDP> filterSendRoutes(
    const std::vector<RouteDP>& routes, int rank) {
  std::function<std::vector<int>(const RouteDP& r)> f = [](const RouteDP& r) {
    return r.sources;
  };
  return filterRoutes(routes, rank, f);
}

std::vector<RouteDP> filterRoutesByGraph(
    const std::vector<RouteDP>& routes,
    const std::vector<std::string>& val_names) {
  std::vector<RouteDP> results;
  for (const auto& route : routes) {
    if (contains(val_names, route.location.value_name)) {
      results.push_back(route);
    }
  }
  return results;
}

std::vector<RouteDP> filterRoutesByGraphInputs(
    const std::vector<RouteDP>& routes, const std::shared_ptr<IRGraph>& graph) {
  return filterRoutesByGraph(routes, graph->getInputNames());
}

std::vector<RouteDP> filterRoutesByGraphOutputs(
    const std::vector<RouteDP>& routes, const std::shared_ptr<IRGraph>& graph) {
  return filterRoutesByGraph(routes, graph->getOutputNames());
}

std::unordered_map<std::string, GraphRoutes> getRoutesByGraph(
    const Deployment& deployment,
    const std::vector<std::shared_ptr<IRGraph>>& subgraphs, int rank) {
  std::unordered_map<std::string, GraphRoutes> results;

  for (const auto& subgraph : subgraphs) {
    GraphRoutes graph_routes;
    graph_routes.fwd_recv_routes = filterRoutesByGraphInputs(
        filterRecvRoutes(deployment.fwd_routes, rank), subgraph);
    graph_routes.fwd_send_routes = filterRoutesByGraphOutputs(
        filterSendRoutes(deployment.fwd_routes, rank), subgraph);
    graph_routes.bwd_recv_routes = filterRoutesByGraphOutputs(
        filterRecvRoutes(deployment.bwd_routes, rank), subgraph);
    graph_routes.bwd_send_routes = filterRoutesByGraphInputs(
        filterSendRoutes(deployment.bwd_routes, rank), subgraph);

    results[subgraph->getName()] = graph_routes;
  }

  return results;
}

UndirectedBGraph toUndirected(const BGraph& g) {
  UndirectedBGraph udg;
  copy_graph(g, udg);
  return udg;
}

std::string createComponentGraphId(const std::string& orig_name, int index) {
  std::stringstream ss;
  ss << orig_name << "_c" << index;
  return ss.str();
}

std::string searchGraphWithValue(
    const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& ir_graphs,
    const std::string& val_name) {
  for (const auto& it : ir_graphs) {
    const std::string& sg_id = it.first;
    const auto& g = it.second;
    if (g->hasValue(val_name)) {
      return sg_id;
    }
  }
  throw std::invalid_argument("No value found in graphs: " + val_name);
}

/**
 * Partition a graph according to VertexInfo::rank.
 *
 * @param g Graph to partition
 * @return Partition
 */
Partition createPartition(const BGraph& g) {
  const auto logger = getLogger(LOGGER_NAME);
  logger->trace("createPartition started. id={}", g[boost::graph_bundle].id);

  std::vector<Vertex> topo_vertices = all_nodes_topo<Vertex, BGraph>(g);
  BDecomposition b_decomp = createSubGraphs(g);

  bool dump_graph = config::Config::get().getVal<bool>(config::DUMP_GRAPH);
  std::string dump_graph_prefix =
      config::Config::get().getVal<std::string>(config::DUMP_GRAPH_PREFIX);

  if (dump_graph) {
    std::ofstream file(dump_graph_prefix + "_r0.dot");
    boost::write_graphviz(file, g, vertex_rank_label_writer<BGraph>(g));
  }

  std::unordered_map<std::string, std::shared_ptr<IRGraph>> comp_graphs;
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::shared_ptr<IRGraph>>>
      comp_graph_groups;

  for (const auto& it : b_decomp.graphs) {
    const std::string& subg_id = it.first;
    const BGraph& subg = it.second;

    if (dump_graph) {
      std::ofstream file1(subg_id + ".dot");
      boost::write_graphviz(
          file1, subg, vertex_rank_label_writer<BGraph>(subg));
    }
    comp_graph_groups[subg_id][subg_id] = comp_graphs[subg_id] = fromBGL(subg);
  }
  // fix connections
  //        spdlog::info("b_decomp.connections={}",
  //        b_decomp.connections.size());

  std::vector<GraphConnection> comp_connection;
  for (const auto& con : b_decomp.connections) {
    //    spdlog::info("  con={}", toString(con));
    std::string comp_src = con.src;
    if (con.src != MASTER_NAME) {
      comp_src = searchGraphWithValue(comp_graph_groups[con.src], con.value);
    }
    std::string comp_dest = con.dest;
    if (con.dest != MASTER_NAME) {
      comp_dest = searchGraphWithValue(comp_graph_groups[con.dest], con.value);
    }

    GraphConnection comp_con = {
        con.value, comp_src, comp_dest, con.id, con.targetId};
    comp_connection.push_back(comp_con);
  }

  size_t cut_size = 0;
  for (const auto& con : b_decomp.connections) {
    const auto& n = find_value_node<Vertex, BGraph>(con.value, g);
    cut_size += g[n].value.getSizeInByte();
  }
  logger->trace(
      "Cut size: {} (cut count={})", cut_size, b_decomp.connections.size());

  auto ret = Partition{
      g[boost::graph_bundle].id, fromBGL(g), comp_graphs, comp_connection,
      b_decomp.order};
  logger->trace("createPartition finished");

  return ret;
}

size_t getSizeInByte(const std::shared_ptr<IRGraph>& ir_graph) {
  size_t size = 0;

  for (const auto& v : ir_graph->getValues()) {
    size += v.second.getSizeInByte();
  }
  return size;
}

std::string toString(const GraphRoutes& routes) {
  std::stringstream ss;

  const auto& fwd_recv_routes = routes.fwd_recv_routes;
  ss << "Fwd recv routes (num=" << fwd_recv_routes.size() << ")" << std::endl;
  for (const auto& r : fwd_recv_routes) {
    ss << "  " << r << std::endl;
  }
  const auto& fwd_send_routes = routes.fwd_send_routes;
  ss << "Fwd send routes (num=" << fwd_send_routes.size() << ")" << std::endl;
  for (const auto& r : fwd_send_routes) {
    ss << "  " << r << std::endl;
  }

  const auto& bwd_recv_routes = routes.bwd_recv_routes;
  ss << "Bwd recv routes (num=" << bwd_recv_routes.size() << ")" << std::endl;
  for (const auto& r : bwd_recv_routes) {
    ss << "  " << r << std::endl;
  }

  const auto& bwd_send_routes = routes.bwd_send_routes;
  ss << "Bwd send routes (num=" << bwd_send_routes.size() << ")" << std::endl;
  for (const auto& r : bwd_send_routes) {
    ss << "  " << r << std::endl;
  }
  return ss.str();
}

std::unordered_map<int, IRType> getDistTensorType(
    const IRType& type, const std::unordered_set<int>& ranks,
    int64_t batch_size) {
  BatchSizeCalculator bs_calc(1, batch_size);
  const auto dp_dim = bs_calc.calcDistBatchDims(type.getTensorDim(), ranks, 0);

  std::unordered_map<int, IRType> ret;
  for (int r : ranks) {
    const auto dp_type = IRType::createTensorType(
        type.getTensorElemType(), dp_dim.at(r), type.requiresGrad());
    ret[r] = dp_type;
  }
  return ret;
}

IRType getDistType(
    const IRType& type, const std::unordered_set<int>& ranks,
    int64_t batch_size, int rank) {
  auto base_type = type.getBaseType();
  switch (base_type) {
    case IRBaseType::SCALAR:
      return type;
    case IRBaseType::TENSOR: {
      const auto& dim = type.getTensorDim();
      if (dim.empty()) { // maybe loss
        return type;
      }
      return getDistTensorType(type, ranks, batch_size).at(rank);
    }
    case IRBaseType::LIST: {
      const auto list_type = type.getListType();
      if (list_type == IRListType::TENSOR) {
        std::vector<IRType> tensor_types;
        for (size_t i = 0; i < type.getListSize(); i++) {
          const auto dist_types = getDistTensorType(
              type.getCompoundTypes().at(i), ranks, batch_size);
          tensor_types.push_back(dist_types.at(rank));
        }
        return IRType::createTensorListType(tensor_types);
      } else if (list_type == IRListType::GENERIC) {
        std::vector<IRType> elem_types;
        for (const auto& et : type.getCompoundTypes()) {
          const auto dist_type = getDistType(et, ranks, batch_size, rank);
          elem_types.push_back(dist_type);
        }
        return IRType::createListType(elem_types);
      }
      return type;
    }
    case IRBaseType::TUPLE: {
      std::vector<IRType> elem_types;
      for (const auto& et : type.getCompoundTypes()) {
        const auto dist_type = getDistType(et, ranks, batch_size, rank);
        elem_types.push_back(dist_type);
      }
      return IRType::createTupleType(elem_types);
    }
    case IRBaseType::STRING:
    case IRBaseType::NONE:
    case IRBaseType::OPTIONAL:
    case IRBaseType::FUNCTION: {
      return type;
    }
  }
}

int64_t guessBatchSize(
    const IRType& type, const IRType& type_scaled, int64_t orig_batch_size) {
  switch (type.getBaseType()) {
    case IRBaseType::TENSOR: {
      const auto& dim = type.getTensorDim();
      const auto& dim_scaled = type_scaled.getTensorDim();
      assert(!dim.empty() && !dim_scaled.empty());
      return dim_scaled.front() / (dim.front() / orig_batch_size);
    }
    case IRBaseType::LIST:
    case IRBaseType::TUPLE: {
      std::vector<IRType> elem_types;
      for (int i = 0; i < type.getCompoundTypes().size(); i++) {
        const auto& t = type.getCompoundTypes().at(i);
        const auto& ts = type_scaled.getCompoundTypes().at(i);
        auto b = guessBatchSize(t, ts, orig_batch_size);
        if (b > 0) {
          return b;
        }
      }
    }
    case IRBaseType::SCALAR:
    case IRBaseType::STRING:
    case IRBaseType::NONE:
    case IRBaseType::OPTIONAL:
    case IRBaseType::FUNCTION:
      break;
  }
  return -1;
}

std::shared_ptr<IRGraph> scaleGraph(
    const std::shared_ptr<IRGraph>& graph, const std::string& name, int num,
    int index, int64_t batch_size) {
  std::vector<IRNode> nodes = graph->getNodes();

  std::unordered_set<int> dummy_ranks;
  for (int i = 0; i < num; i++) {
    dummy_ranks.insert(i);
  }

  int64_t new_batch_size = -1;

  std::unordered_map<std::string, IRValue> values;
  for (auto& v : graph->getValues()) {
    const auto ir_val = v.second;
    if (ir_val.isBatch()) {
      const auto& type = ir_val.getType();
      const auto dp_type = getDistType(type, dummy_ranks, batch_size, index);

      assert(ir_val.getBatchSize() == batch_size);

      if (new_batch_size <= 0) {
        new_batch_size = guessBatchSize(type, dp_type, batch_size);
      }

      assert(new_batch_size == guessBatchSize(type, dp_type, batch_size));

      IRValue batch_val(ir_val.getName(), dp_type);
      batch_val.setBatch(true, new_batch_size);
      values[v.first] = batch_val;
    } else {
      values[v.first] = ir_val;
    }
  }

  return std::make_shared<IRGraph>(
      name, graph->getNodes(), values, graph->getInputNames(),
      graph->getOutputNames(), new_batch_size);
}

PartitionDP replicate(
    const PartitionDP& partition,
    const std::unordered_map<std::string, int>& repl_nums, int pipeline_num,
    int64_t batch_size) {
  const auto logger = getLogger(LOGGER_NAME);

  PartitionDP rep_partition = partition;
  for (const auto& it : repl_nums) {
    assert(contains(rep_partition.subgraphs, it.first));
    const auto& g = rep_partition.subgraphs.at(it.first);
    int num = it.second;
    if (num > 1 && !g->isReplicable()) {
      logger->info(
          "Graph {} is not replicable. Replication num {} is ignored.",
          it.first, it.second);
      num = 1;
    }

    auto replica =
        scaleGraph(g, g->getName(), num * pipeline_num, 0, batch_size);
    replica->setReplicable(g->isReplicable());
    rep_partition.subgraphs[it.first] = replica;
  }
  rep_partition.replica_nums = repl_nums;
  return rep_partition;
}

bool isAllocFeasible(
    const std::unordered_map<int, size_t>& alloc_mem, size_t dev_mem) {
  for (const auto& it : alloc_mem) {
    if (dev_mem < it.second) {
      return false;
    }
  }
  return true;
}

struct AllocState {
  int dev;
  std::unordered_map<std::string, int> alloc;
  std::unordered_map<int, size_t> alloc_mem;

  size_t getMaxAllocMem() const {
    size_t max = 0;
    for (const auto& it : alloc_mem) {
      if (it.second > max) {
        max = it.second;
      }
    }
    return max;
  }
};

bool hasReplica(
    const std::string& sg_name, int dev,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        replications,
    const std::unordered_map<std::string, int>& alloc) {
  for (const auto& it : replications) {
    const auto& repl_set = it.second;
    if (contains(repl_set, sg_name)) {
      for (const auto& repl_id : repl_set) {
        if (contains(alloc, repl_id)) {
          if (alloc.at(repl_id) == dev) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

std::unordered_map<std::string, int> doSearchAllocation(
    std::vector<std::string> sg_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        replications,
    const std::unordered_map<std::string, int>& alloc,
    const std::unordered_map<int, size_t>& alloc_mem,
    const std::unordered_map<std::string, size_t>& graph_sizes,
    const std::vector<int>& devices, size_t dev_mem) {
  const auto logger = getLogger(LOGGER_NAME);

  if (!isAllocFeasible(alloc_mem, dev_mem)) {
    return std::unordered_map<std::string, int>();
  }
  if (sg_names.empty()) {
    return alloc;
  }

  const std::string sg_name = sg_names.back();
  sg_names.pop_back();

  std::vector<AllocState> alloc_states;
  for (int dev : devices) {
    if (hasReplica(sg_name, dev, replications, alloc)) {
      continue;
    }

    auto _alloc = alloc;
    auto _alloc_mem = alloc_mem;

    _alloc[sg_name] = dev;
    _alloc_mem[dev] += graph_sizes.at(sg_name);
    AllocState state = {dev, _alloc, _alloc_mem};
    alloc_states.push_back(state);
  }

  std::sort(
      alloc_states.begin(), alloc_states.end(),
      [](const AllocState& s1, const AllocState& s2) {
        return s1.getMaxAllocMem() < s2.getMaxAllocMem();
      });

  for (const auto& st : alloc_states) {
    const auto ret_alloc = doSearchAllocation(
        sg_names, replications, st.alloc, st.alloc_mem, graph_sizes, devices,
        dev_mem);
    if (!ret_alloc.empty()) {
      return ret_alloc;
    }
  }

  return std::unordered_map<std::string, int>();
}

std::unordered_map<std::string, std::unordered_set<int>> searchAllocation(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem) {
  int repl_num = -1;
  for (const auto& it : partition.replica_nums) {
    repl_num = it.second;
  }

  std::unordered_map<std::string, std::unordered_set<int>> ret;
  int i = 0;
  for (size_t repl_idx = 0; repl_idx < repl_num; repl_idx++) {
    for (const auto& it : partition.subgraphs) {
      ret[it.first].insert(i % dev_count);
      i++;
    }
  }
  return ret;

  //        std::unordered_map<std::string, int> alloc;
  //        std::vector<std::string> sg_names = keys(partition.subgraphs);
  //
  //        std::unordered_map<int, size_t> alloc_mem;
  //        for (size_t i=1; i<=dev_count; i++) {
  //            alloc_mem[i] = 0;
  //        }
  //        std::unordered_map<std::string, size_t> graph_sizes;
  //        for (const auto& sg: partition.subgraphs) {
  //            graph_sizes[sg.first] = getSizeInByte(sg.second);
  //        }
  //
  //        std::vector<int> devices;
  //        for (size_t i=1; i<=dev_count; i++) {
  //            devices.push_back(i);
  //        }
  //
  //        return doSearchAllocation(sg_names, partition.replications, alloc,
  //        alloc_mem, graph_sizes, devices, dev_mem);
}

std::unordered_map<std::string, std::unordered_set<int>> searchAllocationFlat(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem) {
  const auto repl_nums = values(partition.replica_nums);
  int common_repl_num = gcd(repl_nums);

  int dev_idx = 0;
  std::unordered_map<std::string, std::unordered_set<int>> ret;
  for (int i = 0; i < common_repl_num; i++) {
    for (const auto& it : partition.replica_nums) {
      for (int j = 0; j < (it.second / common_repl_num); j++) {
        ret[it.first].insert(dev_idx);
        dev_idx++;
      }
    }
  }

  return ret;
}

std::unordered_map<std::string, std::unordered_set<int>> searchAllocationSimple(
    const PartitionDP& partition, size_t dev_count, size_t dev_mem) {
  std::vector<std::string> sorted_gid = keys(partition.subgraphs);
  std::sort(
      sorted_gid.begin(), sorted_gid.end(),
      [&partition](const std::string& g1, const std::string& g2) {
        assert(contains(partition.replica_nums, g1));
        assert(contains(partition.replica_nums, g2));
        return partition.replica_nums.at(g1) > partition.replica_nums.at(g2);
      });

  std::unordered_map<std::string, std::unordered_set<int>> ret;
  int dev_idx = 0;
  for (const auto& gid : sorted_gid) {
    for (int i = 0; i < partition.replica_nums.at(gid); i++) {
      ret[gid].insert(dev_idx);
      dev_idx++;
    }
  }

  return ret;
}

HGraph toHGraph(const PartitionDP& partition) {
  HGraph hg;
  std::unordered_map<std::string, HVertex> node_map;

  for (const auto& it : partition.subgraphs) {
    auto v = boost::add_vertex(hg);
    hg[v] = it.second;
    node_map[it.second->getName()] = v;
  }

  for (const auto& con : partition.connections) {
    assert(contains(node_map, con.src));
    assert(contains(node_map, con.dest));
    boost::add_edge(node_map.at(con.src), node_map.at(con.dest), hg);
  }

  return hg;
}

std::unordered_map<std::string, std::unordered_set<int>>
searchAllocationFitToDevice(const PartitionDP& partition) {
  HGraph hg = toHGraph(partition);
  std::unordered_map<std::string, std::unordered_set<int>> ret;

  int graph_idx = 0;
  for (const auto& v : all_nodes_topo<HVertex, HGraph>(hg)) {
    const auto& sg = hg[v];
    assert(contains(partition.replica_nums, sg->getName()));
    int repl_num = partition.replica_nums.at(sg->getName());

    for (int i = 0; i < repl_num; i++) {
      ret[sg->getName()].insert(i * partition.subgraphs.size() + graph_idx);
    }
    graph_idx++;
  }
  return ret;
}

// Fix ranks of nodes and values.
// If a value is NOT a batch, replicates the value on the ranks of following
// ops.
void fixNonBatchRanks(BGraph& g) {
  std::vector<Vertex> rev_topo_vertices;
  boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

  // check nodes and values in a reversed topological sorted order
  for (auto& v : rev_topo_vertices) {
    if (g[v].type == VALUE) {
      if (g[v].value.isBatch()) {
        //                    spdlog::info("Value {} {} is batch. ranks: {}",
        //                    g[v].name, g[v].id,
        //                                 join_as_str(g[v].ranks));
        continue;
      }

      std::unordered_set<int> fixed_ranks;
      std::vector<Vertex> out_ops = target_nodes(v, g);
      for (auto& op : out_ops) {
        for (int op_rank : g[op].ranks) {
          fixed_ranks.insert(op_rank);
        }
      }

      //                spdlog::info("Value {} {} changed ranks: {} to {}",
      //                g[v].name, g[v].id,
      //                        join_as_str(g[v].ranks),
      //                             join_as_str(fixed_ranks));
      if (!fixed_ranks.empty()) {
        g[v].ranks = fixed_ranks;
      }
    } else {
      if (g[v].node.isBatch()) {
        //                    spdlog::info("Node {} {} is batch. ranks: {}",
        //                    g[v].name, g[v].id,
        //                                 join_as_str(g[v].ranks));
        continue;
      }

      std::unordered_set<int> fixed_ranks;
      std::vector<Vertex> out_vals = target_nodes(v, g);
      std::vector<std::string> output_names;
      for (auto& val : out_vals) {
        for (int val_rank : g[val].ranks) {
          fixed_ranks.insert(val_rank);
        }
        output_names.push_back(g[val].name);
      }

      //      spdlog::info(
      //          "Node {} {} (out={}) changed ranks: {} to {}", g[v].name,
      //          g[v].id, join_as_str(output_names), join_as_str(g[v].ranks),
      //          join_as_str(fixed_ranks));

      if (!fixed_ranks.empty()) {
        g[v].ranks = fixed_ranks;
      }
    }
  }
}

std::vector<size_t> splitByValueSizes(const BGraph& g, int n_partition) {
  std::vector<Vertex> topo_vertices, rev_topo_vertices;
  boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

  boost::for_each(
      rev_topo_vertices | boost::adaptors::reversed,
      [&topo_vertices](Vertex v) { topo_vertices.push_back(v); });
  size_t total_size = 0;
  for (const auto& v : topo_vertices) {
    if (g[v].type == VALUE) {
      total_size += g[v].value.getSizeInByte();
    }
  }

  // initial decomposition
  size_t current_size = 0;
  size_t index = 0;
  std::vector<size_t> split_indices;
  for (const auto& v : topo_vertices) {
    if (g[v].type == VALUE) {
      current_size += g[v].value.getSizeInByte();
    }
    if (current_size > (total_size / n_partition)) {
      split_indices.push_back(index);
      current_size = 0;
    }
    index++;
  }
  split_indices.push_back(index);

  return split_indices;
}

void setRanksOnGraph(BGraph& g, const std::vector<size_t>& split) {
  std::vector<Vertex> topo_vertices, rev_topo_vertices;
  boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

  boost::for_each(
      rev_topo_vertices | boost::adaptors::reversed,
      [&topo_vertices](Vertex v) { topo_vertices.push_back(v); });

  for (const auto& v : topo_vertices) {
    g[v].ranks.clear();
  }

  int rank = 0;
  size_t index = 0;
  for (const auto& v : topo_vertices) {
    if (split.size() > rank && index >= split.at(rank) && g[v].type == NODE) {
      rank++;
    }
    g[v].ranks.insert(rank);
    index++;

    //            spdlog::info("@setRanksOnGraph {} ranks={}", g[v].name,
    //            join_as_str(g[v].ranks));
  }
}

void verifyDeployment(const Deployment& deployment) {
  std::vector<RouteDP> fwd_routes;
  fwd_routes = addAll(fwd_routes, deployment.fwd_in_routes);
  fwd_routes = addAll(fwd_routes, deployment.fwd_routes);
  fwd_routes = addAll(fwd_routes, deployment.fwd_out_routes);
  std::vector<RouteDP> bwd_routes;
  bwd_routes = addAll(bwd_routes, deployment.bwd_in_routes);
  bwd_routes = addAll(bwd_routes, deployment.bwd_routes);
  bwd_routes = addAll(bwd_routes, deployment.bwd_out_routes);

  for (const auto& it : deployment.subgraphs) {
    const auto& graph_name = it.first;
    const auto& ir_graph = it.second;

    for (const auto& in_name : ir_graph->getInputNames()) {
      const auto& in_val = ir_graph->getValue(in_name);
      if (in_val.isParam()) {
        continue;
      }

      bool used_fwd = false;
      for (const auto& r : fwd_routes) {
        if (graph_name == r.dest_graph &&
            r.ir_value.getName() == in_val.getName()) {
          used_fwd = true;
        }
      }
      if (!used_fwd) {
        std::stringstream ss;
        ss << "No fwd route for input: " << in_name << " in " << graph_name;
        throw std::runtime_error(ss.str());
      }

      if (passedForBackward(in_val.getType())) {
        bool used_bwd = false;
        for (const auto& r : bwd_routes) {
          if (graph_name == r.source_graph &&
              r.ir_value.getName() == in_val.getName()) {
            used_bwd = true;
          }
        }
        if (!used_bwd) {
          std::stringstream ss;
          ss << "No bwd route for input: " << in_name << " in " << graph_name;
          throw std::runtime_error(ss.str());
        }
      }
    }

    for (const auto& out_name : ir_graph->getOutputNames()) {
      const auto& out_val = ir_graph->getValue(out_name);

      bool used_fwd = false;
      for (const auto& r : fwd_routes) {
        if (graph_name == r.source_graph &&
            r.ir_value.getName() == out_val.getName()) {
          used_fwd = true;
        }
      }
      if (!used_fwd) {
        std::stringstream ss;
        ss << "No fwd route for output: " << out_name << " in " << graph_name;
        throw std::runtime_error(ss.str());
      }

      if (passedForBackward(out_val.getType())) {
        bool used_bwd = false;
        for (const auto& r : bwd_routes) {
          if (graph_name == r.dest_graph &&
              r.ir_value.getName() == out_val.getName()) {
            used_bwd = true;
          }
        }
        if (!used_bwd) {
          std::stringstream ss;
          ss << "No bwd route for output: " << out_name << " in " << graph_name;
          throw std::runtime_error(ss.str());
        }
      }
    }
  }
}

Partition createPartition(
    const std::shared_ptr<IRGraph>& ir_graph,
    const std::vector<std::shared_ptr<IRGraph>>& subgraphs) {
  // value name -> graph id
  // a value can be an input of one or more graphs
  std::unordered_map<std::string, std::unordered_set<std::string>> in_vals;
  // Only one graph produces a value
  std::unordered_map<std::string, std::string> out_vals;
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> sg_map;
  std::vector<std::string> sg_order;

  // graph id -> input value names
  std::unordered_map<std::string, std::unordered_set<std::string>> created_cons;

  for (const auto& sg : subgraphs) {
    for (const auto& in : sg->getInputNames()) {
      if (!sg->getValue(in).isParam()) {
        in_vals[in].insert(sg->getName());
      }
    }
    for (const auto& out : sg->getOutputNames()) {
      if (!contains(ir_graph->getInputNames(), out)) {
        out_vals[out] = sg->getName();
      }
    }

    sg_map[sg->getName()] = sg;
    sg_order.push_back(sg->getName());
  }

  std::vector<GraphConnection> connections;
  for (const auto& sg : subgraphs) {
    for (const auto& in : sg->getInputNames()) {
      if (!sg->getValue(in).isParam()) {
        if (contains(out_vals, in)) {
          const auto& src_id = out_vals.at(in);
          const auto& tgt_id = sg->getName();
          if (src_id != tgt_id) {
            GraphConnection con{
                in, src_id, tgt_id, src_id + "_" + in, tgt_id + "_" + in};
            connections.push_back(con);

            created_cons[tgt_id].insert(in);
          }
        }
      }
    }
  }

  for (const auto& in : ir_graph->getInputNames()) {
    if (!ir_graph->getValue(in).isParam()) {
      if (!contains(in_vals, in)) {
        // unused input
        continue;
      }
      for (const auto& tgt_id : in_vals.at(in)) {
        if (!contains(created_cons[tgt_id], in)) {
          ;
          GraphConnection con{
              in, "MASTER", tgt_id, "MASTER_" + in, tgt_id + "_" + in};
          connections.push_back(con);
          created_cons[tgt_id].insert(in);
        }
      }
    }
  }

  for (const auto& out : ir_graph->getOutputNames()) {
    assert(contains(out_vals, out));
    const auto& src_id = out_vals.at(out);
    GraphConnection con{
        out, src_id, "MASTER", src_id + "_" + out, "MASTER_" + out};
    connections.push_back(con);
  }

  return Partition{
      ir_graph->getName(), ir_graph, sg_map, connections, sg_order};
}

std::ostream& operator<<(std::ostream& os, const Deployment& deployment) {
  os << "id: " << deployment.id << std::endl;
  os << " pipeline_num: " << deployment.pipeline_num
     << " checkpointing: " << deployment.checkpointing << std::endl;

  os << " graph: " << *deployment.graph;
  os << " subgraphs: " << std::endl;
  int idx = 0;
  for (const auto& sg_name : deployment.fwd_graph_order) {
    assert(contains(deployment.subgraphs, sg_name));
    const auto graph = deployment.subgraphs.at(sg_name);
    os << "  order: " << idx << " " << *graph;
    os << "  allocation: " << join_as_str(deployment.allocation.at(sg_name))
       << std::endl;

    const auto& part_info = deployment.part_info.at(sg_name);
    os << "  rank_vals: " << std::endl;
    for (const auto& rv : part_info.rank_values) {
      os << "   " << rv.first << "=" << rv.second << std::endl;
    }
  }

  for (const auto& r : deployment.fwd_in_routes) {
    os << " fwd_in_routes: " << r << std::endl;
  }
  for (const auto& r : deployment.fwd_routes) {
    os << " fwd_routes: " << r << std::endl;
  }
  for (const auto& r : deployment.fwd_out_routes) {
    os << " fwd_out_routes: " << r << std::endl;
  }
  for (const auto& r : deployment.bwd_in_routes) {
    os << " bwd_in_routes: " << r << std::endl;
  }
  for (const auto& r : deployment.bwd_routes) {
    os << " bwd_routes: " << r << std::endl;
  }
  for (const auto& r : deployment.bwd_out_routes) {
    os << " bwd_out_routes: " << r << std::endl;
  }

  return os;
}

PartitioningConf makePartitioningConf(
    int dev_num, size_t batch_size, size_t dev_mem, bool use_amp_master_params,
    bool enable_zero, bool offload_params) {
  config::Config& conf = config::Config::get();

  PartitioningConf part_conf;
  part_conf.dev_num = dev_num;
  part_conf.batch_size = batch_size;
  part_conf.dev_mem = dev_mem;
  part_conf.opt_param_factor = conf.getVal<int>(config::OPT_PARAM_FACTOR);
  part_conf.use_amp_master_params = use_amp_master_params;
  part_conf.enable_zero = enable_zero;
  part_conf.offload_params = offload_params;
  part_conf.force_dist_matmul = conf.getVal<bool>(config::FORCE_DIST_MATMUL);
  part_conf.min_pipeline_num = conf.getVal<int>(config::MIN_PIPELINE);
  part_conf.max_pipeline_num = conf.getVal<int>(config::MAX_PIPELINE);
  part_conf.cfg_pipeline_num = conf.getVal<int>(config::PIPELINE_NUM);
  part_conf.min_partition_num = conf.getVal<int>(config::MIN_PARTITION_NUM);
  part_conf.max_partition_num = conf.getVal<int>(config::MAX_PARTITION_NUM);
  part_conf.cfg_stage_num = conf.getVal<int>(config::PARTITION_NUM);
  return part_conf;
}
} // namespace rannc
