//
// Created by Masahiro Tanaka on 2020/02/19.
//

#include <random>
#include <cuda/CudaUtil.h>
#include "Partitioner.h"

namespace rannc {

    long getSize(const std::vector<IRValue>& vals) {
        long size = 0;
        for (const auto& v: vals) {
            size += v.getSizeInByte();
        }
        return size;
    }

    long MLPartitioner::eval(const GraphProfile& prof) {
        if (coarsen_by_time_) {
            return prof.fwd_time + prof.bwd_time;
        }
        return prof.max_allocated_mem;
    }

    GraphProfile MLPartitioner::profile(const std::shared_ptr<IRGraph> &g) {
        int repl_num = dev_num_*max_pipeline_num_;
        if (min_pipeline_bs_ > 0) {
            repl_num = std::min((int) (batch_size_ / min_pipeline_bs_), repl_num);
        }
        return prof_util_.profile(g, batch_size_, repl_num);
    }

    std::vector<MLVertex> MLPartitioner::sortNodesByEval(std::vector<MLVertex> nodes, const MLBGraph &bg) {
        // To reduce time for profiling, we set replica num to dev_num_
        std::stable_sort(nodes.begin(), nodes.end(), [&bg, this](const MLVertex& x, const MLVertex& y) {
            return eval(this->profile(bg[x].graph)) < eval(this->profile(bg[y].graph));
        });
        return nodes;
    }

    MLGraph MLPartitioner::mergeAdjacents(const MLGraph& ml_graph, bool skip_profiling, int min_partition_num) {

        MLBGraph bg = toBGL(ml_graph);
        std::vector<MLVertex> nodes_sort = all_nodes<MLVertex, MLBGraph>(bg);

        // Sort nodes by eval values because random shuffling significantly increases communication time
        nodes_sort = sortNodesByEval(nodes_sort, bg);

        long eval_sum = 0;
        for (const auto& bv: nodes_sort) {
            const auto& v = bg[bv];
            eval_sum += eval(profile(v.graph));
        }
        long eval_ave = eval_sum / nodes_sort.size();
        long eval_var = 0;
        for (const auto& bv: nodes_sort) {
            const auto& v = bg[bv];
            long d = eval(profile(v.graph)) - eval_ave;
            eval_var += d * d;
        }
        long eval_sigma = std::sqrt(eval_var / nodes_sort.size());

        std::unordered_set<MLVertex> matched;
        std::vector<MLNode> merged_nodes;
        std::unordered_map<std::string, std::string> name_map;

        for (const auto& bv: nodes_sort) {
            if (contains(matched, bv)) {
                continue;
            }

            const auto& v = bg[bv];

            long best_imprv = -1;
            MLVertex best_adj;
            MLNode best_merged = liftUp(v);

            std::vector<MLVertex> targets;
            if (nodes_sort.size() - matched.size() + merged_nodes.size() > min_partition_num) {
                targets = target_nodes<MLVertex, MLBGraph>(bv, bg);
            }

            for (const auto &b_adj: targets) {
                const auto& adj = bg[b_adj];

                if (contains(matched, b_adj)) {
//                    spdlog::info("v={} adj={}: adj is already matched", v.id, adj.id);
                    continue;
                }

                if (!isConvex(bv, b_adj, bg)) {
//                    spdlog::info("v={} adj={}: merged graph is not convex", v.id, adj.id);
                    continue;
                }

                const auto cut_values = getCutValues(v, adj);
                long cut_size = getSize(cut_values);
                long comm_time = calcCommTime(cut_size / (dev_num_*max_pipeline_num_));

                long eval_v = 0, eval_adj = 0, eval_merged = 0;

                MLNode merged = merge(v, adj, ml_graph.nodes, ml_graph.edges);

//                std::stringstream ss;
//                ss << "id_v=" << v.id << " v=" << *v.graph
//                   << "id_adj=" << adj.id << " adj=" << *adj.graph
//                   << "id_merged=" << merged.id << " merged=" << *merged.graph;
//                spdlog::info("merge {}", ss.str());

                eval_v = eval(profile(v.graph));
                eval_adj = eval(profile(adj.graph));

                // We do not merge nodes if the estimated time is extremely long
                bool skip_merge = false;
                if (skip_profiling) {
                    skip_merge = (eval_v > eval_ave + eval_sigma*2)
                                || (eval_adj > eval_ave + eval_sigma*2);
                } else {
                    const auto& prof_merged = profile(merged.graph);
                    if (!fitToMem(merged.graph, prof_merged, dev_mem_, use_amp_master_params_)) {
//                    spdlog::info("v={} adj={}: merged graph does not fit to mem. mem={}", v.id, adj.id, prof_merged.max_allocated_mem);
                        continue;
                    }
                    eval_merged = eval(prof_merged);
                    skip_merge = eval_merged > eval_ave + eval_sigma*2;
                }

//                std::stringstream ss;
//                ss << "id_v=" << v.id << "eval_v=" << eval_v << " v=" << *v.graph
//                   << "id_adj=" << adj.id << " eval_adj=" << eval_adj << " adj=" << *adj.graph
//                   << "id_merged=" << merged.id << " eval_merged=" << eval_merged << " merged=" << *merged.graph
//                   << " eval_ave=" << eval_ave << " eval_sigma=" << eval_sigma
//                   << " update=" << skip_merge
//                   << std::endl;
//                spdlog::info(ss.str());

                if (!skip_merge) {
                    long imprv = eval_v + eval_adj + comm_time - eval_merged;

//                spdlog::info("v={} adj={} eval_v={} eval_adj={} eval_merged={} comm={} improve={}",
//                             v.id, adj.id, eval_v, eval_adj, eval_merged, comm_time, imprv);

                    if (best_imprv <= imprv) {
//                    spdlog::info("UPDATED v={} best_imprv={}", v.id, adj.id, imprv);

                        best_adj = b_adj;
                        best_merged = merged;
                        best_imprv = imprv;
                    }
                }
//                spdlog::info("adj {} -> {} cut={} comm_time={} imprv={}",
//                        v.id, adj.id, cut_size, comm_time, imprv);
            }
            merged_nodes.push_back(best_merged);
            matched.insert(bv);
            name_map[v.id] = best_merged.id;

            if (best_imprv >= 0) {
                matched.insert(best_adj);
                name_map[bg[best_adj].id] = best_merged.id;
//                spdlog::info("Merged best adj {}->{} imprv={}", v.id, bg[best_adj].id, best_imprv);
            } else {
//                spdlog::info("No merge {} as {}", v.id, best_merged.id);
            }
        }

        std::vector<MLEdge> merged_edges = mergeEdges(ml_graph.edges, name_map);
        return MLGraph{merged_nodes, merged_edges};
    }

    MLGraph MLPartitioner::coarsen(const MLGraph& ml_graph, int min_partition_num) {
        MLGraph last_graph;
        MLGraph c_graph = ml_graph;

        while (last_graph.nodes.size() != c_graph.nodes.size()) {
            last_graph = c_graph;

            // for debugging
            if (last_graph.nodes.size() <= 3) {
                break;
            }

//            std::stringstream ss;
//            ss << "coarsened_graph_L" << c_graph.getLevel() << ".dot";
//            std::ofstream c_file(ss.str());
//            MLBGraph cg = toBGL(c_graph);
//            boost::write_graphviz(c_file, cg, MLGraphLabelWriter(cg));

            bool skip_profiling = c_graph.nodes.size() > 200;
            c_graph = mergeAdjacents(c_graph, skip_profiling, min_partition_num);
            logger->trace("Coarsened graph. level={} #nodes={} #cut_size={} skip_profiling={}", c_graph.getLevel(),
                    c_graph.nodes.size(), sumEdgeSizes(c_graph.edges), skip_profiling);

            if (c_graph.nodes.size() <= min_partition_num) {
                last_graph = c_graph;
                break;
            }
        }
        logger->trace("Coarsening finished");

        return last_graph;
    }

    // Remove g2's values and ops from g1
    std::shared_ptr<IRGraph> MLPartitioner::removeTailSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2) {

        const auto g = doRemoveSubgraph(g1, g2);

        auto outputs = g->getOutputNames();
        for (const auto& o: g2->getInputNames()) {
            if (!contains(outputs, o) && !g2->getValue(o).isParam()
                    && contains(g->getValues(), o)) {
                outputs.push_back(o);
            }
        }
        return removeUnusedNodes(std::make_shared<IRGraph>(g->getName(), g->getNodes(), g->getValues(),
                                          g->getInputNames(), outputs));
    }

    std::shared_ptr<IRGraph> MLPartitioner::removeHeadSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2) {

        const auto g = doRemoveSubgraph(g1, g2);

        std::vector<std::string> inputs;
        for (const auto& in_name: g->getInputNames()) {
            const auto& in_val = g->getValue(in_name);
            if (!in_val.isParam()) {
                if (!contains(inputs, in_name)) {
                    inputs.push_back(in_name);
                }
            }
        }

        std::unordered_set<std::string> required_val_names;
        for (const auto& n: g->getNodes()) {
            for (const auto& i: n.getInputNames()) {
                required_val_names.insert(i);
            }
        }

        for (const auto& out_name: g2->getOutputNames()) {
            if (contains(required_val_names, out_name)) {
                if (!contains(inputs, out_name)) {
                    inputs.push_back(out_name);
                }
            }
        }

        for (const auto& in_name: g->getInputNames()) {
            const auto& in_val = g->getValue(in_name);
            if (in_val.isParam()) {
                inputs.push_back(in_name);
            }
        }

        assert(inputs.size() == vectorToSet(inputs).size());

        return removeUnusedNodes(std::make_shared<IRGraph>(g->getName(), g->getNodes(), g->getValues(),
                                          inputs, g->getOutputNames()));
    }

    std::shared_ptr<IRGraph> MLPartitioner::doRemoveSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2) {

        std::unordered_set<std::string> removed_nodes;
        // List nodes to remove
        for (const auto& n: g2->getNodes()) {
            removed_nodes.insert(n.getId());
        }

        // List *values* to keep
        // Keep values that are inputs/ouputs of remaining (op) nodes
        std::vector<IRNode> nodes;
        std::unordered_set<std::string> val_names;
        for (const auto& n: g1->getNodes()) {
            if (!contains(removed_nodes, n.getId())) {
                nodes.push_back(n);
                for (const auto& in_name: n.getInputNames()) {
                    val_names.insert(in_name);
                }
                for (const auto& out_name: n.getOutputNames()) {
                    val_names.insert(out_name);
                }
            }
        }

        std::unordered_map<std::string, IRValue> values;
        for (const auto& it: g1->getValues()) {
            if (contains(val_names, it.first)) {
                values[it.first] = it.second;
            }
        }

        std::vector<std::string> input_names;
        for (const auto& in_name: g1->getInputNames()) {
            if (contains(val_names, in_name)) {
                input_names.push_back(in_name);
            }
        }

        std::vector<std::string> output_names;
        for (const auto& out_name: g1->getOutputNames()) {
            if (contains(val_names, out_name)) {
                output_names.push_back(out_name);
            }
        }

        return std::make_shared<IRGraph>(generateName("REMOVED_"), nodes,
                                                  values, input_names, output_names);
    }

    // src_node and tgt_node are nodes at level L,
    // moved_node is a node at level L-1
    MoveResult MLPartitioner::move_to_tgt(const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
                                          const MLEdge& sub_edge, const MLNode& moved_node,
                                          const std::unordered_set<std::string>& required_inputs) {

//        spdlog::info("move_to_tgt {}->{} move_graph={} required={}", src_node.id, tgt_node.id,
//                moved_node.graph->getName(), join_as_str(required_inputs));

        std::shared_ptr<IRGraph> new_src_graph = removeTailSubgraph(src_node.graph, moved_node.graph);
        std::shared_ptr<IRGraph> new_tgt_graph = merge(moved_node.graph, tgt_node.graph, required_inputs);

        std::vector<MLNode> new_src_elem_nodes;
        for (const auto& elem: src_node.sub_nodes) {
            if (elem.id != moved_node.id) {
                new_src_elem_nodes.push_back(elem);
            }
        }

        // Remove edges in src_node @L-1
        std::vector<MLEdge> new_src_edges;
        std::vector<MLEdge> new_sub_edges; // edges to add @L
        for (const auto& se: src_node.sub_edges) {
            if (se.tgt_id == moved_node.id) {
                new_sub_edges.push_back(se);
            } else {
                new_src_edges.push_back(se);
            }
        }
        MLNode new_src_node{src_node.id, new_src_graph, new_src_elem_nodes, new_src_edges};

        std::vector<MLNode> new_tgt_elem_graphs;
        new_tgt_elem_graphs.push_back(moved_node);
        for (const auto& elem: tgt_node.sub_nodes) {
            new_tgt_elem_graphs.push_back(elem);
        }

        // Add edges in tgt_node @L-1
        std::vector<MLEdge> new_tgt_edges = tgt_node.sub_edges;
        new_tgt_edges.push_back(sub_edge);
        MLNode new_tgt_node{tgt_node.id, new_tgt_graph, new_tgt_elem_graphs, new_tgt_edges};

        // Fix edge @L
        for (const auto& se: edge.sub_edges) {
            if (se.src_id != moved_node.id) {
                new_sub_edges.push_back(se);
            }
        }
        MLEdge new_edge{edge.src_id, edge.tgt_id, new_sub_edges, {}};

        return MoveResult{new_src_node, new_tgt_node, new_edge};
    }

    MoveResult MLPartitioner::move_to_src(const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
                                          const MLEdge& sub_edge, const MLNode& moved_node,
                                          const std::unordered_set<std::string>& required_inputs) {

//        spdlog::info("move_to_src {}->{} moved_node.id={} move_graph={} required={}", tgt_node.id, src_node.id,
//                     moved_node.id, moved_node.graph->getName(), join_as_str(required_inputs));

        std::shared_ptr<IRGraph> new_src_graph = merge(src_node.graph, moved_node.graph, required_inputs);
        std::shared_ptr<IRGraph> new_tgt_graph = removeHeadSubgraph(tgt_node.graph, moved_node.graph);

        std::vector<MLNode> new_src_elem_graphs;
        for (const auto& elem: src_node.sub_nodes) {
            new_src_elem_graphs.push_back(elem);
        }
        new_src_elem_graphs.push_back(moved_node);

        std::vector<MLEdge> new_src_edges = src_node.sub_edges;
        new_src_edges.push_back(sub_edge);
        MLNode new_src_node{src_node.id, new_src_graph, new_src_elem_graphs, new_src_edges};

        std::vector<MLNode> new_tgt_elem_graphs;
        for (const auto& elem: tgt_node.sub_nodes) {
            if (elem.id != moved_node.id) {
                new_tgt_elem_graphs.push_back(elem);
            }
        }
        std::vector<MLEdge> new_tgt_edges;
        std::vector<MLEdge> new_sub_edges; // edges to add @L
        for (const auto& se: tgt_node.sub_edges) {
            if (se.src_id == moved_node.id) {
                new_sub_edges.push_back(se);
            } else {
                new_tgt_edges.push_back(se);
            }
        }
        MLNode new_tgt_node{tgt_node.id, new_tgt_graph, new_tgt_elem_graphs, new_tgt_edges};

        for (const auto& se: edge.sub_edges) {
            if (se.tgt_id != moved_node.id) {
                new_sub_edges.push_back(se);
            }
        }
        MLEdge new_edge{edge.src_id, edge.tgt_id, new_sub_edges, {}};

        return MoveResult{new_src_node, new_tgt_node, new_edge};
    }

    struct NodeMove {
        std::string src_id; // level L
        MLEdge edge; // level L
        std::string tgt_id; // level L
        MLNode moved_node; // level L-1
        MLEdge sub_edge; // level L-1
        bool direction; // move src to tgt if true
    };

    MLGraph MLPartitioner::adjustBoundaries(const MLGraph& ml_graph) {
        std::unordered_map<std::string, MLNode> node_map;
        std::unordered_map<std::string, MLNode> lower_node_map;
        std::vector<MLEdge> all_sub_edges;

        for (const auto& n: ml_graph.nodes) {
            node_map[n.id] = n;
            for (const auto& sn: n.sub_nodes) {
                lower_node_map[sn.id] = sn;
            }

            if (!verify(n)) {
                spdlog::info("Node verification failed: {} {}", n.id, dumpMLNode(n));
                throw std::runtime_error("Verification failed");
            }
        }

        MLEdgeMap edge_map;
        for (const auto& e: ml_graph.edges) {
            MLEdgeKey k{e.src_id, e.tgt_id};
            edge_map[k] = e;

            for (const auto& se: e.sub_edges) {
                all_sub_edges.push_back(se);
            }
        }

        // List possible choices of movements
        std::unordered_map<std::string, std::vector<NodeMove>> move_choices;
        for (const auto& e: ml_graph.edges) {
            assert(contains(node_map, e.src_id));
            assert(contains(node_map, e.tgt_id));

            const MLNode& src_node = node_map.at(e.src_id);
            const MLNode& tgt_node = node_map.at(e.tgt_id);

//            spdlog::info("doUncoarsen checking edge: src_node={} #src_elems={} tgt_node={} #tgt_elems={}",
//                         src_node.id, src_node.sub_nodes.size(), tgt_node.id, tgt_node.sub_nodes.size());

            for (const auto& se: e.sub_edges) { // lower level edges
//                spdlog::info("Move candidate src: {}", se.src_id);
                assert(contains(lower_node_map, se.src_id));
//                spdlog::info("Move candidate tgt: {}", se.tgt_id);
                assert(contains(lower_node_map, se.tgt_id));

                const MLNode& moved_node_src = lower_node_map.at(se.src_id);
                move_choices[moved_node_src.id].push_back(
                        NodeMove{src_node.id, e, tgt_node.id, moved_node_src, se, true});

                const MLNode& moved_node_tgt = lower_node_map.at(se.tgt_id);
                move_choices[moved_node_tgt.id].push_back(
                        NodeMove{src_node.id, e, tgt_node.id, moved_node_tgt, se, false});
            }
        }

        std::vector<std::string> move_cand_ids = keys(move_choices);
//        std::vector<std::string> move_cand_ids = shuffle(keys(move_choices));
        std::vector<MLNode> all_nodes = ml_graph.nodes;
        for (const auto& move_node_id: move_cand_ids) {

//            spdlog::info("Checking node move: move_node_id={}", move_node_id);
            const auto& possible_moves =  move_choices.at(move_node_id);
            assert(!possible_moves.empty());

            MoveResult best_moved_result;
            long best_imprv = 0;

            for (const auto& move: possible_moves) {
                assert(contains(node_map, move.src_id));
                const MLNode& src_node = node_map.at(move.src_id);
                assert(contains(node_map, move.tgt_id));
                const MLNode& tgt_node = node_map.at(move.tgt_id);

                if (move.direction) {
                    if (src_node.sub_nodes.size() < 2
                        || countRefTgtInSubgraph(src_node, move.moved_node) != 0
                        || countRefInEdgesSrc(move.moved_node, all_sub_edges) > 1
                        || countRefInEdgesTgt(move.moved_node, all_sub_edges) != 0) {
                        continue;
                    }
                } else {
                    if (tgt_node.sub_nodes.size() < 2
                        || countRefSrcInSubgraph(tgt_node, move.moved_node) != 0
                        || countRefInEdgesTgt(move.moved_node, all_sub_edges) > 1
                        || countRefInEdgesSrc(move.moved_node, all_sub_edges) != 0) {
                        continue;
                    }
                }

                MLEdgeKey k{move.edge.src_id, move.edge.tgt_id};
                // get current edge@L
                // An edge can be modified during this iteration
                MLEdge edge = edge_map.at(k);

                // The edge@L may not contain subedge from/to moved_node anymore
                if (!containsSubedge(edge, move.sub_edge)) {
                    continue;
                }

                MoveResult moved_result;
                if (move.direction) {
                    if (!containsSubgraph(src_node, move.moved_node.id)) {
                        continue;
                    }
                    moved_result = move_to_tgt(src_node, edge, tgt_node, move.sub_edge, move.moved_node,
                            getRequiredInputs(src_node, tgt_node, all_nodes));
                } else {
                    if (!containsSubgraph(tgt_node, move.moved_node.id)) {
                        continue;
                    }
                    moved_result = move_to_src(src_node, edge, tgt_node, move.sub_edge, move.moved_node,
                                               getPreservedOutputs(src_node.id, move.moved_node.graph->getName(), all_nodes));
                }

                if (!verifyNodeInputs(moved_result.src_node.graph) || !verifyNodeInputs(moved_result.tgt_node.graph)) {
                    continue;
                }

                if (!verify(moved_result.src_node) || !verify(moved_result.tgt_node)
                || !verifyNoDuplicatedOutputs(moved_result.src_node.graph)
                || !verifyNoDuplicatedOutputs(moved_result.tgt_node.graph)
                || !verifyNodeInputs(moved_result.src_node.graph, true)
                || !verifyNodeInputs(moved_result.tgt_node.graph, true)
                || !noUnusedValue(moved_result.src_node.graph, true)
                || !noUnusedValue(moved_result.tgt_node.graph, true)) {

                    spdlog::info("Node verification failed after move: direction={} SRC {}={} TGT {}={}",
                                 move.direction,
                                 moved_result.src_node.id, verify(moved_result.src_node),
                                 moved_result.tgt_node.id, verify(moved_result.tgt_node));

                    spdlog::info("Move choice: src={} tgt={} move={} direction={}", move.src_id,
                             move.tgt_id,
                             move.moved_node.id, move.direction);

                    spdlog::info("TO REMOVE: id={} g={} \nBEFORE MOVE: src={}\n tgt={}",
                             move.moved_node.id,
                        toString(*move.moved_node.graph),
                        toString(*src_node.graph),
                        toString(*tgt_node.graph));

                    spdlog::info("MOVE_RESULT: src {}\ntgt ={}",
                                 toString(*moved_result.src_node.graph),
                                 toString(*moved_result.tgt_node.graph));

                    if (move.direction) {
                        spdlog::info("countRefTgtInSubgraph={} countRefInEdgesSrc={} countRefInEdgesTgt={}",
                                     countRefTgtInSubgraph(src_node, move.moved_node),
                                     countRefInEdgesSrc(move.moved_node, all_sub_edges),
                                     countRefInEdgesTgt(move.moved_node, all_sub_edges));
                    } else {
                        spdlog::info("countRefSrcInSubgraph={} countRefInEdgesTgt={} countRefInEdgesSrc={}",
                                     countRefSrcInSubgraph(tgt_node, move.moved_node),
                                     countRefInEdgesTgt(move.moved_node, all_sub_edges),
                                     countRefInEdgesSrc(move.moved_node, all_sub_edges));
                    }

                    for (const auto& sn: src_node.sub_nodes) {
                        spdlog::info("source subnode graph: {}", toString(*sn.graph));
                    }
                    for (const auto& sn: tgt_node.sub_nodes) {
                        spdlog::info("target subnode graph: {}", toString(*sn.graph));
                    }

                    throw std::runtime_error("Verification failed");
                }

//                spdlog::info("AFTER MOVE: src={}", toString(*moved_result.src_node.graph));
//                spdlog::info("AFTER MOVE: tgt={}", toString(*moved_result.tgt_node.graph));

                long eval_src = eval(profile(src_node.graph));
                long eval_tgt = eval(profile(tgt_node.graph));
                long eval_comm = calcCommTime(calcEdgeSize(edge) / (dev_num_*max_pipeline_num_));

                const auto prof_moved_src = profile(moved_result.src_node.graph);
                long eval_moved_src = eval(prof_moved_src);
                if (!fitToMem(moved_result.src_node.graph, prof_moved_src, dev_mem_, use_amp_master_params_)) {
                    continue;
                }

                const auto prof_moved_tgt = profile(moved_result.tgt_node.graph);
                long eval_moved_tgt = eval(prof_moved_tgt);
                if (!fitToMem(moved_result.tgt_node.graph, prof_moved_tgt, dev_mem_, use_amp_master_params_)) {
                    continue;
                }
                long eval_moved_comm = calcCommTime(calcEdgeSize(moved_result.edge) / (dev_num_*max_pipeline_num_));

                long imprv = (eval_src + eval_tgt + eval_comm)
                             - (eval_moved_src + eval_moved_tgt + eval_moved_comm);

                if (imprv > best_imprv) {
                    best_imprv = imprv;
                    best_moved_result = moved_result;
                }
            }

            if (best_imprv > 0) {
//                spdlog::info("MOVE: best_imprv={} src_id={} tgt_id={}", best_imprv,
//                             best_moved_result.src_node.id, best_moved_result.tgt_node.id);

                node_map[best_moved_result.src_node.id] = best_moved_result.src_node;
                node_map[best_moved_result.tgt_node.id] = best_moved_result.tgt_node;
                MLEdgeKey k{best_moved_result.edge.src_id, best_moved_result.edge.tgt_id};

                if (best_moved_result.edge.sub_edges.empty()) {
                    edge_map.erase(k);
                } else {
                    edge_map[k] = best_moved_result.edge;
                }

                all_nodes = values(node_map);

                all_sub_edges.clear();
                for (const auto& e: values(edge_map)) {
                    for (const auto& se: e.sub_edges) {
                        all_sub_edges.push_back(se);
                    }
                }

                assert(verify(MLGraph{values(node_map), values(edge_map)}));
            }
        }

        return MLGraph{values(node_map), values(edge_map)};
    }

    MLGraph MLPartitioner::uncoarsen(const MLGraph& ml_graph) {

        MLGraph graph = adjustBoundaries(ml_graph);
        logger->trace("Uncoarsening starting");

        while (graph.getLevel() > 1) {
            graph = toLowerLevel(graph);
            graph = adjustBoundaries(graph);
            logger->trace("Uncoarsened graph. level={} #nodes={} cut_size={}", graph.getLevel(), graph.nodes.size(),
                    sumEdgeSizes(graph.edges));
        }
        logger->trace("Uncoarsening finished");

        return graph;
    }

    MLGraph MLPartitioner::mergeSmall(const MLGraph& ml_graph, int min_partition_num) {

        std::unordered_map<std::string, long> eval_map;
        MLGraph merged_graph = ml_graph;
        while (merged_graph.nodes.size() > (size_t) min_partition_num) {

            size_t min_node_idx;
            long min_val = LONG_MAX;

            for (size_t idx=0; idx<merged_graph.nodes.size(); idx++) {
                const auto& n = merged_graph.nodes.at(idx);
                if (!contains(eval_map, n.id)) {
                    eval_map[n.id] = eval(profile(n.graph));
                }
                long val = eval_map.at(n.id);
                if (val < min_val) {
                    min_val = val;
                    min_node_idx = idx;
                }
            }

            const MLNode& min_node = merged_graph.nodes.at(min_node_idx);
            long min_adj_eval = LONG_MAX;
            MLNode merged_adj;
            MLNode merged;
            // preceding
            if (min_node_idx > 0) {
                const auto& preceding = merged_graph.nodes.at(min_node_idx-1);
                assert(contains(eval_map, preceding.id));

                merged = merge(preceding, min_node, merged_graph.nodes, merged_graph.edges);
                const auto prof = profile(merged.graph);
                if (!fitToMem(merged.graph, prof, dev_mem_, use_amp_master_params_)) {
                    break;
                }
                min_adj_eval = eval(prof);
                merged_adj = preceding;
            }
            // following
            if (min_node_idx < merged_graph.nodes.size()-1) {
                const auto& following = merged_graph.nodes.at(min_node_idx+1);
                assert(contains(eval_map, following.id));

                const auto merged_fol = merge(min_node, following, merged_graph.nodes, merged_graph.edges);
                const auto prof = profile(merged_fol.graph);
                if (!fitToMem(merged_fol.graph, prof, dev_mem_, use_amp_master_params_)) {
                    break;
                }
                long val_following = eval(prof);

                if (val_following < min_adj_eval) {
                    min_adj_eval = val_following;
                    merged_adj = following;
                    merged = merged_fol;
                    logger->trace("Merging {} (min) with {} (fol) as {} (#nodes={})", min_node.id, following.id,
                            merged_fol.id, merged_graph.nodes.size());
                } else {
                    if (min_adj_eval < LONG_MAX) {
                        const auto& preceding = merged_graph.nodes.at(min_node_idx-1);
                        logger->trace("Merging {} (pre) with {} (min) as {} (#nodes={})", preceding.id,
                                min_node.id, merged.id, merged_graph.nodes.size());
                    }
                }
            }

            // Failed to merge
            if (min_adj_eval == LONG_MAX) {
                logger->trace("Unable to merge any more. #nodes={}", merged_graph.nodes.size());
                break;
            }

            // update ml graph
            std::vector<MLNode> merged_nodes;
            std::unordered_map<std::string, std::string> name_map;
            bool merged_added = false;
            for (const auto& n: merged_graph.nodes) {
                if (n.id == min_node.id || n.id == merged_adj.id) {
                    if (!merged_added) {
                        merged_nodes.push_back(merged);
                        merged_added = true;
                    }
                    name_map[n.id] = merged.id;
                } else {
                    merged_nodes.push_back(n);
                    name_map[n.id] = n.id;
                }
            }

            std::vector<MLEdge> merged_edges = mergeEdgesNoCopy(std::move(merged_graph.edges), name_map);
            merged_graph = MLGraph{merged_nodes, merged_edges};
        }
        return merged_graph;
    }

    MLGraph MLPartitioner::partition(const std::shared_ptr<IRGraph>& ir_graph) {
        logger->trace("MLPartitioner::partition starting: id={}", ir_graph->getName());

        batch_size_ = guessGraphBatchSize(ir_graph);
        assert(batch_size_ > 0);

//        std::stringstream ss;
//        ss << *ir_graph;
//        spdlog::info("ir_graph={}", ss.str());

        BGraph bg = toBGL(ir_graph);

        // label batch ops
        int rank = 0;
        for (auto &v: all_nodes<Vertex, BGraph>(bg)) {
            if (bg[v].type == NODE && (bg[v].node.isBatch() || bg[v].node.isCriterion())) {
                bg[v].ranks = {rank};
                for (auto& tgt: target_nodes(v, bg)) {
                    bg[tgt].ranks = {rank};
                }
                rank++;
            }
        }

        // label graph inputs
        for (auto &in_v: graph_regular_input_nodes<Vertex, BGraph>(bg)) {
            assert(bg[in_v].value.isBatch());
            for (auto& op: target_nodes(in_v, bg)) {
                for (auto op_rank: bg[op].ranks) {
                    bg[in_v].ranks.insert(op_rank);
                }
            }
        }

        fixNonBatchRanks(bg);

        //        std::ofstream file("batch_only_graph.dot");
//        boost::write_graphviz(file, bg, vertex_rank_label_writer<BGraph>(bg));

        const Partition partitions = createPartition(bg);

        logger->trace("Converting graph: id={}", ir_graph->getName());
        const MLGraph ml_graph = convert(partitions);

        bool do_coarsening = config::Config::get().getVal<bool>(config::DO_COARSENING);
        if (!do_coarsening) {
            logger->info("Skipping coarsening: id={}", ir_graph->getName());
            return sortMLGraph(ml_graph);
        }

        logger->trace("Coarsening graph: id={}", ir_graph->getName());
        int min_partition_num = config::Config::get().getVal<int>(config::MIN_PARTITON_NUM);
        MLGraph graph = coarsen(ml_graph, min_partition_num);

        bool do_uncoarsening = config::Config::get().getVal<bool>(config::DO_UNCOARSENING);
        if (do_uncoarsening) {
            logger->trace("Uncoarsening graph: id={}", ir_graph->getName());
            graph = uncoarsen(graph);
        }

        graph = sortMLGraph(graph);

        int max_partition_num = config::Config::get().getVal<int>(config::MAX_PARTITON_NUM);
        return mergeSmall(graph, max_partition_num);
    }
}