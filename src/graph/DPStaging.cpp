//
// Created by Masahiro Tanaka on 2020/04/11.
//

#include <cuda/CudaUtil.h>
#include "DPStaging.h"

#include <json.hpp>

namespace {
    int isPowerOfTwo(size_t x)
    {
        while (((x & 1) == 0) && x > 1) /* While x is even and > 1 */
            x >>= 1;
        return (x == 1);
    }
}
namespace rannc {

    std::string getMergedGraphId(size_t from, size_t to) {
        std::stringstream ss;
        ss << "MERGE_" << from << "_" << to;
        return ss.str();
    }

    MLNode setNodeId(const MLNode &node, const std::string &id) {
        MLNode new_node;

        new_node = node;
        new_node.id = id;
        new_node.graph = std::make_shared<IRGraph>(id, *node.graph);

        return new_node;
    }

    GraphMergeHelper::GraphMergeHelper(MLGraph graph) : graph_(std::move(graph)) {
        for (size_t i = 0; i < graph_.nodes.size(); i++) {
            const auto &n = graph_.nodes.at(i);
            node_map_[n.id] = std::make_shared<MLNode>(n);
            node_ids_.push_back(n.id);

            GraphMergeKey merge_key{i, i};
            graph_merge_cache_[merge_key] = node_map_[n.id];
        }
    }

    std::shared_ptr<IRGraph> GraphMergeHelper::merge(size_t from, size_t to) {
        GraphMergeKey merge_key{from, to};
        if (contains(graph_merge_cache_, merge_key)) {
            return graph_merge_cache_.at(merge_key)->graph;
        }

        // note: includes the elem whose index is "to"
        assert(to < graph_.nodes.size());

        std::unordered_map<std::string, std::shared_ptr<MLNode>> rest_nodes = node_map_;
        std::vector <MLEdge> rest_edges = graph_.edges;
        std::unordered_map <std::string, std::string> name_map;
        for (const auto &name: node_ids_) {
            name_map[name] = name;
        }

        size_t avail_to = from + 1;
        GraphMergeKey part_merge_key{from, avail_to};
        std::shared_ptr<MLNode> base = std::make_shared<MLNode>(graph_.nodes.at(from));
        while (contains(graph_merge_cache_, part_merge_key)) {
            const MLNode &tgt = graph_.nodes.at(avail_to);
            const auto merged_graph_id = getMergedGraphId(from, avail_to);

            name_map[base->id] = merged_graph_id;
            name_map[tgt.id] = merged_graph_id;
            rest_edges = mergeEdgesNoCopy(std::move(rest_edges), name_map);
            name_map.erase(base->id);
            name_map.erase(tgt.id);
            name_map[merged_graph_id] = merged_graph_id;

            rest_nodes.erase(base->id);
            rest_nodes.erase(tgt.id);

            base = graph_merge_cache_.at(part_merge_key);
            rest_nodes[merged_graph_id] = base;
            part_merge_key = GraphMergeKey(from, ++avail_to);
        }

        for (size_t i = avail_to; i <= to; i++) {
            const MLNode &tgt = graph_.nodes.at(i);
            const auto merged_node = std::make_shared<MLNode>(
                    ::rannc::merge(*base, tgt, values(rest_nodes), rest_edges));
            const auto merged_graph_id = getMergedGraphId(from, i);

            merged_node->graph->setName(merged_graph_id);

            name_map[base->id] = merged_graph_id;
            name_map[tgt.id] = merged_graph_id;
            rest_edges = mergeEdgesNoCopy(std::move(rest_edges), name_map);
            name_map.erase(base->id);
            name_map.erase(tgt.id);
            name_map[merged_graph_id] = merged_graph_id;

            rest_nodes.erase(base->id);
            rest_nodes.erase(tgt.id);
            rest_nodes[merged_graph_id] = merged_node;

            GraphMergeKey new_merge_key{from, i};
            graph_merge_cache_[new_merge_key] = merged_node;

            base = merged_node;
        }

        graph_merge_cache_[merge_key] = base;
        return base->graph;
    }

    size_t estimateCommValueSize(const std::vector<std::shared_ptr<IRGraph>>& graphs,
                             const std::function<std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& get_targets,
                             const std::function<std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& to_erase) {

        std::unordered_map<std::string, IRValue> target_values;

        for (const auto& g: graphs) {
            for (const auto& tgt_name: get_targets(g)) {
                target_values[tgt_name] = g->getValue(tgt_name);
            }
        }

        for (const auto& g: graphs) {
            for (const auto& erase_name: to_erase(g)) {
                target_values.erase(erase_name);
            }
        }

        size_t size_sum = 0;
        for (const auto& it: target_values) {
            size_sum += it.second.getSizeInByte();
        }
        return size_sum;
    }

    size_t estimateInputSize(const std::vector<std::shared_ptr<IRGraph>>& graphs) {
        return estimateCommValueSize(graphs,
                              [](const std::shared_ptr<IRGraph>& g) {return g->getInputNames();},
                              [](const std::shared_ptr<IRGraph>& g) {return g->getOutputNames();});
    }

    size_t estimateOutputSize(const std::vector<std::shared_ptr<IRGraph>>& graphs) {
        return estimateCommValueSize(graphs,
                                     [](const std::shared_ptr<IRGraph>& g) {return g->getOutputNames();},
                                     [](const std::shared_ptr<IRGraph>& g) {return g->getInputNames();});
    }

    GraphProfile DPStaging::estimateProf(const MLGraph &graph, size_t from, size_t to, size_t dev_num,
                                         bool checkpointing) {

        std::vector<std::shared_ptr<IRGraph>> graphs;
        for (const auto& node: graph.nodes) {
            graphs.push_back(node.graph);
        }

        return accProfileValues(prof_util_, batch_size_, graphs, from, to, dev_num, checkpointing);
    }

    long estimateEval(const GraphProfile &step_prof,
                      long step_comm_in, long step_comm_out,
                      long prev_fwd_max, long prev_bwd_max, long prev_ar_max) {
        long step_comm = step_comm_in + step_comm_out;
        long max_fwd_val = std::max(step_prof.fwd_time + step_comm, prev_fwd_max);
        long max_bwd_val = std::max(step_prof.bwd_time + step_comm, prev_bwd_max);
        return max_fwd_val + max_bwd_val;
    }

    void DPStaging::dumpNodeProfiles(const std::string& path, const MLGraph &graph, size_t dev_num,
                                     size_t min_pipeline_num, size_t max_pipeline_num) {
        int node_idx = 0;
        nlohmann::ordered_json nodes_prof;
        nlohmann::json graphs;

        for (const auto &node: graph.nodes) {
            nlohmann::json node_obj;
            node_obj["name"] = node.graph->getName();
            node_obj["param_size"] = node.graph->getParamSizeInByte();
            node_obj["input_size"] = calcInputSize(node.graph);
            node_obj["output_size"] = calcOutputSize(node.graph);
            graphs[node.graph->getName()] = toString(*node.graph);

            std::vector<nlohmann::json> prof_dev;

            for (int pipeline_num = std::max(1, (int) min_pipeline_num);
                    pipeline_num <= std::min((int) batch_size_, (int) max_pipeline_num);
                    pipeline_num *= 2) {
                for (size_t d = 1; d <= dev_num; d++) {
                    nlohmann::ordered_json prof_obj;
                    prof_obj["global_batch_size"] = batch_size_;
                    prof_obj["prof_batch_size"] = size_t(ceil(batch_size_ / (double) (d * pipeline_num)));
                    prof_obj["dev_num"] = d;
                    prof_obj["pipeline_num"] = pipeline_num;

                    const auto prof = prof_util_.profile(node.graph, batch_size_, d * pipeline_num, pipeline_num > 1);
                    prof_obj["fwd_time"] = prof.fwd_time;
                    prof_obj["bwd_time"] = prof.bwd_time;
                    prof_obj["max_allocated_mem"] = prof.max_allocated_mem;
                    prof_obj["checkpointing"] = prof.checkpointing;

                    prof_dev.push_back(prof_obj);
                }
            }
            node_obj["profiles"] = prof_dev;
            nodes_prof[std::to_string(node_idx)] = node_obj;

            node_idx++;
        }

        logger->info("Saving node profiles to {}", path);

        std::ofstream out(path, std::ios::out);
        if (!out) {
            throw std::invalid_argument("Failed to open file: " + path);
        }

        nlohmann::ordered_json dump_obj;
        dump_obj["total_dev_num"] = dev_num;
        dump_obj["node_profiles"] = nodes_prof;
        dump_obj["graphs"] = graphs;
        out << dump_obj.dump(4);
        out.close();
    }

    long DPStaging::estimateTime(const AllocSolution &sol, const MLGraph& graph) {

        std::unordered_map<std::string, long> fwd_times;
        std::unordered_map<std::string, long> bwd_times;

        long comp_time = 0;
        for (int step = 0; step < sol.pipeline_num + sol.graphs.size() - 1; step++) {
            size_t g_from = std::max(0, step - sol.pipeline_num + 1);
            size_t g_to_excl = std::min(step + 1, (int) sol.graphs.size());

            long max_fwd_time = 0;
            long max_bwd_time = 0;
            for (size_t g_idx = g_from; g_idx < g_to_excl; g_idx++) {
                const auto &sg = sol.graphs.at(g_idx);

                const auto prof = estimateSolutionGraph(sol, graph, g_idx);

                int repl = sol.repl_nums.at(sg->getName());
                long comm_time = calcInputCommTime(sg, repl * sol.pipeline_num) +
                        calcOutputCommTime(sg, repl * sol.pipeline_num);
                long fwd_time = prof.fwd_time + comm_time;
                max_fwd_time = std::max(max_fwd_time, fwd_time);

                long bwd_time = prof.bwd_time + comm_time;
                max_bwd_time = std::max(max_bwd_time, bwd_time);
            }

            comp_time += max_fwd_time + max_bwd_time;
        }

        long max_ar_time = 0;
        for (const auto &sg: sol.graphs) {
            long ar_time = calcAllReduceTime(sg->getParamSizeInByte());
            max_ar_time = std::max(max_ar_time, ar_time);
        }

        return comp_time + max_ar_time;
    }

    GraphProfile DPStaging::estimateSolutionGraph(const AllocSolution &sol, const MLGraph& graph, size_t g_idx) {
        const auto &sg = sol.graphs.at(g_idx);
        assert(contains(sol.repl_nums, sg->getName()));
        int repl = sol.repl_nums.at(sg->getName());
        return prof_util_.profile(sg, batch_size_, repl * sol.pipeline_num, sol.pipeline_num > 1);
    }

    AllocSolution DPStaging::runDpComm(const MLGraph &graph, size_t dev_num) {

        config::Config& config = config::Config::get();

        // Forcibly set pipeline num for debugging
        if (cfg_pipeline_num_ != 0) {
            min_pipeline_num_ = cfg_pipeline_num_;
            max_pipeline_num_ = cfg_pipeline_num_;
        }

        logger->trace("DPStaging::runDpComm starting: batch_size={} dev_num={} min_pipeline_num={}",
                      batch_size_, dev_num, min_pipeline_num_);

        const bool dp_search_all = config.getVal<bool>(config::DP_SEARCH_ALL);

        int dev_per_node = std::min((int) dev_num, getCudaDeviceCount());

        if (dev_num % dev_per_node != 0) {
            logger->warn("The numbers of devices may differ across nodes");
        }
        int node_num_total = dev_num / dev_per_node;

        if (!dump_dp_node_profiles_.empty()) {
            dumpNodeProfiles(dump_dp_node_profiles_, graph, dev_num, min_pipeline_num_, max_pipeline_num_);
        }

        std::vector <AllocSolution> pl_sols;

        bool sol_found = false;
        size_t MIN_SEARCH_STAGE_NUM = 1;
        size_t prev_stage_num_max = 0;
        for (int node_num_used = 1; node_num_used <= node_num_total; node_num_used++) {
            if (node_num_total % node_num_used != 0) {
                continue;
            }

            size_t stage_num_min, stage_num_max;
            // Forcibly set stage num for debugging
            if (cfg_stage_num_ != 0) {
                stage_num_min = stage_num_max = cfg_stage_num_;
            } else {
                stage_num_min = prev_stage_num_max + 1;
                stage_num_max = prev_stage_num_max = dev_per_node * node_num_used;
            }

            // graph can be very small
            stage_num_min = std::min(stage_num_min, graph.nodes.size());
            stage_num_max = std::min(stage_num_max, graph.nodes.size());

            for (size_t stage_num = stage_num_min; stage_num <= stage_num_max; stage_num++) {
                for (int pipeline_num = std::max(1, min_pipeline_num_);
                     pipeline_num <= std::min((int) batch_size_, max_pipeline_num_);
                     pipeline_num *= 2) {
                    bool checkpointing = pipeline_num > 1;
                    int replica_num = node_num_total / node_num_used;

                    logger->trace(
                            "Searching allocations: #nodes={} #dev_per_node={} #stages={} replica_num={} pipeline_num={}",
                            node_num_used, dev_per_node, stage_num, replica_num, pipeline_num);
                    AllocSolution sol = doRunDpComm(graph, stage_num, dev_per_node * node_num_used,
                                                    replica_num, pipeline_num, checkpointing);

                    // DP found a solution
                    if (!sol.graphs.empty()) {
                        sol_found = true;
                        pl_sols.push_back(sol);
                    }
                }

                // DP found a solution
                if (!dp_search_all && sol_found && stage_num >= MIN_SEARCH_STAGE_NUM) {
                    break;
                }
            }
            if (!dp_search_all && sol_found) {
                break;
            }
        }

        if (!dump_dp_cache_.empty()) {
            DPStagingCache cache;
            cache.graph = graph;
            cache.ml_profile_cache = prof_util_.getProfileCache();
            cache.dev_num = dev_num;
            cache.batch_size = batch_size_;
            cache.dev_mem = dev_mem_;
            cache.min_pipeline_num = min_pipeline_num_;
            cache.max_pipeline_num = max_pipeline_num_;
            cache.cfg_pipeline_num = cfg_pipeline_num_;
            cache.cfg_stage_num = cfg_stage_num_;
            cache.use_amp_master_params = use_amp_master_params_;
            cache.enable_zero = enable_zero_;
            cache.ir_graph = ir_graph_;

            logger->info("Saving DP cache to {}", dump_dp_cache_);
            saveToFile(dump_dp_cache_, cache);
        }

        if (pl_sols.empty()) {
            throw std::runtime_error("Failed to find a feasible allocation.");
        }

        long best_time = LONG_MAX;
        AllocSolution best_sol;
        for (const auto &sol: pl_sols) {
            long est_time = estimateTime(sol, graph);
            if (est_time < best_time) {
                best_time = est_time;
                best_sol = sol;
            }
        }

        std::unordered_map<std::string, GraphProfile> profiles;
        for (size_t g_idx=0; g_idx<best_sol.graphs.size(); g_idx++) {
            profiles[best_sol.graphs.at(g_idx)->getName()] = estimateSolutionGraph(best_sol, graph, g_idx);
        }
        logger->info(displayGraphProfiles(best_sol.graphs,
                             batch_size_, best_sol.pipeline_num,
                             use_amp_master_params_, enable_zero_,
                             best_sol.repl_nums, profiles));

        return best_sol;
    }

    struct DPState {
        DPState() : eval(ProfilerUtil::ERROR_VAL), max_fwd(ProfilerUtil::ERROR_VAL), max_bwd(ProfilerUtil::ERROR_VAL),
                    max_allreduce(ProfilerUtil::ERROR_VAL),
                    pre_boundary(0), pre_dev_num(0) {}

        DPState(long eval, long maxFwd, long maxBwd, long maxAr, size_t preBoundary, size_t preDevNum) :
                eval(eval), max_fwd(maxFwd), max_bwd(maxBwd), max_allreduce(maxAr),
                pre_boundary(preBoundary), pre_dev_num(preDevNum) {}

        long eval;
        long max_fwd;
        long max_bwd;
        long max_allreduce;
        size_t pre_boundary;
        size_t pre_dev_num;
        std::shared_ptr <IRGraph> step_graph;
    };

    AllocSolution DPStaging::doRunDpComm(const MLGraph &graph, size_t stage_num, size_t dev_num_per_group,
                                         int replica_num, int pipeline_num, bool checkpointing) {
        GraphMergeHelper merge_helper(graph);

        const std::vector <MLNode> &nodes = graph.nodes;
        size_t layer_num = nodes.size();
        const int min_pipeline_bs = config::Config::get().getVal<int>(config::MIN_PIPELINE_BS);
        const bool limit_dev_num_pot = config::Config::get().getVal<bool>(config::LIMIT_DEV_NUM_POT);
        const bool limit_dev_num_more_than_bs = config::Config::get().getVal<bool>(config::LIMIT_DEV_NUM_MORE_THAN_BS);

        // 3-dimensional table
        // table[stage][boundary][used_dev]
        using DPTable = std::vector <std::vector<std::vector < DPState>>>;
        DPTable table;

        for (size_t s = 0; s <= stage_num; s++) {
            table.push_back(std::vector < std::vector < DPState >> ());

            for (size_t l = 0; l <= layer_num; l++) {
                table[s].push_back(std::vector<DPState>());

                for (size_t d = 0; d <= dev_num_per_group; d++) {
                    table[s][l].push_back(DPState());
                }
            }
        }

        for (size_t l = 0; l <= layer_num; l++) {
            for (size_t d = 0; d <= dev_num_per_group; d++) {
                DPState stage(0, 0, 0, 0, 0, 0);
                table[0][l][d] = stage;
            }
        }

        for (size_t s = 1; s <= stage_num; s++) {
            // the index of a stage starts from 1

            logger->trace(
                    "DPStaging::doRunDpComm stage_num={} s={} dev_num_per_group={} pipeline_num={} checkpointing={}",
                    stage_num, s, dev_num_per_group, pipeline_num, checkpointing);

            size_t min_d = 1;

            // b must equal to layer_num when s == stage_num
            size_t b_start = s == stage_num ? layer_num : s;

            for (size_t b = b_start; b <= layer_num - stage_num + s; b++) {
                // b: the index of the right boundary of s-th stage
                // the index from "boundary" starts from 0
                bool found_b_sol = false;

                // TODO: d can start from (dev_num_per_group - (stage_num - s))
                for (size_t d = dev_num_per_group; d >= std::max(min_d, s); d--) {
                    bool found_d_sol = false;
                    bool skip_small_bs = false;

                    // d: the number of devices used for stages <= s
                    // b_prev and d_prev must be 0 when s=1
                    size_t b_prev_limit = s == 1 ? 1 : b;
                    size_t d_prev_limit = s == 1 ? 1 : d;

                    // search possible boundaries of (s-1)th stage
                    for (size_t b_prev = (s - 1); b_prev < b_prev_limit; b_prev++) {

                        for (size_t d_prev = (s - 1); d_prev < d_prev_limit; d_prev++) {

                            size_t max_d = ceil(batch_size_ / (double) (replica_num * pipeline_num));
                            if (limit_dev_num_more_than_bs) {
                                if (max_d < (d - d_prev)) {
                                    logger->trace(
                                            "Skip dev_num: stage_num={} s={} b={} d={} b_prev={} d_prev={} bs={} repl={} pl={}",
                                            stage_num, s, b, d, b_prev, d_prev,
                                            batch_size_, replica_num, pipeline_num);
                                    skip_small_bs = true;
                                    continue;
                                }
                            }

                            if (limit_dev_num_pot) {
                                if (!isPowerOfTwo(d - d_prev)) {
                                    logger->trace(
                                            "Skip pot: stage_num={} s={} b={} d={} b_prev={} d_prev={} bs={} repl={} pl={}",
                                            stage_num, s, b, d, b_prev, d_prev,
                                            batch_size_, replica_num, pipeline_num);
                                    skip_small_bs = true;
                                    continue;
                                }
                            }

                            double stage_bs = batch_size_ / (double) (replica_num * pipeline_num);
                            size_t repl_bs = ceil(stage_bs / (d - d_prev));
                            if (repl_bs < min_pipeline_bs) {
                                logger->trace(
                                        "Skip 2: stage_num={} s={} b={} d={} b_prev={} d_prev={} bs={} repl={} pl={} stage_bs={} repl_bs={} min_pipeline_bs={}",
                                        stage_num, s, b, d, b_prev, d_prev,
                                        batch_size_, replica_num, pipeline_num,
                                        stage_bs, repl_bs, min_pipeline_bs);

                                skip_small_bs = true;
                                continue;
                            }

                            if (table[s - 1][b_prev][d_prev].eval >= ProfilerUtil::ERROR_VAL) {
                                logger->trace(
                                        "DPStaging::doRunDpComm: The previous state is infeasible. stage_num={} s={} b={} d={} b_prev={} d_prev={} table[s-1][b_prev][d_prev].eval={}",
                                        stage_num, s, b, d, b_prev, d_prev, table[s - 1][b_prev][d_prev].eval);
                                continue;
                            }

                            long step_val = LONG_MAX;
                            long step_mem = LONG_MAX;
                            long ar_comm = 0;
                            GraphProfile step_prof;

                            bool profile_by_acc = config::Config::get().getVal<bool>(config::PROFILE_BY_ACC);
                            if (profile_by_acc) {
                                // Just estimate time by accumulation
                                step_prof = estimateProf(graph, b_prev, b - 1,
                                                         (d - d_prev) * replica_num * pipeline_num, checkpointing);

                                std::vector<std::shared_ptr<IRGraph>> ir_graphs;
                                assert(graph.nodes.size() > b - 1);
                                for (size_t i = b_prev; i <= b - 1; i++) {
                                    ir_graphs.push_back(graph.nodes.at(i).graph);
                                }

                                size_t input_size = estimateInputSize(ir_graphs);
                                size_t step_in_comm = calcCommTime(input_size / ((d - d_prev) * replica_num * pipeline_num));
                                size_t output_size = estimateOutputSize(ir_graphs);
                                size_t step_out_comm = calcCommTime(output_size / ((d - d_prev) * replica_num * pipeline_num));

                                step_val = ::rannc::estimateEval(step_prof,
                                                                 step_in_comm, step_out_comm,
                                                                 table[s - 1][b_prev][d_prev].max_fwd,
                                                                 table[s - 1][b_prev][d_prev].max_bwd,
                                                                 table[s - 1][b_prev][d_prev].max_allreduce);

                                static int opt_param_factor = config::Config::get().getVal<int>(
                                        config::OPT_PARAM_FACTOR);
                                long opt_mem = 0;
                                for (size_t i = b_prev; i <= (b - 1); i++) {
                                    const auto &node = graph.nodes.at(i);
                                    opt_mem += getOptMemSize(node.graph, opt_param_factor, use_amp_master_params_, enable_zero_, (d - d_prev)*replica_num);
                                }
                                step_mem = step_prof.max_allocated_mem + opt_mem
                                        + (step_in_comm + step_out_comm) * 2;
                            } else {
                                // merge graphs from j+1 to i (inclusive)
                                const auto step_graph = merge_helper.merge(b_prev, b - 1);
                                size_t step_in_comm = calcCommTime(
                                        calcInputSize(step_graph) / ((d - d_prev) * replica_num * pipeline_num));
                                size_t step_out_comm = calcCommTime(
                                        calcOutputSize(step_graph) / ((d - d_prev) * replica_num * pipeline_num));
                                ar_comm = calcAllReduceTime(step_graph->getParamSizeInByte());

                                // run profiler for the merged graph
                                step_prof = prof_util_.profile(step_graph, batch_size_,
                                                               (d - d_prev) * replica_num * pipeline_num,
                                                               checkpointing);
                                step_mem = calcGraphMem(step_graph, step_prof, batch_size_, (d - d_prev) * replica_num, pipeline_num,
                                                        use_amp_master_params_, enable_zero_);
                                step_val = ::rannc::estimateEval(step_prof,
                                                                 step_in_comm, step_out_comm,
                                                                 table[s - 1][b_prev][d_prev].max_fwd,
                                                                 table[s - 1][b_prev][d_prev].max_bwd,
                                                                 table[s - 1][b_prev][d_prev].max_allreduce);
                            }

                            if (step_mem >= dev_mem_) {
                                logger->trace(
                                        "DPStaging::doRunDpComm: The required memory exceeded the limit. stage_num={} s={} b={} d={} b_prev={} d_prev={} mem={}",
                                        stage_num, s, b, d, b_prev, d_prev, step_mem);

                                // we break here, not continue
                                // this is because larger d_prev gives less gpus for the step graph
                                break;
                            }

                            bool update = table[s][b][d].eval > step_val;

                            found_b_sol = true;
                            found_d_sol = true;

                            if (update) {
                                table[s][b][d].eval = std::max(step_val, table[s - 1][b_prev][d_prev].eval);
                                table[s][b][d].max_fwd = std::max(step_prof.fwd_time,
                                                                  table[s - 1][b_prev][d_prev].max_fwd);
                                table[s][b][d].max_bwd = std::max(step_prof.bwd_time,
                                                                  table[s - 1][b_prev][d_prev].max_bwd);
                                table[s][b][d].max_allreduce = std::max(ar_comm,
                                                                        table[s - 1][b_prev][d_prev].max_allreduce);
                                table[s][b][d].pre_boundary = b_prev;
                                table[s][b][d].pre_dev_num = d_prev;
//                                table[s][b][d].step_graph = step_graph;
                                //
                                logger->trace(
                                        "DPStaging::doRunDpComm: UPDATED stage_num={} s={} b={} d={} s'={} b'={} d'={}: step_val={} "
                                        "table[{}][{}][{}]={} table[{}][{}][{}]={} #pre_graphs={} update={}",
                                        stage_num, s, b, d, s - 1, b_prev, d_prev, step_val,
                                        s, b, d, table[s][b][d].eval,
                                        s - 1, b_prev, d_prev, table[s - 1][b_prev][d_prev].eval,
                                        s - 1,
                                        update);
                            } else {
                                logger->trace(
                                        "DPStaging::doRunDpComm: NO_UPDATE stage_num={} s={} b={} d={} s'={} b'={} d'={}: step_val={} "
                                        "table[{}][{}][{}]={} table[{}][{}][{}]={} #pre_graphs={} min_dev_num={} update={}",
                                        stage_num, s, b, d, s - 1, b_prev, d_prev, step_val,
                                        s, b, d, table[s][b][d].eval,
                                        s - 1, b_prev, d_prev, table[s - 1][b_prev][d_prev].eval,
                                        s - 1,
                                        min_d, update);
                            }
                        }
                    }
                    if (!found_d_sol && !skip_small_bs) {
                        logger->trace("solution not found with d={}. exiting", d);
                        min_d = d + 1;
                        break;
                    }
                }
                if (!found_b_sol) {
                    logger->trace("solution not found with b={}. exiting", b);
                    break;
                }
            }
        }

        logger->trace("DP summary (dev_num_per_group={} pipeline_num={})", dev_num_per_group, pipeline_num);
        for (size_t s = 1; s <= stage_num; s++) {
            logger->trace("DP table: stage {}/{}", s, stage_num);
            size_t b_start = s == stage_num ? layer_num : s;
            for (size_t b = b_start; b <= layer_num - stage_num + s; b++) {
                std::vector<long> evals;
                for (size_t d = 0; d <= dev_num_per_group; d++) {
                    evals.push_back(table[s][b][d].eval);
                }
                logger->trace(" v[{}][{}]={}", s, b, join_as_str(evals));
            }
        }

        size_t best_d = 0;
        long best_val = ProfilerUtil::ERROR_VAL;
        for (size_t d = dev_num_per_group; d <= dev_num_per_group; d++) {
            if (table[stage_num][layer_num][d].eval < best_val) {
                best_val = table[stage_num][layer_num][d].eval;
                best_d = d;
            }
        }

        // get solution
        size_t b_sol = layer_num;
        size_t d_sol = best_d;
        std::vector <size_t> boundaries;
        std::vector <size_t> dev_nums;
        boundaries.push_back(b_sol);
        dev_nums.push_back(d_sol);

        std::unordered_map<std::string, int> repl_nums;
        std::vector <std::shared_ptr<IRGraph>> sol_graphs;
        for (size_t s_sol = stage_num; s_sol > 0; s_sol--) {
            const auto &state = table[s_sol][b_sol][d_sol];
            if (state.eval >= ProfilerUtil::ERROR_VAL) {
                return AllocSolution{{}, std::unordered_map<std::string, int>()};
            }

            const auto step_graph = merge_helper.merge(state.pre_boundary, b_sol - 1);

//            const auto& g = state.step_graph;
            sol_graphs.push_back(step_graph);
            repl_nums[step_graph->getName()] = (d_sol - state.pre_dev_num) * replica_num;

            b_sol = state.pre_boundary;
            d_sol = state.pre_dev_num;

            boundaries.push_back(b_sol);
            dev_nums.push_back(d_sol);
        }
        std::reverse(sol_graphs.begin(), sol_graphs.end());
        std::reverse(boundaries.begin(), boundaries.end());
        std::reverse(dev_nums.begin(), dev_nums.end());

        logger->trace("sol boundaries={}", join_as_str(boundaries));
        logger->trace("sol dev_nums={}", join_as_str(dev_nums));

        return AllocSolution{sol_graphs, repl_nums, pipeline_num, checkpointing, boundaries, dev_nums};
    }

    GraphProfile DPDryStaging::estimateSolutionGraph(const AllocSolution &sol, const MLGraph& graph, size_t g_idx) {
        int repl = sol.dev_nums.at(g_idx + 1) - sol.dev_nums.at(g_idx);
        return this->estimateProf(graph,
                                  sol.boundaries.at(g_idx),
                                  sol.boundaries.at(g_idx + 1) - 1,
                                  repl * sol.pipeline_num, sol.pipeline_num > 1);
    }

    Deployment DPDryStaging::partition() {
        const auto sol =  DPStaging::runDpComm(graph_, dev_num_);
        Partition new_part = createPartition(ir_graph_, sol.graphs);

        // graph names in new_part are different with those in sol.repl_nums
        std::vector<std::string> ordered_graph_ids;
        for (const auto& g: sol.graphs) {
            ordered_graph_ids.push_back(g->getName());
        }
        assert(ordered_graph_ids.size() == new_part.order.size());
        std::unordered_map<std::string, int> repl_nums;
        for (const auto& it: new_part.subgraphs) {
            assert(contains(sol.repl_nums, it.first));
            repl_nums[it.first] = sol.repl_nums.at(it.first);
        }

        const auto repl = replicate(new_part, repl_nums, sol.pipeline_num, batch_size_);
        logger->trace("Partitioning finished: id={}", ir_graph_->getName());

        std::unordered_map<std::string, std::unordered_set<int>> alloc;

        if (config::Config::get().getVal<bool>(config::ALLOC_REPL_FLAT)) {
            logger->trace("searchAllocationFlat");
            alloc = searchAllocationFlat(repl, dev_num_, dev_mem_);
        } else {
            logger->trace("searchAllocationSimple");
            alloc = searchAllocationSimple(repl, dev_num_, dev_mem_);
        }

        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto &it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(repl, alloc, dev_num_);
        deployment.pipeline_num = sol.pipeline_num;
        deployment.checkpointing = sol.checkpointing;
        logger->trace("Created deployment finished");

        return deployment;
    }
}
