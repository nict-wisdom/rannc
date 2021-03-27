//
// Created by Masahiro Tanaka on 2019-03-15.
//
#include <cassert>

#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <graph/ir.h>
#include <Config.h>
#include "ProfiledWeightDecomposer.h"
#include "Decomposition.h"


namespace rannc {
    long block_size(const std::vector<size_t>& split, size_t idx) {
        if (idx == 0) {
            return split.at(idx);
        }
        return split.at(idx) - split.at(idx - 1);
    }

    std::vector<size_t> all_block_sizes(const std::vector<size_t>& split) {
        std::vector<size_t> sizes;
        for (size_t i=0; i<split.size(); i++) {
            sizes.push_back(block_size(split, i));
        }
        return sizes;
    }

    long left_block_size(const std::vector<size_t>& split, size_t idx) {
        long sum = 0;
        for (size_t i=0; i<idx; i++) {
            sum += block_size(split, i);
        }
        return sum;
    }

    long right_block_size(const std::vector<size_t>& split, size_t idx) {
        long sum = 0;
        for (size_t i=idx+1; i<split.size(); i++) {
            sum += block_size(split, i);
        }
        return sum;
    }

    long ave_block_vals(const std::vector<long>& vals) {
        return average(vals);
    }

    long block_val(const std::vector<long>& vals, size_t idx) {
        return vals.at(idx);
    }

    long left_blocks_val(const std::vector<long>& vals, size_t idx) {
        long sum = 0;
        for (size_t i=0; i<idx; i++) {
            sum += block_val(vals, i);
        }
        return sum;
    }

    long right_blocks_val(const std::vector<long>& vals, size_t idx) {
        long sum = 0;
        for (size_t i=idx+1; i<vals.size(); i++) {
            sum += block_val(vals, i);
        }
        return sum;
    }

    size_t max_val_diff_block_idx(const std::vector<long>& vals) {
        long val_ave = ave_block_vals(vals);

        long max_val_diff = 0; // needs pos/neg sign
        size_t max_val_diff_idx = 0;
        for (size_t sg_idx=0; sg_idx<vals.size(); sg_idx++) {
            long diff = val_ave - vals.at(sg_idx);
            if (std::abs(diff) > std::abs(max_val_diff)) {
                max_val_diff = diff;
                max_val_diff_idx = sg_idx;
            }
        }
        return max_val_diff_idx;
    }

    long block_diff_to_ave(const std::vector<long>& vals, size_t idx) {
        long val_ave = ave_block_vals(vals);
        return val_ave - vals.at(idx);
    }

    long sum_abs_diff_to_ave(const std::vector<long>& vals) {
        long sum = 0;
        for (size_t i=0; i<vals.size(); i++) {
            sum += std::abs(block_diff_to_ave(vals, i));
        }
        return sum;
    }

    double scale_to_ave(const std::vector<long>& vals, size_t idx, double fix_rate) {
        long val = vals.at(idx);
        long diff = block_diff_to_ave(vals, idx); // if positive, need to extend
        long scaled_val = val + diff * fix_rate;
        return scaled_val / (double) val;
    }

    long blocks_to_move_by_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, double scale) {
        long n_blocks = block_size(split, idx);
        long n_block_scaled = n_blocks * scale;
        return n_block_scaled - n_blocks;
    }

    long blocks_to_move(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, double fix_rate) {
        double scale = scale_to_ave(vals, idx, fix_rate);
        return blocks_to_move_by_scale(split, vals, idx, scale);
    }

    double _scale_by_block_num(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                               long n_move_block, const std::function<long(const std::vector<size_t>&, size_t)> f) {
        long n_block = f(split, idx);
        if (n_block == 0) {
            return 1.0;
        }
        // if n_move_block is positive, the block at i increases and other blocks decrease
        long n_new_block = n_block - n_move_block;
        return n_new_block / (double) n_block;
    }

    double block_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block) {
        return _scale_by_block_num(split, vals, idx, n_move_block, block_size);
    }

    double left_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block) {
        return _scale_by_block_num(split, vals, idx, n_move_block, left_block_size);
    }

    double right_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block) {
        return _scale_by_block_num(split, vals, idx, n_move_block, right_block_size);
    }

    std::vector<size_t> scale_left_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                         long n_move_block) {
        // unable to move block to the left
        if (idx == 0) {
            return split;
        }

        std::vector<size_t> new_split = split;
        size_t offset = 0;
        double scale = left_scale(split, vals, idx, n_move_block);
        for (size_t i=0; i<idx; i++) {
            size_t size = block_size(split, i) * scale;
            offset += size;
            new_split[i] = offset;
        }
        if (idx != vals.size()-1) {
            long size = block_size(split, idx) + n_move_block;
            offset += size;
            new_split[idx] = offset;
        }
        return new_split;
    }

    std::vector<size_t> scale_right_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                          long n_move_block) {
        // unable to move block to the right
        if (idx == vals.size()-1) {
            return split;
        }

        size_t offset = 0;
        if (idx > 0) {
            offset = split[idx - 1];
        }
        offset += block_size(split, idx) + n_move_block;

        std::vector<size_t> new_split = split;
        new_split[idx] = offset;

        double scale = right_scale(split, vals, idx, n_move_block);
        for (size_t i=idx+1; i<split.size()-1; i++) {
            size_t size = block_size(split, i) * scale;
            offset += size;
            new_split[i] = offset;
        }
        return new_split;
    }

    std::vector<long> _est_vals_scale_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                            long n_move_block,
                                            const std::function<std::vector<size_t>(const std::vector<size_t>&, const std::vector<long>&, size_t, long)>& f) {
        const auto& new_split = f(split, vals, idx, n_move_block);
        std::vector<long> est_values;
        for (size_t i=0; i<split.size(); i++) {
            double block_scale = block_size(new_split, i) / (double) block_size(split, i);
            est_values.push_back(vals.at(i) * block_scale);
        }
        return est_values;
    }

    std::vector<long> est_vals_scale_left_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                                long n_move_block) {
        return _est_vals_scale_block(split, vals, idx, n_move_block, scale_left_block);
    }

    std::vector<long> est_vals_scale_right_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                                 long n_move_block) {
        return _est_vals_scale_block(split, vals, idx, n_move_block, scale_right_block);
    }

    std::vector<size_t> balance_blocks(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                       long n_move_node) {

        std::vector<long> est_vals_left = est_vals_scale_left_block(split, vals, idx, n_move_node);
        const auto split_left = scale_left_block(split, vals, idx, n_move_node);
        long est_max_val_left = max(est_vals_left);

        std::vector<long> est_vals_right = est_vals_scale_right_block(split, vals, idx, n_move_node);
        const auto split_right = scale_right_block(split, vals, idx, n_move_node);
        long est_max_val_right = max(est_vals_right);

        bool balance_left = est_max_val_right - est_max_val_left > 0;

        if (balance_left) {
            return split_left;
        }
        return split_right;
    }

    long evalSubgraph(const GraphProfile& prof) {
        return prof.fwd_time * 2;
    }

    long estimate(const std::vector<long>& vals, int pipeline_num) {
//        spdlog::info("@estimate vals={} pl={} pipeline_num+vals.size()-1={}",
//                     join_as_str(vals), pipeline_num, pipeline_num+vals.size()-1);
        long est = 0;
        for (size_t t=0; t<pipeline_num+vals.size()-1; t++) {
            long est_step = 0;
            size_t init_sg_idx = std::max((long)0, ((long) t - pipeline_num + 1));
            size_t fin_sg_idx = std::min(vals.size(), t+1);

//            spdlog::info("t={} init_sg_idx={} fin_sg_idx={}", t, init_sg_idx, fin_sg_idx);
            for (size_t sg_idx=init_sg_idx; sg_idx<fin_sg_idx; sg_idx++) {
                est_step = std::max(est_step, vals.at(sg_idx));
            }
            est += est_step;
        }
        return est;
    }

    std::pair<long, std::vector<size_t>> ProfiledWeightDecomposer::doBalance(const std::vector<size_t>& initial_split, int replica_num,
                                             int pipeline_num, BGraph& g) {

        const int MAX_BALANCE_ITER = 10;
//        const double mem_exceed_scale = 0.9;

        std::vector<size_t> split = initial_split;
        std::vector<size_t> prev_split;

        long est_val = LONG_MAX;
        for (int iter_idx=0; iter_idx < MAX_BALANCE_ITER; iter_idx++) {
            setRanksOnGraph(g, split);
            fixNonBatchRanks(g);

            Partition partition = createPartition(g);
            std::unordered_map<std::string, std::shared_ptr<IRGraph>> scaled_graphs;
            std::stringstream ss;
            for (const auto &it: partition.subgraphs) {
                scaled_graphs[it.first] = scaleGraph(it.second, replica_num*pipeline_num, batch_size_);
            }

            int opt_param_factor = config::Config::get().getVal<int>(config::OPT_PARAM_FACTOR);
            std::unordered_map<std::string, long> mem_limits;
            for (const auto& it: scaled_graphs) {
                const auto& sg = it.second;
                long opt_mem = sg->getParamSizeInByte() * opt_param_factor;
                mem_limits[it.first] = dev_mem_ - opt_mem;

//                spdlog::info("mem_limits[{}]={} param_size={} opt_param_factor={}", it.first, mem_limits[it.first],
//                             sg->getParamSizeInByte(), opt_param_factor);
            }

            ProfilingResult profiles;
            try {
                bool chk_pt = config::Config::get().getVal<bool>(config::CHECKPOINTING);
                int profiling_iter = config::Config::get().getVal<int>(config::PROFILING_ITER);
                profiles = sg_prof_->profile(scaled_graphs, profiling_iter);
            } catch (...) {
                spdlog::info("Encountered an error while profiling.");
                sg_prof_->clear();
                split = prev_split;
                break;
            }

            std::vector<long> subgraph_vals;
            for (const auto &id: partition.order) {
                const auto &prof = profiles.node_profiles.at(id);
                logger->trace("{} fwd_time={} bwd_time={} mem={}", id, prof.fwd_time, prof.bwd_time,
                              prof.max_allocated_mem);
                subgraph_vals.push_back(evalSubgraph(prof));
            }
            est_val = estimate(subgraph_vals, pipeline_num);

            long time_ar = 0;
            if (replica_num > 1) {
                long param_size_sum = 0;
                for (const auto &it: scaled_graphs) {
                    const auto &sg = it.second;
                    param_size_sum += sg->getParamSizeInByte();
                    spdlog::info("{} param size={}", it.first, sg->getParamSizeInByte());
                }
                time_ar = param_size_sum / (200 * 1000);
            }

            spdlog::info("vals={} pl={} est(comp)={} time_ar={}", join_as_str(subgraph_vals), pipeline_num,
                    est_val, time_ar);
            est_val += time_ar;

            // check memory error
            bool mem_exceed = false;
            for (size_t sg_idx=0; sg_idx<partition.order.size(); sg_idx++) {
                const auto& id = partition.order.at(sg_idx);
                const auto& prof = profiles.node_profiles.at(id);

//                spdlog::info("mem_limits[{}]={} alloc_mem={}", id, mem_limits[id], prof.max_allocated_mem);
                if (prof.max_allocated_mem > mem_limits[id]) {
                    mem_exceed = true;
                    break;
                }
            }
            if (mem_exceed) {
                sg_prof_->clear();
                split = prev_split;
                break;
            }

            prev_split = split;
//            if (doBalanceSplitAdjacent(split, subgraph_vals)) {
            if (doBalanceSplitVariance(split, subgraph_vals)) {
                break;
            }
        }

        setRanksOnGraph(g, split);
        fixNonBatchRanks(g);

        return {est_val, split};
    }

    size_t calcCutSize(const Partition& p) {
        size_t size_sum = 0;
        const auto& g = p.graph;
        for (const auto& con: p.connections) {
            const auto& val = g->getValue(con.value);
            size_sum += val.getSizeInByte();
        }
        return size_sum;
    }

    struct CutSearchState {
        bool operator==(const CutSearchState &rhs) const {
            return cut_size == rhs.cut_size &&
                   split == rhs.split;
        }

        bool operator!=(const CutSearchState &rhs) const {
            return !(rhs == *this);
        }

        size_t cut_size;
        std::vector<size_t> split;
    };


    struct CutSearchStateHash {
        std::size_t operator()(const CutSearchState &state) const {
            std::stringstream ss;
            ss << join_as_str(state.split) << "=" << state.cut_size;
            return std::hash<std::string>()(ss.str());
        };
    };

    CutSearchState chooseBestState(const std::unordered_set<CutSearchState, CutSearchStateHash>& open) {
        CutSearchState best_state{SIZE_MAX, {}};
        for (const auto& it: open) {
            if (it.cut_size < best_state.cut_size) {
                best_state = it;
            }
        }
        return best_state;
    }

    CutSearchState doSearchCuts(const std::vector<size_t>& split, BGraph& g) {
        setRanksOnGraph(g, split);
        fixNonBatchRanks(g);
        size_t cut_size = calcCutSize(createPartition(g));

        int MAX_SEARCH_ITER = 10;

        CutSearchState init_state{cut_size, split};
        spdlog::info("initial cut={} split={}", cut_size,join_as_str(split));

        std::unordered_set<CutSearchState, CutSearchStateHash> open;
        std::unordered_set<CutSearchState, CutSearchStateHash> closed;
        std::unordered_set<std::vector<size_t>, SizeVectorHash> visited;
        open.insert(init_state);

        for (int iter_idx = 0; iter_idx < MAX_SEARCH_ITER; iter_idx++) {
            spdlog::info("iter={} open size={}", iter_idx, open.size());

            if (open.empty()) {
                break;
            }
            CutSearchState state = chooseBestState(open);

            for (size_t idx = 0; idx < state.split.size() - 1; idx++) {
                std::vector<int> steps = {-1, +1};
                for (int s: steps) {
                    auto test_split = state.split;
                    test_split[idx] += s;

                    if (contains(visited, test_split)) {
                        continue;
                    }

                    setRanksOnGraph(g, test_split);
                    fixNonBatchRanks(g);
                    size_t test_cut_size = calcCutSize(createPartition(g));

                    CutSearchState test_state{test_cut_size, test_split};
                    spdlog::info("iter={} current={} test_cut={} test_split={}", iter_idx,
                            state.cut_size,
                            test_cut_size,
                            join_as_str(test_split));

                    open.insert(test_state);
                    visited.insert(test_split);
                }
            }
            open.erase(state);
        }

        std::unordered_set<CutSearchState, CutSearchStateHash> all;
        for (const auto& it: open) {
            all.insert(it);
        }
        for (const auto& it: closed) {
            all.insert(it);
        }

        return chooseBestState(all);
    }

    ParallelConf ProfiledWeightDecomposer::balance(BGraph& g) {

        bool disable_balancing = config::Config::get().getVal<bool>(config::DISABLE_BALANCING);
        int cfg_part_num = config::Config::get().getVal<int>(config::PARTITION_NUM);
        int cfg_replica_num = config::Config::get().getVal<int>(config::REPLICA_NUM);
        int cfg_pipeline_num = config::Config::get().getVal<int>(config::PIPELINE_NUM);
        bool cfg_auto_parallel = config::Config::get().getVal<bool>(config::AUTO_PARALLEL);

        ParallelConf pconf;
        if (disable_balancing && !cfg_auto_parallel) {
            pconf.part = cfg_part_num;
            pconf.repl = cfg_replica_num;
            pconf.pipe = cfg_pipeline_num;

            const auto split = splitByValueSizes(g, pconf.part);
            setRanksOnGraph(g, split);
            fixNonBatchRanks(g);
            return pconf;
        }

        long best_est = LONG_MAX;
        std::vector<size_t> best_split;

        if (cfg_auto_parallel) {
            for (size_t part_num = 1; part_num <= mpi::getSize(); part_num *= 2) {
                const auto initial_split = splitByValueSizes(g, part_num);
                int replica_num = mpi::getSize() / part_num;

//        if (initial_split.size() == 1 || disable_balancing) {
//            setRanksOnGraph(g, initial_split);
//            fixNonBatchRanks(g);
//            return pipeline_num;
//        }

//        logger->trace("balance starting. initial_split={} replica_num={} pipeline_num={}",
//                join_as_str(initial_split), replica_num, pipeline_num);

                int max_pipeline_num = std::min((int) 32, (int) batch_size_ / replica_num);
                for (size_t pl = 1; pl <= max_pipeline_num; pl *= 2) {
                    std::pair<long, std::vector<size_t>> balance_est = doBalance(initial_split, replica_num, pl, g);
                    long est = balance_est.first;
                    spdlog::info("est={} P={} R={} L={}", est, part_num, replica_num, pl);
                    if (best_est > est) {
                        best_est = est;
                        pconf.part = part_num;
                        pconf.repl = replica_num;
                        pconf.pipe = pl;
                        best_split = balance_est.second;
                        spdlog::info("BEST est={} P={} R={} L={} split={}", est, part_num, replica_num, pl,
                                     join_as_str(initial_split));
                    }
                }
            }
        } else {
            pconf.part = cfg_part_num;
            pconf.repl = cfg_replica_num;
            pconf.pipe = cfg_pipeline_num;

            const auto split = splitByValueSizes(g, pconf.part);
            std::pair<long, std::vector<size_t>> balance_est = doBalance(split, cfg_replica_num,
                    cfg_pipeline_num, g);

            CutSearchState st = doSearchCuts(balance_est.second, g);
            best_split = st.split;
        }

        setRanksOnGraph(g, best_split);
        fixNonBatchRanks(g);

        return pconf;
    }

    Deployment ProfiledWeightDecomposer::decompose(const std::shared_ptr<IRGraph>& irGraph) {

        int conf_n_partition = config::Config::get().getVal<int>(config::PARTITION_NUM);

        logger->trace("ProfiledWeightDecomposer::decompose starting");

        if (dev_mem_ > 0) {
            dev_mem_ -= 1024L * 1024L * 1024L;
        } else {
            logger->warn("No CUDA device found on workers. Assuming (almost) unlimited host memory when assigning subgraphs.");
            dev_mem_ = 2 * 1024L * 1024L * 1024L * 1024L; // 2TB
        }

        int n_partition = conf_n_partition;
        if (n_partition <= 0) {
            size_t graph_size = irGraph->getSizeInByte();
            n_partition = (int) (graph_size / dev_mem_) + 1;
            logger->info("The partition num was automatically set to {}. To manually set this value, "
                         "add 'partition_num' to ~/.pyrannc/rannc_conf.toml", n_partition);
        }
        assert(n_partition != 0);

        int replica_num = config::Config::get().getVal<int>(config::REPLICA_NUM);

        BGraph g = toBGL(irGraph);

        ParallelConf pconf = balance(g);
        replica_num = pconf.repl;
//
//        setRanksOnGraph(g, split);
//        fixNonBatchRanks(g);

        Partition partition = createPartition(g);

        // Use IRGraph in the argument because createPartition loses the order of inputs/outputs
        partition.graph = irGraph;
        logger->trace("ProfiledWeightDecomposer::decompose created partition. id={}", partition.id);

        logger->trace("ProfiledWeightDecomposer::decompose creating replications. id={} repl_num={}", partition.id, replica_num);

        std::unordered_map<std::string, int> repl_nums;
        for (const auto& it: partition.subgraphs) {
            repl_nums[it.first] = replica_num;
        }
        PartitionDP partitionDp = replicate(partition, repl_nums, batch_size_);
        logger->trace("ProfiledWeightDecomposer::decompose created PartitionDP. id={}", partitionDp.id);

        logger->info("Assigning {} subgraphs to {} device(s) ... (mem: {} per device)",
                     partitionDp.subgraphs.size(), mpi::getSize(), dev_mem_);

        std::unordered_map<std::string, std::unordered_set<int>> alloc = searchAllocation(partitionDp, mpi::getSize(), dev_mem_*2);
        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto& it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(partitionDp, alloc);
        deployment.pipeline_num = pconf.pipe;
        logger->trace("ProfiledWeightDecomposer::decompose finished");

        return deployment;
    }
}
