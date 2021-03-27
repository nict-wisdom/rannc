//
// Created by Masahiro Tanaka on 2019-03-15.
//

#ifndef PYRANNC_PROFWEIGHTDECOMPOSER_H
#define PYRANNC_PROFWEIGHTDECOMPOSER_H


#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {
    class IRGraph;

    long block_size(const std::vector<size_t>& split, size_t idx);
    std::vector<size_t> all_block_sizes(const std::vector<size_t>& split);
    long left_block_size(const std::vector<size_t>& split, size_t idx);
    long right_block_size(const std::vector<size_t>& split, size_t idx);
    long block_val(const std::vector<long>& vals, size_t idx);
    long left_blocks_val(const std::vector<long>& vals, size_t idx);
    long right_blocks_val(const std::vector<long>& vals, size_t idx);
    long ave_block_vals(const std::vector<long>& vals);
    size_t max_val_diff_block_idx(const std::vector<long>& vals);
    long block_diff_to_ave(const std::vector<long>& vals, size_t idx);
    long sum_abs_diff_to_ave(const std::vector<long>& vals);
    double scale_to_ave(const std::vector<long>& vals, size_t idx, double fix_rate);
    long blocks_to_move_by_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, double scale);
    long blocks_to_move(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, double fix_rate);
    double block_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block);
    double left_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block);
    double right_scale(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx, long n_move_block);

    std::vector<size_t> scale_left_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                         long n_move_block);
    std::vector<size_t> scale_right_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                          long n_move_block);
    std::vector<long> est_vals_scale_left_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                              long n_move_block);
    std::vector<long> est_vals_scale_right_block(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                               long n_move_block);
    std::vector<size_t> balance_blocks(const std::vector<size_t>& split, const std::vector<long>& vals, size_t idx,
                                       long n_move_node);

    struct ParallelConf {
        int part;
        int repl;
        int pipe;
    };

    class ProfiledWeightDecomposer {
    public:
        ProfiledWeightDecomposer(std::shared_ptr<GraphProfiler> sg_prof, int worker_num, int64_t batch_size, size_t dev_mem)
            :sg_prof_(std::move(sg_prof)), worker_num_(worker_num), batch_size_(batch_size), dev_mem_(dev_mem) {}

        Deployment decompose(const std::shared_ptr<IRGraph>& irGraph);

    private:
        std::shared_ptr<GraphProfiler> sg_prof_;
        int worker_num_;
        int64_t batch_size_;
        size_t dev_mem_;

        ParallelConf balance(BGraph& g);
        std::pair<long, std::vector<size_t>> doBalance(const std::vector<size_t>& initial_split, int replica_num,
                                                 int pipeline_num, BGraph& g);
        bool doBalanceSplitAdjacent(std::vector<size_t>& split, const std::vector<long>& subgraph_vals);
        bool doBalanceSplitVariance(std::vector<size_t>& split, const std::vector<long>& subgraph_vals);

        const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
    };
}

#endif //PYRANNC_PROFWEIGHTDECOMPOSER_H
