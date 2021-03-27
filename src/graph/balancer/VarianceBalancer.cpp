//
// Created by Masahiro Tanaka on 2020/01/18.
//

#include "graph/ProfiledWeightDecomposer.h"

namespace rannc {
    bool ProfiledWeightDecomposer::doBalanceSplitVariance(std::vector<size_t>& split,
                                                   const std::vector<long>& vals) {

        const double TIME_DIFF_THRES = 0.01;
        const double FIX_RATE = 0.1;

        long ave_val = average(vals);
        size_t max_val_diff_idx = max_val_diff_block_idx(vals);
        long max_val_diff = block_diff_to_ave(vals, max_val_diff_idx);
        long diff_sum = sum_abs_diff_to_ave(vals);

        logger->trace("vals={} sizes={} ave={} max_val_diff_idx={} max_val_diff={}",
                      join_as_str(vals), join_as_str(all_block_sizes(split)), ave_val,
                      max_val_diff_idx, max_val_diff);

        if (diff_sum < sum(vals)*TIME_DIFF_THRES) {
            logger->trace("Exiting optimization max_val_diff={} diff_sum={} sum(subgraph_vals)*TIME_DIFF_THRES={}",
                          max_val_diff, diff_sum,
                          sum(vals)*TIME_DIFF_THRES);

            return true;
        }

        double scale = scale_to_ave(vals, max_val_diff_idx, FIX_RATE);
        long n_move_node = blocks_to_move(split, vals, max_val_diff_idx, FIX_RATE);

        split = balance_blocks(split, vals, max_val_diff_idx, n_move_node);

        logger->trace("max_block_val={} block_size={} n_move_node={} scale={}",  block_val(vals, max_val_diff_idx),
                     block_size(split, max_val_diff_idx), n_move_node, scale);
        logger->trace("scale={} new_split={} new_size={}", scale, join_as_str(split),
                join_as_str(all_block_sizes(split)));

        return false;
    }
}