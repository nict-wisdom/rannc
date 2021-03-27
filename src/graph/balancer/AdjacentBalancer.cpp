//
// Created by Masahiro Tanaka on 2020/01/18.
//

#include "graph/ProfiledWeightDecomposer.h"

namespace rannc {

    bool ProfiledWeightDecomposer::doBalanceSplitAdjacent(std::vector<size_t>& split,
                                                   const std::vector<long>& subgraph_vals) {
        const double TIME_DIFF_THRES = 0.01;
        const double FIX_RATE = 0.3;

        long max_val_diff = 0;
        size_t max_val_diff_idx = 0;
        for (size_t sg_idx=0; sg_idx<subgraph_vals.size()-1; sg_idx++) {
            long diff = subgraph_vals.at(sg_idx+1) - subgraph_vals.at(sg_idx);
            if (std::abs(diff) > std::abs(max_val_diff)) {
                max_val_diff = diff;
                max_val_diff_idx = sg_idx;
            }
        }
        logger->trace("subgraph_vals={} split={} max_val_diff_idx={} max_val_diff={}",
                      join_as_str(subgraph_vals), join_as_str(split),
                      max_val_diff_idx, max_val_diff);

        long eval_sum = subgraph_vals.at(max_val_diff_idx) + subgraph_vals.at(max_val_diff_idx+1);
        if (std::abs(max_val_diff) < eval_sum*TIME_DIFF_THRES) {
            logger->trace("Exiting optimization max_val_diff={} eval_sum={} eval_sum*TIME_DIFF_THRES={}",
                          max_val_diff, eval_sum,
                          eval_sum*TIME_DIFF_THRES);

            return true;
        }
        // sum eval values of adjacent subgraphs
        long block_size1 = block_size(split, max_val_diff_idx);
        long block_size2 = block_size(split, max_val_diff_idx + 1);
        long size_sum = block_size1 + block_size2;
        double ave_eval1 =  subgraph_vals.at(max_val_diff_idx) / (double) block_size(split, max_val_diff_idx);
        double ave_eval2 =  subgraph_vals.at(max_val_diff_idx + 1) / (double) block_size(split, max_val_diff_idx + 1);
        long exp_size1 = (long) (ave_eval2 / (ave_eval1 + ave_eval2) * size_sum);
        long move_size = (long) ((exp_size1 - block_size1) * FIX_RATE);
        split[max_val_diff_idx] += move_size;

        logger->trace("Trial: block_size1={} block_size2={} size_sum={} ave_eval1={} ave_eval2={} exp_size1={} move={}",
                      block_size1, block_size2,
                      size_sum, ave_eval1, ave_eval2, exp_size1, move_size);

        return false;
    }
}