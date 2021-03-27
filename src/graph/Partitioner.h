//
// Created by Masahiro Tanaka on 2020/02/19.
//

#ifndef PYRANNC_PARTITIONER_H
#define PYRANNC_PARTITIONER_H

#include <ostream>
#include <comp/GraphProfiler.h>
#include "ir.h"
#include "Decomposition.h"
#include "MLGraph.h"
#include "ProfilerUtil.h"

namespace rannc {

    struct GraphCacheKeyHash {
        std::size_t operator()(const std::vector<std::string> &key) const {
            return std::hash<std::string>()(join_as_str(key));
        };
    };

    using GraphCache = std::unordered_map<std::vector<std::string>, MLNode, GraphCacheKeyHash>;

    struct MoveResult {
        MLNode src_node;
        MLNode tgt_node;
        MLEdge edge;
    };

    class MLPartitioner {
    public:
        MLPartitioner(std::shared_ptr<GraphProfiler> profiler, size_t dev_num, size_t dev_mem,
                      size_t max_pipeline_num, size_t min_pipeline_bs,
                      bool use_amp_master_params, bool coarsen_by_time):
                prof_util_(std::move(profiler)), dev_num_(dev_num), dev_mem_(dev_mem),
                max_pipeline_num_(max_pipeline_num),
                min_pipeline_bs_(min_pipeline_bs),
                use_amp_master_params_(use_amp_master_params),
                coarsen_by_time_(coarsen_by_time) {}

        MLGraph partition(const std::shared_ptr<IRGraph>& ir_graph);

    private:
        MLGraph coarsen(const MLGraph& ml_graph, int min_partition_num);
        MLGraph uncoarsen(const MLGraph& ml_graph);
        GraphProfile profile(const std::shared_ptr<IRGraph> &g);

        MLGraph mergeAdjacents(const MLGraph& ml_graph, bool skip_profiling, int min_partition_num);
        MLGraph adjustBoundaries(const MLGraph& ml_graph);
        MoveResult move_to_tgt(const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
                               const MLEdge& sub_edge, const MLNode& moved_node,
                               const std::unordered_set<std::string>& required_inputs);
        MoveResult move_to_src(const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
                               const MLEdge& sub_edge, const MLNode& moved_node,
                               const std::unordered_set<std::string>& required_inputs);
        MLGraph mergeSmall(const MLGraph& ml_graph, int min_partition_num);

        std::shared_ptr<IRGraph> removeHeadSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);
        std::shared_ptr<IRGraph> removeTailSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);
        std::shared_ptr<IRGraph> doRemoveSubgraph(const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);

        std::vector<MLVertex> sortNodesByEval(std::vector<MLVertex> nodes, const MLBGraph &bg);
        long eval(const GraphProfile& prof);

        ProfilerUtil prof_util_;
        size_t batch_size_;
        size_t dev_num_;
        size_t dev_mem_;
        size_t max_pipeline_num_;
        size_t min_pipeline_bs_;
        bool use_amp_master_params_;
        bool coarsen_by_time_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("MLPartitioner");
    };
}

#endif //PYRANNC_PARTITIONER_H
