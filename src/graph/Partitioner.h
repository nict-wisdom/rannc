//
// Created by Masahiro Tanaka on 2020/02/19.
//

#ifndef PYRANNC_PARTITIONER_H
#define PYRANNC_PARTITIONER_H

#include <comp/GraphProfiler.h>
#include <ostream>
#include "Decomposition.h"
#include "ir.h"
#include "MLGraph.h"
#include "ProfilerUtil.h"

namespace rannc {

struct GraphCacheKeyHash {
  std::size_t operator()(const std::vector<std::string>& key) const {
    return std::hash<std::string>()(join_as_str(key));
  };
};

using GraphCache =
    std::unordered_map<std::vector<std::string>, MLNode, GraphCacheKeyHash>;

struct MoveResult {
  MLNode src_node;
  MLNode tgt_node;
  MLEdge edge;
};

class MLPartitioner {
 public:
  MLPartitioner(
      std::shared_ptr<GraphProfiler> profiler, PartitioningConf conf,
      bool coarsen_by_time)
      : prof_util_(std::move(profiler)),
        conf_(std::move(conf)),
        coarsen_by_time_(coarsen_by_time) {
    max_repl_num_ = conf.dev_num;
  }

  MLGraph partition(const std::shared_ptr<IRGraph>& ir_graph);

 private:
  MLGraph coarsen(const MLGraph& ml_graph, int min_partition_num);
  MLGraph uncoarsen(const MLGraph& ml_graph);
  GraphProfile profile(const std::shared_ptr<IRGraph>& g);
  //  GraphProfile profileDist(const std::shared_ptr<IRGraph>& g);
  ProfilingInput makeProfileDistInput(const std::shared_ptr<IRGraph>& g) const;

  MLGraph mergeAdjacents(
      const MLGraph& ml_graph, bool skip_profiling, int min_partition_num);
  MLGraph adjustBoundaries(const MLGraph& ml_graph);
  MoveResult move_to_tgt(
      const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
      const MLEdge& sub_edge, const MLNode& moved_node,
      const std::unordered_set<std::string>& required_inputs);
  MoveResult move_to_src(
      const MLNode& src_node, const MLEdge& edge, const MLNode& tgt_node,
      const MLEdge& sub_edge, const MLNode& moved_node,
      const std::unordered_set<std::string>& required_inputs);
  MLGraph mergeSmall(const MLGraph& ml_graph, int min_partition_num);

  std::shared_ptr<IRGraph> removeHeadSubgraph(
      const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);
  std::shared_ptr<IRGraph> removeTailSubgraph(
      const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);
  std::shared_ptr<IRGraph> doRemoveSubgraph(
      const std::shared_ptr<IRGraph>& g1, const std::shared_ptr<IRGraph>& g2);

  std::vector<MLVertex> sortNodesByEval(
      std::vector<MLVertex> nodes, const MLBGraph& bg);
  long eval(const GraphProfile& prof);

  bool fitToMem(
      const std::shared_ptr<IRGraph>& g, const GraphProfile& prof,
      long capacity, bool use_amp_master_params, bool enable_zero,
      int zero_dist_num) const;
  bool fitToMem(
      const std::shared_ptr<IRGraph>& g, const GraphProfile& prof,
      long capacity, const ProfilingInput& prof_in) const;

  ProfilerUtil prof_util_;
  PartitioningConf conf_;
  bool coarsen_by_time_;
  int max_repl_num_;

  static const int DEFALUT_ITERATION_NUM;

  const std::shared_ptr<spdlog::logger> logger = getLogger("MLPartitioner");
};
} // namespace rannc

#endif // PYRANNC_PARTITIONER_H
