//
// Created by Masahiro Tanaka on 2020/04/11.
//

#ifndef PYRANNC_DPSTAGING_H
#define PYRANNC_DPSTAGING_H

#include "MLGraph.h"
#include "ProfilerUtil.h"

namespace rannc {
class GraphMergeHelper {
 public:
  explicit GraphMergeHelper(MLGraph graph);
  std::shared_ptr<IRGraph> merge(size_t from, size_t to);

 private:
  MLGraph graph_;
  std::unordered_map<std::string, std::shared_ptr<MLNode>> node_map_;
  std::vector<std::string> node_ids_;
  GraphMergeCache graph_merge_cache_;
};

struct AllocSolution {
  std::vector<std::shared_ptr<IRGraph>> graphs;
  std::unordered_map<std::string, int> repl_nums;
  std::unordered_map<std::string, TensorPartitioningGraphInfo> part_info;
  int pipeline_num;
  bool checkpointing;
  std::vector<size_t> boundaries;
  std::vector<size_t> dev_nums;

  MSGPACK_DEFINE(
      graphs, repl_nums, pipeline_num, checkpointing, boundaries, dev_nums);
};

class DPStaging {
 public:
  DPStaging(
      std::shared_ptr<GraphProfiler> profiler,
      std::shared_ptr<IRGraph> ir_graph, PartitioningConf conf)
      : prof_util_(std::move(profiler)),
        ir_graph_(std::move(ir_graph)),
        conf_(conf) {
    config::Config& config = config::Config::get();
    dump_dp_node_profiles_ =
        config.getVal<std::string>(config::DUMP_DP_NODE_PROFILES);
    dump_dp_cache_ = config.getVal<std::string>(config::DUMP_DP_CACHE);
  }

  AllocSolution runDpComm(const MLGraph& graph);

 protected:
  AllocSolution doRunDpComm(
      const MLGraph& graph, size_t stage_num, size_t dev_num_per_group,
      int replica_num, int pipeline_num, bool checkpointing);
  long estimateTime(const AllocSolution& sol, const MLGraph& graph);
  virtual GraphProfile estimateSolutionGraph(
      const AllocSolution& sol, const MLGraph& graph, size_t g_idx);

  GraphProfile estimateProf(const ProfilingInput& prof_in);

  void dumpNodeProfiles(const std::string& path, const MLGraph& graph);

  TensorPartitioningGraphInfo partitionParams(
      std::shared_ptr<IRGraph> g, int repl_num) const;

  ProfilerUtil prof_util_;
  PartitioningConf conf_;

  std::string dump_dp_node_profiles_;
  std::string dump_dp_cache_;
  std::shared_ptr<IRGraph> ir_graph_;

  static const int DEFALUT_ITERATION_NUM;

  const std::shared_ptr<spdlog::logger> logger = getLogger("DPStaging");
};

struct DPStagingCache {
  MLGraph graph;
  MLProfileCache ml_profile_cache;
  std::shared_ptr<IRGraph> ir_graph;
  PartitioningConf conf;

  MSGPACK_DEFINE(graph, ml_profile_cache, ir_graph, conf);
};

class DPDryStaging : public DPStaging {
 public:
  DPDryStaging(const DPStagingCache& cache)
      : graph_(cache.graph),
        conf_(cache.conf),
        DPStaging(
            std::shared_ptr<GraphProfiler>(nullptr), cache.ir_graph,
            cache.conf) {
    prof_util_.setProfileCache(cache.ml_profile_cache);
    dump_dp_node_profiles_.clear();
    dump_dp_cache_.clear();
  }

  Deployment partition();

 protected:
  GraphProfile estimateSolutionGraph(
      const AllocSolution& sol, const MLGraph& graph, size_t g_idx) override;

 private:
  MLGraph graph_;
  std::shared_ptr<IRGraph> ir_graph_;
  PartitioningConf conf_;
};
} // namespace rannc

#endif // PYRANNC_DPSTAGING_H
