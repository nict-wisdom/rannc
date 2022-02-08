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
  GraphMergeHelper(MLGraph graph);
  std::shared_ptr<IRGraph> merge(size_t from, size_t to);

 private:
  MLGraph graph_;
  std::unordered_map<std::string, std::shared_ptr<MLNode>> node_map_;
  std::vector<std::string> node_ids_;
  GraphMergeCache graph_merge_cache_;
};

class DPStaging {
 public:
  DPStaging(
      std::shared_ptr<GraphProfiler> profiler,
      std::shared_ptr<IRGraph> ir_graph, size_t batch_size, size_t dev_mem,
      bool use_amp_master_params, bool enable_zero, bool force_dist_matmul)
      : prof_util_(std::move(profiler), force_dist_matmul),
        ir_graph_(ir_graph),
        batch_size_(batch_size),
        dev_mem_(dev_mem),
        use_amp_master_params_(use_amp_master_params),
        enable_zero_(enable_zero),
        force_dist_matmul_(force_dist_matmul) {
    config::Config& config = config::Config::get();
    dump_dp_node_profiles_ =
        config.getVal<std::string>(config::DUMP_DP_NODE_PROFILES);
    dump_dp_cache_ = config.getVal<std::string>(config::DUMP_DP_CACHE);
    min_pipeline_num_ = config.getVal<int>(config::MIN_PIPELINE);
    max_pipeline_num_ = config.getVal<int>(config::MAX_PIPELINE);
    cfg_pipeline_num_ = config.getVal<int>(config::PIPELINE_NUM);
    cfg_stage_num_ = config.getVal<int>(config::PARTITION_NUM);
  }

  AllocSolution runDpComm(const MLGraph& graph, size_t dev_num);

 protected:
  AllocSolution doRunDpComm(
      const MLGraph& graph, size_t stage_num, size_t dev_num_per_group,
      int replica_num, int pipeline_num, bool checkpointing);
  long estimateTime(const AllocSolution& sol, const MLGraph& graph);
  virtual GraphProfile estimateSolutionGraph(
      const AllocSolution& sol, const MLGraph& graph, size_t g_idx);

  GraphProfile estimateProf(
      const MLGraph& graph, size_t from, size_t to, size_t dev_num,
      size_t pipeline_num, bool checkpointing);

  void dumpNodeProfiles(
      const std::string& path, const MLGraph& graph, size_t dev_num,
      size_t min_pipeline_num, size_t max_pipeline_num);

  ProfilerUtil prof_util_;
  size_t batch_size_;
  size_t dev_mem_;
  int min_pipeline_num_;
  int max_pipeline_num_;
  int cfg_pipeline_num_;
  size_t cfg_stage_num_;

  bool use_amp_master_params_;
  bool enable_zero_;
  bool force_dist_matmul_;

  GraphMergeCache graph_merge_cache_;

  std::string dump_dp_node_profiles_;
  std::string dump_dp_cache_;
  std::shared_ptr<IRGraph> ir_graph_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("DPStaging");
};

struct DPStagingCache {
  MLGraph graph;
  MLProfileCache ml_profile_cache;
  size_t dev_num;
  size_t batch_size;
  size_t dev_mem;
  int min_pipeline_num;
  int max_pipeline_num;
  int cfg_pipeline_num;
  size_t cfg_stage_num;
  bool use_amp_master_params;
  bool enable_zero;
  std::shared_ptr<IRGraph> ir_graph;

  MSGPACK_DEFINE(
      graph, ml_profile_cache, dev_num, batch_size, dev_mem, min_pipeline_num,
      max_pipeline_num, cfg_pipeline_num, cfg_stage_num, use_amp_master_params,
      enable_zero, ir_graph);
};

class DPDryStaging : public DPStaging {
 public:
  DPDryStaging(const DPStagingCache& cache)
      : graph_(cache.graph),
        dev_num_(cache.dev_num),
        batch_size_(cache.batch_size),
        ir_graph_(cache.ir_graph),
        DPStaging(
            std::shared_ptr<GraphProfiler>(nullptr), cache.ir_graph,
            cache.batch_size, cache.dev_mem, cache.use_amp_master_params,
            cache.enable_zero, false) {
    prof_util_.setProfileCache(cache.ml_profile_cache);

    min_pipeline_num_ = cache.min_pipeline_num;
    max_pipeline_num_ = cache.max_pipeline_num;
    cfg_pipeline_num_ = cache.cfg_pipeline_num;
    cfg_stage_num_ = cache.cfg_stage_num;

    dump_dp_node_profiles_.clear();
    dump_dp_cache_.clear();
  }

  Deployment partition();

 protected:
  virtual GraphProfile estimateSolutionGraph(
      const AllocSolution& sol, const MLGraph& graph, size_t g_idx);

 private:
  MLGraph graph_;
  size_t batch_size_;
  std::shared_ptr<IRGraph> ir_graph_;
  size_t dev_num_;
};
} // namespace rannc

#endif // PYRANNC_DPSTAGING_H
