//
// Created by Masahiro Tanaka on 2019/12/20.
//

#ifndef PYRANNC_GRAPHPROFILER_H
#define PYRANNC_GRAPHPROFILER_H

#include <torch/torch.h>
#include <ostream>
#include "distop/PartitionTensor.h"
#include "graph/ir.h"
#include "torch/TorchDriver.h"
#include "torch/TorchUtil.h"

namespace rannc {

class ParamStorage;
class FunctionStorage;

struct GraphProfile {
  std::string name;
  long fwd_time;
  long bwd_time;
  long max_allocated_mem;
  bool checkpointing;

  friend std::ostream& operator<<(
      std::ostream& os, const GraphProfile& profile) {
    os << "name: " << profile.name << " fwd_time: " << profile.fwd_time
       << " bwd_time: " << profile.bwd_time
       << " max_allocated_mem: " << profile.max_allocated_mem
       << " checkpointing: " << profile.checkpointing;
    return os;
  }

  MSGPACK_DEFINE(name, fwd_time, bwd_time, max_allocated_mem, checkpointing);
};

struct ProfilingInput {
  ProfilingInput() = default;

  ProfilingInput(
      const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& irGraphs,
      size_t batchSize, int iteration,
      std::unordered_map<std::string, size_t> replicaNums, size_t pipelineNum,
      bool checkpointing, int optParamFactor, bool useAmpMasterParams,
      bool enableZero, bool offloadParams, bool forceDistMatmul,
      const std::unordered_map<std::string, TensorPartitioningGraphInfo>&
          partInfo)
      : ir_graphs(irGraphs),
        batch_size(batchSize),
        iteration(iteration),
        replica_nums(replicaNums),
        pipeline_num(pipelineNum),
        checkpointing(checkpointing),
        opt_param_factor(optParamFactor),
        use_amp_master_params(useAmpMasterParams),
        enable_zero(enableZero),
        offload_params(offloadParams),
        force_dist_matmul(forceDistMatmul),
        part_info(partInfo) {}

  ProfilingInput(
      const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& irGraphs,
      int iteration, const std::unordered_map<std::string, size_t>& replicaNums,
      size_t pipelineNum, int checkpointing,
      std::unordered_map<std::string, TensorPartitioningGraphInfo>& partInfo,
      const PartitioningConf& conf)
      : ir_graphs(irGraphs),
        batch_size(conf.batch_size),
        iteration(iteration),
        replica_nums(replicaNums),
        pipeline_num(pipelineNum),
        checkpointing(checkpointing),
        opt_param_factor(conf.opt_param_factor),
        use_amp_master_params(conf.use_amp_master_params),
        enable_zero(conf.enable_zero),
        offload_params(conf.offload_params),
        force_dist_matmul(conf.force_dist_matmul),
        part_info(partInfo) {}

  ProfilingInput(
      const std::shared_ptr<IRGraph>& ir_graph, int iteration,
      size_t replicaNum, size_t pipelineNum, int checkpointing,
      const TensorPartitioningGraphInfo& partInfo, const PartitioningConf& conf)
      : ir_graphs({{ir_graph->getName(), ir_graph}}),
        batch_size(conf.batch_size),
        iteration(iteration),
        replica_nums({{ir_graph->getName(), replicaNum}}),
        pipeline_num(pipelineNum),
        checkpointing(checkpointing),
        opt_param_factor(conf.opt_param_factor),
        use_amp_master_params(conf.use_amp_master_params),
        enable_zero(conf.enable_zero),
        offload_params(conf.offload_params),
        force_dist_matmul(conf.force_dist_matmul),
        part_info({{ir_graph->getName(), partInfo}}) {}

  std::unordered_map<std::string, std::shared_ptr<IRGraph>> ir_graphs;
  size_t batch_size = 0;
  int iteration = 0;
  std::unordered_map<std::string, size_t> replica_nums;
  size_t pipeline_num = 0;
  bool checkpointing = false;
  int opt_param_factor = 2;
  bool use_amp_master_params;
  bool enable_zero;
  bool offload_params = false;
  bool force_dist_matmul = false;
  std::unordered_map<std::string, TensorPartitioningGraphInfo> part_info;

  MSGPACK_DEFINE(
      ir_graphs, batch_size, iteration, replica_nums, pipeline_num,
      checkpointing, opt_param_factor, use_amp_master_params, enable_zero,
      offload_params, force_dist_matmul, part_info);
};

struct ProfilingResult {
  std::unordered_map<std::string, GraphProfile> node_profiles;
  std::unordered_map<std::string, IRType> value_types;

  MSGPACK_DEFINE(node_profiles, value_types);
};

struct ProfileItemKey {
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> ir_graphs;
  size_t batch_size;
  std::unordered_map<std::string, size_t> repl_nums;
  int iteration;
  bool checkpointing;

  MSGPACK_DEFINE(ir_graphs, batch_size, repl_nums, iteration, checkpointing);
};

struct ProfileItem {
  ProfileItemKey key;
  ProfilingResult profile;

  MSGPACK_DEFINE(key, profile);
};

class ProfileDB {
 public:
  void add(const ProfileItem& prof_item);
  bool hasRecord(const ProfileItemKey& prof_key);
  ProfilingResult get(const ProfileItemKey& prof_key);
  const std::unordered_map<size_t, ProfileItem>& getItems() const;

 private:
  std::unordered_map<size_t, ProfileItem> items_;
};

class GraphProfiler {
 public:
  GraphProfiler(
      std::shared_ptr<ParamStorage> param_storage,
      std::shared_ptr<IRGraph> base_graph,
      std::unordered_map<std::string, torch::jit::IValue> non_param_inputs,
      std::unordered_map<std::string, long> graph_params, IValueMap constants,
      std::shared_ptr<FunctionStorage> functions, size_t batch_size,
      int dev_num, size_t min_pipeline_num)
      : param_storage_(std::move(param_storage)),
        base_graph_(std::move(base_graph)),
        non_param_inputs_(std::move(non_param_inputs)),
        graph_params_(std::move(graph_params)),
        constants_(std::move(constants)),
        functions_(std::move(functions)),
        batch_size_(batch_size),
        dev_num_(dev_num),
        min_pipeline_num_(min_pipeline_num) {
  }

  ~GraphProfiler() {
    clear();
  }

  ProfilingResult init(bool trace_dim_names);
  ProfilingResult profile(const ProfilingInput& input);
  ProfilingResult profile(const ProfilingInput& input, IValueMap values);
  void clear();
  void load(const std::string& file);
  void save(const std::string& file);

  bool hasConstant(const IValueLocation& loc) const;
  void updateConstants(const IValueMap& constants);
  void removeConstant(const IValueLocation& loc);

  const IValueMap& getValues() const {
    return values_;
  }

 private:
  std::shared_ptr<ParamStorage> param_storage_;
  TorchDriver driver_;
  std::shared_ptr<IRGraph> base_graph_;
  std::unordered_map<std::string, torch::jit::IValue> non_param_inputs_;
  std::unordered_map<std::string, long> graph_params_;
  std::unordered_map<
      IValueLocation, std::vector<at::Dimname>, IValueLocationHash>
      dim_names_;
  IValueMap constants_;
  std::shared_ptr<FunctionStorage> functions_;
  size_t batch_size_;
  int dev_num_;
  size_t min_pipeline_num_;

  IValueMap values_;
  ProfileDB profile_db_;

  void backward(
      const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs,
      int split_idx);
  ProfilingResult compute(
      const ProfilingInput& input, IValueMap& values, int split_index,
      bool trace_dim_names);
  std::pair<IValueMap, GraphProfile> computeGraph(
      const std::shared_ptr<IRGraph>& subgraph, const IValueMap& graph_inputs,
      const std::unordered_map<std::string, at::Tensor>& graph_params,
      int iteration, IValueMap& values, int split_index, bool trace_dim_names,
      const DriverExecConf& conf);
  ProfilingResult doProfile(
      const ProfilingInput& input, IValueMap& values, bool trace_dim_names);
  size_t setRequiresGrad(
      const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs);
  std::unordered_map<std::string, at::Tensor> getGraphParams(
      const std::shared_ptr<IRGraph>& graph);

  const std::shared_ptr<spdlog::logger> logger = getLogger("GraphProfiler");
};
} // namespace rannc

#endif // PYRANNC_GRAPHPROFILER_H
