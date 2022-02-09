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
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> ir_graphs;
  size_t batch_size = 0;
  int iteration = 0;
  size_t replica_num = 0;
  size_t pipeline_num = 0;
  bool checkpointing = false;
  TensorPartioningGraphInfo part_info;

  MSGPACK_DEFINE(
      ir_graphs, batch_size, iteration, replica_num, pipeline_num,
      checkpointing, part_info);
};

struct ProfilingResult {
  std::unordered_map<std::string, GraphProfile> node_profiles;
  std::unordered_map<std::string, IRType> value_types;

  MSGPACK_DEFINE(node_profiles, value_types);
};

struct ProfileItemKey {
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> ir_graphs;
  size_t batch_size;
  size_t repl_num;
  int iteration;
  bool checkpointing;

  MSGPACK_DEFINE(ir_graphs, batch_size, repl_num, iteration, checkpointing);
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

class GraphValueCache {
 public:
  GraphValueCache(size_t batch_size) : batch_size_(batch_size) {}

  void put(const std::string& name, const torch::jit::IValue& value);
  torch::jit::IValue get(const std::string& name, size_t batch_size);

 private:
  std::unordered_map<std::string, torch::jit::IValue> values_;
  size_t batch_size_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("GraphValueCache");
};

class GraphProfiler {
 public:
  GraphProfiler(
      std::shared_ptr<ParamStorage> param_storage,
      std::shared_ptr<IRGraph> base_graph,
      std::unordered_map<std::string, torch::jit::IValue> non_param_inputs,
      std::unordered_map<std::string, long> graph_params, IValueMap constants,
      std::shared_ptr<FunctionStorage> functions, size_t batch_size,
      int dev_num, size_t min_pipeline_num, bool offload_params)
      : param_storage_(std::move(param_storage)),
        base_graph_(std::move(base_graph)),
        non_param_inputs_(std::move(non_param_inputs)),
        graph_params_(std::move(graph_params)),
        constants_(std::move(constants)),
        functions_(std::move(functions)),
        batch_size_(batch_size),
        dev_num_(dev_num),
        min_pipeline_num_(min_pipeline_num),
        driver_(offload_params) {
    cache_param_values_ = false;
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

  bool isCacheParamValues() const {
    return cache_param_values_;
  }

  void setCacheParamValues(bool cacheParamValues) {
    cache_param_values_ = cacheParamValues;
    if (!cacheParamValues) {
      param_cache_.clear();
    }
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

  bool cache_param_values_;
  std::unordered_map<std::string, at::Tensor> param_cache_;

  void backward(
      const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs,
      int split_idx);
  ProfilingResult compute(
      const ProfilingInput& input, IValueMap& values, int split_index,
      bool trace_dim_names);
  std::pair<IValueMap, GraphProfile> computeGraph(
      const std::shared_ptr<IRGraph>& subgraph, const IValueMap& graph_inputs,
      const std::unordered_map<std::string, at::Tensor>& graph_params,
      int iteration, IValueMap& values, int split_index, bool checkpointing,
      bool trace_dim_names);
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
