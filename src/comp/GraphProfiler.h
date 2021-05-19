//
// Created by Masahiro Tanaka on 2019/12/20.
//

#ifndef PYRANNC_GRAPHPROFILER_H
#define PYRANNC_GRAPHPROFILER_H

#include <torch/torch.h>
#include <ostream>
#include "graph/ir.h"
#include "torch/TorchUtil.h"
#include "torch/TorchDriver.h"

namespace rannc {

    class ParamStorage;
    class FunctionStorage;

    struct GraphProfile {
        std::string name;
        long fwd_time;
        long bwd_time;
        long max_allocated_mem;
        bool checkpointing;

        friend std::ostream &operator<<(std::ostream &os, const GraphProfile &profile) {
            os << "name: " << profile.name << " fwd_time: " << profile.fwd_time << " bwd_time: " << profile.bwd_time
               << " max_allocated_mem: " << profile.max_allocated_mem
               << " checkpointing: " << profile.checkpointing
               ;
            return os;
        }

        MSGPACK_DEFINE(name, fwd_time, bwd_time, max_allocated_mem, checkpointing);
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
        const std::unordered_map<size_t, ProfileItem> &getItems() const;

    private:
        std::unordered_map<size_t, ProfileItem> items_;
    };

    class GraphProfiler {
    public:
        GraphProfiler(std::shared_ptr<ParamStorage> param_storage,
                      std::shared_ptr<IRGraph> base_graph,
                      std::unordered_map<std::string, torch::jit::IValue> non_param_inputs,
                      const std::unordered_map<std::string, long>& graph_params,
                      IValueMap constants,
                      const FunctionStorage &functions,
                      size_t batch_size, int dev_num, size_t min_pipeline_num)
                :param_storage_(std::move(param_storage)),
                 base_graph_(std::move(base_graph)),
                 non_param_inputs_(std::move(non_param_inputs)),
                 graph_params_(std::move(graph_params)),
                 constants_(std::move(constants)),
                 functions_(functions),
                 batch_size_(batch_size),
                 dev_num_(dev_num),
                 min_pipeline_num_(min_pipeline_num) {
        }

        ~GraphProfiler() {
            clear();
        }

        ProfilingResult init();
        ProfilingResult profile(
                const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& ir_graphs,
                int iteration, size_t replica_num=1, bool checkpointing=false);
        void clear();
        void load(const std::string& file);
        void save(const std::string& file);

    private:
        std::shared_ptr<ParamStorage> param_storage_;
        TorchDriver driver_;
        std::shared_ptr<IRGraph> base_graph_;
        std::unordered_map<std::string, torch::jit::IValue> non_param_inputs_;
        std::unordered_map<std::string, long> graph_params_;
        IValueMap constants_;
        const FunctionStorage & functions_;
        size_t batch_size_;
        int dev_num_;
        size_t min_pipeline_num_;

        IValueMap values_;
        ProfileDB profile_db_;

        void backward(const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs, int split_idx);
        ProfilingResult compute(
                const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& ir_graphs,
                int iteration, IValueMap& values, int split_index, bool checkpointing);
        ProfilingResult doProfile(
                const std::unordered_map<std::string, std::shared_ptr<IRGraph>>& ir_graphs,
                IValueMap& values, int iteration, size_t replica_num, bool checkpointing);
        size_t setRequiresGrad(const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs);
        std::unordered_map<std::string, at::Tensor> getGraphParams(const std::shared_ptr<IRGraph>& graph);

        const std::shared_ptr<spdlog::logger> logger = getLogger("GraphProfiler");
    };
}

#endif //PYRANNC_GRAPHPROFILER_H
