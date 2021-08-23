//
// Created by Masahiro Tanaka on 2019-06-16.
//

#ifndef PYRANNC_GRAPHCONNECTOR_H
#define PYRANNC_GRAPHCONNECTOR_H

#include <graph/Decomposition.h>
#include <torch/TorchDriver.h>

#include "GraphValueStorage.h"
#include "ParamStorage.h"

#include <ATen/cuda/CUDAEvent.h>

namespace rannc {

    //
    //    Forward declarations
    //
    class FunctionStorage;


    class GraphConnector {
    public:
        GraphConnector(std::string id, std::shared_ptr<ParamStorage> param_storage,
                       std::shared_ptr<GraphValueStorage> value_storage,
                       const FunctionStorage & functions)
                : id_(std::move(id)), param_storage_(std::move(param_storage)),
                  value_storage_(std::move(value_storage)),
                  functions_(functions)
        {

            enable_profiling_ = config::Config::get().getVal<bool>(config::PROFILING);
            time_counter_.enable(enable_profiling_);

            verify_recomp_ = config::Config::get().getVal<bool>(config::VERIFY_RECOMP);

        }

        GraphConnector(const GraphConnector&) = delete;

        ~GraphConnector() {
            if (enable_profiling_) {
                logger->info("Rank {} profiling summary\n{}", mpi::getRank(),
                             time_counter_.summary<std::chrono::milliseconds>());
            }
        }

        void deployGraph(const Deployment& deployment);
        std::unordered_map<std::string, IValueMap> forward(
                const std::string& id, const std::unordered_map<std::string, IValueMap>& inputs,
                int split_index, bool grad_mode);
        std::unordered_map<std::string, IValueMap> backward(const std::string& id, const std::unordered_map<std::string, IValueMap>& inputs, int split_index);

    private:
        const std::string id_;
        std::unordered_map<std::string, std::shared_ptr<IRGraph>> graphs_;

        std::shared_ptr<ParamStorage> param_storage_;
        std::shared_ptr<GraphValueStorage> value_storage_;
        const FunctionStorage & functions_;
        TorchDriver driver_;

        std::unordered_map<std::string, std::unordered_map<int, IValueMap>> inputs_;

        // for debugging
        std::unordered_map<std::string, std::unordered_map<int, IValueMap>> last_cp_outputs_;

        std::unordered_map<std::string, int> fwd_graph_order_;
        std::vector<std::string> fwd_sorted_graph_ids_;
        std::unordered_map<std::string, std::vector<RouteDP>> fwd_recv_routes_;
        std::unordered_map<std::string, std::vector<RouteDP>> fwd_send_routes_;
        std::unordered_map<std::string, int> bwd_graph_order_;
        std::vector<std::string> bwd_sorted_graph_ids_;
        std::unordered_map<std::string, std::vector<RouteDP>> bwd_recv_routes_;
        std::unordered_map<std::string, std::vector<RouteDP>> bwd_send_routes_;

        TimeCounter time_counter_;
        bool enable_profiling_;
        bool verify_recomp_;

        std::unordered_map<std::string, bool> checkpointing_;
        // split index -> graph id -> values
        std::unordered_map<int, std::unordered_map<std::string, IValueMap>> split_values_;
        int pipeline_num_;
        int max_fwd_delay_;
        int max_bwd_delay_;
        std::unordered_map<std::string, std::unordered_map<IValueLocation, size_t, IValueLocationHash>> fwd_inc_edges_count_;
        std::unordered_map<std::string, std::unordered_map<IValueLocation, size_t, IValueLocationHash>> bwd_inc_edges_count_;

        std::unordered_map<std::string, std::unordered_map<int, RngState>> rng_states_;
        std::unordered_map<int, bool> skip_fwd_split_;
        std::unordered_map<std::string, std::unordered_map<int, at::cuda::CUDAEvent>> copy_events_;

        torch::jit::IValue distributeOutput(bool is_bwd, const RouteDP& r, int split_index, int flush_offset,
                                            const std::unordered_map<std::string, int>& graph_order);
        std::unordered_map<std::string, IValueMap> compute(const std::string &id,
                bool is_bwd, const std::unordered_map<std::string, IValueMap>& inputs, int split_index,
                const std::unordered_map<std::string, std::vector<RouteDP>>& recv_routes,
                const std::unordered_map<std::string, std::vector<RouteDP>>& send_routes,
                const std::unordered_map<std::string, int>& graph_order,
                const std::vector<std::string>& sorted_graph_ids,
                int max_delay,
                const std::function<IValueMap(const std::string &, const IValueMap &, int)>& func,
                const std::function<void(std::unordered_map<std::string, IValueMap>&,
                                         std::unordered_map<std::string, std::unordered_map<IValueLocation, std::vector<torch::jit::IValue>,IValueLocationHash>>&)>& aggr,
                                         const std::function<std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& input_names_getter,
                const std::function<bool(const IValueMap&, int)>& skip);
        void runDriver(std::unordered_set<std::string>& graphs_done,
                       std::unordered_map<std::string, IValueMap>& values, int split_index,
                       const std::function<IValueMap(const std::string&, const IValueMap&, int)>& f,
                       const std::function<std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& input_names_getter,
                       const std::function<bool(const IValueMap&, int)>& skip);

        const std::shared_ptr<spdlog::logger> logger = getLogger("GraphConnector");
    };

}


#endif //PYRANNC_GRAPHCONNECTOR_H
