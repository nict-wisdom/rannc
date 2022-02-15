//
// Created by Masahiro Tanaka on 2019-06-13.
//

#ifndef PYRANNC_GRAPHLAUNCHER_H
#define PYRANNC_GRAPHLAUNCHER_H

#include <Config.h>
#include <Logging.h>

#include <graph/Decomposition.h>
#include <torch/TorchDriver.h>

#include "comp/GraphConnector.h"
#include "GraphValueStorage.h"

namespace rannc {

//
//    Forward declarations
//
class FunctionStorage;

class GraphLauncher {
 public:
  GraphLauncher(
      std::shared_ptr<ParamStorage> param_storage,
      std::shared_ptr<GraphValueStorage> value_storage,
      std::shared_ptr<FunctionStorage> function_storage, Deployment deployment,
      bool gather_inputs)
      : param_storage_(std::move(param_storage)),
        value_storage_(std::move(value_storage)),
        function_storage_(std::move(function_storage)),
        deployment_(std::move(deployment)),
        gather_inputs_(gather_inputs) {
    enable_profiling_ = config::Config::get().getVal<bool>(config::PROFILING);

    time_counter_.enable(enable_profiling_);
  }

  void deployGraph();
  void undeployGraph(const std::string& id);
  torch::jit::IValue forward(const std::string& id, const IValueMap& inputs);
  IValueMap backward(const std::string& id, const IValueMap& inputs);
  void enableDropout(const std::string& id, bool enable);

 private:
  IValueMap compute(
      const std::string& id, bool is_bwd, int64_t batch_size,
      const IValueMap& inputs, std::vector<RouteDP>& in_routes,
      std::vector<RouteDP>& out_routes);

  IValueMap alignBatch(
      const IValueMap& input, int batch_size,
      const std::shared_ptr<IRGraph>& graph, bool zero_pad);

  std::shared_ptr<ParamStorage> param_storage_;
  std::shared_ptr<GraphValueStorage> value_storage_;
  std::shared_ptr<FunctionStorage> function_storage_;
  std::unordered_map<std::string, std::shared_ptr<GraphConnector>> driver_;
  std::unordered_map<std::string, std::atomic_bool> bwd_running_;

  Deployment deployment_;

  std::unordered_map<std::string, std::shared_ptr<Blob>> buffer;
  RouteDP bcast_route_;
  int64_t last_batch_size_;

  TimeCounter time_counter_;
  bool enable_profiling_;
  bool gather_inputs_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("GraphLauncher");
};
} // namespace rannc

#endif // PYRANNC_GRAPHLAUNCHER_H
