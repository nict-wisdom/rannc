//
// Created by Masahiro Tanaka on 2018-11-30.
//

#ifndef PT_RANNC_TORCHDRIVER_H
#define PT_RANNC_TORCHDRIVER_H

#include <torch/csrc/jit/ir/ir.h>
#include <torch/optim/optimizer.h>

#include <comp/TimeCounter.h>
#include <Config.h>
#include <graph/ConvertGraph.h>
#include <Logging.h>
#include <torch/TorchUtil.h>

namespace rannc {

//
//    Forward declarations
//
class FunctionStorage;

class TorchDriver {
 public:
  TorchDriver(bool offload_params) : offload_params_(offload_params) {
    config::Config& config = config::Config::get();
    enable_profiling_ = config.getVal<bool>(config::PROFILING);
    time_counter_.enable(enable_profiling_);
    display_comm_values_ = config.getVal<bool>(config::DISPLAY_COMM_VALUE);
    display_act_values_ = config.getVal<bool>(config::DISPLAY_ACT_VALUE);
  }

  TorchDriver(const TorchDriver&) = delete;

  ~TorchDriver() {
    if (enable_profiling_) {
      logger->info(
          "Rank {} profiling summary\n{}", mpi::getRank(),
          time_counter_.summary<std::chrono::microseconds>());
    }
  }

  /**
   * Creates a Torch module from an *IRGraph*. The *IRGraph* given as an
   * argument is converted to *torch::jit::Graph* using *GraphConv::toTorch()*.
   *
   * @param id ID of the given Graph.
   * @param irGraph Graph to compute.
   * @param constants Constant values used within the graph.
   * @param [in] functions   Functions used within the graph.
   * @param parameters Parameters given to the graph as inputs.
   */
  void createModule(
      const std::string& id, const std::shared_ptr<rannc::IRGraph>& irGraph,
      const IValueMap& constants, const FunctionStorage& functions,
      const std::unordered_map<std::string, at::Tensor>& parameters);

  /**
   * Computes forward propagation.
   *
   * @param id ID of the given Graph.
   * @param inputs Inputs of forward propagation.
   * @return Outputs of forward propagation.
   */
  IValueMap forward(
      const std::string& id, const IValueMap& inputs, int split_idx);

  /**
   * Computes backward propagation.
   *
   * @param id ID of the given Graph.
   * @param inputs Inputs of backward propagation.
   * @return Outputs of backward propagation.
   */
  IValueMap backward(
      const std::string& id, const IValueMap& inputs, int split_idx);

  void destroyModule(const std::string& id);

  void destroy();

  // for profiling
  bool isProfilingEnabled() const;
  void enableProfiling(bool enableProfiling);
  std::string getProfilingSummary() const;

  static void setKeepGraph(bool keep_graph);
  static bool getKeepGraph();

 private:
  void displayValue(
      const std::string& prefix, size_t count, int split_index, bool grad_mode,
      const IValueMap& vals);
  std::vector<at::Tensor> getParamInputTensors(
      const std::string& id, bool init);

  /**
   * Inputs used in the previous forward().
   */
  std::unordered_map<std::string, IValueMap> last_inputs_;
  /**
   * Outputs of the previous forward().
   */
  std::unordered_map<std::string, IValueMap> last_outputs_;

  /**
   * Graphs in IR.  The key is a graph ID.
   */
  std::unordered_map<std::string, std::shared_ptr<rannc::IRGraph>> ir_graphs_;
  std::unordered_map<std::string, std::shared_ptr<rannc::IRGraph>>
      clone_input_ir_graphs_;

  /**
   * Parameter tensors.
   */
  std::unordered_map<std::string, std::unordered_map<std::string, at::Tensor>>
      param_tensors_;
  std::unordered_map<std::string, std::vector<std::string>>
      ordered_param_names_;
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<std::string>>>
      input_clone_names_;
  std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<at::Tensor>>>
      clone_params_;

  /**
   * Script modules built from Torch IR. The key is a graph ID.
   */
  std::unordered_map<std::string, std::shared_ptr<torch::jit::Function>>
      functions_;

  std::unordered_map<std::string, BufferTensorCache> buffer_cache_;

  int last_split_idx_ = INT32_MAX;

  TimeCounter time_counter_;
  bool enable_profiling_;
  bool display_comm_values_;
  bool display_act_values_;
  bool offload_params_;

  // for debugging
  size_t fwd_count_ = 0;
  size_t bwd_count_ = 0;

  const std::shared_ptr<spdlog::logger> logger = getLogger("TorchDriver");

  static bool keep_graph_;
};
} // namespace rannc

#endif // PT_RANNC_TORCHDRIVER_H
