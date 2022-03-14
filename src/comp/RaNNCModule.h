//
// Created by Masahiro Tanaka on 2019-02-25.
//

#ifndef PYRANNC_RANNCMODULE_H
#define PYRANNC_RANNCMODULE_H

#include <pybind11/pybind11.h>

#undef USE_DISTRIBUTED
#undef USE_RPC
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/torch.h>

#include <Common.h>
#include <Logging.h>

#include <comp/ParamStorage.h>
#include "FunctionStorage.h"
#include "GraphValueStorage.h"

namespace py = pybind11;

namespace rannc {
class IRGraph;
class RaNNCProcess;
class GraphLauncher;
class DistributedGradLocator;

class RaNNCModule {
 public:
  RaNNCModule(
      bool use_amp_master_params, bool allreduce_amp_master_param,
      bool enable_zero, bool check_unused_values, bool offload_params);
  ~RaNNCModule();

  /**
   * Runs forward(). For the first invocation, forward() is traced and the
   * output graph is deployed on workers.
   *
   * @param args Arguments of forward().
   * @param kwargs Unused.
   * @return Output tensor.
   */
  py::object operator()(const py::args& args, const py::kwargs& kwargs);

  std::vector<long> init(
      const py::function& fwdFunc, const std::vector<py::tuple>& py_params,
      const std::vector<py::tuple>& py_buffers,
      const py::function& var_lookup_fn, const py::args& args,
      bool gather_inputs);
  bool isCheckpointingEnabled() const;

  void allReduceParamGrads();
  void allReduceParamGradsZero(double loss_scale);
  void clearParamGrads();
  void clipGrad(float max_grad_norm);
  double calcGradL2Norm();

  at::Tensor syncParam(long param_id);
  at::Tensor syncParamGrad(long param_id);
  void syncParamZero(bool grad);
  at::Tensor getLocalParamSegment(long param_id);
  std::tuple<int64_t, int64_t> getLocalParamRange(long param_id);
  at::Tensor getParam(long param_id, bool amp_master_param);
  at::Tensor getParamGrad(long param_id, bool amp_master_param);

  void saveDeployment(const std::string& deployment_file);

  void setLoadDeployment(bool loadDeployment) {
    load_deployment_ = loadDeployment;
  }

  void setDeploymentFile(const std::string& deploymentFile) {
    deployment_file_ = deploymentFile;
  }

  bool useAmpMasterParams() const;

  void destroy();

  void enableDropout(bool enable);

 private:
  std::string id_;
  std::shared_ptr<RaNNCProcess> master_;

  std::shared_ptr<IRGraph> ir_graph_;

  std::shared_ptr<GraphLauncher> driver_;
  std::shared_ptr<ParamStorage> param_storage_;
  std::shared_ptr<GraphValueStorage> value_storage_;
  std::shared_ptr<FunctionStorage> func_storage_;
  std::vector<long> param_ids_on_rank_;

  Deployment deployment_;
  bool load_deployment_;
  bool save_deployment_;
  std::string deployment_file_;
  int dry_run_np_;
  bool load_profile_;
  std::string graph_profile_file_;
  bool use_named_tensors_;
  std::string decomp_name_;
  bool save_profile_;
  bool verify_partitioning_;
  size_t prof_cache_size_;

  bool checkpointing_enabled_ = false;
  bool use_amp_master_params_;
  bool check_unused_values_;
  bool allreduce_amp_master_param_;
  bool enable_zero_;
  bool offload_params_;

  bool destroyed_ = false;

  std::unordered_map<std::string, BufferTensorCache> buffer_cache_;
  void setGradFn(
      torch::jit::IValue& out, const std::string& graph_id,
      const std::string& name,
      const std::vector<IValueLocation>& ordered_inputs);
  void doRegisterParams(
      const std::vector<py::tuple>& py_params, bool is_buffer);
  at::Tensor doGetParam(long param_id, bool grad, bool amp_master_param);

  const std::shared_ptr<spdlog::logger> logger = getLogger("RaNNCModule");
};

} // namespace rannc
#endif // PYRANNC_RANNCMODULE_H
