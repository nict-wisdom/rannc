//
// Created by Masahiro Tanaka on 2019-02-25.
//

#ifndef PYRANNC_RANNCMODULE_H
#define PYRANNC_RANNCMODULE_H

#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <Logging.h>
#include <Common.h>

#include <comp/ParamStorage.h>

namespace py = pybind11;

namespace rannc {
    class IRGraph;
    class RaNNCProcess;
    class GraphLauncher;

    class RaNNCModule {
    public:
        RaNNCModule(bool use_amp_master_params, bool allreduce_amp_master_param, bool check_unused_values);
        ~RaNNCModule();

        /**
         * Runs forward(). For the first invocation, forward() is traced and the output graph is deployed on workers.
         *
         * @param args Arguments of forward().
         * @param kwargs Unused.
         * @return Output tensor.
         */
        py::object operator()(const py::args& args, const py::kwargs& kwargs);

        std::vector<long> init(const py::function& fwdFunc,
                               const std::vector<py::tuple>& py_params, const std::vector<py::tuple>& py_buffers,
                               const py::function& var_lookup_fn, const py::args& args);
        bool isCheckpointingEnabled() const;

        void allReduceParamGrads();
        void clearParamGrads();
        void clipGrad(float max_grad_norm);
        double calcGradL2Norm();

        at::Tensor syncParam(long pid);
        at::Tensor gatherParam(long param_id, int dest);
        at::Tensor syncParamGrad(long pid);
        at::Tensor gatherParamGrad(long param_id, int dest);

        void saveDeployment(const std::string& deployment_file);

        void setLoadDeployment(bool loadDeployment) {
            load_deployment_ = loadDeployment;
        }

        void setDeploymentFile(const std::string &deploymentFile) {
            deployment_file_ = deploymentFile;
        }

        bool useAmpMasterParams() const;

        void destroy();

    private:
        std::string id_;
        std::shared_ptr<RaNNCProcess> master_;

        std::shared_ptr<IRGraph> ir_graph_;

        std::shared_ptr<GraphLauncher> driver_;
        std::shared_ptr<ParamStorage> param_storage_;
        std::vector<long> param_ids_on_rank_;

        Deployment deployment_;
        bool load_deployment_;
        bool save_deployment_;
        std::string deployment_file_;

        bool checkpointing_enabled_ = false;
        bool use_amp_master_params_;
        bool check_unused_values_;
        bool skip_grad_scaling_;
        bool allreduce_amp_master_param_;

        bool destroyed_ = false;

        std::unordered_map<std::string, BufferTensorCache> buffer_cache_;
        void setGradFn(torch::jit::IValue& out, const std::string& graph_id, const std::string& name,
                const std::vector<IValueLocation>& ordered_inputs);

        const std::shared_ptr<spdlog::logger> logger = getLogger("RaNNCModule");
    };

}
#endif //PYRANNC_RANNCMODULE_H
