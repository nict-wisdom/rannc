//
// Created by Masahiro Tanaka on 2019-02-25.
//

#include "RaNNCModule.h"

#include <torch/csrc/jit/python/python_ivalue.h>

#include <graph/ir.h>
#include <Common.h>
#include <Config.h>
#include <cuda/CudaUtil.h>
#include <comm/ObjectComm.h>
#include <bind/RaNNCFactory.h>
#include <bind/RaNNCProcess.h>
#include <bind/PybindUtil.h>
#include <graph/ConvertGraph.h>
#include <bind/Tracer.h>
#include <comp/FunctionStorage.h>
#include <comp/DistributedParamLocator.h>
#include <comp/DistributedGradLocator.h>
#include <graph/GuessValueTypes.h>
#include <graph/MetaDecomposer.h>
#include <graph/Partitioner.h>
#include <graph/DeploymentSerializer.h>

#include "Backward.h"
#include "GraphProfiler.h"
#include "EventRecorder.h"
#include "Validator.h"

namespace {

    /*
     * Matches param names to param ids.
     */
    std::unordered_map<std::string, long> matchParamNames(const std::shared_ptr<torch::jit::Graph>& graph,
                                                          const size_t real_input,
                                                          const std::vector<py::tuple> &py_params,
                                                          const std::vector<py::tuple> &py_buffers) {

        std::unordered_map<std::string, long> result;
        const auto& inputs = graph->inputs();
        std::unordered_map<std::string, torch::jit::IValue> params_with_names;

        for (size_t i=0; i<py_params.size(); i++) {
            const auto& name = inputs.at(i+real_input)->debugName();
            result[name] = rannc::getPythonObjId(py_params.at(i)[1]);
        }

        for (size_t i=0; i<py_buffers.size(); i++) {
            const auto& name = inputs.at(i+real_input+py_params.size())->debugName();
            result[name] = rannc::getPythonObjId(py_buffers.at(i)[1]);
        }

        return result;
    }
}


namespace rannc {

    void syncDebugName(std::shared_ptr<torch::jit::Graph>& graph) {
        ObjectComm& ocomm = ObjectComm::get();

        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<torch::jit::Value*> in_nodes;
        std::vector<torch::jit::Value*> out_nodes;

        for (auto i: graph->inputs()) {
            input_names.push_back(i->debugName());
            in_nodes.push_back(i);
        }
        input_names = ocomm.bcast(input_names);

        for (auto n: graph->nodes()) {
            for (auto o: n->outputs()) {
                output_names.push_back(o->debugName());
                out_nodes.push_back(o);
            }
        }
        output_names = ocomm.bcast(output_names);

        assert(in_nodes.size() == input_names.size());
        for (size_t idx=0; idx<in_nodes.size(); idx++) {
            in_nodes.at(idx)->setDebugName(input_names.at(idx));
        }

        assert(out_nodes.size() == output_names.size());
        for (size_t idx=0; idx<out_nodes.size(); idx++) {
            out_nodes.at(idx)->setDebugName(output_names.at(idx));
        }
    }

    RaNNCModule::RaNNCModule(bool use_amp_master_params, bool allreduce_amp_master_param,
                             bool enable_zero, bool check_unused_values):
            id_(generateName("mod_")), master_(RaNNCFactory::get()),
            use_amp_master_params_(use_amp_master_params),
            allreduce_amp_master_param_(allreduce_amp_master_param),
            enable_zero_(enable_zero),
            check_unused_values_(check_unused_values) {

        ObjectComm& ocomm = ObjectComm::get();
        id_ = ocomm.bcast(id_);

        config::Config& conf = config::Config::get();
        load_deployment_ = conf.getVal<bool>(config::LOAD_DEPLOYMENT);
        save_deployment_ = conf.getVal<bool>(config::SAVE_DEPLOYMENT);
        deployment_file_ = conf.getVal<std::string>(config::DEPLOYMENT_FILE);
        skip_grad_scaling_ = conf.getVal<bool>(config::SKIP_GRAD_SCALING);

        param_storage_ = master_->getParamStorage();
        param_storage_->useAmpMasterParams(id_, use_amp_master_params_);
        bool consolidate_grads = conf.getVal<bool>(config::CONSOLIDATE_GRADS);

        param_storage_->setConsolidate(consolidate_grads);
        param_storage_->setAllreduceAmpMasterParams(allreduce_amp_master_param);
        master_->registerModule(id_, this);
    }

    RaNNCModule::~RaNNCModule() {
        if (!destroyed_) {
            destroy();
        }
    }

    std::vector<IValueLocation> orderInputs(const py::args& args, const std::shared_ptr<IRGraph>& graph) {
        std::vector<IValueLocation> ordered_locs;
        for (size_t i=0; i<args.size(); i++) {
            ordered_locs.emplace_back(graph->getInputNames().at(i));
        }
        return ordered_locs;
    }


    void RaNNCModule::doRegisterParams(const std::vector<py::tuple>& py_params, bool is_buffer) {
        for (const py::tuple& p: py_params) {
            bool distributed = false;
            if (pybind11::hasattr(p[1], "distributed")) {
                distributed = py::cast<bool>(pybind11::getattr(p[1], "distributed"));
            }

            const auto ten = py::cast<at::Tensor>(p[1]);
            long pid = getPythonObjId(p[1]);
            param_storage_->registerParam(pid, ten, is_buffer, distributed);
        }
    }

    std::vector<long> RaNNCModule::init(const py::function& fwdFunc,
                                        const std::vector<py::tuple>& py_params, const std::vector<py::tuple>& py_buffers,
                                        const py::function& var_lookup_fn, const py::args& args,
                                        bool gather_inputs) {

        doRegisterParams(py_params, false);
        doRegisterParams(py_buffers, true);

        std::vector<torch::jit::IValue> input_ivals = torch::jit::_toTypeInferredIValue(args).toTuple()->elements();
        int64_t local_batch_size = guessBatchSize(input_ivals);
        int64_t batch_size = gather_inputs ? mpi::allReduceSumBatchSize(local_batch_size) : local_batch_size;

        if (mpi::isMaster()) {
            logger->info("Tracing model ...");
        }
        std::shared_ptr<torch::jit::Graph> graph = trace(args, fwdFunc, py_params, py_buffers, var_lookup_fn, 2);
        syncDebugName(graph);

        std::unordered_map<std::string, long> graph_params = matchParamNames(graph, input_ivals.size(), py_params, py_buffers);
        bool distributed = false;
        for (const auto &it: graph_params) {
            if (param_storage_->distributed(it.second)) {
                distributed = true;
            } else {
                auto p = param_storage_->getParamTensor(it.second);
                toCPUInPlace(p);
            }
        }

        if (mpi::isMaster()) {
            logger->debug("Traced graph: {}", graph->toString());
            logger->info("Converting torch model to IR ...");
        }
        ir_graph_ = fromTorch(id_, graph, args.size());
        if (ir_graph_->getNodes().empty()) {
            std::stringstream ss;
            ss << "The target model is empty: " << graph->toString();
            throw std::invalid_argument(ss.str());
        }

        IValueMap input_map = createInputMap(input_ivals, ir_graph_);

        /////////////////
        // call profiler to exactly know tensor sizes
        // The shape of some tensors in a graph is unknown after we migrated to pt120
        std::unordered_map<std::string, torch::jit::IValue> non_param_inputs;
        for (const auto& it: input_map) {
            non_param_inputs[it.first.value_name] = it.second;
        }

        std::unordered_map<std::string, torch::jit::IValue> param_inputs;
        if (!distributed) {
            for (const auto &it: graph_params) {
                param_inputs[it.first] = param_storage_->getParamTensor(it.second);
            }
        }

        auto value_storage = std::make_shared<GraphValueStorage>();
        value_storage->deploy(graph);

        FunctionStorage function_storage;
        function_storage.deploy(graph);

        DistributedParamLocator& zpl = DistributedParamLocator::get();
        zpl.fetchStart();
        if (mpi::isMaster()) {
            const int min_pipeline = config::Config::get().getVal<int>(config::MIN_PIPELINE);
            std::shared_ptr<GraphProfiler> sg_prof =
                    std::make_shared<GraphProfiler>(param_storage_, ir_graph_, non_param_inputs, graph_params,
                                                    value_storage->getValues(),
                                                    function_storage,
                                                    batch_size, mpi::getSize(), min_pipeline);

            bool load_profile = config::Config::get().getVal<bool>(config::LOAD_GRAPH_PROFILE);
            std::string graph_profile_file = config::Config::get().getVal<std::string>(config::GRAPH_PROFILE_FILE);
            if (load_profile) {
                logger->info("Loading graph profiles from {}", graph_profile_file);
                sg_prof->load(graph_profile_file);
            }

            logger->info("Running profiler ...");
            pybind11::gil_scoped_release no_gil;
            ProfilingResult prof_results;
            try {
                prof_results = sg_prof->init();
            } catch (std::exception& e) {
                std::stringstream ss;
                ss << "Failed to profile graph."
                    << " Try to reduce the batch size or increase min_pipline_num."
                    << " message=" << e.what();
                throw std::runtime_error(ss.str());
            }
            logger->info("Profiling finished");
            ir_graph_ = setValueTypes(ir_graph_, prof_results.value_types);
            ir_graph_ = guessValueTypes(ir_graph_);
            ir_graph_->setBatchSize(batch_size);
            logger->info("Assuming batch size: {}", batch_size);

            if (check_unused_values_) {
                const auto unused_vals = findUnusedValue(ir_graph_);
                if (!unused_vals.empty()) {
                    logger->warn("Unused value(s) found in graph. It is likely that the autograd graph will not "
                                 "be properly constructed. "
                                 "To ignore this warning, set check_unused_values=True when creating RaNNCModule. "
                                 "unused_values={} graph={}",
                                 join_as_str(unused_vals),
                                 graph->toString());
                    throw std::runtime_error("The graph is invalid.");
                }
            }

            const auto dev_info = ::rannc::getCudaDeviceInfo(getCurrentCudaDeviceId());

            if (load_deployment_) {
                logger->info("Loading deployment state from {}", deployment_file_);
                deployment_ = load(deployment_file_, mpi::getSize(), dev_info.total_mem);
                deployment_.id = id_;

                ProfilerUtil prof_util(sg_prof);
                logger->info("Estimated profiles of subgraphs: batch_size={} ranks={} pipeline_num={}",
                        batch_size, mpi::getSize(), deployment_.pipeline_num);
                for (const auto& it: deployment_.subgraphs) {
                    assert(contains(deployment_.allocation, it.first));
                    int repl_num = deployment_.allocation.at(it.first).size();
                    const auto& g = it.second;
                    const auto prof = prof_util.profile(g, batch_size, repl_num*deployment_.pipeline_num,
                                                        deployment_.checkpointing);

                    int opt_param_factor = config::Config::get().getVal<int>(config::OPT_PARAM_FACTOR);
                    size_t opt_mem = getOptMemSize(g, opt_param_factor, use_amp_master_params_, enable_zero_, repl_num);

                    size_t bs = ceil(batch_size / (double) (repl_num*deployment_.pipeline_num));
                    auto scaled = std::make_shared<IRGraph>("scaled", *g);
                    scaled->setBatchSize(bs);
                    size_t comm_buf = calcCommBufSize(scaled, deployment_.pipeline_num);

                    long ar_time = calcAllReduceTime(g->getParamSizeInByte());

                    size_t total = prof.max_allocated_mem + opt_mem + comm_buf;

                    size_t fp32params = 0;
                    size_t fp16params = 0;
                    for (const auto& in_name: g->getInputNames()) {
                        const auto& in_v = g->getValue(in_name);
                        if (in_v.isParam()) {
                            assert(in_v.getType().getBaseType() == IRBaseType::TENSOR);
                            if (in_v.getType().getTensorElemType() == IRTensorElemType::FLOAT) {
                                fp32params += in_v.getSizeInByte();
                            } else if (in_v.getType().getTensorElemType() == IRTensorElemType::HALF
                                    || in_v.getType().getTensorElemType() == IRTensorElemType::BFLOAT16) {
                                fp16params += in_v.getSizeInByte();
                            } else {
                                logger->debug("Unknown elem type of parameter {}: {}", in_name,
                                             toString(in_v.getType().getTensorElemType()));
                            }
                        }
                    }

                    logger->info("  graph={} repl={} fwd_time={} bwd_time={} ar_time={} in_size={} out_size={}"
                                 " fp32param_size={} fp16params_size={} mem={} (fwd+bwd={} opt={} comm={})",
                                  g->getName(), repl_num,
                                  prof.fwd_time, prof.bwd_time, ar_time,
                                  calcInputSize(scaled), calcOutputSize(scaled),
                                  fp32params, fp16params,
                                  total, prof.max_allocated_mem, opt_mem, comm_buf);
                }

                logger->info("Allocations: dev_num={}", mpi::getSize());
                for (const auto& it: deployment_.subgraphs) {
                    assert(contains(deployment_.allocation, it.first));
                    logger->info("   {} ranks={}", it.first, join_as_str(deployment_.allocation.at(it.first)));
                }
            } else {
                MetaDecomposer decomposer(sg_prof, mpi::getSize(), batch_size, prof_results.node_profiles,
                                          dev_info.total_mem, use_amp_master_params_);
                const auto decomp_name = config::Config::get().getVal<std::string>(config::DECOMPOSER);
                deployment_ = decomposer.decompose(decomp_name, ir_graph_);

                if (save_deployment_) {
                    logger->info("Saving deployment state to {}", deployment_file_);
                    save(deployment_file_, deployment_, mpi::getSize(), dev_info.total_mem);
                }
            }

            for (const auto& it: deployment_.subgraphs) {
                logger->trace("subgraph {}", toString(*it.second));
            }
            for (const auto& r: deployment_.fwd_in_routes) {
                logger->trace("fwd_in_routes {}", toString(r));
            }
            for (const auto& r: deployment_.fwd_routes) {
                logger->trace("fwd_routes {}", toString(r));
            }
            for (const auto& r: deployment_.fwd_out_routes) {
                logger->trace("fwd_out_routes {}", toString(r));
            }
            for (const auto& r: deployment_.bwd_in_routes) {
                logger->trace("bwd_in_routes {}", toString(r));
            }
            for (const auto& r: deployment_.bwd_routes) {
                logger->trace("bwd_routes {}", toString(r));
            }
            for (const auto& r: deployment_.bwd_out_routes) {
                logger->trace("bwd_out_routes {}", toString(r));
            }

            verifyDeployment(deployment_);
            logger->info("Routes verification passed.");

            bool save_profile = config::Config::get().getVal<bool>(config::SAVE_GRAPH_PROFILE);
            if (save_profile) {
                logger->info("Saving graph profiles to {}", graph_profile_file);
                sg_prof->save(graph_profile_file);
            }

            if (config::Config::get().getVal<bool>(config::VERIFY_PARTITIONING)) {
                if (distributed) {
                    logger->warn("Verification was disabled because zero param distribution is enabled.");
                } else {
                    Validator validator;
                    if (!validator.validate(graph, input_ivals, param_inputs, value_storage->getValues(),
                                            function_storage, deployment_)) {
                        throw std::runtime_error("Partitioning verification failed.");
                    }
                    logger->info("Partitioning verification passed.");
                }
            }
        }

        emptyCache();
        zpl.fetchEnd();
        MPI_Barrier(MPI_COMM_WORLD);

        ObjectComm& ocomm = ObjectComm::get();
        if (mpi::getRank() == 0) {
            logger->debug("Broadcasting deployment ...");
        }
        deployment_ = ocomm.bcast(deployment_);
        ir_graph_ = deployment_.graph;
        checkpointing_enabled_ = deployment_.checkpointing;

        if (mpi::getRank() == 0) {
            logger->debug("Deploying parameters ...");
        }
        param_storage_->deploy(deployment_, graph_params, enable_zero_);

        driver_ = std::make_shared<GraphLauncher>(
                        param_storage_, value_storage,
                        function_storage,
                        deployment_.pipeline_num,
                        gather_inputs);
        logger->debug("Calling driver->deployGraph");
        driver_->deployGraph(deployment_);
        logger->debug("Finished calling driver->deployGraph");

        // List params unused in graphs on this rank
        for (const auto& it: deployment_.subgraphs) {
            const auto& sg_name = it.first;
            const auto& sg = it.second;

            assert(contains(deployment_.allocation, sg_name));
            std::unordered_set<int> sg_ranks = deployment_.allocation.at(sg_name);
            if (!contains(sg_ranks, mpi::getRank())) {
                continue;
            }

            for (const auto& in_name: sg->getInputNames()) {
                const auto& ir_in = sg->getValue(in_name);
                if (ir_in.isParam()) {
                    long pid = param_storage_->getParamID(id_, in_name);
                    param_ids_on_rank_.push_back(pid);
                }
            }
        }
        logger->debug("Found {} param(s) required on this rank.", param_ids_on_rank_.size());

        for (const auto& it: graph_params) {
            long pid = it.second;
            if (!contains(param_ids_on_rank_, pid)) {
                param_storage_->unregisterParam(pid);
            }
        }

        logger->info("RaNNCModule is ready. (rank{})", mpi::getRank());

        return param_ids_on_rank_;
    }

    py::object RaNNCModule::operator()(const py::args& args, const py::kwargs& kwargs) {

        const std::string key = "RaNNCModule::operator()";
        recordStart(key);

        std::vector<torch::jit::IValue> stack = torch::jit::_toTypeInferredIValue(args).toTuple()->elements();

        // forward here
        const auto inputs = createInputMap(stack, ir_graph_);
        torch::jit::IValue out = driver_->forward(id_, inputs);

        if (torch::autograd::GradMode::is_enabled()) {
            const auto ordered_locs = orderInputs(args, ir_graph_);
            setGradFn(out, id_, ir_graph_->getOutputNames().at(0), ordered_locs);
        }

        recordEnd(key);

        return torch::jit::toPyObject(std::move(out));
    }

    void RaNNCModule::allReduceParamGrads() {
        if (!skip_grad_scaling_) {
            param_storage_->scaleGrads(id_, allreduce_amp_master_param_);
        }
        param_storage_->allReduceParamGrads(id_);
    }

    void RaNNCModule::clearParamGrads() {
        param_storage_->clearParamGrads(id_);
    }

    bool RaNNCModule::useAmpMasterParams() const {
        return use_amp_master_params_;
    }

    void RaNNCModule::clipGrad(float max_grad_norm) {
        param_storage_->clipGradNorm(id_, max_grad_norm, use_amp_master_params_);
    }

    double RaNNCModule::calcGradL2Norm() {
        return param_storage_->calcGradGlobalL2Norm(id_, use_amp_master_params_);
    }

    bool RaNNCModule::isCheckpointingEnabled() const{
        return checkpointing_enabled_;
    }

    void RaNNCModule::saveDeployment(const std::string& deployment_file) {
        logger->info("Saving deployment state to {}", deployment_file);
        const auto dev_info = ::rannc::getCudaDeviceInfo(getCurrentCudaDeviceId());
        save(deployment_file, deployment_, mpi::getSize(), dev_info.total_mem);
    }

    void RaNNCModule::setGradFn(torch::jit::IValue& out, const std::string& graph_id, const std::string& name,
                                const std::vector<IValueLocation>& ordered_inputs) {

        std::stringstream ss;
        ss << "out_clone_buf_" << name;
        const auto key = ss.str();
        const auto clone_out = cloneTensorsInIValueWithBuffer(out, key, buffer_cache_[graph_id]);

        const auto& paths = findPathsToTensorInIValue(out);
        for (const auto& path: paths) {
            auto elem = getElemInIValue(out, path);
            assert(elem.isTensor());
            auto var = elem.toTensor();

            auto func = std::shared_ptr<RaNNCTensorBackward>(
                    new RaNNCTensorBackward(driver_, param_storage_, graph_id, name, clone_out,
                            path, ordered_inputs, param_ids_on_rank_));
            func->add_input_metadata(var);

            torch::autograd::Edge e(func, 0);
            torch::autograd::impl::set_gradient_edge(var, e);
        }
    }

    at::Tensor RaNNCModule::syncParam(long param_id) {
        return param_storage_->syncParam(param_id);
    }

    at::Tensor RaNNCModule::syncParamGrad(long param_id) {
        return param_storage_->syncParamGrad(param_id);
    }

    void RaNNCModule::syncParamZero(bool grad) {
        param_storage_->bcastParamsZero(id_, grad);
    }

    at::Tensor RaNNCModule::getLocalParamSegment(long param_id) {
        return param_storage_->getLocalParamSegment(param_id);
    }

    void RaNNCModule::destroy() {
        if (driver_) {
            driver_->undeployGraph(id_);
        }
        if (param_storage_) {
            param_storage_->releaseGraphParams(id_);
        }
        ir_graph_.reset();
        driver_.reset();
        param_ids_on_rank_.clear();
        if (master_) {
            master_->unregisterModule(id_);
        }

        destroyed_ = true;
    }
}
