//
// Created by Masahiro Tanaka on 2018-11-30.
//
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/engine.h>

#include <Common.h>
#include "ConfiguredTorch.h"
#include <graph/ConvertGraph.h>
#include <comp/EventRecorder.h>
#include <torch/CustomOps.h>
#include <cuda/CudaUtil.h>

#include "TorchDriver.h"



namespace rannc {

    const torch::jit::IValue matchIValue(const IValueMap &ival_map, const IValueLocation& loc) {

        IValueLocation key_loc{loc.value_name};
        if (contains(ival_map, key_loc)) {
            return getElemInIValue(ival_map.at(key_loc), loc.path);
        }

        auto path_rest = loc.path;
        for (const auto& step: loc.path) {
            key_loc.path.push_back(step);
            path_rest.erase(path_rest.begin());
            if (contains(ival_map, key_loc)) {
                return getElemInIValue(ival_map.at(key_loc), path_rest);
            }
        }
        throw std::invalid_argument("Location not found: " + toString(loc));
    }

    void processTensorInIValue(const torch::jit::IValue &ivalue, const std::function<void(at::Tensor)>& f) {
        if (ivalue.isTensor()) {
            f(ivalue.toTensor());
        } else if (ivalue.isTensorList()) {
            for (const auto &t: ivalue.toTensorVector()) {
                f(t);
            }
        } else if (ivalue.isList()) {
            for (auto e: ivalue.toListRef()) {
                processTensorInIValue(e, f);
            }
        } else if (ivalue.isTuple()) {
            for (auto &e: ivalue.toTuple()->elements()) {
                processTensorInIValue(e, f);
            }
        }
    }

    void clearGradIfDefined(at::Tensor t) {
        if (t.grad().defined()) {
            t.grad().zero_();
        }
    }

    void clearGradsInIValue(torch::jit::IValue &ivalue) {
        processTensorInIValue(ivalue, clearGradIfDefined);
    }

    void setZerosToGradInIValueIfUndefined(torch::jit::IValue &ivalue,
                                           BufferTensorCache& buf_cache, const std::string& key) {
        processTensorInIValue(ivalue, [&buf_cache, &key](at::Tensor t) {
            if (!t.grad().defined()) {
                auto type = toIRType(t);
                type.setRequiresGrad(false);
                getMutableGradRef(t) = buf_cache.get(key, type);
                t.grad().zero_();
            }
        });
    }

    torch::jit::IValue sumGradTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues,
            BufferTensorCache& buf_cache, const std::string& key) {

        return aggregateTensorsInIValues(ivalues, [&buf_cache, &key](const std::vector<at::Tensor>& tensors) {
            at::Tensor grad_sum;
            for (const auto& input_ten: tensors) {
                if (!grad_sum.defined()) {
                    auto type = toIRType(input_ten);
                    type.setRequiresGrad(false);
                    grad_sum = buf_cache.get(key, type);
                    grad_sum.zero_();
                }

                if (input_ten.grad().defined()) {
                    torch::NoGradGuard no_grad;
                    grad_sum.add_(input_ten.grad());
                }
            }

            return grad_sum;
        });
    }

    std::vector<torch::jit::IValue> getNonParamInputElems(const std::shared_ptr<IRGraph>& ir_graph,
                                                          const std::unordered_map<std::string, std::vector<std::string>>& input_clone_names,
                                                          const IValueMap& graphIn, bool fwd) {
        std::vector<torch::jit::IValue> non_param_input_elems;
        auto &in_names = ir_graph->getInputNames();
        for (const std::string &in: in_names) {
            auto &val = ir_graph->getValue(in);
            if(val.isFunction()) {
                //  Exception handling - Function in input(s).
                continue;
            }
            if (!val.isParam()) {
                if (fwd || passedForBackward(val.getType())) {
                    if (contains(input_clone_names, in)) {
                        for (const auto &cl_name: input_clone_names.at(in)) {
                            non_param_input_elems.push_back(graphIn.at(cl_name));
                        }
                    } else {
                        non_param_input_elems.push_back(graphIn.at(in));
                    }
                }
            }
        }
        return non_param_input_elems;
    }

    std::vector<at::Tensor> TorchDriver::getParamInputTensors(const std::string& id, bool init) {
        std::vector<at::Tensor> res;

        for (const auto& param_name: ordered_param_names_[id]) {
            const std::unordered_map<std::string, std::vector<std::string>>& clone_names = input_clone_names_[id];
            const auto& p = param_tensors_[id].at(param_name);
            if (contains(clone_names, param_name)) {
                for (const auto& cl_name: clone_names.at(param_name)) {
                    std::stringstream ss;
                    ss << "[PARAM_CLONE]" << id << "_" << cl_name;
                    at::Tensor t_copy = buffer_cache_[id].get(ss.str(), toIRType(p));

                    if (init) {
                        torch::NoGradGuard no_grad_guard;
                        t_copy.copy_(p, false);
                        if (t_copy.grad().defined()) {
                            t_copy.grad().zero_();
                        }
                        clone_params_[id][param_name].push_back(t_copy);
                    }
                    t_copy.set_requires_grad(p.requires_grad());
                    res.emplace_back(t_copy);
                }
            } else {
                res.emplace_back(p);
            }
        }
        return res;
    }

    std::shared_ptr<IRGraph> insertValueHook(const std::shared_ptr<IRGraph>& g, IValueMap &constants) {
        std::vector<IRNode> new_nodes;
        const std::unordered_map<std::string, IRValue> &vals = g->getValues();
        std::unordered_map<std::string, IRValue> new_values;
        std::unordered_map<std::string, std::string> hook_out_names;

        for (const auto& in_name: g->getInputNames()) {
            assert(contains(vals, in_name));
            new_values[in_name] = vals.at(in_name);
        }

        for (const auto &n: g->getNodes()) {
            std::vector<std::string> input_names;
            for (const auto &in_name: n.getInputNames()) {
                if (contains(hook_out_names, in_name)) {
                    input_names.push_back(hook_out_names.at(in_name));
                } else {
                    input_names.push_back(in_name);
                }
            }

            IRNode new_node(n.getName(), input_names, n.getOutputNames());
            new_node.setBatch(n.isBatch());
            new_node.setCriterion(n.isCriterion());
            new_nodes.push_back(new_node);

            for (const auto &out_name: n.getOutputNames()) {
                assert(contains(vals, out_name));
                const IRValue& out_val = vals.at(out_name);

                new_values[out_name] = out_val;
                if (out_val.isBatch()) {
                    const std::string out_name_var = out_name + "_name";
                    IRNode var_name_node("prim::Constant", {}, {out_name_var});
                    new_nodes.push_back(var_name_node);

                    IRValue var_name_val(out_name_var, IRType::createStringType());
                    new_values[out_name_var] = var_name_val;

                    const std::string hook_out_name = out_name + "_hook_out";
                    IRNode hook("rannc::valueHook", {out_name, out_name_var}, {hook_out_name});
                    hook.setBatch(n.isBatch());
                    hook.setCriterion(n.isCriterion());
                    new_nodes.push_back(hook);

                    hook_out_names[out_name] = hook_out_name;
                    new_values[hook_out_name] = IRValue(hook_out_name, out_val);

                    constants[out_name_var] = torch::jit::IValue(out_name);
                }
            }
        }

        std::vector<std::string> output_names;
        for (const auto& out_name: g->getOutputNames()) {
            assert(contains(vals, out_name));
            if (contains(hook_out_names, out_name)) {
                output_names.push_back(hook_out_names.at(out_name));
            } else {
                output_names.push_back(out_name);
            }
        }

        return std::make_shared<IRGraph>(g->getName(), new_nodes, new_values, g->getInputNames(), output_names);
    }

    bool TorchDriver::keep_graph_ = false;

    void TorchDriver::createModule(const std::string &id,
                                   const std::shared_ptr<rannc::IRGraph> &irGraph,
                                   const IValueMap &constants,
                                   const FunctionStorage & functions,
                                   const std::unordered_map<std::string, at::Tensor> &parameters) {
        logger->trace("TorchDriver::createModule starting");

        time_counter_.start("TorchDriver::createModule");

        ir_graphs_[id] = irGraph;
        param_tensors_[id] = parameters;

        const auto& clone_results = cloneSharedInputs(irGraph);
        clone_input_ir_graphs_[id] = clone_results.first;
        input_clone_names_[id] = clone_results.second;

        IValueMap constants_mod = constants;

        if (display_act_values_) {
            clone_input_ir_graphs_[id] = insertValueHook(clone_input_ir_graphs_[id], constants_mod);
        }

        ConvertGraph cg;
        auto graph = cg.toTorch(clone_input_ir_graphs_[id], constants_mod, functions);

        logger->trace("Finished to convert graph.");
        logger->debug("Subgraph {} deployed: {}", id, graph->toString());

        const auto &input_names = irGraph->getInputNames();
        size_t input_idx = input_names.size() - parameters.size();
        for (size_t i = input_idx; i < input_names.size(); i++) {
            const std::string& param_name = input_names.at(i);
            ordered_param_names_[id].push_back(param_name);
            assert(contains(parameters, param_name));
            param_tensors_[id][param_name] = parameters.at(param_name);
        }

        // Clear gradients for safety.
        // If you profiled a graph and ran backward, the gradients of the params still remain non-zero
        for (auto& it: param_tensors_[id]) {
            auto& grad = it.second.grad();
            if (grad.defined()) {
                grad.zero_();
            }
        }

        logger->trace("TorchDriver::createModule creating function.");
        functions_[id] = std::make_shared<torch::jit::GraphFunction>("forward", graph, nullptr);

        syncStream();
        time_counter_.stop("TorchDriver::createModule");

        logger->trace("TorchDriver::createModule finished");
    }

    void TorchDriver::displayValue(const std::string& prefix, size_t count, int split_index,
            bool grad_mode, const IValueMap &vals) {

        if (display_comm_values_) {
            const auto tid = std::this_thread::get_id();
            if (vals.empty()) {
                this->logger->info("@rank{} {} no data to display", mpi::getRank(), prefix);
            } else {
                for (const auto &it: vals) {
                    const auto &loc = it.first;
                    const std::function<at::Tensor(const at::Tensor &, const IValueLocation)> f =
                            [this, &prefix, &tid, count, split_index, grad_mode](const at::Tensor &t,
                                                                                 const IValueLocation &loc) {
                                this->logger->info("@rank{} {} count={} tid={} split={} grad={}: {} {} {} v={}",
                                                   mpi::getRank(), prefix,
                                                   count, toString(tid), split_index, grad_mode,
                                                   toString(loc), toString(toIRType(t)),
                                                   t.device().str(),
                                                   tensorToString(t));
                                return at::Tensor();
                            };
                    transformTensorsInIValueWithPath(it.second, loc.value_name, f);
                }
            }
        }
    }

    IValueMap TorchDriver::forward(const std::string &id, const IValueMap &inputs, int split_idx) {

        bool grad_mode = torch::autograd::GradMode::is_enabled();
        logger->trace("TorchDriver::forward starting. id={} split={} grad_mode={}", id, split_idx, grad_mode);

        recordStart(getFuncKey("TorchDriver", "forward", id, split_idx, grad_mode));

        assert(contains(ir_graphs_, id));

        const auto tc_key = "TorchDriver::forward_" + id;
        time_counter_.start(tc_key);

        displayValue("forward input", fwd_count_, split_idx, grad_mode, inputs);

        recordStart(getFuncKey("TorchDriver", "forward_copy_in", id, split_idx, grad_mode));

        IValueMap graphIn;
        std::unordered_map<std::string, std::vector<std::string>>& clone_names = input_clone_names_[id];
        for (const auto &in: inputs) {
            if (contains(clone_names, in.first.value_name)) {
                for (const auto& cl_name: clone_names.at(in.first.value_name)) {
                    std::stringstream ss;
                    ss << "[SHARED_IN]" << id << "_" << cl_name;

                    const auto cl_ivalue = cloneTensorsInIValueWithBuffer(in.second, ss.str(), buffer_cache_[id]);
                    graphIn[cl_name] = cl_ivalue;
                }
            } else {
                graphIn[in.first] = in.second;
            }
        }

        assert(contains(clone_input_ir_graphs_, id));
        const auto& ir_g = clone_input_ir_graphs_.at(id);
        const auto& values = ir_g->getValues();

        for (const auto& it: graphIn) {
            assert(contains(values, it.first.value_name));

            const auto& type = values.at(it.first.value_name).getType();
            for (const auto& path: findPathsToTensorInIValue(it.second)) {
                const auto elem_type = getElemInIRType(type, path);
                auto elem = getElemInIValue(it.second, path);

                assert(elem.isTensor());
                auto t = elem.toTensor();
                t.set_requires_grad(elem_type.requiresGrad());
            }
        }

        recordEnd(getFuncKey("TorchDriver", "forward_copy_in", id, split_idx, grad_mode));

        IValueMap &graphOut = last_outputs_[id];
        graphOut.clear();

        // get the order of inputs from IRGraph and create an input tuple
        std::vector<torch::jit::IValue> non_param_input_elems = getNonParamInputElems(ir_graphs_[id], input_clone_names_[id], graphIn, true);
        auto non_param_inputs = c10::ivalue::Tuple::create(non_param_input_elems);
        std::vector<torch::jit::IValue> stack;
        stack.emplace_back(non_param_inputs);

        recordStart(getFuncKey("TorchDriver", "forward_copy_param", id, split_idx, grad_mode));

        auto& graph_clone_params = clone_params_[id];
        if (split_idx <= last_split_idx_) {
            graph_clone_params.clear();
        }

        for (const auto &p: getParamInputTensors(id, split_idx <= last_split_idx_)) {
            stack.emplace_back(p);
        }

        recordEnd(getFuncKey("TorchDriver", "forward_copy_param", id, split_idx, grad_mode));

        logger->trace("TorchDriver::forward starting torch engine. id={}", id);
        functions_[id]->run(stack);
        torch::jit::IValue out = stack.front();
        logger->trace("TorchDriver::forward finished torch engine. id={}", id);

        auto &outElems = out.toTuple()->elements();
        auto &outNames = ir_graphs_[id]->getOutputNames();
        for (size_t i = 0; i < outElems.size(); i++) {
            auto elem = outElems.at(i);
            graphOut[outNames.at(i)] = contiguous(elem);
        }

        syncStream();

        time_counter_.stop(tc_key);

        displayValue("forward output", fwd_count_, split_idx, grad_mode, graphOut);

        recordEnd(getFuncKey("TorchDriver", "forward", id, split_idx, grad_mode));

        fwd_count_++;

        last_inputs_[id] = graphIn;
        last_split_idx_ = split_idx;

        logger->trace("TorchDriver::forward finished. id={} split={}", id, split_idx);

        return graphOut;
    }

    void matchGrads(const torch::jit::IValue& out_iv, const torch::jit::IValue& grad_iv,
                    std::vector<at::Tensor>& out_tensors, std::vector<at::Tensor>& grad_tensors) {

        if (out_iv.isTensor()) {
            assert(grad_iv.isTensor());
            auto grad_ten = grad_iv.toTensor();
            if (!grad_ten.defined()) {
                return;
            }
            out_tensors.push_back(out_iv.toTensor());
            grad_ten.set_requires_grad(false);
            grad_tensors.push_back(grad_ten);
        } else if (out_iv.isTensorList()) {
            assert(grad_iv.isTensorList());
            const auto& out_iv_elems = out_iv.toTensorList();
            const auto& grad_iv_elems = grad_iv.toTensorList();
            assert(out_iv_elems.size() == grad_iv_elems.size());

            for (size_t i=0; i<out_iv_elems.size(); i++) {
                matchGrads(out_iv_elems.get(i), grad_iv_elems.get(i), out_tensors, grad_tensors);
            }
        } else if (out_iv.isList()) {
            const auto& out_iv_elems = out_iv.toList();
            const auto& grad_iv_elems = grad_iv.toList();
            assert(out_iv_elems.size() == grad_iv_elems.size());

            for (size_t i=0; i<out_iv_elems.size(); i++) {
                matchGrads(out_iv_elems.get(i), grad_iv_elems.get(i), out_tensors, grad_tensors);
            }
        } else if (out_iv.isTuple()) {
            assert(grad_iv.isTuple());
            auto out_iv_elems = out_iv.toTuple()->elements();
            auto grad_iv_elems = grad_iv.toTuple()->elements();
            assert(out_iv_elems.size() == grad_iv_elems.size());

            for (size_t i=0; i<out_iv_elems.size(); i++) {
                matchGrads(out_iv_elems.at(i), grad_iv_elems.at(i), out_tensors, grad_tensors);
            }
        } else {
            throw std::invalid_argument("Unsupported type of backward input.");
        }
    }

    IValueMap TorchDriver::backward(const std::string &id, const IValueMap &inputs, int split_idx) {
        const auto tc_key = "TorchDriver::backward_" + id;

        recordStart(getFuncKey("TorchDriver", "backward", id, split_idx, false));

        assert(contains(ir_graphs_, id));

        logger->trace("TorchDriver::backward starting. id={} split={}", id, split_idx);
        time_counter_.start(tc_key);

        // Filter unnecessary inputs
        IValueMap required_inputs;
        auto &irGraph = ir_graphs_[id];
        const auto& output_names = irGraph->getOutputNames();
        for (const auto& it: inputs) {
            const auto& loc = it.first;
            if (contains(output_names, loc.value_name)) {
                if (passedForBackward(irGraph->getValue(loc.value_name).getType())) {
                    required_inputs[loc] = inputs.at(loc);
                }
            }
        }

        IValueMap &graphOut = last_outputs_[id];

        displayValue("backward input", bwd_count_, split_idx, false, required_inputs);

        std::vector<at::Tensor> out_tensors;
        std::vector<at::Tensor> grad_tensors;
        for (const auto &in_iv: required_inputs) {
            auto out_iv = matchIValue(graphOut, in_iv.first);
            matchGrads(out_iv, in_iv.second, out_tensors, grad_tensors);
        }

        IValueMap inGrads;
        if (out_tensors.empty()) {
            return inGrads;
        }

        auto &graphLastIn = last_inputs_[id];

        for (const auto& in_name: ir_graphs_[id]->getInputNames()) {
            const auto& val = irGraph->getValue(in_name);
            if (val.isParam()) {
                continue;
            }
            const auto& type = val.getType();

            if (passedForBackward(type)) {
                std::unordered_map<std::string, std::vector<std::string>>& clone_names = input_clone_names_[id];
                if (contains(clone_names, in_name)) {
                    for (const auto& cl_name: clone_names.at(in_name)) {
                        assert(contains(graphLastIn, cl_name));
                        clearGradsInIValue(graphLastIn.at(cl_name));
                    }
                } else {
                    assert(contains(graphLastIn, in_name));
                    clearGradsInIValue(graphLastIn.at(in_name));
                }
            }
        }

        std::vector<torch::autograd::Edge> edges;
        std::vector<torch::autograd::Variable> grad_vars;
        for (size_t i=0; i<out_tensors.size(); i++) {
            auto out_var = out_tensors.at(i);
            auto autograd_meta = torch::autograd::impl::get_autograd_meta(out_var);

            if (autograd_meta == nullptr || autograd_meta->grad_fn_ == nullptr) {
                getMutableGradRef(out_tensors.at(i)) = grad_tensors.at(i).clone();
            } else {
                grad_vars.push_back(grad_tensors.at(i));
                edges.emplace_back(autograd_meta->grad_fn_, autograd_meta->output_nr_);
            }
        }

        recordStart(getFuncKey("TorchDriver", "backward_engine", id, split_idx, false));
        logger->trace("TorchDriver::backward starting torch engine. id={}", id);
        torch::autograd::Engine::get_default_engine().execute(edges, grad_vars, getKeepGraph(), false, true);
        logger->trace("TorchDriver::backward finished torch engine. id={}", id);
        recordEnd(getFuncKey("TorchDriver", "backward_engine", id, split_idx, false));

        recordStart(getFuncKey("TorchDriver", "backward_sum_ingrad", id, split_idx, false));

        for (const auto& in_name: ir_graphs_[id]->getInputNames()) {
            const auto& val = irGraph->getValue(in_name);
            if (val.isParam()) {
                continue;
            }
            const auto& type = val.getType();

            if (passedForBackward(type)) {
                at::Tensor grad_ten;
                std::unordered_map<std::string, std::vector<std::string>>& clone_names = input_clone_names_[id];
                std::vector<torch::jit::IValue> input_ivals;
                std::stringstream ss;
                ss << "[IN_GRAD]" <<  id << "_" << in_name;

                if (contains(clone_names, in_name)) {
                    for (const auto& cl_name: clone_names.at(in_name)) {
                        assert(contains(graphLastIn, cl_name));
                        input_ivals.push_back(graphLastIn.at(cl_name));
                    }
                    inGrads[in_name] = sumGradTensorsInIValues(input_ivals, buffer_cache_[id], ss.str());

                    // Clear grads of shared input
                    for (const auto& cl_name: clone_names.at(in_name)) {
                        const auto& cl_in = graphLastIn.at(cl_name);
                        if (cl_in.isTensor()) {
                            auto cl_ten = cl_in.toTensor();
                            if (cl_ten.grad().defined()) {
                                getMutableGradRef(cl_ten) = at::Tensor();
                            }
                        }
                    }
                } else {
                    assert(contains(graphLastIn, in_name));
                    auto iv = graphLastIn.at(in_name);
                    setZerosToGradInIValueIfUndefined(iv, buffer_cache_[id], ss.str());
                    const auto grad_iv = transformTensorsInIValue(iv, [](const at::Tensor &t) {
                        return t.grad().contiguous();
                    });
                    inGrads[in_name] = grad_iv;
                }
            }
        }
        recordEnd(getFuncKey("TorchDriver", "backward_sum_ingrad", id, split_idx, false));

        recordStart(getFuncKey("TorchDriver", "backward_sum_paramgrad", id, split_idx, false));

        size_t param_idx = 0;
        auto& graph_clone_params = clone_params_[id];
        for (const auto& param_name: ordered_param_names_[id]) {
            std::unordered_map<std::string, std::vector<std::string>>& clone_names = input_clone_names_[id];
            if (contains(clone_names, param_name)) {
                torch::NoGradGuard no_grad;

                at::Tensor grad_sum;
                assert(contains(graph_clone_params, param_name));
                for (const auto& cl_param: graph_clone_params.at(param_name)) {
                    if (!grad_sum.defined()) {
                        std::stringstream ss;
                        ss << "[GRAD_SUM]" << id << "_" << param_name;
                        auto type = toIRType(cl_param);
                        type.setRequiresGrad(false);
                        grad_sum = buffer_cache_[id].get(ss.str(), type);
                        grad_sum.zero_();
                    }
                    if (cl_param.grad().defined()) {
                        grad_sum.add_(cl_param.grad());
                        cl_param.grad().zero_();
                    }
                    param_idx++;
                }

                assert(contains(param_tensors_[id], param_name));
                auto& grad = getMutableGradRef( param_tensors_[id].at(param_name) );
                if (grad.defined()) {
                    grad.add_(grad_sum);
                } else {
                    std::stringstream ss;
                    ss << "[GRAD]" << id << "_" << param_name;
                    auto type = toIRType(grad_sum);
                    type.setRequiresGrad(false);
                    grad = buffer_cache_[id].get(ss.str(), type);
                    grad.copy_(grad_sum);
                }
            } else {
                param_idx++;
            }
        }

        recordEnd(getFuncKey("TorchDriver", "backward_sum_paramgrad", id, split_idx, false));

        recordStart(getFuncKey("TorchDriver", "backward_sync", id, split_idx, false));
        syncStream();
        recordEnd(getFuncKey("TorchDriver", "backward_sync", id, split_idx, false));

        if (!getKeepGraph()) {
            graphOut.clear();
        }

        displayValue("backward output", bwd_count_, split_idx, false, inGrads);

        time_counter_.stop(tc_key);

        logger->trace("TorchDriver::backward finished. id={} split={}", id, split_idx);

        recordEnd(getFuncKey("TorchDriver", "backward", id, split_idx, false));

        bwd_count_++;

        return inGrads;
    }

    void TorchDriver::destroyModule(const std::string &id) {
        last_inputs_.erase(id);
        last_outputs_.erase(id);
        ir_graphs_.erase(id);

        param_tensors_.erase(id);
        ordered_param_names_.erase(id);
        input_clone_names_.erase(id);
        clone_params_.erase(id);

        functions_.erase(id);

        buffer_cache_.erase(id);
    }

    void TorchDriver::destroy() {
        const auto all_ids = keys(ir_graphs_);
        for (const auto& id: all_ids) {
            destroyModule(id);
        }
    }

    bool TorchDriver::isProfilingEnabled() const {
        return time_counter_.isEnabled();
    }

    void TorchDriver::enableProfiling(bool enableProfiling) {
        time_counter_.enable(enableProfiling);
    }

    std::string TorchDriver::getProfilingSummary() const {
        return time_counter_.summary<std::chrono::milliseconds>();
    }

    void TorchDriver::setKeepGraph(bool keep_graph) {
        keep_graph_ = keep_graph;
    }

    bool TorchDriver::getKeepGraph() {
        return keep_graph_;
    }
}
