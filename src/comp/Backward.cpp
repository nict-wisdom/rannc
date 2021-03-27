//
// Created by Masahiro Tanaka on 2019-01-08.
//
#include <memory>

#include <pybind11/pybind11.h>

#include <graph/ir.h>
#include "Backward.h"
#include "bind/RaNNCProcess.h"
#include "EventRecorder.h"

namespace py = pybind11;


namespace rannc {
    torch::jit::IValue doCreateZeroPad(const torch::jit::IValue &value, const PathInIValue& current_path,
                                       const at::Tensor& grad, const PathInIValue &path) {

        if (value.isTensor()) {
            if (current_path == path) {
                return grad;
            }
            return torch::zeros_like(value.toTensor());
        } else if (value.isTensorList()) {
            c10::List<at::Tensor> tensor_list;
            int i=0;
            for (const auto& t: value.toTensorVector()) {
                auto path_elem = current_path;
                path_elem.emplace_back(StepTypeInIValue::LIST, i);
                if (path_elem == path) {
                    tensor_list.push_back(t);
                } else {
                    tensor_list.push_back(torch::zeros_like(t));
                }
                i++;
            }
            return tensor_list;
        } else if (value.isList()) {
            c10::impl::GenericList list(at::AnyType::get());
            std::vector<torch::jit::IValue> ival_list;
            int i=0;
            for (const auto& elem: value.toListRef()) {
                auto path_elem = current_path;
                path_elem.emplace_back(StepTypeInIValue::LIST, i);
                list.push_back(doCreateZeroPad(elem, path_elem, grad, path));
                i++;
            }
            return list;
        } else if (value.isTuple()) {
            std::vector<torch::jit::IValue> ival_list;
            int i=0;
            for (const auto& elem: value.toTuple()->elements()) {
                auto path_elem = current_path;
                path_elem.emplace_back(StepTypeInIValue::TUPLE, i);
                ival_list.push_back(doCreateZeroPad(elem, path_elem, grad, path));
                i++;
            }
            return c10::ivalue::Tuple::create(ival_list);
        }
        throw std::invalid_argument("Output contains non-tensor value");
    }

    torch::jit::IValue createZeroPad(const torch::jit::IValue &ivalue, const PathInIValue& path,
                                     const at::Tensor& grad) {
        return doCreateZeroPad(ivalue, {}, grad, path);
    }

    RaNNCTensorBackward::RaNNCTensorBackward(std::shared_ptr<GraphLauncher> driver,
                                             std::shared_ptr<ParamStorage> param_storage,
                                             std::string graph_id, std::string name,
                                             torch::jit::IValue output, PathInIValue path,
                                             std::vector<IValueLocation> ordered_inputs,
                                             std::vector<long> param_ids_on_rank)
            :output_(std::move(output)),
             ordered_inputs_(std::move(ordered_inputs)), driver_(std::move(driver)),
             param_storage_(std::move(param_storage)),
             graph_id_(std::move(graph_id)), value_name_(std::move(name)), path_(std::move(path)),
             param_ids_on_rank_(std::move(param_ids_on_rank)) {
        skip_grad_scaling_ = config::Config::get().getVal<bool>(config::SKIP_GRAD_SCALING);
    }

    bool RaNNCTensorBackward::delay_grad_allreduce_ = false;

    variable_list RaNNCTensorBackward::apply(variable_list &&grads) {

        try {
            std::stringstream ss;
            ss << "RaNNCTensorBackward::apply_" << toString(value_name_);
            recordStart(ss.str());

            if (grads.size() != 1) {
                throw std::invalid_argument("The size of grads must be one.");
            }

            if (!skip_grad_scaling_) {
                std::stringstream ss_scale;
                ss_scale << "RaNNCTensorBackward::apply_" << toString(value_name_) << "_scaleGrads";
                recordStart(ss_scale.str());
                param_storage_->scaleGrads(graph_id_, false);
                recordEnd(ss_scale.str());
            }

            inputs_[value_name_] = createZeroPad(output_, path_, grads.at(0));
            driver_->backward(graph_id_, inputs_);

            if (delay_grad_allreduce_) {
                if (!skip_grad_scaling_) {
                    param_storage_->unscaleGrads(graph_id_, false);
                }
            } else {
                param_storage_->allReduceParamGrads(graph_id_);
            }

            recordEnd(ss.str());

            variable_list grad_inputs;
            return grad_inputs;
        } catch (c10::Error& e) {
            std::cerr << "Torch exception caught: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        } catch (std::runtime_error& e) {
            std::cerr << "Runtime error caught: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -2);
        } catch (std::invalid_argument& e) {
            std::cerr << "Invalid argument exception caught: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -3);
        } catch (std::exception& e) {
            std::cerr << "Unknown exception caught: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -4);
        }
        std::cerr << "Failed to compute backward. exiting." << std::endl;
        std::exit(-5);
    }
}