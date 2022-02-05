//
// Created by Masahiro Tanaka on 2021/02/16.
//

#include "Validator.h"
#include <torch/TorchDriver.h>
#include "FunctionStorage.h"

namespace rannc {

bool sameShape(const at::Tensor& t1, const at::Tensor& t2) {
  const auto dim1 = getTensorDim(t1);
  const auto dim2 = getTensorDim(t2);
  if (dim1.size() != dim2.size()) {
    return false;
  }
  for (size_t i = 0; i < dim1.size(); i++) {
    if (dim1.at(i) != dim2.at(i)) {
      return false;
    }
  }
  return true;
}

bool almostEqual(
    const at::Tensor& t1, const at::Tensor& t2, double tolerance,
    double tolerance_ratio) {
  if (t1.scalar_type() != t2.scalar_type()) {
    return false;
  }

  switch (t1.scalar_type()) {
    case c10::ScalarType::Float:
      return almostEqualTensorsWithTolerance<float>(
          t1, t2, tolerance, tolerance_ratio);
    case c10::ScalarType::Double:
      return almostEqualTensorsWithTolerance<double>(
          t1, t2, tolerance, tolerance_ratio);
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return almostEqualTensorsWithTolerance<float>(
          t1.to(c10::ScalarType::Float, false, true),
          t2.to(c10::ScalarType::Float, false, true), tolerance,
          tolerance_ratio);
    default:
      return torch::equal(t1, t2);
  }
}

std::vector<torch::jit::IValue> orderOutputs(
    const IValueMap& outputs, const std::shared_ptr<IRGraph>& graph) {
  std::vector<torch::jit::IValue> ordered_outputs;
  for (const auto& out_name : graph->getOutputNames()) {
    IValueLocation loc(out_name);
    assert(contains(outputs, loc));
    ordered_outputs.push_back(outputs.at(loc));
  }
  return ordered_outputs;
}

std::unordered_map<std::string, at::Tensor> graphParamTensors(
    const std::shared_ptr<IRGraph>& graph,
    const std::unordered_map<std::string, torch::jit::IValue>& params) {
  auto ir_params = graphParamValues(graph);
  std::unordered_map<std::string, at::Tensor> param_tensors;
  for (const auto& irp : ir_params) {
    assert(contains(params, irp.getName()));
    const auto param_iv = params.at(irp.getName());
    assert(param_iv.isTensor());
    param_tensors[irp.getName()] = param_iv.toTensor();
  }
  return param_tensors;
}

IValueMap compute(
    const IValueMap& inputs, const std::shared_ptr<IRGraph>& graph,
    const std::unordered_map<std::string, torch::jit::IValue>& params,
    const IValueMap& const_vals,
    const std::shared_ptr<rannc::FunctionStorage>& functions,
    bool offload_params) {
  std::unordered_map<std::string, at::Tensor> graph_params =
      graphParamTensors(graph, params);

  IValueMap inputs_gpu;
  for (const auto& it : inputs) {
    inputs_gpu[it.first] = toCUDAIfAvailable(it.second, true);
  }
  std::unordered_map<std::string, at::Tensor> params_gpu;
  for (const auto& it : graph_params) {
    params_gpu[it.first] =
        it.second.to(torch::Device(torch::kCUDA), false, true).detach();
  }

  TorchDriver driver(offload_params);
  driver.createModule(
      graph->getName(), graph->getName(), graph, const_vals, functions,
      params_gpu);

  const auto out = driver.forward(graph->getName(), inputs_gpu, 0);
  driver.destroy();

  IValueMap outputs_cpu;
  for (const auto& it : out) {
    outputs_cpu[it.first] = toCPU(it.second, true);
  }

  return outputs_cpu;
}

bool Validator::validate(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const std::vector<torch::jit::IValue>& input_ivals,
    const std::unordered_map<std::string, torch::jit::IValue>& param_inputs,
    const IValueMap& const_vals,
    const std::shared_ptr<FunctionStorage>& functions,
    const Deployment& deployment) {
  torch::NoGradGuard no_grad;

  logger->info("Running torch traced model with dropout disabled ...");
  const auto no_dropout_graph = enableDropout(graph, false);
  auto func = std::make_shared<torch::jit::GraphFunction>(
      "forward", no_dropout_graph, nullptr);
  std::vector<torch::jit::IValue> stack;

  try {
    for (const auto& iv : input_ivals) {
      stack.push_back(toCUDAIfAvailable(iv, true));
    }
    const auto graph_inputs = no_dropout_graph->inputs();
    for (size_t i = input_ivals.size(); i < func->num_inputs(); i++) {
      const auto& in_val = graph_inputs.at(i);
      assert(contains(param_inputs, in_val->debugName()));
      stack.push_back(
          toCUDAIfAvailable(param_inputs.at(in_val->debugName()), true));
    }

    func->run(stack);
  } catch (std::exception& e) {
    logger->error(
        "Failed to run traced model. "
        "The validator may cause OOM because it puts all parameters on CUDA devices. "
        "Disable the validator if Torch reports CUDA device OOM.");
    throw e;
  }
  logger->info("Finished torch traced model");

  std::unordered_set<std::string> dropout_flag_names;
  for (const auto& it : deployment.subgraphs) {
    for (const auto& node : it.second->getNodes()) {
      if (node.getName() == "aten::dropout") {
        assert(node.getInputNames().size() > 2);
        dropout_flag_names.insert(node.getInputNames().at(2));
      }
    }
  }

  IValueMap const_vals_no_do;
  for (const auto& it : const_vals) {
    if (contains(dropout_flag_names, it.first.value_name)) {
      const_vals_no_do[it.first] = 0;
    } else {
      const_vals_no_do[it.first] = it.second;
    }
  }

  const auto initial_inputs = createInputMap(input_ivals, deployment.graph);
  std::unordered_map<std::string, IValueMap> sg_output_vals;
  IValueMap output_vals;

  logger->info("Starting to compute subgraphs with dropout disabled ...");
  for (const auto& sg_name : deployment.fwd_graph_order) {
    assert(contains(deployment.subgraphs, sg_name));
    const auto& sg = deployment.subgraphs.at(sg_name);

    IValueMap sg_inputs;
    for (const auto& r : deployment.fwd_in_routes) {
      if (r.dest_graph == sg_name) {
        assert(contains(initial_inputs, r.location));
        sg_inputs[r.location] = initial_inputs.at(r.location);
      }
    }
    for (const auto& r : deployment.fwd_routes) {
      if (r.dest_graph == sg_name) {
        assert(contains(sg_output_vals, r.source_graph));
        const auto& prev_out = sg_output_vals.at(r.source_graph);
        assert(contains(prev_out, r.location));
        sg_inputs[r.location] = prev_out.at(r.location);
      }
    }

    sg_output_vals[sg_name] = compute(
        sg_inputs, sg, param_inputs, const_vals_no_do, functions,
        deployment.offload_params);

    for (const auto& r : deployment.fwd_out_routes) {
      if (r.source_graph == sg_name) {
        assert(contains(sg_output_vals.at(sg_name), r.location));
        output_vals[r.location] = sg_output_vals[sg_name].at(r.location);
      }
    }
  }

  const auto actual_outputs = orderOutputs(output_vals, deployment.graph);
  assert(stack.size() == actual_outputs.size());
  for (size_t i = 0; i < stack.size(); i++) {
    assert(actual_outputs.at(i).isTensor());
    assert(stack.at(i).isTensor());
    const auto actual = actual_outputs.at(i).toTensor();
    const auto expected = stack.at(i).toTensor();

    if (!sameShape(expected, actual)) {
      logger->error(
          "Resulting shapes do not match. expected={} actual={}",
          join_as_str(getTensorDim(expected)),
          join_as_str(getTensorDim(actual)));
      return false;
    }

    if (!almostEqual(expected, actual, 0.001, 0.01)) {
      logger->error(
          "Results do not match. expected={} actual={}",
          tensorToString(expected), tensorToString(actual));
      return false;
    }
  }
  return true;
}
} // namespace rannc