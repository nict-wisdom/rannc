//
// Created by Masahiro Tanaka on 2018/11/30.
//
#include <iostream>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <comp/FunctionStorage.h>
#include "Common.h"
#include "ConvertGraph.h"
#include "graph/Decomposition.h"
#include "graph/ir.h"
#include "torch/TorchUtil.h"

namespace {

bool isNumber(const std::string& name) {
  return !name.empty() &&
      name.find_first_not_of("0123456789") == std::string::npos;
}

bool containsIRValue(
    const std::vector<rannc::IRValue>& value_set, const std::string& name) {
  for (const auto& v : value_set) {
    if (v.getName() == name) {
      return true;
    }
  }
  return false;
}

const std::shared_ptr<spdlog::logger> logger() {
  return rannc::getLogger("ConvertGraph");
}
} // namespace

namespace rannc {

torch::jit::TypePtr fromIRListType(const IRType& ir_type) {
  assert(ir_type.getBaseType() == IRBaseType::LIST);
  switch (ir_type.getListType()) {
    case IRListType::INT:
      return torch::jit::IntType::get();
    case IRListType::FLOAT:
      return torch::jit::FloatType::get();
    case IRListType::BOOL:
      return torch::jit::BoolType::get();
    case IRListType::TENSOR:
      return torch::jit::TensorType::get();
    case IRListType::GENERIC: {
      assert(ir_type.getCompoundTypes().size() == 1);
      return fromIRType(ir_type.getCompoundTypes().front());
    }
  }
}

torch::jit::Value* createFunctionConstantNode(
    const std::shared_ptr<FunctionStorage>& functions, const IRValue& irv,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<std::string, torch::jit::Value*>& regValues) {
  const FunctionStorage::FunctionTable& functable = functions->getFunctions();

  const std::string func_name = irv.getFunctionName();
  assert(contains(functable, func_name));
  torch::jit::Function* const callee = functable.at(func_name);

  const std::string out_name = irv.getName();
  const std::string& attr_name = functions->getAttrName(func_name);
  auto node = graph->insertNode(graph->create(c10::prim::Constant));

  torch::jit::Value* fn_constant =
      node->s_(c10::attr::name, attr_name)
          ->output()
          ->setType(c10::FunctionType::create(callee));
  regValues[out_name] = fn_constant->setDebugName(out_name);

  return fn_constant;
}

void ConvertGraph::doToTorch(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<std::string, torch::jit::Value*>& regValues,
    const std::shared_ptr<IRGraph>& irGraph, const IValueMap& constants,
    const std::shared_ptr<FunctionStorage>& function_storage,
    const std::vector<IRValue>& no_param_inputs) {
  const std::unordered_map<std::string, IRValue>& values = irGraph->getValues();
  const FunctionStorage::FunctionTable& functions =
      function_storage->getFunctions();

  for (const auto& irNode : irGraph->getNodes()) {
    //            logger()->info("Adding node. name={}", irNode.getName());

    if (irGraph->isFunctionNode(irNode)) {
      assert(irNode.getInputNames().size() == 0);
      assert(irNode.getOutputNames().size() == 1);

      const std::string& out_name = irNode.getOutputNames().at(0);
      const IRValue& ov = irGraph->getValue(out_name);

      createFunctionConstantNode(function_storage, ov, graph, regValues);
    } else if (irNode.getName() == "prim::CallFunction") {
      std::vector<torch::jit::Value*> inputs = {};

      for (const std::string& argName : irNode.getInputNames()) {
        if (!contains(regValues, argName)) {
          throw std::runtime_error(
              "Value (for argname) is not in the graph. "
              "Nodes may not be topologically sorted: " +
              argName);
        }
        const auto& argVal = regValues.at(argName);
        inputs.push_back(argVal);
      }

      auto n =
          graph->insertNode(graph->create(c10::prim::CallFunction, inputs));
      const std::string& outName = irNode.getOutputNames().at(0);
      assert(contains(values, outName));
      const auto& val = values.at(outName);
      regValues[outName] = n->output()
                               ->setType(fromIRType(val.getType()))
                               ->setDebugName(outName);
    } else if (irNode.getName() == "prim::Constant") {
      //                logger()->info("  Inserting constant. name={}
      //                outputs={}", irNode.getName(),
      //                            join_as_str(irNode.getOutputNames()));
      for (const std::string& outName : irNode.getOutputNames()) {
        assert(contains(values, outName));
        const auto& val = values.at(outName);

        std::vector<std::string> const_keys;
        for (const auto& it : constants) {
          const_keys.push_back(toString(it.first));
        }

        assert(contains(constants, IValueLocation(val.getName())));
        const c10::IValue& iv = constants.at(val.getName());
        auto v = graph->insertConstant(iv);
        v->setType(fromIRType(val.getType()));
        v->setDebugName(outName);
        regValues[val.getName()] = v;
      }
    } else if (irNode.getName() == "prim::ListConstruct") {
      //                logger()->info("  Inserting ListConstruct. name={}
      //                elem_num={}", irNode.getName(),
      //                        irNode.getInputNames().size());

      std::vector<torch::jit::Value*> value_vec;
      for (const std::string& inName : irNode.getInputNames()) {
        auto v = regValues[inName];
        //                    logger()->info("  adding list input. name={}",
        //                    v->debugName());
        value_vec.push_back(v);
      }
      at::ArrayRef<torch::jit::Value*> vals(value_vec);

      const auto& output_names = irNode.getOutputNames();
      assert(!output_names.empty());
      assert(contains(values, output_names.at(0)));
      const auto& output_ir_val = values.at(output_names.at(0));
      assert(output_ir_val.getType().getBaseType() == IRBaseType::LIST);
      const auto list_type = fromIRListType(output_ir_val.getType());
      auto n = graph->createList(list_type, vals);
      graph->appendNode(n);

      auto nodeOutput = n->outputs();
      for (size_t i = 0; i < irNode.getOutputNames().size(); i++) {
        const auto& out_name = irNode.getOutputNames().at(i);
        assert(contains(values, out_name));
        const auto& val = values.at(out_name);
        const auto& name = irNode.getOutputNames().at(i);
        const auto& v = nodeOutput.at(i);

        //                    logger()->info("  adding list output. name={}
        //                    irtype={}", v->debugName(),
        //                    toString(val.getType()));

        v->setType(fromIRType(val.getType()));
        regValues[name] = v;
        if (!isNumber(name)) {
          v->setDebugName(name);
        }
        //                    logger()->info("  added list output. name={}
        //                    type={}", v->uniqueName(),
        //                                 typeKindToString(fromIRType(val.getType())->kind()));
      }
    } else {
      //                logger()->info("  Inserting op. name={}",
      //                irNode.getName());

      auto n = graph->create(
          torch::jit::Symbol::fromQualString(irNode.getName()),
          irNode.getOutputNames().size());

      for (const std::string& inName : irNode.getInputNames()) {
        if (!contains(regValues, inName)) {
          throw std::runtime_error(
              "Value is not in the graph. "
              "Nodes may not be topologically sorted: " +
              inName);
        }

        assert(contains(regValues, inName));
        const auto& in_val = regValues.at(inName);
        assert(contains(values, inName));
        const auto& in_ir_val = values.at(inName);

        // The input to this node is an input to the graph
        // AND this node is an in_place operation
        if (containsIRValue(no_param_inputs, inName) &&
            in_ir_val.getType().getBaseType() == IRBaseType::TENSOR &&
            inPlaceOpName(irNode.getName())) {
          // Then clone
          // An in-place operation on a leaf variable is prohibited
          // AND reuse of inputs for checkpointing may be corrupted due to
          // in-place operations
          torch::jit::Value* none =
              graph->insertNode(graph->createNone())->output();
          auto clone_node = graph->create(
              torch::jit::Symbol::fromQualString("aten::clone"), 1);
          graph->appendNode(clone_node);
          clone_node->addInput(in_val);
          clone_node->addInput(none);

          const auto& clone_outputs = clone_node->outputs();
          assert(clone_outputs.size() == 1);
          const auto& cl_out_val = clone_outputs.at(0);

          std::string cl_name = inName + "_clone";
          cl_out_val->setDebugName(cl_name);

          cl_out_val->setType(fromIRType(in_ir_val.getType()));
          regValues[cl_name] = cl_out_val;

          n->addInput(cl_out_val);
        } else {
          //                        logger()->info("    adding node input.
          //                        name={}", irNode.getName());
          n->addInput(in_val);
        }
      }
      graph->appendNode(n);

      const auto& nodeOutput = n->outputs();
      for (size_t i = 0; i < irNode.getOutputNames().size(); i++) {
        const auto& name = irNode.getOutputNames().at(i);
        const auto& v = nodeOutput.at(i);
        if (!isNumber(name)) {
          v->setDebugName(name);
        }
        //                    logger()->info("    adding node output. name={}",
        //                    v->uniqueName());

        if (!contains(values, name)) {
          std::stringstream ss;
          ss << "Failed to find output. node=" << irNode.getName()
             << " out=" << name << " values=" << join_as_str(keys(values));
          throw std::runtime_error(ss.str());
        }

        assert(contains(values, name));
        const auto& val = values.at(name);
        v->setType(fromIRType(val.getType()));
        regValues[name] = v;
      }
    }
  }
}

std::shared_ptr<torch::jit::Graph> ConvertGraph::toTorch(
    const std::shared_ptr<IRGraph>& irGraph, const IValueMap& constants,
    const std::shared_ptr<FunctionStorage>& functions) {
  //        std::cout << "ConvertGraph::toTorch graph=" <<
  //            *irGraph << std::endl;

  auto graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, IRValue> values = irGraph->getValues();
  std::unordered_map<std::string, torch::jit::Value*> regValues;

  std::vector<IRValue> no_param_inputs_org = graphNonParamInputValues(irGraph);
  std::vector<IRValue> param_inputs = graphParamInputValues(irGraph);

  std::vector<IRValue> no_param_inputs;
  std::vector<IRValue> no_param_functions;
  no_param_inputs.reserve(no_param_inputs_org.size());
  for (IRValue& iv : no_param_inputs_org) {
    if (iv.isFunction()) {
      no_param_functions.push_back(iv);
    } else {
      no_param_inputs.push_back(iv);
    }
  }

  const std::string merged_input_name = "merged_input";
  torch::jit::Value* merged_input = graph->addInput(merged_input_name);
  regValues[merged_input_name] = merged_input;

  // set type of merged_input here
  // because createTupleUnpack requires the number of elements
  std::vector<at::TypePtr> elements;
  elements.reserve(no_param_inputs.size());
  for (auto& irVal : no_param_inputs) {
    elements.push_back(fromIRType(irVal.getType()));
  }
  merged_input->setType(
      static_cast<at::TypePtr>(at::TupleType::create(elements)));

  // insert param inputs here
  for (auto& irVal : param_inputs) {
    torch::jit::Value* param_input = graph->addInput(irVal.getName());
    param_input->setType(fromIRType(irVal.getType()));
    regValues[irVal.getName()] = param_input;
  }

  // Unpack non-param inputs
  torch::jit::Node* unpackInput = graph->createTupleUnpack(merged_input);
  graph->appendNode(unpackInput);
  size_t input_idx = 0;
  for (torch::jit::Value* in : unpackInput->outputs()) {
    auto& irVal = no_param_inputs.at(input_idx);
    in->setDebugName(irVal.getName());
    in->setType(fromIRType(irVal.getType()));
    regValues[irVal.getName()] = in;
    input_idx++;
  }

  //  Re-construct Function(s).
  // const FunctionStorage::FunctionTable & funcs = functions.getFunctions();
  for (const IRValue& irv : no_param_functions) {
    createFunctionConstantNode(functions, irv, graph, regValues);
  }

  doToTorch(graph, regValues, irGraph, constants, functions, no_param_inputs);

  const std::vector<std::string>& outNames = irGraph->getOutputNames();

  std::vector<torch::jit::Value*> out_values;
  out_values.reserve(outNames.size());
  for (const auto& out : outNames) {
    torch::jit::Value* const outVal = regValues[out];
    if (outVal->type()->kind() == c10::TypeKind::FunctionType) {
      //  Exception handling - Function in output(s).
      for (const auto& irNode : irGraph->getNodes()) {}
      for (const auto& iv : constants) {}
      continue;
    }
    out_values.push_back(regValues[out]);
  }

  at::ArrayRef<torch::jit::Value*> outValuesArray(out_values);
  torch::jit::Node* outTuple = graph->createTuple(outValuesArray);

  graph->appendNode(outTuple);
  const auto& graph_out_value = outTuple->output();
  const std::string merged_output_name = "merged_output";
  graph_out_value->setDebugName(merged_output_name);
  graph->registerOutput(graph_out_value);

  return graph;
}

torch::jit::Value* addInput(
    std::shared_ptr<torch::jit::Graph>& graph, const IRValue& ir_value) {
  torch::jit::Value* in_val = graph->addInput(ir_value.getName());
  at::TypePtr type = fromIRType(ir_value.getType());
  in_val->setType(type);
  return in_val;
}

std::shared_ptr<torch::jit::Graph> ConvertGraph::toTorchNoMerge(
    const std::shared_ptr<IRGraph>& irGraph, const IValueMap& constants,
    const std::shared_ptr<FunctionStorage>& functions) {
  auto graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, torch::jit::Value*> regValues;

  std::vector<IRValue> no_param_inputs = graphNonParamInputValues(irGraph);
  std::vector<IRValue> param_inputs = graphParamInputValues(irGraph);

  for (const auto& in : no_param_inputs) {
    regValues[in.getName()] = addInput(graph, in);
  }
  for (const auto& in : param_inputs) {
    regValues[in.getName()] = addInput(graph, in);
  }
  doToTorch(graph, regValues, irGraph, constants, functions, no_param_inputs);

  for (const auto& out : irGraph->getOutputNames()) {
    assert(contains(regValues, out));
    graph->registerOutput(regValues.at(out));
  }

  return graph;
}

torch::jit::TypePtr fromIRType(const IRType& ir_type) {
  auto base_type = ir_type.getBaseType();
  switch (base_type) {
    case IRBaseType::SCALAR:
      return fromIRScalarType(ir_type.getScalarType());
    case IRBaseType::TENSOR: {
      if (ir_type.getTensorElemType() == IRTensorElemType::UNDEF) {
        return torch::jit::TensorType::get();
      } else {
        at::ScalarType tensor_elem_type =
            fromIRTensorElemTypeToScalarType(ir_type.getTensorElemType());
        auto& tensor_dim = ir_type.getTensorDim();
        return torch::jit::TensorType::createContiguous(
            tensor_elem_type, at::kCPU, tensor_dim);
      }
    }
    case IRBaseType::LIST: {
      torch::jit::TypePtr type;
      switch (ir_type.getListType()) {
        case IRListType::INT:
          type = torch::jit::IntType::get();
          break;
        case IRListType::FLOAT:
          type = torch::jit::FloatType::get();
          break;
        case IRListType::BOOL:
          type = torch::jit::BoolType::get();
          break;
        case IRListType::TENSOR:
          type = torch::jit::TensorType::get();
          break;
        case IRListType::GENERIC: {
          assert(ir_type.getCompoundTypes().size() == 1);
          type = fromIRType(ir_type.getCompoundTypes().front());
          break;
        }
      }
      return torch::jit::ListType::create(type);
    }
    case IRBaseType::TUPLE: {
      const auto& ir_elem_types = ir_type.getCompoundTypes();
      std::vector<torch::jit::TypePtr> elem_types;
      elem_types.reserve(ir_elem_types.size());
      for (const auto& elem_type : ir_elem_types) {
        elem_types.push_back(fromIRType(elem_type));
      }
      return torch::jit::TupleType::create(elem_types);
    }
    case IRBaseType::OPTIONAL: {
      const auto& ir_elem_types = ir_type.getCompoundTypes();
      assert(ir_elem_types.size() == 1);
      const auto elem_type = fromIRType(ir_elem_types.at(0));
      return torch::jit::OptionalType::create(elem_type);
    }
    case IRBaseType::STRING: {
      return torch::jit::StringType::get();
    }
    case IRBaseType::NONE: {
      return torch::jit::NoneType::get();
    }
  }
}

bool isBatch(const IRType& type, int batch_size) {
  switch (type.getBaseType()) {
    case IRBaseType::SCALAR:
      return false;
    case IRBaseType::TENSOR: {
      const auto& dim = type.getTensorDim();

      if (dim.empty()) {
        return false;
      }
      if (batch_size < 1) {
        return true;
      }
      int64_t a_bsize = dim.front();
      return batch_size == a_bsize;
    }
    case IRBaseType::LIST: {
      IRListType list_type = type.getListType();
      if (list_type == IRListType::TENSOR || list_type == IRListType::GENERIC) {
        const auto& elem_types = type.getCompoundTypes();
        bool ret = false;
        for (const auto& et : elem_types) {
          ret |= isBatch(et, batch_size);
        }
        return ret;
      }
      return false;
    }
    case IRBaseType::TUPLE: {
      const auto& elem_types = type.getCompoundTypes();
      bool ret = false;
      for (const auto& et : elem_types) {
        ret |= isBatch(et, batch_size);
      }
      return ret;
    }
    case IRBaseType::OPTIONAL:
      return false;
    case IRBaseType::NONE:
      return false;
  }
}

std::shared_ptr<IRGraph> guessBatchValuesByReachability(
    const std::shared_ptr<IRGraph>& g) {
  const auto& bg = toBGL(g);

  std::unordered_map<std::string, IRValue> values = g->getValues();

  for (const auto& in : graph_regular_input_nodes<Vertex, BGraph>(bg)) {
    for (const auto& tgt : all_nodes<Vertex, BGraph>(bg)) {
      if (bg[tgt].type == VALUE) {
        if (is_reachable(in, tgt, bg)) {
          //                        spdlog::info("{} is reachable from {}",
          //                        bg[tgt].name, bg[in].name);
          IRValue& val = values[bg[tgt].name];
          val.setBatch(isBatch(val.getType(), 0));
        }
      }
    }
  }

  std::vector<IRNode> nodes = g->getNodes();
  for (auto& n : nodes) {
    for (const auto& in_name : n.getInputNames()) {
      const auto& in_val = values.at(in_name);

      if (in_val.isBatch()) {
        n.setBatch(!(n.getName() == "aten::size"));
      }
    }
  }

  return std::make_shared<IRGraph>(
      g->getName(), nodes, values, g->getInputNames(), g->getOutputNames());
}

std::shared_ptr<IRGraph> guessBatchValues(const std::shared_ptr<IRGraph>& g) {
  int64_t bsize = 0;
  for (const auto& in_name : g->getInputNames()) {
    const IRValue& in_val = g->getValue(in_name);
    if (in_val.isParam()) {
      continue;
    }

    const auto& type = in_val.getType();
    if (type.getBaseType() == IRBaseType::TENSOR) {
      const auto& dim = type.getTensorDim();
      assert(!dim.empty());

      int64_t a_bsize = dim.front();
      if (bsize < 1) {
        bsize = a_bsize;
      } else if (bsize != a_bsize) {
        logger()->info(
            "Failed to guess a batch size. Detected different batch sizes: {} and {}",
            bsize, a_bsize);
        return g;
      }
    }
  }
  logger()->trace("Setting batch size in traced graph to {}", bsize);

  std::unordered_map<std::string, IRValue> values = g->getValues();
  for (auto& it : values) {
    IRValue& v = it.second;

    if (v.isParam()) {
      continue;
    }

    const auto& type = v.getType();
    v.setBatch(isBatch(type, bsize));
  }

  std::vector<IRNode> nodes = g->getNodes();
  for (auto& n : nodes) {
    for (const auto& in_name : n.getInputNames()) {
      const auto& in_val = values.at(in_name);

      if (in_val.isBatch()) {
        n.setBatch(true);
      }
    }
  }

  return std::make_shared<IRGraph>(
      g->getName(), nodes, values, g->getInputNames(), g->getOutputNames());
}

std::shared_ptr<IRGraph> setValueTypes(
    const std::shared_ptr<IRGraph>& g,
    const std::unordered_map<std::string, IRType>& value_types) {
  std::unordered_map<std::string, IRValue> values = g->getValues();
  for (auto& it : values) {
    IRValue& v = it.second;

    if (v.isFunction()) {
      continue;
    }
    if (!contains(value_types, it.first)) {
      throw std::invalid_argument(
          "No type information in profiling results: " + it.first);
    }
    v.setType(value_types.at(it.first));
  }

  return std::make_shared<IRGraph>(
      g->getName(), g->getNodes(), values, g->getInputNames(),
      g->getOutputNames());
}

std::shared_ptr<IRGraph> fromTorch(
    const std::string& name, const std::shared_ptr<torch::jit::Graph>& graph,
    size_t real_input_num) {
  std::unordered_map<std::string, IRValue> values;
  std::vector<IRNode> nodes;

  size_t input_idx = 0;
  for (torch::jit::Value* graph_in : graph->inputs()) {
    IRValue irVal = toIRValue(graph_in);
    if (input_idx >= real_input_num) {
      //                logger()->info("{} is param", irVal.getName());
      irVal.setParam(true);
    }
    values[graph_in->debugName()] = irVal;
    input_idx++;
  }

  for (auto node : graph->nodes()) {
    std::string nodeKind = node->kind().toQualString();
    //            logger()->info("node: {}", nodeKind);

    //            for (torch::jit::Value* node_in: node->inputs()) {
    //                logger()->info("   node_in: {}", node_in->debugName());
    //            }
    //            for (torch::jit::Value* node_out: node->outputs()) {
    //                logger()->info("   node_out: {} type={} ir_type={}
    //                torch_type={}", node_out->debugName(),
    //                typeKindToString(node_out->type()->kind()),
    //                               toString(toIRValue(node_out)),
    //                               node_out->type()->annotation_str());
    //            }
    for (torch::jit::Value* node_out : node->outputs()) {
      values[node_out->debugName()] = toIRValue(node_out);
    }

    std::vector<std::string> inputNames(node->inputs().size());
    std::transform(
        node->inputs().begin(), node->inputs().end(), inputNames.begin(),
        [](torch::jit::Value* n) { return n->debugName(); });
    std::vector<std::string> outputNames(node->outputs().size());
    std::transform(
        node->outputs().begin(), node->outputs().end(), outputNames.begin(),
        [](torch::jit::Value* n) { return n->debugName(); });

    nodes.emplace_back(node->kind().toQualString(), inputNames, outputNames);
  }

  std::vector<std::string> graphInputNames(graph->inputs().size());
  std::transform(
      graph->inputs().begin(), graph->inputs().end(), graphInputNames.begin(),
      [](torch::jit::Value* n) { return n->debugName(); });
  std::vector<std::string> graphOutputNames(graph->outputs().size());
  std::transform(
      graph->outputs().begin(), graph->outputs().end(),
      graphOutputNames.begin(),
      [](torch::jit::Value* n) { return n->debugName(); });

  auto ir_graph = std::make_shared<IRGraph>(
      name, nodes, values, graphInputNames, graphOutputNames);

  //        std::cout << *ir_graph << std::endl;
  return ir_graph;
}

std::shared_ptr<torch::jit::Graph> enableDropout(
    const std::shared_ptr<torch::jit::Graph>& g, bool flag) {
  auto copy_graph = g->copy();

  for (auto node : copy_graph->nodes()) {
    std::string nodeKind = node->kind().toQualString();
    if (node->kind() == c10::Symbol::fromQualString("aten::dropout")) {
      auto inputs = node->inputs();
      assert(inputs.size() > 2);
      auto flag_out_val = inputs.at(2);
      auto flag_node = flag_out_val->node();
      flag_node->i_(c10::Symbol::attr("value"), flag ? 1 : 0);
    }
  }
  return copy_graph;
}

IValueMap createInputMap(
    const std::vector<torch::jit::IValue>& input_ivals,
    const std::shared_ptr<IRGraph>& graph) {
  IValueMap inputs;
  for (size_t i = 0; i < input_ivals.size(); i++) {
    inputs[graph->getInputNames().at(i)] = input_ivals.at(i);
  }
  return inputs;
}
} // namespace rannc
