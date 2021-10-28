//
// Created by Masahiro Tanaka on 2019-02-25.
//

#undef USE_DISTRIBUTED
#undef USE_RPC

#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/python/python_tracer.h>

#include <torch/TorchUtil.h>

#include "PybindUtil.h"

using namespace torch::autograd;

namespace rannc {

std::vector<int> with_rank_stack;

void enterRank(int rank) {
  with_rank_stack.push_back(rank);
}

int exitRank() {
  int prev = with_rank_stack.back();
  with_rank_stack.pop_back();
  return prev;
}

int getCurrentRank() {
  if (with_rank_stack.empty())
    return 0;
  return with_rank_stack.back();
}

torch::jit::TypedIValue tranformTensorsInTypedIValue(
    const torch::jit::TypedIValue& typed_ivalue,
    const std::function<at::Tensor(const at::Tensor&)>& f,
    const std::function<torch::jit::TypePtr(const torch::jit::TypePtr&)>& tf) {
  const auto& ivalue = typed_ivalue.first;
  const auto& type = typed_ivalue.second;

  if (ivalue.isTensor()) {
    return {f(ivalue.toTensor()), tf(type)};
  } else if (ivalue.isTensorList()) {
    std::vector<at::Tensor> tensors;
    std::vector<torch::jit::TypePtr> elem_types;

    torch::jit::TypePtr listType = nullptr;
    for (const auto& t : ivalue.toTensorVector()) {
      tensors.push_back(f(t));
      elem_types.push_back(tf(type));
      auto unify = torch::jit::unifyOrInitializeType(listType, tf(type));
      if (!unify) {
        AT_ERROR(
            "List inputs to traced functions must have consistent element type");
      }
      listType = *unify;
    }
    return {tensors, torch::jit::ListType::create(listType)};
  } else if (ivalue.isList()) {
    const c10::ArrayRef<torch::jit::IValue>& elems = ivalue.toListRef();
    const auto& elem_types = type->containedTypes();
    assert(elems.size() == elem_types.size());

    //            std::vector<torch::jit::IValue> trans_elems;
    c10::impl::GenericList list(at::AnyType::get());
    std::vector<torch::jit::TypePtr> trans_types;
    torch::jit::TypePtr listType = nullptr;
    for (size_t i = 0; i < elems.size(); i++) {
      auto tiv =
          tranformTensorsInTypedIValue({elems.at(i), elem_types.at(i)}, f, tf);
      list.push_back(tiv.ivalue());
      auto unify = torch::jit::unifyOrInitializeType(listType, tiv.type());
      if (!unify) {
        AT_ERROR(
            "List inputs to traced functions must have consistent element type");
      }
      listType = *unify;
    }
    return {list, listType};
  } else if (ivalue.isTuple()) {
    const std::vector<torch::jit::IValue>& elems = ivalue.toTuple()->elements();
    const auto& elem_types = type->containedTypes();
    assert(elems.size() == elem_types.size());

    std::vector<torch::jit::IValue> trans_elems;
    std::vector<torch::jit::TypePtr> trans_types;

    for (size_t i = 0; i < elems.size(); i++) {
      auto tiv =
          tranformTensorsInTypedIValue({elems.at(i), elem_types.at(i)}, f, tf);
      trans_elems.push_back(tiv.ivalue());
      trans_types.push_back(tiv.type());
    }
    return {
        c10::ivalue::Tuple::create(trans_elems),
        c10::TupleType::create(trans_types)};
  }
  return {ivalue, type};
}
//
//    torch::jit::TypePtr toFloatTypeIfHalfType(const torch::jit::TypePtr& type)
//    {
//        assert(type->kind() == c10::TypeKind::CompleteTensorType);
//
//        const auto ctt = type->expect<torch::jit::CompleteTensorType>();
//        if (ctt->scalarType() == at::ScalarType::Half) {
//            return
//            torch::jit::CompleteTensorType::create(at::ScalarType::Float,
//            ctt->device(), ctt->sizes());
//        }
//        return type;
//    }
//
//    torch::jit::TypedIValue toFloatTensorsInTypedIValue(const
//    torch::jit::TypedIValue &ivalue) {
//        return tranformTensorsInTypedIValue(ivalue, toFloatIfHalf,
//        toFloatTypeIfHalfType);
//    }
//
torch::jit::TypedIValue sliceBatchTensorsInTypedIValue(
    const torch::jit::TypedIValue& ivalue, int offset, int count) {
  return tranformTensorsInTypedIValue(
      ivalue,
      [offset, count](const at::Tensor& t) {
        return t.slice(0, offset, offset + count);
      },
      [count](const torch::jit::TypePtr& type) {
        const auto ctt = type->expect<torch::jit::TensorType>();
        assert(!ctt->sizes().concrete_sizes().value().empty());
        ctt->sizes().concrete_sizes().value()[0] = count;
        return torch::jit::TensorType::create(
            ctt->scalarType(), ctt->device(), ctt->sizes(), ctt->strides(),
            c10::nullopt);
      });
}

std::vector<std::pair<std::string, at::Tensor>> toNamedTensors(
    const std::vector<py::tuple>& pytuple_tensors) {
  std::vector<std::pair<std::string, at::Tensor>> named_tensors;
  for (const py::tuple& p : pytuple_tensors) {
    named_tensors.emplace_back(
        py::cast<std::string>(p[0]), py::cast<at::Tensor>(p[1]));
  }
  return named_tensors;
}

std::vector<at::Tensor> toTensors(
    const std::vector<py::tuple>& params,
    const std::vector<py::tuple>& buffers) {
  std::vector<at::Tensor> params_and_bufs;
  for (const py::tuple& p : params) {
    const at::Tensor& ten = py::cast<at::Tensor>(p[1]);
    params_and_bufs.push_back(ten);
  }
  for (const py::tuple& b : buffers) {
    const at::Tensor& ten = py::cast<at::Tensor>(b[1]);
    params_and_bufs.push_back(ten);
  }

  return params_and_bufs;
}

static bool CompareIValue(
    const std::pair<torch::jit::IValue, torch::jit::IValue>& aWrap,
    const std::pair<torch::jit::IValue, torch::jit::IValue>& bWrap) {
  const auto a = aWrap.first;
  const auto b = bWrap.first;
  if (a.isString() && b.isString()) {
    return a.toStringRef().compare(b.toStringRef()) < 0;
  } else if (a.isInt() && b.isInt()) {
    return a.toInt() < b.toInt();
  } else if (a.isDouble() && b.isDouble()) {
    return a.toDouble() < b.toDouble();
  }
  AT_ERROR("Illegal dict key");
}

std::vector<std::pair<torch::jit::IValue, torch::jit::IValue>> iterationOrder(
    const c10::Dict<torch::jit::IValue, torch::jit::IValue>& dict) {
  std::vector<std::pair<torch::jit::IValue, torch::jit::IValue>> ordered;
  for (auto& element : dict) {
    ordered.emplace_back(element.key(), element.value());
  }
  std::sort(ordered.begin(), ordered.end(), CompareIValue);
  return ordered;
}

// Not modified. Copied since the original is static.
// XXX: this function mutates input
static torch::jit::IValue addInput(
    const std::shared_ptr<torch::jit::tracer::TracingState>& state,
    const torch::jit::IValue& input, const torch::jit::TypePtr& type,
    torch::jit::Value* value) {
  value->setType(type);
  if (type->isSubtypeOf(torch::jit::TensorType::get())) {
    auto input_tensor = input.toTensor();
    auto name = Variable(input_tensor).name();
    if (state->hasValue(input)) {
      input_tensor = input_tensor.view(input_tensor.sizes());
    }
    value->setDebugName(name);
    state->setValue(input_tensor, value);
    return input_tensor;
  } else if (auto tuple_type = type->cast<torch::jit::TupleType>()) {
    auto unpack_node =
        state->graph->insertNode(state->graph->createTupleUnpack(value));
    auto elem_values = unpack_node->outputs();
    auto elem_types = tuple_type->elements();
    auto tuple = input.toTuple();
    auto elems = tuple->elements();
    size_t num_elems = elems.size();
    AT_ASSERT(
        elem_values.size() == num_elems && elem_types.size() == num_elems);
    for (size_t i = 0; i < num_elems; ++i) {
      elems[i] = addInput(state, elems.at(i), elem_types[i], elem_values[i]);
    }
    return std::move(tuple);
  } else if (auto dict_type = type->cast<torch::jit::DictType>()) {
    auto dict = input.toGenericDict();

    auto dict_size = dict.size();
    auto unpack_to_list = state->graph->insert(c10::aten::values, {value});
    auto list_unpack =
        state->graph->createListUnpack(unpack_to_list, dict_size);
    auto unpack_node = state->graph->insertNode(list_unpack);
    auto elem_values = unpack_node->outputs();

    const auto order = rannc::iterationOrder(dict);
    AT_ASSERT(order.size() == elem_values.size());

    size_t i = 0;
    for (const auto& pair : order) {
      dict.insert_or_assign(
          pair.first,
          addInput(
              state, pair.second, dict_type->getValueType(), elem_values[i++]));
    }

    return std::move(dict);
  } else if (auto list_type = type->cast<torch::jit::ListType>()) {
    size_t num_elems = input.isList() ? input.toListRef().size()
                                      : input.toTensorVector().size();
    auto list_unpack = state->graph->insertNode(
        state->graph->createListUnpack(value, num_elems));
    auto unpack_outputs = list_unpack->outputs();

    if (input.isTensorList()) {
      auto elems = input.toTensorList();
      for (size_t i = 0; i < num_elems; i++) {
        elems[i] = addInput(
                       state, elems.get(i), list_type->getElementType(),
                       unpack_outputs[i])
                       .toTensor();
      }
      return elems;
    } else {
      auto elems = input.toList();
      for (size_t i = 0; i < num_elems; i++) {
        elems[i] = addInput(
            state, elems.get(i), list_type->getElementType(),
            unpack_outputs[i]);
      }
      return elems;
    }
  } else {
    AT_ERROR(
        "Only tensors or (possibly nested) dict or tuples of tensors can be "
        "inputs to traced functions. Got ",
        type->repr_str());
  }
}

std::shared_ptr<torch::jit::tracer::TracingState> _trace(
    torch::jit::Stack inputs, c10::TupleTypePtr ttp,
    const std::function<torch::jit::Stack(torch::jit::Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool force_outplace, torch::jit::script::Module* self) {
  try {
    // Start tracing, treating 'inputs' as inputs to the trace, which can be
    // varied on subsequent invocations of the trace.  Any other variables
    // will be treated as constants.
    if (torch::jit::tracer::isTracing()) {
      AT_ERROR("Tracing can't be nested");
    }
    auto state = std::make_shared<torch::jit::tracer::TracingState>();
    torch::jit::tracer::setTracingState(state);

    // if we are a module, then make sure the modules parameters are in the map
    // and mapped to accesses to the self object
    //        if (self) {
    //          Value* self_value = state->graph->insertInput(0,
    //          "self")->setType(
    //              self->_ivalue()->type());
    //          gatherParametersAndBuffers(state, self_value, *self,
    //          {"__module"});
    //        }

    size_t i = 0;
    auto input_types = ttp->elements();
    for (torch::jit::IValue& input : inputs) {
      input =
          addInput(state, input, input_types[i++], state->graph->addInput());
    }

    for (const torch::jit::NameValue& s :
         self->named_attributes(/*recurse=*/false)) {
      auto param_in = state->graph->insertInput(i);
      const auto& param_iv = s.value;
      auto param_tensor = param_iv.toTensor();
      param_in->setType(param_iv.type());

      auto name = s.name;
      if (state->hasValue(param_tensor)) {
        param_tensor = param_tensor.view(param_tensor.sizes());
      }
      state->setValue(param_tensor, param_in);
      param_in->setDebugName(name);

      i++;
    }

    auto graph = state->graph;

    torch::jit::tracer::getTracingState()->lookup_var_name_fn =
        std::move(var_name_lookup_fn);
    torch::jit::tracer::getTracingState()->force_outplace = force_outplace;

    // Invoke the traced function
    auto out_stack = traced_fn(inputs);

    // Exit a trace, treating 'out_stack' as the outputs of the trace.  These
    // are the variables whose values will be computed upon subsequent
    // invocations of the trace.
    // size_t i = 0;
    i = 0;
    for (auto& output : out_stack) {
      // NB: The stack is in "reverse" order, so when we pass the diagnostic
      // number we need to flip it based on size.
      state->graph->registerOutput(
          state->getOutput(output, out_stack.size() - i));
      i++;
    }
    torch::jit::tracer::setTracingState(nullptr);

    //        if (script::getInlineEverythingMode()) {
    //          Inline(*graph);
    //        }
    FixupTraceScopeBlocks(graph, self);

    return state;
  } catch (...) {
    torch::jit::tracer::abandon();
    throw;
  }
}

/*
 * This is a copy of createGraphByTracing in python_tracer.cpp.
 * The caster for IValue was added.
 */
// std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
std::shared_ptr<torch::jit::Graph> _createGraphByTracing(
    const py::function& func, torch::jit::Stack trace_inputs,
    c10::TupleTypePtr ttp, const py::function& var_name_lookup_fn,
    bool force_outplace, torch::jit::script::Module* self) {
  C10_LOG_API_USAGE_ONCE("torch.tracer");

  auto lookup_fn_adapter =
      [var_name_lookup_fn](const Variable& var) -> std::string {
    AutoGIL ag;
    std::stringstream ss;
    int rank = getCurrentRank();
    if (rank == 0) {
      ss << py::cast<std::string>(var_name_lookup_fn(var));
    } else {
      ss << "r" << getCurrentRank() << "_"
         << py::cast<std::string>(var_name_lookup_fn(var));
    }
    return ss.str();
  };

  auto outs = _trace(
      std::move(trace_inputs), ttp,
      [&func](torch::jit::Stack inputs) -> torch::jit::Stack {
        size_t num_func_inputs = inputs.size();
        py::tuple py_inputs(num_func_inputs);

        for (size_t i = 0; i < num_func_inputs; ++i) {
          // py_inputs[i] = py::cast(inputs[i]);
          py_inputs[i] = torch::jit::toPyObject(std::move(inputs[i])).release();
        }
        auto out = func(*py_inputs);
        if (out.ptr() == Py_None) {
          AT_ERROR(
              "The traced function didn't return any values! Side-effects are not "
              "captured in traces, so it would be a no-op.");
        }
        return {torch::jit::_toTypeInferredIValue(out)};
      },
      lookup_fn_adapter, force_outplace, self);
  // return std::make_pair(std::get<0>(outs)->graph, std::get<1>(outs));
  return outs->graph;
}

void stackParams(
    std::vector<torch::jit::IValue>& param_ivalues,
    std::shared_ptr<torch::jit::script::Module>& module,
    const std::vector<py::tuple>& params, bool is_buffer) {
  for (const auto& param : toNamedTensors(params)) {
    const auto& p = param.second;
    module->register_parameter(param.first, p, is_buffer);
    const auto iv = c10::IValue(p).toIValue();
    param_ivalues.push_back(iv);
  }
}

std::shared_ptr<torch::jit::Graph> trace(
    const py::tuple& input_tuple, const py::function& fwdFunc,
    const std::vector<py::tuple>& params, const std::vector<py::tuple>& buffers,
    const py::function& var_lookup_fn, int64_t batch_size) {
  // Create dummy module for tracing
  std::shared_ptr<torch::jit::script::Module> module =
      std::make_shared<torch::jit::script::Module>("dummy");

  std::vector<torch::jit::IValue> param_ivalues;
  stackParams(param_ivalues, module, params, false);
  stackParams(param_ivalues, module, buffers, true);

  torch::jit::Stack typed_inputs;
  const auto iv = torch::jit::_toTypeInferredIValue(input_tuple);
  torch::jit::TypedIValue info =
      sliceBatchTensorsInTypedIValue({iv, iv.type()}, 0, batch_size);
  c10::TupleTypePtr ttp = info.type()->expect<torch::jit::TupleType>();

  if (batch_size > 0) {
    typed_inputs = info.ivalue().toTuple()->elements();
  } else {
    typed_inputs = torch::jit::_toTraceableStack(input_tuple);
  }

  torch::jit::Stack trace_inputs;
  for (const auto& iv : typed_inputs) {
    trace_inputs.push_back(toCUDAIfAvailable(iv, true));
  }

  std::shared_ptr<torch::jit::Graph> g;
  {
    torch::NoGradGuard no_grad;
    g = _createGraphByTracing(
        fwdFunc, trace_inputs, ttp, var_lookup_fn, false, module.get());
  }

  for (size_t i = 0; i < param_ivalues.size(); i++) {
    auto& val = g->inputs().at(i + input_tuple.size());
    val->inferTypeFrom(param_ivalues.at(i).toTensor());
  }

  for (auto i : g->inputs()) {
    i->setDebugName("_" + i->debugName());
  }
  for (auto n : g->nodes()) {
    for (auto o : n->outputs()) {
      o->setDebugName("_" + o->debugName());
    }
  }

  return g;
}
} // namespace rannc
