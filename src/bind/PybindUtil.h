//
// Created by Masahiro Tanaka on 2019-07-07.
//

#ifndef PYRANNC_PYBINDUTIL_H
#define PYRANNC_PYBINDUTIL_H

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#undef USE_DISTRIBUTED
#undef USE_RPC

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_ivalue.h>

namespace py = pybind11;

namespace rannc {
long getPythonObjId(py::object obj);
}

namespace torch {
namespace jit {
IValue _toTypeInferredIValue(py::handle input);
Stack _toTraceableStack(const py::tuple& inputs);
} // namespace jit
} // namespace torch
#endif // PYRANNC_PYBINDUTIL_H
