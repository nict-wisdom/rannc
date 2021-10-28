//
// Created by Masahiro Tanaka on 2019-07-07.
//

#include "PybindUtil.h"

namespace rannc {
py::object builtin_id = py::module::import("builtins").attr("id");

long getPythonObjId(py::object obj) {
  return builtin_id(obj).cast<long>();
}
} // namespace rannc

// The following part was copied from csrc/jit/python/pybind_utils.cpp
// because the symbols of the functions are invisible from an application.
namespace torch {
namespace jit {
IValue _toTypeInferredIValue(py::handle input) {
  auto match = tryToInferType(input);
  if (!match.success()) {
    AT_ERROR(
        "Tracer cannot infer type of ", py::str(input), "\n:", match.reason());
  }
  return toIValue(input, match.type());
}
} // namespace jit
} // namespace torch
