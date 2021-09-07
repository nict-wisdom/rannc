//
// Created by Masahiro Tanaka on 2019-02-25.
//

#ifndef PYRANNC_TRACER_H
#define PYRANNC_TRACER_H

#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace rannc {
void enterRank(int rank);
int exitRank();

std::shared_ptr<torch::jit::Graph> trace(
    const py::tuple& input, const py::function& fwdFunc,
    const std::vector<py::tuple>& params, const std::vector<py::tuple>& buffers,
    const py::function& var_lookup_fn, int64_t batch_size);

std::unordered_map<std::string, torch::jit::IValue> matchParams(
    const std::shared_ptr<torch::jit::Graph>& graph, const size_t real_input,
    const std::vector<py::tuple>& params,
    const std::vector<py::tuple>& buffers);
} // namespace rannc

#endif // PYRANNC_TRACER_H
