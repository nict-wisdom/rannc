//
// Created by Masahiro Tanaka on 2018/11/30.
//

#ifndef PT_RANNC_RANNC_COMMON_H
#define PT_RANNC_RANNC_COMMON_H

#include <torch/csrc/jit/ir/ir.h>
#include <torch/TorchUtil.h>

#include "graph/ir.h"

namespace rannc {

    //
    //    Forward declarations
    //
    class FunctionStorage;


    class ConvertGraph {
    public:
        ConvertGraph() {}

        std::shared_ptr<torch::jit::Graph> toTorch(
                const std::shared_ptr<IRGraph>& irGraph,
                const IValueMap& constants,
                const FunctionStorage & functions);

        std::shared_ptr<torch::jit::Graph> toTorchNoMerge(
                const std::shared_ptr<IRGraph>& irGraph,
                const IValueMap& constants,
                const FunctionStorage & functions);

    private:
        void doToTorch(
                std::shared_ptr<torch::jit::Graph>& graph,
                std::unordered_map<std::string, torch::jit::Value*>& regValues,
                const std::shared_ptr<IRGraph>& irGraph,
                const IValueMap& constants,
                const FunctionStorage & functions,
                const std::vector<IRValue>& no_param_inputs);
    };

    IRValue toIRValue(torch::jit::Value* value);
    std::shared_ptr<IRGraph> fromTorch(const std::string &name, const std::shared_ptr<torch::jit::Graph> &graph,
            size_t real_input_num);
    torch::jit::TypePtr fromIRType(const IRType& ir_type);

    std::shared_ptr<IRGraph> guessBatchValuesByReachability(const std::shared_ptr<IRGraph>& g);
    std::shared_ptr<IRGraph> setValueTypes(const std::shared_ptr<IRGraph>& g,
                                           const std::unordered_map<std::string, IRType>& value_types);

    std::shared_ptr<torch::jit::Graph> disableDropout(const std::shared_ptr<torch::jit::Graph>& g);
    IValueMap createInputMap(const std::vector<torch::jit::IValue>& input_ivals,
                             const std::shared_ptr<IRGraph>& graph);

}
#endif //PT_RANNC_RANNC_COMMON_H
