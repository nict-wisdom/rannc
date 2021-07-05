//
// Created by Masahiro Tanaka on 2019-01-08.
//

#ifndef PYRANNC_BACKWARD_H
#define PYRANNC_BACKWARD_H

#include <torch/csrc/autograd/function.h>

#include <Common.h>
#include "GraphLauncher.h"

using namespace torch::autograd;


namespace rannc {
    class RaNNCProcess;

struct RaNNCTensorBackward : public torch::autograd::TraceableFunction {

    public:
        // output is an tensor
        RaNNCTensorBackward(std::shared_ptr<GraphLauncher> driver,
                            std::shared_ptr<ParamStorage> param_storage,
                            std::string graph_id, std::string name,
                            torch::jit::IValue output, PathInIValue path,
                            std::vector<IValueLocation> ordered_inputs,
                            std::vector<long> param_ids_on_rank,
                            bool enable_zero);

        variable_list apply(variable_list &&grads) override;

        std::string name() const override { return "RaNNCTensorBackward"; }

        void release_variables() override {
        }

        static void setDelayGradAllreduce(bool delayAllreduce) {
            delay_grad_allreduce_ = delayAllreduce;
        }

    private:
        torch::jit::IValue output_;
        IValueMap inputs_;
        std::vector<IValueLocation> ordered_inputs_;
        std::shared_ptr<GraphLauncher> driver_;
        std::shared_ptr<ParamStorage> param_storage_;
        const std::string graph_id_;
        const std::string value_name_;
        PathInIValue path_;
        std::vector<long> param_ids_on_rank_;
        bool skip_grad_scaling_;
        bool enable_zero_;

        static bool delay_grad_allreduce_;
    };

}

#endif //PYRANNC_BACKWARD_H
