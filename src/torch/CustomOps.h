//
// Created by Masahiro Tanaka on 2021/09/03.
//

#ifndef PYRANNC_CUSTOMOPS_H
#define PYRANNC_CUSTOMOPS_H

#include <comp/OffloadedParamMap.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include "TorchUtil.h"

namespace rannc {

class OffloadingHookFunction
    : public torch::autograd::Function<OffloadingHookFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor input,
      const std::string& param_name, bool to_cuda) {
    ctx->saved_data["param_name"] = param_name;
    ctx->saved_data["to_cuda"] = to_cuda;
    OffloadedParamMap& param_map = OffloadedParamMap::get();
    at::Tensor param = param_map.getParam(param_name);

    if (to_cuda) {
      toCUDAInPlace(param);
    } else {
      toCPUInPlace(param);
    }

    return input;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const torch::jit::IValue iv_param_name = ctx->saved_data["param_name"];
    assert(iv_param_name.isString());
    OffloadedParamMap& param_map = OffloadedParamMap::get();
    at::Tensor param = param_map.getParam(iv_param_name.toStringRef());

    const torch::jit::IValue iv_to_cuda = ctx->saved_data["to_cuda"];
    assert(iv_to_cuda.isBool());
    bool to_cuda = iv_to_cuda.toBool();

    if (to_cuda) {
      toCPUInPlace(param);
    } else {
      toCUDAInPlace(param);
    }

    grad_outputs.push_back(torch::autograd::Variable());
    grad_outputs.push_back(torch::autograd::Variable());
    return grad_outputs;
  }
};

class OffloadingPostHookFunction
    : public torch::autograd::Function<OffloadingPostHookFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx, torch::Tensor input,
      const std::string& param_name) {
    ctx->saved_data["param_name"] = param_name;
    OffloadedParamMap& param_map = OffloadedParamMap::get();
    at::Tensor param = param_map.getParam(param_name);

    toCPUInPlace(param);

    return input;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const torch::jit::IValue iv_param_name = ctx->saved_data["param_name"];
    assert(iv_param_name.isString());
    OffloadedParamMap& param_map = OffloadedParamMap::get();
    at::Tensor param = param_map.getParam(iv_param_name.toStringRef());

    toCUDAInPlace(param);

    grad_outputs.push_back(torch::autograd::Variable());
    return grad_outputs;
  }
};

at::Tensor offloadingPreHook(const at::Tensor& tensor, const std::string& name);
at::Tensor offloadingPostHook(
    const at::Tensor& tensor, const std::string& name);

} // namespace rannc

#endif // PYRANNC_CUSTOMOPS_H
