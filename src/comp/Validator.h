//
// Created by Masahiro Tanaka on 2021/02/16.
//

#ifndef PYRANNC_VALIDATOR_H
#define PYRANNC_VALIDATOR_H

#include "graph/Decomposition.h"

namespace rannc {

class Validator {
 public:
  Validator() {}

  bool validate(
      const std::shared_ptr<torch::jit::Graph>& graph,
      const std::vector<torch::jit::IValue>& input_ivals,
      const std::unordered_map<std::string, torch::jit::IValue>& param_inputs,
      const IValueMap& const_vals, const FunctionStorage& functions,
      const Deployment& deployment);

 private:
  const std::shared_ptr<spdlog::logger> logger = getLogger("Validator");
};
} // namespace rannc

#endif // PYRANNC_VALIDATOR_H
