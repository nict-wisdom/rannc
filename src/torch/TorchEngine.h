//
// Created by Masahiro Tanaka on 2019/09/19.
//

#ifndef PYRANNC_TORCHENGINE_H
#define PYRANNC_TORCHENGINE_H

#include <torch/csrc/autograd/engine.h>

namespace rannc {
class TorchEngine {
 public:
  static torch::autograd::Engine& get() {
    return torch::autograd::Engine::get_base_engine();
  }
};
} // namespace rannc

#endif // PYRANNC_TORCHENGINE_H
