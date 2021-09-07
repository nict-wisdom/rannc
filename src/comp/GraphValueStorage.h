//
// Created by Masahiro Tanaka on 2019-06-14.
//

#ifndef PYRANNC_GRAPHVALUESTORAGE_H
#define PYRANNC_GRAPHVALUESTORAGE_H

#include <graph/ir.h>
#include <Logging.h>
#include <torch/TorchUtil.h>

namespace rannc {

class GraphValueStorage {
 public:
  void deploy(const std::shared_ptr<torch::jit::Graph>& graph);
  const torch::jit::IValue& getValue(const IValueLocation& loc) const;

  const IValueMap& getValues() const {
    return values_;
  }

 private:
  IValueMap values_;
  const std::shared_ptr<spdlog::logger> logger = getLogger("GraphValueStorage");
};
} // namespace rannc

#endif // PYRANNC_GRAPHVALUESTORAGE_H
