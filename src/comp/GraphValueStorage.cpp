//
// Created by Masahiro Tanaka on 2019-06-14.
//

#include <comm/ObjectComm.h>
#include <comm/SComm.h>

#include "GraphValueStorage.h"

namespace rannc {

const torch::jit::IValue& GraphValueStorage::getValue(
    const IValueLocation& loc) const {
  if (!contains(values_, loc)) {
    std::stringstream ss;
    ss << "No value found: " << toString(loc);
    throw std::runtime_error(ss.str());
  }

  return values_.at(loc);
}

void GraphValueStorage::deploy(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  for (const auto& it : getGraphConstantValues(graph)) {
    if (auto iv = toIValue(it.second)) {
      if (iv->isDevice()) {
        const auto& dev = iv->toDevice();
        if (dev.is_cpu()) {
          values_[it.first] = c10::Device(c10::DeviceType::CUDA);
        } else {
          values_[it.first] = *iv;
        }
      } else {
        values_[it.first] = *iv;
      }
    }
  }
}

void GraphValueStorage::add(
    const IValueLocation& loc, const torch::jit::IValue& val) {
  values_[loc] = val;
}

} // namespace rannc
