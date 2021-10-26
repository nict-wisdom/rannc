//
// Created by Masahiro Tanaka on 2021/10/26.
//

#ifndef PYRANNC_OFFLOADEDPARAMMAP_H
#define PYRANNC_OFFLOADEDPARAMMAP_H

#include <torch/torch.h>

namespace rannc {

class OffloadedParamMap {
 public:
  OffloadedParamMap(const OffloadedParamMap&) = delete;
  OffloadedParamMap& operator=(const OffloadedParamMap&) = delete;
  OffloadedParamMap(OffloadedParamMap&&) = delete;
  OffloadedParamMap& operator=(OffloadedParamMap&&) = delete;

  static OffloadedParamMap& get() {
    static OffloadedParamMap instance;
    return instance;
  }

  void registerParam(const std::string& name, const at::Tensor& param);
  at::Tensor getParam(const std::string& name);

 private:
  OffloadedParamMap(){};
  ~OffloadedParamMap() = default;

  std::unordered_map<std::string, at::Tensor> param_map_;
};
} // namespace rannc

#endif // PYRANNC_OFFLOADEDPARAMMAP_H
