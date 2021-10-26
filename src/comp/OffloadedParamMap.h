//
// Created by Masahiro Tanaka on 2021/10/26.
//

#ifndef PYRANNC_OFFLOADEDPARAMMAP_H
#define PYRANNC_OFFLOADEDPARAMMAP_H

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

 private:
  OffloadedParamMap();
  ~OffloadedParamMap() = default;
};
} // namespace rannc

#endif // PYRANNC_OFFLOADEDPARAMMAP_H
