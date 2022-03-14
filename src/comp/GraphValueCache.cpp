//
// Created by Masahiro Tanaka on 2022/03/11.
//

#include "GraphValueCache.h"

namespace rannc {
size_t ParamCache::getValueSize(const at::Tensor& v) const {
  return v.numel() * elementSize(v.scalar_type());
}
} // namespace rannc