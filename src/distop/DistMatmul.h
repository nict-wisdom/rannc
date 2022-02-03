//
// Created by Masahiro Tanaka on 2021/05/11.
//

#ifndef TPTESTS_DISTMATMUL_H
#define TPTESTS_DISTMATMUL_H

#include <comp/TimeCounter.h>
#include <torch/torch.h>
#include <torch/TorchUtil.h>

namespace rannc {

class DistMatmul {
 public:
  at::Tensor run(const at::Tensor& x, const at::Tensor& y);

 private:
  at::Tensor out_buf_;
};
} // namespace rannc

#endif // TPTESTS_DISTMATMUL_H
