//
// Created by Masahiro Tanaka on 2021/09/03.
//

#ifndef PYRANNC_CUSTOMOPS_H
#define PYRANNC_CUSTOMOPS_H

#include <torch/torch.h>

namespace rannc {
at::Tensor displayValueHook(const at::Tensor& tensor, const std::string& name);
}

#endif // PYRANNC_CUSTOMOPS_H
