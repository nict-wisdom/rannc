
#ifndef PYRANNC_CONFIGURED_TORCH_H
#define PYRANNC_CONFIGURED_TORCH_H

#include <torch/torch.h>

namespace rannc {

inline at::Tensor& getMutableGradRef(at::Tensor &t)
{
    return t.mutable_grad();
}

inline at::Tensor& getMutableGradRef(at::Tensor &&t)
{
    return t.mutable_grad();
}

}   //  End of namespace rannc.

#endif
