import torch
import torch.utils.cpp_extension

op_source = r"""
#include <torch/script.h>

torch::Tensor d_sigmoid(torch::Tensor  z)
{
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

TORCH_LIBRARY(test_ops, m) {
    m.def("d_sigmoid",   d_sigmoid);
}
"""

torch.utils.cpp_extension.load_inline(
    name="testop",
    cpp_sources=op_source,
    extra_ldflags=[],
    is_python_module=False,
    verbose=True
)


def native_compute_trace(z):
    return torch.ops.test_ops.d_sigmoid(z)


@torch.jit.script
def native_compute_script(z):
    return torch.ops.test_ops.d_sigmoid(z)


class NativeCallModel01(torch.nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()

    def forward(self, x):
        nx1 = native_compute_trace(x)
        nx2 = native_compute_script(x)
        return  nx1 * nx2
