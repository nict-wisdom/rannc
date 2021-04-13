import torch

from . import common, models


def test_small_param(init_dist, batch_size, iteration):
    print("test_small_param")
    common.run(models.SmallParamModel(), batch_size, iteration, dtype=torch.bfloat16, rtol=0.1)

