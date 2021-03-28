from . import common, models


def test_basic_fp16(init_dist, batch_size, iteration):
    common.run(models.BasicModel(), batch_size, iteration, fp16=True, rtol=1e-1, atol=1e-2)


def test_small_param_fp16(init_dist, batch_size, iteration):
    common.run(models.SmallParamModel(), batch_size, iteration, fp16=True, rtol=1e-1, atol=1e-2)

