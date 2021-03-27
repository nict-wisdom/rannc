from . import common, models


def test_basic(init_dist, batch_size, iteration):
    print("test_basic")
    common.run(models.BasicModel(), batch_size, iteration)


# def test_small_param(init_dist, batch_size, iteration):
#     print("test_small_param")
#     common.run(models.SmallParamModel(), batch_size, iteration)
