from . import common, models


def test_buffer(init_dist, batch_size, iteration):
    common.run(models.BufferModel1(), batch_size, iteration, check_unused_values=False)


def test_buffer2(init_dist, batch_size, iteration):
    common.run(models.BufferModel2(), batch_size, iteration, check_unused_values=False)


def test_buffer3(init_dist, batch_size, iteration):
    common.run(models.BufferModel3(), batch_size, iteration, check_unused_values=False)

