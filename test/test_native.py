
from . import common, native_models


def run(m, batch_size, iteration, **kwargs):
    common.run(m, batch_size, m.INPUT_DIM, m.OUTPUT_DIM, iteration, **kwargs)


def run_loss(m, batch_size, iteration, **kwargs):
    common.run_loss(m, batch_size, m.INPUT_DIM, m.OUTPUT_DIM, iteration, **kwargs)


def  test_nativecall_01(init_dist, batch_size, iteration):
    run(native_models.NativeCallModel01(), batch_size, iteration)
