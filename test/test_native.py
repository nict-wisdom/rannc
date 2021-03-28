from . import common, native_models

def test_native(init_dist, batch_size, iteration):
    common.run(native_models.NativeCallModel01(), batch_size, iteration)
