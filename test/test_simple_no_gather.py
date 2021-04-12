from . import common, models


def test_basic(init_dist, batch_size, iteration):
    print("test_basic")
    common.run(models.BasicModel(), batch_size, iteration, gather_inputs=False)


def test_small_param(init_dist, batch_size, iteration):
    print("test_small_param")
    common.run(models.SmallParamModel(), batch_size, iteration, gather_inputs=False)


def test_fork_join(init_dist, batch_size, iteration):
    print("test_fork_join")
    common.run(models.ForkJoinModel(), batch_size, iteration, gather_inputs=False)


def test_shared_param(init_dist, batch_size, iteration):
    print("test_shared_param")
    common.run(models.SharedParamModel(), batch_size, iteration, gather_inputs=False)


def test_one_op(init_dist, batch_size, iteration):
    print("test_one_op")
    common.run(models.OneOpModel(), batch_size, iteration, gather_inputs=False)


def test_tensor_mul(init_dist, batch_size, iteration):
    print("test_tensor_mul")
    common.run(models.TensorMulModel(), batch_size, iteration, gather_inputs=False)


def test_emb(init_dist, batch_size, iteration):
    print("test_emb")
    common.run(models.EmbeddingModel(), batch_size, iteration, get_dataset=models.EmbeddingModel.get_dataset, gather_inputs=False)


def test_function(init_dist, batch_size, iteration):
    print("test_function")
    common.run(models.FunctionModel(), batch_size, iteration, get_dataset=models.FunctionModel.get_dataset, gather_inputs=False)


# def test_identity(init_dist, batch_size, iteration):
#     print("test_identity")
#     common.run(models.IdentityModel(), batch_size, iteration)


def test_loss_out(init_dist, batch_size, iteration):
    print("test_loss_out")
    common.run_loss(models.LossOutModel(), batch_size, iteration, gather_inputs=False)



