import pytest

from . import common, models

default_vals = {
    "rtol": 1e-2,
    "atol": 0,
    "get_dataset": None,
    "loss_out": False
}

test_models = [
    {"model": models.BasicModel(), "rtol": 5e-2},
    {"model": models.SmallParamModel()},
    {"model": models.ForkJoinModel()},
    {"model": models.SharedParamModel()},
    {"model": models.OneOpModel()},
    {"model": models.TensorMulModel()},
    {"model": models.EmbeddingModel(), "get_dataset": models.EmbeddingModel.get_dataset},
    {"model": models.FunctionModel(), "get_dataset": models.FunctionModel.get_dataset},
    {"model": models.LossOutModel(), "loss_out": True}
]


@pytest.mark.parametrize("test_model", test_models)
@pytest.mark.parametrize("use_amp", [False, True])
def test_match(init_dist, batch_size, iteration, test_model, use_amp):

    for k, v in default_vals.items():
        if k not in test_model:
            test_model[k] = v

    common.run(test_model["model"], batch_size, iteration,
               loss_out=test_model["loss_out"],
               use_amp=use_amp,
               rtol=test_model["rtol"],
               atol=test_model["atol"],
               get_dataset=test_model["get_dataset"]
               )

