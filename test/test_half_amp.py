import copy
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from . import common
from apex import amp

import pyrannc
from pyrannc.amp import allreduce_grads, allreduce_grads_rannc

ASSERT_DECIMAL = 3
seed = 0
RELATIVE_TOLERANCE = 1.0e-1
ABSOLUTE_TOLERANCE = 1.0e-2
LOSS_SCALE = 1.0


if not torch.cuda.is_available():
    print("This test is valid only on a cuda environment.")
    sys.exit(0)


class Net(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 2, bias=False)
        w1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
        self.fc1.weight = torch.nn.Parameter(w1)
        self.fc2 = nn.Linear(2, 3, bias=False)
        w2 = torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], requires_grad=True)
        self.fc2.weight = torch.nn.Parameter(w2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x*2
        x = x*3
        return x


def do_run(model_base, batch_size_per_proc, input_dim, output_dim, num_iter,
           fp16, rtol, atol, get_dataset,
           **kwargs):

    device = torch.device("cuda")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    data_loader = common.get_loader(
        batch_size_per_proc, input_dim, output_dim, num_iter, get_dataset)

    lr = 0.01

    #model_base = Net().to(device)
    model_base = model_base.to(device)
    #model = copy.deepcopy(model_base)

    rmodel_base = copy.deepcopy(model_base)

    opt_base = optim.SGD(model_base.parameters(), lr=lr)
    model, opt = amp.initialize(model_base, opt_base, opt_level="O2",
                                loss_scale=LOSS_SCALE, master_weights=True)

    ropt_base = optim.SGD(rmodel_base.parameters(), lr=lr)
    rmodel_base, ropt = amp.initialize(rmodel_base, ropt_base, opt_level="O2",
                                       loss_scale=LOSS_SCALE, master_weights=True)
    rmodel = pyrannc.RaNNCModule(rmodel_base, ropt)

    pyrannc.delay_grad_allreduce(True)
    for x, tgt in data_loader:
        # Create test input
        x = x.to(device)

        p_out = model(x)
        r_out = rmodel(x)

        # Verify the equality of outputs
        np.testing.assert_equal(p_out.size(), r_out.size())
        np.testing.assert_allclose(p_out.tolist(), r_out.tolist(), rtol=rtol, atol=atol)

        tgt = tgt.to(device)
        criterion = nn.MSELoss()
        loss = criterion(tgt, p_out)
        rloss = criterion(tgt, r_out)

        with amp.scale_loss(loss, opt, delay_overflow_check=False, delay_unscale=False) as scaled_loss:
            scaled_loss.backward()
        allreduce_grads(opt, prescale=pyrannc.get_world_size())

        for p in amp.master_params(opt):
            print("AFTER AR: original grad={}".format(torch.flatten(p.grad)[0:3]))

        with amp.scale_loss(rloss, ropt, delay_overflow_check=False, delay_unscale=False) as scaled_loss:
            scaled_loss.backward()
        allreduce_grads_rannc(rmodel, ropt)

        for p in amp.master_params(ropt):
            print("AFTER AR: rannc grad={}".format(torch.flatten(p.grad)[0:3]))

        opt.step()
        ropt.step()

        expected_params = {n: p for n, p in model.named_parameters()}
        for n, p in rmodel.named_parameters():
            np.testing.assert_equal(p.size(), expected_params[n].size())
            np.testing.assert_allclose(p.tolist(), expected_params[n].tolist(), rtol=rtol, atol=atol)

        opt.zero_grad()
        rmodel.zero_grad()

    pyrannc.clear()


def run(model_base, batch_size_per_proc, num_iter,
        fp16=False,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        get_dataset=None,
        **kwargs):
    do_run(model_base, batch_size_per_proc,
           model_base.INPUT_DIM, model_base.OUTPUT_DIM, num_iter,
           fp16, rtol, atol, get_dataset,
           **kwargs)

def test_half_amp(init_dist, batch_size, iteration):
    print("test_half_amp")
    run(Net(), batch_size, iteration)
