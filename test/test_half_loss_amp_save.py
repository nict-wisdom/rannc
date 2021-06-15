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
LOSS_SCALE = 16.0


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
        self.criterion = nn.MSELoss()

    def forward(self, x, tgt):
        x = self.fc1(x)
        x = self.fc2(x)
        loss = self.criterion(x, tgt)
        return loss


def do_run(model_base, batch_size_per_proc, input_dim, output_dim, num_iter,
           rtol, atol, get_dataset, **kwargs):

    device = torch.device("cuda")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    data_loader = common.get_loader(
        batch_size_per_proc, input_dim, output_dim, num_iter, get_dataset, True)

    lr = 0.01

    model_base = model_base.to(device)
    rmodel_base = copy.deepcopy(model_base)

    opt_base = optim.Adam(model_base.parameters(), lr=lr)
    model, opt = amp.initialize(model_base, opt_base, opt_level="O2",
                                loss_scale=LOSS_SCALE, master_weights=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[pyrannc.get_rank()],
                                                      output_device=pyrannc.get_rank())

    ropt_base = optim.Adam(rmodel_base.parameters(), lr=lr)
    rmodel_base, ropt = amp.initialize(rmodel_base, ropt_base, opt_level="O2",
                                       loss_scale=LOSS_SCALE, master_weights=True)
    rmodel = pyrannc.RaNNCModule(rmodel_base, ropt, use_amp_master_params=True)

    # we manually run allreduce
    pyrannc.delay_grad_allreduce(True)

    for x, tgt in data_loader:
        # Create test input
        x = x.to(device)
        tgt = tgt.to(device)

        p_loss = model(x, tgt)
        tmp_loss = p_loss.clone()
        torch.distributed.all_reduce(tmp_loss)
        tmp_loss /= pyrannc.get_world_size()

        r_loss = rmodel(x, tgt)

        # Verify the equality of outputs
        np.testing.assert_equal(tmp_loss.size(), r_loss.size())
        np.testing.assert_allclose(tmp_loss.tolist(), r_loss.tolist(), rtol=rtol, atol=atol)

        with amp.scale_loss(p_loss, opt, delay_overflow_check=False, delay_unscale=False) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(opt), 1.0)

        with amp.scale_loss(r_loss, ropt, delay_overflow_check=False, delay_unscale=False) as scaled_loss:
            scaled_loss.backward()
        allreduce_grads_rannc(rmodel, ropt)
        rmodel.clip_grad_norm(1.0)

        common.compare_grads(model, rmodel, rtol, atol, fp16=True, zero=False, opt_exp=opt, opt_act=ropt)

        opt.step()
        ropt.step()

        common.compare_params(model, rmodel, rtol, atol, fp16=True, zero=False, opt_exp=opt, opt_act=ropt)

        opt.zero_grad()
        ropt.zero_grad()

    # Save model & opt
    # state_dict should run on all ranks
    model_state_dict = rmodel.state_dict()
    global_opt_state_dict = ropt.state_dict(from_global=True)

    if pyrannc.get_rank() == 0:
        torch.save(model_state_dict, 'model.pt')
        torch.save(global_opt_state_dict, 'opt_state.pt')
        rmodel.save_deployment(str("rannc_deployment.bin"))

    pyrannc.barrier()

    ld_model = Net().to(device)

    loaded_state_dict = torch.load('model.pt')
    ld_model.load_state_dict(loaded_state_dict)
    ld_opt = optim.Adam(ld_model.parameters(), lr=lr)
    ld_model, ld_opt = amp.initialize(ld_model, ld_opt, opt_level="O2",
                                      loss_scale="dynamic", master_weights=True)
    ld_model = pyrannc.RaNNCModule(ld_model, ld_opt, use_amp_master_params=True)

    # Verify parameters
    r_params = {n: p for n, p in rmodel.named_parameters()}
    ld_params = {n: p for n, p in ld_model.named_parameters()}

    for n, rp in r_params.items():
        ld_p = ld_params[n]
        np.testing.assert_equal(ld_p.size(), rp.size())
        np.testing.assert_almost_equal(ld_p.tolist(), rp.tolist(), decimal=ASSERT_DECIMAL)

    global_opt_state_dict = torch.load('opt_state.pt')
    opt_state_dict = opt.state_dict()

    for ld_grp, pt_grp in zip(global_opt_state_dict['param_groups'], opt_state_dict['param_groups']):
        np.testing.assert_(ld_grp.keys(), pt_grp.keys())
        for k in pt_grp.keys():
            if k == 'params':
                np.testing.assert_equal(len(ld_grp['params']), len(pt_grp['params']))
            else:
                np.testing.assert_(ld_grp[k] == pt_grp[k])

        for ld_pid, pt_pid in zip(ld_grp['params'], pt_grp['params']):
            ld_param_state = global_opt_state_dict['state'][ld_pid]
            pt_param_state = opt_state_dict['state'][pt_pid]
            np.testing.assert_(ld_param_state.keys(), pt_param_state.keys())
            for k in pt_param_state.keys():
                ldv = ld_param_state[k]
                pv = pt_param_state[k]
                if isinstance(ldv, torch.Tensor):
                    np.testing.assert_equal(ldv.size(), pv.size())
                    np.testing.assert_almost_equal(ldv.tolist(), pv.tolist(), decimal=ASSERT_DECIMAL)
                else:
                    np.testing.assert_(ldv == pv)

    print("Done")


def run(model_base, batch_size_per_proc, num_iter,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        get_dataset=None,
        **kwargs):
    do_run(model_base, batch_size_per_proc,
           model_base.INPUT_DIM, model_base.OUTPUT_DIM, num_iter,
           rtol, atol, get_dataset,
           **kwargs)


def test_half_loss_amp(init_dist, batch_size, iteration):
    print("test_half_loss_amp_save")
    run(Net(), batch_size, iteration)
