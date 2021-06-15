import copy

import numpy as np
import torch
import torch.cuda.random
import torch.distributed as dist
import torch.optim as optim
try:
    from apex import amp
except ImportError:
    print("Failed to import apex. Tests with FP16 will fail.")
import pyrannc
import pyrannc.amp


ASSERT_DECIMAL = 3
seed = 0
RELATIVE_TOLERANCE = 1e-2
ABSOLUTE_TOLERANCE = 1e-4

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_dataset_default(dataset_size, input_dim, output_dim):
    ds_x = torch.randn((dataset_size,) + input_dim)
    ds_tgt = torch.randn((dataset_size,) + output_dim)
    return ds_x, ds_tgt


def get_loader(batch_size_per_proc, input_dim, output_dim, num_iter, get_dataset, gather_inputs):
    DATASET_SIZE = pyrannc.get_world_size() * batch_size_per_proc * num_iter
    if get_dataset is None:
        get_dataset = get_dataset_default
    ds_x, ds_tgt = get_dataset(DATASET_SIZE, input_dim, output_dim)
    ds = torch.utils.data.TensorDataset(ds_x, ds_tgt)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=pyrannc.get_world_size(), rank=pyrannc.get_rank(), shuffle=False) if gather_inputs else torch.utils.data.SequentialSampler(ds)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size_per_proc, sampler=sampler)
    return data_loader


def compare_tensors(v1, v2, rtol, atol):
    np.testing.assert_equal(v1.size(), v2.size())
    np.testing.assert_allclose(v1.tolist(), v2.tolist(), rtol=rtol, atol=atol)


def do_compare_params(model_exp, model_act, f, rtol, atol, fp16, opt_exp, opt_act):
    if hasattr(model_exp, "module"): # ddp model
        model_exp = model_exp.module

    expected_params = {n: p for n, p in model_exp.named_parameters()}
    actual_params = {n: p for n, p in model_act.named_parameters()}

    for n, rp in actual_params.items():
        p = expected_params[n]
        compare_tensors(f(rp), f(p), rtol, atol)

    if fp16:
        expected_amp_param_map = {model_p: master_p for master_p, model_p in pyrannc.amp.zip_params(opt_exp)}
        expected_master_params = {n: expected_amp_param_map[p] for n, p in expected_params.items()}

        actual_amp_param_map = {model_p: master_p for master_p, model_p in pyrannc.amp.zip_params(opt_act)}
        actual_master_params = {n: actual_amp_param_map[p] for n, p in actual_params.items()}

        for n, rp in actual_master_params.items():
            p = expected_master_params[n]
            compare_tensors(f(rp), f(p), rtol, atol)


def compare_params(model_exp, model_act, rtol, atol, fp16, zero=False, opt_exp=None, opt_act=None):
    do_compare_params(model_exp, model_act, lambda p: p, rtol, atol, fp16, opt_exp, opt_act)


def compare_grads(model_exp, model_act, rtol, atol, fp16, zero=False, opt_exp=None, opt_act=None):
    do_compare_params(model_exp, model_act, lambda p: p.grad, rtol, atol, fp16, opt_exp, opt_act)


def reset_running_stats(model):
    for name, _module in model.named_modules():
        if hasattr(_module, "reset_running_stats"):
            _module.reset_running_stats()


def backward_loss(loss, optimizer, fp16):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


def bwd_with_criterion(out, tgt, optimizer, fp16):
    criterion = torch.nn.MSELoss()
    loss = criterion(tgt, out)
    backward_loss(loss, optimizer, fp16)


def bwd_loss_output(loss, tgt, optimizer, fp16):
    backward_loss(loss, optimizer, fp16)


def convert_dtype(t, dtype):
    if t.is_floating_point():
        return t.to(dtype)
    return t


def do_run(model_base, batch_size_per_proc, input_dim, output_dim, num_iter,
           trace, fwd, aggregate, bwd, dtype, use_amp, rtol, atol, get_dataset,
           **kwargs):
    device = torch.cuda.current_device()

    model = copy.deepcopy(model_base)
    model = model.to(device)
    ddp_model = None
    lr = 0.01
    opt = None
    has_param = len(list(model.parameters())) > 0
    if has_param:
        opt = optim.Adam(model.parameters(), lr=lr)
        if use_amp:
            model, opt = amp.initialize(model, opt, opt_level="O2",
                                              max_loss_scale=2.**4,
                                              min_loss_scale=1)

    # We copy the model for verification of parameter update.
    # You may not need to copy the model in your application.
    rmodel = copy.deepcopy(model_base)
    rmodel.to(dtype)

    r_opt = None
    if has_param:
        r_opt = optim.Adam(rmodel.parameters(), lr=lr)
        if use_amp:
            rmodel = rmodel.to(device)
            rmodel, r_opt = amp.initialize(rmodel, r_opt, opt_level="O2",
                                            max_loss_scale=2.**4,
                                            min_loss_scale=1)

    module_args = {}
    if "check_unused_values" in kwargs:
        module_args["check_unused_values"] = kwargs["check_unused_values"]

    gather_inputs = True
    if "gather_inputs" in kwargs:
        gather_inputs = module_args["gather_inputs"] = kwargs["gather_inputs"]

    enable_zero = False
    if "enable_zero" in kwargs:
        enable_zero = module_args["enable_zero"] = kwargs["enable_zero"]

    rmodel = pyrannc.RaNNCModule(rmodel, r_opt, use_amp_master_params=use_amp, **module_args)

    compare_params(model, rmodel, rtol, atol, use_amp)

    data_loader = get_loader(batch_size_per_proc, input_dim, output_dim, num_iter, get_dataset, gather_inputs)
    for x, tgt in data_loader:
        # Create test input
        x = x.to(device)
        tgt = tgt.to(device)

        if not ddp_model:
            jit_model = trace(model, x, tgt)
            reset_running_stats(model)
            if has_param:
                if gather_inputs:
                    ddp_model = torch.nn.parallel.DistributedDataParallel(jit_model, device_ids=[device],
                                                                          output_device=device)
                else:
                    ddp_model = jit_model
            else:
                ddp_model = jit_model

        with torch.random.fork_rng(devices=[device]):
            p_out = fwd(ddp_model, x, tgt)
            agg_out = aggregate(p_out)

        r_out = fwd(rmodel, convert_dtype(x, dtype), convert_dtype(tgt, dtype))

        # Verify the equality of outputs
        if gather_inputs or pyrannc.get_rank() == 0:
            compare_tensors(r_out, agg_out, rtol, atol)

        # Create test target
        if has_param:
            bwd(p_out, tgt, opt, use_amp)
            bwd(r_out, convert_dtype(tgt, dtype), r_opt, use_amp)

            if gather_inputs or pyrannc.get_rank() == 0:
                if enable_zero:
                    rmodel._sync_orig_params(sync_grad=True)
                compare_grads(model, rmodel, rtol, atol, use_amp)

            opt.step()
            r_opt.step()
            opt.zero_grad()
            r_opt.zero_grad()

        if gather_inputs or pyrannc.get_rank() == 0:
            if enable_zero:
                rmodel._sync_orig_params()
            compare_params(model, rmodel, rtol, atol, use_amp)

    if gather_inputs or pyrannc.get_rank() == 0:
        compare_params(model, rmodel, rtol, atol, use_amp)

    ddp_model = None
    model.eval()
    rmodel.eval()

    for x, tgt in data_loader:
        x = x.to(device)
        tgt = tgt.to(device)

        if not ddp_model:
            jit_model = trace(model, x, tgt)
            if has_param:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    jit_model, device_ids=[device],
                    output_device=device)
            else:
                ddp_model = jit_model

        p_out = fwd(ddp_model, x, tgt)
        agg_out = aggregate(p_out)
        r_out = fwd(rmodel, convert_dtype(x, dtype), convert_dtype(tgt, dtype))

        if gather_inputs or pyrannc.get_rank() == 0:
            compare_tensors(r_out, agg_out, rtol, atol)

    pyrannc.clear()


def run(model_base, batch_size_per_proc, num_iter,
        dtype=torch.float, use_amp=False,
        rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
        get_dataset=None, **kwargs):
    do_run(model_base, batch_size_per_proc, model_base.INPUT_DIM, model_base.OUTPUT_DIM, num_iter,
           lambda model, x, tgt: torch.jit.trace(model, (x,)),
           lambda model, x, tgt: model(x),
           lambda out: out,
           bwd_with_criterion,
           dtype, use_amp, rtol, atol, get_dataset, **kwargs)


def run_loss(model_base, batch_size_per_proc, num_iter,
             dtype=torch.float, use_amp=False,
             rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
             get_dataset=None, **kwargs):
    def aggregate_out_loss(out):
        tmp_loss = out.clone()
        torch.distributed.all_reduce(tmp_loss)
        tmp_loss /= pyrannc.get_world_size()
        return tmp_loss
    do_run(model_base, batch_size_per_proc, model_base.INPUT_DIM, model_base.OUTPUT_DIM, num_iter,
           lambda model, x, tgt: torch.jit.trace(model, (x, tgt)),
           lambda model, x, tgt: model(x, tgt),
           aggregate_out_loss,
           bwd_loss_output,
           dtype, use_amp, rtol, atol, get_dataset, **kwargs)
