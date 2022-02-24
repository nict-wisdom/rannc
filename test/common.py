import copy

import numpy as np
import torch
import torch.cuda.random
import torch.distributed as dist
import torch.optim as optim

try:
    from apex import amp
    import pyrannc.amp
except ImportError:
    print("Failed to import apex. Tests with FP16 will fail.")
import pyrannc

RELATIVE_TOLERANCE = 1e-2
ABSOLUTE_TOLERANCE = 0
LOSS_SCALE = 2 ** 10
MAX_NORM = 1.0

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
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=pyrannc.get_world_size(),
                                                              rank=pyrannc.get_rank(),
                                                              shuffle=False) if gather_inputs else torch.utils.data.SequentialSampler(
        ds)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size_per_proc, sampler=sampler)
    return data_loader


def compare_tensors(v1, v2, rtol, atol):
    np.testing.assert_equal(v1.size(), v2.size())
    np.testing.assert_allclose(v1.tolist(), v2.tolist(), rtol=rtol, atol=atol)


def compare_dist_params(model_exp, model_act, rtol, atol):
    if hasattr(model_exp, "module"):  # ddp model
        model_exp = model_exp.module

    expected_params = {n: p for n, p in model_exp.named_parameters()}
    actual_params = {n: p for n, p in model_act.named_parameters()}

    dist_param_ranges = {n: pyrannc.get_dist_param_range(id(p)) for n, p in model_act.named_parameters()}

    for n, rp in actual_params.items():
        p = expected_params[n]
        v_a = rp.flatten()
        v_e = p.flatten()[dist_param_ranges[n]]
        compare_tensors(v_a, v_e, rtol, atol)


def do_compare_params(expected_params, actual_params, rtol, atol):
    for n, rp in actual_params.items():
        p = expected_params[n]
        compare_tensors(rp, p, rtol, atol)


def compare_params(model_exp, model_act, rtol, atol, fp16, zero=False, opt_exp=None, opt_act=None):
    if hasattr(model_exp, "module"):  # ddp model
        model_exp = model_exp.module

    expected_params = {n: p for n, p in pyrannc.amp.named_master_params(model_exp, opt_exp)} if fp16 \
        else {n: p for n, p in model_exp.named_parameters()}
    actual_params = {n: model_act.get_param(n, fp16) for n in sorted(model_act.name_to_param.keys())}
    do_compare_params(expected_params, actual_params, rtol, atol)


def compare_grads(model_exp, model_act, rtol, atol, fp16, zero=False, opt_exp=None, opt_act=None):
    if hasattr(model_exp, "module"):  # ddp model
        model_exp = model_exp.module

    expected_grads = {n: p.grad for n, p in pyrannc.amp.named_master_params(model_exp, opt_exp)} if fp16 \
        else {n: p.grad for n, p in model_exp.named_parameters()}
    actual_grads = {n: model_act.get_param_grad(n, fp16) for n in sorted(model_act.name_to_param.keys())}
    do_compare_params(expected_grads, actual_grads, rtol, atol)


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


def do_run(model_cls, batch_size_per_proc, num_iter,
           trace, fwd, aggregate, bwd, dtype, preprocess, gradient_accumulation_steps,
           use_amp, allreduce_amp_master_params,
           enable_zero, dist_params, offload_params, rtol, atol, get_dataset,
           **kwargs):
    print("Starting test using {}".format(model_cls.__name__))

    model_base = model_cls()
    input_dim = model_base.INPUT_DIM
    output_dim = model_base.OUTPUT_DIM

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
                                        max_loss_scale=LOSS_SCALE,
                                        min_loss_scale=1)
        if preprocess:
            preprocess(model)

    if dist_params:
        with pyrannc.DistributeModelParams(model_cls):
            rmodel = model_cls()

        base_named_params = {n: p for n, p in model.named_parameters()}
        for n, p in rmodel.named_parameters():
            pyrannc.set_dist_param(id(p), base_named_params[n])
    else:
        # We copy the model for verification of parameter update.
        rmodel = copy.deepcopy(model_base)

    rmodel.to(dtype)

    r_opt = None
    if has_param:
        r_opt = optim.Adam(rmodel.parameters(), lr=lr)
        if use_amp:
            rmodel = rmodel.to(device)
            rmodel, r_opt = amp.initialize(rmodel, r_opt, opt_level="O2",
                                           max_loss_scale=LOSS_SCALE,
                                           min_loss_scale=1)
        if preprocess:
            preprocess(rmodel)

    module_args = {}
    if "check_unused_values" in kwargs:
        module_args["check_unused_values"] = kwargs["check_unused_values"]

    if "offload_params" in kwargs:
        module_args["offload_params"] = kwargs["offload_params"]

    gather_inputs = True
    if "gather_inputs" in kwargs:
        gather_inputs = module_args["gather_inputs"] = kwargs["gather_inputs"]

    rmodel = pyrannc.RaNNCModule(rmodel, r_opt, enable_apex_amp=use_amp, enable_zero=enable_zero,
                                 allreduce_amp_master_params=allreduce_amp_master_params,
                                 offload_params=offload_params, **module_args)

    if dist_params:
        compare_dist_params(model, rmodel, rtol, atol)
    else:
        compare_params(model, rmodel, rtol, atol, False)

    delay_grad_allreduce = allreduce_amp_master_params or gradient_accumulation_steps > 1
    pyrannc.delay_grad_allreduce(delay_grad_allreduce)

    data_loader = get_loader(batch_size_per_proc, input_dim, output_dim, num_iter, get_dataset, gather_inputs)
    for step, (x, tgt) in enumerate(data_loader):

        run_update = step % gradient_accumulation_steps == 0 if delay_grad_allreduce else True

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
            if run_update or (not has_param):
                p_out = fwd(ddp_model, x, tgt)
            else:  # delay allreduce by ddp
                with ddp_model.no_sync():
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

            if delay_grad_allreduce and run_update:
                pyrannc.allreduce_grads(rmodel, r_opt)

            if use_amp:
                torch.nn.utils.clip_grad_norm_(amp.master_params(opt), MAX_NORM)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

            rmodel.clip_grad_norm(MAX_NORM)

            if run_update:
                if enable_zero:
                    rmodel._sync_orig_params(sync_grad=True)
                compare_grads(model, rmodel, rtol, atol, use_amp, zero=enable_zero, opt_exp=opt, opt_act=r_opt)

                opt.step()
                r_opt.step()
                opt.zero_grad()
                r_opt.zero_grad()

        if enable_zero:
            rmodel._sync_orig_params()
        compare_params(model, rmodel, rtol, atol, has_param and use_amp, zero=enable_zero, opt_exp=opt,
                       opt_act=r_opt)

    if gather_inputs or pyrannc.get_rank() == 0:
        compare_params(model, rmodel, rtol, atol, has_param and use_amp, zero=enable_zero, opt_exp=opt, opt_act=r_opt)

    if not has_param:
        print("Done")
        return

    # Save model & opt
    # state_dict should run on all ranks
    model_state_dict = rmodel.state_dict()
    global_opt_state_dict = r_opt.state_dict(from_global=True)

    if pyrannc.get_rank() == 0:
        torch.save(model_state_dict, 'model.pt')
        torch.save(global_opt_state_dict, 'opt_state.pt')

    pyrannc.barrier()

    ld_model = copy.deepcopy(model_base)

    loaded_state_dict = torch.load('model.pt')
    ld_model.load_state_dict(loaded_state_dict)
    ld_opt = optim.Adam(ld_model.parameters(), lr=lr)

    if use_amp:
        ld_model = ld_model.to(device)
        ld_model, ld_opt = amp.initialize(ld_model, ld_opt, opt_level="O2",
                                          max_loss_scale=2. ** 4,
                                          min_loss_scale=1)

    ld_model = pyrannc.RaNNCModule(ld_model, ld_opt, enable_apex_amp=use_amp, **module_args)

    # Verify parameters
    r_params = {n: p for n, p in rmodel.named_parameters()}
    ld_params = {n: p for n, p in ld_model.named_parameters()}

    for n, rp in r_params.items():
        ld_p = ld_params[n]
        compare_tensors(rp, ld_p, rtol, atol)

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
            np.testing.assert_(ld_param_state.keys() == pt_param_state.keys())
            for k in pt_param_state.keys():
                ldv = ld_param_state[k]
                pv = pt_param_state[k]
                if isinstance(ldv, torch.Tensor):
                    compare_tensors(ldv, pv, rtol, atol)
                else:
                    np.testing.assert_(ldv == pv)

    r_opt.load_state_dict(global_opt_state_dict, from_global=True)

    pyrannc.clear()
    pyrannc.barrier()
    print("Done")


def run(model_base, batch_size_per_proc, num_iter,
        dtype=torch.float, loss_out=False, preprocess=False, gradient_accumulation_steps=1,
        use_amp=False, allreduce_amp_master_params=False, enable_zero=False,
        dist_params=False, offload_params=False, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
        get_dataset=None, **kwargs):
    if loss_out:
        def aggregate_out_loss(out):
            tmp_loss = out.clone()
            torch.distributed.all_reduce(tmp_loss)
            tmp_loss /= pyrannc.get_world_size()
            return tmp_loss

        do_run(model_base, batch_size_per_proc, num_iter,
               lambda model, x, tgt: torch.jit.trace(model, (x, tgt)),
               lambda model, x, tgt: model(x, tgt),
               aggregate_out_loss,
               bwd_loss_output,
               dtype, preprocess, gradient_accumulation_steps,
               use_amp, allreduce_amp_master_params, enable_zero, dist_params, offload_params,
               rtol, atol, get_dataset, **kwargs)

    else:
        do_run(model_base, batch_size_per_proc, num_iter,
               lambda model, x, tgt: torch.jit.trace(model, (x,)),
               lambda model, x, tgt: model(x),
               lambda out: out,
               bwd_with_criterion,
               dtype, preprocess, gradient_accumulation_steps,
               use_amp, allreduce_amp_master_params, enable_zero, dist_params, offload_params,
               rtol, atol, get_dataset, **kwargs)
