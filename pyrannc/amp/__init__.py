import amp_C
import apex_C
import torch
import torch.distributed as dist
from apex import amp
from apex.amp import _amp_state

from .. import _pyrannc
from ..tensor_coll import _allreduce_sum, _allreduce_min


def allreduce_grads(optimizer, prescale=1.0):
    overflow_buf = torch.cuda.IntTensor([0])

    # 1. allocate an uninitialized buffer for flattened gradient
    scaler = _amp_state.loss_scalers[0]
    master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
    flat_grad_size = sum(p.numel() for p in master_grads)
    allreduce_dtype = torch.float16
    flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
    # 2. combine unflattening and predivision of unscaled 'raw' gradient
    allreduced_views = apex_C.unflatten(flat_raw, master_grads)
    overflow_buf.zero_()
    amp_C.multi_tensor_scale(65536,
                             overflow_buf,
                             [master_grads, allreduced_views],
                             scaler.loss_scale() / prescale)
    # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
    torch.distributed.all_reduce(flat_raw)
    # 4. combine unscaling and unflattening of allreduced gradient
    overflow_buf.zero_()
    amp_C.multi_tensor_scale(65536,
                             overflow_buf,
                             [allreduced_views, master_grads],
                             1./scaler.loss_scale())
    # 5. update loss scale
    scaler = _amp_state.loss_scalers[0]
    old_overflow_buf = scaler._overflow_buf
    scaler._overflow_buf = overflow_buf
    had_overflow = scaler.update_scale()
    scaler._overfloat_buf = old_overflow_buf
    return had_overflow


def zip_params(optimizer):
    stash = optimizer._amp_stash
    return zip(stash.all_fp32_from_fp16_params, stash.all_fp16_params)


def zip_grads(optimizer):
    return [(master_param.grad, model_param.grad) for master_param, model_param in zip_params(optimizer)]


def register_amp_params(optimizer):
    for master_p, model_p in zip_params(optimizer):
        if model_p in optimizer.param_zero_segment_to_id:
            _pyrannc.register_amp_master_param(optimizer.param_zero_segment_to_id[model_p], master_p)
        else:
            _pyrannc.register_amp_master_param(id(model_p), master_p)


def convert_and_scale_params(source_grads, dest_grads, scale):
    overflow_buf = torch.cuda.IntTensor([0])
    if len(source_grads) == 0:
        return overflow_buf

    amp_C.multi_tensor_scale(65536,
                             overflow_buf,
                             [source_grads, dest_grads],
                             scale)

    return overflow_buf


def master_grads_to_model_grads(optimizer, scale):
    zipped_grads = zip_grads(optimizer)
    return convert_and_scale_params(list(map(lambda t: t[0], zipped_grads)), list(map(lambda t: t[1], zipped_grads)), scale)


def model_grads_to_master_grads(optimizer, scale):
    zipped_grads = zip_grads(optimizer)
    return convert_and_scale_params(list(map(lambda t: t[1], zipped_grads)), list(map(lambda t: t[0], zipped_grads)), scale)


def named_master_params(model, optimizer, zero=False):
    amp_param_map = {model_p: master_p for master_p, model_p in zip_params(optimizer)}

    if zero:
        pid_to_name = {pid: n for n, pid in model.name_to_pid.items()}
        return {pid_to_name[optimizer.param_zero_segment_to_id[model_p]]: master_p for model_p, master_p in amp_param_map.items()}

    return {n: amp_param_map[p] for n, p in model.named_parameters()}


def allreduce_grads_rannc(rmodel, optimizer, prescale=1.0, use_amp_master_param=True):

    if use_amp_master_param:
        scaler = _amp_state.loss_scalers[0]
        if rmodel.allreduce_amp_master_param:
            overflow_buf = torch.cuda.IntTensor([0])
            master_grads = [param.grad for param in amp.master_params(optimizer) if param.grad is not None]
            if len(master_grads) > 0:
                amp_C.multi_tensor_scale(65536,
                                         overflow_buf,
                                         [master_grads, master_grads],
                                         prescale)
            torch.cuda.synchronize()
        else:
            master_grads_to_model_grads(optimizer, scaler.loss_scale()*prescale)

    # rannc's allreduce
    rmodel.allreduce_grads()

    if use_amp_master_param:
        if rmodel.allreduce_amp_master_param:
            had_overflow = scaler.update_scale()
        else:
            overflow_buf = model_grads_to_master_grads(optimizer, 1./scaler.loss_scale())
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overflow_buf = old_overflow_buf

        return had_overflow

    return False


def patch_amp_scaler():
    scaler = _amp_state.loss_scalers[0]
    def decorate(func):
        def wrapper():
            had_overflow = func()

            # Share if overflow happens on any of ranks
            flag_overflow = torch.IntTensor([0]).cuda()
            if had_overflow:
                flag_overflow = torch.IntTensor([1]).cuda()
            _allreduce_sum(flag_overflow)

            if flag_overflow.item() > 0:
                scale_buf = torch.cuda.IntTensor([scaler._loss_scale])
                _allreduce_min(scale_buf)
                scaler._loss_scale = scale_buf.item()
                scaler._unskipped = 0
                return True
            return False
        return wrapper

    scaler.update_scale = decorate(scaler.update_scale)
