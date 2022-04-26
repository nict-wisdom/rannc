import torch
from apex import amp
from apex.amp import _amp_state

from .. import _pyrannc
from ..tensor_coll import _allreduce_sum, _allreduce_min


def zip_params(optimizer):
    stash = optimizer._amp_stash
    return zip(stash.all_fp32_from_fp16_params, stash.all_fp16_params)

def register_amp_params(optimizer):
    for master_p, model_p in zip_params(optimizer):
        if model_p in optimizer.param_zero_segment_to_id:
            _pyrannc.register_amp_master_param(optimizer.param_zero_segment_to_id[model_p], master_p)
        else:
            _pyrannc.register_amp_master_param(id(model_p), master_p)


def named_master_params(model, optimizer, zero=False):
    if hasattr(optimizer._amp_stash, "all_fp32_from_fp16_params"):
        amp_param_map = {model_p: master_p for master_p, model_p in zip_params(optimizer)}

        if zero:
            pid_to_name = {pid: n for n, pid in model.name_to_pid.items()}
            pid_to_segment = {pid: segment for segment, pid in optimizer.param_zero_segment_to_id.items()}
            for pid, segment in pid_to_segment.items():
                if segment.dtype == torch.float:
                    amp_param_map[segment] = segment

            for model_p, master_p in amp_param_map.items():
                yield pid_to_name[optimizer.param_zero_segment_to_id[model_p]], master_p

        else:
            for p in model.parameters():
                if p.dtype == torch.float:
                    amp_param_map[p] = p

            for n, p in model.named_parameters():
                yield n, amp_param_map[p]


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
                scale_buf = torch.cuda.IntTensor([int(scaler._loss_scale)])
                _allreduce_min(scale_buf)
                scaler._loss_scale = scale_buf.item()
                scaler._unskipped = 0
                return True
            return False
        return wrapper

    scaler.update_scale = decorate(scaler.update_scale)


def unset_master_grads(optimizer):
    for p in amp.master_params(optimizer):
        p.grad = None
