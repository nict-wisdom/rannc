import pickle
import types

import torch

import pyrannc
from . import _pyrannc, comm_utils, tensor_coll
from .amp import register_amp_params, unset_master_grads


def replace_ids_in_param_group(param_group, order_local_to_global):
    new_group = {}
    for k, v in param_group.items():
        if k == 'params':
            new_group['params'] = [order_local_to_global[pid] for pid in v]
        else:
            new_group[k] = v
    return new_group


def to_cpu_tensor(s):
    if isinstance(s, list):
        return [to_cpu_tensor(v) for v in s]
    if isinstance(s, dict):
        return {to_cpu_tensor(k): to_cpu_tensor(v) for k, v in s.items()}
    if isinstance(s, torch.Tensor):
        return s.cpu()
    return s


def to_cuda_tensor(s):
    if isinstance(s, list):
        return [to_cuda_tensor(v) for v in s]
    if isinstance(s, dict):
        return {to_cuda_tensor(k): to_cuda_tensor(v) for k, v in s.items()}
    if isinstance(s, torch.Tensor):
        return s.cuda()
    return s


def replace_param_ids(opt_state_dict, order_local_to_global):
    new_state = {}
    for k, v in opt_state_dict.items():
        if k == 'state':
            new_state['state'] = {order_local_to_global[pid]: sv for pid, sv in v.items()}
        elif k == 'param_groups':
            new_state['param_groups'] = [replace_ids_in_param_group(g, order_local_to_global) for g in opt_state_dict['param_groups']]
        else:
            new_state[k] = v
    return new_state


def merge_state_dict(s1, s2):
    ret = s1.copy()
    for k, v in s2.items():
        if k == 'state':
            for pid, sv in v.items():
                if pid not in ret['state'].keys():
                    ret['state'][pid] = sv
        elif k == 'param_groups':
            ret['param_groups'] = merge_param_groups(s1['param_groups'], s2['param_groups'])
        else:
            ret[k] = v
    return ret


def append_param_ranks(param_ranks, state, rank):
    pids = []
    for grp in state['param_groups']:
        pids.extend(grp['params'])
    for pid in pids:
        if pid not in param_ranks:
            param_ranks[pid] = []
        param_ranks[pid].append(rank)


def remove_params_from_state(state, param_ids):
    ret = {'state': {}}
    for k, v in state.items():
        if k == 'state':
            for pid, sv in v.items():
                if pid not in param_ids:
                    ret['state'][pid] = sv
        else:
            ret[k] = v
    return ret


def merge_param_group(pg_all, pg_sub):
    for k, v in pg_sub.items():
        if k != 'params':
            pg_all[k] = pg_sub[k]
    return pg_all


def merge_param_groups(target_groups, sub_groups):
    return [merge_param_group(pg_all.copy(), pg_sub) for pg_all, pg_sub in zip(target_groups, sub_groups)]


def gather_optimizer_state_dict(optimizer, use_amp_master_param=False, enable_zero=False, to_cpu=True, root=0):
    if use_amp_master_param:
        register_amp_params(optimizer)

    state_dict = optimizer.state_dict(from_global=False)
    # replace local ids with global ones
    state_dict = replace_param_ids(state_dict, optimizer.order_local_to_global)
    state_dict = to_cpu_tensor(state_dict) if to_cpu else state_dict

    param_num = sum([len(pg["params"]) for pg in optimizer.original_param_groups])
    new_state = {}
    for global_order in range(0, param_num):
        pid = optimizer.global_order_to_id[global_order]
        ranks = sorted(list(_pyrannc.get_param_ranks(pid)))
        assert(len(ranks) > 0)
        param_root = ranks[0]

        tensor_item_names = []
        non_tensor_item_names = []
        if _pyrannc.get_rank() == param_root:
            param_state = state_dict["state"][global_order]
            tensor_item_names = [k for k, v in param_state.items() if torch.is_tensor(v)]
            non_tensor_item_names = [k for k, v in param_state.items() if not torch.is_tensor(v)]
        tensor_item_names = comm_utils.bcast_obj(tensor_item_names, param_root)
        non_tensor_item_names = comm_utils.bcast_obj(non_tensor_item_names, param_root)

        new_param_state = {}
        all_item_names = tensor_item_names + non_tensor_item_names
        for k in sorted(all_item_names):
            v = None
            if _pyrannc.get_rank() in ranks:
                if global_order in state_dict["state"]:
                    v = state_dict["state"][global_order][k]
                else:
                    # The size of the segment is zero
                    v = optimizer.param_zero_dummy[global_order]
            if k in tensor_item_names:
                if enable_zero:
                    if pyrannc.get_rank() in ranks:
                        v = _pyrannc.gather_tensor_zero(v, pid).cpu()
                v = tensor_coll.bcast(v, param_root).cpu()
            else:
                v = comm_utils.bcast_obj(v, param_root)
            if _pyrannc.get_rank() == root:
                new_param_state[k] = v
        new_state[global_order] = new_param_state

    if _pyrannc.get_rank() == root:
        new_state_dict = {}
        new_state_dict['param_groups'] = optimizer.original_param_groups
        new_state_dict['state'] = new_state
        return new_state_dict, None

    return None, None


def _get_local_optimizer_state_dict(global_state_dict, used_param_global_order):

    local_state_dict = {'state': {}, 'param_groups': []}

    for global_order, sv in global_state_dict['state'].items():
        if global_order in used_param_global_order:
            local_state_dict['state'][global_order] = sv

    for grp in global_state_dict['param_groups']:
        new_grp = grp.copy()
        new_grp['params'] = [global_order for global_order in grp['params'] if global_order in used_param_global_order]
        local_state_dict['param_groups'].append(new_grp)

    for k, v in global_state_dict.items():
        if k not in local_state_dict.keys():
            local_state_dict[k] = v
    return local_state_dict


def _optimizer_state_to_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _slice_optimizer_state(local_state_dict, param_zero_range, global_order_to_id):
    sliced_state_dict = {'state': {}}
    for k, v in local_state_dict.items():
        if k == 'state':
            for global_order, param_state in v.items():
                new_state_vals = {}
                for state_k, state_v in param_state.items():
                    if torch.is_tensor(state_v):
                        param_slice = param_zero_range[global_order_to_id[global_order]]
                        new_state_vals[state_k] = state_v.flatten().detach().clone()[param_slice]
                    else:
                        new_state_vals[state_k] = state_v
                sliced_state_dict[k][global_order] = new_state_vals
        else:
            sliced_state_dict[k] = v

    return sliced_state_dict


def patch_optimizer(model, optimizer):
    # preserve param groups and order
    optimizer.original_param_groups = optimizer.state_dict()['param_groups']
    new_param_groups = []

    order_local_to_global = {}
    global_order_to_id = {}
    used_param_global_order = []
    local_order = 0
    global_order = 0
    param_zero_range = {}
    param_zero_segment_to_id = {}
    param_zero_dummy = {}
    for param_group in optimizer.param_groups:
        params = []

        for p in param_group['params']:
            pid = id(p)
            if pid in model.used_param_ids:
                skip = False
                if model.enable_zero:
                    p = model.get_local_param_segment(pid)
                    range = model.get_local_param_range(pid)
                    param_zero_range[pid] = slice(range[0], range[1])
                    param_zero_segment_to_id[p] = pid

                    # We skip parameter segments whose size is zero.
                    # PyTorch's optimizers can process params without an element, but
                    # Apex Amp's unscaling using multi_tensor_apply produces incorrect results.
                    if range[0] == range[1]:
                        skip = True
                        # Record tensor with no element. This is used when saving state_dict.
                        # Keeping only dtype should work.
                        param_zero_dummy[global_order] = torch.empty_like(p)

                if not skip:
                    params.append(p)
                    order_local_to_global[local_order] = global_order
                    used_param_global_order.append(global_order)
                    local_order += 1

            global_order_to_id[global_order] = pid
            global_order += 1
        # Need to add a param group even when this rank has no param.
        # Otherwise load_state_dict() of the optimizer will fail because the numbers of param groups do not match.
        param_group['params'] = params
        new_param_groups.append(param_group)
    optimizer.param_groups = new_param_groups
    optimizer.order_local_to_global = order_local_to_global
    optimizer.global_order_to_id = global_order_to_id
    optimizer.param_zero_segment_to_id = param_zero_segment_to_id
    optimizer.param_zero_range = param_zero_range
    optimizer.param_zero_dummy = param_zero_dummy

    # replace state_dict and load_state_dict
    old_state_dict = optimizer.state_dict

    def new_state_dict(opt, from_global=False, **kwargs):
        if from_global:
            global_opt_state_dict, _ = gather_optimizer_state_dict(opt, use_amp_master_param=model.enable_apex_amp,
                                                                   enable_zero=model.enable_zero, **kwargs)
            return global_opt_state_dict
        else:
            return old_state_dict(**kwargs)

    optimizer.state_dict = types.MethodType(new_state_dict, optimizer)

    old_load_state_dict = optimizer.load_state_dict

    def new_load_state_dict(opt, state_dict, from_global=False, **kwargs):
        if from_global:
            # `local_state_dict' uses global order
            local_state_dict = _get_local_optimizer_state_dict(state_dict, used_param_global_order)

            if model.enable_zero:
                local_state_dict = _slice_optimizer_state(local_state_dict, param_zero_range, opt.global_order_to_id)

            global_to_local = {global_order: local_order for local_order, global_order in optimizer.order_local_to_global.items()}
            local_state_dict = replace_param_ids(local_state_dict, global_to_local)
            old_load_state_dict(local_state_dict)
            _optimizer_state_to_cuda(opt, model.device)
        else:
            old_load_state_dict(state_dict, **kwargs)

    optimizer.load_state_dict = types.MethodType(new_load_state_dict, optimizer)

    # replace zero_grad
    if model.enable_zero:
        def new_zero_grad(opt, **kwargs):
            if model.use_amp_master_params:
                unset_master_grads(opt)
            model.zero_grad(**kwargs)

        optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)

