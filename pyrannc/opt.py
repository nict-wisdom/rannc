import pickle
import types

import torch

from . import _pyrannc
from .amp import register_amp_params


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


def gather_optimizer_state_dict(optimizer, use_amp_master_param=False, to_cpu=True, root=0):
    if use_amp_master_param:
        register_amp_params(optimizer)

    state_dict = optimizer.state_dict(from_global=False)
    # replace local ids with global ones
    state_dict = replace_param_ids(state_dict, optimizer.order_local_to_global)
    state_dict = to_cpu_tensor(state_dict) if to_cpu else state_dict

    if _pyrannc.get_rank() == root:
        param_ranks = {}
        append_param_ranks(param_ranks, state_dict, root)

        state_dict['param_groups'] = merge_param_groups(optimizer.original_param_groups, state_dict['param_groups'])

        for r in range(0, _pyrannc.get_world_size()):
            if r == root:
                continue

            # Send pids that are already available on root rank
            current_pids = list(state_dict['state'].keys())
            pck_current_pids = pickle.dumps(current_pids)
            _pyrannc.send_bytes(pck_current_pids, r)

            data = _pyrannc.recv_bytes(r)
            rank_state = pickle.loads(data)
            append_param_ranks(param_ranks, rank_state, r)
            state_dict = merge_state_dict(state_dict, rank_state)

        _pyrannc.barrier()

        return state_dict, param_ranks

    else:
        pck_current_root_pids = _pyrannc.recv_bytes(root)
        current_root_pids = pickle.loads(pck_current_root_pids)
        state_dict = remove_params_from_state(state_dict, current_root_pids)
        pickled_state = pickle.dumps(state_dict)
        _pyrannc.send_bytes(pickled_state, root)

        _pyrannc.barrier()

    return None, None


def _get_local_optimizer_state_dict(global_state_dict, pids):
    local_state_dict = {}
    for k, v in global_state_dict.items():
        if k == 'state':
            local_state_dict['state'] = {pid: sv for pid, sv in v.items() if pid in pids}
        elif k == 'param_groups':
            local_state_dict['param_groups'] = []
            for grp in v:
                new_grp = grp.copy()
                new_grp['params'] = [pid for pid in grp['params'] if pid in pids]
                local_state_dict['param_groups'].append(new_grp)
        else:
            local_state_dict[k] = v
    return local_state_dict


def _optimizer_state_to_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _slice_optimizer_state(local_state_dict, param_zero_range):
    sliced_state_dict = {}
    for k, v in local_state_dict.items():
        if k == 'state':
            for pid, param_state in v.items():
                new_state_vals = {}
                for state_k, state_v in param_state.items():
                    if torch.is_tensor(state_v):
                        # slice here
                        slice = param_zero_range[pid]
                        new_state_vals[state_k] = state_v.detach().clone()[slice]
                    else:
                        new_state_vals[state_k] = state_v
                sliced_state_dict[k][pid] = new_state_vals
        else:
            sliced_state_dict[k] = v

    return sliced_state_dict


def patch_optimizer(model, optimizer):
    # preserve param groups and order
    optimizer.original_param_groups = optimizer.state_dict()['param_groups']
    print(" self.optimizer.original_param_groups={}".format(optimizer.original_param_groups))
    print(" self.optimizer.param_groups={}".format(optimizer.param_groups))
    new_param_groups = []

    order_local_to_global = {}
    used_param_global_order = []
    local_order = 0
    global_order = 0
    param_zero_range = {}
    param_zero_segment_to_id = {}
    for param_group in optimizer.param_groups:
        params = []

        for p in param_group['params']:
            if id(p) in model.used_param_ids:
                if model.enable_zero:
                    pid = id(p)
                    p = model.get_local_param_segment(pid)
                    range = model.get_local_param_range(pid)
                    param_zero_range[pid] = slice(range[0], range[1])
                    param_zero_segment_to_id[p] = pid

                params.append(p)
                order_local_to_global[local_order] = global_order
                used_param_global_order.append(global_order)
                local_order += 1
            global_order += 1
        # Need to add a param group even when this rank has no param.
        # Otherwise load_state_dict() of the optimizer will fail because the numbers of param groups do not match.
        param_group['params'] = params
        new_param_groups.append(param_group)
    optimizer.param_groups = new_param_groups
    optimizer.order_local_to_global = order_local_to_global
    optimizer.param_zero_segment_to_id = param_zero_segment_to_id
    optimizer.param_zero_range = param_zero_range

    # replace state_dict and load_state_dict
    old_state_dict = optimizer.state_dict

    def new_state_dict(opt, from_global=False, **kwargs):
        if from_global:
            global_opt_state_dict, _ = gather_optimizer_state_dict(opt, use_amp_master_param=model.use_amp_master_params, **kwargs)
            return global_opt_state_dict
        else:
            return old_state_dict(**kwargs)

    optimizer.state_dict = types.MethodType(new_state_dict, optimizer)

    old_load_state_dict = optimizer.load_state_dict

    def new_load_state_dict(opt, state_dict, from_global=False, **kwargs):
        if from_global:
            local_state_dict = _get_local_optimizer_state_dict(state_dict, used_param_global_order)

            if model.enable_zero:
                local_state_dict = _slice_optimizer_state(local_state_dict, param_zero_range)

            old_load_state_dict(local_state_dict)
            _optimizer_state_to_cuda(opt, model.device)
        else:
            old_load_state_dict(state_dict, **kwargs)

    optimizer.load_state_dict = types.MethodType(new_load_state_dict, optimizer)

    # replace zero_grad
    if model.enable_zero:
        def new_zero_grad(opt, **kwargs):
            model.zero_grad(**kwargs)

        optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)

