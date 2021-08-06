import functools
import types

import torch

from . import _pyrannc
from .tensor_coll import _allreduce_sum


def store_dist_param(p):
    if _pyrannc.dist_param_registered(id(p)):
        return

    p.data = _pyrannc.store_dist_param(p)
    p.distributed = True

    old_del = None
    if hasattr(p, "__del__"):
        old_del = p.__del__

    def new_del(param):
        remove_dist_param(id(param))
        if old_del:
            old_del()

    p.__del__ = types.MethodType(new_del, p)


def load_dist_param(pid):
    return _pyrannc.load_dist_param(pid)


def set_dist_param(pid, src):
    _pyrannc.set_dist_param(pid, src)


def get_dist_param_segment(pid):
    return _pyrannc.get_dist_param_segment(pid)


def get_dist_param_range(pid):
    range = _pyrannc.get_dist_param_range(pid)
    return slice(range[0], range[1])


def set_dist_param_dtype(pid, dtype):
    _pyrannc.set_dist_param_dtype(pid, dtype)


def remove_dist_param(pid):
    _pyrannc.remove_dist_param(pid)


class DistributeModelParams(object):

    def __init__(self, enable=True):
        self.enable = enable
        self.hooks = []

        if enable:
            # Creation of NCCL communicator failed during tracing.
            # So we create a communicator including all ranks on initialization.
            _allreduce_sum(torch.zeros(3, 3).cuda())


    def __enter__(self):
        if not self.enable:
            return

        def add_post_process(f):
            @functools.wraps(f)
            def wrapper(model, *args, **kwargs):
                f(model, *args, **kwargs)
                self._store_dist_params(model)
                self._set_hooks(model)

            return wrapper

        for subclass in torch.nn.modules.module.Module.__subclasses__():
            subclass._old_init = subclass.__init__
            subclass.__init__ = add_post_process(subclass.__init__)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enable:
            return

        for subclass in torch.nn.modules.module.Module.__subclasses__():
            subclass.__init__ = subclass._old_init

    def _store_dist_params(self, model):
        for p in model.parameters(recurse=False):
            store_dist_param(p)
        for b in model.buffers(recurse=False):
            store_dist_param(b)

    def _set_hooks(self, model):
        # Get param tensors
        def _pre_hook_for_tracing(_model, input):
            _pyrannc.set_tracing_state(False)
            for p in _model.parameters(recurse=False):
                # Convert data type for amp
                set_dist_param_dtype(id(p), p.dtype)
                p.data = load_dist_param(id(p))
            for b in _model.buffers(recurse=False):
                set_dist_param_dtype(id(b), b.dtype)
                b.data = load_dist_param(id(b))
            _pyrannc.set_tracing_state(True)
            return input

        def _post_hook_for_tracing(_model, input, output):
            _pyrannc.set_tracing_state(False)
            for p in _model.parameters(recurse=False):
                p.data = get_dist_param_segment(id(p))
            for b in _model.buffers(recurse=False):
                b.data = get_dist_param_segment(id(b))
            _pyrannc.set_tracing_state(True)

        model.register_forward_pre_hook(_pre_hook_for_tracing)
        model.register_forward_hook(_post_hook_for_tracing)
