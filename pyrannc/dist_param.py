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
    r"""
     Distributes model parameters on initialization.
    """

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
            def wrapper(model, *args, **kwargs):
                f(model, *args, **kwargs)
                self._store_dist_params(model)
                self._set_hooks(model)
                model._dist_params = True

            return wrapper

        def fq_classname(cls):
            module = cls.__module__
            name = cls.__qualname__
            if module is not None and module != "__builtin__":
                name = module + "." + name
            return name

        def has_custom_init(cls):
            return "__init__" in vars(cls)

        for subclass in self._all_subclasses(torch.nn.modules.module.Module):
            if "torch.jit" not in fq_classname(subclass) and has_custom_init(subclass):
                subclass._old_init = subclass.__init__
                subclass.__init__ = add_post_process(subclass.__init__)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enable:
            return

        for subclass in self._all_subclasses(torch.nn.modules.module.Module):
            if hasattr(subclass, "_old_init"):
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

        self.hooks.append(model.register_forward_pre_hook(_pre_hook_for_tracing))
        self.hooks.append(model.register_forward_hook(_post_hook_for_tracing))

    def _all_subclasses(self, cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in self._all_subclasses(c)])
