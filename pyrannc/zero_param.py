import functools

import torch

from . import _pyrannc


def _remove_param_if_unnecessary(p):
    if _pyrannc.get_rank() != p.owner:
        p.data = torch.ones(1, dtype=p.dtype).to(p.device)


def store_dist_param(p):
    owner = _pyrannc.store_dist_param(p)
    p.owner = owner
    p.distributed = True
    _remove_param_if_unnecessary(p)


def load_dist_param(pid):
    return _pyrannc.load_dist_param(pid)


class DistributeModelParams(object):
    def __init__(self):
       print("DistributeModelParams init")

    def __enter__(self):
        print("DistributeModelParams __enter__")

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
        print("DistributeModelParams __exit__")

    def _store_dist_params(self, model):
        print("Storing params by DistributeModelParams: {}".format(model.__class__.__name__))
        for p in model.parameters(recurse=False):
            store_dist_param(p)
        for b in model.buffers(recurse=False):
            store_dist_param(b)

    def _set_hooks(self, model):
        # Get param tensors
        def _pre_hook_for_tracing(_model, input):
            for p in _model.parameters(recurse=False):
                p.data = load_dist_param(id(p))
            for b in _model.buffers(recurse=False):
                b.data = load_dist_param(id(b))
            return input

        # Remove param tensors
        def _post_hook_for_tracing(_model, input, output):
            for p in _model.parameters(recurse=False):
                _remove_param_if_unnecessary(p)
            for b in _model.buffers(recurse=False):
                _remove_param_if_unnecessary(b)

        model.register_forward_pre_hook(_pre_hook_for_tracing)
        model.register_forward_hook(_post_hook_for_tracing)
