import functools

import torch

from . import _pyrannc


def store_zero_param(p):
    owner = _pyrannc.store_zero_param(p)
    if _pyrannc.get_rank() != owner:
        p.data = torch.ones(1, dtype=p.dtype).to(p.device)


def load_zero_param(pid):
    return _pyrannc.load_zero_param(pid)


class DistributeModelParams(object):
    def __init__(self):
       print("DistributeModelParams init")

    def __enter__(self):
        print("DistributeModelParams __enter__")

        def add_post_process(f):
            @functools.wraps(f)
            def wrapper(model, *args, **kwargs):
                f(model, *args, **kwargs)
                self._store_zero_params(model)

            return wrapper

        for subclass in torch.nn.modules.module.Module.__subclasses__():
            subclass._old_init = subclass.__init__
            subclass.__init__ = add_post_process(subclass.__init__)

    def __exit__(self, exc_type, exc_value, traceback):
        print("DistributeModelParams __exit__")

    def _store_zero_params(self, model):
        print("Storing params by DistributeModelParams: {}".format(model.__class__.__name__))
        for p in model.parameters(recurse=False):
            store_zero_param(p)

