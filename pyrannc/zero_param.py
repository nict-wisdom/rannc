import torch

from . import _pyrannc


def store_zero_param(p):
    owner = _pyrannc.store_zero_param(p)
    if _pyrannc.get_rank() != owner:
        p.data = torch.ones(1, dtype=p.dtype).to(p.device)


