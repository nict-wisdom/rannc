import torch


def _to_in_place(tensors, device):
    for p in tensors:
        with torch.no_grad():
            p.data = p.to(device, dtype=p.dtype)
