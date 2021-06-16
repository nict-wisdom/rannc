from . import _pyrannc


def _allreduce_sum(t):
    return _pyrannc.allreduce_tensor(t, True)


def _allreduce_min(t):
    return _pyrannc.allreduce_tensor(t, False)
