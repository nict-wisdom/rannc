from . import _pyrannc
import pickle


def bcast_obj(obj, root):
    bytes_data = pickle.dumps(obj)
    recv_bytes = _pyrannc.bcast_bytes(bytes_data, root)
    return pickle.loads(recv_bytes)

