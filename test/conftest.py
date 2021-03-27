import pytest
import torch
import torch.distributed as dist

import pyrannc


def pytest_addoption(parser):
    parser.addoption("--master-address", action="store", default="localhost")
    parser.addoption("--master-port", action="store", default=28888)
    parser.addoption("--batch-size", action="store", default=64)
    parser.addoption("--iteration", action="store", default=2)


@pytest.fixture(scope="session")
def init_dist(request):
    assert torch.cuda.is_available()

    master_addr = request.config.getoption('--master-address')
    master_port = request.config.getoption('--master-port')
    torch.backends.cudnn.enabled = True
    # torch.set_deterministic(True)
    comm_backend = "nccl"
    init_method = "tcp://{}:{}".format(master_addr, master_port)

    dist.init_process_group(
        comm_backend,
        init_method=init_method,
        rank=pyrannc.get_rank(),
        world_size=pyrannc.get_world_size())


@pytest.fixture
def batch_size(request):
    return int(request.config.getoption("--batch-size"))


@pytest.fixture
def iteration(request):
    return int(request.config.getoption("--iteration"))
