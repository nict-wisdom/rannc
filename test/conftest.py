import pytest
import numpy as np
import random
import torch
import torch.distributed as dist


def pytest_addoption(parser):
    parser.addoption("--master-address", action="store", default="localhost")
    parser.addoption("--master-port", action="store", default=28888)
    parser.addoption("--rank", action="store", default=-1)
    parser.addoption("--world-size", action="store", default=-1)
    parser.addoption("--batch-size", action="store", default=64)
    parser.addoption("--iteration", action="store", default=2)
    parser.addoption("--seed", action="store", default=0)


@pytest.fixture(scope="session")
def init_dist(request):
    assert torch.cuda.is_available()

    master_addr = request.config.getoption('--master-address')
    master_port = request.config.getoption('--master-port')
    rank = int(request.config.getoption('--rank'))
    world_size = int(request.config.getoption('--world-size'))

    torch.backends.cudnn.enabled = True
    # torch.set_deterministic(True)
    comm_backend = "nccl"
    init_method = "tcp://{}:{}".format(master_addr, master_port)

    dist.init_process_group(
        comm_backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size)


@pytest.fixture(scope="function")
def init_seed(request):
    seed = int(request.config.getoption('--seed'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@pytest.fixture
def batch_size(request):
    return int(request.config.getoption("--batch-size"))


@pytest.fixture
def iteration(request):
    return int(request.config.getoption("--iteration"))
