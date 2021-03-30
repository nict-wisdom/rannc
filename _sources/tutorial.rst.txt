Tutorial
======================================

RaNNC partitions a PyTorch and computes it using multiple GPUs.
Follow the steps below to learn the basic usage of RaNNC.

Steps to use RaNNC
~~~~~~~~~~~~~~~~~~

0. Set up environment
-------------------------

Ensure the required tools and libraries (CUDA, NCCL, OpenMPI, etc.) are available.
The libraries must be included in ``LD_LIBRARY_PATH`` at runtime.


1. Import RaNNC
---------------

Insert ``import`` in your script.

.. code-block:: python

  import pyrannc


2. Wrap your model
------------------

Wrap your model by using ``pyrannc.RaNNCModule`` with your optimizer.
You can use the wrapped model in almost the same manner as the original model (see below).
Note that the original model must be on a CUDA device.

.. code-block:: python

  model = Net()
  model.to(torch.device("cuda"))
  opt = optim.SGD(model.parameters(), lr=0.01)
  model = pyrannc.RaNNCModule(model, optimizer)

If you do not use an optimizer, pass only the model.

.. code-block:: python

  model = pyrannc.RaNNCModule(model)



3. Run forward/backward passes
------------------------------

``RaNNCModule`` can run forward/backward passes, as with ``torch.nn.Module``.

.. code-block:: python

  x = torch.randn(64, 3, requires_grad=True).to(torch.device("cuda"))
  out = model(x)
  out.backward(torch.randn_like(out))

Inputs to ``RaNNCModule`` must be CUDA tensors.
RaNNCModule has several more limitations regarding a wrapped model and inputs/outputs.
See :doc:`limitations` for details.
The optimizer can update model parameters simply by calling ``step()``.

The program below shows the above usage with a very simple model.

.. code-block:: python

  import torch
  import torch.nn as nn
  import torch.optim as optim

  import pyrannc


  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.fc1 = nn.Linear(3, 2, bias=False)
          self.fc2 = nn.Linear(2, 3, bias=False)

      def forward(self, x):
          x = self.fc1(x)
          x = self.fc2(x)
          return x


  model = Net()
  model.to(torch.device("cuda"))
  opt = optim.SGD(model.parameters(), lr=0.01)
  model = pyrannc.RaNNCModule(model, opt)

  x = torch.randn(64, 3, requires_grad=True).to(torch.device("cuda"))
  out = model(x)

  target = torch.randn_like(out)
  out.backward(target)

  opt.step()


4. Launch
---------

A program using RaNNC must be launched using ``mpirun``.
You can launch the above example script by

.. code-block:: bash

  mpirun -np 2 python tutorial_usage.py

``-np`` indicates the number of ranks (processes).
RaNNC allocates one CUDA device for each rank.
In the above example, there must be two available CUDA devices.

The following output shows that RaNNC uses only data parallelism to train this model.

.. code-block:: bash

  ...
  [DPStaging] [info] Estimated profiles of subgraphs (#partition(s))=1: batch_size=128 ranks=2 pipeline_num=2
  [DPStaging] [info]   graph=MERGE_0_1 repl=2 cp=true fwd_time=99 bwd_time=356 ar_time=0 in_size=1536 out_size=1536 mem=4752 (fwd+bwd=4656 opt=96)
  [Decomposer] [info]  Assigned subgraph MERGE_0_1 to rank[1,0]
  ...

RaNNC automatically partitions a given model if the model is too large to be stored on one CUDA device's memory.
The model in this example is so small that data parallelism works well.

.. note::

  Each process launched by MPI is expected to load different (mini-)batches. RaNNC automatically gathers the batches from all ranks and computes them as a single batch. ``torch.utils.data.distributed.DistributedSampler`` is useful for this purpose.


To see how the partitioning works, you can force RaNNC to partition a model using model parallelism.
The environment variable ``RANNC_PARTITION_NUM`` forces the number of partitions for model parallelism (See also :doc:`config`).

.. code-block:: bash

  mpirun -np 2 -x RANNC_PARTITION_NUM=2 python tutorial_usage.py

The output shows that RaNNC produced two partitions and allocated GPUs to them.

.. code-block:: bash

  ...
  [DPStaging] [info] Estimated profiles of subgraphs (#partition(s))=2: batch_size=128 ranks=2 pipeline_num=2
  [DPStaging] [info]   graph=ML_mod_jy9757acx1eve7nm_p0 repl=1 cp=true fwd_time=83 bwd_time=295 ar_time=0 in_size=1536 out_size=1024 mem=3656 (fwd+bwd=3608 opt=48)
  [DPStaging] [info]   graph=ML_mod_jy9757acx1eve7nm_p1 repl=1 cp=true fwd_time=83 bwd_time=296 ar_time=0 in_size=1024 out_size=1536 mem=5192 (fwd+bwd=5144 opt=48)
  [Decomposer] [info]  Assigned subgraph ML_mod_jy9757acx1eve7nm_p1 to rank[1]
  [Decomposer] [info]  Assigned subgraph ML_mod_jy9757acx1eve7nm_p0 to rank[0]
  ...

For more practical usages, scripts in ``test/`` and `examples <https://github.com/nict-wisdom/rannc-examples/>`_ will be helpful.
