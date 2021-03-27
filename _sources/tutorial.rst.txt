Tutorial
======================================

RaNNC partitions a PyTorch and computes it using multiple GPUs.
Follow the steps below to learn the basic usage of RaNNC.

Steps to use RaNNC
~~~~~~~~~~~~~~~~~~

0. Set up environment
-------------------------

Ensure required tools and libraries (CUDA, NCCL, OpenMPI, etc.) are available.
The libraries must be included in ``LD_LIBRARY_PATH`` at runtime.


1. Import RaNNC
---------------

Insert ``import`` in your script.

.. code-block:: python

  import pyrannc


2. Wrap your model
------------------

Wrap your model by ``pyrannc.RaNNCModule`` with your optimizer.
You can use the wrapped model in almost same manner as the original model (See below).
Note that the original model must be on a CUDA device.

.. code-block:: python

  model = Net()
  model.to(torch.device("cuda"))
  opt = optim.SGD(model.parameters(), lr=0.01)
  model = pyrannc.RaNNCModule(model, optimizer)

If you don't use an optimizer, pass only the model.

.. code-block:: python

  model = pyrannc.RaNNCModule(model)



3. Run forward/backward passes
------------------------------

A ``RaNNCModule`` can run forward/backward passes as with a ``torch.nn.Module``.

.. code-block:: python

  x = torch.randn(64, 3, requires_grad=True).to(torch.device("cuda"))
  out = model(x)
  out.backward(torch.randn_like(out))

Inputs to ``RaNNCModule`` must be CUDA tensors.
RaNNCModule has several more limitations about a wrapped model and inputs/outputs.
See :doc:`limitations` for details.
The optimizer can update model parameters just by calling ``step()``.

The program below (``examples/tutorial_usage.py``) shows the above usage with a very simple model.

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

A program using RaNNC requires to be launched by ``mpirun``.
You can launch the above example script by:

.. code-block:: bash

  mpirun -np 2 python tutorial_usage.py

``-np`` indicates the number of ranks (processes).
RaNNC allocates one CUDA device for each rank.
In the above example, there must be two available CUDA devices.


.. note::

  Each process launched by MPI is expected to load different (mini-)batches. RaNNC automatically gathers the batches from all ranks and compute them as one batch. ``torch.utils.data.distributed.DistributedSampler`` will be useful for this purpose.
