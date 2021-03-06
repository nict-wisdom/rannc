Tutorial
========

The greatest feature of RaNNC is that it can automatically partition a model written for PyTorch and train it using
multiple GPUs (model parallelism).
Unlike other frameworks for model parallelism, users do not need to modify the model for partitioning.

In this tutorial, you will learn how to use RaNNC to train very large models that cannot be trained using data parallelism only.

Steps to use RaNNC
~~~~~~~~~~~~~~~~~~

0. Set up environment
-------------------------

Ensure the required tools and libraries (CUDA, NCCL, OpenMPI, etc.) are available in your environment.
The libraries must be included in ``LD_LIBRARY_PATH`` at runtime.
Then install ``pyrannc`` following the commands shown in :doc:`installation` page.


1. Import RaNNC
---------------

Insert ``import`` in your script.

.. code-block:: python

  import pyrannc


2. Wrap your model
------------------

Wrap your model by using ``pyrannc.RaNNCModule`` with your optimizer.
You can use the wrapped model in almost the same manner as the original model (see below).

.. code-block:: python

  model = Net()
  opt = optim.SGD(model.parameters(), lr=0.01)
  model = pyrannc.RaNNCModule(model, optimizer)


Note that the original model does not need to be on a CUDA device.
Thus you can declare a very large model that does not fit to the memory of a GPU.

If you do not use an optimizer, pass only the model.

.. code-block:: python

  model = pyrannc.RaNNCModule(model)


3. Run forward/backward passes
------------------------------

``RaNNCModule`` can run forward/backward passes, as with ``torch.nn.Module``.

.. code-block:: python

  x = torch.randn(batch_size, hidden_size, requires_grad=True).to(torch.device("cuda"))
  out = model(x)
  out.backward(torch.randn_like(out))

Inputs to ``RaNNCModule`` must be CUDA tensors.
RaNNCModule has several more limitations regarding a wrapped model and inputs/outputs.
See :doc:`limitations` for details.
The optimizer can update model parameters simply by calling ``step()``.

The `script below <https://github.com/nict-wisdom/rannc/blob/main/examples/tutorial.py>`_ shows the usage with a very simple model.

.. code-block:: python

  import sys
  import torch
  import torch.nn as nn
  import torch.optim as optim

  import pyrannc


  class Net(nn.Module):
      def __init__(self, hidden, layers):
          super(Net, self).__init__()
          self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(layers)])

      def forward(self, x):
          for l in self.layers:
              x = l(x)
          return x


  batch_size = int(sys.argv[1])
  hidden = int(sys.argv[2])
  layers = int(sys.argv[3])

  model = Net(hidden, layers)
  if pyrannc.get_rank() == 0:
      print("#Parameters={}".format(sum(p.numel() for p in model.parameters())))

  opt = optim.SGD(model.parameters(), lr=0.01)
  model = pyrannc.RaNNCModule(model, opt)

  x = torch.randn(batch_size, hidden, requires_grad=True).to(torch.device("cuda"))
  out = model(x)
  target = torch.randn_like(out)
  out.backward(target)
  opt.step()
  print("Finished on rank{}".format(pyrannc.get_rank()))


4. Launch (with a small model)
------------------------------

A program using RaNNC must be launched using ``mpirun``.
Begin with launching the above script with a very small model using two GPUs.

.. code-block:: bash

  # The arguments are: [batch_size] [hidden] [layers]
  mpirun -np 2 python tutorial.py 64 512 10


(Ensure MPI is properly configured in your environment before you run RaNNC. You may need more options for MPI like
``--mca pml ucx --mca btl ^vader,tcp,openib ...``)

``-np`` indicates the number of processes (ranks).
RaNNC allocates one CUDA device for each process.
In the above example, there must be two available CUDA devices.
By properly setting nodes for MPI, you can run processes using RaNNC across multiple nodes
(Ensure that you have the equal or more number of GPUs than processes).

The following shows the output on our compute node that has eight NVIDIA A100's (40GB memory).

.. code-block:: bash

  $ mpirun -np 2 --mca pml ucx --mca btl ^vader,tcp,openib --mca coll ^hcoll python tutorial.py 64 512 10
  [RaNNCProcess] [info] RaNNC started on rank 1 (gpunode001)
  [RaNNCProcess] [info] RaNNC started on rank 0 (gpunode001)
  [RaNNCProcess] [info] CUDA device assignments:
  [RaNNCProcess] [info]  Worker 0: device0@gpunode001
  [RaNNCProcess] [info]  Worker 1: device1@gpunode001
  #Parameters=2626560
  [RaNNCModule] [info] Tracing model ...
  [RaNNCModule] [info] Converting torch model to IR ...
  [RaNNCModule] [info] Running profiler ...
  [RaNNCModule] [info] Profiling finished
  [RaNNCModule] [info] Assuming batch size: 128
  [Decomposer] [info] Decomposer: ml_part
  [Decomposer] [info] Available device memory: 38255689728
  [Decomposer] [info] Starting model partitioning ... (this may take a very long time)
  [DPStaging] [info] Estimated profiles of subgraphs: batch_size=128 np=2 pipeline=1 use_amp=0 zero=0
    graph=MERGE_0_9 repl=2 fwd_time=4722 bwd_time=24237 ar_time=978 in_size=131072 out_size=131072 fp32param_size=10506240 fp16param_size=0 total_mem=54759424 (fwd+bwd=33353728 opt=21012480 comm=393216)
  [Decomposer] [info]  Assigned subgraph MERGE_0_9 to rank[1,0]
  [RaNNCModule] [info] Routes verification passed.
  [ParamStorage] [info] Synchronizing parameters ...
  [RaNNCModule] [info] RaNNCModule is ready. (rank0)
  [RaNNCModule] [info] RaNNCModule is ready. (rank1)
  Finished on rank0
  Finished on rank1

Since this model is very small, RaNNC determines to train it using only data parallelism.
You can see the partitioning result in the following part.
The computational graph that is equivalent to the model was named ``MERGE_0_9`` and assigned to ranks 0 and 1
(replicated for data parallelism).

.. code-block:: bash

  [DPStaging] [info] Estimated profiles of subgraphs: batch_size=128 np=2 pipeline=1 use_amp=0 zero=0
    graph=MERGE_0_9 repl=2 fwd_time=4722 bwd_time=24237 ar_time=978 in_size=131072 out_size=131072 fp32param_size=10506240 fp16param_size=0 total_mem=54759424 (fwd+bwd=33353728 opt=21012480 comm=393216)
  [Decomposer] [info]  Assigned subgraph MERGE_0_9 to rank[1,0]


.. note::

  Each process launched by MPI is expected to load different (mini-)batches.
  RaNNC automatically gathers the batches from all ranks and computes them as a single batch.
  Therefore, the effective (global) batch size is [number of processes (np)] * [batch size per process].
  ``torch.utils.data.distributed.DistributedSampler`` is useful to properly take batches in each process.



5. Model partitioning for very large models
-------------------------------------------

If the number of parameters of a model is very large, you cannot train the model only with data parallelism.
RaNNC automatically partitions such models for *model parallelism*.

To see how RaNNC partitions such a large model, set ``hidden`` and ``layers`` to 5000 and 100 respectively.
Given the configuration, the model has more than 2.5 billion parameters.

You cannot train this model using only data parallelism because the size of parameters, gradients
and optimizer states exceeds the memory of the GPU (40GB). (The model requires 10GB for parameters, 10GB for gradients,
20GB for optimizer states, and more for activations)

Let's use all the GPUs on the node (eight GPUs) for this configuration.

.. code-block:: bash

  $ mpirun -np 8 --mca pml ucx --mca btl ^vader,tcp,openib --mca coll ^hcoll python tutorial.py 64 5000 100
  [RaNNCProcess] [info] RaNNC started on rank 0 (gpunode001)
  [RaNNCProcess] [info] RaNNC started on rank 1 (gpunode001)
  ...
  Parameters=2500500000
  ..
  [Decomposer] [info] Starting model partitioning ... (this may take a very long time)
  [DPStaging] [info] Estimated profiles of subgraphs: batch_size=512 np=8 pipeline=1 use_amp=0 zero=0
  graph=MERGE_0_4 repl=4 fwd_time=27516 bwd_time=126756 ar_time=437809 in_size=2560000 out_size=2560000 fp32param_size=4700940000 fp16param_size=0 total_mem=23707792544 (fwd+bwd=14298232544 opt=9401880000 comm=7680000)
  graph=MERGE_5_9 repl=4 fwd_time=31228 bwd_time=153762 ar_time=493699 in_size=2560000 out_size=2560000 fp32param_size=5301060000 fp16param_size=0 total_mem=26732209376 (fwd+bwd=16122409376 opt=10602120000 comm=7680000)
  [Decomposer] [info]  Assigned subgraph MERGE_5_9 to rank[7,5,1,3]
  [Decomposer] [info]  Assigned subgraph MERGE_0_4 to rank[6,4,0,2]
  ...

The partitioning may take a long time when the model is very large. (It took around five minutes in our environment)

The model was partitioned into two computational graphs (``MERGE_0_4`` and ``MERGE_5_9``) for model parallelism and they were assigned to rank[6,4,0,2] and ranks[7,5,1,3] respectively for data parallelism
(hybrid model/data parallelism).
Note that RaNNC may set different numbers of replicas for data parallelism for each computational graph to optimize the training throughput.

For more practical usages, ``test/test_simple.py`` and `examples <https://github.com/nict-wisdom/rannc-examples/>`_ will be helpful.
