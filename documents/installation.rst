Installation
=================

Prerequisites
-------------

RaNNC works only with CUDA devices (CPU only/TPU environments are not supported).
RaNNC requires the following libraries and tools at runtime.

* *CUDA*: A CUDA runtime must be available at the runtime environment. Currently RaNNC is tested with CUDA 10.2 and 11.0.
* *NCCL*: NCCL (Version >= 2.7.3 is required) must be available at the runtime environment. RaNNC uses NCCL both for allreduce and P2P communications.
* *MPI*: A program using RaNNC must be launched with MPI. MPI libraries must also be available at runtime. RaNNC is tested with OpenMPI v4.0.5.
* *libstd++*: ``libstd++`` must support ``GLIBCXX_3.4.21`` to use the distributed ``pip`` packages (The packages are built with gcc 5.4.0).

Installation
------------

This version of RaNNC requires PyTorch v1.8.0.
``pip`` packages for ``linux_x86_64`` are available from the following links.

* :download:`For Python 3.7 <resources/pyrannc-0.5-cp37-cp37m-linux_x86_64.whl>`
* :download:`For Python 3.8 <resources/pyrannc-0.5-cp38-cp38m-linux_x86_64.whl>`

You can create a new conda environment and install RaNNC by the following commands.
Set a CUDA version available in your environment.

.. code-block:: bash

  conda create -n rannc python=3.8
  conda activate rannc
  conda install pytorch==1.8.0 cudatoolkit=10.2 -c pytorch
  pip install pyrannc-0.5-cp38-cp38m-linux_x86_64.whl
