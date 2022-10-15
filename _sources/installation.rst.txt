Installation
=================

Prerequisites
-------------

RaNNC works only with CUDA devices (CPU only or TPU environments are not supported).
RaNNC requires the following libraries and tools at runtime.

* *CUDA*: A CUDA runtime (Version 11) must be available in the runtime environment.
* *NCCL*: NCCL (Version >= 2.11.4) must be available in the runtime environment.
* *MPI*: A program using RaNNC must be launched with MPI. MPI libraries must also be available at runtime. RaNNC has been tested with OpenMPI v4.0.7.
* *libstd++*: ``libstd++`` must support ``GLIBCXX_3.4.21`` to use the distributed ``pip`` packages (these packages are built with gcc 7.5.0).


Installation
------------

The current version (``0.7.5``) of RaNNC requires PyTorch v1.11.0.
``pip`` packages for ``linux_x86_64`` are available for the following combinations of Python and CUDA versions.

* Python version: 3,7, 3.8, 3.9, 3.10
* CUDA version: 11.3

The following commands install PyToch and RaNNC with pip.
(The package version should be specified as ``0.7.5+cu[CUDA_VERSION_WITHOUT_DOT]``)

.. code-block:: bash

  pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
  pip install pyrannc==0.7.5+cu113 -f https://nict-wisdom.github.io/rannc/installation.html


Use the following links to manually download the packages.

* :download:`For Python 3.7 / CUDA 11.3 <https://github.com/nict-wisdom/rannc/releases/download/v0.7.5/pyrannc-0.7.5+cu113-cp37-cp37m-linux_x86_64.whl>`
* :download:`For Python 3.8 / CUDA 11.3 <https://github.com/nict-wisdom/rannc/releases/download/v0.7.5/pyrannc-0.7.5+cu113-cp38-cp38-linux_x86_64.whl>`
* :download:`For Python 3.9 / CUDA 11.3 <https://github.com/nict-wisdom/rannc/releases/download/v0.7.5/pyrannc-0.7.5+cu113-cp39-cp39-linux_x86_64.whl>`
* :download:`For Python 3.10 / CUDA 11.3 <https://github.com/nict-wisdom/rannc/releases/download/v0.7.5/pyrannc-0.7.5+cu113-cp310-cp310-linux_x86_64.whl>`

If the above packages do not match your Python/CUDA versions, create a suitable package using ``Makefile``
in ``docker/``. ``make.sh`` shows the commands to create wheel packages.
