Building from source
====================

Compiler version
----------------

You must use GCC v5.4 or newer. RaNNC has been tested with GCC v8.2.
Note that RaNNC must be built in compliance with ABI of PyTorch.

RaNNC is built with *Pre-cxx11 ABI* (``_GLIBCXX_USE_CXX11_ABI=0``) as default because PyTorch installed via conda is built with it.
You can change the ABI setting in ``CMakeLists.txt``.
PyTorch provides you with `function <https://pytorch.org/docs/stable/generated/torch.compiled_with_cxx11_abi.html>`_ below to understand how the binary is compiled.


Build and Install
-----------------

Clone the repository with the submodules (``--recursive`` is required).

.. code-block:: bash

    git clone --recursive https://github.com/nict-wisdom/rannc


You need to set some environment variables before building RaNNC to help cmake find the dependent libraries.

.. list-table:: Variables for building configurations

   * - Variable
     -
   * - CUDA_HOME
     - Path to a CUDA runtime directory.
   * - MPI_DIR
     - Path to an MPI installation directory.
   * - BOOST_DIR
     - Path to a Boost libraries directory.
   * - CUDNN_ROOT_DIR
     - Path to a cuDNN libraries directory.
   * - LD_LIBRARY_PATH
     - Must contain the path to NCCL lib directory.

The building process refers to PyTorch installed with conda.
Therefore, install PyTorch using your python and run ``setup.py``.
The following script shows the configurations used to install RaNNC from the source.

.. code-block:: bash

 #!/usr/bin/env bash

  # Activate conda
  source [CONDA_PATH]/etc/profile.d/conda.sh
  conda activate rannc

  # Set dependencies
  export CUDA_HOME="$(dirname $(which nvcc))/../"
  export MPI_DIR="$(dirname $(which ompi_info))/../"
  export BOOST_DIR=[BOOST_DIR_PATH]
  export CUDNN_ROOT_DIR=[YOUR_CUDNN_DIR_PATH]

  python setup.py build -g install


*Makefiles* under ``docker/`` show the complete process used to build and install RaNNC.
These are used to build the pip packages.
