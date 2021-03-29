ARG OS_NAME=ubuntu
ARG OS_VERSION
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-${OS_NAME}${OS_VERSION}

ARG PYTHON_VERSION

ENV DOCKER_BUILD_DIR /build
ENV BOOST_VERSION 1.74.0
ENV OPENUCX_VERSION 1.9.0
ENV OPENMPI_VERSION 4.0.5
ENV CONDA_VERSION 4.9.2
ENV PYTORCH_VERSION 1.8.0

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y wget git libnuma-dev libibverbs-dev librdmacm-dev libssl-dev && apt-get clean

RUN mkdir -p $DOCKER_BUILD_DIR

# Boost
RUN cd ${DOCKER_BUILD_DIR} \
    && export BOOST_VERSION_FILE=${BOOST_VERSION//./_} \
    && wget --quiet https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_FILE}.tar.gz \
    && tar -xzf boost_${BOOST_VERSION_FILE}.tar.gz \
    && cd boost_${BOOST_VERSION_FILE} \
    && /bin/bash ./bootstrap.sh \
    && ./b2 cxxflags=-fPIC cflags=-fPIC link=static -a \
            define=_GLIBCXX_USE_CXX11_ABI=0 --with-graph --with-filesystem --with-system \
            -j2 install

# UCX
RUN cd ${DOCKER_BUILD_DIR} \
    && wget --quiet https://github.com/openucx/ucx/releases/download/v${OPENUCX_VERSION}/ucx-${OPENUCX_VERSION}.tar.gz \
    && tar -xzf ucx-${OPENUCX_VERSION}.tar.gz \
    && cd ucx-${OPENUCX_VERSION} \
    && ./configure --prefix=/usr/local/ucx --disable-dependency-tracking --disable-optimizations --disable-logging \
          --disable-debug --disable-assertions --enable-mt --disable-params-check --without-java --enable-cma \
          --with-verbs --without-cm --with-rdmacm --without-rocm --without-xpmem --without-ugni \
          --disable-doxygen-doc --with-cuda=/usr/local/cuda \
    && make -j 4 \
    && make install

# OpenMPI
RUN cd ${DOCKER_BUILD_DIR} \
    && wget --quiet https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION%.*}/openmpi-${OPENMPI_VERSION}.tar.gz \
    && tar -xzf openmpi-${OPENMPI_VERSION}.tar.gz \
    && cd openmpi-${OPENMPI_VERSION} \
    && ./configure --enable-mpi-cxx --enable-mpi-cxx-seek --with-cuda \
        --with-ucx=/usr/local/ucx --disable-dependency-tracking \
        --disable-getpwuid --without-usnic --enable-orterun-prefix-by-default \
    && make -j 4 && make install

# Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_${CONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc

## Env & PyTorch
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda create -y -n rannc python=${PYTHON_VERSION} \
    && conda activate rannc \
    && if [[ $CUDA_VERSION =~ ^11. ]]; then \
          export CUDA_TOOLKIT_VERSION="11.1"; \
       else \
          export CUDA_TOOLKIT_VERSION="10.2"; \
       fi \
    && conda install -y pytorch=${PYTORCH_VERSION} torchvision torchaudio cudatoolkit=${CUDA_TOOLKIT_VERSION} -c pytorch -c conda-forge \
    && conda clean --all \
    && echo "conda activate rannc" >> /root/.bashrc

RUN ldconfig
