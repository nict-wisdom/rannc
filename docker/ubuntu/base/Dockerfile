ARG OS_NAME=ubuntu
ARG OS_VERSION
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-${OS_NAME}${OS_VERSION}

ARG PYTHON_VERSION

ENV DOCKER_BUILD_DIR /build
ENV BOOST_VERSION 1.78.0
ENV OPENUCX_VERSION 1.9.0
ENV OPENMPI_VERSION 4.0.7
ENV CONDA_VERSION 4.9.2
ENV PYTORCH_VERSION 1.12.1
ENV NCCL_VERSION 2.14.3-1

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y wget git libnuma-dev libibverbs-dev librdmacm-dev libssl-dev libffi-dev zlib1g-dev && apt-get clean

RUN mkdir -p $DOCKER_BUILD_DIR

# Boost
RUN cd ${DOCKER_BUILD_DIR} \
    && export BOOST_VERSION_FILE=${BOOST_VERSION//./_} \
    && wget --quiet https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_FILE}.tar.gz \
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
    && ./configure --with-cuda \
        --with-ucx=/usr/local/ucx --disable-dependency-tracking \
        --disable-getpwuid --without-usnic --enable-orterun-prefix-by-default \
    && make -j 4 && make install

# Python
RUN cd ${DOCKER_BUILD_DIR} \
    && wget --quiet https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure \
    && make \
    && make install

# PyTorch
RUN CUDA_VERSION_SHORT=${CUDA_VERSION//.} \
    && CUDA_VERSION_SHORT=${CUDA_VERSION_SHORT:0:3} \
    && pip3 install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}

# NCCL
RUN cd ${DOCKER_BUILD_DIR} \
    && git clone https://github.com/NVIDIA/nccl \
    && cd nccl \
    && git checkout v${NCCL_VERSION} \
    && make -j src.build \
    && make install

RUN ldconfig
