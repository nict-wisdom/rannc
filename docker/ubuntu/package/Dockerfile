ARG OS_NAME=ubuntu
ARG OS_VERSION
ARG CUDA_VERSION
ARG PYTHON_VERSION
FROM rannc_base_cuda${CUDA_VERSION}-${OS_NAME}${OS_VERSION}-py${PYTHON_VERSION}:latest

ENV CMAKE_VERSION 3.19.1

# CMake
RUN cd ${DOCKER_BUILD_DIR} \
    && wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
    && tar -xzf cmake-${CMAKE_VERSION}.tar.gz \
    && cd cmake-${CMAKE_VERSION} \
    && ./configure \
    && make -j 4 \
    && make install

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rannc \
    && pip install bdist_wheel_name
