ARG OS_NAME=ubuntu
ARG OS_VERSION
ARG CUDA_VERSION
ARG PYTHON_VERSION
FROM rannc_base_cuda${CUDA_VERSION}-${OS_NAME}${OS_VERSION}-py${PYTHON_VERSION}:latest

ARG PKG_FILE

# apex
RUN cd ${DOCKER_BUILD_DIR} \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rannc \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 \
    && export CUDA_HOME="$(dirname $(which nvcc))/../" \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# RaNNC
COPY ./${PKG_FILE} ${DOCKER_BUILD_DIR}
RUN cd ${DOCKER_BUILD_DIR} \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rannc \
    && pip install ${DOCKER_BUILD_DIR}/${PKG_FILE}

# pytest
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rannc \
    && pip install pytest
