FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup

## Go
RUN export VERSION=1.14.12 OS=linux ARCH=amd64 \
    && wget --quiet https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz  \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz \
    && rm go$VERSION.$OS-$ARCH.tar.gz

# Singularity
ENV PATH /usr/local/go/bin:$PATH
RUN export VERSION=3.7.0 \
    && wget --quiet https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz \
    && tar -xzf singularity-${VERSION}.tar.gz \
    && cd singularity \
    && ./mconfig \
    && make -C builddir \
    && make -C builddir install

ENTRYPOINT ["singularity"]
