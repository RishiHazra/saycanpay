FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
FROM continuumio/miniconda3
# Add Conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"
# Install Python 3.8 using Conda
RUN conda install -y python=3.8 && \
    conda clean -a
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get update \
    && \
    echo "------------------------------------------------------ essentials" \
    && \
    apt-get install -y --no-install-recommends -y \
    build-essential \
    apt-utils \
    python3-setuptools \
    git-all \
    && \
    echo "------------------------------------------------------ editors" \
    && \
    apt-get install -y --no-install-recommends -y \
    emacs \
    vim \
    nano \
    && \
    echo "------------------------------------------------------ software" \
    && \
    apt-get install -y --no-install-recommends -y \
    python3-pip \
    tmux \
    screen \
    graphviz \
    cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig \
    sumo \
    sumo-tools \
    sumo-doc \
    && \
    echo "------------------------------------------------------ cleanup" \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install cudatoolkit -y
