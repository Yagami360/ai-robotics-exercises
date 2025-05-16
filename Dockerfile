FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# install basic libs
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt install -y wget curl && \
    apt install --no-install-recommends -y \
    sudo git make cmake build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    python3-dev \
    pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libx11-6 \
    ca-certificates \
    jq

# define ARG and ENV
ARG WORKDIR="/app"
ENV WORKDIR=${WORKDIR}

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/usr/local/miniconda3/bin:${PATH}"
RUN conda init bash

RUN conda create -y -n lerobot python=3.10
RUN echo "source activate lerobot" > ~/.bashrc
ENV PATH="/usr/local/miniconda3/envs/lerobot/bin:${PATH}"

# Install OpenCV
RUN apt install --no-install-recommends -y libgl1-mesa-dev libglib2.0-0

# Install LeRobot
RUN git clone https://github.com/huggingface/lerobot.git
RUN cd lerobot && pip install -e .
RUN cd lerobot && pip install -e ".[pusht]"
RUN cd lerobot && pip install -e ".[aloha]"
RUN cd lerobot && pip install -e ".[xarm]"
RUN cd lerobot && pip install -e ".[pi0]"

# Install other packages
RUN pip install pytest

# Install Rendering libraries in simulator
ENV MUJOCO_GL=egl
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev

# COPY scripts
COPY . ${WORKDIR}

# set workdir
WORKDIR ${WORKDIR}

# expose wandb connect port
EXPOSE 443

# config & cleanup
RUN apt purge -y build-essential

RUN ldconfig && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
