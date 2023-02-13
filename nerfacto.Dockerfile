# Specify your base image here
FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000
ARG DEBIAN_FRONTEND=noninteractive
# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME

# Nerfacto
RUN cd /home/${USERNAME} && git clone https://github.com/nerfstudio-project/nerfstudio.git && \
    cd /home/${USERNAME}/nerfstudio && \
    $CONDA_DIR/bin/python -m pip install --upgrade pip setuptools && \
    $CONDA_DIR/bin/python -m pip install -e . 