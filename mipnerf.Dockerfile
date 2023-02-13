ARG IMAGE_NAME
# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 as base
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 as base
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 curl 
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv

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

# Install mipnerf dependencies
RUN /opt/conda/bin/conda init bash \
    && /opt/conda/bin/conda install \
    && /opt/conda/bin/conda install pip; /opt/conda/bin/python -m pip install --upgrade pip
RUN /opt/conda/bin/python -m pip install --upgrade jax jaxlib==0.3.20+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN /opt/conda/bin/python -m pip install \ 
# "jax==0.2.12" \
# "jaxlib>=0.1.65" \
# "flax>=0.2.2" \
# "opencv-python>=4.4.0" \
# "Pillow>=7.2.0" \
# "tensorboard>=2.4.0" \
# "tensorflow>=2.3.1" \
# "gin-config" \
# "joblib"
RUN /opt/conda/bin/python -m pip install flax opencv-python Pillow tensorboard tensorflow gin-config 
RUN /opt/conda/bin/python -m pip install dm_pix rawpy mediapy
RUN /opt/conda/bin/python -m pip install --upgrade jax jaxlib==0.3.20+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN /opt/conda/bin/python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
