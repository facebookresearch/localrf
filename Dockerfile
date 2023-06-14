FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv

# conda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf /tmp/*

# Create the user
RUN addgroup --gid $GROUP_ID user
RUN useradd --create-home -s /bin/bash --uid $USER_ID --gid $GROUP_ID docker
RUN adduser docker sudo
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN chown docker /opt/conda -R 
USER docker

# Setup localrf
RUN /opt/conda/bin/python -m ensurepip
RUN /opt/conda/bin/python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN /opt/conda/bin/python -m pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict matplotlib scipy==1.6.1 kornia plyfile joblib timm