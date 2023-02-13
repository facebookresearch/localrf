# Specify your base image here
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive
# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv

# Install miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf /tmp/*

# Create the user
# # RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
# #     chown $USERNAME /opt/conda -R && \
# #     adduser $USERNAME sudo && \
# #     echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # USER $USERNAME
# RUN useradd --create-home -s /bin/bash --no-user-group -u $USER_ID docker
# # RUN addgroup --gid $GROUP_ID user 
# RUN adduser docker sudo 
# RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN addgroup --gid $GROUP_ID user
RUN useradd --create-home -s /bin/bash --uid $USER_ID --gid $GROUP_ID docker
RUN adduser docker sudo
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN chown docker /opt/conda -R 
# RUN addgroup --gid $GROUP_ID user && \
#     adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user  && \
#     chown $USER_ID:$GROUP_ID /opt/conda -R
# RUN adduser --uid $USER_ID:$GROUP_ID sudo 
USER docker

RUN /opt/conda/bin/conda install ffmpeg
RUN /opt/conda/bin/conda update ffmpeg

## Basic conda for localTensoRF
RUN /opt/conda/bin/python -m ensurepip
RUN /opt/conda/bin/python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
# RUN /opt/conda/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -c conda-forge

RUN /opt/conda/bin/python -m pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
RUN /opt/conda/bin/python -m pip install imageio easydict matplotlib scipy kornia plyfile joblib
RUN /opt/conda/bin/python -m pip install timm

# ## FuSta
# RUN /opt/conda/bin/python -m pip install cffi olefile pycparser
# RUN /opt/conda/bin/conda install -c conda-forge cupy cudatoolkit=11.6

## COLMAP
# Prepare and empty machine for building
RUN sudo apt-get update 
RUN sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# Build and install ceres solver
RUN sudo apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
ARG CERES_SOLVER_VERSION=2.1.0
RUN cd /home/docker && git clone https://github.com/ceres-solver/ceres-solver.git --branch ${CERES_SOLVER_VERSION}
RUN cd /home/docker/ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j16 && \
	sudo make install

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.7.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN cd /home/docker && git clone https://github.com/colmap/colmap.git #--branch 3.7

RUN cd /home/docker/colmap && \
	git checkout dev && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j16 && \
	sudo make install

# # FuSta
# RUN /opt/conda/bin/python -m pip install certifi Pillow scikit-video six

# ## SCNeRF
# RUN /opt/conda/bin/python -m pip install piqa pyquaternion wandb pytorch-lightning

# ## NeRF++
# RUN /opt/conda/bin/conda install _libgcc_mutex ca-certificates certifi ld_impl_linux-64 libedit libffi libgcc-ng libstdcxx-ng ncurses openssl pip python readline setuptools sqlite tk wheel xz zlib
# RUN /opt/conda/bin/python -m pip install absl-py astunparse backcall cachetools chardet configargparse cycler decorator future gast google-auth google-auth-oauthlib google-pasta grpcio h5py idna imageio importlib-metadata ipython ipython-genutils jedi keras-preprocessing kiwisolver lpips markdown networkx oauthlib opt-einsum parso pexpect pickleshare pillow prompt-toolkit protobuf ptyprocess pyasn1 pyasn1-modules pygments pymcubes pyparsing python-dateutil pywavelets requests requests-oauthlib rsa scikit-image scipy tensorboard-plugin-wit tensorboardx tensorflow six termcolor tifffile tqdm traitlets trimesh urllib3 wcwidth werkzeug wrapt zipp
# RUN /opt/conda/bin/python -m pip install open3d


# # ## Instant-NGP
# RUN sudo apt-get install -y libopenexr-dev libxi-dev libglfw3-dev libomp-dev libxinerama-dev libxcursor-dev
# RUN /opt/conda/bin/python -m pip install pybind11 commentjson

# Image preprocessing
RUN sudo apt-get install -y imagemagick
RUN sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5

## BaRF
RUN /opt/conda/bin/conda install ipdb visdom scikit-video trimesh pyyaml gdown termcolor -c pytorch -c conda-forge
RUN /opt/conda/bin/python -m pip install pymcubes 

## Nerfacto
# RUN /opt/conda/bin/python -m pip install open3d nerfacc pymeshlab aiortc 
# RUN /opt/conda/bin/python -m pip install pyngrok torchmetrics[image] nuscenes-devkit wandb  jupyterlab aiohttp functorch msgpack  h5py Shapely docstring-parser mypy
# RUN /opt/conda/bin/python -m pip install jsonpatch xatlas 
# # RUN /opt/conda/bin/python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# # RUN cd /home/docker && git clone https://github.com/nerfstudio-project/nerfstudio.git && \
# #     cd /home/docker/nerfstudio && \
# #     /opt/conda/bin/python -m pip install --upgrade pip setuptools && \
# #     /opt/conda/bin/python -m pip install -e . 

# ## Fix
# RUN /opt/conda/bin/conda install -c conda-forge gcc=12.1.0
# RUN /opt/conda/bin/conda install -c conda-forge gcc=12.1.0
RUN /opt/conda/bin/python -m pip install scipy==1.9.1 
