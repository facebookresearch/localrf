# Specify your base image here
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
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

## Basic conda for localTensoRF
RUN $CONDA_DIR/bin/python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# RUN $CONDA_DIR/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -c conda-forge
RUN $CONDA_DIR/bin/conda install ffmpeg
RUN $CONDA_DIR/bin/conda update ffmpeg

RUN $CONDA_DIR/bin/python -m ensurepip
RUN $CONDA_DIR/bin/python -m pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
RUN $CONDA_DIR/bin/python -m pip install imageio easydict matplotlib scipy kornia plyfile joblib
RUN $CONDA_DIR/bin/python -m pip install timm

## FuSta
RUN $CONDA_DIR/bin/python -m pip install cffi olefile pycparser
RUN $CONDA_DIR/bin/conda install -c conda-forge cupy cudatoolkit=11.6

## COLMAP
# Prepare and empty machine for building
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
RUN cd /home/${USERNAME} && git clone https://github.com/ceres-solver/ceres-solver.git --branch ${CERES_SOLVER_VERSION}
RUN cd /home/${USERNAME}/ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	sudo make install

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.7.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN cd /home/${USERNAME} && git clone https://github.com/colmap/colmap.git #--branch 3.7

RUN cd /home/${USERNAME}/colmap && \
	git checkout dev && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j4 && \
	sudo make install

## FuSta
RUN $CONDA_DIR/bin/python -m pip install certifi Pillow scikit-video six

# ## SCNeRF
# RUN $CONDA_DIR/bin/python -m pip install piqa pyquaternion wandb pytorch-lightning

# ## NeRF++
# RUN $CONDA_DIR/bin/conda install _libgcc_mutex ca-certificates certifi ld_impl_linux-64 libedit libffi libgcc-ng libstdcxx-ng ncurses openssl pip python readline setuptools sqlite tk wheel xz zlib
# RUN $CONDA_DIR/bin/python -m pip install absl-py astunparse backcall cachetools chardet configargparse cycler decorator future gast google-auth google-auth-oauthlib google-pasta grpcio h5py idna imageio importlib-metadata ipython ipython-genutils jedi keras-preprocessing kiwisolver lpips markdown networkx oauthlib opt-einsum parso pexpect pickleshare pillow prompt-toolkit protobuf ptyprocess pyasn1 pyasn1-modules pygments pymcubes pyparsing python-dateutil pywavelets requests requests-oauthlib rsa scikit-image scipy tensorboard-plugin-wit tensorboardx tensorflow six termcolor tifffile tqdm traitlets trimesh urllib3 wcwidth werkzeug wrapt zipp
# RUN $CONDA_DIR/bin/python -m pip install open3d


# # ## Instant-NGP
# RUN sudo apt-get install -y libopenexr-dev libxi-dev libglfw3-dev libomp-dev libxinerama-dev libxcursor-dev
# RUN $CONDA_DIR/bin/python -m pip install pybind11 commentjson

# # Image preprocessing
# RUN sudo apt-get install -y imagemagick
# RUN sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5

# # BaRF
# RUN $CONDA_DIR/bin/conda install ipdb visdom scikit-video pyyaml gdown -c pytorch -c conda-forge

# ## Fix
# RUN $CONDA_DIR/bin/conda install -c conda-forge gcc=12.1.0