# Specify your base image here
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
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

## meganerf stuff
RUN $CONDA_DIR/bin/conda install -c pytorch -c conda-forge -c default \
    _libgcc_mutex=0.1=main \
    _openmp_mutex=4.5=1_gnu \
    absl-py=1.0.0=pyhd8ed1ab_0 \
    aiohttp=3.7.4.post0=py39h3811e60_0 \
    async-timeout=3.0.1=py_1000 \
    attrs=21.2.0=pyhd8ed1ab_0 \
    blas=1.0=mkl \
    blinker=1.4=py_1 \
    brotlipy=0.7.0=py39h3811e60_1001 \
    bzip2=1.0.8=h7b6447c_0 \
    c-ares=1.17.1=h27cfd23_0 \
    ca-certificates=2021.10.8=ha878542_0 \
    cachetools=4.2.4=pyhd8ed1ab_0 \
    certifi=2021.10.8=py39hf3d152e_1 \
    cffi=1.14.6=py39h400218f_0 \
    chardet=4.0.0=py39hf3d152e_2 \
    charset-normalizer=2.0.8=pyhd8ed1ab_0 \
    click=8.0.3=py39hf3d152e_1 \
    colorama=0.4.4=pyh9f0ad1d_0 \
    cryptography=3.4.8=py39hbca0aa6_0 \
    cudatoolkit=11.3.1=h2bc3f7f_2 \
    dataclasses=0.8=pyhc8e2a94_3 \
    ffmpeg=4.3=hf484d3e_0 \
    freetype=2.11.0=h70c0345_0 \
    giflib=5.2.1=h7b6447c_0 \
    gmp=6.2.1=h2531618_2 \
    gnutls=3.6.15=he1e5248_0 \
    google-auth=2.3.3=pyh6c4a22f_0 \
    google-auth-oauthlib=0.4.6=pyhd8ed1ab_0 \
    grpcio=1.42.0=py39hce63b2e_0 \
    idna=3.1=pyhd3deb0d_0 \
    importlib-metadata=4.8.2=py39hf3d152e_0 \
    intel-openmp=2021.4.0=h06a4308_3561 \
    jpeg=9d=h7f8727e_0 \
    lame=3.100=h7b6447c_0 \
    lcms2=2.12=h3be6417_0 \
    ld_impl_linux-64=2.35.1=h7274673_9 \
    libffi=3.3=he6710b0_2 \
    libgcc-ng=9.3.0=h5101ec6_17 \
    libgomp=9.3.0=h5101ec6_17 \
    libiconv=1.15=h63c8f33_5 \
    libidn2=2.3.2=h7f8727e_0 \
    libpng=1.6.37=hbc83047_0 \
    libprotobuf=3.15.8=h780b84a_0 \
    libstdcxx-ng=9.3.0=hd4cf53a_17 \
    libtasn1=4.16.0=h27cfd23_0 \
    libtiff=4.2.0=h85742a9_0 \
    libunistring=0.9.10=h27cfd23_0 \
    libuv=1.40.0=h7b6447c_0 \
    libwebp=1.2.0=h89dd481_0 \
    libwebp-base=1.2.0=h27cfd23_0 \
    lz4-c=1.9.3=h295c915_1 \
    markdown=3.3.6=pyhd8ed1ab_0 \
    mkl=2021.4.0=h06a4308_640 \
    mkl-service=2.4.0=py39h7f8727e_0 \
    mkl_fft=1.3.1=py39hd3c417c_0 \
    mkl_random=1.2.2=py39h51133e4_0 \
    multidict=5.1.0=py39h27cfd23_2 \
    ncurses=6.3=h7f8727e_2 \
    nettle=3.7.3=hbbd107a_1 \
    npy-append-array=0.9.13=pyhd8ed1ab_0 \
    numpy=1.21.2=py39h20f2e39_0 \
    numpy-base=1.21.2=py39h79a1101_0 \
    oauthlib=3.1.1=pyhd8ed1ab_0 \
    olefile=0.46=pyhd3eb1b0_0 \
    openh264=2.1.0=hd408876_0 \
    openssl=1.1.1l=h7f8727e_0 \
    pillow=8.4.0=py39h5aabda8_0 \
    pip=21.2.4=py39h06a4308_0 \
    protobuf=3.15.8=py39he80948d_0 \
    pyasn1=0.4.8=py_0 \
    pyasn1-modules=0.2.7=py_0 \
    pycparser=2.21=pyhd8ed1ab_0 \
    pyjwt=2.3.0=pyhd8ed1ab_0 \
    pyopenssl=21.0.0=pyhd8ed1ab_0 \
    pysocks=1.7.1=py39hf3d152e_4 \
    python=3.9.7=h12debd9_1 \
    python_abi=3.9=2_cp39 \
    pytorch=1.10.0=py3.9_cuda11.3_cudnn8.2.0_0 \
    pytorch-mutex=1.0=cuda \
    pyu2f=0.1.5=pyhd8ed1ab_0 \
    readline=8.1=h27cfd23_0 \
    requests=2.26.0=pyhd8ed1ab_1 \
    requests-oauthlib=1.3.0=pyh9f0ad1d_0 \
    rsa=4.8=pyhd8ed1ab_0 \
    setuptools=58.0.4=py39h06a4308_0 \
    six=1.16.0=pyhd3eb1b0_0 \
    sqlite=3.36.0=hc218d9a_0 \
    tensorboard=2.7.0=pyhd8ed1ab_0 \
    tensorboard-data-server=0.6.0=py39h3da14fd_0 \
    tensorboard-plugin-wit=1.8.0=pyh44b312d_0 \
    tk=8.6.11=h1ccaba5_0 \
    torchvision=0.11.1=py39_cu113 \
    tqdm=4.62.3=pyhd8ed1ab_0 \
    typing-extensions=3.10.0.2=hd3eb1b0_0 \
    typing_extensions=3.10.0.2=pyh06a4308_0 \
    tzdata=2021e=hda174b7_0 \
    urllib3=1.26.7=pyhd8ed1ab_0 \
    werkzeug=2.0.1=pyhd8ed1ab_0 \
    wheel=0.37.0=pyhd3eb1b0_1 \
    xz=5.2.5=h7b6447c_0 \
    yarl=1.6.3=py39h3811e60_2 \
    zipp=3.6.0=pyhd8ed1ab_0 \
    zlib=1.2.11=h7b6447c_3 \
    zstd=1.4.9=haebb681_0

RUN $CONDA_DIR/bin/python -m pip install \
    configargparse==1.5.3 \
    lpips==0.1.4 \
    opencv-python-headless==4.5.4.60 \
    pandas==1.4.2 \
    parscript==0.0.2 \
    portalocker==2.3.2 \
    pyarrow==8.0.0 \
    python-dateutil==2.8.2 \
    pytz==2022.1 \
    pyyaml==6.0 \
    scipy==1.7.3