FROM gaetanlandreau/pytorch3d:latest AS base

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    build-essential \
    byobu \
    ca-certificates \
    git-core git \
    htop \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcupti-dev \
    openssh-server \
    software-properties-common \
    vim \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip install tk
RUN pip install opencv-python
RUN pip install kornia==0.5.5
RUN pip install dominate==2.6.0
RUN pip install trimesh==3.9.20
RUN pip install onnxruntime-gpu==1.7.0
RUN pip install pyfakewebcam


COPY . app/

WORKDIR app/

CMD ["python", "main.py"]