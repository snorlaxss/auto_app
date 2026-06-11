FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ninja-build \
    build-essential \
    openexr \
    libopenexr-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn\n" > /root/.pip/pip.conf

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY FlowPose/ ./FlowPose/

WORKDIR /workspace/FlowPose/networks/pts_encoder/pointnet2_utils/pointnet2

ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN python3 setup.py install

WORKDIR /workspace/FlowPose

EXPOSE 6666
CMD ["uvicorn", "server:workspace", "--host", "0.0.0.0", "--port", "6666"]