# Training on a local NVIDIA GPU (CUDA 12.4). Needs the NVIDIA driver on the host
# and the NVIDIA Container Toolkit; run with --gpus all (scripts/train.sh gpu does).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch, torchvision; assert torch.__version__.startswith('2.5.1'), torch.__version__"
ENV HOME=/tmp/home USER=whoosh YOLO_CONFIG_DIR=/tmp/home/ultralytics MPLCONFIGDIR=/tmp/home/mpl PYTHONUNBUFFERED=1
RUN mkdir -p /tmp/home && chmod 1777 /tmp/home
WORKDIR /work
