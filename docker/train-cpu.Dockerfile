# Training and evaluation on the CPU; also the client for training on Modal.
# The repository is mounted at /work at run time (scripts/train.sh), nothing is copied in.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# runs as the calling user (files in the mounted repo stay theirs), who has no home here
ENV HOME=/tmp/home USER=whoosh YOLO_CONFIG_DIR=/tmp/home/ultralytics MPLCONFIGDIR=/tmp/home/mpl PYTHONUNBUFFERED=1
RUN mkdir -p /tmp/home && chmod 1777 /tmp/home
WORKDIR /work
