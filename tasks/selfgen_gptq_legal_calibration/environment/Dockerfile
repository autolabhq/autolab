FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir numpy==2.1.3
RUN python3 -m pip install --no-cache-dir \
    --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1

COPY . /app

RUN mkdir -p /orig && \
    cp /app/run_quant.py /orig/run_quant.py && \
    cp /app/calibrate.py /orig/calibrate.py && \
    cp /app/quantize.py /orig/quantize.py && \
    cp /app/benchmark.py /orig/benchmark.py

CMD ["bash"]
