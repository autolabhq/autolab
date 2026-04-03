FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin versions for reproducibility
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    triton==3.0.0 \
    numpy \
    pyyaml \
    --index-url https://download.pytorch.org/whl/cu124

# Install DeepGEMM (baseline reference — agents must NOT import this in solve.py)
RUN git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM /opt/deep_gemm && \
    cd /opt/deep_gemm && pip3 install --no-cache-dir -e . && \
    echo "DEEP_GEMM_PATH=/opt/deep_gemm" >> /etc/environment

ENV DEEP_GEMM_PATH=/opt/deep_gemm

WORKDIR /app
COPY environment/ /app/

# Pre-warm Triton JIT cache directory
RUN mkdir -p /root/.triton/cache

# Log output directory
RUN mkdir -p /logs/verifier
