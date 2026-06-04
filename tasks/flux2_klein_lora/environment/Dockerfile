FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# ── System packages ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl wget tmux vim libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv (test.sh needs it, no network at runtime)
RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── PyTorch (CUDA 12.4) ──
RUN uv pip install --system \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Clone musubi-tuner (pinned commit for reproducibility) ──
# Upstream main as of 2026-04-20. Pinned so the image is deterministic and
# upstream breaking changes don't silently change the calibration.
ARG MUSUBI_COMMIT=6306e839608cd6525def3e6b0d8ec8cd17ff459e
RUN git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
    cd /opt/musubi-tuner && \
    git checkout "$MUSUBI_COMMIT"

RUN cd /opt/musubi-tuner && uv pip install --system .

# ── Evaluation dependencies ──
RUN uv pip install --system \
    open_clip_torch \
    timm \
    datasets \
    peft \
    safetensors \
    pytest \
    protobuf

# ── Download models ──
RUN mkdir -p /models/dit /models/vae /models/text_encoder

# DiT: FLUX.2 klein base 9B (~18GB)
RUN huggingface-cli download zhangchenxu/FLUX.2-klein-base-9B \
    flux-2-klein-base-9b.safetensors \
    --local-dir /models/dit --quiet && \
    mv /models/dit/flux-2-klein-base-9b.safetensors /models/dit/flux2-klein-base-9b.safetensors

# VAE: shared ae.safetensors (~0.3GB)
RUN huggingface-cli download zhangchenxu/FLUX.2-klein-base-9B \
    ae.safetensors \
    --local-dir /models/vae --quiet

# Text Encoder: Qwen3 8B split files (~16GB)
RUN huggingface-cli download zhangchenxu/FLUX.2-klein-base-9B \
    --include "text_encoder/*" \
    --local-dir /tmp/klein9b --quiet && \
    cp /tmp/klein9b/text_encoder/* /models/text_encoder/ && \
    rm -rf /tmp/klein9b

# Pre-cache Qwen3-8B tokenizer (used by musubi-tuner text encoder)
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')"

# ── Pre-cache evaluation models ──
# CLIP ViT-B/32
RUN python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"

# DINOv2 ViT-S/14
RUN python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

# ── Initialize accelerate config (single GPU, bf16) ──
RUN mkdir -p /root/.cache/huggingface/accelerate && \
    echo '{"compute_environment": "LOCAL_MACHINE", "distributed_type": "NO", "mixed_precision": "bf16", "num_processes": 1, "use_cpu": false}' \
    > /root/.cache/huggingface/accelerate/default_config.yaml

# Force offline mode at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ── Prepare training data (15 naruto images) ──
COPY prepare_data.py /tmp/prepare_data.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 /tmp/prepare_data.py && rm /tmp/prepare_data.py

# ── Copy app files ──
RUN mkdir -p /app /logs/verifier /app/output
WORKDIR /app

COPY dataset.toml /app/dataset.toml
COPY train.sh /app/train.sh
COPY evaluate.py /app/evaluate.py
COPY eval_prompts.txt /app/eval_prompts.txt
RUN chmod +x /app/train.sh

# ── Save originals for baseline comparison / tamper detection ──
RUN mkdir -p /orig && cp -r /app /orig/app

CMD ["bash"]
