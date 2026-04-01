FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git tmux curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch (2.6.0 for vllm 0.8.5 compatibility)
RUN uv pip install --system setuptools==82.0.1
RUN uv pip install --system \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install LLaMA-Factory with pinned deps
RUN uv pip install --system \
    "llamafactory[torch,metrics]==0.9.3" \
    peft==0.15.2 \
    trl==0.9.6 \
    transformers==4.52.4 \
    datasets==3.6.0 \
    accelerate==1.7.0 \
    bitsandbytes==0.45.5 \
    sentencepiece==0.2.0 \
    protobuf==5.29.5 \
    pandas

# Install vLLM for fast inference + lm-evaluation-harness for IFEval
RUN uv pip install --system "vllm==0.8.5.post1"
RUN uv pip install --system "lm-eval[vllm]==0.4.10" langdetect immutabledict
# Pin transformers back (vllm may upgrade it beyond llamafactory's constraint)
RUN uv pip install --system "transformers==4.52.4"
ENV DISABLE_VERSION_CHECK=1

# Pre-cache IFEval dataset and nltk data for offline evaluation
RUN python3 -c "\
from datasets import load_dataset; \
ds = load_dataset('google/IFEval', trust_remote_code=True); \
print(f'IFEval cached: {ds}'); \
"
RUN python3 -c "import nltk; nltk.download('punkt_tab')"

# Download Qwen2.5-3B-Instruct (~6GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='/models/Qwen2.5-3B-Instruct')"

RUN mkdir -p /app /logs/verifier /orig /data /app/data /app/output

# Prepare data pool at build time
COPY prepare_data.py /tmp/prepare_data.py
RUN python3 /tmp/prepare_data.py && rm /tmp/prepare_data.py

# Copy scripts
COPY select_data.py /app/select_data.py
COPY convert_to_sharegpt.py /app/convert_to_sharegpt.py
COPY train_config.yaml /app/train_config.yaml
COPY train.sh /app/train.sh
COPY evaluate_local.py /app/evaluate_local.py
RUN chmod +x /app/train.sh

# Save originals for tamper detection
RUN mkdir -p /orig && \
    cp /app/select_data.py /orig/select_data.py && \
    cp /app/train_config.yaml /orig/train_config.yaml && \
    cp /app/convert_to_sharegpt.py /orig/convert_to_sharegpt.py && \
    cp /app/train.sh /orig/train.sh && \
    sha256sum /app/train_config.yaml > /orig/train_config.yaml.sha256 && \
    sha256sum /app/convert_to_sharegpt.py > /orig/convert_to_sharegpt.py.sha256 && \
    sha256sum /app/train.sh > /orig/train.sh.sha256

# Force offline mode at runtime (all downloads happen during build)
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app
CMD ["bash"]
