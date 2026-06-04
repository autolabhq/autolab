FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux bash curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip list

WORKDIR /app
COPY model.py train_model.py main.py solve.py /app/

# Train the frozen CIFAR-10 model (deterministic, ~3-5 min on CPU)
RUN python3 /app/train_model.py

RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which python3 && python3 --version || true) && \
    (pip list 2>/dev/null || true) && \
    ls -la /app /data/cifar10 || true && \
    echo "=== End System Info ==="

CMD ["bash"]
