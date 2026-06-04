FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash=5.1-6ubuntu1.1 \
    build-essential=12.9ubuntu3 \
    make=4.3-4.1build1 \
    python3=3.10.6-1~22.04.1 \
    tmux=3.2a-4ubuntu0.2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile solve.h main.cu solve.cu /app/

RUN mkdir -p /orig /logs/verifier && \
    sha256sum /app/Makefile /app/main.cu /app/solve.h > /orig/protected.sha256

RUN make

RUN echo "=== System Info ===" && \
    (which nvcc && nvcc --version || true) && \
    (nvidia-smi || true) && \
    (which g++ && g++ --version | head -1 || true) && \
    (which python3 && python3 --version || true) && \
    echo "=== End System Info ==="

CMD ["bash"]
