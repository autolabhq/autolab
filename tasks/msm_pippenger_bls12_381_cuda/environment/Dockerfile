FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    make \
    python3 \
    tmux \
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
