FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        build-essential make python3 tmux bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY solve.h main.cu solve.cu Makefile /app/

RUN mkdir -p /orig && \
    sha256sum /app/Makefile | cut -d' ' -f1 > /orig/makefile.sha256 && \
    sha256sum /app/main.cu  | cut -d' ' -f1 > /orig/main_cu.sha256 && \
    sha256sum /app/solve.h  | cut -d' ' -f1 > /orig/solve_h.sha256

RUN make

RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which nvcc && nvcc --version || true) && \
    (nvidia-smi || true) && \
    (which python3 && python3 --version || true) && \
    echo "=== End System Info ==="

CMD ["bash"]
