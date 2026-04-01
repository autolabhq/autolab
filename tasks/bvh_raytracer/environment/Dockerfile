FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── Build tools + profilers ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends tmux \
        build-essential \
        g++ \
        make \
        python3 \
        linux-tools-generic \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

# ─── Copy task source ─────────────────────────────────────────────────────────
COPY main.cpp  /app/main.cpp
COPY solve.cpp /app/solve.cpp
COPY solve.h   /app/solve.h
COPY Makefile  /app/Makefile
RUN mkdir -p /orig && sha256sum /app/Makefile | cut -d' ' -f1 > /orig/build_file.sha256
WORKDIR /app

# ─── Pre-build baseline binary ────────────────────────────────────────────────
RUN make

# ─── Pre-compute reference checksum with the baseline implementation ──────────
# Stored in /app so it survives after tests/ is mounted at runtime.
RUN /app/solve --verify > /app/reference_checksum.txt && \
    echo "Reference checksum: $(cat /app/reference_checksum.txt)"

# ─── Logs directory expected by Harbor ────────────────────────────────────────
RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which gcc && gcc --version | head -1 || true) && \
    (which g++ && g++ --version | head -1 || true) && \
    (which rustc && rustc --version || true) && \
    (which go && go version || true) && \
    (which python3 && python3 --version || true) && \
    (pip list 2>/dev/null || pip3 list 2>/dev/null || true) && \
    echo "=== End System Info ==="
CMD ["bash"]
