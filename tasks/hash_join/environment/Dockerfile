FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y tmux \
        gcc \
        make \
        python3 \
        linux-tools-generic \
        binutils \
        time \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

# ─── Copy task files ──────────────────────────────────────────────────────────
COPY solve.c     /app/solve.c
COPY solve.h     /app/solve.h
COPY main.c      /app/main.c
COPY Makefile    /app/Makefile
RUN mkdir -p /orig && sha256sum /app/Makefile | cut -d' ' -f1 > /orig/build_file.sha256
WORKDIR /app

# ─── Pre-build baseline binary and store benchmark reference ─────────────────
RUN make
RUN mkdir -p /tests && \
    BENCH_OUT=$(/app/join 2>/dev/null) && \
    echo "$BENCH_OUT" | grep -oP 'matches=\K[0-9]+'  > /tests/benchmark_matches.txt && \
    echo "$BENCH_OUT" | grep -oP 'checksum=\K[0-9]+' > /tests/benchmark_checksum.txt && \
    echo "Reference: matches=$(cat /tests/benchmark_matches.txt) checksum=$(cat /tests/benchmark_checksum.txt)"

# ─── Log directory for Harbor verifier ────────────────────────────────────────
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
