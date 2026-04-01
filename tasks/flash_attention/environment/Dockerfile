FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# gcc with AVX2 support + python3-numpy for the correctness verifier
RUN apt-get update && apt-get install -y tmux \
        gcc \
        make \
        python3 \
        python3-numpy \
        linux-tools-generic \
        binutils \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

COPY solve.c     /app/solve.c
COPY solve.h     /app/solve.h
COPY main.c      /app/main.c
COPY Makefile    /app/Makefile
RUN mkdir -p /orig && sha256sum /app/Makefile | cut -d' ' -f1 > /orig/build_file.sha256
WORKDIR /app

# Pre-build, sanity-check, and store reference benchmark checksum
RUN make && ./attn --verify | grep -E '^verify'
RUN mkdir -p /tests && \
    /app/attn 2>/dev/null | sed 's/.*checksum=\([0-9.eE+\-]*\).*/\1/' > /tests/benchmark_checksum.txt && \
    echo "Reference benchmark checksum: $(cat /tests/benchmark_checksum.txt)"

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
