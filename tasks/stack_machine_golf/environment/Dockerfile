FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        gcc \
        make \
        python3 \
        tmux \
    && rm -rf /var/lib/apt/lists/*

COPY main.c /app/main.c
COPY program.stk /app/program.stk
COPY Makefile /app/Makefile
RUN mkdir -p /orig && sha256sum /app/Makefile | cut -d' ' -f1 > /orig/build_file.sha256

WORKDIR /app

RUN make && ./stack_vm

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
