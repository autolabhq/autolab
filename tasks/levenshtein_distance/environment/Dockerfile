FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        tmux \
        gcc \
        make \
        python3 \
        linux-tools-generic \
        binutils \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

COPY solve.c   /app/solve.c
COPY solve.h   /app/solve.h
COPY main.c    /app/main.c
COPY Makefile  /app/Makefile
RUN mkdir -p /orig && \
    sha256sum /app/Makefile | cut -d' ' -f1 > /orig/Makefile.sha256 && \
    sha256sum /app/main.c | cut -d' ' -f1 > /orig/main.c.sha256 && \
    sha256sum /app/solve.h | cut -d' ' -f1 > /orig/solve.h.sha256

WORKDIR /app

RUN make && echo "Build OK"

RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which gcc && gcc --version | head -1 || true) && \
    (which python3 && python3 --version || true) && \
    echo "=== End System Info ==="

CMD ["bash"]
