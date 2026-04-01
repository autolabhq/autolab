FROM golang:1.22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY go.mod .
RUN mkdir -p /orig && sha256sum /app/go.mod | cut -d' ' -f1 > /orig/build_file.sha256
RUN go mod download

COPY *.go .
COPY Makefile .

RUN go build -o /app/solve .
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
