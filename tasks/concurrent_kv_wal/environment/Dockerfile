FROM golang:1.22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends tmux \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY go.mod              /app/go.mod
RUN mkdir -p /orig && sha256sum /app/go.mod | cut -d' ' -f1 > /orig/build_file.sha256
COPY main.go             /app/main.go
COPY types.go            /app/types.go
COPY store.go            /app/store.go
COPY index.go            /app/index.go
COPY wal.go              /app/wal.go
COPY Makefile            /app/Makefile
COPY generate_workload.go /tmp/generate_workload.go

WORKDIR /app

RUN go build -o /app/solve .

RUN mkdir -p /tests && go run /tmp/generate_workload.go \
        -workers 4 \
        -load 3000 \
        -mixed 9000 \
        -query 4000 \
        -seed 99 \
        -output /tests/verify_input.kv

RUN go run /tmp/generate_workload.go \
        -workers 4 \
        -load 32000 \
        -mixed 96000 \
        -query 48000 \
        -seed 42 \
        -output /tests/benchmark_input.kv

RUN /app/solve -mode verify -input /tests/verify_input.kv > /tests/reference_output.txt

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
