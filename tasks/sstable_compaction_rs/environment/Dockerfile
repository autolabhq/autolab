FROM rust:1.82-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux python3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml .
RUN mkdir -p /orig && sha256sum /app/Cargo.toml | cut -d' ' -f1 > /orig/build_file.sha256
RUN mkdir -p src && \
    printf 'fn main(){}' > src/main.rs && \
    cargo build --release && \
    rm -f target/release/solve

COPY src/ src/

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
