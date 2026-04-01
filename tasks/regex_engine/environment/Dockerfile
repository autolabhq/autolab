FROM rust:1.82-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tmux \
        python3 \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml /app/Cargo.toml
RUN mkdir -p /orig && sha256sum /app/Cargo.toml | cut -d' ' -f1 > /orig/build_file.sha256
COPY Cargo.lock /app/Cargo.lock
COPY src/       /app/src/

WORKDIR /app

RUN cargo build --release && ./target/release/solve --verify | grep -E '^verify'

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
