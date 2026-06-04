FROM rust:1.89-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        bash \
        tmux \
        python3 \
        coreutils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
WORKDIR /app

COPY Cargo.toml /app/Cargo.toml
COPY Cargo.lock /app/Cargo.lock
COPY main.rs    /app/main.rs
COPY solve.rs   /app/solve.rs

ARG BUILD_SEED_RAW=""

ENV CARGO_NET_OFFLINE=true

RUN set -e; \
    seed_hex="${BUILD_SEED_RAW}"; \
    if [ -z "$seed_hex" ]; then \
        seed_hex=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n'); \
    fi; \
    seed_hex=$(echo "$seed_hex" | tr -cd '0-9a-fA-F' | head -c 16); \
    if [ "${#seed_hex}" -lt 16 ]; then \
        seed_hex=$(printf '%s%s' "$seed_hex" "$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')" | head -c 16); \
    fi; \
    echo "$seed_hex" > /app/.task_seed; \
    export TASK_SEED="$seed_hex"; \
    cargo build --release --offline && \
    /app/target/release/range_count_task --verify

RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which rustc && rustc --version || true) && \
    (which cargo && cargo --version || true) && \
    (which python3 && python3 --version || true) && \
    (uname -a || true) && \
    echo "=== End System Info ==="

CMD ["bash"]
