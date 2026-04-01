FROM rust:1.82-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux python3 \
    && rm -rf /var/lib/apt/lists/*

# Harbor commands commonly run under `bash -lc`, whose login-shell PATH on this
# base image drops `/usr/local/cargo/bin`. Mirror the Rust toolchain into
# `/usr/local/bin` so cargo remains available during verification.
RUN for tool in /usr/local/cargo/bin/*; do \
        ln -sf "$tool" "/usr/local/bin/$(basename "$tool")"; \
    done

WORKDIR /app

# ── Pre-fetch and compile Cargo dependencies ──────────────────────────────────
# This layer is cached: only re-runs if Cargo.toml changes.
# The actual agent code (src/) is compiled later (fast incremental rebuild).
COPY Cargo.toml .
RUN mkdir -p /orig && sha256sum /app/Cargo.toml | cut -d' ' -f1 > /orig/build_file.sha256
RUN mkdir -p src && \
    printf 'mod solve;\nfn main(){}' > src/main.rs && \
    printf 'pub fn fft(_: &[f64]) -> Vec<[f64; 2]> { vec![] }' > src/solve.rs && \
    cargo build --release && \
    rm -f target/release/deps/fft_rust* target/release/fft-rust*

# ── Copy task sources ─────────────────────────────────────────────────────────
COPY src/ src/

# Ensure reward output directory exists
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
