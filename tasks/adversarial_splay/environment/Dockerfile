FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY main.py /app/main.py
COPY accesses.txt /app/accesses.txt

# Integrity: hash main.py at build time. The verifier rejects any run that
# tampers with the splay-tree implementation.
RUN mkdir -p /orig && sha256sum /app/main.py > /orig/protected_files.sha256

RUN mkdir -p /logs/verifier

RUN echo "=== System Info ===" && \
    (which python3 && python3 --version || true) && \
    echo "=== End System Info ==="
CMD ["bash"]
