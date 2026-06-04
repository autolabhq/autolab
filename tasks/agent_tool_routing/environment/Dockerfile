FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY retriever.py /app/retriever.py
COPY main.py      /app/main.py

RUN chmod +x /app/main.py && \
    mkdir -p /orig /logs/verifier && \
    sha256sum /app/main.py | cut -d' ' -f1 > /orig/main.py.sha256 && \
    chmod 0444 /orig/main.py.sha256 && \
    (chattr +i /orig/main.py.sha256 2>/dev/null || true)

RUN python3 -m py_compile /app/main.py /app/retriever.py && \
    python3 /app/main.py --verify | grep -E '^verify'

RUN echo "=== System Info ===" && \
    (which python3 && python3 --version || true) && \
    echo "=== End System Info ==="

CMD ["bash"]
