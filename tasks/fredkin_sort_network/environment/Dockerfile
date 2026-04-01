FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY main.py /app/main.py
COPY circuit.txt /app/circuit.txt

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
