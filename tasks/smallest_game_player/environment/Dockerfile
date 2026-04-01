FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy==1.26.4" \
    --index-url https://pypi.org/simple && \
    pip install --no-cache-dir "torch==2.2.2" \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY generate_data.py /tmp/generate_data.py
RUN python3 /tmp/generate_data.py && rm /tmp/generate_data.py

COPY solve.py /app/solve.py

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
