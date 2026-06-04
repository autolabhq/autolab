FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy==1.26.4"

WORKDIR /app

COPY main.py /app/main.py
COPY predictor.py /app/predictor.py

# Generate visible data using a build-only copy of datagen.py — the generator
# is intentionally absent from /app at run time. The hash is recorded so the
# verifier can detect drift between this copy and tests/datagen.py.
COPY datagen.py /tmp/datagen.py
RUN mkdir -p /orig && \
    sha256sum /tmp/datagen.py | awk '{print $1}' > /orig/datagen.sha256 && \
    python3 /tmp/datagen.py --seed 2024 --output-dir /app/data/visible && \
    rm /tmp/datagen.py

# Tamper detection for the local evaluator.
RUN sha256sum /app/main.py > /orig/main.py.sha256

# Smoke test the baseline predictor.
RUN python3 /app/main.py

RUN mkdir -p /logs/verifier

CMD ["bash"]
