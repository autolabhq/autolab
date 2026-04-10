#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG_DIR:-/logs/verifier}"
APP_DIR="${APP_DIR:-/app}"
mkdir -p "$LOG"

write_reward() {
    local reward="$1" metric="$2" correctness="$3"
    LOG_PATH="$LOG" python3 - "$reward" "$metric" "$correctness" <<'PY'
import json
import os
import sys

reward = float(sys.argv[1])
metric = None if sys.argv[2] == "null" else int(sys.argv[2])
correct = sys.argv[3] == "true"
log_path = os.environ["LOG_PATH"]

result = {
    "reward": round(reward, 4),
    "correctness": correct,
    "metric": metric,
}

with open(f"{log_path}/reward.json", "w") as f:
    json.dump(result, f, indent=2)
with open(f"{log_path}/reward.txt", "w") as f:
    f.write(str(round(reward, 4)))
PY
}

echo "=== Syntax ===" >&2
if ! python3 -m py_compile "$APP_DIR/solve.py" "$APP_DIR/main.py" >"$LOG/py_compile.txt" 2>&1; then
    cat "$LOG/py_compile.txt" >&2
    write_reward 0.0 null false
    exit 0
fi

echo "=== Correctness + metric ===" >&2
if ! RESULT=$(APP_DIR="$APP_DIR" python3 "$APP_DIR/main.py" --verify 2>&1); then
    echo "$RESULT" >"$LOG/verify.txt"
    cat "$LOG/verify.txt" >&2
    write_reward 0.0 null false
    exit 0
fi
echo "$RESULT" | tee "$LOG/verify.txt" >&2

if ! echo "$RESULT" | grep -q "result=ok"; then
    write_reward 0.0 null false
    exit 0
fi

COUNT=$(echo "$RESULT" | grep -oP 'comparators=\K[0-9]+')
if [[ -z "$COUNT" ]]; then
    echo "could not parse comparator count" >&2
    write_reward 0.0 null false
    exit 0
fi

BASELINE=80
BEST_KNOWN=60

REWARD=$(python3 - "$COUNT" "$BASELINE" "$BEST_KNOWN" <<'PY'
import sys

count = int(sys.argv[1])
baseline = int(sys.argv[2])
best_known = int(sys.argv[3])

if count > baseline:
    reward = 0.0
elif count <= best_known:
    reward = 1.0
else:
    reward = (baseline - count) / (baseline - best_known)

print(f"{reward:.4f}")
PY
)

echo "comparators=$COUNT reward=$REWARD" >&2
write_reward "$REWARD" "$COUNT" true
