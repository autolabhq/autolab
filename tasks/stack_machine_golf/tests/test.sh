#!/usr/bin/env bash
set -euo pipefail

LOG=${LOG_DIR:-/logs/verifier}
if ! mkdir -p "$LOG" 2>/dev/null; then
    LOG="$(pwd)/.local_logs/verifier"
    mkdir -p "$LOG"
fi
mkdir -p "$LOG"

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/Makefile 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [ -n "$ORIG_HASH" ] && [ "$ORIG_HASH" != "$CURR_HASH" ]; then
    echo "Build file was modified — scoring 0" >&2
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > "$LOG/reward.json"
    echo "0.0" > "$LOG/reward.txt"
    exit 0
fi

APP_DIR=${APP_DIR:-/app}
if [[ ! -d "$APP_DIR" ]]; then
    APP_DIR=$(pwd)
fi
cd "$APP_DIR"
if ! make >"$LOG/build.txt" 2>&1; then
    python3 - <<PY
import json
with open("$LOG/reward.json", "w") as f:
    json.dump({"reward": 0.0, "reason": "build_failed"}, f)
PY
    exit 0
fi

# ── 1. Run with default seed (seed=0) for metric ────────────────────────────
OUT=$(./stack_vm 0)
echo "$OUT" > "$LOG/run.txt"

if ! echo "$OUT" | grep -q "verify=ok"; then
    python3 - <<PY
import json
with open("$LOG/reward.json", "w") as f:
    json.dump({"reward": 0.0, "reason": "wrong_answer"}, f)
PY
    exit 0
fi

INS=$(echo "$OUT" | sed -n 's/.*instructions=\([0-9]*\).*/\1/p')

# ── 2. Run with extra seeds to reject hardcoded answers ─────────────────────
for SEED in 42 137 999; do
    SOUT=$(./stack_vm $SEED)
    echo "seed=$SEED: $SOUT" >> "$LOG/run.txt"
    if ! echo "$SOUT" | grep -q "verify=ok"; then
        python3 - <<PY
import json
with open("$LOG/reward.json", "w") as f:
    json.dump({"reward": 0.0, "reason": "wrong_answer_seed_${SEED}"}, f)
PY
        exit 0
    fi
done

# ── 3. Compute reward ──────────────────────────────────────────────────────
BASELINE=5132
BEST_KNOWN=1026

REWARD=$(python3 - "$INS" "$BASELINE" "$BEST_KNOWN" <<'PY'
import sys

count = int(sys.argv[1])
baseline = int(sys.argv[2])
best_known = int(sys.argv[3])

if count >= baseline:
    reward = 0.0
elif count <= best_known:
    reward = 1.0
else:
    reward = (baseline - count) / (baseline - best_known)

print(f"{round(reward, 4)}")
PY
)

echo "instructions=$INS reward=$REWARD"
python3 - <<PY
import json
obj = {"reward": $REWARD, "correctness": True, "metric": int("$INS")}
with open("$LOG/reward.json", "w") as f:
    json.dump(obj, f, indent=2)
with open("$LOG/reward.txt", "w") as f:
    f.write(str($REWARD))
PY
