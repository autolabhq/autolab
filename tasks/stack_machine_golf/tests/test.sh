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

OUT=$(./stack_vm)
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
BASELINE=5132
REFERENCE=3530

python3 - <<PY
import json, math
score = float("$INS")
baseline = float("$BASELINE")
reference = float("$REFERENCE")
speedup = baseline / score if score > 0 else 0.0
ref_speedup = baseline / reference if reference > 0 else 1.0
reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
reward = max(0.0, reward)
obj = {"reward": round(reward, 4), "metric": int(score), "speedup": round(speedup, 2)}
with open("$LOG/reward.json", "w") as f:
    json.dump(obj, f)
with open("$LOG/reward.txt", "w") as f:
    f.write(str(round(reward, 4)))
print(json.dumps(obj))
PY
