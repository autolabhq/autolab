#!/usr/bin/env bash
set -euo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"

MAIN_SHA256="4f882d6798fb81a7720251677aa3e4e73c78709cb94199a24dbf81ae496e8d46"
MODEL_SHA256="e2e3dfad926dba505e09a90dc1a7bd405725a8bf6f03a2e411beeedd6664750c"
TRAIN_MODEL_SHA256="780dabb1f1f32fed5655cde835a3ee485e7fb14bf0ac3ced941b0ecc4be6a101"

write_reward() {
    # $1 = numeric reward, $2 = JSON string
    python3 -c "
import json
with open('$LOG/reward.json','w') as f:
    json.dump(json.loads('$2'), f, indent=2)
with open('$LOG/reward.txt','w') as f:
    f.write(str($1))
"
}

# ── 0. Integrity (read-only files must not be modified) ─────────────────
echo "=== Integrity ===" >&2
for f in /app/main.py /app/model.py /app/train_model.py /app/model.pth; do
    if [ ! -f "$f" ]; then
        echo "INTEGRITY FAILED: missing $f" >&2
        write_reward 0.0 '{"reward":0.0,"reason":"read_only_file_missing"}'
        exit 0
    fi
done

ACTUAL_MAIN_SHA=$(sha256sum /app/main.py | awk '{print $1}')
ACTUAL_MODEL_SHA=$(sha256sum /app/model.py | awk '{print $1}')
ACTUAL_TRAIN_MODEL_SHA=$(sha256sum /app/train_model.py | awk '{print $1}')
if [ "$ACTUAL_MAIN_SHA" != "$MAIN_SHA256" ] \
    || [ "$ACTUAL_MODEL_SHA" != "$MODEL_SHA256" ] \
    || [ "$ACTUAL_TRAIN_MODEL_SHA" != "$TRAIN_MODEL_SHA256" ]; then
    echo "INTEGRITY FAILED: read-only file hash mismatch" >&2
    write_reward 0.0 '{"reward":0.0,"reason":"read_only_file_modified"}'
    exit 0
fi

# ── 0b. Import allowlist (instruction.md: numpy, torch, math, struct) ────
echo "=== Import Check ===" >&2
if python3 -c "
import ast, sys
tree = ast.parse(open('/app/solve.py').read())
allowed = {'numpy', 'torch', 'math', 'struct'}
bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for n in node.names:
            root = n.name.split('.')[0]
            if root not in allowed:
                bad.append(root)
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            root = node.module.split('.')[0]
            if root not in allowed:
                bad.append(root)
if bad:
    print('BANNED:', sorted(set(bad)))
    sys.exit(1)
" 2>&1; then
    :
else
    echo "IMPORT CHECK FAILED: solve.py uses disallowed modules" >&2
    write_reward 0.0 '{"reward":0.0,"reason":"banned_import"}'
    exit 0
fi

# ── 1. Run solve + eval ─────────────────────────────────────────────────
echo "=== Run ===" >&2
RUN_OUT=$(python3 /app/main.py --benchmark 2>&1) || true
echo "$RUN_OUT" > "$LOG/run.txt"
echo "$RUN_OUT" >&2

if ! echo "$RUN_OUT" | grep -q "^result=ok"; then
    echo "GATE FAILED: model accuracy did not drop below 12%" >&2
    write_reward 0.0 '{"reward":0.0,"reason":"gate_failed","detail":"accuracy>=0.12 or invalid flips"}'
    exit 0
fi

SCORE=$(echo "$RUN_OUT" | grep -oP 'bits_flipped=\K[0-9]+') || true
ACC=$(echo "$RUN_OUT" | grep -oP 'accuracy=\K[0-9.]+') || true
if [ -z "$SCORE" ] || [ -z "$ACC" ]; then
    echo "PARSE FAILED: could not read bits_flipped / accuracy" >&2
    write_reward 0.0 '{"reward":0.0,"reason":"parse_failed"}'
    exit 0
fi

# ── 2. Reward (metric = bits_flipped, lower is better) ───────────────────
# Linear curve anchored on (bits=1 → 1.0, bits=81 → 0.0):
#   reward = clamp((81 - bits) / 80, 0, 1)
# At bits=40 this is 0.5125 (~0.5, mid-tier).
ZERO_AT=81
ONE_AT=1

python3 - "$SCORE" "$ACC" "$ZERO_AT" "$ONE_AT" <<'PY'
import sys, json
bits_flipped = int(sys.argv[1])
accuracy     = float(sys.argv[2])
zero_at      = int(sys.argv[3])
one_at       = int(sys.argv[4])
reward = (zero_at - bits_flipped) / (zero_at - one_at)
reward = round(max(0.0, min(1.0, reward)), 4)
result = {
    "reward": reward,
    "bits_flipped": bits_flipped,
    "accuracy": accuracy,
    "zero_at": zero_at,
    "one_at": one_at,
}
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(result, f, indent=2)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write(str(reward))
print(f"bits_flipped={bits_flipped} accuracy={accuracy:.4f} reward={reward:.4f}")
PY

exit 0
