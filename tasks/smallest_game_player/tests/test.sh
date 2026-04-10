#!/usr/bin/env bash
set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

write_reward() {
    local reward="$1" correctness="$2" metric="$3" score="$4"
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$reward" "$correctness" "$metric" "$score" > "$REWARD_FILE"
}

echo "=== Running correctness check ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

echo ""
echo "=== Running main evaluation ==="

RESULT=$(python3 - <<'PYEOF'
import importlib.util
import sys
import numpy as np

X_train = np.load("/app/X_train.npy")
y_train = np.load("/app/y_train.npy")
X_test = np.load("/tests/X_test.npy")
y_test = np.load("/tests/y_test.npy")

spec = importlib.util.spec_from_file_location("solve", "/app/solve.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.train(X_train, y_train)
preds = mod.predict(model, X_test)

if not isinstance(model, dict):
    print("ERROR: train() must return dict")
    sys.exit(1)
if not isinstance(preds, np.ndarray) or preds.shape != (len(X_test),):
    print("ERROR: predict() returned invalid shape")
    sys.exit(1)

acc = float((preds.astype(np.int64) == y_test).mean())
params = sum(v.size for v in model.values() if isinstance(v, np.ndarray))
print(f"accuracy={acc:.6f} params={params}")
PYEOF
)

echo "$RESULT"

ACC=$(echo "$RESULT" | grep -oP 'accuracy=\K[0-9.]+')
PARAMS=$(echo "$RESULT" | grep -oP 'params=\K[0-9]+')

if [[ -z "$ACC" || -z "$PARAMS" ]]; then
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

ACC_OK=$(python3 -c "print('yes' if float('$ACC') >= 0.95 else 'no')")
if [[ "$ACC_OK" != "yes" ]]; then
    write_reward 0.0 false "$PARAMS" null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

BASELINE=17924

REWARD=$(python3 - "$PARAMS" "$BASELINE" <<'PYEOF'
import sys

params = int(sys.argv[1])
baseline = int(sys.argv[2])

if params >= baseline:
    reward = 0.0
elif params <= 0:
    reward = 1.0
else:
    reward = (baseline - params) / baseline

print(f"{round(reward, 4)}")
PYEOF
)

echo "params=$PARAMS reward=$REWARD"
write_reward "$REWARD" true "$PARAMS" null
echo "$REWARD" > /logs/verifier/reward.txt
