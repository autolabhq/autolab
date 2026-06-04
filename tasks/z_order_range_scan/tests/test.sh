#!/usr/bin/env bash
set -euo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"

write_reward() {
    local reward="$1" correctness="$2" metric="$3" speedup="$4"
    python3 - "$reward" "$correctness" "$metric" "$speedup" "$LOG/reward.json" "$LOG/reward.txt" <<'PY'
import json
import sys

reward = float(sys.argv[1])
correctness = sys.argv[2]
metric = sys.argv[3]
speedup = sys.argv[4]
rjson = sys.argv[5]
rtxt = sys.argv[6]

def parse(value):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value

payload = {
    "reward": reward,
    "correctness": parse(correctness),
    "metric": parse(metric),
    "speedup": parse(speedup),
}
with open(rjson, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
with open(rtxt, "w", encoding="utf-8") as f:
    f.write(str(round(reward, 4)))
PY
}

ORIGINALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/originals"
if [ ! -d "$ORIGINALS_DIR" ]; then
    echo "tests/originals missing at $ORIGINALS_DIR" >&2
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Layout ===" >&2
EXPECTED_APP_TOP=$(cat <<'EOF' | sort
/app
/app/.task_seed
/app/Cargo.lock
/app/Cargo.toml
/app/main.rs
/app/solve.rs
/app/target
EOF
)
ACTUAL_APP_TOP=$(find /app -mindepth 0 -maxdepth 1 \( -type f -o -type d \) 2>/dev/null | sort)
FILTERED_ACTUAL=$(echo "$ACTUAL_APP_TOP" | grep -v '^/app/target$' || true)
FILTERED_EXPECTED=$(echo "$EXPECTED_APP_TOP" | grep -v '^/app/target$')
if [ "$FILTERED_ACTUAL" != "$FILTERED_EXPECTED" ]; then
    echo "Unexpected file layout under /app:" >&2
    diff <(echo "$FILTERED_EXPECTED") <(echo "$FILTERED_ACTUAL") >&2 || true
    write_reward 0.0 false null null
    exit 0
fi

# Recursive whitelist: nothing besides /app/target/* may live outside the
# canonical files. Catches stashed lookup tables under /app/.cache/, etc.
EXTRA_ENTRIES=$(find /app -mindepth 1 \( -path /app/target -prune \) -o \
    \( -type f -o -type d \) -print 2>/dev/null | \
    grep -v '^/app/\.task_seed$' | \
    grep -v '^/app/Cargo\.lock$' | \
    grep -v '^/app/Cargo\.toml$' | \
    grep -v '^/app/main\.rs$' | \
    grep -v '^/app/solve\.rs$' | \
    grep -v '^/app/target$' || true)
if [ -n "$EXTRA_ENTRIES" ]; then
    echo "Unexpected extra entries under /app:" >&2
    echo "$EXTRA_ENTRIES" >&2
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Integrity ===" >&2
pushd "$ORIGINALS_DIR" >/dev/null
ORIG_FILES=$(find . -type f | sed 's|^\./||' | sort)
popd >/dev/null

while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if [ ! -f "/app/$rel" ]; then
        echo "Missing /app/$rel" >&2
        write_reward 0.0 false null null
        exit 0
    fi
    if ! cmp -s "$ORIGINALS_DIR/$rel" "/app/$rel"; then
        echo "tampered: /app/$rel differs from tests/originals" >&2
        write_reward 0.0 false null null
        exit 0
    fi
done <<< "$ORIG_FILES"

if [ ! -f /app/solve.rs ]; then
    echo "Missing editable /app/solve.rs" >&2
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Source checks ===" >&2
EDITABLE=/app/solve.rs
if grep -qE '__VERIFIER__|SlowSet|make_cases|runtime_salt|TASK_SEED|main\.rs|target/release|/dev/urandom|/proc/' "$EDITABLE"; then
    echo "solve.rs references task runner internals" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\bgetenv\b|\benviron\b' "$EDITABLE"; then
    echo "solve.rs reads process environment" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\bcrate::|\bsuper::|include!|include_str!|include_bytes!|env!|option_env!' "$EDITABLE"; then
    echo "solve.rs uses disallowed crate/include/environment access" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\bstd[[:space:]]*::[[:space:]]*(fs|process|env|thread|net|os)\b|\bCommand[[:space:]]*::[[:space:]]*new\b' "$EDITABLE"; then
    echo "solve.rs uses disallowed std module" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\bunsafe\b|\bextern[[:space:]]+"|#\[link\(|global_asm!|asm!' "$EDITABLE"; then
    echo "solve.rs uses disallowed unsafe or FFI escape" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\brayon\b|\btokio\b|\bcrossbeam\b|\bserde\b|\bmemmap\b|\bmmap\b' "$EDITABLE"; then
    echo "solve.rs references disallowed external/runtime facility" >&2
    write_reward 0.0 false null null
    exit 0
fi
if grep -qE '\bstatic[[:space:]]+mut\b|\bthread_local\!|\bOnceCell\b|\bOnceLock\b|\blazy_static\b|\bMutex\b|\bRwLock\b|\bRefCell\b|\bCell\b|\bUnsafeCell\b|\bSyncUnsafeCell\b|\bAtomic[A-Za-z0-9]*\b|\bBox[[:space:]]*::[[:space:]]*leak\b' "$EDITABLE"; then
    echo "solve.rs uses disallowed global/cross-call mutable state" >&2
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Build ===" >&2
if [ ! -f /app/.task_seed ]; then
    echo "Missing task seed" >&2
    write_reward 0.0 false null null
    exit 0
fi
export TASK_SEED="$(cat /app/.task_seed)"
export CARGO_NET_OFFLINE=true
if ! (cd /app && cargo clean) >"$LOG/clean.txt" 2>&1; then
    echo "Clean failed" >&2
    tail -n 80 "$LOG/clean.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi
if ! (cd /app && cargo build --release --offline) >"$LOG/build.txt" 2>&1; then
    echo "Build failed" >&2
    tail -n 120 "$LOG/build.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi

BIN=/app/target/release/range_count_task
if [ ! -x "$BIN" ]; then
    echo "Missing executable $BIN" >&2
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Verify ===" >&2
if ! "$BIN" --verify >"$LOG/verify.txt" 2>&1; then
    echo "Verify failed" >&2
    cat "$LOG/verify.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi
if ! grep -q '^OK' "$LOG/verify.txt"; then
    echo "Verify did not report OK" >&2
    cat "$LOG/verify.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Task run ===" >&2
if ! "$BIN" >"$LOG/run.txt" 2>"$LOG/run_stderr.txt"; then
    echo "Task run failed" >&2
    cat "$LOG/run_stderr.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi
cat "$LOG/run_stderr.txt" >&2 || true

SENTINEL_COUNT=$(grep -c '^__VERIFIER__' "$LOG/run.txt" || true)
if [ "$SENTINEL_COUNT" != "1" ]; then
    echo "Unexpected number of verifier lines: $SENTINEL_COUNT" >&2
    cat "$LOG/run.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi
if grep -q '^__VERIFIER__ .* result=wrong' "$LOG/run.txt"; then
    echo "Wrong output" >&2
    cat "$LOG/run.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi
if ! grep -q '^__VERIFIER__ .* result=ok' "$LOG/run.txt"; then
    echo "Missing success line" >&2
    cat "$LOG/run.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi

CAND_TIME=$(grep -oP '^__VERIFIER__[[:space:]]+.*\bcandidate_time=\K[0-9.]+' "$LOG/run.txt")
if [ -z "$CAND_TIME" ]; then
    echo "Could not parse candidate_time" >&2
    cat "$LOG/run.txt" >&2 || true
    write_reward 0.0 false null null
    exit 0
fi

BASELINE=2.0
REFERENCE=0.02
RESULT=$(python3 - "$CAND_TIME" "$BASELINE" "$REFERENCE" <<'PY'
import math
import sys

score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])
speedup = baseline / score
ref_speedup = baseline / reference
reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 4)}")
PY
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP_FMT=$(echo "$RESULT" | cut -d' ' -f2)

echo "runtime=${CAND_TIME}s speedup=${SPEEDUP_FMT}x reward=${REWARD}" >&2
write_reward "$REWARD" true "$CAND_TIME" "$SPEEDUP_FMT"
exit 0
