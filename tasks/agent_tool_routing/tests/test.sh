#!/usr/bin/env bash
# tests/test.sh - verifier for agent_tool_routing
#
# DO NOT MODIFY THIS FILE.

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
REWARD_TXT="/logs/verifier/reward.txt"
mkdir -p /logs/verifier

write_zero() {
    local reason="$1"
    printf '{"reward": 0.0, "correctness": false, "reason": "%s"}\n' "$reason" > "$REWARD_FILE"
    echo "0.0" > "$REWARD_TXT"
}

# Expected sha256 is hard-coded here (not read from /orig/) so the agent — even as
# root inside the container — cannot tamper with it: tests/ is mounted by the
# verifier after the agent's filesystem is frozen.
EXPECTED_MAIN_PY_SHA256="feb28a6c160fce1d1b135ea093dcc17a9e4447778de6fbbf023238bd8bddaa61"
curr=$(sha256sum /app/main.py 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [[ "$EXPECTED_MAIN_PY_SHA256" != "$curr" ]]; then
    echo "main.py was modified - scoring 0" >&2
    write_zero "main_py_modified"
    exit 0
fi

echo "=== Syntax check ==="
if ! python3 -m py_compile /app/main.py /app/retriever.py 2>&1; then
    write_zero "syntax_failed"
    exit 0
fi

echo ""
echo "=== Public smoke workload ==="
if ! python3 /app/main.py --verify 2>&1; then
    write_zero "public_smoke_failed"
    exit 0
fi

echo ""
echo "=== Hidden verifier workload ==="
if ! python3 /tests/verify_and_benchmark.py \
        --reward-file "$REWARD_FILE" \
        --reward-txt "$REWARD_TXT" 2>&1; then
    write_zero "hidden_verifier_failed"
    exit 0
fi

echo "Results written to $REWARD_FILE"
