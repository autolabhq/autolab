from __future__ import annotations

"""Reward function for selfgen_gptq_legal_calibration."""

import json
import math
import sys


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compute_reward.py <baseline_json> <optimized_json> <reference_bpb>")

    b = json.loads(sys.argv[1])
    o = json.loads(sys.argv[2])
    reference_bpb = float(sys.argv[3])

    if not b.get("all_legal", False):
        print(json.dumps({"reward": 0.0, "reason": "baseline_illegal"}))
        return
    if not o.get("all_legal", False):
        print(json.dumps({"reward": 0.0, "reason": "optimized_illegal"}))
        return

    bpb_b = float(b["val_bpb_mean"])
    bpb_o = float(o["val_bpb_mean"])
    bytes_o = int(o["artifact_total_bytes_max"])

    if bytes_o > 16_000_000:
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "reason": "artifact_too_large",
                    "artifact_total_bytes": bytes_o,
                    "size_cap_bytes": 16_000_000,
                }
            )
        )
        return

    improvement = bpb_b / bpb_o if bpb_o > 0 else 0.0
    ref_improvement = bpb_b / reference_bpb if reference_bpb > 0 else 1.0
    if improvement <= 1.0:
        reward = 0.0
    else:
        reward = min(1.0, 0.5 * math.log(improvement) / max(math.log(ref_improvement), 1e-9))
    reward = round(max(0.0, reward), 4)

    out = {
        "reward": reward,
        "baseline_val_bpb": round(bpb_b, 6),
        "optimized_val_bpb": round(bpb_o, 6),
        "optimized_val_bpb_std": round(float(o.get("val_bpb_std", 0.0)), 6),
        "improvement": round(improvement, 6),
        "reference_val_bpb": reference_bpb,
        "artifact_total_bytes": bytes_o,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
