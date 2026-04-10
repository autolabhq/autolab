from __future__ import annotations

"""Benchmark entrypoint for selfgen_gptq_legal_calibration.

This script aggregates multiple seeds for a selected mode and emits one JSON line
consumed by the verifier.
"""

import argparse
import json
import statistics

from run_quant import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "eval"], default="eval")
    ap.add_argument("--seeds", type=str, default="42,314,999")
    ap.add_argument("--enforce-legal", type=int, default=1)
    args = ap.parse_args()

    # Parse a comma-separated seed list, e.g. "42,314,999".
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # Evaluate each seed independently through the same pipeline.
    outs = [
        run_pipeline(mode=args.mode, enforce_legal=bool(args.enforce_legal), seed=s)
        for s in seeds
    ]
    bpb_list = [float(o["val_bpb"]) for o in outs]
    bytes_list = [int(o["artifact_total_bytes"]) for o in outs]
    legal_list = [bool(o["legal"]) for o in outs]

    # Verifier expects this schema to compute legality/quality/size statistics.
    summary = {
        "mode": args.mode,
        "seed_results": outs,
        "val_bpb_mean": float(statistics.mean(bpb_list)),
        "val_bpb_std": float(statistics.pstdev(bpb_list) if len(bpb_list) > 1 else 0.0),
        "artifact_total_bytes_max": int(max(bytes_list)),
        "all_legal": bool(all(legal_list)),
        "seeds": seeds,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
