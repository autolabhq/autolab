from __future__ import annotations

"""Main task pipeline for legality-constrained GPTQ calibration."""

import argparse
import json
from pathlib import Path

import torch

from calibrate import build_calibration_pack
from dataset import DatasetConfig, ForbiddenProbe, build_validation_tokens, eval_val_bpb
from model import TinyLM
from quantize import (
    apply_selective_prune,
    artifact_size_bytes,
    dequantize_state_dict,
    quantize_state_dict,
)


def build_model(device: torch.device, seed: int) -> TinyLM:
    """Initialize model and load baseline checkpoint if present."""

    torch.manual_seed(seed)
    model = TinyLM(vocab_size=256, d_model=128, n_layers=4).to(device)
    ckpt_path = Path(__file__).resolve().parent / "baseline_ckpt.pt"
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    return model


def _editable_code_bytes() -> int:
    """Return bytes of editable files counted toward artifact size."""

    here = Path(__file__).resolve().parent
    targets = ["run_quant.py", "calibrate.py", "quantize.py"]
    return sum(len((here / t).read_bytes()) for t in targets)


def run_pipeline(mode: str, enforce_legal: bool, seed: int = 42) -> dict:
    """Run one full baseline/eval trial and return verifier-facing metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DatasetConfig(seed=12345 + seed)
    model = build_model(device, seed=20260325 + seed)
    val_tokens = build_validation_tokens(cfg)

    # Legality probe: calibration must not touch this object.
    forbidden = ForbiddenProbe()

    if mode == "baseline":
        calib_pack = build_calibration_pack(
            model,
            strategy="random",
            vocab_size=cfg.vocab_size,
            num_seqs=16,
            seq_len=256,
            seed=seed,
            temperature=1.0,
        )
        quant_method = "gptq_lite"
    else:
        calib_pack = build_calibration_pack(
            model,
            strategy="selfgen_ar",
            vocab_size=cfg.vocab_size,
            num_seqs=32,
            seq_len=256,
            temperature=0.8,
            seed=seed,
        )
        quant_method = "hessian_proxy"

    # Illegal-use simulation hook: if legal mode is disabled, marker is touched.
    if not enforce_legal:
        forbidden.touch()

    sd = model.state_dict()
    q = quantize_state_dict(sd, method=quant_method, calib_stats=calib_pack["col_sensitivity"])
    code_bytes = _editable_code_bytes()
    q = apply_selective_prune(q, target_total_bytes=16_000_000, code_bytes=code_bytes)
    total_bytes = artifact_size_bytes(q, code_bytes)
    deq = dequantize_state_dict(q, sd)
    model.load_state_dict(deq, strict=False)
    bpb = eval_val_bpb(model, val_tokens, seq_len=cfg.eval_seq_len, stride=cfg.stride)

    return {
        "val_bpb": float(bpb),
        "artifact_total_bytes": int(total_bytes),
        "legal": (not forbidden.touched),
        "quant_method": quant_method,
        "calibration_strategy": str(calib_pack["strategy"]),
        "calibration_tokens": int(calib_pack["num_tokens"]),
        "seed": int(seed),
        "mode": mode,
    }


def main():
    """CLI wrapper around run_pipeline()."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "eval"], default="eval")
    ap.add_argument("--enforce-legal", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    result = run_pipeline(mode=args.mode, enforce_legal=bool(args.enforce_legal), seed=args.seed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
