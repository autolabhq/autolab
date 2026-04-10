# Self-Generated Legal Calibration for GPTQ

Optimize post-training quantization under a strict legality constraint: calibration must not read held-out validation tokens.

## Problem

Minimize validation bits-per-byte after quantization:

```text
val_bpb
```

`val_bpb` is computed from byte-normalized NLL on a fixed validation token stream.

## Setup

| Item | Path |
|------|------|
| Main entrypoint (editable) | `/app/run_quant.py` |
| Calibration logic (editable) | `/app/calibrate.py` |
| Quantization logic (editable) | `/app/quantize.py` |
| Benchmark entrypoint (read-only) | `/app/benchmark.py` |
| Model definition (read-only) | `/app/model.py` |
| Dataset utilities (read-only) | `/app/dataset.py` |
| Baseline checkpoint (read-only) | `/app/baseline_ckpt.pt` |
| Local eval command | `python3 /app/benchmark.py --mode eval --seeds "42,314,999"` |
| Runtime stack | Python 3.12 + PyTorch 2.9.1 (CUDA 12.8) |

## Files

| Path | Permission |
|------|-----------|
| `/app/run_quant.py` | **Edit** |
| `/app/calibrate.py` | **Edit** |
| `/app/quantize.py` | **Edit** |
| `/app/benchmark.py` | Read-only |
| `/app/model.py` | Read-only |
| `/app/dataset.py` | Read-only |

## Rules

- During calibration, do **not** access validation tokens directly.
- Allowed calibration data sources:
  - synthetic random tokens
  - autoregressively self-generated tokens from the model
- Forbidden:
  - importing/calling validation builders/evaluators from `dataset.py` inside calibration
  - touching the provided forbidden probe in illegal mode
- Do not modify files outside the editable set above.
- Modifying `/app/benchmark.py` is invalid and fails verifier.

Any legality violation yields reward `0.0`.

## Evaluation

Verifier pipeline:

1. legality + correctness gate
2. baseline measurement from `/orig` with seeds `42,314,999`
3. optimized measurement from `/app` with seeds `42,314,999`
4. reward computation

Reward is `0.0` if correctness fails, legality fails, or artifact size exceeds `16,000,000` bytes.

Otherwise:

```text
improvement = baseline_bpb / optimized_bpb
ref_improvement = baseline_bpb / 8.2360
reward = min(1.0, 0.5 * log(improvement) / log(ref_improvement))
```

## Suggested Optimization Directions

- Replace random calibration with autoregressive self-generated calibration
- Upgrade from GPTQ-lite quantization to stronger Hessian-aware reconstruction
- Add size-aware selective pruning under the artifact cap
- Improve calibration diversity (temperature / sequence shape / seed policy)
- Improve mean `val_bpb` while keeping cross-seed variance controlled
