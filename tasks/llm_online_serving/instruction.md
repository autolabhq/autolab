# Optimize Online LLM Serving in SimpleLLM

Improve the serving performance of SimpleLLM. The engine serves gpt-oss-20b, a 21B MoE model with 3.6B active parameters per token.

## Setup

| Item | Path |
|------|------|
| Engine source (editable) | `/app/llm.py` |
| Benchmark | `python3 /app/benchmark.py` |
| Model (read-only) | `/models/gpt-oss-20b` |
| Model code (read-only) | `/app/model/` |
| Kernels (read-only) | `/app/kernels/` |

## Your Goal

Maximize the combined serving score on 96 requests arriving with Poisson inter-arrival times:

```
serving_score = 0.5 × throughput_ratio + 0.5 × completion_time_ratio
```

Where `throughput_ratio = optimized/baseline` (higher is better) and `completion_time_ratio = baseline/optimized` (lower latency is better).

## Evaluation

```bash
python3 /app/benchmark.py
```

Outputs JSON with `throughput_tok_per_sec` and `mean_completion_sec`. The verifier runs this benchmark twice (baseline vs your code) and computes:

```
reward = 0.5 × (your_throughput / baseline_throughput) + 0.5 × (baseline_completion / your_completion)
```

A correctness check (`pytest /tests/test_state.py`) runs first — the engine must still produce valid output.

## Key Information

The engine architecture:
- **Async request queue**: Callers submit via `generate()`, background `_inference_loop` processes
- **Continuous batching**: Sequences join/leave dynamically as slots free up
- **Slot-based KV cache**: Fixed regions per sequence, released on completion

## Rules

- Edit `/app/llm.py` only
- Do NOT modify `/app/model/`, `/app/kernels/`, or `/orig/`
- No external network access
- Single GPU (H100 80GB)
- Time budget: 2 hours
