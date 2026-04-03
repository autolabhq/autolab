#!/usr/bin/env python3
"""
AutoLab verifier for fp8_m_grouped_gemm.

Usage:
    python3 run_benchmark.py [--warmup N] [--repeat N] [--output PATH]

Writes JSON to --output (default: /logs/verifier/reward.json) with schema:
    {
        "reward":       float,   # 0.0 – 1.0
        "correct":      bool,
        "agent_tflops": float,
        "deepgemm_tflops": float,
        "ratio":        float,   # agent / deepgemm
        "message":      str
    }
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.testing

TESTS_DIR = Path(__file__).resolve().parent


# ── helpers ──────────────────────────────────────────────────────────────────

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark(model, inputs, warmup: int, repeat: int) -> float:
    """Return average latency in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(*inputs)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    return sum(times) / len(times)


def tflops(flops: int, ms: float) -> float:
    return flops / ms / 1e9


def write_result(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup",   type=int, default=5)
    parser.add_argument("--repeat",   type=int, default=10)
    parser.add_argument("--output",   type=str, default="/logs/verifier/reward.json")
    parser.add_argument("--app-dir",  type=str, default=None,
                        help="Directory containing solve.py and friends "
                             "(default: $APP_DIR env var, then /app)")
    args = parser.parse_args()

    # resolve app directory
    if args.app_dir:
        app_dir = Path(args.app_dir).resolve()
    elif "APP_DIR" in os.environ:
        app_dir = Path(os.environ["APP_DIR"]).resolve()
    else:
        app_dir = Path("/app")

    sys.path.insert(0, str(app_dir))

    # ── load modules ─────────────────────────────────────────────────────────
    try:
        solve_mod    = load_module("solve",      app_dir / "solve.py")
        ref_mod      = load_module("torch_ref",  app_dir / "torch_ref.py")
        baseline_mod = load_module("baseline",   app_dir / "baseline.py")
        tc_mod       = load_module("test_cases", app_dir / "test_cases.py")
    except Exception as e:
        write_result(args.output, {
            "reward": 0.0, "correct": False,
            "agent_tflops": 0.0, "deepgemm_tflops": 0.0, "ratio": 0.0,
            "message": f"Import error: {e}\n{traceback.format_exc()}"
        })
        sys.exit(1)

    test_cases = tc_mod.test_cases
    rtol, atol = 3e-2, 3e-2

    agent_tflops_list    = []
    deepgemm_tflops_list = []
    correct = True
    fail_msg = ""

    for idx, tc in enumerate(test_cases):
        init_inputs = tc.get_init_inputs()
        inputs      = tc.get_inputs()
        flops       = tc.workload.flops if tc.workload else None

        # build models
        ref_model      = ref_mod.Model(*init_inputs).cuda()
        baseline_model = baseline_mod.Model(*init_inputs).cuda()
        try:
            agent_model = solve_mod.Model(*init_inputs).cuda()
        except Exception as e:
            write_result(args.output, {
                "reward": 0.0, "correct": False,
                "agent_tflops": 0.0, "deepgemm_tflops": 0.0, "ratio": 0.0,
                "message": f"[case {idx}] Failed to instantiate solve.Model: {e}"
            })
            sys.exit(1)

        # forward passes
        with torch.no_grad():
            ref_out   = ref_model(*inputs)
            base_out  = baseline_model(*inputs)
            try:
                agent_out = agent_model(*inputs)
            except Exception as e:
                write_result(args.output, {
                    "reward": 0.0, "correct": False,
                    "agent_tflops": 0.0, "deepgemm_tflops": 0.0, "ratio": 0.0,
                    "message": f"[case {idx}] solve.Model.forward() raised: {e}\n{traceback.format_exc()}"
                })
                sys.exit(1)

        # correctness gate
        try:
            torch.testing.assert_close(
                agent_out.float(), ref_out.float(), rtol=rtol, atol=atol
            )
        except AssertionError as e:
            correct = False
            fail_msg = (
                f"[case {idx}] Correctness check failed "
                f"(rtol={rtol}, atol={atol}): {e}"
            )
            write_result(args.output, {
                "reward": 0.0, "correct": False,
                "agent_tflops": 0.0, "deepgemm_tflops": 0.0, "ratio": 0.0,
                "message": fail_msg
            })
            sys.exit(1)

        # benchmark
        agent_ms    = benchmark(agent_model,    inputs, args.warmup, args.repeat)
        deepgemm_ms = benchmark(baseline_model, inputs, args.warmup, args.repeat)

        if flops:
            a_tf = tflops(flops, agent_ms)
            d_tf = tflops(flops, deepgemm_ms)
            agent_tflops_list.append(a_tf)
            deepgemm_tflops_list.append(d_tf)
            print(f"[case {idx}] agent={a_tf:.2f} TF  deepgemm={d_tf:.2f} TF  "
                  f"ratio={a_tf/d_tf:.3f}  agent_ms={agent_ms:.2f}  dg_ms={deepgemm_ms:.2f}")
        else:
            # fallback: use reciprocal latency as proxy
            agent_tflops_list.append(1.0 / agent_ms)
            deepgemm_tflops_list.append(1.0 / deepgemm_ms)

    # ── aggregate ─────────────────────────────────────────────────────────────
    avg_agent    = sum(agent_tflops_list) / len(agent_tflops_list)
    avg_deepgemm = sum(deepgemm_tflops_list) / len(deepgemm_tflops_list)
    ratio        = avg_agent / avg_deepgemm if avg_deepgemm > 0 else 0.0

    # reward = clamp(agent / (deepgemm * 0.80), 0, 1)
    TARGET_FRACTION = 0.80
    reward = min(1.0, max(0.0, ratio / TARGET_FRACTION))

    result = {
        "reward":          round(reward, 4),
        "correct":         True,
        "agent_tflops":    round(avg_agent, 4),
        "deepgemm_tflops": round(avg_deepgemm, 4),
        "ratio":           round(ratio, 4),
        "message": (
            f"agent={avg_agent:.2f} TFLOPs  "
            f"deepgemm={avg_deepgemm:.2f} TFLOPs  "
            f"ratio={ratio:.3f}  "
            f"target>=0.80  reward={reward:.4f}"
        )
    }
    write_result(args.output, result)


if __name__ == "__main__":
    main()
