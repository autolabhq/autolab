from __future__ import annotations

"""Correctness and anti-triviality checks for verifier gate."""

import ast
import json
import os
import sys
from pathlib import Path
import subprocess
import importlib


def _assert_run_quant_structure(run_quant_path: Path) -> None:
    """Static checks to reduce trivial hardcoded-score bypasses."""
    tree = ast.parse(run_quant_path.read_text(encoding="utf-8"), filename=str(run_quant_path))
    imported_from_dataset = set()
    calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dataset":
            for alias in node.names:
                imported_from_dataset.add(alias.asname or alias.name)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.add(fn.attr)

    required_imports = {"build_validation_tokens", "eval_val_bpb"}
    if not required_imports.issubset(imported_from_dataset):
        missing = sorted(required_imports - imported_from_dataset)
        raise RuntimeError(f"run_quant.py missing required dataset imports: {missing}")

    required_calls = {"build_validation_tokens", "eval_val_bpb", "quantize_state_dict"}
    if not required_calls.issubset(calls):
        missing = sorted(required_calls - calls)
        raise RuntimeError(f"run_quant.py missing required calls: {missing}")


def _run(cmd):
    """Run command and parse the last stdout line as JSON."""

    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    line = p.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    app_root = Path(os.environ.get("TASK_APP_ROOT", "/app"))
    if not (app_root / "run_quant.py").exists():
        app_root = Path(__file__).resolve().parents[1] / "environment"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    _assert_run_quant_structure(app_root / "run_quant.py")

    out = _run(["python3", str(app_root / "run_quant.py"), "--mode", "eval", "--enforce-legal", "1"])
    required = {
        "val_bpb",
        "artifact_total_bytes",
        "legal",
        "quant_method",
        "calibration_strategy",
        "calibration_tokens",
        "seed",
        "mode",
    }
    if not required.issubset(set(out.keys())):
        raise RuntimeError("missing required fields in run output")
    if not (0.0 < float(out["val_bpb"]) < 20.0):
        raise RuntimeError("val_bpb out of expected range")
    if int(out["artifact_total_bytes"]) <= 0:
        raise RuntimeError("artifact size invalid")
    if out["legal"] is not True:
        raise RuntimeError("legal flag false")
    if out["mode"] != "eval":
        raise RuntimeError("unexpected mode")
    if int(out["seed"]) != 42:
        raise RuntimeError("seed field does not match requested seed")

    # Runtime hook check: ensure run_pipeline actually uses dataset.eval_val_bpb.
    os.environ["TASK_APP_ROOT"] = str(app_root)
    import dataset  # type: ignore
    import run_quant  # type: ignore

    marker = {"called": False}
    expected_bpb = 12.345678
    orig_eval = dataset.eval_val_bpb

    def _fake_eval(*args, **kwargs):
        marker["called"] = True
        return expected_bpb

    dataset.eval_val_bpb = _fake_eval
    importlib.reload(run_quant)
    hooked = run_quant.run_pipeline(mode="eval", enforce_legal=True, seed=42)
    dataset.eval_val_bpb = orig_eval

    if not marker["called"]:
        raise RuntimeError("run_pipeline did not call dataset.eval_val_bpb")
    if abs(float(hooked["val_bpb"]) - expected_bpb) > 1e-9:
        raise RuntimeError("run_pipeline does not propagate eval_val_bpb result")

    # Anti-constant-output check: different seeds should not return identical payloads.
    out_seed_314 = _run(
        ["python3", str(app_root / "run_quant.py"), "--mode", "eval", "--enforce-legal", "1", "--seed", "314"]
    )
    if int(out_seed_314.get("seed", -1)) != 314:
        raise RuntimeError("seed field mismatch for second run")
    if abs(float(out_seed_314["val_bpb"]) - float(out["val_bpb"])) < 1e-12:
        raise RuntimeError("val_bpb identical across different seeds; suspicious hardcoding")
    if (
        int(out_seed_314["artifact_total_bytes"]) == int(out["artifact_total_bytes"])
        and str(out_seed_314["quant_method"]) == str(out["quant_method"])
        and str(out_seed_314["calibration_strategy"]) == str(out["calibration_strategy"])
    ):
        # This condition alone is acceptable, but combined with identical val_bpb is suspicious.
        pass

    print("correctness_ok")


if __name__ == "__main__":
    main()
