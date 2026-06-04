import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


BASELINE_PARAMS = 16641
MIN_ACCURACY = 0.64
MIN_UNSAFE_RECALL = 0.66
MIN_SAFE_RECALL = 0.57

EXPECTED_HASHES = {
    "data/train.npz": "220d806aed8013e554c21c592ec8d94ff99e1a817634ccd849aa441e3887db20",
    "data/val.npz": "186881b6f6e34cdc4817e953e9288836f72813ff78864e322e91e428a7b4d691",
    "data/test_public.npz": "fa099af9421c6639c6a0137cf8858ccdcbd004491d2d4fad9c55d006d537657e",
    "model.py": "a121f8f92e57493a25984adc7768da13ee1587faaaebc6567daac6d6967bf2cc",
}
PRIVATE_HASH = "200e504630fc8c08aae36230449294053a3a3208285b85cfa1b09d27204ad98b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_model_module(app_dir: Path):
    model_path = app_dir / "model.py"
    spec = importlib.util.spec_from_file_location("candidate_model", str(model_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import model module from {model_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifier for smallest_safety_router")
    parser.add_argument("--app-dir", type=str, default="/app")
    parser.add_argument("--private-data", type=str, default="/tests/data/test_private.npz")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=os.environ.get("LOG_DIR", "/tmp/autolab_logs"))
    return parser.parse_args()


def find_default_app_dir(passed: str) -> Path:
    candidate = Path(passed)
    if (candidate / "train.py").exists() and (candidate / "model.py").exists():
        return candidate
    fallback = Path(__file__).resolve().parents[1] / "environment"
    if (fallback / "train.py").exists() and (fallback / "model.py").exists():
        return fallback
    return candidate


def write_reward(log_dir: Path, payload: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "reward.json").write_text(json.dumps(payload, indent=2))
    (log_dir / "reward.txt").write_text(f"{payload['reward']:.10f}\n")


def main() -> None:
    args = parse_args()
    app_dir = find_default_app_dir(args.app_dir)
    private_data = Path(args.private_data)
    if not private_data.exists():
        alt = Path(__file__).resolve().parent / "data" / "test_private.npz"
        if alt.exists():
            private_data = alt

    model_module = import_model_module(app_dir)

    integrity_ok = True
    integrity_errors: list[str] = []

    for rel_path, expected in EXPECTED_HASHES.items():
        path = app_dir / rel_path
        if not path.exists() or sha256(path) != expected:
            integrity_ok = False
            integrity_errors.append(f"hash mismatch: {path}")

    if not private_data.exists() or sha256(private_data) != PRIVATE_HASH:
        integrity_ok = False
        integrity_errors.append(f"hash mismatch: {private_data}")

    if not integrity_ok:
        payload = {
            "reward": 0.0,
            "passed_constraints": False,
            "integrity_ok": False,
            "integrity_errors": integrity_errors,
        }
        write_reward(Path(args.log_dir), payload)
        return

    with np.load(private_data) as data:
        X = data["X"]
        y = data["y"].astype(np.int64)

    model = model_module.load_model(args.model_path)
    y_pred = model_module.predict_labels(model, X)
    metrics = model_module.compute_metrics(y, y_pred)

    # Compute total_params directly from the saved .npz weight tensors,
    # bypassing any model_module helper. Even with model.py hash-pinned this
    # provides defense-in-depth: agent-emitted weight files cannot lie about
    # parameter count, regardless of what model.py reports.
    with np.load(args.model_path, allow_pickle=True) as data:
        total_params = int(
            data["W1"].size + data["b1"].size + data["W2"].size + data["b2"].size
        )

    passed_constraints = (
        metrics.accuracy >= MIN_ACCURACY
        and metrics.unsafe_recall >= MIN_UNSAFE_RECALL
        and metrics.safe_recall >= MIN_SAFE_RECALL
        and total_params > 0
    )

    # Reward scales linearly in [0, 1] with compression, matching the
    # puzzle_and_challenge convention (cf. smallest_game_player).
    if not passed_constraints:
        reward = 0.0
    elif total_params >= BASELINE_PARAMS:
        reward = 0.0
    else:
        reward = float((BASELINE_PARAMS - total_params) / BASELINE_PARAMS)

    payload = {
        "reward": reward,
        "total_params": total_params,
        "accuracy": float(metrics.accuracy),
        "unsafe_recall": float(metrics.unsafe_recall),
        "safe_recall": float(metrics.safe_recall),
        "passed_constraints": bool(passed_constraints),
        "integrity_ok": True,
        "baseline_params": BASELINE_PARAMS,
        "thresholds": {
            "accuracy_min": MIN_ACCURACY,
            "unsafe_recall_min": MIN_UNSAFE_RECALL,
            "safe_recall_min": MIN_SAFE_RECALL,
        },
    }
    write_reward(Path(args.log_dir), payload)


if __name__ == "__main__":
    main()
