import argparse
import json

import numpy as np

from model import compute_metrics, count_trainable_params, load_model, predict_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate safety router on a local split")
    parser.add_argument("--model", type=str, default="model_artifacts/router_model.npz")
    parser.add_argument("--split", type=str, default="data/test_public.npz")
    parser.add_argument("--min-accuracy", type=float, default=0.64)
    parser.add_argument("--min-unsafe-recall", type=float, default=0.66)
    parser.add_argument("--min-safe-recall", type=float, default=0.57)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(args.model)
    with np.load(args.split) as data:
        X = data["X"]
        y = data["y"].astype(np.int64)

    pred = predict_labels(model, X)
    metrics = compute_metrics(y, pred)

    passed = (
        metrics.accuracy >= args.min_accuracy
        and metrics.unsafe_recall >= args.min_unsafe_recall
        and metrics.safe_recall >= args.min_safe_recall
    )

    result = {
        "split": args.split,
        "total_params": int(count_trainable_params(model)),
        "accuracy": float(metrics.accuracy),
        "unsafe_recall": float(metrics.unsafe_recall),
        "safe_recall": float(metrics.safe_recall),
        "passed_constraints": bool(passed),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
