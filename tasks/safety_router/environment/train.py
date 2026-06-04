import argparse
import json
from pathlib import Path

import numpy as np

from model import count_trainable_params, save_model, train_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train safety router baseline")
    parser.add_argument("--train-path", type=str, default="data/train.npz")
    parser.add_argument("--val-path", type=str, default="data/val.npz")
    parser.add_argument("--output", type=str, default="model_artifacts/router_model.npz")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with np.load(args.train_path) as train_data:
        X_train = train_data["X"]
        y_train = train_data["y"]

    with np.load(args.val_path) as val_data:
        X_val = val_data["X"]
        y_val = val_data["y"]

    model = train_router(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_path))

    summary = {
        "model_path": str(out_path),
        "hidden_dim": int(args.hidden_dim),
        "trainable_params": int(count_trainable_params(model)),
        "threshold": float(model["threshold"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
