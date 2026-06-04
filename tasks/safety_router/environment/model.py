import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Metrics:
    accuracy: float
    unsafe_recall: float
    safe_recall: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    accuracy = float((y_true == y_pred).mean())

    unsafe_mask = y_true == 1
    safe_mask = y_true == 0
    unsafe_recall = float((y_pred[unsafe_mask] == 1).mean()) if unsafe_mask.any() else 0.0
    safe_recall = float((y_pred[safe_mask] == 0).mean()) if safe_mask.any() else 0.0
    return Metrics(accuracy=accuracy, unsafe_recall=unsafe_recall, safe_recall=safe_recall)


def predict_proba(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xn = (X - model["feature_mean"]) / model["feature_std"]
    h_pre = Xn @ model["W1"] + model["b1"]
    h = np.maximum(h_pre, 0.0)
    logits = (h @ model["W2"] + model["b2"]).reshape(-1)
    return _sigmoid(logits)


def predict_labels(model: Dict[str, np.ndarray], X: np.ndarray, threshold: float | None = None) -> np.ndarray:
    if threshold is None:
        threshold = float(model["threshold"])
    p = predict_proba(model, X)
    return (p >= threshold).astype(np.int64)


def count_trainable_params(model: Dict[str, np.ndarray]) -> int:
    return int(model["W1"].size + model["b1"].size + model["W2"].size + model["b2"].size)


def save_model(model: Dict[str, np.ndarray], output_path: str) -> None:
    np.savez_compressed(
        output_path,
        W1=model["W1"].astype(np.float32),
        b1=model["b1"].astype(np.float32),
        W2=model["W2"].astype(np.float32),
        b2=model["b2"].astype(np.float32),
        feature_mean=model["feature_mean"].astype(np.float32),
        feature_std=model["feature_std"].astype(np.float32),
        threshold=np.array([float(model["threshold"])], dtype=np.float32),
        hidden_dim=np.array([int(model["W1"].shape[1])], dtype=np.int64),
        meta=np.array(
            [
                json.dumps(
                    {
                        "format": "smallest_safety_router_v1",
                        "input_dim": int(model["W1"].shape[0]),
                        "output_dim": 1,
                    }
                )
            ],
            dtype=object,
        ),
    )


def load_model(model_path: str) -> Dict[str, np.ndarray]:
    with np.load(model_path, allow_pickle=True) as data:
        model = {
            "W1": data["W1"].astype(np.float64),
            "b1": data["b1"].astype(np.float64),
            "W2": data["W2"].astype(np.float64),
            "b2": data["b2"].astype(np.float64),
            "feature_mean": data["feature_mean"].astype(np.float64),
            "feature_std": np.where(data["feature_std"].astype(np.float64) < 1e-6, 1.0, data["feature_std"].astype(np.float64)),
            "threshold": float(data["threshold"][0]),
        }
    return model


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std, mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def train_router(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 2026,
    hidden_dim: int = 128,
    epochs: int = 240,
    lr_start: float = 0.045,
    lr_end: float = 0.02,
    weight_decay: float = 1e-4,
    positive_weight: float = 1.15,
) -> Dict[str, np.ndarray]:
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_val = y_val.astype(np.float64)

    Xn_train, mean, std = _standardize_fit(X_train)
    Xn_val = _standardize_apply(X_val, mean, std)

    rng = np.random.default_rng(seed)
    input_dim = Xn_train.shape[1]

    W1 = rng.normal(0.0, 0.12, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    W2 = rng.normal(0.0, 0.08, size=(hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)

    best_score = -1.0
    best_bundle = None

    for epoch in range(epochs):
        # Forward pass.
        h_pre = Xn_train @ W1 + b1
        h = np.maximum(h_pre, 0.0)
        logits = (h @ W2 + b2).reshape(-1)
        probs = _sigmoid(logits)

        # Backprop for weighted BCE loss.
        weights = np.where(y_train > 0.5, positive_weight, 1.0)
        d_logits = (probs - y_train) * weights / y_train.shape[0]

        dW2 = h.T @ d_logits[:, None] + weight_decay * W2
        db2 = d_logits.sum(keepdims=True)

        d_hidden = d_logits[:, None] @ W2.T
        d_h_pre = d_hidden * (h_pre > 0.0)

        dW1 = Xn_train.T @ d_h_pre + weight_decay * W1
        db1 = d_h_pre.sum(axis=0)

        lr = lr_start if epoch < (epochs // 2) else lr_end
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 8 != 0:
            continue

        val_h = np.maximum(Xn_val @ W1 + b1, 0.0)
        val_probs = _sigmoid((val_h @ W2 + b2).reshape(-1))

        for threshold in np.linspace(0.35, 0.70, 36):
            val_pred = (val_probs >= threshold).astype(np.int64)
            m = compute_metrics(y_val.astype(np.int64), val_pred)
            # Heavier weight on unsafe recall to keep safety behavior stable.
            score = m.accuracy + 0.25 * m.unsafe_recall + 0.15 * m.safe_recall
            if score > best_score:
                best_score = score
                best_bundle = {
                    "W1": W1.copy(),
                    "b1": b1.copy(),
                    "W2": W2.copy(),
                    "b2": b2.copy(),
                    "feature_mean": mean.copy(),
                    "feature_std": std.copy(),
                    "threshold": float(threshold),
                }

    if best_bundle is None:
        raise RuntimeError("training failed to produce a model")

    return best_bundle
