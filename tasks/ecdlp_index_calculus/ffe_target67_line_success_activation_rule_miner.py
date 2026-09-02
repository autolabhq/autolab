#!/usr/bin/env python3
"""Mine public activation rules for target-67 line-gated rank-2 replays.

The input feature probe is already restricted to cases where exact FFE found a
preserving degree-1 line.  This miner asks a narrower question: from public
x-match metadata available after the line gate, which cases should keep the
rank-aware ``all`` replay baseline because they are likely to derive the key?
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SOURCE = DEFAULT_STATE_DIR / "ffe_target67_orientation_feature_probe_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_line_success_activation_rule_miner_328_511.json"

PUBLIC_NUMERIC_KEYS = (
    "top_k",
    "xmatch_count",
    "scheduled1_count",
    "candidate_eq_scheduled_xmatch_count",
    "factor_zero_profile_count",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def public_features(record: dict[str, Any]) -> dict[str, Any]:
    features = {
        key: record.get(key)
        for key in PUBLIC_NUMERIC_KEYS
        if record.get(key) is not None
    }
    features["row_xmatch_counts"] = list(record.get("row_xmatch_counts") or [])
    features["leaf_signature"] = str(record.get("leaf_signature") or "")
    features["leaf_selector"] = str(record.get("leaf_selector") or "")
    return features


def rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for record in source.get("records") or []:
        out.append(
            {
                "record_id": (
                    f"transfer{record.get('transfer_index')}_top{record.get('top_k')}_"
                    f"{record.get('leaf_signature')}"
                ),
                "transfer_index": int(record.get("transfer_index") or 0),
                "label_success": bool(record.get("label_replay_success")),
                "label_verified_rule_count": int(record.get("original_verified_rule_count") or 0),
                "label_best_rule": record.get("original_best_verified_rule"),
                "features": public_features(record),
            }
        )
    return out


def numeric_feature(features: dict[str, Any], key: str) -> int:
    return int(features[key])


def predicate_matches(predicate: str, features: dict[str, Any]) -> bool:
    if ">=" in predicate:
        key, value = predicate.split(">=", 1)
        return numeric_feature(features, key) >= int(value)
    if "<=" in predicate:
        key, value = predicate.split("<=", 1)
        return numeric_feature(features, key) <= int(value)
    key, value = predicate.split("=", 1)
    if key == "row_xmatch_counts":
        return ",".join(str(part) for part in features.get(key) or []) == value
    if key in PUBLIC_NUMERIC_KEYS:
        return numeric_feature(features, key) == int(value)
    return str(features.get(key) or "") == value


def clause_matches(clause: str, features: dict[str, Any]) -> bool:
    return all(
        predicate_matches(part.strip(), features)
        for part in clause.split("&")
        if part.strip()
    )


def rule_matches(rule: str, features: dict[str, Any]) -> bool:
    expression = rule.removeprefix("activate:").strip()
    return any(clause_matches(clause, features) for clause in expression.split("|") if clause.strip())


def feature_values(data: list[dict[str, Any]], key: str) -> list[int]:
    values = set()
    for row in data:
        value = row["features"].get(key)
        if value is not None:
            values.add(int(value))
    return sorted(values)


def candidate_clauses(data: list[dict[str, Any]], include_leaf_signature: bool) -> list[str]:
    clauses = set()
    for key in PUBLIC_NUMERIC_KEYS:
        for value in feature_values(data, key):
            if key == "top_k":
                clauses.add(f"{key}={value}")
                continue
            clauses.add(f"{key}>={value}")
            clauses.add(f"{key}<={value}")
    for top_k in feature_values(data, "top_k"):
        for threshold in feature_values(data, "xmatch_count"):
            clauses.add(f"top_k={top_k}&xmatch_count>={threshold}")
        for threshold in feature_values(data, "scheduled1_count"):
            clauses.add(f"top_k={top_k}&scheduled1_count>={threshold}")
    for row in data:
        pattern = ",".join(str(part) for part in row["features"].get("row_xmatch_counts") or [])
        if pattern:
            clauses.add(f"row_xmatch_counts={pattern}")
        if include_leaf_signature and row["features"].get("leaf_signature"):
            clauses.add(f"leaf_signature={row['features']['leaf_signature']}")
    return sorted(clauses)


def candidate_rules(data: list[dict[str, Any]], max_clauses: int, include_leaf_signature: bool) -> list[str]:
    clauses = candidate_clauses(data, include_leaf_signature)
    rules = {f"activate:{clause}" for clause in clauses}
    if max_clauses >= 2:
        for left, right in combinations(clauses, 2):
            rules.add(f"activate:{left}|{right}")
    if max_clauses >= 3:
        for first, second, third in combinations(clauses, 3):
            rules.add(f"activate:{first}|{second}|{third}")
    return sorted(rules)


def evaluate_rule(rule: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    case_results = []
    for row in data:
        matched = rule_matches(rule, row["features"])
        case_results.append(
            {
                "record_id": row["record_id"],
                "transfer_index": row["transfer_index"],
                "label_success": row["label_success"],
                "label_best_rule": row["label_best_rule"],
                "rule_matched": matched,
                "features": row["features"],
            }
        )
    tp = sum(1 for row in case_results if row["rule_matched"] and row["label_success"])
    fp = sum(1 for row in case_results if row["rule_matched"] and not row["label_success"])
    fn = sum(1 for row in case_results if not row["rule_matched"] and row["label_success"])
    tn = sum(1 for row in case_results if not row["rule_matched"] and not row["label_success"])
    return {
        "rule": rule,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "selected_count": tp + fp,
        "case_results": case_results,
    }


def score(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["false_positive_count"]),
        int(row["false_negative_count"]),
        -int(row["true_positive_count"]),
        int(row["selected_count"]),
        len(str(row["rule"])),
        str(row["rule"]),
    )


def validation_score(row: dict[str, Any], train_row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["false_positive_count"]),
        int(row["false_negative_count"]),
        -int(row["true_positive_count"]),
        int(train_row["false_positive_count"]),
        int(train_row["false_negative_count"]),
        -int(train_row["true_positive_count"]),
        int(row["selected_count"]),
        len(str(row["rule"])),
        str(row["rule"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-max-transfer", type=int, default=463)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--include-leaf-signature", action="store_true")
    parser.add_argument("--top-rules", type=int, default=25)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.source)
    all_rows = rows(source)
    train_rows = [row for row in all_rows if row["transfer_index"] <= args.train_max_transfer]
    validation_rows = [row for row in all_rows if row["transfer_index"] > args.train_max_transfer]
    evaluations = [
        evaluate_rule(rule, train_rows)
        for rule in candidate_rules(
            train_rows,
            max(1, int(args.max_clauses)),
            bool(args.include_leaf_signature),
        )
    ]
    evaluations.sort(key=score)
    best = evaluations[0] if evaluations else None
    validation = evaluate_rule(best["rule"], validation_rows) if best else None
    validation_pairs = [
        (train_eval, evaluate_rule(train_eval["rule"], validation_rows))
        for train_eval in evaluations
    ]
    validation_pairs.sort(key=lambda pair: validation_score(pair[1], pair[0]))
    validation_best_train, validation_best = validation_pairs[0] if validation_pairs else (None, None)
    output = {
        "schema": "ecdlp_target67_line_success_activation_rule_miner_v1",
        "method": "train_public_line_gate_activation_rule_then_score_later_transfers",
        "parameters": {
            "source": str(args.source),
            "train_max_transfer": int(args.train_max_transfer),
            "max_clauses": int(args.max_clauses),
            "include_leaf_signature": bool(args.include_leaf_signature),
            "orientation_baseline": "all",
            "label_use": (
                "Replay verification labels rank public activation rules. "
                "Validation rows are later transfers than the training cutoff."
            ),
        },
        "summary": {
            "case_count": len(all_rows),
            "train_case_count": len(train_rows),
            "validation_case_count": len(validation_rows),
            "train_success_count": sum(1 for row in train_rows if row["label_success"]),
            "validation_success_count": sum(1 for row in validation_rows if row["label_success"]),
            "candidate_rule_count": len(evaluations),
            "best_rule": best["rule"] if best else None,
            "train_true_positive_count": best["true_positive_count"] if best else 0,
            "train_false_positive_count": best["false_positive_count"] if best else 0,
            "train_false_negative_count": best["false_negative_count"] if best else 0,
            "validation_true_positive_count": (
                validation["true_positive_count"] if validation else 0
            ),
            "validation_false_positive_count": (
                validation["false_positive_count"] if validation else 0
            ),
            "validation_false_negative_count": (
                validation["false_negative_count"] if validation else 0
            ),
            "validation_best_rule": validation_best["rule"] if validation_best else None,
            "validation_best_true_positive_count": (
                validation_best["true_positive_count"] if validation_best else 0
            ),
            "validation_best_false_positive_count": (
                validation_best["false_positive_count"] if validation_best else 0
            ),
            "validation_best_false_negative_count": (
                validation_best["false_negative_count"] if validation_best else 0
            ),
            "validation_best_train_true_positive_count": (
                validation_best_train["true_positive_count"] if validation_best_train else 0
            ),
            "validation_best_train_false_positive_count": (
                validation_best_train["false_positive_count"] if validation_best_train else 0
            ),
            "validation_best_train_false_negative_count": (
                validation_best_train["false_negative_count"] if validation_best_train else 0
            ),
            "interpretation": (
                "The best rule is an activation rule for the public line-gated all-xmatch "
                "replay baseline, not a degree-1 line predictor by itself."
            ),
        },
        "best_rule_training_evaluation": best,
        "best_rule_validation_evaluation": validation,
        "validation_best_rule_training_evaluation": validation_best_train,
        "validation_best_rule_validation_evaluation": validation_best,
        "top_training_rules": evaluations[: max(1, int(args.top_rules))],
        "top_validation_rules": [
            {
                "training_evaluation": train_eval,
                "validation_evaluation": validation_eval,
            }
            for train_eval, validation_eval in validation_pairs[: max(1, int(args.top_rules))]
        ],
        "non_claims": [
            "The feature table contains only line-present cases found by exact FFE.",
            "This does not remove the need for public line prediction or amortized line confirmation.",
            "A fresh future line-present holdout is still required before treating the rule as general.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
