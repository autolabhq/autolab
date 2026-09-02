#!/usr/bin/env python3
"""Mine public activation rules for target-67 line-family replays.

The line-family replay audit already separates public x-match features from
verifier labels.  This helper freezes a small ``activate:`` rule over the same
public features before a future validation window is inspected.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_line_family_activation_miner_target67_known_lines_328_463.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_line_family_activation_rule_miner_target67.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def activation_numeric_feature(features: dict[str, Any], key: str) -> int:
    if key.startswith("transfer_mod"):
        modulus = int(key.removeprefix("transfer_mod"))
        return int(features["transfer_index"]) % modulus
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        prefix, residue_text = key.removesuffix("_count").split("_residue", 1)
        return int(features[f"{prefix}_residue_counts"].get(residue_text, 0))
    return int(features[key])


def predicate_matches(predicate: str, features: dict[str, Any]) -> bool:
    if ">=" in predicate:
        key, value = predicate.split(">=", 1)
        return activation_numeric_feature(features, key) >= int(value)
    if "<=" in predicate:
        key, value = predicate.split("<=", 1)
        return activation_numeric_feature(features, key) <= int(value)
    key, value = predicate.split("=", 1)
    if key in {"row_xmatch_counts", "salt_mod2_pattern", "salt_mod3_pattern", "salt_mod4_pattern"}:
        return ",".join(str(part) for part in features[key]) == value
    if key.startswith("transfer_mod"):
        return activation_numeric_feature(features, key) == int(value)
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        return activation_numeric_feature(features, key) == int(value)
    if key in {
        "transfer_index",
        "top_k",
        "row_count",
        "factor_zero_profile_count",
        "xmatch_count",
        "support05_count",
        "support15_count",
        "scheduled1_count",
    }:
        return int(features[key]) == int(value)
    return str(features.get(key)) == value


def clause_matches(clause: str, features: dict[str, Any]) -> bool:
    return all(
        predicate_matches(part.strip(), features)
        for part in clause.split("&")
        if part.strip()
    )


def rule_matches(rule: str, features: dict[str, Any]) -> bool:
    expression = rule.removeprefix("activate:").strip()
    return any(clause_matches(clause, features) for clause in expression.split("|") if clause.strip())


def case_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in source.get("records") or []:
        for case in record.get("cases") or []:
            features = dict(case.get("activation_features") or {})
            if not features:
                continue
            label_success = int(case.get("label_verified_rule_count") or 0) > 0
            rows.append(
                {
                    "record_label": record.get("label"),
                    "source": record.get("source"),
                    "factor": record.get("factor"),
                    "features": features,
                    "label_success": label_success,
                    "label_verified_rules": case.get("label_verified_rules") or [],
                    "label_max_rank": int(case.get("label_max_rank") or 0),
                    "label_valid_relation_count": int(case.get("label_valid_relation_count") or 0),
                }
            )
    return rows


def feature_values(rows: list[dict[str, Any]], key: str) -> list[int]:
    return sorted({int(row["features"][key]) for row in rows if key in row["features"]})


def candidate_clauses(rows: list[dict[str, Any]]) -> list[str]:
    clauses = set()
    for top_k in feature_values(rows, "top_k"):
        clauses.add(f"top_k={top_k}")
        for threshold in feature_values(rows, "xmatch_count"):
            clauses.add(f"top_k={top_k}&xmatch_count>={threshold}")
        for threshold in feature_values(rows, "scheduled1_count"):
            clauses.add(f"top_k={top_k}&scheduled1_count>={threshold}")
    for threshold in feature_values(rows, "xmatch_count"):
        clauses.add(f"xmatch_count>={threshold}")
    for threshold in feature_values(rows, "scheduled1_count"):
        clauses.add(f"scheduled1_count>={threshold}")
    for row in rows:
        features = row["features"]
        if features.get("row_xmatch_counts") is not None:
            pattern = ",".join(str(part) for part in features["row_xmatch_counts"])
            clauses.add(f"row_xmatch_counts={pattern}")
    return sorted(clauses)


def candidate_rules(rows: list[dict[str, Any]], max_clauses: int) -> list[str]:
    clauses = candidate_clauses(rows)
    rules = {f"activate:{clause}" for clause in clauses}
    if max_clauses >= 2:
        for left, right in combinations(clauses, 2):
            rules.add(f"activate:{left}|{right}")
    return sorted(rules)


def evaluate_rule(rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    for row in rows:
        matched = rule_matches(rule, row["features"])
        selected.append({**row, "rule_matched": matched})
    tp = sum(1 for row in selected if row["rule_matched"] and row["label_success"])
    fp = sum(1 for row in selected if row["rule_matched"] and not row["label_success"])
    fn = sum(1 for row in selected if not row["rule_matched"] and row["label_success"])
    tn = sum(1 for row in selected if not row["rule_matched"] and not row["label_success"])
    return {
        "rule": rule,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "selected_count": tp + fp,
        "matched_cases": [
            {
                "record_label": row["record_label"],
                "label_success": row["label_success"],
                "label_verified_rules": row["label_verified_rules"],
                "features": row["features"],
            }
            for row in selected
            if row["rule_matched"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.source)
    rows = case_rows(source)
    evaluations = [evaluate_rule(rule, rows) for rule in candidate_rules(rows, max(1, args.max_clauses))]
    evaluations.sort(
        key=lambda row: (
            int(row["false_positive_count"]),
            -int(row["true_positive_count"]),
            int(row["false_negative_count"]),
            int(row["selected_count"]),
            len(str(row["rule"])),
            str(row["rule"]),
        )
    )
    zero_fp = [row for row in evaluations if int(row["false_positive_count"]) == 0]
    best = evaluations[0] if evaluations else None
    output = {
        "schema": "ecdlp_target67_line_family_activation_rule_miner_v1",
        "method": "public_activate_rule_grid_search_over_line_family_xmatch_features",
        "parameters": {
            "source": str(args.source),
            "max_clauses": int(args.max_clauses),
            "label_use": "verifier labels are used only to choose a rule before future-window validation",
        },
        "summary": {
            "source_case_count": len(rows),
            "candidate_rule_count": len(evaluations),
            "zero_false_positive_rule_count": len(zero_fp),
            "best_rule": best["rule"] if best else None,
            "best_rule_true_positive_count": best["true_positive_count"] if best else 0,
            "best_rule_false_positive_count": best["false_positive_count"] if best else 0,
            "best_rule_false_negative_count": best["false_negative_count"] if best else 0,
            "interpretation": (
                "Freeze best_rule before inspecting future windows. The rule may be used as "
                "the --activation-rule argument of ffe_public_linear_factor_xmatch_orientation_audit.py."
            ),
        },
        "best_rule_evaluation": best,
        "top_rules": evaluations[:25],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
