#!/usr/bin/env python3
"""Mine public target-67 axis/orientation activation rules.

This helper consumes exact-profile axis-root artifacts plus line-gated
x-match orientation artifacts.  It builds a compact public feature table and
mines small conjunctive activation rules that should be frozen before testing
future windows.

Labels are read only after each public feature row is built.  The candidate
rule vocabulary intentionally excludes transfer ids, salts, exact line
coefficients, and verifier labels.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_axis_orientation_feature_rule_miner.json"

NUMERIC_RULE_KEYS = (
    "top_k",
    "source_ops_over_rho_millirhos",
    "xmatch_count",
    "scheduled1_count",
    "row_xmatch_count_min",
    "row_xmatch_count_max",
    "candidate_pos_unique_count",
    "scheduled_trial_unique_count",
    "factor_zero_profile_count",
    "factor_zero_leaf_min",
    "factor_zero_leaf_max",
    "axis_candidate_line_count",
    "axis_line_lift_millirhos",
    "axis_line_leaf_count",
)

STRING_RULE_KEYS = (
    "policy",
    "leaf_selector",
    "row_xmatch_counts",
    "candidate_pos_counts",
    "scheduled_trial_counts",
    "term_shape_counts",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def parse_record(raw: str) -> tuple[str, str, Path, Path]:
    parts = raw.split("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "record must be label|bucket|axis_json|orientation_json"
        )
    return parts[0], parts[1], resolve(parts[2]), resolve(parts[3])


def factor_line_key(artifact: dict[str, Any]) -> tuple[int, int, int, int | None]:
    factor = (artifact.get("parameters") or {}).get("factor") or {}
    return (
        int(factor.get("b_coeff") or 0),
        int(factor.get("c_coeff") or 0),
        int(factor.get("constant") or 0),
        int(factor["p"]) if factor.get("p") is not None else None,
    )


def flatten_xmatches(case: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in case.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for match in row.get("xmatches") or []:
            if isinstance(match, dict):
                out.append(dict(match))
    return out


def count_values(values: list[Any]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def int_list_pattern(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(values))


def verified_rules(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(result)
        for result in case.get("rule_results") or []
        if bool(result.get("public_key_verified"))
    ]


def charged_over_rho(result: dict[str, Any]) -> float | None:
    value = (result.get("charged_models") or {}).get("measured_oriented_ops_over_rho")
    return float(value) if value is not None else None


def best_verified_rule(case: dict[str, Any]) -> dict[str, Any] | None:
    rules = verified_rules(case)
    if not rules:
        return None
    return min(
        rules,
        key=lambda result: (
            charged_over_rho(result) if charged_over_rho(result) is not None else 10**18,
            int(result.get("selected_xmatch_count") or 0),
            str(result.get("rule") or ""),
        ),
    )


def axis_record_for_case(
    axis: dict[str, Any],
    case: dict[str, Any],
    line: tuple[int, int, int, int | None],
) -> dict[str, Any]:
    transfer = int(case.get("transfer_index") or 0)
    top_k = int(case.get("top_k") or 0)
    leaf_selector = str(case.get("leaf_selector") or "")
    b_coeff, c_coeff, constant, _p = line
    for record in axis.get("records") or []:
        if int(record.get("transfer_index") or 0) != transfer:
            continue
        if int(record.get("top_k") or 0) != top_k:
            continue
        if str(record.get("leaf_selector") or "") != leaf_selector:
            continue
        for candidate in record.get("candidate_lines") or []:
            if (
                int(candidate.get("slope") or 0) == b_coeff
                and int(candidate.get("constant") or 0) == constant
                and c_coeff == 1
            ):
                return dict(record)
    return {}


def axis_line_for_record(
    axis_record: dict[str, Any],
    line: tuple[int, int, int, int | None],
) -> dict[str, Any]:
    b_coeff, c_coeff, constant, _p = line
    for candidate in axis_record.get("candidate_lines") or []:
        if (
            int(candidate.get("slope") or 0) == b_coeff
            and int(candidate.get("constant") or 0) == constant
            and c_coeff == 1
        ):
            return dict(candidate)
    return {}


def feature_row(
    label: str,
    bucket: str,
    axis_path: Path,
    orientation_path: Path,
) -> dict[str, Any]:
    axis = load_json(axis_path)
    orientation = load_json(orientation_path)
    cases = [case for case in orientation.get("cases") or [] if isinstance(case, dict)]
    if len(cases) != 1:
        raise ValueError(f"expected one case in {orientation_path}, found {len(cases)}")
    case = cases[0]
    line = factor_line_key(orientation)
    axis_record = axis_record_for_case(axis, case, line)
    axis_line = axis_line_for_record(axis_record, line)
    features = dict(case.get("activation_features") or {})
    xmatches = flatten_xmatches(case)
    row_xmatch_counts = [len(row.get("xmatches") or []) for row in case.get("rows") or []]
    candidate_positions = [int(match.get("candidate_pos") or 0) for match in xmatches]
    scheduled_trials = [int(match.get("scheduled_trial") or 0) for match in xmatches]
    term_shapes = [str(match.get("term_shape") or "") for match in xmatches]
    factor_zero_leaves = [
        int(profile.get("leaf_index"))
        for profile in case.get("factor_zero_profiles") or []
        if isinstance(profile, dict) and profile.get("leaf_index") is not None
    ]
    best = best_verified_rule(case)
    verified = verified_rules(case)
    line_lift = axis_record.get("line_lift_ops_over_rho")
    line_lift_millirhos = (
        int(round(float(line_lift) * 1000)) if line_lift is not None else None
    )
    public_features = {
        "target": case.get("target"),
        "top_k": int(case.get("top_k") or features.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector"),
        "source_ops_over_rho_millirhos": int(
            round(float(case.get("source_ops_over_rho") or 0.0) * 1000)
        ),
        "factor_zero_profile_count": int(features.get("factor_zero_profile_count") or len(factor_zero_leaves)),
        "factor_zero_leaf_min": min(factor_zero_leaves) if factor_zero_leaves else -1,
        "factor_zero_leaf_max": max(factor_zero_leaves) if factor_zero_leaves else -1,
        "xmatch_count": int(features.get("xmatch_count") or len(xmatches)),
        "scheduled1_count": int(features.get("scheduled1_count") or 0),
        "row_xmatch_count_min": min(row_xmatch_counts) if row_xmatch_counts else 0,
        "row_xmatch_count_max": max(row_xmatch_counts) if row_xmatch_counts else 0,
        "row_xmatch_counts": int_list_pattern(row_xmatch_counts),
        "candidate_pos_unique_count": len(set(candidate_positions)),
        "candidate_pos_counts": count_values(candidate_positions),
        "scheduled_trial_unique_count": len(set(scheduled_trials)),
        "scheduled_trial_counts": count_values(scheduled_trials),
        "term_shape_counts": count_values(term_shapes),
        "axis_candidate_line_count": int(axis_record.get("candidate_line_count") or 0),
        "axis_line_lift_millirhos": line_lift_millirhos if line_lift_millirhos is not None else -1,
        "axis_line_leaf_count": len(axis_line.get("leaf_indices") or []),
    }
    return {
        "label": label,
        "bucket": bucket,
        "axis_source": str(axis_path.relative_to(WORKTREE_ROOT)),
        "orientation_source": str(orientation_path.relative_to(WORKTREE_ROOT)),
        "transfer_index": int(case.get("transfer_index") or 0),
        "line": {
            "b_coeff": line[0],
            "c_coeff": line[1],
            "constant": line[2],
            "p": line[3],
        },
        "public_features": public_features,
        "labels": {
            "source_public_key_verified": bool(case.get("source_public_key_verified")),
            "orientation_verified_rule_count": len(verified),
            "orientation_success": bool(verified),
            "best_verified_rule": best.get("rule") if best else None,
            "best_verified_ops_over_rho": charged_over_rho(best) if best else None,
            "best_verified_rank": int(best.get("rank") or 0) if best else 0,
            "best_verified_relation_count": int(best.get("relation_count") or 0) if best else 0,
        },
    }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[int]:
    values = set()
    for row in rows:
        value = row["public_features"].get(key)
        if isinstance(value, int):
            values.add(value)
    return sorted(values)


def candidate_predicates(rows: list[dict[str, Any]]) -> list[str]:
    predicates: set[str] = set()
    for key in NUMERIC_RULE_KEYS:
        for value in numeric_values(rows, key):
            predicates.add(f"{key}>={value}")
            predicates.add(f"{key}<={value}")
            predicates.add(f"{key}={value}")
    for key in STRING_RULE_KEYS:
        values = sorted({str(row["public_features"].get(key) or "") for row in rows})
        for value in values:
            if value:
                predicates.add(f"{key}={value}")
    return sorted(predicates)


def predicate_matches(predicate: str, features: dict[str, Any]) -> bool:
    if ">=" in predicate:
        key, value = predicate.split(">=", 1)
        return int(features.get(key) or 0) >= int(value)
    if "<=" in predicate:
        key, value = predicate.split("<=", 1)
        return int(features.get(key) or 0) <= int(value)
    key, value = predicate.split("=", 1)
    feature_value = features.get(key)
    if isinstance(feature_value, int):
        return int(feature_value) == int(value)
    return str(feature_value or "") == value


def clause_matches(clause: str, features: dict[str, Any]) -> bool:
    return all(
        predicate_matches(part.strip(), features)
        for part in clause.split("&")
        if part.strip()
    )


def rule_matches(rule: str, features: dict[str, Any]) -> bool:
    expression = rule.removeprefix("activate:").strip()
    return any(
        clause_matches(clause, features)
        for clause in expression.split("|")
        if clause.strip()
    )


def candidate_rules(rows: list[dict[str, Any]], max_predicates: int) -> list[str]:
    predicates = candidate_predicates(rows)
    clauses: set[str] = set(predicates)
    for width in range(2, max(2, int(max_predicates)) + 1):
        for combo in combinations(predicates, width):
            clauses.add("&".join(combo))
    return [f"activate:{clause}" for clause in sorted(clauses)]


def row_label(row: dict[str, Any], label_mode: str) -> bool:
    labels = row.get("labels") or {}
    if label_mode == "orientation_success":
        return bool(labels.get("orientation_success"))
    if label_mode == "measured_below_rho":
        value = labels.get("best_verified_ops_over_rho")
        return bool(labels.get("orientation_success")) and value is not None and float(value) < 1.0
    raise ValueError(f"unknown label mode: {label_mode}")


def evaluate_rule(rule: str, rows: list[dict[str, Any]], label_mode: str) -> dict[str, Any]:
    case_results = []
    for row in rows:
        matched = rule_matches(rule, row["public_features"])
        success = row_label(row, label_mode)
        case_results.append(
            {
                "label": row["label"],
                "bucket": row["bucket"],
                "transfer_index": row["transfer_index"],
                "rule_matched": matched,
                "label_success": success,
                "orientation_success": bool(row["labels"].get("orientation_success")),
                "best_verified_rule": row["labels"].get("best_verified_rule"),
                "best_verified_ops_over_rho": row["labels"].get("best_verified_ops_over_rho"),
                "public_features": row["public_features"],
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


def score(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(result["false_positive_count"]),
        int(result["false_negative_count"]),
        -int(result["true_positive_count"]),
        int(result["selected_count"]),
        len(str(result["rule"])),
        str(result["rule"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--max-predicates", type=int, default=2)
    parser.add_argument("--top-rules", type=int, default=25)
    parser.add_argument("--frozen-rule")
    parser.add_argument(
        "--label-mode",
        choices=["measured_below_rho", "orientation_success"],
        default="measured_below_rho",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [feature_row(*record) for record in args.record]
    evaluations = [
        evaluate_rule(rule, rows, args.label_mode)
        for rule in candidate_rules(rows, max_predicates=max(1, int(args.max_predicates)))
    ]
    evaluations.sort(key=score)
    best = evaluations[0] if evaluations else None
    frozen = evaluate_rule(args.frozen_rule, rows, args.label_mode) if args.frozen_rule else None
    output = {
        "schema": "ecdlp_target67_axis_orientation_feature_rule_miner_v1",
        "method": "public_axis_root_and_xmatch_feature_rule_mining",
        "parameters": {
            "records": [
                {
                    "label": label,
                    "bucket": bucket,
                    "axis_source": str(axis.relative_to(WORKTREE_ROOT)),
                    "orientation_source": str(orientation.relative_to(WORKTREE_ROOT)),
                }
                for label, bucket, axis, orientation in args.record
            ],
            "max_predicates": int(args.max_predicates),
            "frozen_rule": args.frozen_rule,
            "label_mode": args.label_mode,
            "excluded_from_rule_search": [
                "transfer_index",
                "salt",
                "line coefficients",
                "verifier labels",
                "valid_relation",
            ],
            "label_use": "orientation verification labels rank mined rules after public features are built",
        },
        "summary": {
            "record_count": len(rows),
            "success_count": sum(1 for row in rows if row_label(row, args.label_mode)),
            "failure_count": sum(1 for row in rows if not row_label(row, args.label_mode)),
            "orientation_success_count": sum(
                1 for row in rows if bool(row["labels"].get("orientation_success"))
            ),
            "measured_below_rho_success_count": sum(
                1 for row in rows if row_label(row, "measured_below_rho")
            ),
            "candidate_rule_count": len(evaluations),
            "best_rule": best["rule"] if best else None,
            "best_rule_true_positive_count": best["true_positive_count"] if best else 0,
            "best_rule_false_positive_count": best["false_positive_count"] if best else 0,
            "best_rule_false_negative_count": best["false_negative_count"] if best else 0,
            "frozen_rule_true_positive_count": frozen["true_positive_count"] if frozen else None,
            "frozen_rule_false_positive_count": frozen["false_positive_count"] if frozen else None,
            "frozen_rule_false_negative_count": frozen["false_negative_count"] if frozen else None,
            "interpretation": (
                "Freeze best_rule before inspecting future-window verifier labels. "
                "It is an activation gate for line-present target-67 candidates, "
                "not a public line predictor by itself."
            ),
        },
        "feature_records": rows,
        "best_rule_evaluation": best,
        "frozen_rule_evaluation": frozen,
        "top_rules": evaluations[: max(1, int(args.top_rules))],
        "non_claims": [
            "The mined rule is retrospective unless validated on later windows.",
            "This does not prove axis-root line prediction is fully charged below rho.",
            "A selected candidate still needs public replay verification after the frozen gate.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
