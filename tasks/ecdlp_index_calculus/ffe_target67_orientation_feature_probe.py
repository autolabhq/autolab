#!/usr/bin/env python3
"""Summarize target-67 line-orientation features across known replay cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_orientation_feature_probe_328_511.json"
WHERE_RULE = "where:candidate_eq_scheduled"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def case_key(case: dict[str, Any]) -> str:
    return str(case.get("case_key") or "")


def flatten_xmatches(case: dict[str, Any]) -> list[dict[str, Any]]:
    xmatches: list[dict[str, Any]] = []
    for row in case.get("rows") or []:
        for xmatch in row.get("xmatches") or []:
            xmatches.append(dict(xmatch))
    return xmatches


def rule_result(case: dict[str, Any], rule: str) -> dict[str, Any]:
    for result in case.get("rule_results") or []:
        if result.get("rule") == rule:
            return dict(result)
    return {}


def charged_over_rho(result: dict[str, Any], key: str) -> float | None:
    charged = result.get("charged_models") or {}
    value = charged.get(key)
    return float(value) if value is not None else None


def selected_valid_count(result: dict[str, Any]) -> int:
    return sum(1 for xmatch in result.get("selected_xmatches") or [] if xmatch.get("valid_relation"))


def selected_invalid_count(result: dict[str, Any]) -> int:
    return sum(1 for xmatch in result.get("selected_xmatches") or [] if not xmatch.get("valid_relation"))


def verified_rules(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(result)
        for result in case.get("rule_results") or []
        if result.get("public_key_verified")
    ]


def best_verified_rule(case: dict[str, Any]) -> dict[str, Any] | None:
    rules = verified_rules(case)
    if not rules:
        return None
    return min(
        rules,
        key=lambda result: (
            charged_over_rho(result, "measured_oriented_ops_over_rho")
            if charged_over_rho(result, "measured_oriented_ops_over_rho") is not None
            else float("inf"),
            int(result.get("selected_xmatch_count") or 0),
            str(result.get("rule") or ""),
        ),
    )


def stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "values": []}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "values": sorted_values,
    }


def group_feature_stats(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for bucket in sorted({str(row["bucket"]) for row in rows}):
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        grouped[bucket] = {
            key: stats(
                [
                    row[key]
                    for row in bucket_rows
                    if isinstance(row.get(key), int | float)
                ]
            )
            for key in keys
        }
    return grouped


def load_rule_replays(paths: list[Path]) -> dict[str, dict[str, Any]]:
    replays: dict[str, dict[str, Any]] = {}
    for path in paths:
        artifact = load_json(path)
        for case in artifact.get("cases") or []:
            if case_key(case):
                replays[case_key(case)] = {
                    "path": str(path.relative_to(WORKTREE_ROOT)),
                    "summary": artifact.get("summary") or {},
                    "case": case,
                }
    return replays


def default_where_artifacts() -> list[Path]:
    return sorted(
        DEFAULT_STATE_DIR.glob(
            "ffe_public_linear_factor_xmatch_orientation_target67_*where_candidate_eq_scheduled*.json"
        )
    )


def summarize_line_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_line[str(row.get("preserving_line") or "")].append(row)
    families = []
    for line, line_rows in sorted(by_line.items()):
        families.append(
            {
                "line": line,
                "case_count": len(line_rows),
                "success_count": sum(1 for row in line_rows if row["label_replay_success"]),
                "failure_count": sum(1 for row in line_rows if not row["label_replay_success"]),
                "transfers": sorted({int(row["transfer_index"]) for row in line_rows}),
                "leaf_signatures": sorted({str(row["leaf_signature"]) for row in line_rows}),
                "where_candidate_eq_scheduled_verified_count": sum(
                    1 for row in line_rows if row["where_public_key_verified"]
                ),
            }
        )
    return families


def row_from_record(record: dict[str, Any], rule_replays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    replay_path = resolve_path(str(record["replay_source"]))
    replay_artifact = load_json(replay_path)
    cases = replay_artifact.get("cases") or []
    if len(cases) != 1:
        raise ValueError(f"expected one replay case in {replay_path}, found {len(cases)}")
    original_case = cases[0]
    original_key = case_key(original_case)
    original_features = original_case.get("activation_features") or {}
    xmatches = flatten_xmatches(original_case)
    eq_scheduled = [
        xmatch
        for xmatch in xmatches
        if int(xmatch.get("candidate_pos") or -1) == int(xmatch.get("scheduled_trial") or -2)
    ]
    best = best_verified_rule(original_case)
    where_entry = rule_replays.get(original_key, {})
    where_case = where_entry.get("case") or {}
    where_result = rule_result(where_case, WHERE_RULE)
    preserving_line = record.get("preserving_line") or {}
    return {
        "bucket": record.get("bucket"),
        "label_replay_success": record.get("bucket") == "line_present_replay_success",
        "target": record.get("target"),
        "transfer_index": record.get("transfer_index"),
        "top_k": record.get("top_k"),
        "salt": record.get("salt"),
        "leaf_selector": record.get("leaf_selector"),
        "leaf_signature": record.get("leaf_signature"),
        "preserving_line": preserving_line.get("line"),
        "preserving_factor_index": record.get("preserving_factor_index"),
        "line_root_scan_ops_over_rho": record.get("line_root_scan_ops_over_rho"),
        "line_plus_replay_ops_over_rho": record.get("line_plus_replay_ops_over_rho"),
        "row_line_replay_ops_over_rho": record.get("row_line_replay_ops_over_rho"),
        "replay_source": str(replay_path.relative_to(WORKTREE_ROOT)),
        "replay_case_key": original_key,
        "xmatch_count": int(original_features.get("xmatch_count") or len(xmatches)),
        "scheduled1_count": int(original_features.get("scheduled1_count") or 0),
        "factor_zero_profile_count": int(original_features.get("factor_zero_profile_count") or 0),
        "row_xmatch_counts": original_features.get("row_xmatch_counts") or [],
        "valid_xmatch_count": sum(1 for xmatch in xmatches if xmatch.get("valid_relation")),
        "invalid_xmatch_count": sum(1 for xmatch in xmatches if not xmatch.get("valid_relation")),
        "candidate_eq_scheduled_xmatch_count": len(eq_scheduled),
        "candidate_eq_scheduled_valid_count": sum(
            1 for xmatch in eq_scheduled if xmatch.get("valid_relation")
        ),
        "candidate_eq_scheduled_invalid_count": sum(
            1 for xmatch in eq_scheduled if not xmatch.get("valid_relation")
        ),
        "original_verified_rule_count": len(verified_rules(original_case)),
        "original_best_verified_rule": best.get("rule") if best else None,
        "original_best_verified_selected_xmatch_count": (
            int(best.get("selected_xmatch_count") or 0) if best else None
        ),
        "original_best_verified_rank": int(best.get("rank") or 0) if best else None,
        "original_best_verified_ops_over_rho": (
            charged_over_rho(best, "measured_oriented_ops_over_rho") if best else None
        ),
        "where_artifact": where_entry.get("path"),
        "where_rule": WHERE_RULE if where_result else None,
        "where_selected_xmatch_count": int(where_result.get("selected_xmatch_count") or 0),
        "where_selected_valid_relation_count": selected_valid_count(where_result),
        "where_selected_invalid_relation_count": selected_invalid_count(where_result),
        "where_rank": int(where_result.get("rank") or 0),
        "where_relation_count": int(where_result.get("relation_count") or 0),
        "where_public_key_verified": bool(where_result.get("public_key_verified")),
        "where_ops_over_rho": charged_over_rho(where_result, "measured_oriented_ops_over_rho"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--where-artifact", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    line_stage = load_json(args.line_stage_audit)
    where_paths = args.where_artifact or default_where_artifacts()
    rule_replays = load_rule_replays(where_paths)
    rows = [
        row_from_record(record, rule_replays)
        for record in line_stage.get("records") or []
        if record.get("has_preserving_line") and record.get("replay_source")
    ]
    buckets = Counter(str(row["bucket"]) for row in rows)
    where_verified_rows = [row for row in rows if row["where_public_key_verified"]]
    feature_keys = [
        "xmatch_count",
        "scheduled1_count",
        "valid_xmatch_count",
        "candidate_eq_scheduled_xmatch_count",
        "candidate_eq_scheduled_valid_count",
        "where_selected_xmatch_count",
        "where_selected_valid_relation_count",
        "where_rank",
        "original_verified_rule_count",
        "original_best_verified_selected_xmatch_count",
    ]
    output = {
        "schema": "ecdlp_target67_orientation_feature_probe_v1",
        "method": "join_line_stage_audit_original_replays_and_candidate_eq_scheduled_rule_replays",
        "parameters": {
            "line_stage_audit": str(args.line_stage_audit),
            "where_rule": WHERE_RULE,
            "where_artifacts": [str(path.relative_to(WORKTREE_ROOT)) for path in where_paths],
        },
        "summary": {
            "line_present_case_count": len(rows),
            "bucket_counts": dict(sorted(buckets.items())),
            "where_candidate_eq_scheduled_activated_count": sum(
                1 for row in rows if row["where_artifact"]
            ),
            "where_candidate_eq_scheduled_verified_count": len(where_verified_rows),
            "where_candidate_eq_scheduled_success_verified_count": sum(
                1 for row in where_verified_rows if row["label_replay_success"]
            ),
            "where_candidate_eq_scheduled_failure_verified_count": sum(
                1 for row in where_verified_rows if not row["label_replay_success"]
            ),
            "success_case_count": sum(1 for row in rows if row["label_replay_success"]),
            "failure_case_count": sum(1 for row in rows if not row["label_replay_success"]),
            "original_replay_success_verified_count": sum(
                1 for row in rows if row["original_verified_rule_count"]
            ),
            "interpretation": (
                "candidate_eq_scheduled is a public x-match filter that often selects valid "
                "single relations, but it does not produce rank-2 public-key verification on "
                "any known target-67 line-present case in this window."
            ),
        },
        "feature_stats_by_bucket": group_feature_stats(rows, feature_keys),
        "line_families": summarize_line_families(rows),
        "records": rows,
        "non_claims": [
            "This probe is retrospective and does not claim a fresh public selector.",
            "The where-rule replay uses verification labels only after the public filter is fixed.",
            "A target-67 speedup still needs public line prediction, amortized line confirmation, or a rank-aware orientation rule that validates on holdout windows.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
