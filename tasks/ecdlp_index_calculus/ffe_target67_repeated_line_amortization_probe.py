#!/usr/bin/env python3
"""Test whether repeated target-67 FFE lines support two-row amortization.

The line-stage audit shows a theoretical break-even: if exact line
confirmation can be reused across two replay-success rows, then
``line_confirmation / 2 + replay`` beats rho.  This probe checks the harder
observed condition: do any public degree-1 line families actually recur with
two verifier-backed replay recoveries?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LINE_STAGE_SOURCE = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_495.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_repeated_line_amortization_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def min_or_none(values: list[float]) -> float | None:
    return round(min(values), 8) if values else None


def max_or_none(values: list[float]) -> float | None:
    return round(max(values), 8) if values else None


def line_key(record: dict[str, Any]) -> tuple[int, int, int, int] | None:
    line = record.get("preserving_line") or {}
    try:
        return (
            int(line["b_coeff"]),
            int(line["c_coeff"]),
            int(line["constant"]),
            int(line["p"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def line_label(key: tuple[int, int, int, int]) -> str:
    b_coeff, c_coeff, constant, p = key
    return f"{b_coeff}*b + {c_coeff}*c + {constant} mod {p}"


def cost_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "bucket": record.get("bucket"),
        "transfer_index": int(record.get("transfer_index") or 0),
        "top_k": int(record.get("top_k") or 0),
        "salt": record.get("salt"),
        "leaf_signature": record.get("leaf_signature"),
        "row_public_key_verified": bool(record.get("row_public_key_verified")),
        "row_rank": int(record.get("row_rank") or 0),
        "row_relation_count": int(record.get("row_relation_count") or 0),
        "line_root_scan_ops_over_rho": record.get("line_root_scan_ops_over_rho"),
        "replay_best_rule": record.get("replay_best_rule"),
        "replay_secret": record.get("replay_secret"),
        "replay_measured_ops_over_rho": record.get("replay_measured_ops_over_rho"),
        "line_plus_replay_ops_over_rho": record.get("line_plus_replay_ops_over_rho"),
        "source": record.get("source"),
    }


def amortized_success_costs(
    success_rows: list[dict[str, Any]],
    fixed_line_cost: float,
    denominator: int,
) -> list[dict[str, Any]]:
    rows = []
    if denominator <= 0:
        return rows
    fixed_share = fixed_line_cost / denominator
    for row in success_rows:
        replay = cost_float(row.get("replay_measured_ops_over_rho"))
        if replay is None:
            continue
        total = replay + fixed_share
        rows.append(
            {
                "transfer_index": int(row.get("transfer_index") or 0),
                "leaf_signature": row.get("leaf_signature"),
                "replay_secret": row.get("replay_secret"),
                "replay_measured_ops_over_rho": round(replay, 8),
                "amortized_line_share_over_rho": round(fixed_share, 8),
                "amortized_line_plus_replay_ops_over_rho": round(total, 8),
                "below_rho": total < 1.0,
            }
        )
    return rows


def group_summary(key: tuple[int, int, int, int], rows: list[dict[str, Any]], required_successes: int) -> dict[str, Any]:
    success_rows = [row for row in rows if row.get("bucket") == "line_present_replay_success"]
    failure_rows = [row for row in rows if row.get("bucket") == "line_present_replay_failure"]
    verified_rows = [row for row in rows if bool(row.get("row_public_key_verified"))]
    root_costs = [
        float(row["line_root_scan_ops_over_rho"])
        for row in rows
        if cost_float(row.get("line_root_scan_ops_over_rho")) is not None
    ]
    replay_costs = [
        float(row["replay_measured_ops_over_rho"])
        for row in success_rows
        if cost_float(row.get("replay_measured_ops_over_rho")) is not None
    ]
    conservative_line_cost = max(root_costs) if root_costs else None
    success_only_rows = (
        amortized_success_costs(success_rows, conservative_line_cost, len(success_rows))
        if conservative_line_cost is not None
        else []
    )
    line_present_rows = (
        amortized_success_costs(success_rows, conservative_line_cost, len(rows))
        if conservative_line_cost is not None
        else []
    )
    return {
        "line_key": list(key),
        "line": line_label(key),
        "line_present_count": len(rows),
        "row_verified_count": len(verified_rows),
        "replay_success_count": len(success_rows),
        "replay_failure_count": len(failure_rows),
        "transfer_indices": sorted({int(row.get("transfer_index") or 0) for row in rows}),
        "salts": sorted({int(row["salt"]) for row in rows if row.get("salt") is not None}),
        "leaf_signatures": sorted({str(row.get("leaf_signature")) for row in rows}),
        "min_line_root_scan_ops_over_rho": min_or_none(root_costs),
        "mean_line_root_scan_ops_over_rho": mean_or_none(root_costs),
        "max_line_root_scan_ops_over_rho": max_or_none(root_costs),
        "min_replay_ops_over_rho": min_or_none(replay_costs),
        "max_replay_ops_over_rho": max_or_none(replay_costs),
        "meets_required_success_count": len(success_rows) >= required_successes,
        "repeated_line_present": len(rows) >= required_successes,
        "success_only_amortized_rows": success_only_rows,
        "success_only_amortized_below_rho_count": sum(
            bool(row.get("below_rho")) for row in success_only_rows
        ),
        "line_present_amortized_rows": line_present_rows,
        "line_present_amortized_below_rho_count": sum(
            bool(row.get("below_rho")) for row in line_present_rows
        ),
        "records": [compact_record(row) for row in sorted(rows, key=lambda row: int(row.get("transfer_index") or 0))],
        "interpretation": (
            "success_only amortization counts only rows that actually recover; "
            "line_present amortization shares confirmation over all observed rows "
            "with the same line, including replay failures."
        ),
    }


def summarize(groups: list[dict[str, Any]], required_successes: int) -> dict[str, Any]:
    repeated = [group for group in groups if int(group["line_present_count"]) >= required_successes]
    repeated_success = [
        group for group in groups if int(group["replay_success_count"]) >= required_successes
    ]
    repeated_with_any_success = [
        group
        for group in repeated
        if int(group["replay_success_count"]) > 0
    ]
    success_only_below = [
        group
        for group in repeated_success
        if int(group["success_only_amortized_below_rho_count"]) == int(group["replay_success_count"])
    ]
    line_present_below = [
        group
        for group in repeated_with_any_success
        if int(group["line_present_amortized_below_rho_count"]) == int(group["replay_success_count"])
    ]
    bucket_counts = Counter()
    for group in groups:
        for record in group.get("records") or []:
            bucket_counts[str(record.get("bucket"))] += 1
    return {
        "line_family_count": len(groups),
        "required_successes": required_successes,
        "repeated_line_present_family_count": len(repeated),
        "repeated_line_with_any_success_count": len(repeated_with_any_success),
        "repeated_line_with_required_success_count": len(repeated_success),
        "success_only_amortized_below_rho_family_count": len(success_only_below),
        "line_present_amortized_below_rho_family_count": len(line_present_below),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "repeated_line_labels": [group["line"] for group in repeated],
        "repeated_success_line_labels": [group["line"] for group in repeated_success],
        "line_present_amortized_below_rho_labels": [group["line"] for group in line_present_below],
        "interpretation": (
            "The n=2 break-even is only operational when a repeated line has at "
            "least two replay-success rows.  Repeated line-present families with "
            "one success and one failure are mechanism evidence, not an amortized "
            "ECDLP speedup."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-source", type=Path, default=DEFAULT_LINE_STAGE_SOURCE)
    parser.add_argument("--required-successes", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.line_stage_source)
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in source.get("records") or []:
        if not isinstance(record, dict) or not record.get("has_preserving_line"):
            continue
        key = line_key(record)
        if key is None:
            continue
        grouped[key].append(record)
    groups = [
        group_summary(key, rows, int(args.required_successes))
        for key, rows in sorted(grouped.items(), key=lambda item: line_label(item[0]))
    ]
    output = {
        "schema": "ecdlp_target67_repeated_line_amortization_probe_v1",
        "method": "group_preserving_degree1_lines_and_test_observed_two_row_reuse",
        "parameters": {
            "line_stage_source": str(args.line_stage_source),
            "required_successes": int(args.required_successes),
        },
        "summary": summarize(groups, int(args.required_successes)),
        "line_groups": groups,
        "non_claims": [
            "This probe does not create a new line predictor.",
            "line_present amortization is diagnostic when failed replays share the line; success_only amortization is the stricter recovery criterion.",
            "A repeated line with fewer than two replay successes does not satisfy the n=2 amortized recovery target.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
