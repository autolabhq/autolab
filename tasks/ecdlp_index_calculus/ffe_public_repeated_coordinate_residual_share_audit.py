#!/usr/bin/env python3
"""Audit shared residual charges for repeated-coordinate near misses.

The repeated-coordinate decompositions already identify relation-bearing row
subsets.  Some fresh target-67 subsets are just above rho because each row pays
the same selected-leaf / hit-event filter charges independently.  This probe
does not select rows and does not claim a speedup.  It builds a transparent
ledger for public grouping keys and small shared-charge models so follow-up
FFE/summation-polynomial work has a concrete target to validate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_residual_share_audit.json"
SALT_RE = re.compile(r":salt(\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_named_path(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("value must be name|path")
    return parts[0], Path(parts[1])


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def salt_from_row_key(row_key: str) -> int | None:
    match = SALT_RE.search(row_key)
    return int(match.group(1)) if match else None


def signature(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def event_signature(row: dict[str, Any]) -> dict[str, str]:
    events = [event for event in row.get("event_summaries") or [] if isinstance(event, dict)]
    return {
        "candidate_pos_signature": signature(sorted(as_int(event.get("candidate_pos"), -1) for event in events)),
        "scheduled_trial_signature": signature(sorted(as_int(event.get("scheduled_trial"), -1) for event in events)),
        "term_shape_signature": signature(sorted(str(event.get("term_shape") or "") for event in events)),
        "factor_support_signature": signature(
            sorted(signature(event.get("factor_support") or []) for event in events)
        ),
        "term_signature": signature(sorted(signature(event.get("terms") or []) for event in events)),
    }


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    ops = as_int(row.get("preassociation_filter_ops"))
    leaf_count = as_int(row.get("selected_leaf_count"))
    hit_events = as_int(row.get("selected_hit_events"))
    hit_roots = as_int(row.get("selected_hit_roots"))
    core_including_roots = ops - leaf_count - (2 * hit_events)
    return {
        "row_key": row.get("row_key"),
        "salt": salt_from_row_key(str(row.get("row_key") or "")),
        "selected_leaf_indices": [as_int(leaf) for leaf in row.get("selected_leaf_indices") or []],
        "selected_leaf_count": leaf_count,
        "selected_hit_roots": hit_roots,
        "selected_hit_events": hit_events,
        "candidate_verifications": as_int(row.get("candidate_verifications")),
        "relation_event_count": as_int(row.get("relation_event_count")),
        "preassociation_filter_ops": ops,
        "core_ops_including_hit_roots": core_including_roots,
        "core_ops_excluding_hit_roots": core_including_roots - hit_roots,
        "event_signature": event_signature(row),
    }


def best_subset_rows(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    best = recovery.get("best_verified_subset") or {}
    wanted = {str(row_key) for row_key in best.get("row_keys") or []}
    rows = []
    for row in recovery.get("event_positive_rows") or []:
        row_key = str(row.get("row_key") or "")
        if row_key in wanted:
            rows.append(row_summary(row))
    return rows


def public_leaf_signature(rows: list[dict[str, Any]]) -> str:
    return signature(sorted({leaf for row in rows for leaf in row["selected_leaf_indices"]}))


def diagnostic_event_group_signature(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "term_shape_signature": signature(
            sorted({row["event_signature"]["term_shape_signature"] for row in rows})
        ),
        "factor_support_signature": signature(
            sorted({row["event_signature"]["factor_support_signature"] for row in rows})
        ),
        "candidate_pos_signature": signature(
            sorted(row["event_signature"]["candidate_pos_signature"] for row in rows)
        ),
        "scheduled_trial_signature": signature(
            sorted(row["event_signature"]["scheduled_trial_signature"] for row in rows)
        ),
    }


def shared_duplicate_sum(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return sum(values) - max(values)


def modeled_ops(
    baseline_ops: int,
    rows: list[dict[str, Any]],
    share_leaf: bool,
    share_hit_root: bool,
    share_one_event_pass: bool,
    share_two_event_passes: bool,
) -> tuple[int, dict[str, int]]:
    leaf_share = shared_duplicate_sum([as_int(row.get("selected_leaf_count")) for row in rows]) if share_leaf else 0
    hit_root_share = shared_duplicate_sum([as_int(row.get("selected_hit_roots")) for row in rows]) if share_hit_root else 0
    event_duplicate = shared_duplicate_sum([as_int(row.get("selected_hit_events")) for row in rows])
    one_event_share = event_duplicate if share_one_event_pass else 0
    two_event_share = event_duplicate if share_two_event_passes else 0
    total_shared = leaf_share + hit_root_share + one_event_share + two_event_share
    return (
        baseline_ops - total_shared,
        {
            "shared_leaf_ops": leaf_share,
            "shared_hit_root_ops": hit_root_share,
            "shared_one_event_pass_ops": one_event_share,
            "shared_second_event_pass_ops": two_event_share,
            "total_shared_ops": total_shared,
        },
    )


def model_record(
    name: str,
    baseline_ops: int,
    rho: int,
    rows: list[dict[str, Any]],
    share_leaf: bool = False,
    share_hit_root: bool = False,
    share_one_event_pass: bool = False,
    share_two_event_passes: bool = False,
    public_before_events: bool = False,
) -> dict[str, Any]:
    ops, shares = modeled_ops(
        baseline_ops,
        rows,
        share_leaf=share_leaf,
        share_hit_root=share_hit_root,
        share_one_event_pass=share_one_event_pass,
        share_two_event_passes=share_two_event_passes,
    )
    return {
        "model": name,
        "ops": ops,
        "ops_over_rho": clean_ratio(ops, rho),
        "beats_rho": bool(rho and ops < rho),
        "ops_gap_to_rho": ops - rho if rho else None,
        "public_before_events": bool(public_before_events),
        **shares,
    }


def recovery_record(source_name: str, source_path: Path, recovery: dict[str, Any]) -> dict[str, Any] | None:
    best = recovery.get("best_verified_subset") or {}
    if not best.get("public_key_verified"):
        return None
    rows = best_subset_rows(recovery)
    if not rows:
        return None
    coordinate = recovery.get("coordinate") or {}
    rho = as_int(best.get("generic_rho_steps") or (recovery.get("full_replay") or {}).get("generic_rho_steps"))
    baseline_ops = as_int(best.get("ops"))
    if not baseline_ops:
        baseline_ops = sum(as_int(row.get("preassociation_filter_ops")) for row in rows)
    public_group_key = {
        "target": recovery.get("target"),
        "coordinate_key": f"{as_int(coordinate.get('b'))},{as_int(coordinate.get('c'))}",
        "top_k": as_int(recovery.get("top_k")),
        "policy": recovery.get("policy"),
        "leaf_selector": recovery.get("leaf_selector"),
        "leaf_signature": public_leaf_signature(rows),
        "selected_row_count": len(rows),
    }
    models = [
        model_record("baseline_independent_rows", baseline_ops, rho, rows, public_before_events=True),
        model_record("share_duplicate_leaf_only", baseline_ops, rho, rows, share_leaf=True, public_before_events=True),
        model_record("share_duplicate_leaf_and_hit_root", baseline_ops, rho, rows, share_leaf=True, share_hit_root=True),
        model_record("share_one_hit_event_pass", baseline_ops, rho, rows, share_one_event_pass=True),
        model_record(
            "share_leaf_hit_root_and_one_event_pass",
            baseline_ops,
            rho,
            rows,
            share_leaf=True,
            share_hit_root=True,
            share_one_event_pass=True,
        ),
        model_record(
            "share_two_hit_event_passes_diagnostic_upper_bound",
            baseline_ops,
            rho,
            rows,
            share_two_event_passes=True,
        ),
    ]
    return {
        "source_name": source_name,
        "source_path": str(source_path),
        "target": recovery.get("target"),
        "transfer_index": as_int(recovery.get("transfer_index")),
        "coordinate": {"b": as_int(coordinate.get("b")), "c": as_int(coordinate.get("c"))},
        "public_group_key": public_group_key,
        "diagnostic_event_group_signature": diagnostic_event_group_signature(rows),
        "generic_rho_steps": rho,
        "baseline_ops": baseline_ops,
        "baseline_ops_over_rho": clean_ratio(baseline_ops, rho),
        "baseline_ops_gap_to_rho": baseline_ops - rho if rho else None,
        "baseline_below_rho": bool(best.get("below_rho")),
        "derived_secret": best.get("derived_secret"),
        "rank": as_int(best.get("rank")),
        "relation_count": as_int(best.get("relation_count")),
        "rows": rows,
        "shared_charge_models": models,
        "best_public_candidate_model": min(
            (model for model in models if model["public_before_events"]),
            key=lambda model: model["ops"],
        ),
        "best_event_share_model": min(models, key=lambda model: model["ops"]),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    near = [record for record in records if record.get("baseline_ops_gap_to_rho") is not None and record["baseline_ops_gap_to_rho"] >= 0]
    event_win = [
        record
        for record in near
        if any(model["model"] == "share_one_hit_event_pass" and model["beats_rho"] for model in record["shared_charge_models"])
    ]
    public_win = [
        record
        for record in near
        if any(model["public_before_events"] and model["beats_rho"] for model in record["shared_charge_models"])
    ]
    gaps = [as_int(record["baseline_ops_gap_to_rho"]) for record in near]
    event_model_ops = [
        model["ops_over_rho"]
        for record in records
        for model in record["shared_charge_models"]
        if model["model"] == "share_one_hit_event_pass" and model["ops_over_rho"] is not None
    ]
    return {
        "input_verified_recovery_count": len(records),
        "near_or_over_rho_verified_recovery_count": len(near),
        "public_before_events_model_below_rho_count": len(public_win),
        "one_event_pass_share_below_rho_count": len(event_win),
        "min_near_baseline_ops_gap_to_rho": min(gaps) if gaps else None,
        "mean_one_event_pass_share_ops_over_rho": round(mean(event_model_ops), 8) if event_model_ops else None,
        "interpretation": (
            "Public-before-events models are selector-safe but may be too weak. "
            "Event-pass sharing is a concrete FFE/summation-polynomial target, "
            "but it remains diagnostic until the shared computation is derived "
            "from public row/leaf data before relation labels are inspected."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=parse_named_path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for source_name, source_path in args.decomposition:
        artifact = load_json(source_path)
        for recovery in artifact.get("recoveries") or []:
            if not isinstance(recovery, dict):
                continue
            record = recovery_record(source_name, source_path, recovery)
            if record is not None:
                records.append(record)

    output = {
        "schema": "ecdlp_repeated_coordinate_residual_share_audit_v1",
        "method": "diagnostic_shared_charge_models_for_repeated_coordinate_relation_rows",
        "parameters": {
            "decompositions": [{"name": name, "path": str(path)} for name, path in args.decomposition],
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                as_int(record.get("baseline_ops_gap_to_rho")),
                str((record.get("public_group_key") or {}).get("coordinate_key") or ""),
                as_int(record.get("transfer_index")),
            ),
        ),
        "non_claims": [
            "The shared event-pass models are cost ledgers, not a public selector.",
            "Rows still come from verifier-informed decomposition unless paired with a frozen public row rule.",
            "A promoted result must materialize the shared FFE/summation-polynomial computation and replay below rho.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
