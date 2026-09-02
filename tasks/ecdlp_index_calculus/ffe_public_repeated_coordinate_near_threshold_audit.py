#!/usr/bin/env python3
"""Audit repeated-coordinate decompositions that land just above rho.

The pair-rule replay path only has a chance to prove a public speedup when the
verifier-informed relation subset is itself below rho.  The 720 and 728 fresh
target-67 windows instead repeat the same coordinate with a two-row relation
system that is only a few operations above rho.  This audit makes that boundary
machine-readable and groups recurring diagnostic form signatures so the next
experiment can target a small FFE/summation-polynomial cost shave instead of
mining another salt-congruence rule blindly.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_near_threshold_audit.json"
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def salt_from_row_key(row_key: str) -> int | None:
    match = SALT_RE.search(row_key)
    return int(match.group(1)) if match else None


def signature(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def event_rows(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in recovery.get("event_positive_rows") or []:
        events = [event for event in row.get("event_summaries") or [] if isinstance(event, dict)]
        rows.append(
            {
                "row_key": row.get("row_key"),
                "salt": salt_from_row_key(str(row.get("row_key") or "")),
                "selected_leaf_indices": [as_int(leaf) for leaf in row.get("selected_leaf_indices") or []],
                "preassociation_filter_ops": as_int(row.get("preassociation_filter_ops")),
                "ops_over_rho": as_float(row.get("ops_over_rho")),
                "relation_event_count": as_int(row.get("relation_event_count")),
                "candidate_pos_values": [as_int(event.get("candidate_pos"), -1) for event in events],
                "scheduled_trial_values": [as_int(event.get("scheduled_trial"), -1) for event in events],
                "term_shapes": sorted(str(event.get("term_shape") or "") for event in events),
                "factor_supports": sorted(signature(event.get("factor_support") or []) for event in events),
                "term_signatures": sorted(signature(event.get("terms") or []) for event in events),
                "event_count": len(events),
            }
        )
    return rows


def compact_recovery(source_name: str, source_path: Path, recovery: dict[str, Any]) -> dict[str, Any]:
    coordinate = recovery.get("coordinate") or {}
    best = recovery.get("best_verified_subset") or {}
    full = recovery.get("full_replay") or {}
    row_keys = [str(row_key) for row_key in best.get("row_keys") or []]
    salts = [salt for salt in (salt_from_row_key(row_key) for row_key in row_keys) if salt is not None]
    rho = as_int(best.get("generic_rho_steps") or full.get("generic_rho_steps"))
    ops = as_int(best.get("ops"))
    rows = event_rows(recovery)
    leaf_signature = signature(sorted({leaf for row in rows for leaf in row["selected_leaf_indices"]}))
    term_shape_signature = signature(sorted({shape for row in rows for shape in row["term_shapes"]}))
    factor_support_signature = signature(sorted({support for row in rows for support in row["factor_supports"]}))
    candidate_pos_signature = signature(sorted(pos for row in rows for pos in row["candidate_pos_values"]))
    scheduled_trial_signature = signature(sorted(pos for row in rows for pos in row["scheduled_trial_values"]))
    return {
        "source_name": source_name,
        "source_path": str(source_path),
        "target": recovery.get("target"),
        "transfer_index": as_int(recovery.get("transfer_index")),
        "coordinate": {"b": as_int(coordinate.get("b")), "c": as_int(coordinate.get("c"))},
        "coordinate_key": f"{as_int(coordinate.get('b'))},{as_int(coordinate.get('c'))}",
        "top_k": as_int(recovery.get("top_k")),
        "policy": recovery.get("policy"),
        "leaf_selector": recovery.get("leaf_selector"),
        "full_ops": as_int(full.get("ops")),
        "full_ops_over_rho": as_float(full.get("ops_over_rho")),
        "best_subset_ops": ops,
        "generic_rho_steps": rho,
        "ops_gap_to_rho": ops - rho if rho else None,
        "best_subset_ops_over_rho": as_float(best.get("ops_over_rho")),
        "best_subset_public_key_verified": bool(best.get("public_key_verified")),
        "best_subset_below_rho": bool(best.get("below_rho")),
        "best_subset_rank": as_int(best.get("rank")),
        "best_subset_relation_count": as_int(best.get("relation_count")),
        "best_subset_row_keys": row_keys,
        "best_subset_salts": sorted(salts),
        "event_positive_row_count": len(rows),
        "event_positive_rows": rows,
        "diagnostic_form_signature": {
            "leaf_signature": leaf_signature,
            "term_shape_signature": term_shape_signature,
            "factor_support_signature": factor_support_signature,
            "candidate_pos_signature": candidate_pos_signature,
            "scheduled_trial_signature": scheduled_trial_signature,
        },
    }


def summarize_groups(records: list[dict[str, Any]], near_ops_gap: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        form = record["diagnostic_form_signature"]
        key = (
            str(record.get("coordinate_key") or ""),
            str(form.get("leaf_signature") or ""),
            str(form.get("term_shape_signature") or ""),
            str(form.get("factor_support_signature") or ""),
        )
        grouped[key].append(record)

    summaries = []
    for (coordinate_key, leaf_sig, term_sig, factor_sig), rows in grouped.items():
        if len(rows) < 2:
            continue
        gaps = [as_int(row.get("ops_gap_to_rho")) for row in rows if row.get("ops_gap_to_rho") is not None]
        near_rows = [row for row in rows if row.get("ops_gap_to_rho") is not None and as_int(row["ops_gap_to_rho"]) <= near_ops_gap]
        summaries.append(
            {
                "coordinate_key": coordinate_key,
                "leaf_signature": leaf_sig,
                "term_shape_signature": term_sig,
                "factor_support_signature": factor_sig,
                "occurrence_count": len(rows),
                "near_threshold_occurrence_count": len(near_rows),
                "transfer_indices": sorted(as_int(row.get("transfer_index")) for row in rows),
                "source_names": [str(row.get("source_name")) for row in rows],
                "best_ops_gap_to_rho": min(gaps) if gaps else None,
                "mean_ops_gap_to_rho": round(mean(gaps), 8) if gaps else None,
                "best_subset_ops_over_rho_values": sorted(
                    value for value in (as_float(row.get("best_subset_ops_over_rho")) for row in rows) if value is not None
                ),
                "best_subset_salt_signatures": sorted(
                    signature(row.get("best_subset_salts") or []) for row in rows
                ),
                "candidate_pos_signatures": sorted(
                    str((row.get("diagnostic_form_signature") or {}).get("candidate_pos_signature") or "")
                    for row in rows
                ),
                "scheduled_trial_signatures": sorted(
                    str((row.get("diagnostic_form_signature") or {}).get("scheduled_trial_signature") or "")
                    for row in rows
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -(item["near_threshold_occurrence_count"]),
            item["best_ops_gap_to_rho"] if item["best_ops_gap_to_rho"] is not None else 10**9,
            item["coordinate_key"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=parse_named_path, action="append", required=True)
    parser.add_argument("--near-ops-gap", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for name, path in args.decomposition:
        artifact = load_json(path)
        for recovery in artifact.get("recoveries") or []:
            if isinstance(recovery, dict):
                records.append(compact_recovery(name, path, recovery))

    near = [
        record
        for record in records
        if record.get("ops_gap_to_rho") is not None
        and 0 <= as_int(record["ops_gap_to_rho"]) <= int(args.near_ops_gap)
    ]
    below = [record for record in records if bool(record.get("best_subset_below_rho")) and bool(record.get("best_subset_public_key_verified"))]
    output = {
        "schema": "ecdlp_repeated_coordinate_near_threshold_audit_v1",
        "method": "diagnostic_grouping_of_verifier_informed_repeated_coordinate_decompositions",
        "parameters": {
            "decompositions": [{"name": name, "path": str(path)} for name, path in args.decomposition],
            "near_ops_gap": int(args.near_ops_gap),
        },
        "summary": {
            "input_recovery_count": len(records),
            "verified_below_rho_subset_count": len(below),
            "near_threshold_over_rho_count": len(near),
            "min_ops_gap_to_rho": min([as_int(record["ops_gap_to_rho"]) for record in near] or [None]),
            "repeated_diagnostic_group_count": len(summarize_groups(records, int(args.near_ops_gap))),
            "interpretation": (
                "Near-threshold records are verifier-informed diagnostics. "
                "A speedup still requires a public predictor and a replay that "
                "beats generic rho without using relation-event labels."
            ),
        },
        "near_threshold_records": sorted(
            near,
            key=lambda record: (
                as_int(record.get("ops_gap_to_rho")),
                str(record.get("coordinate_key") or ""),
                as_int(record.get("transfer_index")),
            ),
        ),
        "repeated_diagnostic_groups": summarize_groups(records, int(args.near_ops_gap)),
        "non_claims": [
            "Event signatures such as factor_support and term_shape are verifier diagnostics, not public selectors by themselves.",
            "This audit identifies small cost gaps and recurring forms; it does not prove an ECDLP speedup.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
