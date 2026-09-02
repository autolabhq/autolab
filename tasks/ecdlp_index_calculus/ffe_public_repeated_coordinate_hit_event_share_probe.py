#!/usr/bin/env python3
"""Audit public hit-event row selection and shared-pass cost models.

The repeated-coordinate decompositions expose two different questions:

1. Can a public row rule select the rows that later recover the target secret?
2. If so, how much cost would a materialized shared FFE event pass need to
   remove before the replay beats generic rho?

This probe uses only row-level public scan counters to select rows.  The replay
verification and relation counts are labels used for auditing the rule, not for
selection.  Shared-pass models are cost ledgers, not a proof that the shared
computation has been implemented.
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
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_hit_event_share_probe.json"
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


def row_key_set(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(row.get("row_key") or "") for row in rows if row.get("row_key")))


def subset_key(row_keys: list[Any]) -> tuple[str, ...]:
    return tuple(sorted(str(row_key) for row_key in row_keys))


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    row_key = str(row.get("row_key") or "")
    leaves = [as_int(leaf) for leaf in row.get("selected_leaf_indices") or []]
    return {
        "row_key": row_key,
        "salt": salt_from_row_key(row_key),
        "selected_leaf_indices": leaves,
        "selected_leaf_signature": signature(sorted(leaves)),
        "selected_leaf_count": as_int(row.get("selected_leaf_count")),
        "selected_hit_roots": as_int(row.get("selected_hit_roots")),
        "selected_hit_root_values": row.get("selected_hit_root_values") or [],
        "selected_hit_events": as_int(row.get("selected_hit_events")),
        "hit_event_summaries": row.get("hit_event_summaries") or [],
        "candidate_verifications": as_int(row.get("candidate_verifications")),
        "relation_event_count": as_int(row.get("relation_event_count")),
        "preassociation_filter_ops": as_int(row.get("preassociation_filter_ops")),
        "ops_over_rho": row.get("ops_over_rho"),
    }


def subset_lookup(recovery: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    lookup: dict[tuple[str, ...], dict[str, Any]] = {}
    for subset in recovery.get("subset_records") or []:
        if not isinstance(subset, dict):
            continue
        lookup[subset_key(subset.get("row_keys") or [])] = subset
    return lookup


def public_group_key(recovery: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    coordinate = recovery.get("coordinate") or {}
    return {
        "target": recovery.get("target"),
        "transfer_index": as_int(recovery.get("transfer_index")),
        "coordinate_key": f"{as_int(coordinate.get('b'))},{as_int(coordinate.get('c'))}",
        "policy": recovery.get("policy"),
        "leaf_selector": recovery.get("leaf_selector"),
        "top_k": as_int(recovery.get("top_k")),
        "selected_row_count": len(rows),
        "leaf_signatures": sorted({str(row.get("selected_leaf_signature") or "") for row in rows}),
        "hit_event_signature": signature(sorted(as_int(row.get("selected_hit_events")) for row in rows)),
        "hit_root_signature": signature(sorted(as_int(row.get("selected_hit_roots")) for row in rows)),
    }


def choose_largest_leaf_hit_event_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if as_int(row.get("selected_hit_events")) > 0:
            grouped[str(row.get("selected_leaf_signature") or "")].append(row)
    if not grouped:
        return []
    return sorted(
        grouped.values(),
        key=lambda group: (
            -len(group),
            -sum(as_int(row.get("selected_hit_events")) for row in group),
            str(group[0].get("selected_leaf_signature") or ""),
        ),
    )[0]


def choose_rows(rows: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    if rule == "all_hit_event_rows":
        return [row for row in rows if as_int(row.get("selected_hit_events")) > 0]
    if rule == "all_hit_root_rows":
        return [row for row in rows if as_int(row.get("selected_hit_roots")) > 0]
    if rule == "largest_leaf_hit_event_group":
        return choose_largest_leaf_hit_event_group(rows)
    if rule == "top2_hit_event_rows":
        return sorted(
            [row for row in rows if as_int(row.get("selected_hit_events")) > 0],
            key=lambda row: (
                -as_int(row.get("selected_hit_events")),
                -as_int(row.get("selected_hit_roots")),
                str(row.get("row_key") or ""),
            ),
        )[:2]
    raise ValueError(f"unknown rule: {rule}")


def shared_duplicate_sum(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return sum(values) - max(values)


def modeled_ops(
    baseline_ops: int,
    rows: list[dict[str, Any]],
    share_leaf: bool = False,
    share_hit_root: bool = False,
    share_one_event_pass: bool = False,
    share_two_event_passes: bool = False,
) -> tuple[int, dict[str, int]]:
    leaf_share = shared_duplicate_sum([as_int(row.get("selected_leaf_count")) for row in rows]) if share_leaf else 0
    hit_root_share = shared_duplicate_sum([as_int(row.get("selected_hit_roots")) for row in rows]) if share_hit_root else 0
    hit_event_duplicate = shared_duplicate_sum([as_int(row.get("selected_hit_events")) for row in rows])
    one_event_share = hit_event_duplicate if share_one_event_pass else 0
    two_event_share = hit_event_duplicate if share_two_event_passes else 0
    total_share = leaf_share + hit_root_share + one_event_share + two_event_share
    return (
        baseline_ops - total_share,
        {
            "shared_leaf_ops": leaf_share,
            "shared_hit_root_ops": hit_root_share,
            "shared_one_event_pass_ops": one_event_share,
            "shared_second_event_pass_ops": two_event_share,
            "total_shared_ops": total_share,
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
        **shares,
    }


def share_models(baseline_ops: int, rho: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        model_record("baseline_selected_rows", baseline_ops, rho, rows),
        model_record("share_duplicate_leaf_only", baseline_ops, rho, rows, share_leaf=True),
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


def model_by_name(models: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for model in models:
        if model.get("model") == name:
            return model
    return None


def charged_replay(
    replay: dict[str, Any],
    charge_model: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if charge_model is None:
        return None
    rho = as_int(replay.get("generic_rho_steps"))
    charged_ops = as_int(charge_model.get("ops"))
    return {
        "charge_policy": charge_model.get("model"),
        "materialization_status": "accounting_only_not_algebraic_kernel",
        "public_key_verified": bool(replay.get("public_key_verified")),
        "derived_secret": replay.get("derived_secret"),
        "rank": as_int(replay.get("rank")),
        "relation_count": as_int(replay.get("relation_count")),
        "baseline_ops": as_int(replay.get("ops")),
        "charged_ops": charged_ops,
        "generic_rho_steps": rho,
        "charged_ops_over_rho": clean_ratio(charged_ops, rho),
        "charged_below_rho": bool(rho and charged_ops < rho),
        "ops_saved": as_int(replay.get("ops")) - charged_ops,
        "shared_leaf_ops": as_int(charge_model.get("shared_leaf_ops")),
        "shared_hit_root_ops": as_int(charge_model.get("shared_hit_root_ops")),
        "shared_one_event_pass_ops": as_int(charge_model.get("shared_one_event_pass_ops")),
        "shared_second_event_pass_ops": as_int(charge_model.get("shared_second_event_pass_ops")),
        "total_shared_ops": as_int(charge_model.get("total_shared_ops")),
    }


def rule_record(
    source_name: str,
    source_path: Path,
    recovery: dict[str, Any],
    rule: str,
    charge_policy: str | None,
) -> dict[str, Any] | None:
    rows = [row_summary(row) for row in recovery.get("row_costs") or [] if isinstance(row, dict)]
    selected = choose_rows(rows, rule)
    if not selected:
        return None
    lookup = subset_lookup(recovery)
    replay = lookup.get(row_key_set(selected))
    if replay is None:
        return None
    rho = as_int(replay.get("generic_rho_steps") or (recovery.get("full_replay") or {}).get("generic_rho_steps"))
    baseline_ops = as_int(replay.get("ops"))
    models = share_models(baseline_ops, rho, selected)
    selected_charge_model = model_by_name(models, charge_policy) if charge_policy else None
    return {
        "source_name": source_name,
        "source_path": str(source_path),
        "rule": rule,
        "selector_stage": "public_ffe_hit_scan_before_relation_labels",
        "public_group_key": public_group_key(recovery, selected),
        "selected_rows": selected,
        "selected_row_keys": list(row_key_set(selected)),
        "replay": {
            "public_key_verified": bool(replay.get("public_key_verified")),
            "below_rho": bool(replay.get("below_rho")),
            "derived_secret": replay.get("derived_secret"),
            "rank": as_int(replay.get("rank")),
            "relation_count": as_int(replay.get("relation_count")),
            "ops": baseline_ops,
            "generic_rho_steps": rho,
            "ops_over_rho": replay.get("ops_over_rho"),
        },
        "shared_charge_models": models,
        "charged_replay": charged_replay(replay, selected_charge_model),
        "best_model": min(models, key=lambda model: as_int(model.get("ops"), 10**9)),
    }


def summarize(records: list[dict[str, Any]], charge_policy: str | None) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rule[str(record.get("rule") or "")].append(record)

    rule_summaries = []
    for rule, rows in sorted(by_rule.items()):
        verified = [row for row in rows if bool((row.get("replay") or {}).get("public_key_verified"))]
        below = [row for row in verified if bool((row.get("replay") or {}).get("below_rho"))]
        one_event = [
            row
            for row in verified
            if any(model["model"] == "share_one_hit_event_pass" and model["beats_rho"] for model in row["shared_charge_models"])
        ]
        leaf_root_event = [
            row
            for row in verified
            if any(
                model["model"] == "share_leaf_hit_root_and_one_event_pass" and model["beats_rho"]
                for model in row["shared_charge_models"]
            )
        ]
        gaps = [
            as_int((row.get("replay") or {}).get("ops")) - as_int((row.get("replay") or {}).get("generic_rho_steps"))
            for row in verified
            if (row.get("replay") or {}).get("ops") is not None
        ]
        one_event_ratios = [
            model["ops_over_rho"]
            for row in verified
            for model in row["shared_charge_models"]
            if model["model"] == "share_one_hit_event_pass" and model["ops_over_rho"] is not None
        ]
        charged = [row for row in verified if isinstance(row.get("charged_replay"), dict)]
        charged_below = [
            row
            for row in charged
            if bool((row.get("charged_replay") or {}).get("charged_below_rho"))
        ]
        summary = {
            "rule": rule,
            "record_count": len(rows),
            "verified_replay_count": len(verified),
            "actual_below_rho_replay_count": len(below),
            "one_event_pass_model_below_rho_count": len(one_event),
            "leaf_root_event_model_below_rho_count": len(leaf_root_event),
            "min_verified_baseline_gap_to_rho": min(gaps) if gaps else None,
            "mean_one_event_pass_ops_over_rho": round(mean(one_event_ratios), 8) if one_event_ratios else None,
        }
        if charge_policy:
            charged_ratios = [
                (row.get("charged_replay") or {}).get("charged_ops_over_rho")
                for row in charged
                if (row.get("charged_replay") or {}).get("charged_ops_over_rho") is not None
            ]
            summary.update(
                {
                    "charge_policy": charge_policy,
                    "charged_replay_count": len(charged),
                    "charged_below_rho_count": len(charged_below),
                    "charged_verified_secret_count": sum(
                        1 for row in charged_below if bool((row.get("charged_replay") or {}).get("public_key_verified"))
                    ),
                    "mean_charged_ops_over_rho": round(mean(charged_ratios), 8) if charged_ratios else None,
                }
            )
        rule_summaries.append(summary)

    return {
        "record_count": len(records),
        "charge_policy": charge_policy,
        "rule_summaries": rule_summaries,
        "interpretation": (
            "Rules select rows from public FFE hit-scan counters.  Replay "
            "verification and shared-pass savings are labels/model ledgers; "
            "charged replay is accounting-only until the shared computation is "
            "implemented as an algebraic kernel."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=parse_named_path, action="append", required=True)
    parser.add_argument(
        "--rule",
        action="append",
        choices=[
            "all_hit_event_rows",
            "all_hit_root_rows",
            "largest_leaf_hit_event_group",
            "top2_hit_event_rows",
        ],
        default=None,
    )
    parser.add_argument(
        "--charge-policy",
        choices=[
            "share_duplicate_leaf_only",
            "share_duplicate_leaf_and_hit_root",
            "share_one_hit_event_pass",
            "share_leaf_hit_root_and_one_event_pass",
            "share_two_hit_event_passes_diagnostic_upper_bound",
        ],
        default=None,
        help="Attach an accounting-only charged replay using the selected shared-charge policy.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rules = args.rule or [
        "all_hit_event_rows",
        "all_hit_root_rows",
        "largest_leaf_hit_event_group",
        "top2_hit_event_rows",
    ]
    records: list[dict[str, Any]] = []
    for source_name, source_path in args.decomposition:
        artifact = load_json(source_path)
        for recovery in artifact.get("recoveries") or []:
            if not isinstance(recovery, dict):
                continue
            for rule in rules:
                record = rule_record(source_name, source_path, recovery, str(rule), args.charge_policy)
                if record is not None:
                    records.append(record)

    output = {
        "schema": "ecdlp_public_repeated_coordinate_hit_event_share_probe_v1",
        "method": "public_hit_event_row_selection_with_shared_pass_cost_ledgers",
        "parameters": {
            "decompositions": [{"name": name, "path": str(path)} for name, path in args.decomposition],
            "rules": rules,
            "charge_policy": args.charge_policy,
        },
        "summary": summarize(records, args.charge_policy),
        "records": sorted(
            records,
            key=lambda record: (
                str(record.get("rule") or ""),
                as_int(((record.get("public_group_key") or {}).get("transfer_index"))),
                str((record.get("public_group_key") or {}).get("coordinate_key") or ""),
                str(record.get("selected_row_keys") or []),
            ),
        ),
        "non_claims": [
            "The row selector uses public FFE hit-scan counters, but the replay verification labels are not part of selection.",
            "Shared event-pass rows are modeled as cost ledgers; this script does not implement a new algebraic shared-pass kernel.",
            "A promoted ECDLP result must replay below rho with the shared computation materialized, not only modeled.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
