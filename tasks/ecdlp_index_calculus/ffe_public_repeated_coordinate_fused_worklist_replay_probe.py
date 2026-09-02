#!/usr/bin/env python3
"""Replay ECDLP relations from a public fused repeated-coordinate worklist.

`ffe_public_repeated_coordinate_fused_worklist_audit.py` emits the public
leaf/root/x-match work a fused shared pass should consume.  This probe consumes
that worklist, reconstructs the compact linear forms emitted by accepted
row-specific second-pass relation instances, derives the target secret, and
recomputes fused work counters from the worklist itself.

This is deliberately one step short of the final algebraic kernel: the compact
relation summaries were produced by the existing scanner, and this script does
not re-evaluate elliptic-curve candidate equality.  It proves that the public
worklist is sufficient for the replay/linear-algebra layer and that the fused
work counters no longer need `charged_ops` or `shared_*_ops` inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_fused_worklist_replay.json"
DEFAULT_FRONTIER_TARGETS = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state/frontier_targets.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def parse_order(target: Any) -> int:
    raw = str(target or "")
    _label, sep, order = raw.rpartition("@")
    if not sep:
        return 0
    return as_int(order)


def load_order_map(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    orders: dict[str, int] = {}
    for item in data.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or "")
        order = as_int(item.get("base_order") or item.get("order"))
        if target and order:
            orders[target] = order
    return orders


def target_order(target: Any, order_map: dict[str, int]) -> tuple[int, str]:
    raw = str(target or "")
    if raw in order_map:
        return order_map[raw], "frontier_targets_base_order"
    parsed = parse_order(raw)
    return parsed, "target_suffix_fallback"


def inv_mod(value: int, modulus: int) -> int:
    return pow(value % modulus, -1, modulus)


def rref_mod(rows: list[list[int]], modulus: int) -> tuple[list[list[int]], list[int], bool]:
    if not rows:
        return [], [], True
    rows = [[value % modulus for value in row] for row in rows]
    nvars = len(rows[0]) - 1
    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(nvars):
        pivot = None
        for row_index in range(pivot_row, len(rows)):
            if rows[row_index][col] % modulus:
                pivot = row_index
                break
        if pivot is None:
            continue

        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = inv_mod(rows[pivot_row][col], modulus)
        rows[pivot_row] = [(value * inverse) % modulus for value in rows[pivot_row]]

        for row_index in range(len(rows)):
            if row_index == pivot_row:
                continue
            scale = rows[row_index][col] % modulus
            if scale:
                rows[row_index] = [
                    (rows[row_index][i] - scale * rows[pivot_row][i]) % modulus
                    for i in range(nvars + 1)
                ]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == len(rows):
            break

    for row in rows:
        if all(row[col] % modulus == 0 for col in range(nvars)) and row[-1] % modulus:
            return rows, pivot_cols, False
    return rows, pivot_cols, True


def derive_secret_from_forms(forms: list[tuple[tuple[int, ...], int, list[int]]], modulus: int) -> dict[str, Any]:
    if not forms or modulus <= 1:
        return {
            "rank": 0,
            "derived": False,
            "derived_secret": None,
            "consistent": True,
        }
    rows = [list(coeffs) + [rhs] for coeffs, rhs, _terms in forms]
    reduced, pivot_cols, consistent = rref_mod(rows, modulus)
    if not consistent or 0 not in pivot_cols:
        return {
            "rank": len(pivot_cols),
            "derived": False,
            "derived_secret": None,
            "consistent": consistent,
        }

    nvars = len(rows[0]) - 1
    free_cols = set(range(nvars)) - set(pivot_cols)
    row_index = pivot_cols.index(0)
    if any(reduced[row_index][col] % modulus for col in free_cols):
        return {
            "rank": len(pivot_cols),
            "derived": False,
            "derived_secret": None,
            "consistent": consistent,
        }
    return {
        "rank": len(pivot_cols),
        "derived": True,
        "derived_secret": reduced[row_index][-1] % modulus,
        "consistent": consistent,
    }


def relation_summaries(record: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    seen: set[tuple[Any, ...]] = set()
    for event in record.get("event_worklist") or []:
        if not isinstance(event, dict):
            continue
        for summary in event.get("relation_summaries") or []:
            if not isinstance(summary, dict):
                continue
            key = (
                as_int(summary.get("q_coeff")),
                as_int(summary.get("rhs")),
                tuple(as_int(term) for term in summary.get("terms") or []),
                as_int(summary.get("scheduled_trial")),
                as_int(summary.get("original_trial")),
            )
            if key in seen:
                continue
            seen.add(key)
            summaries.append(summary)
    return summaries


def factor_variable_count(summaries: list[dict[str, Any]]) -> int:
    max_index = -1
    for summary in summaries:
        for term in summary.get("terms") or []:
            max_index = max(max_index, as_int(term, -1))
        for index in summary.get("factor_support") or []:
            max_index = max(max_index, as_int(index, -1))
    return max_index + 1 if max_index >= 0 else 0


def summary_to_form(
    summary: dict[str, Any],
    factor_count: int,
    modulus: int,
) -> tuple[tuple[int, ...], int, list[int]]:
    terms = [as_int(term) for term in summary.get("terms") or []]
    coeffs = [0] * (1 + factor_count)
    coeffs[0] = as_int(summary.get("q_coeff")) % modulus
    for term in terms:
        if 0 <= term < factor_count:
            coeffs[1 + term] = (coeffs[1 + term] - 1) % modulus
    return tuple(coeffs), as_int(summary.get("rhs")) % modulus, terms


def replay_forms(record: dict[str, Any], order_map: dict[str, int]) -> dict[str, Any]:
    public_group = record.get("public_group_key") or {}
    replay = record.get("replay") or {}
    target_modulus, order_source = target_order(public_group.get("target"), order_map)
    summaries = relation_summaries(record)
    factor_count = factor_variable_count(summaries)
    forms = [summary_to_form(summary, factor_count, target_modulus) for summary in summaries]
    derived = derive_secret_from_forms(forms, target_modulus)
    recorded_secret = replay.get("derived_secret")
    return {
        "target_order": target_modulus,
        "target_order_source": order_source,
        "factor_variable_count": factor_count,
        "relation_summary_count": len(summaries),
        "form_count": len(forms),
        "rank": derived["rank"],
        "consistent": bool(derived["consistent"]),
        "derived": bool(derived["derived"]),
        "derived_secret": derived["derived_secret"],
        "recorded_replay_secret": recorded_secret,
        "matches_recorded_replay_secret": (
            derived["derived_secret"] is not None
            and recorded_secret is not None
            and as_int(derived["derived_secret"]) == as_int(recorded_secret)
        ),
        "recorded_replay_public_key_verified": bool(replay.get("public_key_verified")),
    }


def recompute_work_from_worklist(record: dict[str, Any]) -> dict[str, Any]:
    fused = record.get("fused_worklist") or {}
    rho = as_int((record.get("replay") or {}).get("generic_rho_steps"))
    nonshared_base_ops = as_int(fused.get("nonshared_base_ops"))
    leaf_ops = sum(as_int(item.get("shared_leaf_ops")) for item in record.get("leaf_worklist") or [])
    hit_root_ops = sum(as_int(item.get("shared_hit_root_ops")) for item in record.get("hit_root_worklist") or [])
    first_event_pass_ops = len(record.get("event_worklist") or [])
    second_event_pass_ops = sum(as_int(item.get("fanout")) for item in record.get("event_worklist") or [])
    fused_ops = nonshared_base_ops + leaf_ops + hit_root_ops + first_event_pass_ops + second_event_pass_ops
    first_pass_saved_ops = sum(as_int(item.get("first_pass_ops_saved")) for item in record.get("event_worklist") or [])
    return {
        "nonshared_base_ops": nonshared_base_ops,
        "shared_leaf_ops": leaf_ops,
        "shared_hit_root_ops": hit_root_ops,
        "shared_first_event_pass_ops": first_event_pass_ops,
        "second_event_pass_ops": second_event_pass_ops,
        "fused_ops": fused_ops,
        "generic_rho_steps": rho,
        "fused_ops_over_rho": clean_ratio(fused_ops, rho),
        "fused_below_rho": bool(rho and fused_ops < rho),
        "matches_worklist_fused_ops": bool(fused_ops == as_int(fused.get("fused_ops"))),
        "first_pass_saved_ops": first_pass_saved_ops,
        "reused_first_pass_event_count": sum(
            1 for item in record.get("event_worklist") or [] if bool(item.get("first_pass_reused"))
        ),
    }


def consumed_record(record: dict[str, Any], order_map: dict[str, int]) -> dict[str, Any]:
    replay = replay_forms(record, order_map)
    counters = recompute_work_from_worklist(record)
    ready = str((record.get("fused_worklist") or {}).get("status")) == "ready_for_kernel_implementation"
    consumed = bool(
        ready
        and counters["matches_worklist_fused_ops"]
        and replay["matches_recorded_replay_secret"]
        and replay["recorded_replay_public_key_verified"]
    )
    return {
        "source_name": record.get("source_name"),
        "rule": record.get("rule"),
        "public_group_key": record.get("public_group_key"),
        "selected_row_keys": record.get("selected_row_keys"),
        "consumer_status": "worklist_consumed_and_relation_replayed" if consumed else "worklist_consumer_failed_check",
        "work_counters": counters,
        "relation_form_replay": replay,
        "event_reuse_target": bool(
            counters["fused_below_rho"]
            and as_int(counters.get("first_pass_saved_ops")) > 0
        ),
        "implementation_boundary": (
            "Consumes public fused worklist and derives from compact relation "
            "forms; low-level candidate equality and relation-form emission are "
            "still supplied by the existing scanner artifacts."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    consumed = [record for record in records if record.get("consumer_status") == "worklist_consumed_and_relation_replayed"]
    derived = [record for record in records if bool((record.get("relation_form_replay") or {}).get("derived"))]
    matched = [
        record
        for record in records
        if bool((record.get("relation_form_replay") or {}).get("matches_recorded_replay_secret"))
    ]
    below = [record for record in consumed if bool((record.get("work_counters") or {}).get("fused_below_rho"))]
    event_reuse = [record for record in below if bool(record.get("event_reuse_target"))]
    ratios = [
        (record.get("work_counters") or {}).get("fused_ops_over_rho")
        for record in below
        if (record.get("work_counters") or {}).get("fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "worklist_consumed_count": len(consumed),
        "relation_form_derived_count": len(derived),
        "derived_secret_match_count": len(matched),
        "consumed_below_rho_count": len(below),
        "consumed_event_reuse_below_rho_count": len(event_reuse),
        "mean_consumed_below_rho_ops_over_rho": round(mean(ratios), 8) if ratios else None,
        "interpretation": (
            "The public fused worklist is sufficient to drive the replay "
            "linear-algebra layer and recompute fused work counters.  It still "
            "does not replace the low-level x-match verifier scanner."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-source", type=Path, required=True)
    parser.add_argument("--frontier-targets", type=Path, default=DEFAULT_FRONTIER_TARGETS)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.worklist_source)
    order_map = load_order_map(args.frontier_targets)
    wanted = set(args.source_name or [])
    records = []
    for record in source.get("records") or []:
        if not isinstance(record, dict):
            continue
        if wanted and str(record.get("source_name") or "") not in wanted:
            continue
        records.append(consumed_record(record, order_map))

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_worklist_replay_v1",
        "method": "consume_public_fused_worklist_and_replay_relation_forms",
        "parameters": {
            "worklist_source": str(args.worklist_source),
            "frontier_targets": str(args.frontier_targets),
            "source_names": sorted(wanted),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                not bool(record.get("event_reuse_target")),
                not bool((record.get("work_counters") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This consumes the public worklist and relation-form summaries; it is not yet the low-level fused FFE kernel.",
            "Recorded public-key verification comes from the upstream replay artifact and is checked here by matching the derived secret.",
            "A promoted speedup still requires moving candidate equality and relation-form emission into the fused scanner path.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
