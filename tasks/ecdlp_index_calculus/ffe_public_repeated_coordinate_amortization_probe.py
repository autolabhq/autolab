#!/usr/bin/env python3
"""Audit whether repeated public coordinate gates support amortization.

The repeated-coordinate miner can find public monic ``(b,c)`` gates that replay
to verified target-67 secrets, but the measured replays are currently over rho.
This probe groups those gates, dedupes policy/selector aliases down to unique
transfer-secret recoveries, and asks a narrower question: how much of the
measured replay charge would need to be reusable before a repeated coordinate
family could beat rho?

The output is an audit, not a speedup claim.  It does not invent a measured
confirmation timer; it reports direct replay costs, additive source+replay
proxies, and reusable-fraction break-even requirements.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_amortization_probe.json"
REUSABLE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("window must be name|artifact_path")
    return parts[0], Path(parts[1])


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def min_or_none(values: list[float]) -> float | None:
    return round(min(values), 8) if values else None


def max_or_none(values: list[float]) -> float | None:
    return round(max(values), 8) if values else None


def policy_family(policy: str) -> str:
    if "target_cap3" in policy:
        return "target_cap3"
    if "global_cap3" in policy:
        return "global_cap3"
    if "target_cap1" in policy:
        return "target_cap1"
    return policy or "unknown"


def selector_total(leaf_selector: str) -> int | None:
    if "total3" in leaf_selector:
        return 3
    if "total4" in leaf_selector:
        return 4
    return None


def clause_text(clause: tuple[str, ...]) -> str:
    return "&".join(clause)


def matching_clause_texts(
    rule: tuple[tuple[str, ...], ...] | None,
    feature_record: dict[str, Any],
) -> list[str]:
    if rule is None:
        return ["all"]
    matches = [
        clause_text(clause)
        for clause in rule
        if activation_miner.clause_matches(clause, feature_record)
    ]
    return matches


def compact_candidate(record: dict[str, Any], window_name: str) -> dict[str, Any]:
    coordinate = record.get("coordinate") or {}
    salts = sorted(int(salt) for salt in record.get("salts") or [])
    leaf_indices = sorted(int(leaf) for leaf in record.get("leaf_indices") or [])
    leaf_selector = str(record.get("leaf_selector") or "")
    policy = str(record.get("policy") or "")
    return {
        "window": window_name,
        "target": str(record.get("target") or ""),
        "transfer_index": int(record.get("transfer_index") or 0),
        "top_k": int(record.get("top_k") or 0),
        "policy": policy,
        "policy_family": policy_family(policy),
        "leaf_selector": leaf_selector,
        "leaf_total": selector_total(leaf_selector),
        "coordinate": {
            "b": int(record.get("b") if record.get("b") is not None else coordinate.get("b") or 0),
            "c": int(record.get("c") if record.get("c") is not None else coordinate.get("c") or 0),
        },
        "p": int(record.get("p") or 9803),
        "row_count": int(record.get("row_count") or 0),
        "salt_count": int(record.get("salt_count") or 0),
        "profile_count": int(record.get("profile_count") or 0),
        "leaf_signature": ",".join(str(leaf) for leaf in leaf_indices),
        "salt_signature": ",".join(str(salt) for salt in salts),
        "source_ops_over_rho": round_or_none(record.get("source_ops_over_rho")),
        "source_public_key_verified": bool(record.get("source_public_key_verified")),
        "source_rank": int(record.get("source_rank") or 0),
        "source_relation_count": int(record.get("source_relation_count") or 0),
        "case_key": record.get("case_key"),
    }


def gate_entries(
    record: dict[str, Any],
    window_name: str,
    source_path: str,
    rule: tuple[tuple[str, ...], ...] | None,
    include_axis_gates: bool,
) -> list[dict[str, Any]]:
    feature_record = activation_miner.feature_record(record, window_name, source_path)
    clauses = matching_clause_texts(rule, feature_record)
    activated = bool(clauses)
    candidate = compact_candidate(record, window_name)
    b_value = int(candidate["coordinate"]["b"])
    c_value = int(candidate["coordinate"]["c"])
    replay_specs = [
        ("exact_coordinate", f"({b_value},{c_value})", record.get("exact_coordinate_replay") or {})
    ]
    if include_axis_gates:
        replay_specs.extend(
            [
                ("b_axis", f"b={b_value}", record.get("b_axis_replay") or {}),
                ("c_axis", f"c={c_value}", record.get("c_axis_replay") or {}),
            ]
        )
    entries: list[dict[str, Any]] = []
    for gate_kind, gate_label, replay in replay_specs:
        if not replay:
            continue
        verified = bool(replay.get("public_key_verified"))
        ops = round_or_none(replay.get("ops_over_rho"))
        source_ops = candidate.get("source_ops_over_rho")
        additive = (
            round(float(source_ops) + float(ops), 8)
            if source_ops is not None and ops is not None
            else None
        )
        entries.append(
            {
                **candidate,
                "gate_kind": gate_kind,
                "gate_label": gate_label,
                "activation_selected": activated,
                "matching_activation_clauses": clauses,
                "replay_public_key_verified": verified,
                "replay_below_rho": verified and bool(replay.get("below_rho")),
                "replay_rank": replay.get("rank"),
                "replay_relation_count": replay.get("relation_count"),
                "replay_derived_secret": replay.get("derived_secret"),
                "replay_ops_over_rho": ops,
                "source_plus_replay_ops_over_rho": additive,
            }
        )
    return entries


def unique_recovery_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("gate_kind"),
        entry.get("gate_label"),
        entry.get("target"),
        int(entry.get("transfer_index") or 0),
        entry.get("replay_derived_secret"),
    )


def unique_success_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        if not entry.get("replay_public_key_verified"):
            continue
        key = unique_recovery_key(entry)
        current = best.get(key)
        current_ops = current.get("replay_ops_over_rho") if current else None
        entry_ops = entry.get("replay_ops_over_rho")
        if current is None or (
            entry_ops is not None
            and (current_ops is None or float(entry_ops) < float(current_ops))
        ):
            best[key] = entry
    return sorted(
        best.values(),
        key=lambda item: (
            str(item.get("gate_kind")),
            str(item.get("gate_label")),
            str(item.get("target")),
            int(item.get("transfer_index") or 0),
            int(item.get("replay_derived_secret") or -1),
        ),
    )


def required_reusable_charge(max_cost: float, unique_success_count: int) -> float | None:
    if max_cost < 1.0:
        return 0.0
    if unique_success_count <= 1:
        return None
    return (max_cost - 1.0) / (1.0 - (1.0 / unique_success_count))


def fraction_needed(max_cost: float, unique_success_count: int) -> float | None:
    reusable = required_reusable_charge(max_cost, unique_success_count)
    if reusable is None:
        return None
    if max_cost <= 0:
        return None
    return reusable / max_cost


def reuse_count_needed(max_cost: float, reusable_fraction: float) -> int | None:
    if max_cost < 1.0:
        return 1
    fixed = max_cost * reusable_fraction
    variable = max_cost - fixed
    if fixed <= 0 or variable >= 1.0:
        return None
    threshold = fixed / (1.0 - variable)
    # Need n > threshold because strict below-rho requires fixed/n below the
    # remaining gap.
    return max(2, math.floor(threshold) + 1)


def break_even_table(max_cost: float | None, unique_success_count: int) -> list[dict[str, Any]]:
    if max_cost is None:
        return []
    rows = []
    for fraction in REUSABLE_FRACTIONS:
        needed = reuse_count_needed(max_cost, fraction)
        rows.append(
            {
                "reusable_fraction": fraction,
                "unique_success_count_needed": needed,
                "observed_unique_success_count": unique_success_count,
                "observed_count_satisfies": needed is not None and unique_success_count >= needed,
            }
        )
    return rows


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "window",
        "target",
        "transfer_index",
        "top_k",
        "policy",
        "leaf_selector",
        "coordinate",
        "gate_kind",
        "gate_label",
        "matching_activation_clauses",
        "leaf_signature",
        "salt_signature",
        "source_ops_over_rho",
        "replay_derived_secret",
        "replay_ops_over_rho",
        "source_plus_replay_ops_over_rho",
        "replay_below_rho",
    )
    return {key: entry.get(key) for key in keys}


def group_summary(group_key: tuple[Any, ...], entries: list[dict[str, Any]]) -> dict[str, Any]:
    success_entries = [entry for entry in entries if entry.get("replay_public_key_verified")]
    unique_successes = unique_success_entries(entries)
    replay_costs = [
        float(entry["replay_ops_over_rho"])
        for entry in unique_successes
        if entry.get("replay_ops_over_rho") is not None
    ]
    additive_costs = [
        float(entry["source_plus_replay_ops_over_rho"])
        for entry in unique_successes
        if entry.get("source_plus_replay_ops_over_rho") is not None
    ]
    max_replay = max(replay_costs) if replay_costs else None
    max_additive = max(additive_costs) if additive_costs else None
    unique_count = len(unique_successes)
    activation_clauses = sorted(
        {
            clause
            for entry in entries
            for clause in (entry.get("matching_activation_clauses") or [])
        }
    )
    policy_counts = Counter(str(entry.get("policy_family")) for entry in entries)
    transfer_mod8_counts = Counter(str(int(entry.get("transfer_index") or 0) % 8) for entry in entries)
    replay_fraction = (
        fraction_needed(max_replay, unique_count) if max_replay is not None else None
    )
    additive_fraction = (
        fraction_needed(max_additive, unique_count) if max_additive is not None else None
    )
    return {
        "group_key": list(group_key),
        "gate_kind": group_key[0],
        "gate_label": group_key[1],
        "activation_clauses": activation_clauses,
        "candidate_record_count": len(entries),
        "activated_record_count": sum(1 for entry in entries if entry.get("activation_selected")),
        "verified_record_count": len(success_entries),
        "unique_verified_recovery_count": unique_count,
        "verified_below_rho_record_count": sum(
            1 for entry in success_entries if entry.get("replay_below_rho")
        ),
        "unique_verified_below_rho_count": sum(
            1 for entry in unique_successes if entry.get("replay_below_rho")
        ),
        "windows": sorted({str(entry.get("window")) for entry in entries}),
        "transfer_indices": sorted({int(entry.get("transfer_index") or 0) for entry in entries}),
        "derived_secrets": sorted(
            {
                int(entry["replay_derived_secret"])
                for entry in success_entries
                if entry.get("replay_derived_secret") is not None
            }
        ),
        "policy_family_counts": dict(sorted(policy_counts.items())),
        "transfer_mod8_counts": dict(sorted(transfer_mod8_counts.items())),
        "min_unique_replay_ops_over_rho": min_or_none(replay_costs),
        "mean_unique_replay_ops_over_rho": mean_or_none(replay_costs),
        "max_unique_replay_ops_over_rho": max_or_none(replay_costs),
        "min_unique_source_plus_replay_ops_over_rho": min_or_none(additive_costs),
        "max_unique_source_plus_replay_ops_over_rho": max_or_none(additive_costs),
        "direct_replay_all_unique_below_rho": bool(replay_costs) and max(replay_costs) < 1.0,
        "replay_charge_required_reusable_fraction": (
            round(replay_fraction, 8) if replay_fraction is not None else None
        ),
        "replay_charge_reusable_fraction_feasible_in_principle": (
            replay_fraction is not None and replay_fraction <= 1.0
        ),
        "source_plus_replay_required_reusable_fraction": (
            round(additive_fraction, 8) if additive_fraction is not None else None
        ),
        "source_plus_replay_reusable_fraction_feasible_in_principle": (
            additive_fraction is not None and additive_fraction <= 1.0
        ),
        "replay_charge_break_even_table": break_even_table(max_replay, unique_count),
        "source_plus_replay_break_even_table": break_even_table(max_additive, unique_count),
        "unique_successes": [compact_entry(entry) for entry in unique_successes],
        "sample_records": [compact_entry(entry) for entry in entries[:16]],
    }


def summarize(groups: list[dict[str, Any]], all_entries: list[dict[str, Any]]) -> dict[str, Any]:
    activated_entries = [entry for entry in all_entries if entry.get("activation_selected")]
    verified_entries = [entry for entry in all_entries if entry.get("replay_public_key_verified")]
    unique_successes = unique_success_entries(all_entries)
    activated_unique_successes = unique_success_entries(activated_entries)
    groups_with_unique_two = [
        group for group in groups if int(group["unique_verified_recovery_count"]) >= 2
    ]
    groups_direct_below = [
        group for group in groups if bool(group["direct_replay_all_unique_below_rho"])
    ]
    replay_feasible = [
        group for group in groups if bool(group["replay_charge_reusable_fraction_feasible_in_principle"])
    ]
    additive_feasible = [
        group
        for group in groups
        if bool(group["source_plus_replay_reusable_fraction_feasible_in_principle"])
    ]
    return {
        "gate_record_count": len(all_entries),
        "activated_gate_record_count": len(activated_entries),
        "verified_gate_record_count": len(verified_entries),
        "unique_verified_recovery_count": len(unique_successes),
        "activated_unique_verified_recovery_count": len(activated_unique_successes),
        "group_count": len(groups),
        "groups_with_at_least_two_unique_recoveries": len(groups_with_unique_two),
        "groups_direct_replay_below_rho": len(groups_direct_below),
        "groups_replay_charge_feasible_with_reuse": len(replay_feasible),
        "groups_source_plus_replay_feasible_with_reuse": len(additive_feasible),
        "best_replay_reusable_fraction_needed": min_or_none(
            [
                float(group["replay_charge_required_reusable_fraction"])
                for group in replay_feasible
                if group.get("replay_charge_required_reusable_fraction") is not None
            ]
        ),
        "best_source_plus_replay_reusable_fraction_needed": min_or_none(
            [
                float(group["source_plus_replay_required_reusable_fraction"])
                for group in additive_feasible
                if group.get("source_plus_replay_required_reusable_fraction") is not None
            ]
        ),
        "top_replay_reuse_candidates": [
            {
                "gate_kind": group["gate_kind"],
                "gate_label": group["gate_label"],
                "unique_verified_recovery_count": group["unique_verified_recovery_count"],
                "max_unique_replay_ops_over_rho": group["max_unique_replay_ops_over_rho"],
                "replay_charge_required_reusable_fraction": group[
                    "replay_charge_required_reusable_fraction"
                ],
                "transfer_indices": group["transfer_indices"],
                "derived_secrets": group["derived_secrets"],
            }
            for group in sorted(
                replay_feasible,
                key=lambda item: (
                    float(item.get("replay_charge_required_reusable_fraction") or 10**9),
                    str(item.get("gate_kind")),
                    str(item.get("gate_label")),
                ),
            )[:8]
        ],
        "interpretation": (
            "direct replay below rho is measured evidence; reusable-fraction "
            "feasibility is only a break-even audit over the measured cost, not "
            "proof that the cost actually decomposes into reusable and per-row parts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--activated-only", action="store_true")
    parser.add_argument("--include-axis-gates", action="store_true")
    parser.add_argument("--top-groups", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rule = args.activation_rule
    entries: list[dict[str, Any]] = []
    candidate_count = 0
    for window_name, path in args.window:
        artifact = load_json(path)
        for record in artifact.get("candidates") or []:
            if not isinstance(record, dict):
                continue
            candidate_count += 1
            for entry in gate_entries(
                record,
                window_name,
                str(path),
                rule,
                bool(args.include_axis_gates),
            ):
                if args.activated_only and not entry.get("activation_selected"):
                    continue
                entries.append(entry)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["gate_kind"], entry["gate_label"])].append(entry)

    groups = [
        group_summary(key, sorted(rows, key=lambda item: json.dumps(compact_entry(item), sort_keys=True)))
        for key, rows in grouped.items()
    ]
    groups = sorted(
        groups,
        key=lambda group: (
            -int(group["unique_verified_recovery_count"]),
            float(group.get("replay_charge_required_reusable_fraction") or 10**9),
            str(group["gate_kind"]),
            str(group["gate_label"]),
        ),
    )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_amortization_probe_v1",
        "method": "group_public_coordinate_and_axis_gates_with_unique_transfer_secret_break_even_audit",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "activation_rule": (
                activation_miner.rule_text(rule) if rule is not None else None
            ),
            "activated_only": bool(args.activated_only),
            "include_axis_gates": bool(args.include_axis_gates),
            "top_groups": int(args.top_groups),
            "reusable_fractions": list(REUSABLE_FRACTIONS),
        },
        "input_candidate_count": candidate_count,
        "summary": summarize(groups, entries),
        "groups": groups[: int(args.top_groups)],
        "group_count_before_truncation": len(groups),
        "groups_truncated": len(groups) > int(args.top_groups),
        "non_claims": [
            "This probe does not measure a separate coordinate-confirmation timer.",
            "Policy, selector, and top-k aliases are deduped to unique transfer-secret recoveries for break-even counts.",
            "Reusable-fraction feasibility is not a speedup unless a later implementation proves that much of the measured charge is reusable.",
            "source_plus_replay is an additive upper-bound proxy and may double count work already included in replay measurements.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
