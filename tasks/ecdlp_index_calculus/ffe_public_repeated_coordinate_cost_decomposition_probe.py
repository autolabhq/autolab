#!/usr/bin/env python3
"""Decompose repeated-coordinate replay cost into row-level subset costs.

The coordinate amortization audit showed that the best target-67 coordinate
family still replays over rho as a whole.  This probe reruns selected
coordinate-gate recoveries with full row scan details and evaluates every row
subset, so we can tell whether the over-rho cost comes from the algebraic gate
itself or from carrying extra public rows that do not contribute relations.

Subset wins are diagnostics unless the row-pruning rule is public and frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_cost_decomposition_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_source(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("source must be name|artifact_path")
    return parts[0], Path(parts[1])


def parse_coordinate(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinate must be b,c")
    return int(parts[0]), int(parts[1])


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def replay_args(params: dict[str, Any], event_summary_limit: int) -> argparse.Namespace:
    return argparse.Namespace(
        radius=int(params.get("radius") or 4),
        event_summary_limit=int(event_summary_limit),
        row_pool=512,
        row_count=128,
        scout_limit=192,
        scout_mode="s3_coeff_spread",
        scout_order="eval_cover_hits_high",
        selected_limit=64,
        factor_base_size=16,
        max_relations=96,
        min_distinct_indices=4,
        min_unsigned_distinct_indices=2,
        require_unit_coefficients=True,
        row_factor=512,
        product_factor=4096,
        seed=str(params.get("seed") or "ecdlp-frontier-signed-dual-sieve-v1"),
    )


def profile_row_leaves(
    profiles: list[dict[str, Any]],
    coordinate: tuple[int, int],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    b_value, c_value = coordinate
    grouped: dict[str, set[int]] = {}
    kept: list[dict[str, Any]] = []
    for profile in profiles:
        if int(profile.get("b") or 0) != b_value or int(profile.get("c") or 0) != c_value:
            continue
        row_key = str(profile.get("row_key") or "")
        if not row_key:
            continue
        leaf_index = int(profile.get("leaf_index") or 0)
        grouped.setdefault(row_key, set()).add(leaf_index)
        kept.append(
            {
                "row_key": row_key,
                "salt": profile.get("salt"),
                "surface_id": profile.get("surface_id"),
                "leaf_indices": [leaf_index],
            }
        )
    return grouped, kept


def candidate_verified(candidate: dict[str, Any], gate_kind: str) -> bool:
    replay_key = {
        "exact_coordinate": "exact_coordinate_replay",
        "b_axis": "b_axis_replay",
        "c_axis": "c_axis_replay",
    }[gate_kind]
    return bool((candidate.get(replay_key) or {}).get("public_key_verified"))


def candidate_replay(candidate: dict[str, Any], gate_kind: str) -> dict[str, Any]:
    replay_key = {
        "exact_coordinate": "exact_coordinate_replay",
        "b_axis": "b_axis_replay",
        "c_axis": "c_axis_replay",
    }[gate_kind]
    return candidate.get(replay_key) or {}


def gate_label(coordinate: tuple[int, int], gate_kind: str) -> str:
    b_value, c_value = coordinate
    if gate_kind == "b_axis":
        return f"b={b_value}"
    if gate_kind == "c_axis":
        return f"c={c_value}"
    return f"({b_value},{c_value})"


def candidate_key(candidate: dict[str, Any], coordinate: tuple[int, int], gate_kind: str) -> tuple[Any, ...]:
    replay = candidate_replay(candidate, gate_kind)
    return (
        gate_kind,
        gate_label(coordinate, gate_kind),
        str(candidate.get("target") or ""),
        int(candidate.get("transfer_index") or 0),
        replay.get("derived_secret"),
    )


def candidate_sort_key(candidate: dict[str, Any], gate_kind: str) -> tuple[Any, ...]:
    replay = candidate_replay(candidate, gate_kind)
    return (
        float(replay.get("ops_over_rho") or 10**9),
        float(candidate.get("source_ops_over_rho") or 10**9),
        int(candidate.get("top_k") or 0),
        str(candidate.get("policy") or ""),
        str(candidate.get("leaf_selector") or ""),
        str(candidate.get("case_key") or ""),
    )


def selected_unique_candidates(
    sources: list[tuple[str, dict[str, Any]]],
    coordinate: tuple[int, int],
    gate_kind: str,
    max_recoveries: int,
) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    b_value, c_value = coordinate
    for source_name, artifact in sources:
        for candidate in artifact.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            coord = candidate.get("coordinate") or {}
            if int(coord.get("b") or 0) != b_value or int(coord.get("c") or 0) != c_value:
                continue
            if not candidate_verified(candidate, gate_kind):
                continue
            enriched = dict(candidate)
            enriched["_source_name"] = source_name
            enriched["_artifact_parameters"] = artifact.get("parameters") or {}
            key = candidate_key(enriched, coordinate, gate_kind)
            current = best.get(key)
            if current is None or candidate_sort_key(enriched, gate_kind) < candidate_sort_key(current, gate_kind):
                best[key] = enriched
    selected = sorted(best.values(), key=lambda item: candidate_sort_key(item, gate_kind))
    return selected[:max_recoveries] if max_recoveries > 0 else selected


def row_ops_summary(row: dict[str, Any], generic_rho_steps: int) -> dict[str, Any]:
    scan = row.get("scan") or {}
    ops = int(scan.get("preassociation_filter_ops") or 0)
    return {
        "row_key": row.get("row_key"),
        "challenge_seed": row.get("challenge_seed"),
        "generic_rho_steps": int(row.get("generic_rho_steps") or generic_rho_steps),
        "selected_leaf_indices": scan.get("selected_leaf_indices") or [],
        "selected_leaf_count": scan.get("selected_leaf_count"),
        "selected_hit_roots": int(scan.get("selected_hit_roots") or 0),
        "selected_hit_root_values": scan.get("selected_hit_root_values") or [],
        "selected_hit_events": int(scan.get("selected_hit_events") or 0),
        "hit_event_summaries": scan.get("hit_event_summaries") or [],
        "candidate_verifications": int(scan.get("candidate_verifications") or 0),
        "row_relation_count": int(scan.get("relation_count") or 0),
        "row_rank": int(scan.get("rank") or 0),
        "row_public_key_verified": bool(scan.get("row_public_key_verified")),
        "row_derived_secret": scan.get("derived_secret"),
        "preassociation_filter_ops": ops,
        "ops_over_rho": round_or_none(ops / generic_rho_steps if generic_rho_steps else None),
        "relation_event_count": int(scan.get("event_summary_count") or 0),
        "event_summaries": scan.get("event_summaries") or [],
        "event_summaries_truncated": bool(scan.get("event_summaries_truncated")),
    }


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_row_count": replay.get("selected_row_count"),
        "materialized_row_count": replay.get("materialized_row_count"),
        "selected_leaf_count": replay.get("selected_leaf_count"),
        "relation_count": replay.get("relation_count"),
        "rank": replay.get("rank"),
        "unique_form_count": replay.get("unique_form_count"),
        "duplicate_form_count": replay.get("duplicate_form_count"),
        "public_key_verified": bool(replay.get("public_key_verified")),
        "derived_secret": replay.get("derived_secret"),
        "ops": replay.get("ops"),
        "generic_rho_steps": replay.get("generic_rho_steps"),
        "ops_over_rho": replay.get("ops_over_rho"),
        "below_rho": bool(replay.get("below_rho")),
    }


def subset_records(
    verifier: Any,
    row_leaves: dict[str, set[int]],
    contexts: dict[str, dict[str, Any]],
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
    event_summary_limit: int,
) -> list[dict[str, Any]]:
    row_keys = sorted(row_leaves)
    records: list[dict[str, Any]] = []
    for size in range(1, len(row_keys) + 1):
        for subset in itertools.combinations(row_keys, size):
            subset_leaves = {row_key: set(row_leaves[row_key]) for row_key in subset}
            replay, _events = replay_probe.replay_selection(
                verifier,
                subset_leaves,
                contexts,
                scan_cache,
                event_summary_limit,
            )
            compact = compact_replay(replay)
            compact["row_keys"] = list(subset)
            compact["relation_event_count_sum"] = sum(
                int(((row.get("scan") or {}).get("event_summary_count")) or 0)
                for row in replay.get("rows") or []
            )
            records.append(compact)
    return sorted(
        records,
        key=lambda item: (
            not bool(item.get("public_key_verified")),
            float(item.get("ops_over_rho") or 10**9),
            len(item.get("row_keys") or []),
            item.get("row_keys") or [],
        ),
    )


def recovery_decomposition(
    candidate: dict[str, Any],
    coordinate: tuple[int, int],
    gate_kind: str,
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    event_summary_limit: int,
) -> dict[str, Any]:
    params = candidate.get("_artifact_parameters") or {}
    args = replay_args(params, event_summary_limit)
    row_leaves, kept_profiles = profile_row_leaves(candidate.get("profiles") or [], coordinate)
    case = {
        "target": candidate.get("target"),
        "transfer_index": int(candidate.get("transfer_index") or 0),
        "top_k": int(candidate.get("top_k") or 0),
    }
    contexts, errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        case,
        sorted(row_leaves),
        args,
        context_cache,
    )
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    full_replay, _events = replay_probe.replay_selection(
        verifier,
        row_leaves,
        contexts,
        scan_cache,
        event_summary_limit,
    )
    subsets = subset_records(
        verifier,
        row_leaves,
        contexts,
        scan_cache,
        event_summary_limit,
    )
    generic_rho_steps = int(full_replay.get("generic_rho_steps") or 0)
    row_costs = [
        row_ops_summary(row, generic_rho_steps)
        for row in sorted(full_replay.get("rows") or [], key=lambda item: str(item.get("row_key")))
    ]
    verified_subsets = [row for row in subsets if row.get("public_key_verified")]
    below_rho_verified_subsets = [
        row for row in verified_subsets if bool(row.get("below_rho"))
    ]
    event_positive_rows = [
        row for row in row_costs if int(row.get("relation_event_count") or 0) > 0
    ]
    replay = candidate_replay(candidate, gate_kind)
    return {
        "source_name": candidate.get("_source_name"),
        "case_key": candidate.get("case_key"),
        "target": candidate.get("target"),
        "transfer_index": int(candidate.get("transfer_index") or 0),
        "top_k": int(candidate.get("top_k") or 0),
        "policy": candidate.get("policy"),
        "leaf_selector": candidate.get("leaf_selector"),
        "coordinate": {"b": coordinate[0], "c": coordinate[1]},
        "gate_kind": gate_kind,
        "gate_label": gate_label(coordinate, gate_kind),
        "source_ops_over_rho": candidate.get("source_ops_over_rho"),
        "original_compact_replay": {
            "derived_secret": replay.get("derived_secret"),
            "public_key_verified": bool(replay.get("public_key_verified")),
            "ops_over_rho": replay.get("ops_over_rho"),
            "relation_count": replay.get("relation_count"),
            "rank": replay.get("rank"),
            "selected_row_count": replay.get("selected_row_count"),
            "selected_leaf_count": replay.get("selected_leaf_count"),
        },
        "full_replay": compact_replay(full_replay),
        "row_costs": row_costs,
        "kept_profiles": kept_profiles,
        "context_error_count": len(errors),
        "context_errors": errors,
        "subset_count": len(subsets),
        "verified_subset_count": len(verified_subsets),
        "below_rho_verified_subset_count": len(below_rho_verified_subsets),
        "best_verified_subset": verified_subsets[0] if verified_subsets else None,
        "best_below_rho_verified_subset": (
            below_rho_verified_subsets[0] if below_rho_verified_subsets else None
        ),
        "event_positive_row_count": len(event_positive_rows),
        "event_positive_row_ops_over_rho_sum": round_or_none(
            sum(float(row.get("ops_over_rho") or 0.0) for row in event_positive_rows)
        ),
        "event_positive_rows": event_positive_rows,
        "subset_records": subsets,
        "non_claim": (
            "Subsets are verifier-informed diagnostics unless a public row-pruning "
            "rule selects the same rows before relation events are inspected."
        ),
    }


def summarize(recoveries: list[dict[str, Any]]) -> dict[str, Any]:
    full_costs = [
        float((item.get("full_replay") or {}).get("ops_over_rho"))
        for item in recoveries
        if (item.get("full_replay") or {}).get("ops_over_rho") is not None
    ]
    best_subset_costs = [
        float((item.get("best_verified_subset") or {}).get("ops_over_rho"))
        for item in recoveries
        if (item.get("best_verified_subset") or {}).get("ops_over_rho") is not None
    ]
    event_row_costs = [
        float(item["event_positive_row_ops_over_rho_sum"])
        for item in recoveries
        if item.get("event_positive_row_ops_over_rho_sum") is not None
    ]
    below_subset = [
        item for item in recoveries if item.get("below_rho_verified_subset_count")
    ]
    event_row_below = [
        item
        for item in recoveries
        if item.get("event_positive_row_ops_over_rho_sum") is not None
        and float(item["event_positive_row_ops_over_rho_sum"]) < 1.0
    ]
    row_event_counts = Counter()
    for item in recoveries:
        for row in item.get("row_costs") or []:
            row_event_counts[str(row.get("relation_event_count"))] += 1
    return {
        "recovery_count": len(recoveries),
        "min_full_replay_ops_over_rho": round(min(full_costs), 8) if full_costs else None,
        "mean_full_replay_ops_over_rho": mean_or_none(full_costs),
        "min_best_verified_subset_ops_over_rho": (
            round(min(best_subset_costs), 8) if best_subset_costs else None
        ),
        "max_best_verified_subset_ops_over_rho": (
            round(max(best_subset_costs), 8) if best_subset_costs else None
        ),
        "below_rho_verified_subset_recovery_count": len(below_subset),
        "event_positive_rows_below_rho_recovery_count": len(event_row_below),
        "min_event_positive_row_ops_over_rho_sum": (
            round(min(event_row_costs), 8) if event_row_costs else None
        ),
        "row_relation_event_count_distribution": dict(sorted(row_event_counts.items())),
        "diagnostic_below_rho_subset_recoveries": [
            {
                "transfer_index": item.get("transfer_index"),
                "derived_secret": (item.get("full_replay") or {}).get("derived_secret"),
                "full_ops_over_rho": (item.get("full_replay") or {}).get("ops_over_rho"),
                "best_subset": item.get("best_below_rho_verified_subset"),
                "event_positive_row_ops_over_rho_sum": item.get(
                    "event_positive_row_ops_over_rho_sum"
                ),
            }
            for item in below_subset
        ],
        "interpretation": (
            "If below-rho subsets exist, the algebraic relation replay can be "
            "cheap once relation-bearing rows are known.  This is not a public "
            "speedup until the same pruning is predicted without verifier events."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=parse_source, action="append", required=True)
    parser.add_argument("--coordinate", type=parse_coordinate, required=True)
    parser.add_argument(
        "--gate-kind",
        choices=["exact_coordinate"],
        default="exact_coordinate",
        help="Only exact-coordinate reruns are currently reconstructed from compact artifacts.",
    )
    parser.add_argument("--max-recoveries", type=int, default=0)
    parser.add_argument("--event-summary-limit", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    loaded_sources = [(name, load_json(path)) for name, path in args.candidate_source]
    selected = selected_unique_candidates(
        loaded_sources,
        args.coordinate,
        str(args.gate_kind),
        int(args.max_recoveries),
    )
    if not selected:
        raise SystemExit("no verified candidate recoveries matched the requested coordinate")

    first_params = selected[0].get("_artifact_parameters") or {}
    bank_source = load_json(Path(first_params["bank_source"]))
    config_source = load_json(Path(first_params["config_source"]))
    direct_source = load_json(Path(first_params["direct_source"]))
    transfer_source = load_json(Path(first_params["transfer_source"]))
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(first_params.get("radius") or (params or {}).get("radius") or 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    recoveries = [
        recovery_decomposition(
            candidate,
            args.coordinate,
            str(args.gate_kind),
            verifier,
            records,
            config_source,
            specs_by_target,
            context_cache,
            int(args.event_summary_limit),
        )
        for candidate in selected
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_cost_decomposition_probe_v1",
        "method": "rerun_verified_coordinate_recoveries_with_row_subset_costs",
        "parameters": {
            "candidate_sources": [
                {"name": name, "artifact": str(path)} for name, path in args.candidate_source
            ],
            "coordinate": {"b": args.coordinate[0], "c": args.coordinate[1]},
            "gate_kind": args.gate_kind,
            "max_recoveries": int(args.max_recoveries),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "summary": summarize(recoveries),
        "recoveries": recoveries,
        "non_claims": [
            "Subset replays are verifier-informed diagnostics unless a public row-pruning rule selects the same subset.",
            "The probe decomposes measured row scan costs; it does not prove that coordinate confirmation is reusable.",
            "Only exact-coordinate gates are rerun because compact candidate artifacts do not preserve full case profiles for wider axis gates.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
