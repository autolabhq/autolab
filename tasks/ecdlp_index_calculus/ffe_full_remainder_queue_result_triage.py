#!/usr/bin/env python3
"""Summarize top-k12/root0 queue exact results and rank remaining batches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_QUEUE = DEFAULT_STATE_DIR / "ffe_full_remainder_topk12_root0_candidate_queue.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_full_remainder_topk12_root0_queue_result_triage.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def leaf_tuple(values: Any) -> tuple[int, ...]:
    leaves: list[int] = []
    for value in values or []:
        leaf = as_int(value)
        if leaf is not None:
            leaves.append(leaf)
    return tuple(sorted(set(leaves)))


def materialization_key(target: str, transfer_index: int, row_key: str, leaves: tuple[int, ...]) -> str:
    return f"{target}|{transfer_index}|{row_key}|{','.join(str(leaf) for leaf in leaves)}"


def candidate_key(candidate: dict[str, Any]) -> str | None:
    key = candidate.get("materialization_key") or {}
    target = key.get("target") or candidate.get("target")
    transfer_index = as_int(key.get("transfer_index") or candidate.get("transfer_index"))
    row_key = key.get("row_key") or candidate.get("row_key")
    leaves = leaf_tuple(key.get("leaf_indices") or candidate.get("leaf_indices"))
    if target is None or transfer_index is None or row_key is None or not leaves:
        return None
    return materialization_key(str(target), transfer_index, str(row_key), leaves)


def surface_key(surface: dict[str, Any]) -> str | None:
    exact = surface.get("exact_profile") or {}
    target = exact.get("target") or surface.get("target")
    transfer_index = as_int(exact.get("transfer_index") or surface.get("transfer_index"))
    row_key = exact.get("row_key") or surface.get("row_key")
    leaves = leaf_tuple(exact.get("leaf_indices") or surface.get("selected_leaf_indices"))
    if target is None or transfer_index is None or row_key is None or not leaves:
        return None
    return materialization_key(str(target), transfer_index, str(row_key), leaves)


def compact_surface(surface: dict[str, Any], source: Path) -> dict[str, Any]:
    exact = surface.get("exact_profile") or {}
    best = surface.get("best_preserving_candidate") or {}
    target = str(exact.get("target") or surface.get("target"))
    transfer_index = as_int(exact.get("transfer_index") or surface.get("transfer_index"))
    row_key = str(exact.get("row_key") or surface.get("row_key"))
    leaves = leaf_tuple(exact.get("leaf_indices") or surface.get("selected_leaf_indices"))
    full_ops = as_float(best.get("remainder_ffe_ops_over_rho") or surface.get("full_remainder_ffe_ops_over_rho"))
    factor_ops = as_float(best.get("factor_root_scan_ops_over_rho"))
    surface_ops = as_float(best.get("surface_ffe_ops_over_rho"))
    root_count = as_int(surface.get("original_selected_root_pair_count"))
    return {
        "source": str(source),
        "materialization_key": materialization_key(target, transfer_index or 0, row_key, leaves),
        "surface_profile_id": surface.get("surface_profile_id"),
        "target": target,
        "transfer_index": transfer_index,
        "row_key": row_key,
        "leaf_indices": list(leaves),
        "leaf_count": len(leaves),
        "original_selected_root_pair_count": root_count,
        "root0": root_count == 0,
        "full_remainder_ops_over_rho": full_ops,
        "full_remainder_below_rho": full_ops is not None and full_ops < 1.0,
        "factor_root_scan_ops_over_rho": factor_ops,
        "factor_root_scan_below_rho": factor_ops is not None and factor_ops < 1.0,
        "surface_ffe_ops_over_rho": surface_ops,
        "surface_ffe_below_rho": surface_ops is not None and surface_ops < 1.0,
        "selected_root_pair_count": as_int(best.get("selected_root_pair_count")),
        "selected_valid_root_leaves": as_int(best.get("selected_valid_root_leaves")),
        "selected_missed_leaves": as_int(best.get("selected_missed_leaves")),
    }


def dedupe_surfaces(exact_artifacts: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for path in exact_artifacts:
        data = load_json(path)
        for surface in data.get("surfaces") or []:
            if not isinstance(surface, dict):
                continue
            key = surface_key(surface)
            if key is None:
                continue
            compact = compact_surface(surface, path)
            records.append(compact)
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = compact
                continue
            prev_full = previous.get("full_remainder_ops_over_rho")
            cur_full = compact.get("full_remainder_ops_over_rho")
            if cur_full is not None and (prev_full is None or cur_full < prev_full):
                by_key[key] = compact
    return records, sorted(by_key.values(), key=surface_sort_key)


def surface_sort_key(surface: dict[str, Any]) -> tuple[Any, ...]:
    full_ops = surface.get("full_remainder_ops_over_rho")
    factor_ops = surface.get("factor_root_scan_ops_over_rho")
    return (
        full_ops if full_ops is not None else 10**9,
        factor_ops if factor_ops is not None else 10**9,
        str(surface.get("target")),
        int(surface.get("transfer_index") or 0),
        surface.get("leaf_indices") or [],
    )


def stats_for_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    full_values = [
        float(surface["full_remainder_ops_over_rho"])
        for surface in surfaces
        if surface.get("full_remainder_ops_over_rho") is not None
    ]
    factor_values = [
        float(surface["factor_root_scan_ops_over_rho"])
        for surface in surfaces
        if surface.get("factor_root_scan_ops_over_rho") is not None
    ]
    by_target = []
    for target, group in grouped(surfaces, "target").items():
        by_target.append({"target": target, **basic_group_stats(group)})
    by_row_key = []
    for row_key, group in grouped(surfaces, "row_key").items():
        by_row_key.append({"row_key": row_key, **basic_group_stats(group)})
    by_leaf_count = []
    for leaf_count, group in grouped(surfaces, "leaf_count").items():
        by_leaf_count.append({"leaf_count": int(leaf_count), **basic_group_stats(group)})
    return {
        "profile_record_count": len(surfaces),
        "root0_count": sum(1 for surface in surfaces if surface.get("root0")),
        "nonroot0_count": sum(1 for surface in surfaces if surface.get("root0") is False),
        "full_remainder_below_rho_count": sum(1 for surface in surfaces if surface.get("full_remainder_below_rho")),
        "factor_root_scan_below_rho_count": sum(1 for surface in surfaces if surface.get("factor_root_scan_below_rho")),
        "surface_ffe_below_rho_count": sum(1 for surface in surfaces if surface.get("surface_ffe_below_rho")),
        "min_full_remainder_ops_over_rho": min(full_values) if full_values else None,
        "min_factor_root_scan_ops_over_rho": min(factor_values) if factor_values else None,
        "by_target": sorted(by_target, key=lambda item: (item.get("min_full_remainder_ops_over_rho") or 10**9, item["target"])),
        "by_row_key": sorted(by_row_key, key=lambda item: (item.get("min_full_remainder_ops_over_rho") or 10**9, item["row_key"])),
        "by_leaf_count": sorted(by_leaf_count, key=lambda item: item["leaf_count"]),
    }


def grouped(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key))].append(item)
    return dict(groups)


def basic_group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
    full_values = [
        float(item["full_remainder_ops_over_rho"])
        for item in group
        if item.get("full_remainder_ops_over_rho") is not None
    ]
    factor_values = [
        float(item["factor_root_scan_ops_over_rho"])
        for item in group
        if item.get("factor_root_scan_ops_over_rho") is not None
    ]
    return {
        "count": len(group),
        "root0_count": sum(1 for item in group if item.get("root0")),
        "full_remainder_below_rho_count": sum(1 for item in group if item.get("full_remainder_below_rho")),
        "factor_root_scan_below_rho_count": sum(1 for item in group if item.get("factor_root_scan_below_rho")),
        "min_full_remainder_ops_over_rho": min(full_values) if full_values else None,
        "min_factor_root_scan_ops_over_rho": min(factor_values) if factor_values else None,
    }


def row_minima(surfaces: list[dict[str, Any]]) -> dict[str, float]:
    minima: dict[str, float] = {}
    for surface in surfaces:
        row_key = str(surface.get("row_key"))
        full_ops = surface.get("full_remainder_ops_over_rho")
        if full_ops is None:
            continue
        value = float(full_ops)
        minima[row_key] = min(value, minima.get(row_key, value))
    return minima


def candidate_priority(candidate: dict[str, Any], observed_row_minima: dict[str, float]) -> tuple[Any, ...]:
    row_key = str(candidate.get("row_key"))
    row_observed = observed_row_minima.get(row_key, 10**9)
    leaves = leaf_tuple(candidate.get("leaf_indices"))
    bank_ops = as_float(candidate.get("bank_best_filter_ops_over_rho"))
    selector_ops = as_float(candidate.get("selector_ops_over_rho_min"))
    equivalent_count = len(candidate.get("equivalent_profiles") or [])
    return (
        row_observed,
        len(leaves),
        bank_ops if bank_ops is not None else 10**9,
        selector_ops if selector_ops is not None else 10**9,
        -equivalent_count,
        str(candidate.get("target")),
        int(candidate.get("transfer_index") or 0),
        list(leaves),
    )


def ranked_candidates(queue: dict[str, Any], surfaces: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    minima = row_minima(surfaces)
    candidates = [
        candidate
        for candidate in queue.get("candidates") or []
        if isinstance(candidate, dict) and candidate_key(candidate) is not None
    ]
    candidates = sorted(candidates, key=lambda candidate: candidate_priority(candidate, minima))
    return compact_ranked_candidates(candidates[:limit], minima)


def compact_ranked_candidates(candidates: list[dict[str, Any]], minima: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for candidate in candidates:
        out.append(
            {
                "profile": candidate.get("profile"),
                "profile_from_signature": candidate.get("profile_from_signature"),
                "target": candidate.get("target"),
                "transfer_index": candidate.get("transfer_index"),
                "row_key": candidate.get("row_key"),
                "leaf_indices": candidate.get("leaf_indices"),
                "leaf_count": len(candidate.get("leaf_indices") or []),
                "source_artifacts": candidate.get("source_artifacts") or [],
                "equivalent_profile_count": len(candidate.get("equivalent_profiles") or []),
                "observed_row_min_full_remainder_ops_over_rho": minima.get(str(candidate.get("row_key"))),
                "bank_best_filter_ops_over_rho": candidate.get("bank_best_filter_ops_over_rho"),
                "selector_ops_over_rho_min": candidate.get("selector_ops_over_rho_min"),
            }
        )
    return out


def recommended_batches(ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in ranked:
        sources = candidate.get("source_artifacts") or []
        if not sources:
            continue
        batches[str(sources[0])].append(candidate)
    out = []
    for source, candidates in batches.items():
        out.append(
            {
                "signature_source": source,
                "candidate_count": len(candidates),
                "profiles": [candidate["profile"] for candidate in candidates[:limit]],
                "profile_preview": candidates[: min(limit, len(candidates))],
            }
        )
    return sorted(out, key=lambda item: (-item["candidate_count"], item["signature_source"]))[:limit]


def recommended_batches_by_row(
    queue: dict[str, Any],
    surfaces: list[dict[str, Any]],
    batch_preview_limit: int,
    per_row_rank_limit: int,
) -> list[dict[str, Any]]:
    minima = row_minima(surfaces)
    exact_by_row = grouped(surfaces, "row_key")
    candidates_by_row: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in queue.get("candidates") or []:
        if not isinstance(candidate, dict) or candidate_key(candidate) is None:
            continue
        candidates_by_row[str(candidate.get("row_key"))].append(candidate)

    out: list[dict[str, Any]] = []
    for row_key, candidates in candidates_by_row.items():
        ranked_raw = sorted(candidates, key=lambda candidate: candidate_priority(candidate, minima))
        ranked = compact_ranked_candidates(ranked_raw[:per_row_rank_limit], minima)
        exact_group = exact_by_row.get(row_key, [])
        out.append(
            {
                "row_key": row_key,
                "remaining_candidate_count": len(candidates),
                "unique_exact_count": len(exact_group),
                "unique_root0_count": sum(1 for surface in exact_group if surface.get("root0")),
                "unique_full_remainder_below_rho_count": sum(
                    1 for surface in exact_group if surface.get("full_remainder_below_rho")
                ),
                "unique_factor_root_scan_below_rho_count": sum(
                    1 for surface in exact_group if surface.get("factor_root_scan_below_rho")
                ),
                "observed_min_full_remainder_ops_over_rho": minima.get(row_key),
                "ranked_candidate_preview": ranked[:batch_preview_limit],
                "recommended_batches": recommended_batches(ranked, batch_preview_limit),
            }
        )
    return sorted(
        out,
        key=lambda item: (
            item["unique_full_remainder_below_rho_count"] <= 0,
            item["unique_exact_count"],
            -(item["remaining_candidate_count"]),
            item["row_key"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--exact-artifact", type=Path, action="append", required=True)
    parser.add_argument("--rank-limit", type=int, default=32)
    parser.add_argument("--batch-preview-limit", type=int, default=8)
    parser.add_argument("--per-row-rank-limit", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    queue = load_json(args.queue)
    exact_records, exact_unique = dedupe_surfaces(args.exact_artifact)
    ranked = ranked_candidates(queue, exact_unique, args.rank_limit)
    output = {
        "schema": "ecdlp_full_remainder_topk12_root0_queue_result_triage_v1",
        "method": "deduped_exact_result_summary_and_remaining_queue_rank",
        "inputs": {
            "queue": str(args.queue),
            "exact_artifacts": [str(path) for path in args.exact_artifact],
        },
        "summary": {
            "exact_profile_record_count": len(exact_records),
            "unique_materialized_surface_count": len(exact_unique),
            "duplicate_exact_profile_record_count": len(exact_records) - len(exact_unique),
            "remaining_queue_candidate_count": len(queue.get("candidates") or []),
            "unique_exact": stats_for_surfaces(exact_unique),
            "all_profile_records": stats_for_surfaces(exact_records),
        },
        "best_unique_surfaces": exact_unique[: min(16, len(exact_unique))],
        "ranked_remaining_candidates": ranked,
        "recommended_batches": recommended_batches(ranked, args.batch_preview_limit),
        "recommended_batches_by_row": recommended_batches_by_row(
            queue,
            exact_unique,
            args.batch_preview_limit,
            args.per_row_rank_limit,
        ),
        "boundary": (
            "Ranked candidates are triage suggestions only; promotion requires a fresh exact "
            "surface with full_remainder_ops_over_rho < 1.0, not merely factor-root-scan below rho."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "unique_materialized_surface_count": len(exact_unique),
                "min_full_remainder_ops_over_rho": output["summary"]["unique_exact"]["min_full_remainder_ops_over_rho"],
                "full_remainder_below_rho_count": output["summary"]["unique_exact"]["full_remainder_below_rho_count"],
                "remaining_queue_candidate_count": output["summary"]["remaining_queue_candidate_count"],
                "top_recommended_source": (output["recommended_batches"][0]["signature_source"] if output["recommended_batches"] else None),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
