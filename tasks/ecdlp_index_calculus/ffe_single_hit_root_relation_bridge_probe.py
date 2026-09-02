#!/usr/bin/env python3
"""Join retained single-hit-root FFE hyperplanes to verifier signature cases.

The single-hit-root policy is now a replicated component: it publicly selects a
preserving quadratic root hyperplane below rho on fresh total2 windows.  This
probe starts the next proof obligation by joining those retained hyperplanes
back to the verifier-backed low-term signature cases that produced them.

The output is deliberately conservative.  It counts relation/rank evidence that
already exists in source signature cases, but it does not claim an end-to-end
ECDLP derivation unless a later verifier path assembles those rows into an
independent scalar-free proof.
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
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_single_hit_root_relation_bridge_152_175.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "window must be label:signature_json:policy_json"
        )
    label, signature, policy = parts
    if not label:
        raise argparse.ArgumentTypeError("window label must be nonempty")
    return label, Path(signature), Path(policy)


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    row_leaf_key = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector")),
        row_leaf_key,
    )


def compact_case(case: dict[str, Any], window: str) -> dict[str, Any]:
    row_leaf_keys = [
        {
            "row_key": item.get("row_key"),
            "leaf_indices": [int(leaf) for leaf in item.get("leaf_indices") or []],
            "salt": item.get("salt"),
            "surface_id": item.get("surface_id"),
        }
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    ]
    return {
        "window": window,
        "case_key": "|".join(str(part) for part in case_key(case)),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "public_key_verified": bool(case.get("public_key_verified")),
        "relation_count": int(case.get("relation_count") or 0),
        "rank": int(case.get("rank") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "row_leaf_keys": row_leaf_keys,
        "row_salts": [int(salt) for salt in case.get("row_salts") or []],
        "unique_leaf_indices": [int(leaf) for leaf in case.get("unique_leaf_indices") or []],
        "surface_ids": [str(surface_id) for surface_id in case.get("surface_ids") or []],
    }


def compact_policy_row(row: dict[str, Any], window: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = row.get("chosen_candidate") or {}
    return {
        "window": window,
        "surface_id": row.get("surface_id"),
        "target": row.get("target"),
        "transfer_index": int(row.get("transfer_index") or 0),
        "row_key": row.get("row_key"),
        "p": int(row.get("p") or 0),
        "chosen_root": chosen.get("root"),
        "chosen_factor_index": chosen.get("factor_index"),
        "pre_factor_selected_hit_roots": row.get("pre_factor_selected_hit_roots") or [],
        "selector_eval_ops": int(row.get("selector_eval_ops") or 0),
        "root_scan_ops": row.get("root_scan_ops"),
        "direct_root_recovery_ops": row.get("direct_root_recovery_ops"),
        "generic_rho_steps": int(row.get("generic_rho_steps") or 0),
        "total_ops_over_rho": row.get("total_ops_over_rho"),
        "direct_total_ops_over_rho": row.get("direct_total_ops_over_rho"),
        "chosen_preserves_selected_root_pairs": bool(row.get("chosen_preserves_selected_root_pairs")),
        "chosen_false_positive": bool(row.get("chosen_false_positive")),
        "below_rho": bool(row.get("below_rho")),
        "direct_below_rho": bool(row.get("direct_below_rho")),
        "source_case_count": len(cases),
        "source_case_keys": [case["case_key"] for case in cases],
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def summarize_target(surfaces: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["total_ops_over_rho"])
        for row in surfaces
        if row.get("total_ops_over_rho") is not None
    ]
    direct_ratios = [
        float(row["direct_total_ops_over_rho"])
        for row in surfaces
        if row.get("direct_total_ops_over_rho") is not None
    ]
    transfers = sorted(
        {
            int(row.get("transfer_index") or 0)
            for row in surfaces + cases
            if row.get("transfer_index") is not None
        }
    )
    return {
        "surface_count": len(surfaces),
        "source_case_count": len(cases),
        "transfer_indices": transfers,
        "window_count": len({str(row.get("window")) for row in surfaces}),
        "relation_count_sum_from_source_cases": sum(int(case.get("relation_count") or 0) for case in cases),
        "rank_sum_from_source_cases": sum(int(case.get("rank") or 0) for case in cases),
        "max_source_case_rank": max([int(case.get("rank") or 0) for case in cases] or [0]),
        "min_source_case_ops_over_rho": round(
            min(float(case["ops_over_rho"]) for case in cases if case.get("ops_over_rho") is not None),
            8,
        )
        if any(case.get("ops_over_rho") is not None for case in cases)
        else None,
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": mean_or_none(ratios),
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
        "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
    }


def build_bridge(windows: list[tuple[str, Path, Path]]) -> dict[str, Any]:
    retained_surfaces: list[dict[str, Any]] = []
    retained_cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    window_summaries = []
    for label, signature_path, policy_path in windows:
        signature = load_json(signature_path)
        policy = load_json(policy_path)
        cases_by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in signature.get("positive_cases") or []:
            if not isinstance(case, dict):
                continue
            compact = compact_case(case, label)
            retained_cases_by_key[case_key(case)] = compact
            for surface_id in compact["surface_ids"]:
                cases_by_surface[str(surface_id)].append(compact)
        window_rows = []
        unjoined = []
        for row in policy.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if not (
                row.get("chosen_preserves_selected_root_pairs")
                and not row.get("chosen_false_positive")
                and row.get("below_rho")
            ):
                continue
            surface_id = str(row.get("surface_id"))
            cases = cases_by_surface.get(surface_id, [])
            compact = compact_policy_row(row, label, cases)
            retained_surfaces.append(compact)
            window_rows.append(compact)
            if not cases:
                unjoined.append(surface_id)
        window_summaries.append(
            {
                "window": label,
                "signature_source": str(signature_path),
                "policy_source": str(policy_path),
                "retained_surface_count": len(window_rows),
                "retained_source_case_count": len(
                    {
                        key
                        for key, case in retained_cases_by_key.items()
                        if case.get("window") == label
                        and any(surface_id in {row["surface_id"] for row in window_rows} for surface_id in case.get("surface_ids") or [])
                    }
                ),
                "unjoined_retained_surface_count": len(unjoined),
                "unjoined_retained_surface_ids": unjoined,
            }
        )

    retained_surface_ids = {str(row["surface_id"]) for row in retained_surfaces}
    retained_cases = [
        case
        for case in retained_cases_by_key.values()
        if any(surface_id in retained_surface_ids for surface_id in case.get("surface_ids") or [])
    ]
    by_target_surfaces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target_cases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in retained_surfaces:
        by_target_surfaces[str(row.get("target"))].append(row)
    for case in retained_cases:
        by_target_cases[str(case.get("target"))][str(case.get("case_key"))] = case

    target_summaries = {
        target: summarize_target(surfaces, list(by_target_cases.get(target, {}).values()))
        for target, surfaces in sorted(by_target_surfaces.items())
    }
    ratios = [
        float(row["total_ops_over_rho"])
        for row in retained_surfaces
        if row.get("total_ops_over_rho") is not None
    ]
    direct_ratios = [
        float(row["direct_total_ops_over_rho"])
        for row in retained_surfaces
        if row.get("direct_total_ops_over_rho") is not None
    ]
    target_order = sorted(
        target_summaries,
        key=lambda target: (
            -int(target_summaries[target]["surface_count"]),
            -int(target_summaries[target]["source_case_count"]),
            float(target_summaries[target]["min_total_ops_over_rho"] or 10**9),
            target,
        ),
    )
    return {
        "schema": "ecdlp_ffe_single_hit_root_relation_bridge_probe_v1",
        "method": "join_retained_single_hit_root_hyperplanes_to_verifier_signature_cases",
        "parameters": {
            "windows": [
                {
                    "label": label,
                    "signature_source": str(signature),
                    "policy_source": str(policy),
                }
                for label, signature, policy in windows
            ],
            "retention_rule": [
                "chosen_preserves_selected_root_pairs",
                "not chosen_false_positive",
                "below_rho"
            ],
        },
        "summary": {
            "window_count": len(windows),
            "retained_surface_count": len(retained_surfaces),
            "retained_source_case_count": len(retained_cases),
            "target_count": len(target_summaries),
            "all_retained_surfaces_below_rho": all(bool(row.get("below_rho")) for row in retained_surfaces),
            "all_retained_surfaces_direct_below_rho": all(
                bool(row.get("direct_below_rho")) for row in retained_surfaces
            ),
            "chosen_false_positive_count": sum(bool(row.get("chosen_false_positive")) for row in retained_surfaces),
            "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
            "mean_total_ops_over_rho": mean_or_none(ratios),
            "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
            "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
            "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
            "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
            "strongest_targets": target_order,
            "bridge_status": "component_join_only_no_scalar_free_ecdlp_derivation_yet",
            "next_obligation": (
                "Build a verifier-facing relation assembly that proves whether "
                "the retained hyperplanes give enough independent equations to "
                "derive the ECDLP secret without returning a scalar."
            ),
        },
        "window_summaries": window_summaries,
        "target_summaries": target_summaries,
        "retained_surfaces": sorted(
            retained_surfaces,
            key=lambda row: (
                str(row.get("target")),
                int(row.get("transfer_index") or 0),
                str(row.get("surface_id")),
            ),
        ),
        "retained_source_cases": sorted(
            retained_cases,
            key=lambda row: (
                str(row.get("target")),
                int(row.get("transfer_index") or 0),
                str(row.get("case_key")),
            ),
        ),
        "non_claims": [
            "This does not prove an end-to-end ECDLP speedup.",
            "This does not prove scalar-free derivation.",
            "Source case ranks are counted as existing verifier evidence, not as a proven independent global relation matrix.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        type=parse_window,
        action="append",
        required=True,
        help="Window as label:signature_json:policy_json.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output = build_bridge(args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
