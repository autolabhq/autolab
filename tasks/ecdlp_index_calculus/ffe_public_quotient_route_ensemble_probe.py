#!/usr/bin/env python3
"""Combine public FFE quotient routes before verifier-label auditing.

The current FFE quotient lead has two complementary public selectors:

* a direct public factor predictor, which is strong on the 67.a1@9803 surfaces;
* a reusable factor-fingerprint leaf locator, which is strong on the
  22050.cf1@11731 surfaces.

This probe does not introduce a new verifier oracle.  It loads those public
route outputs, selects a route per surface by public estimated cost only, and
then audits verifier labels such as preservation and below-rho status.
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
DEFAULT_SUPPORT_SOURCE = DEFAULT_STATE_DIR / "ffe_support_unique_diag.json"
DEFAULT_MISS_CLOSURE_SOURCE = DEFAULT_STATE_DIR / "ffe_public_quotient_miss_closure_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_quotient_route_ensemble_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def first_existing_factor_source(state_dir: Path) -> Path:
    candidates = sorted(state_dir.glob("*low_term_total2_ffe_public_factor_predictor_probe.json"))
    if not candidates:
        raise SystemExit(f"no public factor predictor JSON under {state_dir}")
    return candidates[0]


def clean_ratio(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def surface_target(surface_id: str, fallback: str | None = None) -> str:
    if "|" in surface_id:
        return surface_id.split("|", 1)[0]
    return fallback or ""


def factor_routes(factor_source: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    universe: dict[str, dict[str, Any]] = {}
    routes = []
    for surface in factor_source.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        surface_id = str(surface.get("surface_id") or "")
        if not surface_id:
            continue
        target = str(surface.get("target") or surface_target(surface_id))
        universe[surface_id] = {
            "surface_id": surface_id,
            "target": target,
            "row_key": surface.get("row_key"),
            "challenge_seed": surface.get("challenge_seed"),
        }
        selection = surface.get("best_public_factor_selection") or {}
        ratio = clean_ratio(selection.get("public_factor_all_hit_ops_over_rho"))
        routes.append(
            {
                "route": "direct_public_factor",
                "surface_id": surface_id,
                "target": target,
                "row_key": surface.get("row_key"),
                "challenge_seed": surface.get("challenge_seed"),
                "estimated_ops_over_rho": ratio,
                "estimated_ops": selection.get("public_factor_all_hit_ops"),
                "estimated_beats_rho": bool(selection.get("public_factor_all_hit_beats_rho")),
                "preserves_selected_root_pairs": bool(selection.get("preserves_selected_root_pairs")),
                "same_selected_root_pairs": bool(selection.get("same_selected_root_pairs")),
                "selector": {
                    "rule": selection.get("rule"),
                    "top_k": selection.get("top_k"),
                    "chosen_count": selection.get("chosen_count"),
                    "factor_work": selection.get("factor_work"),
                    "quadratic_root_work": selection.get("quadratic_root_work"),
                    "recovered_root_count": selection.get("recovered_root_count"),
                },
            }
        )
    return universe, routes


def support_routes(support_source: dict[str, Any]) -> list[dict[str, Any]]:
    routes = []
    for split in support_source.get("splits") or []:
        if not isinstance(split, dict):
            continue
        split_name = str(split.get("split") or "")
        for row in split.get("test_rows") or []:
            if not isinstance(row, dict):
                continue
            surface_id = str(row.get("surface_id") or "")
            if not surface_id:
                continue
            ratio = clean_ratio(row.get("public_locator_all_hit_ops_over_rho"))
            routes.append(
                {
                    "route": "cv_fingerprint_leaf_locator",
                    "split": split_name,
                    "surface_id": surface_id,
                    "target": str(row.get("target") or surface_target(surface_id)),
                    "row_key": row.get("row_key"),
                    "challenge_seed": row.get("challenge_seed"),
                    "estimated_ops_over_rho": ratio,
                    "estimated_ops": row.get("public_locator_all_hit_ops"),
                    "estimated_beats_rho": bool(row.get("public_locator_all_hit_beats_rho")),
                    "preserves_selected_root_pairs": bool(row.get("preserves_selected_root_pairs")),
                    "same_selected_root_pairs": bool(row.get("same_selected_root_pairs")),
                    "selector": {
                        "locator_policy": row.get("policy"),
                        "support_policy": row.get("support_policy"),
                        "predicted_leaf_count": row.get("predicted_leaf_count"),
                        "support_matched_candidate_count": row.get("support_matched_candidate_count"),
                        "support_rejected_candidate_count": row.get("support_rejected_candidate_count"),
                        "support_selector_extra_ops": row.get("support_selector_extra_ops"),
                    },
                    "chosen_support_signature": row.get("chosen_support_signature"),
                    "selected_root_support_diagnostic": row.get("selected_root_support_diagnostic"),
                }
            )
    return routes


def incremental_support_routes(miss_closure_source: dict[str, Any]) -> list[dict[str, Any]]:
    routes = []
    for row in miss_closure_source.get("surface_rows") or []:
        if not isinstance(row, dict):
            continue
        surface_id = str(row.get("surface_id") or "")
        if not surface_id:
            continue
        for selection in row.get("selections") or []:
            if not isinstance(selection, dict):
                continue
            if selection.get("order_policy") != "low_factor_index":
                continue
            if selection.get("cost_model") != "incremental_selector_scan_all_hit":
                continue
            chosen = selection.get("chosen") or {}
            charge = chosen.get("charge") or {}
            ratio = clean_ratio(charge.get("ops_over_rho"))
            routes.append(
                {
                    "route": "incremental_factor_index_support",
                    "split": row.get("split"),
                    "surface_id": surface_id,
                    "target": str(row.get("target") or surface_target(surface_id)),
                    "row_key": row.get("row_key"),
                    "challenge_seed": row.get("challenge_seed"),
                    "estimated_ops_over_rho": ratio,
                    "estimated_ops": charge.get("ops"),
                    "estimated_beats_rho": bool(charge.get("beats_rho")),
                    "preserves_selected_root_pairs": bool(chosen.get("preserves_selected_root_pairs")),
                    "same_selected_root_pairs": bool(chosen.get("same_selected_root_pairs")),
                    "selector": {
                        "order_policy": selection.get("order_policy"),
                        "cost_model": selection.get("cost_model"),
                        "candidate_name": chosen.get("candidate_name"),
                        "factor_index": chosen.get("factor_index"),
                        "predicted_valid_root_indices": chosen.get("predicted_valid_root_indices"),
                        "incremental_selector_ops": chosen.get("incremental_selector_ops"),
                        "support_check_ops": chosen.get("support_check_ops"),
                        "support_scan_extra_ops": chosen.get("scan_extra_ops"),
                        "effective_selector_ops": charge.get("effective_selector_ops"),
                    },
                    "chosen_support_signature": {
                        "predicted_leaf_indices": chosen.get("predicted_leaf_indices"),
                        "predicted_valid_root_indices": chosen.get("predicted_valid_root_indices"),
                        "predicted_recovered_root_count": chosen.get("predicted_recovered_root_count"),
                    },
                }
            )
    return routes


def route_sort_key(route: dict[str, Any]) -> tuple[Any, ...]:
    ratio = clean_ratio(route.get("estimated_ops_over_rho"))
    return (
        not bool(route.get("estimated_beats_rho")),
        ratio if ratio is not None else 10**18,
        str(route.get("route") or ""),
        str(route.get("split") or ""),
    )


def choose_routes(universe: dict[str, dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        by_surface[str(route.get("surface_id") or "")].append(route)

    selected = []
    missing = []
    for surface_id, surface in sorted(universe.items()):
        candidates = [
            route
            for route in by_surface.get(surface_id, [])
            if route.get("estimated_beats_rho") and clean_ratio(route.get("estimated_ops_over_rho")) is not None
        ]
        if candidates:
            selected.append(sorted(candidates, key=route_sort_key)[0])
        else:
            best_any = sorted(by_surface.get(surface_id, []), key=route_sort_key)
            missing.append({**surface, "best_available_route": best_any[0] if best_any else None})
    return {"selected": selected, "missing": missing, "by_surface": by_surface}


def summarize(
    universe: dict[str, dict[str, Any]],
    factor_route_rows: list[dict[str, Any]],
    support_route_rows: list[dict[str, Any]],
    incremental_route_rows: list[dict[str, Any]],
    selected_routes: list[dict[str, Any]],
    missing_surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_ratios = [
        float(route["estimated_ops_over_rho"])
        for route in selected_routes
        if route.get("estimated_ops_over_rho") is not None
    ]
    factor_below = {
        str(route["surface_id"])
        for route in factor_route_rows
        if route.get("estimated_beats_rho")
    }
    support_below = {
        str(route["surface_id"])
        for route in support_route_rows
        if route.get("estimated_beats_rho")
    }
    incremental_below = {
        str(route["surface_id"])
        for route in incremental_route_rows
        if route.get("estimated_beats_rho")
    }
    selected_false = [
        route for route in selected_routes if not route.get("preserves_selected_root_pairs")
    ]
    by_target: dict[str, dict[str, Any]] = {}
    for target in sorted({surface["target"] for surface in universe.values()}):
        target_universe = [sid for sid, surface in universe.items() if surface["target"] == target]
        target_selected = [route for route in selected_routes if route.get("target") == target]
        target_ratios = [
            float(route["estimated_ops_over_rho"])
            for route in target_selected
            if route.get("estimated_ops_over_rho") is not None
        ]
        by_target[target] = {
            "surface_count": len(target_universe),
            "selected_below_rho_count": len(target_selected),
            "selected_preserving_count": sum(1 for route in target_selected if route.get("preserves_selected_root_pairs")),
            "selected_false_positive_count": sum(
                1 for route in target_selected if not route.get("preserves_selected_root_pairs")
            ),
            "min_selected_ops_over_rho": min(target_ratios) if target_ratios else None,
            "max_selected_ops_over_rho": max(target_ratios) if target_ratios else None,
        }
    return {
        "surface_universe_count": len(universe),
        "factor_route_below_rho_unique_count": len(factor_below),
        "support_route_below_rho_unique_count": len(support_below),
        "incremental_support_route_below_rho_unique_count": len(incremental_below),
        "public_route_union_below_rho_count": len(factor_below | support_below | incremental_below),
        "selected_route_count": len(selected_routes),
        "selected_preserving_count": sum(1 for route in selected_routes if route.get("preserves_selected_root_pairs")),
        "selected_false_positive_count": len(selected_false),
        "selected_same_surface_count": sum(1 for route in selected_routes if route.get("same_selected_root_pairs")),
        "missing_surface_count": len(missing_surfaces),
        "min_selected_ops_over_rho": min(selected_ratios) if selected_ratios else None,
        "mean_selected_ops_over_rho": round(mean(selected_ratios), 8) if selected_ratios else None,
        "max_selected_ops_over_rho": max(selected_ratios) if selected_ratios else None,
        "selected_route_counts": dict(sorted(Counter(route["route"] for route in selected_routes).items())),
        "selected_target_counts": dict(sorted(Counter(route["target"] for route in selected_routes).items())),
        "by_target": by_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--factor-source", type=Path)
    parser.add_argument("--support-source", type=Path, default=DEFAULT_SUPPORT_SOURCE)
    parser.add_argument("--miss-closure-source", type=Path, default=DEFAULT_MISS_CLOSURE_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    factor_source_path = args.factor_source or first_existing_factor_source(args.state_dir)
    support_source_path = args.support_source
    miss_closure_source_path = args.miss_closure_source
    factor_source = load_json(factor_source_path)
    support_source = load_json(support_source_path)
    miss_closure_source = load_json(miss_closure_source_path)
    universe, factor_route_rows = factor_routes(factor_source)
    support_route_rows = support_routes(support_source)
    incremental_route_rows = incremental_support_routes(miss_closure_source)
    choice = choose_routes(universe, factor_route_rows + support_route_rows + incremental_route_rows)
    selected_routes = choice["selected"]
    missing_surfaces = choice["missing"]
    output = {
        "method": "public_ffe_quotient_route_ensemble",
        "parameters": {
            "factor_source": str(factor_source_path),
            "support_source": str(support_source_path),
            "miss_closure_source": str(miss_closure_source_path),
            "selection_rule": "choose any public route with estimated ops/rho < 1, then minimize estimated ops/rho",
            "label_fields_used_only_for_audit": [
                "preserves_selected_root_pairs",
                "same_selected_root_pairs",
            ],
        },
        "summary": summarize(
            universe,
            factor_route_rows,
            support_route_rows,
            incremental_route_rows,
            selected_routes,
            missing_surfaces,
        ),
        "selected_routes": sorted(selected_routes, key=lambda row: (row["target"], row["surface_id"], route_sort_key(row))),
        "missing_surfaces": missing_surfaces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
