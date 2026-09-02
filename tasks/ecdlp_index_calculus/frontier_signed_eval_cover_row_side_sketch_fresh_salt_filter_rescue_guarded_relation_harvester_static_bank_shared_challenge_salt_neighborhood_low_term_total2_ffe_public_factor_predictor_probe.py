#!/usr/bin/env python3
"""Factor-level public predictor audit for the low-term total-2 FFE quotient lead.

The public slice predictor keeps every low-degree factor on a chosen slice, so
its public selections can preserve roots while still losing to rho on factor
work.  This follow-up ranks individual public slice factors instead.  It keeps
the selector public by using only fixed coordinates, factor shape, and
factor-zero counts over public monic leaves, then audits whether the selected
factor set recovers verifier-backed roots through quadratic solving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_resultant_surface_probe as resultant_surface_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_cross_surface_root_map_probe as cross_surface_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_remainder_factor_probe as remainder_factor_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_resultant_slice_factor_probe as slice_factor_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_slice_quadratic_root_probe as slice_quadratic_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = cross_surface_probe.DEFAULT_SIGNATURE_SOURCE
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_predictor_probe.json"
)

RULES = (
    "factor_sparse_zero",
    "factor_low_work",
    "factor_singleton_low_leaf",
    "factor_singleton_low_coord",
    "balanced_factor",
)
TOP_KS = (1, 2, 4, 8, 16, 32, 64)


def load_json(path: Path) -> dict[str, Any]:
    return cross_surface_probe.load_json(path)


def leaf_coords(surface_record: dict[str, Any]) -> list[dict[str, int]]:
    p = int(surface_record["p"])
    coords = []
    for leaf_index, leaf in enumerate(surface_record["components"]["leaves"]):
        coeffs = resultant_surface_probe.monic_coeffs(leaf, p)
        if coeffs is None:
            continue
        b_value, c_value = coeffs
        coords.append({"leaf_index": leaf_index, "b": int(b_value), "c": int(c_value)})
    return coords


def evaluate_factor_candidate(
    surface_record: dict[str, Any],
    axis: str,
    fixed_value: int,
    factor: dict[str, Any],
    coords: list[dict[str, int]],
) -> dict[str, Any] | None:
    p = int(surface_record["p"])
    selected = {int(leaf) for leaf in surface_record["selected_leaf_indices"]}
    terms = {int(degree): int(coeff) % p for degree, coeff in factor["terms"]}
    selected_pairs: set[tuple[int, int]] = set()
    recovered_root_count = 0
    factor_zero_leaf_count = 0
    valid_root_leaf_count = 0
    false_zero_leaf_count = 0
    selected_valid_leaf_count = 0
    factor_zero_leaf_indices = []
    variable_values = []
    zero_variable_values = []

    for coord in coords:
        leaf_index = int(coord["leaf_index"])
        b_value = int(coord["b"])
        c_value = int(coord["c"])
        variable = int(c_value if axis == "b" else b_value) % p
        variable_values.append(variable)
        if slice_factor_probe.eval_univariate(terms, variable, p) != 0:
            continue
        factor_zero_leaf_count += 1
        factor_zero_leaf_indices.append(leaf_index)
        zero_variable_values.append(variable)
        leaf = surface_record["components"]["leaves"][leaf_index]
        roots = slice_quadratic_probe.recover_roots_for_leaf(surface_record, leaf, b_value, c_value)
        if not roots:
            false_zero_leaf_count += 1
            continue
        valid_root_leaf_count += 1
        recovered_root_count += len(roots)
        if leaf_index in selected:
            selected_valid_leaf_count += 1
            for root in roots:
                selected_pairs.add((leaf_index, int(root)))

    if factor_zero_leaf_count == 0:
        return None
    return {
        "axis": axis,
        "fixed_value": int(fixed_value),
        "fixed_value_mod": int(fixed_value) % p,
        "factor_index": int(factor["factor_index"]),
        "exponent": int(factor["exponent"]),
        "degree": int(factor["degree"]),
        "monomials": int(factor["monomials"]),
        "terms": factor["terms"],
        "relevant_leaf_count": len(coords),
        "factor_zero_leaf_count": factor_zero_leaf_count,
        "factor_zero_density": round(factor_zero_leaf_count / max(1, len(coords)), 8),
        "valid_root_leaf_count": valid_root_leaf_count,
        "false_zero_leaf_count": false_zero_leaf_count,
        "selected_valid_leaf_count": selected_valid_leaf_count,
        "recovered_root_count": recovered_root_count,
        "selected_pairs": sorted(selected_pairs),
        "min_leaf_index": min(int(coord["leaf_index"]) for coord in coords),
        "min_zero_leaf_index": min(factor_zero_leaf_indices),
        "factor_zero_leaf_indices": factor_zero_leaf_indices[:16],
        "min_variable_value": min(variable_values),
        "max_variable_value": max(variable_values),
        "min_zero_variable_value": min(zero_variable_values),
        "max_zero_variable_value": max(zero_variable_values),
    }


def build_public_factor_candidates(
    surface_record: dict[str, Any],
    max_factor_degree: int,
) -> list[dict[str, Any]]:
    p = int(surface_record["p"])
    surface = surface_record["surface"]
    coords = leaf_coords(surface_record)
    coords_by_axis_value: dict[tuple[str, int], list[dict[str, int]]] = defaultdict(list)
    for coord in coords:
        coords_by_axis_value[("b", int(coord["b"]))].append(coord)
        coords_by_axis_value[("c", int(coord["c"]))].append(coord)

    candidates = []
    for (axis, fixed_value), slice_coords in sorted(coords_by_axis_value.items()):
        specialized = slice_factor_probe.specialize_resultant(surface, axis, fixed_value, p)
        if specialized.is_zero:
            continue
        for factor in slice_factor_probe.factor_rows(specialized, p):
            if not (0 < int(factor["degree"]) <= max_factor_degree):
                continue
            candidate = evaluate_factor_candidate(surface_record, axis, fixed_value, factor, slice_coords)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def public_sort_key(candidate: dict[str, Any], rule: str) -> tuple[Any, ...]:
    if rule == "factor_sparse_zero":
        return (
            int(candidate["factor_zero_leaf_count"]),
            int(candidate["relevant_leaf_count"]),
            int(candidate["monomials"]),
            int(candidate["degree"]),
            int(candidate["min_zero_leaf_index"]),
            str(candidate["axis"]),
            int(candidate["fixed_value_mod"]),
            int(candidate["factor_index"]),
        )
    if rule == "factor_low_work":
        return (
            int(candidate["monomials"]),
            int(candidate["degree"]),
            int(candidate["factor_zero_leaf_count"]),
            int(candidate["relevant_leaf_count"]),
            int(candidate["min_zero_leaf_index"]),
            str(candidate["axis"]),
            int(candidate["fixed_value_mod"]),
        )
    if rule == "factor_singleton_low_leaf":
        return (
            abs(int(candidate["factor_zero_leaf_count"]) - 1),
            int(candidate["min_zero_leaf_index"]),
            int(candidate["monomials"]),
            int(candidate["degree"]),
            str(candidate["axis"]),
            int(candidate["fixed_value_mod"]),
        )
    if rule == "factor_singleton_low_coord":
        return (
            abs(int(candidate["factor_zero_leaf_count"]) - 1),
            int(candidate["fixed_value_mod"]),
            int(candidate["min_zero_variable_value"]),
            int(candidate["monomials"]),
            str(candidate["axis"]),
            int(candidate["factor_index"]),
        )
    if rule == "balanced_factor":
        return (
            int(candidate["factor_zero_leaf_count"]) + int(candidate["relevant_leaf_count"]) + int(candidate["monomials"]),
            int(candidate["min_zero_leaf_index"]),
            int(candidate["degree"]),
            str(candidate["axis"]),
            int(candidate["fixed_value_mod"]),
        )
    raise ValueError(f"unknown rule: {rule}")


def evaluate_selection(
    surface_record: dict[str, Any],
    candidates: list[dict[str, Any]],
    rule: str,
    top_k: int,
    row_factor: int,
    product_factor: int,
) -> dict[str, Any]:
    chosen = sorted(candidates, key=lambda candidate: public_sort_key(candidate, rule))[:top_k]
    original_pairs = remainder_factor_probe.selected_pair_set(surface_record, surface_record["surface"])
    selected_pairs = {
        (int(leaf), int(root))
        for candidate in chosen
        for leaf, root in candidate.get("selected_pairs") or []
    }
    missing_pairs = sorted(original_pairs - selected_pairs)
    extra_pairs = sorted(selected_pairs - original_pairs)
    relevant_leaf_evals = sum(int(candidate["relevant_leaf_count"]) for candidate in chosen)
    factor_zero_leaf_evals = sum(int(candidate["factor_zero_leaf_count"]) for candidate in chosen)
    recovered_root_count = sum(int(candidate["recovered_root_count"]) for candidate in chosen)
    factor_work = sum(int(candidate["monomials"]) for candidate in chosen)
    cost = surface_record["cost_inputs"]
    selected_hit_events = int(cost["selected_hit_events"])
    selected_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / max(1, row_factor))
        + ceil(relevant_leaf_evals / max(1, product_factor))
        + int(cost["selected_hit_roots"])
    )
    all_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / max(1, row_factor))
        + ceil(relevant_leaf_evals / max(1, product_factor))
        + len(surface_record["components"]["hit_roots"])
    )
    quadratic_root_work = 2 * factor_zero_leaf_evals

    def charge(core: int) -> dict[str, Any]:
        ops = (
            core
            + factor_work
            + relevant_leaf_evals
            + quadratic_root_work
            + recovered_root_count
            + 2 * selected_hit_events
        )
        rho = int(cost["generic_rho_steps"])
        return {
            "ops": ops,
            "ops_over_rho": round(ops / max(1, rho), 8),
            "beats_rho": bool(ops < rho),
        }

    selected_hit = charge(selected_hit_core)
    all_hit = charge(all_hit_core)
    return {
        "rule": rule,
        "top_k": int(top_k),
        "candidate_count": len(candidates),
        "chosen_count": len(chosen),
        "chosen_factors": [
            {
                "axis": candidate.get("axis"),
                "fixed_value": candidate.get("fixed_value"),
                "factor_index": candidate.get("factor_index"),
                "degree": candidate.get("degree"),
                "monomials": candidate.get("monomials"),
                "relevant_leaf_count": candidate.get("relevant_leaf_count"),
                "factor_zero_leaf_count": candidate.get("factor_zero_leaf_count"),
                "min_zero_leaf_index": candidate.get("min_zero_leaf_index"),
            }
            for candidate in chosen[:12]
        ],
        "original_selected_root_pair_count": len(original_pairs),
        "selected_root_pair_count": len(selected_pairs),
        "preserves_selected_root_pairs": not missing_pairs,
        "same_selected_root_pairs": not missing_pairs and not extra_pairs,
        "missing_selected_root_pairs": [[leaf, root] for leaf, root in missing_pairs[:8]],
        "extra_selected_root_pairs": [[leaf, root] for leaf, root in extra_pairs[:8]],
        "relevant_leaf_evals": relevant_leaf_evals,
        "factor_zero_leaf_evals": factor_zero_leaf_evals,
        "factor_work": factor_work,
        "recovered_root_count": recovered_root_count,
        "selected_hit_core_ops": selected_hit_core,
        "all_hit_core_ops": all_hit_core,
        "quadratic_root_work": quadratic_root_work,
        "public_factor_selected_hit_ops": selected_hit["ops"],
        "public_factor_selected_hit_ops_over_rho": selected_hit["ops_over_rho"],
        "public_factor_selected_hit_beats_rho": selected_hit["beats_rho"],
        "public_factor_all_hit_ops": all_hit["ops"],
        "public_factor_all_hit_ops_over_rho": all_hit["ops_over_rho"],
        "public_factor_all_hit_beats_rho": all_hit["beats_rho"],
    }


def evaluate_surface(surface_record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates = build_public_factor_candidates(surface_record, int(args.max_factor_degree))
    selections = [
        evaluate_selection(surface_record, candidates, rule, top_k, int(args.row_factor), int(args.product_factor))
        for rule in RULES
        for top_k in TOP_KS
        if candidates
    ]
    preserving = [selection for selection in selections if selection.get("preserves_selected_root_pairs")]
    preserving.sort(
        key=lambda row: (
            float(row.get("public_factor_all_hit_ops_over_rho") or 10**18),
            float(row.get("public_factor_selected_hit_ops_over_rho") or 10**18),
            int(row.get("top_k") or 10**18),
            str(row.get("rule")),
        )
    )
    return {
        "surface_id": surface_record["surface_id"],
        "target": surface_record["target"],
        "row_key": surface_record["row_key"],
        "challenge_seed": surface_record["challenge_seed"],
        "p": int(surface_record["p"]),
        "selected_leaf_indices": surface_record["selected_leaf_indices"],
        "public_factor_candidate_count": len(candidates),
        "selection_count": len(selections),
        "preserving_selection_count": len(preserving),
        "best_public_factor_selection": preserving[0] if preserving else None,
        "public_factor_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"selected_pairs", "terms"}
            }
            for candidate in sorted(candidates, key=lambda candidate: public_sort_key(candidate, "balanced_factor"))[:32]
        ],
        "selection_results": selections,
    }


def summarize(surfaces: list[dict[str, Any]], case_results: list[dict[str, Any]]) -> dict[str, Any]:
    verified_cases = [case for case in case_results if case.get("public_key_verified")]
    preserving_surfaces = [surface for surface in surfaces if surface.get("best_public_factor_selection")]
    selected_wins = [
        surface
        for surface in preserving_surfaces
        if bool((surface.get("best_public_factor_selection") or {}).get("public_factor_selected_hit_beats_rho"))
    ]
    all_hit_wins = [
        surface
        for surface in preserving_surfaces
        if bool((surface.get("best_public_factor_selection") or {}).get("public_factor_all_hit_beats_rho"))
    ]
    selected_ratios = [
        float((surface.get("best_public_factor_selection") or {}).get("public_factor_selected_hit_ops_over_rho"))
        for surface in preserving_surfaces
        if (surface.get("best_public_factor_selection") or {}).get("public_factor_selected_hit_ops_over_rho") is not None
    ]
    all_hit_ratios = [
        float((surface.get("best_public_factor_selection") or {}).get("public_factor_all_hit_ops_over_rho"))
        for surface in preserving_surfaces
        if (surface.get("best_public_factor_selection") or {}).get("public_factor_all_hit_ops_over_rho") is not None
    ]
    rule_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"preserving": 0, "selected_wins": 0, "all_hit_wins": 0})
    for surface in surfaces:
        for selection in surface.get("selection_results") or []:
            if not selection.get("preserves_selected_root_pairs"):
                continue
            stats = rule_counts[str(selection.get("rule"))]
            stats["preserving"] += 1
            if selection.get("public_factor_selected_hit_beats_rho"):
                stats["selected_wins"] += 1
            if selection.get("public_factor_all_hit_beats_rho"):
                stats["all_hit_wins"] += 1
    best = sorted(
        preserving_surfaces,
        key=lambda surface: (
            float((surface.get("best_public_factor_selection") or {}).get("public_factor_all_hit_ops_over_rho") or 10**18),
            str(surface.get("surface_id")),
        ),
    )[:8]
    return {
        "case_count": len(case_results),
        "verified_case_count": len(verified_cases),
        "surface_count": len(surfaces),
        "surfaces_with_public_factor_candidates_count": sum(
            1 for surface in surfaces if int(surface.get("public_factor_candidate_count") or 0)
        ),
        "surfaces_with_public_factor_preserving_selection": len(preserving_surfaces),
        "public_factor_selected_hit_below_rho_count": len(selected_wins),
        "public_factor_all_hit_below_rho_count": len(all_hit_wins),
        "min_public_factor_selected_hit_ops_over_rho": round(min(selected_ratios), 8) if selected_ratios else None,
        "min_public_factor_all_hit_ops_over_rho": round(min(all_hit_ratios), 8) if all_hit_ratios else None,
        "rule_counts": dict(sorted(rule_counts.items())),
        "best_public_factor_surfaces": [
            {
                "surface_id": surface.get("surface_id"),
                "target": surface.get("target"),
                "row_key": surface.get("row_key"),
                "p": surface.get("p"),
                "best_public_factor_selection": surface.get("best_public_factor_selection"),
            }
            for surface in best
        ],
        "interpretation": (
            "This narrows the public selector from whole slices to individual "
            "low-degree resultant factors. A below-rho all-hit result would be "
            "a stronger public quotient candidate; if only selected-hit wins "
            "survive, the remaining gap is honest hit-stream publicness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=cross_surface_probe.compress_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=cross_surface_probe.compress_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=cross_surface_probe.compress_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=cross_surface_probe.compress_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--row-pool", type=int, default=512)
    parser.add_argument("--row-count", type=int, default=128)
    parser.add_argument("--scout-limit", type=int, default=192)
    parser.add_argument("--scout-mode", default="s3_coeff_spread")
    parser.add_argument("--scout-order", default="eval_cover_hits_high")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--factor-base-size", type=int, default=16)
    parser.add_argument("--max-relations", type=int, default=96)
    parser.add_argument("--min-distinct-indices", type=int, default=4)
    parser.add_argument("--min-unsigned-distinct-indices", type=int, default=2)
    parser.add_argument("--allow-combined-coefficients", dest="require_unit_coefficients", action="store_false")
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--max-factor-degree", type=int, default=1)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = load_json(args.signature_source)
    positive_cases = [case for case in signature.get("positive_cases") or [] if isinstance(case, dict)]
    if args.max_cases and args.max_cases > 0:
        positive_cases = positive_cases[: args.max_cases]

    bank = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    bank_rows = {
        cross_surface_probe.compress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict) and cross_surface_probe.compress_probe.row_key(row)
    }
    specs_by_target = cross_surface_probe.leaf_trim_probe.specs_by_target_and_key(
        cross_surface_probe.salt_neighborhood_probe.witness_specs(direct_source, bank_rows, radius)
    )
    verifier = cross_surface_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    surface_records, case_results = cross_surface_probe.materialize_surface_records(
        verifier,
        records,
        config_source,
        specs_by_target,
        positive_cases,
        args,
    )
    surfaces = [evaluate_surface(surface_record, args) for surface_record in surface_records]
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_predictor_probe_v1",
        "method": "signature_coupled_public_individual_factor_predictor_audit",
        "parameters": {
            "campaign_task_dir": str(CAMPAIGN_TASK_DIR),
            "signature_source": str(args.signature_source),
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "radius": radius,
            "max_cases": args.max_cases,
            "row_pool": args.row_pool,
            "scout_limit": args.scout_limit,
            "scout_mode": args.scout_mode,
            "scout_order": args.scout_order,
            "selected_limit": args.selected_limit,
            "factor_base_size": args.factor_base_size,
            "seed": args.seed,
            "rules": list(RULES),
            "top_ks": list(TOP_KS),
            "max_factor_degree": args.max_factor_degree,
        },
        "summary": summarize(surfaces, case_results),
        "surfaces": surfaces,
        "cases": case_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
