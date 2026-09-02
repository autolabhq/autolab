#!/usr/bin/env python3
"""Diagnose public ways to close the remaining FFE quotient-route miss.

The route ensemble covers 17/18 current FFE quotient surfaces below rho.  The
remaining miss has public support for both a false candidate and a preserving
candidate, so this probe audits whether simple public re-rankers or cost
accounting variants can close it without introducing false positives.

This is explicitly a diagnostic.  Policies are public, while preservation
labels are used only in the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_support_probe as support_probe


base_probe = support_probe.base_probe
locator_probe = support_probe.locator_probe

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_quotient_miss_closure_probe.json"
DEFAULT_ROUTE_ENSEMBLE_SOURCE = DEFAULT_STATE_DIR / "ffe_public_quotient_route_ensemble_probe.json"

SUPPORT_POLICY = "predicted_valid_single_root_unique"
ORDER_POLICIES = (
    "current_rank",
    "low_predicted_leaf",
    "low_factor_index",
)
COST_MODELS = (
    "strict_unique_full_scan_all_hit",
    "ordered_first_accept_scan_all_hit",
    "incremental_selector_scan_all_hit",
    "ranked_first_all_hit",
    "fingerprint_table_all_hit",
    "ordered_first_accept_scan_selected_hit",
    "ranked_first_selected_hit",
)
KNOWN_ROUTE_MISS_SURFACE_ID = (
    "22050.cf1@11731|22050.cf1@11731:uniform:256:salt164|"
    "ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:57:22050.cf1@11731"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def clean_ratio(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transfer_index(surface: dict[str, Any]) -> int | None:
    return base_probe.transfer_index(surface)


def selected_hit_charge(
    surface_record: dict[str, Any],
    signature: dict[str, Any],
    selector_ops: int,
) -> dict[str, Any]:
    cost = surface_record["cost_inputs"]
    core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(int(signature.get("predicted_leaf_count") or 0) / 4096)
        + int(cost["selected_hit_roots"])
    )
    ops = (
        core
        + int(selector_ops)
        + int(signature.get("predicted_leaf_eval_ops") or 0)
        + int(signature.get("predicted_quadratic_root_work") or 0)
        + int(signature.get("predicted_recovered_root_count") or 0)
        + 2 * int(cost["selected_hit_events"])
    )
    rho = int(cost["generic_rho_steps"])
    return {
        "estimated_selected_hit_ops": ops,
        "estimated_selected_hit_ops_over_rho": round(ops / max(1, rho), 8),
        "estimated_selected_hit_beats_rho": bool(ops < rho),
    }


def support_row(
    surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any],
    learned_row: dict[str, Any],
    current_rank: int,
    selector_ops: int,
    locator_policy: str,
) -> dict[str, Any]:
    signature = support_probe.predicted_support(surface, surface_record, candidate, learned_row, locator_policy)
    support_ops = support_probe.support_check_ops(signature)
    all_hit_ranked = support_probe.estimated_all_hit_charge(surface_record, signature, selector_ops)
    selected_hit_ranked = selected_hit_charge(surface_record, signature, selector_ops)
    factor_index = base_probe.factor_index(candidate)
    predicted_leaf_indices = [int(index) for index in signature.get("predicted_valid_root_indices") or []]
    if not predicted_leaf_indices:
        predicted_leaf_indices = [int(index) for index in signature.get("predicted_leaf_indices") or []]
    min_predicted_leaf = min(predicted_leaf_indices, default=10**9)
    return {
        "candidate_name": candidate.get("candidate_name"),
        "candidate": base_probe.compact_candidate(surface, candidate),
        "current_rank": int(current_rank),
        "factor_index": factor_index,
        "fingerprint_key": base_probe.fingerprint_key(surface, candidate),
        "learned_support": int(learned_row.get("support") or 0),
        "learned_preserving": int(learned_row.get("preserving") or 0),
        "learned_false_positive": int(learned_row.get("false_positive") or 0),
        "support_accepts": support_probe.support_accepts(signature, SUPPORT_POLICY),
        "support_check_ops": support_ops,
        "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
        "same_selected_root_pairs": bool(candidate.get("same_selected_root_pairs")),
        "min_predicted_leaf": min_predicted_leaf,
        "signature": signature,
        "ranked_first_all_hit": all_hit_ranked,
        "ranked_first_selected_hit": selected_hit_ranked,
    }


def order_key(row: dict[str, Any], policy: str) -> tuple[Any, ...]:
    if policy == "current_rank":
        return (int(row["current_rank"]),)
    if policy == "low_predicted_leaf":
        return (
            int(row.get("min_predicted_leaf") or 10**9),
            int(row["current_rank"]),
            str(row.get("candidate_name") or ""),
        )
    if policy == "low_factor_index":
        return (
            int(row.get("factor_index") if row.get("factor_index") is not None else 10**9),
            int(row["current_rank"]),
            str(row.get("candidate_name") or ""),
        )
    raise ValueError(f"unknown order policy: {policy}")


def candidate_charge(
    surface_record: dict[str, Any],
    row: dict[str, Any],
    effective_selector_ops: int,
    cost_model: str,
) -> dict[str, Any]:
    effective_selector_ops = max(0, int(effective_selector_ops))
    if cost_model.endswith("_selected_hit"):
        charge = selected_hit_charge(surface_record, row["signature"], effective_selector_ops)
        return {
            "ops": charge["estimated_selected_hit_ops"],
            "ops_over_rho": charge["estimated_selected_hit_ops_over_rho"],
            "beats_rho": charge["estimated_selected_hit_beats_rho"],
            "membership_scope": "selected_hit_roots",
            "effective_selector_ops": effective_selector_ops,
        }
    charge = support_probe.estimated_all_hit_charge(surface_record, row["signature"], effective_selector_ops)
    return {
        "ops": charge["estimated_all_hit_ops"],
        "ops_over_rho": charge["estimated_all_hit_ops_over_rho"],
        "beats_rho": charge["estimated_all_hit_beats_rho"],
        "membership_scope": "all_hit_roots",
        "effective_selector_ops": effective_selector_ops,
    }


def choose_for_policy(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    surface_record: dict[str, Any],
    selector_ops: int,
    order_policy: str,
    cost_model: str,
) -> dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: order_key(row, order_policy))
    accepted = [row for row in ordered if row.get("support_accepts")]
    if cost_model == "strict_unique_full_scan_all_hit":
        if len(accepted) != 1:
            return None
        chosen = accepted[0]
        support_scan_extra = sum(int(row.get("support_check_ops") or 0) for row in rows) - int(
            chosen.get("support_check_ops") or 0
        )
        charge = candidate_charge(surface_record, chosen, int(selector_ops) + support_scan_extra, cost_model)
        if charge["beats_rho"]:
            return {
                **chosen,
                "charge": charge,
                "ordered_index": ordered.index(chosen),
                "scan_extra_ops": support_scan_extra,
            }
        return None
    if cost_model.startswith("ordered_first_accept_scan_"):
        support_scan_extra = 0
        for index, row in enumerate(ordered):
            if not row.get("support_accepts"):
                support_scan_extra += int(row.get("support_check_ops") or 0)
                continue
            charge = candidate_charge(surface_record, row, int(selector_ops) + support_scan_extra, cost_model)
            if charge["beats_rho"]:
                return {**row, "charge": charge, "ordered_index": index, "scan_extra_ops": support_scan_extra}
            return None
        return None
    if cost_model == "incremental_selector_scan_all_hit":
        if order_policy != "low_factor_index":
            return None
        row_by_name = {str(row.get("candidate_name") or ""): row for row in rows}
        incremental_selector_ops = 0
        support_scan_extra = 0
        for index, candidate in enumerate(
            sorted(
                candidates,
                key=lambda candidate: (
                    int(base_probe.factor_index(candidate) if base_probe.factor_index(candidate) is not None else 10**9),
                    int(candidate.get("factor_total_degree") or 10**9),
                    int(candidate.get("factor_monomials") or 10**9),
                    str(candidate.get("candidate_name") or ""),
                ),
            )
        ):
            incremental_selector_ops += int(candidate.get("factor_monomials") or 0)
            row = row_by_name.get(str(candidate.get("candidate_name") or ""))
            if row is None:
                continue
            if not row.get("support_accepts"):
                support_scan_extra += int(row.get("support_check_ops") or 0)
                continue
            charge = candidate_charge(
                surface_record,
                row,
                incremental_selector_ops + support_scan_extra,
                cost_model,
            )
            if charge["beats_rho"]:
                return {
                    **row,
                    "charge": charge,
                    "ordered_index": index,
                    "scan_extra_ops": support_scan_extra,
                    "incremental_selector_ops": incremental_selector_ops,
                }
            return None
        return None
    if cost_model.startswith("ranked_first_"):
        for index, row in enumerate(accepted):
            charge = candidate_charge(surface_record, row, int(selector_ops), cost_model)
            if charge["beats_rho"]:
                return {**row, "charge": charge, "ordered_index": index, "scan_extra_ops": 0}
        return None
    if cost_model == "fingerprint_table_all_hit":
        for index, row in enumerate(accepted):
            charge = candidate_charge(surface_record, row, 0, cost_model)
            if charge["beats_rho"]:
                return {**row, "charge": charge, "ordered_index": index, "scan_extra_ops": -int(selector_ops)}
        return None
    raise ValueError(f"unknown cost model: {cost_model}")


def evaluate_surface(
    surface: dict[str, Any],
    surface_record: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    locator_policy: str,
    max_factor_degree: int,
    factor_index_prior: str,
) -> dict[str, Any]:
    candidates, selector_ops, matched = support_probe.ranked_candidates_for_support(
        surface,
        learned,
        max_factor_degree,
        factor_index_prior,
    )
    rows = []
    for rank, candidate in enumerate(matched):
        key = base_probe.fingerprint_key(surface, candidate)
        rows.append(support_row(surface, surface_record, candidate, learned[key], rank, selector_ops, locator_policy))
    accepted = [row for row in rows if row.get("support_accepts")]
    selections = []
    for order_policy in ORDER_POLICIES:
        for cost_model in COST_MODELS:
            chosen = choose_for_policy(candidates, rows, surface_record, selector_ops, order_policy, cost_model)
            selections.append(
                {
                    "order_policy": order_policy,
                    "cost_model": cost_model,
                    "chosen": compact_candidate_row(chosen) if chosen else None,
                }
            )
    return {
        "surface_id": surface.get("surface_id"),
        "target": surface.get("target"),
        "row_key": surface.get("row_key"),
        "challenge_seed": surface.get("challenge_seed"),
        "transfer_index": transfer_index(surface),
        "candidate_count": len(candidates),
        "matched_candidate_count": len(matched),
        "selector_ops": int(selector_ops),
        "support_accepted_candidate_count": len(accepted),
        "support_accepted_preserving_count": sum(1 for row in accepted if row.get("preserves_selected_root_pairs")),
        "support_accepted_false_positive_count": sum(1 for row in accepted if not row.get("preserves_selected_root_pairs")),
        "best_accepted_candidates": [compact_candidate_row(row) for row in accepted[:8]],
        "selections": selections,
    }


def compact_candidate_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    signature = row.get("signature") or {}
    return {
        "candidate_name": row.get("candidate_name"),
        "current_rank": row.get("current_rank"),
        "ordered_index": row.get("ordered_index"),
        "factor_index": row.get("factor_index"),
        "min_predicted_leaf": row.get("min_predicted_leaf"),
        "learned_support": row.get("learned_support"),
        "learned_preserving": row.get("learned_preserving"),
        "learned_false_positive": row.get("learned_false_positive"),
        "support_check_ops": row.get("support_check_ops"),
        "scan_extra_ops": row.get("scan_extra_ops"),
        "incremental_selector_ops": row.get("incremental_selector_ops"),
        "preserves_selected_root_pairs": row.get("preserves_selected_root_pairs"),
        "same_selected_root_pairs": row.get("same_selected_root_pairs"),
        "predicted_leaf_indices": signature.get("predicted_leaf_indices"),
        "predicted_valid_root_indices": signature.get("predicted_valid_root_indices"),
        "predicted_recovered_root_count": signature.get("predicted_recovered_root_count"),
        "ranked_first_all_hit_ops_over_rho": (row.get("ranked_first_all_hit") or {}).get(
            "estimated_all_hit_ops_over_rho"
        ),
        "ranked_first_selected_hit_ops_over_rho": (row.get("ranked_first_selected_hit") or {}).get(
            "estimated_selected_hit_ops_over_rho"
        ),
        "charge": row.get("charge"),
    }


def split_specs(surfaces: list[dict[str, Any]], min_train_transfers: int) -> list[tuple[str, set[str], set[str]]]:
    surface_ids = {str(surface.get("surface_id")) for surface in surfaces}
    specs: list[tuple[str, set[str], set[str]]] = []
    transfers = sorted({transfer_index(surface) for surface in surfaces if transfer_index(surface) is not None})
    for transfer in transfers:
        test = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) == transfer}
        specs.append((f"holdout_transfer_{transfer}", surface_ids - test, test))
    for idx in range(int(min_train_transfers), len(transfers)):
        train_transfers = set(transfers[:idx])
        holdout = transfers[idx]
        train = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) in train_transfers}
        test = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) == holdout}
        specs.append((f"rolling_to_transfer_{holdout}", train, test))
    return specs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for selection in row["selections"]:
            key = (selection["order_policy"], selection["cost_model"])
            if selection.get("chosen"):
                policy_rows[key].append({**selection["chosen"], "surface_id": row["surface_id"], "target": row["target"]})
    summaries = []
    for (order_policy, cost_model), values in sorted(policy_rows.items()):
        unique: dict[str, dict[str, Any]] = {}
        for value in values:
            current = unique.get(str(value["surface_id"]))
            if current is None or float(value["charge"]["ops_over_rho"]) < float(current["charge"]["ops_over_rho"]):
                unique[str(value["surface_id"])] = value
        selected = list(unique.values())
        ratios = [float(row["charge"]["ops_over_rho"]) for row in selected]
        known_miss = unique.get(KNOWN_ROUTE_MISS_SURFACE_ID)
        summaries.append(
            {
                "order_policy": order_policy,
                "cost_model": cost_model,
                "selected_unique_surface_count": len(selected),
                "selected_preserving_count": sum(1 for row in selected if row.get("preserves_selected_root_pairs")),
                "selected_false_positive_count": sum(1 for row in selected if not row.get("preserves_selected_root_pairs")),
                "known_route_miss_selected": bool(known_miss),
                "known_route_miss_preserving": bool(known_miss and known_miss.get("preserves_selected_root_pairs")),
                "known_route_miss_ops_over_rho": (known_miss.get("charge") or {}).get("ops_over_rho")
                if known_miss
                else None,
                "min_ops_over_rho": min(ratios) if ratios else None,
                "mean_ops_over_rho": round(mean(ratios), 8) if ratios else None,
                "max_ops_over_rho": max(ratios) if ratios else None,
                "selected_target_counts": dict(sorted(Counter(row["target"] for row in selected).items())),
            }
        )
    summaries.sort(
        key=lambda row: (
            int(row["selected_false_positive_count"]),
            -int(row["selected_preserving_count"]),
            float(row["mean_ops_over_rho"] or 10**18),
        )
    )
    return {
        "surface_evaluations": len(rows),
        "unique_surface_count": len({str(row["surface_id"]) for row in rows}),
        "known_route_miss_surface_id": KNOWN_ROUTE_MISS_SURFACE_ID,
        "policy_summaries": summaries,
        "best_false_positive_free_policy": next(
            (row for row in summaries if int(row["selected_false_positive_count"]) == 0),
            None,
        ),
    }


def selected_by_policy(rows: list[dict[str, Any]], order_policy: str, cost_model: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        for selection in row["selections"]:
            if selection["order_policy"] != order_policy or selection["cost_model"] != cost_model:
                continue
            chosen = selection.get("chosen")
            if not chosen:
                continue
            surface_id = str(row["surface_id"])
            current = selected.get(surface_id)
            if current is None or float(chosen["charge"]["ops_over_rho"]) < float(
                current["chosen"]["charge"]["ops_over_rho"]
            ):
                selected[surface_id] = {
                    "surface_id": surface_id,
                    "target": row["target"],
                    "split": row.get("split"),
                    "chosen": chosen,
                }
    return selected


def hybrid_union_summary(
    rows: list[dict[str, Any]],
    route_ensemble: dict[str, Any],
    order_policy: str,
    cost_model: str,
) -> dict[str, Any]:
    baseline_routes = {
        str(route.get("surface_id")): route
        for route in route_ensemble.get("selected_routes") or []
        if isinstance(route, dict)
    }
    support_routes = selected_by_policy(rows, order_policy, cost_model)
    added = {surface_id: row for surface_id, row in support_routes.items() if surface_id not in baseline_routes}
    hybrid_routes: dict[str, dict[str, Any]] = {
        surface_id: {"source": "route_ensemble", "route": route}
        for surface_id, route in baseline_routes.items()
    }
    for surface_id, row in added.items():
        hybrid_routes[surface_id] = {"source": "incremental_support", **row}
    false_positive_count = 0
    preserving_count = 0
    for row in hybrid_routes.values():
        if row["source"] == "route_ensemble":
            preserves = bool(row["route"].get("preserves_selected_root_pairs"))
        else:
            preserves = bool(row["chosen"].get("preserves_selected_root_pairs"))
        preserving_count += int(preserves)
        false_positive_count += int(not preserves)
    surface_universe_count = int((route_ensemble.get("summary") or {}).get("surface_universe_count") or 0)
    return {
        "baseline_selected_count": len(baseline_routes),
        "support_policy_order": order_policy,
        "support_policy_cost_model": cost_model,
        "support_selected_count": len(support_routes),
        "support_added_count": len(added),
        "support_added_surfaces": [
            {
                "surface_id": surface_id,
                "target": row["target"],
                "split": row.get("split"),
                "candidate_name": row["chosen"].get("candidate_name"),
                "factor_index": row["chosen"].get("factor_index"),
                "ops_over_rho": row["chosen"]["charge"]["ops_over_rho"],
                "preserves_selected_root_pairs": row["chosen"].get("preserves_selected_root_pairs"),
            }
            for surface_id, row in sorted(added.items())
        ],
        "hybrid_selected_count": len(hybrid_routes),
        "hybrid_preserving_count": preserving_count,
        "hybrid_false_positive_count": false_positive_count,
        "surface_universe_count": surface_universe_count,
        "hybrid_closes_surface_universe": bool(
            surface_universe_count and len(hybrid_routes) == surface_universe_count and false_positive_count == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-factor-source", type=Path, default=base_probe.DEFAULT_SAGE_FACTOR_SOURCE)
    parser.add_argument("--signature-source", type=Path, default=base_probe.DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=base_probe.cross_surface_probe.compress_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=base_probe.cross_surface_probe.compress_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=base_probe.cross_surface_probe.compress_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=base_probe.cross_surface_probe.compress_probe.DEFAULT_TRANSFER_SOURCE)
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
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--require-clean-fingerprint", action="store_true")
    parser.add_argument("--allow-noisy-fingerprint", action="store_true")
    parser.add_argument("--fingerprint-mode", choices=support_probe.FINGERPRINT_MODES, default="raw")
    parser.add_argument("--factor-index-prior", choices=support_probe.FACTOR_INDEX_PRIORS, default="none")
    parser.add_argument("--locator-policy", choices=locator_probe.LOCATOR_POLICIES, default="mode1")
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--route-ensemble-source", type=Path, default=DEFAULT_ROUTE_ENSEMBLE_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    support_probe.install_fingerprint_mode(str(args.fingerprint_mode))
    sage_source = load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = base_probe.materialize_records(args)
    rows = []
    surface_by_id = {str(surface.get("surface_id")): surface for surface in surfaces}
    for split_name, train_ids, test_ids in split_specs(surfaces, int(args.min_train_transfers)):
        learned = support_probe.learn_leaf_tables_for_support(
            surfaces,
            surface_records,
            train_ids,
            int(args.max_factor_degree),
            int(args.min_support),
            bool(args.require_clean_fingerprint),
            bool(args.allow_noisy_fingerprint),
        )
        for surface_id in sorted(test_ids):
            if surface_id not in surface_records or surface_id not in surface_by_id:
                continue
            evaluated = evaluate_surface(
                surface_by_id[surface_id],
                surface_records[surface_id],
                learned,
                str(args.locator_policy),
                int(args.max_factor_degree),
                str(args.factor_index_prior),
            )
            evaluated["split"] = split_name
            rows.append(evaluated)
    route_ensemble = load_json(args.route_ensemble_source)
    summary = summarize(rows)
    summary["hybrid_union_with_route_ensemble"] = hybrid_union_summary(
        rows,
        route_ensemble,
        "low_factor_index",
        "incremental_selector_scan_all_hit",
    )
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_quotient_miss_closure_probe_v1",
        "method": "support_candidate_public_rerank_and_cost_sensitivity",
        "parameters": {
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "route_ensemble_source": str(args.route_ensemble_source),
            "fingerprint_mode": args.fingerprint_mode,
            "factor_index_prior": args.factor_index_prior,
            "locator_policy": args.locator_policy,
            "support_policy": SUPPORT_POLICY,
            "order_policies": list(ORDER_POLICIES),
            "cost_models": list(COST_MODELS),
            "label_fields_used_only_for_audit": [
                "preserves_selected_root_pairs",
                "same_selected_root_pairs",
                "selected_surface_zero_leaves",
            ],
        },
        "case_count": len(case_results),
        "summary": summary,
        "surface_rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
