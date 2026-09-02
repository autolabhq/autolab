#!/usr/bin/env python3
"""Order-free public support guard for FFE factor-fingerprint leaf locators.

The factor-index guard showed that public factor order can reject some bad
holdout factors, but factor order is a brittle proxy for algebraic structure.
This probe instead asks whether a matched factor has the public support shape
predicted by calibration: it must vanish, and optionally recover a hit root, at
the learned leaf index before it is accepted.

Holdout selection may use the factor fingerprint, the learned leaf-index table,
and public evaluation/root recovery at those predicted leaves.  It may not use
the selected-leaf label or preservation labels until the audit row is scored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_probe as locator_probe


base_probe = locator_probe.base_probe

DEFAULT_OUT = (
    base_probe.DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_support_probe.json"
)

SUPPORT_POLICIES = (
    "none",
    "predicted_zero_any",
    "predicted_zero_all",
    "predicted_valid_any",
    "predicted_valid_any_below_rho",
    "predicted_valid_all",
    "predicted_valid_single_root",
    "predicted_valid_single_root_unique",
    "predicted_valid_single_root_unique_below_rho",
)

FINGERPRINT_MODES = (
    "raw",
    "degree_shape",
    "linear_axis_shape",
    "linear_coeff_shape",
    "signed_coeff_shape",
)

FACTOR_INDEX_PRIORS = (
    "none",
    "preserving_hist",
)


RAW_FINGERPRINT_KEY = base_probe.fingerprint_key


def _normalized_terms(
    surface: dict[str, Any],
    candidate: dict[str, Any],
    *,
    signed: bool = False,
) -> list[tuple[int, int, int]]:
    p = int(surface.get("p") or 0)
    terms = base_probe.factor_terms(surface, candidate)
    if not terms or p <= 0:
        return []
    nonconstant = [
        ((int(bd), int(cd)), int(coeff) % p)
        for (bd, cd), coeff in terms.items()
        if int(bd) or int(cd)
    ]
    if not nonconstant:
        return []
    pivot = None
    for degrees, coeff in sorted(nonconstant):
        if coeff % p:
            pivot = coeff % p
            break
    if not pivot:
        return []
    inv_pivot = pow(pivot, -1, p)
    out = []
    for (bd, cd), coeff in terms.items():
        value = (int(coeff) % p) * inv_pivot % p
        centered = value - p if signed and value > p // 2 else min(value, p - value)
        out.append((int(bd), int(cd), int(centered)))
    return sorted(out)


def fingerprint_key_for_mode(
    surface: dict[str, Any],
    candidate: dict[str, Any],
    fingerprint_mode: str,
) -> tuple[Any, ...] | None:
    if fingerprint_mode == "raw":
        return RAW_FINGERPRINT_KEY(surface, candidate)
    normalized = _normalized_terms(surface, candidate)
    if not normalized:
        return None
    degree_shape = tuple(sorted((bd, cd) for bd, cd, _coeff in normalized))
    if fingerprint_mode == "degree_shape":
        return ("degree_shape", degree_shape)
    nonconstant = tuple(sorted((bd, cd) for bd, cd, _coeff in normalized if bd or cd))
    if fingerprint_mode == "linear_axis_shape":
        return ("linear_axis_shape", nonconstant, bool(any((bd, cd) == (0, 0) for bd, cd, _ in normalized)))
    if fingerprint_mode == "linear_coeff_shape":
        coeff_classes = tuple(
            sorted(
                (
                    bd,
                    cd,
                    "zero" if coeff == 0 else "unit" if coeff == 1 else "small" if coeff <= 64 else "other",
                )
                for bd, cd, coeff in normalized
            )
        )
        return ("linear_coeff_shape", coeff_classes)
    if fingerprint_mode == "signed_coeff_shape":
        return ("signed_coeff_shape", tuple(_normalized_terms(surface, candidate, signed=True)))
    raise ValueError(f"unknown fingerprint mode: {fingerprint_mode}")


def install_fingerprint_mode(fingerprint_mode: str) -> None:
    def selected_key(surface: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, ...] | None:
        return fingerprint_key_for_mode(surface, candidate, fingerprint_mode)

    base_probe.fingerprint_key = selected_key
    locator_probe.base_probe.fingerprint_key = selected_key


def learn_leaf_tables_for_support(
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    train_ids: set[str],
    max_degree: int,
    min_support: int,
    require_clean: bool,
    allow_noisy: bool,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    tables: dict[tuple[Any, ...], dict[str, Any]] = {}
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in train_ids or surface_id not in surface_records:
            continue
        surface_record = surface_records[surface_id]
        for candidate in base_probe.public_candidates(surface, max_degree):
            key = base_probe.fingerprint_key(surface, candidate)
            if key is None:
                continue
            row = tables.setdefault(
                key,
                {
                    "fingerprint_key": key,
                    "support": 0,
                    "preserving": 0,
                    "false_positive": 0,
                    "selected_leaf_histogram": Counter(),
                    "zero_leaf_histogram": Counter(),
                    "factor_index_histogram": Counter(),
                },
            )
            row["support"] += 1
            if candidate.get("preserves_selected_root_pairs"):
                row["preserving"] += 1
                factor_index = base_probe.factor_index(candidate)
                if factor_index is not None:
                    row["factor_index_histogram"][int(factor_index)] += 1
                for leaf_index in locator_probe.selected_recovering_zero_indices(surface_record, surface, candidate):
                    row["selected_leaf_histogram"][leaf_index] += 1
                for leaf_index in locator_probe.candidate_zero_indices(surface_record, surface, candidate):
                    row["zero_leaf_histogram"][leaf_index] += 1
            else:
                row["false_positive"] += 1
    return {
        key: row
        for key, row in tables.items()
        if int(row["preserving"]) >= min_support
        and (allow_noisy or int(row["preserving"]) > int(row["false_positive"]))
        and (not require_clean or int(row["false_positive"]) == 0)
        and row["selected_leaf_histogram"]
    }


def predicted_support(
    surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any],
    learned_row: dict[str, Any],
    locator_policy: str,
) -> dict[str, Any]:
    p = int(surface_record["p"])
    coords_by_index = {int(coord["leaf_index"]): coord for coord in base_probe.leaf_coords(surface_record)}
    max_leaf_index = max(coords_by_index, default=-1)
    indices = locator_probe.predicted_indices(learned_row, locator_policy, max_leaf_index)
    terms = base_probe.factor_terms(surface, candidate)
    zero_indices = []
    valid_root_indices = []
    recovered_roots: dict[int, list[int]] = {}
    false_zero_count = 0
    for leaf_index in indices:
        coord = coords_by_index.get(int(leaf_index))
        if coord is None:
            continue
        if not terms or base_probe.eval_factor(terms, int(coord["b"]), int(coord["c"]), p) != 0:
            continue
        zero_indices.append(int(leaf_index))
        leaf = surface_record["components"]["leaves"][int(leaf_index)]
        roots = base_probe.slice_quadratic_probe.recover_roots_for_leaf(
            surface_record,
            leaf,
            int(coord["b"]),
            int(coord["c"]),
        )
        if roots:
            valid_root_indices.append(int(leaf_index))
            recovered_roots[int(leaf_index)] = [int(root) for root in roots]
        else:
            false_zero_count += 1
    factor_monomials = int(candidate.get("factor_monomials") or 0)
    return {
        "predicted_leaf_count": len(indices),
        "predicted_leaf_indices": indices[:32],
        "predicted_zero_count": len(zero_indices),
        "predicted_zero_indices": zero_indices[:32],
        "predicted_valid_root_leaf_count": len(valid_root_indices),
        "predicted_valid_root_indices": valid_root_indices[:32],
        "predicted_recovered_root_count": sum(len(roots) for roots in recovered_roots.values()),
        "predicted_false_zero_count": false_zero_count,
        "predicted_leaf_eval_ops": len(indices) * factor_monomials,
        "predicted_quadratic_root_work": 2 * len(zero_indices),
        "predicted_root_membership_work": sum(len(roots) for roots in recovered_roots.values()),
    }


def support_check_ops(signature: dict[str, Any]) -> int:
    return (
        int(signature.get("predicted_leaf_eval_ops") or 0)
        + int(signature.get("predicted_quadratic_root_work") or 0)
        + int(signature.get("predicted_root_membership_work") or 0)
    )


def support_accepts(signature: dict[str, Any], support_policy: str) -> bool:
    predicted = int(signature.get("predicted_leaf_count") or 0)
    zero = int(signature.get("predicted_zero_count") or 0)
    valid = int(signature.get("predicted_valid_root_leaf_count") or 0)
    recovered = int(signature.get("predicted_recovered_root_count") or 0)
    if support_policy == "none":
        return True
    if predicted <= 0:
        return False
    if support_policy == "predicted_zero_any":
        return zero > 0
    if support_policy == "predicted_zero_all":
        return zero == predicted
    if support_policy == "predicted_valid_any":
        return valid > 0
    if support_policy == "predicted_valid_any_below_rho":
        return valid > 0
    if support_policy == "predicted_valid_all":
        return valid == predicted
    if support_policy in {
        "predicted_valid_single_root",
        "predicted_valid_single_root_unique",
        "predicted_valid_single_root_unique_below_rho",
    }:
        return valid > 0 and recovered == 1
    raise ValueError(f"unknown support policy: {support_policy}")


def factor_index_prior_rank(
    candidate: dict[str, Any],
    learned_row: dict[str, Any],
    factor_index_prior: str,
) -> tuple[Any, ...]:
    if factor_index_prior == "none":
        return (0,)
    factor_index = base_probe.factor_index(candidate)
    if factor_index is None:
        return (1, 10**9)
    hist = learned_row.get("factor_index_histogram") or {}
    hist_items = {int(key): int(value) for key, value in dict(hist).items()}
    if not hist_items:
        return (0, int(factor_index))
    exact = hist_items.get(int(factor_index), 0)
    distance = min(abs(int(factor_index) - key) for key in hist_items)
    if factor_index_prior == "preserving_hist":
        return (-exact, distance, int(factor_index))
    raise ValueError(f"unknown factor index prior: {factor_index_prior}")


def ranked_candidates_for_support(
    surface: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
    factor_index_prior: str,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    ranked_keys = {
        key: rank
        for rank, key in enumerate(sorted(learned, key=lambda key: locator_probe.learned_rank(learned[key])))
    }
    candidates = base_probe.public_candidates(surface, max_degree)
    selector_ops = sum(int(candidate.get("factor_monomials") or 0) for candidate in candidates)
    matched = [
        candidate
        for candidate in candidates
        if base_probe.fingerprint_key(surface, candidate) in ranked_keys
    ]
    matched.sort(
        key=lambda candidate: (
            ranked_keys[base_probe.fingerprint_key(surface, candidate)],
            factor_index_prior_rank(candidate, learned[base_probe.fingerprint_key(surface, candidate)], factor_index_prior),
            int(candidate.get("factor_total_degree") or 10**9),
            int(candidate.get("factor_monomials") or 10**9),
            repr(base_probe.fingerprint_key(surface, candidate)),
            str(candidate.get("candidate_name") or ""),
        )
    )
    return candidates, selector_ops, matched


def estimated_all_hit_charge(
    surface_record: dict[str, Any],
    signature: dict[str, Any],
    selector_ops: int,
) -> dict[str, Any]:
    cost = surface_record["cost_inputs"]
    all_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(int(signature.get("predicted_leaf_count") or 0) / 4096)
        + len(surface_record["components"]["hit_roots"])
    )
    ops = (
        all_hit_core
        + int(selector_ops)
        + int(signature.get("predicted_leaf_eval_ops") or 0)
        + int(signature.get("predicted_quadratic_root_work") or 0)
        + int(signature.get("predicted_recovered_root_count") or 0)
        + 2 * int(cost["selected_hit_events"])
    )
    rho = int(cost["generic_rho_steps"])
    return {
        "estimated_all_hit_ops": ops,
        "estimated_all_hit_ops_over_rho": round(ops / max(1, rho), 8),
        "estimated_all_hit_beats_rho": bool(ops < rho),
    }


def choose_supported_candidate(
    surface: dict[str, Any],
    surface_record: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
    locator_policy: str,
    support_policy: str,
    factor_index_prior: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, int, int, int, int, dict[str, Any] | None]:
    candidates, selector_ops, matched = ranked_candidates_for_support(
        surface,
        learned,
        max_degree,
        factor_index_prior,
    )
    rejected = 0
    support_selector_extra_ops = 0
    if support_policy in {
        "predicted_valid_single_root_unique",
        "predicted_valid_single_root_unique_below_rho",
    }:
        accepted: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        scanned_ops = 0
        for candidate in matched:
            key = base_probe.fingerprint_key(surface, candidate)
            learned_row = learned[key]
            signature = predicted_support(surface, surface_record, candidate, learned_row, locator_policy)
            scanned_ops += support_check_ops(signature)
            if support_accepts(signature, support_policy):
                accepted.append((candidate, learned_row, signature))
            else:
                rejected += 1
        if len(accepted) != 1:
            return None, None, len(candidates), selector_ops + scanned_ops, len(matched), len(matched), scanned_ops, None
        candidate, learned_row, signature = accepted[0]
        support_selector_extra_ops = scanned_ops - support_check_ops(signature)
        estimated_charge = estimated_all_hit_charge(
            surface_record,
            signature,
            selector_ops + support_selector_extra_ops,
        )
        signature = {**signature, **estimated_charge}
        if support_policy.endswith("_below_rho") and not estimated_charge["estimated_all_hit_beats_rho"]:
            return (
                None,
                None,
                len(candidates),
                selector_ops + scanned_ops,
                len(matched),
                len(matched),
                scanned_ops,
                None,
            )
        return (
            candidate,
            learned_row,
            len(candidates),
            selector_ops + support_selector_extra_ops,
            len(matched),
            len(matched) - 1,
            support_selector_extra_ops,
            signature,
        )
    for candidate in matched:
        key = base_probe.fingerprint_key(surface, candidate)
        learned_row = learned[key]
        signature = predicted_support(surface, surface_record, candidate, learned_row, locator_policy)
        if support_accepts(signature, support_policy):
            estimated_charge = estimated_all_hit_charge(
                surface_record,
                signature,
                selector_ops + support_selector_extra_ops,
            )
            signature = {**signature, **estimated_charge}
            if support_policy.endswith("_below_rho") and not estimated_charge["estimated_all_hit_beats_rho"]:
                rejected += 1
                support_selector_extra_ops += support_check_ops(signature)
                continue
            return (
                candidate,
                learned_row,
                len(candidates),
                selector_ops + support_selector_extra_ops,
                len(matched),
                rejected,
                support_selector_extra_ops,
                signature,
            )
        rejected += 1
        support_selector_extra_ops += support_check_ops(signature)
    return None, None, len(candidates), selector_ops + support_selector_extra_ops, len(matched), rejected, support_selector_extra_ops, None


def selected_root_pairs(surface_record: dict[str, Any]) -> set[tuple[int, int]]:
    return base_probe.remainder_factor_probe.selected_pair_set(surface_record, surface_record["surface"])


def candidate_recovered_selected_pairs(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> set[tuple[int, int]]:
    if candidate is None:
        return set()
    p = int(surface_record["p"])
    terms = base_probe.factor_terms(sage_surface, candidate)
    if not terms:
        return set()
    selected = {int(leaf) for leaf in surface_record["selected_leaf_indices"]}
    recovered: set[tuple[int, int]] = set()
    for coord in base_probe.leaf_coords(surface_record):
        leaf_index = int(coord["leaf_index"])
        if leaf_index not in selected:
            continue
        if base_probe.eval_factor(terms, int(coord["b"]), int(coord["c"]), p) != 0:
            continue
        leaf = surface_record["components"]["leaves"][leaf_index]
        roots = base_probe.slice_quadratic_probe.recover_roots_for_leaf(
            surface_record,
            leaf,
            int(coord["b"]),
            int(coord["c"]),
        )
        recovered.update((leaf_index, int(root)) for root in roots)
    return recovered


def selected_root_support_diagnostic(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    support_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    original_pairs = selected_root_pairs(surface_record)
    original_leaves = sorted({leaf for leaf, _root in original_pairs})
    candidate_pairs = candidate_recovered_selected_pairs(sage_surface, surface_record, candidate)
    candidate_leaves = sorted({leaf for leaf, _root in candidate_pairs})
    predicted_valid = {
        int(leaf)
        for leaf in (support_signature or {}).get("predicted_valid_root_indices") or []
    }
    missing_candidate_pairs = sorted(original_pairs - candidate_pairs)
    if candidate is None:
        failure_mode = "no_candidate"
    elif not missing_candidate_pairs and set(original_leaves).issubset(predicted_valid):
        failure_mode = "public_selector_preserves"
    elif not missing_candidate_pairs:
        failure_mode = "leaf_locator_miss"
    elif not candidate_pairs:
        failure_mode = "factor_misses_selected_leaf"
    else:
        failure_mode = "factor_wrong_selected_root"
    return {
        "failure_mode": failure_mode,
        "original_selected_root_leaf_indices": original_leaves[:32],
        "candidate_selected_valid_leaf_indices": candidate_leaves[:32],
        "predicted_valid_selected_leaf_indices": sorted(predicted_valid & set(original_leaves))[:32],
        "candidate_missing_selected_root_pair_count": len(missing_candidate_pairs),
        "candidate_missing_selected_root_pairs": [[leaf, root] for leaf, root in missing_candidate_pairs[:8]],
        "candidate_recovered_selected_root_pair_count": len(candidate_pairs),
        "predicted_selected_leaf_overlap_count": len(predicted_valid & set(original_leaves)),
    }


def support_pool_diagnostic(
    surface: dict[str, Any],
    surface_record: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
    locator_policy: str,
    support_policy: str,
    factor_index_prior: str,
) -> dict[str, Any]:
    candidates, selector_ops, matched = ranked_candidates_for_support(
        surface,
        learned,
        max_degree,
        factor_index_prior,
    )
    diagnostic: dict[str, Any] = {
        "candidate_count": len(candidates),
        "matched_candidate_count": len(matched),
        "matched_preserving_candidate_count": 0,
        "matched_false_positive_candidate_count": 0,
        "support_accepted_candidate_count": 0,
        "support_accepted_preserving_candidate_count": 0,
        "support_accepted_false_positive_candidate_count": 0,
        "below_rho_support_accepted_candidate_count": 0,
        "below_rho_support_accepted_preserving_candidate_count": 0,
        "below_rho_support_accepted_false_positive_candidate_count": 0,
        "first_preserving_candidate_rank": None,
        "first_preserving_support_accepts": False,
        "first_preserving_below_rho_accepts": False,
        "first_support_accepted_candidate_rank": None,
        "first_support_accepted_candidate_preserves": None,
        "first_below_rho_support_accepted_candidate_rank": None,
        "first_below_rho_support_accepted_candidate_preserves": None,
    }
    support_selector_extra_ops = 0
    for rank, candidate in enumerate(matched):
        key = base_probe.fingerprint_key(surface, candidate)
        learned_row = learned[key]
        signature = predicted_support(surface, surface_record, candidate, learned_row, locator_policy)
        accepts = support_accepts(signature, support_policy)
        charge = estimated_all_hit_charge(
            surface_record,
            signature,
            selector_ops + support_selector_extra_ops,
        )
        ranked_first_charge = estimated_all_hit_charge(surface_record, signature, selector_ops)
        charged_signature = {
            **signature,
            **charge,
            "ranked_first_estimated_all_hit_ops": ranked_first_charge["estimated_all_hit_ops"],
            "ranked_first_estimated_all_hit_ops_over_rho": ranked_first_charge[
                "estimated_all_hit_ops_over_rho"
            ],
            "ranked_first_estimated_all_hit_beats_rho": ranked_first_charge[
                "estimated_all_hit_beats_rho"
            ],
        }
        below = bool(charge["estimated_all_hit_beats_rho"])
        preserves = bool(candidate.get("preserves_selected_root_pairs"))
        if preserves:
            diagnostic["matched_preserving_candidate_count"] += 1
        else:
            diagnostic["matched_false_positive_candidate_count"] += 1
        if preserves and diagnostic["first_preserving_candidate_rank"] is None:
            diagnostic["first_preserving_candidate_rank"] = rank
            diagnostic["first_preserving_support_accepts"] = bool(accepts)
            diagnostic["first_preserving_below_rho_accepts"] = bool(accepts and below)
            diagnostic["first_preserving_candidate"] = base_probe.compact_candidate(surface, candidate)
            diagnostic["first_preserving_support_signature"] = charged_signature
        if accepts:
            diagnostic["support_accepted_candidate_count"] += 1
            if preserves:
                diagnostic["support_accepted_preserving_candidate_count"] += 1
            else:
                diagnostic["support_accepted_false_positive_candidate_count"] += 1
            if diagnostic["first_support_accepted_candidate_rank"] is None:
                diagnostic["first_support_accepted_candidate_rank"] = rank
                diagnostic["first_support_accepted_candidate_preserves"] = preserves
                diagnostic["first_support_accepted_candidate"] = base_probe.compact_candidate(surface, candidate)
                diagnostic["first_support_accepted_signature"] = charged_signature
            if below:
                diagnostic["below_rho_support_accepted_candidate_count"] += 1
                if preserves:
                    diagnostic["below_rho_support_accepted_preserving_candidate_count"] += 1
                else:
                    diagnostic["below_rho_support_accepted_false_positive_candidate_count"] += 1
                if diagnostic["first_below_rho_support_accepted_candidate_rank"] is None:
                    diagnostic["first_below_rho_support_accepted_candidate_rank"] = rank
                    diagnostic["first_below_rho_support_accepted_candidate_preserves"] = preserves
                    diagnostic["first_below_rho_support_accepted_candidate"] = base_probe.compact_candidate(
                        surface,
                        candidate,
                    )
                    diagnostic["first_below_rho_support_accepted_signature"] = charged_signature
        if not accepts or (support_policy.endswith("_below_rho") and not below):
            support_selector_extra_ops += support_check_ops(signature)
    return diagnostic


def audit_supported_locator(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    learned_row: dict[str, Any] | None,
    candidate_count: int,
    selector_ops: int,
    split_name: str,
    locator_policy: str,
    support_policy: str,
    matched_count: int,
    rejected_count: int,
    support_selector_extra_ops: int,
    support_signature: dict[str, Any] | None,
    pool_diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    row = locator_probe.audit_locator(
        sage_surface,
        surface_record,
        candidate,
        learned_row,
        candidate_count,
        selector_ops,
        split_name,
        locator_policy,
    )
    row["support_policy"] = support_policy
    row["support_matched_candidate_count"] = matched_count
    row["support_rejected_candidate_count"] = rejected_count
    row["support_selector_extra_ops"] = support_selector_extra_ops
    row["chosen_support_signature"] = support_signature
    row["selected_root_support_diagnostic"] = selected_root_support_diagnostic(
        sage_surface,
        surface_record,
        candidate,
        support_signature,
    )
    row["support_pool_diagnostic"] = pool_diagnostic
    return row


def rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = locator_probe.rows_summary(rows)
    usable_relation_rows = [
        row
        for row in rows
        if row.get("chosen_candidate") and int(row.get("valid_root_leaf_count") or 0) > 0
    ]
    alternate_relation_rows = [
        row
        for row in usable_relation_rows
        if not row.get("preserves_selected_root_pairs")
    ]
    usable_ratios = [
        float(row["public_locator_all_hit_ops_over_rho"])
        for row in usable_relation_rows
        if row.get("public_locator_all_hit_ops_over_rho") is not None
    ]
    summary["support_rejected_candidate_count"] = sum(
        int(row.get("support_rejected_candidate_count") or 0) for row in rows
    )
    summary["support_selector_extra_ops"] = sum(int(row.get("support_selector_extra_ops") or 0) for row in rows)
    summary["usable_public_relation_count"] = len(usable_relation_rows)
    summary["usable_public_relation_below_rho_count"] = sum(
        1 for row in usable_relation_rows if row.get("public_locator_all_hit_beats_rho")
    )
    summary["alternate_public_relation_count"] = len(alternate_relation_rows)
    summary["alternate_public_relation_below_rho_count"] = sum(
        1 for row in alternate_relation_rows if row.get("public_locator_all_hit_beats_rho")
    )
    summary["min_usable_public_relation_ops_over_rho"] = min(usable_ratios) if usable_ratios else None
    pool_rows = [row.get("support_pool_diagnostic") or {} for row in rows]
    summary["matched_preserving_candidate_count"] = sum(
        int(pool.get("matched_preserving_candidate_count") or 0) for pool in pool_rows
    )
    summary["support_accepted_preserving_candidate_count"] = sum(
        int(pool.get("support_accepted_preserving_candidate_count") or 0) for pool in pool_rows
    )
    summary["support_accepted_false_positive_candidate_count"] = sum(
        int(pool.get("support_accepted_false_positive_candidate_count") or 0) for pool in pool_rows
    )
    summary["below_rho_support_accepted_preserving_candidate_count"] = sum(
        int(pool.get("below_rho_support_accepted_preserving_candidate_count") or 0) for pool in pool_rows
    )
    summary["below_rho_support_accepted_false_positive_candidate_count"] = sum(
        int(pool.get("below_rho_support_accepted_false_positive_candidate_count") or 0) for pool in pool_rows
    )
    summary["rows_with_matched_preserving_candidate_count"] = sum(
        1 for pool in pool_rows if int(pool.get("matched_preserving_candidate_count") or 0) > 0
    )
    summary["rows_with_matched_preserving_but_no_chosen_preserving_count"] = sum(
        1
        for row in rows
        if int((row.get("support_pool_diagnostic") or {}).get("matched_preserving_candidate_count") or 0) > 0
        and not row.get("preserves_selected_root_pairs")
    )
    summary["selected_root_failure_mode_counts"] = dict(
        sorted(
            Counter(
                str((row.get("selected_root_support_diagnostic") or {}).get("failure_mode") or "unknown")
                for row in rows
            ).items()
        )
    )
    return summary


def support_summary_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
    preserving = int(summary.get("preserving_surface_count") or 0)
    false_positive = int(summary.get("false_positive_surface_count") or 0)
    below_rho = int(summary.get("all_hit_below_rho_count") or 0)
    chosen = int(summary.get("chosen_factor_count") or 0)
    return (
        preserving <= 0,
        false_positive,
        -preserving,
        -below_rho,
        float(summary.get("mean_all_hit_ops_over_rho") or 10**18),
        -chosen,
        float(summary.get("mean_predicted_leaf_count") or 10**18),
    )


def evaluate_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    ids: set[str],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    locator_policy: str,
    support_policy: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in ids or surface_id not in surface_records:
            continue
        (
            candidate,
            learned_row,
            candidate_count,
            selector_ops,
            matched_count,
            rejected_count,
            support_selector_extra_ops,
            support_signature,
        ) = choose_supported_candidate(
            surface,
            surface_records[surface_id],
            learned,
            int(args.max_factor_degree),
            locator_policy,
            support_policy,
            str(args.factor_index_prior),
        )
        pool_diagnostic = support_pool_diagnostic(
            surface,
            surface_records[surface_id],
            learned,
            int(args.max_factor_degree),
            locator_policy,
            support_policy,
            str(args.factor_index_prior),
        )
        rows.append(
            audit_supported_locator(
                surface,
                surface_records[surface_id],
                candidate,
                learned_row,
                candidate_count,
                selector_ops,
                split_name,
                locator_policy,
                support_policy,
                matched_count,
                rejected_count,
                support_selector_extra_ops,
                support_signature,
                pool_diagnostic,
            )
        )
    return rows


def split_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    learned = learn_leaf_tables_for_support(
        surfaces,
        surface_records,
        train_ids,
        int(args.max_factor_degree),
        int(args.min_support),
        bool(args.require_clean_fingerprint),
        bool(args.allow_noisy_fingerprint),
    )
    support_policies = [args.force_support_policy] if args.force_support_policy else list(SUPPORT_POLICIES)
    locator_policies = [args.force_locator_policy] if args.force_locator_policy else list(locator_probe.LOCATOR_POLICIES)
    calibration = []
    for support_policy in support_policies:
        for locator_policy in locator_policies:
            rows = evaluate_rows(
                f"{split_name}:calibration",
                surfaces,
                surface_records,
                train_ids,
                learned,
                locator_policy,
                support_policy,
                args,
            )
            calibration.append(
                {
                    "support_policy": support_policy,
                    "locator_policy": locator_policy,
                    "summary": rows_summary(rows),
                }
            )
    calibration.sort(
        key=lambda row: (
            support_summary_rank(row["summary"]),
            row["support_policy"],
            row["locator_policy"],
        )
    )
    selected = calibration[0] if calibration else {
        "support_policy": "none",
        "locator_policy": "mode1",
        "summary": {},
    }
    test_rows = evaluate_rows(
        split_name,
        surfaces,
        surface_records,
        test_ids,
        learned,
        str(selected["locator_policy"]),
        str(selected["support_policy"]),
        args,
    )
    return {
        "split": split_name,
        "train_surface_count": len(train_ids),
        "test_surface_count": len(test_ids),
        "learned_fingerprint_count": len(learned),
        "selected_locator_policy": selected["locator_policy"],
        "selected_support_policy": selected["support_policy"],
        "selected_training_summary": selected["summary"],
        "test_summary": rows_summary(test_rows),
        "test_rows": test_rows,
        "top_calibration_rows": calibration[:12],
    }


def holdout_splits(
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    surface_ids = {str(surface.get("surface_id")) for surface in surfaces}
    splits = []
    targets = sorted({str(surface.get("target")) for surface in surfaces})
    for target in targets:
        test = {str(surface.get("surface_id")) for surface in surfaces if str(surface.get("target")) == target}
        splits.append(split_rows(f"holdout_target_{target}", surfaces, surface_records, surface_ids - test, test, args))
    transfers = sorted(
        {base_probe.transfer_index(surface) for surface in surfaces if base_probe.transfer_index(surface) is not None}
    )
    for transfer in transfers:
        test = {str(surface.get("surface_id")) for surface in surfaces if base_probe.transfer_index(surface) == transfer}
        splits.append(split_rows(f"holdout_transfer_{transfer}", surfaces, surface_records, surface_ids - test, test, args))
    for idx in range(int(args.min_train_transfers), len(transfers)):
        train_transfers = set(transfers[:idx])
        holdout = transfers[idx]
        train = {
            str(surface.get("surface_id"))
            for surface in surfaces
            if base_probe.transfer_index(surface) in train_transfers
        }
        test = {
            str(surface.get("surface_id"))
            for surface in surfaces
            if base_probe.transfer_index(surface) == holdout
        }
        splits.append(split_rows(f"rolling_to_transfer_{holdout}", surfaces, surface_records, train, test, args))
    return splits


def summarize(case_results: list[dict[str, Any]], splits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for split in splits
        for row in split.get("test_rows", [])
        if isinstance(row, dict)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in splits:
        name = str(split.get("split", ""))
        if name.startswith("holdout_target_"):
            grouped["target"].extend(split.get("test_rows") or [])
        elif name.startswith("holdout_transfer_"):
            grouped["transfer"].extend(split.get("test_rows") or [])
        elif name.startswith("rolling_to_transfer_"):
            grouped["rolling"].extend(split.get("test_rows") or [])
    return {
        "case_count": len(case_results),
        "verified_case_count": sum(1 for case in case_results if case.get("public_key_verified")),
        "split_count": len(splits),
        "aggregate_summary": rows_summary(rows),
        "split_family_summaries": {key: rows_summary(value) for key, value in sorted(grouped.items())},
        "positive_split_count": sum(
            1 for split in splits if int((split.get("test_summary") or {}).get("all_hit_below_rho_count") or 0) > 0
        ),
        "all_splits_preserve": all(
            int((split.get("test_summary") or {}).get("preserving_surface_count") or 0)
            == int((split.get("test_summary") or {}).get("surface_count") or 0)
            for split in splits
        ),
        "all_splits_false_positive_free": all(
            int((split.get("test_summary") or {}).get("false_positive_surface_count") or 0) == 0
            for split in splits
        ),
        "selected_policy_counts": dict(
            sorted(
                Counter(
                    f"{split.get('selected_support_policy')}+{split.get('selected_locator_policy')}"
                    for split in splits
                ).items()
            )
        ),
        "support_policy_counts": dict(
            sorted(Counter(str(split.get("selected_support_policy")) for split in splits).items())
        ),
        "locator_policy_counts": dict(
            sorted(Counter(str(split.get("selected_locator_policy")) for split in splits).items())
        ),
        "best_rows": sorted(
            rows,
            key=lambda row: (
                float(row.get("public_locator_all_hit_ops_over_rho") or 10**18),
                str(row.get("surface_id")),
            ),
        )[:10],
        "false_positive_rows": [
            row
            for row in rows
            if row.get("chosen_candidate") and not row.get("preserves_selected_root_pairs")
        ][:12],
        "interpretation": (
            "This is an order-free public support guard for the factor-fingerprint "
            "leaf locator. Holdout selection uses only learned fingerprints, learned "
            "leaf indices, and public factor/root support at those predicted leaves."
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
    parser.add_argument("--fingerprint-mode", choices=FINGERPRINT_MODES, default="raw")
    parser.add_argument("--factor-index-prior", choices=FACTOR_INDEX_PRIORS, default="none")
    parser.add_argument("--force-support-policy", choices=SUPPORT_POLICIES)
    parser.add_argument("--force-locator-policy", choices=locator_probe.LOCATOR_POLICIES)
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    install_fingerprint_mode(str(args.fingerprint_mode))

    sage_source = base_probe.load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = base_probe.materialize_records(args)
    splits = holdout_splits(surfaces, surface_records, args)
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_support_probe_v2",
        "method": "calibrated_factor_fingerprint_leaf_index_locator_with_public_predicted_leaf_support_guard",
        "parameters": {
            "campaign_task_dir": str(base_probe.CAMPAIGN_TASK_DIR),
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "max_factor_degree": args.max_factor_degree,
            "min_support": args.min_support,
            "require_clean_fingerprint": bool(args.require_clean_fingerprint),
            "allow_noisy_fingerprint": bool(args.allow_noisy_fingerprint),
            "fingerprint_mode": args.fingerprint_mode,
            "factor_index_prior": args.factor_index_prior,
            "support_policies": list(SUPPORT_POLICIES),
            "locator_policies": list(locator_probe.LOCATOR_POLICIES),
            "force_support_policy": args.force_support_policy,
            "force_locator_policy": args.force_locator_policy,
            "forbidden_holdout_selection_fields": [
                "selected_surface_zero_leaves",
                "preserves_selected_root_pairs",
                "same_selected_root_pairs",
                "selected_valid_root_leaves",
                "missing_selected_root_pairs",
                "extra_selected_root_pairs",
            ],
        },
        "summary": summarize(case_results, splits),
        "splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
