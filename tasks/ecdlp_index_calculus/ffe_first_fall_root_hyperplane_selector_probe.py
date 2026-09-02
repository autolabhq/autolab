#!/usr/bin/env python3
"""Probe held-out root-hyperplane selectors for first-fall FFE surfaces.

The first-fall audit showed that the Sage factors in the current FFE surface
bank are all linear hyperplanes

    c + r*b + r^2

for the monic leaf quadratic ``x^2 + b*x + c``.  This follow-up treats factor
choice as a root-ordering problem: rank candidate roots using public factor
features or held-out calibration priors, evaluate factors until the first
public zero on the selected leaf point, then charge that selector work plus the
selected root scan against the recorded generic Pollard-rho step count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AUDIT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_all_public_leaf_plus_fresh_72_79.json"
)
DEFAULT_SAGE_FACTOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_sage_factor_all_public_leaf_plus_fresh_72_79.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_first_fall_root_hyperplane_selector_all_public_leaf_plus_fresh_72_79.json"
)

STATIC_POLICIES = (
    "sage_factor_order",
    "low_constant",
    "low_root_norm",
    "global_root_hash",
    "target_root_hash",
    "summax_sage_low_constant_target_hash",
    "minsum_low_constant_target_hash",
    "summax_low_root_norm_target_hash",
)
LEARNED_POLICIES = (
    "learned_target_zero_root_prior",
    "learned_global_zero_root_prior",
    "learned_target_zero_candidate_prior",
    "learned_target_zero_residue16_prior",
    "learned_target_zero_factor_index_prior",
    "learned_salt_weighted_zero_prior",
    "learned_transfer_weighted_zero_prior",
    "learned_zero_ensemble_prior",
)
HYBRID_POLICIES = (
    "residue16_then_global_hash_cap1",
    "residue16_then_global_hash_cap2",
    "residue16_then_global_hash_cap3",
    "residue16_then_global_hash_cap4",
)
POLICIES = STATIC_POLICIES + LEARNED_POLICIES + HYBRID_POLICIES


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_factor_index(candidate_name: str | None) -> int | None:
    if not candidate_name:
        return None
    try:
        return int(str(candidate_name).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def row_salt(row_key: str | None) -> int | None:
    if not row_key:
        return None
    match = re.search(r"salt([0-9]+)$", str(row_key))
    return int(match.group(1)) if match else None


def transfer_index(surface: dict[str, Any]) -> int | None:
    if surface.get("transfer_index") is not None:
        return int(surface["transfer_index"])
    challenge_seed = str(surface.get("challenge_seed") or "")
    marker = ":shared-transfer:"
    if marker not in challenge_seed:
        return None
    try:
        return int(challenge_seed.split(marker, 1)[1].split(":", 1)[0])
    except (IndexError, ValueError):
        return None


def centered_mod(value: int, p: int) -> int:
    value %= p
    return min(value, p - value)


def fingerprint_terms(factor: dict[str, Any], p: int) -> dict[tuple[int, int], int]:
    return {
        (int(b_degree), int(c_degree)): int(coeff) % p
        for b_degree, c_degree, coeff in factor.get("fingerprint") or []
    }


def root_hyperplane_info(factor: dict[str, Any], p: int) -> dict[str, Any]:
    terms = fingerprint_terms(factor, p)
    root = terms.get((1, 0))
    constant = terms.get((0, 0))
    c_coeff = terms.get((0, 1))
    is_root_hyperplane = (
        set(terms) == {(0, 0), (0, 1), (1, 0)}
        and c_coeff == 1
        and root is not None
        and constant is not None
        and (int(root) * int(root) - int(constant)) % p == 0
    )
    coeff_norms = [centered_mod(int(coeff), p) for coeff in terms.values()] if p else []
    return {
        "is_root_hyperplane": is_root_hyperplane,
        "root": int(root) if root is not None else None,
        "constant": int(constant) if constant is not None else None,
        "c_coeff": int(c_coeff) if c_coeff is not None else None,
        "coeff_norm_sum": sum(coeff_norms),
        "coeff_norm_max": max(coeff_norms) if coeff_norms else None,
        "constant_norm": centered_mod(int(constant), p)
        if constant is not None and p
        else None,
        "root_norm": centered_mod(int(root), p) if root is not None and p else None,
        "fingerprint": factor.get("fingerprint") or [],
    }


def candidate_by_index(surface: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for candidate in surface.get("sage_resultant_factor_candidates") or []:
        index = parse_factor_index(candidate.get("candidate_name"))
        if index is not None:
            out[index] = candidate
    return out


def candidate_selector_eval_ops(candidate: dict[str, Any]) -> int:
    selected_leaf_count = max(1, int(candidate.get("selected_leaf_count") or 0))
    return selected_leaf_count * int(candidate.get("factor_monomials") or 0)


def build_candidates(surface: dict[str, Any]) -> list[dict[str, Any]]:
    p = int(surface.get("p") or 0)
    candidates_by_index = candidate_by_index(surface)
    out = []
    for index, factor in enumerate((surface.get("sage_resultant_factorization") or {}).get("factors") or []):
        candidate = candidates_by_index.get(index, {})
        info = root_hyperplane_info(factor, p)
        if info["root"] is None:
            continue
        out.append(
            {
                "candidate_name": f"sage_resultant_factor_{index}",
                "factor_index": index,
                "p": p,
                **info,
                "factor_total_degree": int(candidate.get("factor_total_degree") or factor.get("total_degree") or 0),
                "factor_monomials": int(candidate.get("factor_monomials") or factor.get("monomials") or 0),
                "surface_ffe_ops": int(candidate.get("surface_ffe_ops") or 0),
                "surface_monomials": int(candidate.get("surface_monomials") or 0),
                "selected_leaf_count": int(candidate.get("selected_leaf_count") or 0),
                "selected_surface_zero_leaves": int(candidate.get("selected_surface_zero_leaves") or 0),
                "selected_valid_root_leaves": int(candidate.get("selected_valid_root_leaves") or 0),
                "selected_root_pair_count": int(candidate.get("selected_root_pair_count") or 0),
                "public_zero": int(candidate.get("selected_surface_zero_leaves") or 0) > 0,
                "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
                "same_selected_root_pairs": bool(candidate.get("same_selected_root_pairs")),
                "factor_root_scan_ops": int(candidate.get("factor_root_scan_ops") or 0),
                "generic_rho_steps": int(candidate.get("generic_rho_steps") or 0),
                "selector_eval_ops": candidate_selector_eval_ops(candidate),
            }
        )
    return out


def build_surface_rows(audit_source: dict[str, Any], sage_source: dict[str, Any]) -> list[dict[str, Any]]:
    audit_by_surface = {
        str(row.get("surface_id")): row
        for row in audit_source.get("surfaces") or []
        if isinstance(row, dict)
    }
    rows = []
    for surface in sage_source.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        surface_id = str(surface.get("surface_id") or "")
        audit_row = audit_by_surface.get(surface_id, {})
        candidates = build_candidates(surface)
        generic_rho_steps = max(
            [int(candidate.get("generic_rho_steps") or 0) for candidate in candidates] + [0]
        )
        if generic_rho_steps <= 0:
            best = surface.get("best_preserving_candidate") or {}
            generic_rho_steps = int(best.get("generic_rho_steps") or 0)
        rows.append(
            {
                "surface_id": surface_id,
                "target": str(surface.get("target") or audit_row.get("target") or ""),
                "row_key": surface.get("row_key") or audit_row.get("row_key"),
                "salt": row_salt(surface.get("row_key") or audit_row.get("row_key")),
                "challenge_seed": surface.get("challenge_seed"),
                "transfer_index": transfer_index(surface) if transfer_index(surface) is not None else audit_row.get("transfer_index"),
                "p": int(surface.get("p") or audit_row.get("p") or 0),
                "selected_leaf_indices": surface.get("selected_leaf_indices")
                or audit_row.get("selected_leaf_indices")
                or [],
                "original_selected_root_pair_count": int(
                    surface.get("original_selected_root_pair_count") or 0
                ),
                "generic_rho_steps": generic_rho_steps,
                "candidate_count": len(candidates),
                "public_zero_root_count": sum(1 for candidate in candidates if candidate["public_zero"]),
                "preserving_root_count": sum(
                    1 for candidate in candidates if candidate["preserves_selected_root_pairs"]
                ),
                "candidates": candidates,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or -1),
            str(row.get("row_key")),
            str(row.get("surface_id")),
        ),
    )


def fingerprint_digest(row: dict[str, Any], candidate: dict[str, Any], salt: str) -> str:
    payload = {
        "fingerprint": candidate.get("fingerprint") or [],
        "p": int(row.get("p") or candidate.get("p") or 0),
        "salt": salt,
        "target": row.get("target") if salt == "target" else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def base_sort_key(policy: str, row: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, ...]:
    p = int(row.get("p") or candidate.get("p") or 0)
    if policy == "sage_factor_order":
        return (int(candidate["factor_index"]),)
    if policy == "low_constant":
        return (
            int(candidate.get("constant_norm") if candidate.get("constant_norm") is not None else p),
            int(candidate.get("coeff_norm_max") if candidate.get("coeff_norm_max") is not None else 10**18),
            int(candidate.get("factor_total_degree") or 10**9),
            int(candidate.get("factor_monomials") or 10**9),
            int(candidate["factor_index"]),
        )
    if policy == "low_root_norm":
        return (
            int(candidate.get("root_norm") if candidate.get("root_norm") is not None else p),
            int(candidate.get("constant_norm") if candidate.get("constant_norm") is not None else p),
            int(candidate["factor_index"]),
        )
    if policy == "global_root_hash":
        return (fingerprint_digest(row, candidate, "global"),)
    if policy == "target_root_hash":
        return (fingerprint_digest(row, candidate, "target"),)
    raise ValueError(f"unsupported base policy: {policy}")


def rank_map(policy: str, row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[int, int]:
    ordered = sorted(candidates, key=lambda candidate: base_sort_key(policy, row, candidate))
    return {id(candidate): index for index, candidate in enumerate(ordered)}


def rank_ensemble_key(policy: str, row: dict[str, Any], candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[Any, ...]:
    if policy == "summax_sage_low_constant_target_hash":
        bases = ("sage_factor_order", "low_constant", "target_root_hash")
        mode = "summax"
    elif policy == "minsum_low_constant_target_hash":
        bases = ("low_constant", "target_root_hash")
        mode = "minsum"
    elif policy == "summax_low_root_norm_target_hash":
        bases = ("low_root_norm", "target_root_hash")
        mode = "summax"
    else:
        raise ValueError(f"unsupported rank ensemble policy: {policy}")
    ranks = [rank_map(base, row, candidates)[id(candidate)] for base in bases]
    if mode == "summax":
        head = (sum(ranks), max(ranks), min(ranks))
    elif mode == "minsum":
        head = (min(ranks), sum(ranks), max(ranks))
    else:
        raise ValueError(f"unsupported rank ensemble mode: {mode}")
    return head + (
        int(candidate.get("factor_total_degree") or 10**9),
        int(candidate.get("factor_monomials") or 10**9),
        int(candidate["factor_index"]),
    )


def training_rows(surface_rows: list[dict[str, Any]], row: dict[str, Any], leave_out_mode: str) -> list[dict[str, Any]]:
    out = []
    for other in surface_rows:
        if other.get("surface_id") == row.get("surface_id"):
            continue
        if "transfer" in leave_out_mode and other.get("transfer_index") == row.get("transfer_index"):
            continue
        if "row" in leave_out_mode and other.get("row_key") == row.get("row_key"):
            continue
        if "target" in leave_out_mode and other.get("target") == row.get("target"):
            continue
        out.append(other)
    return out


def train_stats(train: list[dict[str, Any]]) -> dict[str, Any]:
    target_zero_roots: dict[str, Counter[int]] = defaultdict(Counter)
    global_zero_roots: Counter[int] = Counter()
    target_candidate_roots: dict[str, Counter[int]] = defaultdict(Counter)
    global_candidate_roots: Counter[int] = Counter()
    target_zero_residue16: dict[str, Counter[int]] = defaultdict(Counter)
    global_zero_residue16: Counter[int] = Counter()
    target_zero_factor_index: dict[str, Counter[int]] = defaultdict(Counter)
    global_zero_factor_index: Counter[int] = Counter()
    for row in train:
        target = str(row.get("target"))
        p = int(row.get("p") or 0)
        for candidate in row.get("candidates") or []:
            root = int(candidate["root"])
            target_candidate_roots[target][root] += 1
            global_candidate_roots[root] += 1
            if not candidate.get("public_zero"):
                continue
            target_zero_roots[target][root] += 1
            global_zero_roots[root] += 1
            residue = root % 16 if p else root % 16
            target_zero_residue16[target][residue] += 1
            global_zero_residue16[residue] += 1
            target_zero_factor_index[target][int(candidate["factor_index"])] += 1
            global_zero_factor_index[int(candidate["factor_index"])] += 1
    return {
        "target_zero_roots": target_zero_roots,
        "global_zero_roots": global_zero_roots,
        "target_candidate_roots": target_candidate_roots,
        "global_candidate_roots": global_candidate_roots,
        "target_zero_residue16": target_zero_residue16,
        "global_zero_residue16": global_zero_residue16,
        "target_zero_factor_index": target_zero_factor_index,
        "global_zero_factor_index": global_zero_factor_index,
    }


def weighted_zero_score(
    row: dict[str, Any],
    train: list[dict[str, Any]],
    root: int,
    mode: str,
) -> float:
    target = str(row.get("target"))
    row_value = row.get("salt") if mode == "salt" else row.get("transfer_index")
    if row_value is None:
        return 0.0
    score = 0.0
    for other in train:
        if str(other.get("target")) != target:
            continue
        other_value = other.get("salt") if mode == "salt" else other.get("transfer_index")
        if other_value is None:
            continue
        distance = abs(int(row_value) - int(other_value))
        weight = 1.0 / (1.0 + float(distance))
        if any(int(candidate["root"]) == root and candidate.get("public_zero") for candidate in other.get("candidates") or []):
            score += weight
    return score


def learned_sort_key(
    policy: str,
    row: dict[str, Any],
    candidate: dict[str, Any],
    train: list[dict[str, Any]],
    stats: dict[str, Any],
) -> tuple[Any, ...]:
    target = str(row.get("target"))
    root = int(candidate["root"])
    residue16 = root % 16
    factor_index = int(candidate["factor_index"])
    target_zero = int(stats["target_zero_roots"][target][root])
    global_zero = int(stats["global_zero_roots"][root])
    target_candidate = int(stats["target_candidate_roots"][target][root])
    global_candidate = int(stats["global_candidate_roots"][root])
    target_residue = int(stats["target_zero_residue16"][target][residue16])
    global_residue = int(stats["global_zero_residue16"][residue16])
    target_factor_index = int(stats["target_zero_factor_index"][target][factor_index])
    global_factor_index = int(stats["global_zero_factor_index"][factor_index])
    low_constant = base_sort_key("low_constant", row, candidate)
    root_hash = base_sort_key("target_root_hash", row, candidate)
    if policy == "learned_target_zero_root_prior":
        return (-target_zero, -global_zero, -target_candidate, -global_candidate) + low_constant + root_hash
    if policy == "learned_global_zero_root_prior":
        return (-global_zero, -target_zero, -global_candidate, -target_candidate) + low_constant + root_hash
    if policy == "learned_target_zero_candidate_prior":
        return (-target_zero, -target_candidate, -global_zero, -global_candidate) + low_constant + root_hash
    if policy == "learned_target_zero_residue16_prior":
        return (-target_residue, -global_residue, -target_zero, -global_zero) + low_constant + root_hash
    if policy == "learned_target_zero_factor_index_prior":
        return (-target_factor_index, -global_factor_index, -target_zero, -global_zero) + low_constant + root_hash
    if policy == "learned_salt_weighted_zero_prior":
        weighted = weighted_zero_score(row, train, root, "salt")
        return (-weighted, -target_zero, -global_zero) + low_constant + root_hash
    if policy == "learned_transfer_weighted_zero_prior":
        weighted = weighted_zero_score(row, train, root, "transfer")
        return (-weighted, -target_zero, -global_zero) + low_constant + root_hash
    if policy == "learned_zero_ensemble_prior":
        weighted_salt = weighted_zero_score(row, train, root, "salt")
        weighted_transfer = weighted_zero_score(row, train, root, "transfer")
        return (
            -(4 * target_zero + 2 * global_zero + target_candidate),
            -(target_residue + target_factor_index),
            -weighted_salt,
            -weighted_transfer,
            -global_candidate,
        ) + low_constant + root_hash
    raise ValueError(f"unsupported learned policy: {policy}")


def hybrid_order_candidates(
    policy: str,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    train: list[dict[str, Any]],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    match = re.fullmatch(r"residue16_then_global_hash_cap([1-9][0-9]*)", policy)
    if not match:
        raise ValueError(f"unsupported hybrid policy: {policy}")
    cap = int(match.group(1))
    residue_ordered = sorted(
        candidates,
        key=lambda candidate: learned_sort_key(
            "learned_target_zero_residue16_prior",
            row,
            candidate,
            train,
            stats,
        ),
    )
    head = residue_ordered[:cap]
    head_ids = {id(candidate) for candidate in head}
    tail = sorted(
        [candidate for candidate in candidates if id(candidate) not in head_ids],
        key=lambda candidate: base_sort_key("global_root_hash", row, candidate),
    )
    return head + tail


def order_candidates(
    policy: str,
    row: dict[str, Any],
    surface_rows: list[dict[str, Any]],
    leave_out_mode: str,
) -> tuple[list[dict[str, Any]], int]:
    candidates = list(row.get("candidates") or [])
    if policy in STATIC_POLICIES:
        if policy in {"sage_factor_order", "low_constant", "low_root_norm", "global_root_hash", "target_root_hash"}:
            return sorted(candidates, key=lambda candidate: base_sort_key(policy, row, candidate)), 0
        return sorted(candidates, key=lambda candidate: rank_ensemble_key(policy, row, candidate, candidates)), 0
    train = training_rows(surface_rows, row, leave_out_mode)
    stats = train_stats(train)
    if policy in HYBRID_POLICIES:
        return hybrid_order_candidates(policy, row, candidates, train, stats), len(train)
    return (
        sorted(candidates, key=lambda candidate: learned_sort_key(policy, row, candidate, train, stats)),
        len(train),
    )


def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_name": candidate.get("candidate_name"),
        "factor_index": int(candidate.get("factor_index") or 0),
        "root": int(candidate.get("root") or 0),
        "constant": int(candidate.get("constant") or 0),
        "constant_norm": candidate.get("constant_norm"),
        "root_norm": candidate.get("root_norm"),
        "selected_surface_zero_leaves": candidate.get("selected_surface_zero_leaves"),
        "selected_valid_root_leaves": candidate.get("selected_valid_root_leaves"),
        "selected_root_pair_count": candidate.get("selected_root_pair_count"),
        "public_zero": bool(candidate.get("public_zero")),
        "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
        "factor_root_scan_ops": candidate.get("factor_root_scan_ops"),
        "direct_root_recovery_ops": direct_root_recovery_ops(candidate),
    }


def direct_root_recovery_ops(candidate: dict[str, Any] | None) -> int | None:
    if candidate is None:
        return None
    # Mirrors the earlier quadratic-root accounting shape but uses the
    # hyperplane root r directly after the public zero test.
    return (
        int(candidate.get("surface_ffe_ops") or 0)
        + int(candidate.get("factor_monomials") or 0)
        + max(1, int(candidate.get("selected_leaf_count") or 0))
        + 2 * int(candidate.get("selected_surface_zero_leaves") or 0)
        + int(candidate.get("selected_valid_root_leaves") or 0)
        + 2 * int(candidate.get("selected_root_pair_count") or 0)
    )


def audit_policy_row(
    policy: str,
    row: dict[str, Any],
    surface_rows: list[dict[str, Any]],
    leave_out_mode: str,
) -> dict[str, Any]:
    ordered, train_row_count = order_candidates(policy, row, surface_rows, leave_out_mode)
    selector_ops = 0
    evaluated = 0
    chosen: dict[str, Any] | None = None
    for candidate in ordered:
        evaluated += 1
        selector_ops += int(candidate.get("selector_eval_ops") or 0)
        if candidate.get("public_zero"):
            chosen = candidate
            break
    if chosen is None:
        selector_ops = sum(int(candidate.get("selector_eval_ops") or 0) for candidate in ordered)
    root_scan_ops = int(chosen.get("factor_root_scan_ops") or 0) if chosen else None
    total_ops = selector_ops + root_scan_ops if root_scan_ops is not None else None
    direct_ops = direct_root_recovery_ops(chosen)
    direct_total_ops = selector_ops + direct_ops if direct_ops is not None else None
    rho = int(row.get("generic_rho_steps") or 0)
    ratio = round(total_ops / rho, 8) if total_ops is not None and rho else None
    direct_ratio = round(direct_total_ops / rho, 8) if direct_total_ops is not None and rho else None
    chosen_preserves = bool(chosen and chosen.get("preserves_selected_root_pairs"))
    return {
        "policy": policy,
        "policy_type": "static_public_order" if policy in STATIC_POLICIES else "heldout_zero_prior",
        "surface_id": row.get("surface_id"),
        "target": row.get("target"),
        "row_key": row.get("row_key"),
        "salt": row.get("salt"),
        "transfer_index": row.get("transfer_index"),
        "p": row.get("p"),
        "candidate_count": int(row.get("candidate_count") or 0),
        "public_zero_root_count": int(row.get("public_zero_root_count") or 0),
        "preserving_root_count": int(row.get("preserving_root_count") or 0),
        "original_selected_root_pair_count": int(row.get("original_selected_root_pair_count") or 0),
        "train_row_count": train_row_count,
        "evaluated_root_count": evaluated,
        "selector_eval_ops": selector_ops,
        "root_scan_ops": root_scan_ops,
        "total_ops": total_ops,
        "generic_rho_steps": rho,
        "total_ops_over_rho": ratio,
        "direct_root_recovery_ops": direct_ops,
        "direct_total_ops": direct_total_ops,
        "direct_total_ops_over_rho": direct_ratio,
        "direct_ops_saved_vs_scan": (total_ops - direct_total_ops)
        if total_ops is not None and direct_total_ops is not None
        else None,
        "public_zero_recovered": chosen is not None,
        "chosen_preserves_selected_root_pairs": chosen_preserves,
        "chosen_false_positive": bool(chosen and not chosen_preserves),
        "below_rho": bool(chosen_preserves and total_ops is not None and rho and total_ops < rho),
        "direct_below_rho": bool(
            chosen_preserves and direct_total_ops is not None and rho and direct_total_ops < rho
        ),
        "chosen_candidate": compact_candidate(chosen),
        "top_root_sample": [
            {
                "factor_index": int(candidate.get("factor_index") or 0),
                "root": int(candidate.get("root") or 0),
                "public_zero": bool(candidate.get("public_zero")),
                "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
            }
            for candidate in ordered[:8]
        ],
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(row["total_ops_over_rho"]) for row in rows if row.get("total_ops_over_rho") is not None]
    direct_ratios = [
        float(row["direct_total_ops_over_rho"])
        for row in rows
        if row.get("direct_total_ops_over_rho") is not None
    ]
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = str(row.get("target"))
        by_target[target]["surface_count"] += 1
        if row.get("public_zero_recovered"):
            by_target[target]["public_zero_recovered_count"] += 1
        if row.get("chosen_preserves_selected_root_pairs"):
            by_target[target]["chosen_preserving_count"] += 1
        if row.get("chosen_false_positive"):
            by_target[target]["chosen_false_positive_count"] += 1
        if row.get("below_rho"):
            by_target[target]["below_rho_count"] += 1
        if row.get("direct_below_rho"):
            by_target[target]["direct_below_rho_count"] += 1
    return {
        "surface_count": len(rows),
        "selected_pair_surface_count": sum(
            int(row.get("original_selected_root_pair_count") or 0) > 0 for row in rows
        ),
        "public_zero_capable_surface_count": sum(
            int(row.get("public_zero_root_count") or 0) > 0 for row in rows
        ),
        "vacuous_preserving_surface_count": sum(
            int(row.get("original_selected_root_pair_count") or 0) == 0
            and int(row.get("preserving_root_count") or 0) > 0
            for row in rows
        ),
        "public_zero_recovered_count": sum(bool(row.get("public_zero_recovered")) for row in rows),
        "chosen_preserving_count": sum(bool(row.get("chosen_preserves_selected_root_pairs")) for row in rows),
        "chosen_false_positive_count": sum(bool(row.get("chosen_false_positive")) for row in rows),
        "below_rho_count": sum(bool(row.get("below_rho")) for row in rows),
        "direct_below_rho_count": sum(bool(row.get("direct_below_rho")) for row in rows),
        "all_surfaces_preserving": bool(rows)
        and all(bool(row.get("chosen_preserves_selected_root_pairs")) for row in rows),
        "all_surfaces_false_positive_free": bool(rows)
        and not any(bool(row.get("chosen_false_positive")) for row in rows),
        "all_surfaces_below_rho": bool(rows)
        and all(bool(row.get("below_rho")) for row in rows),
        "all_surfaces_direct_below_rho": bool(rows)
        and all(bool(row.get("direct_below_rho")) for row in rows),
        "all_surfaces_below_rho_either_route": bool(rows)
        and all(
            bool(row.get("below_rho")) or bool(row.get("direct_below_rho"))
            for row in rows
        ),
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": mean_or_none(ratios),
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
        "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_ops_saved_vs_scan": mean_or_none(
            [float(row.get("direct_ops_saved_vs_scan") or 0) for row in rows]
        ),
        "mean_evaluated_root_count": mean_or_none([float(row.get("evaluated_root_count") or 0) for row in rows]),
        "mean_selector_eval_ops": mean_or_none([float(row.get("selector_eval_ops") or 0) for row in rows]),
        "mean_train_row_count": mean_or_none([float(row.get("train_row_count") or 0) for row in rows]),
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
    }


def best_policy_name(policy_summaries: dict[str, dict[str, Any]]) -> str | None:
    if not policy_summaries:
        return None
    return min(
        policy_summaries,
        key=lambda name: (
            -int(policy_summaries[name].get("below_rho_count") or 0),
            -int(policy_summaries[name].get("chosen_preserving_count") or 0),
            int(policy_summaries[name].get("chosen_false_positive_count") or 0),
            float(policy_summaries[name].get("max_total_ops_over_rho") or 10**18),
            float(policy_summaries[name].get("mean_total_ops_over_rho") or 10**18),
            float(policy_summaries[name].get("mean_evaluated_root_count") or 10**18),
            name,
        ),
    )


def diagnostic_gate_predicates() -> dict[str, dict[str, Any]]:
    return {
        "nonvacuous_selected_root_pair": {
            "description": "Surfaces with at least one original selected root pair.",
            "ex_post_diagnostic": False,
            "predicate": lambda row: int(row.get("original_selected_root_pair_count") or 0) > 0,
        },
        "public_zero_capable": {
            "description": "Surfaces with at least one root hyperplane that is public-zero capable.",
            "ex_post_diagnostic": True,
            "predicate": lambda row: int(row.get("public_zero_root_count") or 0) > 0,
        },
        "nonvacuous_public_zero_capable": {
            "description": (
                "Surfaces with both an original selected root pair and at least "
                "one public-zero-capable root hyperplane."
            ),
            "ex_post_diagnostic": True,
            "predicate": lambda row: (
                int(row.get("original_selected_root_pair_count") or 0) > 0
                and int(row.get("public_zero_root_count") or 0) > 0
            ),
        },
    }


def summarize_diagnostic_gates(
    policy_rows: dict[str, list[dict[str, Any]]],
    full_best_policy: str | None,
) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    for gate_name, gate in diagnostic_gate_predicates().items():
        predicate = gate["predicate"]
        gated_policy_rows = {
            policy: [row for row in rows if predicate(row)]
            for policy, rows in policy_rows.items()
        }
        policy_summaries = {
            policy: summarize_rows(rows) for policy, rows in sorted(gated_policy_rows.items())
        }
        nonempty_policy_summaries = {
            policy: summary
            for policy, summary in policy_summaries.items()
            if int(summary.get("surface_count") or 0) > 0
        }
        best_name = best_policy_name(nonempty_policy_summaries)
        static_summaries = {
            policy: summary
            for policy, summary in nonempty_policy_summaries.items()
            if policy in STATIC_POLICIES
        }
        learned_summaries = {
            policy: summary
            for policy, summary in nonempty_policy_summaries.items()
            if policy in LEARNED_POLICIES
        }
        gates[gate_name] = {
            "description": gate["description"],
            "ex_post_diagnostic": bool(gate["ex_post_diagnostic"]),
            "policy_selection_note": (
                "best_policy_under_gate is selected after applying this reporting "
                "gate and is diagnostic unless the gate is preregistered for a "
                "fresh bank. full_best_policy_summary keeps the all-surface "
                "policy fixed before gate reporting."
            ),
            "surface_count": next(
                (
                    int(summary.get("surface_count") or 0)
                    for summary in nonempty_policy_summaries.values()
                ),
                0,
            ),
            "full_best_policy": full_best_policy,
            "full_best_policy_summary": (
                policy_summaries.get(full_best_policy or "", {})
                if full_best_policy
                else {}
            ),
            "best_policy_under_gate": best_name,
            "best_policy_under_gate_summary": nonempty_policy_summaries.get(best_name or "", {}),
            "best_static_policy_under_gate": best_policy_name(static_summaries),
            "best_static_policy_under_gate_summary": static_summaries.get(
                best_policy_name(static_summaries) or "",
                {},
            ),
            "best_heldout_policy_under_gate": best_policy_name(learned_summaries),
            "best_heldout_policy_under_gate_summary": learned_summaries.get(
                best_policy_name(learned_summaries) or "",
                {},
            ),
        }
    return gates


def summarize_per_transfer_oracle_upper_bound(policy_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    best_by_transfer = []
    all_rows = next(iter(policy_rows.values()), [])
    transfers = sorted({row.get("transfer_index") for row in all_rows if row.get("transfer_index") is not None})
    for transfer in transfers:
        candidate_summaries = {
            policy: summarize_rows([row for row in rows if row.get("transfer_index") == transfer])
            for policy, rows in policy_rows.items()
        }
        candidate_summaries = {
            policy: summary
            for policy, summary in candidate_summaries.items()
            if int(summary.get("surface_count") or 0) > 0
        }
        selected = best_policy_name(candidate_summaries)
        best_by_transfer.append(
            {
                "transfer_index": transfer,
                "selected_policy": selected,
                "test_summary": candidate_summaries.get(selected or "", {}),
            }
        )
    selected_rows = []
    for split in best_by_transfer:
        selected = split.get("selected_policy")
        transfer = split.get("transfer_index")
        selected_rows.extend(
            row
            for row in policy_rows.get(str(selected), [])
            if row.get("transfer_index") == transfer
        )
    summary = summarize_rows(selected_rows)
    summary["split_count"] = len(best_by_transfer)
    summary["selected_policy_counts"] = dict(Counter(str(split.get("selected_policy")) for split in best_by_transfer))
    return {"summary": summary, "splits": best_by_transfer}


def run_probe(
    audit_source: dict[str, Any],
    sage_source: dict[str, Any],
    leave_out_mode: str,
) -> dict[str, Any]:
    surface_rows = build_surface_rows(audit_source, sage_source)
    policy_rows = {
        policy: [
            audit_policy_row(policy, row, surface_rows, leave_out_mode)
            for row in surface_rows
        ]
        for policy in POLICIES
    }
    policy_summaries = {
        policy: summarize_rows(rows) for policy, rows in sorted(policy_rows.items())
    }
    best_name = best_policy_name(policy_summaries)
    best_rows = policy_rows.get(best_name or "", [])
    static_summaries = {
        policy: summary
        for policy, summary in policy_summaries.items()
        if policy in STATIC_POLICIES
    }
    learned_summaries = {
        policy: summary
        for policy, summary in policy_summaries.items()
        if policy in LEARNED_POLICIES
    }
    return {
        "schema": "ecdlp_ffe_first_fall_root_hyperplane_selector_probe_v1",
        "method": "heldout_root_hyperplane_order_with_public_zero_early_stop",
        "parameters": {
            "leave_out_mode": leave_out_mode,
            "policies": list(POLICIES),
            "selection_features": [
                "root hyperplane fingerprint c + r*b + r^2",
                "factorization order",
                "centered root and constant norms",
                "deterministic target/global root hashes",
                "public zero tests on selected leaf coefficients",
                "held-out calibration counts of public-zero roots",
            ],
            "forbidden_test_selection_fields": [
                "preserves_selected_root_pairs",
                "same_selected_root_pairs",
                "selected_valid_root_leaves",
                "missing_selected_root_pairs",
                "extra_selected_root_pairs",
                "generic_rho_steps",
                "total_ops_over_rho",
                "below_rho",
            ],
            "cost_model": (
                "sum(factor_monomials * selected_leaf_count) until first public-zero "
                "root hyperplane, plus selected candidate factor_root_scan_ops, "
                "compared with generic_rho_steps; direct-root companion cost "
                "replaces factor_root_scan_ops with surface_ffe_ops plus direct "
                "validation of the hyperplane root r"
            ),
        },
        "summary": {
            "surface_count": len(surface_rows),
            "policy_count": len(policy_rows),
            "static_policy_count": len(STATIC_POLICIES),
            "heldout_policy_count": len(LEARNED_POLICIES),
            "best_policy": best_name,
            "best_policy_summary": policy_summaries.get(best_name or "", {}),
            "best_static_policy": best_policy_name(static_summaries),
            "best_static_policy_summary": static_summaries.get(best_policy_name(static_summaries) or "", {}),
            "best_heldout_policy": best_policy_name(learned_summaries),
            "best_heldout_policy_summary": learned_summaries.get(best_policy_name(learned_summaries) or "", {}),
            "all_best_policy_surfaces_below_rho": bool(
                policy_summaries.get(best_name or "", {}).get("below_rho_count") == len(surface_rows)
            ),
            "all_best_policy_surfaces_preserve": bool(
                policy_summaries.get(best_name or "", {}).get("chosen_preserving_count") == len(surface_rows)
            ),
            "all_best_policy_false_positive_free": bool(
                policy_summaries.get(best_name or "", {}).get("chosen_false_positive_count") == 0
            ),
            "best_policy_worst_rows": sorted(
                best_rows,
                key=lambda row: (
                    -float(row.get("total_ops_over_rho") or -1),
                    str(row.get("surface_id")),
                ),
            )[:8],
            "diagnostic_gates": summarize_diagnostic_gates(policy_rows, best_name),
            "per_transfer_policy_oracle_upper_bound": summarize_per_transfer_oracle_upper_bound(policy_rows),
            "interpretation": (
                "The factored FFE bank can be viewed as a public zero early-stop "
                "problem over root hyperplanes. Static public rank ensembles remain "
                "strongest on this measured bank; held-out zero-root priors have "
                "transfer signal but are not yet a complete replacement for the "
                "public fingerprint order. This still needs fresh hit-stream "
                "generation before it becomes an ECDLP algorithm claim."
            ),
        },
        "policy_summaries": policy_summaries,
        "policy_rows": policy_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-source", type=Path, default=DEFAULT_AUDIT_SOURCE)
    parser.add_argument("--sage-factor-source", type=Path, default=DEFAULT_SAGE_FACTOR_SOURCE)
    parser.add_argument("--leave-out-mode", default="transfer_row")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output = run_probe(
        load_json(args.audit_source),
        load_json(args.sage_factor_source),
        str(args.leave_out_mode),
    )
    output["parameters"]["audit_source"] = str(args.audit_source)
    output["parameters"]["sage_factor_source"] = str(args.sage_factor_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
