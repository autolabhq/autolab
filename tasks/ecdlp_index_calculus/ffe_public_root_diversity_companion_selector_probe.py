#!/usr/bin/env python3
"""Select public root-diverse companion assemblies for relation replay.

The retained-root selector proved that the FFE pre-factor root policy can find
cheap single-hit root hyperplanes, but retaining only those anchor surfaces can
drop the companion rows that carry independent relation forms.  This probe uses
only public selector fields to choose source cases around root-policy anchors,
then retains every row/leaf surface in each selected case for replay.

Verifier outcomes are copied as labels only.  They are not used for filtering,
ranking, grouping, or tie-breaking.
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
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total3_total4_public_stress_selector_192_199.json"
DEFAULT_POLICY_SOURCE = (
    DEFAULT_STATE_DIR
    / "ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_stress_selector_192_199.json"
)
DEFAULT_GATE_SOURCE = DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_public_stress_selector_192_199.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_root_diversity_companion_selector_total3_total4_public_stress_192_199.json"
DEFAULT_SEED = "ecdlp-frontier-signed-dual-sieve-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def row_schedule_key(row_key: str) -> str:
    text = str(row_key)
    if ":salt" in text:
        return text.rsplit(":salt", 1)[0]
    return text


def surface_id_parts(surface_id: str) -> dict[str, Any]:
    parts = str(surface_id).split("|")
    target = parts[0] if parts else ""
    row_key = parts[1] if len(parts) > 1 else ""
    challenge_seed = parts[2] if len(parts) > 2 else ""
    transfer_index = None
    marker = ":shared-transfer:"
    if marker in challenge_seed:
        try:
            transfer_index = int(challenge_seed.split(marker, 1)[1].split(":", 1)[0])
        except (IndexError, ValueError):
            transfer_index = None
    return {
        "surface_id": surface_id,
        "target": target,
        "transfer_index": transfer_index,
        "row_key": row_key,
        "row_schedule_key": row_schedule_key(row_key),
        "salt": row_salt(row_key),
        "challenge_seed": challenge_seed,
    }


def signature_surface_id(case: dict[str, Any], item: dict[str, Any], seed: str) -> str:
    if item.get("surface_id"):
        return str(item["surface_id"])
    target = str(case.get("target"))
    transfer_index = int(case.get("transfer_index") or 0)
    row_key = str(item.get("row_key"))
    challenge_seed = f"{seed}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    row_leaf_signature = tuple(
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
        str(case.get("leaf_selector") or case.get("selector")),
        row_leaf_signature,
    )


def case_key_string(case: dict[str, Any]) -> str:
    return "|".join(str(part) for part in case_key(case))


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_list(values: Any) -> list[int]:
    out = []
    for value in values or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def public_root_policy_rows(
    policy_source: dict[str, Any],
    max_total_ops_over_rho: float,
    require_direct_below_rho: bool,
    allow_root_ambiguity: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in policy_source.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if not row.get("public_zero_recovered"):
            continue
        roots = int_list(row.get("pre_factor_selected_hit_roots"))
        if not roots:
            continue
        if not allow_root_ambiguity and len(set(roots)) != 1:
            continue
        ratio_field = "direct_total_ops_over_rho" if require_direct_below_rho else "total_ops_over_rho"
        ratio = float_or_none(row.get(ratio_field))
        if ratio is None or ratio >= max_total_ops_over_rho:
            continue
        surface_id = str(row.get("surface_id") or "")
        if not surface_id:
            continue
        rows[surface_id] = row
    return rows


def policy_root(row: dict[str, Any]) -> int | None:
    chosen = row.get("chosen_candidate") or {}
    try:
        return int(chosen.get("root"))
    except (TypeError, ValueError):
        roots = int_list(row.get("pre_factor_selected_hit_roots"))
        return roots[0] if roots else None


def policy_ratio(row: dict[str, Any], require_direct_below_rho: bool) -> float:
    field = "direct_total_ops_over_rho" if require_direct_below_rho else "total_ops_over_rho"
    ratio = float_or_none(row.get(field))
    return ratio if ratio is not None else float("inf")


def compact_row_leaf_keys(case: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    out = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        leaves = [int(leaf) for leaf in item.get("leaf_indices") or []]
        if not row_key or not leaves:
            continue
        surface_id = signature_surface_id(case, item, seed)
        out.append(
            {
                "row_key": row_key,
                "row_schedule_key": row_schedule_key(row_key),
                "leaf_indices": leaves,
                "salt": item.get("salt") if item.get("salt") is not None else row_salt(row_key),
                "surface_id": surface_id,
            }
        )
    return out


def leaf_signatures(row_leaf_keys: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    signatures = set()
    for item in row_leaf_keys:
        signatures.add(
            (
                str(item.get("row_key")),
                tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
            )
        )
    return signatures


def compact_case(
    case: dict[str, Any],
    public_rows_by_surface: dict[str, dict[str, Any]],
    seed: str,
) -> dict[str, Any] | None:
    row_leaf_keys = compact_row_leaf_keys(case, seed)
    if not row_leaf_keys:
        return None
    surface_ids = sorted({str(item["surface_id"]) for item in row_leaf_keys})
    anchor_surface_ids = sorted(surface_id for surface_id in surface_ids if surface_id in public_rows_by_surface)
    if not anchor_surface_ids:
        return None
    anchor_roots = sorted(
        {
            root
            for surface_id in anchor_surface_ids
            for root in [policy_root(public_rows_by_surface[surface_id])]
            if root is not None
        }
    )
    ops_over_rho = float_or_none(case.get("ops_over_rho"))
    selector_key = (
        int(case.get("top_k") or 0),
        str(case.get("leaf_selector") or case.get("selector")),
        str(case.get("policy")),
    )
    return {
        "case_key": case_key_string(case),
        "target": str(case.get("target")),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector") or case.get("selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "source_ops_over_rho_float": ops_over_rho,
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "anchor_surface_ids": anchor_surface_ids,
        "anchor_roots": anchor_roots,
        "companion_surface_ids": sorted(set(surface_ids) - set(anchor_surface_ids)),
        "surface_ids": surface_ids,
        "row_leaf_keys": row_leaf_keys,
        "row_keys": sorted({str(item.get("row_key")) for item in row_leaf_keys}),
        "row_schedule_keys": sorted({str(item.get("row_schedule_key")) for item in row_leaf_keys}),
        "row_salts": sorted(
            {
                int(item["salt"])
                for item in row_leaf_keys
                if item.get("salt") is not None
            }
        ),
        "leaf_signatures": sorted([list(signature) for signature in leaf_signatures(row_leaf_keys)], key=str),
        "selector_key": selector_key,
        "signature_public_key_verified": bool(case.get("public_key_verified")),
        "signature_relation_count": int(case.get("relation_count") or 0),
        "signature_rank": int(case.get("rank") or 0),
    }


def collect_candidate_cases(
    signature: dict[str, Any],
    public_rows_by_surface: dict[str, dict[str, Any]],
    seed: str,
) -> list[dict[str, Any]]:
    cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in signature.get("positive_cases") or []:
        if not isinstance(case, dict):
            continue
        compact = compact_case(case, public_rows_by_surface, seed)
        if compact is None:
            continue
        key = case_key(case)
        old = cases_by_key.get(key)
        if old is None or public_case_sort_key(compact) < public_case_sort_key(old):
            cases_by_key[key] = compact
    return sorted(cases_by_key.values(), key=public_case_sort_key)


def public_case_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(case.get("source_ops_over_rho_float") if case.get("source_ops_over_rho_float") is not None else 10**9),
        -len(case.get("anchor_surface_ids") or []),
        int(case.get("selected_row_count") or 0),
        int(case.get("selected_leaf_count") or 0),
        int(case.get("top_k") or 0),
        str(case.get("leaf_selector")),
        str(case.get("policy")),
        str(case.get("case_key")),
    )


def row_key_tuple(case: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item.get("row_key")) for item in case.get("row_leaf_keys") or [])


def total3_base_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        row_key_tuple(case),
    )


def lift_partner_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    selector = str(case.get("leaf_selector") or "")
    return (
        0 if "total4" in selector else 1,
        float(case.get("source_ops_over_rho_float") if case.get("source_ops_over_rho_float") is not None else 10**9),
        str(case.get("case_key")),
    )


def total4_lift_partner(
    base: dict[str, Any],
    remaining: list[dict[str, Any]],
) -> dict[str, Any] | None:
    base_selector = str(base.get("leaf_selector") or "")
    if "total3" not in base_selector:
        return None
    base_rows = row_key_tuple(base)
    base_leaf_count = int(base.get("selected_leaf_count") or 0)
    partners = []
    for case in remaining:
        if str(case.get("target")) != str(base.get("target")):
            continue
        if int(case.get("transfer_index") or 0) != int(base.get("transfer_index") or 0):
            continue
        if int(case.get("top_k") or 0) != int(base.get("top_k") or 0):
            continue
        if str(case.get("policy")) != str(base.get("policy")):
            continue
        if row_key_tuple(case) != base_rows:
            continue
        if int(case.get("selected_leaf_count") or 0) <= base_leaf_count:
            continue
        if "total4" not in str(case.get("leaf_selector") or ""):
            continue
        partners.append(case)
    return min(partners, key=lift_partner_sort_key) if partners else None


def annotate_total4_lift_margins(cases: list[dict[str, Any]]) -> None:
    bases: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in cases:
        selector = str(case.get("leaf_selector") or "")
        if "total3" not in selector:
            continue
        key = total3_base_key(case)
        old = bases.get(key)
        if old is None or public_case_sort_key(case) < public_case_sort_key(old):
            bases[key] = case

    for case in cases:
        case["total4_lift_base_case_key"] = None
        case["marginal_lift_surface_ids"] = []
        case["marginal_lift_row_leaf_keys"] = []
        selector = str(case.get("leaf_selector") or "")
        if "total4" not in selector:
            continue
        base = bases.get(total3_base_key(case))
        if base is None:
            continue
        case["total4_lift_base_case_key"] = base.get("case_key")
        base_by_row = {
            str(item.get("row_key")): {int(leaf) for leaf in item.get("leaf_indices") or []}
            for item in base.get("row_leaf_keys") or []
        }
        marginal_ids = []
        marginal_rows = []
        for item in case.get("row_leaf_keys") or []:
            row_key = str(item.get("row_key"))
            leaves = {int(leaf) for leaf in item.get("leaf_indices") or []}
            base_leaves = base_by_row.get(row_key, set())
            added_leaves = sorted(leaves - base_leaves)
            if not added_leaves:
                continue
            marginal_ids.append(str(item.get("surface_id")))
            marginal = dict(item)
            marginal["base_leaf_indices"] = sorted(base_leaves)
            marginal["added_leaf_indices"] = added_leaves
            marginal_rows.append(marginal)
        case["marginal_lift_surface_ids"] = sorted(set(marginal_ids))
        case["marginal_lift_row_leaf_keys"] = marginal_rows


def update_public_diversity_state(
    case: dict[str, Any],
    used_anchor_surface_ids: set[str],
    used_roots: set[int],
    used_row_salts: set[int],
    used_row_keys: set[str],
    used_leaf_signatures: set[tuple[Any, ...]],
    used_selector_keys: set[tuple[Any, ...]],
) -> None:
    used_anchor_surface_ids.update(str(surface_id) for surface_id in case.get("anchor_surface_ids") or [])
    used_roots.update(int(root) for root in case.get("anchor_roots") or [])
    used_row_salts.update(int(salt) for salt in case.get("row_salts") or [])
    used_row_keys.update(str(key) for key in case.get("row_keys") or [])
    for signature in case.get("leaf_signatures") or []:
        if len(signature) == 2:
            used_leaf_signatures.add(
                (
                    str(signature[0]),
                    tuple(int(leaf) for leaf in signature[1]),
                )
            )
    used_selector_keys.add(tuple(case.get("selector_key") or ()))


def greedy_score(
    case: dict[str, Any],
    used_anchor_surface_ids: set[str],
    used_roots: set[int],
    used_row_salts: set[int],
    used_row_keys: set[str],
    used_leaf_signatures: set[tuple[Any, ...]],
    used_selector_keys: set[tuple[Any, ...]],
) -> tuple[Any, ...]:
    anchor_ids = {str(surface_id) for surface_id in case.get("anchor_surface_ids") or []}
    roots = {int(root) for root in case.get("anchor_roots") or []}
    row_salts = {int(salt) for salt in case.get("row_salts") or []}
    row_keys = {str(key) for key in case.get("row_keys") or []}
    signatures = {
        (
            str(signature[0]),
            tuple(int(leaf) for leaf in signature[1]),
        )
        for signature in case.get("leaf_signatures") or []
        if len(signature) == 2
    }
    selector_key = tuple(case.get("selector_key") or ())
    return (
        -len(anchor_ids - used_anchor_surface_ids),
        -len(roots - used_roots),
        -len(row_salts - used_row_salts),
        -len(row_keys - used_row_keys),
        -len(signatures - used_leaf_signatures),
        -int(selector_key not in used_selector_keys),
        float(case.get("source_ops_over_rho_float") if case.get("source_ops_over_rho_float") is not None else 10**9),
        str(case.get("case_key")),
    )


def select_public_diverse_cases(
    candidate_cases: list[dict[str, Any]],
    max_cases_per_challenge: int,
    pair_total4_lifts: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in candidate_cases:
        grouped[(str(case.get("target")), int(case.get("transfer_index") or 0))].append(case)

    selected: list[dict[str, Any]] = []
    cap = max(1, int(max_cases_per_challenge))
    for _challenge, rows in sorted(grouped.items()):
        remaining = list(rows)
        used_anchor_surface_ids: set[str] = set()
        used_roots: set[int] = set()
        used_row_salts: set[int] = set()
        used_row_keys: set[str] = set()
        used_leaf_signatures: set[tuple[Any, ...]] = set()
        used_selector_keys: set[tuple[Any, ...]] = set()
        chosen = []
        while remaining and len(chosen) < cap:
            best = min(
                remaining,
                key=lambda case: greedy_score(
                    case,
                    used_anchor_surface_ids,
                    used_roots,
                    used_row_salts,
                    used_row_keys,
                    used_leaf_signatures,
                    used_selector_keys,
                ),
            )
            remaining.remove(best)
            chosen.append(best)
            update_public_diversity_state(
                best,
                used_anchor_surface_ids,
                used_roots,
                used_row_salts,
                used_row_keys,
                used_leaf_signatures,
                used_selector_keys,
            )
            if pair_total4_lifts and len(chosen) < cap:
                partner = total4_lift_partner(best, remaining)
                if partner is not None:
                    remaining.remove(partner)
                    chosen.append(partner)
                    update_public_diversity_state(
                        partner,
                        used_anchor_surface_ids,
                        used_roots,
                        used_row_salts,
                        used_row_keys,
                        used_leaf_signatures,
                        used_selector_keys,
                    )
        selected.extend(chosen)

    return sorted(
        selected,
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
            float(row.get("source_ops_over_rho_float") if row.get("source_ops_over_rho_float") is not None else 10**9),
            str(row.get("case_key")),
        ),
    )


def strip_internal_case_fields(case: dict[str, Any]) -> dict[str, Any]:
    out = dict(case)
    out.pop("source_ops_over_rho_float", None)
    out.pop("selector_key", None)
    return out


def projected_retention_case(case: dict[str, Any], retain_margins_only: bool) -> dict[str, Any] | None:
    out = strip_internal_case_fields(case)
    if not retain_margins_only:
        return out
    marginal_ids = [str(surface_id) for surface_id in case.get("marginal_lift_surface_ids") or []]
    if not marginal_ids:
        return None
    marginal_rows = [
        item
        for item in case.get("marginal_lift_row_leaf_keys") or []
        if str(item.get("surface_id")) in set(marginal_ids)
    ]
    anchor_ids = set(out.get("anchor_surface_ids") or [])
    out["retention_mode"] = "total4_lift_marginal_surfaces_only"
    out["surface_ids"] = sorted(set(marginal_ids))
    out["anchor_surface_ids"] = sorted(anchor_ids.intersection(marginal_ids))
    out["companion_surface_ids"] = sorted(set(marginal_ids) - anchor_ids)
    out["row_leaf_keys"] = marginal_rows
    out["marginal_lift_surface_ids"] = sorted(set(marginal_ids))
    out["marginal_lift_row_leaf_keys"] = marginal_rows
    return out


def compact_anchor_surface(row: dict[str, Any], source_case_keys: list[str]) -> dict[str, Any]:
    chosen = row.get("chosen_candidate") or {}
    row_key = str(row.get("row_key") or "")
    return {
        "surface_id": row.get("surface_id"),
        "surface_role": "root_policy_anchor",
        "target": row.get("target"),
        "transfer_index": int(row.get("transfer_index") or 0),
        "row_key": row_key,
        "row_schedule_key": row_schedule_key(row_key),
        "salt": row.get("salt") if row.get("salt") is not None else row_salt(row_key),
        "p": int(row.get("p") or 0),
        "pre_factor_selected_hit_roots": int_list(row.get("pre_factor_selected_hit_roots")),
        "pre_factor_selected_hit_root_ambiguity": row.get("pre_factor_selected_hit_root_ambiguity") or {},
        "public_zero_recovered": bool(row.get("public_zero_recovered")),
        "chosen_root": chosen.get("root"),
        "chosen_factor_index": chosen.get("factor_index"),
        "evaluated_root_count": int(row.get("evaluated_root_count") or 0),
        "selector_eval_ops": int(row.get("selector_eval_ops") or 0),
        "root_scan_ops": row.get("root_scan_ops"),
        "total_ops_over_rho": row.get("total_ops_over_rho"),
        "direct_root_recovery_ops": row.get("direct_root_recovery_ops"),
        "direct_total_ops_over_rho": row.get("direct_total_ops_over_rho"),
        "generic_rho_steps": int(row.get("generic_rho_steps") or 0),
        "source_case_count": len(source_case_keys),
        "source_case_keys": sorted(source_case_keys),
    }


def compact_companion_surface(surface_id: str, source_case_keys: list[str]) -> dict[str, Any]:
    parts = surface_id_parts(surface_id)
    return {
        **parts,
        "surface_role": "public_companion",
        "source_case_count": len(source_case_keys),
        "source_case_keys": sorted(source_case_keys),
        "note": "Companion surface retained because a public root-diverse source case selected it.",
    }


def retained_surfaces(
    cases: list[dict[str, Any]],
    public_rows_by_surface: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    case_keys_by_surface: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for surface_id in case.get("surface_ids") or []:
            case_keys_by_surface[str(surface_id)].append(str(case.get("case_key")))

    surfaces = []
    for surface_id in sorted(case_keys_by_surface):
        if surface_id in public_rows_by_surface:
            surfaces.append(
                compact_anchor_surface(public_rows_by_surface[surface_id], case_keys_by_surface[surface_id])
            )
        else:
            surfaces.append(compact_companion_surface(surface_id, case_keys_by_surface[surface_id]))
    return surfaces


def summarize_cases(cases: list[dict[str, Any]], all_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(case["ops_over_rho"])
        for case in cases
        if case.get("ops_over_rho") is not None
    ]
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    by_leaf_selector: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    for case in cases:
        target = str(case.get("target"))
        by_target[target]["selected_source_case_count"] += 1
        by_target[target]["anchor_surface_reference_count"] += len(case.get("anchor_surface_ids") or [])
        by_target[target]["retained_surface_reference_count"] += len(case.get("surface_ids") or [])
        by_leaf_selector[str(case.get("leaf_selector"))] += 1
        by_policy[str(case.get("policy"))] += 1
    return {
        "candidate_source_case_count": len(all_candidates),
        "selected_source_case_count": len(cases),
        "selected_signature_verified_label_count": sum(
            bool(case.get("signature_public_key_verified")) for case in cases
        ),
        "selected_source_min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "selected_source_mean_ops_over_rho": mean_or_none(ratios),
        "selected_source_max_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "selected_challenge_count": len(
            {
                (str(case.get("target")), int(case.get("transfer_index") or 0))
                for case in cases
            }
        ),
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
        "leaf_selector_counts": dict(by_leaf_selector.most_common()),
        "policy_counts": dict(by_policy.most_common()),
    }


def summarize_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_ratios = [
        float(surface["total_ops_over_rho"])
        for surface in surfaces
        if surface.get("surface_role") == "root_policy_anchor"
        and surface.get("total_ops_over_rho") is not None
    ]
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for surface in surfaces:
        target = str(surface.get("target"))
        role = str(surface.get("surface_role"))
        by_target[target]["retained_surface_count"] += 1
        by_target[target][f"{role}_surface_count"] += 1
    return {
        "retained_surface_count": len(surfaces),
        "root_policy_anchor_surface_count": sum(
            surface.get("surface_role") == "root_policy_anchor" for surface in surfaces
        ),
        "public_companion_surface_count": sum(
            surface.get("surface_role") == "public_companion" for surface in surfaces
        ),
        "anchor_min_total_ops_over_rho": round(min(anchor_ratios), 8) if anchor_ratios else None,
        "anchor_mean_total_ops_over_rho": mean_or_none(anchor_ratios),
        "anchor_max_total_ops_over_rho": round(max(anchor_ratios), 8) if anchor_ratios else None,
        "surface_target_summaries": {
            target: dict(counter) for target, counter in sorted(by_target.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--policy-source", type=Path, default=DEFAULT_POLICY_SOURCE)
    parser.add_argument("--gate-source", type=Path, default=DEFAULT_GATE_SOURCE)
    parser.add_argument("--window-label", default="total3_total4_public_192_199")
    parser.add_argument("--max-total-ops-over-rho", type=float, default=1.0)
    parser.add_argument("--require-direct-below-rho", action="store_true")
    parser.add_argument("--allow-root-ambiguity", action="store_true")
    parser.add_argument(
        "--pair-total4-lifts",
        action="store_true",
        help="After selecting a total3 assembly, immediately retain a same-row total4 public lift if one is available.",
    )
    parser.add_argument(
        "--retain-total4-lift-margins-only",
        action="store_true",
        help="For selected total4 lifts, retain only row surfaces whose leaf set strictly extends a same-row total3 partner.",
    )
    parser.add_argument("--max-cases-per-challenge", type=int, default=8)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = load_json(args.signature_source)
    policy_source = load_json(args.policy_source)
    public_rows_by_surface = public_root_policy_rows(
        policy_source,
        float(args.max_total_ops_over_rho),
        bool(args.require_direct_below_rho),
        bool(args.allow_root_ambiguity),
    )
    candidate_cases = collect_candidate_cases(
        signature,
        public_rows_by_surface,
        str(args.seed),
    )
    annotate_total4_lift_margins(candidate_cases)
    selected_cases_internal = select_public_diverse_cases(
        candidate_cases,
        int(args.max_cases_per_challenge),
        bool(args.pair_total4_lifts),
    )
    selected_cases = [
        projected
        for case in selected_cases_internal
        for projected in [
            projected_retention_case(
                case,
                bool(args.retain_total4_lift_margins_only),
            )
        ]
        if projected is not None
    ]
    surfaces = retained_surfaces(selected_cases, public_rows_by_surface)
    output = {
        "schema": "ecdlp_ffe_public_root_diversity_companion_selector_probe_v1",
        "method": "public_root_diversity_complete_companion_assembly_selector",
        "parameters": {
            "signature_source": str(args.signature_source),
            "policy_source": str(args.policy_source),
            "gate_source": str(args.gate_source),
            "windows": [
                {
                    "label": str(args.window_label),
                    "signature_source": str(args.signature_source),
                    "policy_source": str(args.policy_source),
                    "gate_source": str(args.gate_source),
                }
            ],
            "max_total_ops_over_rho": args.max_total_ops_over_rho,
            "require_direct_below_rho": bool(args.require_direct_below_rho),
            "allow_root_ambiguity": bool(args.allow_root_ambiguity),
            "pair_total4_lifts": bool(args.pair_total4_lifts),
            "retain_total4_lift_margins_only": bool(args.retain_total4_lift_margins_only),
            "max_cases_per_challenge": args.max_cases_per_challenge,
            "seed": args.seed,
            "selection_rule": [
                "filter root-policy anchors by public_zero_recovered and public root-cost threshold",
                "candidate cases must intersect at least one root-policy anchor surface",
                "within each target/transfer, greedily prefer new anchor surfaces, roots, salts, row keys, leaf signatures, and selector keys",
                "optionally pair selected total3 assemblies with same-row public total4 lifts",
                "optionally retain only the marginal row surfaces where a total4 lift adds leaves over its total3 partner",
                "retain every row/leaf surface from each selected public case",
            ],
            "forbidden_selection_fields": [
                "public_key_verified",
                "relation_count",
                "rank",
                "chosen_preserves_selected_root_pairs",
                "chosen_false_positive",
                "below_rho",
                "direct_below_rho",
                "derived",
                "derived_secret",
            ],
        },
        "summary": {
            **summarize_surfaces(surfaces),
            **summarize_cases(selected_cases, candidate_cases),
            "root_policy_public_candidate_surface_count": len(public_rows_by_surface),
            "assembly_status": "public_root_diversity_companion_pre_replay_selection_ready",
            "next_obligation": "Replay complete companion assemblies and compare rank recovery against retained-root-only replay.",
        },
        "retained_surfaces": surfaces,
        "retained_source_cases": selected_cases,
        "non_claims": [
            "This selector does not inspect verifier relation outcomes.",
            "Copied signature verification fields are labels only and are not part of selection.",
            "A replay success remains a component result until repeated on fresh windows and generalized beyond signature-provided row/leaf keys.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
