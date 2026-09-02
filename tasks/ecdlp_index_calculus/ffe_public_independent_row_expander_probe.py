#!/usr/bin/env python3
"""Expand public retained-root assemblies with independent row candidates.

The retained-root and companion selectors test whether a public single-hit FFE
surface can replay known low-term relation assemblies.  This probe takes the
next conservative step: start from the public anchor-selected challenge groups,
then greedily add below-rho source cases that diversify row salts, row keys,
leaf signatures, and surface IDs.

The selector intentionally does not use public-key verification, relation
count, rank, preserving labels, or false-positive labels for selection.  Those
fields are copied only as evaluation labels so replay can measure whether the
public diversity heuristic lifts rank after the fact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_assembly_selector_probe as anchor_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_ANCHOR_SELECTOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_single_hit_root_assembly_selector_184_191.json"
)
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total2_candidate_signature_fixed_selector_184_191.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_independent_row_expander_selector_184_191.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


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
        "row_key": row_key,
        "salt": anchor_probe.row_salt(row_key),
        "challenge_seed": challenge_seed,
        "transfer_index": transfer_index,
    }


def anchor_case_keys(anchor_selector: dict[str, Any]) -> set[str]:
    return {
        str(case.get("case_key"))
        for case in anchor_selector.get("retained_source_cases") or []
        if isinstance(case, dict) and case.get("case_key") is not None
    }


def anchor_surfaces(anchor_selector: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("surface_id")): row
        for row in anchor_selector.get("retained_surfaces") or []
        if isinstance(row, dict) and row.get("surface_id") is not None
    }


def compact_row_leaf_keys(case: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    out = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        leaves = [int(leaf) for leaf in item.get("leaf_indices") or []]
        if not row_key or not leaves:
            continue
        surface_id = anchor_probe.signature_surface_id(case, item, seed)
        out.append(
            {
                "row_key": row_key,
                "leaf_indices": leaves,
                "salt": anchor_probe.row_salt(row_key),
                "surface_id": surface_id,
            }
        )
    return out


def compact_case(
    case: dict[str, Any],
    seed: str,
    role: str,
    anchor_by_surface: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_leaf_keys = compact_row_leaf_keys(case, seed)
    surface_ids = sorted({str(item["surface_id"]) for item in row_leaf_keys})
    anchor_surface_ids = sorted(surface_id for surface_id in surface_ids if surface_id in anchor_by_surface)
    return {
        "case_key": anchor_probe.case_key_string(case),
        "case_role": role,
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector") or case.get("selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "signature_public_key_verified": bool(case.get("public_key_verified")),
        "signature_relation_count": int(case.get("relation_count") or 0),
        "signature_rank": int(case.get("rank") or 0),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "anchor_surface_ids": anchor_surface_ids,
        "expander_surface_ids": sorted(set(surface_ids) - set(anchor_surface_ids)),
        "surface_ids": surface_ids,
        "row_leaf_keys": row_leaf_keys,
    }


def candidate_is_public_below_rho(case: dict[str, Any], max_ops_over_rho: float) -> bool:
    try:
        ops_over_rho = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        return False
    return ops_over_rho < max_ops_over_rho


def case_row_keys(case: dict[str, Any]) -> set[str]:
    return {
        str(item.get("row_key"))
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict) and item.get("row_key")
    }


def case_salts(case: dict[str, Any]) -> set[int]:
    salts = set()
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        salt = item.get("salt")
        if salt is None:
            salt = anchor_probe.row_salt(str(item.get("row_key") or ""))
        if salt is not None:
            salts.add(int(salt))
    return salts


def case_leaf_signatures(case: dict[str, Any]) -> set[tuple[int, ...]]:
    return {
        tuple(int(leaf) for leaf in item.get("leaf_indices") or [])
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict) and item.get("leaf_indices")
    }


def case_surface_ids(case: dict[str, Any]) -> set[str]:
    return {
        str(surface_id)
        for surface_id in case.get("surface_ids") or []
        if surface_id is not None
    }


def public_diversity_score(
    case: dict[str, Any],
    selected: list[dict[str, Any]],
) -> tuple[int, int, int, int, float, str]:
    selected_salts = set().union(*(case_salts(row) for row in selected)) if selected else set()
    selected_row_keys = set().union(*(case_row_keys(row) for row in selected)) if selected else set()
    selected_leaf_signatures = (
        set().union(*(case_leaf_signatures(row) for row in selected)) if selected else set()
    )
    selected_surface_ids = set().union(*(case_surface_ids(row) for row in selected)) if selected else set()
    salts = case_salts(case)
    row_keys = case_row_keys(case)
    leaf_signatures = case_leaf_signatures(case)
    surface_ids = case_surface_ids(case)
    try:
        ops_over_rho = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        ops_over_rho = 10**9
    return (
        len(salts - selected_salts),
        len(row_keys - selected_row_keys),
        len(leaf_signatures - selected_leaf_signatures),
        len(surface_ids - selected_surface_ids),
        -ops_over_rho,
        str(case.get("case_key")),
    )


def select_cases(
    signature: dict[str, Any],
    anchor_selector: dict[str, Any],
    seed: str,
    max_ops_over_rho: float,
    extra_cases_per_challenge: int,
) -> list[dict[str, Any]]:
    anchors = anchor_surfaces(anchor_selector)
    selected_anchor_keys = anchor_case_keys(anchor_selector)
    cases_by_key = {
        anchor_probe.case_key_string(case): case
        for case in signature.get("positive_cases") or []
        if isinstance(case, dict)
    }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for key in selected_anchor_keys:
        source_case = cases_by_key.get(key)
        if source_case is None:
            continue
        compact = compact_case(source_case, seed, "anchor", anchors)
        grouped[(str(compact.get("target")), int(compact.get("transfer_index") or 0))].append(compact)

    for group_key, selected in list(grouped.items()):
        target, transfer_index = group_key
        already_selected = {str(case.get("case_key")) for case in selected}
        pool = []
        for source_case in cases_by_key.values():
            if str(source_case.get("target")) != target:
                continue
            if int(source_case.get("transfer_index") or 0) != transfer_index:
                continue
            key = anchor_probe.case_key_string(source_case)
            if key in already_selected:
                continue
            if not candidate_is_public_below_rho(source_case, max_ops_over_rho):
                continue
            compact = compact_case(source_case, seed, "expander", anchors)
            if not compact.get("row_leaf_keys"):
                continue
            pool.append(compact)

        for _ in range(max(0, int(extra_cases_per_challenge))):
            if not pool:
                break
            best = max(pool, key=lambda row: public_diversity_score(row, selected))
            selected.append(best)
            already_selected.add(str(best.get("case_key")))
            pool = [case for case in pool if str(case.get("case_key")) not in already_selected]

    return sorted(
        [case for rows in grouped.values() for case in rows],
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
            0 if row.get("case_role") == "anchor" else 1,
            float(row.get("ops_over_rho") or 10**9),
            str(row.get("case_key")),
        ),
    )


def compact_anchor_surface(row: dict[str, Any], case_keys: list[str]) -> dict[str, Any]:
    out = dict(row)
    out["surface_role"] = "anchor"
    out["source_case_count"] = len(case_keys)
    out["source_case_keys"] = sorted(case_keys)
    return out


def compact_expander_surface(surface_id: str, case_keys: list[str]) -> dict[str, Any]:
    return {
        **surface_id_parts(surface_id),
        "surface_role": "expander",
        "source_case_count": len(case_keys),
        "source_case_keys": sorted(case_keys),
        "note": "Expander surface retained by public row/leaf diversity around an anchor challenge group.",
    }


def retained_surfaces(
    cases: list[dict[str, Any]],
    anchor_by_surface: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    case_keys_by_surface: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for surface_id in case.get("surface_ids") or []:
            case_keys_by_surface[str(surface_id)].append(str(case.get("case_key")))

    out = []
    for surface_id in sorted(case_keys_by_surface):
        if surface_id in anchor_by_surface:
            out.append(compact_anchor_surface(anchor_by_surface[surface_id], case_keys_by_surface[surface_id]))
        else:
            out.append(compact_expander_surface(surface_id, case_keys_by_surface[surface_id]))
    return out


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(case["ops_over_rho"])
        for case in cases
        if case.get("ops_over_rho") is not None
    ]
    return {
        "selected_source_case_count": len(cases),
        "anchor_case_count": sum(case.get("case_role") == "anchor" for case in cases),
        "added_case_count": sum(case.get("case_role") == "expander" for case in cases),
        "selected_signature_verified_case_count": sum(
            bool(case.get("signature_public_key_verified")) for case in cases
        ),
        "selected_source_min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "selected_source_mean_ops_over_rho": mean_or_none(ratios),
        "selected_source_max_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "challenge_count": len(
            {
                (str(case.get("target")), int(case.get("transfer_index") or 0))
                for case in cases
            }
        ),
    }


def summarize_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for surface in surfaces:
        target = str(surface.get("target"))
        role = str(surface.get("surface_role"))
        by_target[target]["surface_count"] += 1
        by_target[target][f"{role}_surface_count"] += 1
    return {
        "retained_surface_count": len(surfaces),
        "anchor_surface_count": sum(surface.get("surface_role") == "anchor" for surface in surfaces),
        "expander_surface_count": sum(surface.get("surface_role") == "expander" for surface in surfaces),
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-selector-source", type=Path, default=DEFAULT_ANCHOR_SELECTOR_SOURCE)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--window-label", default="total2_candidates_184_191")
    parser.add_argument("--max-ops-over-rho", type=float, default=1.0)
    parser.add_argument("--extra-cases-per-challenge", type=int, default=4)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    anchor_selector = load_json(args.anchor_selector_source)
    signature = load_json(args.signature_source)
    cases = select_cases(
        signature,
        anchor_selector,
        str(args.seed),
        float(args.max_ops_over_rho),
        int(args.extra_cases_per_challenge),
    )
    surfaces = retained_surfaces(cases, anchor_surfaces(anchor_selector))
    output = {
        "schema": "ecdlp_ffe_public_independent_row_expander_probe_v1",
        "method": "public_anchor_seeded_independent_row_diversity_expander",
        "parameters": {
            "anchor_selector_source": str(args.anchor_selector_source),
            "signature_source": str(args.signature_source),
            "windows": [
                {
                    "label": str(args.window_label),
                    "signature_source": str(args.signature_source),
                    "anchor_selector_source": str(args.anchor_selector_source),
                }
            ],
            "selection_rule": [
                "seed each target/transfer challenge with public anchor-selected source cases",
                "candidate pool is same target/transfer cases with ops_over_rho below threshold",
                "greedily add candidates with new row salts, row keys, leaf signatures, and surface IDs",
            ],
            "max_ops_over_rho": args.max_ops_over_rho,
            "extra_cases_per_challenge": args.extra_cases_per_challenge,
            "seed": args.seed,
            "forbidden_selection_fields": [
                "public_key_verified",
                "relation_count",
                "rank",
                "chosen_preserves_selected_root_pairs",
                "chosen_false_positive",
                "below_rho",
                "direct_below_rho",
            ],
        },
        "summary": {
            **summarize_surfaces(surfaces),
            **summarize_cases(cases),
            "assembly_status": "public_independent_row_expander_pre_replay_selection_ready",
            "next_obligation": "Replay expanded assemblies and test whether public diversity lifts retained rank.",
        },
        "retained_surfaces": surfaces,
        "retained_source_cases": cases,
        "non_claims": [
            "This selector expands only challenge groups already selected by the public anchor selector.",
            "Verification, relation count, and rank are copied as labels only and are not part of the scoring rule.",
            "A replay win remains a component result until it repeats across fresh windows and beats the Pollard-rho baseline end to end.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
