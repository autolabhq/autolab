#!/usr/bin/env python3
"""Promote complete public companion assemblies around retained root anchors.

The single-hit-root anchor selector intentionally replays only the retained
root surface.  On 176-183 that exposed a rank-1 boundary: the anchor is real,
but the companion row/leaf from the public low-term case carries the second
independent equation.

This probe keeps the same public anchor selection, then promotes every
row/leaf surface in those selected source cases into the replay set.  It does
not choose cases from verifier outcomes; public-key verification, relation
count, rank, preserving labels, and false-positive labels are copied only as
evaluation fields.
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
    DEFAULT_STATE_DIR / "ffe_public_single_hit_root_assembly_selector_176_183.json"
)
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total2_candidate_signature_fixed_selector_176_183.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_companion_assembly_selector_176_183.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def selected_case_keys(anchor_selector: dict[str, Any]) -> set[str]:
    return {
        str(case.get("case_key"))
        for case in anchor_selector.get("retained_source_cases") or []
        if isinstance(case, dict) and case.get("case_key") is not None
    }


def anchor_rows(anchor_selector: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("surface_id")): row
        for row in anchor_selector.get("retained_surfaces") or []
        if isinstance(row, dict) and row.get("surface_id") is not None
    }


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


def selected_cases_from_signature(
    signature: dict[str, Any],
    chosen_case_keys: set[str],
    anchor_by_surface: dict[str, dict[str, Any]],
    seed: str,
) -> list[dict[str, Any]]:
    cases = []
    for case in signature.get("positive_cases") or []:
        if not isinstance(case, dict):
            continue
        key = anchor_probe.case_key_string(case)
        if key not in chosen_case_keys:
            continue
        row_leaf_keys = compact_row_leaf_keys(case, seed)
        surface_ids = sorted({str(item["surface_id"]) for item in row_leaf_keys})
        anchor_surface_ids = sorted(surface_id for surface_id in surface_ids if surface_id in anchor_by_surface)
        if not anchor_surface_ids:
            continue
        cases.append(
            {
                "case_key": key,
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
                "companion_surface_ids": sorted(set(surface_ids) - set(anchor_surface_ids)),
                "surface_ids": surface_ids,
                "row_leaf_keys": row_leaf_keys,
            }
        )
    return sorted(
        cases,
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
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


def compact_companion_surface(surface_id: str, case_keys: list[str]) -> dict[str, Any]:
    parts = surface_id_parts(surface_id)
    return {
        **parts,
        "surface_role": "companion",
        "source_case_count": len(case_keys),
        "source_case_keys": sorted(case_keys),
        "note": "Companion surface retained because it is part of a selected public source assembly.",
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
            out.append(compact_companion_surface(surface_id, case_keys_by_surface[surface_id]))
    return out


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(case["ops_over_rho"])
        for case in cases
        if case.get("ops_over_rho") is not None
    ]
    return {
        "selected_source_case_count": len(cases),
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
        "companion_surface_count": sum(surface.get("surface_role") == "companion" for surface in surfaces),
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-selector-source", type=Path, default=DEFAULT_ANCHOR_SELECTOR_SOURCE)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--window-label", default="total2_candidates_176_183")
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    anchor_selector = load_json(args.anchor_selector_source)
    signature = load_json(args.signature_source)
    anchors = anchor_rows(anchor_selector)
    cases = selected_cases_from_signature(
        signature,
        selected_case_keys(anchor_selector),
        anchors,
        str(args.seed),
    )
    surfaces = retained_surfaces(cases, anchors)
    output = {
        "schema": "ecdlp_ffe_public_companion_assembly_selector_probe_v1",
        "method": "public_anchor_plus_complete_companion_row_leaf_selector",
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
                "use source cases already selected by the public retained-root anchor selector",
                "retain every row/leaf surface in each selected source case",
                "rank and case limits are inherited from the anchor selector",
            ],
            "forbidden_selection_fields": [
                "public_key_verified",
                "relation_count",
                "rank",
                "chosen_preserves_selected_root_pairs",
                "chosen_false_positive",
                "below_rho",
                "direct_below_rho",
            ],
            "seed": args.seed,
        },
        "summary": {
            **summarize_surfaces(surfaces),
            **summarize_cases(cases),
            "assembly_status": "public_companion_pre_replay_selection_ready",
            "next_obligation": "Replay complete companion assemblies and compare rank recovery against retained-root-only replay.",
        },
        "retained_surfaces": surfaces,
        "retained_source_cases": cases,
        "non_claims": [
            "This selector inherits the public anchor cases; it does not inspect relation derivation outcomes.",
            "Companion rows are retained only because they are part of a selected public source assembly.",
            "A companion replay win is still a component result until repeated on fresh windows.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
