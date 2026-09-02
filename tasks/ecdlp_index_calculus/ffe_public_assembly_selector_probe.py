#!/usr/bin/env python3
"""Select public retained-root assemblies before relation replay.

This probe is the bridge between the single-hit-root FFE component and a more
algorithmic relation assembly test.  It reads a public below-rho candidate
signature plus the preregistered pre-factor root policy output, then selects
surface assemblies using only public/root-cost fields:

* exactly one pre-factor selected hit root;
* a public-zero root hyperplane was recovered;
* root-scan or direct-root public work is below the rho proxy;
* one candidate per target/transfer/row/leaf signature is retained.

It intentionally does not use public-key verification, relation count, rank,
chosen-preserving labels, false-positive labels, or below-rho labels that are
conditioned on preserving factors to decide which assemblies to replay.
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
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total2_candidate_signature_fixed_selector_176_183.json"
DEFAULT_POLICY_SOURCE = DEFAULT_STATE_DIR / "ffe_prefactor_unique_leaf_single_hit_root_policy_total2_candidates_176_183.json"
DEFAULT_GATE_SOURCE = DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total2_candidates_176_183.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_single_hit_root_assembly_selector_176_183.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    leaf_signature = tuple(
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
        leaf_signature,
    )


def case_key_string(case: dict[str, Any]) -> str:
    return "|".join(str(part) for part in case_key(case))


def signature_surface_id(case: dict[str, Any], item: dict[str, Any], seed: str) -> str:
    if item.get("surface_id"):
        return str(item["surface_id"])
    target = str(case.get("target"))
    transfer_index = int(case.get("transfer_index") or 0)
    row_key = str(item.get("row_key"))
    challenge_seed = f"{seed}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def public_policy_rows(
    policy_source: dict[str, Any],
    max_total_ops_over_rho: float,
    require_direct_below_rho: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in policy_source.get("rows") or []:
        if not isinstance(row, dict):
            continue
        roots = [int(root) for root in row.get("pre_factor_selected_hit_roots") or []]
        if len(set(roots)) != 1:
            continue
        if not row.get("public_zero_recovered"):
            continue
        total_ratio = row.get("total_ops_over_rho")
        direct_ratio = row.get("direct_total_ops_over_rho")
        public_ratio = direct_ratio if require_direct_below_rho else total_ratio
        if public_ratio is None:
            continue
        if float(public_ratio) >= max_total_ops_over_rho:
            continue
        surface_id = str(row.get("surface_id"))
        rows[surface_id] = row
    return rows


def compact_surface(row: dict[str, Any], source_case_keys: list[str]) -> dict[str, Any]:
    chosen = row.get("chosen_candidate") or {}
    return {
        "surface_id": row.get("surface_id"),
        "target": row.get("target"),
        "transfer_index": int(row.get("transfer_index") or 0),
        "row_key": row.get("row_key"),
        "salt": row_salt(str(row.get("row_key") or "")),
        "p": int(row.get("p") or 0),
        "pre_factor_selected_hit_roots": [
            int(root) for root in row.get("pre_factor_selected_hit_roots") or []
        ],
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


def compact_case(case: dict[str, Any], retained_ids: set[str], seed: str) -> dict[str, Any] | None:
    row_leaf_keys = []
    surface_ids = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        surface_id = signature_surface_id(case, item, seed)
        if surface_id not in retained_ids:
            continue
        leaves = [int(leaf) for leaf in item.get("leaf_indices") or []]
        if not leaves:
            continue
        row_key = str(item.get("row_key") or "")
        row_leaf_keys.append(
            {
                "row_key": row_key,
                "leaf_indices": leaves,
                "salt": row_salt(row_key),
                "surface_id": surface_id,
            }
        )
        surface_ids.append(surface_id)
    if not row_leaf_keys:
        return None
    return {
        "case_key": case_key_string(case),
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
        "retained_row_leaf_keys": row_leaf_keys,
        "surface_ids": sorted(set(surface_ids)),
    }


def select_cases(
    signature: dict[str, Any],
    public_rows_by_surface: dict[str, dict[str, Any]],
    seed: str,
    max_cases_per_challenge: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    retained_ids = set(public_rows_by_surface)
    cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in signature.get("positive_cases") or []:
        if not isinstance(case, dict):
            continue
        compact = compact_case(case, retained_ids, seed)
        if compact is None:
            continue
        cases_by_key[case_key(case)] = compact

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases_by_key.values():
        grouped[(str(case.get("target")), int(case.get("transfer_index") or 0))].append(case)

    selected: list[dict[str, Any]] = []
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row.get("ops_over_rho") or 10**9),
                len(row.get("surface_ids") or []),
                str(row.get("case_key")),
            ),
        )
        selected.extend(ranked[: max(1, int(max_cases_per_challenge))])

    case_keys_by_surface: dict[str, list[str]] = defaultdict(list)
    for case in selected:
        for surface_id in case.get("surface_ids") or []:
            case_keys_by_surface[str(surface_id)].append(str(case.get("case_key")))
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
            float(row.get("ops_over_rho") or 10**9),
            str(row.get("case_key")),
        ),
    ), case_keys_by_surface


def summarize_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
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
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in surfaces:
        by_target[str(row.get("target"))]["surface_count"] += 1
        by_target[str(row.get("target"))]["source_case_count"] += int(row.get("source_case_count") or 0)
    return {
        "retained_surface_count": len(surfaces),
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": mean_or_none(ratios),
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
        "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["ops_over_rho"])
        for row in cases
        if row.get("ops_over_rho") is not None
    ]
    return {
        "selected_source_case_count": len(cases),
        "selected_signature_verified_case_count": sum(
            bool(row.get("signature_public_key_verified")) for row in cases
        ),
        "selected_source_min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "selected_source_mean_ops_over_rho": mean_or_none(ratios),
        "selected_source_max_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "challenge_count": len(
            {
                (str(row.get("target")), int(row.get("transfer_index") or 0))
                for row in cases
            }
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--policy-source", type=Path, default=DEFAULT_POLICY_SOURCE)
    parser.add_argument("--gate-source", type=Path, default=DEFAULT_GATE_SOURCE)
    parser.add_argument("--window-label", default="total2_candidates_176_183")
    parser.add_argument("--max-total-ops-over-rho", type=float, default=1.0)
    parser.add_argument("--require-direct-below-rho", action="store_true")
    parser.add_argument("--max-cases-per-challenge", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = load_json(args.signature_source)
    policy_source = load_json(args.policy_source)
    public_rows_by_surface = public_policy_rows(
        policy_source,
        float(args.max_total_ops_over_rho),
        bool(args.require_direct_below_rho),
    )
    selected_cases, case_keys_by_surface = select_cases(
        signature,
        public_rows_by_surface,
        str(args.seed),
        int(args.max_cases_per_challenge),
    )
    selected_surface_ids = {
        str(surface_id)
        for case in selected_cases
        for surface_id in case.get("surface_ids") or []
    }
    retained_surfaces = [
        compact_surface(public_rows_by_surface[surface_id], case_keys_by_surface.get(surface_id, []))
        for surface_id in sorted(selected_surface_ids)
    ]
    output = {
        "schema": "ecdlp_ffe_public_single_hit_root_assembly_selector_probe_v1",
        "method": "public_single_hit_root_root_cost_assembly_selector",
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
            "selection_rule": [
                "exactly one pre_factor_selected_hit_root",
                "public_zero_recovered",
                (
                    "direct_total_ops_over_rho < threshold"
                    if args.require_direct_below_rho
                    else "total_ops_over_rho < threshold"
                ),
                "rank by public source ops/rho then retained surface count",
            ],
            "max_total_ops_over_rho": args.max_total_ops_over_rho,
            "max_cases_per_challenge": args.max_cases_per_challenge,
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
            **summarize_surfaces(retained_surfaces),
            **summarize_cases(selected_cases),
            "assembly_status": "public_pre_replay_selection_ready",
            "next_obligation": "Replay the selected assemblies and count verifier-derived public-key successes.",
        },
        "retained_surfaces": retained_surfaces,
        "retained_source_cases": selected_cases,
        "non_claims": [
            "This selector does not inspect relation derivation outcomes.",
            "This selector does not use preserving-factor labels or false-positive labels.",
            "This is not an ECDLP speedup until replayed assemblies publicly derive fresh challenge keys.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
