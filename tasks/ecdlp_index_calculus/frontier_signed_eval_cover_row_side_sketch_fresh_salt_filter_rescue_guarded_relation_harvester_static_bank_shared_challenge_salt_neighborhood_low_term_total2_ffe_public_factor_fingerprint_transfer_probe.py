#!/usr/bin/env python3
"""Leaf-agnostic fingerprint transfer audit for Sage-factored FFE surfaces.

The live public factor selector can stop on the first factor vanishing at the
verifier-selected leaf point.  That is a useful diagnostic, but it still knows
the relation-bearing leaf.  This probe removes that leaf from selection:

* learn reusable factor fingerprints only from calibration surfaces,
* choose a matching factor on holdout surfaces without selected-leaf zero tests,
* scan all public monic leaves for that chosen factor, and
* recover roots by quadratic solving plus hit-root membership.

Labels are used to score calibration splits and to audit holdout preservation,
not to select a factor within a holdout surface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
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
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_selector_probe as selector_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_remainder_factor_probe as remainder_factor_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_slice_quadratic_root_probe as slice_quadratic_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = cross_surface_probe.DEFAULT_SIGNATURE_SOURCE
DEFAULT_SAGE_FACTOR_SOURCE = (
    Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_transfer_probe.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def transfer_index(surface: dict[str, Any]) -> int | None:
    match = re.search(r"shared-transfer:(\d+):", str(surface.get("challenge_seed") or ""))
    return int(match.group(1)) if match else None


def factor_index(candidate: dict[str, Any]) -> int | None:
    match = re.search(r"_(\d+)$", str(candidate.get("candidate_name") or ""))
    return int(match.group(1)) if match else None


def factor_terms(surface: dict[str, Any], candidate: dict[str, Any]) -> dict[tuple[int, int], int]:
    index = factor_index(candidate)
    factors = ((surface.get("sage_resultant_factorization") or {}).get("factors") or [])
    if index is None or index < 0 or index >= len(factors):
        return {}
    row = factors[index]
    if int(row.get("monomials") or 0) > int(len(row.get("fingerprint") or [])):
        return {}
    return {
        (int(item[0]), int(item[1])): int(item[2]) % int(surface["p"])
        for item in row.get("fingerprint") or []
        if isinstance(item, list) and len(item) == 3
    }


def fingerprint_key(surface: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, ...] | None:
    terms = factor_terms(surface, candidate)
    if not terms:
        return None
    return (
        int(surface.get("p") or 0),
        tuple(sorted((int(bd), int(cd), int(coeff)) for (bd, cd), coeff in terms.items())),
    )


def centered_coeff_norms(surface: dict[str, Any], candidate: dict[str, Any]) -> list[int]:
    p = int(surface.get("p") or 0)
    norms = []
    for _degrees, coeff in factor_terms(surface, candidate).items():
        value = int(coeff) % p
        norms.append(min(value, p - value))
    return norms


def eval_factor(terms: dict[tuple[int, int], int], b_value: int, c_value: int, p: int) -> int:
    total = 0
    for (b_degree, c_degree), coeff in terms.items():
        total += int(coeff) * pow(int(b_value), int(b_degree), p) * pow(int(c_value), int(c_degree), p)
    return total % p


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


def compact_candidate(surface: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_name": candidate.get("candidate_name"),
        "factor_index": factor_index(candidate),
        "fingerprint_key": fingerprint_key(surface, candidate),
        "factor_total_degree": candidate.get("factor_total_degree"),
        "factor_monomials": candidate.get("factor_monomials"),
        "preserves_selected_root_pairs": candidate.get("preserves_selected_root_pairs"),
        "same_selected_root_pairs": candidate.get("same_selected_root_pairs"),
        "selected_surface_zero_leaves": candidate.get("selected_surface_zero_leaves"),
    }


def public_candidates(surface: dict[str, Any], max_degree: int) -> list[dict[str, Any]]:
    rows = []
    for candidate in surface.get("sage_resultant_factor_candidates") or []:
        if int(candidate.get("factor_total_degree") or 0) <= 0:
            continue
        if int(candidate.get("factor_total_degree") or 0) > max_degree:
            continue
        key = fingerprint_key(surface, candidate)
        if key is None:
            continue
        rows.append(candidate)
    return rows


def learned_fingerprints(
    surfaces: list[dict[str, Any]],
    train_surface_ids: set[str],
    max_degree: int,
    min_support: int,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    stats: dict[tuple[Any, ...], dict[str, Any]] = {}
    for surface in surfaces:
        if str(surface.get("surface_id")) not in train_surface_ids:
            continue
        for candidate in public_candidates(surface, max_degree):
            key = fingerprint_key(surface, candidate)
            if key is None:
                continue
            row = stats.setdefault(
                key,
                {
                    "fingerprint_key": key,
                    "support": 0,
                    "preserving": 0,
                    "false_positive": 0,
                    "min_coeff_norm": None,
                    "sum_coeff_norm": None,
                },
            )
            row["support"] += 1
            if candidate.get("preserves_selected_root_pairs"):
                row["preserving"] += 1
            else:
                row["false_positive"] += 1
            norms = centered_coeff_norms(surface, candidate)
            if norms:
                max_norm = max(norms)
                sum_norm = sum(norms)
                row["min_coeff_norm"] = max_norm if row["min_coeff_norm"] is None else min(row["min_coeff_norm"], max_norm)
                row["sum_coeff_norm"] = sum_norm if row["sum_coeff_norm"] is None else min(row["sum_coeff_norm"], sum_norm)
    return {
        key: row
        for key, row in stats.items()
        if int(row["preserving"]) >= min_support and int(row["preserving"]) > int(row["false_positive"])
    }


def fingerprint_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["false_positive"]),
        -int(row["preserving"]),
        -int(row["support"]),
        int(row["min_coeff_norm"] if row["min_coeff_norm"] is not None else 10**18),
        int(row["sum_coeff_norm"] if row["sum_coeff_norm"] is not None else 10**18),
        repr(row["fingerprint_key"]),
    )


def choose_candidate(
    surface: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
) -> tuple[dict[str, Any] | None, int, int]:
    ranked_keys = {key: rank for rank, key in enumerate(sorted(learned, key=lambda key: fingerprint_rank(learned[key])))}
    candidates = public_candidates(surface, max_degree)
    selector_ops = sum(int(candidate.get("factor_monomials") or 0) for candidate in candidates)
    matched = [
        candidate
        for candidate in candidates
        if fingerprint_key(surface, candidate) in ranked_keys
    ]
    matched.sort(
        key=lambda candidate: (
            ranked_keys[fingerprint_key(surface, candidate)],
            int(candidate.get("factor_total_degree") or 10**9),
            int(candidate.get("factor_monomials") or 10**9),
            factor_index(candidate) if factor_index(candidate) is not None else 10**9,
        )
    )
    return (matched[0] if matched else None, len(candidates), selector_ops)


def audit_chosen_candidate(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    candidate_count: int,
    selector_ops: int,
    split_name: str,
    selected_policy: str,
) -> dict[str, Any]:
    p = int(surface_record["p"])
    original_pairs = remainder_factor_probe.selected_pair_set(surface_record, surface_record["surface"])
    selected_pairs: set[tuple[int, int]] = set()
    zero_leaf_count = 0
    valid_root_leaf_count = 0
    false_zero_leaf_count = 0
    recovered_root_count = 0
    selected_valid_leaf_count = 0
    coords = leaf_coords(surface_record)
    terms = factor_terms(sage_surface, candidate) if candidate else {}

    for coord in coords:
        leaf_index = int(coord["leaf_index"])
        b_value = int(coord["b"])
        c_value = int(coord["c"])
        if not terms or eval_factor(terms, b_value, c_value, p) != 0:
            continue
        zero_leaf_count += 1
        leaf = surface_record["components"]["leaves"][leaf_index]
        roots = slice_quadratic_probe.recover_roots_for_leaf(surface_record, leaf, b_value, c_value)
        if not roots:
            false_zero_leaf_count += 1
            continue
        valid_root_leaf_count += 1
        recovered_root_count += len(roots)
        if leaf_index in set(int(leaf) for leaf in surface_record["selected_leaf_indices"]):
            selected_valid_leaf_count += 1
            for root in roots:
                selected_pairs.add((leaf_index, int(root)))

    missing = sorted(original_pairs - selected_pairs)
    extra = sorted(selected_pairs - original_pairs)
    cost = surface_record["cost_inputs"]
    selected_hit_events = int(cost["selected_hit_events"])
    leaf_eval_ops = len(coords) * int((candidate or {}).get("factor_monomials") or 0)
    quadratic_root_work = 2 * zero_leaf_count
    selected_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(len(coords) / 4096)
        + int(cost["selected_hit_roots"])
    )
    all_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(len(coords) / 4096)
        + len(surface_record["components"]["hit_roots"])
    )

    def charge(core: int) -> dict[str, Any]:
        if candidate is None:
            return {
                "ops": None,
                "ops_over_rho": None,
                "beats_rho": False,
            }
        ops = (
            core
            + selector_ops
            + leaf_eval_ops
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

    selected_charge = charge(selected_hit_core)
    all_hit_charge = charge(all_hit_core)
    return {
        "split": split_name,
        "selected_policy": selected_policy,
        "surface_id": sage_surface.get("surface_id"),
        "target": sage_surface.get("target"),
        "row_key": sage_surface.get("row_key"),
        "challenge_seed": sage_surface.get("challenge_seed"),
        "transfer_index": transfer_index(sage_surface),
        "p": p,
        "candidate_count": candidate_count,
        "selector_eval_ops": selector_ops,
        "public_leaf_count": len(coords),
        "leaf_eval_ops": leaf_eval_ops,
        "chosen_candidate": compact_candidate(sage_surface, candidate),
        "chosen_preserves_label": bool(candidate and candidate.get("preserves_selected_root_pairs")),
        "original_selected_root_pair_count": len(original_pairs),
        "selected_root_pair_count": len(selected_pairs),
        "preserves_selected_root_pairs": not missing,
        "same_selected_root_pairs": not missing and not extra,
        "missing_selected_root_pairs": [[leaf, root] for leaf, root in missing[:8]],
        "extra_selected_root_pairs": [[leaf, root] for leaf, root in extra[:8]],
        "zero_leaf_count": zero_leaf_count,
        "valid_root_leaf_count": valid_root_leaf_count,
        "false_zero_leaf_count": false_zero_leaf_count,
        "selected_valid_leaf_count": selected_valid_leaf_count,
        "recovered_root_count": recovered_root_count,
        "quadratic_root_work": quadratic_root_work,
        "selected_hit_core_ops": selected_hit_core,
        "all_hit_core_ops": all_hit_core,
        "public_fingerprint_selected_hit_ops": selected_charge["ops"],
        "public_fingerprint_selected_hit_ops_over_rho": selected_charge["ops_over_rho"],
        "public_fingerprint_selected_hit_beats_rho": selected_charge["beats_rho"],
        "public_fingerprint_all_hit_ops": all_hit_charge["ops"],
        "public_fingerprint_all_hit_ops_over_rho": all_hit_charge["ops_over_rho"],
        "public_fingerprint_all_hit_beats_rho": all_hit_charge["beats_rho"],
        "generic_rho_steps": int(cost["generic_rho_steps"]),
    }


def materialize_records(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
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
    return {str(surface["surface_id"]): surface for surface in surface_records}, case_results


def rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["public_fingerprint_all_hit_ops_over_rho"])
        for row in rows
        if row.get("public_fingerprint_all_hit_ops_over_rho") is not None
    ]
    return {
        "surface_count": len(rows),
        "chosen_factor_count": sum(1 for row in rows if row.get("chosen_candidate")),
        "preserving_surface_count": sum(1 for row in rows if row.get("preserves_selected_root_pairs")),
        "same_surface_count": sum(1 for row in rows if row.get("same_selected_root_pairs")),
        "false_positive_surface_count": sum(
            1 for row in rows if row.get("chosen_candidate") and not row.get("preserves_selected_root_pairs")
        ),
        "selected_hit_below_rho_count": sum(
            1 for row in rows if row.get("public_fingerprint_selected_hit_beats_rho")
        ),
        "all_hit_below_rho_count": sum(
            1 for row in rows if row.get("public_fingerprint_all_hit_beats_rho")
        ),
        "min_all_hit_ops_over_rho": min(ratios) if ratios else None,
        "mean_all_hit_ops_over_rho": round(sum(ratios) / len(ratios), 8) if ratios else None,
        "max_all_hit_ops_over_rho": max(ratios) if ratios else None,
    }


def split_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    learned = learned_fingerprints(surfaces, train_ids, int(args.max_factor_degree), int(args.min_support))
    rows = []
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in test_ids or surface_id not in surface_records:
            continue
        candidate, candidate_count, selector_ops = choose_candidate(surface, learned, int(args.max_factor_degree))
        rows.append(
            audit_chosen_candidate(
                surface,
                surface_records[surface_id],
                candidate,
                candidate_count,
                selector_ops,
                split_name,
                "calibrated_fingerprint_reuse",
            )
        )
    return {
        "split": split_name,
        "train_surface_count": len(train_ids),
        "test_surface_count": len(test_ids),
        "learned_fingerprint_count": len(learned),
        "summary": rows_summary(rows),
        "rows": rows,
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
    transfers = sorted({transfer_index(surface) for surface in surfaces if transfer_index(surface) is not None})
    for transfer in transfers:
        test = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) == transfer}
        splits.append(split_rows(f"holdout_transfer_{transfer}", surfaces, surface_records, surface_ids - test, test, args))
    for idx in range(int(args.min_train_transfers), len(transfers)):
        train_transfers = set(transfers[:idx])
        holdout = transfers[idx]
        train = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) in train_transfers}
        test = {str(surface.get("surface_id")) for surface in surfaces if transfer_index(surface) == holdout}
        splits.append(split_rows(f"rolling_to_transfer_{holdout}", surfaces, surface_records, train, test, args))
    return splits


def summarize(case_results: list[dict[str, Any]], splits: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for split in splits
        for row in split.get("rows", [])
        if isinstance(row, dict)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in splits:
        if str(split.get("split", "")).startswith("holdout_target_"):
            grouped["target"].extend(split.get("rows") or [])
        elif str(split.get("split", "")).startswith("holdout_transfer_"):
            grouped["transfer"].extend(split.get("rows") or [])
        elif str(split.get("split", "")).startswith("rolling_to_transfer_"):
            grouped["rolling"].extend(split.get("rows") or [])
    return {
        "case_count": len(case_results),
        "verified_case_count": sum(1 for case in case_results if case.get("public_key_verified")),
        "split_count": len(splits),
        "aggregate_summary": rows_summary(rows),
        "split_family_summaries": {key: rows_summary(value) for key, value in sorted(grouped.items())},
        "positive_split_count": sum(
            1 for split in splits if int((split.get("summary") or {}).get("all_hit_below_rho_count") or 0) > 0
        ),
        "all_splits_preserve": all(
            int((split.get("summary") or {}).get("preserving_surface_count") or 0)
            == int((split.get("summary") or {}).get("surface_count") or 0)
            for split in splits
        ),
        "all_splits_false_positive_free": all(
            int((split.get("summary") or {}).get("false_positive_surface_count") or 0) == 0
            for split in splits
        ),
        "best_rows": sorted(
            rows,
            key=lambda row: (
                float(row.get("public_fingerprint_all_hit_ops_over_rho") or 10**18),
                str(row.get("surface_id")),
            ),
        )[:10],
        "interpretation": (
            "This is a leaf-agnostic transfer gate: factor fingerprints are learned "
            "from calibration surfaces, selected on holdout surfaces without the "
            "selected-leaf zero predicate, and then charged for scanning all public "
            "monic leaves before quadratic root recovery."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--sage-factor-source", type=Path, default=DEFAULT_SAGE_FACTOR_SOURCE)
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
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sage_source = load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = materialize_records(args)
    splits = holdout_splits(surfaces, surface_records, args)
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_fingerprint_transfer_probe_v1",
        "method": "leaf_agnostic_calibrated_factor_fingerprint_transfer",
        "parameters": {
            "campaign_task_dir": str(CAMPAIGN_TASK_DIR),
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "max_factor_degree": args.max_factor_degree,
            "min_support": args.min_support,
            "min_train_transfers": args.min_train_transfers,
            "selection_features": [
                "calibration-preserving factor fingerprints",
                "holdout factorization fingerprints",
            ],
            "forbidden_holdout_selection_fields": [
                "selected_surface_zero_leaves",
                "preserves_selected_root_pairs",
                "same_selected_root_pairs",
                "selected_valid_root_leaves",
                "missing_selected_root_pairs",
                "extra_selected_root_pairs",
            ],
            "cost_model": "selector fingerprint scan + chosen factor evaluation on all public monic leaves + quadratic root checks",
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
