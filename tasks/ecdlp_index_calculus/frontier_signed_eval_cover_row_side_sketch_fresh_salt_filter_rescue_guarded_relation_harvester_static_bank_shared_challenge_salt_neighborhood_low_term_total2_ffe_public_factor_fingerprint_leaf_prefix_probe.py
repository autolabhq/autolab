#!/usr/bin/env python3
"""Public leaf-prefix audit for calibrated FFE factor fingerprints.

The fingerprint transfer probe removes selected-leaf factor choice, but scans
all public leaves for the chosen factor.  This follow-up keeps the same
leaf-agnostic factor selector and calibrates a public leaf ordering/prefix on
training rows only, then audits whether holdout root recovery can avoid the
all-leaf scan cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_transfer_probe as base_probe


DEFAULT_OUT = (
    base_probe.DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_prefix_probe.json"
)

LEAF_POLICIES = (
    "leaf_index",
    "low_b",
    "low_c",
    "low_b_plus_c",
    "low_discriminant",
    "low_centered_b",
    "low_centered_c",
    "leaf_hash",
)
TOP_KS = (1, 2, 4, 8, 16, 32, 64)


def centered(value: int, p: int) -> int:
    value %= p
    return min(value, p - value)


def leaf_hash(coord: dict[str, int], p: int) -> str:
    payload = f"{p}:{coord['leaf_index']}:{coord['b']}:{coord['c']}".encode()
    return hashlib.sha256(payload).hexdigest()


def leaf_sort_key(coord: dict[str, int], p: int, policy: str) -> tuple[Any, ...]:
    b_value = int(coord["b"]) % p
    c_value = int(coord["c"]) % p
    if policy == "leaf_index":
        return (int(coord["leaf_index"]),)
    if policy == "low_b":
        return (b_value, int(coord["leaf_index"]))
    if policy == "low_c":
        return (c_value, int(coord["leaf_index"]))
    if policy == "low_b_plus_c":
        return ((b_value + c_value) % p, int(coord["leaf_index"]))
    if policy == "low_discriminant":
        return (centered(b_value * b_value - 4 * c_value, p), int(coord["leaf_index"]))
    if policy == "low_centered_b":
        return (centered(b_value, p), int(coord["leaf_index"]))
    if policy == "low_centered_c":
        return (centered(c_value, p), int(coord["leaf_index"]))
    if policy == "leaf_hash":
        return (leaf_hash(coord, p),)
    raise ValueError(f"unknown leaf policy: {policy}")


def audit_candidate_prefix(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    candidate_count: int,
    selector_ops: int,
    split_name: str,
    leaf_policy: str,
    top_k: int,
) -> dict[str, Any]:
    p = int(surface_record["p"])
    original_pairs = base_probe.remainder_factor_probe.selected_pair_set(
        surface_record,
        surface_record["surface"],
    )
    all_coords = base_probe.leaf_coords(surface_record)
    ordered = sorted(all_coords, key=lambda coord: leaf_sort_key(coord, p, leaf_policy))
    coords = ordered[: min(top_k, len(ordered))]
    selected_pairs: set[tuple[int, int]] = set()
    zero_leaf_count = 0
    valid_root_leaf_count = 0
    false_zero_leaf_count = 0
    recovered_root_count = 0
    selected_valid_leaf_count = 0
    terms = base_probe.factor_terms(sage_surface, candidate) if candidate else {}
    selected_leaf_indices = {int(leaf) for leaf in surface_record["selected_leaf_indices"]}

    for coord in coords:
        leaf_index = int(coord["leaf_index"])
        if not terms or base_probe.eval_factor(terms, coord["b"], coord["c"], p) != 0:
            continue
        zero_leaf_count += 1
        leaf = surface_record["components"]["leaves"][leaf_index]
        roots = base_probe.slice_quadratic_probe.recover_roots_for_leaf(
            surface_record,
            leaf,
            int(coord["b"]),
            int(coord["c"]),
        )
        if not roots:
            false_zero_leaf_count += 1
            continue
        valid_root_leaf_count += 1
        recovered_root_count += len(roots)
        if leaf_index in selected_leaf_indices:
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
            return {"ops": None, "ops_over_rho": None, "beats_rho": False}
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
        "surface_id": sage_surface.get("surface_id"),
        "target": sage_surface.get("target"),
        "row_key": sage_surface.get("row_key"),
        "challenge_seed": sage_surface.get("challenge_seed"),
        "transfer_index": base_probe.transfer_index(sage_surface),
        "p": p,
        "leaf_policy": leaf_policy,
        "top_k": int(top_k),
        "candidate_count": candidate_count,
        "selector_eval_ops": selector_ops,
        "public_leaf_count": len(all_coords),
        "prefix_leaf_count": len(coords),
        "leaf_eval_ops": leaf_eval_ops,
        "chosen_candidate": base_probe.compact_candidate(sage_surface, candidate),
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
        "public_prefix_selected_hit_ops": selected_charge["ops"],
        "public_prefix_selected_hit_ops_over_rho": selected_charge["ops_over_rho"],
        "public_prefix_selected_hit_beats_rho": selected_charge["beats_rho"],
        "public_prefix_all_hit_ops": all_hit_charge["ops"],
        "public_prefix_all_hit_ops_over_rho": all_hit_charge["ops_over_rho"],
        "public_prefix_all_hit_beats_rho": all_hit_charge["beats_rho"],
        "generic_rho_steps": int(cost["generic_rho_steps"]),
    }


def rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["public_prefix_all_hit_ops_over_rho"])
        for row in rows
        if row.get("public_prefix_all_hit_ops_over_rho") is not None
    ]
    return {
        "surface_count": len(rows),
        "chosen_factor_count": sum(1 for row in rows if row.get("chosen_candidate")),
        "preserving_surface_count": sum(1 for row in rows if row.get("preserves_selected_root_pairs")),
        "same_surface_count": sum(1 for row in rows if row.get("same_selected_root_pairs")),
        "false_positive_surface_count": sum(
            1 for row in rows if row.get("chosen_candidate") and not row.get("preserves_selected_root_pairs")
        ),
        "selected_hit_below_rho_count": sum(1 for row in rows if row.get("public_prefix_selected_hit_beats_rho")),
        "all_hit_below_rho_count": sum(1 for row in rows if row.get("public_prefix_all_hit_beats_rho")),
        "min_all_hit_ops_over_rho": min(ratios) if ratios else None,
        "mean_all_hit_ops_over_rho": round(sum(ratios) / len(ratios), 8) if ratios else None,
        "max_all_hit_ops_over_rho": max(ratios) if ratios else None,
    }


def summary_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary.get("preserving_surface_count") or 0),
        int(summary.get("false_positive_surface_count") or 0),
        -int(summary.get("all_hit_below_rho_count") or 0),
        float(summary.get("mean_all_hit_ops_over_rho") or 10**18),
    )


def evaluate_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    ids: set[str],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    leaf_policy: str,
    top_k: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in ids or surface_id not in surface_records:
            continue
        candidate, candidate_count, selector_ops = base_probe.choose_candidate(
            surface,
            learned,
            int(args.max_factor_degree),
        )
        rows.append(
            audit_candidate_prefix(
                surface,
                surface_records[surface_id],
                candidate,
                candidate_count,
                selector_ops,
                split_name,
                leaf_policy,
                top_k,
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
    learned = base_probe.learned_fingerprints(
        surfaces,
        train_ids,
        int(args.max_factor_degree),
        int(args.min_support),
    )
    calibration = []
    for policy in LEAF_POLICIES:
        for top_k in TOP_KS:
            train_rows = evaluate_rows(
                f"{split_name}:calibration",
                surfaces,
                surface_records,
                train_ids,
                learned,
                policy,
                top_k,
                args,
            )
            calibration.append(
                {
                    "leaf_policy": policy,
                    "top_k": top_k,
                    "summary": rows_summary(train_rows),
                }
            )
    calibration.sort(key=lambda row: (summary_rank(row["summary"]), row["top_k"], row["leaf_policy"]))
    selected = calibration[0] if calibration else {"leaf_policy": "leaf_index", "top_k": 1, "summary": {}}
    test_rows = evaluate_rows(
        split_name,
        surfaces,
        surface_records,
        test_ids,
        learned,
        str(selected["leaf_policy"]),
        int(selected["top_k"]),
        args,
    )
    return {
        "split": split_name,
        "train_surface_count": len(train_ids),
        "test_surface_count": len(test_ids),
        "learned_fingerprint_count": len(learned),
        "selected_leaf_policy": selected["leaf_policy"],
        "selected_top_k": selected["top_k"],
        "selected_training_summary": selected["summary"],
        "test_summary": rows_summary(test_rows),
        "test_rows": test_rows,
        "top_calibration_rows": calibration[:8],
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
    transfers = sorted({base_probe.transfer_index(surface) for surface in surfaces if base_probe.transfer_index(surface) is not None})
    for transfer in transfers:
        test = {str(surface.get("surface_id")) for surface in surfaces if base_probe.transfer_index(surface) == transfer}
        splits.append(split_rows(f"holdout_transfer_{transfer}", surfaces, surface_records, surface_ids - test, test, args))
    for idx in range(int(args.min_train_transfers), len(transfers)):
        train_transfers = set(transfers[:idx])
        holdout = transfers[idx]
        train = {str(surface.get("surface_id")) for surface in surfaces if base_probe.transfer_index(surface) in train_transfers}
        test = {str(surface.get("surface_id")) for surface in surfaces if base_probe.transfer_index(surface) == holdout}
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
        "selected_leaf_policy_counts": dict(
            sorted(
                Counter(
                    f"{split.get('selected_leaf_policy')}@{split.get('selected_top_k')}"
                    for split in splits
                ).items()
            )
        ),
        "best_rows": sorted(
            rows,
            key=lambda row: (
                float(row.get("public_prefix_all_hit_ops_over_rho") or 10**18),
                str(row.get("surface_id")),
            ),
        )[:10],
        "interpretation": (
            "This keeps factor choice leaf-agnostic and calibrates only a public "
            "leaf ordering plus prefix length on training rows. It tests whether "
            "public leaf prefilters can remove the all-leaf scan cost without "
            "reintroducing selected-leaf leakage."
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
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sage_source = base_probe.load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = base_probe.materialize_records(args)
    splits = holdout_splits(surfaces, surface_records, args)
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_fingerprint_leaf_prefix_probe_v1",
        "method": "calibrated_factor_fingerprint_with_public_leaf_prefix",
        "parameters": {
            "campaign_task_dir": str(base_probe.CAMPAIGN_TASK_DIR),
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "max_factor_degree": args.max_factor_degree,
            "min_support": args.min_support,
            "leaf_policies": list(LEAF_POLICIES),
            "top_ks": list(TOP_KS),
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
