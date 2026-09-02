#!/usr/bin/env python3
"""Public leaf-locator audit for reusable Sage factor fingerprints.

The leaf-prefix probe showed that a leaf-agnostic Sage-factor choice still
needs too many public leaf evaluations.  This probe asks a sharper question:
when a preserving factor fingerprint transfers, does calibration also predict a
small set of public leaf indices where that factor should be tested?

The holdout selector may use only calibration-derived fingerprint and leaf-index
tables plus the holdout factorization fingerprint.  It may not use the holdout
selected-leaf zero predicate or preservation labels until after the audit row is
chosen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_transfer_probe as base_probe


DEFAULT_OUT = (
    base_probe.DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_probe.json"
)

LOCATOR_POLICIES = (
    "mode1",
    "mode2",
    "mode4",
    "union",
    "band1",
    "band2",
    "band4",
)


def candidate_zero_indices(
    surface_record: dict[str, Any],
    sage_surface: dict[str, Any],
    candidate: dict[str, Any],
) -> list[int]:
    p = int(surface_record["p"])
    terms = base_probe.factor_terms(sage_surface, candidate)
    if not terms:
        return []
    indices = []
    for coord in base_probe.leaf_coords(surface_record):
        if base_probe.eval_factor(terms, int(coord["b"]), int(coord["c"]), p) == 0:
            indices.append(int(coord["leaf_index"]))
    return sorted(set(indices))


def selected_recovering_zero_indices(
    surface_record: dict[str, Any],
    sage_surface: dict[str, Any],
    candidate: dict[str, Any],
) -> list[int]:
    p = int(surface_record["p"])
    terms = base_probe.factor_terms(sage_surface, candidate)
    selected = {int(leaf) for leaf in surface_record["selected_leaf_indices"]}
    indices = []
    for coord in base_probe.leaf_coords(surface_record):
        leaf_index = int(coord["leaf_index"])
        if leaf_index not in selected:
            continue
        if not terms or base_probe.eval_factor(terms, int(coord["b"]), int(coord["c"]), p) != 0:
            continue
        leaf = surface_record["components"]["leaves"][leaf_index]
        roots = base_probe.slice_quadratic_probe.recover_roots_for_leaf(
            surface_record,
            leaf,
            int(coord["b"]),
            int(coord["c"]),
        )
        if roots:
            indices.append(leaf_index)
    return sorted(set(indices))


def learn_leaf_tables(
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    train_ids: set[str],
    max_degree: int,
    min_support: int,
    require_clean: bool,
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
                },
            )
            row["support"] += 1
            if candidate.get("preserves_selected_root_pairs"):
                row["preserving"] += 1
                for leaf_index in selected_recovering_zero_indices(surface_record, surface, candidate):
                    row["selected_leaf_histogram"][leaf_index] += 1
                for leaf_index in candidate_zero_indices(surface_record, surface, candidate):
                    row["zero_leaf_histogram"][leaf_index] += 1
            else:
                row["false_positive"] += 1
    return {
        key: row
        for key, row in tables.items()
        if int(row["preserving"]) >= min_support
        and int(row["preserving"]) > int(row["false_positive"])
        and (not require_clean or int(row["false_positive"]) == 0)
        and row["selected_leaf_histogram"]
    }


def learned_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["false_positive"]),
        -int(row["preserving"]),
        -int(row["support"]),
        repr(row["fingerprint_key"]),
    )


def predicted_indices(row: dict[str, Any], policy: str, max_leaf_index: int) -> list[int]:
    selected_hist = row["selected_leaf_histogram"]
    zero_hist = row["zero_leaf_histogram"]
    if policy.startswith("mode"):
        limit = int(policy.removeprefix("mode"))
        return sorted(int(leaf) for leaf, _count in selected_hist.most_common(limit))
    if policy == "union":
        return sorted(int(leaf) for leaf in selected_hist)
    if policy.startswith("band"):
        radius = int(policy.removeprefix("band"))
        seeds = {int(leaf) for leaf in selected_hist}
        out = set()
        for leaf in seeds:
            for delta in range(-radius, radius + 1):
                value = leaf + delta
                if 0 <= value <= max_leaf_index:
                    out.add(value)
        return sorted(out)
    raise ValueError(f"unknown locator policy: {policy}")


def choose_candidate(
    surface: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, int]:
    ranked_keys = {
        key: rank
        for rank, key in enumerate(sorted(learned, key=lambda key: learned_rank(learned[key])))
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
            int(candidate.get("factor_total_degree") or 10**9),
            int(candidate.get("factor_monomials") or 10**9),
            base_probe.factor_index(candidate)
            if base_probe.factor_index(candidate) is not None
            else 10**9,
        )
    )
    if not matched:
        return None, None, len(candidates), selector_ops
    candidate = matched[0]
    key = base_probe.fingerprint_key(surface, candidate)
    return candidate, learned.get(key), len(candidates), selector_ops


def audit_locator(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    learned_row: dict[str, Any] | None,
    candidate_count: int,
    selector_ops: int,
    split_name: str,
    policy: str,
) -> dict[str, Any]:
    p = int(surface_record["p"])
    coords_by_index = {int(coord["leaf_index"]): coord for coord in base_probe.leaf_coords(surface_record)}
    max_leaf_index = max(coords_by_index, default=-1)
    indices = predicted_indices(learned_row, policy, max_leaf_index) if learned_row else []
    original_pairs = base_probe.remainder_factor_probe.selected_pair_set(surface_record, surface_record["surface"])
    selected_pairs: set[tuple[int, int]] = set()
    terms = base_probe.factor_terms(sage_surface, candidate) if candidate else {}
    zero_leaf_count = 0
    valid_root_leaf_count = 0
    false_zero_leaf_count = 0
    recovered_root_count = 0
    selected_valid_leaf_count = 0
    selected = {int(leaf) for leaf in surface_record["selected_leaf_indices"]}

    for leaf_index in indices:
        coord = coords_by_index.get(int(leaf_index))
        if coord is None:
            continue
        if not terms or base_probe.eval_factor(terms, int(coord["b"]), int(coord["c"]), p) != 0:
            continue
        zero_leaf_count += 1
        leaf = surface_record["components"]["leaves"][int(leaf_index)]
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
        if int(leaf_index) in selected:
            selected_valid_leaf_count += 1
            for root in roots:
                selected_pairs.add((int(leaf_index), int(root)))

    missing = sorted(original_pairs - selected_pairs)
    extra = sorted(selected_pairs - original_pairs)
    cost = surface_record["cost_inputs"]
    selected_hit_events = int(cost["selected_hit_events"])
    factor_monomials = int((candidate or {}).get("factor_monomials") or 0)
    leaf_eval_ops = len(indices) * factor_monomials
    quadratic_root_work = 2 * zero_leaf_count
    selected_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(len(indices) / 4096)
        + int(cost["selected_hit_roots"])
    )
    all_hit_core = (
        int(surface_record["built"]["selected_signed_pair_count"])
        + ceil(int(surface_record["built"]["row_pool"]) / 512)
        + ceil(len(indices) / 4096)
        + len(surface_record["components"]["hit_roots"])
    )

    def charge(core: int) -> dict[str, Any]:
        if candidate is None or learned_row is None:
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
        "policy": policy,
        "surface_id": sage_surface.get("surface_id"),
        "target": sage_surface.get("target"),
        "row_key": sage_surface.get("row_key"),
        "challenge_seed": sage_surface.get("challenge_seed"),
        "transfer_index": base_probe.transfer_index(sage_surface),
        "p": p,
        "candidate_count": candidate_count,
        "selector_eval_ops": selector_ops,
        "predicted_leaf_count": len(indices),
        "predicted_leaf_indices": indices[:32],
        "leaf_eval_ops": leaf_eval_ops,
        "chosen_candidate": base_probe.compact_candidate(sage_surface, candidate),
        "chosen_preserves_label": bool(candidate and candidate.get("preserves_selected_root_pairs")),
        "learned_support": int((learned_row or {}).get("support") or 0),
        "learned_preserving": int((learned_row or {}).get("preserving") or 0),
        "learned_false_positive": int((learned_row or {}).get("false_positive") or 0),
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
        "public_locator_selected_hit_ops": selected_charge["ops"],
        "public_locator_selected_hit_ops_over_rho": selected_charge["ops_over_rho"],
        "public_locator_selected_hit_beats_rho": selected_charge["beats_rho"],
        "public_locator_all_hit_ops": all_hit_charge["ops"],
        "public_locator_all_hit_ops_over_rho": all_hit_charge["ops_over_rho"],
        "public_locator_all_hit_beats_rho": all_hit_charge["beats_rho"],
        "generic_rho_steps": int(cost["generic_rho_steps"]),
    }


def rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["public_locator_all_hit_ops_over_rho"])
        for row in rows
        if row.get("public_locator_all_hit_ops_over_rho") is not None
    ]
    return {
        "surface_count": len(rows),
        "chosen_factor_count": sum(1 for row in rows if row.get("chosen_candidate")),
        "preserving_surface_count": sum(1 for row in rows if row.get("preserves_selected_root_pairs")),
        "same_surface_count": sum(1 for row in rows if row.get("same_selected_root_pairs")),
        "false_positive_surface_count": sum(
            1 for row in rows if row.get("chosen_candidate") and not row.get("preserves_selected_root_pairs")
        ),
        "selected_hit_below_rho_count": sum(1 for row in rows if row.get("public_locator_selected_hit_beats_rho")),
        "all_hit_below_rho_count": sum(1 for row in rows if row.get("public_locator_all_hit_beats_rho")),
        "min_all_hit_ops_over_rho": min(ratios) if ratios else None,
        "mean_all_hit_ops_over_rho": round(sum(ratios) / len(ratios), 8) if ratios else None,
        "max_all_hit_ops_over_rho": max(ratios) if ratios else None,
        "mean_predicted_leaf_count": round(
            sum(int(row.get("predicted_leaf_count") or 0) for row in rows) / max(1, len(rows)),
            8,
        ),
    }


def summary_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary.get("preserving_surface_count") or 0),
        int(summary.get("false_positive_surface_count") or 0),
        -int(summary.get("all_hit_below_rho_count") or 0),
        float(summary.get("mean_all_hit_ops_over_rho") or 10**18),
        float(summary.get("mean_predicted_leaf_count") or 10**18),
    )


def evaluate_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    ids: set[str],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    policy: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in ids or surface_id not in surface_records:
            continue
        candidate, learned_row, candidate_count, selector_ops = choose_candidate(
            surface,
            learned,
            int(args.max_factor_degree),
        )
        rows.append(
            audit_locator(
                surface,
                surface_records[surface_id],
                candidate,
                learned_row,
                candidate_count,
                selector_ops,
                split_name,
                policy,
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
    learned = learn_leaf_tables(
        surfaces,
        surface_records,
        train_ids,
        int(args.max_factor_degree),
        int(args.min_support),
        bool(args.require_clean_fingerprint),
    )
    calibration = []
    for policy in LOCATOR_POLICIES:
        rows = evaluate_rows(
            f"{split_name}:calibration",
            surfaces,
            surface_records,
            train_ids,
            learned,
            policy,
            args,
        )
        calibration.append({"policy": policy, "summary": rows_summary(rows)})
    calibration.sort(key=lambda row: (summary_rank(row["summary"]), row["policy"]))
    selected = calibration[0] if calibration else {"policy": "mode1", "summary": {}}
    test_rows = evaluate_rows(
        split_name,
        surfaces,
        surface_records,
        test_ids,
        learned,
        str(selected["policy"]),
        args,
    )
    return {
        "split": split_name,
        "train_surface_count": len(train_ids),
        "test_surface_count": len(test_ids),
        "learned_fingerprint_count": len(learned),
        "selected_locator_policy": selected["policy"],
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
        "selected_locator_policy_counts": dict(
            sorted(Counter(str(split.get("selected_locator_policy")) for split in splits).items())
        ),
        "best_rows": sorted(
            rows,
            key=lambda row: (
                float(row.get("public_locator_all_hit_ops_over_rho") or 10**18),
                str(row.get("surface_id")),
            ),
        )[:10],
        "interpretation": (
            "This tests whether calibration-derived factor fingerprints also "
            "predict a small set of public leaf indices. Holdout selection uses "
            "only the factorization fingerprint and learned leaf-index table."
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
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sage_source = base_probe.load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = base_probe.materialize_records(args)
    splits = holdout_splits(surfaces, surface_records, args)
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_probe_v1",
        "method": "calibrated_factor_fingerprint_leaf_index_locator",
        "parameters": {
            "campaign_task_dir": str(base_probe.CAMPAIGN_TASK_DIR),
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "max_factor_degree": args.max_factor_degree,
            "min_support": args.min_support,
            "require_clean_fingerprint": bool(args.require_clean_fingerprint),
            "locator_policies": list(LOCATOR_POLICIES),
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
