#!/usr/bin/env python3
"""Public factor-index guard for reusable FFE factor fingerprints.

The leaf-locator probe found below-rho holdout wins, but it also accepted a few
wrong factors with the right public fingerprint and learned leaf index.  This
probe keeps the same holdout boundary and adds one more public discriminator:
the Sage factor index of preserving calibration factors for each fingerprint.

The guard is deliberately diagnostic.  Factor order is public once the
resultant is factored, but it is a weaker algebraic invariant than the
coefficient fingerprint.  A good result here is evidence for a narrower
candidate window to replace with a more intrinsic selector.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_probe as locator_probe


base_probe = locator_probe.base_probe

DEFAULT_OUT = (
    base_probe.DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_guard_probe.json"
)

GUARD_POLICIES = (
    "none",
    "index_exact",
    "index_mode1",
    "index_mode2",
    "index_range",
    "index_band1",
    "index_band2",
)


def learn_guarded_leaf_tables(
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
                    "preserving_factor_index_histogram": Counter(),
                    "false_factor_index_histogram": Counter(),
                    "all_factor_index_histogram": Counter(),
                },
            )
            factor_index = base_probe.factor_index(candidate)
            row["support"] += 1
            if factor_index is not None:
                row["all_factor_index_histogram"][int(factor_index)] += 1
            if candidate.get("preserves_selected_root_pairs"):
                row["preserving"] += 1
                if factor_index is not None:
                    row["preserving_factor_index_histogram"][int(factor_index)] += 1
                for leaf_index in locator_probe.selected_recovering_zero_indices(surface_record, surface, candidate):
                    row["selected_leaf_histogram"][leaf_index] += 1
                for leaf_index in locator_probe.candidate_zero_indices(surface_record, surface, candidate):
                    row["zero_leaf_histogram"][leaf_index] += 1
            else:
                row["false_positive"] += 1
                if factor_index is not None:
                    row["false_factor_index_histogram"][int(factor_index)] += 1
    return {
        key: row
        for key, row in tables.items()
        if int(row["preserving"]) >= min_support
        and int(row["preserving"]) > int(row["false_positive"])
        and (not require_clean or int(row["false_positive"]) == 0)
        and row["selected_leaf_histogram"]
        and row["preserving_factor_index_histogram"]
    }


def preserving_indices(row: dict[str, Any]) -> list[int]:
    return sorted(int(index) for index in row.get("preserving_factor_index_histogram", {}))


def common_preserving_indices(row: dict[str, Any], limit: int) -> list[int]:
    histogram = row.get("preserving_factor_index_histogram", Counter())
    return sorted(int(index) for index, _count in histogram.most_common(limit))


def guard_accepts(row: dict[str, Any], candidate: dict[str, Any], guard_policy: str) -> bool:
    if guard_policy == "none":
        return True
    factor_index = base_probe.factor_index(candidate)
    if factor_index is None:
        return False
    known = preserving_indices(row)
    if not known:
        return False
    factor_index = int(factor_index)
    if guard_policy == "index_exact":
        return factor_index in known
    if guard_policy == "index_range":
        return min(known) <= factor_index <= max(known)
    if guard_policy.startswith("index_band"):
        radius = int(guard_policy.removeprefix("index_band"))
        return any(abs(factor_index - index) <= radius for index in known)
    if guard_policy.startswith("index_mode"):
        limit = int(guard_policy.removeprefix("index_mode"))
        return factor_index in common_preserving_indices(row, limit)
    raise ValueError(f"unknown guard policy: {guard_policy}")


def choose_guarded_candidate(
    surface: dict[str, Any],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    max_degree: int,
    guard_policy: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, int, int, int]:
    ranked_keys = {
        key: rank
        for rank, key in enumerate(sorted(learned, key=lambda key: locator_probe.learned_rank(learned[key])))
    }
    candidates = base_probe.public_candidates(surface, max_degree)
    selector_ops = sum(int(candidate.get("factor_monomials") or 0) for candidate in candidates)
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected = 0
    matched = 0
    for candidate in candidates:
        key = base_probe.fingerprint_key(surface, candidate)
        if key not in ranked_keys:
            continue
        matched += 1
        learned_row = learned[key]
        if guard_accepts(learned_row, candidate, guard_policy):
            accepted.append((candidate, learned_row))
        else:
            rejected += 1
    accepted.sort(
        key=lambda item: (
            ranked_keys[base_probe.fingerprint_key(surface, item[0])],
            int(item[0].get("factor_total_degree") or 10**9),
            int(item[0].get("factor_monomials") or 10**9),
            base_probe.factor_index(item[0]) if base_probe.factor_index(item[0]) is not None else 10**9,
        )
    )
    if not accepted:
        return None, None, len(candidates), selector_ops, matched, rejected
    candidate, learned_row = accepted[0]
    return candidate, learned_row, len(candidates), selector_ops, matched, rejected


def audit_guarded_locator(
    sage_surface: dict[str, Any],
    surface_record: dict[str, Any],
    candidate: dict[str, Any] | None,
    learned_row: dict[str, Any] | None,
    candidate_count: int,
    selector_ops: int,
    split_name: str,
    locator_policy: str,
    guard_policy: str,
    matched_count: int,
    rejected_count: int,
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
    row["guard_policy"] = guard_policy
    row["guard_matched_candidate_count"] = matched_count
    row["guard_rejected_candidate_count"] = rejected_count
    row["learned_preserving_factor_indices"] = preserving_indices(learned_row or {})
    row["learned_false_factor_indices"] = sorted(
        int(index) for index in (learned_row or {}).get("false_factor_index_histogram", {})
    )
    return row


def evaluate_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    ids: set[str],
    learned: dict[tuple[Any, ...], dict[str, Any]],
    locator_policy: str,
    guard_policy: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for surface in surfaces:
        surface_id = str(surface.get("surface_id"))
        if surface_id not in ids or surface_id not in surface_records:
            continue
        candidate, learned_row, candidate_count, selector_ops, matched_count, rejected_count = choose_guarded_candidate(
            surface,
            learned,
            int(args.max_factor_degree),
            guard_policy,
        )
        rows.append(
            audit_guarded_locator(
                surface,
                surface_records[surface_id],
                candidate,
                learned_row,
                candidate_count,
                selector_ops,
                split_name,
                locator_policy,
                guard_policy,
                matched_count,
                rejected_count,
            )
        )
    return rows


def guarded_summary_rank(summary: dict[str, Any]) -> tuple[Any, ...]:
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


def split_rows(
    split_name: str,
    surfaces: list[dict[str, Any]],
    surface_records: dict[str, dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    learned = learn_guarded_leaf_tables(
        surfaces,
        surface_records,
        train_ids,
        int(args.max_factor_degree),
        int(args.min_support),
        bool(args.require_clean_fingerprint),
    )
    guard_policies = [args.force_guard_policy] if args.force_guard_policy else list(GUARD_POLICIES)
    locator_policies = [args.force_locator_policy] if args.force_locator_policy else list(locator_probe.LOCATOR_POLICIES)
    calibration = []
    for guard_policy in guard_policies:
        for locator_policy in locator_policies:
            rows = evaluate_rows(
                f"{split_name}:calibration",
                surfaces,
                surface_records,
                train_ids,
                learned,
                locator_policy,
                guard_policy,
                args,
            )
            calibration.append(
                {
                    "guard_policy": guard_policy,
                    "locator_policy": locator_policy,
                    "summary": locator_probe.rows_summary(rows),
                }
            )
    calibration.sort(
        key=lambda row: (
            guarded_summary_rank(row["summary"]),
            row["guard_policy"],
            row["locator_policy"],
        )
    )
    selected = calibration[0] if calibration else {
        "guard_policy": "none",
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
        str(selected["guard_policy"]),
        args,
    )
    return {
        "split": split_name,
        "train_surface_count": len(train_ids),
        "test_surface_count": len(test_ids),
        "learned_fingerprint_count": len(learned),
        "selected_locator_policy": selected["locator_policy"],
        "selected_guard_policy": selected["guard_policy"],
        "selected_training_summary": selected["summary"],
        "test_summary": locator_probe.rows_summary(test_rows),
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
        "aggregate_summary": locator_probe.rows_summary(rows),
        "split_family_summaries": {key: locator_probe.rows_summary(value) for key, value in sorted(grouped.items())},
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
                    f"{split.get('selected_guard_policy')}+{split.get('selected_locator_policy')}"
                    for split in splits
                ).items()
            )
        ),
        "guard_policy_counts": dict(sorted(Counter(str(split.get("selected_guard_policy")) for split in splits).items())),
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
            "This is a guarded variant of the public factor-fingerprint leaf locator. "
            "Holdout selection uses only the factorization fingerprint, the public "
            "candidate factor index, and calibration-derived leaf/index tables."
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
    parser.add_argument("--force-guard-policy", choices=GUARD_POLICIES)
    parser.add_argument("--force-locator-policy", choices=locator_probe.LOCATOR_POLICIES)
    parser.add_argument("--min-train-transfers", type=int, default=3)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sage_source = base_probe.load_json(args.sage_factor_source)
    surfaces = [surface for surface in sage_source.get("surfaces") or [] if isinstance(surface, dict)]
    surface_records, case_results = base_probe.materialize_records(args)
    splits = holdout_splits(surfaces, surface_records, args)
    output = {
        "schema": "ecdlp_low_term_total2_ffe_public_factor_fingerprint_leaf_locator_guard_probe_v1",
        "method": "calibrated_factor_fingerprint_leaf_index_locator_with_public_factor_index_guard",
        "parameters": {
            "campaign_task_dir": str(base_probe.CAMPAIGN_TASK_DIR),
            "sage_factor_source": str(args.sage_factor_source),
            "signature_source": str(args.signature_source),
            "max_factor_degree": args.max_factor_degree,
            "min_support": args.min_support,
            "require_clean_fingerprint": bool(args.require_clean_fingerprint),
            "guard_policies": list(GUARD_POLICIES),
            "locator_policies": list(locator_probe.LOCATOR_POLICIES),
            "force_guard_policy": args.force_guard_policy,
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
