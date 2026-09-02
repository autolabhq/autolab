#!/usr/bin/env sage --python
"""Sage-factor exact row/leaf profiles from a signature source.

The row-level Sage subset wrapper filters by ``surface_id`` after the shared
materializer has already de-duplicated row surfaces.  That is unsafe when the
same row surface appears with multiple selected leaf profiles: the first
profile wins, and a later exact profile can be silently evaluated under the
wrong selected leaves.

This wrapper accepts explicit row/leaf profile specs, clones only those profiles
from the source signature cases, materializes one profile at a time, and then
delegates the actual finite-field factorization to the live Sage evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe as sage_factor_probe
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total3_total4_public_bounded_full_selector_208_215.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_sage_factor_exact_profile_subset.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def leaf_signature(leaves: list[int]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def parse_profile(raw: str) -> dict[str, Any]:
    parts = raw.split("|", 6)
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "profile must be target|transfer_index|top_k|policy|leaf_selector|row_key|leaf1,leaf2"
        )
    leaves: list[int] = []
    for item in parts[6].split(","):
        item = item.strip()
        if item:
            leaves.append(int(item))
    if not leaves:
        raise argparse.ArgumentTypeError("profile leaf list cannot be empty")
    return {
        "target": parts[0],
        "transfer_index": int(parts[1]),
        "top_k": int(parts[2]),
        "policy": parts[3],
        "leaf_selector": parts[4],
        "row_key": parts[5],
        "leaf_indices": sorted(set(leaves)),
    }


def parse_signature_profile_selector(raw: str) -> dict[str, Any]:
    parts = raw.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "profile selector must be target|transfer_index|row_key"
        )
    return {
        "target": parts[0],
        "transfer_index": int(parts[1]),
        "row_key": parts[2],
    }


def profiles_from_signature_selector(
    signature: dict[str, Any],
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for case in signature.get("positive_cases") or []:
        if not isinstance(case, dict):
            continue
        if str(case.get("target")) != selector["target"]:
            continue
        if int(case.get("transfer_index") or 0) != int(selector["transfer_index"]):
            continue
        for item in case.get("row_leaf_keys") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("row_key") or "") != selector["row_key"]:
                continue
            leaves = sorted({int(leaf) for leaf in item.get("leaf_indices") or []})
            if not leaves:
                continue
            profile = {
                "target": selector["target"],
                "transfer_index": int(selector["transfer_index"]),
                "top_k": int(case.get("top_k") or 0),
                "policy": str(case.get("policy")),
                "leaf_selector": str(case.get("leaf_selector") or case.get("selector")),
                "row_key": selector["row_key"],
                "leaf_indices": leaves,
            }
            key = (
                profile["target"],
                profile["transfer_index"],
                profile["top_k"],
                profile["policy"],
                profile["leaf_selector"],
                profile["row_key"],
                tuple(profile["leaf_indices"]),
            )
            if key in seen:
                continue
            seen.add(key)
            profiles.append(profile)
    return profiles


def case_matches(case: dict[str, Any], spec: dict[str, Any]) -> bool:
    return (
        str(case.get("target")) == spec["target"]
        and int(case.get("transfer_index") or 0) == int(spec["transfer_index"])
        and int(case.get("top_k") or 0) == int(spec["top_k"])
        and str(case.get("policy")) == spec["policy"]
        and str(case.get("leaf_selector") or case.get("selector")) == spec["leaf_selector"]
    )


def matching_row_item(case: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    expected_leaves = list(spec["leaf_indices"])
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        leaves = sorted({int(leaf) for leaf in item.get("leaf_indices") or []})
        if row_key == spec["row_key"] and leaves == expected_leaves:
            return dict(item)
    return None


def compact_case_key(case: dict[str, Any]) -> str:
    row_leaf_key = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return "|".join(
        str(part)
        for part in (
            case.get("target"),
            int(case.get("transfer_index") or 0),
            int(case.get("top_k") or 0),
            case.get("policy"),
            case.get("leaf_selector") or case.get("selector"),
            row_leaf_key,
        )
    )


def clone_profile_case(
    case: dict[str, Any],
    item: dict[str, Any],
    spec: dict[str, Any],
    profile_index: int,
) -> dict[str, Any]:
    leaves = list(spec["leaf_indices"])
    surface_id = str(item.get("surface_id") or "")
    cloned_item = {
        **item,
        "row_key": spec["row_key"],
        "leaf_indices": leaves,
    }
    cloned = {
        **case,
        "row_leaf_keys": [cloned_item],
        "surface_ids": [surface_id] if surface_id else [],
        "leaf_indices": leaves,
        "unique_leaf_indices": leaves,
        "selected_leaf_count": len(leaves),
        "selected_row_count": 1,
        "row_salts": [item.get("salt")] if item.get("salt") is not None else [],
        "exact_profile_source_case_key": compact_case_key(case),
        "exact_profile_index": profile_index,
        "exact_profile": {
            **spec,
            "surface_id": surface_id or None,
            "selected_signature": leaf_signature(leaves),
        },
    }
    return cloned


def exact_profile_cases(
    signature: dict[str, Any],
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_cases = [case for case in signature.get("positive_cases") or [] if isinstance(case, dict)]
    profile_cases: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        matches: list[dict[str, Any]] = []
        for case in positive_cases:
            if not case_matches(case, spec):
                continue
            item = matching_row_item(case, spec)
            if item is not None:
                matches.append(clone_profile_case(case, item, spec, index))
        if not matches:
            missing.append(spec)
            continue
        profile_cases.append(matches[0])
    return profile_cases, missing


def build_specs_by_target(args: argparse.Namespace, radius: int) -> dict[str, dict[str, dict[str, Any]]]:
    bank = load_json(args.bank_source)
    direct_source = load_json(args.direct_source)
    bank_rows = {
        sage_factor_probe.cross_surface_probe.compress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict) and sage_factor_probe.cross_surface_probe.compress_probe.row_key(row)
    }
    return sage_factor_probe.cross_surface_probe.leaf_trim_probe.specs_by_target_and_key(
        sage_factor_probe.cross_surface_probe.salt_neighborhood_probe.witness_specs(
            direct_source,
            bank_rows,
            radius,
        )
    )


def replay_materialize_profile_record(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    profile_case: dict[str, Any],
    args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    exact_profile = profile_case.get("exact_profile") or {}
    row_key = str(exact_profile.get("row_key") or "")
    leaf_indices = [int(leaf) for leaf in exact_profile.get("leaf_indices") or []]
    source_leaves, _source_profiles = replay_probe.row_leaf_groups(profile_case, None)
    contexts, errors = replay_probe.materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        profile_case,
        sorted(source_leaves),
        args,
        context_cache,
    )
    case_result = {
        "target": profile_case.get("target"),
        "transfer_index": int(profile_case.get("transfer_index") or 0),
        "top_k": int(profile_case.get("top_k") or 0),
        "policy": profile_case.get("policy"),
        "row_selector": profile_case.get("row_selector"),
        "leaf_selector": profile_case.get("leaf_selector") or profile_case.get("selector"),
        "source_ops_over_rho": profile_case.get("ops_over_rho"),
        "surface_ids": [],
        "public_key_verified": False,
        "union_rank": 0,
        "union_relation_count": 0,
        "reconstructed_errors": errors,
        "fallback_materializer": "single_hit_root_replay",
    }
    if row_key not in contexts:
        case_result["reconstructed_errors"] = [
            *errors,
            {
                "row_key": row_key,
                "error": "row_key_missing_from_replay_context",
                "available_row_keys": sorted(contexts),
            },
        ]
        return None, case_result
    context = contexts[row_key]
    built = context["built"]
    components = context["components"]
    local_args = context["local_args"]
    selected = set(leaf_indices)
    direct_scan = sage_factor_probe.cross_surface_probe.direct_witness_probe.scan_selected(
        verifier,
        built,
        components,
        selected,
        local_args,
    )
    surface = sage_factor_probe.cross_surface_probe.resultant_surface_probe.hit_remainder_surface(
        components["hit_poly"],
        int(built["p"]),
    )
    surface_id = f"{built['target']}|{row_key}|{built.get('challenge_seed')}"
    remainder_a_stats = sage_factor_probe.cross_surface_probe.resultant_surface_probe.poly2_stats(
        surface["remainder_a"]
    )
    remainder_b_stats = sage_factor_probe.cross_surface_probe.resultant_surface_probe.poly2_stats(
        surface["remainder_b"]
    )
    remainder_monomials = int(remainder_a_stats["monomials"]) + int(remainder_b_stats["monomials"])
    full_costs = sage_factor_probe.cross_surface_probe.hit_stream_probe.surface_costs(
        built,
        len(selected),
        int(direct_scan.get("selected_hit_roots") or 0),
        int(direct_scan.get("selected_hit_events") or 0),
        int(sage_factor_probe.cross_surface_probe.resultant_surface_probe.poly2_stats(surface["resultant"])["monomials"]),
        remainder_monomials,
        len(selected),
        int(local_args.row_factor),
        int(local_args.product_factor),
    )
    union = sage_factor_probe.cross_surface_probe.direct_witness_probe.union_derive(
        verifier,
        [{"row_key": row_key, "_direct": direct_scan}],
        {row_key: built},
        "_direct",
    )
    case_result.update(
        {
            "surface_ids": [surface_id],
            "public_key_verified": bool(union.get("union_public_key_verified")),
            "union_rank": int(union.get("union_rank") or 0),
            "union_relation_count": int(union.get("union_relation_count") or 0),
        }
    )
    record = {
        "surface_id": surface_id,
        "target": built["target"],
        "row_key": row_key,
        "row_schedule_key": sage_factor_probe.cross_surface_probe.row_schedule_key(row_key),
        "challenge_seed": built.get("challenge_seed"),
        "selected_leaf_indices": sorted(selected),
        "selected_signature": sage_factor_probe.cross_surface_probe.selected_signature(leaf_indices),
        "p": int(built["p"]),
        "built": built,
        "components": components,
        "surface": surface,
        "cost_inputs": sage_factor_probe.cross_surface_probe.surface_cost_inputs(
            built,
            selected,
            direct_scan,
            local_args,
        ),
        "full_remainder_monomials": remainder_monomials,
        "full_remainder_ffe_ops_over_rho": full_costs["remainder_ffe_ops_over_rho"],
        "source_cases": [
            {
                "target": profile_case.get("target"),
                "transfer_index": int(profile_case.get("transfer_index") or 0),
                "top_k": int(profile_case.get("top_k") or 0),
                "policy": profile_case.get("policy"),
                "row_selector": profile_case.get("row_selector"),
                "leaf_selector": profile_case.get("leaf_selector") or profile_case.get("selector"),
                "source_ops_over_rho": profile_case.get("ops_over_rho"),
                "fallback_materializer": "single_hit_root_replay",
            }
        ],
    }
    return annotate_record(record, profile_case), case_result


def annotate_record(record: dict[str, Any], profile_case: dict[str, Any]) -> dict[str, Any]:
    exact_profile = dict(profile_case.get("exact_profile") or {})
    leaves = [int(leaf) for leaf in exact_profile.get("leaf_indices") or []]
    surface_id = str(record.get("surface_id") or "")
    profile_id = f"{surface_id}#leaves={leaf_signature(leaves)}"
    record["surface_profile_id"] = profile_id
    record["exact_profile"] = exact_profile
    record["exact_profile_source_case_key"] = profile_case.get("exact_profile_source_case_key")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--profile", action="append", type=parse_profile, default=[])
    parser.add_argument(
        "--profile-from-signature",
        action="append",
        type=parse_signature_profile_selector,
        default=[],
        help="Add all exact row/leaf profiles matching target|transfer_index|row_key in the signature source.",
    )
    parser.add_argument("--state-dir", type=Path, default=sage_factor_probe.DEFAULT_STATE_DIR)
    parser.add_argument("--bank-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
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
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument(
        "--allow-materialization-errors",
        action="store_true",
        help="Write a diagnostic artifact instead of aborting when exact profiles cannot be materialized.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = load_json(args.signature_source)
    profile_specs = list(args.profile)
    for selector in args.profile_from_signature:
        profile_specs.extend(profiles_from_signature_selector(signature, selector))
    if not profile_specs:
        raise SystemExit("no profiles requested")

    profile_cases, missing_profiles = exact_profile_cases(signature, profile_specs)
    if missing_profiles:
        raise SystemExit(f"missing exact profiles: {json.dumps(missing_profiles, sort_keys=True)}")

    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    config_source = load_json(args.config_source)
    specs_by_target = build_specs_by_target(args, radius)
    verifier = sage_factor_probe.cross_surface_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()

    surface_records: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    materialization_errors: list[dict[str, Any]] = []
    replay_context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for profile_case in profile_cases:
        records_for_profile, cases_for_profile = sage_factor_probe.cross_surface_probe.materialize_surface_records(
            verifier,
            records,
            config_source,
            specs_by_target,
            [profile_case],
            args,
        )
        expected_profile = profile_case.get("exact_profile") or {}
        expected_row = str(expected_profile.get("row_key") or "")
        expected_leaves = [int(leaf) for leaf in expected_profile.get("leaf_indices") or []]
        matched_records = [
            record
            for record in records_for_profile
            if str(record.get("row_key") or "") == expected_row
            and [int(leaf) for leaf in record.get("selected_leaf_indices") or []] == expected_leaves
        ]
        if not matched_records:
            fallback_record, fallback_case_result = replay_materialize_profile_record(
                verifier,
                records,
                config_source,
                specs_by_target,
                profile_case,
                args,
                replay_context_cache,
            )
            case_results.append(
                {
                    **fallback_case_result,
                    "exact_profile": expected_profile,
                    "exact_profile_source_case_key": profile_case.get("exact_profile_source_case_key"),
                }
            )
            if fallback_record is not None:
                surface_records.append(fallback_record)
                continue
            materialization_errors.append(
                {
                    "exact_profile": expected_profile,
                    "materialized_surface_count": len(records_for_profile),
                    "materialized_records": [
                        {
                            "surface_id": record.get("surface_id"),
                            "row_key": record.get("row_key"),
                            "selected_leaf_indices": record.get("selected_leaf_indices"),
                        }
                        for record in records_for_profile
                    ],
                    "fallback_errors": fallback_case_result.get("reconstructed_errors"),
                },
            )
        for record in matched_records:
            surface_records.append(annotate_record(record, profile_case))
        for case_result in cases_for_profile:
            case_results.append(
                {
                    **case_result,
                    "exact_profile": expected_profile,
                    "exact_profile_source_case_key": profile_case.get("exact_profile_source_case_key"),
                }
            )
    if materialization_errors and not args.allow_materialization_errors:
        raise SystemExit(f"exact profile materialization mismatch: {json.dumps(materialization_errors, sort_keys=True)}")

    surfaces = []
    for surface_record in surface_records:
        evaluated = sage_factor_probe.evaluate_surface(surface_record)
        evaluated["surface_profile_id"] = surface_record.get("surface_profile_id")
        evaluated["exact_profile"] = surface_record.get("exact_profile")
        evaluated["exact_profile_source_case_key"] = surface_record.get("exact_profile_source_case_key")
        evaluated["cost_inputs"] = surface_record.get("cost_inputs")
        evaluated["source_cases"] = surface_record.get("source_cases") or []
        surfaces.append(evaluated)

    output = {
        "schema": "ecdlp_ffe_sage_factor_exact_profile_subset_probe_v1",
        "method": "sage_backed_factorization_for_exact_row_leaf_profiles",
        "parameters": {
            "signature_source": str(args.signature_source),
            "profiles": profile_specs,
            "profile_from_signature": args.profile_from_signature,
            "radius": radius,
            "seed": args.seed,
        },
        "summary": {
            **sage_factor_probe.summarize(surfaces, case_results),
            "requested_profile_count": len(profile_specs),
            "materialized_exact_profile_count": len(surface_records),
            "missing_exact_profile_count": len(missing_profiles),
            "materialization_error_count": len(materialization_errors),
            "surface_profile_ids": [surface.get("surface_profile_id") for surface in surfaces],
        },
        "surfaces": surfaces,
        "cases": case_results,
        "materialization_errors": materialization_errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
