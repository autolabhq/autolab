#!/usr/bin/env python3
"""Mine repeated public monic-coordinate gates before exact FFE factorization.

The input is a public selector artifact whose cases already choose rows and
leaf indices without verifier labels.  This probe materializes those selected
leaves, records their monic ``(b,c)`` coordinates, ranks coordinates that repeat
across rows, and can replay the top exact-coordinate and axis gates.

This is meant to separate a reusable public coordinate selector from a
post-hoc Sage factor such as ``s*b + c + t``.  Verifier fields from the source
case are copied only as labels; they are not used for ranking.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_linear_factor_gate_replay_probe as gate_probe
import relation_probe

import ffe_single_hit_root_relation_replay_probe as replay_probe


DEFAULT_STATE_DIR = gate_probe.DEFAULT_STATE_DIR
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_gate_miner.json"


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def parse_coord(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinate must be b,c")
    return int(parts[0]), int(parts[1])


def case_selected(case: dict[str, Any], args: argparse.Namespace) -> bool:
    return gate_probe.case_selected(case, args)


def public_case_key(case: dict[str, Any]) -> str:
    return replay_probe.case_key_string(case)


def source_cases(signature: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = [
        case
        for case in signature.get("positive_cases") or []
        if isinstance(case, dict) and case_selected(case, args)
    ]
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    return cases


def profile_rows_for_case(
    case: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profiles: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        context = contexts.get(row_key)
        if not context:
            errors.append(
                {
                    "row_key": row_key,
                    "surface_id": item.get("surface_id"),
                    "error": "context not materialized",
                }
            )
            continue
        built = context["built"]
        p = int(built["p"])
        components = context["components"]
        for leaf_index in sorted({int(leaf) for leaf in item.get("leaf_indices") or []}):
            if leaf_index < 0 or leaf_index >= len(components["leaves"]):
                errors.append(
                    {
                        "row_key": row_key,
                        "surface_id": item.get("surface_id"),
                        "leaf_index": leaf_index,
                        "error": "leaf index out of range",
                    }
                )
                continue
            monic = gate_probe.resultant_surface_probe.monic_coeffs(
                components["leaves"][leaf_index],
                p,
            )
            if monic is None:
                errors.append(
                    {
                        "row_key": row_key,
                        "surface_id": item.get("surface_id"),
                        "leaf_index": leaf_index,
                        "error": "monic coefficients unavailable",
                    }
                )
                continue
            profiles.append(
                {
                    "target": case.get("target"),
                    "transfer_index": int(case.get("transfer_index") or 0),
                    "top_k": int(case.get("top_k") or 0),
                    "policy": case.get("policy"),
                    "leaf_selector": case.get("leaf_selector") or case.get("selector"),
                    "row_key": row_key,
                    "surface_id": item.get("surface_id"),
                    "salt": item.get("salt"),
                    "leaf_index": leaf_index,
                    "p": p,
                    "b": int(monic[0]),
                    "c": int(monic[1]),
                }
            )
    return profiles, errors


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_row_count": replay.get("selected_row_count"),
        "selected_leaf_count": replay.get("selected_leaf_count"),
        "relation_count": replay.get("relation_count"),
        "rank": replay.get("rank"),
        "public_key_verified": bool(replay.get("public_key_verified")),
        "derived_secret": replay.get("derived_secret"),
        "ops_over_rho": replay.get("ops_over_rho"),
        "below_rho": bool(replay.get("below_rho")),
    }


def replay_profiles(
    verifier: Any,
    profiles: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
    event_summary_limit: int,
) -> dict[str, Any]:
    row_leaves: dict[str, set[int]] = defaultdict(set)
    for profile in profiles:
        row_leaves[str(profile["row_key"])].add(int(profile["leaf_index"]))
    replay, _events = replay_probe.replay_selection(
        verifier,
        dict(row_leaves),
        contexts,
        scan_cache,
        event_summary_limit,
    )
    return compact_replay(replay)


def candidate_record(
    case: dict[str, Any],
    case_index: int,
    coord_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    first = coord_profiles[0]
    source_ops = round_or_none(case.get("ops_over_rho"))
    row_keys = sorted({str(profile["row_key"]) for profile in coord_profiles})
    salts = sorted(
        {
            int(profile["salt"])
            for profile in coord_profiles
            if profile.get("salt") is not None
        }
    )
    leaf_indices = sorted({int(profile["leaf_index"]) for profile in coord_profiles})
    return {
        "case_index": case_index,
        "case_key": public_case_key(case),
        "target": first.get("target"),
        "transfer_index": int(first["transfer_index"]),
        "top_k": int(first["top_k"]),
        "policy": first.get("policy"),
        "leaf_selector": first.get("leaf_selector"),
        "p": int(first["p"]),
        "b": int(first["b"]),
        "c": int(first["c"]),
        "coordinate": {"b": int(first["b"]), "c": int(first["c"])},
        "row_count": len(row_keys),
        "salt_count": len(salts),
        "profile_count": len(coord_profiles),
        "leaf_indices": leaf_indices,
        "row_keys": row_keys,
        "salts": salts,
        "source_ops_over_rho": source_ops,
        "source_below_rho": bool(case.get("below_rho")),
        "source_public_key_verified": bool(case.get("public_key_verified")),
        "source_rank": int(case.get("rank") or 0),
        "source_relation_count": int(case.get("relation_count") or 0),
        "profiles": coord_profiles,
    }


def candidate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    source_ops = record.get("source_ops_over_rho")
    return (
        -int(record.get("row_count") or 0),
        -int(record.get("profile_count") or 0),
        float(source_ops if source_ops is not None else 10**9),
        str(record.get("target") or ""),
        int(record.get("transfer_index") or 0),
        int(record.get("top_k") or 0),
        str(record.get("policy") or ""),
        str(record.get("leaf_selector") or ""),
        int(record.get("b") or 0),
        int(record.get("c") or 0),
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    replayed = [record for record in records if record.get("exact_coordinate_replay")]
    verified = [
        record
        for record in replayed
        if bool((record.get("exact_coordinate_replay") or {}).get("public_key_verified"))
    ]
    source_verified = [record for record in records if bool(record.get("source_public_key_verified"))]
    replay_ratios = [
        float((record.get("exact_coordinate_replay") or {}).get("ops_over_rho"))
        for record in replayed
        if (record.get("exact_coordinate_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "candidate_count": len(records),
        "source_verified_label_count": len(source_verified),
        "replayed_candidate_count": len(replayed),
        "replayed_public_key_verified_count": len(verified),
        "replayed_verified_below_rho_count": sum(
            1
            for record in verified
            if bool((record.get("exact_coordinate_replay") or {}).get("below_rho"))
        ),
        "min_replayed_ops_over_rho": round(min(replay_ratios), 8) if replay_ratios else None,
        "mean_replayed_ops_over_rho": mean_or_none(replay_ratios),
        "best_replayed_verified_candidates": [
            {
                key: record.get(key)
                for key in (
                    "target",
                    "transfer_index",
                    "top_k",
                    "policy",
                    "leaf_selector",
                    "coordinate",
                    "row_count",
                    "salt_count",
                    "profile_count",
                    "source_ops_over_rho",
                    "exact_coordinate_replay",
                    "b_axis_replay",
                    "c_axis_replay",
                )
            }
            for record in sorted(
                verified,
                key=lambda row: (
                    float((row.get("exact_coordinate_replay") or {}).get("ops_over_rho") or 10**9),
                    candidate_sort_key(row),
                ),
            )[:8]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=gate_probe.DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=gate_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=gate_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=gate_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=gate_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--target")
    parser.add_argument("--transfer-index", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--leaf-selector")
    parser.add_argument("--radius", type=int)
    parser.add_argument("--min-row-count", type=int, default=2)
    parser.add_argument("--min-profile-count", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--replay-top", type=int, default=0)
    parser.add_argument("--include-axis-replay", action="store_true")
    parser.add_argument("--highlight-coordinate", type=parse_coord, action="append", default=[])
    parser.add_argument("--event-summary-limit", type=int, default=8)
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = gate_probe.load_json(args.signature_source)
    cases = source_cases(signature, args)
    bank_source = gate_probe.load_json(args.bank_source)
    config_source = gate_probe.load_json(args.config_source)
    direct_source = gate_probe.load_json(args.direct_source)
    transfer_source = gate_probe.load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()

    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    case_contexts: dict[int, dict[str, dict[str, Any]]] = {}
    case_profiles: dict[int, list[dict[str, Any]]] = {}
    context_errors: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases):
        source_leaves, _source_profiles = replay_probe.row_leaf_groups(case, None)
        contexts, errors = replay_probe.materialize_contexts(
            verifier,
            verifier_records,
            config_source,
            specs_by_target,
            case,
            sorted(source_leaves),
            args,
            context_cache,
        )
        context_errors.extend(errors)
        profiles, profile_errors = profile_rows_for_case(case, contexts)
        context_errors.extend(profile_errors)
        case_contexts[case_index] = contexts
        case_profiles[case_index] = profiles
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for profile in profiles:
            grouped[(int(profile["p"]), int(profile["b"]), int(profile["c"]))].append(profile)
        for (_p, _b, _c), coord_profiles in grouped.items():
            row_count = len({str(profile["row_key"]) for profile in coord_profiles})
            if row_count < args.min_row_count or len(coord_profiles) < args.min_profile_count:
                continue
            candidates.append(candidate_record(case, case_index, coord_profiles))

    candidates = sorted(candidates, key=candidate_sort_key)
    highlights = {(int(b), int(c)) for b, c in args.highlight_coordinate}
    if args.max_candidates > 0:
        top = candidates[: args.max_candidates]
        highlighted = [
            record
            for record in candidates
            if (int(record["b"]), int(record["c"])) in highlights and record not in top
        ]
        candidates = top + highlighted

    replay_limit = max(0, int(args.replay_top))
    replay_indices = set(range(min(replay_limit, len(candidates))))
    replay_indices.update(
        index
        for index, record in enumerate(candidates)
        if (int(record["b"]), int(record["c"])) in highlights
    )
    for index in sorted(replay_indices):
        record = candidates[index]
        case_index = int(record["case_index"])
        contexts = case_contexts.get(case_index, {})
        profiles = case_profiles.get(case_index, [])
        b_value = int(record["b"])
        c_value = int(record["c"])
        exact_profiles = [
            profile
            for profile in profiles
            if int(profile["b"]) == b_value and int(profile["c"]) == c_value
        ]
        record["exact_coordinate_replay"] = replay_profiles(
            verifier,
            exact_profiles,
            contexts,
            scan_cache,
            int(args.event_summary_limit),
        )
        if args.include_axis_replay:
            b_profiles = [profile for profile in profiles if int(profile["b"]) == b_value]
            c_profiles = [profile for profile in profiles if int(profile["c"]) == c_value]
            record["b_axis_replay"] = replay_profiles(
                verifier,
                b_profiles,
                contexts,
                scan_cache,
                int(args.event_summary_limit),
            )
            record["c_axis_replay"] = replay_profiles(
                verifier,
                c_profiles,
                contexts,
                scan_cache,
                int(args.event_summary_limit),
            )

    for record in candidates:
        record.pop("case_index", None)

    output = {
        "schema": "ecdlp_public_repeated_coordinate_gate_miner_v1",
        "method": "public_selected_leaf_monic_coordinate_repeat_mining",
        "parameters": {
            "signature_source": str(args.signature_source),
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "target": args.target,
            "transfer_index": args.transfer_index,
            "top_k": args.top_k,
            "policy": args.policy,
            "leaf_selector": args.leaf_selector,
            "radius": radius,
            "min_row_count": args.min_row_count,
            "min_profile_count": args.min_profile_count,
            "max_cases": args.max_cases,
            "max_candidates": args.max_candidates,
            "replay_top": args.replay_top,
            "include_axis_replay": bool(args.include_axis_replay),
            "highlight_coordinates": [
                {"b": int(b), "c": int(c)} for b, c in args.highlight_coordinate
            ],
            "seed": args.seed,
        },
        "summary": summarize(candidates),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "candidates": candidates,
        "non_claims": [
            "Coordinate repetition is a public pre-exact gate, but replay must still derive the secret below rho before it is an end-to-end speedup.",
            "Source verifier labels are copied only for audit and are not used by candidate ranking.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
