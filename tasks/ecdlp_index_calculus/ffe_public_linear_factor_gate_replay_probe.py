#!/usr/bin/env python3
"""Replay a public linear-factor gate over signature row/leaf cases.

Given a public linear factor in the monic leaf coefficients ``(b,c)``, this
probe filters each compact signature case to leaves satisfying

    b_coeff * b + c_coeff * c + constant == 0 (mod p)

and reruns the verifier-facing relation replay on the filtered leaves.  It is
designed for follow-up on exact-profile Sage factors: the factor is public, but
the replay still has to prove that the resulting relation system derives below
rho without using verifier labels in the selector.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

WORKTREE_TASK_DIR = Path(__file__).resolve().parent
if str(WORKTREE_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(WORKTREE_TASK_DIR))

import frontier_signed_eval_cover_resultant_surface_probe as resultant_surface_probe
import relation_probe

import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total3_total4_public_bounded_full_selector_208_215.json"
DEFAULT_LIVE_STATE_DIR = Path(
    os.environ.get("ECDLP_LIVE_STATE_DIR", "/Volumes/Volume/autolab/ecdlp_index_calculus_state")
)
DEFAULT_BANK_SOURCE = DEFAULT_LIVE_STATE_DIR / replay_probe.leaf_trim_probe.DEFAULT_BANK_SOURCE.name
DEFAULT_CONFIG_SOURCE = DEFAULT_LIVE_STATE_DIR / replay_probe.leaf_trim_probe.DEFAULT_CONFIG_SOURCE.name
DEFAULT_DIRECT_SOURCE = DEFAULT_LIVE_STATE_DIR / replay_probe.leaf_trim_probe.DEFAULT_DIRECT_SOURCE.name
DEFAULT_TRANSFER_SOURCE = DEFAULT_LIVE_STATE_DIR / replay_probe.leaf_trim_probe.DEFAULT_TRANSFER_SOURCE.name
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_linear_factor_gate_replay.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def parse_factor(raw: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("factor must be b_coeff,c_coeff,constant")
    return int(parts[0]), int(parts[1]), int(parts[2])


def case_selected(case: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.target and str(case.get("target")) != args.target:
        return False
    if args.transfer_index is not None and int(case.get("transfer_index") or 0) != args.transfer_index:
        return False
    if args.policy and str(case.get("policy")) != args.policy:
        return False
    if args.leaf_selector and str(case.get("leaf_selector") or case.get("selector")) != args.leaf_selector:
        return False
    if args.top_k is not None and int(case.get("top_k") or 0) != args.top_k:
        return False
    return True


def factor_value(leaf: dict[str, Any], p: int, coeffs: tuple[int, int, int]) -> tuple[bool, dict[str, Any]]:
    monic = resultant_surface_probe.monic_coeffs(leaf, p)
    if monic is None:
        return False, {"monic_coefficients": None, "factor_value": None}
    b_value, c_value = (int(monic[0]), int(monic[1]))
    b_coeff, c_coeff, constant = coeffs
    value = (int(b_coeff) * b_value + int(c_coeff) * c_value + int(constant)) % int(p)
    return value == 0, {
        "monic_coefficients": {"b": b_value, "c": c_value},
        "factor_value": int(value),
    }


def gated_row_leaves(
    case: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    coeffs: tuple[int, int, int],
    factor_p: int | None,
) -> tuple[dict[str, set[int]], list[dict[str, Any]], list[dict[str, Any]]]:
    row_leaves: dict[str, set[int]] = {}
    kept_profiles: list[dict[str, Any]] = []
    rejected_profiles: list[dict[str, Any]] = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        context = contexts.get(row_key)
        if not context:
            continue
        built = context["built"]
        p = int(built["p"])
        if factor_p is not None and p != factor_p:
            continue
        components = context["components"]
        for leaf_index in sorted({int(leaf) for leaf in item.get("leaf_indices") or []}):
            if leaf_index < 0 or leaf_index >= len(components["leaves"]):
                continue
            leaf = components["leaves"][leaf_index]
            is_zero, details = factor_value(leaf, p, coeffs)
            profile = {
                "row_key": row_key,
                "surface_id": item.get("surface_id"),
                "salt": item.get("salt"),
                "leaf_index": leaf_index,
                **details,
            }
            if is_zero:
                row_leaves.setdefault(row_key, set()).add(leaf_index)
                kept_profiles.append(profile)
            else:
                rejected_profiles.append(profile)
    return row_leaves, kept_profiles, rejected_profiles


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    gated = [case for case in cases if int((case.get("factor_gate_replay") or {}).get("selected_leaf_count") or 0)]
    verified = [
        case for case in gated if bool((case.get("factor_gate_replay") or {}).get("public_key_verified"))
    ]
    ratios = [
        float((case.get("factor_gate_replay") or {}).get("ops_over_rho"))
        for case in gated
        if (case.get("factor_gate_replay") or {}).get("ops_over_rho") is not None
    ]
    verified_ratios = [
        float((case.get("factor_gate_replay") or {}).get("ops_over_rho"))
        for case in verified
        if (case.get("factor_gate_replay") or {}).get("ops_over_rho") is not None
    ]
    best_verified = sorted(
        verified,
        key=lambda case: (
            float((case.get("factor_gate_replay") or {}).get("ops_over_rho") or 10**18),
            str(case.get("case_key")),
        ),
    )[:8]
    return {
        "source_case_count": len(cases),
        "factor_gate_nonempty_case_count": len(gated),
        "factor_gate_public_key_verified_count": len(verified),
        "factor_gate_below_rho_count": sum(
            1 for case in gated if bool((case.get("factor_gate_replay") or {}).get("below_rho"))
        ),
        "factor_gate_verified_below_rho_count": sum(
            1 for case in verified if bool((case.get("factor_gate_replay") or {}).get("below_rho"))
        ),
        "min_factor_gate_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_factor_gate_ops_over_rho": mean_or_none(ratios),
        "min_verified_factor_gate_ops_over_rho": round(min(verified_ratios), 8) if verified_ratios else None,
        "best_verified_factor_gate_cases": [
            {
                "case_key": case.get("case_key"),
                "target": case.get("target"),
                "transfer_index": case.get("transfer_index"),
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "source_ops_over_rho": case.get("source_ops_over_rho"),
                "factor_gate_replay": {
                    key: (case.get("factor_gate_replay") or {}).get(key)
                    for key in (
                        "selected_row_count",
                        "selected_leaf_count",
                        "relation_count",
                        "rank",
                        "public_key_verified",
                        "derived_secret",
                        "ops_over_rho",
                        "below_rho",
                    )
                },
                "factor_zero_profiles": case.get("factor_zero_profiles"),
            }
            for case in best_verified
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--factor", type=parse_factor, required=True)
    parser.add_argument("--factor-p", type=int)
    parser.add_argument("--target")
    parser.add_argument("--transfer-index", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--leaf-selector")
    parser.add_argument("--radius", type=int)
    parser.add_argument("--max-cases", type=int, default=0)
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

    signature = load_json(args.signature_source)
    source_cases = [
        case
        for case in signature.get("positive_cases") or []
        if isinstance(case, dict) and case_selected(case, args)
    ]
    if args.max_cases and args.max_cases > 0:
        source_cases = source_cases[: args.max_cases]

    bank_source = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = relation_probe.load_verifier_module()
    records = verifier.load_records()

    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    cases = []
    for case in source_cases:
        source_leaves, source_profiles = replay_probe.row_leaf_groups(case, None)
        contexts, errors = replay_probe.materialize_contexts(
            verifier,
            records,
            config_source,
            specs_by_target,
            case,
            sorted(source_leaves),
            args,
            context_cache,
        )
        factor_leaves, kept_profiles, rejected_profiles = gated_row_leaves(
            case,
            contexts,
            args.factor,
            args.factor_p,
        )
        factor_replay, _events = replay_probe.replay_selection(
            verifier,
            factor_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
        cases.append(
            {
                "case_key": replay_probe.case_key_string(case),
                "target": case.get("target"),
                "transfer_index": int(case.get("transfer_index") or 0),
                "top_k": int(case.get("top_k") or 0),
                "policy": case.get("policy"),
                "row_selector": case.get("row_selector"),
                "leaf_selector": case.get("leaf_selector") or case.get("selector"),
                "source_public_key_verified": bool(case.get("public_key_verified")),
                "source_relation_count": int(case.get("relation_count") or 0),
                "source_rank": int(case.get("rank") or 0),
                "source_ops_over_rho": case.get("ops_over_rho"),
                "source_selected_row_count": int(case.get("selected_row_count") or 0),
                "source_selected_leaf_count": int(case.get("selected_leaf_count") or 0),
                "source_row_leaf_keys": source_profiles,
                "factor_zero_profile_count": len(kept_profiles),
                "factor_rejected_profile_count": len(rejected_profiles),
                "factor_zero_profiles": kept_profiles,
                "factor_rejected_profiles": rejected_profiles[:16],
                "context_error_count": len(errors),
                "context_errors": errors,
                "factor_gate_replay": factor_replay,
            }
        )

    output = {
        "schema": "ecdlp_ffe_public_linear_factor_gate_replay_probe_v1",
        "method": "public_linear_factor_leaf_gate_relation_replay",
        "parameters": {
            "signature_source": str(args.signature_source),
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "factor": {
                "b_coeff": int(args.factor[0]),
                "c_coeff": int(args.factor[1]),
                "constant": int(args.factor[2]),
                "p": args.factor_p,
            },
            "target": args.target,
            "transfer_index": args.transfer_index,
            "top_k": args.top_k,
            "policy": args.policy,
            "leaf_selector": args.leaf_selector,
            "radius": radius,
            "max_cases": args.max_cases,
            "seed": args.seed,
        },
        "summary": summarize(cases),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
