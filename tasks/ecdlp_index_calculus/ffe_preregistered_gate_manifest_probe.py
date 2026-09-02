#!/usr/bin/env python3
"""Build a pre-factor FFE gate manifest for root-hyperplane selector tests.

The root-hyperplane selector has a strong diagnostic slice: surfaces with
original selected root pairs and public-zero-capable factors are below rho on
the current 80+ control.  Public-zero capability is known only after
factorization, so this probe records the closest pre-factor gate available from
materialized FFE surfaces:

* the full surface already has a selected root pair, and
* the selected public leaf has a known hit root that could become a
  ``c + r*b + r^2`` root hyperplane after factorization.

The gate is therefore preregisterable for a fresh bank before Sage
factorization or root-policy scoring.  When a selector output is provided, this
script joins the manifest to the later selector rows only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_cross_surface_root_map_probe as cross_surface_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LIVE_STATE_DIR = Path(
    os.environ.get("ECDLP_LIVE_STATE_DIR", "/Volumes/Volume/autolab/ecdlp_index_calculus_state")
)
DEFAULT_SIGNATURE_SOURCE = (
    DEFAULT_STATE_DIR / "low_term_total3_total4_verified_over_rho_diagnostic_signature_80_87.json"
)
DEFAULT_SELECTOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_first_fall_root_hyperplane_selector_total3_total4_verified_over_rho_80_87.json"
)
DEFAULT_BANK_SOURCE = DEFAULT_LIVE_STATE_DIR / cross_surface_probe.compress_probe.DEFAULT_BANK_SOURCE.name
DEFAULT_CONFIG_SOURCE = DEFAULT_LIVE_STATE_DIR / cross_surface_probe.compress_probe.DEFAULT_CONFIG_SOURCE.name
DEFAULT_DIRECT_SOURCE = DEFAULT_LIVE_STATE_DIR / cross_surface_probe.compress_probe.DEFAULT_DIRECT_SOURCE.name
DEFAULT_TRANSFER_SOURCE = DEFAULT_LIVE_STATE_DIR / cross_surface_probe.salt_neighborhood_probe.DEFAULT_OUT.name
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_verified_over_rho_80_87.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def selector_rows(selector_source: dict[str, Any], policy: str | None) -> tuple[str | None, dict[str, dict[str, Any]]]:
    summary = selector_source.get("summary") or {}
    selected_policy = policy or summary.get("best_policy")
    rows = selector_source.get("policy_rows") or {}
    if not selected_policy or selected_policy not in rows:
        return selected_policy, {}
    return str(selected_policy), {
        str(row.get("surface_id")): row
        for row in rows.get(str(selected_policy)) or []
        if isinstance(row, dict) and row.get("surface_id")
    }


def selected_pair_set(
    surface_record: dict[str, Any],
    candidate_surface: dict[str, Any],
) -> set[tuple[int, int]]:
    samples = cross_surface_probe.root_map_probe.surface_root_samples(
        surface_record["components"],
        candidate_surface,
        int(surface_record["p"]),
        set(surface_record["selected_leaf_indices"]),
        [(0, 0)],
    )
    return {
        (int(sample["leaf_index"]), int(sample["root"]))
        for sample in samples
        if bool(sample.get("selected"))
    }


def selected_zero_stats(
    surface_record: dict[str, Any],
    candidate_surface: dict[str, Any],
) -> dict[str, Any]:
    p = int(surface_record["p"])
    components = surface_record["components"]
    hit_root_set = {int(root) % p for root in components["hit_roots"]}
    stats = {
        "selected_leaf_count": len(surface_record["selected_leaf_indices"]),
        "selected_surface_zero_leaves": 0,
        "selected_valid_root_leaves": 0,
        "selected_false_zero_leaves": 0,
        "selected_missed_leaves": 0,
        "selected_linear_root_recoveries": 0,
        "selected_degenerate_root_scans": 0,
    }
    for leaf_index in surface_record["selected_leaf_indices"]:
        leaf = components["leaves"][int(leaf_index)]
        coeffs = cross_surface_probe.resultant_surface_probe.monic_coeffs(leaf, p)
        if coeffs is None:
            stats["selected_missed_leaves"] += 1
            continue
        b_value, c_value = coeffs
        if cross_surface_probe.resultant_surface_probe.poly2_eval(
            candidate_surface["resultant"],
            b_value,
            c_value,
            p,
        ) != 0:
            stats["selected_missed_leaves"] += 1
            continue
        stats["selected_surface_zero_leaves"] += 1
        a_value = cross_surface_probe.resultant_surface_probe.poly2_eval(
            candidate_surface["remainder_a"],
            b_value,
            c_value,
            p,
        )
        b_value_remainder = cross_surface_probe.resultant_surface_probe.poly2_eval(
            candidate_surface["remainder_b"],
            b_value,
            c_value,
            p,
        )
        if a_value % p:
            stats["selected_linear_root_recoveries"] += 1
            roots = [(-b_value_remainder * pow(a_value, -1, p)) % p]
        else:
            stats["selected_degenerate_root_scans"] += 1
            roots = [
                root
                for root in hit_root_set
                if cross_surface_probe.resultant_surface_probe.polyops.poly_eval(leaf["poly"], root, p) == 0
            ]
        valid = [
            root
            for root in roots
            if root in hit_root_set
            and cross_surface_probe.resultant_surface_probe.polyops.poly_eval(leaf["poly"], root, p) == 0
        ]
        if valid:
            stats["selected_valid_root_leaves"] += 1
        else:
            stats["selected_false_zero_leaves"] += 1
    return stats


def selected_hit_root_pairs(surface_record: dict[str, Any]) -> list[dict[str, int]]:
    p = int(surface_record["p"])
    components = surface_record["components"]
    hit_roots = {int(root) % p for root in components.get("hit_roots") or []}
    out = []
    for leaf_index in surface_record.get("selected_leaf_indices") or []:
        leaf = components["leaves"][int(leaf_index)]
        coeffs = cross_surface_probe.resultant_surface_probe.monic_coeffs(leaf, p)
        if coeffs is None:
            continue
        b_value, c_value = (int(coeffs[0]) % p, int(coeffs[1]) % p)
        for root in sorted(hit_roots):
            if cross_surface_probe.resultant_surface_probe.polyops.poly_eval(leaf["poly"], root, p) == 0:
                out.append(
                    {
                        "leaf_index": int(leaf_index),
                        "root": int(root),
                        "b": b_value,
                        "c": c_value,
                    }
                )
    return out


def compact_source_cases(surface_record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for case in surface_record.get("source_cases") or []:
        out.append(
            {
                "target": case.get("target"),
                "transfer_index": int(case.get("transfer_index") or 0),
                "top_k": int(case.get("top_k") or 0),
                "policy": case.get("policy"),
                "row_selector": case.get("row_selector"),
                "leaf_selector": case.get("leaf_selector"),
                "source_ops_over_rho": case.get("source_ops_over_rho"),
            }
        )
    return out


def compact_surface(surface_record: dict[str, Any], selector_row: dict[str, Any] | None) -> dict[str, Any]:
    surface = surface_record["surface"]
    original_pairs = selected_pair_set(surface_record, surface)
    zero_stats = selected_zero_stats(surface_record, surface)
    hit_pairs = selected_hit_root_pairs(surface_record)
    pre_factor_nonvacuous = len(original_pairs) > 0
    pre_factor_public_zero_proxy = len(hit_pairs) > 0
    selected = bool(pre_factor_nonvacuous and pre_factor_public_zero_proxy)
    row = {
        "surface_id": surface_record["surface_id"],
        "target": surface_record["target"],
        "row_key": surface_record["row_key"],
        "row_schedule_key": surface_record.get("row_schedule_key"),
        "challenge_seed": surface_record["challenge_seed"],
        "transfer_index": int((compact_source_cases(surface_record) or [{}])[0].get("transfer_index") or 0),
        "p": int(surface_record["p"]),
        "selected_leaf_indices": list(surface_record.get("selected_leaf_indices") or []),
        "selected_signature": surface_record.get("selected_signature"),
        "source_case_count": len(surface_record.get("source_cases") or []),
        "source_cases": compact_source_cases(surface_record),
        "generic_rho_steps": int((surface_record.get("cost_inputs") or {}).get("generic_rho_steps") or 0),
        "full_remainder_ffe_ops_over_rho": surface_record.get("full_remainder_ffe_ops_over_rho"),
        "pre_factor_original_selected_root_pair_count": len(original_pairs),
        "pre_factor_selected_hit_root_pair_count": len(hit_pairs),
        "pre_factor_selected_hit_roots": sorted({int(pair["root"]) for pair in hit_pairs}),
        "pre_factor_selected_hit_root_pairs": hit_pairs[:16],
        "pre_factor_selected_zero_stats": zero_stats,
        "pre_factor_nonvacuous": pre_factor_nonvacuous,
        "pre_factor_public_zero_proxy": pre_factor_public_zero_proxy,
        "pre_factor_gate_selected": selected,
        "gate_label": "pre_factor_nonvacuous_hit_root_proxy" if selected else None,
    }
    if selector_row is not None:
        row["post_factor_selector"] = {
            "policy": selector_row.get("policy"),
            "public_zero_root_count": int(selector_row.get("public_zero_root_count") or 0),
            "public_zero_recovered": bool(selector_row.get("public_zero_recovered")),
            "original_selected_root_pair_count": int(selector_row.get("original_selected_root_pair_count") or 0),
            "chosen_preserves_selected_root_pairs": bool(
                selector_row.get("chosen_preserves_selected_root_pairs")
            ),
            "chosen_false_positive": bool(selector_row.get("chosen_false_positive")),
            "below_rho": bool(selector_row.get("below_rho")),
            "direct_below_rho": bool(selector_row.get("direct_below_rho")),
            "total_ops_over_rho": selector_row.get("total_ops_over_rho"),
            "direct_total_ops_over_rho": selector_row.get("direct_total_ops_over_rho"),
            "evaluated_root_count": selector_row.get("evaluated_root_count"),
            "selector_eval_ops": selector_row.get("selector_eval_ops"),
        }
    return row


def summarize_selector(rows: list[dict[str, Any]]) -> dict[str, Any]:
    joined = [row for row in rows if row.get("post_factor_selector")]
    ratios = [
        float((row["post_factor_selector"] or {}).get("total_ops_over_rho"))
        for row in joined
        if (row["post_factor_selector"] or {}).get("total_ops_over_rho") is not None
    ]
    direct_ratios = [
        float((row["post_factor_selector"] or {}).get("direct_total_ops_over_rho"))
        for row in joined
        if (row["post_factor_selector"] or {}).get("direct_total_ops_over_rho") is not None
    ]
    target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in joined:
        selector = row["post_factor_selector"]
        target = str(row.get("target"))
        target_counts[target]["surface_count"] += 1
        if selector.get("public_zero_root_count"):
            target_counts[target]["public_zero_capable_count"] += 1
        if selector.get("public_zero_recovered"):
            target_counts[target]["public_zero_recovered_count"] += 1
        if selector.get("chosen_preserves_selected_root_pairs"):
            target_counts[target]["chosen_preserving_count"] += 1
        if selector.get("below_rho"):
            target_counts[target]["below_rho_count"] += 1
        if selector.get("direct_below_rho"):
            target_counts[target]["direct_below_rho_count"] += 1
    return {
        "joined_surface_count": len(joined),
        "public_zero_capable_count": sum(
            int((row["post_factor_selector"] or {}).get("public_zero_root_count") or 0) > 0
            for row in joined
        ),
        "public_zero_recovered_count": sum(
            bool((row["post_factor_selector"] or {}).get("public_zero_recovered"))
            for row in joined
        ),
        "chosen_preserving_count": sum(
            bool((row["post_factor_selector"] or {}).get("chosen_preserves_selected_root_pairs"))
            for row in joined
        ),
        "chosen_false_positive_count": sum(
            bool((row["post_factor_selector"] or {}).get("chosen_false_positive"))
            for row in joined
        ),
        "below_rho_count": sum(bool((row["post_factor_selector"] or {}).get("below_rho")) for row in joined),
        "direct_below_rho_count": sum(
            bool((row["post_factor_selector"] or {}).get("direct_below_rho")) for row in joined
        ),
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": mean_or_none(ratios),
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
        "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
        "target_summaries": {target: dict(counter) for target, counter in sorted(target_counts.items())},
    }


def summarize_surfaces(rows: list[dict[str, Any]], selector_policy: str | None) -> dict[str, Any]:
    selected = [row for row in rows if row.get("pre_factor_gate_selected")]
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = str(row.get("target"))
        by_target[target]["surface_count"] += 1
        if row.get("pre_factor_nonvacuous"):
            by_target[target]["pre_factor_nonvacuous_count"] += 1
        if row.get("pre_factor_public_zero_proxy"):
            by_target[target]["pre_factor_public_zero_proxy_count"] += 1
        if row.get("pre_factor_gate_selected"):
            by_target[target]["pre_factor_gate_selected_count"] += 1
    return {
        "surface_count": len(rows),
        "pre_factor_nonvacuous_count": sum(bool(row.get("pre_factor_nonvacuous")) for row in rows),
        "pre_factor_public_zero_proxy_count": sum(
            bool(row.get("pre_factor_public_zero_proxy")) for row in rows
        ),
        "pre_factor_gate_selected_count": len(selected),
        "selector_policy": selector_policy,
        "all_surface_selector_summary": summarize_selector(rows),
        "gate_selector_summary": summarize_selector(selected),
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
        "interpretation": (
            "The gate is chosen before Sage factorization: full-surface selected "
            "root-pair presence plus selected-leaf hit-root presence. Joined "
            "selector metrics are evaluation-only and must not be used to choose "
            "the gate on a fresh bank."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--selector-source", type=Path, default=DEFAULT_SELECTOR_SOURCE)
    parser.add_argument("--selector-policy")
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=DEFAULT_TRANSFER_SOURCE)
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
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

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
    selector_policy, rows_by_surface = selector_rows(load_json(args.selector_source), args.selector_policy)
    surfaces = [
        compact_surface(record, rows_by_surface.get(str(record.get("surface_id"))))
        for record in sorted(surface_records, key=lambda row: str(row.get("surface_id")))
    ]
    output = {
        "schema": "ecdlp_ffe_preregistered_gate_manifest_probe_v1",
        "method": "pre_factor_nonvacuous_hit_root_proxy_gate_for_root_hyperplane_selector",
        "parameters": {
            "signature_source": str(args.signature_source),
            "selector_source": str(args.selector_source),
            "selector_policy": selector_policy,
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "radius": radius,
            "max_cases": args.max_cases,
            "gate_definition": {
                "name": "pre_factor_nonvacuous_hit_root_proxy",
                "pre_factor_nonvacuous": "full surface has at least one selected root pair",
                "pre_factor_public_zero_proxy": (
                    "selected public leaf has at least one known hit root before "
                    "Sage factorization"
                ),
                "post_factor_fields_forbidden_for_gate": [
                    "public_zero_root_count",
                    "public_zero_recovered",
                    "chosen_preserves_selected_root_pairs",
                    "below_rho",
                    "direct_below_rho",
                    "total_ops_over_rho",
                ],
            },
        },
        "summary": {
            "input_case_count": len(positive_cases),
            "verified_case_count": sum(bool(case.get("public_key_verified")) for case in case_results),
            **summarize_surfaces(surfaces, selector_policy),
        },
        "case_results": case_results,
        "surfaces": surfaces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
