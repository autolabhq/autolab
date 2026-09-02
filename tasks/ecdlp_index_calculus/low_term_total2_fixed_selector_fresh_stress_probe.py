#!/usr/bin/env python3
"""Stress fresh low-term-total2 leaves with a frozen public row selector.

The full low-monic stress probe retrains row selectors on calibration windows.
For fresh transfer windows that can be expensive, so this wrapper freezes a
previously validated public row selector and evaluates only the requested fresh
transfer indices plus fixed public leaf selectors.
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

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_compress_probe as compress_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_loto_leaf_trim_probe as leaf_trim_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_loto_selector_probe as loto_selector_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_monic_c_stress_probe as stress_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_transfer_probe as salt_neighborhood_probe
import relation_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LIVE_STATE_DIR = Path(
    os.environ.get("ECDLP_LIVE_STATE_DIR", "/Volumes/Volume/autolab/ecdlp_index_calculus_state")
)
DEFAULT_BANK_SOURCE = DEFAULT_LIVE_STATE_DIR / stress_probe.DEFAULT_BANK_SOURCE.name
DEFAULT_CONFIG_SOURCE = DEFAULT_LIVE_STATE_DIR / stress_probe.DEFAULT_CONFIG_SOURCE.name
DEFAULT_DIRECT_SOURCE = DEFAULT_LIVE_STATE_DIR / stress_probe.DEFAULT_DIRECT_SOURCE.name
DEFAULT_TRANSFER_SOURCE = DEFAULT_LIVE_STATE_DIR / stress_probe.DEFAULT_TRANSFER_SOURCE.name
DEFAULT_COMPRESS_SOURCE = DEFAULT_LIVE_STATE_DIR / stress_probe.DEFAULT_COMPRESS_SOURCE.name
DEFAULT_ROW_SELECTOR = "target_cap3_ow0_hw1_lw0_sw0_cw0_aw0"
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_80_87_probe.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def selector_by_name(name: str) -> loto_selector_probe.SelectorSpec:
    for spec in loto_selector_probe.selector_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"unknown fixed row selector: {name}")


def fixed_choice(spec: loto_selector_probe.SelectorSpec) -> dict[str, Any]:
    return {
        "selector": spec.name,
        "spec": spec.__dict__,
        "train": {
            "policy": "fixed_prevalidated",
            "case_count": 0,
            "source_verified_case_count": 0,
            "verified_case_count": 0,
            "cases_below_rho": 0,
            "best_ops_over_rho": None,
            "mean_ops_over_rho": None,
            "mean_selected_row_count": None,
            "note": (
                "Selector was frozen from prior public-selector windows; this "
                "fresh stress run does not rescan calibration verifier outcomes."
            ),
        },
    }


def compact_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stress_probe.compact_training_cases(cases)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--compress-source", type=Path, default=DEFAULT_COMPRESS_SOURCE)
    parser.add_argument("--stress-transfer-indices", default="80,81,82,83,84,85,86,87")
    parser.add_argument("--top-k-grid")
    parser.add_argument("--fixed-row-selector", default=DEFAULT_ROW_SELECTOR)
    parser.add_argument("--fixed-row-selectors", default="")
    parser.add_argument("--leaf-selectors", default="mode_cost_low_term_support_total2")
    parser.add_argument("--leaf-residue-modulus", type=int, default=16)
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
    parser.add_argument(
        "--allow-combined-coefficients",
        dest="require_unit_coefficients",
        action="store_false",
    )
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(params, dict):
        params = {}
    radius = int(args.radius if args.radius is not None else params.get("radius") or 4)
    top_k_grid = compress_probe.parse_ints(
        args.top_k_grid,
        [int(item) for item in params.get("top_k_grid") or [4, 7, 12, 16]],
    )
    stress_indices = compress_probe.parse_ints(
        args.stress_transfer_indices,
        [80, 81, 82, 83, 84, 85, 86, 87],
    )
    leaf_specs = [
        stress_probe.parse_leaf_selector(item.strip())
        for item in str(args.leaf_selectors).split(",")
        if item.strip()
    ]
    row_selector_names = [
        item.strip()
        for item in str(args.fixed_row_selectors).split(",")
        if item.strip()
    ] or [str(args.fixed_row_selector)]
    row_specs = [selector_by_name(name) for name in row_selector_names]

    bank = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    compress_source = load_json(args.compress_source)
    labels = loto_selector_probe.oracle_labels(compress_source)
    bank_rows = {
        compress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict) and compress_probe.row_key(row)
    }
    target_specs = salt_neighborhood_probe.witness_specs(direct_source, bank_rows, radius)
    specs_by_target = leaf_trim_probe.specs_by_target_and_key(target_specs)
    verifier = relation_probe.load_verifier_module()
    records = verifier.load_records()

    stress_cases = loto_selector_probe.scan_cases(
        verifier,
        records,
        config_source,
        target_specs,
        stress_indices,
        top_k_grid,
        labels,
        args,
    )
    policy_results = {}
    policy_summaries = []
    source_verified_count = sum(1 for case in stress_cases if case.get("source_verified"))
    for row_spec in row_specs:
        chosen = fixed_choice(row_spec)
        stress_row_results = stress_probe.evaluate_row_selector(
            verifier,
            stress_cases,
            [],
            chosen,
            labels,
            int(args.leaf_residue_modulus),
        )
        stress_leaf_results = []
        if len(stress_cases) != len(stress_row_results):
            raise RuntimeError(
                "row selector result count mismatch: "
                f"{len(stress_row_results)} results for {len(stress_cases)} cases"
            )
        for case, row_result in zip(stress_cases, stress_row_results):
            stress_leaf_results.extend(
                stress_probe.scan_case_for_leaf_selectors(
                    verifier,
                    records,
                    config_source,
                    specs_by_target,
                    case,
                    row_result,
                    leaf_specs,
                    int(args.leaf_residue_modulus),
                    args,
                )
            )

        row_summary = loto_selector_probe.metrics(stress_row_results, source_verified_count)
        leaf_summary = stress_probe.summarize_rows(stress_leaf_results)
        policy_name = f"fixed_{row_spec.name}"
        policy_results[policy_name] = {
            "row_selector": chosen,
            "stress_row_summary": row_summary,
            "stress_leaf_summary": leaf_summary,
            "stress_leaf_results": stress_leaf_results,
        }
        policy_summaries.append(
            {
                "policy": policy_name,
                "row_selector": row_spec.name,
                "stress_row_verified": row_summary.get("verified_case_count"),
                "stress_row_below_rho": row_summary.get("cases_below_rho"),
                "stress_leaf_verified": leaf_summary.get("verified_case_count"),
                "stress_leaf_below_rho": leaf_summary.get("cases_below_rho"),
                "stress_leaf_best_ops_over_rho": leaf_summary.get("best_ops_over_rho"),
            }
        )
    best_policy_summary = max(
        policy_summaries,
        key=lambda row: (
            int(row.get("stress_leaf_below_rho") or 0),
            int(row.get("stress_leaf_verified") or 0),
            -float(row.get("stress_leaf_best_ops_over_rho") or 10**9),
            int(row.get("stress_row_below_rho") or 0),
            int(row.get("stress_row_verified") or 0),
        ),
    )
    output = {
        "schema": "ecdlp_low_term_total2_fixed_selector_fresh_stress_probe_v1",
        "method": "fixed_public_row_selector_fresh_low_term_total2_stress",
        "parameters": {
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "compress_source": str(args.compress_source),
            "stress_transfer_indices": stress_indices,
            "top_k_grid": top_k_grid,
            "fixed_row_selectors": [spec.name for spec in row_specs],
            "leaf_selectors": [spec.name for spec in leaf_specs],
            "leaf_residue_modulus": args.leaf_residue_modulus,
            "radius": radius,
            "row_pool": args.row_pool,
            "row_count": args.row_count,
            "scout_limit": args.scout_limit,
            "scout_mode": args.scout_mode,
            "scout_order": args.scout_order,
            "selected_limit": args.selected_limit,
            "factor_base_size": args.factor_base_size,
            "seed": args.seed,
        },
        "summary": {
            "policy_summaries": policy_summaries,
            "best_policy_summary": best_policy_summary,
            "row_selector": best_policy_summary.get("row_selector"),
            "stress_leaf_verified": best_policy_summary.get("stress_leaf_verified"),
            "stress_leaf_below_rho": best_policy_summary.get("stress_leaf_below_rho"),
            "stress_leaf_best_ops_over_rho": best_policy_summary.get("stress_leaf_best_ops_over_rho"),
            "interpretation": (
                "This run freezes the public row selector selected in earlier "
                "windows and evaluates only fresh transfer indices. No verifier "
                "outcomes from the requested fresh window are used to choose the row "
                "selector or fixed leaf selector."
            ),
        },
        "policies": policy_results,
        "stress_cases": compact_cases(stress_cases),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
