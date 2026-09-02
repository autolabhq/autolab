#!/usr/bin/env python3
"""Audit whether the slice-quadratic FFE diagnostic is a public relation source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.p1436_slice_quadratic_public_source_preflight.v1"
EXPECTED_SCHEMAS = {
    "slice_probe": "ecdlp_low_term_total2_ffe_slice_quadratic_root_probe_v1",
    "signature": "ecdlp_low_term_total2_signature_probe_v1",
    "p1228": "ecdlp.low_term_total2_p1228_slice_quadratic_ffe_factor_fingerprint_predictor.v1",
    "p1324": "ecdlp.low_term_total2_p1324_public_prefix_slice_qroot_route_join_after_p1323.v1",
    "r68": "p1553.ffe_fixed_sum_information_conservation.r68.v1",
}
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = Path("/Volumes/Volume/git/autolab")
LIVE_STATE = LIVE_ROOT / "ecdlp_index_calculus_state"
LIVE_TASK = LIVE_ROOT / "tasks" / "ecdlp_index_calculus"
STEM = (
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_"
    "guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_"
    "low_term_total2"
)
DEFAULT_PATHS = {
    "slice_probe": LIVE_STATE / f"{STEM}_ffe_slice_quadratic_root_probe.json",
    "signature": LIVE_STATE / f"{STEM}_signature_probe.json",
    "p1228": LIVE_STATE
    / "p1228_slice_quadratic_ffe_factor_fingerprint_predictor_probe.json",
    "p1324": LIVE_STATE
    / "p1324_public_prefix_slice_qroot_route_join_after_p1323_probe.json",
    "r68": WORKTREE_ROOT
    / "p1553_ffe_fixed_sum_information_conservation_report_r68.json",
}
DEFAULT_FACTOR_SOURCE = LIVE_TASK / f"{STEM}_ffe_resultant_slice_factor_probe.py"
DEFAULT_SLICE_SOURCE = LIVE_TASK / f"{STEM}_ffe_slice_quadratic_root_probe.py"
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state"
    / "p1436_slice_quadratic_public_source_preflight.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_schemas(inputs: dict[str, dict[str, Any]]) -> None:
    mismatches = {
        name: {
            "expected": EXPECTED_SCHEMAS[name],
            "observed": value.get("schema"),
        }
        for name, value in inputs.items()
        if value.get("schema") != EXPECTED_SCHEMAS[name]
    }
    if mismatches:
        raise ValueError(f"input schema mismatch: {json.dumps(mismatches, sort_keys=True)}")


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def source_dependency_audit(factor_source: str) -> dict[str, Any]:
    tree = ast.parse(factor_source)
    chooser = function_node(tree, "choose_slice_factors")
    coordinates = function_node(tree, "selected_coordinates")
    chooser_calls_selected_coordinates = bool(
        chooser
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "selected_coordinates"
            for node in ast.walk(chooser)
        )
    )
    selected_coordinates_reads_selected_leaves = bool(
        coordinates
        and any(
            isinstance(node, ast.Constant)
            and node.value == "selected_leaf_indices"
            for node in ast.walk(coordinates)
        )
    )
    return {
        "chooser_function_found": chooser is not None,
        "selected_coordinates_function_found": coordinates is not None,
        "chooser_calls_selected_coordinates": chooser_calls_selected_coordinates,
        "selected_coordinates_reads_selected_leaf_indices": (
            selected_coordinates_reads_selected_leaves
        ),
        "oracle_label_dependency": (
            chooser_calls_selected_coordinates
            and selected_coordinates_reads_selected_leaves
        ),
    }


def rho_by_target(surfaces: list[dict[str, Any]]) -> dict[str, int]:
    result = {}
    for surface in surfaces:
        for candidate in surface.get("candidates") or []:
            rho = candidate.get("generic_rho_steps")
            if rho is not None:
                result[str(surface["target"])] = int(rho)
                break
    return result


def blind_slice_audit(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    target_rhos = rho_by_target(surfaces)
    rows = []
    for surface in surfaces:
        p = int(surface["p"])
        axis_count = 2
        universe = axis_count * p
        success_count = int(surface.get("preserving_candidate_count") or 0)
        rho = target_rhos.get(str(surface["target"]))
        expected_draws = (
            (universe + 1) / (success_count + 1) if success_count else None
        )
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "target": surface["target"],
                "p": p,
                "axis_value_universe": universe,
                "oracle_successful_axis_value_count": success_count,
                "optimistic_expected_draws_without_replacement": expected_draws,
                "generic_rho_steps": rho,
                "optimistic_expected_draws_over_rho": (
                    expected_draws / rho if expected_draws is not None and rho else None
                ),
                "assumption": "one free operation per axis/value candidate",
            }
        )
    finite_ratios = [
        float(row["optimistic_expected_draws_over_rho"])
        for row in rows
        if row["optimistic_expected_draws_over_rho"] is not None
    ]
    by_target: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "surface_count": 0,
            "preserving_surface_count": 0,
            "row_keys": set(),
            "challenge_seeds": set(),
            "oracle_coordinate_pairs": set(),
            "ratios": [],
        }
    )
    surface_lookup = {surface["surface_id"]: surface for surface in surfaces}
    for row in rows:
        target = str(row["target"])
        aggregate = by_target[target]
        aggregate["surface_count"] += 1
        surface = surface_lookup[row["surface_id"]]
        aggregate["row_keys"].add(str(surface["row_key"]))
        aggregate["challenge_seeds"].add(str(surface["challenge_seed"]))
        if row["oracle_successful_axis_value_count"]:
            aggregate["preserving_surface_count"] += 1
        if row["optimistic_expected_draws_over_rho"] is not None:
            aggregate["ratios"].append(
                float(row["optimistic_expected_draws_over_rho"])
            )
        for candidate in surface.get("candidates") or []:
            fixed_values = tuple(
                sorted(
                    {
                        int(factor["fixed_value"])
                        for factor in candidate.get("factors") or []
                    }
                )
            )
            aggregate["oracle_coordinate_pairs"].add(
                (str(candidate.get("axis")), fixed_values)
            )
    target_summaries = []
    for target, aggregate in sorted(by_target.items()):
        target_summaries.append(
            {
                "target": target,
                "surface_count": aggregate["surface_count"],
                "preserving_surface_count": aggregate[
                    "preserving_surface_count"
                ],
                "unique_row_key_count": len(aggregate["row_keys"]),
                "unique_challenge_seed_count": len(aggregate["challenge_seeds"]),
                "distinct_oracle_axis_fixed_value_count": len(
                    aggregate["oracle_coordinate_pairs"]
                ),
                "minimum_optimistic_expected_draws_over_rho": (
                    min(aggregate["ratios"]) if aggregate["ratios"] else None
                ),
            }
        )
    return {
        "surface_rows": rows,
        "target_summaries": target_summaries,
        "minimum_optimistic_expected_draws_over_rho": (
            min(finite_ratios) if finite_ratios else None
        ),
        "maximum_optimistic_expected_draws_over_rho": (
            max(finite_ratios) if finite_ratios else None
        ),
        "surfaces_without_any_oracle_slice": sum(
            row["oracle_successful_axis_value_count"] == 0 for row in rows
        ),
        "interpretation": (
            "This is an optimistic lower bound for a label-blind uniform slice "
            "schedule, not a generic-group lower bound. Real specialization, "
            "factorization, rejection, and leaf-scan costs are omitted."
        ),
    }


def evaluate(
    inputs: dict[str, dict[str, Any]],
    factor_source: str,
) -> dict[str, Any]:
    validate_schemas(inputs)
    probe = inputs["slice_probe"]
    signature = inputs["signature"]
    p1228 = inputs["p1228"]
    p1324 = inputs["p1324"]
    r68 = inputs["r68"]
    source_audit = source_dependency_audit(factor_source)
    surfaces = list(probe.get("surfaces") or [])
    blind_audit = blind_slice_audit(surfaces)

    target_names = sorted({str(surface["target"]) for surface in surfaces})
    explicit_splits = sorted(
        {str(surface["split"]) for surface in surfaces if surface.get("split")}
    )
    positive_cases = list(signature.get("positive_cases") or [])
    negative_cases = list(signature.get("negative_cases") or [])
    prior_predictor = p1228["leave_one_target_out_structural_rule"]
    prior_join = p1324["summary"]
    summary = probe["summary"]

    public_selector_pass = (
        not source_audit["oracle_label_dependency"]
        and probe.get("parameters", {}).get("selector_mode") == "public_pre_choice"
    )
    heldout_pass = len(target_names) >= 4 and {"train", "heldout"} <= set(
        explicit_splits
    )
    negative_control_pass = (
        bool(positive_cases)
        and bool(negative_cases)
        and int(prior_predictor["covered_negative_count"]) > 0
    )
    complete_cost_pass = bool(summary.get("all_public_candidates_charged"))
    measured_public_ratio = summary.get("public_source_total_ops_over_rho_max")
    public_source_below_rho_pass = (
        public_selector_pass
        and complete_cost_pass
        and measured_public_ratio is not None
        and float(measured_public_ratio) < 1.0
    )
    new_rows = int(summary.get("new_independent_fixed_sum_row_count") or 0)
    fresh_rank_delta = int(summary.get("fresh_rank_delta") or 0)
    new_row_pass = (
        new_rows > 0
        and fresh_rank_delta > 0
        and bool(prior_join["exact_route_join_gate_passed"])
    )

    obligations = {
        "public_pre_choice_slice_selector": {
            "pass": public_selector_pass,
            "selector_mode": probe.get("parameters", {}).get("selector_mode"),
            **source_audit,
        },
        "frozen_disjoint_target_heldout_split": {
            "pass": heldout_pass,
            "unique_target_count": len(target_names),
            "targets": target_names,
            "explicit_splits": explicit_splits,
            "minimum_target_count": 4,
        },
        "covered_positive_and_negative_controls": {
            "pass": negative_control_pass,
            "signature_positive_case_count": len(positive_cases),
            "signature_negative_case_count": len(negative_cases),
            "prior_predictor_covered_positive_count": prior_predictor[
                "covered_positive_count"
            ],
            "prior_predictor_covered_negative_count": prior_predictor[
                "covered_negative_count"
            ],
            "prior_predictor_promotion_passed": p1228["promotion_gate"]["passed"],
        },
        "complete_rejected_candidate_costing": {
            "pass": complete_cost_pass,
            "all_public_candidates_charged": summary.get(
                "all_public_candidates_charged"
            ),
            "emitted_oracle_candidate_count": summary[
                "surface_candidate_count"
            ],
            "public_pre_choice_candidate_count": summary.get(
                "public_pre_choice_candidate_count", 0
            ),
        },
        "public_source_below_rho": {
            "pass": public_source_below_rho_pass,
            "measured_public_source_total_ops_over_rho_max": measured_public_ratio,
            "oracle_selected_hit_min_ops_over_rho": summary[
                "min_slice_quadratic_selected_hit_ops_over_rho"
            ],
            "oracle_all_hit_min_ops_over_rho": summary[
                "min_slice_quadratic_all_hit_ops_over_rho"
            ],
            "blind_uniform_slice_audit": blind_audit,
        },
        "new_independent_fixed_sum_rows_and_route_join": {
            "pass": new_row_pass,
            "new_independent_fixed_sum_row_count": new_rows,
            "fresh_rank_delta": fresh_rank_delta,
            "prior_exact_route_join_gate_passed": prior_join[
                "exact_route_join_gate_passed"
            ],
            "prior_target_independent_gate_passed": prior_join[
                "additional_or_target_independent_gate_passed"
            ],
        },
    }
    failed = [name for name, gate in obligations.items() if not gate["pass"]]
    source_lane_admitted = not failed
    return {
        "classification": (
            "SLICE_QUADRATIC_PUBLIC_SOURCE_LANE_ADMITTED"
            if source_lane_admitted
            else "SLICE_QUADRATIC_ORACLE_DIAGNOSTIC_ONLY"
        ),
        "diagnostic_result": {
            "surfaces_with_preserving_slice_quadratic": summary[
                "surfaces_with_preserving_slice_quadratic"
            ],
            "surface_count": summary["surface_count"],
            "oracle_all_hit_below_rho_count": summary[
                "slice_quadratic_all_hit_below_rho_count"
            ],
            "oracle_selected_hit_below_rho_count": summary[
                "slice_quadratic_selected_hit_below_rho_count"
            ],
            "status": "valid_post_selector_component_diagnostic",
        },
        "obligations": obligations,
        "admission": {
            "source_lane_admitted": source_lane_admitted,
            "failed_obligations": failed,
            "passed_obligation_count": len(obligations) - len(failed),
            "obligation_count": len(obligations),
        },
        "r68_binding": {
            "information_conservation_pass": bool(r68["pass"]),
            "product_or_quotient_relation_rank_credit": 0,
            "new_fixed_sum_rows_required": r68["result"][
                "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows"
            ],
        },
        "claim_boundary": {
            "algorithm_breakthrough": False,
            "generic_prime_field_speedup": False,
            "shoup_bound_improvement": False,
            "oracle_component_below_rho_is_full_attack": False,
        },
        "fatal_obstruction": (
            "The emitted slice is selected from verifier-backed leaf coefficients. "
            "The corpus contains no public pre-choice candidate stream, no covered "
            "negative cases, no frozen target-level heldout split, and no fresh "
            "independent-row delta. The oracle-selected component cannot be charged "
            "as a relation source."
        ),
        "next_action": (
            "Modify the surface producer to emit one label-separated corpus over at "
            "least four target families: deterministic public pre-choice axis/value/"
            "factor candidates, sequestered positive and negative outcomes, every "
            "rejected-candidate cost, and fresh fixed-sum row/rank deltas; then run "
            "leave-one-target-family-out exactly once."
        ),
    }


def build_payload(
    paths: dict[str, Path],
    factor_source_path: Path,
    slice_source_path: Path,
) -> dict[str, Any]:
    inputs = {name: read_json(path) for name, path in paths.items()}
    factor_source = factor_source_path.read_text(encoding="utf-8")
    result = evaluate(inputs, factor_source)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "public pre-choice slice-quadratic FFE relation source",
        "source_bindings": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "schema": inputs[name]["schema"],
                "claim_status": inputs[name].get("claim_status"),
            }
            for name, path in sorted(paths.items())
        },
        "code_bindings": {
            "factor_source": {
                "path": str(factor_source_path),
                "sha256": sha256_file(factor_source_path),
            },
            "slice_source": {
                "path": str(slice_source_path),
                "sha256": sha256_file(slice_source_path),
            },
        },
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path)
    parser.add_argument("--factor-source", type=Path, default=DEFAULT_FACTOR_SOURCE)
    parser.add_argument("--slice-source", type=Path, default=DEFAULT_SLICE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {name: getattr(args, name) for name in DEFAULT_PATHS}
    payload = build_payload(paths, args.factor_source, args.slice_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission = payload["admission"]
    blind = payload["obligations"]["public_source_below_rho"][
        "blind_uniform_slice_audit"
    ]
    print(
        f"output={args.output} admitted={admission['source_lane_admitted']} "
        f"passed={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} optimistic_blind_min_rho="
        f"{blind['minimum_optimistic_expected_draws_over_rho']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
