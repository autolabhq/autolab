#!/usr/bin/env python3
"""Deduplicate the R177 marked locator against the R174 signed norm gcd."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_marked_fitting_signed_norm_dedup.r178.v1"

R177_PRODUCER = ROOT / "p1553_m6_global_marked_fitting_locator_probe_r177.py"
R177_REPORT = ROOT / "p1553_m6_global_marked_fitting_locator_probe_report_r177.json"
R177_FROZEN = ROOT / "frozen_m6_global_marked_fitting_locator.json"
R177_COST = ROOT / "m6_global_marked_fitting_locator_cost_ledger.json"
R177_REPLAY = ROOT / "m6_global_marked_fitting_locator_replay.json"
R177_CONTROLS = ROOT / "m6_global_marked_fitting_locator_controls.json"
R177_MARKER = ROOT / "global_marked_fitting_locator_r177.json"
R177_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_global_marked_fitting_locator_probe_r177.py"
R177_GATE = ROOT / "p1553_m6_global_marked_fitting_locator_probe_gate_r177.md"
R177_PARENT = ROOT / "p1553_m6_global_marked_fitting_locator_probe_parent_report_r177.yaml"

R174_REPORT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_report_r174.json"
R174_CONTROLS = ROOT / "m6_confluent_signed_dual_chow_pushforward_controls.json"
R174_GATE = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_gate_r174.md"
R174_PARENT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_parent_report_r174.yaml"

R171_REPORT = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_report_r171.json"
R171_GATE = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_gate_r171.md"
R171_PARENT = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_parent_report_r171.yaml"

R170_REPORT = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_report_r170.json"
R170_GATE = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_gate_r170.md"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r177_producer", R177_PRODUCER, "6aad043e4279175f5b7838eb536b003108b11a0c278df11f9b0737db75ffe923"),
    ("r177_report", R177_REPORT, "d37a2f5a79a41150127d6f1be540522d06c1cb02553e76a43936d58995e95394"),
    ("r177_frozen", R177_FROZEN, "101a2dd4eccc198acbdecd7c76219c1a268a9061815434db6c8f3a0583a0bac7"),
    ("r177_cost", R177_COST, "dc4649daa8069f7bb393929f6cf433b6610a875f644293829a9b9be3f1971a74"),
    ("r177_replay", R177_REPLAY, "41ce8a59968d8fbc493d39e9fdf55aec6a8c916ae46a9a834e76671b5b07d782"),
    ("r177_controls", R177_CONTROLS, "f8a1cb00056fec06b7b54db6c648a9ed4d615c3b808595ad79d98d7583926feb"),
    ("r177_marker", R177_MARKER, "d5d4f83e2e145c13f93bb9fa0f8132850b6bfb6bdc3fa58ad5b3779acff40980"),
    ("r177_test", R177_TEST, "8d8b17d48c39842597abdc9d58c6cd135a8b2c1f8224a60d3a2ff86a49483717"),
    ("r177_gate", R177_GATE, "a670f43be024d898f67d37d5a0cb6f26aa033406633ee4017ab6a340f0c35c5c"),
    ("r177_parent", R177_PARENT, "179d8a8010fd231e6bce5b868185876f30180b41b60c91ec2c5c7e64b440a538"),
    ("r174_report", R174_REPORT, "a8b1d1d5fd4ffeaef17726325ebd85c343285ef61d16ca4dd1ce91dfcce28496"),
    ("r174_controls", R174_CONTROLS, "6d69f7146857cfd017281550e7a44b99aa4c53623b767c852da40b81e8bc9d40"),
    ("r174_gate", R174_GATE, "46597f7afd91ada661b5822031489b7d08445b041dfd4399b4343199457afebb"),
    ("r174_parent", R174_PARENT, "cac550192f9da4090c1e5459a5baed1f0ea3437c89ec012436995fcfe5bb4911"),
    ("r171_report", R171_REPORT, "12c34212fa3e07321ec7e9524a32a845233b420eae28f16e2dc3abfebd4efa92"),
    ("r171_gate", R171_GATE, "0f862466fa7816c438c6ef8526e66e350b1091c02f2e631d5b28a4105c10d680"),
    ("r171_parent", R171_PARENT, "3fee4660f7b558116299174194657da9464435f402a9f3fb289489474fbb1f93"),
    ("r170_report", R170_REPORT, "a354427601094c8346253ac38bc0fd53f878e67d71962846561088ee6495162c"),
    ("r170_gate", R170_GATE, "564ad5ffd47317768209244b1432f13cc2a0d1245a3d582e59c9bb0174375e02"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_report_r178.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_marked_fitting_signed_norm_dedup.json"
DEFAULT_COST = ROOT / "m6_marked_fitting_signed_norm_dedup_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_marked_fitting_signed_norm_dedup_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_marked_fitting_signed_norm_dedup_controls.json"
DEFAULT_EQUIVALENCE = ROOT / "marked_fitting_signed_norm_equivalence_r178.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R177 = load_module("p1553_r177_for_r178", R177_PRODUCER)
R167 = R177.R167
R175 = R177.R175
R161 = R177.R161


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> None:
    failures = [
        name
        for name, path, expected in SOURCE_BINDINGS
        if sha256_file(path) != expected
    ]
    if failures:
        raise AssertionError(f"R178 source binding mismatch: {failures}")


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def control_row(payload: dict[str, Any], family_id: str, seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in payload["controls"]["controls"]
        if row["family_id"] == family_id and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one control for {family_id}:{seed}")
    return matches[0]


def root_polynomial(roots: list[int], prime: int) -> list[int]:
    result = [1]
    for root in roots:
        result = R161.poly_mul(result, [(-root) % prime, 1], prime)
    return R177.normalize_monic(result, prime)


def finite_control(
    curve: dict[str, Any],
    seed: int,
    r177_report: dict[str, Any],
    r174_report: dict[str, Any],
) -> dict[str, Any]:
    family_id = curve["family_id"]
    marked = control_row(r177_report, family_id, seed)
    chow = control_row(r174_report, family_id, seed)
    r167 = R167.finite_control(curve, seed)
    _, divisor, _ = R175.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    prime = int(curve["field_prime"])

    row_norm_points: list[tuple[int, int]] = []
    multiplicities: dict[int, int] = {}
    pair_evaluation_count = 0
    for left in selected:
        row_norm = 1
        multiplicity = 0
        for right in selected:
            pair_sum = R167.point_add(left, right, curve)
            value = R167.rational_value(
                r167["numerator_witness"],
                r167["denominator_witness"],
                pair_sum,
                prime,
            )
            row_norm = row_norm * value % prime
            multiplicity += value == 0
            pair_evaluation_count += 1
        left_x = int(left[0])
        row_norm_points.append((left_x, row_norm))
        if multiplicity:
            multiplicities[left_x] = multiplicity

    expected_multiplicities = {
        int(row["candidate_x"]): int(row["incidence_multiplicity"])
        for row in marked["candidate_multiplicities"]
    }
    if multiplicities != expected_multiplicities:
        raise AssertionError("row-norm valuations differ from R177")

    aggregate = R161.interpolate(row_norm_points, prime)
    if any(
        R161.poly_eval(aggregate, x_value, prime) != row_norm
        for x_value, row_norm in row_norm_points
    ):
        raise AssertionError("aggregate signed row norm interpolation failed")
    candidate_factor = R161.poly_gcd(divisor["u"], aggregate, prime)
    candidate_roots = sorted(multiplicities)
    if candidate_factor != root_polynomial(candidate_roots, prime):
        raise AssertionError("aggregate norm gcd differs from row-norm support")

    maximum_multiplicity = max(multiplicities.values())
    threshold_factors: list[list[int]] = []
    threshold_rows: list[dict[str, Any]] = []
    for level in range(1, maximum_multiplicity + 1):
        roots = sorted(
            root for root, value in multiplicities.items() if value >= level
        )
        factor = root_polynomial(roots, prime)
        threshold_factors.append(factor)
        threshold_rows.append(
            {
                "minimum_incidence_multiplicity": level,
                "degree": R161.poly_degree(factor),
                "roots": roots,
                "factor_sha256": sha256_json(factor),
            }
        )

    marker_from_filtration = [1]
    for factor in threshold_factors:
        marker_from_filtration = R161.poly_mul(
            marker_from_filtration, factor, prime
        )
    marker_from_filtration = R177.normalize_monic(marker_from_filtration, prime)
    if sha256_json(marker_from_filtration) != marked["marker_polynomial_sha256"]:
        raise AssertionError("threshold filtration does not reconstruct R177 marker")

    candidate_factor_sha256 = sha256_json(candidate_factor)
    if candidate_factor_sha256 != marked["candidate_factor_sha256"]:
        raise AssertionError("R178 candidate factor differs from R177")
    if candidate_factor_sha256 != chow["candidate_factor_sha256"]:
        raise AssertionError("R177 candidate factor differs from R174 signed norm")

    row_norm_zero_roots = sorted(
        x_value for x_value, value in row_norm_points if value == 0
    )
    return {
        "control_id": f"{family_id}_marked_fitting_norm_dedup_seed{seed}",
        "family_id": family_id,
        "field_prime": prime,
        "seed": seed,
        "selected_divisor_degree": len(selected),
        "pair_evaluation_count": pair_evaluation_count,
        "row_norm_count": len(row_norm_points),
        "aggregate_output_slot_count": len(selected),
        "aggregate_nonzero_coefficient_count": sum(value != 0 for value in aggregate),
        "aggregate_polynomial_sha256": sha256_json(aggregate),
        "aggregate_interpolation_exact": True,
        "row_norm_zero_roots": row_norm_zero_roots,
        "row_norm_zero_roots_match_r177": row_norm_zero_roots == candidate_roots,
        "candidate_root_count": len(candidate_roots),
        "candidate_roots": candidate_roots,
        "candidate_factor_degree": R161.poly_degree(candidate_factor),
        "candidate_factor_sha256": candidate_factor_sha256,
        "candidate_factor_equals_r177": True,
        "candidate_factor_equals_r174_signed_norm": True,
        "maximum_incidence_multiplicity": maximum_multiplicity,
        "incidence_multiplicity_sum": sum(multiplicities.values()),
        "threshold_factor_count": len(threshold_factors),
        "threshold_degree_sum": sum(row["degree"] for row in threshold_rows),
        "threshold_degrees": [row["degree"] for row in threshold_rows],
        "threshold_factors": threshold_rows,
        "threshold_product_equals_r177_marker": True,
        "first_threshold_equals_candidate_factor": threshold_factors[0]
        == candidate_factor,
        "finite_pair_scan_and_interpolation_receive_asymptotic_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "signed_row_norm": (
            "For each selected P define C_h(P)=product_(Q in D) h(P+Q). "
            "Its zero set on the squarefree selected divisor is exactly the set "
            "of P having at least one signed target incidence. Thus "
            "G_1=gcd(U,C_h) is the R174 signed aggregate-norm candidate factor."
        ),
        "fitting_filtration": (
            "Let mu(P)=#{Q:h(P+Q)=0} and define G_r(A)=product_(mu(P)>=r) "
            "(A-x(P)). Then the R177 restricted-kernel characteristic polynomial "
            "is L(A)=product_(r>=1) G_r(A). In particular G_1=gcd(U,L)."
        ),
        "locator_deduplication": (
            "The ECDLP locator uses only distinct candidate roots, hence only G_1. "
            "R177's higher Fitting layers record incidence multiplicities but do "
            "not change candidate output, R163 labels, or signed verification."
        ),
        "algorithmic_equivalence": (
            "A constructor for the full R177 marker emits G_1 after gcd with U. "
            "For the ECDLP contract, directly emitting the R174 signed aggregate "
            "norm modulo U or its gcd G_1 is sufficient. Since reading compact "
            "U,V already costs Theta(n), both desired interfaces have the same "
            "softly O(n+N) campaign envelope."
        ),
        "remaining_primitive": (
            "The surviving primitive is the existing nonlocal signed elliptic "
            "translate-product or fused dual-Chow outer norm: emit C_h mod U or "
            "G_1 from compact U,V and the target divisor/principal witness in "
            "softly O(n+N), without nN or n^2 represented state."
        ),
        "scope": (
            "This is an exact mechanism-level deduplication and standard-route "
            "cost boundary, not a lower bound against custom arithmetic circuits, "
            "an ECDLP algorithm, or a Pollard-rho or Shoup improvement."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_marked_fitting_signed_norm_dedup.cost.r178.v1",
        "selected_divisor_input_exponent_B": fraction_record(Fraction(9, 4)),
        "target_witness_input_exponent_B": fraction_record(Fraction(5, 4)),
        "distinct_candidate_factor_exponent_B": fraction_record(Fraction(3, 4)),
        "marked_multiplicity_output_exponent_B": fraction_record(Fraction(3, 4)),
        "represented_aggregate_output_exponent_B": fraction_record(Fraction(9, 4)),
        "explicit_signed_target_grid_exponent_B": fraction_record(Fraction(7, 2)),
        "r177_pair_algebra_exponent_B": fraction_record(Fraction(9, 2)),
        "r177_explicit_marker_interpolation_exponent_B": fraction_record(Fraction(6)),
        "conditional_nonlocal_signed_norm_total_exponent_B": fraction_record(Fraction(9, 4)),
        "r163_label_and_backpointer_postprocessing_exponent_B": fraction_record(Fraction(2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "full_marker_is_distinct_ecdlp_primitive": False,
        "standard_explicit_routes_inside_rho": False,
        "conditional_nonlocal_signed_norm_strictly_inside_rho": True,
        "nonlocal_signed_norm_constructor_supplied": False,
        "standard_route_negative_claimed_as_circuit_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r177_report = json.loads(R177_REPORT.read_text())
    r174_report = json.loads(R174_REPORT.read_text())
    rows = [
        finite_control(curve, seed, r177_report, r174_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["aggregate_interpolation_exact"]
        and row["row_norm_zero_roots_match_r177"]
        and row["candidate_factor_equals_r177"]
        and row["candidate_factor_equals_r174_signed_norm"]
        and row["threshold_product_equals_r177_marker"]
        and row["first_threshold_equals_candidate_factor"]
        for row in rows
    )
    threshold_degree_sums = [
        sum(
            row["threshold_degrees"][level]
            for row in rows
            if level < len(row["threshold_degrees"])
        )
        for level in range(max(row["threshold_factor_count"] for row in rows))
    ]
    controls = {
        "schema": "p1553.m6_marked_fitting_signed_norm_dedup.controls.r178.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_aggregate_interpolations_exact": all(
            row["aggregate_interpolation_exact"] for row in rows
        ),
        "all_row_norm_zero_sets_match_r177": all(
            row["row_norm_zero_roots_match_r177"] for row in rows
        ),
        "all_candidate_factors_equal_r177": all(
            row["candidate_factor_equals_r177"] for row in rows
        ),
        "all_candidate_factors_equal_r174_signed_norm": all(
            row["candidate_factor_equals_r174_signed_norm"] for row in rows
        ),
        "all_threshold_products_equal_r177_markers": all(
            row["threshold_product_equals_r177_marker"] for row in rows
        ),
        "all_first_thresholds_equal_candidate_factors": all(
            row["first_threshold_equals_candidate_factor"] for row in rows
        ),
        "pair_evaluation_count": sum(row["pair_evaluation_count"] for row in rows),
        "row_norm_count": sum(row["row_norm_count"] for row in rows),
        "aggregate_output_slot_count": sum(
            row["aggregate_output_slot_count"] for row in rows
        ),
        "aggregate_nonzero_coefficient_count": sum(
            row["aggregate_nonzero_coefficient_count"] for row in rows
        ),
        "candidate_root_count": sum(row["candidate_root_count"] for row in rows),
        "candidate_factor_degree_sum": sum(
            row["candidate_factor_degree"] for row in rows
        ),
        "incidence_multiplicity_sum": sum(
            row["incidence_multiplicity_sum"] for row in rows
        ),
        "threshold_degree_sums": threshold_degree_sums,
        "threshold_degree_sum": sum(threshold_degree_sums),
        "maximum_incidence_multiplicity": max(
            row["maximum_incidence_multiplicity"] for row in rows
        ),
        "finite_pair_scan_and_interpolation_receive_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r177_r174_r171_r170_shoup_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "signed_row_norm_interpolation_complete": controls[
            "all_aggregate_interpolations_exact"
        ],
        "row_norm_zero_set_equals_r177_complete": controls[
            "all_row_norm_zero_sets_match_r177"
        ],
        "candidate_factor_equals_r177_complete": controls[
            "all_candidate_factors_equal_r177"
        ],
        "candidate_factor_equals_r174_signed_norm_complete": controls[
            "all_candidate_factors_equal_r174_signed_norm"
        ],
        "fitting_threshold_filtration_complete": controls[
            "all_threshold_products_equal_r177_markers"
        ],
        "first_threshold_is_candidate_factor_complete": controls[
            "all_first_thresholds_equal_candidate_factors"
        ],
        "multiplicity_layers_receive_no_locator_credit": True,
        "mechanism_level_deduplication_complete": all_exact,
        "r174_nN_standard_cost_charged": True,
        "r177_n2_and_marker_costs_charged": True,
        "represented_aggregate_output_charged": True,
        "r163_output_and_label_postprocessing_charged": True,
        "finite_controls_scoped_without_asymptotic_credit": True,
        "nonlocal_signed_norm_constructor_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    classification = (
        "ADMIT_MARKED_FITTING_FILTRATION_L_EQUALS_PRODUCT_G_R__G1_IS_GCD_U_"
        "SIGNED_ROW_NORM__SIX_EXACT_CONTROLS__THRESHOLD_DEGREES_140_71_23_7__"
        "G1_IDENTICAL_TO_R174_AND_R177_CANDIDATE_FACTOR__HIGHER_MULTIPLICITIES_"
        "ADD_NO_ECDLP_ROOTS__MARKED_FITTING_NOT_DISTINCT_ALGORITHMIC_LANE__"
        "UNIFIED_NONLOCAL_SIGNED_TRANSLATE_PRODUCT_OPEN__NO_CIRCUIT_LOWER_BOUND__"
        "NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Return to the unified nonlocal signed elliptic translate-product primitive: "
        "from compact U,V and the target divisor or principal witness emit the "
        "aggregate signed norm modulo U or G_1=gcd(U,C_h) in softly O(n+N) work. "
        "Reject nN target grids, N dense quotient-ring elements, n^2 pair or "
        "Fitting state, the full marked determinant body, candidate inversions, "
        "and unit-cost norm, resultant, multipoint, root, count, marginal, rank, "
        "source, or generic locator oracles. Higher multiplicity layers are optional "
        "diagnostics and receive no ECDLP attack credit."
    )
    passed = sum(obligations.values())
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether the R177 restricted-kernel marker is a distinct "
            "below-rho primitive or a multiplicity refinement of the existing "
            "R174 signed aggregate-norm candidate factor."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "r170_lambda_zero_fitting": {
                "fit": (
                    "R170 already deduplicates the unmarked lambda-zero Fitting "
                    "support against an aggregate target norm. R178 performs the "
                    "signed multiplicity-refined analogue for R177 and R174."
                )
            },
            "r171_balanced_miller_streaming": {
                "fit": (
                    "R171 shows that node-local Miller norm streaming telescopes "
                    "to the same target leaves at nN standard cost. R178 therefore "
                    "returns to its nonlocal batched translate-product primitive."
                )
            },
            "novelty_scope": (
                "The filtration identity is elementary in the reduced evaluation "
                "algebra and is used here for mechanism-level deduplication. No "
                "novel complexity theorem is claimed."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "marked_fitting_filtration_admitted": all_exact,
            "mechanism_level_deduplication_admitted": all_exact,
            "nonlocal_signed_norm_constructor_admitted": False,
            "lane_admitted": False,
        },
        "classification": classification,
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_marked_fitting_signed_norm_dedup.frozen.r178.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "deduplication": {
            "r177_distinct_ecdlp_lane": False,
            "surviving_primitive": "nonlocal_signed_elliptic_translate_product_mod_u",
            "required_total_work": "softly_O(n+N)",
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_marked_fitting_signed_norm_dedup.replay.r178.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "aggregate_polynomial_sha256": row["aggregate_polynomial_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "threshold_degrees": row["threshold_degrees"],
                "threshold_factor_sha256s": [
                    factor["factor_sha256"] for factor in row["threshold_factors"]
                ],
                "candidate_roots": row["candidate_roots"],
            }
            for row in rows
        ],
    }
    equivalence = {
        "schema": "p1553.m6_marked_fitting_signed_norm_dedup.equivalence.r178.v1",
        "signed_row_norm": theorem["signed_row_norm"],
        "fitting_filtration": theorem["fitting_filtration"],
        "locator_deduplication": theorem["locator_deduplication"],
        "global_threshold_degree_sums": threshold_degree_sums,
        "candidate_factor_degree_sum": controls["candidate_factor_degree_sum"],
        "marker_degree_sum": controls["threshold_degree_sum"],
        "controls": [
            {
                "control_id": row["control_id"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "threshold_factors": row["threshold_factors"],
            }
            for row in rows
        ],
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "equivalence": equivalence,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--equivalence-output", type=Path, default=DEFAULT_EQUIVALENCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    outputs = (
        (args.report_output, bundle["report"]),
        (args.frozen_output, bundle["frozen"]),
        (args.cost_output, bundle["cost"]),
        (args.replay_output, bundle["replay"]),
        (args.controls_output, bundle["controls"]),
        (args.equivalence_output, bundle["equivalence"]),
    )
    for path, value in outputs:
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
