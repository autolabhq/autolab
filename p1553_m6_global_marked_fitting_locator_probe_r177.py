#!/usr/bin/env python3
"""Extract a candidate locator from a globally marked Pontryagin norm."""

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
SCHEMA = "p1553.m6_global_marked_fitting_locator.r177.v1"

R176_PRODUCER = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_r176.py"
R176_REPORT = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_report_r176.json"
R176_FROZEN = ROOT / "frozen_m6_principal_target_pontryagin_resultant.json"
R176_COST = ROOT / "m6_principal_target_pontryagin_resultant_cost_ledger.json"
R176_REPLAY = ROOT / "m6_principal_target_pontryagin_resultant_replay.json"
R176_CONTROLS = ROOT / "m6_principal_target_pontryagin_resultant_controls.json"
R176_RESULTANT = ROOT / "principal_target_pontryagin_resultant_r176.json"
R176_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_principal_target_pontryagin_resultant_probe_r176.py"
R176_GATE = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_gate_r176.md"
R176_PARENT = ROOT / "p1553_m6_principal_target_pontryagin_resultant_probe_parent_report_r176.yaml"

R169_PRODUCER = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_r169.py"
R169_REPORT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_report_r169.json"
R169_FROZEN = ROOT / "frozen_m6_regularized_log_trace_displacement_rank.json"
R169_COST = ROOT / "m6_regularized_log_trace_displacement_rank_cost_ledger.json"
R169_REPLAY = ROOT / "m6_regularized_log_trace_displacement_rank_replay.json"
R169_CONTROLS = ROOT / "m6_regularized_log_trace_displacement_rank_controls.json"
R169_PENCIL = ROOT / "regularized_trace_pencil_and_displacement_r169.json"
R169_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_regularized_log_trace_displacement_rank_probe_r169.py"
R169_GATE = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_gate_r169.md"
R169_PARENT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_parent_report_r169.yaml"

R163_REPORT = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_report_r163.json"
R163_GATE = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_gate_r163.md"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r176_producer", R176_PRODUCER, "6d487f22177941a03c5533c0ee185456a4ea831b31df389e679f03783afe73ef"),
    ("r176_report", R176_REPORT, "6d00706a225c0e1e2bb087a20f2e20ca327acfa25067f63241bb4e7302521632"),
    ("r176_frozen", R176_FROZEN, "743d55aeb444a5b05d566b0178e3bb1d9582cef8d6d8d3127e5bb8e76d12c874"),
    ("r176_cost", R176_COST, "a76a53b4e0c0f59815617f329e36b61675b17e8fa050f173a361e688526fb8f5"),
    ("r176_replay", R176_REPLAY, "387040fa5d26ef5798f1db553632010b556e9b3c1ffce031e14336350b60d9a4"),
    ("r176_controls", R176_CONTROLS, "1fd6f58d2ae0f0fe16407af76bb16b22e740398b692595c3f58642a344e514ae"),
    ("r176_resultant", R176_RESULTANT, "967f07d990aa8296f3256daacd5632b8ad8c8410e5e1e94cefc351901bd1a5d2"),
    ("r176_test", R176_TEST, "dab2a234da7837967b7d7dc0483a257f30bde2a5e5c3f655cc747b591ad50f61"),
    ("r176_gate", R176_GATE, "a64140ac0bd7a0b05cc55edf69a4334c105992e799a1be0534f7b0cf78604a56"),
    ("r176_parent", R176_PARENT, "c9bd7f7782376a4b147efca899fd1ff4ab072edc029798f5e3d9d9e59a11bd3f"),
    ("r169_producer", R169_PRODUCER, "213b8d9ca7b900adef6af241efac79dbe8ca01a2283b3cd166ba56d8cfd7d36f"),
    ("r169_report", R169_REPORT, "6f9123923396d0c6478486c9a669cb73ff759db00fa8997954f5b4a35d11ce86"),
    ("r169_frozen", R169_FROZEN, "df7d50d0bf60835d4513950c0961c65cb8901c20afe3b4dcea75a5a9e488cbf5"),
    ("r169_cost", R169_COST, "bab32516614e139ed989a6d7186aa1daed08c8e2efb2815d1b273c3b1a685a1f"),
    ("r169_replay", R169_REPLAY, "f6538b5474971bb9dbdf594b3efd8eb3c47bf202b518942b35871e8407a27044"),
    ("r169_controls", R169_CONTROLS, "954308ce9d715fb0d904297923b0bb7316e7d1ff625831368a261e55daba5e5b"),
    ("r169_pencil", R169_PENCIL, "cc36dc603f9c28da1194a1b699a71f670566a40aad2af12038d8f18131648d18"),
    ("r169_test", R169_TEST, "fe149185aa0eddc72b896d48bf3ceadd05c6e5ab9502b9b809c4ab396badc2ac"),
    ("r169_gate", R169_GATE, "7aba9b4c16a35dc7db1f1ad5953ef50e597a882644260c028d8b6d7a412d185f"),
    ("r169_parent", R169_PARENT, "90778d88d7c268d7fdc5c4a5257a350e8f3b15cff5f527537bf9ca7710248e78"),
    ("r163_report", R163_REPORT, "0a35b4fee30c7abf4ba69232e0f7af65d48ec785aef832c934bf03d5366645ab"),
    ("r163_gate", R163_GATE, "3064a108bf063ab59a0991be6910b70bf3a2e498974add780f6a57e157a63212"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_global_marked_fitting_locator_probe_report_r177.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_global_marked_fitting_locator.json"
DEFAULT_COST = ROOT / "m6_global_marked_fitting_locator_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_global_marked_fitting_locator_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_global_marked_fitting_locator_controls.json"
DEFAULT_MARKER = ROOT / "global_marked_fitting_locator_r177.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R176 = load_module("p1553_r176_for_r177", R176_PRODUCER)
R167 = R176.R167
R175 = R176.R175
R161 = R176.R161


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


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R177 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def modular_product(values: list[int], prime: int) -> int:
    result = 1
    for value in values:
        result = result * value % prime
    return result


def normalize_monic(poly: list[int], prime: int) -> list[int]:
    poly = R176.R175.R174.poly_trim(poly, prime)
    if poly == [0]:
        raise AssertionError("cannot normalize the zero polynomial")
    return R161.poly_scale(poly, pow(poly[-1], -1, prime), prime)


def r176_control(family_id: str, seed: int) -> dict[str, Any]:
    report = json.loads(R176_REPORT.read_text())
    control_id = f"{family_id}_principal_pontryagin_seed{seed}"
    matches = [
        row
        for row in report["controls"]["controls"]
        if row["control_id"] == control_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one R176 control for {control_id}")
    return matches[0]


def truncated_lambda_coefficients(
    entries: list[tuple[int, int]],
    marker: int,
    precision: int,
    prime: int,
) -> tuple[list[int], int]:
    coefficients = [1] + [0] * precision
    active_degree = 0
    update_count = 0
    for kernel_value, left_x in entries:
        marker_value = (marker - left_x) % prime
        upper = min(active_degree + 1, precision)
        for degree in range(upper, 0, -1):
            coefficients[degree] = (
                coefficients[degree] * kernel_value
                + coefficients[degree - 1] * marker_value
            ) % prime
            update_count += 1
        coefficients[0] = coefficients[0] * kernel_value % prime
        update_count += 1
        active_degree = upper
    return coefficients, update_count


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r176 = r176_control(curve["family_id"], seed)
    r167 = R167.finite_control(curve, seed)
    _, divisor, _ = R175.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    prime = int(curve["field_prime"])
    entries: list[tuple[int, int]] = []
    zero_left_coordinates: list[int] = []
    nonzero_values: list[int] = []
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(selected):
        for right_index, right in enumerate(selected):
            pair_sum = R167.point_add(left, right, curve)
            kernel_value = R167.rational_value(
                r167["numerator_witness"],
                r167["denominator_witness"],
                pair_sum,
                prime,
            )
            left_x = int(left[0])
            entries.append((kernel_value, left_x))
            if kernel_value == 0:
                zero_left_coordinates.append(left_x)
            else:
                nonzero_values.append(kernel_value)
            pair_rows.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_x": left_x,
                    "kernel_value": kernel_value,
                }
            )

    kernel_dimension = len(zero_left_coordinates)
    if kernel_dimension != int(r176["principal_pair_incidence_count"]):
        raise AssertionError("marked kernel dimension differs from R176 incidence")
    multiplicities = Counter(zero_left_coordinates)
    candidate_roots = sorted(multiplicities)
    if candidate_roots != r176["principal_leaf_roots"]:
        raise AssertionError("marked kernel roots differ from R176")
    if kernel_dimension >= prime:
        raise AssertionError("finite marker interpolation exceeds the field")

    marker_polynomial = [1]
    for left_x in zero_left_coordinates:
        marker_polynomial = R161.poly_mul(
            marker_polynomial, [(-left_x) % prime, 1], prime
        )
    marker_polynomial = normalize_monic(marker_polynomial, prime)
    nonzero_pseudodeterminant = modular_product(nonzero_values, prime)
    if nonzero_pseudodeterminant == 0:
        raise AssertionError("nonzero kernel complement product vanished")
    expected_lowest_coefficient = R161.poly_scale(
        marker_polynomial, nonzero_pseudodeterminant, prime
    )

    interpolation_rows: list[dict[str, Any]] = []
    interpolation_points: list[tuple[int, int]] = []
    all_lower_coefficients_zero = True
    truncated_update_count = 0
    for marker in range(kernel_dimension + 1):
        coefficients, updates = truncated_lambda_coefficients(
            entries, marker, kernel_dimension, prime
        )
        lower_zero = all(value == 0 for value in coefficients[:kernel_dimension])
        all_lower_coefficients_zero &= lower_zero
        if not lower_zero:
            raise AssertionError("marked norm has a coefficient below kernel nullity")
        coefficient = coefficients[kernel_dimension]
        interpolation_points.append((marker, coefficient))
        truncated_update_count += updates
        interpolation_rows.append(
            {
                "marker": marker,
                "lower_coefficients_zero": lower_zero,
                "lowest_nonzero_lambda_degree": kernel_dimension,
                "lowest_coefficient": coefficient,
                "truncated_update_count": updates,
                "truncated_polynomial_sha256": sha256_json(coefficients),
            }
        )

    interpolated_coefficient = R161.interpolate(interpolation_points, prime)
    if interpolated_coefficient != expected_lowest_coefficient:
        raise AssertionError("interpolated marked coefficient failed exact formula")
    recovered_marker_polynomial = normalize_monic(
        interpolated_coefficient, prime
    )
    if recovered_marker_polynomial != marker_polynomial:
        raise AssertionError("marked Fitting polynomial recovery failed")

    candidate_factor = R161.poly_gcd(
        divisor["u"], recovered_marker_polynomial, prime
    )
    factor_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    if factor_roots != candidate_roots:
        raise AssertionError("marked Fitting gcd lost candidate roots")

    multiplicity_rows = [
        {"candidate_x": root, "incidence_multiplicity": multiplicities[root]}
        for root in candidate_roots
    ]
    return {
        "control_id": f"{curve['family_id']}_global_marked_fitting_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": len(selected),
        "target_witness_degree": int(r176["target_witness_degree"]),
        "pair_algebra_dimension": len(entries),
        "kernel_dimension": kernel_dimension,
        "r176_principal_pair_incidence_count": r176[
            "principal_pair_incidence_count"
        ],
        "kernel_dimension_matches_r176_incidence": True,
        "candidate_root_count": len(candidate_roots),
        "candidate_roots": candidate_roots,
        "r176_candidate_roots": r176["principal_leaf_roots"],
        "candidate_roots_match_r176": True,
        "maximum_candidate_incidence_multiplicity": max(multiplicities.values()),
        "candidate_multiplicities": multiplicity_rows,
        "nonzero_pseudodeterminant": nonzero_pseudodeterminant,
        "nonzero_pseudodeterminant_is_unit": True,
        "marker_polynomial_degree": R161.poly_degree(marker_polynomial),
        "marker_polynomial_sha256": sha256_json(marker_polynomial),
        "expected_lowest_coefficient_sha256": sha256_json(
            expected_lowest_coefficient
        ),
        "interpolated_lowest_coefficient_sha256": sha256_json(
            interpolated_coefficient
        ),
        "all_lower_lambda_coefficients_zero": all_lower_coefficients_zero,
        "lowest_lambda_degree_equals_kernel_dimension": True,
        "interpolated_lowest_coefficient_exact": True,
        "recovered_marker_polynomial_exact": True,
        "marker_interpolation_sample_count": len(interpolation_points),
        "truncated_lambda_update_count": truncated_update_count,
        "candidate_factor_degree": R161.poly_degree(candidate_factor),
        "candidate_factor_roots": factor_roots,
        "candidate_factor_roots_match_r176": True,
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "pair_transcript_sha256": sha256_json(pair_rows),
        "interpolation_transcript_sha256": sha256_json(interpolation_rows),
        "finite_pair_scan_and_marker_interpolation_receive_asymptotic_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "marked_global_norm": (
            "Let T_D be the reduced pair algebra of D x D, let K be multiplication "
            "by k(P,Q)=h(P+Q), and let X_1 be multiplication by x(P). Define "
            "F(A,lambda)=det(K+lambda*(A*I-X_1))=product_(P,Q) "
            "(h(P+Q)+lambda*(A-x(P)))."
        ),
        "lowest_lambda_coefficient": (
            "If M is the number of signed incidence pairs, then ord_lambda F=M. "
            "The lambda^M coefficient is pdet(K)*product_(h(P+Q)=0) "
            "(A-x(P)), where pdet(K) is the nonzero product on the complement "
            "of ker(K). It is a unit times a monic degree-M marker polynomial."
        ),
        "restricted_kernel_characteristic_polynomial": (
            "Because the finite pair algebra is reduced, K and X_1 are diagonal "
            "in the pair-evaluation basis. The monic marker polynomial is exactly "
            "det(A*I-X_1|ker(K)); each candidate x(P) occurs with its signed "
            "incidence multiplicity. This interpretation extends as a Fitting "
            "or generalized characteristic polynomial when a nonreduced branch "
            "is retained scheme-theoretically."
        ),
        "candidate_locator": (
            "The selected divisor U is squarefree. Therefore gcd(U,monic([lambda^M]F)) "
            "removes incidence multiplicities and returns exactly the distinct "
            "R176 candidate roots. No subset query or opposite-sign verification "
            "branch is required."
        ),
        "represented_boundary": (
            "The standard pair algebra has dimension n^2=B^(9/2), already above "
            "rho. Explicit symbolic truncation through lambda degree M scans n^2 "
            "factors and carries Theta(M^2) marker slots; marker interpolation "
            "costs Theta(n^2*M^2). The full bivariate determinant body has "
            "Theta(n^4) coefficient slots."
        ),
        "open_output_sensitive_fitting_constructor": (
            "The surviving primitive is a fraction-free marked Fitting or "
            "subresultant constructor that emits only M and the degree-M restricted-"
            "kernel characteristic polynomial from compact D and h in softly "
            "O(n+N+M) work, without materializing the n^2 pair algebra."
        ),
        "scope": (
            "R177 gives an exact direct locator interface and finite verification, "
            "not its below-rho constructor. The represented and explicit-truncation "
            "costs are standard-route negatives, not arithmetic-circuit lower bounds. "
            "No unconditional ECDLP, Pollard-rho, or Shoup improvement follows."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_global_marked_fitting_locator.cost.r177.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "r163_candidate_output_exponent_B": fraction_record(Fraction(3, 4)),
        "pair_algebra_dimension_exponent_B": fraction_record(Fraction(9, 2)),
        "candidate_marker_polynomial_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "full_bivariate_marked_norm_body_exponent_B": fraction_record(Fraction(9)),
        "explicit_symbolic_truncated_marker_state_exponent_B": fraction_record(Fraction(3, 2)),
        "explicit_pair_scan_exponent_B": fraction_record(Fraction(9, 2)),
        "explicit_marker_interpolation_work_exponent_B": fraction_record(Fraction(6)),
        "generic_pair_algebra_element_state_exponent_B": fraction_record(Fraction(9, 2)),
        "conditional_output_sensitive_fitting_total_exponent_B": fraction_record(Fraction(9, 4)),
        "r163_label_and_backpointer_postprocessing_exponent_B": fraction_record(Fraction(2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_pair_algebra_route_inside_rho": False,
        "explicit_truncated_marker_route_inside_rho": False,
        "conditional_output_sensitive_fitting_strictly_inside_rho": True,
        "output_sensitive_marked_fitting_constructor_supplied": False,
        "standard_route_negative_claimed_as_circuit_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["kernel_dimension_matches_r176_incidence"]
        and row["candidate_roots_match_r176"]
        and row["all_lower_lambda_coefficients_zero"]
        and row["lowest_lambda_degree_equals_kernel_dimension"]
        and row["interpolated_lowest_coefficient_exact"]
        and row["recovered_marker_polynomial_exact"]
        and row["candidate_factor_roots_match_r176"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_global_marked_fitting_locator.controls.r177.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_kernel_dimensions_match_r176_incidences": all(
            row["kernel_dimension_matches_r176_incidence"] for row in rows
        ),
        "all_candidate_roots_match_r176": all(
            row["candidate_roots_match_r176"] for row in rows
        ),
        "all_lower_lambda_coefficients_zero": all(
            row["all_lower_lambda_coefficients_zero"] for row in rows
        ),
        "all_lowest_lambda_degrees_equal_kernel_dimensions": all(
            row["lowest_lambda_degree_equals_kernel_dimension"] for row in rows
        ),
        "all_interpolated_lowest_coefficients_exact": all(
            row["interpolated_lowest_coefficient_exact"] for row in rows
        ),
        "all_recovered_marker_polynomials_exact": all(
            row["recovered_marker_polynomial_exact"] for row in rows
        ),
        "all_candidate_factor_roots_match_r176": all(
            row["candidate_factor_roots_match_r176"] for row in rows
        ),
        "pair_algebra_dimension_sum": sum(row["pair_algebra_dimension"] for row in rows),
        "kernel_dimension_sum": sum(row["kernel_dimension"] for row in rows),
        "candidate_root_count": sum(row["candidate_root_count"] for row in rows),
        "maximum_candidate_incidence_multiplicity": max(
            row["maximum_candidate_incidence_multiplicity"] for row in rows
        ),
        "marker_polynomial_degree_sum": sum(
            row["marker_polynomial_degree"] for row in rows
        ),
        "marker_interpolation_sample_count": sum(
            row["marker_interpolation_sample_count"] for row in rows
        ),
        "truncated_lambda_update_count": sum(
            row["truncated_lambda_update_count"] for row in rows
        ),
        "candidate_factor_degree_sum": sum(
            row["candidate_factor_degree"] for row in rows
        ),
        "finite_pair_scan_and_marker_interpolation_receive_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r176_r169_r163_shoup_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "global_marked_norm_identity_complete": all_exact,
        "kernel_nullity_equals_signed_incidence_count_complete": controls[
            "all_kernel_dimensions_match_r176_incidences"
        ],
        "lowest_lambda_valuation_complete": controls[
            "all_lowest_lambda_degrees_equal_kernel_dimensions"
        ],
        "all_lower_lambda_coefficients_vanish_complete": controls[
            "all_lower_lambda_coefficients_zero"
        ],
        "lowest_coefficient_product_formula_complete": controls[
            "all_interpolated_lowest_coefficients_exact"
        ],
        "restricted_kernel_characteristic_polynomial_complete": controls[
            "all_recovered_marker_polynomials_exact"
        ],
        "candidate_gcd_locator_complete": controls[
            "all_candidate_factor_roots_match_r176"
        ],
        "r176_signed_root_replay_complete": controls[
            "all_candidate_roots_match_r176"
        ],
        "pair_algebra_n2_cost_charged": True,
        "full_bivariate_n4_body_charged": True,
        "explicit_truncated_marker_cost_charged": True,
        "r163_output_and_label_postprocessing_charged": True,
        "finite_enumeration_scoped_without_asymptotic_credit": True,
        "output_sensitive_marked_fitting_constructor_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_GLOBAL_MARKED_LAMBDA_NORM__KERNEL_NULLITY_EQUALS_241_SIGNED_"
        "INCIDENCES__LOWEST_COEFFICIENT_IS_RESTRICTED_X1_CHARACTERISTIC_"
        "POLYNOMIAL__DEGREE_241_WITH_MULTIPLICITY__GCD_U_RETURNS_140_R176_"
        "ROOTS__NO_SUBSET_QUERIES__STANDARD_PAIR_ALGEBRA_N2_B9O2__FULL_BODY_"
        "N4_B9__EXPLICIT_TRUNCATION_B6__OUTPUT_SENSITIVE_MARKED_FITTING_OPEN__"
        "NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute a fraction-free output-sensitive marked Fitting or "
        "subresultant operator that accepts compact U,V and the degree-N principal "
        "target witness h, computes M=dim ker(K), and emits det(AI-X_1|ker K) in "
        "softly O(n+N+M) total work. Reject n^2 pair enumeration or tensor state, "
        "the full lambda/A determinant body, M^2 marker interpolation, generic "
        "n^2 matrix pencils, candidate inversions, and unit-cost Fitting, kernel, "
        "resultant, root, count, marginal, rank, source, or generic locator oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Replace R175 subset queries by one globally marked lambda-adic norm "
            "whose first nonzero coefficient directly emits the signed candidate "
            "locator with incidence multiplicities."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "r169_scalar_resolvent": {
                "fit": (
                    "R169 proves the lambda valuation separately for every selected "
                    "P and finds full ordinary displacement rank. R177 introduces "
                    "the global A-x(P) marker so one lowest coefficient aggregates "
                    "all candidate coordinates directly."
                )
            },
            "r176_principal_pontryagin_resultant": {
                "fit": (
                    "R176 supplies the exact signed kernel values h(P+Q) and its "
                    "represented n^2 boundary. R177 retains those values only as "
                    "finite controls and asks for their low-nullity Fitting output."
                )
            },
            "novelty_scope": (
                "The globally marked restricted-kernel locator formulation has not "
                "been established as novel by a complete literature review. No "
                "complexity result is attributed to the bound sources."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "global_marked_locator_identity_admitted": all_exact,
            "candidate_gcd_locator_admitted": all_exact,
            "output_sensitive_marked_fitting_constructor_admitted": False,
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
        "schema": "p1553.m6_global_marked_fitting_locator.frozen.r177.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The restricted-kernel characteristic polynomial can be emitted "
                "from factored D,h state in softly O(n+N+M) work."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route materializes n^2 pair values or tensor state, the full "
                "bivariate determinant, M^2 marker samples, or a generic n^2 pencil."
            ),
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_global_marked_fitting_locator.replay.r177.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "pair_transcript_sha256": row["pair_transcript_sha256"],
                "marker_polynomial_sha256": row["marker_polynomial_sha256"],
                "interpolated_lowest_coefficient_sha256": row[
                    "interpolated_lowest_coefficient_sha256"
                ],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "interpolation_transcript_sha256": row[
                    "interpolation_transcript_sha256"
                ],
                "candidate_roots": row["candidate_roots"],
            }
            for row in rows
        ],
    }
    marker = {
        "schema": "p1553.m6_global_marked_fitting_locator.marker.r177.v1",
        "marked_global_norm": theorem["marked_global_norm"],
        "lowest_lambda_coefficient": theorem["lowest_lambda_coefficient"],
        "restricted_kernel_characteristic_polynomial": theorem[
            "restricted_kernel_characteristic_polynomial"
        ],
        "candidate_locator": theorem["candidate_locator"],
        "controls": [
            {
                "control_id": row["control_id"],
                "pair_algebra_dimension": row["pair_algebra_dimension"],
                "kernel_dimension": row["kernel_dimension"],
                "candidate_multiplicities": row["candidate_multiplicities"],
                "marker_polynomial_sha256": row["marker_polynomial_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "candidate_factor_roots": row["candidate_factor_roots"],
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
        "marker": marker,
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
    parser.add_argument("--marker-output", type=Path, default=DEFAULT_MARKER)
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
        (args.marker_output, bundle["marker"]),
    )
    for path, value in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
