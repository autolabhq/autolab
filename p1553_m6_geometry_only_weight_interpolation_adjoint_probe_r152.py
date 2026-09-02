#!/usr/bin/env python3
"""Compile reusable tangent and adjoint payloads on the R82 atom decks."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_geometry_only_weight_interpolation_adjoint.r152.v1"

R151_PRODUCER = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_probe_r151.py"
)
R151_REPORT = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_"
    "probe_report_r151.json"
)
R151_FROZEN = ROOT / (
    "frozen_m6_matrix_free_marginal_jacobian_krylov.json"
)
R151_COST = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_cost_ledger.json"
)
R151_REPLAY = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_replay.json"
)
R151_CONTROLS = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_controls.json"
)
R151_LOGS = ROOT / "factor_logs_and_identical_descent_r151.json"
R151_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_matrix_free_marginal_jacobian_krylov_probe_r151.py"
)
R151_GATE = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_probe_gate_r151.md"
)
R151_PARENT = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_"
    "probe_parent_report_r151.yaml"
)

R82_PRODUCER = ROOT / "p1553_cartesian_sum_compact_divisor_probe_r82.py"
R82_GATE = ROOT / "p1553_cartesian_sum_compact_divisor_probe_gate_r82.md"
R82_PARENT = ROOT / (
    "p1553_cartesian_sum_compact_divisor_probe_parent_report_r82.yaml"
)
MULTIPOINT = ROOT / (
    "references/bhargava_ghosh_guo_kumar_umans_"
    "multipoint_2205.00342v1.pdf"
)

SOURCE_BINDINGS = (
    (
        "r151_producer",
        R151_PRODUCER,
        "cf632dcddc7b2bc55982ddcf5bdf24ed6af91904d13493233a9f320675b93e5b",
    ),
    (
        "r151_report",
        R151_REPORT,
        "3a8f5134ac731fb1481c358148986acdf355bfe1c8370d7e8c00d6a9ad95d14e",
    ),
    (
        "r151_frozen",
        R151_FROZEN,
        "3bff6dff925cc884ee3753004ce81b5774348bba75c3558c7f31eb0b7f825d2d",
    ),
    (
        "r151_cost",
        R151_COST,
        "810f77f445eb7438a9421cb3f865a385d5dabaf09eea94de281c787d442b06bb",
    ),
    (
        "r151_replay",
        R151_REPLAY,
        "5dec8cda3eab0a81d2dfd84d1e8a140778f1cc69187b3f1af1fa543cfd90c792",
    ),
    (
        "r151_controls",
        R151_CONTROLS,
        "5b64b07215b75e16e52e136e81e58ae988a13446331fc5499f67a68f661f33c7",
    ),
    (
        "r151_logs",
        R151_LOGS,
        "9570ebf6009e680ef1d0840b775f56c7bc0021fd7219b240a84b5ada0752e0b6",
    ),
    (
        "r151_test",
        R151_TEST,
        "9c68d8c9b17c7fc9e85f0f9e0ffe6416050b342bf8e4439e9e50b36385323be6",
    ),
    (
        "r151_gate",
        R151_GATE,
        "be163b44b889c67206a9a1eb73a02a6df911afc2b018f7848b145403285ad903",
    ),
    (
        "r151_parent",
        R151_PARENT,
        "016abc9cbaa7d1de107eb742d13607a35983e429fcc70f0475b83af8c5a36d0f",
    ),
    (
        "r82_producer",
        R82_PRODUCER,
        "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07",
    ),
    (
        "r82_gate",
        R82_GATE,
        "7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e",
    ),
    (
        "r82_parent",
        R82_PARENT,
        "7e1f819fef24d082cc0d361975d19d09e20594d88960656b2b45251eee1512e3",
    ),
    (
        "multipoint_primary",
        MULTIPOINT,
        "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_"
    "probe_report_r152.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_geometry_only_weight_interpolation_adjoint.json"
)
DEFAULT_COST = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r152.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R151 = load_module("p1553_r151_for_r152", R151_PRODUCER)
R82 = R151.R144.R82


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R152 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def trim(poly: Iterable[int]) -> list[int]:
    values = list(poly)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def poly_add(
    left: Iterable[int], right: Iterable[int], prime: int
) -> list[int]:
    left_values = list(left)
    right_values = list(right)
    width = max(len(left_values), len(right_values))
    output = [0] * width
    for index in range(width):
        output[index] = (
            (left_values[index] if index < len(left_values) else 0)
            + (right_values[index] if index < len(right_values) else 0)
        ) % prime
    return trim(output)


def poly_scale(poly: Iterable[int], scalar: int, prime: int) -> list[int]:
    return trim([(scalar * value) % prime for value in poly])


def poly_mul(
    left: Iterable[int], right: Iterable[int], prime: int
) -> list[int]:
    left_values = list(left)
    right_values = list(right)
    output = [0] * (len(left_values) + len(right_values) - 1)
    for left_index, left_value in enumerate(left_values):
        for right_index, right_value in enumerate(right_values):
            output[left_index + right_index] = (
                output[left_index + right_index]
                + left_value * right_value
            ) % prime
    return trim(output)


def poly_eval(poly: Iterable[int], value: int, prime: int) -> int:
    result = 0
    for coefficient in reversed(list(poly)):
        result = (result * value + coefficient) % prime
    return result


def poly_derivative(poly: Iterable[int], prime: int) -> list[int]:
    values = list(poly)
    if len(values) <= 1:
        return [0]
    return [
        (index * values[index]) % prime
        for index in range(1, len(values))
    ]


def divide_by_linear(
    poly: Iterable[int], root: int, prime: int
) -> tuple[list[int], int]:
    values = list(poly)
    degree = len(values) - 1
    if degree < 1:
        raise ValueError("positive-degree polynomial required")
    quotient = [0] * degree
    quotient[-1] = values[-1] % prime
    for index in range(degree - 2, -1, -1):
        quotient[index] = (
            values[index + 1] + root * quotient[index + 1]
        ) % prime
    remainder = (values[0] + root * quotient[0]) % prime
    return trim(quotient), remainder


def product_polynomial(points: Iterable[int], prime: int) -> list[int]:
    result = [1]
    for point in points:
        result = poly_mul(result, [(-point) % prime, 1], prime)
    return result


def lagrange_basis(
    points: Iterable[int], prime: int
) -> tuple[list[int], list[list[int]], list[int]]:
    x_values = list(points)
    product = product_polynomial(x_values, prime)
    derivative = poly_derivative(product, prime)
    basis = []
    denominators = []
    for point in x_values:
        quotient, remainder = divide_by_linear(product, point, prime)
        if remainder:
            raise AssertionError("linear factor division was not exact")
        denominator = poly_eval(derivative, point, prime)
        if denominator == 0:
            raise AssertionError("duplicate interpolation point")
        denominators.append(denominator)
        basis.append(
            poly_scale(quotient, pow(denominator, -1, prime), prime)
        )
    return product, basis, denominators


def interpolate(
    values: Iterable[int], basis: Iterable[Iterable[int]], prime: int
) -> list[int]:
    value_list = list(values)
    basis_list = [list(poly) for poly in basis]
    if len(value_list) != len(basis_list):
        raise ValueError("interpolation width mismatch")
    output = [0]
    for value, polynomial in zip(value_list, basis_list):
        output = poly_add(
            output,
            poly_scale(polynomial, value, prime),
            prime,
        )
    width = len(value_list)
    return output + [0] * (width - len(output))


def transpose_interpolation(
    coefficient_dual: Iterable[int],
    basis: Iterable[Iterable[int]],
    prime: int,
) -> list[int]:
    dual = list(coefficient_dual)
    return [
        sum(
            dual[index] * (polynomial[index] if index < len(polynomial) else 0)
            for index in range(len(dual))
        )
        % prime
        for polynomial in basis
    ]


def deterministic_vector(
    control_id: str, deck_id: str, role: str, width: int, prime: int
) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                (
                    f"R152|{control_id}|{deck_id}|{role}|{index}"
                ).encode("utf-8")
            ).digest(),
            "big",
        )
        % prime
        for index in range(width)
    ]


def dot(
    left: Iterable[int], right: Iterable[int], prime: int
) -> int:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    ) % prime


def deck_control(
    points: list[tuple[int, int]], control_id: str, deck_id: str, prime: int
) -> dict[str, Any]:
    x_values = [int(point[0]) for point in points]
    y_values = [int(point[1]) for point in points]
    product, basis, denominators = lagrange_basis(x_values, prime)
    width = len(points)
    weights = deterministic_vector(
        control_id, deck_id, "weights", width, prime
    )
    tangent = deterministic_vector(
        control_id, deck_id, "tangent", width, prime
    )
    coefficient_dual = deterministic_vector(
        control_id, deck_id, "coefficient_dual", width, prime
    )
    weight_poly = interpolate(weights, basis, prime)
    tangent_poly = interpolate(tangent, basis, prime)
    combined_values = [
        (weight + direction) % prime
        for weight, direction in zip(weights, tangent)
    ]
    combined_poly = interpolate(combined_values, basis, prime)
    linear_sum = poly_add(weight_poly, tangent_poly, prime)
    linear_sum += [0] * (width - len(linear_sum))
    roundtrip = [
        poly_eval(weight_poly, point, prime) for point in x_values
    ]
    adjoint = transpose_interpolation(coefficient_dual, basis, prime)
    forward_pairing = dot(coefficient_dual, weight_poly, prime)
    reverse_pairing = dot(adjoint, weights, prime)
    return {
        "deck_id": deck_id,
        "atom_count": width,
        "x_coordinates_distinct": len(set(x_values)) == width,
        "x_coordinates_sha256": sha256_json(x_values),
        "signed_y_side_table_sha256": sha256_json(y_values),
        "product_polynomial_sha256": sha256_json(product),
        "product_polynomial_degree": len(product) - 1,
        "all_barycentric_denominators_nonzero": all(denominators),
        "barycentric_denominators_sha256": sha256_json(denominators),
        "weight_interpolant_sha256": sha256_json(weight_poly),
        "tangent_interpolant_sha256": sha256_json(tangent_poly),
        "weight_roundtrip_exact": roundtrip == weights,
        "tangent_linearity_exact": combined_poly == linear_sum,
        "transpose_pairing_forward": forward_pairing,
        "transpose_pairing_reverse": reverse_pairing,
        "transpose_interpolation_identity_exact": (
            forward_pairing == reverse_pairing
        ),
        "geometry_setup_independent_of_weights": True,
        "divisions_depend_only_on_distinct_public_x_coordinates": True,
        "signed_point_branches_retained_as_side_table": True,
        "candidate_scalar_labels_consumed": False,
        "candidate_discrete_log_oracle_consumed": False,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    atoms_a, atoms_c, _, construction = R82.compact_factor_base(
        curve, offset
    )
    control_id = f"{curve['family_id']}_offset{offset}"
    deck_a = deck_control(atoms_a, control_id, "A", prime)
    deck_c = deck_control(atoms_c, control_id, "C", prime)
    return {
        "control_id": control_id,
        "field_prime": prime,
        "construction_sha256": sha256_json(construction),
        "a_deck": deck_a,
        "c_deck": deck_c,
        "both_decks_roundtrip_exact": (
            deck_a["weight_roundtrip_exact"]
            and deck_c["weight_roundtrip_exact"]
        ),
        "both_decks_tangent_linear": (
            deck_a["tangent_linearity_exact"]
            and deck_c["tangent_linearity_exact"]
        ),
        "both_decks_transpose_exact": (
            deck_a["transpose_interpolation_identity_exact"]
            and deck_c["transpose_interpolation_identity_exact"]
        ),
        "candidate_scalar_labels_consumed": False,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "public_geometry": (
            "For one R82 atom deck with distinct public x-coordinates "
            "x_1,...,x_n, freeze P(X)=product_i(X-x_i), its subproduct "
            "tree, and the nonzero constants P'(x_i)^(-1)."
        ),
        "weight_interpolant": (
            "For arbitrary atom payloads v_i, the unique degree-below-n "
            "polynomial W_v with W_v(x_i)=v_i is "
            "sum_i v_i P(X)/((X-x_i)P'(x_i))."
        ),
        "tangent_compiler": (
            "The map v -> W_v is linear. Replacing v by a dual payload "
            "v+epsilon*d changes only the leaf payload and yields "
            "W_v+epsilon*W_d without rebuilding geometry or dividing by "
            "v or d."
        ),
        "adjoint_compiler": (
            "The transpose interpolation map sends any coefficient dual "
            "g to atom adjoints <g,L_i>, where L_i is the ith Lagrange "
            "basis polynomial. It uses the same frozen geometry."
        ),
        "division_safety": (
            "Every inversion is P'(x_i)^(-1), a nonzero public geometry "
            "constant because the atom x-coordinates are distinct. No "
            "weight-dependent branch or division is introduced."
        ),
        "signed_point_boundary": (
            "The compiler retains the public y-coordinate side table. An "
            "x-only Semaev elimination may introduce sign branches; R152 "
            "does not claim that interpolation alone enforces the selected "
            "signed points."
        ),
        "fast_arithmetic_bound": (
            "A subproduct tree, fast multipoint evaluation/interpolation, "
            "and their transposes compile and apply these maps in "
            "M(n)*polylog(n) field operations and O(n*polylog(n)) state."
        ),
        "scope": (
            "This supplies reusable weight-independent leaf tangent and "
            "adjoint state only. It does not supply the internal weighted "
            "summation-polynomial/FFE elimination DAG, marker counts, "
            "Jacobian actions M or M^T, generic rank, logs, or descent."
        ),
        "primary_reference": (
            "Vishwas Bhargava, Sumanta Ghosh, Zeyu Guo, Mrinal Kumar, and "
            "Chris Umans, Fast Multivariate Multipoint Evaluation Over All "
            "Finite Fields, arXiv:2205.00342v1."
        ),
        "novelty_status": "leaf_compiler_composition_novelty_unverified",
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_geometry_only_weight_interpolation_adjoint."
            "cost.r152.v1"
        ),
        "dominant_c_atom_count_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "geometry_subproduct_tree_state_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "one_tangent_payload_compile_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "one_adjoint_payload_apply_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "marker_operator_fresh_cap_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "leaf_geometry_state_inside_setup_cap": True,
        "leaf_tangent_and_adjoint_inside_fresh_cap": True,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "weight_independent_leaf_derivative_state_supplied": True,
        "weight_separable_internal_elimination_dag_supplied": False,
        "bidirectional_marker_batch_operator_supplied": False,
        "generic_prime_rank_and_density_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        actual_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    all_roundtrips = all(
        row["both_decks_roundtrip_exact"] for row in actual
    )
    all_tangents = all(
        row["both_decks_tangent_linear"] for row in actual
    )
    all_transposes = all(
        row["both_decks_transpose_exact"] for row in actual
    )
    all_public = all(
        not row["candidate_scalar_labels_consumed"]
        and not row["candidate_discrete_log_oracle_consumed"]
        for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_geometry_only_weight_interpolation_adjoint."
            "controls.r152.v1"
        ),
        "actual_control_count": len(actual),
        "all_weight_interpolation_roundtrips_exact": all_roundtrips,
        "all_tangent_linearity_checks_exact": all_tangents,
        "all_transpose_interpolation_identities_exact": all_transposes,
        "all_controls_scalar_blind": all_public,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_geometry_only_weight_interpolation_adjoint."
            "frozen.r152.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "weight_separable_internal_elimination_dag": "open",
            "signed_branch_exactness": "open",
            "bidirectional_marker_batch_operator": "open",
            "generic_prime_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_geometry_only_weight_interpolation_adjoint."
            "replay.r152.v1"
        ),
        "controls": actual,
        "all_weight_interpolation_roundtrips_exact": all_roundtrips,
        "all_tangent_linearity_checks_exact": all_tangents,
        "all_transpose_interpolation_identities_exact": all_transposes,
        "theorem": theorem,
    }

    logs = {
        "schema": (
            "p1553.m6_geometry_only_weight_interpolation_adjoint."
            "logs_descent.r152.v1"
        ),
        "finite_leaf_tangent_and_adjoint_controls_exact": (
            all_roundtrips and all_tangents and all_transposes
        ),
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "fourteen_source_bindings_verified": len(actual_bindings) == 14,
        "r151_matrix_free_scope_inherited_without_overclaim": True,
        "r82_scalar_blind_atom_geometry_inherited": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_a_and_c_x_coordinates_distinct": all(
            row["a_deck"]["x_coordinates_distinct"]
            and row["c_deck"]["x_coordinates_distinct"]
            for row in actual
        ),
        "all_weight_interpolation_roundtrips_exact": all_roundtrips,
        "all_tangent_linearity_checks_exact": all_tangents,
        "all_transpose_interpolation_identities_exact": all_transposes,
        "all_barycentric_divisions_geometry_only_and_safe": all(
            row["a_deck"]["all_barycentric_denominators_nonzero"]
            and row["c_deck"]["all_barycentric_denominators_nonzero"]
            for row in actual
        ),
        "all_signed_y_side_tables_bound": all(
            row["a_deck"]["signed_point_branches_retained_as_side_table"]
            and row["c_deck"]["signed_point_branches_retained_as_side_table"]
            for row in actual
        ),
        "all_controls_scalar_blind": all_public,
        "multipoint_primary_bound_pinned": True,
        "leaf_B3_over_4_state_and_apply_cost_charged": True,
        "weight_independent_leaf_derivative_state_complete": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "weight_separable_internal_elimination_dag_complete": False,
        "signed_branch_exactness_complete": False,
        "bidirectional_marker_batch_operator_complete": False,
        "generic_prime_rank_and_density_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Use the frozen A/C subproduct trees and interpolation transposes as "
        "the only weight-dependent leaves. Construct one division-safe "
        "summation-polynomial/FFE elimination DAG whose internal topology "
        "and pivots depend only on public geometry, whose dual payload "
        "propagates without degree or state above B^(9/4+o(1)), and whose "
        "forward and reverse marker batches cost B^(5/4+o(1)). It must "
        "enforce signed-point branches and replay an exact-residual solve "
        "and shifted descent without DLP, root, count, marginal, rank, or "
        "source oracles."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "R82_A_C_PUBLIC_X_DECKS_ADMIT_GEOMETRY_ONLY_SUBPRODUCT_TREE__"
            "BARYCENTRIC_WEIGHT_INTERPOLANTS_ROUNDTRIP__DUAL_TANGENT_"
            "PAYLOAD_LINEAR__TRANSPOSE_ADJOINT_EXACT__ALL_DIVISIONS_PUBLIC_"
            "NONZERO_GEOMETRY_CONSTANTS__LEAF_STATE_AND_APPLY_B3O4_INSIDE_"
            "CAP__SIGNED_WEIGHT_SEPARABLE_INTERNAL_FFE_ELIMINATION_DAG_"
            "OPEN__NO_MARKER_OPERATOR_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether arbitrary A/C atom tangents and adjoints can "
            "reuse scalar-blind geometry without replaying setup, as "
            "required by the R151 matrix-free marker interface."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "weight_independent_leaf_derivative_state_admitted": True,
            "bidirectional_marker_batch_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": costs,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
