#!/usr/bin/env python3
"""Probe four-minor nonrigidity and its mode-claw construction cost."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_order_two_four_minor_claw.r140.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SUBGROUP_EXPONENT_B = Fraction(5)
GENERIC_COLLISION_EXPONENT_B = Fraction(5, 2)
FULL_SCAN_EXPONENT_B = Fraction(5)
MAX_FINITE_MODE_SCAN = 30_000
PROGRESSION_ROWS = (0, 1, 3, 4)

R139_PRODUCER = pathlib.Path(
    "p1553_torus_c5_order_two_three_minor_rigidity_probe_r139.py"
)
R139_PRODUCER_SHA256 = (
    "16f87b9249e368f80c7d85169c5a969ebb059e7d03dd0fbf84bf1951bf64ab18"
)
R139_REPORT = pathlib.Path(
    "p1553_torus_c5_order_two_three_minor_"
    "rigidity_probe_report_r139.json"
)
R139_REPORT_SHA256 = (
    "a6a63b4dc9dcff419103ed4db4884c23c53c8c4d8d2c607a9da7244a96a5a2ce"
)
R139_FROZEN = pathlib.Path(
    "frozen_torus_c5_order_two_three_minor_rigidity.json"
)
R139_FROZEN_SHA256 = (
    "b3bbc3eb99aad5c03e5feee56c90a979ec086361375d353a297ea72fa4837456"
)
R139_COST = pathlib.Path(
    "torus_c5_order_two_three_minor_rigidity_cost_ledger.json"
)
R139_COST_SHA256 = (
    "660ca8cca2de9f2887b764c6d6054d99c1fdd540e2573c1293be9142f3ecb7f4"
)
R139_REPLAY = pathlib.Path(
    "torus_c5_order_two_three_minor_rigidity_replay.json"
)
R139_REPLAY_SHA256 = (
    "96ba9fc10f4637ebc545ec9e1f39f8d2b687ed81548914bf8f0102713335fc17"
)
R139_CONTROLS = pathlib.Path(
    "torus_c5_order_two_three_minor_rigidity_controls.json"
)
R139_CONTROLS_SHA256 = (
    "7e033a78d9b351529b5427087e8a8da5a28ee5840a9bc61919f21206ee51b6b0"
)
R139_LOGS = pathlib.Path("factor_logs_and_identical_descent_r139.json")
R139_LOGS_SHA256 = (
    "0f080f3cbad1317f7e0084c810e1e34ef45a0fc6ebda3398256f14fca3964be6"
)
R139_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_order_two_three_minor_rigidity_probe_r139.py"
)
R139_TEST_SHA256 = (
    "04c617baf2efcab875d2f550b9e448cf89028960b5d990aae37bb53dcb944098"
)
R139_GATE = pathlib.Path(
    "p1553_torus_c5_order_two_three_minor_rigidity_probe_gate_r139.md"
)
R139_GATE_SHA256 = (
    "068211b726698d32ec0c3ad91ded28e814b2331a9b99a7c74a9327c9bfc8d76d"
)
R139_PARENT = pathlib.Path(
    "p1553_torus_c5_order_two_three_minor_"
    "rigidity_probe_parent_report_r139.yaml"
)
R139_PARENT_SHA256 = (
    "a495e6be54c566f2f38174798963b6cd59c12be11cda1f160a89843ad2508888"
)
R134_CONTROLS = pathlib.Path(
    "torus_c5_two_atom_geometric_progression_controls.json"
)
R134_CONTROLS_SHA256 = (
    "dfc7c9d0f13d21cd3bee1bba5a0dd7d727ef5df0fb7969ff66d9384293ea0a56"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R139 = load_module("p1553_r139_for_r140", R139_PRODUCER)
Field = R139.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r139_producer", R139_PRODUCER, R139_PRODUCER_SHA256),
        ("r139_report", R139_REPORT, R139_REPORT_SHA256),
        ("r139_frozen", R139_FROZEN, R139_FROZEN_SHA256),
        ("r139_cost", R139_COST, R139_COST_SHA256),
        ("r139_replay", R139_REPLAY, R139_REPLAY_SHA256),
        ("r139_controls", R139_CONTROLS, R139_CONTROLS_SHA256),
        ("r139_logs", R139_LOGS, R139_LOGS_SHA256),
        ("r139_test", R139_TEST, R139_TEST_SHA256),
        ("r139_gate", R139_GATE, R139_GATE_SHA256),
        ("r139_parent", R139_PARENT, R139_PARENT_SHA256),
        ("r134_controls", R134_CONTROLS, R134_CONTROLS_SHA256),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R140 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def fp2_sum(values: Iterable[Fp2], field: Field) -> Fp2:
    result = field.zero
    for value in values:
        result = field.add(result, value)
    return result


def mobius_image(value: Fp2, root: Fp2, field: Field) -> Fp2:
    denominator = fp2_sum((value, root, field.one), field)
    if denominator == field.zero:
        raise ZeroDivisionError("Mobius denominator is zero")
    numerator = fp2_sum(
        (field.mul(value, root), value, root),
        field,
    )
    return field.neg(field.mul(numerator, field.inv(denominator)))


def residual(
    mode_value: Fp2,
    row: int,
    root: Fp2,
    field: Field,
) -> Fp2:
    geometric_sum = fp2_sum(
        (field.pow(root, index) for index in range(row)),
        field,
    )
    return field.sub(
        field.sub(field.pow(mode_value, row), field.one),
        field.mul(
            geometric_sum,
            field.sub(mode_value, field.one),
        ),
    )


def four_minor_factor_certificate(
    root: Fp2,
    left_mode_value: Fp2,
    right_mode_value: Fp2,
    field: Field,
) -> dict[str, Any]:
    y = left_mode_value
    w = right_mode_value
    a_y = residual(y, 3, root, field)
    b_y = residual(y, 4, root, field)
    a_w = residual(w, 3, root, field)
    b_w = residual(w, 4, root, field)
    collision_numerator = field.sub(
        field.mul(b_y, a_w),
        field.mul(b_w, a_y),
    )
    linear_factor = fp2_sum(
        (
            field.mul(w, y),
            field.mul(w, root),
            w,
            field.mul(y, root),
            y,
            root,
        ),
        field,
    )
    factored_value = field.one
    for factor in (
        field.sub(y, w),
        field.sub(w, root),
        field.sub(w, field.one),
        field.sub(y, field.one),
        field.sub(y, root),
        linear_factor,
    ):
        factored_value = field.mul(factored_value, factor)
    return {
        "collision_numerator": field.json(collision_numerator),
        "factored_value": field.json(factored_value),
        "factorization_identity_exact": (
            collision_numerator == factored_value
        ),
        "mobius_linear_factor": field.json(linear_factor),
        "mobius_linear_factor_zero": linear_factor == field.zero,
        "all_excluded_factors_nonzero": all(
            factor != field.zero
            for factor in (
                field.sub(y, w),
                field.sub(w, root),
                field.sub(w, field.one),
                field.sub(y, field.one),
                field.sub(y, root),
            )
        ),
    }


def matrix_null_vector(
    matrix: Iterable[Iterable[Fp2]],
    field: Field,
) -> tuple[Fp2, ...]:
    rows = [list(row) for row in matrix]
    column_count = len(rows[0])
    pivot_columns: list[int] = []
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(rank, len(rows))
                if rows[index][column] != field.zero
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = field.inv(rows[rank][column])
        rows[rank] = [
            field.mul(value, inverse) for value in rows[rank]
        ]
        for row_index, row in enumerate(rows):
            if row_index == rank or row[column] == field.zero:
                continue
            scale = row[column]
            rows[row_index] = [
                field.sub(value, field.mul(scale, pivot_value))
                for value, pivot_value in zip(row, rows[rank])
            ]
        pivot_columns.append(column)
        rank += 1
    free_column = next(
        (
            column
            for column in range(column_count)
            if column not in pivot_columns
        ),
        None,
    )
    if free_column is None:
        raise ValueError("matrix has trivial nullspace")
    result = [field.zero] * column_count
    result[free_column] = field.one
    for row_index, pivot_column in reversed(
        list(enumerate(pivot_columns))
    ):
        result[pivot_column] = field.neg(
            rows[row_index][free_column]
        )
    return tuple(result)


def polynomial_value(
    value: Fp2,
    modes: tuple[int, ...],
    coefficients: tuple[Fp2, ...],
    field: Field,
) -> Fp2:
    return fp2_sum(
        (
            field.mul(coefficient, field.pow(value, mode))
            for coefficient, mode in zip(coefficients, modes)
        ),
        field,
    )


def find_known_mode_claw(
    root: Fp2,
    subgroup_order: int,
    field: Field,
    limit: int = MAX_FINITE_MODE_SCAN,
) -> dict[str, Any]:
    upper = min(subgroup_order - 1, limit)
    seen = {field.one: 0, root: 1}
    value = root
    for mode in range(2, upper + 1):
        value = field.mul(value, root)
        image = mobius_image(value, root, field)
        previous = seen.get(image)
        if previous is not None and previous not in (0, 1):
            return {
                "left_mode": previous,
                "right_mode": mode,
                "scanned_mode_count": mode - 1,
                "scan_limit": upper,
                "claw_found": True,
            }
        seen[value] = mode
    return {
        "left_mode": None,
        "right_mode": None,
        "scanned_mode_count": upper - 1,
        "scan_limit": upper,
        "claw_found": False,
    }


def progression_control(
    parent_control: dict[str, Any],
    witness: dict[str, Any],
) -> dict[str, Any]:
    field = Field(parent_control["field_prime"])
    subgroup_order = parent_control["subgroup_order"]
    root = tuple(witness["ratio"])
    claw = find_known_mode_claw(root, subgroup_order, field)
    if not claw["claw_found"]:
        raise AssertionError("bounded actual mode claw not found")
    left_mode = claw["left_mode"]
    right_mode = claw["right_mode"]
    if not isinstance(left_mode, int) or not isinstance(right_mode, int):
        raise AssertionError("invalid claw modes")
    modes = (0, 1, left_mode, right_mode)
    row_values = tuple(
        field.pow(root, row) for row in PROGRESSION_ROWS
    )
    matrix = R139.R135.evaluation_matrix(
        row_values,
        modes,
        field,
    )
    rank = R139.R135.fp2_matrix_rank(matrix, field)
    coefficients = matrix_null_vector(matrix, field)
    progression = tuple(
        field.pow(root, row) for row in range(6)
    )
    progression_values = tuple(
        polynomial_value(value, modes, coefficients, field)
        for value in progression
    )
    roots_in_progression = tuple(
        index
        for index, value in enumerate(progression_values)
        if value == field.zero
    )
    y = field.pow(root, left_mode)
    w = field.pow(root, right_mode)
    factor = four_minor_factor_certificate(root, y, w, field)
    t_y = mobius_image(y, root, field)
    t_w = mobius_image(w, root, field)
    selected_sources = [
        witness["sources"][index] for index in PROGRESSION_ROWS
    ]
    selected_targets = [
        witness["targets"][index] for index in PROGRESSION_ROWS
    ]
    return {
        "control_id": parent_control["control_id"],
        "color": witness["color"],
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "progression_ratio": field.json(root),
        "progression_rows": list(PROGRESSION_ROWS),
        "modes": list(modes),
        "claw": claw,
        "left_mode_value": field.json(y),
        "right_mode_value": field.json(w),
        "mobius_left_to_right": t_y == w,
        "mobius_right_to_left": t_w == y,
        "mobius_is_involution_on_claw": t_y == w and t_w == y,
        "left_mode_value_norm_one": (
            field.mul(y, field.pow(y, field.p)) == field.one
        ),
        "right_mode_value_norm_one": (
            field.mul(w, field.pow(w, field.p)) == field.one
        ),
        "factor_certificate": factor,
        "matrix_rank": rank,
        "matrix_singular": rank == 3,
        "kernel_coefficients": [
            field.json(value) for value in coefficients
        ],
        "all_kernel_coefficients_nonzero": all(
            value != field.zero for value in coefficients
        ),
        "kernel_annihilates_matrix": all(
            fp2_sum(
                (
                    field.mul(entry, coefficient)
                    for entry, coefficient in zip(row, coefficients)
                ),
                field,
            )
            == field.zero
            for row in matrix
        ),
        "progression_values": [
            field.json(value) for value in progression_values
        ],
        "roots_in_six_point_progression": list(roots_in_progression),
        "exactly_forced_four_progression_roots": (
            roots_in_progression == PROGRESSION_ROWS
        ),
        "selected_sources": selected_sources,
        "selected_targets": selected_targets,
        "parent_sources_replay": witness["all_sources_in_c5_replay"],
        "parent_sources_accepted": witness[
            "all_sources_accepted_by_color"
        ],
        "candidate_discrete_log_oracle_consumed": False,
        "finite_scan_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    parent = json.loads(R134_CONTROLS.read_text(encoding="utf-8"))
    controls = [
        progression_control(control, witness)
        for control in parent["controls"]
        for witness in control["color_witnesses"]
        if witness["witness_available"]
    ]
    return {
        "schema": (
            "p1553.torus_c5_order_two_four_minor_claw."
            "controls.r140.v1"
        ),
        "control_count": len(controls),
        "controls": controls,
        "all_bounded_mode_claws_found": all(
            row["claw"]["claw_found"] for row in controls
        ),
        "all_mobius_involutions_exact": all(
            row["mobius_is_involution_on_claw"] for row in controls
        ),
        "all_mode_values_norm_one": all(
            row["left_mode_value_norm_one"]
            and row["right_mode_value_norm_one"]
            for row in controls
        ),
        "all_factorization_identities_exact": all(
            row["factor_certificate"]["factorization_identity_exact"]
            for row in controls
        ),
        "all_mobius_linear_factors_zero": all(
            row["factor_certificate"]["mobius_linear_factor_zero"]
            for row in controls
        ),
        "all_excluded_factors_nonzero": all(
            row["factor_certificate"]["all_excluded_factors_nonzero"]
            for row in controls
        ),
        "all_matrices_rank_three": all(
            row["matrix_rank"] == 3 for row in controls
        ),
        "all_kernel_coefficients_nonzero": all(
            row["all_kernel_coefficients_nonzero"] for row in controls
        ),
        "all_kernels_exact": all(
            row["kernel_annihilates_matrix"] for row in controls
        ),
        "all_progression_root_sets_exactly_forced_four": all(
            row["exactly_forced_four_progression_roots"]
            for row in controls
        ),
        "all_parent_sources_replay_and_are_accepted": all(
            row["parent_sources_replay"]
            and row["parent_sources_accepted"]
            for row in controls
        ),
        "maximum_scanned_mode_count": max(
            row["claw"]["scanned_mode_count"] for row in controls
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls = finite_controls()
    obligations = {
        "eleven_source_bindings_verified": len(actual_bindings) == 11,
        "r139_three_minor_rigidity_inherited": True,
        "r134_actual_progressions_hash_bound": True,
        "four_minor_factorization_derived": True,
        "mobius_claw_formula_derived": True,
        "mobius_map_is_involution": controls[
            "all_mobius_involutions_exact"
        ],
        "order_two_frobenius_preserves_norm_one_image": controls[
            "all_mode_values_norm_one"
        ],
        "twelve_actual_progression_controls_complete": (
            controls["control_count"] == 12
        ),
        "all_actual_bounded_mode_claws_found": controls[
            "all_bounded_mode_claws_found"
        ],
        "all_actual_four_minors_have_rank_three": controls[
            "all_matrices_rank_three"
        ],
        "all_actual_kernel_vectors_exact": controls[
            "all_kernels_exact"
        ],
        "all_actual_progression_root_sets_exactly_four": controls[
            "all_progression_root_sets_exactly_forced_four"
        ],
        "full_spark_extension_to_four_columns_rejected": True,
        "finite_scans_receive_no_asymptotic_credit": True,
        "mode_encoding_dlp_boundary_explicit": True,
        "generic_collision_baseline_above_setup_cap": (
            GENERIC_COLLISION_EXPONENT_B > SETUP_CAP
        ),
        "subcap_asymptotic_known_mode_claw": False,
        "asymptotically_dense_atom_root_set": False,
        "inside_cap_four_mode_selector": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Exploit the explicit Mobius claw without recovering a hidden "
        "Fourier-mode discrete logarithm, prove a sub-q^(9/20) known-mode "
        "claw algorithm from non-generic structure, or move to a genuinely "
        "nonzero-value Frobenius-coordinate selector. Freeze every mode, "
        "coefficient, branch, source pointer, and empty path; fit "
        "B^(9/4+o(1)) state and polylogarithmic arbitrary-target work; "
        "avoid field DLP; and charge rank, logs, identical descent, memory, "
        "field operations, extension degree, and bits."
    )
    theorem = {
        "normalized_rows": list(PROGRESSION_ROWS),
        "normalized_modes": ["0", "1", "a", "b"],
        "residual_factorization": (
            "Let y=z^a and w=z^b. After eliminating modes 0 and 1, "
            "the determinant numerator factors as "
            "(y-w)(w-z)(w-1)(y-1)(y-z)"
            "(wy+wz+w+yz+y+z)."
        ),
        "mobius_claw": (
            "For distinct nontrivial modes, singularity is equivalent to "
            "w=T_z(y)=-(yz+y+z)/(y+z+1)."
        ),
        "involution": (
            "The fractional-linear matrix of T_z squares to "
            "(z^2+z+1)I, so T_z is an involution for prime q>3."
        ),
        "order_two_norm": (
            "If y^(p+1)=z^(p+1)=1, direct conjugation gives "
            "T_z(y)^p=T_z(y)^-1. Thus the image remains in the full "
            "norm-one group of order p+1=6q."
        ),
        "mode_encoding_boundary": (
            "A directly computed w in the order-q subgroup does not expose "
            "the integer mode b with w=z^b. Encoding X^b for arbitrary X "
            "therefore requires a field-subgroup discrete logarithm or a "
            "known-exponent claw search."
        ),
        "charged_generic_baseline": (
            "The standard birthday or generic DLP baseline is "
            "q^(1/2)=B^(5/2), above the B^(9/4) setup cap. This is a "
            "baseline, not a lower bound for every structured algorithm."
        ),
        "selector_consequence": (
            "Four-column full spark is false even on every available "
            "actual six-point color progression. The witnesses hit exactly "
            "four of those six points and supply neither asymptotically "
            "dense atom coverage nor a cap-compliant selector."
        ),
        "literature_novelty": "unverified",
    }
    frozen = {
        "schema": (
            "p1553.torus_c5_order_two_four_minor_claw."
            "frozen.r140.v1"
        ),
        "source_bindings": source_binding_records(),
        "interface": {
            "input": (
                "A generator z of the prime order q subgroup and known "
                "integer modes a,b."
            ),
            "claw_equation": (
                "z^b=-(z^(a+1)+z^a+z)/(z^a+z+1)"
            ),
            "output": (
                "A rank-three four-by-four Fourier minor and its unique "
                "four-term kernel polynomial."
            ),
            "finite_scan_limit": MAX_FINITE_MODE_SCAN,
            "asymptotic_credit": False,
        },
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "arbitrary_target_query_exponent_B": fraction_record(QUERY_CAP),
        },
        "required_open_outputs": failures,
    }
    cost = {
        "schema": (
            "p1553.torus_c5_order_two_four_minor_claw."
            "cost.r140.v1"
        ),
        "subgroup_order_exponent_B": fraction_record(
            SUBGROUP_EXPONENT_B
        ),
        "setup_cap_exponent_B": fraction_record(SETUP_CAP),
        "generic_birthday_or_dlp_exponent_B": fraction_record(
            GENERIC_COLLISION_EXPONENT_B
        ),
        "generic_baseline_inside_setup_cap": False,
        "full_known_mode_scan_exponent_B": fraction_record(
            FULL_SCAN_EXPONENT_B
        ),
        "full_scan_inside_setup_cap": False,
        "known_witness_evaluation_cost": "four fixed exponentiations",
        "known_witness_evaluation_polylogarithmic": True,
        "known_witness_selector_complete": False,
        "finite_maximum_scanned_mode_count": controls[
            "maximum_scanned_mode_count"
        ],
        "finite_scan_receives_asymptotic_credit": False,
        "candidate_field_dlp_charged_if_used": True,
        "candidate_field_dlp_used": False,
        "rank_cost_supplied": False,
        "factor_log_cost_supplied": False,
        "identical_descent_cost_supplied": False,
        "total_attack_cost_supplied": False,
    }
    replay_rows = [
        {
            "control_id": row["control_id"],
            "color": row["color"],
            "subgroup_order": row["subgroup_order"],
            "progression_rows": row["progression_rows"],
            "modes": row["modes"],
            "selected_sources": row["selected_sources"],
            "selected_targets": row["selected_targets"],
            "matrix_rank": row["matrix_rank"],
            "roots_in_six_point_progression": row[
                "roots_in_six_point_progression"
            ],
            "source_replay": row["parent_sources_replay"],
            "source_acceptance": row["parent_sources_accepted"],
        }
        for row in controls["controls"]
    ]
    replay = {
        "schema": (
            "p1553.torus_c5_order_two_four_minor_claw."
            "replay.r140.v1"
        ),
        "row_count": len(replay_rows),
        "rows": replay_rows,
        "all_sources_replay_and_are_accepted": controls[
            "all_parent_sources_replay_and_are_accepted"
        ],
        "all_four_minor_kernels_exact": controls["all_kernels_exact"],
        "all_root_sets_exactly_forced_four": controls[
            "all_progression_root_sets_exactly_forced_four"
        ],
    }
    logs_descent = {
        "schema": (
            "p1553.torus_c5_order_two_four_minor_claw."
            "logs_descent.r140.v1"
        ),
        "candidate_field_dlp_used": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "failures": failures,
        "four_column_full_spark_negative_admitted": True,
        "mobius_claw_factorization_admitted": True,
        "lane_admitted": False,
    }
    classification = (
        "ORDER_TWO_FOUR_COLUMN_FULL_SPARK_FAILS_ON_ALL_TWELVE_AVAILABLE_"
        "ACTUAL_SIX_POINT_COLOR_PROGRESSIONS__FOUR_MINOR_DETERMINANT_"
        "FACTORS_AS_AN_EXPLICIT_NORM_ONE_MOBIUS_CLAW__ALL_KERNELS_RANK_"
        "THREE_AND_HIT_EXACTLY_FOUR_PROGRESS_POINT_SOURCES__DIRECT_IMAGE_"
        "TO_INTEGER_MODE_NEEDS_FIELD_DLP_AND_STANDARD_KNOWN_MODE_COLLISION_"
        "BASELINE_B_FIVE_HALVES_EXCEEDS_SETUP_CAP__NO_DENSE_SELECTOR_RANK_"
        "LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": (
            "P1553-TORUS-C5-ORDER-TWO-FOUR-MINOR-CLAW-R140"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "verified_source_hashes": actual_bindings,
        "theorem": theorem,
        "controls_summary": {
            key: value
            for key, value in controls.items()
            if key not in ("controls",)
        },
        "cost_summary": cost,
        "admission": admission,
        "artifacts": {
            "frozen": "frozen_torus_c5_order_two_four_minor_claw.json",
            "cost": (
                "torus_c5_order_two_four_minor_claw_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_order_two_four_minor_claw_replay.json"
            ),
            "controls": (
                "torus_c5_order_two_four_minor_claw_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r140.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The witnesses refute four-column full spark but do not construct a complete selector.",
            "The B^(5/2) collision cost is a standard generic baseline, not a universal structured lower bound.",
            "The finite mode scans and constant four-of-six root sets receive no asymptotic credit.",
            "No general circuit, RAM, cell-probe, or four-mode root-density lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_FOUR_MINOR_MOBIUS_CLAW_NONRIGIDITY__REJECT_"
            "UNIVERSAL_FOUR_COLUMN_FULL_SPARK_EXTENSION__WITHHOLD_"
            "ASYMPTOTIC_SELECTOR_CREDIT__PRESERVE_SUBCAP_STRUCTURED_CLAW_"
            "AND_NONZERO_VALUE_ROUTES__NO_RANK__NO_LOGS__NO_DESCENT__"
            "NO_SHOUP__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_torus_c5_order_two_four_minor_"
            "claw_probe_report_r140.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_order_two_four_minor_claw.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_four_minor_claw_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_four_minor_claw_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_order_two_four_minor_claw_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r140.json"),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R140 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
