#!/usr/bin/env python3
"""Test scalar-pencil and ordinary displacement routes for the R168 trace."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_regularized_log_trace_displacement_rank.r169.v1"

R168_PRODUCER = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_r168.py"
R168_REPORT = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_report_r168.json"
R168_FROZEN = ROOT / "frozen_m6_log_derivative_elliptic_cauchy_trace.json"
R168_COST = ROOT / "m6_log_derivative_elliptic_cauchy_trace_cost_ledger.json"
R168_REPLAY = ROOT / "m6_log_derivative_elliptic_cauchy_trace_replay.json"
R168_CONTROLS = ROOT / "m6_log_derivative_elliptic_cauchy_trace_controls.json"
R168_TRACE = ROOT / "log_derivative_candidate_poles_and_trace_r168.json"
R168_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_log_derivative_elliptic_cauchy_trace_probe_r168.py"
R168_GATE = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_gate_r168.md"
R168_PARENT = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_parent_report_r168.yaml"
BOSTAN_DISPLACEMENT = ROOT / "references/bostan_jeannerod_mouilleron_schost_displacement_2017.pdf"
EAGEN_PAPER = ROOT / "references/eagen_ecip_weil_reciprocity_2022_596.pdf"
R167_REPORT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_report_r167.json"
R167_GATE = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_gate_r167.md"
R150_REPORT = ROOT / "p1553_m6_rational_convolution_subalgebra_rigidity_probe_report_r150.json"
R150_GATE = ROOT / "p1553_m6_rational_convolution_subalgebra_rigidity_probe_gate_r150.md"

SOURCE_BINDINGS = (
    ("r168_producer", R168_PRODUCER, "2bf35a1e4624d0832b710d36d25cdb50d5ff855793ed3bd3a80a42a3809b7400"),
    ("r168_report", R168_REPORT, "82fbb81b136af357d4a78e5ef7bf29e7ce97002b8a95a7c93b42cc0bda7ed2c3"),
    ("r168_frozen", R168_FROZEN, "150dc58ef81575191a86497a43ceef092603c8661339acaa587f91a3f9d43645"),
    ("r168_cost", R168_COST, "48bf265d3c6b19f72b0a242e63dec7aae75ac2026f641468d0e235c30f2d1412"),
    ("r168_replay", R168_REPLAY, "0b8d9231a701f5e85001245ffdd91e13a9bd79fc34162fa8b34109c6f0a76bae"),
    ("r168_controls", R168_CONTROLS, "e348e8a4b17d41775cbb6fed917f62f0c24b2fecec37abc5c59c9f9c85d490ab"),
    ("r168_trace", R168_TRACE, "12731e6627c812951b35f53fccfa1610a763f3bf14c745544ff255affbf8660d"),
    ("r168_test", R168_TEST, "05d7ec8b9add06c4ea452ed1f61e8c8e83113593c8e5c24af354ece36af293df"),
    ("r168_gate", R168_GATE, "7063a5f6ba686e38682ab6e29f386ab56e7fb56097e288949cce73e130b2fcae"),
    ("r168_parent", R168_PARENT, "390b4ec16a24b79742901f4d8a9894a23bf387edc80998010a0fc0128dc4d68c"),
    ("bostan_displacement_2017", BOSTAN_DISPLACEMENT, "e0bff4ecbd9309d4e9c1a050426546ca21ca8eba6d6097a5fca932f5578ecd05"),
    ("eagen_2022_596", EAGEN_PAPER, "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e"),
    ("r167_report", R167_REPORT, "1d2f009525ef9d54a0f538d3a0a8cefe8451d4d97933b91b31ed1bb5c0b47e3c"),
    ("r167_gate", R167_GATE, "41c4869bb231dfa2d0de1bb0d280aa34bd95903ed425792b80a84c762b845e3b"),
    ("r150_report", R150_REPORT, "598fd0b95622354f22bc6ce531fc02aa2b5f61a1578d06dae36fbd39086ae6d3"),
    ("r150_gate", R150_GATE, "1def188f9b0c5834889a937f7de9629f78b98c0e4f3b3249b1ba170a404483b6"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_report_r169.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_regularized_log_trace_displacement_rank.json"
DEFAULT_COST = ROOT / "m6_regularized_log_trace_displacement_rank_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_regularized_log_trace_displacement_rank_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_regularized_log_trace_displacement_rank_controls.json"
DEFAULT_PENCIL = ROOT / "regularized_trace_pencil_and_displacement_r169.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R168 = load_module("p1553_r168_for_r169", R168_PRODUCER)
R167 = R168.R167
R166 = R168.R166
R161 = R168.R161


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
        raise AssertionError(f"R169 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def modular_rank(matrix: list[list[int]], prime: int) -> int:
    if not matrix:
        return 0
    reduced = [[value % prime for value in row] for row in matrix]
    rank = 0
    for column in range(len(reduced[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(reduced))
                if reduced[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        inverse = pow(reduced[rank][column], -1, prime)
        reduced[rank] = [value * inverse % prime for value in reduced[rank]]
        for row in range(len(reduced)):
            if row == rank or not reduced[row][column]:
                continue
            scale = reduced[row][column]
            reduced[row] = [
                (left - scale * right) % prime
                for left, right in zip(reduced[row], reduced[rank])
            ]
        rank += 1
        if rank == len(reduced):
            break
    return rank


def local_leading_pair(witness: dict[str, Any], prime: int) -> tuple[int, int]:
    pole_order = int(witness["pole_order"])
    weight_index = {
        int(monomial["weight"]): index
        for index, monomial in enumerate(witness["basis"])
    }
    leading_index = weight_index[pole_order]
    leading = int(witness["coefficients"][leading_index])
    if witness["basis"][leading_index]["kind"] == "yx":
        leading = -leading % prime
    next_coefficient = 0
    if pole_order - 1 in weight_index:
        next_index = weight_index[pole_order - 1]
        next_coefficient = int(witness["coefficients"][next_index])
        if witness["basis"][next_index]["kind"] == "yx":
            next_coefficient = -next_coefficient % prime
    return leading % prime, next_coefficient % prime


def h_value(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    point: tuple[int, int] | None,
    prime: int,
) -> int:
    return R167.rational_value(numerator, denominator, point, prime)


def regularized_log_value(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    point: tuple[int, int] | None,
    scalar: int,
    curve: dict[str, Any],
) -> int:
    prime = int(curve["field_prime"])
    if point is None:
        numerator_lead, numerator_next = local_leading_pair(numerator, prime)
        denominator_lead, denominator_next = local_leading_pair(
            denominator, prime
        )
        pencil_lead = (numerator_lead + scalar * denominator_lead) % prime
        pencil_next = (numerator_next + scalar * denominator_next) % prime
        if not pencil_lead or not denominator_lead:
            raise ZeroDivisionError("regularized pencil is singular at infinity")
        return (
            pencil_next * pow(pencil_lead, -1, prime)
            - denominator_next * pow(denominator_lead, -1, prime)
        ) % prime
    numerator_value = R167.witness_value(numerator, point, prime)
    denominator_value = R167.witness_value(denominator, point, prime)
    numerator_derivative = R168.invariant_derivative_witness_value(
        numerator, point, curve
    )
    denominator_derivative = R168.invariant_derivative_witness_value(
        denominator, point, curve
    )
    pencil_value = (numerator_value + scalar * denominator_value) % prime
    pencil_derivative = (
        numerator_derivative + scalar * denominator_derivative
    ) % prime
    if not pencil_value or not denominator_value:
        raise ZeroDivisionError("regularized log kernel reached a nonunit")
    return (
        pencil_derivative * pow(pencil_value, -1, prime)
        - denominator_derivative * pow(denominator_value, -1, prime)
    ) % prime


def regularized_kernel_matrix(
    curve: dict[str, Any],
    selected: list[tuple[int, int]],
    signed_support: list[tuple[int, int]],
    numerator: dict[str, Any],
    denominator: dict[str, Any],
) -> tuple[int, list[list[int]]]:
    prime = int(curve["field_prime"])
    scalar = 1
    while scalar < prime:
        matrix: list[list[int]] = []
        try:
            for left in selected:
                matrix.append(
                    [
                        regularized_log_value(
                            numerator,
                            denominator,
                            R167.point_add(support_point, left, curve),
                            scalar,
                            curve,
                        )
                        for support_point in signed_support
                    ]
                )
        except ZeroDivisionError:
            scalar += 1
            continue
        return scalar, matrix
    raise AssertionError("unable to regularize the logarithmic pencil")


def polynomial_mul(left: list[int], right: list[int], prime: int) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + left_value * right_value
            ) % prime
    return result


def lambda_characteristic_polynomial(values: list[int], prime: int) -> list[int]:
    result = [1]
    for value in values:
        result = polynomial_mul(result, [value, 1], prime)
    return result


def lambda_valuation(poly: list[int]) -> int:
    return next(
        (index for index, coefficient in enumerate(poly) if coefficient),
        len(poly),
    )


def displacement_matrices(
    matrix: list[list[int]],
    selected: list[tuple[int, int]],
    signed_support: list[tuple[int, int]],
    prime: int,
) -> dict[str, list[list[int]]]:
    operators = {
        "x_sylvester_minus": lambda left, right: left[0] - right[0],
        "x_sylvester_plus": lambda left, right: left[0] + right[0],
        "y_sylvester_minus": lambda left, right: left[1] - right[1],
        "y_sylvester_plus": lambda left, right: left[1] + right[1],
        "x_stein": lambda left, right: 1 - left[0] * right[0],
        "y_stein": lambda left, right: 1 - left[1] * right[1],
    }
    return {
        name: [
            [
                coefficient(left, right) * matrix[row_index][column_index]
                % prime
                for column_index, right in enumerate(signed_support)
            ]
            for row_index, left in enumerate(selected)
        ]
        for name, coefficient in operators.items()
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r167_control = R167.finite_control(curve, seed)
    r168_control = R168.finite_control(curve, seed)
    _, divisor, _ = R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    signed_support = selected + [
        R167.point_negate(point, curve) for point in selected
    ]
    numerator = r167_control["numerator_witness"]
    denominator = r167_control["denominator_witness"]
    prime = int(curve["field_prime"])
    scalar, matrix = regularized_kernel_matrix(
        curve, selected, signed_support, numerator, denominator
    )
    raw_rank = modular_rank(matrix, prime)
    displacements = displacement_matrices(
        matrix, selected, signed_support, prime
    )
    displacement_ranks = {
        name: modular_rank(displaced, prime)
        for name, displaced in displacements.items()
    }
    x_power_ranks = {}
    for power in range(1, 5):
        displaced = [
            [
                pow((left[0] - right[0]) % prime, power, prime)
                * matrix[row_index][column_index]
                % prime
                for column_index, right in enumerate(signed_support)
            ]
            for row_index, left in enumerate(selected)
        ]
        x_power_ranks[str(power)] = modular_rank(displaced, prime)

    pencil_rows = []
    all_pencil_valuations_exact = True
    all_pencils_monic_full_degree = True
    for left in selected:
        values = [
            h_value(
                numerator,
                denominator,
                R167.point_add(support_point, left, curve),
                prime,
            )
            for support_point in signed_support
        ]
        polynomial = lambda_characteristic_polynomial(values, prime)
        valuation = lambda_valuation(polynomial)
        expected_multiplicity = sum(value == 0 for value in values)
        all_pencil_valuations_exact &= valuation == expected_multiplicity
        all_pencils_monic_full_degree &= (
            len(polynomial) == 2 * len(selected) + 1
            and polynomial[-1] == 1
        )
        pencil_rows.append(
            {
                "left_endpoint": R167.point_list(left),
                "lambda_valuation": valuation,
                "expected_candidate_multiplicity": expected_multiplicity,
                "candidate_pole": valuation > 0,
                "coefficient_sha256": sha256_json(polynomial),
            }
        )
    pencil_candidate_roots = sorted(
        int(row["left_endpoint"][0])
        for row in pencil_rows
        if row["candidate_pole"]
    )
    row_count = len(selected)
    return {
        "control_id": f"{curve['family_id']}_regularized_displacement_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": row_count,
        "target_witness_degree": r167_control["target_witness_degree"],
        "regularizing_scalar": scalar,
        "kernel_row_count": row_count,
        "kernel_column_count": len(signed_support),
        "kernel_rank": raw_rank,
        "kernel_full_row_rank": raw_rank == row_count,
        "displacement_ranks": displacement_ranks,
        "all_six_natural_displacements_full_row_rank": all(
            rank == row_count for rank in displacement_ranks.values()
        ),
        "x_sylvester_power_ranks": x_power_ranks,
        "all_four_x_sylvester_powers_full_row_rank": all(
            rank == row_count for rank in x_power_ranks.values()
        ),
        "regularized_kernel_sha256": sha256_json(matrix),
        "displacement_rank_transcript_sha256": sha256_json(
            {"natural": displacement_ranks, "x_powers": x_power_ranks}
        ),
        "lambda_characteristic_degree": 2 * row_count,
        "generic_lambda_sample_count_for_interpolation": 2 * row_count + 1,
        "lambda_coefficient_slots_for_all_selected_points": (
            row_count * (2 * row_count + 1)
        ),
        "all_lambda_pencils_monic_full_degree": all_pencils_monic_full_degree,
        "all_lambda_valuations_equal_candidate_multiplicity": (
            all_pencil_valuations_exact
        ),
        "pencil_candidate_roots": pencil_candidate_roots,
        "r168_candidate_roots": r168_control["candidate_roots"],
        "pencil_candidates_match_r168": pencil_candidate_roots
        == r168_control["candidate_roots"],
        "lambda_pencil_rows_sha256": sha256_json(pencil_rows),
        "finite_rank_receives_asymptotic_lower_bound_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def prefix_degree_sweep(curve: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    factor_base, divisor, target_records = R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    signed_support = selected + [
        R167.point_negate(point, curve) for point in selected
    ]
    generator = tuple(factor_base["generator"])
    prime = int(curve["field_prime"])
    rows = []
    for witness_degree in range(2, len(targets) + 1):
        prefix = targets[:witness_degree]
        target_sum = R167.point_sum(prefix, curve)
        if target_sum is None:
            raise AssertionError("prefix target sum reached infinity")
        common_zero = R167.point_negate(target_sum, curve)
        if common_zero is None or common_zero in prefix:
            raise AssertionError("prefix common zero is not admissible")
        numerator = R167.function_witness([*prefix, common_zero], curve)
        auxiliary = None
        for offset in range(1, 4097):
            auxiliary = R167.try_auxiliary_witness(
                curve,
                generator,
                divisor,
                prefix,
                target_sum,
                common_zero,
                numerator,
                offset,
            )
            if auxiliary is not None:
                break
        if auxiliary is None:
            raise AssertionError("prefix auxiliary divisor search failed")
        scalar, matrix = regularized_kernel_matrix(
            curve,
            selected,
            signed_support,
            numerator,
            auxiliary["denominator"],
        )
        displaced = displacement_matrices(
            matrix, selected, signed_support, prime
        )["x_sylvester_minus"]
        kernel_rank = modular_rank(matrix, prime)
        displacement_rank = modular_rank(displaced, prime)
        rows.append(
            {
                "family_id": curve["family_id"],
                "seed": seed,
                "witness_degree": witness_degree,
                "c3_divisor_degree": len(selected),
                "auxiliary_offset": auxiliary["offset"],
                "regularizing_scalar": scalar,
                "kernel_rank": kernel_rank,
                "x_sylvester_displacement_rank": displacement_rank,
                "kernel_full_row_rank": kernel_rank == len(selected),
                "x_sylvester_displacement_full_row_rank": (
                    displacement_rank == len(selected)
                ),
            }
        )
    return rows


def theorem_record() -> dict[str, Any]:
    return {
        "scalar_resolvent_pencil": (
            "For each selected P define chi_P(lambda)=product_(Q in S union "
            "-S)(h(Q+P)+lambda). Because every R167 h denominator is a unit "
            "on the finite controls, the lambda-adic valuation of chi_P at "
            "zero equals the number of translated numerator zeros, hence the "
            "R168 candidate multiplicity. The regularized logarithmic kernel "
            "Dlog((F_num+lambda F_den)/F_den) is defined away from the finite "
            "roots of this pencil."
        ),
        "pencil_materialization_boundary": (
            "Each chi_P has degree 2n. Generic scalar interpolation needs "
            "2n+1 values per P, and storing all coefficients or samples over n "
            "selected points costs Theta(n^2)=B^(9/2). The exact pencil is a "
            "candidate-safe Fitting interface, not a sub-rho constructor."
        ),
        "ordinary_displacement_test": (
            "For a regularizing lambda, form the n-by-2n matrix K_lambda with "
            "entries Dlog((F_num+lambda F_den)/F_den)(Q+P). R169 measures "
            "Sylvester displacements induced by diagonal x and y coordinates, "
            "their plus-sign variants, and Stein displacements 1-x_P x_Q and "
            "1-y_P y_Q. Full row displacement rank means these ordinary "
            "generators are not succinct on the tested controls."
        ),
        "scope": (
            "The finite full-rank results close only the tested diagonal x/y "
            "Sylvester and Stein operators and generic scalar-pencil "
            "materialization. They are not an arithmetic-circuit lower bound "
            "and do not exclude a custom elliptic companion displacement, a "
            "fraction-free subresultant, or another denominator-aware trace."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_regularized_log_trace_displacement_rank.cost.r169.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_witness_degree_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_log_witness_state_exponent_B": fraction_record(Fraction(5, 4)),
        "regularized_kernel_matrix_exponent_B": fraction_record(Fraction(9, 2)),
        "full_rank_displacement_generator_state_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "lambda_pencil_coefficient_or_sample_state_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "preferred_fraction_free_fitting_work_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "compact_log_witness_inside_rho": True,
        "regularized_matrix_inside_rho": False,
        "full_rank_generator_inside_rho": False,
        "generic_lambda_interpolation_inside_rho": False,
        "ordinary_diagonal_displacement_compression_observed": False,
        "custom_elliptic_companion_displacement_refuted": False,
        "fraction_free_fitting_subresultant_supplied": False,
        "finite_rank_receives_asymptotic_lower_bound_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    largest_curve = R161.R159.R82.FAMILIES[R161.R160.FAMILY_COUNT - 1]
    sweep_rows = [
        row
        for seed in R161.R160.SEEDS
        for row in prefix_degree_sweep(largest_curve, seed)
    ]
    all_pencils = all(
        row["all_lambda_pencils_monic_full_degree"]
        and row["all_lambda_valuations_equal_candidate_multiplicity"]
        and row["pencil_candidates_match_r168"]
        for row in rows
    )
    all_natural_full = all(
        row["all_six_natural_displacements_full_row_rank"] for row in rows
    )
    all_power_full = all(
        row["all_four_x_sylvester_powers_full_row_rank"] for row in rows
    )
    all_sweeps_full = all(
        row["kernel_full_row_rank"]
        and row["x_sylvester_displacement_full_row_rank"]
        for row in sweep_rows
    )
    controls = {
        "schema": "p1553.m6_regularized_log_trace_displacement_rank.controls.r169.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "degree_sweep_control_count": len(sweep_rows),
        "degree_sweep_family_id": largest_curve["family_id"],
        "all_lambda_pencils_exact": all_pencils,
        "all_regularized_kernels_full_row_rank": all(
            row["kernel_full_row_rank"] for row in rows
        ),
        "all_six_natural_displacements_full_row_rank": all_natural_full,
        "all_four_x_sylvester_powers_full_row_rank": all_power_full,
        "all_degree_sweep_kernels_and_x_displacements_full_row_rank": (
            all_sweeps_full
        ),
        "maximum_tested_c3_divisor_degree": max(
            row["c3_divisor_degree"] for row in rows
        ),
        "minimum_tested_witness_degree": min(
            row["witness_degree"] for row in sweep_rows
        ),
        "maximum_tested_witness_degree": max(
            row["witness_degree"] for row in sweep_rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_asymptotic_lower_bound_credit": False,
        "degree_sweep_controls": sweep_rows,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "sixteen_source_bindings_verified": len(actual_bindings) == 16,
        "r168_denominator_aware_trace_interface_inherited": True,
        "r167_target_divisor_witness_inherited": True,
        "bostan_displacement_definitions_and_cost_scope_bound": True,
        "eagen_logarithmic_pencil_scope_deduplicated": True,
        "r150_rational_convolution_lane_deduplicated": True,
        "scalar_resolvent_pencil_identity_complete": True,
        "lambda_valuation_candidate_multiplicity_biconditional_complete": all_pencils,
        "six_finite_lambda_pencil_controls_complete": len(rows) == 6,
        "all_finite_pencils_monic_full_degree": all_pencils,
        "all_finite_pencil_candidates_match_r168": all_pencils,
        "six_natural_displacement_operators_tested": True,
        "all_six_natural_displacements_full_row_rank": all_natural_full,
        "four_x_displacement_powers_tested": True,
        "all_four_x_displacement_powers_full_row_rank": all_power_full,
        "degree_two_through_eight_prefix_sweep_complete": len(sweep_rows) == 14,
        "all_prefix_sweep_displacements_full_row_rank": all_sweeps_full,
        "generic_lambda_n2_materialization_cost_charged": True,
        "full_rank_generator_n2_state_charged": True,
        "candidate_oracles_avoided": True,
        "finite_rank_scoped_without_lower_bound_credit": True,
        "custom_elliptic_companion_displacement_complete": False,
        "fraction_free_fitting_subresultant_mod_u_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "scalar_resolvent_fitting_interface_admitted": True,
        "ordinary_diagonal_displacement_route_closed_on_controls": True,
        "fraction_free_fitting_constructor_admitted": False,
        "lane_admitted": False,
    }
    pencil = {
        "schema": "p1553.m6_regularized_trace_pencil_and_displacement.r169.v1",
        "theorem": theorem,
        "finite_control_records": [
            {
                "control_id": row["control_id"],
                "regularizing_scalar": row["regularizing_scalar"],
                "kernel_rank": row["kernel_rank"],
                "displacement_ranks": row["displacement_ranks"],
                "lambda_characteristic_degree": row[
                    "lambda_characteristic_degree"
                ],
                "lambda_pencil_rows_sha256": row[
                    "lambda_pencil_rows_sha256"
                ],
            }
            for row in rows
        ],
        "degree_sweep_records": sweep_rows,
        "fraction_free_fitting_subresultant_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_regularized_log_trace_displacement_rank.replay.r169.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "regularized_kernel_sha256": row["regularized_kernel_sha256"],
                "displacement_rank_transcript_sha256": row[
                    "displacement_rank_transcript_sha256"
                ],
                "lambda_pencil_rows_sha256": row[
                    "lambda_pencil_rows_sha256"
                ],
            }
            for row in rows
        ],
        "degree_sweep_sha256": sha256_json(sweep_rows),
        "all_replay_invariants_pass": all_pencils
        and all_natural_full
        and all_power_full
        and all_sweeps_full,
    }
    frozen = {
        "schema": "p1553.m6_regularized_log_trace_displacement_rank.frozen.r169.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "U,V, compact h and Dh/h, the scalar resolvent pencil, all "
                "auxiliary logarithmic corrections, and public equality removals"
            ),
            "required_output": (
                "the lambda-zero Fitting/denominator factor modulo U by a "
                "fraction-free elliptic subresultant or a custom succinct "
                "elliptic companion operator"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "generic 2n+1 lambda samples per P, full-rank diagonal x/y "
                "displacement generator, nN or n^2 table, tensor quotient, "
                "candidate inversion, unit-cost Fitting/subresultant/trace/"
                "resultant/root/count/rank/source oracle"
            ),
            "open_primitive": (
                "fraction-free elliptic Fitting/subresultant modulo U or a "
                "proved low-rank custom elliptic companion displacement"
            ),
        },
    }
    next_action = (
        "Construct or refute a fraction-free elliptic Fitting/subresultant "
        "modulo U directly from F_num,F_den and their invariant derivatives, "
        "below B^(5/2) and preferably B^(9/4+o(1)). Do not interpolate the "
        "degree-2n lambda pencil or reuse the full-rank diagonal x/y "
        "displacements. A custom elliptic companion displacement is admissible "
        "only with an explicit low-rank generator and charged application cost."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Test whether scalar resolvent interpolation or ordinary x/y "
            "displacement structure makes the R168 denominator-aware trace "
            "sub-rho, while preserving an exact Fitting interface."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r168": (
                "R168 proves candidate poles in an additive trace. R169 "
                "regularizes that trace as a scalar pencil and tests the "
                "ordinary displacement route without weakening denominator "
                "semantics."
            ),
            "bostan_displacement_2017": (
                "The primary source defines Sylvester/Stein displacement rank "
                "and fast algorithms when a short generator is supplied. R169 "
                "finds no short generator for six natural diagonal x/y "
                "operators and attributes no lower bound to the source."
            ),
            "eagen_2022_596": (
                "Eagen motivates logarithmic linearization. R169 tests a "
                "candidate-safe scalar resolvent; it does not attribute an "
                "ECDLP algorithm or displacement claim to that paper."
            ),
            "r150": (
                "R150 closes a different fixed rational convolution algebra. "
                "R169 tests regularized elliptic addition kernels and does not "
                "inherit R150 as a general displacement lower bound."
            ),
            "r167": (
                "R167 supplies the compact target witness. R169 shows that "
                "compact witness degree does not yield ordinary diagonal "
                "displacement compression on the finite controls."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_EXACT_SCALAR_RESOLVENT_FITTING_INTERFACE__LAMBDA_VALUATION_"
            "EQUALS_R168_MULTIPLICITY__SIX_NATURAL_XY_SYLVESTER_STEIN_"
            "DISPLACEMENTS_FULL_ROW_RANK__DEGREE_TWO_THROUGH_EIGHT_SWEEP_FULL_"
            "RANK__GENERIC_PENCIL_AND_FULL_GENERATOR_B9O2__CUSTOM_ELLIPTIC_"
            "COMPANION_OR_FRACTION_FREE_SUBRESULTANT_OPEN__NO_RHO_SHOUP_"
            "BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "pencil": pencil,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--pencil-output", type=Path, default=DEFAULT_PENCIL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.pencil_output, bundle["pencil"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
