#!/usr/bin/env python3
"""Audit complex-to-finite-field uncertainty transfer for torus predicates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_sparse_fourier_predicate_transfer.r130.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C5_EXPONENT = Fraction(15, 4)
GROUP_ORDER_EXPONENT = Fraction(5)

R129_PRODUCER = pathlib.Path(
    "p1553_torus_c5_piecewise_selector_decision_dag_probe_r129.py"
)
R129_PRODUCER_SHA256 = (
    "0296c7a71284b8d376720791eb73a768fe6446ea90ac0ce29a7b6a3b9435f1f5"
)
R129_REPORT = pathlib.Path(
    "p1553_torus_c5_piecewise_selector_decision_dag_"
    "probe_report_r129.json"
)
R129_REPORT_SHA256 = (
    "3ff1da9d4eb76fe023432492ad0ad6452d1aa69dd51cea98e6469b3191de9402"
)
R129_FROZEN = pathlib.Path(
    "frozen_torus_c5_piecewise_selector_decision_dag.json"
)
R129_FROZEN_SHA256 = (
    "a10da885fbae4070e5a88591d53e1e716e40b26f31848661980aeb13a8e49e3d"
)
R129_COST = pathlib.Path(
    "torus_c5_piecewise_selector_decision_dag_cost_ledger.json"
)
R129_COST_SHA256 = (
    "2948aea2107d53d5c9b700f20611c28f28c3a3933378510c9d625b80272334b6"
)
R129_REPLAY = pathlib.Path(
    "torus_c5_piecewise_selector_decision_dag_replay.json"
)
R129_REPLAY_SHA256 = (
    "cb3f53214494107fe9b81cc73a11131da23743fa65c290d19d376236c4593bd4"
)
R129_CONTROLS = pathlib.Path(
    "torus_c5_piecewise_selector_decision_dag_controls.json"
)
R129_CONTROLS_SHA256 = (
    "72d32caf2ee1fa748fc90481cf8061bec62e37e6b61f598e83b77e841d39aaa0"
)
R129_LOGS = pathlib.Path("factor_logs_and_identical_descent_r129.json")
R129_LOGS_SHA256 = (
    "d4d28cc412d139ed9f9dc25a050b43cce225a9c0a87dca307b6a955f04a37a40"
)
R129_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_piecewise_selector_decision_dag_probe_r129.py"
)
R129_TEST_SHA256 = (
    "a8c34c8d3f173328390735d2cae1a5172436c1cd9b99712aaa99860e3903c621"
)
R129_GATE = pathlib.Path(
    "p1553_torus_c5_piecewise_selector_decision_dag_probe_gate_r129.md"
)
R129_GATE_SHA256 = (
    "35026763fe6805e22935bb8f62480609f67a1f7e0c303c67f3d302a14d853c32"
)
R129_PARENT = pathlib.Path(
    "p1553_torus_c5_piecewise_selector_decision_dag_"
    "probe_parent_report_r129.yaml"
)
R129_PARENT_SHA256 = (
    "b9f9d21e9572e9978511b34e01a90dc4b9df62087aa8f915342386610783970e"
)
TAO_PAPER = pathlib.Path(
    "references/tao_uncertainty_cyclic_prime_math0308286.pdf"
)
TAO_PAPER_SHA256 = (
    "244f4e79e667ee831e65d0cb0d8d42de4e354e83d576c65acaf391b0da519880"
)
FINITE_FIELD_CHEBOTAREV_PAPER = pathlib.Path(
    "references/emmrich_kunis_finite_field_chebotarev_2506.02947.pdf"
)
FINITE_FIELD_CHEBOTAREV_PAPER_SHA256 = (
    "5a3b29fbaf5bd6f4833de1a63272bb7510179a92dc9b855d69b8b3dcb92cd627"
)

GF2_10_DEGREE = 10
GF2_10_MODULUS = 0x409
GF2_10_MASK = (1 << GF2_10_DEGREE) - 1


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R129 = load_module("p1553_r129_for_r130", R129_PRODUCER)
R82 = R129.R82


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r129_producer", R129_PRODUCER, R129_PRODUCER_SHA256),
        ("r129_report", R129_REPORT, R129_REPORT_SHA256),
        ("r129_frozen", R129_FROZEN, R129_FROZEN_SHA256),
        ("r129_cost", R129_COST, R129_COST_SHA256),
        ("r129_replay", R129_REPLAY, R129_REPLAY_SHA256),
        ("r129_controls", R129_CONTROLS, R129_CONTROLS_SHA256),
        ("r129_logs", R129_LOGS, R129_LOGS_SHA256),
        ("r129_test", R129_TEST, R129_TEST_SHA256),
        ("r129_gate", R129_GATE, R129_GATE_SHA256),
        ("r129_parent", R129_PARENT, R129_PARENT_SHA256),
        ("tao_paper", TAO_PAPER, TAO_PAPER_SHA256),
        (
            "finite_field_chebotarev_paper",
            FINITE_FIELD_CHEBOTAREV_PAPER,
            FINITE_FIELD_CHEBOTAREV_PAPER_SHA256,
        ),
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
        raise AssertionError(f"R130 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def binary_poly_mod(value: int, modulus: int) -> int:
    modulus_degree = modulus.bit_length() - 1
    while value and value.bit_length() - 1 >= modulus_degree:
        value ^= modulus << (
            value.bit_length() - 1 - modulus_degree
        )
    return value


def binary_poly_gcd(left: int, right: int) -> int:
    while right:
        left, right = right, binary_poly_mod(left, right)
    return left


def gf2_10_mul(left: int, right: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        left <<= 1
        if left & (1 << GF2_10_DEGREE):
            left ^= GF2_10_MODULUS
    return result & GF2_10_MASK


def gf2_10_pow(value: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = gf2_10_mul(result, value)
        value = gf2_10_mul(value, value)
        exponent >>= 1
    return result


def gf2_10_inv(value: int) -> int:
    if value == 0:
        raise ZeroDivisionError("zero has no inverse")
    return gf2_10_pow(value, (1 << GF2_10_DEGREE) - 2)


def gf2_10_modulus_is_irreducible() -> bool:
    x_value = 0b10
    frobenius = x_value
    for step in range(1, GF2_10_DEGREE + 1):
        frobenius = gf2_10_mul(frobenius, frobenius)
        if (
            step <= GF2_10_DEGREE // 2
            and binary_poly_gcd(
                frobenius ^ x_value,
                GF2_10_MODULUS,
            )
            != 1
        ):
            return False
    return frobenius == x_value


def gf2_10_rank(matrix: Iterable[Iterable[int]]) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    rank = 0
    for column in range(len(rows[0])):
        pivot = next(
            (
                index
                for index in range(rank, len(rows))
                if rows[index][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = gf2_10_inv(rows[rank][column])
        rows[rank] = [
            gf2_10_mul(value, inverse) for value in rows[rank]
        ]
        for row_index, row in enumerate(rows):
            if row_index == rank or row[column] == 0:
                continue
            scale = row[column]
            rows[row_index] = [
                value ^ gf2_10_mul(scale, pivot_value)
                for value, pivot_value in zip(row, rows[rank])
            ]
        rank += 1
    return rank


def gf2_10_null_vector(
    matrix: Iterable[Iterable[int]],
) -> tuple[int, ...]:
    rows = [list(row) for row in matrix]
    if not rows:
        raise ValueError("matrix must be nonempty")
    column_count = len(rows[0])
    pivot_columns: list[int] = []
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(rank, len(rows))
                if rows[index][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = gf2_10_inv(rows[rank][column])
        rows[rank] = [
            gf2_10_mul(value, inverse) for value in rows[rank]
        ]
        for row_index, row in enumerate(rows):
            if row_index == rank or row[column] == 0:
                continue
            scale = row[column]
            rows[row_index] = [
                value ^ gf2_10_mul(scale, pivot_value)
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
    result = [0] * column_count
    result[free_column] = 1
    for row_index, pivot_column in reversed(
        list(enumerate(pivot_columns))
    ):
        result[pivot_column] = rows[row_index][free_column]
    return tuple(result)


def finite_field_uncertainty_counterexample() -> dict[str, Any]:
    if not gf2_10_modulus_is_irreducible():
        raise AssertionError("GF(2^10) modulus is reducible")
    group_order = 11
    multiplicative_order = (1 << GF2_10_DEGREE) - 1
    omega = next(
        value
        for candidate in range(2, 1 << GF2_10_DEGREE)
        if (
            (value := gf2_10_pow(
                candidate,
                multiplicative_order // group_order,
            ))
            != 1
            and gf2_10_pow(value, group_order) == 1
        )
    )
    zero_exponents = (0, 1, 2, 3, 7)
    mode_exponents = (0, 1, 2, 4, 7)
    matrix = tuple(
        tuple(
            gf2_10_pow(
                omega,
                (row * column) % group_order,
            )
            for column in mode_exponents
        )
        for row in zero_exponents
    )
    rank = gf2_10_rank(matrix)
    coefficients = gf2_10_null_vector(matrix)
    evaluations = []
    for exponent in range(group_order):
        value = 0
        for coefficient, mode in zip(coefficients, mode_exponents):
            value ^= gf2_10_mul(
                coefficient,
                gf2_10_pow(
                    omega,
                    (exponent * mode) % group_order,
                ),
            )
        evaluations.append(value)
    observed_zeros = tuple(
        index for index, value in enumerate(evaluations) if value == 0
    )
    return {
        "field": "GF(2^10)",
        "irreducible_modulus": "x^10+x^3+1",
        "irreducible_modulus_bits_hex": hex(GF2_10_MODULUS),
        "modulus_irreducible": True,
        "cyclic_subgroup_order": group_order,
        "primitive_root_encoding": omega,
        "primitive_root_has_exact_order_11": (
            omega != 1
            and gf2_10_pow(omega, group_order) == 1
        ),
        "zero_exponents": list(zero_exponents),
        "mode_exponents": list(mode_exponents),
        "mode_coefficients": list(coefficients),
        "minor_rank": rank,
        "minor_size": len(zero_exponents),
        "minor_singular": rank < len(zero_exponents),
        "evaluations": evaluations,
        "observed_zero_exponents": list(observed_zeros),
        "zero_set_exact": observed_zeros == zero_exponents,
        "nonzero_mode_count": sum(
            coefficient != 0 for coefficient in coefficients
        ),
        "zero_count": len(observed_zeros),
        "complex_sharp_zero_bound_violated_after_reduction": (
            len(observed_zeros)
            >= sum(coefficient != 0 for coefficient in coefficients)
        ),
        "candidate_discrete_logs_consumed": False,
        "verifier_exponents_used_only_for_exact_subgroup_control": True,
    }


def actual_pairing_field_transfer_controls() -> list[dict[str, Any]]:
    controls = []
    for curve in R82.FAMILIES:
        characteristic = curve["field_prime"]
        group_order = curve["subgroup_order"]
        residue = characteristic % group_order
        controls.append(
            {
                "family_id": curve["family_id"],
                "field_characteristic": characteristic,
                "pairing_subgroup_order": group_order,
                "cofactor_identity": (
                    characteristic == curve["cofactor"] * group_order - 1
                ),
                "characteristic_mod_subgroup_order": residue,
                "characteristic_is_minus_one_mod_subgroup_order": (
                    residue == group_order - 1
                ),
                "multiplicative_order_of_characteristic_mod_subgroup": 2,
                "order_two_verified": (
                    residue != 1
                    and pow(characteristic, 2, group_order) == 1
                ),
                "primitive_chebotarev_order_required": group_order - 1,
                "primitive_chebotarev_order_condition_passes": (
                    group_order - 1 == 2
                ),
                "subgroup_embeds_in_quadratic_extension": (
                    (characteristic * characteristic - 1)
                    % group_order
                    == 0
                ),
            }
        )
    return controls


def theorem_record() -> dict[str, Any]:
    return {
        "tao_complex_theorem": (
            "For nonzero f on Z/qZ with q prime and complex coefficients, "
            "|supp(f)|+|supp(fhat)|>=q+1."
        ),
        "complex_sparse_zero_consequence": (
            "A nonzero complex q-cyclic Fourier polynomial with s modes "
            "has at most s-1 zeros."
        ),
        "complex_color_predicate_consequence": (
            "A zero test vanishing on Theta(B^(15/4)) accepted targets "
            "needs B^(15/4+o(1)) complex modes."
        ),
        "complex_exact_indicator_consequence": (
            "An exact indicator supported on at most B^(15/4+o(1)) of "
            "q=B^(5+o(1)) points has B^(5+o(1)) Fourier support."
        ),
        "finite_field_transfer_requirement": (
            "The sharp support inequality requires nonvanishing minors of "
            "the q-by-q Fourier matrix in the circuit coefficient field."
        ),
        "finite_field_primary_source_boundary": (
            "The pinned finite-field Chebotarev sufficient theorem is "
            "stated in the primitive case ord_q(characteristic)=q-1; "
            "finite-characteristic transfer is not automatic."
        ),
        "actual_pairing_field_boundary": (
            "Every R82 family has characteristic=6q-1 and therefore "
            "ord_q(characteristic)=2, outside the primitive condition."
        ),
        "counterexample_boundary": (
            "GF(2^10) with q=11 admits a five-mode polynomial with five "
            "distinct subgroup zeros, so the complex sparse-zero "
            "consequence fails after finite-field reduction in general."
        ),
        "scope_limits": [
            "complex or characteristic-zero cyclic Fourier predicates",
            "finite-field transfer only after a field-specific all-minors proof",
            "sparse Fourier zero tests and exact Fourier indicators",
        ],
        "not_covered": [
            "actual order-two finite-field Fourier matrices",
            "non-Fourier shared-predicate decision DAGs",
            "high-degree low-SLP finite-field selectors",
            "adaptive cell-probe selectors",
            "general arithmetic-circuit or data-structure lower bounds",
        ],
    }


def finite_controls() -> dict[str, Any]:
    actual = actual_pairing_field_transfer_controls()
    counterexample = finite_field_uncertainty_counterexample()
    return {
        "schema": (
            "p1553.torus_c5_sparse_fourier_predicate_transfer_"
            "controls.r130.v1"
        ),
        "actual_pairing_fields": actual,
        "actual_pairing_field_count": len(actual),
        "all_actual_pairing_fields_have_order_two_characteristic": all(
            row["order_two_verified"] for row in actual
        ),
        "all_actual_pairing_fields_fail_primitive_order_condition": all(
            not row["primitive_chebotarev_order_condition_passes"]
            for row in actual
        ),
        "all_actual_subgroups_embed_in_quadratic_extension": all(
            row["subgroup_embeds_in_quadratic_extension"] for row in actual
        ),
        "finite_field_counterexample": counterexample,
        "counterexample_exact": (
            counterexample["modulus_irreducible"]
            and counterexample["primitive_root_has_exact_order_11"]
            and counterexample["minor_singular"]
            and counterexample["zero_set_exact"]
            and counterexample[
                "complex_sharp_zero_bound_violated_after_reduction"
            ]
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_sparse_fourier_predicate_transfer_"
            "cost_ledger.r130.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "complex_sparse_fourier_color_zero_test",
                "minimum_mode_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_or_query_cap": False,
                "transfers_to_actual_finite_fields": False,
                "candidate_work_credit": False,
            },
            {
                "route_id": "complex_exact_color_indicator",
                "minimum_mode_exponent_B": fraction_record(
                    GROUP_ORDER_EXPONENT
                ),
                "inside_setup_or_query_cap": False,
                "transfers_to_actual_finite_fields": False,
                "candidate_work_credit": False,
            },
            {
                "route_id": (
                    "primitive_finite_field_chebotarev_uncertainty_transfer"
                ),
                "required_order": "ord_q(characteristic)=q-1",
                "actual_order": 2,
                "condition_passes": False,
                "status": "inapplicable_to_actual_pairing_fields",
            },
            {
                "route_id": "order_two_finite_field_fourier_minor_theorem",
                "all_minors_theorem_supplied": False,
                "sparse_zero_lower_bound_supplied": False,
                "status": "open",
            },
            {
                "route_id": (
                    "nonfourier_shared_predicate_or_low_slp_selector_dag"
                ),
                "exact_circuit_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R129_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R129 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r129_shared_predicate_interface_inherited": (
            inherited["admission"]["scoped_turan_branch_negative_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "tao_complex_theorem_scope_frozen": (
            "complex coefficients" in theorem["tao_complex_theorem"]
        ),
        "finite_field_transfer_requirement_explicit": (
            "nonvanishing minors"
            in theorem["finite_field_transfer_requirement"]
        ),
        "four_actual_pairing_fields_checked": (
            controls["actual_pairing_field_count"] == 4
        ),
        "all_actual_characteristic_orders_equal_two": controls[
            "all_actual_pairing_fields_have_order_two_characteristic"
        ],
        "all_actual_fields_fail_primitive_condition": controls[
            "all_actual_pairing_fields_fail_primitive_order_condition"
        ],
        "all_actual_subgroups_embed_quadratically": controls[
            "all_actual_subgroups_embed_in_quadratic_extension"
        ],
        "gf2_10_modulus_irreducible": controls[
            "finite_field_counterexample"
        ]["modulus_irreducible"],
        "gf2_10_order11_root_exact": controls[
            "finite_field_counterexample"
        ]["primitive_root_has_exact_order_11"],
        "five_mode_five_zero_counterexample_exact": controls[
            "counterexample_exact"
        ],
        "unjustified_complex_to_finite_transfer_rejected": (
            not cost["routes"][0]["transfers_to_actual_finite_fields"]
            and not cost["routes"][1]["transfers_to_actual_finite_fields"]
        ),
        "actual_order_two_fourier_minor_theorem_complete": False,
        "inside_cap_shared_predicate_dag_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct a finite-field shared-predicate selector DAG directly, "
        "or prove a field-specific uncertainty/minor theorem for the actual "
        "ord_q(characteristic)=2 pairing families. It must avoid importing "
        "complex Chebotarev bounds without transfer, choose a valid C2 "
        "branch in polylogarithmic arbitrary-target work, return exact "
        "C2+C3 sources or an empty certificate, fit B^(9/4+o(1)) state, "
        "avoid field DLP, and include rank, logs, identical descent, memory, "
        "field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_sparse_fourier_predicate_"
            "transfer.r130.v1"
        ),
        "source_bindings": source_binding_records(),
        "required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
            "exact_empty_rejection_required": True,
            "five_projective_backpointers_required": True,
            "field_discrete_logarithms_allowed": False,
        },
        "closed_scoped_grammars": [
            "complex sparse Fourier color zero tests",
            "complex exact Fourier color indicators",
            "unproved reduction of complex Chebotarev to finite fields",
        ],
        "preserved_interface": (
            "order-two finite-field Fourier predicate theorem or direct "
            "non-Fourier shared-predicate low-SLP selector DAG"
        ),
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_sparse_fourier_predicate_transfer_"
            "replay.r130.v1"
        ),
        "actual_pairing_field_count": controls[
            "actual_pairing_field_count"
        ],
        "all_actual_characteristic_orders_equal_two": controls[
            "all_actual_pairing_fields_have_order_two_characteristic"
        ],
        "all_actual_fields_fail_primitive_condition": controls[
            "all_actual_pairing_fields_fail_primitive_order_condition"
        ],
        "finite_field_counterexample_exact": controls[
            "counterexample_exact"
        ],
        "actual_order_two_minor_theorem_constructed": False,
        "inside_cap_shared_predicate_dag_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r130.v1",
        "r129_piecewise_selector_branch_audit_complete": True,
        "r130_fourier_uncertainty_transfer_audit_complete": True,
        "inside_cap_target_specialized_source_index_complete": False,
        "relation_independence_theorem_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    classification = (
        "TAO_SHARP_UNCERTAINTY_CLOSES_COMPLEX_SPARSE_FOURIER_PREDICATES_"
        "ONLY__ALL_FOUR_ACTUAL_PAIRING_FIELDS_HAVE_ORD_Q_CHARACTERISTIC_"
        "TWO_AND_FAIL_PRIMITIVE_CHEBOTAREV_CONDITION__GF2E10_Q11_FIVE_"
        "MODE_FIVE_ZERO_COUNTEREXAMPLE_EXACT__FINITE_FIELD_TRANSFER_"
        "UNSUPPLIED__ORDER2_FOURIER_OR_NONFOURIER_SHARED_PREDICATE_DAG_"
        "OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_FINITE_FIELD_TRANSFER_CONTROLS_AND_SCOPED_COMPLEX_"
            "FOURIER_BOUND_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "actual_pairing_field_count": controls[
                "actual_pairing_field_count"
            ],
            "all_actual_characteristic_orders_equal_two": controls[
                "all_actual_pairing_fields_have_order_two_characteristic"
            ],
            "all_actual_fields_fail_primitive_condition": controls[
                "all_actual_pairing_fields_fail_primitive_order_condition"
            ],
            "five_mode_five_zero_counterexample_exact": controls[
                "counterexample_exact"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "complex_fourier_predicate_negative_admitted": True,
            "finite_field_transfer_rejection_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_sparse_fourier_predicate_transfer.json"
            ),
            "cost": (
                "torus_c5_sparse_fourier_predicate_"
                "transfer_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_sparse_fourier_predicate_transfer_replay.json"
            ),
            "controls": (
                "torus_c5_sparse_fourier_predicate_transfer_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r130.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Tao's sharp theorem is not transferred to the actual fields.",
            "The finite-field counterexample is a transfer control only.",
            "No order-two all-minors theorem is supplied.",
            "Non-Fourier and low-SLP selector DAGs remain open.",
            "Finite controls receive no asymptotic credit.",
            "No rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_COMPLEX_FOURIER_UNCERTAINTY_SCOPE_AND_EXACT_FINITE_"
            "FIELD_TRANSFER_COUNTEREXAMPLE_ONLY__REJECT_UNJUSTIFIED_"
            "COMPLEX_TO_FINITE_LOWER_BOUND__PRESERVE_ORDER2_FINITE_FIELD_"
            "FOURIER_AND_NONFOURIER_SHARED_PREDICATE_DAG__NO_LOCATOR__NO_"
            "RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_sparse_fourier_predicate_transfer_"
            "probe_report_r130.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_sparse_fourier_predicate_transfer.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_fourier_predicate_"
            "transfer_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_fourier_predicate_transfer_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_fourier_predicate_transfer_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r130.json"),
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
        f"R130 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
