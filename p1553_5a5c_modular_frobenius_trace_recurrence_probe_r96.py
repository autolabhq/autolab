#!/usr/bin/env python3
"""Audit the standard split-quotient Frobenius trace route after R95."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_modular_frobenius_trace_recurrence.r96.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SMALLER_ROOT_EXPONENT = Fraction(12, 5)
LARGER_ROOT_EXPONENT = Fraction(13, 5)
EXPLICIT_MATRIX_EXPONENT = 2 * SMALLER_ROOT_EXPONENT

R95_PRODUCER = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_r95.py"
)
R95_PRODUCER_SHA256 = (
    "494ed8e091bd965d235dfb5b3b350be2f9493eaeb1bc676229116259f59126f9"
)
R95_REPORT = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_report_r95.json"
)
R95_REPORT_SHA256 = (
    "50441deecd5fadcb05d30c09532d2b5fe6e8893ea2530243c20a52761e0a1742"
)
R95_GATE = pathlib.Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_gate_r95.md"
)
R95_GATE_SHA256 = (
    "6645a20b942f4c86abafed39fa14538b1bb80edee3ffa94a2e7d914e12c0c763"
)
R94_REPORT = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_report_r94.json"
)
R94_REPORT_SHA256 = (
    "a9966f1fb407e720ce1e9aaafc82ab4a38eade9f338cda287cb737445d099df0"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R84_GATE = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_gate_r84.md"
)
R84_GATE_SHA256 = (
    "4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661"
)
R90_REPORT = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_"
    "probe_report_r90.json"
)
R90_REPORT_SHA256 = (
    "02bece6fe25e335bd061eec784e237b6d9d2ab57f34bd8dd823217a137a56c2b"
)
R90_GATE = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_"
    "probe_gate_r90.md"
)
R90_GATE_SHA256 = (
    "b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b"
)
R9_GATE = pathlib.Path("p1553_projector_trace_router_gate_r9.md")
R9_GATE_SHA256 = (
    "400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81"
)

Polynomial = list[int]
Matrix = list[list[int]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R95_PRODUCER: R95_PRODUCER_SHA256,
        R95_REPORT: R95_REPORT_SHA256,
        R95_GATE: R95_GATE_SHA256,
        R94_REPORT: R94_REPORT_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R90_REPORT: R90_REPORT_SHA256,
        R90_GATE: R90_GATE_SHA256,
        R9_GATE: R9_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R96 source binding mismatch: {failures}")
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


def trim(polynomial: Polynomial, prime: int) -> Polynomial:
    result = [value % prime for value in polynomial]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def poly_mul(
    left: Polynomial,
    right: Polynomial,
    prime: int,
) -> Polynomial:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + left_value * right_value
            ) % prime
    return trim(result, prime)


def poly_mod(
    polynomial: Polynomial,
    modulus: Polynomial,
    prime: int,
) -> Polynomial:
    result = trim(polynomial, prime)
    modulus = trim(modulus, prime)
    if modulus[-1] != 1:
        inverse = pow(modulus[-1], -1, prime)
        modulus = [value * inverse % prime for value in modulus]
    while len(result) >= len(modulus):
        shift = len(result) - len(modulus)
        coefficient = result[-1]
        for index, value in enumerate(modulus):
            result[index + shift] = (
                result[index + shift] - coefficient * value
            ) % prime
        result = trim(result, prime)
    return result


def poly_pow_mod(
    polynomial: Polynomial,
    exponent: int,
    modulus: Polynomial,
    prime: int,
) -> Polynomial:
    result = [1]
    base = poly_mod(polynomial, modulus, prime)
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = poly_mod(
                poly_mul(result, base, prime),
                modulus,
                prime,
            )
        base = poly_mod(
            poly_mul(base, base, prime),
            modulus,
            prime,
        )
        remaining >>= 1
    return result


def polynomial_from_roots(
    roots: list[int],
    prime: int,
) -> Polynomial:
    result = [1]
    for root in roots:
        result = poly_mul(result, [-root, 1], prime)
    return result


def pad(polynomial: Polynomial, dimension: int) -> list[int]:
    return polynomial + [0] * (dimension - len(polynomial))


def multiplication_matrix(
    multiplier: Polynomial,
    modulus: Polynomial,
    prime: int,
) -> Matrix:
    dimension = len(modulus) - 1
    columns = []
    for degree in range(dimension):
        basis = [0] * degree + [1]
        column = poly_mod(
            poly_mul(multiplier, basis, prime),
            modulus,
            prime,
        )
        columns.append(pad(column, dimension))
    return [
        [columns[column][row] for column in range(dimension)]
        for row in range(dimension)
    ]


def identity_matrix(dimension: int) -> Matrix:
    return [
        [int(row == column) for column in range(dimension)]
        for row in range(dimension)
    ]


def matrix_mul(
    left: Matrix,
    right: Matrix,
    prime: int,
) -> Matrix:
    dimension = len(left)
    return [
        [
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(dimension)
            )
            % prime
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]


def matrix_pow(matrix: Matrix, exponent: int, prime: int) -> Matrix:
    result = identity_matrix(len(matrix))
    base = matrix
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = matrix_mul(result, base, prime)
        base = matrix_mul(base, base, prime)
        remaining >>= 1
    return result


def matrix_rank(matrix: Matrix, prime: int) -> int:
    work = [[value % prime for value in row] for row in matrix]
    row_index = 0
    for column in range(len(work[0]) if work else 0):
        pivot = next(
            (
                candidate
                for candidate in range(row_index, len(work))
                if work[candidate][column]
            ),
            None,
        )
        if pivot is None:
            continue
        work[row_index], work[pivot] = work[pivot], work[row_index]
        inverse = pow(work[row_index][column], -1, prime)
        work[row_index] = [
            value * inverse % prime for value in work[row_index]
        ]
        for other in range(len(work)):
            if other == row_index:
                continue
            multiplier = work[other][column]
            if multiplier:
                work[other] = [
                    (left - multiplier * right) % prime
                    for left, right in zip(
                        work[other],
                        work[row_index],
                    )
                ]
        row_index += 1
        if row_index == len(work):
            break
    return row_index


def matrix_trace(matrix: Matrix, prime: int) -> int:
    return sum(
        matrix[index][index] for index in range(len(matrix))
    ) % prime


def frobenius_matrix(
    modulus: Polynomial,
    prime: int,
) -> Matrix:
    dimension = len(modulus) - 1
    x_polynomial = [0, 1]
    columns = []
    for degree in range(dimension):
        column = poly_pow_mod(
            x_polynomial,
            degree * prime,
            modulus,
            prime,
        )
        columns.append(pad(column, dimension))
    return [
        [columns[column][row] for column in range(dimension)]
        for row in range(dimension)
    ]


def quotient_projector_trace(
    roots: list[int],
    zero_polynomial: Polynomial,
    prime: int,
) -> dict[str, Any]:
    modulus = polynomial_from_roots(roots, prime)
    multiplication = multiplication_matrix(
        zero_polynomial,
        modulus,
        prime,
    )
    nonzero_projector = matrix_pow(
        multiplication,
        prime - 1,
        prime,
    )
    dimension = len(roots)
    zero_projector = [
        [
            (
                int(row == column)
                - nonzero_projector[row][column]
            )
            % prime
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    trace = matrix_trace(zero_projector, prime)
    direct_count = sum(
        poly_mod(zero_polynomial, [-root, 1], prime) == [0]
        for root in roots
    )
    frobenius = frobenius_matrix(modulus, prime)
    return {
        "roots_with_occurrences": roots,
        "distinct_root_count": len(set(roots)),
        "quotient_dimension": dimension,
        "modulus": modulus,
        "zero_polynomial": zero_polynomial,
        "projector_trace_mod_prime": trace,
        "direct_integer_zero_count": direct_count,
        "dimension_below_field_prime": dimension < prime,
        "trace_equals_direct_mod_prime": (
            trace == direct_count % prime
        ),
        "trace_equals_direct_integer": (
            dimension < prime and trace == direct_count
        ),
        "frobenius_matrix": frobenius,
        "frobenius_rank": matrix_rank(frobenius, prime),
        "frobenius_is_identity": (
            frobenius == identity_matrix(dimension)
        ),
        "explicit_multiplication_matrix_words": dimension * dimension,
    }


def dyadic_quotient_source(
    roots: list[int],
    zero_polynomial: Polynomial,
    prime: int,
) -> dict[str, Any]:
    lower = 0
    upper = len(roots)
    root = quotient_projector_trace(
        roots,
        zero_polynomial,
        prime,
    )
    transcript = [
        {
            "range": [lower, upper],
            "quotient_dimension": root["quotient_dimension"],
            "zero_count": root["direct_integer_zero_count"],
        }
    ]
    queried_dimensions = root["quotient_dimension"]
    if root["direct_integer_zero_count"] == 0:
        return {
            "source_index": None,
            "all_trace_counts_exact": root[
                "trace_equals_direct_integer"
            ],
            "queried_dimension_sum": queried_dimensions,
            "transcript": transcript,
        }
    all_exact = root["trace_equals_direct_integer"]
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        left = quotient_projector_trace(
            roots[lower:midpoint],
            zero_polynomial,
            prime,
        )
        queried_dimensions += left["quotient_dimension"]
        all_exact &= left["trace_equals_direct_integer"]
        if left["direct_integer_zero_count"]:
            upper = midpoint
            selected_count = left["direct_integer_zero_count"]
        else:
            lower = midpoint
            selected = quotient_projector_trace(
                roots[lower:upper],
                zero_polynomial,
                prime,
            )
            queried_dimensions += selected["quotient_dimension"]
            all_exact &= selected["trace_equals_direct_integer"]
            selected_count = selected["direct_integer_zero_count"]
        transcript.append(
            {
                "range": [lower, upper],
                "quotient_dimension": upper - lower,
                "zero_count": selected_count,
            }
        )
    return {
        "source_index": lower,
        "source_root": roots[lower],
        "all_trace_counts_exact": all_exact,
        "returned_source_is_zero": (
            poly_mod(
                zero_polynomial,
                [-roots[lower], 1],
                prime,
            )
            == [0]
        ),
        "queried_dimension_sum": queried_dimensions,
        "root_dimension": len(roots),
        "query_dimension_sum_below_three_root_bodies": (
            queried_dimensions < 3 * len(roots)
        ),
        "transcript": transcript,
    }


@functools.lru_cache(maxsize=1)
def quotient_trace_controls() -> dict[str, Any]:
    prime = 11
    distinct_roots = [1, 2, 4, 7]
    occurrence_roots = [1, 1, 2, 4, 7]
    zero_polynomial = polynomial_from_roots([1, 4], prime)
    reduced = quotient_projector_trace(
        distinct_roots,
        zero_polynomial,
        prime,
    )
    nonreduced = quotient_projector_trace(
        occurrence_roots,
        zero_polynomial,
        prime,
    )
    blind = quotient_projector_trace(
        distinct_roots,
        [0, 1],
        prime,
    )
    source = dyadic_quotient_source(
        occurrence_roots,
        zero_polynomial,
        prime,
    )
    return {
        "field_prime": prime,
        "reduced_split_quotient": reduced,
        "nonreduced_occurrence_quotient": nonreduced,
        "blind_split_quotient": blind,
        "radicalization_loses_duplicate_zero_occurrence": (
            nonreduced["direct_integer_zero_count"]
            > reduced["direct_integer_zero_count"]
        ),
        "nonreduced_frobenius_loses_nilpotent_rank": (
            nonreduced["frobenius_rank"]
            < nonreduced["quotient_dimension"]
        ),
        "dyadic_source": source,
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "r84_root_source_sides": {
            "smaller_exponent_B": fraction_record(
                SMALLER_ROOT_EXPONENT
            ),
            "larger_exponent_B": fraction_record(
                LARGER_ROOT_EXPONENT
            ),
        },
        "source_complete_split_quotient": {
            "basis_dimension_exponent_B": fraction_record(
                SMALLER_ROOT_EXPONENT
            ),
            "inside_setup_cap": SMALLER_ROOT_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SMALLER_ROOT_EXPONENT <= ONLINE_CAP,
        },
        "explicit_multiplication_or_frobenius_matrix": {
            "state_exponent_B": fraction_record(
                EXPLICIT_MATRIX_EXPONENT
            ),
            "inside_setup_cap": EXPLICIT_MATRIX_EXPONENT <= SETUP_CAP,
        },
        "dyadic_range_tree": {
            "coefficient_state": "B^(12/5) polylog(B)",
            "changes_exponent": False,
            "inside_setup_cap": False,
        },
        "reduced_split_frobenius_compresses_state": False,
        "nonreduced_frobenius_preserves_source_multiplicity": False,
        "factored_trace_without_quotient_basis_supplied": False,
        "scope": (
            "charges the standard explicit split-quotient basis, matrices, "
            "and dyadic range quotients only; it is not a lower bound on a "
            "factored transposed trace or other succinct source index"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = quotient_trace_controls()
    costs = asymptotic_cost_control()
    reduced = controls["reduced_split_quotient"]
    nonreduced = controls["nonreduced_occurrence_quotient"]
    source = controls["dyadic_source"]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_modular_frobenius_"
            "trace_recurrence.r96.v1"
        ),
        "trace_identity": "Tr_A(1-M_h^(p-1))",
        "quotient": "A=F_p[x]/F(x)",
        "reduced_frobenius": "identity when F splits squarefree over F_p",
        "occurrence_policy": (
            "duplicate sources require nonradical multiplicity or tagged "
            "product factors"
        ),
        "caps": costs["caps"],
    }
    state_ledger = {
        "schema": (
            "p1553.frobenius_trace_state_transition_ledger.r96.v1"
        ),
        "finite_controls": controls,
        "asymptotic_cost": costs,
        "standard_quotient_trace_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.integer_lift_dyadic_source_replay.r96.v1"
        ),
        "integer_no_wrap_controls": {
            "reduced": reduced["trace_equals_direct_integer"],
            "nonreduced": nonreduced["trace_equals_direct_integer"],
        },
        "dyadic_source": source,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.quotient_trace_exceptional_controls.r96.v1"
        ),
        "duplicate_occurrence_replayed": controls[
            "radicalization_loses_duplicate_zero_occurrence"
        ],
        "nilpotent_frobenius_rank_loss_replayed": controls[
            "nonreduced_frobenius_loses_nilpotent_rank"
        ],
        "blind_zero_replayed": (
            controls["blind_split_quotient"][
                "direct_integer_zero_count"
            ]
            == 0
            and controls["blind_split_quotient"][
                "trace_equals_direct_integer"
            ]
        ),
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_source_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r96.v1",
        "standard_quotient_trace_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "nine_source_bindings_verified": len(bindings) == 9,
        "reduced_projector_trace_exact_mod_prime": reduced[
            "trace_equals_direct_mod_prime"
        ],
        "reduced_projector_trace_exact_integer": reduced[
            "trace_equals_direct_integer"
        ],
        "reduced_split_frobenius_is_identity": reduced[
            "frobenius_is_identity"
        ],
        "nonreduced_projector_trace_counts_multiplicity": nonreduced[
            "trace_equals_direct_integer"
        ],
        "radicalization_duplicate_loss_replayed": controls[
            "radicalization_loses_duplicate_zero_occurrence"
        ],
        "nonreduced_frobenius_rank_loss_replayed": controls[
            "nonreduced_frobenius_loses_nilpotent_rank"
        ],
        "finite_dyadic_trace_source_exact": (
            source["all_trace_counts_exact"]
            and source["returned_source_is_zero"]
        ),
        "dyadic_query_dimension_charged": source[
            "query_dimension_sum_below_three_root_bodies"
        ],
        "root_quotient_dimension_charged_as_B12O5": (
            costs["source_complete_split_quotient"][
                "basis_dimension_exponent_B"
            ]["exact"]
            == "12/5"
        ),
        "explicit_matrix_state_charged_as_B24O5": (
            costs["explicit_multiplication_or_frobenius_matrix"][
                "state_exponent_B"
            ]["exact"]
            == "24/5"
        ),
        "r84_r90_semantic_deduplication_recorded": True,
        "quotient_basis_inside_setup_cap": False,
        "quotient_basis_inside_online_cap": False,
        "explicit_matrix_inside_setup_cap": False,
        "factored_trace_without_quotient_basis_supplied": False,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_unranking_complete": False,
        "blind_zero_source_complete": exceptional[
            "blind_zero_replayed"
        ],
        "projective_infinity_source_complete": False,
        "proper_subsum_source_complete": False,
        "tangent_multiplicity_source_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "SPLIT_QUOTIENT_PROJECTOR_TRACE_EXACT__"
            "REDUCED_FROBENIUS_IDENTITY__NONREDUCED_FROBENIUS_LOSES_"
            "NILPOTENT_SOURCE_STATE__STANDARD_QUOTIENT_B12O5"
        ),
        "source_bindings": {
            "r95_producer": {
                "path": str(R95_PRODUCER),
                "sha256": R95_PRODUCER_SHA256,
            },
            "r95_report": {
                "path": str(R95_REPORT),
                "sha256": R95_REPORT_SHA256,
            },
            "r95_gate": {
                "path": str(R95_GATE),
                "sha256": R95_GATE_SHA256,
            },
            "r94_report": {
                "path": str(R94_REPORT),
                "sha256": R94_REPORT_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "r90_report": {
                "path": str(R90_REPORT),
                "sha256": R90_REPORT_SHA256,
            },
            "r90_gate": {
                "path": str(R90_GATE),
                "sha256": R90_GATE_SHA256,
            },
            "r9_gate": {
                "path": str(R9_GATE),
                "sha256": R9_GATE_SHA256,
            },
        },
        "novelty_scope": (
            "R96 is the first campaign receipt to instantiate the R95 "
            "projector as an exact quotient-algebra trace and measure the "
            "reduced versus nonreduced Frobenius transition."
        ),
        "quotient_trace_controls": controls,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_modular_frobenius_"
                "trace_recurrence.json"
            ),
            "state_ledger": (
                "frobenius_trace_state_and_transition_ledger.json"
            ),
            "source_replay": (
                "integer_lift_dyadic_source_replay.json"
            ),
            "exceptional": (
                "quotient_trace_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r96.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "This closes only the standard explicit split-quotient basis, "
            "multiplication/Frobenius matrices, and dyadic range quotients. "
            "It is not a lower bound on a factored transposed trace, modular "
            "character sum, or another succinct source-reporting circuit."
        ),
        "next_action": (
            "Construct or refute one factored transposed projector-trace "
            "functional over the compact A/C divisor circuits. It must "
            "compute the integer count and one nonzero dyadic child without "
            "materializing a B^(12/5) quotient basis, root polynomial, "
            "moment vector, or endpoint/source table; fit B^(9/4) setup and "
            "B^(5/4) fresh work/workspace; and replay blind-zero, duplicate, "
            "nilpotent, infinity, proper-subsum, tangent, and multiplicity "
            "branches without DLP labels or verifier oracles."
        ),
        "disposition": (
            "REJECT_STANDARD_SPLIT_QUOTIENT_FROBENIUS_TRACE_ONLY__"
            "PROJECTOR_TRACE_AND_DYADIC_TOY_SOURCE_EXACT__REDUCED_"
            "FROBENIUS_IDENTITY_NO_COMPRESSION__RADICAL_LOSES_DUPLICATE__"
            "NONREDUCED_FROBENIUS_LOSES_NILPOTENT_SOURCE_STATE__QUOTIENT_"
            "B12O5_AND_MATRIX_B24O5__FACTORED_TRANSPOSED_TRACE_OPEN__"
            "PROJECTIVE_AND_FULL_5A5C_SOURCE_INCOMPLETE__NO_RANK__NO_"
            "FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "state_ledger": state_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_modular_frobenius_trace_"
            "recurrence_probe_report_r96.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_modular_frobenius_"
            "trace_recurrence.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frobenius_trace_state_and_transition_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "integer_lift_dyadic_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "quotient_trace_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r96.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.state_ledger_output, bundle["state_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
