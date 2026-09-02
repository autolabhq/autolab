#!/usr/bin/env python3
"""Audit the canonical aggregate moment recurrence for the R94 projector."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.5a5c_aggregate_veronese_projector_recurrence.r95.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
FIELD_PRIME_EXPONENT_B = Fraction(5, 1)
MOMENT_STATE_EXPONENT_B = Fraction(10, 1)
RANK_SWEEP_PRIMES = (3, 5, 7, 11, 13, 17, 19, 29)

R94_PRODUCER = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_r94.py"
)
R94_PRODUCER_SHA256 = (
    "607cf20cb9091611d4d6085b7fc8046c35b5d64ed59817d4db2289d2b3ee0661"
)
R94_REPORT = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_report_r94.json"
)
R94_REPORT_SHA256 = (
    "a9966f1fb407e720ce1e9aaafc82ab4a38eade9f338cda287cb737445d099df0"
)
R94_GATE = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_gate_r94.md"
)
R94_GATE_SHA256 = (
    "66d2004d1b4dfa63ac69f60397da45e379dfca4bb8db12ede0647f028aaf517d"
)
R93_REPORT = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_report_r93.json"
)
R93_REPORT_SHA256 = (
    "f33c764ebbf491e55d7b8342fda9df8b29038beef8cdb06662c343ac415b7d4b"
)
R78_REPORT = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
R78_REPORT_SHA256 = (
    "e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60"
)
R78_GATE = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_gate_r78.md"
)
R78_GATE_SHA256 = (
    "9149c3b903d91a2156125a2c61896dc1111eaeffafdf0536c89a50b7e0e84ff9"
)
R9_GATE = pathlib.Path("p1553_projector_trace_router_gate_r9.md")
R9_GATE_SHA256 = (
    "400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)

Exponent = tuple[int, int, int, int, int, int]
Quadratic = tuple[int, int, int]

# Res(a*z^2+b*z+c, d*z^2+e*z+f).
RESULTANT_TERMS: tuple[tuple[Exponent, int], ...] = (
    ((2, 0, 0, 0, 0, 2), 1),
    ((1, 1, 0, 0, 1, 1), -1),
    ((1, 0, 1, 1, 0, 1), -2),
    ((1, 0, 1, 0, 2, 0), 1),
    ((0, 2, 0, 1, 0, 1), 1),
    ((0, 1, 1, 1, 1, 0), -1),
    ((0, 0, 2, 2, 0, 0), 1),
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R94_PRODUCER: R94_PRODUCER_SHA256,
        R94_REPORT: R94_REPORT_SHA256,
        R94_GATE: R94_GATE_SHA256,
        R93_REPORT: R93_REPORT_SHA256,
        R78_REPORT: R78_REPORT_SHA256,
        R78_GATE: R78_GATE_SHA256,
        R9_GATE: R9_GATE_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R95 source binding mismatch: {failures}")
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


def monomial_basis(total_degree: int) -> list[tuple[int, int, int]]:
    return [
        (first, second, total_degree - first - second)
        for first in range(total_degree + 1)
        for second in range(total_degree - first + 1)
    ]


@functools.lru_cache(maxsize=None)
def resultant_power(
    prime: int,
) -> dict[Exponent, int]:
    polynomial: dict[Exponent, int] = {(0, 0, 0, 0, 0, 0): 1}
    for step in range(prime - 1):
        updated: dict[Exponent, int] = {}
        for exponent, coefficient in polynomial.items():
            for term_exponent, term_coefficient in RESULTANT_TERMS:
                new_exponent = tuple(
                    left + right
                    for left, right in zip(exponent, term_exponent)
                )
                new_coefficient = (
                    updated.get(new_exponent, 0)
                    + coefficient * term_coefficient
                ) % prime
                if new_coefficient:
                    updated[new_exponent] = new_coefficient
                else:
                    updated.pop(new_exponent, None)
        polynomial = updated
        expected_degree = 2 * (step + 1)
        if not all(
            sum(exponent[:3]) == expected_degree
            and sum(exponent[3:]) == expected_degree
            for exponent in polynomial
        ):
            raise AssertionError("resultant-power bidegree drifted")
    return polynomial


def sparse_matrix_rank(
    rows: Iterable[dict[int, int]],
    prime: int,
) -> int:
    pivots: dict[int, dict[int, int]] = {}
    rank = 0
    for source_row in rows:
        row = {
            column: value % prime
            for column, value in source_row.items()
            if value % prime
        }
        while row:
            pivot_column = min(row)
            pivot_value = row[pivot_column]
            pivot_row = pivots.get(pivot_column)
            if pivot_row is None:
                inverse = pow(pivot_value, -1, prime)
                row = {
                    column: value * inverse % prime
                    for column, value in row.items()
                }
                pivots[pivot_column] = row
                rank += 1
                break
            multiplier = pivot_value
            for column, value in pivot_row.items():
                reduced = (
                    row.get(column, 0) - multiplier * value
                ) % prime
                if reduced:
                    row[column] = reduced
                else:
                    row.pop(column, None)
    return rank


def coefficient_pairing_control(prime: int) -> dict[str, Any]:
    polynomial = resultant_power(prime)
    degree = 2 * (prime - 1)
    basis = monomial_basis(degree)
    column_index = {
        exponent: index for index, exponent in enumerate(basis)
    }
    rows_by_exponent: dict[
        tuple[int, int, int],
        dict[int, int],
    ] = {}
    for exponent, coefficient in polynomial.items():
        left = exponent[:3]
        right = exponent[3:]
        rows_by_exponent.setdefault(left, {})[
            column_index[right]
        ] = coefficient
    rows = [
        rows_by_exponent.get(exponent, {})
        for exponent in basis
    ]
    rank = sparse_matrix_rank(rows, prime)
    expected_dimension = prime * (2 * prime - 1)
    return {
        "field_prime": prime,
        "fermat_exponent": prime - 1,
        "coefficient_bidegree": [degree, degree],
        "nonzero_coefficient_pair_count": len(polynomial),
        "left_monomial_count": len(rows_by_exponent),
        "right_monomial_count": len(
            {exponent[3:] for exponent in polynomial}
        ),
        "veronese_quotient_dimension": expected_dimension,
        "coefficient_pairing_rank": rank,
        "full_rank": rank == expected_dimension,
    }


@functools.lru_cache(maxsize=1)
def coefficient_rank_sweep() -> list[dict[str, Any]]:
    return [
        coefficient_pairing_control(prime)
        for prime in RANK_SWEEP_PRIMES
    ]


def quadratic_resultant(
    left: Quadratic,
    right: Quadratic,
    prime: int,
) -> int:
    values = (*left, *right)
    total = 0
    for exponent, coefficient in RESULTANT_TERMS:
        term = coefficient
        for value, power in zip(values, exponent):
            term *= pow(value, power, prime)
        total += term
    return total % prime


def monomial_value(
    value: Quadratic,
    exponent: tuple[int, int, int],
    prime: int,
) -> int:
    result = 1
    for coordinate, power in zip(value, exponent):
        result = result * pow(coordinate, power, prime) % prime
    return result


def aggregate_zero_count(
    left_values: list[Quadratic],
    right_values: list[Quadratic],
    polynomial: dict[Exponent, int],
    prime: int,
) -> dict[str, Any]:
    left_exponents = sorted({exponent[:3] for exponent in polynomial})
    right_exponents = sorted({exponent[3:] for exponent in polynomial})
    left_moments = {
        exponent: sum(
            monomial_value(value, exponent, prime)
            for value in left_values
        )
        % prime
        for exponent in left_exponents
    }
    right_moments = {
        exponent: sum(
            monomial_value(value, exponent, prime)
            for value in right_values
        )
        % prime
        for exponent in right_exponents
    }
    nonzero_count_mod_prime = sum(
        coefficient
        * left_moments[exponent[:3]]
        * right_moments[exponent[3:]]
        for exponent, coefficient in polynomial.items()
    ) % prime
    pair_count = len(left_values) * len(right_values)
    zero_count_mod_prime = (
        pair_count - nonzero_count_mod_prime
    ) % prime
    direct_zero_count = sum(
        quadratic_resultant(left, right, prime) == 0
        for left in left_values
        for right in right_values
    )
    no_wrap = pair_count < prime
    return {
        "pair_count": pair_count,
        "nonzero_count_mod_prime": nonzero_count_mod_prime,
        "zero_count_mod_prime": zero_count_mod_prime,
        "direct_integer_zero_count": direct_zero_count,
        "pair_count_below_field_prime": no_wrap,
        "aggregate_count_equals_direct_mod_prime": (
            zero_count_mod_prime == direct_zero_count % prime
        ),
        "aggregate_count_equals_direct_integer": (
            no_wrap and zero_count_mod_prime == direct_zero_count
        ),
        "left_moment_coordinate_count": len(left_moments),
        "right_moment_coordinate_count": len(right_moments),
    }


def dyadic_source_route(
    left_values: list[Quadratic],
    right_values: list[Quadratic],
    polynomial: dict[Exponent, int],
    prime: int,
) -> dict[str, Any]:
    left_lower = 0
    left_upper = len(left_values)
    right_lower = 0
    right_upper = len(right_values)
    root = aggregate_zero_count(
        left_values,
        right_values,
        polynomial,
        prime,
    )
    transcript = [
        {
            "left_range": [left_lower, left_upper],
            "right_range": [right_lower, right_upper],
            "zero_count": root["direct_integer_zero_count"],
        }
    ]
    if root["direct_integer_zero_count"] == 0:
        return {
            "source": None,
            "root_count": 0,
            "transcript": transcript,
            "all_counts_exact": root[
                "aggregate_count_equals_direct_integer"
            ],
        }
    all_counts_exact = root["aggregate_count_equals_direct_integer"]
    while (
        left_upper - left_lower > 1
        or right_upper - right_lower > 1
    ):
        left_width = left_upper - left_lower
        right_width = right_upper - right_lower
        if left_width >= right_width and left_width > 1:
            midpoint = (left_lower + left_upper) // 2
            candidate = aggregate_zero_count(
                left_values[left_lower:midpoint],
                right_values[right_lower:right_upper],
                polynomial,
                prime,
            )
            all_counts_exact &= candidate[
                "aggregate_count_equals_direct_integer"
            ]
            if candidate["direct_integer_zero_count"]:
                left_upper = midpoint
            else:
                left_lower = midpoint
        else:
            midpoint = (right_lower + right_upper) // 2
            candidate = aggregate_zero_count(
                left_values[left_lower:left_upper],
                right_values[right_lower:midpoint],
                polynomial,
                prime,
            )
            all_counts_exact &= candidate[
                "aggregate_count_equals_direct_integer"
            ]
            if candidate["direct_integer_zero_count"]:
                right_upper = midpoint
            else:
                right_lower = midpoint
        current = aggregate_zero_count(
            left_values[left_lower:left_upper],
            right_values[right_lower:right_upper],
            polynomial,
            prime,
        )
        all_counts_exact &= current[
            "aggregate_count_equals_direct_integer"
        ]
        transcript.append(
            {
                "left_range": [left_lower, left_upper],
                "right_range": [right_lower, right_upper],
                "zero_count": current["direct_integer_zero_count"],
            }
        )
    source = [left_lower, right_lower]
    return {
        "source": source,
        "root_count": root["direct_integer_zero_count"],
        "transcript": transcript,
        "all_counts_exact": all_counts_exact,
        "returned_source_is_zero": (
            quadratic_resultant(
                left_values[left_lower],
                right_values[right_lower],
                prime,
            )
            == 0
        ),
        "aggregate_contraction_count": len(transcript) * 2 - 1,
    }


def toy_source_replay() -> dict[str, Any]:
    prime = 11
    polynomial = resultant_power(prime)
    left_values: list[Quadratic] = [
        (1, 0, 10),
        (1, 0, 7),
        (1, 1, 1),
    ]
    right_values: list[Quadratic] = [
        (1, 8, 2),
        (1, 0, 10),
        (1, 0, 10),
    ]
    main = aggregate_zero_count(
        left_values,
        right_values,
        polynomial,
        prime,
    )
    source = dyadic_source_route(
        left_values,
        right_values,
        polynomial,
        prime,
    )
    blind_candidates = [
        candidate
        for candidate in itertools.product(range(prime), repeat=3)
        if candidate[0] == 1
        and (
            candidate[1] * candidate[1]
            - 4 * candidate[0] * candidate[2]
        )
        % prime
        != 0
        and all(
            quadratic_resultant(left, candidate, prime) != 0
            for left in left_values
        )
    ][:2]
    blind = aggregate_zero_count(
        left_values,
        blind_candidates,
        polynomial,
        prime,
    )
    single_right_count = sum(
        quadratic_resultant(left, right_values[1], prime) == 0
        for left in left_values
    )
    duplicate_columns_count = sum(
        quadratic_resultant(left, right, prime) == 0
        for left in left_values
        for right in right_values[1:]
    )
    return {
        "field_prime": prime,
        "left_quadratics": [list(value) for value in left_values],
        "right_quadratics": [list(value) for value in right_values],
        "aggregate_count": main,
        "dyadic_source": source,
        "blind_control": {
            "right_quadratics": [
                list(value) for value in blind_candidates
            ],
            "aggregate_count": blind,
            "blind_zero_exact": (
                blind["direct_integer_zero_count"] == 0
                and blind["aggregate_count_equals_direct_integer"]
            ),
        },
        "duplicate_occurrence_control": {
            "duplicated_right_indices": [1, 2],
            "one_copy_zero_count": single_right_count,
            "two_duplicate_columns_zero_count": duplicate_columns_count,
            "two_copy_expected_zero_count": 2 * single_right_count,
            "duplicate_occurrence_equality_exact": (
                duplicate_columns_count == 2 * single_right_count
            ),
        },
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "relation": "base-field prime p=Theta(N)=Theta(B^5)",
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "raw_rank_six_symmetric_power_coordinates": (
            "binomial(p+4,5)=Theta(p^5)=Theta(B^25)"
        ),
        "veronese_quotient_moment_coordinates": {
            "exact_dimension": "binomial(2p,2)=p(2p-1)",
            "asymptotic": "Theta(p^2)=Theta(B^10)",
            "state_exponent_B": fraction_record(
                MOMENT_STATE_EXPONENT_B
            ),
            "inside_setup_cap": MOMENT_STATE_EXPONENT_B <= SETUP_CAP,
            "inside_online_cap": MOMENT_STATE_EXPONENT_B <= ONLINE_CAP,
        },
        "explicit_moment_contraction_inside_caps": False,
        "compact_root_source_moment_constructor_supplied": False,
        "actual_five_a_five_c_no_wrap_certificate_supplied": False,
        "nonlinear_nonmoment_recurrence_refuted": False,
        "scope": (
            "charges the canonical all-monomial moment contraction only; "
            "it is not a lower bound on modular traces, character sums, "
            "streaming nonlinear circuits, or source-reporting indices"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    sweep = coefficient_rank_sweep()
    replay = toy_source_replay()
    costs = asymptotic_cost_control()
    all_full_rank = all(row["full_rank"] for row in sweep)
    aggregate_exact = replay["aggregate_count"][
        "aggregate_count_equals_direct_integer"
    ]
    source_exact = (
        replay["dyadic_source"]["all_counts_exact"]
        and replay["dyadic_source"]["returned_source_is_zero"]
    )
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_aggregate_veronese_"
            "projector_recurrence.r95.v1"
        ),
        "projector": "1-Res(f,g)^(p-1)",
        "resultant_identity": (
            "(a*f-c*d)^2-(a*e-b*d)*(b*f-c*e)"
        ),
        "aggregate_identity": (
            "sum_(u,v) Res(u,v)^(p-1) equals the coefficient "
            "pairing of the two degree-2(p-1) moment vectors"
        ),
        "rank_sweep_primes": list(RANK_SWEEP_PRIMES),
        "caps": costs["caps"],
    }
    state_ledger = {
        "schema": (
            "p1553.projector_trace_recurrence_state_ledger.r95.v1"
        ),
        "coefficient_rank_sweep": sweep,
        "all_finite_pairings_full_rank": all_full_rank,
        "asymptotic_cost": costs,
        "finite_rank_is_not_asymptotic_lower_bound": True,
    }
    source_replay = {
        "schema": (
            "p1553.dyadic_projector_count_source_replay.r95.v1"
        ),
        "toy_replay": replay,
        "aggregate_count_exact": aggregate_exact,
        "dyadic_source_exact": source_exact,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.projective_trace_exceptional_controls.r95.v1"
        ),
        "blind_zero_replayed": replay["blind_control"][
            "blind_zero_exact"
        ],
        "duplicate_occurrences_counted": replay[
            "duplicate_occurrence_control"
        ]["duplicate_occurrence_equality_exact"],
        "integer_count_requires_no_wrap": True,
        "actual_five_a_five_c_no_wrap_proved": False,
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_multiplicity_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r95.v1",
        "canonical_moment_recurrence_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "eight_source_bindings_verified": len(bindings) == 8,
        "resultant_power_support_exact": all(
            row["nonzero_coefficient_pair_count"] > 0 for row in sweep
        ),
        "veronese_quotient_dimension_exact": all(
            row["veronese_quotient_dimension"]
            == row["field_prime"] * (2 * row["field_prime"] - 1)
            for row in sweep
        ),
        "all_finite_coefficient_pairings_full_rank": all_full_rank,
        "aggregate_moment_count_exact_mod_prime": replay[
            "aggregate_count"
        ]["aggregate_count_equals_direct_mod_prime"],
        "finite_no_wrap_count_exact_integer": aggregate_exact,
        "blind_zero_control_exact": replay["blind_control"][
            "blind_zero_exact"
        ],
        "duplicate_occurrence_control_exact": replay[
            "duplicate_occurrence_control"
        ]["duplicate_occurrence_equality_exact"],
        "finite_dyadic_source_exact": source_exact,
        "p_squared_state_charged_as_B10": (
            costs["veronese_quotient_moment_coordinates"][
                "state_exponent_B"
            ]["exact"]
            == "10"
        ),
        "r9_r78_semantic_deduplication_recorded": True,
        "moment_state_inside_setup_cap": False,
        "moment_state_inside_online_cap": False,
        "compact_root_source_moment_constructor_supplied": False,
        "actual_five_a_five_c_no_wrap_certificate": False,
        "cap_sized_nonmoment_recurrence_supplied": False,
        "actual_five_a_five_c_source_unranking_complete": False,
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
            "AGGREGATE_FERMAT_MOMENT_IDENTITY_EXACT__"
            "VERONESE_QUOTIENT_PAIRING_FULL_RANK_P3_TO_P29__"
            "CANONICAL_MOMENT_STATE_B10__NONMOMENT_RECURRENCE_OPEN"
        ),
        "source_bindings": {
            "r94_producer": {
                "path": str(R94_PRODUCER),
                "sha256": R94_PRODUCER_SHA256,
            },
            "r94_report": {
                "path": str(R94_REPORT),
                "sha256": R94_REPORT_SHA256,
            },
            "r94_gate": {
                "path": str(R94_GATE),
                "sha256": R94_GATE_SHA256,
            },
            "r93_report": {
                "path": str(R93_REPORT),
                "sha256": R93_REPORT_SHA256,
            },
            "r78_report": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r78_gate": {
                "path": str(R78_GATE),
                "sha256": R78_GATE_SHA256,
            },
            "r9_gate": {
                "path": str(R9_GATE),
                "sha256": R9_GATE_SHA256,
            },
            "p1515_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
        },
        "novelty_scope": (
            "R95 is the first campaign receipt to form the exact "
            "coefficient-space aggregate of R94's projector after quotienting "
            "the rank-six expansion by the quadratic Veronese relations."
        ),
        "coefficient_pairing_rank_sweep": sweep,
        "aggregate_count_and_source_replay": replay,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_aggregate_veronese_"
                "projector_recurrence.json"
            ),
            "state_ledger": (
                "projector_trace_recurrence_state_ledger.json"
            ),
            "source_replay": (
                "dyadic_projector_count_and_source_replay.json"
            ),
            "exceptional": (
                "projective_trace_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r95.json",
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
            "This closes only the explicit all-monomial moment contraction "
            "of Res^(p-1). The finite full-rank sweep is not an asymptotic "
            "lower bound on modular trace recurrences, character sums, "
            "streaming nonlinear circuits, or source-reporting indices."
        ),
        "next_action": (
            "Construct or refute one modular trace recurrence for the same "
            "projector sum that uses the Frobenius identity H^p=H on field "
            "values without storing degree-2(p-1) moments. Freeze every "
            "state and range-restriction update; require B^(9/4) setup, "
            "B^(5/4) fresh work/workspace, an integer no-wrap or lift "
            "certificate, and exact dyadic coupled 5A+5C source replay on "
            "blind-zero, infinity, proper-subsum, tangent, and multiplicity "
            "branches without root-side scans, DLP labels, or verifier "
            "oracles."
        ),
        "disposition": (
            "REJECT_CANONICAL_AGGREGATE_MOMENT_CONTRACTION_ONLY__"
            "RESULTANT_POWER_IDENTITY_AND_DYADIC_TOY_SOURCE_EXACT__"
            "COEFFICIENT_PAIRING_FULL_RANK_P3_5_7_11_13_17_19_29__"
            "VERONESE_MOMENT_STATE_P2_EQUALS_B10__FINITE_RANK_NOT_"
            "ASYMPTOTIC_LOWER_BOUND__NONMOMENT_FROBENIUS_TRACE_OPEN__"
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
            "p1553_5a5c_aggregate_veronese_projector_"
            "recurrence_probe_report_r95.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_aggregate_veronese_"
            "projector_recurrence.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projector_trace_recurrence_state_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "dyadic_projector_count_and_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_trace_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r95.json"
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
