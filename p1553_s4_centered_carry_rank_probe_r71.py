#!/usr/bin/env python3
"""Probe centered S4 remainder and carry ranks on exact legal-label grids."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from typing import Any


SCHEMA = "p1553.s4_centered_carry_rank_probe.r71.v1"
DECK_SIZES = tuple(range(3, 9))
AUXILIARY_PRIMES = (1_000_003, 1_000_033)
R8_GATE = pathlib.Path("p1553_integer_valued_quotient_gate_r8.md")
R8_GATE_SHA256 = (
    "78c856187bb43adcd97d0f02f2259c7299874ab93a03954dacb4dd1b8b007ed9"
)
R7_GATE = pathlib.Path("p1553_tensorized_small_root_gate_r7.md")
R7_GATE_SHA256 = (
    "b36870eeb0c7c0a53e6d1714d623629c522f4c58cd308a072e33ea6046a06615"
)
R70_REPORT = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
)
R70_REPORT_SHA256 = (
    "9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89"
)


def load_r70() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_multiplicative_x_s3_closure_screen_r70.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r70_for_r71", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R70 curve controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R70 = load_r70()
CURVES = R70.CURVES
Point = tuple[int, int] | None


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centered_residue(value: int, modulus: int) -> int:
    """Return the unique odd-modulus residue in [-(p-1)/2, (p-1)/2]."""
    return (value + modulus // 2) % modulus - modulus // 2


def semaev_s3_coefficients(
    x_first: int,
    x_second: int,
    curve: dict[str, Any],
) -> tuple[int, int, int]:
    """Return the exact integer coefficients of S3(x_first,x_second,z)."""
    a_coefficient = curve["curve_a"]
    b_coefficient = curve["curve_b"]
    return (
        (x_first - x_second) ** 2,
        -2
        * (
            (x_first + x_second)
            * (x_first * x_second + a_coefficient)
            + 2 * b_coefficient
        ),
        (x_first * x_second - a_coefficient) ** 2
        - 4 * b_coefficient * (x_first + x_second),
    )


def bareiss_determinant(matrix: list[list[int]]) -> int:
    """Compute an exact determinant with fraction-free Bareiss elimination."""
    size = len(matrix)
    if size == 0:
        return 1
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    work = [row[:] for row in matrix]
    sign = 1
    previous_pivot = 1
    for pivot_index in range(size - 1):
        if work[pivot_index][pivot_index] == 0:
            source = next(
                (
                    row
                    for row in range(pivot_index + 1, size)
                    if work[row][pivot_index] != 0
                ),
                None,
            )
            if source is None:
                return 0
            work[pivot_index], work[source] = work[source], work[pivot_index]
            sign *= -1
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                if numerator % previous_pivot:
                    raise AssertionError("Bareiss exact-division invariant failed")
                work[row][column] = numerator // previous_pivot
        for row in range(pivot_index + 1, size):
            work[row][pivot_index] = 0
        previous_pivot = pivot
    return sign * work[-1][-1]


def quadratic_resultant_closed(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> int:
    """Return Res_z(a z^2+b z+c, d z^2+e z+f)."""
    a_coefficient, b_coefficient, c_coefficient = left
    d_coefficient, e_coefficient, f_coefficient = right
    return (
        (a_coefficient * f_coefficient - c_coefficient * d_coefficient) ** 2
        - (
            a_coefficient * e_coefficient
            - b_coefficient * d_coefficient
        )
        * (
            b_coefficient * f_coefficient
            - c_coefficient * e_coefficient
        )
    )


def quadratic_resultant_bareiss(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> int:
    a_coefficient, b_coefficient, c_coefficient = left
    d_coefficient, e_coefficient, f_coefficient = right
    return bareiss_determinant(
        [
            [a_coefficient, b_coefficient, c_coefficient, 0],
            [0, a_coefficient, b_coefficient, c_coefficient],
            [d_coefficient, e_coefficient, f_coefficient, 0],
            [0, d_coefficient, e_coefficient, f_coefficient],
        ]
    )


def semaev_s4_integer(
    x_first: int,
    x_second: int,
    x_third: int,
    x_target: int,
    curve: dict[str, Any],
) -> int:
    left = semaev_s3_coefficients(x_first, x_second, curve)
    right = semaev_s3_coefficients(x_third, x_target, curve)
    return quadratic_resultant_closed(left, right)


def hash_point(
    curve: dict[str, Any],
    stream: str,
    counter: int,
) -> Point:
    digest = hashlib.sha256(
        f"P1553-R71|{curve['family_id']}|{stream}|{counter}".encode()
    ).digest()
    x_coordinate = int.from_bytes(digest, "big") % curve["field_prime"]
    return R70.point_from_x(x_coordinate, digest[0] & 1, curve)


def public_decks_and_targets(
    curve: dict[str, Any],
) -> tuple[list[list[Point]], list[dict[str, Any]]]:
    seen_x_coordinates: set[int] = set()
    decks: list[list[Point]] = []
    for deck_index in range(3):
        deck: list[Point] = []
        counter = 0
        while len(deck) < max(DECK_SIZES):
            point = hash_point(curve, f"deck-{deck_index}", counter)
            counter += 1
            if point is None or point[0] in seen_x_coordinates:
                continue
            seen_x_coordinates.add(point[0])
            deck.append(point)
        decks.append(deck)

    blind_target = None
    counter = 0
    while blind_target is None:
        candidate = hash_point(curve, "blind-target", counter)
        counter += 1
        if candidate is None or candidate[0] in seen_x_coordinates:
            continue
        blind_target = candidate

    forced_target = None
    forced_witness = None
    excluded_target_x = seen_x_coordinates | {blind_target[0]}
    for indices in itertools.product(range(max(DECK_SIZES)), repeat=3):
        total = R70.add(
            R70.add(decks[0][indices[0]], decks[1][indices[1]], curve),
            decks[2][indices[2]],
            curve,
        )
        candidate = R70.negate(total, curve)
        if candidate is None or candidate[0] in excluded_target_x:
            continue
        forced_target = candidate
        forced_witness = list(indices)
        break
    if forced_target is None or forced_witness is None:
        raise AssertionError("unable to construct a distinct forced-positive target")

    targets = [
        {
            "target_id": "blind_hash_target",
            "point": blind_target,
            "construction": "independent_sha256_x_stream_then_cofactor_clear",
            "forced_witness": None,
        },
        {
            "target_id": "forced_positive_target",
            "point": forced_target,
            "construction": "negative_sum_of_three_public_deck_points",
            "forced_witness": forced_witness,
        },
    ]
    return decks, targets


def signed_relation_exists(
    points: tuple[Point, Point, Point],
    target: Point,
    curve: dict[str, Any],
) -> bool:
    for signs in itertools.product((-1, 1), repeat=3):
        total = target
        for point, sign in zip(points, signs):
            signed_point = point if sign == 1 else R70.negate(point, curve)
            total = R70.add(total, signed_point, curve)
        if total is None:
            return True
    return False


def matrix_rank_mod(matrix: list[list[int]], modulus: int) -> int:
    if not matrix:
        return 0
    work = [[value % modulus for value in row] for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (
                row
                for row in range(pivot_row, row_count)
                if work[row][column]
            ),
            None,
        )
        if source is None:
            continue
        work[pivot_row], work[source] = work[source], work[pivot_row]
        inverse = pow(work[pivot_row][column], -1, modulus)
        work[pivot_row] = [
            value * inverse % modulus for value in work[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row:
                continue
            multiplier = work[row][column]
            if not multiplier:
                continue
            work[row] = [
                (value - multiplier * pivot) % modulus
                for value, pivot in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def flatten_tensor(
    tensor: list[list[list[int]]],
    mode: int,
) -> list[list[int]]:
    size = len(tensor)
    if mode == 0:
        return [
            [tensor[first][second][third] for second in range(size) for third in range(size)]
            for first in range(size)
        ]
    if mode == 1:
        return [
            [tensor[first][second][third] for first in range(size) for third in range(size)]
            for second in range(size)
        ]
    if mode == 2:
        return [
            [tensor[first][second][third] for first in range(size) for second in range(size)]
            for third in range(size)
        ]
    raise ValueError("mode must be 0, 1, or 2")


def tensor_rank_profile(
    tensor: list[list[list[int]]],
) -> dict[str, Any]:
    size = len(tensor)
    mode_rows = []
    for mode in range(3):
        matrix = flatten_tensor(tensor, mode)
        ranks = {
            str(prime): matrix_rank_mod(matrix, prime)
            for prime in AUXILIARY_PRIMES
        }
        mode_rows.append(
            {
                "mode": mode,
                "row_count": size,
                "column_count": size * size,
                "auxiliary_prime_ranks": ranks,
                "certified_rational_rank_lower_bound": max(ranks.values()),
                "full_rank_under_both_auxiliary_primes": all(
                    rank == size for rank in ranks.values()
                ),
                "auxiliary_ranks_agree": len(set(ranks.values())) == 1,
            }
        )
    return {
        "mode_profiles": mode_rows,
        "all_modes_full_under_both_auxiliary_primes": all(
            row["full_rank_under_both_auxiliary_primes"] for row in mode_rows
        ),
        "maximum_certified_rational_rank_lower_bound": max(
            row["certified_rational_rank_lower_bound"] for row in mode_rows
        ),
        "minimum_certified_rational_rank_lower_bound": min(
            row["certified_rational_rank_lower_bound"] for row in mode_rows
        ),
    }


def prefix_tensor(
    tensor: list[list[list[int]]],
    size: int,
) -> list[list[list[int]]]:
    return [
        [
            [tensor[first][second][third] for third in range(size)]
            for second in range(size)
        ]
        for first in range(size)
    ]


def analyze_target(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target_spec: dict[str, Any],
) -> dict[str, Any]:
    target = target_spec["point"]
    if target is None:
        raise AssertionError("target may not be identity")
    size = max(DECK_SIZES)
    raw_tensor = [
        [[0 for _ in range(size)] for _ in range(size)] for _ in range(size)
    ]
    centered_tensor = [
        [[0 for _ in range(size)] for _ in range(size)] for _ in range(size)
    ]
    carry_tensor = [
        [[0 for _ in range(size)] for _ in range(size)] for _ in range(size)
    ]
    resultant_mismatches = 0
    predicate_mismatches = 0
    relation_root_count = 0
    divisibility_failures = 0
    for first, second, third in itertools.product(range(size), repeat=3):
        x_coordinates = (
            decks[0][first][0],
            decks[1][second][0],
            decks[2][third][0],
        )
        left = semaev_s3_coefficients(
            x_coordinates[0], x_coordinates[1], curve
        )
        right = semaev_s3_coefficients(x_coordinates[2], target[0], curve)
        raw_value = quadratic_resultant_closed(left, right)
        if raw_value != quadratic_resultant_bareiss(left, right):
            resultant_mismatches += 1
        centered_value = centered_residue(raw_value, curve["field_prime"])
        numerator = raw_value - centered_value
        if numerator % curve["field_prime"]:
            divisibility_failures += 1
        carry_value = numerator // curve["field_prime"]
        raw_tensor[first][second][third] = raw_value
        centered_tensor[first][second][third] = centered_value
        carry_tensor[first][second][third] = carry_value

        relation_exists = signed_relation_exists(
            (
                decks[0][first],
                decks[1][second],
                decks[2][third],
            ),
            target,
            curve,
        )
        polynomial_is_zero = raw_value % curve["field_prime"] == 0
        relation_root_count += relation_exists
        if polynomial_is_zero != relation_exists:
            predicate_mismatches += 1

    prefix_rows = []
    for deck_size in DECK_SIZES:
        raw_profile = tensor_rank_profile(prefix_tensor(raw_tensor, deck_size))
        centered_profile = tensor_rank_profile(
            prefix_tensor(centered_tensor, deck_size)
        )
        carry_profile = tensor_rank_profile(prefix_tensor(carry_tensor, deck_size))
        constant_profile = tensor_rank_profile(
            [
                [[1 for _ in range(deck_size)] for _ in range(deck_size)]
                for _ in range(deck_size)
            ]
        )
        prefix_rows.append(
            {
                "deck_size": deck_size,
                "grid_size": deck_size**3,
                "raw_integer_lift": raw_profile,
                "centered_remainder": centered_profile,
                "centered_carry": carry_profile,
                "rank_one_constant_control": constant_profile,
                "raw_mode_rank_within_s4_degree_bound": all(
                    row["certified_rational_rank_lower_bound"] <= 5
                    for row in raw_profile["mode_profiles"]
                ),
                "constant_control_has_rank_one": all(
                    row["certified_rational_rank_lower_bound"] == 1
                    for row in constant_profile["mode_profiles"]
                ),
            }
        )

    forced_witness_verified = True
    if target_spec["forced_witness"] is not None:
        indices = target_spec["forced_witness"]
        witness_value = raw_tensor[indices[0]][indices[1]][indices[2]]
        forced_witness_verified = (
            witness_value % curve["field_prime"] == 0
            and signed_relation_exists(
                (
                    decks[0][indices[0]],
                    decks[1][indices[1]],
                    decks[2][indices[2]],
                ),
                target,
                curve,
            )
        )
    return {
        "target_id": target_spec["target_id"],
        "target_point": list(target),
        "target_construction": target_spec["construction"],
        "forced_witness": target_spec["forced_witness"],
        "forced_witness_verified": forced_witness_verified,
        "full_grid": {
            "deck_size": size,
            "tuple_count": size**3,
            "relation_root_count": relation_root_count,
            "closed_vs_bareiss_resultant_mismatches": resultant_mismatches,
            "s4_vs_signed_group_relation_mismatches": predicate_mismatches,
            "carry_divisibility_failures": divisibility_failures,
        },
        "prefix_rank_profiles": prefix_rows,
    }


def analyze_family(curve: dict[str, Any]) -> dict[str, Any]:
    decks, targets = public_decks_and_targets(curve)
    return {
        "family": curve,
        "curve_order_replay": R70.curve_order(curve),
        "curve_discriminant_nonzero": R70.curve_discriminant(curve) != 0,
        "public_source": {
            "deck_points": [
                [list(point) for point in deck if point is not None]
                for deck in decks
            ],
            "deck_sizes": list(DECK_SIZES),
            "scalar_labels_consumed": False,
            "construction": (
                "domain-separated_sha256_x_stream_then_canonical_square_root_"
                "and_public_cofactor_clearing"
            ),
        },
        "targets": [
            analyze_target(curve, decks, target_spec) for target_spec in targets
        ],
    }


def run() -> dict[str, Any]:
    bindings = {
        "r8_integer_valued_quotient_gate": {
            "path": str(R8_GATE),
            "sha256": R8_GATE_SHA256,
        },
        "r7_tensorized_small_root_gate": {
            "path": str(R7_GATE),
            "sha256": R7_GATE_SHA256,
        },
        "r70_four_family_curve_control": {
            "path": str(R70_REPORT),
            "sha256": R70_REPORT_SHA256,
        },
    }
    for binding in bindings.values():
        if sha256_file(pathlib.Path(binding["path"])) != binding["sha256"]:
            raise AssertionError(f"source binding mismatch: {binding['path']}")
    family_rows = [analyze_family(dict(curve)) for curve in CURVES]

    all_targets = [
        target
        for family in family_rows
        for target in family["targets"]
    ]
    all_prefixes = [
        prefix
        for target in all_targets
        for prefix in target["prefix_rank_profiles"]
    ]
    largest_prefixes = [
        prefix
        for prefix in all_prefixes
        if prefix["deck_size"] == max(DECK_SIZES)
    ]
    checks = {
        "four_distinct_prime_field_families": len(
            {family["family"]["field_prime"] for family in family_rows}
        )
        == 4
        and all(R70.is_prime(family["family"]["field_prime"]) for family in family_rows),
        "all_subgroup_orders_are_prime": all(
            R70.is_prime(family["family"]["subgroup_order"])
            for family in family_rows
        ),
        "all_curve_orders_replay": all(
            family["curve_order_replay"]
            == family["family"]["subgroup_order"] * family["family"]["cofactor"]
            for family in family_rows
        ),
        "all_curve_discriminants_nonzero": all(
            family["curve_discriminant_nonzero"] for family in family_rows
        ),
        "auxiliary_moduli_are_distinct_primes": (
            len(set(AUXILIARY_PRIMES)) == len(AUXILIARY_PRIMES)
            and all(R70.is_prime(prime) for prime in AUXILIARY_PRIMES)
        ),
        "all_sources_are_scalar_blind": all(
            not family["public_source"]["scalar_labels_consumed"]
            for family in family_rows
        ),
        "all_resultants_match_bareiss": all(
            target["full_grid"]["closed_vs_bareiss_resultant_mismatches"] == 0
            for target in all_targets
        ),
        "all_s4_predicates_match_signed_group_relations": all(
            target["full_grid"]["s4_vs_signed_group_relation_mismatches"] == 0
            for target in all_targets
        ),
        "all_carries_are_exactly_divisible": all(
            target["full_grid"]["carry_divisibility_failures"] == 0
            for target in all_targets
        ),
        "all_forced_witnesses_verify": all(
            target["forced_witness_verified"] for target in all_targets
        ),
        "raw_lifts_respect_degree_four_mode_rank_bound": all(
            prefix["raw_mode_rank_within_s4_degree_bound"]
            for prefix in all_prefixes
        ),
        "all_constant_controls_have_rank_one": all(
            prefix["constant_control_has_rank_one"] for prefix in all_prefixes
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R71 exact probe failed: {checks}")

    largest_carry_full_count = sum(
        prefix["centered_carry"]["all_modes_full_under_both_auxiliary_primes"]
        for prefix in largest_prefixes
    )
    largest_remainder_full_count = sum(
        prefix["centered_remainder"]["all_modes_full_under_both_auxiliary_primes"]
        for prefix in largest_prefixes
    )
    all_largest_carries_full = largest_carry_full_count == len(largest_prefixes)
    all_largest_remainders_full = (
        largest_remainder_full_count == len(largest_prefixes)
    )
    every_carry_prefix_full = all(
        prefix["centered_carry"]["all_modes_full_under_both_auxiliary_primes"]
        for prefix in all_prefixes
    )
    obligations = {
        "exact_scalar_blind_legal_grid_source": checks[
            "all_sources_are_scalar_blind"
        ],
        "four_generic_prime_field_families": checks[
            "four_distinct_prime_field_families"
        ],
        "positive_and_blind_target_predicate_replay": checks[
            "all_s4_predicates_match_signed_group_relations"
        ]
        and checks["all_forced_witnesses_verify"],
        "centered_carry_rank_stays_below_linear_on_all_largest_grids": (
            not all_largest_carries_full
        ),
        "s6_centered_carry_theorem_supplied": False,
        "branch_complete_unit_or_zero_divisor_router_supplied": False,
        "fresh_target_source_recovery_below_rho": False,
    }
    return {
        "schema": SCHEMA,
        "classification": (
            "S4_CENTERED_ELLIPTIC_CARRY_HAS_FULL_MODE_RANK_ON_ALL_FROZEN_LARGEST_GRIDS"
            if all_largest_carries_full
            else "S4_CENTERED_ELLIPTIC_CARRY_HAS_MIXED_MODE_RANK"
        ),
        "candidate": {
            "name": "target_specific_s4_centered_legal_grid_carry",
            "definition": (
                "For three legal x-class decks and one fixed target, evaluate the "
                "exact integer resultant S4=Res_z(S3(x1,x2,z),S3(x3,xR,z)), "
                "center it modulo p, and measure the carry K=(S4-ctr_p(S4))/p."
            ),
            "scope": (
                "This is the k=1, arity-four predecessor of the unresolved R8 "
                "S6 p^k-centered carry. It is a representation probe, not an "
                "ECDLP relation algorithm or asymptotic lower bound."
            ),
        },
        "parameters": {
            "deck_sizes": list(DECK_SIZES),
            "auxiliary_primes": list(AUXILIARY_PRIMES),
            "target_types": [
                "blind_hash_target",
                "forced_positive_target",
            ],
            "centering_interval": "[-(p-1)/2,(p-1)/2]",
        },
        "family_results": family_rows,
        "aggregate": {
            "family_count": len(family_rows),
            "target_instance_count": len(all_targets),
            "prefix_instance_count": len(all_prefixes),
            "largest_prefix_instance_count": len(largest_prefixes),
            "largest_centered_carry_all_mode_full_count": largest_carry_full_count,
            "largest_centered_remainder_all_mode_full_count": (
                largest_remainder_full_count
            ),
            "all_largest_centered_carries_have_full_mode_rank": (
                all_largest_carries_full
            ),
            "all_largest_centered_remainders_have_full_mode_rank": (
                all_largest_remainders_full
            ),
            "every_centered_carry_prefix_has_full_mode_rank": (
                every_carry_prefix_full
            ),
            "certified_cp_rank_lower_bound_on_each_largest_carry": (
                max(DECK_SIZES) if all_largest_carries_full else None
            ),
        },
        "source_bindings": bindings,
        "rank_interpretation": {
            "lower_bound": (
                "A mode-flattening rank r modulo either auxiliary prime proves "
                "rational tensor rank and rational CP rank at least r for the "
                "integer value tensor on that frozen legal grid."
            ),
            "basis_scope": (
                "Factorwise invertible rational label-basis transforms preserve "
                "the flattening ranks. Canonical modular recentering is nonlinear "
                "and is measured after, not inferred through, that transform."
            ),
            "raw_control": (
                "S4 has degree four in each x argument, so the raw fixed-target "
                "value tensor has every mode rank at most five regardless of deck "
                "size. Rank expansion is therefore attributable to centering and "
                "the induced carry in this exact lift."
            ),
        },
        "semantic_dedup": {
            "new_idea_created": False,
            "owners": [
                "P1553 R7 tensorized small-root backend",
                "P1553 R8 integer-valued quotient and centered-carry gate",
                "IDEA-049 bounded root decomposition transducer",
                "IDEA-198 carry-state source unranking",
            ],
            "disposition": (
                "Record as an exact R8 predecessor probe. A full-rank result "
                "rejects only this canonical k=1 S4 lift on the frozen decks; it "
                "does not close elliptic-specific S6 cancellations or non-CP "
                "unit-or-zero-divisor interfaces."
            ),
        },
        "checks": checks,
        "admission": {
            "lane_admitted": all(obligations.values()),
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "obligations": obligations,
        },
        "result": {
            "s4_centered_carry_probe_complete": True,
            "canonical_s4_carry_low_rank_survives": not all_largest_carries_full,
            "s6_centered_carry_rank_resolved": False,
            "branch_complete_unit_or_zero_divisor_router_supplied": False,
            "factor_log_solve_complete": False,
            "fresh_target_descent_complete": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "next_action": (
            "Close the canonical target-specific S4 centered-carry precursor if "
            "its largest legal-grid flattenings are uniformly full. Preserve the "
            "R8 exception only for an S6-specific cancellation, a noncanonical "
            "bounded lift, or an exact non-CP unit-or-zero-divisor router, and "
            "require that successor to expose source recovery and direct caps."
        ),
        "limits": [
            "The experiment uses small curves, public size-three through size-eight prefixes, and two target types per family.",
            "A modular full-rank minor is an exact lower bound for the frozen integer value tensor, not an asymptotic theorem over every deck family.",
            "The probe studies k=1 S4 centering modulo p, not S6 powers centered modulo p^k.",
            "No runtime lower bound, Shoup-bound improvement, or ECDLP breakthrough is claimed.",
        ],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_s4_centered_carry_rank_probe_report_r71.json"
        ),
    )
    args = parser.parse_args()
    payload = run()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = payload["aggregate"]
    print(
        f"output={args.output} targets={aggregate['target_instance_count']} "
        f"full_largest_carries="
        f"{aggregate['largest_centered_carry_all_mode_full_count']}/"
        f"{aggregate['largest_prefix_instance_count']} "
        f"admitted={payload['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
