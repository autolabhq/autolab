#!/usr/bin/env python3
"""Screen multiplicative-subgroup x bases for excess S3 closure collisions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import statistics
from typing import Any


SCHEMA = "p1553.multiplicative_x_s3_closure_screen.r70.v1"
BASE_SIZE = 8
CONTROL_COUNT = 32
CURVES = (
    {
        "family_id": "p193_a2_b3_q103_h2",
        "field_prime": 193,
        "curve_a": 2,
        "curve_b": 3,
        "subgroup_order": 103,
        "cofactor": 2,
        "coordinate_subgroup_order": 64,
    },
    {
        "family_id": "p257_a1_b7_q281_h1",
        "field_prime": 257,
        "curve_a": 1,
        "curve_b": 7,
        "subgroup_order": 281,
        "cofactor": 1,
        "coordinate_subgroup_order": 64,
    },
    {
        "family_id": "p337_a1_b3_q163_h2",
        "field_prime": 337,
        "curve_a": 1,
        "curve_b": 3,
        "subgroup_order": 163,
        "cofactor": 2,
        "coordinate_subgroup_order": 48,
    },
    {
        "family_id": "p449_a1_b3_q463_h1",
        "field_prime": 449,
        "curve_a": 1,
        "curve_b": 3,
        "subgroup_order": 463,
        "cofactor": 1,
        "coordinate_subgroup_order": 64,
    },
)
R69_REPORT = pathlib.Path(
    "p1553_constructive_closure_collision_gate_report_r69.json"
)
R69_REPORT_SHA256 = (
    "401b8ae56607f4962c9ae10ed99889d5c27ac865d0e27b3359f7b163a04dd451"
)


Point = tuple[int, int] | None


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_to_json(point: Point) -> list[int] | None:
    return None if point is None else list(point)


def curve_discriminant(curve: dict[str, Any]) -> int:
    prime = curve["field_prime"]
    return (
        4 * curve["curve_a"] ** 3 + 27 * curve["curve_b"] ** 2
    ) % prime


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def curve_order(curve: dict[str, Any]) -> int:
    prime = curve["field_prime"]
    count = 1
    for x_coordinate in range(prime):
        rhs = (
            x_coordinate**3
            + curve["curve_a"] * x_coordinate
            + curve["curve_b"]
        ) % prime
        if rhs == 0:
            count += 1
        elif pow(rhs, (prime - 1) // 2, prime) == 1:
            count += 2
    return count


def modular_square_root(value: int, prime: int) -> int | None:
    value %= prime
    if value == 0:
        return 0
    if pow(value, (prime - 1) // 2, prime) != 1:
        return None
    odd_part = prime - 1
    two_adic_order = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        two_adic_order += 1
    if two_adic_order == 1:
        root = pow(value, (prime + 1) // 4, prime)
        return min(root, (-root) % prime)
    nonresidue = 2
    while pow(nonresidue, (prime - 1) // 2, prime) != prime - 1:
        nonresidue += 1
    coefficient = pow(nonresidue, odd_part, prime)
    root = pow(value, (odd_part + 1) // 2, prime)
    residue = pow(value, odd_part, prime)
    order = two_adic_order
    while residue != 1:
        exponent = 1
        probe = residue * residue % prime
        while probe != 1:
            probe = probe * probe % prime
            exponent += 1
        update = pow(coefficient, 1 << (order - exponent - 1), prime)
        root = root * update % prime
        residue = residue * update * update % prime
        coefficient = update * update % prime
        order = exponent
    if root * root % prime != value:
        raise AssertionError("Tonelli-Shanks replay failed")
    return min(root, (-root) % prime)


def negate(point: Point, curve: dict[str, Any]) -> Point:
    if point is None:
        return None
    return point[0], (-point[1]) % curve["field_prime"]


def add(left: Point, right: Point, curve: dict[str, Any]) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    prime = curve["field_prime"]
    x_left, y_left = left
    x_right, y_right = right
    if x_left == x_right and (y_left + y_right) % prime == 0:
        return None
    if left == right:
        slope = (
            (3 * x_left * x_left + curve["curve_a"])
            * pow(2 * y_left, -1, prime)
        ) % prime
    else:
        slope = (
            (y_right - y_left) * pow(x_right - x_left, -1, prime)
        ) % prime
    x_result = (slope * slope - x_left - x_right) % prime
    y_result = (slope * (x_left - x_result) - y_left) % prime
    return x_result, y_result


def scalar_mul(scalar: int, point: Point, curve: dict[str, Any]) -> Point:
    result = None
    addend = point
    while scalar:
        if scalar & 1:
            result = add(result, addend, curve)
        addend = add(addend, addend, curve)
        scalar >>= 1
    return result


def prime_factors(value: int) -> list[int]:
    factors = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return factors


def primitive_root(prime: int) -> int:
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // factor, prime) != 1
            for factor in factors
        ):
            return candidate
    raise AssertionError("primitive root not found")


def point_from_x(
    x_coordinate: int,
    sign_bit: int,
    curve: dict[str, Any],
) -> Point:
    prime = curve["field_prime"]
    rhs = (
        x_coordinate**3
        + curve["curve_a"] * x_coordinate
        + curve["curve_b"]
    ) % prime
    y_coordinate = modular_square_root(rhs, prime)
    if y_coordinate is None:
        return None
    if sign_bit:
        y_coordinate = (-y_coordinate) % prime
    point = scalar_mul(curve["cofactor"], (x_coordinate, y_coordinate), curve)
    if point is None:
        return None
    if scalar_mul(curve["subgroup_order"], point, curve) is not None:
        raise AssertionError("cofactor-cleared point missed prime subgroup")
    return point


def multiplicative_x_base(curve: dict[str, Any]) -> tuple[list[Point], dict[str, Any]]:
    prime = curve["field_prime"]
    order = curve["coordinate_subgroup_order"]
    if (prime - 1) % order:
        raise AssertionError("coordinate subgroup order does not divide p-1")
    generator = primitive_root(prime)
    step = pow(generator, (prime - 1) // order, prime)
    coordinates = sorted({pow(step, exponent, prime) for exponent in range(order)})
    points = []
    seen = set()
    accepted_coordinates = []
    for x_coordinate in coordinates:
        point = point_from_x(x_coordinate, 0, curve)
        if point is None or point in seen:
            continue
        seen.add(point)
        points.append(point)
        accepted_coordinates.append(x_coordinate)
        if len(points) == BASE_SIZE:
            break
    if len(points) != BASE_SIZE:
        raise AssertionError(
            f"{curve['family_id']} supplied only {len(points)} subgroup-x points"
        )
    return points, {
        "field_primitive_root": generator,
        "coordinate_subgroup_order": order,
        "coordinate_subgroup_generator": step,
        "accepted_x_coordinates": accepted_coordinates,
        "enumerated_coordinate_count": len(coordinates),
    }


def hash_control_base(curve: dict[str, Any], salt: int) -> list[Point]:
    points = []
    seen = set()
    counter = 0
    while len(points) < BASE_SIZE:
        digest = hashlib.sha256(
            f"P1553-R70|{curve['family_id']}|{salt}|{counter}".encode()
        ).digest()
        counter += 1
        x_coordinate = int.from_bytes(digest, "big") % curve["field_prime"]
        point = point_from_x(x_coordinate, digest[0] & 1, curve)
        if point is None or point in seen:
            continue
        seen.add(point)
        points.append(point)
    return points


def row_reduce(
    rows: list[list[int]],
    column_count: int,
    modulus: int,
) -> tuple[list[list[int]], list[int]]:
    matrix = [
        [value % modulus for value in row]
        for row in rows
        if any(value % modulus for value in row)
    ]
    pivots = []
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column]
            ),
            None,
        )
        if source is None:
            continue
        matrix[pivot_row], matrix[source] = matrix[source], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], -1, modulus)
        matrix[pivot_row] = [
            value * inverse % modulus for value in matrix[pivot_row]
        ]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            multiplier = matrix[row][column]
            matrix[row] = [
                (value - multiplier * pivot) % modulus
                for value, pivot in zip(matrix[row], matrix[pivot_row])
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return matrix[:pivot_row], pivots


def relation_row(indices: tuple[int, ...], column_count: int) -> list[int]:
    row = [0] * column_count
    for index in indices:
        row[index] += 1
    return row


def semaev_s3(
    x_first: int,
    x_second: int,
    x_third: int,
    curve: dict[str, Any],
) -> int:
    prime = curve["field_prime"]
    a_coefficient = curve["curve_a"]
    b_coefficient = curve["curve_b"]
    return (
        (x_first - x_second) ** 2 * x_third**2
        - 2
        * (
            (x_first + x_second) * (x_first * x_second + a_coefficient)
            + 2 * b_coefficient
        )
        * x_third
        + (x_first * x_second - a_coefficient) ** 2
        - 4 * b_coefficient * (x_first + x_second)
    ) % prime


def closure_profile(points: list[Point], curve: dict[str, Any]) -> dict[str, Any]:
    atoms = list(points)
    atom_index = {point: index for index, point in enumerate(atoms)}
    events = []
    s3_checks = 0
    s3_failures = 0
    for first, second in itertools.combinations(range(BASE_SIZE), 2):
        residual = negate(add(atoms[first], atoms[second], curve), curve)
        residual_index = atom_index.get(residual)
        if residual_index is None:
            residual_index = len(atoms)
            atom_index[residual] = residual_index
            atoms.append(residual)
            outcome = "fresh_residual"
        else:
            outcome = "closure_collision"
        if add(add(atoms[first], atoms[second], curve), residual, curve) is not None:
            raise AssertionError("closure relation failed public group replay")
        if all(
            point is not None
            for point in (atoms[first], atoms[second], residual)
        ):
            s3_checks += 1
            if semaev_s3(
                atoms[first][0],
                atoms[second][0],
                residual[0],
                curve,
            ):
                s3_failures += 1
        events.append(
            {
                "parents": [first, second],
                "residual_index": residual_index,
                "outcome": outcome,
            }
        )
    column_count = len(atoms)
    for event in events:
        event["row"] = relation_row(
            (*event["parents"], event["residual_index"]),
            column_count,
        )
    fresh_rows = [
        event["row"] for event in events if event["outcome"] == "fresh_residual"
    ]
    collision_rows = [
        event["row"] for event in events if event["outcome"] == "closure_collision"
    ]
    fresh_rank = len(
        row_reduce(fresh_rows, column_count, curve["subgroup_order"])[1]
    )
    combined_rank = len(
        row_reduce(
            fresh_rows + collision_rows,
            column_count,
            curve["subgroup_order"],
        )[1]
    )
    return {
        "base_size": BASE_SIZE,
        "pair_proposal_count": math.comb(BASE_SIZE, 2),
        "final_atom_count": column_count,
        "fresh_residual_count": len(fresh_rows),
        "closure_collision_count": len(collision_rows),
        "fresh_residual_rank": fresh_rank,
        "independent_collision_rank": combined_rank - fresh_rank,
        "final_relation_rank": combined_rank,
        "final_nullity": column_count - combined_rank,
        "s3_nonidentity_checks": s3_checks,
        "s3_failures": s3_failures,
    }


def analyze_family(curve: dict[str, Any]) -> dict[str, Any]:
    candidate_points, coordinate_source = multiplicative_x_base(curve)
    candidate = closure_profile(candidate_points, curve)
    controls = [
        closure_profile(hash_control_base(curve, salt), curve)
        for salt in range(CONTROL_COUNT)
    ]
    control_ranks = [row["independent_collision_rank"] for row in controls]
    control_collisions = [row["closure_collision_count"] for row in controls]
    mean_rank = statistics.mean(control_ranks)
    rank_stdev = statistics.pstdev(control_ranks)
    rho_operations = math.ceil(
        math.sqrt(math.pi * curve["subgroup_order"] / 2)
    )
    candidate_rank = candidate["independent_collision_rank"]
    return {
        "family": curve,
        "curve_order_replay": curve_order(curve),
        "curve_discriminant_nonzero": curve_discriminant(curve) != 0,
        "candidate_source": {
            **coordinate_source,
            "points": [point_to_json(point) for point in candidate_points],
            "scalar_labels_consumed": False,
            "sign_rule": "canonical_minimum_square_root_before_cofactor_clearing",
        },
        "candidate_profile": candidate,
        "matched_hash_controls": {
            "control_count": CONTROL_COUNT,
            "independent_collision_ranks": control_ranks,
            "closure_collision_counts": control_collisions,
            "mean_independent_collision_rank": mean_rank,
            "max_independent_collision_rank": max(control_ranks),
            "independent_collision_rank_population_stdev": rank_stdev,
        },
        "excess_test": {
            "candidate_minus_control_mean_rank": candidate_rank - mean_rank,
            "candidate_minus_control_max_rank": candidate_rank - max(control_ranks),
            "candidate_exceeds_every_control": candidate_rank > max(control_ranks),
            "candidate_at_least_double_control_mean": candidate_rank
            >= 2 * mean_rank
            and candidate_rank > 0,
            "candidate_has_preregistered_rank_excess": candidate_rank
            > max(control_ranks)
            and candidate_rank >= 2 * mean_rank
            and candidate_rank > 0,
        },
        "cost_floor": {
            "rho_expected_group_operations": rho_operations,
            "pair_proposals": candidate["pair_proposal_count"],
            "pair_proposals_below_rho": candidate["pair_proposal_count"]
            < rho_operations,
            "pair_proposal_ratio_vs_rho": candidate["pair_proposal_count"]
            / rho_operations,
            "subquadratic_locator_supplied": False,
        },
    }


def run() -> dict[str, Any]:
    if sha256_file(R69_REPORT) != R69_REPORT_SHA256:
        raise AssertionError("R69 report hash mismatch")
    family_rows = [analyze_family(dict(curve)) for curve in CURVES]
    checks = {
        "four_distinct_prime_field_families": len(
            {row["family"]["field_prime"] for row in family_rows}
        )
        == 4
        and all(is_prime(row["family"]["field_prime"]) for row in family_rows),
        "all_subgroup_orders_are_prime": all(
            is_prime(row["family"]["subgroup_order"]) for row in family_rows
        ),
        "all_curve_orders_match_prime_subgroup_times_cofactor": all(
            row["curve_order_replay"]
            == row["family"]["subgroup_order"] * row["family"]["cofactor"]
            for row in family_rows
        ),
        "all_curve_discriminants_nonzero": all(
            row["curve_discriminant_nonzero"] for row in family_rows
        ),
        "all_candidate_rows_pass_exact_s3_replay": all(
            row["candidate_profile"]["s3_failures"] == 0
            for row in family_rows
        ),
        "all_controls_pass_exact_s3_replay": all(
            all(control["s3_failures"] == 0 for control in [
                closure_profile(
                    hash_control_base(row["family"], salt),
                    row["family"],
                )
                for salt in range(CONTROL_COUNT)
            ])
            for row in family_rows
        ),
        "candidate_streams_are_scalar_blind": all(
            not row["candidate_source"]["scalar_labels_consumed"]
            for row in family_rows
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R70 multiplicative-x screen failed: {checks}")
    all_rank_excess = all(
        row["excess_test"]["candidate_has_preregistered_rank_excess"]
        for row in family_rows
    )
    all_pair_costs_below_rho = all(
        row["cost_floor"]["pair_proposals_below_rho"] for row in family_rows
    )
    obligations = {
        "public_scalar_blind_coordinate_subgroup_source": checks[
            "candidate_streams_are_scalar_blind"
        ],
        "four_distinct_generic_prime_field_families": checks[
            "four_distinct_prime_field_families"
        ],
        "exact_s3_relation_replay": checks[
            "all_candidate_rows_pass_exact_s3_replay"
        ],
        "independent_collision_rank_exceeds_all_matched_controls_on_every_family": (
            all_rank_excess
        ),
        "complete_pair_probe_below_rho_on_every_family": all_pair_costs_below_rho,
        "subquadratic_s3_locator_supplied": False,
        "fresh_target_descent_below_rho_on_every_family": False,
    }
    return {
        "schema": SCHEMA,
        "classification": (
            "MULTIPLICATIVE_X_S3_STRUCTURE_SCREENED_WITHOUT_UNIFORM_RANK_EXCESS"
            if not all_rank_excess
            else "MULTIPLICATIVE_X_S3_RANK_EXCESS_REQUIRES_LOCATOR"
        ),
        "candidate": {
            "name": "multiplicative_subgroup_x_coordinate_s3_closure",
            "definition": (
                "Take canonical curve points above a public multiplicative "
                "subgroup of F_p^*, cofactor-clear into the prime subgroup, and "
                "measure S3 closure collisions among the frozen prefix."
            ),
            "potential_locator": (
                "A useful successor would need sparse S3 zero-location on the "
                "multiplicative grid. Fast univariate polynomial arithmetic or a "
                "dense p-1 character transform alone is not such a locator."
            ),
        },
        "family_results": family_rows,
        "aggregate": {
            "family_count": len(family_rows),
            "families_with_candidate_rank_above_control_max": sum(
                row["excess_test"]["candidate_exceeds_every_control"]
                for row in family_rows
            ),
            "families_with_preregistered_rank_excess": sum(
                row["excess_test"]["candidate_has_preregistered_rank_excess"]
                for row in family_rows
            ),
            "families_with_pair_probe_below_rho": sum(
                row["cost_floor"]["pair_proposals_below_rho"]
                for row in family_rows
            ),
            "candidate_rank_excess_transfers_to_all_families": all_rank_excess,
            "complete_pair_probe_below_rho_on_all_families": (
                all_pair_costs_below_rho
            ),
        },
        "r69_binding": {
            "path": str(R69_REPORT),
            "sha256": R69_REPORT_SHA256,
            "rank_boundary": (
                "Only independently colliding residuals receive rank-reduction "
                "credit; fresh residual rows receive none."
            ),
        },
        "source_bindings": {
            "r69_constructive_closure_report": {
                "path": str(R69_REPORT),
                "sha256": R69_REPORT_SHA256,
            }
        },
        "semantic_dedup": {
            "new_idea_created": False,
            "owners": [
                "ECDLP-IDEA-057 auxiliary ECFFT and composable-label screens",
                "ECDLP-IDEA-195 non-Cartesian S3 source-router residual",
                "P1553 R10 exact multiplicative-convolution control",
            ],
            "disposition": (
                "Retain as an R69 successor falsification screen. Do not promote "
                "fast field-subgroup arithmetic into an ECDLP source claim without "
                "rank excess and an exact subquadratic locator."
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
            "scalar_blind_four_family_screen_complete": True,
            "transferred_superuniform_collision_rank": all_rank_excess,
            "subquadratic_s3_locator_supplied": False,
            "source_beats_rho": all_pair_costs_below_rho,
            "fresh_target_descent_complete": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "next_action": (
            "Close the raw multiplicative-x prefix candidate unless an independent "
            "review identifies a non-prefix coset family with a prospective density "
            "theorem. Route new work to a source with an explicit sub-pair collision "
            "locator, not merely FFT-compatible coordinates."
        ),
        "limits": [
            "The four-family screen uses small prime-order groups and fixed size-eight prefixes.",
            "Failure to exceed controls is evidence against this frozen candidate, not an impossibility theorem for every multiplicative coset family.",
            "The screen exhausts seed pairs only for falsification and supplies no subquadratic locator.",
            "No Shoup-bound improvement or ECDLP breakthrough is claimed.",
        ],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
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
        f"output={args.output} families={aggregate['family_count']} "
        f"rank_excess={aggregate['families_with_preregistered_rank_excess']} "
        f"below_rho={aggregate['families_with_pair_probe_below_rho']} "
        f"admitted={payload['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
