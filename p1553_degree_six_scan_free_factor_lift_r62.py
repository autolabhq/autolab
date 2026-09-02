#!/usr/bin/env python3
"""Lift R61 product-image roots to factor lines and cubic divisors without scans."""

import argparse
import hashlib
import itertools
import json
import pathlib

import numpy as np
import sympy

import p1553_degree_six_product_hypersurface_locator_r61 as r61


PRIME = r61.PRIME
SUBGROUP_ORDER = r61.SUBGROUP_ORDER
CLASS_SUM_SCALARS = r61.CLASS_SUM_SCALARS


def affine_preimage(multiplication_matrix, target):
    augmented = np.array(
        [row[:] + [target[index]] for index, row in enumerate(multiplication_matrix)],
        dtype=np.int64,
    ) % PRIME
    row = 0
    pivots = []
    for column in range(9):
        candidates = np.flatnonzero(augmented[row:, column])
        if not len(candidates):
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            augmented[[row, pivot_row]] = augmented[[pivot_row, row]]
        augmented[row] = (
            augmented[row] * pow(int(augmented[row, column]), -1, PRIME) % PRIME
        )
        factors = augmented[:, column].copy()
        factors[row] = 0
        augmented = (augmented - factors[:, None] * augmented[row]) % PRIME
        pivots.append(column)
        row += 1
        if row == len(multiplication_matrix):
            break
    if any(
        not any(augmented[index, :9]) and augmented[index, 9]
        for index in range(augmented.shape[0])
    ):
        raise AssertionError("target is not in the multiplication-map image span")
    free_columns = [column for column in range(9) if column not in pivots]
    if len(pivots) != 6 or len(free_columns) != 3:
        raise AssertionError("unexpected affine preimage dimension")

    particular = np.zeros(9, dtype=np.int64)
    for equation_row, pivot in enumerate(pivots):
        particular[pivot] = augmented[equation_row, 9]
    kernel_basis = []
    for free_column in free_columns:
        vector = np.zeros(9, dtype=np.int64)
        vector[free_column] = 1
        for equation_row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -int(
                np.dot(augmented[equation_row, :9], vector)
            ) % PRIME
        kernel_basis.append(vector)
    return (
        tuple(int(value) for value in particular),
        [tuple(int(value) for value in vector) for vector in kernel_basis],
        pivots,
        free_columns,
    )


def rank_one_lift(multiplication_matrix, target):
    particular, kernel_basis, pivots, free_columns = affine_preimage(
        multiplication_matrix,
        target,
    )
    variables = sympy.symbols("s0:3")
    tensor_entries = [
        particular[index]
        + sum(kernel_basis[basis][index] * variables[basis] for basis in range(3))
        for index in range(9)
    ]
    minors = []
    for first_row, second_row in itertools.combinations(range(3), 2):
        for first_column, second_column in itertools.combinations(range(3), 2):
            minors.append(
                sympy.expand(
                    tensor_entries[3 * first_row + first_column]
                    * tensor_entries[3 * second_row + second_column]
                    - tensor_entries[3 * first_row + second_column]
                    * tensor_entries[3 * second_row + first_column]
                )
            )
    basis = sympy.groebner(
        minors,
        *variables,
        order="lex",
        modulus=PRIME,
    )
    if not basis.is_zero_dimensional:
        raise AssertionError("rank-one lift did not give a zero-dimensional fiber")
    solution = {}
    for polynomial in basis.polys:
        expression = polynomial.as_expr()
        involved = [variable for variable in variables if expression.coeff(variable)]
        if len(involved) != 1 or sympy.Poly(expression, *variables).total_degree() != 1:
            raise AssertionError("rank-one lift did not reduce to a unique rational point")
        variable = involved[0]
        coefficient = int(expression.coeff(variable)) % PRIME
        constant = int(expression.subs(variable, 0)) % PRIME
        solution[variable] = -constant * pow(coefficient, -1, PRIME) % PRIME
    if set(solution) != set(variables):
        raise AssertionError("Groebner basis omitted a lift coordinate")

    tensor = tuple(
        (
            particular[index]
            + sum(
                kernel_basis[basis_index][index] * solution[variables[basis_index]]
                for basis_index in range(3)
            )
        )
        % PRIME
        for index in range(9)
    )
    matrix = [tensor[3 * row : 3 * row + 3] for row in range(3)]
    pivot = next(
        (row, column)
        for row in range(3)
        for column in range(3)
        if matrix[row][column]
    )
    pivot_row, pivot_column = pivot
    pivot_value = matrix[pivot_row][pivot_column]
    left = tuple(matrix[row][pivot_column] for row in range(3))
    right = tuple(
        matrix[pivot_row][column] * pow(pivot_value, -1, PRIME) % PRIME
        for column in range(3)
    )
    left = r61.r60.r58.normalized(left)
    right = r61.r60.r58.normalized(right)
    reconstructed = r61.product_output(multiplication_matrix, left, right)
    if r61.r60.r58.normalized(reconstructed) != r61.r60.r58.normalized(target):
        raise AssertionError("rank-one factors do not reconstruct the target section")
    return {
        "affine_preimage_pivots": pivots,
        "affine_preimage_free_columns": free_columns,
        "rank_one_minor_count": len(minors),
        "groebner_basis": [str(polynomial.as_expr()) for polynomial in basis.polys],
        "groebner_basis_size": len(basis.polys),
        "unique_rational_solution": [solution[variable] for variable in variables],
        "tensor": list(tensor),
        "left_coordinates": list(left),
        "right_coordinates": list(right),
    }


def factor_basis_lines(r61_report):
    result = []
    for class_sum, blocks in zip(
        CLASS_SUM_SCALARS,
        r61_report["multiplication_map"]["basis_blocks"],
    ):
        translation = (-class_sum * pow(3, -1, SUBGROUP_ORDER)) % SUBGROUP_ORDER
        lines = [
            r61.r60.r58.r44.r38.fiber_line(tuple(block), translation)
            for block in blocks
        ]
        if any(line is None for line in lines):
            raise AssertionError("stored factor basis block failed replay")
        if r61.r60.r58.r44.section_rank(lines) != 3:
            raise AssertionError("stored factor basis lines are dependent")
        result.append(
            {
                "class_sum_scalar": class_sum,
                "translation": translation,
                "blocks": blocks,
                "lines": lines,
            }
        )
    return result


def line_from_coordinates(coordinates, basis):
    line = tuple(
        sum(coordinates[index] * basis["lines"][index][coordinate] for index in range(3))
        % PRIME
        for coordinate in range(3)
    )
    return r61.r60.r58.r44.r38.normalized(line)


def point_record(point):
    return None if point is None else list(point)


def line_curve_intersection(line, translation):
    x_coefficient, y_coefficient, constant = line
    shifted_points = []
    factorization = None
    if y_coefficient:
        y_square = y_coefficient * y_coefficient % PRIME
        polynomial = (
            (constant * constant - y_square * r61.r60.r58.r44.r38.CURVE_B) % PRIME,
            (
                2 * x_coefficient * constant
                - y_square * r61.r60.r58.r44.r38.CURVE_A
            )
            % PRIME,
            x_coefficient * x_coefficient % PRIME,
            (-y_square) % PRIME,
        )
        unit, factors, roots = r61.factor_univariate(polynomial)
        factorization = {
            "variable": "x",
            "coefficients_low_to_high": list(polynomial),
            "unit": unit,
            "factors": factors,
            "rational_roots_with_multiplicity": roots,
        }
        for x_coordinate in roots:
            y_coordinate = -(
                x_coefficient * x_coordinate + constant
            ) * pow(y_coefficient, -1, PRIME) % PRIME
            shifted_points.append((x_coordinate, y_coordinate))
    else:
        if not x_coefficient:
            raise AssertionError("zero line")
        x_coordinate = -constant * pow(x_coefficient, -1, PRIME) % PRIME
        right_side = (
            x_coordinate**3
            + r61.r60.r58.r44.r38.CURVE_A * x_coordinate
            + r61.r60.r58.r44.r38.CURVE_B
        ) % PRIME
        polynomial = ((-right_side) % PRIME, 0, 1)
        unit, factors, roots = r61.factor_univariate(polynomial)
        factorization = {
            "variable": "y",
            "coefficients_low_to_high": list(polynomial),
            "unit": unit,
            "factors": factors,
            "rational_roots_with_multiplicity": roots,
        }
        shifted_points = [(x_coordinate, y_coordinate) for y_coordinate in roots]
        shifted_points.append(None)

    inverse_translation = r61.r60.r58.r44.r38.scalar_mul(
        (-translation) % SUBGROUP_ORDER,
        r61.r60.r58.r44.r38.GENERATOR,
    )
    source_points = [
        r61.r60.r58.r44.r38.add(point, inverse_translation)
        for point in shifted_points
    ]
    if any(
        point is not None
        and point[1] * point[1] % PRIME
        != (
            point[0] ** 3
            + r61.r60.r58.r44.r38.CURVE_A * point[0]
            + r61.r60.r58.r44.r38.CURVE_B
        )
        % PRIME
        for point in source_points
    ):
        raise AssertionError("recovered source point is not on the curve")
    if any(
        r61.r60.r58.r44.r38.dot(
            line,
            r61.r60.r58.r44.r38.projective(
                r61.r60.r58.r44.r38.add(
                    point,
                    r61.r60.r58.r44.r38.scalar_mul(
                        translation,
                        r61.r60.r58.r44.r38.GENERATOR,
                    ),
                )
            ),
        )
        for point in source_points
    ):
        raise AssertionError("source point does not replay in the translated line")
    projected_points = [
        r61.r60.r58.scalar_mul_integer(r61.r60.r58.COFACTOR_PROJECTOR, point)
        for point in source_points
    ]
    if any(
        r61.r60.r58.scalar_mul_integer(SUBGROUP_ORDER, point) is not None
        for point in projected_points
    ):
        raise AssertionError("publicly projected point did not land in the subgroup")
    return {
        "line": list(line),
        "translation": translation,
        "intersection_factorization": factorization,
        "rational_source_point_count": len(source_points),
        "source_points": [point_record(point) for point in source_points],
        "publicly_projected_subgroup_points": [
            point_record(point) for point in projected_points
        ],
    }


def canonical_points(points):
    return sorted(
        (None if point is None else tuple(point) for point in points),
        key=lambda point: (-1, -1) if point is None else point,
    )


def lift_target(label, target, multiplication_matrix, factor_bases):
    lift = rank_one_lift(multiplication_matrix, target)
    lines = [
        line_from_coordinates(lift["left_coordinates"], factor_bases[0]),
        line_from_coordinates(lift["right_coordinates"], factor_bases[1]),
    ]
    factors = [
        line_curve_intersection(line, basis["translation"])
        for line, basis in zip(lines, factor_bases)
    ]
    return {
        "label": label,
        "target_section": list(target),
        "rank_one_lift": lift,
        "factors": factors,
        "all_six_source_points_rational": sum(
            factor["rational_source_point_count"] for factor in factors
        )
        == 6,
    }


def run():
    r61_report = json.loads(
        pathlib.Path(
            "p1553_degree_six_product_hypersurface_locator_report_r61.json"
        ).read_text()
    )
    r60_path = pathlib.Path(
        "p1553_degree_six_primitive_product_pencil_search_report_r60.json"
    )
    r60_bytes = r60_path.read_bytes()
    r60_report = json.loads(r60_bytes)
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    factor_bases = factor_basis_lines(r61_report)
    first = tuple(r61_report["r60_pencil_restriction"]["first_section"])
    second = tuple(r61_report["r60_pencil_restriction"]["second_section"])
    finite_roots = sorted(
        set(r61_report["r60_pencil_restriction"]["finite_roots_with_multiplicity"])
    )
    lifts = [
        lift_target(
            f"t={root}",
            tuple((left + root * right) % PRIME for left, right in zip(first, second)),
            multiplication_matrix,
            factor_bases,
        )
        for root in finite_roots
    ]
    lifts.append(lift_target("t=infinity", second, multiplication_matrix, factor_bases))

    witness = next(record for record in lifts if record["label"] == "t=29")
    expected_lines = [
        tuple(record["line"])
        for record in r60_report["witness"]["third_factor_lines"]
    ]
    expected_point_coordinates = {
        record["curve_point_index"]: record["point"]
        for record in r60_report["witness"]["third_point_records"]
    }
    expected_factor_points = [
        [expected_point_coordinates[index] for index in block]
        for block in r60_report["witness"]["third_factor_point_indices"]
    ]

    checks = {
        "r61_multiplication_matrix_digest_replays": r61_report["multiplication_map"][
            "sha256"
        ]
        == "95187a49addbd3cb9cfa24e480b4b30b88fc1a928bd0a4ceab6acc6dd92ef871",
        "direct_r60_report_digest_is_pinned": hashlib.sha256(r60_bytes).hexdigest()
        == "b15ac1b6cb5df18b4fb5fdcad4d0982fb8c122777dfc6009cbd5318fc28dd364",
        "four_rational_product_targets_lifted": len(lifts) == 4,
        "every_rank_one_fiber_is_a_unique_rational_point": all(
            record["rank_one_lift"]["groebner_basis_size"] == 3
            for record in lifts
        ),
        "marked_generators_and_t29_split_completely": all(
            record["all_six_source_points_rational"]
            for record in lifts
            if record["label"] in {"t=0", "t=29", "t=infinity"}
        ),
        "t33_has_no_rational_source_points": next(
            record for record in lifts if record["label"] == "t=33"
        )["all_six_source_points_rational"]
        is False
        and sum(
            factor["rational_source_point_count"]
            for factor in next(record for record in lifts if record["label"] == "t=33")[
                "factors"
            ]
        )
        == 0,
        "t29_factor_lines_match_r60": [
            tuple(factor["line"]) for factor in witness["factors"]
        ]
        == expected_lines,
        "t29_source_points_match_r60_without_curve_scan": all(
            canonical_points(actual["source_points"]) == canonical_points(expected)
            for actual, expected in zip(witness["factors"], expected_factor_points)
        ),
        "all_recovered_points_publicly_project_to_prime_subgroup": all(
            len(factor["publicly_projected_subgroup_points"])
            == factor["rational_source_point_count"]
            for record in lifts
            for factor in record["factors"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R62 scan-free factor lift failed: {checks}")

    return {
        "schema": "p1553.degree_six_scan_free_factor_lift.r62.v1",
        "classification": [
            "toy",
            "exact-scan-free-factor-lift",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "prime_subgroup_order": SUBGROUP_ORDER,
        "class_sum_scalars": list(CLASS_SUM_SCALARS),
        "factor_bases": factor_bases,
        "r61_pencil_product_lifts": lifts,
        "checks": checks,
        "cost_boundary": {
            "claim_scope": "conditional lift after module import and precomputed fixed-class tensor and factor bases",
            "import_time_forward_scalar_labelled_subgroup_table_built": True,
            "import_time_setup_cost": "Theta(N) forward subgroup enumeration; uncharged and linearly invertible as a toy lookup table",
            "rank_one_lift": "nine quadratic 2x2 minors in three affine variables; constant-size Groebner basis",
            "factor_divisor_recovery": "factor two degree-three univariate polynomials over F_p",
            "field_element_enumeration": False,
            "curve_point_enumeration": False,
            "conditional_lift_performs_inverse_discrete_log": False,
            "conditional_lift_performs_subgroup_enumeration": False,
            "expected_online_field_operations": "O(log p) for bounded-degree finite-field factorization, excluding bit costs",
            "source_pencil_generation_charged": False,
            "restricted_factor_base_density_charged": False,
            "projected_source_scalar_logs_charged": False,
            "relation_rank_charged": False,
            "factor_base_linear_algebra_charged": False,
            "target_descent_charged": False,
        },
        "result": {
            "r60_parameter_and_factor_recovery_without_field_scan": True,
            "r60_six_source_points_recovered_without_curve_scan": True,
            "public_cofactor_projection_available_without_dlp": True,
            "source_scalar_logs_known_without_dlp": False,
            "asymptotic_relation_algorithm": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the lift is one finite toy conditioned on the precomputed fixed-class multiplication map and R60 pencil",
            "importing the legacy module chain constructs a full forward scalar-labelled subgroup table in Theta(N) work; the conditional lift claim excludes and does not charge this setup",
            "the marked source pencil was generated from scalar-labelled subgroup triples; that source-generation cost is not charged here",
            "public projection returns subgroup points but not their scalar discrete logarithms",
            "restricted factor-base density, relation rank, factor-base logs, and fresh-target descent remain absent",
            "no generic-prime asymptotic, Shoup-bound improvement, or ECDLP breakthrough is claimed",
        ],
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        pathlib.Path(arguments.output).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
