#!/usr/bin/env python3
"""Certify a constant-degree product-image locator for the R60 toy pencil."""

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import random
import warnings

import numpy as np
import sympy
from sympy.utilities.exceptions import SymPyDeprecationWarning

import p1553_degree_six_primitive_product_pencil_search_r60 as r60
import p1553_fixed_class_multiplication_kernel_scan_r56 as r56


PRIME = r60.r58.r44.r38.PRIME
SUBGROUP_ORDER = r60.r58.r44.r38.SUBGROUP_ORDER
CLASS_SUM_SCALARS = r60.CLASS_SUM_SCALARS
IMPLICITIZATION_SEED = 1561
MAX_EXPECTED_IMPLICIT_DEGREE = 6
PRODUCT_REPLAY_SAMPLES = 256


warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)


def homogeneous_exponents(variable_count, degree):
    if variable_count == 1:
        return [(degree,)]
    result = []
    for first in range(degree + 1):
        for tail in homogeneous_exponents(variable_count - 1, degree - first):
            result.append((first,) + tail)
    return result


def evaluate_monomials(point, exponents):
    maximum_degree = max(sum(exponent) for exponent in exponents)
    powers = [
        [pow(coordinate, degree, PRIME) for degree in range(maximum_degree + 1)]
        for coordinate in point
    ]
    return [
        math.prod(powers[index][power] for index, power in enumerate(exponent))
        % PRIME
        for exponent in exponents
    ]


def echelon_mod(matrix):
    reduced = np.array(matrix, dtype=np.int64) % PRIME
    row = 0
    pivots = []
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not len(candidates):
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            reduced[[row, pivot_row]] = reduced[[pivot_row, row]]
        inverse = pow(int(reduced[row, column]), -1, PRIME)
        reduced[row] = reduced[row] * inverse % PRIME
        if row + 1 < reduced.shape[0]:
            factors = reduced[row + 1 :, column].copy()
            reduced[row + 1 :] = (
                reduced[row + 1 :] - factors[:, None] * reduced[row]
            ) % PRIME
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced, pivots


def kernel_vector(echelon, pivots):
    free_columns = [
        column for column in range(echelon.shape[1]) if column not in pivots
    ]
    if len(free_columns) != 1:
        raise AssertionError(f"expected one kernel generator, got {free_columns}")
    vector = np.zeros(echelon.shape[1], dtype=np.int64)
    vector[free_columns[0]] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        vector[pivot] = -int(np.dot(echelon[row, pivot + 1 :], vector[pivot + 1 :]))
        vector[pivot] %= PRIME
    return r60.r58.normalized(tuple(int(value) for value in vector))


def matrix_rank_mod(matrix):
    return len(echelon_mod(matrix)[1])


def factor_catalog(class_sum):
    translation = (-class_sum * pow(3, -1, SUBGROUP_ORDER)) % SUBGROUP_ORDER
    catalog = []
    for block in itertools.combinations(range(SUBGROUP_ORDER), 3):
        if sum(block) % SUBGROUP_ORDER != class_sum:
            continue
        line = r60.r58.r44.r38.fiber_line(block, translation)
        if line is None:
            raise AssertionError("class triple failed to define its line section")
        catalog.append(
            {
                "block": block,
                "mask": sum(1 << scalar for scalar in block),
                "line": line,
                "translation": translation,
            }
        )
    return catalog


def shifted_degree_six_embedding(scalar):
    point = r60.r58.r44.r38.scalar_mul(
        (scalar + r60.TOTAL_TRANSLATION) % SUBGROUP_ORDER,
        r60.r58.r44.r38.GENERATOR,
    )
    return r60.degree_six_embedding(point)


def normalized_product_section(factors):
    block = tuple(sorted(scalar for factor in factors for scalar in factor["block"]))
    if len(set(block)) != r60.MAP_DEGREE:
        raise ValueError("repeated zero in product divisor")
    section = r60.kernel_section(shifted_degree_six_embedding(scalar) for scalar in block)
    return tuple(section), block


def product_value(factors, scalar):
    value = 1
    for factor in factors:
        point = r60.r58.r44.r38.projective(
            r60.r58.r44.r38.scalar_mul(
                (scalar + factor["translation"]) % SUBGROUP_ORDER,
                r60.r58.r44.r38.GENERATOR,
            )
        )
        value = value * r60.r58.r44.r38.dot(factor["line"], point) % PRIME
    return value


def product_trivialization_ratio(factors):
    section, _ = normalized_product_section(factors)
    ratios = {}
    for scalar in range(SUBGROUP_ORDER):
        product = product_value(factors, scalar)
        section_value = r60.r58.r44.r38.dot(
            section,
            shifted_degree_six_embedding(scalar),
        )
        if product == 0 and section_value == 0:
            continue
        if product == 0 or section_value == 0:
            raise AssertionError("reference product and degree-six zero sets differ")
        ratios[scalar] = product * pow(section_value, -1, PRIME) % PRIME
    if len(ratios) < SUBGROUP_ORDER - r60.MAP_DEGREE:
        raise AssertionError("trivialization calibration has too little support")
    return ratios


def scaled_product_section(factors, trivialization_ratio):
    interpolation_rows = []
    interpolation_values = []
    for scalar in range(SUBGROUP_ORDER):
        ratio = trivialization_ratio.get(scalar)
        if ratio is None:
            continue
        product = product_value(factors, scalar)
        row = shifted_degree_six_embedding(scalar)
        if matrix_rank_mod(interpolation_rows + [row]) == len(interpolation_rows) + 1:
            interpolation_rows.append(row)
            interpolation_values.append(product * pow(ratio, -1, PRIME) % PRIME)
        if len(interpolation_rows) == r60.MAP_DEGREE:
            break
    if len(interpolation_rows) != r60.MAP_DEGREE:
        raise AssertionError("calibrated evaluations did not span the degree-six space")
    inverse = r56.matrix_inverse_mod(interpolation_rows)
    section = r56.matrix_vector(inverse, interpolation_values)
    for scalar, ratio in trivialization_ratio.items():
        expected = product_value(factors, scalar) * pow(ratio, -1, PRIME) % PRIME
        actual = r60.r58.r44.r38.dot(
            section,
            shifted_degree_six_embedding(scalar),
        )
        if actual != expected:
            raise AssertionError("interpolated multiplication section failed replay")
    block = tuple(sorted(scalar for factor in factors for scalar in factor["block"]))
    if len(set(block)) == r60.MAP_DEGREE:
        normalized, _ = normalized_product_section(factors)
        if r60.r58.normalized(section) != normalized:
            raise AssertionError("interpolated section does not match its simple zero divisor")
    return tuple(section), block


def product_output(multiplication_matrix, left, right):
    tensor = [
        left[first] * right[second] % PRIME
        for first, second in itertools.product(range(3), repeat=2)
    ]
    return tuple(
        sum(row[column] * tensor[column] for column in range(9)) % PRIME
        for row in multiplication_matrix
    )


def build_multiplication_map(catalogs):
    basis_data = [r56.choose_factor_basis(catalog) for catalog in catalogs]
    basis_indices = [data[0] for data in basis_data]
    factor_coordinates = [data[2] for data in basis_data]
    reference = tuple(
        catalogs[index][basis_indices[index][0]] for index in range(2)
    )
    if reference[0]["mask"] & reference[1]["mask"]:
        reference = next(
            (left, right)
            for left in catalogs[0]
            for right in catalogs[1]
            if not left["mask"] & right["mask"]
        )
    trivialization_ratio = product_trivialization_ratio(reference)

    columns = []
    for first, second in itertools.product(range(3), repeat=2):
        factors = (
            catalogs[0][basis_indices[0][first]],
            catalogs[1][basis_indices[1][second]],
        )
        section, _ = scaled_product_section(factors, trivialization_ratio)
        columns.append(section)
    multiplication_matrix = [
        [columns[column][row] for column in range(9)] for row in range(6)
    ]

    replayed = 0
    for first_index, left in enumerate(catalogs[0]):
        for second_index, right in enumerate(catalogs[1]):
            if left["mask"] & right["mask"]:
                continue
            actual, _ = scaled_product_section((left, right), trivialization_ratio)
            reconstructed = product_output(
                multiplication_matrix,
                factor_coordinates[0][first_index],
                factor_coordinates[1][second_index],
            )
            if actual != reconstructed:
                raise AssertionError("degree-six multiplication tensor replay failed")
            replayed += 1
            if replayed == PRODUCT_REPLAY_SAMPLES:
                break
        if replayed == PRODUCT_REPLAY_SAMPLES:
            break
    return {
        "basis_indices": basis_indices,
        "basis_blocks": [
            [list(catalogs[index][basis]["block"]) for basis in basis_indices[index]]
            for index in range(2)
        ],
        "multiplication_matrix": multiplication_matrix,
        "rank": matrix_rank_mod(multiplication_matrix),
        "kernel_dimension": 9 - matrix_rank_mod(multiplication_matrix),
        "replayed_products": replayed,
    }


def random_product_samples(multiplication_matrix, count, seed):
    source = random.Random(seed)
    samples = []
    while len(samples) < count:
        left = tuple(source.randrange(PRIME) for _ in range(3))
        right = tuple(source.randrange(PRIME) for _ in range(3))
        if not any(left) or not any(right):
            continue
        output = product_output(multiplication_matrix, left, right)
        if any(output):
            samples.append(output)
    return samples


def implicit_equation(multiplication_matrix):
    maximum_monomials = len(homogeneous_exponents(6, MAX_EXPECTED_IMPLICIT_DEGREE))
    samples = random_product_samples(
        multiplication_matrix,
        maximum_monomials + 48,
        IMPLICITIZATION_SEED,
    )
    degree_rows = []
    equation = None
    for degree in range(1, MAX_EXPECTED_IMPLICIT_DEGREE + 1):
        exponents = homogeneous_exponents(6, degree)
        matrix = [
            evaluate_monomials(point, exponents)
            for point in samples[: len(exponents) + 32]
        ]
        echelon, pivots = echelon_mod(matrix)
        nullity = len(exponents) - len(pivots)
        degree_rows.append(
            {
                "degree": degree,
                "monomial_count": len(exponents),
                "sample_count": len(matrix),
                "rank": len(pivots),
                "nullity": nullity,
            }
        )
        if nullity:
            if nullity != 1:
                raise AssertionError("first implicit degree did not have a unique equation")
            equation = {
                "degree": degree,
                "exponents": exponents,
                "coefficients": kernel_vector(echelon, pivots),
            }
            break
    if equation is None:
        raise AssertionError("no product-image equation found through degree six")
    return equation, degree_rows


def evaluate_equation(equation, point):
    values = evaluate_monomials(point, equation["exponents"])
    return sum(
        coefficient * value
        for coefficient, value in zip(equation["coefficients"], values)
    ) % PRIME


def unisolvent_points(degree, seed):
    exponents = homogeneous_exponents(3, degree)
    source = random.Random(seed)
    candidates = []
    while len(candidates) < 3 * len(exponents):
        point = tuple(source.randrange(PRIME) for _ in range(3))
        if any(point):
            candidates.append(point)
    evaluations = [evaluate_monomials(point, exponents) for point in candidates]
    _, pivot_candidates = echelon_mod(np.array(evaluations, dtype=np.int64).T)
    if len(pivot_candidates) != len(exponents):
        raise AssertionError("failed to construct a ternary unisolvent set")
    selected = [candidates[index] for index in pivot_candidates]
    if matrix_rank_mod([evaluate_monomials(point, exponents) for point in selected]) != len(exponents):
        raise AssertionError("selected ternary evaluation set is singular")
    return selected


def certify_pullback_identity(equation, multiplication_matrix):
    left_points = unisolvent_points(equation["degree"], IMPLICITIZATION_SEED + 1)
    right_points = unisolvent_points(equation["degree"], IMPLICITIZATION_SEED + 2)
    nonzero_values = []
    for left in left_points:
        for right in right_points:
            value = evaluate_equation(
                equation,
                product_output(multiplication_matrix, left, right),
            )
            if value:
                nonzero_values.append(value)
    return {
        "ternary_form_dimension": len(left_points),
        "evaluation_count": len(left_points) * len(right_points),
        "nonzero_evaluations": len(nonzero_values),
        "identity_certified": not nonzero_values,
    }


def add_polynomials(left, right):
    size = max(len(left), len(right))
    return tuple(
        ((left[index] if index < len(left) else 0) + (right[index] if index < len(right) else 0))
        % PRIME
        for index in range(size)
    )


def multiply_polynomials(left, right):
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] = (
                result[left_degree + right_degree] + left_value * right_value
            ) % PRIME
    return tuple(result)


def scale_polynomial(polynomial, scalar):
    return tuple(scalar * value % PRIME for value in polynomial)


def restrict_to_line(equation, first, second):
    result = (0,)
    for coefficient, exponent in zip(equation["coefficients"], equation["exponents"]):
        term = (1,)
        for coordinate, power in enumerate(exponent):
            for _ in range(power):
                term = multiply_polynomials(term, (first[coordinate], second[coordinate]))
        result = add_polynomials(result, scale_polynomial(term, coefficient))
    while len(result) > 1 and result[-1] == 0:
        result = result[:-1]
    return result


def factor_univariate(polynomial):
    variable = sympy.symbols("t")
    expression = sum(
        int(coefficient) * variable**degree
        for degree, coefficient in enumerate(polynomial)
    )
    sympy_polynomial = sympy.Poly(expression, variable, modulus=PRIME)
    unit, factors = sympy.factor_list(sympy_polynomial, modulus=PRIME)
    factor_records = []
    roots = []
    for factor, multiplicity in factors:
        coefficients = [int(value) % PRIME for value in factor.all_coeffs()]
        factor_records.append(
            {
                "degree": factor.degree(),
                "multiplicity": multiplicity,
                "coefficients_high_to_low": coefficients,
            }
        )
        if factor.degree() == 1:
            roots.extend(
                [(-coefficients[1] * pow(coefficients[0], -1, PRIME)) % PRIME]
                * multiplicity
            )
    return int(unit) % PRIME, factor_records, sorted(roots)


def reconstruct_r60_sections():
    with open("p1553_degree_six_primitive_product_pencil_search_report_r60.json") as handle:
        report = json.load(handle)
    first_block = tuple(report["witness"]["first_scalar_block"])
    second_block = tuple(report["witness"]["second_scalar_block"])
    first = r60.kernel_section(shifted_degree_six_embedding(scalar) for scalar in first_block)
    second = r60.kernel_section(shifted_degree_six_embedding(scalar) for scalar in second_block)
    return report, tuple(first), tuple(second)


def partition_curve_zeros(indices, points):
    if len(indices) != r60.MAP_DEGREE:
        return None
    class_points = [
        r60.r58.r44.r38.scalar_mul(class_sum, r60.r58.r44.r38.GENERATOR)
        for class_sum in CLASS_SUM_SCALARS
    ]
    index_set = set(indices)
    for first in itertools.combinations(indices, 3):
        if r60.r58.point_sum(first, points) != class_points[0]:
            continue
        second = tuple(sorted(index_set - set(first)))
        if r60.r58.point_sum(second, points) == class_points[1]:
            return tuple(first), second
    return None


def verify_roots_on_curve(roots, first, second):
    points = r60.r58.curve_points()
    translation_point = r60.r58.r44.r38.scalar_mul(
        r60.TOTAL_TRANSLATION,
        r60.r58.r44.r38.GENERATOR,
    )
    embeddings = [
        r60.degree_six_embedding(r60.r58.r44.r38.add(point, translation_point))
        for point in points
    ]
    records = []
    for root in sorted(set(roots)):
        section = tuple(
            (left + root * right) % PRIME for left, right in zip(first, second)
        )
        zero_indices = [
            index
            for index, embedding in enumerate(embeddings)
            if r60.r58.r44.r38.dot(section, embedding) == 0
        ]
        partition = partition_curve_zeros(tuple(zero_indices), points)
        records.append(
            {
                "root": root,
                "rational_zero_count": len(zero_indices),
                "curve_point_indices": zero_indices,
                "bound_class_partition": [list(block) for block in partition]
                if partition
                else None,
            }
        )
    return records


def run():
    catalogs = [factor_catalog(class_sum) for class_sum in CLASS_SUM_SCALARS]
    multiplication = build_multiplication_map(catalogs)
    equation, degree_rows = implicit_equation(multiplication["multiplication_matrix"])
    pullback = certify_pullback_identity(equation, multiplication["multiplication_matrix"])
    r60_report, first, second = reconstruct_r60_sections()
    restricted = restrict_to_line(equation, first, second)
    unit, factor_records, roots = factor_univariate(restricted)
    verifier_roots = [
        parameter
        for parameter in range(PRIME)
        if sum(
            coefficient * pow(parameter, degree, PRIME)
            for degree, coefficient in enumerate(restricted)
        )
        % PRIME
        == 0
    ]
    root_records = verify_roots_on_curve(roots, first, second)
    witness_record = next(record for record in root_records if record["root"] == 29)

    equation_payload = {
        "degree": equation["degree"],
        "exponents": [list(exponent) for exponent in equation["exponents"]],
        "coefficients": list(equation["coefficients"]),
    }
    equation_digest = hashlib.sha256(
        json.dumps(equation_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    multiplication_digest = hashlib.sha256(
        json.dumps(
            multiplication["multiplication_matrix"],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    checks = {
        "class_catalog_sizes_are_1717_each": [len(catalog) for catalog in catalogs]
        == [1717, 1717],
        "factor_spaces_have_dimension_three": all(
            r60.r58.r44.section_rank([factor["line"] for factor in catalog]) == 3
            for catalog in catalogs
        ),
        "multiplication_map_has_rank_six": multiplication["rank"] == 6,
        "multiplication_kernel_has_dimension_three": multiplication["kernel_dimension"] == 3,
        "sampled_products_replay_exactly": multiplication["replayed_products"]
        == PRODUCT_REPLAY_SAMPLES,
        "implicit_equation_first_appears_in_degree_six": equation["degree"] == 6,
        "lower_degrees_have_zero_sample_nullity": all(
            row["nullity"] == 0 for row in degree_rows[:-1]
        ),
        "first_implicit_degree_has_unique_sample_kernel": degree_rows[-1]["nullity"] == 1,
        "pullback_identity_is_exactly_certified": pullback["identity_certified"],
        "restricted_line_polynomial_is_nonzero": any(restricted),
        "sympy_factor_roots_match_exhaustive_verifier": sorted(roots) == verifier_roots,
        "first_generator_is_finite_root_zero": 0 in roots,
        "second_generator_is_root_at_infinity": len(restricted) - 1 < equation["degree"],
        "r60_witness_parameter_is_recovered": r60_report["witness"]["line_parameter"]
        in roots,
        "r60_witness_has_six_rational_zeros": witness_record["rational_zero_count"] == 6,
        "r60_witness_partition_is_recovered": witness_record["bound_class_partition"]
        == r60_report["witness"]["third_factor_point_indices"],
    }
    if not all(checks.values()):
        raise AssertionError(f"R61 hypersurface locator failed: {checks}")

    return {
        "schema": "p1553.degree_six_product_hypersurface_locator.r61.v1",
        "classification": [
            "toy",
            "exact-algebraic-locator",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "curve_order": len(r60.r58.curve_points()),
        "prime_subgroup_order": SUBGROUP_ORDER,
        "class_sum_scalars": list(CLASS_SUM_SCALARS),
        "multiplication_map": {
            **multiplication,
            "sha256": multiplication_digest,
        },
        "implicitization": {
            "seed": IMPLICITIZATION_SEED,
            "degree_trials": degree_rows,
            "equation": equation_payload,
            "equation_sha256": equation_digest,
            "exact_pullback_certificate": pullback,
        },
        "r60_pencil_restriction": {
            "first_section": list(first),
            "second_section": list(second),
            "coefficients_low_to_high": list(restricted),
            "degree": len(restricted) - 1,
            "leading_homogeneous_coefficient_zero": len(restricted) - 1
            < equation["degree"],
            "factorization_unit": unit,
            "factorization": factor_records,
            "finite_roots_with_multiplicity": roots,
            "exhaustive_root_scan_for_verification_only": verifier_roots,
            "root_curve_verification": root_records,
        },
        "cost_boundary": {
            "claim_scope": "online parameter location conditioned on cached multiplication map, sextic equation, and pencil sections",
            "current_toy_preprocessing": "enumerates two fixed-class triple catalogs and performs subgroup-wide calibration and replay",
            "current_toy_preprocessing_cost": "Theta(N^3) triple examination plus subgroup-wide calibration work",
            "preprocessing_cost_charged_or_amortized": False,
            "online_parameter_locator": "factor one fixed-degree univariate polynomial over F_p",
            "online_locator_field_scan": False,
            "online_locator_field_operation_count": "expected O(log p) for fixed-degree finite-field factorization, excluding bit costs",
            "online_locator_field_operation_count_depends_on_p": True,
            "bit_complexity_includes_finite_field_arithmetic_and_polynomial_factorization": True,
            "curve_point_scan_used_only_for_toy_verification": True,
            "scan_free_factor_lift_from_each_hypersurface_root": False,
            "source_factor_selection_cost_charged": False,
            "factor_base_density_charged": False,
            "source_scalar_logs_charged": False,
            "relation_rank_charged": False,
            "target_descent_charged": False,
        },
        "checks": checks,
        "result": {
            "constant_degree_parameter_locator_for_r60_toy": True,
            "linear_field_scan_removed_from_parameter_location": True,
            "scan_free_factor_recovery": False,
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
            "the exact hypersurface certificate and locator are for one finite toy curve and one fixed pair of divisor classes",
            "the online locator claim is conditional on cached preprocessing; the current toy preprocessing enumerates Theta(N^3) triples and is neither charged nor amortized",
            "a hypersurface root is only a product-image membership candidate; this artifact does not recover its factor sections without a scan",
            "the exhaustive curve-point pass is a verifier and is excluded from the claimed online locator",
            "source-factor generation, factor-base density, scalar logs, relation rank, target descent, and generic-prime asymptotics remain uncharged",
            "no Shoup-bound improvement or ECDLP breakthrough is claimed",
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
