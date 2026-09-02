#!/usr/bin/env python3
"""Find a primitive degree-nine product pencil over all rational curve points."""

import collections
import itertools
import json
import random

import p1553_primitive_degree_nine_rank_search_r44 as r44


CLASS_SUM_SCALARS = (46, 54, 96)
TOTAL_SUM_SCALAR = sum(CLASS_SUM_SCALARS) % r44.r38.SUBGROUP_ORDER
TOTAL_TRANSLATION = (
    -TOTAL_SUM_SCALAR * pow(9, -1, r44.r38.SUBGROUP_ORDER)
) % r44.r38.SUBGROUP_ORDER
SEARCH_SEED = 1559
EXPECTED_WITNESS_ATTEMPTS = 25_996
EXPECTED_ACCEPTED_PENCILS = 11_104
COFACTOR_PROJECTOR = 104


def curve_points():
    points = [None]
    for x_coordinate in range(r44.r38.PRIME):
        right_side = (
            x_coordinate**3
            + r44.r38.CURVE_A * x_coordinate
            + r44.r38.CURVE_B
        ) % r44.r38.PRIME
        for y_coordinate in range(r44.r38.PRIME):
            if y_coordinate * y_coordinate % r44.r38.PRIME == right_side:
                points.append((x_coordinate, y_coordinate))
    return points


def scalar_mul_integer(scalar, point):
    result = None
    addend = point
    while scalar:
        if scalar & 1:
            result = r44.r38.add(result, addend)
        addend = r44.r38.add(addend, addend)
        scalar >>= 1
    return result


def embedding(point):
    if point is None:
        return (0, 0, 0, 0, 0, 0, 0, 0, 1)
    x_coordinate, y_coordinate = point
    x_squared = x_coordinate * x_coordinate % r44.r38.PRIME
    x_cubed = x_squared * x_coordinate % r44.r38.PRIME
    x_fourth = x_cubed * x_coordinate % r44.r38.PRIME
    return (
        1,
        x_coordinate,
        x_squared,
        x_cubed,
        x_fourth,
        y_coordinate,
        x_coordinate * y_coordinate % r44.r38.PRIME,
        x_squared * y_coordinate % r44.r38.PRIME,
        x_cubed * y_coordinate % r44.r38.PRIME,
    )


def kernel_section(embeddings):
    reduced, pivots = r44.row_reduce([list(row) for row in embeddings])
    if len(pivots) != 8:
        raise AssertionError("degree-nine divisor did not define one section")
    free_column = next(column for column in range(9) if column not in pivots)
    vector = [0] * 9
    vector[free_column] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        vector[pivot] = -sum(
            reduced[row][column] * vector[column]
            for column in range(pivot + 1, 9)
        ) % r44.r38.PRIME
    first_nonzero = next(value for value in vector if value)
    inverse = pow(first_nonzero, -1, r44.r38.PRIME)
    return tuple(value * inverse % r44.r38.PRIME for value in vector)


def subgroup_section(scalar_block):
    return kernel_section(
        r44.SECTION_EMBEDDINGS[
            (scalar + TOTAL_TRANSLATION) % r44.r38.SUBGROUP_ORDER
        ]
        for scalar in scalar_block
    )


def curve_section(point_indices, points):
    translation_point = r44.r38.scalar_mul(
        TOTAL_TRANSLATION,
        r44.r38.GENERATOR,
    )
    return kernel_section(
        embedding(r44.r38.add(points[index], translation_point))
        for index in point_indices
    )


def section_values(section, shifted_embeddings):
    return tuple(
        sum(left * right for left, right in zip(section, point_embedding))
        % r44.r38.PRIME
        for point_embedding in shifted_embeddings
    )


def normalized(vector):
    first_nonzero = next(value for value in vector if value)
    inverse = pow(first_nonzero, -1, r44.r38.PRIME)
    return tuple(value * inverse % r44.r38.PRIME for value in vector)


def point_sum(indices, points):
    result = None
    for index in indices:
        result = r44.r38.add(result, points[index])
    return result


def class_partition(point_indices, points, class_sum_points):
    point_set = set(point_indices)
    for first in itertools.combinations(point_indices, 3):
        if point_sum(first, points) != class_sum_points[0]:
            continue
        remaining = point_set - set(first)
        for second in itertools.combinations(sorted(remaining), 3):
            if point_sum(second, points) != class_sum_points[1]:
                continue
            third = tuple(sorted(remaining - set(second)))
            if point_sum(third, points) == class_sum_points[2]:
                return tuple(tuple(block) for block in (first, second, third))
    return None


def factor_line(point_indices, points, class_sum_scalar):
    translation = (
        -class_sum_scalar * pow(3, -1, r44.r38.SUBGROUP_ORDER)
    ) % r44.r38.SUBGROUP_ORDER
    translation_point = r44.r38.scalar_mul(translation, r44.r38.GENERATOR)
    shifted = [
        r44.r38.projective(r44.r38.add(points[index], translation_point))
        for index in point_indices
    ]
    line = r44.r38.cross(shifted[0], shifted[1])
    if line == (0, 0, 0) or r44.r38.dot(line, shifted[2]):
        raise AssertionError("class triple did not define a degree-three line")
    return r44.r38.normalized(line), translation


def run():
    points = curve_points()
    if len(points) != 206:
        raise AssertionError("unexpected rational curve order")
    subgroup_points = {
        r44.r38.scalar_mul(scalar, r44.r38.GENERATOR): scalar
        for scalar in range(r44.r38.SUBGROUP_ORDER)
    }
    two_torsion = next(
        point for point in points if point is not None and point[1] == 0
    )
    if two_torsion in subgroup_points:
        raise AssertionError("two-torsion point entered the odd-order subgroup")

    class_catalogs = [
        [
            triple
            for triple in itertools.combinations(
                range(r44.r38.SUBGROUP_ORDER),
                3,
            )
            if sum(triple) % r44.r38.SUBGROUP_ORDER == class_sum
        ]
        for class_sum in CLASS_SUM_SCALARS
    ]
    class_sum_points = [
        r44.r38.scalar_mul(class_sum, r44.r38.GENERATOR)
        for class_sum in CLASS_SUM_SCALARS
    ]
    translation_point = r44.r38.scalar_mul(
        TOTAL_TRANSLATION,
        r44.r38.GENERATOR,
    )
    shifted_embeddings = [
        embedding(r44.r38.add(point, translation_point)) for point in points
    ]

    random_source = random.Random(SEARCH_SEED)
    attempts = 0
    accepted_pencils = 0
    complete_rational_third_fibers = 0
    maximum_finite_fiber_histogram = collections.Counter()
    witness = None

    def random_product():
        while True:
            factors = tuple(
                random_source.choice(catalog) for catalog in class_catalogs
            )
            scalar_block = tuple(
                sorted(scalar for factor in factors for scalar in factor)
            )
            if len(set(scalar_block)) != 9:
                continue
            section = subgroup_section(scalar_block)
            values = section_values(section, shifted_embeddings)
            return scalar_block, factors, section, values

    while witness is None:
        attempts += 1
        first_block, first_factors, first_section, first_values = random_product()
        second_block, second_factors, second_section, second_values = (
            random_product()
        )
        if set(first_block) & set(second_block):
            continue
        accepted_pencils += 1

        fibers = collections.defaultdict(list)
        for index, (first_value, second_value) in enumerate(
            zip(first_values, second_values)
        ):
            if second_value == 0:
                continue
            parameter = (
                -first_value * pow(second_value, -1, r44.r38.PRIME)
            ) % r44.r38.PRIME
            if parameter:
                fibers[parameter].append(index)
        maximum_finite_fiber_histogram[
            max((len(indices) for indices in fibers.values()), default=0)
        ] += 1

        for parameter, point_indices in sorted(fibers.items()):
            if len(point_indices) != 9:
                continue
            complete_rational_third_fibers += 1
            partition = class_partition(
                tuple(point_indices),
                points,
                class_sum_points,
            )
            if partition is None:
                continue
            witness = {
                "first_block": first_block,
                "second_block": second_block,
                "first_factors": first_factors,
                "second_factors": second_factors,
                "first_section": first_section,
                "second_section": second_section,
                "parameter": parameter,
                "third_point_indices": tuple(point_indices),
                "third_partition": partition,
            }
            break

    first_zero_points = {
        r44.r38.scalar_mul(scalar, r44.r38.GENERATOR)
        for scalar in witness["first_block"]
    }
    second_zero_points = {
        r44.r38.scalar_mul(scalar, r44.r38.GENERATOR)
        for scalar in witness["second_block"]
    }
    third_zero_points = {
        points[index] for index in witness["third_point_indices"]
    }
    third_section = normalized(
        tuple(
            (left + witness["parameter"] * right) % r44.r38.PRIME
            for left, right in zip(
                witness["first_section"],
                witness["second_section"],
            )
        )
    )
    third_divisor_section = curve_section(witness["third_point_indices"], points)

    third_factor_lines = [
        factor_line(block, points, class_sum)
        for block, class_sum in zip(
            witness["third_partition"],
            CLASS_SUM_SCALARS,
        )
    ]
    third_point_records = []
    for index in witness["third_point_indices"]:
        point = points[index]
        projected_point = scalar_mul_integer(COFACTOR_PROJECTOR, point)
        if projected_point not in subgroup_points:
            raise AssertionError("public cofactor projection missed the subgroup")
        if point in subgroup_points:
            coset = "H"
            scalar_label = subgroup_points[point]
        else:
            coset = "H_plus_T2"
            scalar_label = subgroup_points[r44.r38.add(point, two_torsion)]
        third_point_records.append(
            {
                "curve_point_index": index,
                "point": None if point is None else list(point),
                "coset": coset,
                "projected_subgroup_point": None
                if projected_point is None
                else list(projected_point),
                "scalar_label_after_torsion_removal": scalar_label,
            }
        )

    partition_coset_counts = []
    for block in witness["third_partition"]:
        outside_count = sum(points[index] not in subgroup_points for index in block)
        partition_coset_counts.append(
            {
                "curve_point_indices": list(block),
                "subgroup_points": 3 - outside_count,
                "outside_coset_points": outside_count,
                "torsion_parity_even": outside_count % 2 == 0,
            }
        )

    checks = {
        "curve_order_is_206": len(points) == 206,
        "prime_subgroup_has_103_points": len(subgroup_points) == 103,
        "rational_two_torsion_is_192_0": two_torsion == (192, 0),
        "factor_catalog_sizes_are_1717_each": [
            len(catalog) for catalog in class_catalogs
        ]
        == [1717, 1717, 1717],
        "first_witness_attempt_matches": attempts == EXPECTED_WITNESS_ATTEMPTS,
        "first_witness_accepted_pencil_matches": accepted_pencils
        == EXPECTED_ACCEPTED_PENCILS,
        "generator_divisors_are_disjoint": not (
            first_zero_points & second_zero_points
        ),
        "all_three_fibers_pairwise_disjoint": not (
            first_zero_points & third_zero_points
            or second_zero_points & third_zero_points
        ),
        "third_fiber_has_nine_distinct_rational_points": len(third_zero_points)
        == 9,
        "third_section_matches_divisor_kernel": third_section
        == third_divisor_section,
        "three_sections_have_rank_two": r44.section_rank(
            [
                witness["first_section"],
                witness["second_section"],
                third_section,
            ]
        )
        == 2,
        "third_relation_parameter_is_33": witness["parameter"] == 33,
        "third_divisor_partitions_into_bound_classes": all(
            point_sum(block, points) == class_sum_point
            for block, class_sum_point in zip(
                witness["third_partition"],
                class_sum_points,
            )
        ),
        "all_third_factors_are_exact_line_sections": len(third_factor_lines) == 3,
        "third_fiber_has_three_H_and_six_H_plus_T2_points": collections.Counter(
            record["coset"] for record in third_point_records
        )
        == {"H": 3, "H_plus_T2": 6},
        "each_third_factor_has_even_torsion_parity": all(
            record["torsion_parity_even"] for record in partition_coset_counts
        ),
        "public_104_projection_recovers_each_subgroup_atom": all(
            scalar_mul_integer(COFACTOR_PROJECTOR, points[record["curve_point_index"]])
            == r44.r38.scalar_mul(
                record["scalar_label_after_torsion_removal"],
                r44.r38.GENERATOR,
            )
            for record in third_point_records
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R58 primitive product-pencil search failed: {checks}")

    return {
        "schema": "p1553.full_curve_primitive_product_pencil_search.r58.v1",
        "classification": [
            "toy",
            "exact-positive-construction-seed",
            "deterministic-search",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r44.r38.PRIME,
        "curve": {"a4": r44.r38.CURVE_A, "a6": r44.r38.CURVE_B},
        "curve_order": len(points),
        "prime_subgroup_order": r44.r38.SUBGROUP_ORDER,
        "two_torsion_point": list(two_torsion),
        "public_cofactor_projector": COFACTOR_PROJECTOR,
        "class_sum_scalars": list(CLASS_SUM_SCALARS),
        "total_sum_scalar": TOTAL_SUM_SCALAR,
        "total_translation": TOTAL_TRANSLATION,
        "factor_catalog_sizes": [len(catalog) for catalog in class_catalogs],
        "search": {
            "seed": SEARCH_SEED,
            "attempts": attempts,
            "accepted_coprime_pencils": accepted_pencils,
            "complete_rational_third_fibers_seen": complete_rational_third_fibers,
            "maximum_finite_fiber_histogram": {
                str(size): maximum_finite_fiber_histogram[size]
                for size in sorted(maximum_finite_fiber_histogram)
            },
        },
        "witness": {
            "first_scalar_block": list(witness["first_block"]),
            "second_scalar_block": list(witness["second_block"]),
            "first_factor_triples": [
                list(block) for block in witness["first_factors"]
            ],
            "second_factor_triples": [
                list(block) for block in witness["second_factors"]
            ],
            "line_parameter": witness["parameter"],
            "third_curve_point_indices": list(witness["third_point_indices"]),
            "third_point_records": third_point_records,
            "third_factor_point_indices": [
                list(block) for block in witness["third_partition"]
            ],
            "third_factor_lines": [
                {"line": list(line), "translation": translation}
                for line, translation in third_factor_lines
            ],
            "third_factor_coset_counts": partition_coset_counts,
            "common_base_divisor_degree": 0,
            "effective_map_degree": 9,
        },
        "checks": checks,
        "result": {
            "primitive_coprime_degree_nine_product_pencil_found": True,
            "third_fiber_source_points_publicly_projectable_without_dlp": True,
            "third_fiber_scalar_logs_known_without_dlp": False,
            "asymptotic_pencil_family": False,
            "low_boundary_family": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the witness is one finite toy and does not supply an asymptotic family",
            "six third-fiber points lie outside the prime subgroup; public cofactor projection recovers subgroup points, but their scalar logs remain unknown factor-base variables",
            "the search does not supply a target locator, low-boundary schedule, or source-invertible descent",
            "no R10 output, relation-rank campaign, factor-base logs, or Shoup-bound improvement is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
