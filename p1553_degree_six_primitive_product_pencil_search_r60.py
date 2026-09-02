#!/usr/bin/env python3
"""Find the predicted constant-rate primitive degree-six product pencil."""

import collections
import itertools
import json
import random

import p1553_full_curve_primitive_product_pencil_search_r58 as r58


CLASS_SUM_SCALARS = (46, 54)
TOTAL_SUM_SCALAR = sum(CLASS_SUM_SCALARS) % r58.r44.r38.SUBGROUP_ORDER
MAP_DEGREE = 6
TOTAL_TRANSLATION = (
    -TOTAL_SUM_SCALAR
    * pow(MAP_DEGREE, -1, r58.r44.r38.SUBGROUP_ORDER)
) % r58.r44.r38.SUBGROUP_ORDER
SEARCH_SEED = 1560
EXPECTED_ATTEMPTS = 24
EXPECTED_ACCEPTED_PENCILS = 18


def degree_six_embedding(point):
    if point is None:
        return (0, 0, 0, 1, 0, 0)
    x_coordinate, y_coordinate = point
    x_squared = x_coordinate * x_coordinate % r58.r44.r38.PRIME
    return (
        1,
        x_coordinate,
        x_squared,
        x_squared * x_coordinate % r58.r44.r38.PRIME,
        y_coordinate,
        x_coordinate * y_coordinate % r58.r44.r38.PRIME,
    )


def kernel_section(embeddings):
    reduced, pivots = r58.r44.row_reduce([list(row) for row in embeddings])
    if len(pivots) != MAP_DEGREE - 1:
        raise AssertionError("degree-six divisor did not define one section")
    free_column = next(
        column for column in range(MAP_DEGREE) if column not in pivots
    )
    vector = [0] * MAP_DEGREE
    vector[free_column] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        vector[pivot] = -sum(
            reduced[row][column] * vector[column]
            for column in range(pivot + 1, MAP_DEGREE)
        ) % r58.r44.r38.PRIME
    return r58.normalized(vector)


def run():
    points = r58.curve_points()
    point_indices = {point: index for index, point in enumerate(points)}
    subgroup_points = {
        r58.r44.r38.scalar_mul(scalar, r58.r44.r38.GENERATOR): scalar
        for scalar in range(r58.r44.r38.SUBGROUP_ORDER)
    }
    two_torsion = tuple((192, 0))
    translation_point = r58.r44.r38.scalar_mul(
        TOTAL_TRANSLATION,
        r58.r44.r38.GENERATOR,
    )
    shifted_embeddings = [
        degree_six_embedding(r58.r44.r38.add(point, translation_point))
        for point in points
    ]
    subgroup_shifted_embeddings = {
        scalar: shifted_embeddings[
            point_indices[
                r58.r44.r38.scalar_mul(scalar, r58.r44.r38.GENERATOR)
            ]
        ]
        for scalar in range(r58.r44.r38.SUBGROUP_ORDER)
    }

    def subgroup_section(scalar_block):
        return kernel_section(
            subgroup_shifted_embeddings[scalar] for scalar in scalar_block
        )

    def curve_section(indices):
        return kernel_section(shifted_embeddings[index] for index in indices)

    def values(section):
        return tuple(
            sum(left * right for left, right in zip(section, point_embedding))
            % r58.r44.r38.PRIME
            for point_embedding in shifted_embeddings
        )

    class_catalogs = [
        [
            triple
            for triple in itertools.combinations(
                range(r58.r44.r38.SUBGROUP_ORDER),
                3,
            )
            if sum(triple) % r58.r44.r38.SUBGROUP_ORDER == class_sum
        ]
        for class_sum in CLASS_SUM_SCALARS
    ]
    class_sum_points = [
        r58.r44.r38.scalar_mul(class_sum, r58.r44.r38.GENERATOR)
        for class_sum in CLASS_SUM_SCALARS
    ]

    def partition(indices):
        index_set = set(indices)
        for first in itertools.combinations(indices, 3):
            if r58.point_sum(first, points) != class_sum_points[0]:
                continue
            second = tuple(sorted(index_set - set(first)))
            if r58.point_sum(second, points) == class_sum_points[1]:
                return tuple(first), second
        return None

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
            scalar_block = tuple(sorted(factors[0] + factors[1]))
            if len(set(scalar_block)) != MAP_DEGREE:
                continue
            section = subgroup_section(scalar_block)
            return scalar_block, factors, section, values(section)

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
                -first_value * pow(second_value, -1, r58.r44.r38.PRIME)
            ) % r58.r44.r38.PRIME
            if parameter:
                fibers[parameter].append(index)
        maximum_finite_fiber_histogram[
            max((len(indices) for indices in fibers.values()), default=0)
        ] += 1
        for parameter, indices in sorted(fibers.items()):
            if len(indices) != MAP_DEGREE:
                continue
            complete_rational_third_fibers += 1
            factor_partition = partition(tuple(indices))
            if factor_partition is None:
                continue
            witness = {
                "first_block": first_block,
                "second_block": second_block,
                "first_factors": first_factors,
                "second_factors": second_factors,
                "first_section": first_section,
                "second_section": second_section,
                "parameter": parameter,
                "third_indices": tuple(indices),
                "third_partition": factor_partition,
            }
            break

    first_zero_points = {
        r58.r44.r38.scalar_mul(scalar, r58.r44.r38.GENERATOR)
        for scalar in witness["first_block"]
    }
    second_zero_points = {
        r58.r44.r38.scalar_mul(scalar, r58.r44.r38.GENERATOR)
        for scalar in witness["second_block"]
    }
    third_zero_points = {points[index] for index in witness["third_indices"]}
    third_section = r58.normalized(
        tuple(
            (left + witness["parameter"] * right) % r58.r44.r38.PRIME
            for left, right in zip(
                witness["first_section"], witness["second_section"]
            )
        )
    )
    third_divisor_section = curve_section(witness["third_indices"])
    third_factor_lines = [
        r58.factor_line(block, points, class_sum)
        for block, class_sum in zip(
            witness["third_partition"], CLASS_SUM_SCALARS
        )
    ]

    third_records = []
    for index in witness["third_indices"]:
        point = points[index]
        projected = r58.scalar_mul_integer(r58.COFACTOR_PROJECTOR, point)
        third_records.append(
            {
                "curve_point_index": index,
                "point": list(point),
                "coset": "H" if point in subgroup_points else "H_plus_T2",
                "projected_subgroup_point": list(projected),
                "projected_scalar_label_for_verification_only": subgroup_points[
                    projected
                ],
            }
        )
    record_by_index = {
        record["curve_point_index"]: record for record in third_records
    }
    partition_cosets = [
        {
            "curve_point_indices": list(block),
            "outside_coset_points": sum(
                record_by_index[index]["coset"] == "H_plus_T2" for index in block
            ),
        }
        for block in witness["third_partition"]
    ]

    checks = {
        "class_catalog_sizes_are_1717_each": [
            len(catalog) for catalog in class_catalogs
        ]
        == [1717, 1717],
        "first_witness_attempt_matches": attempts == EXPECTED_ATTEMPTS,
        "first_witness_accepted_pencil_matches": accepted_pencils
        == EXPECTED_ACCEPTED_PENCILS,
        "generator_divisors_are_disjoint": not (
            first_zero_points & second_zero_points
        ),
        "all_three_fibers_pairwise_disjoint": not (
            first_zero_points & third_zero_points
            or second_zero_points & third_zero_points
        ),
        "third_fiber_has_six_distinct_rational_points": len(third_zero_points)
        == MAP_DEGREE,
        "third_section_matches_divisor_kernel": third_section
        == third_divisor_section,
        "three_sections_have_rank_two": r58.r44.section_rank(
            [witness["first_section"], witness["second_section"], third_section]
        )
        == 2,
        "third_relation_parameter_is_29": witness["parameter"] == 29,
        "third_divisor_partitions_into_bound_classes": all(
            r58.point_sum(block, points) == class_sum_point
            for block, class_sum_point in zip(
                witness["third_partition"], class_sum_points
            )
        ),
        "all_third_factors_are_exact_line_sections": len(third_factor_lines) == 2,
        "each_third_factor_has_even_torsion_parity": all(
            record["outside_coset_points"] % 2 == 0
            for record in partition_cosets
        ),
        "public_projection_recovers_all_six_subgroup_atoms": all(
            tuple(record["projected_subgroup_point"]) in subgroup_points
            for record in third_records
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R60 degree-six search failed: {checks}")

    return {
        "schema": "p1553.degree_six_primitive_product_pencil_search.r60.v1",
        "classification": [
            "toy",
            "exact-positive-construction-seed",
            "deterministic-search",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r58.r44.r38.PRIME,
        "curve_order": len(points),
        "prime_subgroup_order": r58.r44.r38.SUBGROUP_ORDER,
        "class_sum_scalars": list(CLASS_SUM_SCALARS),
        "total_sum_scalar": TOTAL_SUM_SCALAR,
        "total_translation": TOTAL_TRANSLATION,
        "map_degree": MAP_DEGREE,
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
            "third_curve_point_indices": list(witness["third_indices"]),
            "third_factor_point_indices": [
                list(block) for block in witness["third_partition"]
            ],
            "third_factor_lines": [
                {"line": list(line), "translation": translation}
                for line, translation in third_factor_lines
            ],
            "third_point_records": third_records,
            "third_factor_coset_counts": partition_cosets,
            "common_base_divisor_degree": 0,
            "effective_map_degree": MAP_DEGREE,
        },
        "checks": checks,
        "result": {
            "primitive_coprime_degree_six_product_pencil_found": True,
            "observed_constant_rate_model_matches": True,
            "third_fiber_source_points_publicly_projectable_without_dlp": True,
            "third_fiber_scalar_logs_known_without_dlp": False,
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
            "the witness is one finite toy and does not establish a generic-prime family",
            "constant split-class rate does not include the cost of choosing source factors or solving their logarithms",
            "the search supplies no factor-base density analysis, target locator, relation rank, or descent",
            "no R10 output, complete cost model, Shoup-bound improvement, or breakthrough is supplied",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
