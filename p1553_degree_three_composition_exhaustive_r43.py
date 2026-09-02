#!/usr/bin/env python3
"""Exhaust degree-three rational pencils as inner degree-nine toy maps."""

import collections
import json

import p1553_low_boundary_pencil_search_r38 as r38


def all_curve_points():
    points = [None]
    for x_coordinate in range(r38.PRIME):
        right_hand_side = (
            x_coordinate**3
            + r38.CURVE_A * x_coordinate
            + r38.CURVE_B
        ) % r38.PRIME
        for y_coordinate in range(r38.PRIME):
            if y_coordinate * y_coordinate % r38.PRIME == right_hand_side:
                points.append((x_coordinate, y_coordinate))
    return points


def projective_base_points():
    for x_coordinate in range(r38.PRIME):
        for y_coordinate in range(r38.PRIME):
            yield (x_coordinate, y_coordinate, 1)
    for y_coordinate in range(r38.PRIME):
        yield (1, y_coordinate, 0)
    yield (0, 1, 0)


def base_point_on_curve(base_point):
    x_coordinate, y_coordinate, z_coordinate = base_point
    return (
        y_coordinate * y_coordinate * z_coordinate
        - x_coordinate**3
        - r38.CURVE_A * x_coordinate * z_coordinate**2
        - r38.CURVE_B * z_coordinate**3
    ) % r38.PRIME == 0


def complete_triple_coverage(points):
    histogram = collections.Counter()
    maxima = []
    maximum_base_point_count = 0
    maximum_coverage = -1
    valid_base_points = 0
    projective_points = [r38.projective(point) for point in points]

    for base_point in projective_base_points():
        if base_point_on_curve(base_point):
            continue
        valid_base_points += 1
        fibers = {}
        for scalar_index, point in enumerate(projective_points):
            line = r38.normalized(r38.cross(base_point, point))
            fibers.setdefault(line, []).append(scalar_index)
        triple_fibers = sorted(
            sorted(fiber) for fiber in fibers.values() if len(fiber) == 3
        )
        coverage = 3 * len(triple_fibers)
        histogram[coverage] += 1
        if coverage > maximum_coverage:
            maximum_coverage = coverage
            maximum_base_point_count = 1
            maxima = [
                {
                    "base_point": list(base_point),
                    "complete_triple_fibers": triple_fibers,
                }
            ]
        elif coverage == maximum_coverage:
            maximum_base_point_count += 1
            if len(maxima) < 10:
                maxima.append(
                    {
                        "base_point": list(base_point),
                        "complete_triple_fibers": triple_fibers,
                    }
                )

    return {
        "valid_base_points": valid_base_points,
        "coverage_histogram": {
            str(coverage): histogram[coverage] for coverage in sorted(histogram)
        },
        "maximum_complete_triple_coverage": maximum_coverage,
        "maximum_base_point_count": maximum_base_point_count,
        "maximum_base_points": maxima,
    }


def original_pencil_interval_scan(pencil):
    translation = pencil["embedding_translation_scalar"]
    numerator = tuple(pencil["pencil_lines"][0])
    denominator = tuple(pencil["pencil_lines"][1])
    values = [
        r38.pencil_value(scalar, translation, numerator, denominator)
        for scalar in range(r38.SUBGROUP_ORDER)
    ]
    histogram = collections.Counter()
    best = None
    eligible = 0
    for step in range(1, r38.SUBGROUP_ORDER):
        for start in range(r38.SUBGROUP_ORDER):
            interval = [
                (start + offset * step) % r38.SUBGROUP_ORDER
                for offset in range(81)
            ]
            counts = collections.Counter(values[scalar] for scalar in interval)
            distinct_values = len(counts)
            full_fibers = sum(multiplicity == 3 for multiplicity in counts.values())
            histogram[distinct_values] += 1
            if distinct_values <= 27:
                eligible += 1
            candidate = (
                distinct_values,
                -full_fibers,
                step,
                start,
                collections.Counter(counts.values()),
            )
            if best is None or candidate[:4] < best[:4]:
                best = candidate

    return {
        "intervals_tested": (r38.SUBGROUP_ORDER - 1) * r38.SUBGROUP_ORDER,
        "distinct_inner_value_histogram": {
            str(value): histogram[value] for value in sorted(histogram)
        },
        "intervals_with_at_most_27_inner_values": eligible,
        "minimum_distinct_inner_values": best[0],
        "best_full_three_point_fibers": -best[1],
        "best_step": best[2],
        "best_start": best[3],
        "best_inner_multiplicity_histogram": {
            str(multiplicity): best[4][multiplicity]
            for multiplicity in sorted(best[4])
        },
    }


def run():
    pencil = r38.search()
    if not pencil["pass"]:
        raise AssertionError("R38 positive pencil control did not replay")

    curve_points = all_curve_points()
    subgroup = {
        r38.scalar_mul(scalar, r38.GENERATOR)
        for scalar in range(r38.SUBGROUP_ORDER)
    }
    outside_representative = next(
        point for point in curve_points if point not in subgroup
    )
    subgroup_points = [
        r38.scalar_mul(scalar, r38.GENERATOR)
        for scalar in range(r38.SUBGROUP_ORDER)
    ]
    outside_coset_points = [
        r38.add(outside_representative, point) for point in subgroup_points
    ]

    subgroup_coverage = complete_triple_coverage(subgroup_points)
    outside_coverage = complete_triple_coverage(outside_coset_points)
    interval_scan = original_pencil_interval_scan(pencil)

    required_selected_points = 81
    required_inner_values = 27
    checks = {
        "curve_order_is_206": len(curve_points) == 206,
        "subgroup_order_is_103": len(subgroup) == 103,
        "rational_curve_has_two_subgroup_cosets": len(curve_points)
        == 2 * len(subgroup),
        "valid_pencil_centers_count": subgroup_coverage["valid_base_points"]
        == outside_coverage["valid_base_points"]
        == r38.PRIME**2 + r38.PRIME + 1 - len(curve_points),
        "subgroup_maximum_triple_coverage_is_51": subgroup_coverage[
            "maximum_complete_triple_coverage"
        ]
        == 51,
        "outside_coset_has_no_complete_triple": outside_coverage[
            "maximum_complete_triple_coverage"
        ]
        == 0,
        "no_degree_three_inner_map_has_required_81_point_coverage": max(
            subgroup_coverage["maximum_complete_triple_coverage"],
            outside_coverage["maximum_complete_triple_coverage"],
        )
        < required_selected_points,
        "all_original_pencil_intervals_tested": interval_scan[
            "intervals_tested"
        ]
        == 10506,
        "original_pencil_has_no_27_value_interval": interval_scan[
            "intervals_with_at_most_27_inner_values"
        ]
        == 0,
        "original_pencil_minimum_inner_values_is_60": interval_scan[
            "minimum_distinct_inner_values"
        ]
        == 60,
    }
    if not all(checks.values()):
        raise AssertionError("degree-three composition exhaustive check failed")

    return {
        "schema": "p1553.degree_three_composition_exhaustive.r43.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r38.PRIME,
        "curve_order": len(curve_points),
        "subgroup_order": len(subgroup),
        "outside_coset_representative": list(outside_representative),
        "projective_plane_points": r38.PRIME**2 + r38.PRIME + 1,
        "rational_pencil_centers_off_curve": subgroup_coverage[
            "valid_base_points"
        ],
        "degree_nine_composition_requirement": {
            "selected_points": required_selected_points,
            "outer_selected_values": 9,
            "outer_degree": 3,
            "maximum_inner_values": required_inner_values,
            "required_complete_inner_triple_coverage": required_selected_points,
        },
        "subgroup_embedding": subgroup_coverage,
        "outside_coset_embedding": outside_coverage,
        "original_r38_pencil_interval_scan": interval_scan,
        "checks": checks,
        "limits": [
            "the exhaustive result is for rational degree-three maps on the R38 toy curve",
            "it excludes degree-nine maps only when they factor through a degree-three inner map",
            "a primitive degree-nine pencil is not excluded",
            "the result is not an asymptotic lower bound",
            "no target locator, R10 output, rank, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
