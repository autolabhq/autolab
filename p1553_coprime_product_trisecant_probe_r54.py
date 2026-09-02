#!/usr/bin/env python3
"""Probe coprime trisecants in a broadened degree-three product catalog."""

import collections
import itertools
import json
import random

import p1553_primitive_degree_nine_rank_search_r44 as r44


PATH_START = 1
PATH_STEP = 38
WINDOW_COUNT = 9
WINDOW_SIZE = 9
FACTOR_SIZE = 3
PRODUCT_FACTORS = 3
PRODUCT_SIZE = FACTOR_SIZE * PRODUCT_FACTORS
PRODUCT_OFFSET_SUM = 360
SAMPLED_COPRIME_PENCILS = 100_000
SAMPLE_SEED = 1554
PRIME = r44.r38.PRIME


def mask_to_block(mask):
    return tuple(offset for offset in range(81) if mask >> offset & 1)


def normalized_linear_combination(first, second, scalar):
    raw = [
        (left + scalar * right) % PRIME
        for left, right in zip(first, second)
    ]
    first_nonzero = next(value for value in raw if value)
    inverse = pow(first_nonzero, -1, PRIME)
    return bytes(value * inverse % PRIME for value in raw)


def interior_catalog_hits(first, second, section_index):
    hits = []
    for scalar in range(1, PRIME):
        section = normalized_linear_combination(first, second, scalar)
        index = section_index.get(section)
        if index is not None:
            hits.append((scalar, index))
    return hits


def build_catalog():
    factors = []
    for window in range(WINDOW_COUNT):
        window_points = range(
            WINDOW_SIZE * window,
            WINDOW_SIZE * (window + 1),
        )
        for points in itertools.combinations(window_points, FACTOR_SIZE):
            factors.append(
                {
                    "points": points,
                    "point_sum": sum(points),
                    "mask": sum(1 << point for point in points),
                    "window": window,
                }
            )

    factors_by_sum = collections.defaultdict(list)
    for index, factor in enumerate(factors):
        factors_by_sum[factor["point_sum"]].append(index)

    factorizations_per_block = collections.Counter()
    first_factorization = {}
    factorization_count = 0
    for first in range(len(factors)):
        first_factor = factors[first]
        for second in range(first + 1, len(factors)):
            second_factor = factors[second]
            first_two_mask = first_factor["mask"] | second_factor["mask"]
            if first_factor["mask"] & second_factor["mask"]:
                continue
            required_sum = (
                PRODUCT_OFFSET_SUM
                - first_factor["point_sum"]
                - second_factor["point_sum"]
            )
            for third in factors_by_sum.get(required_sum, ()):
                if third <= second:
                    continue
                third_factor = factors[third]
                if first_two_mask & third_factor["mask"]:
                    continue
                block_mask = first_two_mask | third_factor["mask"]
                factorization_count += 1
                factorizations_per_block[block_mask] += 1
                first_factorization.setdefault(block_mask, (first, second, third))

    block_masks = sorted(factorizations_per_block)
    blocks = [mask_to_block(mask) for mask in block_masks]
    sections = []
    section_index = {}
    for index, block in enumerate(blocks):
        section = bytes(r44.block_section(block, PATH_START, PATH_STEP)[0])
        previous = section_index.get(section)
        if previous is not None and previous != index:
            raise AssertionError("two broadened product divisors name one section")
        section_index[section] = index
        sections.append(section)

    return {
        "factors": factors,
        "factorization_count": factorization_count,
        "factorizations_per_block": factorizations_per_block,
        "first_factorization": first_factorization,
        "block_masks": block_masks,
        "blocks": blocks,
        "sections": sections,
        "section_index": section_index,
    }


def inherited_positive_control(catalog):
    with open(
        "p1553_translated_product_shared_factor_audit_report_r52.json",
        encoding="utf-8",
    ) as input_file:
        r52_report = json.load(input_file)
    first_line = r52_report["line_classifications"][0]
    control_blocks = [tuple(block) for block in first_line["residual_point_blocks"]]
    full_blocks = []
    common_points = set(first_line["common_points"])
    for residual in control_blocks:
        full_blocks.append(tuple(sorted(common_points.union(residual))))

    control_indices = [
        catalog["block_masks"].index(sum(1 << point for point in block))
        for block in full_blocks
    ]
    first_index, second_index, third_index = control_indices
    hits = interior_catalog_hits(
        catalog["sections"][first_index],
        catalog["sections"][second_index],
        catalog["section_index"],
    )
    hit_indices = {index for _, index in hits}
    return {
        "block_indices": control_indices,
        "blocks": [list(block) for block in full_blocks],
        "common_point_count": len(set.intersection(*(set(block) for block in full_blocks))),
        "interior_catalog_hits": [
            {"line_scalar": scalar, "block_index": index}
            for scalar, index in hits
        ],
        "expected_third_block_detected": third_index in hit_indices,
    }


def run():
    catalog = build_catalog()
    multiplicity_histogram = collections.Counter(
        catalog["factorizations_per_block"].values()
    )
    maximum_factorization_multiplicity = max(multiplicity_histogram)
    maximum_multiplicity_blocks = [
        mask_to_block(mask)
        for mask, multiplicity in catalog["factorizations_per_block"].items()
        if multiplicity == maximum_factorization_multiplicity
    ]
    positive_control = inherited_positive_control(catalog)

    random_source = random.Random(SAMPLE_SEED)
    sampled_pairs = set()
    attempts = 0
    positive_lines = []
    while len(sampled_pairs) < SAMPLED_COPRIME_PENCILS:
        first, second = sorted(
            random_source.sample(range(len(catalog["blocks"])), 2)
        )
        attempts += 1
        pair = (first, second)
        if pair in sampled_pairs:
            continue
        if catalog["block_masks"][first] & catalog["block_masks"][second]:
            continue
        sampled_pairs.add(pair)

        hits = interior_catalog_hits(
            catalog["sections"][first],
            catalog["sections"][second],
            catalog["section_index"],
        )
        if not hits:
            continue
        positive_lines.append(
            {
                "first_block_index": first,
                "second_block_index": second,
                "first_block": list(catalog["blocks"][first]),
                "second_block": list(catalog["blocks"][second]),
                "interior_hits": [
                    {
                        "line_scalar": scalar,
                        "block_index": index,
                        "block": list(catalog["blocks"][index]),
                    }
                    for scalar, index in hits
                ],
            }
        )

    checks = {
        "local_factor_count_is_756": len(catalog["factors"]) == 756,
        "admissible_factorization_count_is_296600": catalog[
            "factorization_count"
        ]
        == 296_600,
        "unique_product_divisor_count_is_278321": len(catalog["blocks"])
        == 278_321,
        "factorization_multiplicity_histogram_matches": multiplicity_histogram
        == {1: 276_320, 10: 2_000, 280: 1},
        "unique_maximum_multiplicity_block": len(maximum_multiplicity_blocks)
        == 1,
        "all_blocks_have_degree_nine_and_sum_360": all(
            len(block) == PRODUCT_SIZE and sum(block) == PRODUCT_OFFSET_SUM
            for block in catalog["blocks"]
        ),
        "all_product_sections_are_distinct": len(catalog["section_index"])
        == len(catalog["blocks"]),
        "r52_inherited_positive_control_has_six_base_points": positive_control[
            "common_point_count"
        ]
        == 6,
        "line_scanner_detects_r52_inherited_third_section": positive_control[
            "expected_third_block_detected"
        ],
        "sampled_pair_count_reached": len(sampled_pairs)
        == SAMPLED_COPRIME_PENCILS,
        "all_sampled_generators_are_coprime": all(
            not (
                catalog["block_masks"][first]
                & catalog["block_masks"][second]
            )
            for first, second in sampled_pairs
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R54 coprime trisecant probe failed: {checks}")

    return {
        "schema": "p1553.coprime_product_trisecant_probe.r54.v1",
        "classification": [
            "toy",
            "exact-catalog",
            "deterministic-sampled-pencils",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": r44.r38.SUBGROUP_ORDER,
        "path": {
            "start": PATH_START,
            "step": PATH_STEP,
            "selected_points": 81,
            "windows": WINDOW_COUNT,
            "window_size": WINDOW_SIZE,
        },
        "catalog": {
            "local_degree_three_factors": len(catalog["factors"]),
            "product_factors": PRODUCT_FACTORS,
            "product_degree": PRODUCT_SIZE,
            "required_offset_sum": PRODUCT_OFFSET_SUM,
            "admissible_factorizations": catalog["factorization_count"],
            "unique_product_divisors": len(catalog["blocks"]),
            "unique_product_sections": len(catalog["section_index"]),
            "factorizations_per_divisor_histogram": {
                str(multiplicity): multiplicity_histogram[multiplicity]
                for multiplicity in sorted(multiplicity_histogram)
            },
            "maximum_factorization_multiplicity": maximum_factorization_multiplicity,
            "maximum_multiplicity_blocks": [
                list(block) for block in maximum_multiplicity_blocks
            ],
        },
        "pencil_probe": {
            "sample_seed": SAMPLE_SEED,
            "sampled_coprime_pencils": len(sampled_pairs),
            "sampling_attempts": attempts,
            "interior_projective_points_checked_per_pencil": PRIME - 1,
            "interior_projective_points_checked": (PRIME - 1)
            * len(sampled_pairs),
            "positive_coprime_trisecant_lines": len(positive_lines),
            "positive_lines": positive_lines,
        },
        "inherited_positive_control": positive_control,
        "checks": checks,
        "result": {
            "primitive_coprime_trisecant_found": bool(positive_lines),
            "exhaustive_coprime_pair_classification": False,
            "asymptotic_product_line_theorem": False,
            "asymptotic_interval_pencil_family": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the 278321-section product catalog is exact only for three within-window degree-three factors on the frozen 81-point path",
            "the coprime pencil probe is a deterministic sample and not an exhaustive pair classification",
            "a zero sampled survivor count is negative evidence only for the sampled pencils",
            "no asymptotic theorem, target locator, R10 output, rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
