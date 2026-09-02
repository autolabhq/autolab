#!/usr/bin/env python3
"""Classify every three-block R51 line by common and residual divisors."""

import collections
import itertools
import json

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_primitive_degree_nine_rank_search_r44 as r44
import p1553_translated_triple_product_pencil_scan_r51 as r51


def run():
    items = tuple(
        (window, color)
        for window in range(r51.WINDOW_COUNT)
        for color in range(r51.COLOR_COUNT)
    )
    candidate_items = [
        combination
        for combination in itertools.combinations(items, 3)
        if sum(window for window, _ in combination)
        == r51.WINDOW_SUM_REQUIRED
    ]
    blocks = [r51.point_block(combination) for combination in candidate_items]
    sections = [
        r44.block_section(block, r51.PATH_START, r51.PATH_STEP)[0]
        for block in blocks
    ]

    line_blocks = collections.defaultdict(set)
    for first, second in itertools.combinations(range(len(sections)), 2):
        key = r46.plucker_key(sections[first], sections[second])
        line_blocks[key].add(first)
        line_blocks[key].add(second)
    three_block_lines = [
        (key, tuple(sorted(indices)))
        for key, indices in line_blocks.items()
        if len(indices) == 3
    ]
    if len(three_block_lines) != 228:
        raise AssertionError("R51 three-block line count changed")

    common_item_count_histogram = collections.Counter()
    common_point_count_histogram = collections.Counter()
    residual_window_histogram = collections.Counter()
    residual_color_set_histogram = collections.Counter()
    common_factor_pair_count = collections.Counter()
    classifications = []

    for key, indices in sorted(three_block_lines):
        item_sets = [set(candidate_items[index]) for index in indices]
        point_sets = [set(blocks[index]) for index in indices]
        common_items = set.intersection(*item_sets)
        common_points = set.intersection(*point_sets)
        residual_items = [item_set - common_items for item_set in item_sets]
        residual_points = [point_set - common_points for point_set in point_sets]

        common_item_count_histogram[len(common_items)] += 1
        common_point_count_histogram[len(common_points)] += 1
        residual_windows = tuple(
            sorted({window for residual in residual_items for window, _ in residual})
        )
        residual_colors = tuple(
            sorted({color for residual in residual_items for _, color in residual})
        )
        residual_window_histogram[residual_windows] += 1
        residual_color_set_histogram[residual_colors] += 1
        common_factor_pair_count[tuple(sorted(common_items))] += 1

        expected_common_points = {
            9 * window + local_offset
            for window, color in common_items
            for local_offset in r51.LOCAL_FIBER_PATTERNS[color]
        }
        residual_item_list = [next(iter(residual)) for residual in residual_items]
        residual_window = residual_item_list[0][0]
        expected_residual_points = [
            {
                9 * window + local_offset
                for window, color in [residual_item]
                for local_offset in r51.LOCAL_FIBER_PATTERNS[color]
            }
            for residual_item in residual_item_list
        ]
        inherited = (
            len(common_items) == 2
            and len(common_points) == 6
            and common_points == expected_common_points
            and all(len(residual) == 1 for residual in residual_items)
            and len({window for window, _ in residual_item_list}) == 1
            and {color for _, color in residual_item_list} == {0, 1, 2}
            and all(len(points) == 3 for points in residual_points)
            and all(
                points == expected
                for points, expected in zip(
                    residual_points, expected_residual_points
                )
            )
        )
        classifications.append(
            {
                "plucker_key": list(key),
                "block_indices": list(indices),
                "common_items": [list(item) for item in sorted(common_items)],
                "common_points": sorted(common_points),
                "base_divisor_degree": len(common_points),
                "residual_items": [
                    [list(item) for item in sorted(residual)]
                    for residual in residual_items
                ],
                "residual_point_blocks": [sorted(points) for points in residual_points],
                "residual_window": residual_window,
                "residual_colors": sorted(
                    color for _, color in residual_item_list
                ),
                "residual_map_degree": len(residual_points[0]),
                "inherited_translated_r38_pencil": inherited,
            }
        )

    inherited_count = sum(
        classification["inherited_translated_r38_pencil"]
        for classification in classifications
    )
    checks = {
        "all_228_three_block_lines_recovered": len(classifications) == 228,
        "all_lines_have_two_common_local_triples": common_item_count_histogram
        == {2: 228},
        "all_lines_have_six_common_divisor_points": common_point_count_histogram
        == {6: 228},
        "all_residuals_are_one_window_all_three_colors": all(
            classification["residual_colors"] == [0, 1, 2]
            and len(classification["residual_point_blocks"]) == 3
            and all(
                len(points) == 3
                for points in classification["residual_point_blocks"]
            )
            for classification in classifications
        ),
        "all_lines_inherited_translated_r38_pencils": inherited_count == 228,
        "every_common_factor_pair_names_one_line": all(
            count == 1 for count in common_factor_pair_count.values()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"shared-factor audit failed: {checks}")

    return {
        "schema": "p1553.translated_product_shared_factor_audit.r52.v1",
        "classification": [
            "toy",
            "exact",
            "exhaustive-divisor-audit",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r44.r38.PRIME,
        "subgroup_order": r44.r38.SUBGROUP_ORDER,
        "lines_audited": len(classifications),
        "histograms": {
            "common_local_triple_count": {
                str(count): common_item_count_histogram[count]
                for count in sorted(common_item_count_histogram)
            },
            "common_divisor_point_count": {
                str(count): common_point_count_histogram[count]
                for count in sorted(common_point_count_histogram)
            },
            "residual_window": {
                str(windows[0]): residual_window_histogram[windows]
                for windows in sorted(residual_window_histogram)
            },
            "residual_color_set": {
                "_".join(map(str, colors)): residual_color_set_histogram[colors]
                for colors in sorted(residual_color_set_histogram)
            },
        },
        "result": {
            "inherited_translated_r38_pencil_lines": inherited_count,
            "primitive_degree_nine_lines": len(classifications) - inherited_count,
            "base_divisor_degree": 6,
            "residual_map_degree": 3,
            "residual_fibers_per_line": 3,
        },
        "line_classifications": classifications,
        "checks": checks,
        "limits": [
            "the audit classifies only the 228 three-block lines in the R51 translated-product family",
            "it does not exclude non-product primitive degree-nine pencils",
            "the result is one finite toy and supplies no asymptotic product-line theorem",
            "no target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
