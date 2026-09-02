#!/usr/bin/env python3
"""Test degree-nine pencils assembled from translated R38 triple fibers."""

import collections
import itertools
import json

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_primitive_degree_nine_rank_search_r44 as r44


WINDOW_COUNT = 9
COLOR_COUNT = 3
PATH_START = 1
PATH_STEP = 38
LOCAL_FIBER_PATTERNS = (
    (0, 5, 7),
    (1, 3, 8),
    (2, 4, 6),
)
WINDOW_SUM_REQUIRED = 12


def item_index(window, color):
    return COLOR_COUNT * window + color


def point_block(items):
    return tuple(
        sorted(
            9 * window + local_offset
            for window, color in items
            for local_offset in LOCAL_FIBER_PATTERNS[color]
        )
    )


def exact_cover_exists(block_indices, item_masks):
    full_mask = (1 << (WINDOW_COUNT * COLOR_COUNT)) - 1
    candidates_by_item = [[] for _ in range(WINDOW_COUNT * COLOR_COUNT)]
    for block_index in block_indices:
        mask = item_masks[block_index]
        for index in range(WINDOW_COUNT * COLOR_COUNT):
            if mask >> index & 1:
                candidates_by_item[index].append(block_index)

    def search(used_mask, chosen):
        if used_mask == full_mask:
            return tuple(chosen)
        uncovered = [
            index
            for index in range(WINDOW_COUNT * COLOR_COUNT)
            if not (used_mask >> index & 1)
        ]
        next_item = min(
            uncovered,
            key=lambda index: sum(
                item_masks[block_index] & used_mask == 0
                for block_index in candidates_by_item[index]
            ),
        )
        for block_index in candidates_by_item[next_item]:
            mask = item_masks[block_index]
            if mask & used_mask:
                continue
            result = search(used_mask | mask, chosen + [block_index])
            if result is not None:
                return result
        return None

    return search(0, [])


def run():
    with open(
        "p1553_low_boundary_pencil_search_report_r38.json", encoding="utf-8"
    ) as input_file:
        r38_report = json.load(input_file)
    if tuple(r38_report["selected_scalars_in_path_order"]) != tuple(
        (PATH_START + PATH_STEP * offset) % r44.r38.SUBGROUP_ORDER
        for offset in range(9)
    ):
        raise AssertionError("R38 path coordinates changed")

    items = tuple(
        (window, color)
        for window in range(WINDOW_COUNT)
        for color in range(COLOR_COUNT)
    )
    candidate_items = [
        combination
        for combination in itertools.combinations(items, 3)
        if sum(window for window, _ in combination) == WINDOW_SUM_REQUIRED
    ]
    if len(candidate_items) != 253:
        raise AssertionError("unexpected translated triple-union block count")

    blocks = [point_block(combination) for combination in candidate_items]
    item_masks = [
        sum(1 << item_index(window, color) for window, color in combination)
        for combination in candidate_items
    ]
    if len(set(blocks)) != len(blocks) or len(set(item_masks)) != len(item_masks):
        raise AssertionError("triple-union blocks did not stay unique")
    if any(len(block) != 9 or sum(block) != 360 for block in blocks):
        raise AssertionError("triple-union block does not have degree nine and sum 360")

    sections = [
        r44.block_section(block, PATH_START, PATH_STEP)[0] for block in blocks
    ]
    if len(set(sections)) != len(sections):
        raise AssertionError("triple-union section map is not injective")

    line_blocks = collections.defaultdict(set)
    for first, second in itertools.combinations(range(len(sections)), 2):
        key = r46.plucker_key(sections[first], sections[second])
        line_blocks[key].add(first)
        line_blocks[key].add(second)
    line_size_histogram = collections.Counter(
        len(indices) for indices in line_blocks.values()
    )
    maximum_line_size = max(line_size_histogram)
    candidate_cover_lines = []
    for key, indices in line_blocks.items():
        if len(indices) < 9:
            continue
        cover = exact_cover_exists(sorted(indices), item_masks)
        if cover is not None:
            candidate_cover_lines.append(
                {
                    "plucker_key": list(key),
                    "line_block_count": len(indices),
                    "cover_block_indices": list(cover),
                    "cover_point_blocks": [list(blocks[index]) for index in cover],
                }
            )

    first_window_partition = [
        index
        for index, combination in enumerate(candidate_items)
        if {window for window, _ in combination} in (
            {0, 4, 8},
            {1, 5, 6},
            {2, 3, 7},
        )
        and len({color for _, color in combination}) == 1
    ]
    canonical_cover = exact_cover_exists(first_window_partition, item_masks)
    if canonical_cover is None:
        raise AssertionError("canonical translated-product cover is missing")
    canonical_rank = r44.section_rank(
        tuple(sections[index] for index in canonical_cover)
    )

    checks = {
        "r38_local_fiber_patterns_reconstructed": tuple(
            sorted(
                tuple(
                    sorted(
                        r38_report["selected_scalars_in_path_order"].index(scalar)
                        for scalar in block
                    )
                )
                for block in r38_report["fiber_scalar_blocks"]
            )
        )
        == tuple(sorted(LOCAL_FIBER_PATTERNS)),
        "candidate_block_count_is_253": len(candidate_items) == 253,
        "all_blocks_degree_nine_and_sum_360": all(
            len(block) == 9 and sum(block) == 360 for block in blocks
        ),
        "all_sections_distinct": len(set(sections)) == 253,
        "all_section_pairs_accounted": sum(
            count * size * (size - 1) // 2
            for size, count in line_size_histogram.items()
        )
        == len(sections) * (len(sections) - 1) // 2,
        "maximum_pencil_line_size_recorded": maximum_line_size
        == max(len(indices) for indices in line_blocks.values()),
        "canonical_exact_cover_exists": canonical_cover is not None,
        "no_rank_two_exact_cover_line": not candidate_cover_lines,
    }
    if not all(checks.values()):
        raise AssertionError(
            f"translated triple-product pencil scan failed: {checks}"
        )

    return {
        "schema": "p1553.translated_triple_product_pencil_scan.r51.v1",
        "classification": [
            "toy",
            "exact",
            "exhaustive-structured-family",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r44.r38.PRIME,
        "subgroup_order": r44.r38.SUBGROUP_ORDER,
        "r38_local_pattern": {
            "path_start": PATH_START,
            "path_step": PATH_STEP,
            "local_fiber_patterns": [list(pattern) for pattern in LOCAL_FIBER_PATTERNS],
        },
        "construction": {
            "translated_windows": WINDOW_COUNT,
            "local_fibers_per_window": COLOR_COUNT,
            "labeled_local_triples": len(items),
            "triple_unions_per_degree_nine_block": 3,
            "required_window_index_sum": WINDOW_SUM_REQUIRED,
            "candidate_degree_nine_blocks": len(blocks),
            "section_pairs": len(sections) * (len(sections) - 1) // 2,
        },
        "pencil_lines": {
            "distinct_plucker_lines": len(line_blocks),
            "line_size_histogram": {
                str(size): line_size_histogram[size]
                for size in sorted(line_size_histogram)
            },
            "maximum_blocks_on_one_line": maximum_line_size,
            "required_blocks_on_one_line": 9,
            "exact_cover_lines": candidate_cover_lines,
        },
        "canonical_cover_control": {
            "block_indices": list(canonical_cover),
            "point_blocks": [list(blocks[index]) for index in canonical_cover],
            "section_rank": canonical_rank,
            "required_section_rank": 2,
        },
        "checks": checks,
        "limits": [
            "the scan covers only degree-nine blocks that are unions of three translated R38 local triple fibers",
            "other products, sums, and arbitrary primitive degree-nine sections remain open",
            "the result is one finite toy and supplies no asymptotic lifting theorem",
            "no target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
