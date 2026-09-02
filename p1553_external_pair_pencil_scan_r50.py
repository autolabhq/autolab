#!/usr/bin/env python3
"""Exhaust direct pencils from pairs in the R49 external block sample."""

import collections
import itertools
import json

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_fixed_denominator_collision_beam_r47 as r47
import p1553_primitive_degree_nine_rank_search_r44 as r44


PRIME = r44.r38.PRIME
STEP = 42


def run():
    with open(
        "p1553_external_denominator_collision_slice_sweep_report_r49.json",
        encoding="utf-8",
    ) as input_file:
        r49_report = json.load(input_file)
    blocks = [
        tuple(report["denominator_block"])
        for report in r49_report["denominators"]
    ]
    if len(blocks) != 1000 or len(set(blocks)) != 1000:
        raise AssertionError("unexpected external block catalog")
    sections = [r44.block_section(block, 0, STEP)[0] for block in blocks]
    if len(set(sections)) != 1000:
        raise AssertionError("external section catalog is not injective")
    values = [r47.section_values(section) for section in sections]
    masks = [r46.block_mask(block) for block in blocks]
    selected_indices = r47.selected_scalars()
    inverses = [0] + [pow(value, -1, PRIME) for value in range(1, PRIME)]

    overlap_histogram = collections.Counter()
    pencil_keys = set()
    duplicate_pencil_pairs = 0
    complete_fiber_histogram = collections.Counter()
    collision_pair_histogram = collections.Counter()
    best = None
    survivors = []

    for first, second in itertools.combinations(range(len(sections)), 2):
        overlap = (masks[first] & masks[second]).bit_count()
        overlap_histogram[overlap] += 1
        if overlap:
            continue
        key = r46.plucker_key(sections[first], sections[second])
        if key in pencil_keys:
            duplicate_pencil_pairs += 1
            continue
        pencil_keys.add(key)
        labels = r47.labels_from_values(values[first], values[second], inverses)
        if labels is None:
            raise AssertionError("disjoint external pair has subgroup base point")
        score, score_record = r47.score_labels(labels, selected_indices)
        complete_fiber_histogram[score[0]] += 1
        collision_pair_histogram[score[1]] += 1
        record = score_record.copy()
        record["first_block"] = list(blocks[first])
        record["second_block"] = list(blocks[second])
        if best is None or score > best[0] or (
            score == best[0]
            and (blocks[first], blocks[second])
            < (tuple(best[1]["first_block"]), tuple(best[1]["second_block"]))
        ):
            best = (score, record)
        if score[0] >= 3:
            survivors.append(record)

    total_pairs = len(blocks) * (len(blocks) - 1) // 2
    disjoint_pairs = overlap_histogram[0]
    checks = {
        "all_external_pairs_accounted": sum(overlap_histogram.values())
        == total_pairs,
        "all_disjoint_pairs_accounted": len(pencil_keys)
        + duplicate_pencil_pairs
        == disjoint_pairs,
        "all_unique_pencils_scored": sum(complete_fiber_histogram.values())
        == len(pencil_keys),
        "no_three_fiber_survivor": not survivors,
    }
    if not all(checks.values()):
        raise AssertionError("external pair pencil scan failed")

    return {
        "schema": "p1553.external_pair_pencil_scan.r50.v1",
        "classification": [
            "toy",
            "exact",
            "exhaustive-finite-sample-pairs",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": r44.r38.SUBGROUP_ORDER,
        "sample": {
            "external_sections": len(sections),
            "sampling_seed": r49_report["sampling"]["seed"],
            "sampling_attempts": r49_report["sampling"]["attempts"],
        },
        "pair_scan": {
            "total_pairs": total_pairs,
            "overlap_histogram": {
                str(overlap): overlap_histogram[overlap]
                for overlap in sorted(overlap_histogram)
            },
            "disjoint_pairs": disjoint_pairs,
            "duplicate_pencil_pairs": duplicate_pencil_pairs,
            "unique_basepoint_free_pencils": len(pencil_keys),
        },
        "fiber_results": {
            "complete_nine_point_fiber_histogram": {
                str(count): complete_fiber_histogram[count]
                for count in sorted(complete_fiber_histogram)
            },
            "collision_pair_histogram": {
                str(count): collision_pair_histogram[count]
                for count in sorted(collision_pair_histogram)
            },
            "best_witness": best[1],
            "required_complete_fibers": 9,
            "required_collision_pairs": 324,
            "survivors": survivors,
        },
        "checks": checks,
        "limits": [
            "the pair scan is exhaustive only within the deterministic 1000-block R49 sample",
            "the combined 1807-section sample is tiny relative to all eligible blocks",
            "no asymptotic family, target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
