#!/usr/bin/env python3
"""Sweep exact one-row-dropped collision slices for all catalog denominators."""

import collections
import itertools
import json

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_fixed_denominator_collision_beam_r47 as r47
import p1553_primitive_degree_nine_rank_search_r44 as r44


PRIME = r44.r38.PRIME
SUBGROUP_ORDER = r44.r38.SUBGROUP_ORDER
STEP = 42
SLICE_COUNT = 7
SLICE_SIZE = PRIME + 1


def load_catalog():
    with open(
        "p1553_rank_minor_guided_partition_search_report_r45.json",
        encoding="utf-8",
    ) as input_file:
        report = json.load(input_file)
    blocks = sorted(
        {
            tuple(block)
            for partition in report["rank_eight_partition_catalog"]
            for block in partition
        }
    )
    sections = []
    section_blocks = []
    seen = set()
    for block in blocks:
        section = r44.block_section(block, 0, STEP)[0]
        if section in seen:
            raise AssertionError("catalog divisor sections did not stay unique")
        seen.add(section)
        sections.append(section)
        section_blocks.append(block)
    ordered = sorted(zip(sections, section_blocks))
    return (
        [entry[0] for entry in ordered],
        [entry[1] for entry in ordered],
    )


def select_seeds(sections, blocks, values, selected_indices, inverses):
    masks = [r46.block_mask(block) for block in blocks]
    best = [None] * len(sections)
    disjoint_pairs = 0
    for first, second in itertools.combinations(range(len(sections)), 2):
        if masks[first] & masks[second]:
            continue
        disjoint_pairs += 1
        labels = r47.labels_from_values(values[first], values[second], inverses)
        if labels is None:
            raise AssertionError("disjoint catalog seed has subgroup base point")
        score, record = r47.score_labels(labels, selected_indices)
        for denominator, numerator in ((first, second), (second, first)):
            candidate = (score, -numerator, numerator, record)
            if best[denominator] is None or candidate[:2] > best[denominator][:2]:
                best[denominator] = candidate
    if disjoint_pairs != 142286 or any(seed is None for seed in best):
        raise AssertionError("catalog seed selection did not cover every denominator")
    return best, disjoint_pairs


def sweep_denominator(
    denominator_index,
    seed,
    sections,
    blocks,
    values,
    selected_indices,
    inverses,
):
    seed_score, _, numerator_index, seed_score_record = seed
    denominator = sections[denominator_index]
    numerator = sections[numerator_index]
    denominator_values = values[denominator_index]
    seed_labels = r47.labels_from_values(
        denominator_values, values[numerator_index], inverses
    )
    basis = r47.collision_basis(
        denominator_values, seed_labels, selected_indices
    )
    if len(basis) != 7:
        raise AssertionError("catalog seed collision basis is not rank seven")

    seed_key = r46.plucker_key(denominator, numerator)
    seen = {seed_key}
    generated = 0
    basepoint_rejections = 0
    score_histogram = collections.Counter()
    best_score = seed_score
    best_numerator = numerator
    best_record = seed_score_record
    survivors = []

    for dropped_row in range(7):
        six_rows = basis[:dropped_row] + basis[dropped_row + 1 :]
        first_direction, second_direction = r47.quotient_slice_basis(
            denominator, six_rows
        )
        first_values = r47.section_values(first_direction)
        second_values = r47.section_values(second_direction)
        for parameter in list(range(PRIME)) + [None]:
            generated += 1
            if parameter is None:
                candidate_numerator = second_direction
                candidate_values = second_values
            else:
                candidate_numerator = tuple(
                    (
                        first_direction[column]
                        + parameter * second_direction[column]
                    )
                    % PRIME
                    for column in range(9)
                )
                candidate_values = tuple(
                    (
                        first_values[index]
                        + parameter * second_values[index]
                    )
                    % PRIME
                    for index in range(SUBGROUP_ORDER)
                )
            key = r46.plucker_key(denominator, candidate_numerator)
            if key in seen:
                continue
            seen.add(key)
            labels = r47.labels_from_values(
                denominator_values, candidate_values, inverses
            )
            if labels is None:
                basepoint_rejections += 1
                continue
            score, score_record = r47.score_labels(labels, selected_indices)
            score_histogram[score] += 1
            if score > best_score or (
                score == best_score
                and r47.normalize(candidate_numerator)
                < r47.normalize(best_numerator)
            ):
                best_score = score
                best_numerator = r47.normalize(candidate_numerator)
                best_record = score_record
            if score[0] >= 3:
                survivor = score_record.copy()
                survivor["numerator_section"] = list(
                    r47.normalize(candidate_numerator)
                )
                survivors.append(survivor)

    return {
        "denominator_index": denominator_index,
        "denominator_block": list(blocks[denominator_index]),
        "seed_numerator_index": numerator_index,
        "seed_numerator_block": list(blocks[numerator_index]),
        "seed_score": list(seed_score),
        "collision_basis_rank": len(basis),
        "generated_projective_line_instances": generated,
        "unique_pencil_keys_including_seed": len(seen),
        "subgroup_basepoint_rejections": basepoint_rejections,
        "scored_candidates": sum(score_histogram.values()),
        "best_score": list(best_score),
        "best_numerator_section": list(r47.normalize(best_numerator)),
        "best_selected_multiplicity_histogram": best_record[
            "selected_multiplicity_histogram"
        ],
        "survivors": survivors,
        "score_histogram": {
            f"{score[0]}_{score[1]}_{score[2]}": score_histogram[score]
            for score in sorted(score_histogram)
        },
    }


def run():
    sections, blocks = load_catalog()
    if len(sections) != 807:
        raise AssertionError("unexpected catalog section count")
    values = [r47.section_values(section) for section in sections]
    selected_indices = r47.selected_scalars()
    inverses = [0] + [pow(value, -1, PRIME) for value in range(1, PRIME)]
    seeds, disjoint_pairs = select_seeds(
        sections, blocks, values, selected_indices, inverses
    )

    denominator_reports = [
        sweep_denominator(
            index,
            seeds[index],
            sections,
            blocks,
            values,
            selected_indices,
            inverses,
        )
        for index in range(len(sections))
    ]
    survivors = [
        {
            "denominator_index": report["denominator_index"],
            "denominator_block": report["denominator_block"],
            **survivor,
        }
        for report in denominator_reports
        for survivor in report["survivors"]
    ]
    global_best = max(
        denominator_reports,
        key=lambda report: (
            tuple(report["best_score"]),
            tuple(-value for value in report["denominator_block"]),
        ),
    )
    best_score_histogram = collections.Counter(
        tuple(report["best_score"]) for report in denominator_reports
    )
    seed_score_histogram = collections.Counter(
        tuple(report["seed_score"]) for report in denominator_reports
    )
    total_generated = sum(
        report["generated_projective_line_instances"]
        for report in denominator_reports
    )
    total_scored = sum(
        report["scored_candidates"] for report in denominator_reports
    )
    total_basepoint_rejections = sum(
        report["subgroup_basepoint_rejections"]
        for report in denominator_reports
    )
    checks = {
        "all_807_denominators_checked": len(denominator_reports) == 807,
        "all_seed_collision_bases_rank_seven": all(
            report["collision_basis_rank"] == 7
            for report in denominator_reports
        ),
        "all_projective_slices_generated": total_generated
        == 807 * SLICE_COUNT * SLICE_SIZE,
        "all_candidate_outcomes_accounted": all(
            report["unique_pencil_keys_including_seed"]
            == report["scored_candidates"]
            + report["subgroup_basepoint_rejections"]
            + 1
            for report in denominator_reports
        ),
        "no_three_fiber_survivor": not survivors,
    }
    if not all(checks.values()):
        raise AssertionError("all-denominator collision slice sweep failed")

    return {
        "schema": "p1553.all_denominator_collision_slice_sweep.r48.v1",
        "classification": [
            "toy",
            "exact",
            "exhaustive-declared-slices",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": SUBGROUP_ORDER,
        "parameters": {
            "catalog_denominators": len(sections),
            "catalog_disjoint_pairs_scored_for_seeds": disjoint_pairs,
            "dropped_collision_rows_per_denominator": SLICE_COUNT,
            "pencils_per_projective_slice": SLICE_SIZE,
            "generated_instances_per_denominator": SLICE_COUNT * SLICE_SIZE,
        },
        "totals": {
            "generated_projective_line_instances": total_generated,
            "scored_subgroup_basepoint_free_candidates": total_scored,
            "subgroup_basepoint_rejections": total_basepoint_rejections,
            "seed_score_histogram": {
                f"{score[0]}_{score[1]}_{score[2]}": seed_score_histogram[score]
                for score in sorted(seed_score_histogram)
            },
            "best_score_per_denominator_histogram": {
                f"{score[0]}_{score[1]}_{score[2]}": best_score_histogram[score]
                for score in sorted(best_score_histogram)
            },
        },
        "global_best": global_best,
        "required_score": {
            "complete_nine_point_fibers": 9,
            "minimum_collision_pairs": 324,
            "complete_coverage": 81,
        },
        "denominators": denominator_reports,
        "survivors": survivors,
        "checks": checks,
        "limits": [
            "the sweep is exhaustive only for seven one-row-dropped slices around one catalog seed per denominator",
            "numerator pencils outside those slices remain open",
            "subgroup basepoint checks do not certify geometric basepoint freeness over the algebraic closure",
            "no asymptotic family, target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
