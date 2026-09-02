#!/usr/bin/env python3
"""Sweep collision slices for deterministic equal-sum blocks outside R45."""

import collections
import random

import p1553_all_denominator_collision_slice_sweep_r48 as r48
import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_fixed_denominator_collision_beam_r47 as r47
import p1553_primitive_degree_nine_rank_search_r44 as r44


PRIME = r44.r38.PRIME
SUBGROUP_ORDER = r44.r38.SUBGROUP_ORDER
STEP = 42
SAMPLE_SEED = 1553
EXTERNAL_BLOCK_COUNT = 1000
SLICE_COUNT = 7
SLICE_SIZE = PRIME + 1


def sample_external_blocks(catalog_blocks):
    source = random.Random(SAMPLE_SEED)
    target_sum = r44.MAGIC_SUM % SUBGROUP_ORDER
    blocks = set()
    attempts = 0
    congruent_candidates = 0
    catalog_rejections = 0
    duplicate_rejections = 0
    while len(blocks) < EXTERNAL_BLOCK_COUNT:
        attempts += 1
        first_eight = sorted(source.sample(range(81), 8))
        ninth = (target_sum - sum(first_eight)) % SUBGROUP_ORDER
        if ninth >= 81 or ninth in first_eight:
            continue
        congruent_candidates += 1
        block = tuple(sorted(first_eight + [ninth]))
        if block in catalog_blocks:
            catalog_rejections += 1
            continue
        if block in blocks:
            duplicate_rejections += 1
            continue
        blocks.add(block)
    return sorted(blocks), {
        "attempts": attempts,
        "congruent_candidates": congruent_candidates,
        "catalog_rejections": catalog_rejections,
        "duplicate_rejections": duplicate_rejections,
    }


def choose_catalog_seed(
    denominator_values,
    denominator_mask,
    catalog_values,
    catalog_masks,
    selected_indices,
    inverses,
):
    best = None
    disjoint_partners = 0
    for index, numerator_values in enumerate(catalog_values):
        if denominator_mask & catalog_masks[index]:
            continue
        disjoint_partners += 1
        labels = r47.labels_from_values(
            denominator_values, numerator_values, inverses
        )
        if labels is None:
            raise AssertionError("disjoint external seed has subgroup base point")
        score, score_record = r47.score_labels(labels, selected_indices)
        candidate = (score, -index, index, score_record, labels)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise AssertionError("external denominator has no disjoint catalog seed")
    return best, disjoint_partners


def sweep_external_denominator(
    denominator_index,
    denominator_block,
    denominator,
    denominator_values,
    seed,
    catalog_blocks,
    catalog_sections,
    selected_indices,
    inverses,
):
    seed_score, _, numerator_index, seed_record, seed_labels = seed
    numerator = catalog_sections[numerator_index]
    basis = r47.collision_basis(
        denominator_values, seed_labels, selected_indices
    )
    if len(basis) != 7:
        return {
            "denominator_index": denominator_index,
            "denominator_block": list(denominator_block),
            "seed_numerator_index": numerator_index,
            "seed_numerator_block": list(catalog_blocks[numerator_index]),
            "seed_score": list(seed_score),
            "collision_basis_rank": len(basis),
            "underdetermined_seed": True,
        }

    seen = {r46.plucker_key(denominator, numerator)}
    generated = 0
    basepoint_rejections = 0
    scored = 0
    best_score = seed_score
    best_numerator = numerator
    best_score_record = seed_record
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
            scored += 1
            score, score_record = r47.score_labels(labels, selected_indices)
            if score > best_score or (
                score == best_score
                and r47.normalize(candidate_numerator)
                < r47.normalize(best_numerator)
            ):
                best_score = score
                best_numerator = r47.normalize(candidate_numerator)
                best_score_record = score_record
            if score[0] >= 3:
                survivor = score_record.copy()
                survivor["numerator_section"] = list(
                    r47.normalize(candidate_numerator)
                )
                survivors.append(survivor)

    return {
        "denominator_index": denominator_index,
        "denominator_block": list(denominator_block),
        "seed_numerator_index": numerator_index,
        "seed_numerator_block": list(catalog_blocks[numerator_index]),
        "seed_score": list(seed_score),
        "collision_basis_rank": len(basis),
        "underdetermined_seed": False,
        "generated_projective_line_instances": generated,
        "unique_pencil_keys_including_seed": len(seen),
        "subgroup_basepoint_rejections": basepoint_rejections,
        "scored_candidates": scored,
        "best_score": list(best_score),
        "best_numerator_section": list(r47.normalize(best_numerator)),
        "best_selected_multiplicity_histogram": best_score_record[
            "selected_multiplicity_histogram"
        ],
        "survivors": survivors,
    }


def run():
    catalog_sections, catalog_blocks = r48.load_catalog()
    catalog_block_set = set(catalog_blocks)
    external_blocks, sampling = sample_external_blocks(catalog_block_set)
    external_sections = [
        r44.block_section(block, 0, STEP)[0] for block in external_blocks
    ]
    if len(set(external_sections)) != EXTERNAL_BLOCK_COUNT:
        raise AssertionError("external block sections did not stay unique")

    selected_indices = r47.selected_scalars()
    inverses = [0] + [pow(value, -1, PRIME) for value in range(1, PRIME)]
    catalog_values = [
        r47.section_values(section) for section in catalog_sections
    ]
    catalog_masks = [r46.block_mask(block) for block in catalog_blocks]
    external_values = [
        r47.section_values(section) for section in external_sections
    ]

    zero_set_checks = []
    denominator_reports = []
    disjoint_partner_histogram = collections.Counter()
    for index, (block, section, values) in enumerate(
        zip(external_blocks, external_sections, external_values)
    ):
        selected_zeros = {
            offset
            for offset, scalar in enumerate(selected_indices)
            if values[scalar] == 0
        }
        zero_set_checks.append(
            selected_zeros == set(block)
            and sum(value == 0 for value in values) == 9
        )
        seed, disjoint_partners = choose_catalog_seed(
            values,
            r46.block_mask(block),
            catalog_values,
            catalog_masks,
            selected_indices,
            inverses,
        )
        disjoint_partner_histogram[disjoint_partners] += 1
        denominator_reports.append(
            sweep_external_denominator(
                index,
                block,
                section,
                values,
                seed,
                catalog_blocks,
                catalog_sections,
                selected_indices,
                inverses,
            )
        )

    underdetermined = [
        report
        for report in denominator_reports
        if report["underdetermined_seed"]
    ]
    completed = [
        report
        for report in denominator_reports
        if not report["underdetermined_seed"]
    ]
    survivors = [
        {
            "denominator_index": report["denominator_index"],
            "denominator_block": report["denominator_block"],
            **survivor,
        }
        for report in completed
        for survivor in report["survivors"]
    ]
    global_best = max(
        completed,
        key=lambda report: (
            tuple(report["best_score"]),
            tuple(-value for value in report["denominator_block"]),
        ),
    )
    seed_score_histogram = collections.Counter(
        tuple(report["seed_score"]) for report in denominator_reports
    )
    best_score_histogram = collections.Counter(
        tuple(report["best_score"]) for report in completed
    )
    total_generated = sum(
        report["generated_projective_line_instances"] for report in completed
    )
    total_scored = sum(report["scored_candidates"] for report in completed)
    total_basepoint_rejections = sum(
        report["subgroup_basepoint_rejections"] for report in completed
    )
    checks = {
        "sampled_1000_external_blocks": len(external_blocks)
        == EXTERNAL_BLOCK_COUNT,
        "all_external_blocks_outside_catalog": not catalog_block_set.intersection(
            external_blocks
        ),
        "all_external_section_zero_sets_match_blocks": all(zero_set_checks),
        "all_seed_collision_bases_rank_seven": not underdetermined,
        "all_projective_slices_generated": total_generated
        == EXTERNAL_BLOCK_COUNT * SLICE_COUNT * SLICE_SIZE,
        "all_candidate_outcomes_accounted": all(
            report["unique_pencil_keys_including_seed"]
            == report["scored_candidates"]
            + report["subgroup_basepoint_rejections"]
            + 1
            for report in completed
        ),
        "no_three_fiber_survivor": not survivors,
    }
    if not all(checks.values()):
        raise AssertionError("external-denominator collision slice sweep failed")

    return {
        "schema": "p1553.external_denominator_collision_slice_sweep.r49.v1",
        "classification": [
            "toy",
            "exact",
            "deterministic-finite-sample",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": SUBGROUP_ORDER,
        "sampling": {
            "seed": SAMPLE_SEED,
            "accepted_external_blocks": EXTERNAL_BLOCK_COUNT,
            "target_offset_sum_mod_103": r44.MAGIC_SUM % SUBGROUP_ORDER,
            **sampling,
        },
        "parameters": {
            "catalog_seed_sections": len(catalog_sections),
            "dropped_collision_rows_per_denominator": SLICE_COUNT,
            "pencils_per_projective_slice": SLICE_SIZE,
            "generated_instances_per_denominator": SLICE_COUNT * SLICE_SIZE,
        },
        "totals": {
            "completed_denominators": len(completed),
            "underdetermined_seed_denominators": len(underdetermined),
            "generated_projective_line_instances": total_generated,
            "scored_subgroup_basepoint_free_candidates": total_scored,
            "subgroup_basepoint_rejections": total_basepoint_rejections,
            "disjoint_catalog_partner_count_histogram": {
                str(count): disjoint_partner_histogram[count]
                for count in sorted(disjoint_partner_histogram)
            },
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
            "the 1000 external blocks are a deterministic finite sample, not an exhaustive block catalog",
            "each denominator is searched only through seven one-row-dropped slices around one catalog seed",
            "subgroup basepoint checks do not certify geometric basepoint freeness over the algebraic closure",
            "no asymptotic family, target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
