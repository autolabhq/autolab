#!/usr/bin/env python3
"""Search fixed-denominator pencils through exact collision-hyperplane slices."""

import collections
import json

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_primitive_degree_nine_rank_search_r44 as r44


PRIME = r44.r38.PRIME
SUBGROUP_ORDER = r44.r38.SUBGROUP_ORDER
STEP = 42
BEAM_WIDTH = 32
BEAM_LAYERS = 8
PROJECTIVE_LINE_SIZE = PRIME + 1


def normalize(vector):
    first_nonzero = next(value for value in vector if value)
    inverse = pow(first_nonzero, -1, PRIME)
    return tuple(value * inverse % PRIME for value in vector)


def vector_rank(vectors):
    if not vectors:
        return 0
    return r44.section_rank(tuple(tuple(vector) for vector in vectors))


def nullspace(matrix):
    reduced, pivots = r44.row_reduce([list(row) for row in matrix])
    free_columns = [column for column in range(9) if column not in pivots]
    basis = []
    for free_column in free_columns:
        vector = [0] * 9
        vector[free_column] = 1
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -sum(
                reduced[row][column] * vector[column]
                for column in range(pivot + 1, 9)
            ) % PRIME
        basis.append(normalize(vector))
    return tuple(basis)


def collision_row(denominator_values, first_index, second_index):
    first_scale = denominator_values[first_index]
    second_scale = denominator_values[second_index]
    first_embedding = r44.SECTION_EMBEDDINGS[first_index]
    second_embedding = r44.SECTION_EMBEDDINGS[second_index]
    row = tuple(
        (
            first_scale * second_embedding[column]
            - second_scale * first_embedding[column]
        )
        % PRIME
        for column in range(9)
    )
    if not any(row):
        return None
    return normalize(row)


def selected_scalars():
    return tuple(
        STEP * (offset - 40) % SUBGROUP_ORDER for offset in range(81)
    )


def labels_from_values(denominator_values, numerator_values, inverses):
    labels = []
    for denominator, numerator in zip(denominator_values, numerator_values):
        if denominator == 0:
            if numerator == 0:
                return None
            labels.append(PRIME)
        else:
            labels.append(numerator * inverses[denominator] % PRIME)
    return tuple(labels)


def score_labels(labels, selected_indices):
    subgroup_counts = collections.Counter(labels)
    if max(subgroup_counts.values()) > 9:
        raise AssertionError("degree-nine candidate has oversized subgroup fiber")
    selected_counts = collections.Counter(labels[index] for index in selected_indices)
    complete_fibers = sum(count == 9 for count in selected_counts.values())
    collision_pairs = sum(count * (count - 1) // 2 for count in selected_counts.values())
    maximum_multiplicity = max(selected_counts.values())
    histogram = collections.Counter(selected_counts.values())
    return (
        complete_fibers,
        collision_pairs,
        maximum_multiplicity,
    ), {
        "complete_nine_point_fibers": complete_fibers,
        "complete_coverage": complete_fibers * 9,
        "collision_pairs": collision_pairs,
        "maximum_selected_multiplicity": maximum_multiplicity,
        "selected_multiplicity_histogram": {
            str(multiplicity): histogram[multiplicity]
            for multiplicity in sorted(histogram)
        },
    }


def collision_basis(denominator_values, labels, selected_indices):
    rows = []
    rank = 0
    for left_offset in range(len(selected_indices)):
        left_index = selected_indices[left_offset]
        for right_offset in range(left_offset + 1, len(selected_indices)):
            right_index = selected_indices[right_offset]
            if labels[left_index] != labels[right_index]:
                continue
            row = collision_row(
                denominator_values, left_index, right_index
            )
            if row is None:
                continue
            candidate_rank = vector_rank(rows + [row])
            if candidate_rank > rank:
                rows.append(row)
                rank = candidate_rank
                if rank == 7:
                    return tuple(rows)
    return tuple(rows)


def quotient_slice_basis(denominator, six_rows):
    kernel = nullspace(six_rows)
    if len(kernel) != 3:
        raise AssertionError("six collision equations do not have corank three")
    basis = [denominator]
    for vector in kernel:
        if vector_rank(basis + [vector]) > len(basis):
            basis.append(vector)
        if len(basis) == 3:
            break
    if len(basis) != 3:
        raise AssertionError("failed to split denominator from quotient slice")
    return basis[1], basis[2]


def section_values(section):
    return tuple(
        r46.dot(section, embedding) for embedding in r44.SECTION_EMBEDDINGS
    )


def candidate_record(numerator, score_record):
    record = score_record.copy()
    record["numerator_section"] = list(normalize(numerator))
    return record


def search_denominator(denominator, seed_numerator, selected_indices, inverses):
    denominator = normalize(denominator)
    seed_numerator = normalize(seed_numerator)
    denominator_values = section_values(denominator)
    seed_values = section_values(seed_numerator)
    seed_labels = labels_from_values(denominator_values, seed_values, inverses)
    if seed_labels is None:
        raise AssertionError("seed pencil has a subgroup base point")
    seed_score, seed_record = score_labels(seed_labels, selected_indices)
    seed_basis = collision_basis(
        denominator_values, seed_labels, selected_indices
    )
    if len(seed_basis) != 7:
        raise AssertionError("seed collisions do not isolate one pencil")

    seed_key = r46.plucker_key(denominator, seed_numerator)
    beam = [
        {
            "key": seed_key,
            "numerator": seed_numerator,
            "labels": seed_labels,
            "basis": seed_basis,
            "score": seed_score,
            "record": candidate_record(seed_numerator, seed_record),
        }
    ]
    visited = {seed_key}
    best = beam[0]
    survivors = []
    layers = []

    for layer_index in range(BEAM_LAYERS):
        candidates = {}
        generated_instances = 0
        subgroup_basepoint_rejections = 0
        for state in beam:
            basis = state["basis"]
            for dropped_row in range(7):
                six_rows = basis[:dropped_row] + basis[dropped_row + 1 :]
                first_direction, second_direction = quotient_slice_basis(
                    denominator, six_rows
                )
                first_values = section_values(first_direction)
                second_values = section_values(second_direction)
                projective_parameters = list(range(PRIME)) + [None]
                for parameter in projective_parameters:
                    generated_instances += 1
                    if parameter is None:
                        numerator = second_direction
                        numerator_values = second_values
                    else:
                        numerator = tuple(
                            (
                                first_direction[column]
                                + parameter * second_direction[column]
                            )
                            % PRIME
                            for column in range(9)
                        )
                        numerator_values = tuple(
                            (
                                first_values[index]
                                + parameter * second_values[index]
                            )
                            % PRIME
                            for index in range(SUBGROUP_ORDER)
                        )
                    key = r46.plucker_key(denominator, numerator)
                    if key in visited or key in candidates:
                        continue
                    labels = labels_from_values(
                        denominator_values, numerator_values, inverses
                    )
                    if labels is None:
                        subgroup_basepoint_rejections += 1
                        continue
                    score, score_record = score_labels(labels, selected_indices)
                    candidates[key] = {
                        "key": key,
                        "numerator": normalize(numerator),
                        "labels": labels,
                        "score": score,
                        "record": candidate_record(numerator, score_record),
                    }

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (candidate["score"], candidate["key"]),
            reverse=True,
        )
        selected = ordered[:BEAM_WIDTH]
        next_beam = []
        basis_rank_histogram = collections.Counter()
        for candidate in selected:
            basis = collision_basis(
                denominator_values, candidate["labels"], selected_indices
            )
            basis_rank_histogram[len(basis)] += 1
            if len(basis) != 7:
                continue
            candidate["basis"] = basis
            next_beam.append(candidate)
            visited.add(candidate["key"])
            if candidate["score"] > best["score"] or (
                candidate["score"] == best["score"]
                and candidate["key"] < best["key"]
            ):
                best = candidate
            if candidate["score"][0] == 9:
                survivors.append(candidate["record"])

        layers.append(
            {
                "layer": layer_index + 1,
                "input_beam": len(beam),
                "generated_projective_line_instances": generated_instances,
                "unique_unvisited_candidates": len(candidates),
                "subgroup_basepoint_rejections": subgroup_basepoint_rejections,
                "selected_candidate_collision_basis_rank_histogram": {
                    str(rank): basis_rank_histogram[rank]
                    for rank in sorted(basis_rank_histogram)
                },
                "output_beam": len(next_beam),
                "best_score_in_selected": list(selected[0]["score"])
                if selected
                else None,
            }
        )
        beam = next_beam
        if not beam:
            break

    return {
        "denominator_section": list(denominator),
        "seed_numerator_section": list(seed_numerator),
        "seed_score": list(seed_score),
        "layers": layers,
        "visited_isolated_pencils": len(visited),
        "best_score": list(best["score"]),
        "best_witness": best["record"],
        "survivors": survivors,
    }


def run():
    with open(
        "p1553_catalog_pencil_fiber_scan_report_r46.json", encoding="utf-8"
    ) as input_file:
        r46_report = json.load(input_file)
    best_blocks = r46_report["fiber_results"]["best_witness"]
    first_block = tuple(best_blocks["first_block"])
    second_block = tuple(best_blocks["second_block"])
    first_section = r44.block_section(first_block, 0, STEP)[0]
    second_section = r44.block_section(second_block, 0, STEP)[0]
    selected_indices = selected_scalars()
    inverses = [0] + [pow(value, -1, PRIME) for value in range(1, PRIME)]

    searches = [
        search_denominator(
            first_section, second_section, selected_indices, inverses
        ),
        search_denominator(
            second_section, first_section, selected_indices, inverses
        ),
    ]
    global_best = max(
        searches,
        key=lambda search: (
            tuple(search["best_score"]),
            tuple(search["denominator_section"]),
        ),
    )
    survivors = [
        survivor
        for search in searches
        for survivor in search["survivors"]
    ]
    checks = {
        "two_denominator_orientations_checked": len(searches) == 2,
        "all_beam_layers_completed": all(
            len(search["layers"]) == BEAM_LAYERS for search in searches
        ),
        "all_selected_collision_bases_have_rank_seven": all(
            layer["selected_candidate_collision_basis_rank_histogram"]
            == {"7": BEAM_WIDTH}
            for search in searches
            for layer in search["layers"]
        ),
        "no_nine_fiber_survivor": not survivors,
    }
    if not all(checks.values()):
        raise AssertionError("fixed-denominator collision beam check failed")

    return {
        "schema": "p1553.fixed_denominator_collision_beam.r47.v1",
        "classification": [
            "toy",
            "exact",
            "deterministic-finite-search",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": SUBGROUP_ORDER,
        "search_parameters": {
            "denominator_orientations": 2,
            "beam_width": BEAM_WIDTH,
            "beam_layers": BEAM_LAYERS,
            "dropped_collision_rows_per_state": 7,
            "pencils_per_projective_slice": PROJECTIVE_LINE_SIZE,
            "score": [
                "complete nine-point fibers",
                "selected collision pairs",
                "maximum selected multiplicity",
            ],
        },
        "searches": searches,
        "global_best": {
            "denominator_section": global_best["denominator_section"],
            "score": global_best["best_score"],
            "witness": global_best["best_witness"],
        },
        "required_score": {
            "complete_nine_point_fibers": 9,
            "minimum_collision_pairs": 324,
            "complete_coverage": 81,
        },
        "survivors": survivors,
        "checks": checks,
        "limits": [
            "only two fixed denominator sections from the strongest R46 pair are searched",
            "the beam retains 32 isolated pencils per layer and is not exhaustive in projective seven-space",
            "subgroup basepoint checks do not certify geometric basepoint freeness over the algebraic closure",
            "no asymptotic family, target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
