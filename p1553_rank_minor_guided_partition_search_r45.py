#!/usr/bin/env python3
"""Search equal-sum degree-nine partitions using exact determinantal scores."""

import collections
import functools
import json

import p1553_primitive_degree_nine_rank_search_r44 as r44


PRIME = r44.r38.PRIME
SEARCH_STEP = 42
NEGATED_STEP = r44.r38.SUBGROUP_ORDER - SEARCH_STEP
BEAM_WIDTH = 64
BEAM_LAYERS = 8
RANK_EIGHT_CLOSURE_BUDGET_PER_LAYER = 32


def canonical_partition(blocks):
    return tuple(sorted(tuple(sorted(block)) for block in blocks))


def initial_column_partition():
    square = r44.magic_square()
    return canonical_partition(
        [
            [square[row][column] for row in range(r44.MAGIC_ORDER)]
            for column in range(r44.MAGIC_ORDER)
        ]
    )


@functools.lru_cache(maxsize=None)
def pair_positions_by_sum(block):
    positions = collections.defaultdict(list)
    for left in range(r44.MAGIC_ORDER):
        for right in range(left + 1, r44.MAGIC_ORDER):
            positions[block[left] + block[right]].append((left, right))
    return tuple(
        (pair_sum, tuple(pairs))
        for pair_sum, pairs in sorted(positions.items())
    )


def pair_dictionary(block):
    return dict(pair_positions_by_sum(block))


@functools.lru_cache(maxsize=None)
def neighbors(partition):
    children = set()
    for first_block in range(r44.MAGIC_ORDER):
        first_pairs = pair_dictionary(partition[first_block])
        for second_block in range(first_block + 1, r44.MAGIC_ORDER):
            second_pairs = pair_dictionary(partition[second_block])
            for pair_sum in sorted(set(first_pairs).intersection(second_pairs)):
                for first_positions in first_pairs[pair_sum]:
                    for second_positions in second_pairs[pair_sum]:
                        blocks = [list(block) for block in partition]
                        first_values = [
                            blocks[first_block][position]
                            for position in first_positions
                        ]
                        second_values = [
                            blocks[second_block][position]
                            for position in second_positions
                        ]
                        for position, value in zip(
                            first_positions, second_values
                        ):
                            blocks[first_block][position] = value
                        for position, value in zip(
                            second_positions, first_values
                        ):
                            blocks[second_block][position] = value
                        children.add(canonical_partition(blocks))
    return tuple(sorted(children))


@functools.lru_cache(maxsize=None)
def section_vector(block):
    return r44.block_section(block, 0, SEARCH_STEP)[0]


def null_vector(matrix):
    reduced, pivots = r44.row_reduce([list(row) for row in matrix])
    free_columns = [column for column in range(9) if column not in pivots]
    if len(free_columns) != 1:
        raise AssertionError("null-vector helper requires corank one")
    vector = [0] * 9
    vector[free_columns[0]] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        vector[pivot] = -sum(
            reduced[row][column] * vector[column]
            for column in range(pivot + 1, 9)
        ) % PRIME
    first_nonzero = next(value for value in vector if value)
    inverse = pow(first_nonzero, -1, PRIME)
    return tuple(value * inverse % PRIME for value in vector)


def inverse_zero_count(matrix):
    size = len(matrix)
    augmented = [
        list(row) + [int(row_index == column) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = next(
            row
            for row in range(column, size)
            if augmented[row][column] % PRIME
        )
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )
        inverse = pow(augmented[column][column] % PRIME, -1, PRIME)
        augmented[column] = [
            value * inverse % PRIME for value in augmented[column]
        ]
        for row in range(size):
            if row == column or augmented[row][column] == 0:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                (
                    augmented[row][index]
                    - scale * augmented[column][index]
                )
                % PRIME
                for index in range(2 * size)
            ]
    return sum(
        value == 0 for row in augmented for value in row[size:]
    )


@functools.lru_cache(maxsize=None)
def diagnostics(partition):
    vectors = tuple(section_vector(block) for block in partition)
    rank = r44.section_rank(vectors)
    result = {
        "rank": rank,
        "vanishing_8_by_8_minors": 81,
    }
    if rank == 9:
        result["vanishing_8_by_8_minors"] = inverse_zero_count(vectors)
    elif rank == 8:
        right_null = null_vector(vectors)
        left_null = null_vector(tuple(zip(*vectors)))
        left_support = sum(value != 0 for value in left_null)
        right_support = sum(value != 0 for value in right_null)
        result.update(
            {
                "left_null_support": left_support,
                "right_null_support": right_support,
                "vanishing_8_by_8_minors": 81
                - left_support * right_support,
            }
        )
    return result


def rank_eight_closure(seeds, known_rank_eight, state_budget=None):
    queue = collections.deque(
        seed for seed in sorted(seeds) if seed not in known_rank_eight
    )
    added = set()
    boundary = set()
    edge_rank_histogram = collections.Counter()
    while queue and (state_budget is None or len(added) < state_budget):
        partition = queue.popleft()
        if partition in known_rank_eight or partition in added:
            continue
        rank = diagnostics(partition)["rank"]
        if rank < 8:
            raise AssertionError("rank-below-eight survivor reached")
        if rank != 8:
            continue
        added.add(partition)
        for child in neighbors(partition):
            child_rank = diagnostics(child)["rank"]
            edge_rank_histogram[child_rank] += 1
            if child_rank < 8:
                raise AssertionError("rank-below-eight survivor reached")
            if child_rank == 8:
                if child not in known_rank_eight and child not in added:
                    queue.append(child)
            else:
                boundary.add(child)
    return {
        "states": added,
        "rank_nine_boundary": boundary,
        "edge_rank_histogram": edge_rank_histogram,
        "deferred_queue_states": len(queue),
        "truncated": bool(queue),
    }


def histogram(values):
    counts = collections.Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def partition_record(partition):
    record = diagnostics(partition).copy()
    record["blocks"] = [list(block) for block in partition]
    return record


def run():
    root = initial_column_partition()
    if {sum(block) for block in root} != {r44.MAGIC_SUM}:
        raise AssertionError("initial blocks do not have equal sums")

    start_cancellation_checks = []
    for step in (SEARCH_STEP, NEGATED_STEP):
        reference_rank, reference_vectors, reference_translation = (
            r44.partition_rank(root, 0, step)
        )
        for start in range(r44.r38.SUBGROUP_ORDER):
            rank, vectors, translation = r44.partition_rank(root, start, step)
            start_cancellation_checks.append(
                rank == reference_rank
                and vectors == reference_vectors
                and translation
                == (
                    reference_translation - start
                )
                % r44.r38.SUBGROUP_ORDER
            )

    known_rank_eight = set()
    initial = rank_eight_closure({root}, known_rank_eight)
    known_rank_eight.update(initial["states"])
    rank_nine_pool = set(initial["rank_nine_boundary"])
    expanded_rank_nine = set()
    layers = []
    minimum_rank = 8

    for layer_index in range(BEAM_LAYERS):
        candidates = sorted(
            rank_nine_pool - expanded_rank_nine,
            key=lambda partition: (
                -diagnostics(partition)["vanishing_8_by_8_minors"],
                partition,
            ),
        )
        beam = candidates[:BEAM_WIDTH]
        if not beam:
            break
        expanded_rank_nine.update(beam)

        child_set = set()
        for partition in beam:
            child_set.update(neighbors(partition))
        child_rank_histogram = collections.Counter(
            diagnostics(child)["rank"] for child in child_set
        )
        minimum_rank = min(minimum_rank, min(child_rank_histogram))
        if minimum_rank < 8:
            raise AssertionError("rank-below-eight survivor reached")

        new_rank_eight_seeds = {
            child
            for child in child_set
            if diagnostics(child)["rank"] == 8
            and child not in known_rank_eight
        }
        closure = rank_eight_closure(
            new_rank_eight_seeds,
            known_rank_eight,
            RANK_EIGHT_CLOSURE_BUDGET_PER_LAYER,
        )
        known_rank_eight.update(closure["states"])

        rank_nine_pool.update(
            child
            for child in child_set
            if diagnostics(child)["rank"] == 9
        )
        rank_nine_pool.update(closure["rank_nine_boundary"])

        layers.append(
            {
                "layer": layer_index + 1,
                "beam_size": len(beam),
                "beam_vanishing_minor_histogram": histogram(
                    diagnostics(partition)["vanishing_8_by_8_minors"]
                    for partition in beam
                ),
                "unique_children": len(child_set),
                "child_rank_histogram": {
                    str(rank): child_rank_histogram[rank]
                    for rank in sorted(child_rank_histogram)
                },
                "new_rank_eight_seeds": len(new_rank_eight_seeds),
                "rank_eight_closure_added": len(closure["states"]),
                "rank_eight_closure_budget": RANK_EIGHT_CLOSURE_BUDGET_PER_LAYER,
                "rank_eight_closure_deferred_queue_states": closure[
                    "deferred_queue_states"
                ],
                "rank_eight_closure_truncated": closure["truncated"],
                "rank_eight_closure_edge_rank_histogram": {
                    str(rank): closure["edge_rank_histogram"][rank]
                    for rank in sorted(closure["edge_rank_histogram"])
                },
                "rank_nine_pool": len(rank_nine_pool),
            }
        )

    rank_eight_scores = collections.Counter(
        diagnostics(partition)["vanishing_8_by_8_minors"]
        for partition in known_rank_eight
    )
    best_rank_eight_score = max(rank_eight_scores)
    best_rank_eight = min(
        partition
        for partition in known_rank_eight
        if diagnostics(partition)["vanishing_8_by_8_minors"]
        == best_rank_eight_score
    )
    expanded_rank_nine_scores = collections.Counter(
        diagnostics(partition)["vanishing_8_by_8_minors"]
        for partition in expanded_rank_nine
    )

    negation_checks = []
    for partition in sorted(known_rank_eight) + sorted(expanded_rank_nine):
        rank, vectors, _ = r44.partition_rank(partition, 0, NEGATED_STEP)
        negation_checks.append(
            rank == diagnostics(partition)["rank"]
            and (
                inverse_zero_count(vectors)
                if rank == 9
                else 81
                - sum(value != 0 for value in null_vector(vectors))
                * sum(
                    value != 0
                    for value in null_vector(tuple(zip(*vectors)))
                )
            )
            == diagnostics(partition)["vanishing_8_by_8_minors"]
        )

    checks = {
        "root_rank_is_eight": diagnostics(root)["rank"] == 8,
        "all_206_start_cancellation_checks": len(start_cancellation_checks)
        == 206
        and all(start_cancellation_checks),
        "initial_rank_eight_component_has_25_states": len(initial["states"])
        == 25,
        "initial_boundary_has_6647_rank_nine_states": len(
            initial["rank_nine_boundary"]
        )
        == 6647,
        "beam_layers_completed": len(layers) == BEAM_LAYERS,
        "minimum_rank_is_eight": minimum_rank == 8,
        "no_rank_two_survivor": minimum_rank > 2,
        "step_negation_preserves_all_checked_scores": all(negation_checks),
    }
    if not all(checks.values()):
        raise AssertionError("rank-minor-guided search check failed")

    return {
        "schema": "p1553.rank_minor_guided_partition_search.r45.v1",
        "classification": [
            "toy",
            "exact",
            "finite-search",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": r44.r38.SUBGROUP_ORDER,
        "search": {
            "step_representative": SEARCH_STEP,
            "negated_step_representative": NEGATED_STEP,
            "beam_width": BEAM_WIDTH,
            "beam_layers": BEAM_LAYERS,
            "rank_eight_closure_budget_per_layer": RANK_EIGHT_CLOSURE_BUDGET_PER_LAYER,
            "score": "section rank, then number of vanishing 8-by-8 minors",
        },
        "initial_rank_eight_component": {
            "states": len(initial["states"]),
            "rank_nine_boundary": len(initial["rank_nine_boundary"]),
            "edge_rank_histogram": {
                str(rank): initial["edge_rank_histogram"][rank]
                for rank in sorted(initial["edge_rank_histogram"])
            },
            "truncated": initial["truncated"],
        },
        "layers": layers,
        "totals": {
            "rank_eight_states": len(known_rank_eight),
            "rank_eight_vanishing_minor_histogram": {
                str(score): rank_eight_scores[score]
                for score in sorted(rank_eight_scores)
            },
            "expanded_rank_nine_states": len(expanded_rank_nine),
            "expanded_rank_nine_vanishing_minor_histogram": {
                str(score): expanded_rank_nine_scores[score]
                for score in sorted(expanded_rank_nine_scores)
            },
            "rank_nine_pool": len(rank_nine_pool),
            "minimum_rank": minimum_rank,
        },
        "rank_eight_partition_catalog": [
            [list(block) for block in partition]
            for partition in sorted(known_rank_eight)
        ],
        "best_rank_eight_witness": partition_record(best_rank_eight),
        "checks": checks,
        "limits": [
            "only the initial 25-state rank-eight component is exhaustively closed under equal-pair-sum trades",
            "later rank-eight closures use a declared 32-state budget per beam layer",
            "the rank-nine bridge search retains only a deterministic finite beam",
            "cofactor sparsity is a heuristic score and not a rank-descent theorem",
            "partitions outside the reached trade graph and primitive degree-nine pencils remain open",
            "no target locator, R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
