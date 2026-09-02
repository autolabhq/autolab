#!/usr/bin/env python3
"""Search structured primitive degree-nine pencil partitions by section rank."""

import collections
import json
import random

import p1553_low_boundary_pencil_search_r38 as r38


MAGIC_ORDER = 9
MAGIC_SUM = 360
TRADE_COUNT = 20000
TRADE_SEED = 1553
AFFINE_CONTEXTS = (
    (0, 1),
    (1, 1),
    (0, 42),
    (1, 42),
    (0, 43),
    (12, 43),
    (0, 102),
    (50, 17),
)


def magic_square():
    square = [[None] * MAGIC_ORDER for _ in range(MAGIC_ORDER)]
    row = 0
    column = MAGIC_ORDER // 2
    for value in range(MAGIC_ORDER**2):
        square[row][column] = value
        next_row = (row - 1) % MAGIC_ORDER
        next_column = (column + 1) % MAGIC_ORDER
        if square[next_row][next_column] is not None:
            row = (row + 1) % MAGIC_ORDER
        else:
            row, column = next_row, next_column
    return square


def section_embedding(scalar):
    point = r38.scalar_mul(scalar, r38.GENERATOR)
    if point is None:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1]
    x_coordinate, y_coordinate = point
    x_squared = x_coordinate * x_coordinate % r38.PRIME
    x_cubed = x_squared * x_coordinate % r38.PRIME
    x_fourth = x_cubed * x_coordinate % r38.PRIME
    return [
        1,
        x_coordinate,
        x_squared,
        x_cubed,
        x_fourth,
        y_coordinate,
        x_coordinate * y_coordinate % r38.PRIME,
        x_squared * y_coordinate % r38.PRIME,
        x_cubed * y_coordinate % r38.PRIME,
    ]


SECTION_EMBEDDINGS = [
    section_embedding(scalar) for scalar in range(r38.SUBGROUP_ORDER)
]


def row_reduce(matrix):
    reduced = [row[:] for row in matrix]
    pivots = []
    pivot_row = 0
    for column in range(len(reduced[0])):
        swap_row = next(
            (
                row
                for row in range(pivot_row, len(reduced))
                if reduced[row][column] % r38.PRIME
            ),
            None,
        )
        if swap_row is None:
            continue
        reduced[pivot_row], reduced[swap_row] = (
            reduced[swap_row],
            reduced[pivot_row],
        )
        inverse = pow(
            reduced[pivot_row][column] % r38.PRIME,
            -1,
            r38.PRIME,
        )
        reduced[pivot_row] = [
            value * inverse % r38.PRIME for value in reduced[pivot_row]
        ]
        for row in range(len(reduced)):
            if row == pivot_row or reduced[row][column] == 0:
                continue
            scale = reduced[row][column]
            reduced[row] = [
                (reduced[row][index] - scale * reduced[pivot_row][index])
                % r38.PRIME
                for index in range(len(reduced[row]))
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return reduced, pivots


def block_section(block, start, step):
    common_sum = (MAGIC_ORDER * start + step * MAGIC_SUM) % r38.SUBGROUP_ORDER
    translation = (
        -common_sum * pow(MAGIC_ORDER, -1, r38.SUBGROUP_ORDER)
    ) % r38.SUBGROUP_ORDER
    matrix = [
        SECTION_EMBEDDINGS[
            (start + step * index + translation) % r38.SUBGROUP_ORDER
        ]
        for index in block
    ]
    reduced, pivots = row_reduce(matrix)
    if len(pivots) != 8:
        raise AssertionError("equal-sum degree-nine block has wrong kernel rank")
    free_column = next(column for column in range(9) if column not in pivots)
    vector = [0] * 9
    vector[free_column] = 1
    for row, pivot in reversed(list(enumerate(pivots))):
        vector[pivot] = -sum(
            reduced[row][column] * vector[column]
            for column in range(pivot + 1, 9)
        ) % r38.PRIME
    first_nonzero = next(value for value in vector if value)
    scale = pow(first_nonzero, -1, r38.PRIME)
    return tuple(value * scale % r38.PRIME for value in vector), translation


def section_rank(vectors):
    return len(row_reduce([list(vector) for vector in vectors])[1])


def partition_rank(blocks, start, step):
    vectors = []
    translations = set()
    for block in blocks:
        vector, translation = block_section(block, start, step)
        vectors.append(vector)
        translations.add(translation)
    if len(translations) != 1:
        raise AssertionError("equal-sum blocks did not share one translation")
    return section_rank(vectors), vectors, translations.pop()


def exhaustive_affine_scan(blocks):
    histogram = collections.Counter()
    minimum_rank = 10
    witnesses = []
    for step in range(1, r38.SUBGROUP_ORDER):
        for start in range(r38.SUBGROUP_ORDER):
            rank, _, translation = partition_rank(blocks, start, step)
            histogram[rank] += 1
            if rank < minimum_rank:
                minimum_rank = rank
                witnesses = [
                    {
                        "start": start,
                        "step": step,
                        "translation": translation,
                    }
                ]
            elif rank == minimum_rank and len(witnesses) < 20:
                witnesses.append(
                    {
                        "start": start,
                        "step": step,
                        "translation": translation,
                    }
                )
    return {
        "instances": sum(histogram.values()),
        "rank_histogram": {
            str(rank): histogram[rank] for rank in sorted(histogram)
        },
        "minimum_rank": minimum_rank,
        "minimum_rank_witnesses": witnesses,
    }


def pair_positions_by_sum(block):
    positions = collections.defaultdict(list)
    for left in range(MAGIC_ORDER):
        for right in range(left + 1, MAGIC_ORDER):
            positions[block[left] + block[right]].append((left, right))
    return positions


def deterministic_trade_walk(initial_blocks):
    blocks = [sorted(block) for block in initial_blocks]
    random_source = random.Random(TRADE_SEED)
    contexts = list(AFFINE_CONTEXTS)
    context_vectors = {}
    histograms = {context: collections.Counter() for context in contexts}
    minimum = {
        "rank": 10,
        "trade": None,
        "context": None,
        "blocks": None,
    }

    for context in contexts:
        rank, vectors, _ = partition_rank(blocks, *context)
        context_vectors[context] = vectors
        histograms[context][rank] += 1
        if rank < minimum["rank"]:
            minimum = {
                "rank": rank,
                "trade": 0,
                "context": list(context),
                "blocks": [block[:] for block in blocks],
            }

    trades = 0
    attempts = 0
    while trades < TRADE_COUNT:
        attempts += 1
        first_block, second_block = random_source.sample(range(MAGIC_ORDER), 2)
        first_pairs = pair_positions_by_sum(blocks[first_block])
        second_pairs = pair_positions_by_sum(blocks[second_block])
        common_sums = sorted(set(first_pairs).intersection(second_pairs))
        if not common_sums:
            continue
        pair_sum = random_source.choice(common_sums)
        first_positions = random_source.choice(first_pairs[pair_sum])
        second_positions = random_source.choice(second_pairs[pair_sum])
        first_values = [blocks[first_block][position] for position in first_positions]
        second_values = [blocks[second_block][position] for position in second_positions]
        for position, value in zip(first_positions, second_values):
            blocks[first_block][position] = value
        for position, value in zip(second_positions, first_values):
            blocks[second_block][position] = value
        blocks[first_block].sort()
        blocks[second_block].sort()
        trades += 1

        for context in contexts:
            vectors = context_vectors[context]
            vectors[first_block] = block_section(
                blocks[first_block], *context
            )[0]
            vectors[second_block] = block_section(
                blocks[second_block], *context
            )[0]
            rank = section_rank(vectors)
            histograms[context][rank] += 1
            if rank < minimum["rank"]:
                minimum = {
                    "rank": rank,
                    "trade": trades,
                    "context": list(context),
                    "blocks": [block[:] for block in blocks],
                }

    return {
        "seed": TRADE_SEED,
        "trades": trades,
        "attempts": attempts,
        "contexts": [list(context) for context in contexts],
        "rank_histograms": {
            f"start_{context[0]}_step_{context[1]}": {
                str(rank): histograms[context][rank]
                for rank in sorted(histograms[context])
            }
            for context in contexts
        },
        "rank_instances": len(contexts) * (trades + 1),
        "minimum": minimum,
    }


def run():
    square = magic_square()
    row_blocks = [tuple(row) for row in square]
    column_blocks = [
        tuple(square[row][column] for row in range(MAGIC_ORDER))
        for column in range(MAGIC_ORDER)
    ]
    if {sum(block) for block in row_blocks + column_blocks} != {MAGIC_SUM}:
        raise AssertionError("normal magic-square blocks have unequal sums")

    row_scan = exhaustive_affine_scan(row_blocks)
    column_scan = exhaustive_affine_scan(column_blocks)
    trade_walk = deterministic_trade_walk(row_blocks)

    checks = {
        "row_affine_instances": row_scan["instances"] == 10506,
        "row_rank_is_always_nine": row_scan["rank_histogram"] == {"9": 10506},
        "column_affine_instances": column_scan["instances"] == 10506,
        "column_minimum_rank_is_eight": column_scan["minimum_rank"] == 8,
        "column_no_rank_two": all(
            int(rank) > 2 for rank in column_scan["rank_histogram"]
        ),
        "trade_count": trade_walk["trades"] == TRADE_COUNT,
        "trade_rank_instances": trade_walk["rank_instances"]
        == len(AFFINE_CONTEXTS) * (TRADE_COUNT + 1),
        "trade_minimum_rank_is_eight": trade_walk["minimum"]["rank"] == 8,
        "trade_no_rank_two": all(
            int(rank) > 2
            for histogram in trade_walk["rank_histograms"].values()
            for rank in histogram
        ),
    }
    if not all(checks.values()):
        raise AssertionError("primitive degree-nine rank search check failed")

    return {
        "schema": "p1553.primitive_degree_nine_rank_search.r44.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r38.PRIME,
        "subgroup_order": r38.SUBGROUP_ORDER,
        "linear_system": {
            "divisor": "9O",
            "dimension": 9,
            "basis": [
                "1",
                "x",
                "x^2",
                "x^3",
                "x^4",
                "y",
                "x*y",
                "x^2*y",
                "x^3*y",
            ],
            "fiber_partition_requirement": "nine equal-sum blocks with section-vector rank 2",
        },
        "magic_square": square,
        "row_partition_affine_scan": row_scan,
        "column_partition_affine_scan": column_scan,
        "equal_pair_sum_trade_walk": trade_walk,
        "checks": checks,
        "limits": [
            "the row and column affine scans are exhaustive only for two structured partitions",
            "the deterministic trade walk is a finite correlated family, not a random-sampling theorem",
            "rank greater than two is a negative witness only for the tested partition and affine context",
            "primitive degree-nine pencils outside the tested family remain open",
            "no target locator, R10 output, rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
