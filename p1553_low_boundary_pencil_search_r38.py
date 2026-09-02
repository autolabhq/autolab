#!/usr/bin/env python3
"""Search for a degree-three line pencil with a one-path selected union."""

import itertools
import json


PRIME = 193
CURVE_A = 2
CURVE_B = 3
SUBGROUP_ORDER = 103
GENERATOR = (1, 44)
DEGREE = 3
SELECTED_FIBERS = 3
SELECTED_POINTS = DEGREE * SELECTED_FIBERS


def inverse(value):
    return pow(value % PRIME, -1, PRIME)


def negate(point):
    if point is None:
        return None
    return point[0], (-point[1]) % PRIME


def add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    x_left, y_left = left
    x_right, y_right = right
    if x_left == x_right and (y_left + y_right) % PRIME == 0:
        return None
    if left == right:
        if y_left == 0:
            return None
        slope = (3 * x_left * x_left + CURVE_A) * inverse(2 * y_left)
    else:
        slope = (y_right - y_left) * inverse(x_right - x_left)
    slope %= PRIME
    x_sum = (slope * slope - x_left - x_right) % PRIME
    y_sum = (slope * (x_left - x_sum) - y_left) % PRIME
    return x_sum, y_sum


def scalar_mul(scalar, point):
    result = None
    addend = point
    scalar %= SUBGROUP_ORDER
    while scalar:
        if scalar & 1:
            result = add(result, addend)
        addend = add(addend, addend)
        scalar >>= 1
    return result


def projective(point):
    if point is None:
        return (0, 1, 0)
    return (point[0], point[1], 1)


def cross(left, right):
    return (
        (left[1] * right[2] - left[2] * right[1]) % PRIME,
        (left[2] * right[0] - left[0] * right[2]) % PRIME,
        (left[0] * right[1] - left[1] * right[0]) % PRIME,
    )


def dot(left, right):
    return sum(a * b for a, b in zip(left, right)) % PRIME


def normalized(vector):
    for value in vector:
        if value:
            scale = inverse(value)
            return tuple((entry * scale) % PRIME for entry in vector)
    raise ValueError("zero projective vector")


def partitions_into_triples(values):
    values = tuple(values)
    first = values[0]
    tail = values[1:]
    for pair in itertools.combinations(tail, 2):
        block = (first,) + pair
        remaining = tuple(value for value in values if value not in block)
        second_first = remaining[0]
        for second_pair in itertools.combinations(remaining[1:], 2):
            second = (second_first,) + second_pair
            third = tuple(value for value in remaining if value not in second)
            if second < third:
                yield (block, second, third)


def fiber_line(block, translation):
    shifted = [
        projective(scalar_mul((scalar + translation) % SUBGROUP_ORDER, GENERATOR))
        for scalar in block
    ]
    line = cross(shifted[0], shifted[1])
    if line == (0, 0, 0) or dot(line, shifted[2]):
        return None
    return normalized(line)


def line_value(line, numerator, denominator):
    # Find alpha,beta with line=alpha*numerator+beta*denominator.
    for i, j in itertools.combinations(range(3), 2):
        determinant = (
            numerator[i] * denominator[j]
            - numerator[j] * denominator[i]
        ) % PRIME
        if not determinant:
            continue
        alpha = (
            line[i] * denominator[j] - line[j] * denominator[i]
        ) * inverse(determinant) % PRIME
        beta = (
            numerator[i] * line[j] - numerator[j] * line[i]
        ) * inverse(determinant) % PRIME
        if all(
            (line[k] - alpha * numerator[k] - beta * denominator[k]) % PRIME
            == 0
            for k in range(3)
        ):
            if alpha == 0:
                return "infinity"
            return int((-beta * inverse(alpha)) % PRIME)
    return None


def pencil_value(scalar, translation, numerator, denominator):
    point = projective(
        scalar_mul((scalar + translation) % SUBGROUP_ORDER, GENERATOR)
    )
    numerator_value = dot(numerator, point)
    denominator_value = dot(denominator, point)
    if denominator_value == 0:
        if numerator_value == 0:
            raise AssertionError("pencil base point lies on the curve")
        return "infinity"
    return int(numerator_value * inverse(denominator_value) % PRIME)


def search():
    inv_degree = pow(DEGREE, -1, SUBGROUP_ORDER)
    tested_partitions = 0
    equal_sum_partitions = 0
    concurrent_partitions = 0

    for step in range(1, SUBGROUP_ORDER):
        for start in range(SUBGROUP_ORDER):
            interval = tuple(
                (start + offset * step) % SUBGROUP_ORDER
                for offset in range(SELECTED_POINTS)
            )
            if len(set(interval)) != SELECTED_POINTS or 0 in interval:
                continue
            for partition in partitions_into_triples(interval):
                tested_partitions += 1
                sums = [sum(block) % SUBGROUP_ORDER for block in partition]
                if len(set(sums)) != 1:
                    continue
                equal_sum_partitions += 1
                translation = (-sums[0] * inv_degree) % SUBGROUP_ORDER
                lines = [fiber_line(block, translation) for block in partition]
                if any(line is None for line in lines) or len(set(lines)) < 3:
                    continue
                base = cross(lines[0], lines[1])
                if base == (0, 0, 0) or dot(lines[2], base):
                    continue
                concurrent_partitions += 1
                base = normalized(base)
                values = [line_value(line, lines[0], lines[1]) for line in lines]
                if any(value is None for value in values) or len(set(values)) < 3:
                    continue

                selected_fiber_check = {}
                for value, block in zip(values, partition):
                    actual = sorted(
                        scalar
                        for scalar in range(SUBGROUP_ORDER)
                        if pencil_value(
                            scalar,
                            translation,
                            lines[0],
                            lines[1],
                        )
                        == value
                    )
                    expected = sorted(block)
                    selected_fiber_check[str(value)] = {
                        "expected": expected,
                        "actual": actual,
                        "pass": actual == expected,
                    }
                if not all(item["pass"] for item in selected_fiber_check.values()):
                    continue

                selected = set(interval)
                outgoing = [
                    scalar
                    for scalar in interval
                    if (scalar + step) % SUBGROUP_ORDER not in selected
                ]
                incoming = [
                    scalar
                    for scalar in interval
                    if (scalar - step) % SUBGROUP_ORDER not in selected
                ]
                complete = [
                    scalar
                    for scalar in interval
                    if (scalar + step) % SUBGROUP_ORDER in selected
                    and (scalar - step) % SUBGROUP_ORDER in selected
                ]
                return {
                    "schema": "p1553.low_boundary_pencil_search.r38.v1",
                    "classification": [
                        "toy",
                        "exact",
                        "model-bound",
                        "novelty-unverified",
                    ],
                    "field_prime": PRIME,
                    "curve": {"a4": CURVE_A, "a6": CURVE_B},
                    "subgroup_order": SUBGROUP_ORDER,
                    "generator": list(GENERATOR),
                    "degree": DEGREE,
                    "selected_fibers": SELECTED_FIBERS,
                    "selected_points": SELECTED_POINTS,
                    "step_scalar": step,
                    "step_known_log": True,
                    "start_scalar": start,
                    "selected_scalars_in_path_order": list(interval),
                    "fiber_scalar_blocks": [list(block) for block in partition],
                    "common_fiber_sum_scalar": sums[0],
                    "embedding_translation_scalar": translation,
                    "pencil_lines": [list(line) for line in lines],
                    "pencil_base_point": list(base),
                    "selected_values": values,
                    "verified_selected_fibers": selected_fiber_check,
                    "outgoing_boundary_scalars": outgoing,
                    "incoming_boundary_scalars": incoming,
                    "boundary_size": len(outgoing),
                    "complete_plus_minus_domain_size": len(complete),
                    "tested_partitions": tested_partitions,
                    "equal_sum_partitions": equal_sum_partitions,
                    "concurrent_partitions": concurrent_partitions,
                    "pass": len(outgoing) == 1 and len(incoming) == 1,
                }

    return {
        "schema": "p1553.low_boundary_pencil_search.r38.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "subgroup_order": SUBGROUP_ORDER,
        "degree": DEGREE,
        "selected_fibers": SELECTED_FIBERS,
        "selected_points": SELECTED_POINTS,
        "tested_partitions": tested_partitions,
        "equal_sum_partitions": equal_sum_partitions,
        "concurrent_partitions": concurrent_partitions,
        "pass": False,
    }


def main():
    print(json.dumps(search(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
