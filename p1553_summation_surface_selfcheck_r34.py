#!/usr/bin/env python3
"""Deterministic affine-resultant control for the R34 summation surface."""

import json

import sympy as sp


PRIME = 101
CURVE_A = 2
CURVE_B = 3


def curve_points():
    return [
        (x, y)
        for x in range(PRIME)
        for y in range(PRIME)
        if (y * y - x * x * x - CURVE_A * x - CURVE_B) % PRIME == 0
    ]


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


def third_summation(x_left, x_right, x_third, curve_a, curve_b):
    return sp.expand(
        (x_left - x_right) ** 2 * x_third**2
        - 2
        * ((x_left + x_right) * (x_left * x_right + curve_a) + 2 * curve_b)
        * x_third
        + (x_left * x_right - curve_a) ** 2
        - 4 * curve_b * (x_left + x_right)
    )


def main():
    center, plus_value, minus_value, intermediate = sp.symbols(
        "center plus_value minus_value intermediate"
    )
    curve_a, curve_b = sp.symbols("curve_a curve_b")

    recursive_resultant = sp.resultant(
        third_summation(
            plus_value,
            minus_value,
            intermediate,
            curve_a,
            curve_b,
        ),
        third_summation(
            center,
            center,
            intermediate,
            curve_a,
            curve_b,
        ),
        intermediate,
    )
    symbolic = sp.Poly(
        recursive_resultant,
        center,
        plus_value,
        minus_value,
        curve_a,
        curve_b,
    )
    finite = sp.Poly(
        recursive_resultant,
        center,
        plus_value,
        minus_value,
        curve_a,
        curve_b,
        modulus=PRIME,
    )

    def evaluate(center_x, plus_x, minus_x):
        return int(
            finite.eval(
                {
                    center: center_x,
                    plus_value: plus_x,
                    minus_value: minus_x,
                    curve_a: CURVE_A,
                    curve_b: CURVE_B,
                }
            )
        ) % PRIME

    points = curve_points()
    desired_checked = 0
    desired_failures = 0
    degenerate_skipped = 0

    for point in points:
        for shift in points:
            translated_plus = add(point, shift)
            translated_minus = add(point, negate(shift))
            doubled = add(point, point)
            if None in (translated_plus, translated_minus, doubled):
                degenerate_skipped += 1
                continue
            desired_checked += 1
            if evaluate(point[0], translated_plus[0], translated_minus[0]) != 0:
                desired_failures += 1

    if desired_failures:
        raise AssertionError("recursive resultant rejected a regular desired branch")

    cancellation_witness = None
    for point in points:
        for other in points:
            if point[0] == other[0]:
                continue
            value = evaluate(other[0], point[0], point[0])
            if value:
                cancellation_witness = {
                    "first_point": list(point),
                    "second_point": list(other),
                    "argument_x_values": [
                        point[0],
                        point[0],
                        other[0],
                        other[0],
                    ],
                    "resultant_value_mod_p": value,
                }
                break
        if cancellation_witness is not None:
            break

    if cancellation_witness is None:
        raise AssertionError("no omitted proper-subsum cancellation witness found")

    print(
        json.dumps(
            {
                "schema": "p1553.summation_surface_selfcheck.r34.v1",
                "classification": "toy_affine_resultant_strata_control",
                "field_prime": PRIME,
                "curve": {"a4": CURVE_A, "a6": CURVE_B},
                "affine_curve_point_count": len(points),
                "group_order_with_infinity": len(points) + 1,
                "recursive_resultant_degrees": {
                    "center": int(sp.degree(recursive_resultant, center)),
                    "plus": int(sp.degree(recursive_resultant, plus_value)),
                    "minus": int(sp.degree(recursive_resultant, minus_value)),
                    "total_in_three_value_variables": int(
                        sp.Poly(
                            recursive_resultant,
                            center,
                            plus_value,
                            minus_value,
                        ).total_degree()
                    ),
                },
                "recursive_resultant_term_count": len(symbolic.terms()),
                "regular_desired_branches_checked": desired_checked,
                "regular_desired_branch_failures": desired_failures,
                "degenerate_desired_branches_skipped": degenerate_skipped,
                "proper_subsum_cancellation_witness": cancellation_witness,
                "affine_recursive_resultant_is_complete_projective_iff": False,
                "breakthrough": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
