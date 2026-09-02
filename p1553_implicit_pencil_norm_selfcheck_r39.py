#!/usr/bin/env python3
"""Probe composition, coboundary, and recurrence escapes for the R38 pencil."""

import collections
import json

import p1553_low_boundary_pencil_search_r38 as r38


def berlekamp_massey(sequence, modulus):
    """Return the shortest connection polynomial for a finite field sequence."""
    connection = [1]
    previous = [1]
    order = 0
    shift = 1
    previous_discrepancy = 1

    for index in range(len(sequence)):
        discrepancy = sequence[index] % modulus
        for offset in range(1, order + 1):
            discrepancy = (
                discrepancy
                + connection[offset] * sequence[index - offset]
            ) % modulus
        if discrepancy == 0:
            shift += 1
            continue

        old_connection = connection[:]
        scale = discrepancy * pow(previous_discrepancy, -1, modulus) % modulus
        required = len(previous) + shift
        if len(connection) < required:
            connection.extend([0] * (required - len(connection)))
        for offset, coefficient in enumerate(previous):
            connection[offset + shift] = (
                connection[offset + shift] - scale * coefficient
            ) % modulus

        if 2 * order <= index:
            order = index + 1 - order
            previous = old_connection
            previous_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

    return order, connection


def cyclic_linear_complexity(sequence, modulus):
    """Recover and verify the recurrence of a complete periodic sequence."""
    order, connection = berlekamp_massey(sequence + sequence, modulus)
    period = len(sequence)
    for index in range(2 * period, 3 * period):
        residual = sequence[index % period]
        residual += sum(
            connection[offset] * sequence[(index - offset) % period]
            for offset in range(1, order + 1)
        )
        if residual % modulus:
            raise AssertionError("Berlekamp-Massey recurrence did not replay")
    return order


def rational_locator_values(report):
    """Evaluate an intrinsic rational locator with no subgroup poles."""
    translation = report["embedding_translation_scalar"]
    numerator = tuple(report["pencil_lines"][0])
    denominator = tuple(report["pencil_lines"][1])
    pencil_values = [
        r38.pencil_value(scalar, translation, numerator, denominator)
        for scalar in range(r38.SUBGROUP_ORDER)
    ]
    used_finite_values = {
        value for value in pencil_values if value != "infinity"
    }
    reference_value = next(
        value for value in range(r38.PRIME) if value not in used_finite_values
    )

    values = []
    for scalar in range(r38.SUBGROUP_ORDER):
        point = r38.projective(
            r38.scalar_mul(
                (scalar + translation) % r38.SUBGROUP_ORDER,
                r38.GENERATOR,
            )
        )
        numerator_value = r38.dot(numerator, point)
        denominator_value = r38.dot(denominator, point)
        top = numerator_value * denominator_value % r38.PRIME
        top *= (numerator_value - 192 * denominator_value) % r38.PRIME
        top %= r38.PRIME
        bottom = pow(
            (numerator_value - reference_value * denominator_value) % r38.PRIME,
            3,
            r38.PRIME,
        )
        if bottom == 0:
            raise AssertionError("reference fiber meets the subgroup")
        values.append(top * r38.inverse(bottom) % r38.PRIME)

    selected = set(report["selected_scalars_in_path_order"])
    if {index for index, value in enumerate(values) if value == 0} != selected:
        raise AssertionError("rational locator has the wrong subgroup zero set")
    return pencil_values, reference_value, values


def translated_norm_sequence(locator_values, endpoint_support):
    sequence = []
    for target in range(r38.SUBGROUP_ORDER):
        value = 1
        for endpoint in endpoint_support:
            value *= locator_values[
                (target - endpoint) % r38.SUBGROUP_ORDER
            ]
            value %= r38.PRIME
        sequence.append(value)
    return sequence


def endpoint_grid(first_step, second_step):
    return {
        (left * first_step + right * second_step) % r38.SUBGROUP_ORDER
        for left in range(r38.DEGREE)
        for right in range(r38.DEGREE)
    }


def run():
    report = r38.search()
    if not report["pass"]:
        raise AssertionError("R38 positive control did not replay")

    pencil_values, reference_value, locator_values = rational_locator_values(report)
    step = report["step_scalar"]
    start = report["start_scalar"]

    scale_up_path = [
        (start + offset * step) % r38.SUBGROUP_ORDER for offset in range(81)
    ]
    path_value_counts = collections.Counter(
        pencil_values[scalar] for scalar in scale_up_path
    )
    multiplicity_histogram = collections.Counter(path_value_counts.values())

    named_decks = {
        "aligned_path": endpoint_grid(step, 3 * step),
        "grid_1_7": endpoint_grid(1, 7),
        "grid_11_29": endpoint_grid(11, 29),
    }
    named_results = {}
    for name, support in named_decks.items():
        if len(support) != 9:
            raise AssertionError("named endpoint deck is degenerate")
        sequence = translated_norm_sequence(locator_values, support)
        zero_indicator = [int(value == 0) for value in sequence]
        named_results[name] = {
            "endpoint_support": sorted(support),
            "zero_support_size": sum(zero_indicator),
            "norm_cyclic_linear_complexity": cyclic_linear_complexity(
                sequence, r38.PRIME
            ),
            "zero_indicator_cyclic_linear_complexity": cyclic_linear_complexity(
                zero_indicator, r38.PRIME
            ),
        }

    complexity_histogram = collections.Counter()
    tested_grids = 0
    for first_step in range(1, r38.SUBGROUP_ORDER):
        for second_step in range(first_step, r38.SUBGROUP_ORDER):
            support = endpoint_grid(first_step, second_step)
            if len(support) != 9:
                continue
            tested_grids += 1
            sequence = translated_norm_sequence(locator_values, support)
            complexity_histogram[
                cyclic_linear_complexity(sequence, r38.PRIME)
            ] += 1

    expected_named = {
        "aligned_path": (17, 102, 103),
        "grid_1_7": (75, 102, 103),
        "grid_11_29": (45, 102, 103),
    }
    for name, expected in expected_named.items():
        actual = named_results[name]
        observed = (
            actual["zero_support_size"],
            actual["norm_cyclic_linear_complexity"],
            actual["zero_indicator_cyclic_linear_complexity"],
        )
        if observed != expected:
            raise AssertionError(f"unexpected named-deck result for {name}")

    passed = all(
        (
            reference_value == 1,
            len(path_value_counts) == 66,
            dict(multiplicity_histogram) == {1: 57, 2: 3, 3: 6},
            tested_grids == 4896,
            dict(complexity_histogram) == {102: 4896},
        )
    )

    return {
        "schema": "p1553.implicit_pencil_norm_selfcheck.r39.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r38.PRIME,
        "subgroup_order": r38.SUBGROUP_ORDER,
        "pencil_degree": r38.DEGREE,
        "selected_path_size": len(report["selected_scalars_in_path_order"]),
        "step_scalar": step,
        "rational_locator": {
            "homogeneous_numerator": "N*D*(N-192*D)",
            "homogeneous_denominator": "(N-c_ref*D)^3",
            "reference_value": reference_value,
            "reference_fiber_subgroup_preimages": 0,
            "subgroup_zero_support": len(report["selected_scalars_in_path_order"]),
            "subgroup_zero_orbit_valuation_mass": 9,
            "translation_coboundary_possible": False,
            "reason": "nonzero valuation mass on the subgroup translation orbit",
        },
        "naive_degree_nine_scale_up": {
            "path_size": len(scale_up_path),
            "distinct_inner_pencil_values": len(path_value_counts),
            "maximum_inner_values_allowed_by_degree_3_composition": 27,
            "inner_value_multiplicity_histogram": {
                str(key): multiplicity_histogram[key]
                for key in sorted(multiplicity_histogram)
            },
            "full_three_point_inner_fibers": sum(
                count == 3 for count in path_value_counts.values()
            ),
            "necessary_inner_saturation_pass": len(path_value_counts) <= 27,
        },
        "translated_norm_recurrence": {
            "endpoint_deck_size": 9,
            "arithmetic_grids_tested": tested_grids,
            "norm_cyclic_linear_complexity_histogram": {
                str(key): complexity_histogram[key]
                for key in sorted(complexity_histogram)
            },
            "named_decks": named_results,
            "short_constant_recurrence_found": False,
        },
        "limits": [
            "toy evidence is not an asymptotic lower bound",
            "linear complexity does not exclude nonlinear or variable-coefficient recurrences",
            "divisor degree does not lower-bound arithmetic-circuit size",
            "the scalar-labelled exhaustive grid scan is not an ECDLP algorithm",
        ],
        "pass": passed,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
