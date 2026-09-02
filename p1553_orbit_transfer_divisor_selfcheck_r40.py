#!/usr/bin/env python3
"""Find the minimum subgroup-supported transfer divisor for the R38 path."""

import collections
import json
import math

import p1553_low_boundary_pencil_search_r38 as r38


def optimize_simple_interval_transfer(order, interval_length):
    """Exhaust pole multisets by a dynamic program with exact Picard checks."""
    positive_first_moment = sum(range(interval_length))
    positive_second_moment = sum(
        index * (index + 1) for index in range(interval_length)
    ) % order

    minimum_pole_sum = interval_length * interval_length
    maximum_pole_sum = interval_length * (order - 1)
    minimum_k = math.ceil(
        (minimum_pole_sum - positive_first_moment) / order
    )
    maximum_k = math.floor(
        (maximum_pole_sum - positive_first_moment) / order
    )

    candidates = []
    for k in range(minimum_k, maximum_k + 1):
        target_pole_sum = positive_first_moment + k * order
        initial_degree = sum(
            max(-k + index, 0) for index in range(interval_length)
        )
        states = {(0, 0, 0): (initial_degree, ())}

        for position in range(interval_length, order):
            next_states = {}
            second_residue = position * (position + 1) % order
            for (used, pole_sum, residue), (cost, poles) in states.items():
                value_at_position = interval_length - k - used
                base_cost = cost + max(value_at_position, 0)
                for multiplicity in range(interval_length - used + 1):
                    next_used = used + multiplicity
                    next_sum = pole_sum + multiplicity * position
                    if next_sum > target_pole_sum:
                        break
                    remaining = interval_length - next_used
                    if next_sum + remaining * (position + 1) > target_pole_sum:
                        continue
                    if next_sum + remaining * (order - 1) < target_pole_sum:
                        continue
                    next_residue = (
                        residue + multiplicity * second_residue
                    ) % order
                    key = (next_used, next_sum, next_residue)
                    value = (
                        base_cost,
                        poles + (position,) * multiplicity,
                    )
                    if key not in next_states or value < next_states[key]:
                        next_states[key] = value
            states = next_states

        result = states.get(
            (interval_length, target_pole_sum, positive_second_moment)
        )
        if result is not None:
            candidates.append(
                {
                    "k": k,
                    "target_pole_sum": target_pole_sum,
                    "transfer_degree": result[0],
                    "pole_multiset": list(result[1]),
                }
            )

    if not candidates:
        raise AssertionError("no principal transfer divisor was found")
    optimum = min(
        candidates,
        key=lambda item: (
            item["transfer_degree"],
            item["pole_multiset"],
            item["k"],
        ),
    )
    return candidates, optimum


def reconstruct_divisors(order, interval_length, optimum):
    ratio_orders = [0] * order
    for index in range(interval_length):
        ratio_orders[index] = 1
    for position in optimum["pole_multiset"]:
        ratio_orders[position] -= 1

    transfer_orders = [-optimum["k"]]
    for index in range(order - 1):
        transfer_orders.append(
            transfer_orders[-1] + ratio_orders[index]
        )

    return ratio_orders, transfer_orders


def run():
    pencil = r38.search()
    if not pencil["pass"]:
        raise AssertionError("R38 positive control did not replay")

    order = pencil["subgroup_order"]
    step = pencil["step_scalar"]
    inverse_step = pow(step, -1, order)
    selected_q_coordinates = [
        scalar * inverse_step % order
        for scalar in pencil["selected_scalars_in_path_order"]
    ]
    rotation = selected_q_coordinates[0]
    rotated = [(coordinate - rotation) % order for coordinate in selected_q_coordinates]
    if rotated != list(range(9)):
        raise AssertionError("R38 selected points are not one q-interval")

    interval_length = len(rotated)
    candidates, optimum = optimize_simple_interval_transfer(
        order, interval_length
    )
    ratio_orders, transfer_orders = reconstruct_divisors(
        order, interval_length, optimum
    )

    zero_orders = {
        index: value for index, value in enumerate(ratio_orders) if value > 0
    }
    pole_orders = {
        index: -value for index, value in enumerate(ratio_orders) if value < 0
    }
    transfer_nonzero = {
        index: value
        for index, value in enumerate(transfer_orders)
        if value != 0
    }
    positive_degree = sum(max(value, 0) for value in transfer_orders)
    negative_degree = sum(max(-value, 0) for value in transfer_orders)
    picard_moment = sum(
        index * value for index, value in enumerate(transfer_orders)
    )

    expected_degrees = {
        1: 47,
        2: 29,
        3: 24,
        4: 21,
        5: 21,
        6: 24,
        7: 29,
        8: 47,
    }
    actual_degrees = {
        candidate["k"]: candidate["transfer_degree"]
        for candidate in candidates
    }

    checks = {
        "candidate_degree_profile": actual_degrees == expected_degrees,
        "optimal_degree": positive_degree == negative_degree == 21,
        "optimal_support": len(transfer_nonzero) == 19,
        "simple_interval_zeros": zero_orders == {index: 1 for index in range(9)},
        "poles_outside_interval": not set(pole_orders).intersection(range(9)),
        "ratio_degree_balance": sum(zero_orders.values()) == sum(pole_orders.values()) == 9,
        "transfer_degree_balance": sum(transfer_orders) == 0,
        "transfer_picard_principal": picard_moment % order == 0,
        "coboundary_difference": all(
            transfer_orders[(index + 1) % order] - transfer_orders[index]
            == ratio_orders[index]
            for index in range(order)
        ),
        "prime_order_no_proper_subgroup": all(
            order % divisor for divisor in range(2, order)
        ),
    }
    if not all(checks.values()):
        raise AssertionError("orbit transfer-divisor self-check failed")

    candidate_profile = []
    for candidate in candidates:
        counts = collections.Counter(candidate["pole_multiset"])
        candidate_profile.append(
            {
                "k": candidate["k"],
                "target_pole_sum": candidate["target_pole_sum"],
                "transfer_degree": candidate["transfer_degree"],
                "pole_orders": {
                    str(position): counts[position] for position in sorted(counts)
                },
            }
        )

    return {
        "schema": "p1553.orbit_transfer_divisor_selfcheck.r40.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": pencil["field_prime"],
        "subgroup_order": order,
        "step_scalar": step,
        "inverse_step_scalar": inverse_step,
        "selected_q_coordinates": selected_q_coordinates,
        "rotation_to_interval_zero": rotation,
        "rotated_selected_interval": rotated,
        "optimization": {
            "ratio_zero_orders_fixed_simple": True,
            "ratio_extra_zeros_forbidden": True,
            "ratio_poles_restricted_to_subgroup_orbit": True,
            "transfer_divisor_restricted_to_subgroup_orbit": True,
            "transfer_divisor_principal_conditions": [
                "sum(d_i)=0",
                "sum(i*d_i)=0 mod 103",
            ],
            "candidate_profile": candidate_profile,
            "optimal_k": optimum["k"],
            "optimal_transfer_degree": positive_degree,
            "optimal_transfer_support_size": len(transfer_nonzero),
            "optimal_transfer_maximum_absolute_order": max(
                abs(value) for value in transfer_orders
            ),
            "optimal_ratio_zero_orders": {
                str(index): zero_orders[index] for index in sorted(zero_orders)
            },
            "optimal_ratio_pole_orders": {
                str(index): pole_orders[index] for index in sorted(pole_orders)
            },
            "optimal_transfer_orders": {
                str(index): transfer_nonzero[index]
                for index in sorted(transfer_nonzero)
            },
            "transfer_picard_moment": picard_moment,
            "standard_sequential_generalized_miller_line_quotient_upper_bound": 40,
        },
        "asymptotic_interface": {
            "interval_length": "L=B^2",
            "transfer_support_lower_bound": "L",
            "transfer_degree_lower_bound": "Omega(L^2)=Omega(B^4)",
            "explicit_support_miller_evaluation_lower_bound": "Omega(L)=Omega(B^2)",
            "proper_subgroup_kernel_available_in_prime_order_G": False,
            "single_miller_function_interval_support": False,
            "division_polynomial_interval_support": False,
            "remaining_exception": "new partial-orbit elliptic shifted-factorial circuit",
        },
        "checks": checks,
        "limits": [
            "the dynamic-program optimum is for the exact toy and frozen simple-zero grammar",
            "the asymptotic bounds apply to explicit transfer-divisor support",
            "divisor degree and support do not lower-bound unrestricted arithmetic-circuit size",
            "a new partial-orbit product identity or nonlinear marked locator is not excluded",
            "no degree-nine pencil lift or ECDLP relation campaign is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
