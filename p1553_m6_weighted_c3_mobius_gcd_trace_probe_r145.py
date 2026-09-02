#!/usr/bin/env python3
"""Audit an exact weighted C3-pair count through a Mobius gcd trace."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_weighted_c3_mobius_gcd_trace.r145.v1"

R144_PRODUCER = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_r144.py"
)
R144_REPORT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_report_r144.json"
)
R144_FROZEN = ROOT / "frozen_m6_weighted_fiber_marginal_log_operator.json"
R144_COST = ROOT / "m6_weighted_fiber_marginal_log_operator_cost_ledger.json"
R144_REPLAY = ROOT / "m6_weighted_fiber_marginal_log_operator_replay.json"
R144_CONTROLS = ROOT / "m6_weighted_fiber_marginal_log_operator_controls.json"
R144_LOGS = ROOT / "factor_logs_and_identical_descent_r144.json"
R144_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_weighted_fiber_marginal_log_operator_probe_r144.py"
)
R144_GATE = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_gate_r144.md"
)
R144_PARENT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_parent_report_r144.yaml"
)
R116_REPORT = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_"
    "probe_report_r116.json"
)
R116_GATE = ROOT / (
    "p1553_m6_a6_batched_c3_pair_sum_source_locator_"
    "probe_gate_r116.md"
)
R117_REPORT = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_"
    "probe_report_r117.json"
)
R117_GATE = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_"
    "probe_gate_r117.md"
)

SOURCE_BINDINGS = (
    (
        "r144_producer",
        R144_PRODUCER,
        "3cd7a3e40bf696a836ff6b6cad4c59a7ca73a44d72d152232cddeffa14e90a5c",
    ),
    (
        "r144_report",
        R144_REPORT,
        "954bc230dbe28300534a765c3bde389b7ae807767b664dc82f63ede1b3a5805d",
    ),
    (
        "r144_frozen",
        R144_FROZEN,
        "70e6a2d8c6fb9b2eb579d4deb3a398a21ad126a34175d3c8f00d64cd3bda02fc",
    ),
    (
        "r144_cost",
        R144_COST,
        "20a1dd3117e76d7c99160ff4c23efd51286c438e28ba4479a18965cd27a1ab3f",
    ),
    (
        "r144_replay",
        R144_REPLAY,
        "1b2b4533fb7d635cc22ab564758ecfd5134a053f3ee27da6ec666659e63ad327",
    ),
    (
        "r144_controls",
        R144_CONTROLS,
        "0677e3a7220e7258e4dc56bff14f5e083a09ed048948667fe0582dd2fa4c6ea8",
    ),
    (
        "r144_logs",
        R144_LOGS,
        "260511a7965ce75937fbf22c41f6cd16060140b3de4741e091e995ecf1506e4a",
    ),
    (
        "r144_test",
        R144_TEST,
        "34c164e5aeb86b2d0da6bb2ff53da79e692cc0e59df88293375a8a7395b8e6d0",
    ),
    (
        "r144_gate",
        R144_GATE,
        "c967cc10137fe861384a80a8261763c2ecb86ca2951db8c5bf43f348c65e1e9b",
    ),
    (
        "r144_parent",
        R144_PARENT,
        "db906bd74c9c43fd12ab196a36d59ac80bfbbd22de494c4b6b2efb4d0c96c268",
    ),
    (
        "r116_report",
        R116_REPORT,
        "9c5ebb9e99eaada2296c3d4f0fcbdb472995f3ae071bb68138e364e266f93d25",
    ),
    (
        "r116_gate",
        R116_GATE,
        "1a17b8e396affe9ced0a5d589059ec6d3d000ee23f56522e1f55e353bac72542",
    ),
    (
        "r117_report",
        R117_REPORT,
        "449021dd58e6567bb827bcfe41336175d7b628acda4446e1b63d9a46cf48f824",
    ),
    (
        "r117_gate",
        R117_GATE,
        "014f74c772dfc77d8821f658b10ef9bd11a644fe8f0b356c24fda61ef81d9de5",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_weighted_c3_mobius_gcd_trace_probe_report_r145.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_weighted_c3_mobius_gcd_trace.json"
DEFAULT_COST = ROOT / "m6_weighted_c3_mobius_gcd_trace_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_weighted_c3_mobius_gcd_trace_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_weighted_c3_mobius_gcd_trace_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r145.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R144 = load_module("p1553_r144_for_r145", R144_PRODUCER)
R141 = R144.R141


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R145 source binding mismatch: {failures}")
    return actual


def poly_trim(values: Iterable[int], prime: int) -> list[int]:
    result = [value % prime for value in values]
    while result and result[-1] == 0:
        result.pop()
    return result


def poly_add(left: list[int], right: list[int], prime: int) -> list[int]:
    width = max(len(left), len(right))
    return poly_trim(
        [
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
            for index in range(width)
        ],
        prime,
    )


def poly_sub(left: list[int], right: list[int], prime: int) -> list[int]:
    width = max(len(left), len(right))
    return poly_trim(
        [
            (left[index] if index < len(left) else 0)
            - (right[index] if index < len(right) else 0)
            for index in range(width)
        ],
        prime,
    )


def poly_scale(values: list[int], scalar: int, prime: int) -> list[int]:
    return poly_trim((scalar * value for value in values), prime)


def poly_mul(left: list[int], right: list[int], prime: int) -> list[int]:
    if not left or not right:
        return []
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + left_value * right_value
            ) % prime
    return poly_trim(result, prime)


def poly_divmod(
    numerator: list[int],
    denominator: list[int],
    prime: int,
) -> tuple[list[int], list[int]]:
    numerator = poly_trim(numerator, prime)
    denominator = poly_trim(denominator, prime)
    if not denominator:
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return [], numerator
    quotient = [0] * (len(numerator) - len(denominator) + 1)
    inverse_lead = pow(denominator[-1], prime - 2, prime)
    while numerator and len(numerator) >= len(denominator):
        shift = len(numerator) - len(denominator)
        scale = numerator[-1] * inverse_lead % prime
        quotient[shift] = scale
        for index, value in enumerate(denominator):
            numerator[index + shift] = (
                numerator[index + shift] - scale * value
            ) % prime
        numerator = poly_trim(numerator, prime)
    return poly_trim(quotient, prime), numerator


def poly_mod(values: list[int], modulus: list[int], prime: int) -> list[int]:
    return poly_divmod(values, modulus, prime)[1]


def poly_gcd(left: list[int], right: list[int], prime: int) -> list[int]:
    left = poly_trim(left, prime)
    right = poly_trim(right, prime)
    while right:
        left, right = right, poly_mod(left, right, prime)
    if not left:
        return []
    return poly_scale(left, pow(left[-1], prime - 2, prime), prime)


def poly_xgcd(
    left: list[int],
    right: list[int],
    prime: int,
) -> tuple[list[int], list[int], list[int]]:
    old_r, r = poly_trim(left, prime), poly_trim(right, prime)
    old_s, s = [1], []
    old_t, t = [], [1]
    while r:
        quotient, remainder = poly_divmod(old_r, r, prime)
        old_r, r = r, remainder
        old_s, s = s, poly_sub(
            old_s,
            poly_mul(quotient, s, prime),
            prime,
        )
        old_t, t = t, poly_sub(
            old_t,
            poly_mul(quotient, t, prime),
            prime,
        )
    if not old_r:
        return [], [], []
    inverse = pow(old_r[-1], prime - 2, prime)
    return (
        poly_scale(old_r, inverse, prime),
        poly_scale(old_s, inverse, prime),
        poly_scale(old_t, inverse, prime),
    )


def poly_inverse_mod(
    values: list[int],
    modulus: list[int],
    prime: int,
) -> list[int]:
    gcd, coefficient, _ = poly_xgcd(values, modulus, prime)
    if gcd != [1]:
        raise ZeroDivisionError("polynomial is not invertible modulo divisor")
    return poly_mod(coefficient, modulus, prime)


def poly_pow(
    values: list[int],
    exponent: int,
    prime: int,
    modulus: list[int] | None = None,
) -> list[int]:
    result = [1]
    base = values
    power = exponent
    while power:
        if power & 1:
            result = poly_mul(result, base, prime)
            if modulus:
                result = poly_mod(result, modulus, prime)
        power >>= 1
        if power:
            base = poly_mul(base, base, prime)
            if modulus:
                base = poly_mod(base, modulus, prime)
    return result


def root_polynomial(roots: list[int], prime: int) -> list[int]:
    result = [1]
    for root in roots:
        result = poly_mul(result, [(-root) % prime, 1], prime)
    return result


def poly_evaluate(values: list[int], point: int, prime: int) -> int:
    result = 0
    for value in reversed(values):
        result = (result * point + value) % prime
    return result


def interpolate(
    points: list[int],
    values: list[int],
    prime: int,
) -> list[int]:
    result: list[int] = []
    for index, point in enumerate(points):
        basis = [1]
        denominator = 1
        for other_index, other in enumerate(points):
            if other_index == index:
                continue
            basis = poly_mul(basis, [(-other) % prime, 1], prime)
            denominator = denominator * (point - other) % prime
        result = poly_add(
            result,
            poly_scale(
                basis,
                values[index] * pow(denominator, prime - 2, prime),
                prime,
            ),
            prime,
        )
    return result


def mobius_pullback(
    polynomial: list[int],
    target: int,
    nonsquare: int,
    homogeneous_degree: int,
    prime: int,
) -> list[int]:
    numerator = [target % prime, -1 % prime]
    denominator = [1, (-nonsquare * target) % prime]
    numerator_powers = [[1]]
    denominator_powers = [[1]]
    for _ in range(homogeneous_degree):
        numerator_powers.append(
            poly_mul(numerator_powers[-1], numerator, prime)
        )
        denominator_powers.append(
            poly_mul(denominator_powers[-1], denominator, prime)
        )
    result: list[int] = []
    for degree, coefficient in enumerate(polynomial):
        if not coefficient:
            continue
        result = poly_add(
            result,
            poly_scale(
                poly_mul(
                    numerator_powers[degree],
                    denominator_powers[
                        homogeneous_degree - degree
                    ],
                    prime,
                ),
                coefficient,
                prime,
            ),
            prime,
        )
    return result


def quotient_trace(
    value: list[int],
    monic_modulus: list[int],
    prime: int,
) -> int:
    degree = len(monic_modulus) - 1
    if degree <= 0:
        return 0
    value = poly_mod(value, monic_modulus, prime)
    trace = 0
    for basis_index in range(degree):
        product = poly_mod(
            [0] * basis_index + value,
            monic_modulus,
            prime,
        )
        if basis_index < len(product):
            trace = (trace + product[basis_index]) % prime
    return trace


def mobius_parameter(
    left: int,
    target: int,
    nonsquare: int,
    prime: int,
) -> int | None:
    denominator = (1 - nonsquare * target * left) % prime
    if denominator == 0:
        return None
    return (target - left) * pow(denominator, prime - 2, prime) % prime


def count_query(
    target_parameter: int,
    support_polynomial: list[int],
    weight_interpolant: list[int],
    support_parameters: list[int],
    support_weights: dict[int, int],
    nonsquare: int,
    prime: int,
) -> dict[str, Any]:
    support_degree = len(support_parameters)
    transformed_support = mobius_pullback(
        support_polynomial,
        target_parameter,
        nonsquare,
        support_degree,
        prime,
    )
    common_divisor = poly_gcd(
        support_polynomial,
        transformed_support,
        prime,
    )
    common_degree = max(0, len(common_divisor) - 1)
    direct_common = []
    direct_count = 0
    denominator_pole_count = 0
    for parameter in support_parameters:
        partner = mobius_parameter(
            parameter,
            target_parameter,
            nonsquare,
            prime,
        )
        if partner is None:
            denominator_pole_count += 1
            continue
        if partner in support_weights:
            direct_common.append(parameter)
            direct_count += (
                support_weights[parameter] * support_weights[partner]
            )

    if common_degree:
        denominator = [
            1,
            (-nonsquare * target_parameter) % prime,
        ]
        transformed_weight_numerator = mobius_pullback(
            weight_interpolant,
            target_parameter,
            nonsquare,
            support_degree - 1,
            prime,
        )
        denominator_power = poly_pow(
            denominator,
            support_degree - 1,
            prime,
            common_divisor,
        )
        transformed_weight = poly_mul(
            poly_mod(transformed_weight_numerator, common_divisor, prime),
            poly_inverse_mod(
                denominator_power,
                common_divisor,
                prime,
            ),
            prime,
        )
        weighted_value = poly_mul(
            poly_mod(weight_interpolant, common_divisor, prime),
            poly_mod(transformed_weight, common_divisor, prime),
            prime,
        )
        trace_count = quotient_trace(
            weighted_value,
            common_divisor,
            prime,
        )
    else:
        trace_count = 0

    return {
        "target_parameter": target_parameter,
        "support_degree": support_degree,
        "transformed_support_degree": max(
            0,
            len(transformed_support) - 1,
        ),
        "gcd_degree": common_degree,
        "direct_common_support_count": len(direct_common),
        "gcd_degree_matches_direct_intersection": (
            common_degree == len(direct_common)
        ),
        "direct_ordered_pair_count": direct_count,
        "quotient_trace_count_mod_field_prime": trace_count,
        "trace_count_exact_integer": trace_count == direct_count,
        "denominator_pole_count_on_support": denominator_pole_count,
        "root_oracle_consumed": False,
        "discrete_log_oracle_consumed": False,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck = R141.R121.pairing_deck(curve, offset)
    c3_occurrences: collections.Counter[Any] = collections.Counter(
        field.product(source)
        for source in itertools.product(deck, repeat=3)
    )
    minus_one = field.neg(field.one)
    if minus_one in c3_occurrences:
        raise AssertionError("C3 support reaches omitted Cayley point")
    parameter_weights = {
        R141.R121.torus_parameter(value, field): count
        for value, count in c3_occurrences.items()
    }
    if len(parameter_weights) != len(c3_occurrences):
        raise AssertionError("Cayley parameters collide on C3 support")
    parameters = sorted(parameter_weights)
    support_polynomial = root_polynomial(parameters, field.p)
    weight_interpolant = interpolate(
        parameters,
        [parameter_weights[value] for value in parameters],
        field.p,
    )
    roots_exact = all(
        poly_evaluate(support_polynomial, value, field.p) == 0
        for value in parameters
    )
    weights_exact = all(
        poly_evaluate(weight_interpolant, value, field.p)
        == parameter_weights[value] % field.p
        for value in parameters
    )

    c6_occurrences: collections.Counter[Any] = collections.Counter()
    for left, left_count in c3_occurrences.items():
        for right, right_count in c3_occurrences.items():
            c6_occurrences[field.mul(left, right)] += (
                left_count * right_count
            )
    positive_parameters = sorted(
        R141.R121.torus_parameter(value, field)
        for value in c6_occurrences
        if value != minus_one
    )
    positive_sample_parameters = positive_parameters[
        : min(12, len(positive_parameters))
    ]
    positive_queries = [
        count_query(
            target,
            support_polynomial,
            weight_interpolant,
            parameters,
            parameter_weights,
            field.nonsquare,
            field.p,
        )
        for target in positive_sample_parameters
    ]
    positive_target_values = {
        R141.R121.torus_parameter(value, field): value
        for value in c6_occurrences
        if value != minus_one
    }
    for row in positive_queries:
        target_value = positive_target_values[row["target_parameter"]]
        row["direct_c6_counter_count"] = c6_occurrences[target_value]
        row["trace_matches_direct_c6_counter"] = (
            row["quotient_trace_count_mod_field_prime"]
            == c6_occurrences[target_value]
        )

    empty_parameters = []
    candidate = 0
    positive_set = set(positive_parameters)
    while len(empty_parameters) < 6:
        if candidate not in positive_set:
            empty_parameters.append(candidate)
        candidate += 1
    empty_queries = [
        count_query(
            target,
            support_polynomial,
            weight_interpolant,
            parameters,
            parameter_weights,
            field.nonsquare,
            field.p,
        )
        for target in empty_parameters
    ]
    all_queries = [*positive_queries, *empty_queries]
    all_exact = all(
        row["gcd_degree_matches_direct_intersection"]
        and row["trace_count_exact_integer"]
        and not row["denominator_pole_count_on_support"]
        for row in all_queries
    )
    all_positive_exact = all(
        row["trace_matches_direct_c6_counter"]
        and row["direct_c6_counter_count"] > 0
        for row in positive_queries
    )
    all_empty_exact = all(
        row["gcd_degree"] == 0
        and row["direct_ordered_pair_count"] == 0
        and row["quotient_trace_count_mod_field_prime"] == 0
        for row in empty_queries
    )

    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": len(deck),
        "ordered_c3_occurrence_count": len(deck) ** 3,
        "c3_support_degree": len(parameters),
        "c3_support_formula": f"binom({len(deck)}+2,3)",
        "c3_support_in_cayley_chart": True,
        "c3_support_polynomial_roots_exact": roots_exact,
        "c3_weight_interpolant_exact": weights_exact,
        "c6_support_size": len(c6_occurrences),
        "c6_support_formula": f"binom({len(deck)}+5,6)",
        "maximum_ordered_c6_fiber_count": max(c6_occurrences.values()),
        "all_sample_counts_below_field_prime": all(
            row["direct_ordered_pair_count"] < field.p
            for row in all_queries
        ),
        "positive_queries": positive_queries,
        "empty_queries": empty_queries,
        "all_gcd_and_trace_queries_exact": all_exact,
        "all_positive_queries_match_direct_c6_count": all_positive_exact,
        "all_empty_queries_rejected": all_empty_exact,
        "candidate_root_oracle_consumed": False,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def standard_route_cost() -> dict[str, Any]:
    return {
        "c3_support_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "explicit_support_polynomial_state_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "one_mobius_pullback_coefficient_output_exponent_B": (
            fraction_record(Fraction(9, 4))
        ),
        "one_gcd_trace_query_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "a6_batch_size_exponent_B": fraction_record(Fraction(1, 2)),
        "one_full_six_factor_count_exponent_B": fraction_record(
            Fraction(11, 4)
        ),
        "one_full_six_factor_count_exponent_N": fraction_record(
            Fraction(11, 20)
        ),
        "fresh_work_cap_exponent_B": fraction_record(Fraction(5, 4)),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_route_inside_setup_cap": True,
        "standard_one_c6_query_inside_fresh_cap": False,
        "standard_a6_batch_inside_rho": False,
        "scope": (
            "This charges the explicit C3 support polynomial, its dense "
            "Mobius pullback, Euclidean gcd, and quotient trace. It is not "
            "a lower bound for implicit modular resultants, target-batched "
            "remainders, arithmetic circuits, RAM, or cell probes."
        ),
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        actual_control(curve, offset)
        for curve in R144.R82.FAMILIES
        for offset in (0, 1)
    ]
    all_charts = all(
        row["c3_support_in_cayley_chart"] for row in actual
    )
    all_polynomials = all(
        row["c3_support_polynomial_roots_exact"]
        and row["c3_weight_interpolant_exact"]
        for row in actual
    )
    all_queries = all(
        row["all_gcd_and_trace_queries_exact"] for row in actual
    )
    all_positive = all(
        row["all_positive_queries_match_direct_c6_count"]
        for row in actual
    )
    all_empty = all(
        row["all_empty_queries_rejected"] for row in actual
    )
    all_integer_lifts = all(
        row["all_sample_counts_below_field_prime"] for row in actual
    )
    cost_boundary = standard_route_cost()

    controls = {
        "schema": (
            "p1553.m6_weighted_c3_mobius_gcd_trace."
            "controls.r145.v1"
        ),
        "actual_control_count": len(actual),
        "all_c3_supports_in_cayley_chart": all_charts,
        "all_support_polynomials_and_weight_interpolants_exact": (
            all_polynomials
        ),
        "all_gcd_intersections_and_quotient_traces_exact": all_queries,
        "all_positive_queries_match_direct_c6_counts": all_positive,
        "all_empty_queries_rejected": all_empty,
        "all_sample_integer_lifts_unambiguous": all_integer_lifts,
        "candidate_root_oracle_consumed": False,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_weighted_c3_mobius_gcd_trace."
            "frozen.r145.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "cayley_product_law": {
            "product": "x oplus y=(x+y)/(1+d*x*y)",
            "target_partner": "y=(tau-x)/(1-d*tau*x)",
            "chart_point_omitted": "-1 in the norm-one torus",
        },
        "weighted_divisor_query": {
            "support_polynomial": "P(X)=prod_{x in supp(C3)}(X-x)",
            "weight_interpolant": "W(x)=mu_C3(x) on supp(C3)",
            "mobius_pullback": (
                "P_tau(X)=(1-d*tau*X)^n "
                "P((tau-X)/(1-d*tau*X))"
            ),
            "common_divisor": "G_tau=gcd(P,P_tau)",
            "weighted_count": (
                "(mu_C3*mu_C3)(tau) is the quotient-algebra trace over "
                "F_p[X]/G_tau of W(X)*W((tau-X)/(1-d*tau*X))."
            ),
            "root_extraction_required": False,
        },
        "standard_route_cost": cost_boundary,
        "required_open_outputs": {
            "sublinear_implicit_mobius_gcd_trace_query": "open",
            "a6_target_batched_modular_resultant": "open",
            "offline_online_transposed_weight_derivatives": "open",
            "generic_integer_lift": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_weighted_c3_mobius_gcd_trace."
            "replay.r145.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "c3_support_degree": row["c3_support_degree"],
                "positive_queries": row["positive_queries"],
                "empty_queries": row["empty_queries"],
            }
            for row in actual
        ],
        "all_gcd_intersections_and_quotient_traces_exact": all_queries,
        "all_positive_queries_match_direct_c6_counts": all_positive,
        "all_empty_queries_rejected": all_empty,
    }

    cost = {
        "schema": (
            "p1553.m6_weighted_c3_mobius_gcd_trace."
            "cost.r145.v1"
        ),
        "standard_route_cost": cost_boundary,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "finite_direct_support_intersection_charged_to_candidate": False,
        "explicit_mobius_gcd_trace_route_supplied": True,
        "inside_cap_weighted_count_index_supplied": False,
        "offline_online_transposed_derivative_index_supplied": False,
        "structured_rank_density_theorem_supplied": False,
        "factor_log_cost_supplied_unconditionally": False,
        "identical_descent_cost_supplied_unconditionally": False,
        "unconditional_total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.m6_weighted_c3_mobius_gcd_trace."
            "logs_descent.r145.v1"
        ),
        "finite_weighted_c6_counts_exact": all_positive and all_empty,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "fourteen_source_bindings_verified": len(actual_bindings) == 14,
        "r144_source_free_reduction_inherited": True,
        "cayley_product_and_partner_laws_derived": True,
        "weighted_c3_support_divisor_frozen": True,
        "mobius_pullback_common_divisor_identity_derived": True,
        "root_free_quotient_trace_count_identity_derived": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_c3_supports_in_chart": all_charts,
        "all_actual_support_and_weight_polynomials_exact": all_polynomials,
        "all_actual_gcd_and_trace_queries_exact": all_queries,
        "all_actual_positive_counts_exact": all_positive,
        "all_actual_empty_counts_exact": all_empty,
        "all_sample_integer_lifts_unambiguous": all_integer_lifts,
        "candidate_dlp_and_root_oracles_avoided": True,
        "standard_route_coefficient_cost_charged": True,
        "standard_route_a6_batch_exceeds_rho": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "sublinear_implicit_mobius_gcd_trace_query_complete": False,
        "a6_target_batched_modular_resultant_complete": False,
        "offline_online_transposed_weight_derivatives_complete": False,
        "generic_integer_lift_complete": False,
        "structured_rank_and_density_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Leave dense per-target Mobius pullbacks. Construct one implicit "
        "target-batched modular resultant or remainder operator for the "
        "weighted C3 divisor. It must consume the compact C deck and an "
        "A6-shifted target batch without emitting the B^(9/4) transformed "
        "polynomial per target, compute exact integer C6 counts and reusable "
        "weight transposes inside B^(9/4+o(1)) setup and B^(5/4+o(1)) "
        "fresh work, and preserve the R144 known-RHS rank, factor-log, and "
        "shifted-descent identities. It may use no DLP, root, gcd, count, "
        "marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "WEIGHTED_C6_COUNT_IS_EXACT_C3_DIVISOR_MOBIUS_GCD_QUOTIENT_"
            "TRACE__ROOT_FREE_POSITIVE_AND_EMPTY_REPLAY_ON_EIGHT_CONTROLS__"
            "EXPLICIT_C3_POLYNOMIAL_DEGREE_B9O4_FITS_SETUP__ONE_PULLBACK_"
            "FAILS_FRESH_CAP__A6_BATCH_B11O4_EXCEEDS_RHO__IMPLICIT_BATCHED_"
            "MODULAR_RESULTANT_OPEN__NO_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether the standard explicit C3 divisor can realize "
            "the R144 weighted six-factor count and marginals inside the "
            "campaign setup, query, and rho caps."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "cayley_product_law": frozen["cayley_product_law"],
            "weighted_divisor_query": frozen["weighted_divisor_query"],
            "standard_route_cost": cost_boundary,
            "finite_actual_result": (
                "Every sampled positive and empty query on all eight actual "
                "controls has exact common-divisor degree and exact weighted "
                "quotient trace, without root extraction."
            ),
            "literature_novelty": "unverified",
            "scope": cost_boundary["scope"],
        },
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "weighted_gcd_trace_identity_admitted": True,
            "standard_explicit_route_negative_admitted": True,
            "implicit_batched_count_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=DEFAULT_CONTROLS,
    )
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    print(
        "obligations="
        f"{admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
