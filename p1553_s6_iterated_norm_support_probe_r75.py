#!/usr/bin/env python3
"""Measure first-stage coefficient growth in an S6 iterated-norm grammar."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from typing import Any, Iterable


SCHEMA = "p1553.s6_iterated_norm_support_probe.r75.v1"
PREFIX_SIZES = (2, 3, 4, 6, 8)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R74_REPORT = pathlib.Path(
    "p1553_s6_residual_decision_diagram_probe_report_r74.json"
)
R74_REPORT_SHA256 = (
    "1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740"
)
R73_REPORT = pathlib.Path(
    "p1553_resultant_valuation_trace_grammar_report_r73.json"
)
R73_REPORT_SHA256 = (
    "00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915"
)


def load_r72() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_s6_centered_carry_rank_minor_probe_r72.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r72_for_r75", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R72 controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R72 = load_r72()
Monomial = tuple[int, int, int]
Polynomial = dict[Monomial, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant(value: int, modulus: int) -> Polynomial:
    value %= modulus
    return {} if value == 0 else {(0, 0, 0): value}


def variable(axis: int) -> Polynomial:
    exponent = [0, 0, 0]
    exponent[axis] = 1
    return {tuple(exponent): 1}


def polynomial_add(
    left: Polynomial,
    right: Polynomial,
    modulus: int,
    right_scale: int = 1,
) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        value = (
            result.get(monomial, 0) + right_scale * coefficient
        ) % modulus
        if value:
            result[monomial] = value
        else:
            result.pop(monomial, None)
    return result


def polynomial_scale(
    polynomial: Polynomial,
    scalar: int,
    modulus: int,
) -> Polynomial:
    scalar %= modulus
    if scalar == 0:
        return {}
    return {
        monomial: coefficient * scalar % modulus
        for monomial, coefficient in polynomial.items()
        if coefficient * scalar % modulus
    }


def polynomial_multiply(
    left: Polynomial,
    right: Polynomial,
    modulus: int,
) -> Polynomial:
    result: Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(
                left_degree + right_degree
                for left_degree, right_degree in zip(
                    left_monomial,
                    right_monomial,
                )
            )
            value = (
                result.get(monomial, 0)
                + left_coefficient * right_coefficient
            ) % modulus
            if value:
                result[monomial] = value
            else:
                result.pop(monomial, None)
    return result


def polynomial_power_two(
    polynomial: Polynomial,
    modulus: int,
) -> Polynomial:
    return polynomial_multiply(polynomial, polynomial, modulus)


def s4_symbolic_first_two(
    x_third: int,
    curve: dict[str, Any],
) -> Polynomial:
    """Return S4(X,Y,x_third,Z) over the curve field."""

    modulus = curve["field_prime"]
    curve_a = curve["curve_a"] % modulus
    curve_b = curve["curve_b"] % modulus
    x_third %= modulus
    x_variable = variable(0)
    y_variable = variable(1)
    z_variable = variable(2)
    one = constant(1, modulus)

    x_minus_y = polynomial_add(
        x_variable,
        y_variable,
        modulus,
        right_scale=-1,
    )
    s3_a = polynomial_power_two(x_minus_y, modulus)

    x_plus_y = polynomial_add(x_variable, y_variable, modulus)
    xy = polynomial_multiply(x_variable, y_variable, modulus)
    xy_plus_a = polynomial_add(xy, constant(curve_a, modulus), modulus)
    s3_b_inner = polynomial_multiply(x_plus_y, xy_plus_a, modulus)
    s3_b_inner = polynomial_add(
        s3_b_inner,
        constant(2 * curve_b, modulus),
        modulus,
    )
    s3_b = polynomial_scale(s3_b_inner, -2, modulus)

    xy_minus_a = polynomial_add(
        xy,
        constant(curve_a, modulus),
        modulus,
        right_scale=-1,
    )
    s3_c = polynomial_power_two(xy_minus_a, modulus)
    s3_c = polynomial_add(
        s3_c,
        polynomial_scale(x_plus_y, 4 * curve_b, modulus),
        modulus,
        right_scale=-1,
    )

    z_squared = polynomial_power_two(z_variable, modulus)
    d_polynomial = polynomial_add(
        z_squared,
        polynomial_scale(z_variable, -2 * x_third, modulus),
        modulus,
    )
    d_polynomial = polynomial_add(
        d_polynomial,
        constant(x_third * x_third, modulus),
        modulus,
    )

    e_polynomial = polynomial_scale(
        z_squared,
        -2 * x_third,
        modulus,
    )
    e_polynomial = polynomial_add(
        e_polynomial,
        polynomial_scale(
            z_variable,
            -2 * (x_third * x_third + curve_a),
            modulus,
        ),
        modulus,
    )
    e_polynomial = polynomial_add(
        e_polynomial,
        constant(
            -2 * (curve_a * x_third + 2 * curve_b),
            modulus,
        ),
        modulus,
    )

    f_polynomial = polynomial_scale(
        z_squared,
        x_third * x_third,
        modulus,
    )
    f_polynomial = polynomial_add(
        f_polynomial,
        polynomial_scale(
            z_variable,
            -2 * curve_a * x_third - 4 * curve_b,
            modulus,
        ),
        modulus,
    )
    f_polynomial = polynomial_add(
        f_polynomial,
        constant(
            curve_a * curve_a - 4 * curve_b * x_third,
            modulus,
        ),
        modulus,
    )

    first = polynomial_add(
        polynomial_multiply(f_polynomial, s3_a, modulus),
        polynomial_multiply(d_polynomial, s3_c, modulus),
        modulus,
        right_scale=-1,
    )
    second_left = polynomial_add(
        polynomial_multiply(e_polynomial, s3_a, modulus),
        polynomial_multiply(d_polynomial, s3_b, modulus),
        modulus,
        right_scale=-1,
    )
    second_right = polynomial_add(
        polynomial_multiply(f_polynomial, s3_b, modulus),
        polynomial_multiply(e_polynomial, s3_c, modulus),
        modulus,
        right_scale=-1,
    )
    return polynomial_add(
        polynomial_power_two(first, modulus),
        polynomial_multiply(second_left, second_right, modulus),
        modulus,
        right_scale=-1,
    )


def specialize_first_two(
    polynomial: Polynomial,
    x_value: int,
    y_value: int,
    modulus: int,
) -> list[int]:
    coefficients = [0] * 5
    for (x_degree, y_degree, z_degree), coefficient in polynomial.items():
        coefficients[z_degree] = (
            coefficients[z_degree]
            + coefficient
            * pow(x_value, x_degree, modulus)
            * pow(y_value, y_degree, modulus)
        ) % modulus
    return coefficients


def product_support_profile(
    curve: dict[str, Any],
    deck: list[Any],
    size: int,
) -> dict[str, Any]:
    modulus = curve["field_prime"]
    product = constant(1, modulus)
    factor_support_counts = []
    for index in range(size):
        factor = s4_symbolic_first_two(deck[index][0], curve)
        factor_support_counts.append(len(factor))
        product = polynomial_multiply(product, factor, modulus)
    side = 4 * size + 1
    full_cube_count = side**3
    support_count = len(product)
    return {
        "deck_size": size,
        "factor_support_counts": factor_support_counts,
        "all_factors_full_5_cube": all(
            count == 125 for count in factor_support_counts
        ),
        "product_degree_bounds": {
            "x_first": 4 * size,
            "x_second": 4 * size,
            "z": 4 * size,
        },
        "support_count": support_count,
        "full_cube_coefficient_count": full_cube_count,
        "support_is_full_cube": support_count == full_cube_count,
        "support_density": support_count / full_cube_count,
        "suffix_modulus_degree_lower_bound": size**2,
        "z_reduction_active": 4 * size >= size**2,
    }


def factor_specialization_replay(
    curve: dict[str, Any],
    decks: list[list[Any]],
) -> dict[str, Any]:
    symbolic = s4_symbolic_first_two(decks[2][0][0], curve)
    specialized = specialize_first_two(
        symbolic,
        decks[0][1][0],
        decks[1][2][0],
        curve["field_prime"],
    )
    direct = R72.s4_polynomial_last(
        decks[0][1][0],
        decks[1][2][0],
        decks[2][0][0],
        curve["curve_a"],
        curve["curve_b"],
        curve["field_prime"],
    )
    return {
        "symbolic_support_count": len(symbolic),
        "specialization_matches_r72": specialized == direct,
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    decks, _ = R72.public_decks_and_targets(curve)
    return {
        "family_id": curve["family_id"],
        "field_bits": curve["field_prime"].bit_length(),
        "specialization_replay": factor_specialization_replay(curve, decks),
        "prefixes": [
            product_support_profile(curve, decks[2], size)
            for size in prefix_sizes
        ],
    }


def build_report() -> dict[str, Any]:
    families = [probe_curve(dict(curve)) for curve in R72.CURVES]
    prefixes = [
        prefix for family in families for prefix in family["prefixes"]
    ]
    large_prefixes = [
        row for row in prefixes if not row["z_reduction_active"]
    ]
    return {
        "schema": SCHEMA,
        "classification": (
            "S6_FIRST_ITERATED_NORM_HAS_FULL_THREE_DIMENSIONAL_SUPPORT"
        ),
        "source_bindings": {
            "r74_residual_decision_diagram": {
                "path": str(R74_REPORT),
                "sha256": R74_REPORT_SHA256,
            },
            "r73_resultant_valuation_grammar": {
                "path": str(R73_REPORT),
                "sha256": R73_REPORT_SHA256,
            },
        },
        "grammar": {
            "grammar_id": "s6_iterated_norm_mod_suffix_v1",
            "first_stage": (
                "product over a3 of S4(X1,X2,a3,Z), then reduce modulo "
                "the suffix support polynomial"
            ),
            "representation": "expanded trivariate coefficient dictionary",
            "support_adaptive": False,
            "not_product_resultant_in_query_parameter": True,
            "not_residual_key_table": True,
        },
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "prefix_instance_count": len(prefixes),
            "all_symbolic_specializations_match_r72": all(
                family["specialization_replay"][
                    "specialization_matches_r72"
                ]
                for family in families
            ),
            "all_single_factors_full_5_cube": all(
                row["all_factors_full_5_cube"] for row in prefixes
            ),
            "all_products_full_cube": all(
                row["support_is_full_cube"] for row in prefixes
            ),
            "all_B_ge_6_suffix_reductions_inactive_at_first_stage": all(
                not row["z_reduction_active"]
                for row in prefixes
                if row["deck_size"] >= 6
            ),
            "maximum_support_count": max(
                row["support_count"] for row in prefixes
            ),
            "large_prefix_count": len(large_prefixes),
        },
        "cost_ledger": {
            "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_workspace_cap_exponent_B": (
                ONLINE_WORKSPACE_CAP_EXPONENT
            ),
            "first_stage_degree_each_axis": "4B",
            "first_stage_full_support_formula": "(4B+1)^3",
            "first_stage_state_exponent_B": 3.0,
            "first_stage_work_lower_bound_in_frozen_representation_B": 3.0,
            "suffix_polynomial_degree_B": 2.0,
            "suffix_reduction_boundary": (
                "For B>4, first-stage Z degree 4B is below the degree-B^2 "
                "suffix modulus, so reduction changes no coefficient."
            ),
            "lane_inside_caps": False,
            "not_an_arithmetic_circuit_lower_bound": True,
        },
        "admission": {
            "passed_obligation_count": 4,
            "obligation_count": 9,
            "lane_admitted": False,
            "failures": [
                "first unary norm fills (4B+1)^3 coefficient cube",
                "first-stage state exponent 3 exceeds 9/4",
                "suffix reduction is inactive before the B^3 body appears",
                "R73 multiplicity, source, and dyadic controls are not reached",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "next_action": (
            "Leave expanded iterated norms. Freeze a transposed scalar-functional "
            "grammar that computes only the suffix gcd or zero certificate from "
            "the factored unary S4 norm, without representing its trivariate "
            "coefficient body. Require R73 multiplicity and source replay."
        ),
        "disposition": (
            "REJECT_EXPANDED_S6_ITERATED_NORM_MOD_SUFFIX_GRAMMAR_ONLY__"
            "FOUR_STANDARD_CURVES__B2_3_4_6_8__EVERY_S4_FACTOR_FULL_5_CUBE__"
            "EVERY_FIRST_NORM_PRODUCT_FULL_4B_PLUS1_CUBED_SUPPORT__B3_STATE__"
            "SUFFIX_REDUCTION_INACTIVE_FOR_B_GT4__NO_CIRCUIT_LOWER_BOUND__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_s6_iterated_norm_support_probe_report_r75.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = report["aggregate"]
    print(
        f"families={aggregate['family_count']} "
        f"full_products={aggregate['all_products_full_cube']} "
        f"max_support={aggregate['maximum_support_count']} "
        f"lane_admitted={report['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
