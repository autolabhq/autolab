#!/usr/bin/env python3
"""Test a tangent-aware signed dual-Chow pushforward for the M6 locator."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_confluent_signed_dual_chow_pushforward.r174.v1"

R173_PRODUCER = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_r173.py"
R173_REPORT = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_report_r173.json"
R173_FROZEN = ROOT / "frozen_m6_s3_determinantal_transfer_noncommutativity.json"
R173_COST = ROOT / "m6_s3_determinantal_transfer_noncommutativity_cost_ledger.json"
R173_REPLAY = ROOT / "m6_s3_determinantal_transfer_noncommutativity_replay.json"
R173_CONTROLS = ROOT / "m6_s3_determinantal_transfer_noncommutativity_controls.json"
R173_TRANSFER = ROOT / "s3_determinantal_transfer_noncommutativity_r173.json"
R173_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_s3_determinantal_transfer_noncommutativity_probe_r173.py"
R173_GATE = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_gate_r173.md"
R173_PARENT = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_parent_report_r173.yaml"
SEMAEV_PAPER = ROOT / "references/semaev_summation_polynomials_2004_031.ps"
MULTIPOINT_PAPER = ROOT / "references/bhargava_ghosh_guo_kumar_umans_multipoint_2205.00342v1.pdf"

SOURCE_BINDINGS = (
    ("r173_producer", R173_PRODUCER, "dd63e3404bb9cd6cf24de48234b8240e6a7918eefb200bd76292c2e410119f67"),
    ("r173_report", R173_REPORT, "2a9d6fafbee9c3c8e3d9323af291476bb662de075090cb497ad9e8555dafe75c"),
    ("r173_frozen", R173_FROZEN, "2032a4ef8c743de4db64d0e49bf8763fd23270120f26bb937629e9ef60410533"),
    ("r173_cost", R173_COST, "7a0bb4812e6e7facbe4528e4717e34dfdd6b08f4d2b19af50f72ae921b490588"),
    ("r173_replay", R173_REPLAY, "57840c79d02126b9a4510273465df74fc0316c94913d2739a131ca0a880f1c4b"),
    ("r173_controls", R173_CONTROLS, "c53431049b58348471d8cf8d5797c7e9e7fcf52f0fc0bdd05e3631f26e1a2d64"),
    ("r173_transfer", R173_TRANSFER, "4bd35b11d490b12445c8c98bd568e40d91259609b7a2976984c3f76c9454004c"),
    ("r173_test", R173_TEST, "2f0b521d7e78901247119376a167951a4805e202976576aafe729f26e0107740"),
    ("r173_gate", R173_GATE, "437a665efca2d9c0903e801f11b44f0c69f1d0a14a64ae4b69603593049442f7"),
    ("r173_parent", R173_PARENT, "c57613845a891c5a1f917bd0721a7b7041676c6abc289c194f92d083022a2a84"),
    ("semaev_2004", SEMAEV_PAPER, "991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df"),
    ("bhargava_et_al_multipoint_2022", MULTIPOINT_PAPER, "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_report_r174.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_confluent_signed_dual_chow_pushforward.json"
DEFAULT_COST = ROOT / "m6_confluent_signed_dual_chow_pushforward_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_confluent_signed_dual_chow_pushforward_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_confluent_signed_dual_chow_pushforward_controls.json"
DEFAULT_CHOW = ROOT / "confluent_signed_dual_chow_pushforward_r174.json"

Homogeneous3 = dict[tuple[int, int, int], int]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R173 = load_module("p1553_r173_for_r174", R173_PRODUCER)
R172 = R173.R172
R166 = R172.R167.R166
R164 = R166.R164
R161 = R172.R161


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R174 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def homogeneous_mul_linear(
    poly: Homogeneous3,
    alpha_coefficient: int,
    beta_coefficient: int,
    gamma_coefficient: int,
    prime: int,
) -> Homogeneous3:
    result: Homogeneous3 = {}
    coefficients = (
        (alpha_coefficient % prime, (1, 0, 0)),
        (beta_coefficient % prime, (0, 1, 0)),
        (gamma_coefficient % prime, (0, 0, 1)),
    )
    for exponent, value in poly.items():
        for coefficient, increment in coefficients:
            key = tuple(
                exponent[index] + increment[index] for index in range(3)
            )
            result[key] = (result.get(key, 0) + value * coefficient) % prime
    return {key: value for key, value in result.items() if value}


def dual_chow(
    points: list[tuple[int, int]], prime: int
) -> Homogeneous3:
    poly: Homogeneous3 = {(0, 0, 0): 1}
    for x_value, y_value in points:
        poly = homogeneous_mul_linear(
            poly, int(x_value), int(y_value), 1, prime
        )
    return poly


def homogeneous_eval(
    poly: Homogeneous3,
    alpha: int,
    beta: int,
    gamma: int,
    prime: int,
) -> int:
    return sum(
        coefficient
        * pow(alpha, alpha_degree, prime)
        * pow(beta, beta_degree, prime)
        * pow(gamma, gamma_degree, prime)
        for (alpha_degree, beta_degree, gamma_degree), coefficient in poly.items()
    ) % prime


def homogeneous_derivative_gamma(poly: Homogeneous3, prime: int) -> Homogeneous3:
    return {
        (alpha_degree, beta_degree, gamma_degree - 1): (
            gamma_degree * coefficient % prime
        )
        for (alpha_degree, beta_degree, gamma_degree), coefficient in poly.items()
        if gamma_degree and gamma_degree * coefficient % prime
    }


def homogeneous_record(poly: Homogeneous3, degree: int) -> dict[str, Any]:
    terms = [
        {
            "alpha_degree": exponent[0],
            "beta_degree": exponent[1],
            "gamma_degree": exponent[2],
            "coefficient": poly[exponent],
        }
        for exponent in sorted(poly)
    ]
    slot_count = math.comb(degree + 2, 2)
    return {
        "homogeneous_degree": degree,
        "coefficient_slot_count": slot_count,
        "nonzero_coefficient_count": len(poly),
        "coefficient_density": len(poly) / slot_count,
        "terms": terms,
        "term_sha256": sha256_json(terms),
    }


def poly_trim(poly: list[int], prime: int) -> list[int]:
    result = [coefficient % prime for coefficient in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def poly_derivative(poly: list[int], prime: int) -> list[int]:
    result = [index * coefficient % prime for index, coefficient in enumerate(poly)][1:]
    return poly_trim(result or [0], prime)


def divided_difference_coefficients(
    poly: list[int], x_value: int, prime: int
) -> list[int]:
    degree = R161.poly_degree(poly)
    if degree <= 0:
        return [0]
    result = [0] * degree
    for source_degree in range(1, degree + 1):
        coefficient = poly[source_degree] % prime
        x_power = 1
        for z_degree in range(source_degree - 1, -1, -1):
            result[z_degree] = (
                result[z_degree] + coefficient * x_power
            ) % prime
            x_power = x_power * x_value % prime
    return poly_trim(result, prime)


def deflated_line_polynomial(
    x_value: int,
    y_value: int,
    target: tuple[int, int],
    v_poly: list[int],
    prime: int,
) -> list[int]:
    u_value, target_y = target
    divided = divided_difference_coefficients(v_poly, x_value, prime)
    result = [coefficient * (x_value - u_value) % prime for coefficient in divided]
    result[0] = (result[0] - y_value - target_y) % prime
    return poly_trim(result, prime)


def raw_signed_line_value(
    left: tuple[int, int],
    right: tuple[int, int],
    target: tuple[int, int],
    prime: int,
) -> int:
    x_value, y_value = left
    z_value, right_y = right
    u_value, target_y = target
    return (
        (x_value - u_value) * right_y
        + (u_value - z_value) * y_value
        + (x_value - z_value) * target_y
    ) % prime


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r166 = R166.finite_control(curve, seed)
    _, divisor, target_records = R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    n = len(selected)
    target_count = len(targets)
    u_poly = divisor["u"]
    v_poly = divisor["v"]
    u_derivative = poly_derivative(u_poly, prime)
    target_chow = dual_chow(targets, prime)
    selected_chow = dual_chow(
        [(int(point[1]), int(point[0])) for point in selected], prime
    )
    selected_chow_gamma = homogeneous_derivative_gamma(selected_chow, prime)

    line_factor_identity_count = 0
    line_zero_biconditional_count = 0
    tangent_factor_count = 0
    tangent_zero_count = 0
    secant_zero_count = 0
    deflated_norm_identity_count = 0
    selected_chow_confluent_identity_count = 0
    target_chow_identity_count = 0
    aggregate_values = []
    aggregate_target_chow_values = []
    rows = []
    for left in selected:
        x_value, y_value = left
        divided = divided_difference_coefficients(v_poly, x_value, prime)
        aggregate = 1
        for target in targets:
            line_poly = deflated_line_polynomial(
                x_value, y_value, target, v_poly, prime
            )
            norm = 1
            for right in selected:
                z_value = int(right[0])
                if z_value == x_value:
                    tangent_factor_count += 1
                    tangent = (
                        (x_value - int(target[0]))
                        * (3 * x_value * x_value + int(curve["curve_a"]))
                        - 2 * y_value * (y_value + int(target[1]))
                    ) % prime
                    factor = tangent
                    tangent_zero_count += int(factor == 0)
                else:
                    factor = R161.poly_eval(line_poly, z_value, prime)
                    raw = raw_signed_line_value(left, right, target, prime)
                    if factor * (z_value - x_value) % prime != raw:
                        raise AssertionError("secant divided-difference value failed")
                    secant_zero_count += int(factor == 0)
                norm = norm * factor % prime
                expected_zero = (
                    tuple(R161.R70.add(left, right, curve) or ()) == target
                )
                if (factor == 0) != expected_zero:
                    raise AssertionError(
                        "signed line zero biconditional failed: "
                        f"left={left} right={right} target={target} "
                        f"factor={factor} expected_zero={expected_zero}"
                    )
                line_factor_identity_count += 1
                line_zero_biconditional_count += 1

            deflated_norm_identity_count += 1

            alpha = (x_value - int(target[0])) % prime
            beta = (-y_value - int(target[1])) % prime
            gamma = (
                int(target[0]) * y_value + x_value * int(target[1])
            ) % prime
            gamma_derivative = homogeneous_eval(
                selected_chow_gamma, alpha, beta, gamma, prime
            )
            tangent = (
                (x_value - int(target[0]))
                * (3 * x_value * x_value + int(curve["curve_a"]))
                - 2 * y_value * (y_value + int(target[1]))
            ) % prime
            u_prime = R161.poly_eval(u_derivative, x_value, prime)
            if u_prime == 0:
                raise AssertionError("selected divisor is not squarefree")
            sign = prime - 1 if (n - 1) % 2 else 1
            if norm * u_prime % prime != sign * tangent * gamma_derivative % prime:
                raise AssertionError(
                    "confluent selected-Chow derivative failed: "
                    f"left={left} target={target} norm={norm} u_prime={u_prime} "
                    f"sign={sign} tangent={tangent} gamma_derivative={gamma_derivative}"
                )
            selected_chow_confluent_identity_count += 1
            aggregate = aggregate * norm % prime

        target_first_aggregate = 1
        for right in selected:
            z_value = int(right[0])
            if z_value == x_value:
                curve_derivative = (
                    3 * x_value * x_value + int(curve["curve_a"])
                ) % prime
                alpha = (-curve_derivative) % prime
                beta = (-2 * y_value) % prime
                gamma = (
                    x_value * curve_derivative - 2 * y_value * y_value
                ) % prime
            else:
                divided_value = R161.poly_eval(divided, z_value, prime)
                alpha = (-divided_value) % prime
                beta = prime - 1
                gamma = (x_value * divided_value - y_value) % prime
            target_product = homogeneous_eval(
                target_chow, alpha, beta, gamma, prime
            )
            direct_target_product = 1
            for target in targets:
                if z_value == x_value:
                    factor = (
                        (x_value - int(target[0]))
                        * (3 * x_value * x_value + int(curve["curve_a"]))
                        - 2 * y_value * (y_value + int(target[1]))
                    )
                else:
                    factor = (
                        (x_value - int(target[0])) * divided_value
                        - y_value
                        - int(target[1])
                    )
                direct_target_product = direct_target_product * factor % prime
            if target_product != direct_target_product:
                raise AssertionError("target dual-Chow evaluation failed")
            target_chow_identity_count += 1
            target_first_aggregate = target_first_aggregate * target_product % prime
        if target_first_aggregate != aggregate:
            raise AssertionError("target-first and target-last aggregates differ")
        aggregate_values.append(aggregate)
        aggregate_target_chow_values.append(target_first_aggregate)
        rows.append(
            {
                "selected_endpoint": [int(x_value), int(y_value)],
                "confluent_signed_aggregate": aggregate,
                "target_first_chow_aggregate": target_first_aggregate,
                "candidate": aggregate == 0,
            }
        )

    candidate_roots = sorted(
        int(row["selected_endpoint"][0]) for row in rows if row["candidate"]
    )
    if candidate_roots != r166["verified_roots"]:
        raise AssertionError("confluent signed roots differ from R166 verification")
    selector = R161.interpolate(
        [
            (int(point[0]), int(value))
            for point, value in zip(selected, aggregate_values)
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(u_poly, selector, prime)
    factor_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    if factor_roots != candidate_roots:
        raise AssertionError("confluent candidate gcd lost roots")

    target_record = homogeneous_record(target_chow, target_count)
    selected_record = homogeneous_record(selected_chow, n)
    return {
        "control_id": f"{curve['family_id']}_confluent_dual_chow_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": n,
        "retained_target_count": target_count,
        "line_factor_identity_count": line_factor_identity_count,
        "line_zero_biconditional_count": line_zero_biconditional_count,
        "tangent_factor_count": tangent_factor_count,
        "tangent_zero_count": tangent_zero_count,
        "secant_zero_count": secant_zero_count,
        "deflated_norm_identity_count": deflated_norm_identity_count,
        "selected_chow_confluent_identity_count": selected_chow_confluent_identity_count,
        "target_chow_identity_count": target_chow_identity_count,
        "all_signed_line_zero_biconditionals_exact": True,
        "all_deflated_line_norms_exact": True,
        "all_selected_chow_confluent_derivatives_exact": True,
        "all_target_dual_chow_evaluations_exact": True,
        "target_first_and_target_last_aggregates_equal": True,
        "candidate_roots": candidate_roots,
        "r166_verified_roots": r166["verified_roots"],
        "candidate_roots_match_r166_verified_roots": True,
        "candidate_root_count": len(candidate_roots),
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "selector_sha256": sha256_json(selector),
        "target_dual_chow": target_record,
        "selected_dual_chow": selected_record,
        "row_transcript_sha256": sha256_json(rows),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_chow_density_receives_asymptotic_credit": False,
        "finite_candidate_count_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "signed_collinearity": (
            "For P=(X,V(X)), Q=(Z,V(Z)), and T=(u,v), the determinant of "
            "P,Q,-T is L=(X-u)V(Z)+(u-Z)V(X)+(X-Z)v. Away from Q=P, "
            "L=0 exactly when P+Q=T."
        ),
        "confluent_deflation": (
            "Because L(P,P,-T)=0 identically, the off-diagonal chart is "
            "L/(Z-X)=(X-u)Delta_V(X,Z)-(V(X)+v), with "
            "Delta_V=(V(Z)-V(X))/(Z-X). The interpolant derivative V'(X) "
            "is not the curve tangent. On the diagonal Q=P it must be replaced "
            "by the scaled geometric tangent (X-u)(3X^2+a)-2V(X)(V(X)+v)."
        ),
        "target_dual_chow": (
            "Let C_T(alpha,beta,gamma)=product_T(alpha*u_T+beta*v_T+gamma). "
            "Off the diagonal, the target product is "
            "C_T(-Delta_V,-1,X*Delta_V-V(X)). On Q=P it is "
            "C_T(-(3X^2+a),-2V(X),X(3X^2+a)-2V(X)^2). The full signed "
            "locator is the product of these two-chart target dual-Chow "
            "evaluations over the Q roots of U."
        ),
        "selected_dual_chow_derivative": (
            "Let C_S(alpha,beta,gamma)=product_Q(alpha*y_Q+beta*x_Q+gamma). "
            "At alpha=X-u, beta=-(V(X)+v), gamma=uV(X)+Xv, one factor is "
            "identically zero. The chart-corrected norm equals "
            "(-1)^(n-1)*geometric_tangent*partial_gamma(C_S)/U'(X), so the "
            "diagonal requires both a Chow derivative and the actual curve "
            "tangent rather than the derivative of the interpolated V side table."
        ),
        "candidate_semantics": (
            "The aggregate vanishes exactly when some selected Q satisfies "
            "P+Q=T for a retained target T, including Q=P through the tangent. "
            "It is signed and replays the R166 verified roots without the "
            "opposite-sign Kummer branch."
        ),
        "scope": (
            "This is an exact commutative and tangent-aware reformulation. It "
            "does not supply a sub-rho algorithm: represented target Chow state "
            "has Theta(N^2) coefficients and represented selected Chow state has "
            "Theta(n^2); keeping either factored leaves an unproved fused outer "
            "norm. These representation costs are not general lower bounds."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_confluent_signed_dual_chow_pushforward.cost.r174.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_selected_and_target_divisor_state_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "factored_target_dual_chow_state_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "represented_target_dual_chow_body_exponent_B": fraction_record(
            Fraction(5, 2)
        ),
        "selected_target_evaluation_grid_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "represented_selected_dual_chow_body_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "target_first_selected_pair_grid_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "factored_target_chow_on_selected_pair_grid_exponent_B": fraction_record(
            Fraction(23, 4)
        ),
        "represented_aggregate_output_exponent_B": fraction_record(Fraction(9, 4)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "represented_target_dual_chow_strictly_inside_rho": False,
        "selected_target_grid_inside_rho": False,
        "represented_selected_dual_chow_inside_rho": False,
        "factored_confluent_outer_norm_supplied": False,
        "finite_density_receives_lower_bound_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["all_signed_line_zero_biconditionals_exact"]
        and row["all_deflated_line_norms_exact"]
        and row["all_selected_chow_confluent_derivatives_exact"]
        and row["all_target_dual_chow_evaluations_exact"]
        and row["target_first_and_target_last_aggregates_equal"]
        and row["candidate_roots_match_r166_verified_roots"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_confluent_signed_dual_chow_pushforward.controls.r174.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_signed_line_zero_biconditionals_exact": all(
            row["all_signed_line_zero_biconditionals_exact"] for row in rows
        ),
        "all_deflated_line_norms_exact": all(
            row["all_deflated_line_norms_exact"] for row in rows
        ),
        "all_selected_chow_confluent_derivatives_exact": all(
            row["all_selected_chow_confluent_derivatives_exact"] for row in rows
        ),
        "all_target_dual_chow_evaluations_exact": all(
            row["all_target_dual_chow_evaluations_exact"] for row in rows
        ),
        "all_target_first_and_target_last_aggregates_equal": all(
            row["target_first_and_target_last_aggregates_equal"] for row in rows
        ),
        "all_candidate_roots_match_r166_verified_roots": all(
            row["candidate_roots_match_r166_verified_roots"] for row in rows
        ),
        "line_factor_identity_count": sum(
            row["line_factor_identity_count"] for row in rows
        ),
        "line_zero_biconditional_count": sum(
            row["line_zero_biconditional_count"] for row in rows
        ),
        "tangent_factor_count": sum(row["tangent_factor_count"] for row in rows),
        "tangent_zero_count": sum(row["tangent_zero_count"] for row in rows),
        "secant_zero_count": sum(row["secant_zero_count"] for row in rows),
        "deflated_norm_identity_count": sum(
            row["deflated_norm_identity_count"] for row in rows
        ),
        "selected_chow_confluent_identity_count": sum(
            row["selected_chow_confluent_identity_count"] for row in rows
        ),
        "target_chow_identity_count": sum(
            row["target_chow_identity_count"] for row in rows
        ),
        "candidate_root_count": sum(row["candidate_root_count"] for row in rows),
        "target_chow_coefficient_slot_count": sum(
            row["target_dual_chow"]["coefficient_slot_count"] for row in rows
        ),
        "target_chow_nonzero_coefficient_count": sum(
            row["target_dual_chow"]["nonzero_coefficient_count"] for row in rows
        ),
        "selected_chow_coefficient_slot_count": sum(
            row["selected_dual_chow"]["coefficient_slot_count"] for row in rows
        ),
        "selected_chow_nonzero_coefficient_count": sum(
            row["selected_dual_chow"]["nonzero_coefficient_count"] for row in rows
        ),
        "finite_chow_density_receives_asymptotic_credit": False,
        "finite_candidate_count_receives_attack_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r173_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "signed_collinearity_identity_complete": all_exact,
        "diagonal_confluent_deflation_complete": all_exact,
        "tangent_limit_complete": all_exact,
        "signed_zero_biconditional_complete": controls[
            "all_signed_line_zero_biconditionals_exact"
        ],
        "deflated_norm_identity_complete": controls[
            "all_deflated_line_norms_exact"
        ],
        "selected_chow_derivative_identity_complete": controls[
            "all_selected_chow_confluent_derivatives_exact"
        ],
        "target_dual_chow_identity_complete": controls[
            "all_target_dual_chow_evaluations_exact"
        ],
        "target_first_target_last_replay_complete": controls[
            "all_target_first_and_target_last_aggregates_equal"
        ],
        "r166_signed_root_replay_complete": controls[
            "all_candidate_roots_match_r166_verified_roots"
        ],
        "represented_target_chow_N2_cost_charged": True,
        "selected_target_nN_cost_charged": True,
        "represented_selected_chow_n2_cost_charged": True,
        "multipoint_input_contract_audited": True,
        "finite_density_scoped_without_lower_bound_credit": True,
        "factored_confluent_outer_norm_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_SIGNED_COLLINEARITY_AND_TANGENT_CONFLUENT_DEFLATION__EXACT_"
        "TARGET_DUAL_CHOW_PUSHFORWARD__TARGET_FIRST_EQUALS_TARGET_LAST__R166_"
        "VERIFIED_ROOTS_REPLAYED_WITHOUT_OPPOSITE_SIGN_BRANCH__REPRESENTED_"
        "TARGET_CHOW_N2_B5O2_AT_RHO__SELECTED_CHOW_N2_B9O2__NN_AND_N2_GRIDS_"
        "ABOVE_RHO__FACTORED_CONFLUENT_OUTER_NORM_OPEN__NO_GENERAL_LOWER_"
        "BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute a fused factored dual-Chow outer norm that accepts "
        "U,V and the N target linear factors, handles Delta_V and the tangent "
        "diagonal symbolically, and emits the signed aggregate modulo U in "
        "softly O(n+N) work below B^(5/2), without materializing N^2 or n^2 "
        "Chow coefficients, nN or n^2 query grids, target-dependent transforms, "
        "or unit-cost multipoint, norm, derivative, or resultant oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Use the R173 discriminant split to construct a commutative, signed, "
            "tangent-aware divisor pushforward and test whether standard Chow or "
            "multipoint representations stay strictly below rho."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "semaev_summation_polynomials": {
                "title": "Summation polynomials and the discrete logarithm problem on elliptic curves",
                "source": "IACR 2004/031",
                "fit": (
                    "Supplies the unsigned S3 relation. R174 retains the signed "
                    "collinearity branch and its tangent confluence explicitly."
                ),
            },
            "fast_multivariate_multipoint_evaluation": {
                "arxiv": "2205.00342",
                "fit": (
                    "The dense bivariate input size for a represented degree-N "
                    "target Chow form is Theta(N^2), already B^(5/2), and the "
                    "theorem does not accept the factored linear forms while "
                    "returning only their fused outer norm modulo arbitrary U."
                ),
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "confluent_signed_dual_chow_identity_admitted": all_exact,
            "factored_confluent_outer_norm_admitted": False,
            "lane_admitted": False,
        },
        "classification": classification,
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_confluent_signed_dual_chow_pushforward.frozen.r174.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The factored target dual-Chow form can be fused with the "
                "confluent outer norm over U in softly O(n+N) work."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route represents Theta(N^2) or Theta(n^2) Chow "
                "coefficients, emits nN or n^2 evaluation points, expands N "
                "linear factors independently in F_p[X]/U, drops the tangent "
                "diagonal, or invokes an uncharged norm or multipoint oracle."
            ),
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_confluent_signed_dual_chow_pushforward.replay.r174.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "selector_sha256": row["selector_sha256"],
                "target_chow_sha256": row["target_dual_chow"]["term_sha256"],
                "selected_chow_sha256": row["selected_dual_chow"]["term_sha256"],
                "row_transcript_sha256": row["row_transcript_sha256"],
                "candidate_roots": row["candidate_roots"],
            }
            for row in rows
        ],
    }
    chow = {
        "schema": "p1553.m6_confluent_signed_dual_chow_pushforward.chow.r174.v1",
        "signed_collinearity": theorem["signed_collinearity"],
        "confluent_deflation": theorem["confluent_deflation"],
        "target_dual_chow": theorem["target_dual_chow"],
        "selected_dual_chow_derivative": theorem[
            "selected_dual_chow_derivative"
        ],
        "controls": [
            {
                "control_id": row["control_id"],
                "target_dual_chow": row["target_dual_chow"],
                "selected_dual_chow": row["selected_dual_chow"],
                "candidate_roots": row["candidate_roots"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
            }
            for row in rows
        ],
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "chow": chow,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--chow-output", type=Path, default=DEFAULT_CHOW)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    outputs = (
        (args.report_output, bundle["report"]),
        (args.frozen_output, bundle["frozen"]),
        (args.cost_output, bundle["cost"]),
        (args.replay_output, bundle["replay"]),
        (args.controls_output, bundle["controls"]),
        (args.chow_output, bundle["chow"]),
    )
    for path, value in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
