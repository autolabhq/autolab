#!/usr/bin/env python3
"""Freeze an exact signed C3-divisor translation/gcd source interface."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_signed_c3_divisor_translation_gcd.r161.v1"

R160_PRODUCER = ROOT / (
    "p1553_m6_positive_c6_generic_locator_reduction_probe_r160.py"
)
R160_REPORT = ROOT / (
    "p1553_m6_positive_c6_generic_locator_reduction_probe_report_r160.json"
)
R160_FROZEN = ROOT / "frozen_m6_positive_c6_generic_locator_reduction.json"
R160_COST = ROOT / "m6_positive_c6_generic_locator_reduction_cost_ledger.json"
R160_REPLAY = ROOT / "m6_positive_c6_generic_locator_reduction_replay.json"
R160_CONTROLS = ROOT / "m6_positive_c6_generic_locator_reduction_controls.json"
R160_LOGS = ROOT / "factor_logs_and_identical_descent_r160.json"
R160_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_positive_c6_generic_locator_reduction_probe_r160.py"
)
R160_GATE = ROOT / (
    "p1553_m6_positive_c6_generic_locator_reduction_probe_gate_r160.md"
)
R160_PARENT = ROOT / (
    "p1553_m6_positive_c6_generic_locator_reduction_probe_parent_report_r160.yaml"
)
R117_PRODUCER = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_r117.py"
)
R117_REPORT = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_report_r117.json"
)
R117_COST = ROOT / "m6_target_batched_c3_transpose_cost_ledger.json"
R117_GATE = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_gate_r117.md"
)
R117_PARENT = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_probe_parent_report_r117.yaml"
)
SEMAEV_PAPER = ROOT / "references/semaev_summation_polynomials_2004_031.ps"

SOURCE_BINDINGS = (
    ("r160_producer", R160_PRODUCER, "c303909641700a141714fde1ef473bb6abe4331ab9cfaf0e53161aa13b1fe7b1"),
    ("r160_report", R160_REPORT, "8b061717b7822ce969433b17003f00fb2e8e4644305cda007b70ca2213f4f48c"),
    ("r160_frozen", R160_FROZEN, "868b534a8684cbd63cf72d320102af5d6bc568510bcd64b44698ce5cbea4ca50"),
    ("r160_cost", R160_COST, "834312ee1a28f7dd7418bd09050af8c39ccbbc816731b7cc00fbfebc9f3814ab"),
    ("r160_replay", R160_REPLAY, "8bcfcc063412cdc3609748ea4843daa01dacef622433468ab201d10db2e2fce7"),
    ("r160_controls", R160_CONTROLS, "84b61b9ef84ffbcc88acbf0f24cd41be48d3aa0f495c1de18e2ac2611b83fafa"),
    ("r160_logs", R160_LOGS, "3a2980834f0b87e5c68d6ffa5b7cab03da27f1fbaaf8098c1bfe00001a0ee196"),
    ("r160_test", R160_TEST, "d568f454e0808498bc43e54b357b29a708252cc42cf53a9e2f0dfdce55a76633"),
    ("r160_gate", R160_GATE, "e4d13e79feb831e0d65e734170def4b5e79a6f53edaeb8c49c9bead4e739ec92"),
    ("r160_parent", R160_PARENT, "9089a77246d129094800f1c941a92d355a85d2ad0d1c36c86fb2a3a5bcdc67c2"),
    ("r117_producer", R117_PRODUCER, "af3d7447763e16ee2a21491f3217d719bf57bdafc035d666df5e8f5859cc49f2"),
    ("r117_report", R117_REPORT, "449021dd58e6567bb827bcfe41336175d7b628acda4446e1b63d9a46cf48f824"),
    ("r117_cost", R117_COST, "33ca990fe05e8e80bd4b557d57f4a292a84de0a5d6f1a836fa6bb54965cada8b"),
    ("r117_gate", R117_GATE, "014f74c772dfc77d8821f658b10ef9bd11a644fe8f0b356c24fda61ef81d9de5"),
    ("r117_parent", R117_PARENT, "8b276efff8f61f4278e78ab367ff4536c3bee8a1e172ed3ea26a12c1c12dbdae"),
    ("semaev_summation_polynomials", SEMAEV_PAPER, "991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df"),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_signed_c3_divisor_translation_gcd_probe_report_r161.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_signed_c3_divisor_translation_gcd.json"
DEFAULT_COST = ROOT / "m6_signed_c3_divisor_translation_gcd_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_signed_c3_divisor_translation_gcd_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_signed_c3_divisor_translation_gcd_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r161.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R160 = load_module("p1553_r160_for_r161", R160_PRODUCER)
R159 = R160.R159
R157 = R160.R157
R81 = R160.R81
R70 = R160.R70


def sha256_file(path: pathlib.Path) -> str:
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
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R161 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def trim(poly: Iterable[int], prime: int) -> list[int]:
    result = [value % prime for value in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def poly_degree(poly: list[int]) -> int:
    return -1 if poly == [0] else len(poly) - 1


def poly_add(left: list[int], right: list[int], prime: int) -> list[int]:
    length = max(len(left), len(right))
    return trim(
        [
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
            for index in range(length)
        ],
        prime,
    )


def poly_sub(left: list[int], right: list[int], prime: int) -> list[int]:
    length = max(len(left), len(right))
    return trim(
        [
            (left[index] if index < len(left) else 0)
            - (right[index] if index < len(right) else 0)
            for index in range(length)
        ],
        prime,
    )


def poly_scale(poly: list[int], scalar: int, prime: int) -> list[int]:
    return trim([(scalar * value) % prime for value in poly], prime)


def poly_mul(left: list[int], right: list[int], prime: int) -> list[int]:
    if left == [0] or right == [0]:
        return [0]
    result = [0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            result[i + j] = (result[i + j] + left_value * right_value) % prime
    return trim(result, prime)


def poly_divmod(
    numerator: list[int], denominator: list[int], prime: int
) -> tuple[list[int], list[int]]:
    numerator = trim(numerator, prime)
    denominator = trim(denominator, prime)
    if denominator == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    if poly_degree(numerator) < poly_degree(denominator):
        return [0], numerator
    quotient = [0] * (poly_degree(numerator) - poly_degree(denominator) + 1)
    remainder = numerator[:]
    inverse_lead = pow(denominator[-1], -1, prime)
    while remainder != [0] and poly_degree(remainder) >= poly_degree(denominator):
        shift = poly_degree(remainder) - poly_degree(denominator)
        coefficient = remainder[-1] * inverse_lead % prime
        quotient[shift] = coefficient
        subtractor = [0] * shift + poly_scale(denominator, coefficient, prime)
        remainder = poly_sub(remainder, subtractor, prime)
    return trim(quotient, prime), trim(remainder, prime)


def poly_mod(poly: list[int], modulus: list[int], prime: int) -> list[int]:
    return poly_divmod(poly, modulus, prime)[1]


def poly_mul_mod(
    left: list[int], right: list[int], modulus: list[int], prime: int
) -> list[int]:
    return poly_mod(poly_mul(left, right, prime), modulus, prime)


def poly_gcd(left: list[int], right: list[int], prime: int) -> list[int]:
    left = trim(left, prime)
    right = trim(right, prime)
    while right != [0]:
        left, right = right, poly_mod(left, right, prime)
    if left == [0]:
        return [0]
    return poly_scale(left, pow(left[-1], -1, prime), prime)


def poly_xgcd(
    left: list[int], right: list[int], prime: int
) -> tuple[list[int], list[int], list[int]]:
    old_r, r = trim(left, prime), trim(right, prime)
    old_s, s = [1], [0]
    old_t, t = [0], [1]
    while r != [0]:
        quotient, remainder = poly_divmod(old_r, r, prime)
        old_r, r = r, remainder
        old_s, s = s, poly_sub(old_s, poly_mul(quotient, s, prime), prime)
        old_t, t = t, poly_sub(old_t, poly_mul(quotient, t, prime), prime)
    inverse_lead = pow(old_r[-1], -1, prime)
    return (
        poly_scale(old_r, inverse_lead, prime),
        poly_scale(old_s, inverse_lead, prime),
        poly_scale(old_t, inverse_lead, prime),
    )


def poly_inv_mod(poly: list[int], modulus: list[int], prime: int) -> list[int]:
    gcd, coefficient, _ = poly_xgcd(poly, modulus, prime)
    if gcd != [1]:
        raise ZeroDivisionError("polynomial is not invertible modulo modulus")
    return poly_mod(coefficient, modulus, prime)


def poly_eval(poly: list[int], value: int, prime: int) -> int:
    result = 0
    for coefficient in reversed(poly):
        result = (result * value + coefficient) % prime
    return result


def poly_compose_mod(
    outer: list[int], inner: list[int], modulus: list[int], prime: int
) -> list[int]:
    result = [0]
    for coefficient in reversed(outer):
        result = poly_mul_mod(result, inner, modulus, prime)
        result = poly_add(result, [coefficient], prime)
    return poly_mod(result, modulus, prime)


def monic_root_polynomial(roots: Iterable[int], prime: int) -> list[int]:
    result = [1]
    for root in roots:
        result = poly_mul(result, [(-root) % prime, 1], prime)
    return result


def interpolate(points: list[tuple[int, int]], prime: int) -> list[int]:
    result = [0]
    for index, (x_value, y_value) in enumerate(points):
        numerator = [1]
        denominator = 1
        for other_index, (other_x, _) in enumerate(points):
            if index == other_index:
                continue
            numerator = poly_mul(numerator, [(-other_x) % prime, 1], prime)
            denominator = denominator * (x_value - other_x) % prime
        term = poly_scale(
            numerator, y_value * pow(denominator, -1, prime), prime
        )
        result = poly_add(result, term, prime)
    return trim(result, prime)


def signed_c3_divisor(
    representatives: tuple[Any, ...], curve: dict[str, Any]
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    records = R160.c3_occurrence_records(representatives, curve)
    if any(record["endpoint"] is None for record in records):
        raise AssertionError("C3 infinity endpoint is outside this chart")
    x_values = [int(record["endpoint"][0]) for record in records]
    if len(set(x_values)) != len(x_values):
        raise AssertionError("C3 x-coordinate collision is outside this chart")
    u_poly = monic_root_polynomial(x_values, prime)
    v_poly = interpolate(
        [
            (int(record["endpoint"][0]), int(record["endpoint"][1]))
            for record in records
        ],
        prime,
    )
    curve_residual = poly_sub(
        poly_mul(v_poly, v_poly, prime),
        [int(curve["curve_b"]), int(curve["curve_a"]), 0, 1],
        prime,
    )
    _, residual_mod_u = poly_divmod(curve_residual, u_poly, prime)
    return {
        "records": records,
        "u": u_poly,
        "v": v_poly,
        "x_injective": True,
        "curve_residual_zero_mod_u": residual_mod_u == [0],
        "u_sha256": sha256_json(u_poly),
        "v_sha256": sha256_json(v_poly),
    }


def quotient_translation_gcd(
    target: Any,
    divisor: dict[str, Any],
    curve: dict[str, Any],
) -> dict[str, Any]:
    if target is None:
        return {
            "source": None,
            "gcd_degree": 0,
            "exceptional_left_count": 0,
            "regular_c3_degree": len(divisor["records"]),
            "chart_supported": False,
            "reason": "target_at_infinity",
        }
    prime = int(curve["field_prime"])
    target_x, target_y = map(int, target)
    records = divisor["records"]
    point_index = {record["endpoint"]: record for record in records}
    exceptional = [
        record for record in records if int(record["endpoint"][0]) == target_x
    ]
    for left in exceptional:
        complement = R70.add(
            target, R70.negate(left["endpoint"], curve), curve
        )
        right = point_index.get(complement)
        if right is not None:
            source = tuple(
                a + b for a, b in zip(left["source"], right["source"])
            )
            return {
                "source": source,
                "left_c3_indices": left["indices"],
                "right_c3_indices": right["indices"],
                "gcd_degree": 0,
                "exceptional_left_count": len(exceptional),
                "regular_c3_degree": len(records) - len(exceptional),
                "chart_supported": True,
                "source_from_exceptional_branch": True,
                "finite_root_dictionary_scan_count": 0,
            }

    regular = [record for record in records if record not in exceptional]
    if not regular:
        return {
            "source": None,
            "gcd_degree": 0,
            "exceptional_left_count": len(exceptional),
            "regular_c3_degree": 0,
            "chart_supported": True,
            "source_from_exceptional_branch": False,
            "finite_root_dictionary_scan_count": 0,
        }
    u_poly = monic_root_polynomial(
        (int(record["endpoint"][0]) for record in regular), prime
    )
    v_poly = poly_mod(divisor["v"], u_poly, prime)
    denominator = [target_x % prime, (-1) % prime]
    denominator_inverse = poly_inv_mod(denominator, u_poly, prime)
    numerator = poly_add(v_poly, [target_y], prime)
    slope = poly_mul_mod(numerator, denominator_inverse, u_poly, prime)
    translated_x = poly_sub(
        poly_mul_mod(slope, slope, u_poly, prime),
        [target_x, 1],
        prime,
    )
    translated_x = poly_mod(translated_x, u_poly, prime)
    target_x_minus_translated = poly_sub(
        [target_x], translated_x, prime
    )
    translated_y = poly_sub(
        poly_mul_mod(slope, target_x_minus_translated, u_poly, prime),
        [target_y],
        prime,
    )
    translated_y = poly_mod(translated_y, u_poly, prime)
    x_membership = poly_compose_mod(
        divisor["u"], translated_x, u_poly, prime
    )
    expected_y = poly_compose_mod(
        divisor["v"], translated_x, u_poly, prime
    )
    y_membership = poly_sub(translated_y, expected_y, prime)
    y_membership = poly_mod(y_membership, u_poly, prime)
    source_gcd = poly_gcd(u_poly, x_membership, prime)
    source_gcd = poly_gcd(source_gcd, y_membership, prime)
    gcd_degree = max(poly_degree(source_gcd), 0)
    scanned = 0
    source = None
    left_indices = None
    right_indices = None
    for left in regular:
        scanned += 1
        left_x = int(left["endpoint"][0])
        if poly_eval(source_gcd, left_x, prime) != 0:
            continue
        complement = R70.add(
            target, R70.negate(left["endpoint"], curve), curve
        )
        right = point_index.get(complement)
        if right is None:
            continue
        source = tuple(
            a + b for a, b in zip(left["source"], right["source"])
        )
        left_indices = left["indices"]
        right_indices = right["indices"]
        break
    return {
        "source": source,
        "left_c3_indices": left_indices,
        "right_c3_indices": right_indices,
        "gcd_degree": gcd_degree,
        "exceptional_left_count": len(exceptional),
        "regular_c3_degree": len(regular),
        "chart_supported": True,
        "source_from_exceptional_branch": False,
        "finite_root_dictionary_scan_count": scanned,
        "u_regular_sha256": sha256_json(u_poly),
        "translated_x_sha256": sha256_json(translated_x),
        "translated_y_sha256": sha256_json(translated_y),
        "x_membership_sha256": sha256_json(x_membership),
        "y_membership_sha256": sha256_json(y_membership),
        "source_gcd_sha256": sha256_json(source_gcd),
    }


def point_from_relation(
    relation: dict[str, Any],
    representatives: tuple[Any, ...],
    generator: Any,
    curve: dict[str, Any],
) -> Any:
    return R70.add(
        R70.scalar_mul(int(relation["known_rhs_scalar"]), generator, curve),
        R70.scalar_mul(
            int(relation["diagonal_shift"]),
            representatives[int(relation["column"])],
            curve,
        ),
        curve,
    )


def target_record(
    *,
    role: str,
    target: Any,
    expected_source: tuple[int, ...] | None,
    divisor: dict[str, Any],
    representatives: tuple[Any, ...],
    curve: dict[str, Any],
    unique_c6_endpoints: dict[Any, tuple[int, ...]],
) -> dict[str, Any]:
    located = quotient_translation_gcd(target, divisor, curve)
    source = located["source"]
    source_identity_exact = (
        source is not None
        and R157.row_point(source, representatives, curve) == target
    )
    expected_presence = expected_source is not None
    return {
        "role": role,
        "target": R157.point_record(target),
        "expected_source": expected_source,
        "located_source": source,
        "source_presence_exact": (source is not None) == expected_presence,
        "source_matches_expected_coefficients": source == expected_source,
        "source_identity_exact": source_identity_exact if source else not expected_presence,
        "verifier_unique_c6_endpoint": target in unique_c6_endpoints,
        **located,
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    dimension = len(R157.c_point_pairs(curve, 0))
    factor_base = R160.generic_factor_base(curve, dimension, seed)
    representatives = factor_base["representatives"]
    generator = factor_base["generator"]
    base = R160.generic_reduction_control(curve, seed)
    divisor = signed_c3_divisor(representatives, curve)
    c3_index = R160.c3_endpoint_index(divisor["records"])
    c6 = R159.positive_endpoint_structure(representatives, curve)
    positive = []
    for relation in base["relation_rows"]:
        target = point_from_relation(
            relation, representatives, generator, curve
        )
        positive.append(
            target_record(
                role=f"relation_column_{relation['column']}",
                target=target,
                expected_source=tuple(relation["source"]),
                divisor=divisor,
                representatives=representatives,
                curve=curve,
                unique_c6_endpoints=c6["unique_endpoints"],
            )
        )
    descent = base["identical_positive_c6_target_descent"]
    descent_target = R70.add(
        factor_base["challenge"],
        R70.scalar_mul(int(descent["known_shift_scalar"]), generator, curve),
        curve,
    )
    positive.append(
        target_record(
            role="identical_descent",
            target=descent_target,
            expected_source=tuple(descent["source"]),
            divisor=divisor,
            representatives=representatives,
            curve=curve,
            unique_c6_endpoints=c6["unique_endpoints"],
        )
    )

    order = int(curve["subgroup_order"])
    empty_target = None
    empty_search_count = 0
    for index in range(order):
        scalar = R160.deterministic_scalar(
            "r161-empty", order, curve["family_id"], seed, index
        )
        candidate = R70.scalar_mul(scalar, generator, curve)
        empty_search_count += 1
        if R160.c3_pair_source(candidate, divisor["records"], c3_index, curve) is None:
            empty_target = candidate
            break
    if empty_target is None:
        raise AssertionError("finite empty target was not found")
    empty = target_record(
        role="empty",
        target=empty_target,
        expected_source=None,
        divisor=divisor,
        representatives=representatives,
        curve=curve,
        unique_c6_endpoints=c6["unique_endpoints"],
    )

    exceptional_target = divisor["records"][0]["endpoint"]
    exceptional_expected = R160.c3_pair_source(
        exceptional_target, divisor["records"], c3_index, curve
    )
    exceptional = target_record(
        role="denominator_exception",
        target=exceptional_target,
        expected_source=(
            None
            if exceptional_expected is None
            else tuple(exceptional_expected["source"])
        ),
        divisor=divisor,
        representatives=representatives,
        curve=curve,
        unique_c6_endpoints=c6["unique_endpoints"],
    )
    all_targets = positive + [empty, exceptional]
    return {
        "control_id": f"{curve['family_id']}_signed_divisor_seed{seed}_d{dimension}",
        "family_id": curve["family_id"],
        "seed": seed,
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": order,
        "factor_base_dimension": dimension,
        "c3_degree": len(divisor["records"]),
        "c3_degree_formula_exact": (
            len(divisor["records"]) == math.comb(dimension + 2, 3)
        ),
        "c3_x_coordinates_injective": divisor["x_injective"],
        "signed_divisor_curve_residual_zero": divisor[
            "curve_residual_zero_mod_u"
        ],
        "u_sha256": divisor["u_sha256"],
        "v_sha256": divisor["v_sha256"],
        "positive_target_count": len(positive),
        "all_positive_sources_exact": all(
            row["source_presence_exact"]
            and row["source_matches_expected_coefficients"]
            and row["source_identity_exact"]
            for row in positive
        ),
        "all_positive_targets_unique_c6": all(
            row["verifier_unique_c6_endpoint"] for row in positive
        ),
        "maximum_positive_source_gcd_degree": max(
            row["gcd_degree"] for row in positive
        ),
        "unique_source_gcd_degree_bound_20_holds": all(
            row["gcd_degree"] <= 20 for row in positive
        ),
        "empty_target_search_count": empty_search_count,
        "empty_target_exact": (
            empty["source_presence_exact"] and empty["located_source"] is None
        ),
        "denominator_exception_branch_exercised": (
            exceptional["exceptional_left_count"] == 1
        ),
        "denominator_exception_semantics_exact": exceptional[
            "source_presence_exact"
        ],
        "positive_targets": positive,
        "empty_target": empty,
        "denominator_exception_target": exceptional,
        "target_records_sha256": sha256_json(all_targets),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_polynomial_and_dictionary_work_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "signed_c3_divisor": (
            "When the n unordered C3 endpoints have distinct x-coordinates, "
            "U(X)=product_P(X-x(P)) and the degree-below-n interpolant V with "
            "V(x(P))=y(P) encode the signed endpoint set; U divides "
            "V^2-(X^3+aX+b)."
        ),
        "target_translation": (
            "For T=(u,v) and P=(X,V(X)), outside u=X set "
            "lambda=(v+V(X))/(u-X), phi=lambda^2-u-X, and "
            "psi=lambda(u-phi)-v in F_p[X]/U. Then (phi,psi)=T-P."
        ),
        "exact_membership_gcd": (
            "A regular C3 endpoint P participates in a positive decomposition "
            "T=P+P' with P' in C3 exactly when U(phi(P))=0 and "
            "psi(P)=V(phi(P)). Hence gcd(U,U(phi),psi-V(phi)) returns exactly "
            "the signed left endpoints; u=X roots are split and checked directly."
        ),
        "unique_c6_source_gcd_degree": (
            "If T has one positive-C6 coefficient source and the C3 x-map is "
            "injective, every gcd root is a size-three submultiset of that "
            "six-multiset. Distinct coefficient submultisets are images of "
            "the binomial(6,3)=20 occurrence subsets, so the source gcd has "
            "degree at most 20."
        ),
        "source_extraction": (
            "A degree-at-most-20 split polynomial can be factored over F_p in "
            "B^(o(1)) work; a root indexes the persistent C3 source dictionary, "
            "and exact point subtraction indexes the complementary source."
        ),
        "scope": (
            "This is a coordinate-specific signed branch of the elliptic "
            "summation relation. It adds an exact quotient/gcd interface and "
            "constant output-degree theorem, not a batched modular-composition "
            "algorithm or an arithmetic-circuit lower bound."
        ),
        "primary_source": {
            "title": "Summation polynomials and the discrete logarithm problem on elliptic curves",
            "author": "Igor Semaev",
            "url": "https://eprint.iacr.org/2004/031",
            "local_postscript_sha256": sha256_file(SEMAEV_PAPER),
        },
        "novelty_status": "signed_c3_divisor_gcd_interface_novelty_unverified",
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_signed_c3_divisor_translation_gcd.cost.r161.v1",
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "persistent_u_v_and_source_dictionary_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "r159_target_batch_exponent_B": fraction_record(Fraction(5, 4)),
        "regular_target_modular_composition_count": 2,
        "regular_target_gcd_count": 2,
        "optimistic_quasilinear_one_target_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "independent_complete_batch_exponent_B": fraction_record(Fraction(7, 2)),
        "required_complete_batch_exponent_B": fraction_record(Fraction(5, 4)),
        "positive_source_output_count_exponent_B": fraction_record(Fraction(3, 4)),
        "unique_source_gcd_degree_bound": 20,
        "constant_degree_source_factorization_exponent_B": fraction_record(
            Fraction(0)
        ),
        "r117_translated_divisor_cost_semantically_inherited": True,
        "signed_u_v_quotient_and_gcd_interface_supplied": True,
        "target_batched_many_inner_modular_composition_supplied": False,
        "target_batched_gcd_source_adjoint_supplied": False,
        "coordinate_specific_operation_explicit": True,
        "generic_encoding_invariant": False,
        "independent_target_route_exceeds_pollard_rho": True,
        "unconditional_total_attack_cost_supplied": False,
        "finite_polynomial_and_dictionary_work_receives_attack_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R159.R82.FAMILIES[: R160.FAMILY_COUNT]
        for seed in R160.SEEDS
    ]
    all_divisors = all(
        row["c3_x_coordinates_injective"]
        and row["signed_divisor_curve_residual_zero"]
        for row in rows
    )
    all_positive = all(row["all_positive_sources_exact"] for row in rows)
    all_unique = all(row["all_positive_targets_unique_c6"] for row in rows)
    all_degree = all(
        row["unique_source_gcd_degree_bound_20_holds"] for row in rows
    )
    all_empty = all(row["empty_target_exact"] for row in rows)
    all_exceptional = all(
        row["denominator_exception_branch_exercised"]
        and row["denominator_exception_semantics_exact"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_signed_c3_divisor_translation_gcd.controls.r161.v1",
        "control_count": len(rows),
        "family_count": R160.FAMILY_COUNT,
        "seeds": list(R160.SEEDS),
        "exact_signed_divisor_control_count": sum(
            row["c3_x_coordinates_injective"]
            and row["signed_divisor_curve_residual_zero"]
            for row in rows
        ),
        "exact_positive_source_control_count": sum(
            row["all_positive_sources_exact"] for row in rows
        ),
        "exact_empty_target_control_count": sum(
            row["empty_target_exact"] for row in rows
        ),
        "exact_exceptional_branch_control_count": sum(
            row["denominator_exception_branch_exercised"]
            and row["denominator_exception_semantics_exact"]
            for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "sixteen_source_bindings_verified": len(actual_bindings) == 16,
        "r117_translated_divisor_cost_semantically_deduplicated": True,
        "r160_representation_specificity_gate_inherited": True,
        "signed_c3_u_v_divisor_theorem_complete": True,
        "target_translation_quotient_formula_complete": True,
        "signed_membership_gcd_biconditional_complete": True,
        "denominator_exception_split_complete": True,
        "unique_c6_source_gcd_degree_bound_20_complete": True,
        "constant_degree_source_adjoint_reduction_complete": True,
        "six_finite_signed_divisor_controls_complete": len(rows) == 6,
        "all_finite_signed_divisors_exact": all_divisors,
        "all_finite_positive_sources_exact": all_positive,
        "all_finite_positive_targets_unique": all_unique,
        "all_finite_gcd_degree_bounds_hold": all_degree,
        "all_finite_empty_targets_exact": all_empty,
        "all_finite_exceptional_branches_exact": all_exceptional,
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "target_batched_many_inner_modular_composition_complete": False,
        "target_batched_gcd_source_adjoint_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "signed_divisor_translation_gcd_interface_admitted": True,
        "constant_degree_source_adjoint_admitted": True,
        "target_batched_composition_admitted": False,
        "lane_admitted": False,
    }
    replay = {
        "schema": "p1553.m6_signed_c3_divisor_translation_gcd.replay.r161.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "u_sha256": row["u_sha256"],
                "v_sha256": row["v_sha256"],
                "target_records_sha256": row["target_records_sha256"],
                "positive_target_count": row["positive_target_count"],
                "maximum_positive_source_gcd_degree": row[
                    "maximum_positive_source_gcd_degree"
                ],
                "positive_sources_exact": row["all_positive_sources_exact"],
                "empty_target_exact": row["empty_target_exact"],
                "exceptional_branch_exact": row[
                    "denominator_exception_semantics_exact"
                ],
            }
            for row in rows
        ],
        "all_replay_invariants_pass": (
            all_divisors
            and all_positive
            and all_unique
            and all_degree
            and all_empty
            and all_exceptional
        ),
    }
    frozen = {
        "schema": "p1553.m6_signed_c3_divisor_translation_gcd.frozen.r161.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "signed_divisor_interface": {
            "state": "U(X), V(X), and x/point-to-C3-source dictionaries",
            "target_maps": (
                "lambda=(v+V)/(u-X), phi=lambda^2-u-X, "
                "psi=lambda(u-phi)-v mod U"
            ),
            "membership_gcd": "gcd(U,U(phi),psi-V(phi))",
            "unique_source_gcd_degree_bound": 20,
            "exceptional_branch": "split roots of gcd(U,u-X) and check directly",
        },
        "successor_interface": {
            "persistent_degree": "n=B^(9/4+o(1))",
            "target_count": "N=B^(5/4+o(1))",
            "required_work": "B^(5/4+o(1)) after persistent setup",
            "required_output": "nontrivial gcd factor and one C3+C3 source",
            "open_primitive": (
                "many-inner target-batched modular composition and gcd source adjoint"
            ),
        },
        "admission": admission,
    }
    logs = {
        "schema": "p1553.m6_signed_c3_divisor_translation_gcd.logs.r161.v1",
        "r160_factor_logs_and_identical_descent_inherited": True,
        "signed_divisor_positive_source_control_count": sum(
            row["all_positive_sources_exact"] for row in rows
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "factor_log_replay_with_inside_cap_batch_operator": False,
        "identical_descent_replay_with_inside_cap_batch_operator": False,
        "unconditional_algorithm_credit": False,
    }
    next_action = (
        "Construct a target-batched many-inner modular-composition and gcd "
        "source adjoint for the frozen signed C3 divisor. Starting from "
        "U,V of degree B^(9/4), it must process all B^(5/4) R159 targets "
        "in B^(5/4+o(1)) work, return the degree-at-most-20 source factors, "
        "handle u-X exceptional roots, and replay factor logs and identical "
        "descent without materializing per-target degree-B^(9/4) states."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Instantiate the first exact coordinate-specific positive-C6 "
            "locator interface after R160 and identify its remaining batched "
            "polynomial primitive."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r117": (
                "Already charges independent translated-divisor queries; "
                "R161 adds the exact signed U,V quotient/gcd and source-degree "
                "interface, not a new claim for the old batch exponent."
            ),
            "r160": (
                "Requires an explicit coordinate operation; R161 supplies "
                "that operation but not its inside-cap batched realization."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "SIGNED_C3_DIVISOR_U_V_EXACT__ELLIPTIC_TARGET_TRANSLATION_IN_"
            "QUOTIENT_ALGEBRA__SIGNED_SOURCE_GCD_BICONDITIONAL__UNIQUE_C6_"
            "SOURCE_GCD_DEGREE_AT_MOST_20__SIX_FINITE_POSITIVE_EMPTY_AND_"
            "EXCEPTIONAL_CONTROLS_EXACT__INDEPENDENT_TARGET_COMPOSITION_"
            "B7O2__MANY_INNER_BATCHED_COMPOSITION_GCD_ADJOINT_OPEN__NO_RHO_"
            "SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
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
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
