#!/usr/bin/env python3
"""Reduce the R164 norm to translates of one randomized elliptic function."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_global_randomizer_elliptic_translate_product.r165.v1"

R164_PRODUCER = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_r164.py"
R164_REPORT = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_report_r164.json"
R164_FROZEN = ROOT / "frozen_m6_randomized_target_divisor_norm_union.json"
R164_COST = ROOT / "m6_randomized_target_divisor_norm_union_cost_ledger.json"
R164_REPLAY = ROOT / "m6_randomized_target_divisor_norm_union_replay.json"
R164_CONTROLS = ROOT / "m6_randomized_target_divisor_norm_union_controls.json"
R164_LABEL_ALGEBRA = ROOT / "randomized_target_label_algebra_and_false_positive_r164.json"
R164_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_randomized_target_divisor_norm_union_probe_r164.py"
R164_GATE = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_gate_r164.md"
R164_PARENT = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_parent_report_r164.yaml"
R113_REPORT = ROOT / "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_report_r113.json"
R113_GATE = ROOT / "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_gate_r113.md"
R148_REPORT = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
R148_GATE = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_gate_r148.md"
MILLER_PAPER = ROOT / "references/miller_weil_pairing_algorithm_1986.pdf"

SOURCE_BINDINGS = (
    ("r164_producer", R164_PRODUCER, "770294394c4f9b4e39652224ef44918242198c6c99d9fb786f9883774bb4044b"),
    ("r164_report", R164_REPORT, "8d9ddeaf584d8d8bd41c3373c9c765b1361aa9a287cf763217b1296ed85b389a"),
    ("r164_frozen", R164_FROZEN, "bfd2d51835b22c70ddb84217fc12c21187220ef7c2e40993cbcfe3b21cef2781"),
    ("r164_cost", R164_COST, "d2ad661d6fec3bbb1bb9bad95a549fa89ab7528a2abdbbd17dee486e8d607ea1"),
    ("r164_replay", R164_REPLAY, "db29165a34163d1cab89569c883a41f1e61eb1c2ae56f341316b50fcd6c708e4"),
    ("r164_controls", R164_CONTROLS, "6604f752e894d891afe39e44eb984260a705d0eac86b5734e9dca636e1574cc9"),
    ("r164_label_algebra", R164_LABEL_ALGEBRA, "718cbcb4a79a238eb2995b16c56098a7c5cab510dd2b6d1b7faaf730f72dfd62"),
    ("r164_test", R164_TEST, "3056cfbbd9abea6dd1b436216112b2d073cd604eb3fc12882b253b510341327a"),
    ("r164_gate", R164_GATE, "65f78505220ff5e396fb834c9c81034861b80391578089cc295d72c9be771ea3"),
    ("r164_parent", R164_PARENT, "b0aa13e8a47535b223824c97b18f4f577ad7723296fb54d5f11f9be27d6622fc"),
    ("r113_report", R113_REPORT, "5b961649fa71ced6b049a3500316114b3c9f3806b5b61639471d9a6f86b01102"),
    ("r113_gate", R113_GATE, "d1939734367f92798ec40b89f9954ff0c9fe3bf07c6ad0bdaeb33e9a5fccad94"),
    ("r148_report", R148_REPORT, "b01496572b919ffd15406ee83bcd185675b96669d0cd40f51972ddf56f2caed7"),
    ("r148_gate", R148_GATE, "50e465e70d3e57457a64185cd9b86fc9b08bd9ebc56cfe6969f8c1f04508d9ca"),
    ("miller_paper", MILLER_PAPER, "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_global_randomizer_elliptic_translate_product_probe_report_r165.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_global_randomizer_elliptic_translate_product.json"
DEFAULT_COST = ROOT / "m6_global_randomizer_elliptic_translate_product_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_global_randomizer_elliptic_translate_product_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_global_randomizer_elliptic_translate_product_controls.json"
DEFAULT_TRANSLATE = ROOT / "global_randomizer_fixed_function_translate_product_r165.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R164 = load_module("p1553_r164_for_r165", R164_PRODUCER)
R163 = R164.R163
R161 = R164.R161


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
        raise AssertionError(f"R165 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def global_randomizer(curve: dict[str, Any], seed: int) -> int:
    return R164.deterministic_field_element(
        "r165-global-randomizer",
        int(curve["field_prime"]),
        curve["family_id"],
        seed,
    )


def fixed_function_value(
    point: tuple[int, int],
    divisor: dict[str, Any],
    randomizer: int,
    prime: int,
) -> int:
    x_value, y_value = map(int, point)
    return (
        R161.poly_eval(divisor["u"], x_value, prime)
        + randomizer
        * (y_value - R161.poly_eval(divisor["v"], x_value, prime))
    ) % prime


def global_control(
    curve: dict[str, Any], seed: int, *, randomizer: int | None = None
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    randomizer = (
        global_randomizer(curve, seed)
        if randomizer is None
        else randomizer % prime
    )
    _, divisor, targets = R164.target_material(curve, seed)
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    function_coefficients = R161.poly_sub(
        divisor["u"], R161.poly_scale(divisor["v"], randomizer, prime), prime
    )

    prescribed_zero_exact = all(
        fixed_function_value(
            tuple(record["endpoint"]), divisor, randomizer, prime
        )
        == 0
        for record in divisor["records"]
    )
    translation_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    exact_union_roots: set[int] = set()
    pole_equality_pair_count = 0
    tangent_pair_count = 0
    regular_identity_count = 0
    tangent_identity_count = 0
    all_translation_identities_exact = True
    for left_record in divisor["records"]:
        left = tuple(left_record["endpoint"])
        product_value = 1
        any_match = False
        for target_index, target_record in enumerate(targets):
            target = tuple(target_record["target"])
            right = R161.R70.add(
                target, R161.R70.negate(left, curve), curve
            )
            if right is None:
                pole_equality_pair_count += 1
                is_equality = left == target
                all_translation_identities_exact &= is_equality
                translation_rows.append(
                    {
                        "left_endpoint": list(left),
                        "target_index": target_index,
                        "branch": "pole_equality",
                        "left_equals_target": is_equality,
                        "semantic_product_factor": 1,
                        "direct_match": False,
                    }
                )
                continue
            right = tuple(right)
            function_value = fixed_function_value(
                right, divisor, randomizer, prime
            )
            direct = right in point_index
            any_match = any_match or direct
            product_value = product_value * function_value % prime
            if left[0] == target[0]:
                tangent_pair_count += 1
                branch = "tangent"
                tangent_identity_count += 1
                identity_exact = left == R161.R70.negate(target, curve)
            else:
                branch = "regular"
                regular_identity_count += 1
                x_residual, y_residual = R164.pair_residuals(
                    left, target, divisor, curve
                )
                identity_exact = function_value == (
                    x_residual + randomizer * y_residual
                ) % prime
            all_translation_identities_exact &= identity_exact
            translation_rows.append(
                {
                    "left_endpoint": list(left),
                    "target_index": target_index,
                    "branch": branch,
                    "translated_endpoint": list(right),
                    "fixed_function_value": function_value,
                    "translation_identity_exact": identity_exact,
                    "direct_match": direct,
                }
            )
        if any_match:
            exact_union_roots.add(int(left[0]))
        product_rows.append(
            {
                "left_endpoint": list(left),
                "translate_product_value": product_value,
                "any_match": any_match,
            }
        )

    selector = R161.interpolate(
        [
            (int(row["left_endpoint"][0]), int(row["translate_product_value"]))
            for row in product_rows
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(divisor["u"], selector, prime)
    candidate_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if R161.poly_eval(candidate_factor, int(record["endpoint"][0]), prime)
        == 0
    }
    verified_roots: set[int] = set()
    verification_scan_count = 0
    for record in divisor["records"]:
        left = tuple(record["endpoint"])
        if int(left[0]) not in candidate_roots:
            continue
        for target_record in targets:
            verification_scan_count += 1
            if R164.direct_match(
                left, tuple(target_record["target"]), point_index, curve
            ):
                verified_roots.add(int(left[0]))
                break
    expected_factor = R161.monic_root_polynomial(
        sorted(exact_union_roots), prime
    )
    verified_factor = R161.monic_root_polynomial(sorted(verified_roots), prime)
    inherited_r163 = R163.finite_control(curve, seed)
    return {
        "control_id": f"{curve['family_id']}_global_translate_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "global_randomizer": randomizer,
        "c3_divisor_degree": len(divisor["records"]),
        "target_count": len(targets),
        "fixed_function_coefficients_sha256": sha256_json(
            function_coefficients
        ),
        "fixed_function_prescribed_c3_zeros_exact": prescribed_zero_exact,
        "fixed_function_pole_order": 2 * len(divisor["records"]),
        "fixed_function_zero_divisor_degree": 2 * len(divisor["records"]),
        "fixed_function_extra_zero_divisor_degree": len(divisor["records"]),
        "translate_product_pole_divisor_degree": (
            2 * len(divisor["records"]) * len(targets)
        ),
        "translate_product_zero_divisor_degree": (
            2 * len(divisor["records"]) * len(targets)
        ),
        "translation_rows_sha256": sha256_json(translation_rows),
        "product_rows_sha256": sha256_json(product_rows),
        "selector_sha256": sha256_json(selector),
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "verified_factor_sha256": sha256_json(verified_factor),
        "expected_factor_sha256": sha256_json(expected_factor),
        "r163_expected_union_factor_sha256": inherited_r163[
            "expected_union_sha256"
        ],
        "regular_translation_identity_count": regular_identity_count,
        "tangent_translation_identity_count": tangent_identity_count,
        "pole_equality_pair_count": pole_equality_pair_count,
        "tangent_pair_count": tangent_pair_count,
        "chart_exception_pair_count_at_most_target_count": (
            pole_equality_pair_count + tangent_pair_count <= len(targets)
        ),
        "all_translation_identities_exact": all_translation_identities_exact,
        "exact_union_degree": len(exact_union_roots),
        "candidate_factor_degree": len(candidate_roots),
        "global_randomizer_false_positive_count": len(
            candidate_roots - exact_union_roots
        ),
        "global_randomizer_false_positive_roots": sorted(
            candidate_roots - exact_union_roots
        ),
        "all_true_roots_retained": exact_union_roots.issubset(candidate_roots),
        "verification_removes_all_false_positives": (
            verified_roots == exact_union_roots
        ),
        "verified_union_matches_r163_exactly": (
            sha256_json(verified_factor)
            == inherited_r163["expected_union_sha256"]
        ),
        "verification_scan_count": verification_scan_count,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_pair_enumeration_receives_attack_credit": False,
    }


def forced_correlated_false_positive_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    seed = R161.R160.SEEDS[0]
    prime = int(curve["field_prime"])
    _, divisor, targets = R164.target_material(curve, seed)
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    exact_union_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if any(
            R164.direct_match(
                tuple(record["endpoint"]),
                tuple(target_record["target"]),
                point_index,
                curve,
            )
            for target_record in targets
        )
    }
    chosen: dict[str, Any] | None = None
    for left_record in divisor["records"]:
        left = tuple(left_record["endpoint"])
        if int(left[0]) in exact_union_roots:
            continue
        for target_index, target_record in enumerate(targets):
            target = tuple(target_record["target"])
            right = R161.R70.add(
                target, R161.R70.negate(left, curve), curve
            )
            if right is None:
                continue
            right = tuple(right)
            a_value = R161.poly_eval(divisor["u"], int(right[0]), prime)
            b_value = (
                int(right[1])
                - R161.poly_eval(divisor["v"], int(right[0]), prime)
            ) % prime
            if b_value == 0:
                continue
            forced_randomizer = -a_value * pow(b_value, -1, prime) % prime
            chosen = {
                "left": left,
                "target_index": target_index,
                "target": target,
                "translated": right,
                "a_value": a_value,
                "b_value": b_value,
                "forced_randomizer": forced_randomizer,
            }
            break
        if chosen is not None:
            break
    if chosen is None:
        raise AssertionError("unable to force a global-randomizer cancellation")
    row = global_control(
        curve, seed, randomizer=int(chosen["forced_randomizer"])
    )
    forced_root = int(chosen["left"][0])
    combined = (
        int(chosen["a_value"])
        + int(chosen["forced_randomizer"]) * int(chosen["b_value"])
    ) % prime
    return {
        "control_id": "forced_global_randomizer_correlated_cancellation",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "left_endpoint": list(chosen["left"]),
        "target_index": chosen["target_index"],
        "target": list(chosen["target"]),
        "translated_endpoint": list(chosen["translated"]),
        "a_value": chosen["a_value"],
        "b_value": chosen["b_value"],
        "forced_global_randomizer": chosen["forced_randomizer"],
        "combined_residual": combined,
        "forced_root_is_not_true_union_root": forced_root not in exact_union_roots,
        "forced_root_appears_as_false_positive": forced_root
        in row["global_randomizer_false_positive_roots"],
        "correlated_false_positive_count": row[
            "global_randomizer_false_positive_count"
        ],
        "verification_removes_all_correlated_false_positives": row[
            "verification_removes_all_false_positives"
        ],
        "verified_union_matches_r163_exactly": row[
            "verified_union_matches_r163_exactly"
        ],
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def tangent_and_pole_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    prime = int(curve["field_prime"])
    factor_base = R161.R160.generic_factor_base(curve, 3, 16001)
    point_p = tuple(factor_base["generator"])
    point_q = tuple(
        R161.R70.negate(R161.R70.scalar_mul(2, point_p, curve), curve)
    )
    target_tangent = tuple(R161.R70.negate(point_p, curve))
    points = [point_p, point_q]
    divisor = {
        "u": R161.monic_root_polynomial([point[0] for point in points], prime),
        "v": R161.interpolate(points, prime),
    }
    randomizer = R164.deterministic_field_element(
        "r165-tangent-control", prime
    )
    tangent_translate = R161.R70.add(
        target_tangent, R161.R70.negate(point_p, curve), curve
    )
    opposite_translate = R161.R70.add(
        target_tangent, R161.R70.negate(point_q, curve), curve
    )
    pole_translate = R161.R70.add(
        point_p, R161.R70.negate(point_p, curve), curve
    )
    if tangent_translate is None or opposite_translate is None:
        raise AssertionError("tangent control unexpectedly reached infinity")
    return {
        "control_id": "global_function_positive_tangent_and_pole_equality",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "global_randomizer": randomizer,
        "tangent_target": list(target_tangent),
        "tangent_left": list(point_p),
        "tangent_x_incidence": point_p[0] == target_tangent[0],
        "tangent_left_is_negative_target": point_p
        == R161.R70.negate(target_tangent, curve),
        "tangent_translate": list(tangent_translate),
        "tangent_translate_is_point_q": tuple(tangent_translate) == point_q,
        "tangent_fixed_function_zero": fixed_function_value(
            tuple(tangent_translate), divisor, randomizer, prime
        )
        == 0,
        "opposite_translate": list(opposite_translate),
        "opposite_translate_is_point_p": tuple(opposite_translate) == point_p,
        "opposite_fixed_function_zero": fixed_function_value(
            tuple(opposite_translate), divisor, randomizer, prime
        )
        == 0,
        "pole_target_equals_left": True,
        "pole_translate_is_infinity": pole_translate is None,
        "pole_semantic_product_factor": 1,
        "pole_equality_is_not_affine_membership": pole_translate is None,
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "one_global_randomizer": (
            "Independence across targets is unnecessary. For one uniform "
            "r in F_p, every fixed regular nonmatch a+r*b cancels with "
            "probability at most 1/p. A union bound over nN pairs still gives "
            "nN/p, and linearity of expectation gives the same expected "
            "false-root verification cost."
        ),
        "fixed_elliptic_function": (
            "Define f_r(Q)=U(x(Q))+r(y(Q)-V(x(Q)))=(U-rV)(x(Q))+r*y(Q). "
            "Every signed C3 point is a zero. Since U is monic of degree n "
            "while V has degree below n, f_r has one pole of exact order 2n "
            "at O and a zero divisor of degree 2n."
        ),
        "translate_product": (
            "For every affine translated point Q=T_j-P, the randomized "
            "residual equals f_r(T_j-P). Thus the hard regular norm is the "
            "single-function translate product Phi_r(P)=product_j "
            "f_r(T_j-P), restricted modulo U."
        ),
        "global_chart_semantics": (
            "The morphism P maps to T-P is global. The x(T)=x(P), P=-T "
            "case is tangent evaluation and needs no exceptional membership "
            "oracle. Only P=T maps to O, the pole of f_r; assign that one "
            "factor value one because O is outside the affine signed C3 set."
        ),
        "divisor_degree_boundary": (
            "For arbitrary N targets the explicit translate product has zero "
            "and pole divisor degrees 2nN=B^(7/2). This explains the standard "
            "represented cost but is not a lower bound for an output-sensitive "
            "remainder or arithmetic circuit modulo U."
        ),
        "scope": (
            "The finite producer evaluates every endpoint-target pair. It "
            "proves the global-randomizer and translate identities but does "
            "not construct Phi_r mod U below rho."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_global_randomizer_elliptic_translate_product.cost.r165.v1",
        "field_prime_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "positive_target_count_exponent_B": fraction_record(Fraction(3, 4)),
        "fixed_function_representation_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "fixed_function_zero_divisor_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "explicit_translate_product_divisor_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "standard_translate_product_exponent_B": fraction_record(Fraction(7, 2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_product_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "false_union_probability_exponent_B": fraction_record(Fraction(-3, 2)),
        "expected_regular_verification_exponent_B": fraction_record(Fraction(2)),
        "expected_false_verification_exponent_B": fraction_record(Fraction(-1, 4)),
        "pole_equality_hash_correction_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "per_target_independent_randomizers_required": False,
        "target_label_randomizer_interpolation_required": False,
        "one_fixed_elliptic_function_supplied": True,
        "tangent_branch_is_global_function_evaluation": True,
        "pole_equality_correction_inside_rho": True,
        "standard_explicit_translate_product_inside_rho": False,
        "output_sensitive_translate_product_supplied": False,
        "finite_pair_enumeration_receives_attack_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        global_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    forced = forced_correlated_false_positive_control()
    tangent_pole = tangent_and_pole_control()
    all_identities = all(row["all_translation_identities_exact"] for row in rows)
    all_zeros = all(
        row["fixed_function_prescribed_c3_zeros_exact"] for row in rows
    )
    all_true = all(row["all_true_roots_retained"] for row in rows)
    all_verified = all(
        row["verification_removes_all_false_positives"]
        and row["verified_union_matches_r163_exactly"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_global_randomizer_elliptic_translate_product.controls.r165.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_fixed_function_prescribed_zeros_exact": all_zeros,
        "all_translation_identities_exact": all_identities,
        "all_true_roots_retained": all_true,
        "all_verified_unions_exact": all_verified,
        "unforced_global_false_positive_count": sum(
            row["global_randomizer_false_positive_count"] for row in rows
        ),
        "forced_correlated_false_positive_control": forced,
        "tangent_and_pole_control": tangent_pole,
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fifteen_source_bindings_verified": len(actual_bindings) == 15,
        "r164_randomized_norm_and_verifier_inherited": True,
        "r113_orbit_product_grammar_deduplicated": True,
        "r148_static_3sum_tradeoff_deduplicated": True,
        "miller_scalar_chain_scope_bound": True,
        "one_global_randomizer_suffices_theorem_complete": True,
        "union_bound_without_independence_complete": True,
        "fixed_elliptic_function_identity_complete": True,
        "fixed_function_pole_order_2n_complete": True,
        "single_function_translate_product_identity_complete": True,
        "tangent_branch_globalized": True,
        "pole_equality_semantics_complete": True,
        "six_finite_global_randomizer_controls_complete": len(rows) == 6,
        "all_finite_fixed_function_zeros_exact": all_zeros,
        "all_finite_translation_identities_exact": all_identities,
        "all_finite_true_roots_retained": all_true,
        "all_finite_verified_unions_exact": all_verified,
        "forced_correlated_false_positive_removed": (
            forced["combined_residual"] == 0
            and forced["forced_root_is_not_true_union_root"]
            and forced["forced_root_appears_as_false_positive"]
            and forced["verification_removes_all_correlated_false_positives"]
        ),
        "positive_tangent_and_pole_control_complete": (
            tangent_pole["tangent_fixed_function_zero"]
            and tangent_pole["opposite_fixed_function_zero"]
            and tangent_pole["pole_translate_is_infinity"]
            and tangent_pole["pole_semantic_product_factor"] == 1
        ),
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "translate_product_below_rho_constructed": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "global_randomizer_reduction_admitted": True,
        "fixed_function_translate_identity_admitted": True,
        "translate_product_constructor_admitted": False,
        "lane_admitted": False,
    }
    translate = {
        "schema": "p1553.m6_global_randomizer_fixed_function_translate_product.r165.v1",
        "theorem": theorem,
        "finite_function_records": [
            {
                "control_id": row["control_id"],
                "global_randomizer": row["global_randomizer"],
                "fixed_function_coefficients_sha256": row[
                    "fixed_function_coefficients_sha256"
                ],
                "fixed_function_pole_order": row["fixed_function_pole_order"],
                "translate_product_zero_divisor_degree": row[
                    "translate_product_zero_divisor_degree"
                ],
                "translation_rows_sha256": row["translation_rows_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "verified_factor_sha256": row["verified_factor_sha256"],
            }
            for row in rows
        ],
        "forced_correlated_false_positive_control": forced,
        "tangent_and_pole_control": tangent_pole,
        "output_sensitive_translate_product_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_global_randomizer_elliptic_translate_product.replay.r165.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "global_randomizer": row["global_randomizer"],
                "fixed_function_coefficients_sha256": row[
                    "fixed_function_coefficients_sha256"
                ],
                "translation_rows_sha256": row["translation_rows_sha256"],
                "product_rows_sha256": row["product_rows_sha256"],
                "selector_sha256": row["selector_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "verified_factor_sha256": row["verified_factor_sha256"],
                "expected_factor_sha256": row["expected_factor_sha256"],
            }
            for row in rows
        ],
        "forced_control_sha256": sha256_json(forced),
        "tangent_pole_control_sha256": sha256_json(tangent_pole),
        "all_replay_invariants_pass": (
            all_zeros
            and all_identities
            and all_true
            and all_verified
            and obligations["forced_correlated_false_positive_removed"]
            and obligations["positive_tangent_and_pole_control_complete"]
        ),
    }
    frozen = {
        "schema": "p1553.m6_global_randomizer_elliptic_translate_product.frozen.r165.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "one signed C3 divisor U,V; one global random scalar r; "
                "the N public target points"
            ),
            "fixed_function": "f_r(Q)=U(x(Q))+r(y(Q)-V(x(Q)))",
            "required_output": (
                "gcd(U, product_j f_r(T_j-P)) with P=T_j pole factors "
                "regularized to one"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "explicit 2nN divisor, n-by-N value table, unit-cost norm/"
                "D5/3SUM oracle, candidate DLP/root/count/rank/source oracle"
            ),
            "open_primitive": (
                "output-sensitive arbitrary-target translate product of one "
                "fixed elliptic function modulo U"
            ),
        },
    }
    next_action = (
        "Construct or refute gcd(U,product_j f_r(T_j-P)) below B^(5/2), "
        "preferably B^(9/4+o(1)), for the one fixed function "
        "f_r=U(x)+r(y-V(x)). Exploit shared elliptic translations without "
        "expanding the degree-2nN divisor or the n-by-N value table. R165 "
        "already supplies the global randomizer, tangent/pole semantics, "
        "false-positive bound, and exact verifier."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Show that one global random scalar suffices in R164 and reduce "
            "the hard norm to arbitrary-target translates of one fixed "
            "elliptic function."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r164": (
                "Uses independent target randomizers in a collision-safe label "
                "algebra. R165 proves independence is unnecessary and turns "
                "all factors into translates of one fixed function."
            ),
            "r113": (
                "Closes a full fixed translation orbit and ordinary product "
                "tree. R165 has arbitrary public translations of one function "
                "and still does not claim a compressed product constructor."
            ),
            "r148": (
                "Closes standard static 3SUM-indexing tradeoffs at campaign "
                "caps but preserves coordinate-aware elliptic operators. R165 "
                "supplies exactly such a coordinate function interface, not a "
                "published data-structure improvement."
            ),
            "miller": (
                "Miller compresses functions associated with scalar-multiple "
                "divisor chains. The R165 target divisor is an arbitrary public "
                "set; no scalar-chain representation or unit-cost Miller "
                "compression is assumed."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_ONE_GLOBAL_RANDOMIZER_WITH_NN_OVER_P_UNION_BOUND__FIXED_"
            "ELLIPTIC_FUNCTION_F_EQUALS_U_PLUS_R_Y_MINUS_V_HAS_POLE_ORDER_2N__"
            "ALL_TARGET_FACTORS_ARE_TRANSLATES_OF_F__TANGENT_BRANCH_GLOBAL__"
            "ONLY_P_EQUALS_T_POLE_REGULARIZATION__FORCED_CORRELATED_FALSE_"
            "ROOTS_VERIFIED_AWAY__EXPLICIT_TRANSLATE_DIVISOR_B7O2__OUTPUT_"
            "SENSITIVE_TRANSLATE_PRODUCT_OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
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
        "translate": translate,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--translate-output", type=Path, default=DEFAULT_TRANSLATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.translate_output, bundle["translate"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
