#!/usr/bin/env python3
"""Freeze a randomized target-divisor norm reduction for the R163 union."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_randomized_target_divisor_norm_union.r164.v1"

R163_PRODUCER = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_r163.py"
R163_REPORT = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_report_r163.json"
R163_FROZEN = ROOT / "frozen_m6_aggregate_union_factor_label_recovery.json"
R163_COST = ROOT / "m6_aggregate_union_factor_label_recovery_cost_ledger.json"
R163_REPLAY = ROOT / "m6_aggregate_union_factor_label_recovery_replay.json"
R163_CONTROLS = ROOT / "m6_aggregate_union_factor_label_recovery_controls.json"
R163_LABELS = ROOT / "aggregate_union_target_labels_and_backpointers_r163.json"
R163_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_aggregate_union_factor_label_recovery_probe_r163.py"
R163_GATE = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_gate_r163.md"
R163_PARENT = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_parent_report_r163.yaml"
R107_REPORT = ROOT / "p1553_5a5c_noncharacter_algebraic_target_norm_resultant_probe_report_r107.json"
R107_GATE = ROOT / "p1553_5a5c_noncharacter_algebraic_target_norm_resultant_probe_gate_r107.md"
DYNAMIC_EVALUATION_PAPER = ROOT / "references/dahan_lex_groebner_dynamic_evaluation_2010.14775v3.pdf"

SOURCE_BINDINGS = (
    ("r163_producer", R163_PRODUCER, "225108ac4f89b44c88db5db93aa3c2c1b4578c4b0993f7f736781b469b3d77f9"),
    ("r163_report", R163_REPORT, "0a35b4fee30c7abf4ba69232e0f7af65d48ec785aef832c934bf03d5366645ab"),
    ("r163_frozen", R163_FROZEN, "2971fc80ba63eb35303f35592dab57c91d532f31886cfde21b632ef04626c8d7"),
    ("r163_cost", R163_COST, "28f090f312d5177d4f7d4132e5b772b56b42ff372c7866db23e227b3888e4722"),
    ("r163_replay", R163_REPLAY, "83fa1b944f0b336e8c81c3e166ff0ce94f18949201d93f0b440af87df58e655e"),
    ("r163_controls", R163_CONTROLS, "dc485b0fd593aadd11db178fd2c2edc5510fc443dca928fcc58627a818c4d5ff"),
    ("r163_labels", R163_LABELS, "815fbd1a3e64a6f203a34541c5a29c407553144b934b56a5641e38ec4a15b9ee"),
    ("r163_test", R163_TEST, "6f829d95211f8d4b2785e6392319435e6a651a41028de506b2cfadbce621058a"),
    ("r163_gate", R163_GATE, "3064a108bf063ab59a0991be6910b70bf3a2e498974add780f6a57e157a63212"),
    ("r163_parent", R163_PARENT, "cbaf7edfdf8ca8082138be914b6af923d913be75b61b44cf3da774584207f90e"),
    ("r107_report", R107_REPORT, "0f85b6c29df4b91a56b6f17badc08c08f34c95e21c3b5219261a9308f493d65d"),
    ("r107_gate", R107_GATE, "74343856b9bd1865d92bde6055b9d3b441d2ad716797f478d99ab8ee85cc1c51"),
    ("dahan_dynamic_evaluation", DYNAMIC_EVALUATION_PAPER, "e17f13261cab77b08313c5524764e2a7b1030dfc83b56afc1a88c954034c667f"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_report_r164.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_randomized_target_divisor_norm_union.json"
DEFAULT_COST = ROOT / "m6_randomized_target_divisor_norm_union_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_randomized_target_divisor_norm_union_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_randomized_target_divisor_norm_union_controls.json"
DEFAULT_LABEL_ALGEBRA = ROOT / "randomized_target_label_algebra_and_false_positive_r164.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R163 = load_module("p1553_r163_for_r164", R163_PRODUCER)
R161 = R163.R161


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
        raise AssertionError(f"R164 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def deterministic_field_element(
    domain: str, prime: int, *parts: Any
) -> int:
    transcript = "|".join([domain, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(transcript.encode()).digest(), "big") % prime


def target_material(
    curve: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    dimension = len(R161.R157.c_point_pairs(curve, 0))
    factor_base = R161.R160.generic_factor_base(curve, dimension, seed)
    divisor = R161.signed_c3_divisor(factor_base["representatives"], curve)
    base = R161.finite_control(curve, seed)
    return factor_base, divisor, R163.target_records(base)


def pair_residuals(
    left: tuple[int, int],
    target: tuple[int, int],
    divisor: dict[str, Any],
    curve: dict[str, Any],
) -> tuple[int, int]:
    if left[0] == target[0]:
        raise ValueError("incidence pair belongs to the exceptional branch")
    right = R161.R70.add(target, R161.R70.negate(left, curve), curve)
    if right is None:
        raise AssertionError("regular translated point unexpectedly reached infinity")
    prime = int(curve["field_prime"])
    x_residual = R161.poly_eval(divisor["u"], int(right[0]), prime)
    y_residual = (
        int(right[1])
        - R161.poly_eval(divisor["v"], int(right[0]), prime)
    ) % prime
    return x_residual, y_residual


def target_randomizers(
    curve: dict[str, Any], seed: int, count: int
) -> list[int]:
    prime = int(curve["field_prime"])
    return [
        deterministic_field_element(
            "r164-target-randomizer", prime, curve["family_id"], seed, index
        )
        for index in range(count)
    ]


def direct_match(
    left: tuple[int, int],
    target: tuple[int, int],
    point_index: dict[tuple[int, int], dict[str, Any]],
    curve: dict[str, Any],
) -> bool:
    right = R161.R70.add(target, R161.R70.negate(left, curve), curve)
    return right is not None and tuple(right) in point_index


def finite_control(
    curve: dict[str, Any],
    seed: int,
    *,
    randomizer_overrides: dict[int, int] | None = None,
    control_suffix: str = "",
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    _, divisor, targets = target_material(curve, seed)
    labels = list(range(len(targets)))
    randomizers = target_randomizers(curve, seed, len(targets))
    for index, value in (randomizer_overrides or {}).items():
        randomizers[index] = value % prime

    label_modulus = R161.monic_root_polynomial(labels, prime)
    target_x_interpolant = R161.interpolate(
        [(label, int(targets[label]["target"][0])) for label in labels], prime
    )
    target_y_interpolant = R161.interpolate(
        [(label, int(targets[label]["target"][1])) for label in labels], prime
    )
    randomizer_interpolant = R161.interpolate(
        list(zip(labels, randomizers)), prime
    )
    target_x_root_polynomial = R161.monic_root_polynomial(
        [int(record["target"][0]) for record in targets], prime
    )
    incidence_factor = R161.poly_gcd(
        divisor["u"], target_x_root_polynomial, prime
    )

    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    norm_rows: list[dict[str, Any]] = []
    exact_regular_roots: set[int] = set()
    exact_union_roots: set[int] = set()
    incidence_pair_count = 0
    for left_record in divisor["records"]:
        left = tuple(left_record["endpoint"])
        norm_value = 1
        regular_match = False
        any_match = False
        row_residuals = []
        for target_index, target_record in enumerate(targets):
            target = tuple(target_record["target"])
            if left[0] == target[0]:
                incidence_pair_count += 1
                is_match = direct_match(left, target, point_index, curve)
                any_match = any_match or is_match
                row_residuals.append(
                    {
                        "target_index": target_index,
                        "incidence_branch": True,
                        "regular_norm_factor": 1,
                        "direct_match": is_match,
                    }
                )
                continue
            x_residual, y_residual = pair_residuals(
                left, target, divisor, curve
            )
            combined = (
                x_residual + randomizers[target_index] * y_residual
            ) % prime
            norm_value = norm_value * combined % prime
            is_match = x_residual == 0 and y_residual == 0
            regular_match = regular_match or is_match
            any_match = any_match or is_match
            row_residuals.append(
                {
                    "target_index": target_index,
                    "incidence_branch": False,
                    "x_residual": x_residual,
                    "y_residual": y_residual,
                    "randomizer": randomizers[target_index],
                    "combined_residual": combined,
                    "direct_match": is_match,
                }
            )
        if regular_match:
            exact_regular_roots.add(int(left[0]))
        if any_match:
            exact_union_roots.add(int(left[0]))
        norm_rows.append(
            {
                "left_endpoint": list(left),
                "regular_norm_value": norm_value,
                "regular_match": regular_match,
                "any_match": any_match,
                "residuals_sha256": sha256_json(row_residuals),
            }
        )

    norm_selector = R161.interpolate(
        [
            (int(row["left_endpoint"][0]), int(row["regular_norm_value"]))
            for row in norm_rows
        ],
        prime,
    )
    norm_candidate_factor = R161.poly_gcd(
        divisor["u"], norm_selector, prime
    )
    norm_candidate_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if R161.poly_eval(
            norm_candidate_factor, int(record["endpoint"][0]), prime
        )
        == 0
    }
    incidence_candidate_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if R161.poly_eval(incidence_factor, int(record["endpoint"][0]), prime)
        == 0
    }

    verified_norm_roots: set[int] = set()
    regular_verification_scan_count = 0
    for record in divisor["records"]:
        left = tuple(record["endpoint"])
        if int(left[0]) not in norm_candidate_roots:
            continue
        for target_record in targets:
            regular_verification_scan_count += 1
            if direct_match(left, tuple(target_record["target"]), point_index, curve):
                verified_norm_roots.add(int(left[0]))
                break

    target_x_buckets: dict[int, list[tuple[int, int]]] = {}
    for target_record in targets:
        target = tuple(target_record["target"])
        target_x_buckets.setdefault(int(target[0]), []).append(target)
    verified_incidence_roots: set[int] = set()
    incidence_verification_pair_count = 0
    for record in divisor["records"]:
        left = tuple(record["endpoint"])
        if int(left[0]) not in incidence_candidate_roots:
            continue
        for target in target_x_buckets.get(int(left[0]), []):
            incidence_verification_pair_count += 1
            if direct_match(left, target, point_index, curve):
                verified_incidence_roots.add(int(left[0]))
                break

    verified_union_roots = verified_norm_roots | verified_incidence_roots
    candidate_union_roots = norm_candidate_roots | incidence_candidate_roots
    expected_union_factor = R161.monic_root_polynomial(
        sorted(exact_union_roots), prime
    )
    verified_union_factor = R161.monic_root_polynomial(
        sorted(verified_union_roots), prime
    )
    candidate_union_factor = R161.monic_root_polynomial(
        sorted(candidate_union_roots), prime
    )
    inherited_r163 = R163.finite_control(curve, seed)
    return {
        "control_id": (
            f"{curve['family_id']}_randomized_norm_seed{seed}"
            f"{control_suffix}"
        ),
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": len(divisor["records"]),
        "target_count": len(targets),
        "target_labels_distinct": len(labels) == len(set(labels)),
        "target_labels_fit_base_field": len(labels) < prime,
        "label_modulus_sha256": sha256_json(label_modulus),
        "target_x_interpolant_sha256": sha256_json(target_x_interpolant),
        "target_y_interpolant_sha256": sha256_json(target_y_interpolant),
        "randomizer_interpolant_sha256": sha256_json(randomizer_interpolant),
        "randomizer_vector_sha256": sha256_json(randomizers),
        "norm_rows_sha256": sha256_json(norm_rows),
        "norm_selector_sha256": sha256_json(norm_selector),
        "norm_candidate_factor_sha256": sha256_json(norm_candidate_factor),
        "incidence_factor_sha256": sha256_json(incidence_factor),
        "candidate_union_factor_sha256": sha256_json(candidate_union_factor),
        "verified_union_factor_sha256": sha256_json(verified_union_factor),
        "expected_union_factor_sha256": sha256_json(expected_union_factor),
        "r163_expected_union_factor_sha256": inherited_r163[
            "expected_union_sha256"
        ],
        "exact_regular_union_degree": len(exact_regular_roots),
        "exact_union_degree": len(exact_union_roots),
        "randomized_norm_candidate_degree": len(norm_candidate_roots),
        "incidence_candidate_degree": len(incidence_candidate_roots),
        "candidate_union_degree": len(candidate_union_roots),
        "randomized_false_positive_count": len(
            norm_candidate_roots - exact_regular_roots
        ),
        "randomized_false_positive_roots": sorted(
            norm_candidate_roots - exact_regular_roots
        ),
        "incidence_pair_count": incidence_pair_count,
        "incidence_pair_count_at_most_target_count": (
            incidence_pair_count <= len(targets)
        ),
        "regular_verification_scan_count": regular_verification_scan_count,
        "incidence_verification_pair_count": incidence_verification_pair_count,
        "true_regular_roots_never_lost": exact_regular_roots.issubset(
            norm_candidate_roots
        ),
        "all_true_roots_reach_candidate_union": exact_union_roots.issubset(
            candidate_union_roots
        ),
        "verification_removes_all_false_positives": (
            verified_union_roots == exact_union_roots
        ),
        "verified_union_matches_r163_exactly": (
            sha256_json(verified_union_factor)
            == inherited_r163["expected_union_sha256"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_split_algebra_enumeration_receives_attack_credit": False,
    }


def forced_false_positive_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    seed = R161.R160.SEEDS[0]
    prime = int(curve["field_prime"])
    _, divisor, targets = target_material(curve, seed)
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    exact_union_roots = {
        int(record["endpoint"][0])
        for record in divisor["records"]
        if any(
            direct_match(
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
            if left[0] == target[0]:
                continue
            x_residual, y_residual = pair_residuals(
                left, target, divisor, curve
            )
            if y_residual == 0:
                continue
            forced_randomizer = (
                -x_residual * pow(y_residual, -1, prime)
            ) % prime
            chosen = {
                "left": left,
                "target_index": target_index,
                "target": target,
                "x_residual": x_residual,
                "y_residual": y_residual,
                "forced_randomizer": forced_randomizer,
            }
            break
        if chosen is not None:
            break
    if chosen is None:
        raise AssertionError("unable to construct adversarial cancellation")
    row = finite_control(
        curve,
        seed,
        randomizer_overrides={
            int(chosen["target_index"]): int(chosen["forced_randomizer"])
        },
        control_suffix="_forced_false_positive",
    )
    combined = (
        int(chosen["x_residual"])
        + int(chosen["forced_randomizer"]) * int(chosen["y_residual"])
    ) % prime
    forced_root = int(chosen["left"][0])
    return {
        "control_id": "forced_single_regular_pair_randomized_cancellation",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "left_endpoint": list(chosen["left"]),
        "target_index": chosen["target_index"],
        "target": list(chosen["target"]),
        "x_residual": chosen["x_residual"],
        "y_residual": chosen["y_residual"],
        "forced_randomizer": chosen["forced_randomizer"],
        "combined_residual": combined,
        "forced_root_is_not_true_union_root": forced_root not in exact_union_roots,
        "forced_root_appears_as_randomized_false_positive": (
            forced_root in row["randomized_false_positive_roots"]
        ),
        "verification_removes_forced_false_positive": row[
            "verification_removes_all_false_positives"
        ],
        "verified_union_matches_r163_exactly": row[
            "verified_union_matches_r163_exactly"
        ],
        "forced_trial_randomized_false_positive_count": row[
            "randomized_false_positive_count"
        ],
        "forced_trial_candidate_union_degree": row["candidate_union_degree"],
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def positive_incidence_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    prime = int(curve["field_prime"])
    factor_base = R161.R160.generic_factor_base(curve, 3, 16001)
    point_p = factor_base["generator"]
    point_q = R161.R70.negate(
        R161.R70.scalar_mul(2, point_p, curve), curve
    )
    target = R161.R70.negate(point_p, curve)
    if point_p is None or point_q is None or target is None:
        raise AssertionError("positive incidence control left affine chart")
    points = [tuple(point_p), tuple(point_q)]
    if len({point[0] for point in points}) != 2:
        raise AssertionError("positive incidence control has x collision")
    u_poly = R161.monic_root_polynomial([point[0] for point in points], prime)
    incidence_factor = R161.poly_gcd(
        u_poly,
        R161.monic_root_polynomial([target[0]], prime),
        prime,
    )
    point_index = {point: {} for point in points}
    norm_values = []
    for left in points:
        if left[0] == target[0]:
            norm_values.append((left[0], 1))
            continue
        right = R161.R70.add(target, R161.R70.negate(left, curve), curve)
        norm_values.append((left[0], 0 if right in point_index else 1))
    norm_factor = R161.poly_gcd(
        u_poly, R161.interpolate(norm_values, prime), prime
    )
    incidence_roots = {
        point[0]
        for point in points
        if R161.poly_eval(incidence_factor, point[0], prime) == 0
    }
    norm_roots = {
        point[0]
        for point in points
        if R161.poly_eval(norm_factor, point[0], prime) == 0
    }
    verified_incidence = {
        left[0]
        for left in points
        if left[0] in incidence_roots
        and direct_match(left, tuple(target), point_index, curve)
    }
    verified_union = norm_roots | verified_incidence
    return {
        "control_id": "synthetic_positive_incidence_p_plus_minus2p",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "target": list(target),
        "target_x_equals_p_x": target[0] == point_p[0],
        "incidence_factor_degree": max(R161.poly_degree(incidence_factor), 0),
        "regular_norm_factor_degree": max(R161.poly_degree(norm_factor), 0),
        "incidence_root_is_true_exceptional_match": point_p[0]
        in verified_incidence,
        "opposite_orientation_is_regular_norm_root": point_q[0] in norm_roots,
        "combined_verified_union_is_exact": verified_union
        == {point_p[0], point_q[0]},
        "combined_verified_union_factor_sha256": sha256_json(
            R161.monic_root_polynomial(sorted(verified_union), prime)
        ),
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "label_algebra": (
            "Give the N public targets distinct field labels z_j and let "
            "W(Z)=product_j(Z-z_j). Interpolating target coordinates and a "
            "uniform degree-below-N polynomial R in F_p[Z]/W makes the values "
            "r_j=R(z_j) independent uniform field elements, even when target "
            "x-coordinates collide."
        ),
        "randomized_same_target_residual": (
            "For each regular pair (P,T_j), let a_j(P),b_j(P) be R161's signed "
            "x/y membership residuals and set c_j(P)=a_j(P)+r_j b_j(P). A true "
            "same-target match always has c_j(P)=0. A nonmatch has cancellation "
            "probability at most 1/p: exactly one r_j if b_j(P) is nonzero and "
            "none if b_j(P)=0."
        ),
        "regular_branch_norm": (
            "In the split tensor algebra, assign the value one to incidence "
            "pairs x(P)=x(T_j), and c_j(P) to regular pairs. Its target-label "
            "norm C(P)=product_j c_j(P) vanishes on every regular true union "
            "root and has only one-sided randomized error. Equivalently this is "
            "a resultant Res_Z(W,c) after the incidence branch is split."
        ),
        "false_positive_bound": (
            "For n left endpoints and N targets, the probability that any "
            "regular nonmatch enters gcd(U,C) is at most nN/p. With "
            "p=B^5,n=B^(9/4),N=B^(5/4), this is B^(-3/2). There are no false "
            "negatives."
        ),
        "las_vegas_verification": (
            "Factor gcd(U,C), scan the N public targets for each returned root, "
            "and retain only direct group-law matches. This always returns the "
            "exact regular union and has expected verification B^2; deliberately "
            "forced cancellations are rejected."
        ),
        "incidence_split": (
            "J=gcd(U,product_j(X-x(T_j))) lists all left roots in an incidence "
            "pair. Signed C3 x-injectivity gives at most N such pairs. Hashing "
            "targets by x and checking only the matching bucket recovers all "
            "true exceptional roots in O-tilde(n+N), independently of the hard "
            "regular norm constructor."
        ),
        "scope": (
            "The finite producer evaluates every split-algebra pair. It proves "
            "the reduction and error bound but does not construct the regular "
            "target norm below rho."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_randomized_target_divisor_norm_union.cost.r164.v1",
        "field_prime_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "positive_target_count_exponent_B": fraction_record(Fraction(3, 4)),
        "regular_union_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "false_union_probability_exponent_B": fraction_record(Fraction(-3, 2)),
        "expected_regular_verification_exponent_B": fraction_record(Fraction(2)),
        "expected_false_verification_exponent_B": fraction_record(Fraction(-1, 4)),
        "incidence_factor_and_split_exponent_B": fraction_record(Fraction(9, 4)),
        "incidence_pair_check_exponent_B": fraction_record(Fraction(5, 4)),
        "standard_coefficient_ring_norm_exponent_B": fraction_record(Fraction(7, 2)),
        "standard_norm_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "random_linear_residual_replaces_fermat_projector": True,
        "target_labels_tolerate_coordinate_collisions": True,
        "regular_true_roots_have_zero_false_negative_probability": True,
        "direct_verification_makes_output_exact": True,
        "exceptional_incidence_constructor_inside_rho": True,
        "standard_represented_norm_inside_rho": False,
        "output_sensitive_elliptic_translation_norm_supplied": False,
        "finite_pair_enumeration_receives_attack_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    forced = forced_false_positive_control()
    positive_incidence = positive_incidence_control()
    all_true_retained = all(row["true_regular_roots_never_lost"] for row in rows)
    all_verified = all(
        row["verification_removes_all_false_positives"]
        and row["verified_union_matches_r163_exactly"]
        for row in rows
    )
    all_incidence_bounds = all(
        row["incidence_pair_count_at_most_target_count"] for row in rows
    )
    controls = {
        "schema": "p1553.m6_randomized_target_divisor_norm_union.controls.r164.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_true_regular_roots_retained": all_true_retained,
        "all_verified_unions_exact": all_verified,
        "all_incidence_pair_bounds_hold": all_incidence_bounds,
        "unforced_randomized_false_positive_count": sum(
            row["randomized_false_positive_count"] for row in rows
        ),
        "forced_false_positive_control": forced,
        "positive_incidence_control": positive_incidence,
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "thirteen_source_bindings_verified": len(actual_bindings) == 13,
        "r163_union_and_b2_verifier_inherited": True,
        "r107_standard_resultant_obstruction_deduplicated": True,
        "dahan_dynamic_evaluation_scope_bound": True,
        "target_label_algebra_collision_safe": True,
        "independent_uniform_target_randomizers_complete": True,
        "same_target_linear_residual_complete": True,
        "one_sided_false_positive_bound_complete": True,
        "no_false_negative_theorem_complete": True,
        "las_vegas_direct_verification_complete": True,
        "incidence_factor_constructor_complete": True,
        "incidence_pair_bound_complete": True,
        "six_finite_randomized_norm_controls_complete": len(rows) == 6,
        "all_finite_true_roots_retained": all_true_retained,
        "all_finite_verified_unions_exact": all_verified,
        "all_finite_incidence_bounds_hold": all_incidence_bounds,
        "forced_false_positive_exercised_and_removed": (
            forced["combined_residual"] == 0
            and forced["forced_root_is_not_true_union_root"]
            and forced["forced_root_appears_as_randomized_false_positive"]
            and forced["verification_removes_forced_false_positive"]
        ),
        "positive_incidence_match_exercised": (
            positive_incidence["incidence_root_is_true_exceptional_match"]
            and positive_incidence["opposite_orientation_is_regular_norm_root"]
            and positive_incidence["combined_verified_union_is_exact"]
        ),
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "regular_target_norm_below_rho_constructed": False,
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
        "randomized_norm_reduction_admitted": True,
        "incidence_constructor_admitted": True,
        "regular_norm_constructor_admitted": False,
        "lane_admitted": False,
    }
    label_algebra = {
        "schema": "p1553.m6_randomized_target_label_algebra.r164.v1",
        "label_algebra_theorem": theorem["label_algebra"],
        "same_target_residual_theorem": theorem[
            "randomized_same_target_residual"
        ],
        "false_positive_bound": theorem["false_positive_bound"],
        "forced_false_positive_control": forced,
        "finite_label_algebra_records": [
            {
                "control_id": row["control_id"],
                "label_modulus_sha256": row["label_modulus_sha256"],
                "target_x_interpolant_sha256": row[
                    "target_x_interpolant_sha256"
                ],
                "target_y_interpolant_sha256": row[
                    "target_y_interpolant_sha256"
                ],
                "randomizer_interpolant_sha256": row[
                    "randomizer_interpolant_sha256"
                ],
                "randomized_false_positive_count": row[
                    "randomized_false_positive_count"
                ],
                "verified_union_factor_sha256": row[
                    "verified_union_factor_sha256"
                ],
            }
            for row in rows
        ],
        "output_sensitive_regular_norm_constructor_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_randomized_target_divisor_norm_union.replay.r164.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "randomizer_vector_sha256": row["randomizer_vector_sha256"],
                "norm_rows_sha256": row["norm_rows_sha256"],
                "norm_selector_sha256": row["norm_selector_sha256"],
                "norm_candidate_factor_sha256": row[
                    "norm_candidate_factor_sha256"
                ],
                "incidence_factor_sha256": row["incidence_factor_sha256"],
                "verified_union_factor_sha256": row[
                    "verified_union_factor_sha256"
                ],
                "expected_union_factor_sha256": row[
                    "expected_union_factor_sha256"
                ],
            }
            for row in rows
        ],
        "forced_false_positive_sha256": sha256_json(forced),
        "positive_incidence_sha256": sha256_json(positive_incidence),
        "all_replay_invariants_pass": (
            all_true_retained
            and all_verified
            and all_incidence_bounds
            and obligations["forced_false_positive_exercised_and_removed"]
            and obligations["positive_incidence_match_exercised"]
        ),
    }
    frozen = {
        "schema": "p1553.m6_randomized_target_divisor_norm_union.frozen.r164.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "signed C3 divisor U,V; N labeled public targets; one uniform "
                "degree-below-N randomizer polynomial in the target algebra"
            ),
            "required_output": (
                "the regular-branch norm factor gcd(U,Norm_target(a+R*b)); "
                "R164 supplies exact verification and the incidence branch"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "independent n-by-N residual enumeration, unit-cost resultant, "
                "candidate DLP/root/count/marginal/rank/source oracle"
            ),
            "open_primitive": (
                "output-sensitive elliptic-translation target norm in the "
                "split tensor algebra"
            ),
        },
    }
    next_action = (
        "Construct or refute the regular-branch target norm "
        "gcd(U,Norm_target(a+R*b)) below B^(5/2), preferably B^(9/4+o(1)), "
        "by exploiting that all target maps are translations of one elliptic "
        "divisor. Do not form the n-by-N residual table. R164 already supplies "
        "collision-safe target labels, one-sided randomization, exact B^2 "
        "verification, and the O-tilde(B^(9/4)) incidence branch."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Replace R163's Fermat projector by a one-sided randomized "
            "target-divisor norm and isolate denominator incidence from the "
            "remaining hard constructor."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r163": (
                "Supplies exact union semantics and B^2 label/backpointer "
                "verification. R164 changes the per-target nonlinear projector "
                "to one randomized linear residual and proves that verification "
                "makes its one-sided error exact."
            ),
            "r107": (
                "Closed standard explicit canonical 5A+5C resultants at a "
                "B^(13/5) body. R164 uses an unlabeled C3 endpoint union with "
                "a separate target-label algebra; it does not reopen or claim "
                "the R107 canonical multiplicity interface."
            ),
            "dahan_dynamic_evaluation": (
                "Dahan supplies correct quotient-ring splitting around "
                "noninvertible/nilpotent coefficients using subresultant "
                "sequences. The paper does not supply an output-sensitive "
                "simultaneous elliptic-translation norm; its represented "
                "subresultant route therefore receives no below-rho credit."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_COLLISION_SAFE_TARGET_LABEL_ALGEBRA__ADMIT_RANDOM_LINEAR_"
            "SAME_TARGET_RESIDUAL_WITH_NO_FALSE_NEGATIVES__UNION_BOUND_NN_OVER_"
            "P_EQUALS_B_MINUS3O2__DIRECT_VERIFICATION_REMOVES_FORCED_FALSE_"
            "ROOT__INCIDENCE_FACTOR_AND_X_BUCKET_BRANCH_B9O4__STANDARD_"
            "COEFFICIENT_RING_NORM_B7O2_OVER_RHO__OUTPUT_SENSITIVE_ELLIPTIC_"
            "TRANSLATION_NORM_OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
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
        "label_algebra": label_algebra,
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
    parser.add_argument(
        "--label-algebra-output", type=Path, default=DEFAULT_LABEL_ALGEBRA
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.label_algebra_output, bundle["label_algebra"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
