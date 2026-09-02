#!/usr/bin/env python3
"""Test target-sign conjugation and a factored S3 self-resultant interface."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_target_sign_conjugate_s3_self_resultant.r172.v1"

R171_PRODUCER = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_r171.py"
R171_REPORT = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_report_r171.json"
R171_FROZEN = ROOT / "frozen_m6_balanced_miller_tree_norm_streaming.json"
R171_COST = ROOT / "m6_balanced_miller_tree_norm_streaming_cost_ledger.json"
R171_REPLAY = ROOT / "m6_balanced_miller_tree_norm_streaming_replay.json"
R171_CONTROLS = ROOT / "m6_balanced_miller_tree_norm_streaming_controls.json"
R171_SLP = ROOT / "balanced_miller_tree_and_leaf_cancellation_r171.json"
R171_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_balanced_miller_tree_norm_streaming_probe_r171.py"
R171_GATE = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_gate_r171.md"
R171_PARENT = ROOT / "p1553_m6_balanced_miller_tree_norm_streaming_probe_parent_report_r171.yaml"
R161_REPORT = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_report_r161.json"
R161_GATE = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_gate_r161.md"
SEMAEV_PAPER = ROOT / "references/semaev_summation_polynomials_2004_031.ps"
MOROZ_SCHOST_PAPER = ROOT / "references/moroz_schost_truncated_resultant_1609.04259.pdf"
HYUN_NEIGER_SCHOST_PAPER = ROOT / "references/hyun_neiger_schost_bivariate_resultants_1905.04356.pdf"

SOURCE_BINDINGS = (
    ("r171_producer", R171_PRODUCER, "684ab4cf92398b7a7789b8e429356ee712ba255e65f5d172cee1c4ebbe58d39e"),
    ("r171_report", R171_REPORT, "12c34212fa3e07321ec7e9524a32a845233b420eae28f16e2dc3abfebd4efa92"),
    ("r171_frozen", R171_FROZEN, "a80e1d9cbb7893a0a70ee04c95cd4ca2632f45181c9b72d645bd1afaf15abaef"),
    ("r171_cost", R171_COST, "9e9fe2ae4e5ec8e905b60b56227be57e86f0cc10de63e0c0917dc7b5837ebb76"),
    ("r171_replay", R171_REPLAY, "76a793283ce4df296d0ead1dc6826bab40835cefdfcbc809e481bbbe377141be"),
    ("r171_controls", R171_CONTROLS, "d54c4d4db51abab1ccf4d01294bee945d240d637dd9db4a0078a523bef8aa300"),
    ("r171_slp", R171_SLP, "e322d78df66e9edec2fb39b412a98b23077b7648548f4154f30b7a166a5bcc7f"),
    ("r171_test", R171_TEST, "85df97c68496b5ce8beb3c207ab2e6b0bf0a51486be80183e6e92d02d76e74e5"),
    ("r171_gate", R171_GATE, "0f862466fa7816c438c6ef8526e66e350b1091c02f2e631d5b28a4105c10d680"),
    ("r171_parent", R171_PARENT, "3fee4660f7b558116299174194657da9464435f402a9f3fb289489474fbb1f93"),
    ("r161_report", R161_REPORT, "7e73325212e9d8c3cf42ae46cce0a5c1bdebea9784b9bf5c30761c006cb8bedd"),
    ("r161_gate", R161_GATE, "0ca8660b9554c39e2c90a8f29cee05554b9fb3adce0c4c2a1038e739b3888bcd"),
    ("semaev_2004", SEMAEV_PAPER, "991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df"),
    ("moroz_schost_2016", MOROZ_SCHOST_PAPER, "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709"),
    ("hyun_neiger_schost_2019", HYUN_NEIGER_SCHOST_PAPER, "32b73cf0ca7172bdec0f8f1b256adda628a86d6dd7eee8e07e8644e35a9f16f3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_report_r172.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_target_sign_conjugate_s3_self_resultant.json"
DEFAULT_COST = ROOT / "m6_target_sign_conjugate_s3_self_resultant_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_target_sign_conjugate_s3_self_resultant_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_target_sign_conjugate_s3_self_resultant_controls.json"
DEFAULT_RESULTANT = ROOT / "target_conjugate_s3_factored_self_resultant_r172.json"

Bivariate = dict[tuple[int, int], int]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R171 = load_module("p1553_r171_for_r172", R171_PRODUCER)
R167 = R171.R167
R161 = R171.R161


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
        raise AssertionError(f"R172 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def coefficient_density(poly: list[int], size: int) -> dict[str, Any]:
    padded = [*poly, *([0] * (size - len(poly)))]
    nonzero = sum(value != 0 for value in padded)
    return {
        "slot_count": size,
        "nonzero_coefficient_count": nonzero,
        "density": nonzero / size,
        "degree": R161.poly_degree(poly),
        "coefficient_sha256": sha256_json(padded),
    }


def curve_rhs(value: int, curve: dict[str, Any]) -> int:
    prime = int(curve["field_prime"])
    return (
        value * value * value
        + int(curve["curve_a"]) * value
        + int(curve["curve_b"])
    ) % prime


def semaev_s3(
    left_x: int, right_x: int, target_x: int, curve: dict[str, Any]
) -> int:
    prime = int(curve["field_prime"])
    a = int(curve["curve_a"])
    b = int(curve["curve_b"])
    return (
        (left_x - right_x) ** 2 * target_x**2
        - 2
        * (
            (left_x + right_x) * (left_x * right_x + a)
            + 2 * b
        )
        * target_x
        + (left_x * right_x - a) ** 2
        - 4 * b * (left_x + right_x)
    ) % prime


def s3_bivariate_factor(target_x: int, curve: dict[str, Any]) -> Bivariate:
    prime = int(curve["field_prime"])
    a = int(curve["curve_a"])
    b = int(curve["curve_b"])
    u = target_x % prime
    return {
        (2, 2): 1,
        (2, 1): (-2 * u) % prime,
        (1, 2): (-2 * u) % prime,
        (2, 0): (u * u) % prime,
        (0, 2): (u * u) % prime,
        (1, 1): (-2 * (u * u + a)) % prime,
        (1, 0): (-2 * a * u - 4 * b) % prime,
        (0, 1): (-2 * a * u - 4 * b) % prime,
        (0, 0): (a * a - 4 * b * u) % prime,
    }


def bivariate_mul(left: Bivariate, right: Bivariate, prime: int) -> Bivariate:
    result: Bivariate = {}
    for (left_x, left_z), left_value in left.items():
        for (right_x, right_z), right_value in right.items():
            key = (left_x + right_x, left_z + right_z)
            result[key] = (
                result.get(key, 0) + left_value * right_value
            ) % prime
    return {key: value for key, value in result.items() if value}


def bivariate_eval(poly: Bivariate, x_value: int, z_value: int, prime: int) -> int:
    return sum(
        coefficient
        * pow(x_value, x_degree, prime)
        * pow(z_value, z_degree, prime)
        for (x_degree, z_degree), coefficient in poly.items()
    ) % prime


def modular_rank(matrix: list[list[int]], prime: int) -> int:
    reduced = [[value % prime for value in row] for row in matrix]
    rank = 0
    column_count = len(reduced[0]) if reduced else 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, len(reduced))
                if reduced[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        inverse = pow(reduced[rank][column], -1, prime)
        reduced[rank] = [value * inverse % prime for value in reduced[rank]]
        for row in range(len(reduced)):
            if row == rank or reduced[row][column] == 0:
                continue
            scale = reduced[row][column]
            reduced[row] = [
                (left - scale * right) % prime
                for left, right in zip(reduced[row], reduced[rank])
            ]
        rank += 1
    return rank


def bivariate_record(poly: Bivariate, prime: int) -> dict[str, Any]:
    degree_x = max(x_degree for x_degree, _ in poly)
    degree_z = max(z_degree for _, z_degree in poly)
    matrix = [
        [poly.get((x_degree, z_degree), 0) for z_degree in range(degree_z + 1)]
        for x_degree in range(degree_x + 1)
    ]
    terms = [
        {
            "x_degree": x_degree,
            "z_degree": z_degree,
            "coefficient": poly[(x_degree, z_degree)],
        }
        for x_degree, z_degree in sorted(poly)
    ]
    slot_count = (degree_x + 1) * (degree_z + 1)
    return {
        "degree_x": degree_x,
        "degree_z": degree_z,
        "coefficient_slot_count": slot_count,
        "nonzero_coefficient_count": len(poly),
        "coefficient_density": len(poly) / slot_count,
        "coefficient_matrix_rank": modular_rank(matrix, prime),
        "coefficient_matrix_row_count": degree_x + 1,
        "coefficient_matrix_column_count": degree_z + 1,
        "terms": terms,
        "term_sha256": sha256_json(terms),
    }


def homogenized_value(
    u_poly: list[int], numerator: int, denominator: int, prime: int
) -> int:
    degree = R161.poly_degree(u_poly)
    value = u_poly[-1] % prime
    denominator_power = denominator % prime
    for coefficient in reversed(u_poly[:-1]):
        value = (
            value * numerator + coefficient * denominator_power
        ) % prime
        denominator_power = denominator_power * denominator % prime
    if denominator_power != pow(denominator, degree + 1, prime):
        raise AssertionError("homogenized denominator power drifted")
    return value


def poly_pow_mod(
    base: list[int], exponent: int, modulus: list[int], prime: int
) -> list[int]:
    result = [1]
    base = R161.poly_mod(base, modulus, prime)
    while exponent:
        if exponent & 1:
            result = R161.poly_mul_mod(result, base, modulus, prime)
        base = R161.poly_mul_mod(base, base, modulus, prime)
        exponent >>= 1
    return result


def interpolate_values(
    selected: list[tuple[int, int]], values: Iterable[int], prime: int
) -> list[int]:
    return R161.interpolate(
        [(int(point[0]), int(value)) for point, value in zip(selected, values)],
        prime,
    )


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r171 = R171.finite_control(curve, seed)
    _, divisor, target_records = R167.R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    n = len(selected)
    target_x = [int(point[0]) for point in targets]
    if len(set(target_x)) != len(target_x):
        raise AssertionError("target x-coordinates must be distinct")
    if set(target_x) & {int(point[0]) for point in selected}:
        raise AssertionError("target/selected x-equality requires an exceptional split")

    target_u = R161.monic_root_polynomial(target_x, prime)
    target_v = R161.interpolate(
        [(int(point[0]), int(point[1])) for point in targets], prime
    )
    target_curve_residual = R161.poly_sub(
        R161.poly_mul(target_v, target_v, prime),
        [int(curve["curve_b"]), int(curve["curve_a"]), 0, 1],
        prime,
    )
    target_curve_residual = R161.poly_mod(
        target_curve_residual, target_u, prime
    )
    if target_curve_residual != [0]:
        raise AssertionError("compact target divisor left the curve")

    reverse_resultant: Bivariate = {(0, 0): 1}
    for u_value in target_x:
        reverse_resultant = bivariate_mul(
            reverse_resultant,
            s3_bivariate_factor(u_value, curve),
            prime,
        )
    reverse_record = bivariate_record(reverse_resultant, prime)
    if reverse_record["degree_x"] != 2 * len(targets):
        raise AssertionError("reverse resultant lost x-degree")
    if reverse_record["degree_z"] != 2 * len(targets):
        raise AssertionError("reverse resultant lost z-degree")

    rows = []
    plus_values = []
    minus_values = []
    symmetric_values = []
    denominator_values = []
    numerator_values = []
    homogenized_identity_count = 0
    conjugate_s3_identity_count = 0
    reverse_resultant_identity_count = 0
    for selected_point in selected:
        x_value, y_value = selected_point
        plus_product = 1
        minus_product = 1
        denominator_product = 1
        iterated_resultant_value = 1
        for target in targets:
            u_value, v_value = target
            denominator = (u_value - x_value) ** 2 % prime
            if denominator == 0:
                raise AssertionError("unexpected target equality denominator")
            even_numerator = (
                u_value * u_value * x_value
                + u_value * x_value * x_value
                + int(curve["curve_a"]) * (u_value + x_value)
                + 2 * int(curve["curve_b"])
            ) % prime
            plus_numerator = (even_numerator + 2 * v_value * y_value) % prime
            minus_numerator = (even_numerator - 2 * v_value * y_value) % prime
            plus_point = R167.point_subtract(target, selected_point, curve)
            minus_target = R167.point_negate(target, curve)
            if plus_point is None or minus_target is None:
                raise AssertionError("regular target translation reached infinity")
            minus_point = R167.point_subtract(minus_target, selected_point, curve)
            if minus_point is None:
                raise AssertionError("conjugate translation reached infinity")
            plus_factor = R167.kummer_value(divisor, plus_point, prime)
            minus_factor = R167.kummer_value(divisor, minus_point, prime)
            plus_homogenized = homogenized_value(
                divisor["u"], plus_numerator, denominator, prime
            )
            minus_homogenized = homogenized_value(
                divisor["u"], minus_numerator, denominator, prime
            )
            denominator_to_n = pow(denominator, n, prime)
            if plus_homogenized != denominator_to_n * plus_factor % prime:
                raise AssertionError("plus homogenization failed")
            if minus_homogenized != denominator_to_n * minus_factor % prime:
                raise AssertionError("minus homogenization failed")
            homogenized_identity_count += 2

            s3_resultant = 1
            for right_point in selected:
                s3_resultant = (
                    s3_resultant
                    * semaev_s3(
                        x_value, int(right_point[0]), u_value, curve
                    )
                    % prime
                )
            if (
                plus_homogenized * minus_homogenized % prime
                != denominator_to_n * s3_resultant % prime
            ):
                raise AssertionError("target-sign conjugate S3 identity failed")
            conjugate_s3_identity_count += 1
            plus_product = plus_product * plus_factor % prime
            minus_product = minus_product * minus_factor % prime
            denominator_product = (
                denominator_product * denominator_to_n % prime
            )
            iterated_resultant_value = (
                iterated_resultant_value * s3_resultant % prime
            )

        reverse_row_product = 1
        for right_point in selected:
            reverse_row_product = (
                reverse_row_product
                * bivariate_eval(
                    reverse_resultant,
                    x_value,
                    int(right_point[0]),
                    prime,
                )
                % prime
            )
        if reverse_row_product != iterated_resultant_value:
            raise AssertionError("reverse resultant row product failed")
        reverse_resultant_identity_count += n
        target_u_value = R161.poly_eval(target_u, x_value, prime)
        batched_denominator = pow(target_u_value, 2 * n, prime)
        if batched_denominator != denominator_product:
            raise AssertionError("target-x denominator batch failed")
        symmetric_product = plus_product * minus_product % prime
        if symmetric_product * denominator_product % prime != iterated_resultant_value:
            raise AssertionError("iterated resultant quotient failed")
        plus_values.append(plus_product)
        minus_values.append(minus_product)
        symmetric_values.append(symmetric_product)
        denominator_values.append(denominator_product)
        numerator_values.append(iterated_resultant_value)
        rows.append(
            {
                "selected_endpoint": [int(x_value), int(y_value)],
                "plus_target_norm": plus_product,
                "minus_target_norm": minus_product,
                "symmetric_target_norm": symmetric_product,
                "batched_denominator": denominator_product,
                "iterated_s3_resultant_numerator": iterated_resultant_value,
                "plus_candidate": plus_product == 0,
                "conjugate_candidate": minus_product == 0,
                "symmetric_candidate": symmetric_product == 0,
                "identity_exact": True,
            }
        )

    plus_poly = interpolate_values(selected, plus_values, prime)
    minus_poly = interpolate_values(selected, minus_values, prime)
    symmetric_poly = interpolate_values(selected, symmetric_values, prime)
    numerator_poly = interpolate_values(selected, numerator_values, prime)
    denominator_poly = poly_pow_mod(
        target_u, 2 * n, divisor["u"], prime
    )
    if any(
        R161.poly_eval(denominator_poly, int(point[0]), prime) != value
        for point, value in zip(selected, denominator_values)
    ):
        raise AssertionError("denominator polynomial replay failed")
    if R161.poly_mul_mod(plus_poly, minus_poly, divisor["u"], prime) != symmetric_poly:
        raise AssertionError("conjugate aggregate multiplication failed")
    if (
        R161.poly_mul_mod(
            symmetric_poly, denominator_poly, divisor["u"], prime
        )
        != numerator_poly
    ):
        raise AssertionError("self-resultant quotient-ring identity failed")
    if R161.poly_gcd(divisor["u"], denominator_poly, prime) != [1]:
        raise AssertionError("batched denominator is not a unit")

    plus_roots = sorted(
        int(row["selected_endpoint"][0]) for row in rows if row["plus_candidate"]
    )
    minus_roots = sorted(
        int(row["selected_endpoint"][0])
        for row in rows
        if row["conjugate_candidate"]
    )
    symmetric_roots = sorted(
        int(row["selected_endpoint"][0])
        for row in rows
        if row["symmetric_candidate"]
    )
    if plus_roots != r171["candidate_roots"]:
        raise AssertionError("plus branch differs from R171")
    if symmetric_roots != sorted(set(plus_roots) | set(minus_roots)):
        raise AssertionError("symmetric roots are not the branch union")
    extra_roots = sorted(set(symmetric_roots) - set(plus_roots))
    overlap_roots = sorted(set(plus_roots) & set(minus_roots))
    output_polynomials = {
        "plus": coefficient_density(plus_poly, n),
        "minus": coefficient_density(minus_poly, n),
        "symmetric": coefficient_density(symmetric_poly, n),
        "denominator": coefficient_density(denominator_poly, n),
        "numerator": coefficient_density(numerator_poly, n),
    }
    return {
        "control_id": f"{curve['family_id']}_target_conjugate_s3_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": n,
        "retained_target_count": len(targets),
        "target_x_coordinates_distinct": True,
        "target_selected_x_support_disjoint": True,
        "target_divisor_curve_residual_zero": True,
        "target_u": target_u,
        "target_v": target_v,
        "target_u_sha256": sha256_json(target_u),
        "target_v_sha256": sha256_json(target_v),
        "target_divisor_coefficient_slot_count": len(target_u) + len(target_v),
        "homogenized_translate_identity_count": homogenized_identity_count,
        "target_conjugate_s3_identity_count": conjugate_s3_identity_count,
        "reverse_resultant_row_identity_count": reverse_resultant_identity_count,
        "all_target_conjugate_s3_identities_exact": True,
        "all_reverse_resultant_row_identities_exact": True,
        "batched_denominator_exact_and_unit": True,
        "reverse_target_resultant": reverse_record,
        "output_polynomials": output_polynomials,
        "plus_candidate_roots": plus_roots,
        "r171_candidate_roots": r171["candidate_roots"],
        "plus_candidate_roots_match_r171": plus_roots == r171["candidate_roots"],
        "conjugate_candidate_roots": minus_roots,
        "symmetric_candidate_roots": symmetric_roots,
        "extra_conjugate_candidate_roots": extra_roots,
        "overlap_candidate_roots": overlap_roots,
        "plus_candidate_root_count": len(plus_roots),
        "conjugate_candidate_root_count": len(minus_roots),
        "symmetric_candidate_root_count": len(symmetric_roots),
        "extra_conjugate_candidate_root_count": len(extra_roots),
        "overlap_candidate_root_count": len(overlap_roots),
        "row_transcript_sha256": sha256_json(rows),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_density_and_rank_receive_asymptotic_lower_bound_credit": False,
        "finite_candidate_counts_receive_asymptotic_attack_credit": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "homogenized_translation": (
            "For P=(X,V(X)), T=(u,v), D=(u-X)^2, and "
            "A=u^2X+uX^2+a(u+X)+2b, the x-coordinate of T-P is "
            "(A+2vV)/D modulo U. If H_U(K,D)=D^n U(K/D), then the "
            "plus and target-sign-conjugate factors are H_U(A+2vV,D) "
            "and H_U(A-2vV,D)."
        ),
        "semaev_conjugate_identity": (
            "The polynomial identity (A-DZ)^2-4(u^3+au+b)(X^3+aX+b) "
            "= D*S3(X,Z,u) implies H_U(A+2vV,D)H_U(A-2vV,D) "
            "= D^n Res_Z(U(Z),S3(X,Z,u)) modulo U(X)."
        ),
        "compact_target_x_batch": (
            "For W(Y)=product_j(Y-u_j), multiplying over targets gives a "
            "symmetric locator G_+(X)G_-(X)=R(X)/W(X)^(2n), where "
            "R(X)=Res_Z(U(Z),Res_Y(W(Y),S3(X,Z,Y))). The denominator "
            "is one exponentiation modulo U and is a unit after the public "
            "target/selected x-equality split."
        ),
        "candidate_semantics": (
            "The plus branch is the R171 Kummer locator. The conjugate branch "
            "tests target equations with -P in the selected slot. Their product "
            "returns the union; all roots retain a signed-verifier backstop. "
            "Finite extra-root counts are diagnostics, not a density theorem."
        ),
        "reverse_resultant_boundary": (
            "The reverse resultant Res_Y(W,S3) is the factored product of N "
            "quadratic S3 kernels. Its represented bivariate coefficient grid "
            "has (2N+1)^2 slots, hence N^2=B^(5/2) scale before the final "
            "self-resultant. The six controls are fully dense and full "
            "coefficient rank, but this is not a circuit or resultant lower bound."
        ),
        "surviving_interface": (
            "The surviving unproved interface must keep Res_Y(W,S3) factored "
            "and supply an algorithm for Res_Z(U(Z),product_j "
            "S3(X,Z,u_j)) modulo U(X) in softly O(n+N) work, without the "
            "N^2 reverse body, nN pair body, or candidate inversions."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_target_sign_conjugate_s3_self_resultant.cost.r172.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_target_u_v_state_exponent_B": fraction_record(Fraction(5, 4)),
        "factored_reverse_s3_resultant_state_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "batched_denominator_work_exponent_B": fraction_record(Fraction(9, 4)),
        "represented_reverse_resultant_body_exponent_B": fraction_record(
            Fraction(5, 2)
        ),
        "standard_pair_or_factor_local_work_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "represented_aggregate_output_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "expected_plus_candidate_exponent_B": fraction_record(Fraction(3, 4)),
        "expected_conjugate_candidate_upper_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "signed_candidate_verification_exponent_B": fraction_record(Fraction(2)),
        "preferred_factored_self_resultant_work_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "batched_denominator_inside_rho": True,
        "represented_reverse_resultant_strictly_inside_rho": False,
        "standard_pair_or_factor_local_work_inside_rho": False,
        "factored_self_s3_resultant_mod_u_supplied": False,
        "finite_reverse_density_receives_lower_bound_credit": False,
        "finite_conjugate_candidate_count_receives_attack_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["all_target_conjugate_s3_identities_exact"]
        and row["all_reverse_resultant_row_identities_exact"]
        and row["batched_denominator_exact_and_unit"]
        and row["plus_candidate_roots_match_r171"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_target_sign_conjugate_s3_self_resultant.controls.r172.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_target_x_coordinates_distinct": all(
            row["target_x_coordinates_distinct"] for row in rows
        ),
        "all_target_selected_x_supports_disjoint": all(
            row["target_selected_x_support_disjoint"] for row in rows
        ),
        "all_compact_target_divisors_exact": all(
            row["target_divisor_curve_residual_zero"] for row in rows
        ),
        "all_target_conjugate_s3_identities_exact": all(
            row["all_target_conjugate_s3_identities_exact"] for row in rows
        ),
        "all_reverse_resultant_row_identities_exact": all(
            row["all_reverse_resultant_row_identities_exact"] for row in rows
        ),
        "all_batched_denominators_exact_and_units": all(
            row["batched_denominator_exact_and_unit"] for row in rows
        ),
        "all_plus_candidate_roots_match_r171": all(
            row["plus_candidate_roots_match_r171"] for row in rows
        ),
        "all_reverse_resultants_full_grid_dense": all(
            row["reverse_target_resultant"]["coefficient_density"] == 1.0
            for row in rows
        ),
        "all_reverse_resultant_coefficient_matrices_full_rank": all(
            row["reverse_target_resultant"]["coefficient_matrix_rank"]
            == row["reverse_target_resultant"]["coefficient_matrix_row_count"]
            for row in rows
        ),
        "target_divisor_coefficient_slot_count": sum(
            row["target_divisor_coefficient_slot_count"] for row in rows
        ),
        "homogenized_translate_identity_count": sum(
            row["homogenized_translate_identity_count"] for row in rows
        ),
        "target_conjugate_s3_identity_count": sum(
            row["target_conjugate_s3_identity_count"] for row in rows
        ),
        "reverse_resultant_row_identity_count": sum(
            row["reverse_resultant_row_identity_count"] for row in rows
        ),
        "reverse_resultant_coefficient_slot_count": sum(
            row["reverse_target_resultant"]["coefficient_slot_count"]
            for row in rows
        ),
        "reverse_resultant_nonzero_coefficient_count": sum(
            row["reverse_target_resultant"]["nonzero_coefficient_count"]
            for row in rows
        ),
        "reverse_resultant_coefficient_rank_sum": sum(
            row["reverse_target_resultant"]["coefficient_matrix_rank"]
            for row in rows
        ),
        "plus_candidate_root_count": sum(
            row["plus_candidate_root_count"] for row in rows
        ),
        "conjugate_candidate_root_count": sum(
            row["conjugate_candidate_root_count"] for row in rows
        ),
        "symmetric_candidate_root_count": sum(
            row["symmetric_candidate_root_count"] for row in rows
        ),
        "extra_conjugate_candidate_root_count": sum(
            row["extra_conjugate_candidate_root_count"] for row in rows
        ),
        "overlap_candidate_root_count": sum(
            row["overlap_candidate_root_count"] for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_asymptotic_attack_or_lower_bound_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fifteen_source_bindings_verified": len(actual_bindings) == 15,
        "semaev_s3_primary_source_bound": True,
        "r161_signed_divisor_interface_inherited": True,
        "r171_target_norm_and_specialization_interface_inherited": True,
        "six_compact_target_u_v_divisors_constructed": len(rows) == 6,
        "all_target_divisor_curve_residuals_zero": controls[
            "all_compact_target_divisors_exact"
        ],
        "all_target_selected_x_equality_branches_split": controls[
            "all_target_selected_x_supports_disjoint"
        ],
        "homogenized_translation_identity_complete": True,
        "target_sign_conjugate_s3_identity_complete": controls[
            "all_target_conjugate_s3_identities_exact"
        ],
        "iterated_resultant_identity_complete": controls[
            "all_reverse_resultant_row_identities_exact"
        ],
        "target_x_denominator_batch_complete": controls[
            "all_batched_denominators_exact_and_units"
        ],
        "plus_candidate_roots_equal_r171": controls[
            "all_plus_candidate_roots_match_r171"
        ],
        "conjugate_candidate_branch_explicit": True,
        "signed_verifier_backstop_preserved": True,
        "six_reverse_resultant_bodies_constructed": len(rows) == 6,
        "reverse_resultant_degree_two_N_complete": all(
            row["reverse_target_resultant"]["degree_x"]
            == 2 * row["retained_target_count"]
            and row["reverse_target_resultant"]["degree_z"]
            == 2 * row["retained_target_count"]
            for row in rows
        ),
        "reverse_resultant_finite_density_recorded": controls[
            "all_reverse_resultants_full_grid_dense"
        ],
        "reverse_resultant_finite_full_rank_recorded": controls[
            "all_reverse_resultant_coefficient_matrices_full_rank"
        ],
        "represented_N2_body_B5O2_charged": True,
        "standard_nN_route_B7O2_charged": True,
        "moroz_schost_truncated_input_contract_audited": True,
        "hyun_neiger_schost_represented_resultant_contract_audited": True,
        "finite_density_and_rank_scoped_without_lower_bound_credit": True,
        "finite_conjugate_counts_scoped_without_attack_credit": True,
        "factored_self_s3_resultant_mod_u_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_TARGET_SIGN_CONJUGATE_SEMAEV_S3_IDENTITY__COMPACT_TARGET_X_"
        "DIVISOR_AND_SINGLE_DENOMINATOR_BATCH__PLUS_BRANCH_EQUALS_R171__"
        "CONJUGATE_BRANCH_EXPLICIT_WITH_SIGNED_VERIFIER__SIX_ITERATED_"
        "RESULTANT_REPLAYS__REVERSE_RESULTANT_FULL_N2_BODY_B5O2_AT_RHO__"
        "STANDARD_FINAL_NN_B7O2__FACTORED_SELF_S3_RESULTANT_MOD_U_OPEN__"
        "NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute a factored self-S3 resultant operator that emits "
        "Res_Z(U(Z),product_j S3(X,Z,u_j)) modulo U(X) in softly O(n+N) "
        "work, preferably B^(9/4+o(1)), while retaining the N-factor S3 "
        "representation and avoiding the N^2 reverse-resultant coefficient "
        "body, nN point/factor grid, candidate inversions, or unit-cost "
        "resultant and multipoint oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Remove target-y dependence by target-sign conjugation and test "
            "whether the resulting Semaev S3 self-resultant creates a sub-rho "
            "nonlocal batch interface."
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
                    "Supplies the S3 relation used in the exact conjugate identity; "
                    "it does not supply the factored self-resultant modulo U."
                ),
            },
            "moroz_schost_truncated_resultant": {
                "arxiv": "1609.04259",
                "fit": (
                    "Computes a represented bivariate resultant modulo a local "
                    "power x^k in soft-O(kd); it does not accept this factored "
                    "three-variable self-resultant or arbitrary squarefree U "
                    "remainder as its stated input/output contract."
                ),
            },
            "hyun_neiger_schost_bivariate_resultants": {
                "arxiv": "1905.04356",
                "fit": (
                    "Implements Villard-style generic bivariate resultants from "
                    "represented polynomial-matrix inputs. It does not avoid "
                    "forming the N^2 reverse body or provide the required "
                    "factored self-resultant remainder."
                ),
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "target_sign_conjugate_s3_identity_admitted": all_exact,
            "single_denominator_batch_admitted": all_exact,
            "factored_self_s3_resultant_mod_u_admitted": False,
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
        "schema": "p1553.m6_target_sign_conjugate_s3_self_resultant.frozen.r172.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The factored symmetric S3 kernel has a self-resultant remainder "
                "algorithm softly linear in n+N despite its dense represented "
                "reverse resultant."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route materializes the (2N+1)^2 reverse coefficient grid, "
                "visits nN point/factor pairs, forms an nN or n^2 polynomial "
                "matrix, inverts candidate nonunits, or assumes a resultant, "
                "norm, root, count, or multipoint oracle."
            ),
        },
        "promotion_allowed": False,
    }
    resultant = {
        "schema": "p1553.m6_target_sign_conjugate_s3_self_resultant.resultant.r172.v1",
        "homogenized_identity": theorem["homogenized_translation"],
        "conjugate_identity": theorem["semaev_conjugate_identity"],
        "iterated_resultant": theorem["compact_target_x_batch"],
        "surviving_interface": theorem["surviving_interface"],
        "controls": [
            {
                "control_id": row["control_id"],
                "target_u_sha256": row["target_u_sha256"],
                "target_v_sha256": row["target_v_sha256"],
                "reverse_target_resultant": row["reverse_target_resultant"],
                "plus_candidate_roots": row["plus_candidate_roots"],
                "conjugate_candidate_roots": row["conjugate_candidate_roots"],
                "symmetric_candidate_roots": row["symmetric_candidate_roots"],
                "row_transcript_sha256": row["row_transcript_sha256"],
            }
            for row in rows
        ],
    }
    replay = {
        "schema": "p1553.m6_target_sign_conjugate_s3_self_resultant.replay.r172.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "target_u_sha256": row["target_u_sha256"],
                "target_v_sha256": row["target_v_sha256"],
                "reverse_resultant_sha256": row["reverse_target_resultant"][
                    "term_sha256"
                ],
                "row_transcript_sha256": row["row_transcript_sha256"],
                "plus_candidate_roots": row["plus_candidate_roots"],
                "conjugate_candidate_roots": row["conjugate_candidate_roots"],
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
        "resultant": resultant,
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
    parser.add_argument("--resultant-output", type=Path, default=DEFAULT_RESULTANT)
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
        (args.resultant_output, bundle["resultant"]),
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
