#!/usr/bin/env python3
"""Test whether the exact 2x2 S3 pencil yields a compact resultant transfer."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_s3_determinantal_transfer_noncommutativity.r173.v1"

R172_PRODUCER = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_r172.py"
R172_REPORT = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_report_r172.json"
R172_FROZEN = ROOT / "frozen_m6_target_sign_conjugate_s3_self_resultant.json"
R172_COST = ROOT / "m6_target_sign_conjugate_s3_self_resultant_cost_ledger.json"
R172_REPLAY = ROOT / "m6_target_sign_conjugate_s3_self_resultant_replay.json"
R172_CONTROLS = ROOT / "m6_target_sign_conjugate_s3_self_resultant_controls.json"
R172_RESULTANT = ROOT / "target_conjugate_s3_factored_self_resultant_r172.json"
R172_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_target_sign_conjugate_s3_self_resultant_probe_r172.py"
R172_GATE = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_gate_r172.md"
R172_PARENT = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_parent_report_r172.yaml"
SEMAEV_PAPER = ROOT / "references/semaev_summation_polynomials_2004_031.ps"
HYUN_NEIGER_SCHOST_PAPER = ROOT / "references/hyun_neiger_schost_bivariate_resultants_1905.04356.pdf"

SOURCE_BINDINGS = (
    ("r172_producer", R172_PRODUCER, "297dc3d3195926341fbbd79c7d0346e59bfcb1cd721e5b4f0d4b3e2cb4d615c8"),
    ("r172_report", R172_REPORT, "66f986b68c6d31576c688da4509d828139275331009dba4bb05abd45aef5ce5e"),
    ("r172_frozen", R172_FROZEN, "5a6938348733ec3d103ec2044198bcb5aef06c2119da01d45d5f746ed9e737bf"),
    ("r172_cost", R172_COST, "b8ceb6c4d9cc9d6365ce861ba17b6ac631c0e8f4f951d56dc4d94d0e351602f0"),
    ("r172_replay", R172_REPLAY, "7975e14f92f9fedd73193ace641d3ea6e2d97387b512be16ad9b01133fb15920"),
    ("r172_controls", R172_CONTROLS, "d176c75580c5810c37e6e9debb0a2a91636fe17f28bbec0b516f2a3cd164bc99"),
    ("r172_resultant", R172_RESULTANT, "281fcf996ceddeeef5b5da57930e6b5dfc44da6da743719726fc2ebffd7f299d"),
    ("r172_test", R172_TEST, "b9b663f3e502971257047a6b39d09a4d969030ebfd928927b14be4f474385dd1"),
    ("r172_gate", R172_GATE, "14fb9459c75510f26177e2ef795266a825f58c2cdd11797d6ec6e60bd31dc951"),
    ("r172_parent", R172_PARENT, "45e476d650cb7c54bb3f9859175cded032d823881d76c706f60722c3e9d6a1bb"),
    ("semaev_2004", SEMAEV_PAPER, "991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df"),
    ("hyun_neiger_schost_2019", HYUN_NEIGER_SCHOST_PAPER, "32b73cf0ca7172bdec0f8f1b256adda628a86d6dd7eee8e07e8644e35a9f16f3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_report_r173.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_s3_determinantal_transfer_noncommutativity.json"
DEFAULT_COST = ROOT / "m6_s3_determinantal_transfer_noncommutativity_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_s3_determinantal_transfer_noncommutativity_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_s3_determinantal_transfer_noncommutativity_controls.json"
DEFAULT_TRANSFER = ROOT / "s3_determinantal_transfer_noncommutativity_r173.json"

Bivariate = dict[tuple[int, int], int]
Matrix2 = tuple[tuple[Bivariate, Bivariate], tuple[Bivariate, Bivariate]]
NumericMatrix2 = tuple[tuple[int, int], tuple[int, int]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R172 = load_module("p1553_r172_for_r173", R172_PRODUCER)
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
        raise AssertionError(f"R173 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def poly_normalize(poly: Bivariate, prime: int) -> Bivariate:
    return {key: value % prime for key, value in poly.items() if value % prime}


def poly_add(left: Bivariate, right: Bivariate, prime: int) -> Bivariate:
    result = dict(left)
    for key, value in right.items():
        result[key] = (result.get(key, 0) + value) % prime
    return poly_normalize(result, prime)


def poly_neg(poly: Bivariate, prime: int) -> Bivariate:
    return poly_normalize({key: -value for key, value in poly.items()}, prime)


def poly_sub(left: Bivariate, right: Bivariate, prime: int) -> Bivariate:
    return poly_add(left, poly_neg(right, prime), prime)


def poly_mul(left: Bivariate, right: Bivariate, prime: int) -> Bivariate:
    return R172.bivariate_mul(left, right, prime)


def zero_matrix() -> Matrix2:
    return (({}, {}), ({}, {}))


def identity_matrix() -> Matrix2:
    return (({(0, 0): 1}, {}), ({}, {(0, 0): 1}))


def matrix_add(left: Matrix2, right: Matrix2, prime: int) -> Matrix2:
    return (
        (
            poly_add(left[0][0], right[0][0], prime),
            poly_add(left[0][1], right[0][1], prime),
        ),
        (
            poly_add(left[1][0], right[1][0], prime),
            poly_add(left[1][1], right[1][1], prime),
        ),
    )


def matrix_neg(matrix: Matrix2, prime: int) -> Matrix2:
    return (
        (poly_neg(matrix[0][0], prime), poly_neg(matrix[0][1], prime)),
        (poly_neg(matrix[1][0], prime), poly_neg(matrix[1][1], prime)),
    )


def matrix_sub(left: Matrix2, right: Matrix2, prime: int) -> Matrix2:
    return matrix_add(left, matrix_neg(right, prime), prime)


def matrix_mul(left: Matrix2, right: Matrix2, prime: int) -> Matrix2:
    return (
        (
            poly_add(
                poly_mul(left[0][0], right[0][0], prime),
                poly_mul(left[0][1], right[1][0], prime),
                prime,
            ),
            poly_add(
                poly_mul(left[0][0], right[0][1], prime),
                poly_mul(left[0][1], right[1][1], prime),
                prime,
            ),
        ),
        (
            poly_add(
                poly_mul(left[1][0], right[0][0], prime),
                poly_mul(left[1][1], right[1][0], prime),
                prime,
            ),
            poly_add(
                poly_mul(left[1][0], right[0][1], prime),
                poly_mul(left[1][1], right[1][1], prime),
                prime,
            ),
        ),
    )


def matrix_det(matrix: Matrix2, prime: int) -> Bivariate:
    return poly_sub(
        poly_mul(matrix[0][0], matrix[1][1], prime),
        poly_mul(matrix[0][1], matrix[1][0], prime),
        prime,
    )


def matrix_is_zero(matrix: Matrix2) -> bool:
    return all(not entry for row in matrix for entry in row)


def target_pencil(target_x: int, curve: dict[str, Any]) -> Matrix2:
    prime = int(curve["field_prime"])
    a = int(curve["curve_a"])
    b = int(curve["curve_b"])
    u = target_x % prime
    diagonal = poly_normalize(
        {(1, 0): u, (0, 0): -a, (1, 1): 1, (0, 1): u}, prime
    )
    upper = poly_normalize({(0, 0): 2 * b, (1, 1): 2 * u}, prime)
    lower = poly_normalize({(1, 0): 2, (0, 0): 2 * u, (0, 1): 2}, prime)
    return ((diagonal, upper), (lower, diagonal))


def pencil_parts_at_x(
    x_value: int, target_x: int, curve: dict[str, Any]
) -> tuple[NumericMatrix2, NumericMatrix2]:
    prime = int(curve["field_prime"])
    a = int(curve["curve_a"])
    b = int(curve["curve_b"])
    x_value %= prime
    target_x %= prime
    a_matrix = (
        ((x_value * target_x - a) % prime, 2 * b % prime),
        (2 * (x_value + target_x) % prime, (x_value * target_x - a) % prime),
    )
    b_matrix = (
        ((x_value + target_x) % prime, 2 * x_value * target_x % prime),
        (2, (x_value + target_x) % prime),
    )
    return a_matrix, b_matrix


def numeric_matrix_add(
    left: NumericMatrix2, right: NumericMatrix2, prime: int
) -> NumericMatrix2:
    return tuple(
        tuple((left[row][column] + right[row][column]) % prime for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def numeric_matrix_sub(
    left: NumericMatrix2, right: NumericMatrix2, prime: int
) -> NumericMatrix2:
    return tuple(
        tuple((left[row][column] - right[row][column]) % prime for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def numeric_matrix_scale(
    scalar: int, matrix: NumericMatrix2, prime: int
) -> NumericMatrix2:
    return tuple(
        tuple(scalar * matrix[row][column] % prime for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def numeric_matrix_mul(
    left: NumericMatrix2, right: NumericMatrix2, prime: int
) -> NumericMatrix2:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(2))
            % prime
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


def numeric_pencil_value(
    a_matrix: NumericMatrix2,
    b_matrix: NumericMatrix2,
    z_value: int,
    prime: int,
) -> NumericMatrix2:
    return numeric_matrix_add(
        a_matrix, numeric_matrix_scale(z_value, b_matrix, prime), prime
    )


def bivariate_record(
    poly: Bivariate, prime: int, degree_bound: int
) -> dict[str, Any]:
    matrix = [
        [poly.get((x_degree, z_degree), 0) for z_degree in range(degree_bound + 1)]
        for x_degree in range(degree_bound + 1)
    ]
    terms = [
        {
            "x_degree": x_degree,
            "z_degree": z_degree,
            "coefficient": poly[(x_degree, z_degree)],
        }
        for x_degree, z_degree in sorted(poly)
    ]
    slot_count = (degree_bound + 1) ** 2
    return {
        "degree_bound_x": degree_bound,
        "degree_bound_z": degree_bound,
        "actual_degree_x": max((key[0] for key in poly), default=-1),
        "actual_degree_z": max((key[1] for key in poly), default=-1),
        "coefficient_slot_count": slot_count,
        "nonzero_coefficient_count": len(poly),
        "coefficient_density": len(poly) / slot_count,
        "coefficient_matrix_rank": R172.modular_rank(matrix, prime),
        "terms": terms,
        "term_sha256": sha256_json(terms),
    }


def matrix_record(
    matrix: Matrix2, prime: int, degree_bound: int
) -> dict[str, Any]:
    entries = {
        f"m{row}{column}": bivariate_record(
            matrix[row][column], prime, degree_bound
        )
        for row in range(2)
        for column in range(2)
    }
    return {
        "degree_bound_each_axis": degree_bound,
        "entry_count": 4,
        "coefficient_slot_count": sum(
            entry["coefficient_slot_count"] for entry in entries.values()
        ),
        "nonzero_coefficient_count": sum(
            entry["nonzero_coefficient_count"] for entry in entries.values()
        ),
        "coefficient_rank_sum": sum(
            entry["coefficient_matrix_rank"] for entry in entries.values()
        ),
        "entries": entries,
        "matrix_sha256": sha256_json(entries),
    }


def terms_to_poly(record: dict[str, Any]) -> Bivariate:
    return {
        (int(term["x_degree"]), int(term["z_degree"])): int(term["coefficient"])
        for term in record["terms"]
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r172 = R172.finite_control(curve, seed)
    _, divisor, target_records = R172.R167.R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    target_x_values = [int(target[0]) for target in targets]
    n = len(selected)
    target_count = len(targets)

    determinant_identity_count = 0
    discriminant_identity_count = 0
    pencil_pair_order_identity_count = 0
    pencil_pair_noncommuting_count = 0
    exceptional_commutator_evaluation_count = 0
    for selected_point in selected:
        x_value = int(selected_point[0])
        for target in targets:
            u_value = int(target[0])
            factor = R172.s3_bivariate_factor(u_value, curve)
            pencil = target_pencil(u_value, curve)
            if matrix_det(pencil, prime) != factor:
                raise AssertionError("S3 determinant polynomial identity failed")
            coefficients = [0, 0, 0]
            for (x_degree, z_degree), coefficient in factor.items():
                coefficients[z_degree] = (
                    coefficients[z_degree]
                    + coefficient * pow(x_value, x_degree, prime)
                ) % prime
            discriminant = (
                coefficients[1] * coefficients[1]
                - 4 * coefficients[2] * coefficients[0]
            ) % prime
            expected_discriminant = (
                16
                * R172.curve_rhs(x_value, curve)
                * R172.curve_rhs(u_value, curve)
            ) % prime
            if discriminant != expected_discriminant:
                raise AssertionError("S3 z-discriminant identity failed")
            discriminant_identity_count += 1

            a_matrix, b_matrix = pencil_parts_at_x(x_value, u_value, curve)
            commutator = numeric_matrix_sub(
                numeric_matrix_mul(a_matrix, b_matrix, prime),
                numeric_matrix_mul(b_matrix, a_matrix, prime),
                prime,
            )
            delta = 4 * (
                int(curve["curve_b"])
                - x_value * u_value * (x_value + u_value)
            ) % prime
            expected_commutator = ((delta, 0), (0, -delta % prime))
            if commutator != expected_commutator:
                raise AssertionError("pencil commutator formula failed")
            if delta == 0:
                exceptional_commutator_evaluation_count += 1

            for right_point in selected:
                z_value = int(right_point[0])
                pencil_value = numeric_pencil_value(
                    a_matrix, b_matrix, z_value, prime
                )
                determinant = (
                    pencil_value[0][0] * pencil_value[1][1]
                    - pencil_value[0][1] * pencil_value[1][0]
                ) % prime
                if determinant != R172.semaev_s3(
                    x_value, z_value, u_value, curve
                ):
                    raise AssertionError("evaluated S3 determinant failed")
                determinant_identity_count += 1

            for left_index, left_point in enumerate(selected):
                z_left = int(left_point[0])
                left_matrix = numeric_pencil_value(
                    a_matrix, b_matrix, z_left, prime
                )
                for right_point in selected[left_index + 1 :]:
                    z_right = int(right_point[0])
                    right_matrix = numeric_pencil_value(
                        a_matrix, b_matrix, z_right, prime
                    )
                    observed = numeric_matrix_sub(
                        numeric_matrix_mul(left_matrix, right_matrix, prime),
                        numeric_matrix_mul(right_matrix, left_matrix, prime),
                        prime,
                    )
                    expected = numeric_matrix_scale(
                        z_right - z_left, commutator, prime
                    )
                    if observed != expected:
                        raise AssertionError("ordered pencil pair identity failed")
                    pencil_pair_order_identity_count += 1
                    if observed != ((0, 0), (0, 0)):
                        pencil_pair_noncommuting_count += 1

    forward = identity_matrix()
    for u_value in target_x_values:
        forward = matrix_mul(forward, target_pencil(u_value, curve), prime)
    reverse = identity_matrix()
    for u_value in reversed(target_x_values):
        reverse = matrix_mul(reverse, target_pencil(u_value, curve), prime)
    forward_record = matrix_record(forward, prime, target_count)
    reverse_record = matrix_record(reverse, prime, target_count)
    changed_entries = sum(
        forward[row][column] != reverse[row][column]
        for row in range(2)
        for column in range(2)
    )
    if target_count > 1 and changed_entries == 0:
        raise AssertionError("target transfer unexpectedly became order invariant")

    reverse_resultant = terms_to_poly(r172["reverse_target_resultant"])
    if matrix_det(forward, prime) != reverse_resultant:
        raise AssertionError("forward transfer determinant lost reverse resultant")
    if matrix_det(reverse, prime) != reverse_resultant:
        raise AssertionError("reverse transfer determinant lost reverse resultant")

    target_pair_count = 0
    target_pair_noncommuting_count = 0
    for left_index, left_u in enumerate(target_x_values):
        left_matrix = target_pencil(left_u, curve)
        for right_u in target_x_values[left_index + 1 :]:
            right_matrix = target_pencil(right_u, curve)
            commutator = matrix_sub(
                matrix_mul(left_matrix, right_matrix, prime),
                matrix_mul(right_matrix, left_matrix, prime),
                prime,
            )
            target_pair_count += 1
            if not matrix_is_zero(commutator):
                target_pair_noncommuting_count += 1

    return {
        "control_id": f"{curve['family_id']}_s3_determinantal_transfer_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": n,
        "retained_target_count": target_count,
        "elementary_symmetric_determinant_identity_exact": True,
        "s3_z_discriminant_identity_exact": True,
        "determinant_identity_count": determinant_identity_count,
        "discriminant_identity_count": discriminant_identity_count,
        "pencil_pair_order_identity_count": pencil_pair_order_identity_count,
        "pencil_pair_noncommuting_count": pencil_pair_noncommuting_count,
        "exceptional_commutator_evaluation_count": exceptional_commutator_evaluation_count,
        "target_factor_pair_count": target_pair_count,
        "target_factor_noncommuting_pair_count": target_pair_noncommuting_count,
        "all_target_factor_pairs_noncommuting": (
            target_pair_noncommuting_count == target_pair_count
        ),
        "forward_reverse_target_product_changed_entry_count": changed_entries,
        "forward_reverse_target_products_differ": changed_entries > 0,
        "forward_transfer": forward_record,
        "reverse_transfer_sha256": reverse_record["matrix_sha256"],
        "forward_transfer_determinant_equals_r172_reverse_resultant": True,
        "reverse_transfer_determinant_equals_r172_reverse_resultant": True,
        "r172_reverse_resultant_sha256": r172["reverse_target_resultant"][
            "term_sha256"
        ],
        "candidate_discrete_log_oracle_consumed": False,
        "finite_noncommutativity_receives_general_lower_bound_credit": False,
        "finite_transfer_density_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "elementary_symmetric_s3": (
            "For s1=X+Z+u, s2=XZ+Xu+Zu, and s3=XZu, S3(X,Z,u) "
            "=(s2-a)^2-4*s1*(s3+b). This is the determinant of the 2 by 2 "
            "matrix [[s2-a,2(s3+b)],[2s1,s2-a]]."
        ),
        "affine_pencil": (
            "Writing the determinant matrix as A(X,u)+Z B(X,u), A has rows "
            "[Xu-a,2b] and [2(X+u),Xu-a], while B has rows "
            "[X+u,2Xu] and [2,X+u]."
        ),
        "commutator": (
            "The exact commutator [A,B] is diagonal with entries delta and "
            "-delta, where delta=4(b-Xu(X+u)). Hence M(z1)M(z2)-M(z2)M(z1) "
            "=(z2-z1)[A,B], and the naive root-matrix product is generically "
            "order sensitive rather than a symmetric function of the roots of U."
        ),
        "separable_discriminant": (
            "The discriminant of S3(X,Z,u) as a quadratic in Z is "
            "16(X^3+aX+b)(u^3+au+b). On the two compact curve divisors its "
            "square root is 4 V(X) V_T(u); diagonalizing the pencil therefore "
            "recovers the two target-sign branches already exposed in R172."
        ),
        "determinant_product": (
            "For any explicit ordering of target x-coordinates, det(product_j "
            "M_{u_j}(X,Z))=product_j S3(X,Z,u_j)=Res_Y(W(Y),S3(X,Z,Y)). "
            "The determinant is order invariant although the matrix product is not."
        ),
        "scope": (
            "This refutes only the naive order-independent 2 by 2 transfer and "
            "charges its represented product. It is not a resultant lower bound; "
            "a custom commutative, transposed, or implicit squarefree-modulus "
            "algorithm remains possible."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_s3_determinantal_transfer_noncommutativity.cost.r173.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "factored_pencil_input_state_exponent_B": fraction_record(Fraction(5, 4)),
        "represented_ordered_matrix_transfer_body_exponent_B": fraction_record(
            Fraction(5, 2)
        ),
        "represented_scalar_determinant_body_exponent_B": fraction_record(
            Fraction(5, 2)
        ),
        "standard_outer_root_or_factor_grid_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "represented_aggregate_output_exponent_B": fraction_record(Fraction(9, 4)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "represented_ordered_transfer_strictly_inside_rho": False,
        "standard_outer_route_inside_rho": False,
        "order_independent_two_by_two_root_transfer_supplied": False,
        "commutative_diagonalization_beyond_r172_sign_split_supplied": False,
        "custom_implicit_resultant_lower_bound_claimed": False,
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
        row["elementary_symmetric_determinant_identity_exact"]
        and row["s3_z_discriminant_identity_exact"]
        and row["forward_transfer_determinant_equals_r172_reverse_resultant"]
        and row["reverse_transfer_determinant_equals_r172_reverse_resultant"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_s3_determinantal_transfer_noncommutativity.controls.r173.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_elementary_symmetric_determinant_identities_exact": all(
            row["elementary_symmetric_determinant_identity_exact"] for row in rows
        ),
        "all_s3_z_discriminant_identities_exact": all(
            row["s3_z_discriminant_identity_exact"] for row in rows
        ),
        "all_transfer_determinants_equal_r172_reverse_resultants": all(
            row["forward_transfer_determinant_equals_r172_reverse_resultant"]
            and row["reverse_transfer_determinant_equals_r172_reverse_resultant"]
            for row in rows
        ),
        "all_forward_reverse_target_products_differ": all(
            row["forward_reverse_target_products_differ"] for row in rows
        ),
        "all_target_factor_pairs_noncommuting": all(
            row["all_target_factor_pairs_noncommuting"] for row in rows
        ),
        "determinant_identity_count": sum(
            row["determinant_identity_count"] for row in rows
        ),
        "discriminant_identity_count": sum(
            row["discriminant_identity_count"] for row in rows
        ),
        "pencil_pair_order_identity_count": sum(
            row["pencil_pair_order_identity_count"] for row in rows
        ),
        "pencil_pair_noncommuting_count": sum(
            row["pencil_pair_noncommuting_count"] for row in rows
        ),
        "exceptional_commutator_evaluation_count": sum(
            row["exceptional_commutator_evaluation_count"] for row in rows
        ),
        "target_factor_pair_count": sum(
            row["target_factor_pair_count"] for row in rows
        ),
        "target_factor_noncommuting_pair_count": sum(
            row["target_factor_noncommuting_pair_count"] for row in rows
        ),
        "represented_transfer_coefficient_slot_count": sum(
            row["forward_transfer"]["coefficient_slot_count"] for row in rows
        ),
        "represented_transfer_nonzero_coefficient_count": sum(
            row["forward_transfer"]["nonzero_coefficient_count"] for row in rows
        ),
        "represented_transfer_coefficient_rank_sum": sum(
            row["forward_transfer"]["coefficient_rank_sum"] for row in rows
        ),
        "finite_noncommutativity_receives_general_lower_bound_credit": False,
        "finite_transfer_density_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r172_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "elementary_symmetric_s3_identity_complete": controls[
            "all_elementary_symmetric_determinant_identities_exact"
        ],
        "affine_two_by_two_pencil_complete": all_exact,
        "separable_discriminant_identity_complete": controls[
            "all_s3_z_discriminant_identities_exact"
        ],
        "pencil_commutator_formula_complete": all_exact,
        "ordered_pair_formula_complete": all_exact,
        "target_factor_noncommutativity_observed": controls[
            "all_target_factor_pairs_noncommuting"
        ],
        "forward_reverse_product_order_dependence_observed": controls[
            "all_forward_reverse_target_products_differ"
        ],
        "transfer_determinant_replays_r172": controls[
            "all_transfer_determinants_equal_r172_reverse_resultants"
        ],
        "represented_matrix_N2_body_charged": True,
        "standard_outer_nN_route_charged": True,
        "finite_observation_scoped_without_lower_bound_credit": True,
        "r172_sign_split_equivalence_stated": True,
        "order_independent_two_by_two_transfer_complete": False,
        "factored_self_s3_resultant_mod_u_complete": False,
        "custom_resultant_lower_bound_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_ELEMENTARY_SYMMETRIC_S3_DETERMINANT_AND_SEPARABLE_"
        "DISCRIMINANT__EXACT_2X2_AFFINE_PENCIL__GENERIC_PENCIL_AND_TARGET_"
        "FACTOR_NONCOMMUTATIVITY__ORDERED_TRANSFER_DETERMINANT_REPLAYS_R172__"
        "REPRESENTED_MATRIX_BODY_N2_B5O2_AT_RHO__DIAGONALIZATION_RETURNS_R172_"
        "SIGN_SPLIT__NAIVE_2X2_SYMMETRIC_TRANSFER_CLOSED__CUSTOM_IMPLICIT_"
        "RESULTANT_OPEN__NO_GENERAL_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Exploit the discriminant square root 4*V(X)*V_T(u) directly: formulate "
        "the two diagonal target-sign branches as compact divisor pushforwards "
        "modulo arbitrary squarefree U, then test a transposed modular-composition "
        "or multipoint implementation with fully charged precomputation and work "
        "strictly below B^(5/2), rejecting N^2 coefficients, nN pairs, target-"
        "dependent setup, and unit-cost norm or resultant oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Test whether the elementary-symmetric 2 by 2 determinantal form of "
            "Semaev S3 supplies the softly linear factored self-resultant operator "
            "left open by R172."
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
                    "Supplies S3. The determinant, discriminant, and commutator "
                    "identities here are exact algebraic rewrites, not an algorithm "
                    "claimed by the paper."
                ),
            },
            "hyun_neiger_schost_bivariate_resultants": {
                "arxiv": "1905.04356",
                "fit": (
                    "Its represented polynomial-matrix setting does not turn this "
                    "order-sensitive product into an order-independent linear-state "
                    "transfer; represented multiplication still exposes N^2 slots."
                ),
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "determinantal_identity_admitted": all_exact,
            "naive_order_independent_two_by_two_transfer_admitted": False,
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
        "schema": "p1553.m6_s3_determinantal_transfer_noncommutativity.frozen.r173.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The 2 by 2 S3 determinant pencil composes over the roots or target "
                "support through an order-independent transfer of softly linear size."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The transfer is order sensitive, materializes Theta(N^2) field "
                "coefficients, visits nN pairs, or merely diagonalizes into the two "
                "already charged R172 target-sign branches."
            ),
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_s3_determinantal_transfer_noncommutativity.replay.r173.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "r172_reverse_resultant_sha256": row[
                    "r172_reverse_resultant_sha256"
                ],
                "forward_transfer_sha256": row["forward_transfer"][
                    "matrix_sha256"
                ],
                "reverse_transfer_sha256": row["reverse_transfer_sha256"],
                "changed_entry_count": row[
                    "forward_reverse_target_product_changed_entry_count"
                ],
            }
            for row in rows
        ],
    }
    transfer = {
        "schema": "p1553.m6_s3_determinantal_transfer_noncommutativity.transfer.r173.v1",
        "elementary_symmetric_identity": theorem["elementary_symmetric_s3"],
        "affine_pencil": theorem["affine_pencil"],
        "commutator": theorem["commutator"],
        "discriminant": theorem["separable_discriminant"],
        "controls": [
            {
                "control_id": row["control_id"],
                "retained_target_count": row["retained_target_count"],
                "forward_transfer": row["forward_transfer"],
                "reverse_transfer_sha256": row["reverse_transfer_sha256"],
                "changed_entry_count": row[
                    "forward_reverse_target_product_changed_entry_count"
                ],
                "r172_reverse_resultant_sha256": row[
                    "r172_reverse_resultant_sha256"
                ],
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
        "transfer": transfer,
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
    parser.add_argument("--transfer-output", type=Path, default=DEFAULT_TRANSFER)
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
        (args.transfer_output, bundle["transfer"]),
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
