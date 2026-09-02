#!/usr/bin/env python3
"""Probe S6 centered-carry rank minors on large prime-order curves."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Callable


SCHEMA = "p1553.s6_centered_carry_rank_minor_probe.r72.v1"
DECK_SIZE = 18
PREDICATE_REPLAY_SIZE = 3
RAW_MODE_RANK_UPPER_BOUND = 17
MAX_MINOR_COLUMNS = 12 * DECK_SIZE
AUXILIARY_PRIMES = (1_000_003, 1_000_033)

R71_REPORT = pathlib.Path(
    "p1553_s4_centered_carry_rank_probe_report_r71.json"
)
R71_REPORT_SHA256 = (
    "6ee62d61f6c567a5a65947b6b47026182bd961565dde3c1aee540cf3cbe5b72f"
)
R71_GATE = pathlib.Path("p1553_s4_centered_carry_rank_probe_gate_r71.md")
R71_GATE_SHA256 = (
    "f194ed996a24659e2b11d762dd471c0ff75ab98d6e64a9df8b782c676acca2f7"
)
R9_GATE = pathlib.Path("p1553_projector_trace_router_gate_r9.md")
R9_GATE_SHA256 = (
    "400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81"
)
R10_GATE = pathlib.Path("p1553_factorized_pullback_gate_r10.md")
R10_GATE_SHA256 = (
    "49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1"
)


def load_r71() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_s4_centered_carry_rank_probe_r71.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r71_for_r72", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R71 controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R71 = load_r71()
R70 = R71.R70
Point = tuple[int, int] | None


CURVES = (
    {
        "family_id": "secp256k1",
        "field_prime": int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
            16,
        ),
        "curve_a": 0,
        "curve_b": 7,
        "subgroup_order": int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            16,
        ),
        "cofactor": 1,
        "generator": (
            int(
                "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                16,
            ),
            int(
                "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
                16,
            ),
        ),
        "parameter_source": "SEC 2 secp256k1",
    },
    {
        "family_id": "nist_p256",
        "field_prime": int(
            "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
            16,
        ),
        "curve_a": int(
            "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC",
            16,
        ),
        "curve_b": int(
            "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
            16,
        ),
        "subgroup_order": int(
            "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
            16,
        ),
        "cofactor": 1,
        "generator": (
            int(
                "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
                16,
            ),
            int(
                "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
                16,
            ),
        ),
        "parameter_source": "NIST SP 800-186 P-256",
    },
    {
        "family_id": "nist_p384",
        "field_prime": int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE"
            "FFFFFFFF0000000000000000FFFFFFFF",
            16,
        ),
        "curve_a": int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE"
            "FFFFFFFF0000000000000000FFFFFFFC",
            16,
        ),
        "curve_b": int(
            "B3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE8141120314088F5013875A"
            "C656398D8A2ED19D2A85C8EDD3EC2AEF",
            16,
        ),
        "subgroup_order": int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF"
            "581A0DB248B0A77AECEC196ACCC52973",
            16,
        ),
        "cofactor": 1,
        "generator": (
            int(
                "AA87CA22BE8B05378EB1C71EF320AD746E1D3B628BA79B9859F741E082542A38"
                "5502F25DBF55296C3A545E3872760AB7",
                16,
            ),
            int(
                "3617DE4A96262C6F5D9E98BF9292DC29F8F41DBD289A147CE9DA3113B5F0B8C0"
                "0A60B1CE1D7E819D7A431D7C90EA0E5F",
                16,
            ),
        ),
        "parameter_source": "NIST SP 800-186 P-384",
    },
    {
        "family_id": "nist_p521",
        "field_prime": (1 << 521) - 1,
        "curve_a": (1 << 521) - 4,
        "curve_b": int(
            "51953EB9618E1C9A1F929A21A0B68540EEA2DA725B99B315F3B8B489918EF109"
            "E156193951EC7E937B1652C0BD3BB1BF073573DF883D2C34F1EF451FD46B503F00",
            16,
        ),
        "subgroup_order": int(
            "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
            "FA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409",
            16,
        ),
        "cofactor": 1,
        "generator": (
            int(
                "C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF606B4D3D"
                "BAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF97E7E31C2E5BD66",
                16,
            ),
            int(
                "11839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD17273E6"
                "62C97EE72995EF42640C550B9013FAD0761353C7086A272C24088BE94769FD16650",
                16,
            ),
        ),
        "parameter_source": "NIST SP 800-186 P-521",
    },
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_probable_prime(value: int) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    two_adic_order = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        two_adic_order += 1
    for base in small_primes:
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(two_adic_order - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def curve_discriminant_nonzero(curve: dict[str, Any]) -> bool:
    prime = curve["field_prime"]
    return (
        4 * pow(curve["curve_a"], 3, prime)
        + 27 * pow(curve["curve_b"], 2, prime)
    ) % prime != 0


def point_is_on_curve(point: Point, curve: dict[str, Any]) -> bool:
    if point is None:
        return True
    prime = curve["field_prime"]
    x_coordinate, y_coordinate = point
    return (
        y_coordinate * y_coordinate
        - x_coordinate**3
        - curve["curve_a"] * x_coordinate
        - curve["curve_b"]
    ) % prime == 0


def point_from_x(
    x_coordinate: int,
    sign_bit: int,
    curve: dict[str, Any],
) -> Point:
    prime = curve["field_prime"]
    rhs = (
        x_coordinate**3
        + curve["curve_a"] * x_coordinate
        + curve["curve_b"]
    ) % prime
    y_coordinate = R70.modular_square_root(rhs, prime)
    if y_coordinate is None:
        return None
    if sign_bit:
        y_coordinate = (-y_coordinate) % prime
    point = (x_coordinate, y_coordinate)
    if not point_is_on_curve(point, curve):
        raise AssertionError("large-curve point construction failed")
    return point


def polynomial_add(
    left: list[int],
    right: list[int],
    modulus: int,
) -> list[int]:
    size = max(len(left), len(right))
    return [
        (
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
        )
        % modulus
        for index in range(size)
    ]


def polynomial_subtract(
    left: list[int],
    right: list[int],
    modulus: int,
) -> list[int]:
    size = max(len(left), len(right))
    return [
        (
            (left[index] if index < len(left) else 0)
            - (right[index] if index < len(right) else 0)
        )
        % modulus
        for index in range(size)
    ]


def polynomial_scale(
    polynomial: list[int],
    scalar: int,
    modulus: int,
) -> list[int]:
    return [coefficient * scalar % modulus for coefficient in polynomial]


def polynomial_multiply(
    left: list[int],
    right: list[int],
    modulus: int,
) -> list[int]:
    product = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            product[left_index + right_index] = (
                product[left_index + right_index]
                + left_value * right_value
            ) % modulus
    return product


def polynomial_pad(polynomial: list[int], size: int) -> list[int]:
    if len(polynomial) > size:
        raise AssertionError("polynomial exceeded fixed Semaev degree")
    return polynomial + [0] * (size - len(polynomial))


def s4_polynomial_last(
    x_first: int,
    x_second: int,
    x_third: int,
    a_coefficient: int,
    b_coefficient: int,
    modulus: int,
) -> list[int]:
    """Return S4(x_first,x_second,x_third,z), ascending in z."""
    first_s3 = R71.semaev_s3_coefficients(
        x_first,
        x_second,
        {"curve_a": a_coefficient, "curve_b": b_coefficient},
    )
    a_value, b_value, c_value = (
        coefficient % modulus for coefficient in first_s3
    )
    x_third %= modulus
    a_coefficient %= modulus
    b_coefficient %= modulus
    d_polynomial = [
        x_third * x_third % modulus,
        -2 * x_third % modulus,
        1,
    ]
    e_polynomial = [
        -2 * (a_coefficient * x_third + 2 * b_coefficient) % modulus,
        -2 * (x_third * x_third + a_coefficient) % modulus,
        -2 * x_third % modulus,
    ]
    f_polynomial = [
        (a_coefficient * a_coefficient - 4 * b_coefficient * x_third)
        % modulus,
        (-2 * a_coefficient * x_third - 4 * b_coefficient) % modulus,
        x_third * x_third % modulus,
    ]
    first_term = polynomial_subtract(
        polynomial_scale(f_polynomial, a_value, modulus),
        polynomial_scale(d_polynomial, c_value, modulus),
        modulus,
    )
    second_left = polynomial_subtract(
        polynomial_scale(e_polynomial, a_value, modulus),
        polynomial_scale(d_polynomial, b_value, modulus),
        modulus,
    )
    second_right = polynomial_subtract(
        polynomial_scale(f_polynomial, b_value, modulus),
        polynomial_scale(e_polynomial, c_value, modulus),
        modulus,
    )
    resultant = polynomial_subtract(
        polynomial_multiply(first_term, first_term, modulus),
        polynomial_multiply(second_left, second_right, modulus),
        modulus,
    )
    return polynomial_pad(resultant, 5)


def determinant_mod(matrix: list[list[int]], modulus: int) -> int:
    work = [[value % modulus for value in row] for row in matrix]
    size = len(work)
    determinant = 1
    for pivot_index in range(size):
        source = next(
            (
                row
                for row in range(pivot_index, size)
                if work[row][pivot_index]
            ),
            None,
        )
        if source is None:
            return 0
        if source != pivot_index:
            work[pivot_index], work[source] = work[source], work[pivot_index]
            determinant = -determinant % modulus
        pivot = work[pivot_index][pivot_index]
        determinant = determinant * pivot % modulus
        inverse = pow(pivot, -1, modulus)
        for row in range(pivot_index + 1, size):
            multiplier = work[row][pivot_index] * inverse % modulus
            if not multiplier:
                continue
            for column in range(pivot_index, size):
                work[row][column] = (
                    work[row][column]
                    - multiplier * work[pivot_index][column]
                ) % modulus
    return determinant


def fixed_degree_resultant_mod(
    left: list[int],
    right: list[int],
    degree: int,
    modulus: int,
) -> int:
    left_descending = list(reversed(polynomial_pad(left, degree + 1)))
    right_descending = list(reversed(polynomial_pad(right, degree + 1)))
    size = 2 * degree
    matrix = []
    for shift in range(degree):
        matrix.append(
            [0] * shift
            + left_descending
            + [0] * (size - shift - degree - 1)
        )
    for shift in range(degree):
        matrix.append(
            [0] * shift
            + right_descending
            + [0] * (size - shift - degree - 1)
        )
    return determinant_mod(matrix, modulus)


def semaev_s6_mod(
    x_coordinates: tuple[int, int, int, int, int, int],
    a_coefficient: int,
    b_coefficient: int,
    modulus: int,
) -> int:
    left = s4_polynomial_last(
        x_coordinates[0],
        x_coordinates[1],
        x_coordinates[2],
        a_coefficient,
        b_coefficient,
        modulus,
    )
    right = s4_polynomial_last(
        x_coordinates[3],
        x_coordinates[4],
        x_coordinates[5],
        a_coefficient,
        b_coefficient,
        modulus,
    )
    return fixed_degree_resultant_mod(left, right, 4, modulus)


def hash_point(
    curve: dict[str, Any],
    stream: str,
    counter: int,
) -> Point:
    digest = hashlib.sha256(
        f"P1553-R72|{curve['family_id']}|{stream}|{counter}".encode()
    ).digest()
    x_coordinate = int.from_bytes(digest, "big") % curve["field_prime"]
    return point_from_x(x_coordinate, digest[0] & 1, curve)


def public_decks_and_targets(
    curve: dict[str, Any],
) -> tuple[list[list[Point]], list[dict[str, Any]]]:
    decks = []
    for deck_index in range(5):
        deck = []
        seen_x_coordinates: set[int] = set()
        counter = 0
        while len(deck) < DECK_SIZE:
            point = hash_point(curve, f"deck-{deck_index}", counter)
            counter += 1
            if point is None or point[0] in seen_x_coordinates:
                continue
            seen_x_coordinates.add(point[0])
            deck.append(point)
        decks.append(deck)

    blind_target = None
    counter = 0
    while blind_target is None:
        blind_target = hash_point(curve, "blind-target", counter)
        counter += 1

    forced_target = None
    forced_witness = None
    for indices in itertools.product(range(PREDICATE_REPLAY_SIZE), repeat=5):
        total = None
        for mode, index in enumerate(indices):
            total = R70.add(total, decks[mode][index], curve)
        candidate = R70.negate(total, curve)
        if candidate is None:
            continue
        forced_target = candidate
        forced_witness = list(indices)
        break
    if forced_target is None or forced_witness is None:
        raise AssertionError("unable to construct forced S6 target")

    return decks, [
        {
            "target_id": "blind_hash_target",
            "point": blind_target,
            "construction": "independent_sha256_x_stream",
            "forced_witness": None,
        },
        {
            "target_id": "forced_positive_target",
            "point": forced_target,
            "construction": "negative_sum_of_five_public_deck_points",
            "forced_witness": forced_witness,
        },
    ]


def signed_relation_exists(
    points: tuple[Point, Point, Point, Point, Point],
    target: Point,
    curve: dict[str, Any],
) -> bool:
    for signs in itertools.product((-1, 1), repeat=5):
        total = target
        for point, sign in zip(points, signs):
            signed_point = point if sign == 1 else R70.negate(point, curve)
            total = R70.add(total, signed_point, curve)
        if total is None:
            return True
    return False


def centered_coefficient(value: int, prime: int) -> int:
    return R71.centered_residue(value, prime)


def lift_schedules(curve: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        "least_nonnegative_coefficients": (
            curve["curve_a"],
            curve["curve_b"],
        ),
        "centered_coefficients": (
            centered_coefficient(curve["curve_a"], curve["field_prime"]),
            centered_coefficient(curve["curve_b"], curve["field_prime"]),
        ),
    }


def deterministic_column_indices(
    family_id: str,
    target_id: str,
    mode: int,
) -> list[tuple[int, int, int, int]]:
    column_count = DECK_SIZE**4
    digest = hashlib.sha256(
        f"P1553-R72-COLUMNS|{family_id}|{target_id}|{mode}".encode()
    ).digest()
    offset = int.from_bytes(digest[:8], "big") % column_count
    step = int.from_bytes(digest[8:16], "big") % column_count
    if step == 0:
        step = 1
    while math.gcd(step, column_count) != 1:
        step += 1
    columns = []
    for counter in range(min(MAX_MINOR_COLUMNS, column_count)):
        encoded = (offset + counter * step) % column_count
        coordinates = []
        for _ in range(4):
            coordinates.append(encoded % DECK_SIZE)
            encoded //= DECK_SIZE
        columns.append(tuple(coordinates))
    return columns


class IncrementalColumnRank:
    def __init__(self, modulus: int) -> None:
        self.modulus = modulus
        self.basis: dict[int, list[int]] = {}

    @property
    def rank(self) -> int:
        return len(self.basis)

    def add(self, column: list[int]) -> bool:
        vector = [value % self.modulus for value in column]
        for pivot in sorted(self.basis):
            if not vector[pivot]:
                continue
            multiplier = vector[pivot]
            vector = [
                (value - multiplier * basis_value) % self.modulus
                for value, basis_value in zip(vector, self.basis[pivot])
            ]
        pivot = next((index for index, value in enumerate(vector) if value), None)
        if pivot is None:
            return False
        inverse = pow(vector[pivot], -1, self.modulus)
        vector = [value * inverse % self.modulus for value in vector]
        self.basis[pivot] = vector
        return True


def tuple_from_column(
    mode: int,
    row_index: int,
    column: tuple[int, int, int, int],
) -> tuple[int, int, int, int, int]:
    indices = []
    column_index = 0
    for current_mode in range(5):
        if current_mode == mode:
            indices.append(row_index)
        else:
            indices.append(column[column_index])
            column_index += 1
    return tuple(indices)


def make_value_oracle(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target: Point,
) -> tuple[
    Callable[[tuple[int, int, int, int, int]], dict[str, Any]],
    dict[tuple[int, int, int, int, int], dict[str, Any]],
]:
    cache: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    prime = curve["field_prime"]
    schedules = lift_schedules(curve)

    def value(indices: tuple[int, int, int, int, int]) -> dict[str, Any]:
        if indices in cache:
            return cache[indices]
        x_coordinates = tuple(
            decks[mode][index][0] for mode, index in enumerate(indices)
        ) + (target[0],)
        field_value = semaev_s6_mod(
            x_coordinates,
            curve["curve_a"],
            curve["curve_b"],
            prime,
        )
        centered_value = R71.centered_residue(field_value, prime)
        lift_values = {}
        for lift_id, (a_lift, b_lift) in schedules.items():
            auxiliary_values = {}
            for auxiliary_prime in AUXILIARY_PRIMES:
                raw_value = semaev_s6_mod(
                    x_coordinates,
                    a_lift,
                    b_lift,
                    auxiliary_prime,
                )
                carry_value = (
                    (raw_value - centered_value)
                    * pow(prime, -1, auxiliary_prime)
                ) % auxiliary_prime
                auxiliary_values[str(auxiliary_prime)] = {
                    "raw": raw_value,
                    "carry": carry_value,
                }
            lift_values[lift_id] = auxiliary_values
        row = {
            "field_value": field_value,
            "centered_remainder": centered_value,
            "lifts": lift_values,
        }
        cache[indices] = row
        return row

    return value, cache


def rank_minor_profile(
    curve: dict[str, Any],
    target_id: str,
    mode: int,
    value_oracle: Callable[
        [tuple[int, int, int, int, int]], dict[str, Any]
    ],
) -> dict[str, Any]:
    ranks: dict[str, IncrementalColumnRank] = {}
    target_ranks: dict[str, int] = {}
    for auxiliary_prime in AUXILIARY_PRIMES:
        remainder_key = f"remainder|{auxiliary_prime}"
        ranks[remainder_key] = IncrementalColumnRank(auxiliary_prime)
        target_ranks[remainder_key] = DECK_SIZE
        for lift_id in (
            "least_nonnegative_coefficients",
            "centered_coefficients",
        ):
            for tensor_kind, target_rank in (
                ("raw", RAW_MODE_RANK_UPPER_BOUND),
                ("carry", DECK_SIZE),
            ):
                key = f"{tensor_kind}|{lift_id}|{auxiliary_prime}"
                ranks[key] = IncrementalColumnRank(auxiliary_prime)
                target_ranks[key] = target_rank

    columns_scanned = 0
    witness_columns: dict[str, list[int]] = {}
    for column_number, column in enumerate(
        deterministic_column_indices(curve["family_id"], target_id, mode)
    ):
        rows = [
            value_oracle(tuple_from_column(mode, row_index, column))
            for row_index in range(DECK_SIZE)
        ]
        for auxiliary_prime in AUXILIARY_PRIMES:
            remainder_key = f"remainder|{auxiliary_prime}"
            if ranks[remainder_key].rank < target_ranks[remainder_key]:
                if ranks[remainder_key].add(
                    [
                        row["centered_remainder"] % auxiliary_prime
                        for row in rows
                    ]
                ):
                    witness_columns.setdefault(remainder_key, []).append(
                        column_number
                    )
            for lift_id in (
                "least_nonnegative_coefficients",
                "centered_coefficients",
            ):
                for tensor_kind in ("raw", "carry"):
                    key = f"{tensor_kind}|{lift_id}|{auxiliary_prime}"
                    if ranks[key].rank >= target_ranks[key]:
                        continue
                    if ranks[key].add(
                        [
                            row["lifts"][lift_id][str(auxiliary_prime)][
                                tensor_kind
                            ]
                            for row in rows
                        ]
                    ):
                        witness_columns.setdefault(key, []).append(column_number)
        columns_scanned = column_number + 1
        if all(ranks[key].rank >= target_ranks[key] for key in ranks):
            break

    remainder_profiles = {}
    lift_profiles = {}
    for auxiliary_prime in AUXILIARY_PRIMES:
        key = f"remainder|{auxiliary_prime}"
        remainder_profiles[str(auxiliary_prime)] = {
            "rank": ranks[key].rank,
            "full_row_rank": ranks[key].rank == DECK_SIZE,
            "independent_column_numbers": witness_columns.get(key, []),
        }
    for lift_id in (
        "least_nonnegative_coefficients",
        "centered_coefficients",
    ):
        lift_profiles[lift_id] = {}
        for tensor_kind in ("raw", "carry"):
            prime_profiles = {}
            for auxiliary_prime in AUXILIARY_PRIMES:
                key = f"{tensor_kind}|{lift_id}|{auxiliary_prime}"
                prime_profiles[str(auxiliary_prime)] = {
                    "rank": ranks[key].rank,
                    "target_rank": target_ranks[key],
                    "target_rank_reached": (
                        ranks[key].rank == target_ranks[key]
                    ),
                    "full_row_rank": ranks[key].rank == DECK_SIZE,
                    "independent_column_numbers": witness_columns.get(key, []),
                }
            lift_profiles[lift_id][tensor_kind] = prime_profiles
    return {
        "mode": mode,
        "row_count": DECK_SIZE,
        "ambient_column_count": DECK_SIZE**4,
        "columns_scanned": columns_scanned,
        "centered_remainder": remainder_profiles,
        "lifts": lift_profiles,
    }


def replay_predicate(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target: Point,
) -> dict[str, Any]:
    mismatch_count = 0
    root_count = 0
    permutation_mismatch_count = 0
    tuple_count = PREDICATE_REPLAY_SIZE**5
    for indices in itertools.product(range(PREDICATE_REPLAY_SIZE), repeat=5):
        points = tuple(decks[mode][index] for mode, index in enumerate(indices))
        x_coordinates = tuple(point[0] for point in points) + (target[0],)
        value = semaev_s6_mod(
            x_coordinates,
            curve["curve_a"],
            curve["curve_b"],
            curve["field_prime"],
        )
        relation_exists = signed_relation_exists(points, target, curve)
        polynomial_is_zero = value == 0
        root_count += relation_exists
        mismatch_count += polynomial_is_zero != relation_exists
        permuted_value = semaev_s6_mod(
            (
                x_coordinates[1],
                x_coordinates[0],
                x_coordinates[2],
                x_coordinates[4],
                x_coordinates[3],
                x_coordinates[5],
            ),
            curve["curve_a"],
            curve["curve_b"],
            curve["field_prime"],
        )
        permutation_mismatch_count += permuted_value != value
    return {
        "prefix_size": PREDICATE_REPLAY_SIZE,
        "tuple_count": tuple_count,
        "relation_root_count": root_count,
        "s6_vs_signed_group_relation_mismatches": mismatch_count,
        "sampled_symmetry_mismatches": permutation_mismatch_count,
    }


def analyze_target(
    curve: dict[str, Any],
    decks: list[list[Point]],
    target_spec: dict[str, Any],
) -> dict[str, Any]:
    target = target_spec["point"]
    if target is None:
        raise AssertionError("target may not be identity")
    predicate_replay = replay_predicate(curve, decks, target)
    value_oracle, cache = make_value_oracle(curve, decks, target)
    mode_profiles = [
        rank_minor_profile(
            curve,
            target_spec["target_id"],
            mode,
            value_oracle,
        )
        for mode in range(5)
    ]
    forced_witness_verified = True
    if target_spec["forced_witness"] is not None:
        indices = tuple(target_spec["forced_witness"])
        points = tuple(decks[mode][index] for mode, index in enumerate(indices))
        forced_witness_verified = (
            value_oracle(indices)["field_value"] == 0
            and signed_relation_exists(points, target, curve)
        )
    return {
        "target_id": target_spec["target_id"],
        "target_point": list(target),
        "target_construction": target_spec["construction"],
        "forced_witness": target_spec["forced_witness"],
        "forced_witness_verified": forced_witness_verified,
        "predicate_replay": predicate_replay,
        "mode_rank_minor_profiles": mode_profiles,
        "unique_s6_tuple_evaluations": len(cache),
    }


def analyze_family(curve: dict[str, Any]) -> dict[str, Any]:
    decks, targets = public_decks_and_targets(curve)
    generator = curve["generator"]
    return {
        "family": {
            key: value
            for key, value in curve.items()
            if key != "generator"
        },
        "parameter_checks": {
            "field_prime_probable_prime": is_probable_prime(
                curve["field_prime"]
            ),
            "subgroup_order_probable_prime": is_probable_prime(
                curve["subgroup_order"]
            ),
            "cofactor_one": curve["cofactor"] == 1,
            "curve_discriminant_nonzero": curve_discriminant_nonzero(curve),
            "generator_on_curve": point_is_on_curve(generator, curve),
            "generator_has_stated_order": (
                R70.scalar_mul(curve["subgroup_order"], generator, curve)
                is None
            ),
            "r9_no_wrap_threshold_at_probe_size": (
                32 * DECK_SIZE**4 < curve["field_prime"]
            ),
            "probe_size_below_asymptotic_factor_bound": (
                DECK_SIZE**5 < curve["subgroup_order"]
            ),
        },
        "public_source": {
            "deck_count": len(decks),
            "deck_size": DECK_SIZE,
            "deck_points": [
                [list(point) for point in deck if point is not None]
                for deck in decks
            ],
            "scalar_labels_consumed": False,
            "construction": (
                "domain-separated_sha256_x_stream_then_canonical_square_root"
            ),
        },
        "lift_schedules": {
            lift_id: {"curve_a_lift": values[0], "curve_b_lift": values[1]}
            for lift_id, values in lift_schedules(curve).items()
        },
        "targets": [
            analyze_target(curve, decks, target_spec) for target_spec in targets
        ],
    }


def run() -> dict[str, Any]:
    bindings = {
        "r71_s4_centered_carry_report": {
            "path": str(R71_REPORT),
            "sha256": R71_REPORT_SHA256,
        },
        "r71_s4_centered_carry_gate": {
            "path": str(R71_GATE),
            "sha256": R71_GATE_SHA256,
        },
        "r9_projector_trace_router": {
            "path": str(R9_GATE),
            "sha256": R9_GATE_SHA256,
        },
        "r10_factorized_pullback_gate": {
            "path": str(R10_GATE),
            "sha256": R10_GATE_SHA256,
        },
    }
    for binding in bindings.values():
        if sha256_file(pathlib.Path(binding["path"])) != binding["sha256"]:
            raise AssertionError(f"source binding mismatch: {binding['path']}")

    family_rows = [analyze_family(dict(curve)) for curve in CURVES]
    target_rows = [
        target for family in family_rows for target in family["targets"]
    ]
    mode_rows = [
        mode
        for target in target_rows
        for mode in target["mode_rank_minor_profiles"]
    ]
    parameter_checks = [
        check
        for family in family_rows
        for check in family["parameter_checks"].values()
    ]
    checks = {
        "four_large_prime_order_curve_families": len(family_rows) == 4
        and all(parameter_checks),
        "all_public_sources_are_scalar_blind": all(
            not family["public_source"]["scalar_labels_consumed"]
            for family in family_rows
        ),
        "all_s6_predicates_match_signed_group_relations": all(
            target["predicate_replay"][
                "s6_vs_signed_group_relation_mismatches"
            ]
            == 0
            for target in target_rows
        ),
        "all_sampled_s6_symmetries_match": all(
            target["predicate_replay"]["sampled_symmetry_mismatches"] == 0
            for target in target_rows
        ),
        "all_forced_witnesses_verify": all(
            target["forced_witness_verified"] for target in target_rows
        ),
        "all_centered_remainder_modes_are_full_under_both_primes": all(
            all(
                row["full_row_rank"]
                for row in mode["centered_remainder"].values()
            )
            for mode in mode_rows
        ),
        "all_raw_modes_reach_degree_bound_under_both_primes": all(
            all(
                profile["rank"] == RAW_MODE_RANK_UPPER_BOUND
                for profile in mode["lifts"][lift_id]["raw"].values()
            )
            for mode in mode_rows
            for lift_id in mode["lifts"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R72 exact probe failed: {checks}")

    all_carry_modes_full = all(
        all(
            profile["full_row_rank"]
            for profile in mode["lifts"][lift_id]["carry"].values()
        )
        for mode in mode_rows
        for lift_id in mode["lifts"]
    )
    carry_profile_count = (
        len(mode_rows) * 2 * len(AUXILIARY_PRIMES)
    )
    full_carry_profile_count = sum(
        profile["full_row_rank"]
        for mode in mode_rows
        for lift_id in mode["lifts"]
        for profile in mode["lifts"][lift_id]["carry"].values()
    )
    obligations = {
        "exact_scalar_blind_five_deck_source": checks[
            "all_public_sources_are_scalar_blind"
        ],
        "four_large_prime_order_curve_families": checks[
            "four_large_prime_order_curve_families"
        ],
        "exact_positive_and_blind_s6_predicate_replay": checks[
            "all_s6_predicates_match_signed_group_relations"
        ]
        and checks["all_forced_witnesses_verify"],
        "s6_centered_carry_rank_stays_at_most_raw_degree_bound": (
            not all_carry_modes_full
        ),
        "branch_complete_trace_contraction_supplied": False,
        "fresh_target_source_recovery_below_rho": False,
        "factor_log_and_descent_path_complete": False,
    }
    return {
        "schema": SCHEMA,
        "classification": (
            "S6_CANONICAL_AND_CENTERED_COEFFICIENT_CARRIES_HAVE_FULL_B18_MODE_RANK"
            if all_carry_modes_full
            else "S6_CENTERED_CARRY_RANK_IS_LIFT_OR_FAMILY_MIXED"
        ),
        "candidate": {
            "name": "s6_centered_carry_natural_lift_schedules",
            "definition": (
                "Use the exact split resultant "
                "Res_z(S4(x1,x2,x3,z),S4(x4,x5,xR,z)), center its field "
                "value modulo p, and certify mode-rank minors of the induced "
                "carry under least-nonnegative and centered curve coefficients."
            ),
            "scope": (
                "This is a rank-minor test of two named integer lifts. It does "
                "not represent the Fermat projector, compute the R9 trace count, "
                "or lower-bound non-CP and noncanonical S6 interfaces."
            ),
        },
        "parameters": {
            "deck_size": DECK_SIZE,
            "predicate_replay_prefix_size": PREDICATE_REPLAY_SIZE,
            "raw_s6_mode_rank_upper_bound": RAW_MODE_RANK_UPPER_BOUND,
            "maximum_minor_columns_scanned": MAX_MINOR_COLUMNS,
            "auxiliary_primes": list(AUXILIARY_PRIMES),
            "lift_schedules": [
                "least_nonnegative_coefficients",
                "centered_coefficients",
            ],
        },
        "family_results": family_rows,
        "aggregate": {
            "family_count": len(family_rows),
            "target_instance_count": len(target_rows),
            "mode_instance_count": len(mode_rows),
            "carry_rank_profile_count": carry_profile_count,
            "full_carry_rank_profile_count": full_carry_profile_count,
            "all_carry_modes_full_under_both_primes_and_both_lifts": (
                all_carry_modes_full
            ),
            "certified_rational_cp_rank_lower_bound_per_carry": (
                DECK_SIZE if all_carry_modes_full else None
            ),
            "raw_rational_mode_rank_exact": (
                RAW_MODE_RANK_UPPER_BOUND
                if checks["all_raw_modes_reach_degree_bound_under_both_primes"]
                else None
            ),
            "predicate_tuple_count": sum(
                target["predicate_replay"]["tuple_count"]
                for target in target_rows
            ),
            "unique_s6_tuple_evaluation_count": sum(
                target["unique_s6_tuple_evaluations"]
                for target in target_rows
            ),
        },
        "source_bindings": bindings,
        "rank_interpretation": {
            "raw_upper_bound": (
                "Fixed-target S6 has degree 16 in each x argument, so every "
                "raw legal-grid mode flattening has rational rank at most 17."
            ),
            "minor_certificate": (
                "A rank-r minor modulo either auxiliary prime proves rational "
                "matrix rank and rational CP rank at least r for the exact "
                "integer value tensor of that named lift."
            ),
            "decisive_gap": (
                "At B=18, raw rank 17 versus carry rank 18 distinguishes "
                "centering-induced rank expansion from the raw degree ceiling."
            ),
            "lift_boundary": (
                "Carry rank changes when the integer lift changes. Testing two "
                "natural coefficient schedules rejects only those schedules."
            ),
        },
        "semantic_dedup": {
            "new_idea_created": False,
            "owners": [
                "P1553 R7-R8 centered-carry backend",
                "P1553 R9 projector-trace router",
                "P1553 R10 factorized pullback",
                "IDEA-049 and IDEA-198",
            ],
            "disposition": (
                "Use R72 only to close the S6-specific low-carry-rank exception "
                "for the two frozen resultant lifts. Route surviving work to the "
                "R9 exact trace contraction or another explicit non-CP interface."
            ),
        },
        "checks": checks,
        "admission": {
            "lane_admitted": all(obligations.values()),
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "obligations": obligations,
        },
        "result": {
            "s6_rank_minor_probe_complete": True,
            "named_s6_carry_low_rank_survives": not all_carry_modes_full,
            "projector_trace_contraction_supplied": False,
            "factor_log_solve_complete": False,
            "fresh_target_descent_complete": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "next_action": (
            "Close explicit centered-carry CP construction for the two natural "
            "S6 resultant lifts if every B=18 carry mode is full. Return to R9 "
            "and R10: require one exact non-CP balanced trace contraction that "
            "computes restricted root counts and a dyadic source without a B^3 "
            "triple table, and reject any restatement of the carry tensor."
        ),
        "limits": [
            "The rank certificate uses size-18 scalar-blind decks, not asymptotically growing deck families.",
            "The four curves are standardized prime-order controls rather than a random-curve distribution.",
            "Only the least-nonnegative and coefficient-centered resultant lifts are tested.",
            "The experiment does not construct the projector, root count, factor logs, or descent.",
            "No runtime lower bound, Shoup-bound improvement, or ECDLP breakthrough is claimed.",
        ],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
        ),
    )
    args = parser.parse_args()
    payload = run()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = payload["aggregate"]
    print(
        f"output={args.output} families={aggregate['family_count']} "
        f"carry_full={aggregate['full_carry_rank_profile_count']}/"
        f"{aggregate['carry_rank_profile_count']} "
        f"admitted={payload['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
