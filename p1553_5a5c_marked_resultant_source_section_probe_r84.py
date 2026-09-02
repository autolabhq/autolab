#!/usr/bin/env python3
"""Audit an explicit marked-resultant source section for R82's 5A+5C join."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_marked_resultant_source_section.r84.v1"
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
ATOM_A_EXPONENT = 2 / 5
ATOM_C_EXPONENT = 3 / 5
LEFT_A_COUNT = 2
LEFT_C_COUNT = 3
RIGHT_A_COUNT = 3
RIGHT_C_COUNT = 2
GCD_SAMPLE_COUNT = 4

R83_REPORT = pathlib.Path(
    "p1553_5a5c_coordinate_filtration_probe_report_r83.json"
)
R83_REPORT_SHA256 = (
    "1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c"
)
R83_GATE = pathlib.Path("p1553_5a5c_coordinate_filtration_probe_gate_r83.md")
R83_GATE_SHA256 = (
    "7907df5232c7a6322c797ec33b7600042e223c2c21cb5c832b285995a77fafdf"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
P1510_PRODUCER_SHA256 = (
    "20c7e26c55801aba57d2095254823f20de66ca499c8281ac761603391e8f0d68"
)
P1510_AUDIT_SHA256 = (
    "e89c11c5a57ae2ac90f4d42b3d33558cb1c1ba1765d7409a7940accba3098452"
)
P1511_ACTUAL_GATE_SHA256 = (
    "4b393c9805e5d7bc008451a9e275e5b41cf917fd9d87c7f99237783a0f4440d6"
)
P1510_PRODUCER = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1510_global_truncated_marked_resultant_compiler.md"
)
P1510_AUDIT = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1510_global_truncated_marked_resultant_compiler_audit.md"
)
P1511_GATE = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1511_factorized_semijoin_gate.md"
)
R16_GATE = pathlib.Path("p1553_block_common_right_factor_compiler_gate_r16.md")
R16_GATE_SHA256 = (
    "9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c"
)

Point = tuple[int, int] | None
Fp2 = tuple[int, int]
Source = tuple[tuple[int, ...], tuple[int, ...]]


def load_r82() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_cartesian_sum_compact_divisor_probe_r82.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r82_for_r84", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R82 geometry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R82 = load_r82()
R70 = R82.R70


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def point_sort_key(point: Point) -> tuple[int, int, int]:
    return (-1, 0, 0) if point is None else (0, point[0], point[1])


def support_exponent(size: int, base_size: int) -> float:
    if size <= 1 or base_size <= 1:
        return 0.0
    return math.log(size) / math.log(base_size)


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R83_REPORT: R83_REPORT_SHA256,
        R83_GATE: R83_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        P1510_PRODUCER: P1510_PRODUCER_SHA256,
        P1510_AUDIT: P1510_AUDIT_SHA256,
        P1511_GATE: P1511_ACTUAL_GATE_SHA256,
        R16_GATE: R16_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R84 source binding mismatch: {failures}")
    return actual


def row_digest(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(compact_json(row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def least_nonsquare(prime: int) -> int:
    for value in range(2, prime):
        if pow(value, (prime - 1) // 2, prime) == prime - 1:
            return value
    raise AssertionError("odd prime has no nonsquare")


def f2_zero() -> Fp2:
    return (0, 0)


def f2_one() -> Fp2:
    return (1, 0)


def f2_add(left: Fp2, right: Fp2, prime: int) -> Fp2:
    return ((left[0] + right[0]) % prime, (left[1] + right[1]) % prime)


def f2_neg(value: Fp2, prime: int) -> Fp2:
    return ((-value[0]) % prime, (-value[1]) % prime)


def f2_sub(left: Fp2, right: Fp2, prime: int) -> Fp2:
    return f2_add(left, f2_neg(right, prime), prime)


def f2_mul(left: Fp2, right: Fp2, prime: int, nonsquare: int) -> Fp2:
    return (
        (
            left[0] * right[0]
            + nonsquare * left[1] * right[1]
        )
        % prime,
        (left[0] * right[1] + left[1] * right[0]) % prime,
    )


def f2_scale(value: Fp2, scalar: int, prime: int) -> Fp2:
    return (value[0] * scalar % prime, value[1] * scalar % prime)


def f2_inv(value: Fp2, prime: int, nonsquare: int) -> Fp2:
    norm = (
        value[0] * value[0] - nonsquare * value[1] * value[1]
    ) % prime
    if norm == 0:
        raise ZeroDivisionError("zero in Fp2")
    inverse_norm = pow(norm, prime - 2, prime)
    return (
        value[0] * inverse_norm % prime,
        -value[1] * inverse_norm % prime,
    )


def f2_div(
    numerator: Fp2,
    denominator: Fp2,
    prime: int,
    nonsquare: int,
) -> Fp2:
    return f2_mul(
        numerator,
        f2_inv(denominator, prime, nonsquare),
        prime,
        nonsquare,
    )


def point_key(point: Point) -> Fp2:
    # (0,0) is reserved for O. It is not an affine point on y^2=x^3+1.
    return (0, 0) if point is None else point


def poly_trim(coefficients: Sequence[Fp2]) -> list[Fp2]:
    output = list(coefficients)
    while len(output) > 1 and output[-1] == f2_zero():
        output.pop()
    return output


def poly_add(
    left: Sequence[Fp2],
    right: Sequence[Fp2],
    prime: int,
) -> list[Fp2]:
    width = max(len(left), len(right))
    output = [f2_zero()] * width
    for index in range(width):
        output[index] = f2_add(
            left[index] if index < len(left) else f2_zero(),
            right[index] if index < len(right) else f2_zero(),
            prime,
        )
    return poly_trim(output)


def poly_mul(
    left: Sequence[Fp2],
    right: Sequence[Fp2],
    prime: int,
    nonsquare: int,
) -> list[Fp2]:
    output = [f2_zero()] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        if left_value == f2_zero():
            continue
        for right_index, right_value in enumerate(right):
            if right_value == f2_zero():
                continue
            product = f2_mul(
                left_value,
                right_value,
                prime,
                nonsquare,
            )
            slot = left_index + right_index
            output[slot] = f2_add(output[slot], product, prime)
    return poly_trim(output)


def poly_scale(
    value: Sequence[Fp2],
    scalar: Fp2,
    prime: int,
    nonsquare: int,
) -> list[Fp2]:
    return poly_trim(
        [
            f2_mul(coefficient, scalar, prime, nonsquare)
            for coefficient in value
        ]
    )


def poly_eval(
    coefficients: Sequence[Fp2],
    value: Fp2,
    prime: int,
    nonsquare: int,
) -> Fp2:
    result = f2_zero()
    for coefficient in reversed(coefficients):
        result = f2_add(
            f2_mul(result, value, prime, nonsquare),
            coefficient,
            prime,
        )
    return result


def poly_derivative(
    coefficients: Sequence[Fp2],
    prime: int,
) -> list[Fp2]:
    if len(coefficients) <= 1:
        return [f2_zero()]
    return poly_trim(
        [
            f2_scale(coefficients[index], index, prime)
            for index in range(1, len(coefficients))
        ]
    )


def poly_from_roots(
    roots: Sequence[Fp2],
    prime: int,
    nonsquare: int,
) -> list[Fp2]:
    coefficients = [f2_one()]
    for root in roots:
        coefficients = poly_mul(
            coefficients,
            [f2_neg(root, prime), f2_one()],
            prime,
            nonsquare,
        )
    return coefficients


def synthetic_division(
    coefficients: Sequence[Fp2],
    root: Fp2,
    prime: int,
    nonsquare: int,
) -> tuple[list[Fp2], Fp2]:
    if len(coefficients) <= 1:
        raise ValueError("cannot divide a constant polynomial")
    quotient = [f2_zero()] * (len(coefficients) - 1)
    quotient[-1] = coefficients[-1]
    for index in range(len(coefficients) - 2, 0, -1):
        quotient[index - 1] = f2_add(
            coefficients[index],
            f2_mul(
                root,
                quotient[index],
                prime,
                nonsquare,
            ),
            prime,
        )
    remainder = f2_add(
        coefficients[0],
        f2_mul(root, quotient[0], prime, nonsquare),
        prime,
    )
    return poly_trim(quotient), remainder


def poly_divmod(
    numerator: Sequence[Fp2],
    denominator: Sequence[Fp2],
    prime: int,
    nonsquare: int,
) -> tuple[list[Fp2], list[Fp2]]:
    remainder = poly_trim(numerator)
    divisor = poly_trim(denominator)
    if divisor == [f2_zero()]:
        raise ZeroDivisionError("polynomial division by zero")
    if len(remainder) < len(divisor):
        return [f2_zero()], remainder
    quotient = [f2_zero()] * (len(remainder) - len(divisor) + 1)
    divisor_lead_inverse = f2_inv(divisor[-1], prime, nonsquare)
    while len(remainder) >= len(divisor) and remainder != [f2_zero()]:
        shift = len(remainder) - len(divisor)
        lead = f2_mul(
            remainder[-1],
            divisor_lead_inverse,
            prime,
            nonsquare,
        )
        quotient[shift] = lead
        for index, coefficient in enumerate(divisor):
            slot = shift + index
            remainder[slot] = f2_sub(
                remainder[slot],
                f2_mul(lead, coefficient, prime, nonsquare),
                prime,
            )
        remainder = poly_trim(remainder)
    return poly_trim(quotient), remainder


def poly_gcd_monic(
    left: Sequence[Fp2],
    right: Sequence[Fp2],
    prime: int,
    nonsquare: int,
) -> list[Fp2]:
    first = poly_trim(left)
    second = poly_trim(right)
    while second != [f2_zero()]:
        _, remainder = poly_divmod(
            first,
            second,
            prime,
            nonsquare,
        )
        first, second = second, remainder
    return poly_scale(
        first,
        f2_inv(first[-1], prime, nonsquare),
        prime,
        nonsquare,
    )


def selector_interpolant(
    roots: Sequence[Fp2],
    codes: Sequence[int],
    prime: int,
    nonsquare: int,
    root_polynomial: Sequence[Fp2] | None = None,
) -> list[Fp2]:
    if len(roots) != len(codes) or not roots:
        raise ValueError("selector roots and codes must be nonempty and aligned")
    root_poly = (
        list(root_polynomial)
        if root_polynomial is not None
        else poly_from_roots(roots, prime, nonsquare)
    )
    derivative = poly_derivative(root_poly, prime)
    output = [f2_zero()] * len(roots)
    for root, code in zip(roots, codes):
        quotient, remainder = synthetic_division(
            root_poly,
            root,
            prime,
            nonsquare,
        )
        if remainder != f2_zero():
            raise AssertionError("root polynomial synthetic division failed")
        denominator = poly_eval(derivative, root, prime, nonsquare)
        scale = f2_div(
            (code % prime, 0),
            denominator,
            prime,
            nonsquare,
        )
        scaled = poly_scale(quotient, scale, prime, nonsquare)
        for index, coefficient in enumerate(scaled):
            output[index] = f2_add(output[index], coefficient, prime)
    return poly_trim(output)


def polynomial_receipt(coefficients: Sequence[Fp2]) -> dict[str, Any]:
    nonzero = sum(coefficient != f2_zero() for coefficient in coefficients)
    return {
        "degree": len(coefficients) - 1,
        "coefficient_count": len(coefficients),
        "nonzero_coefficient_count": nonzero,
        "coefficient_density": nonzero / len(coefficients),
        "coefficient_sha256": row_digest(coefficients),
        "leading_coefficient": list(coefficients[-1]),
    }


def encode_source(source: Source, size_a: int, size_c: int) -> int:
    code = 0
    for index in source[0]:
        code = code * size_a + index
    for index in source[1]:
        code = code * size_c + index
    return code


def decode_source(
    code: int,
    size_a: int,
    size_c: int,
    count_a: int,
    count_c: int,
) -> Source:
    right = [0] * count_c
    left = [0] * count_a
    value = code
    for index in range(count_c - 1, -1, -1):
        right[index] = value % size_c
        value //= size_c
    for index in range(count_a - 1, -1, -1):
        left[index] = value % size_a
        value //= size_a
    if value:
        raise AssertionError("source code exceeds frozen mixed radix")
    return tuple(left), tuple(right)


def source_endpoint(
    source: Source,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R82.add_many(
        [
            *(atoms_a[index] for index in source[0]),
            *(atoms_c[index] for index in source[1]),
        ],
        curve,
    )


def multiset_endpoint_section(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    count_a: int,
    count_c: int,
    curve: dict[str, Any],
) -> tuple[
    collections.Counter[Point],
    dict[Point, Source],
    int,
]:
    histogram: collections.Counter[Point] = collections.Counter()
    first: dict[Point, Source] = {}
    source_count = 0
    for left in itertools.combinations_with_replacement(
        range(len(atoms_a)),
        count_a,
    ):
        for right in itertools.combinations_with_replacement(
            range(len(atoms_c)),
            count_c,
        ):
            source = (left, right)
            endpoint = source_endpoint(
                source,
                atoms_a,
                atoms_c,
                curve,
            )
            histogram[endpoint] += 1
            first.setdefault(endpoint, source)
            source_count += 1
    if source_count != sum(histogram.values()):
        raise AssertionError("multiset endpoint source count drifted")
    return histogram, first, source_count


def build_side_section(
    first: dict[Point, Source],
    histogram: collections.Counter[Point],
    source_count: int,
    count_a: int,
    count_c: int,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
    nonsquare: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prime = curve["field_prime"]
    size_a = len(atoms_a)
    size_c = len(atoms_c)
    points = sorted(first, key=point_sort_key)
    roots = [point_key(point) for point in points]
    if len(set(roots)) != len(points):
        raise AssertionError("Fp2 point key is not injective")
    sources = [first[point] for point in points]
    codes = [
        encode_source(source, size_a, size_c)
        for source in sources
    ]
    if len(set(codes)) != len(codes):
        # Endpoint collisions can select distinct codes, but one code must not
        # describe two different canonical endpoints.
        code_to_endpoint: dict[int, Point] = {}
        for point, code in zip(points, codes):
            previous = code_to_endpoint.setdefault(code, point)
            if previous != point:
                raise AssertionError("one source code names two endpoints")
    root_polynomial = poly_from_roots(roots, prime, nonsquare)
    selector = selector_interpolant(
        roots,
        codes,
        prime,
        nonsquare,
        root_polynomial,
    )
    all_roots_vanish = all(
        poly_eval(root_polynomial, root, prime, nonsquare) == f2_zero()
        for root in roots
    )
    all_selectors_exact = True
    all_sources_replay = True
    for point, root, source, code in zip(points, roots, sources, codes):
        selected = poly_eval(selector, root, prime, nonsquare)
        decoded = decode_source(
            selected[0],
            size_a,
            size_c,
            count_a,
            count_c,
        )
        all_selectors_exact &= selected == (code, 0) and decoded == source
        all_sources_replay &= source_endpoint(
            decoded,
            atoms_a,
            atoms_c,
            curve,
        ) == point
    max_code = size_a**count_a * size_c**count_c - 1
    section = {
        "source_arity": {
            "atom_a_count": count_a,
            "atom_c_count": count_c,
        },
        "unordered_multiset_source_count": source_count,
        "distinct_endpoint_count": len(points),
        "source_collision_excess": source_count - len(points),
        "maximum_endpoint_source_fiber": max(histogram.values(), default=0),
        "point_key": {
            "field": "F_p[w]/(w^2-nonsquare)",
            "encoding": "O maps to 0; affine (x,y) maps to x+w*y",
            "injective_on_all_subgroup_points": True,
            "candidate_scalar_labels_consumed": False,
        },
        "packed_source_code": {
            "radices": [*[size_a] * count_a, *[size_c] * count_c],
            "maximum_possible_code": max_code,
            "code_fits_base_field": max_code < prime,
            "jointly_coupled": True,
        },
        "root_polynomial": polynomial_receipt(root_polynomial),
        "source_selector_interpolant": polynomial_receipt(selector),
        "all_roots_vanish": all_roots_vanish,
        "all_selector_values_exact": all_selectors_exact,
        "all_selected_sources_group_replay": all_sources_replay,
        "explicit_fp2_coefficient_count": (
            len(root_polynomial) + len(selector)
        ),
        "explicit_base_field_word_count": 2 * (
            len(root_polynomial) + len(selector)
        ),
    }
    private = {
        "points": points,
        "roots": roots,
        "sources": sources,
        "codes": codes,
        "root_polynomial": root_polynomial,
        "selector": selector,
        "point_to_index": {
            point: index for index, point in enumerate(points)
        },
    }
    return section, private


def translated_right_polynomial(
    target: Point,
    right_points: Sequence[Point],
    curve: dict[str, Any],
    nonsquare: int,
) -> list[Fp2]:
    roots = [
        point_key(
            R70.add(
                target,
                R70.negate(point, curve),
                curve,
            )
        )
        for point in right_points
    ]
    return poly_from_roots(
        roots,
        curve["field_prime"],
        nonsquare,
    )


def exact_all_target_join(
    left_private: dict[str, Any],
    right_private: dict[str, Any],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
    nonsquare: int,
) -> dict[str, Any]:
    left_points: list[Point] = left_private["points"]
    right_points: list[Point] = right_private["points"]
    target_first: dict[Point, tuple[int, int]] = {}
    target_histogram: collections.Counter[Point] = collections.Counter()
    pair_count = 0
    samples: list[Point] = []
    for left_index, left_point in enumerate(left_points):
        for right_index, right_point in enumerate(right_points):
            target = R70.add(left_point, right_point, curve)
            pair_count += 1
            target_histogram[target] += 1
            if target not in target_first:
                target_first[target] = (left_index, right_index)
                if len(samples) < GCD_SAMPLE_COUNT:
                    samples.append(target)
    direct_histogram, direct_first, direct_source_count = (
        multiset_endpoint_section(
            atoms_a,
            atoms_c,
            LEFT_A_COUNT + RIGHT_A_COUNT,
            LEFT_C_COUNT + RIGHT_C_COUNT,
            curve,
        )
    )
    direct_target_set_matches = set(direct_first) == set(target_first)
    if not direct_target_set_matches:
        raise AssertionError("split join lost a direct 5A+5C target")
    all_target_sources_exact = True
    for target, (left_index, right_index) in target_first.items():
        left_source = decode_source(
            left_private["codes"][left_index],
            len(atoms_a),
            len(atoms_c),
            LEFT_A_COUNT,
            LEFT_C_COUNT,
        )
        right_source = decode_source(
            right_private["codes"][right_index],
            len(atoms_a),
            len(atoms_c),
            RIGHT_A_COUNT,
            RIGHT_C_COUNT,
        )
        left_endpoint = source_endpoint(
            left_source,
            atoms_a,
            atoms_c,
            curve,
        )
        right_endpoint = source_endpoint(
            right_source,
            atoms_a,
            atoms_c,
            curve,
        )
        all_target_sources_exact &= (
            left_endpoint == left_points[left_index]
            and right_endpoint == right_points[right_index]
            and R70.add(left_endpoint, right_endpoint, curve) == target
        )

    prime = curve["field_prime"]
    left_point_to_index = left_private["point_to_index"]
    sample_receipts = []
    for target in samples:
        translated = translated_right_polynomial(
            target,
            right_points,
            curve,
            nonsquare,
        )
        gcd = poly_gcd_monic(
            left_private["root_polynomial"],
            translated,
            prime,
            nonsquare,
        )
        matches = []
        for right_index, right_point in enumerate(right_points):
            left_point = R70.add(
                target,
                R70.negate(right_point, curve),
                curve,
            )
            left_index = left_point_to_index.get(left_point)
            if left_index is not None:
                matches.append((left_index, right_index))
        expected_roots = [
            left_private["roots"][left_index]
            for left_index, _ in matches
        ]
        all_expected_roots_vanish = all(
            poly_eval(gcd, root, prime, nonsquare) == f2_zero()
            for root in expected_roots
        )
        if len(gcd) - 1 != len(matches) or not all_expected_roots_vanish:
            raise AssertionError("coefficient gcd disagrees with exact join")
        selected_left, selected_right = matches[0]
        left_root = left_private["roots"][selected_left]
        left_code = poly_eval(
            left_private["selector"],
            left_root,
            prime,
            nonsquare,
        )
        right_root = right_private["roots"][selected_right]
        right_code = poly_eval(
            right_private["selector"],
            right_root,
            prime,
            nonsquare,
        )
        coupled_source_exact = (
            left_code == (left_private["codes"][selected_left], 0)
            and right_code == (right_private["codes"][selected_right], 0)
        )
        sample_receipts.append(
            {
                "target": point_json(target),
                "translated_right_polynomial": polynomial_receipt(translated),
                "coefficient_gcd": polynomial_receipt(gcd),
                "exact_join_match_count": len(matches),
                "gcd_degree_equals_match_count": (
                    len(gcd) - 1 == len(matches)
                ),
                "all_expected_roots_vanish": all_expected_roots_vanish,
                "one_jointly_coupled_source_exact": coupled_source_exact,
            }
        )

    target_rows = (
        (
            point_json(target),
            left_private["codes"][indices[0]],
            right_private["codes"][indices[1]],
        )
        for target, indices in sorted(
            target_first.items(),
            key=lambda item: point_sort_key(item[0]),
        )
    )
    return {
        "side_endpoint_pair_count": pair_count,
        "attained_target_count": len(target_first),
        "target_collision_excess": pair_count - len(target_first),
        "maximum_split_decomposition_multiplicity": max(
            target_histogram.values(),
            default=0,
        ),
        "direct_5a5c_multiset_source_count": direct_source_count,
        "direct_5a5c_distinct_target_count": len(direct_histogram),
        "direct_5a5c_source_collision_excess": (
            direct_source_count - len(direct_histogram)
        ),
        "split_target_set_equals_direct_5a5c_target_set": (
            direct_target_set_matches
        ),
        "one_canonical_joint_source_per_attained_target": True,
        "all_attained_target_sources_group_replay": all_target_sources_exact,
        "canonical_target_source_sha256": row_digest(target_rows),
        "coefficient_gcd_samples": sample_receipts,
        "all_sampled_coefficient_gcds_exact": all(
            sample["gcd_degree_equals_match_count"]
            and sample["all_expected_roots_vanish"]
            and sample["one_jointly_coupled_source_exact"]
            for sample in sample_receipts
        ),
        "candidate_credit": False,
        "audit_construction_cost": (
            "explicit Cartesian product of both endpoint supports"
        ),
    }


def p1510_output_sensitive_control() -> dict[str, Any]:
    return {
        "producer_sha256": P1510_PRODUCER_SHA256,
        "independent_audit_sha256": P1510_AUDIT_SHA256,
        "verified_frozen_family": (
            "degree-two marked resultant from pure-left, pure-right, and "
            "r^2 constant-size pair resultants"
        ),
        "work": "O(r^2 polylog r)",
        "peak_coefficient_state": "O(r^2)",
        "explicit_output": "15 dense coefficient polynomials of degree O(r^2)",
        "output_sensitive_exception_preserved": True,
        "r84_transfer": (
            "An output-linear compiler still emits B^2.6 left or B^2.4 "
            "right endpoint/source coefficients in R84's explicit grammar."
        ),
        "not_claimed": [
            "a lower bound on target-uniform pre-coefficient circuits",
            "a lower bound on arithmetic circuits or arbitrary resultants",
            "failure of P1510 on its frozen family",
        ],
    }


def asymptotic_cost_ledger() -> dict[str, Any]:
    left_exponent = (
        LEFT_A_COUNT * ATOM_A_EXPONENT
        + LEFT_C_COUNT * ATOM_C_EXPONENT
    )
    right_exponent = (
        RIGHT_A_COUNT * ATOM_A_EXPONENT
        + RIGHT_C_COUNT * ATOM_C_EXPONENT
    )
    return {
        "caps": {
            "setup_state_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
            "fresh_workspace_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "split": {
            "left": "2A+3C",
            "right": "3A+2C",
            "left_multiset_source_exponent_B": left_exponent,
            "right_multiset_source_exponent_B": right_exponent,
            "source_pair_exponent_B": left_exponent + right_exponent,
        },
        "explicit_radical_endpoint_polynomial": {
            "prospective_generic_left_degree_exponent_B": left_exponent,
            "prospective_generic_right_degree_exponent_B": right_exponent,
            "smaller_side_degree_exponent_B": min(
                left_exponent,
                right_exponent,
            ),
            "smaller_side_inside_setup_cap": (
                min(left_exponent, right_exponent)
                <= SETUP_STATE_CAP_EXPONENT
            ),
            "smaller_side_inside_fresh_cap": (
                min(left_exponent, right_exponent)
                <= ONLINE_CAP_EXPONENT
            ),
            "source_product_degrees_before_radicalization": {
                "left_exponent_B": left_exponent,
                "right_exponent_B": right_exponent,
            },
            "finite_support_is_not_asymptotic_injectivity_theorem": True,
        },
        "packed_source_interpolant": {
            "one_fp2_value_returns_all_indices_jointly": True,
            "left_coefficient_exponent_B": left_exponent,
            "right_coefficient_exponent_B": right_exponent,
            "explicit_coefficient_body_inside_setup_cap": False,
        },
        "fresh_target_translated_right_polynomial": {
            "work_and_output_exponent_B": right_exponent,
            "inside_fresh_cap": right_exponent <= ONLINE_CAP_EXPONENT,
        },
        "all_target_audit": {
            "work_exponent_B": left_exponent + right_exponent,
            "candidate_credit": False,
            "purpose": "finite completeness and containment audit only",
        },
        "generic_collision_baseline": {
            "work_exponent_B": 2.5,
            "work_exponent_N": 0.5,
        },
        "p1510_control": p1510_output_sensitive_control(),
        "p1511_boundary": {
            "bound_gate_sha256": P1511_ACTUAL_GATE_SHA256,
            "standard_provenance_leaves_before_gcd_are_charged": True,
            "target_uniform_representation_before_leaf_emission_open": True,
        },
        "r16_boundary": {
            "bound_gate_sha256": R16_GATE_SHA256,
            "common_right_factor_or_compact_identity_exception_open": True,
        },
        "explicit_marked_resultant_section_inside_caps": False,
        "fresh_query_inside_caps": False,
        "failure_reasons": [
            "the smaller explicit side has B^2.4 coefficients, above B^2.25 setup",
            "a fresh translated-right coefficient body costs B^2.4, above B^1.25",
            "the exact all-target containment audit enumerates B^5 side pairs",
            "the passing finite selector is an over-cap representation, not a solver",
        ],
    }


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    prime = curve["field_prime"]
    nonsquare = least_nonsquare(prime)
    left_histogram, left_first, left_source_count = (
        multiset_endpoint_section(
            atoms_a,
            atoms_c,
            LEFT_A_COUNT,
            LEFT_C_COUNT,
            curve,
        )
    )
    right_histogram, right_first, right_source_count = (
        multiset_endpoint_section(
            atoms_a,
            atoms_c,
            RIGHT_A_COUNT,
            RIGHT_C_COUNT,
            curve,
        )
    )
    left_section, left_private = build_side_section(
        left_first,
        left_histogram,
        left_source_count,
        LEFT_A_COUNT,
        LEFT_C_COUNT,
        atoms_a,
        atoms_c,
        curve,
        nonsquare,
    )
    right_section, right_private = build_side_section(
        right_first,
        right_histogram,
        right_source_count,
        RIGHT_A_COUNT,
        RIGHT_C_COUNT,
        atoms_a,
        atoms_c,
        curve,
        nonsquare,
    )
    join = exact_all_target_join(
        left_private,
        right_private,
        atoms_a,
        atoms_c,
        curve,
        nonsquare,
    )
    base_size = len(factors)
    return {
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": curve["subgroup_order"],
        "offset": offset,
        "factor_base_size_B": base_size,
        "atom_a_size": len(atoms_a),
        "atom_c_size": len(atoms_c),
        "factor_base_injective": geometry["factor_base_injective"],
        "fp2_nonsquare": nonsquare,
        "scalar_labels_consumed": False,
        "left_section": {
            **left_section,
            "finite_endpoint_support_exponent_B": support_exponent(
                left_section["distinct_endpoint_count"],
                base_size,
            ),
        },
        "right_section": {
            **right_section,
            "finite_endpoint_support_exponent_B": support_exponent(
                right_section["distinct_endpoint_count"],
                base_size,
            ),
        },
        "joint_target_replay": join,
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = R82.FAMILIES,
    offsets: Sequence[int] = R82.INSTANCE_OFFSETS,
) -> dict[str, dict[str, Any]]:
    verified_bindings = verify_source_bindings()
    instances = [
        analyze_instance(dict(curve), offset)
        for curve in families
        for offset in offsets
    ]
    all_side_sections_exact = all(
        instance["left_section"]["all_roots_vanish"]
        and instance["left_section"]["all_selector_values_exact"]
        and instance["left_section"]["all_selected_sources_group_replay"]
        and instance["right_section"]["all_roots_vanish"]
        and instance["right_section"]["all_selector_values_exact"]
        and instance["right_section"]["all_selected_sources_group_replay"]
        for instance in instances
    )
    all_target_sources_exact = all(
        instance["joint_target_replay"][
            "all_attained_target_sources_group_replay"
        ]
        and instance["joint_target_replay"][
            "split_target_set_equals_direct_5a5c_target_set"
        ]
        for instance in instances
    )
    all_coefficient_gcds_exact = all(
        instance["joint_target_replay"][
            "all_sampled_coefficient_gcds_exact"
        ]
        for instance in instances
    )
    cost_ledger = asymptotic_cost_ledger()

    frozen = {
        "schema": "p1553.frozen_5a5c_marked_resultant_section.r84.v1",
        "split": "2A+3C versus 3A+2C",
        "point_key": (
            "injective scalar-blind Fp2 key x+w*y with O mapped to zero"
        ),
        "endpoint_polynomial": (
            "squarefree product over distinct attained side endpoints"
        ),
        "source_section": (
            "one packed mixed-radix source-code interpolant per side"
        ),
        "target_join": (
            "gcd(L(Z), product_R(Z-key(T-R))) with exact point replay"
        ),
        "candidate_scalar_labels_consumed": False,
        "verifier_dlp_consumed": False,
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "factor_base_size_B": instance["factor_base_size_B"],
                "atom_a_size": instance["atom_a_size"],
                "atom_c_size": instance["atom_c_size"],
                "fp2_nonsquare": instance["fp2_nonsquare"],
            }
            for instance in instances
        ],
    }
    coefficients = {
        "schema": (
            "p1553.resultant_coefficient_containment_receipts.r84.v1"
        ),
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "left_section": instance["left_section"],
                "right_section": instance["right_section"],
                "coefficient_gcd_samples": instance[
                    "joint_target_replay"
                ]["coefficient_gcd_samples"],
            }
            for instance in instances
        ],
        "all_side_polynomial_roots_exact": all_side_sections_exact,
        "all_sampled_coefficient_gcds_exact": all_coefficient_gcds_exact,
        "p1510_output_sensitive_control": p1510_output_sensitive_control(),
        "containment_statement": (
            "Every side root is an exact EC endpoint of its decoded atom "
            "source; the injective Fp2 key makes common roots equivalent to "
            "the exact target equality."
        ),
    }
    replay = {
        "schema": "p1553.joint_source_section_replay.r84.v1",
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                **instance["joint_target_replay"],
            }
            for instance in instances
        ],
        "all_side_sections_exact": all_side_sections_exact,
        "all_attained_target_sources_exact": all_target_sources_exact,
        "all_sampled_coefficient_gcds_exact": all_coefficient_gcds_exact,
        "joint_source_coordinates": (
            "one left packed code and one right packed code sharing the "
            "same exact gcd root and target equality"
        ),
        "all_target_audit_candidate_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r84.v1",
        "inherited_public_factor_identity": (
            "log(F_ij)=log(A_i)+log(C_j)"
        ),
        "inherited_public_rectangle_rank_exact": True,
        "finite_joint_source_section_exact": (
            all_side_sections_exact
            and all_target_sources_exact
            and all_coefficient_gcds_exact
        ),
        "finite_joint_source_section_inside_caps": False,
        "algorithmic_known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "descent_obstruction": (
            "the same fresh translated-right polynomial emits B^2.4 "
            "coefficients before its gcd/source can be opened"
        ),
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "four_prime_order_relation_scale_families": (
            len(families) == 4
            and len(instances) == 4 * len(offsets)
        ),
        "eight_frozen_instances": len(instances) == 8,
        "factor_bases_frozen_scalar_blind": all(
            instance["factor_base_injective"]
            and not instance["scalar_labels_consumed"]
            for instance in instances
        ),
        "injective_fp2_point_key": True,
        "left_and_right_radical_polynomials_exact": all_side_sections_exact,
        "packed_joint_source_selectors_exact": all_side_sections_exact,
        "every_finite_attained_target_has_joint_source": (
            all_target_sources_exact
        ),
        "sampled_coefficient_gcds_match_exact_join": (
            all_coefficient_gcds_exact
        ),
        "p1510_output_sensitive_control_preserved": True,
        "explicit_coefficient_body_inside_setup_cap": False,
        "fresh_translated_polynomial_inside_online_cap": False,
        "all_target_containment_audit_inside_online_cap": False,
        "algorithmic_known_rhs_rank_complete": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_theorem": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "EXPLICIT_MARKED_SOURCE_SECTION_EXACT__COEFFICIENT_BODY_OVER_CAP"
        ),
        "source_bindings": {
            "r83_coordinate_filtration": {
                "path": str(R83_REPORT),
                "sha256": R83_REPORT_SHA256,
            },
            "r83_gate": {
                "path": str(R83_GATE),
                "sha256": R83_GATE_SHA256,
            },
            "r82_cartesian_sum_geometry": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "p1510_output_sensitive_compiler": {
                "path": str(P1510_PRODUCER),
                "sha256": P1510_PRODUCER_SHA256,
            },
            "p1510_independent_audit": {
                "path": str(P1510_AUDIT),
                "sha256": P1510_AUDIT_SHA256,
            },
            "p1511_factorized_semijoin_gate": {
                "path": str(P1511_GATE),
                "sha256": P1511_ACTUAL_GATE_SHA256,
            },
            "r16_common_right_factor_gate": {
                "path": str(R16_GATE),
                "sha256": R16_GATE_SHA256,
            },
        },
        "all_source_bindings_verified": (
            len(verified_bindings) == 7
        ),
        "novelty_scope": (
            "R84 instantiates the R83 marked-resultant residual with an "
            "injective scalar-blind Fp2 endpoint key, exact radical side "
            "polynomials, packed coupled source interpolants, coefficient "
            "gcds, and exhaustive finite target containment."
        ),
        "instances": instances,
        "aggregate": {
            "instance_count": len(instances),
            "all_side_sections_exact": all_side_sections_exact,
            "all_attained_target_sources_exact": all_target_sources_exact,
            "all_sampled_coefficient_gcds_exact": (
                all_coefficient_gcds_exact
            ),
            "attained_target_count_total": sum(
                instance["joint_target_replay"]["attained_target_count"]
                for instance in instances
            ),
            "side_endpoint_pair_count_total": sum(
                instance["joint_target_replay"]["side_endpoint_pair_count"]
                for instance in instances
            ),
            "explicit_smaller_side_exponent_B": cost_ledger[
                "explicit_radical_endpoint_polynomial"
            ]["smaller_side_degree_exponent_B"],
            "setup_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "fresh_cap_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "side_artifacts": {
            "frozen": "frozen_5a5c_marked_resultant_section.json",
            "coefficients": (
                "resultant_coefficient_and_containment_receipts.json"
            ),
            "joint_replay": "joint_source_section_replay.json",
            "cost_ledger": "fresh_query_state_cost_ledger.json",
            "logs_descent": (
                "factor_logs_and_identical_descent_r84.json"
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This closes only explicit dense endpoint-polynomial and packed "
            "interpolant marked-resultant sections for the frozen split. It "
            "does not reject a P1510/P1511-style target-uniform circuit "
            "before coefficient or provenance-leaf emission, a compact "
            "common right factor, or an arbitrary FFE/Semaev solver."
        ),
        "next_action": (
            "Construct or refute one target-uniform pre-coefficient circuit "
            "for the 2A+3C versus 3A+2C join. It must consume only the compact "
            "D_A,D_C inputs, specialize a fresh target inside B^(5/4), emit "
            "no B^(2.4) side polynomial or provenance leaves, return one "
            "jointly coupled source, and preserve exact containment, rank, "
            "factor-log, "
            "and identical-descent receipts."
        ),
        "disposition": (
            "REJECT_EXPLICIT_MARKED_RESULTANT_COEFFICIENT_BODY_ONLY__"
            "SCALAR_BLIND_FP2_POINT_KEY_INJECTIVE__RADICAL_SIDE_POLYNOMIALS_"
            "EXACT__PACKED_JOINT_SOURCE_INTERPOLANTS_EXACT__EVERY_FINITE_"
            "ATTAINED_TARGET_REPLAYED__COEFFICIENT_GCD_SAMPLES_EXACT__"
            "LEFT_B2P6__RIGHT_B2P4__SMALLER_SIDE_EXCEEDS_B2P25_SETUP__"
            "FRESH_TRANSLATION_EXCEEDS_B1P25__ALL_TARGET_AUDIT_B5_NO_CREDIT__"
            "P1510_OUTPUT_SENSITIVE_EXCEPTION_PRESERVED__PRECOEFFICIENT_"
            "CIRCUIT_OPEN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__"
            "NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "coefficients": coefficients,
        "joint_replay": replay,
        "cost_ledger": cost_ledger,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_marked_resultant_source_section_"
            "probe_report_r84.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_marked_resultant_section.json"
        ),
    )
    parser.add_argument(
        "--coefficient-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "resultant_coefficient_and_containment_receipts.json"
        ),
    )
    parser.add_argument(
        "--joint-output",
        type=pathlib.Path,
        default=pathlib.Path("joint_source_section_replay.json"),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("fresh_query_state_cost_ledger.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r84.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.coefficient_output, bundle["coefficients"])
    write_json(args.joint_output, bundle["joint_replay"])
    write_json(args.cost_output, bundle["cost_ledger"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
