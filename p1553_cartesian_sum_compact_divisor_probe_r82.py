#!/usr/bin/env python3
"""Test a Cartesian-sum compact divisor through S4 and full relation source."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Iterator, Sequence


SCHEMA = "p1553.cartesian_sum_compact_divisor_probe.r82.v1"
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
ATOM_A_EXPONENT = 2 / 5
ATOM_C_EXPONENT = 3 / 5
TRIPLE_ARITY = 3
RELATION_ARITY = 5
CONTROL_COUNT = 1
INSTANCE_OFFSETS = (0, 1)

FAMILIES = (
    {
        "family_id": "p98561_j0_q16427_h6_u2_v3",
        "field_prime": 98_561,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 16_427,
        "cofactor": 6,
        "atom_a_size": 2,
        "atom_c_size": 3,
    },
    {
        "family_id": "p3148097_j0_q524683_h6_u3_v5",
        "field_prime": 3_148_097,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 524_683,
        "cofactor": 6,
        "atom_a_size": 3,
        "atom_c_size": 5,
    },
    {
        "family_id": "p9603641_j0_q1600607_h6_u3_v6",
        "field_prime": 9_603_641,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 1_600_607,
        "cofactor": 6,
        "atom_a_size": 3,
        "atom_c_size": 6,
    },
    {
        "family_id": "p100683137_j0_q16780523_h6_u4_v7",
        "field_prime": 100_683_137,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 16_780_523,
        "cofactor": 6,
        "atom_a_size": 4,
        "atom_c_size": 7,
    },
)

R81_REPORT = pathlib.Path(
    "p1553_full_multiplicative_x_coset_endpoint_probe_report_r81.json"
)
R81_REPORT_SHA256 = (
    "e556efa7c1e639f76915f152ebdcc3d00db2a932d8ef207ae8653c94117f026f"
)
R14_GATE = pathlib.Path("p1553_tensor_trace_minpoly_compiler_gate_r14.md")
R14_GATE_SHA256 = (
    "da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac"
)
R11_GATE = pathlib.Path("p1553_cartesian_kummer_rigidity_gate_r11.md")
R11_GATE_SHA256 = (
    "6c79b486bfa4cfd14674033a438db3d91ddf7bbe4d2c4aaf309f1a0706f0df4e"
)
R31_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R31_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R3_GATE_SHA256 = (
    "b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e"
)

Point = tuple[int, int] | None


def load_r81() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_full_multiplicative_x_coset_endpoint_probe_r81.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r81_for_r82", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R81 curve controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R81 = load_r81()
R70 = R81.R70


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def point_sort_key(point: Point) -> tuple[int, int, int]:
    return (-1, 0, 0) if point is None else (0, point[0], point[1])


def support_exponent(size: int, base_size: int) -> float:
    if size <= 1 or base_size <= 1:
        return 0.0
    return math.log(size) / math.log(base_size)


def validate_family(curve: dict[str, Any], generator: Point) -> dict[str, bool]:
    prime = curve["field_prime"]
    order = curve["subgroup_order"]
    return {
        "field_prime_probable_prime": R70.is_prime(prime),
        "subgroup_order_probable_prime": R70.is_prime(order),
        "j_zero_supersingular_order_identity": (
            prime % 3 == 2
            and curve["curve_a"] == 0
            and prime + 1 == curve["cofactor"] * order
        ),
        "curve_discriminant_nonzero": R70.curve_discriminant(curve) != 0,
        "generator_nonidentity": generator is not None,
        "generator_has_stated_order": (
            generator is not None
            and R70.scalar_mul(order, generator, curve) is None
        ),
    }


def hash_point_candidates(
    curve: dict[str, Any],
    salt: str,
) -> Iterator[Point]:
    counter = 0
    seen: set[Point] = set()
    while True:
        digest = hashlib.sha256(
            f"P1553-R82|{curve['family_id']}|{salt}|{counter}".encode()
        ).digest()
        counter += 1
        x_coordinate = int.from_bytes(digest, "big") % curve["field_prime"]
        point = R70.point_from_x(x_coordinate, digest[0] & 1, curve)
        if point is None:
            continue
        point = R81.canonical_point(point, curve)
        if point is None or point in seen:
            continue
        seen.add(point)
        yield point


def compact_factor_base(
    curve: dict[str, Any],
    offset: int,
) -> tuple[list[Point], list[Point], list[Point], dict[str, Any]]:
    size_a = curve["atom_a_size"]
    size_c = curve["atom_c_size"]
    atoms_a = list(
        itertools.islice(
            hash_point_candidates(curve, f"A|{offset}"),
            size_a,
        )
    )
    atoms_c: list[Point] = []
    factor_seen: set[Point] = set()
    rejected = 0
    for candidate in hash_point_candidates(curve, f"C|{offset}"):
        if candidate in atoms_a or candidate in atoms_c:
            rejected += 1
            continue
        new_points = [R70.add(left, candidate, curve) for left in atoms_a]
        if (
            any(point is None for point in new_points)
            or len(set(new_points)) != len(new_points)
            or any(point in factor_seen for point in new_points)
        ):
            rejected += 1
            continue
        atoms_c.append(candidate)
        factor_seen.update(new_points)
        if len(atoms_c) == size_c:
            break
    if len(atoms_c) != size_c:
        raise AssertionError("unable to freeze collision-free C deck")
    expected = size_a * size_c
    factor_points = [
        R70.add(left, right, curve)
        for left in atoms_a
        for right in atoms_c
    ]
    if len(factor_points) != expected or len(factor_seen) != expected:
        raise AssertionError("Cartesian-sum factor base lost injectivity")
    return atoms_a, atoms_c, factor_points, {
        "offset": offset,
        "atom_a_size": size_a,
        "atom_c_size": size_c,
        "factor_base_size": expected,
        "rejected_c_candidates": rejected,
        "factor_base_injective": True,
        "factor_base_contains_identity": False,
        "construction": "F_(i,j)=A_i+C_j",
        "input_atom_count": size_a + size_c,
        "optional_materialization_group_additions": expected,
        "divisor_slp": {
            "input_divisors": "D_A=sum_i[A_i], D_C=sum_j[C_j]",
            "correspondence": "addition_pushforward(D_A x D_C)",
            "output_divisor": "D_F=sum_(i,j)[A_i+C_j]",
            "input_leaf_count": size_a + size_c,
            "output_degree": expected,
            "straight_line_gate_count_before_output_materialization": 1,
        },
        "source_rules": {
            "hash_to_curve_and_cofactor_clear": True,
            "candidate_scalar_labels_consumed": False,
            "outcome_adaptive_selection": False,
        },
    }


def add_many(
    points: Iterable[Point],
    curve: dict[str, Any],
) -> Point:
    result: Point = None
    for point in points:
        result = R70.add(result, point, curve)
    return result


def ordered_endpoint_map(
    points: Sequence[Point],
    arity: int,
    curve: dict[str, Any],
) -> tuple[collections.Counter[Point], dict[Point, tuple[int, ...]]]:
    histogram: collections.Counter[Point] = collections.Counter()
    first: dict[Point, tuple[int, ...]] = {}
    for indices in itertools.product(range(len(points)), repeat=arity):
        endpoint = add_many((points[index] for index in indices), curve)
        histogram[endpoint] += 1
        first.setdefault(endpoint, indices)
    if sum(histogram.values()) != len(points) ** arity:
        raise AssertionError("ordered endpoint occurrence count drifted")
    return histogram, first


def sumset_source(
    target: Point,
    left: dict[Point, tuple[int, ...]],
    right: dict[Point, tuple[int, ...]],
    curve: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if len(left) <= len(right):
        for endpoint, source in left.items():
            complement = R70.add(target, R70.negate(endpoint, curve), curve)
            other = right.get(complement)
            if other is not None or complement in right:
                return source, right[complement]
        return None
    result = sumset_source(target, right, left, curve)
    return None if result is None else (result[1], result[0])


def convolution_support(
    left: Iterable[Point],
    right: Iterable[Point],
    curve: dict[str, Any],
) -> set[Point]:
    return {
        R70.add(left_endpoint, right_endpoint, curve)
        for left_endpoint in left
        for right_endpoint in right
    }


def source_replay(
    target: Point,
    source: tuple[tuple[int, ...], tuple[int, ...]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    indices_a, indices_c = source
    endpoint = add_many(
        [
            *(atoms_a[index] for index in indices_a),
            *(atoms_c[index] for index in indices_c),
        ],
        curve,
    )
    return endpoint == target


def multinomial_weight(indices: Sequence[int]) -> int:
    value = math.factorial(len(indices))
    for multiplicity in collections.Counter(indices).values():
        value //= math.factorial(multiplicity)
    return value


def weighted_multiset_histogram(
    labels: Sequence[int],
    arity: int,
    modulus: int,
) -> tuple[collections.Counter[int], dict[int, tuple[int, ...]]]:
    histogram: collections.Counter[int] = collections.Counter()
    first: dict[int, tuple[int, ...]] = {}
    for indices in itertools.combinations_with_replacement(
        range(len(labels)),
        arity,
    ):
        endpoint = sum(labels[index] for index in indices) % modulus
        histogram[endpoint] += multinomial_weight(indices)
        first.setdefault(endpoint, indices)
    if sum(histogram.values()) != len(labels) ** arity:
        raise AssertionError("multiset weighting did not recover ordered count")
    return histogram, first


def convolve_label_histograms(
    left: collections.Counter[int],
    right: collections.Counter[int],
    left_first: dict[int, tuple[int, ...]],
    right_first: dict[int, tuple[int, ...]],
    modulus: int,
) -> tuple[
    collections.Counter[int],
    dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
]:
    histogram: collections.Counter[int] = collections.Counter()
    first: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for left_endpoint, left_count in left.items():
        for right_endpoint, right_count in right.items():
            endpoint = (left_endpoint + right_endpoint) % modulus
            histogram[endpoint] += left_count * right_count
            first.setdefault(
                endpoint,
                (left_first[left_endpoint], right_first[right_endpoint]),
            )
    return histogram, first


def coefficient_row(indices: Sequence[int], width: int) -> tuple[int, ...]:
    row = [0] * width
    for index in indices:
        row[index] += 1
    return tuple(row)


def atom_row(
    source: tuple[tuple[int, ...], tuple[int, ...]],
    size_a: int,
    size_c: int,
) -> tuple[int, ...]:
    left, right = source
    return (
        *coefficient_row(left, size_a),
        *coefficient_row(right, size_c),
    )


def factor_row(
    source: tuple[tuple[int, ...], tuple[int, ...]],
    size_a: int,
    size_c: int,
) -> tuple[int, ...]:
    left, right = source
    row = [0] * (size_a * size_c)
    for left_index, right_index in zip(sorted(left), sorted(right)):
        row[left_index * size_c + right_index] += 1
    return tuple(row)


def rectangle_identity_rows(size_a: int, size_c: int) -> list[tuple[int, ...]]:
    rows = []
    for left_index in range(size_a - 1):
        for right_index in range(size_c - 1):
            row = [0] * (size_a * size_c)
            row[left_index * size_c + right_index] += 1
            row[(left_index + 1) * size_c + right_index + 1] += 1
            row[left_index * size_c + right_index + 1] -= 1
            row[(left_index + 1) * size_c + right_index] -= 1
            rows.append(tuple(row))
    return rows


def independent_source_rows(
    sources: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    size_a: int,
    size_c: int,
    modulus: int,
) -> list[tuple[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    selected: list[tuple[int, tuple[tuple[int, ...], tuple[int, ...]]]] = []
    rows: list[tuple[int, ...]] = []
    target_rank = size_a + size_c - 1
    for endpoint in sorted(sources):
        source = sources[endpoint]
        row = atom_row(source, size_a, size_c)
        if R81.rank_mod([*rows, row], modulus) > len(rows):
            rows.append(row)
            selected.append((endpoint, source))
            if len(rows) == target_rank:
                break
    return selected


def labels_match_points(
    points: Sequence[Point],
    labels: Sequence[int],
    generator: Point,
    curve: dict[str, Any],
) -> bool:
    return all(
        R70.scalar_mul(label, generator, curve) == point
        for point, label in zip(points, labels)
    )


def source_split_ledger() -> dict[str, Any]:
    splits = []
    for count_a in range(RELATION_ARITY + 1):
        for count_c in range(RELATION_ARITY + 1):
            left = (
                count_a * ATOM_A_EXPONENT
                + count_c * ATOM_C_EXPONENT
            )
            right = RELATION_ARITY - left
            splits.append(
                {
                    "left_a_count": count_a,
                    "left_c_count": count_c,
                    "left_exponent_B": left,
                    "right_exponent_B": right,
                    "maximum_exponent_B": max(left, right),
                    "minimum_exponent_B": min(left, right),
                }
            )
    best = min(
        splits,
        key=lambda row: (
            row["maximum_exponent_B"],
            row["minimum_exponent_B"],
            row["left_a_count"],
            row["left_c_count"],
        ),
    )
    return {
        "all_integer_equality_joins": splits,
        "best_explicit_equality_join": best,
        "generic_collision_baseline": {
            "work_exponent_B": 2.5,
            "group_order_exponent_B": 5.0,
            "equivalent_group_order_exponent": 0.5,
            "description": (
                "rho-scale collision search; no algorithmic improvement"
            ),
        },
        "wagner_route": {
            "credited": False,
            "reason": (
                "Wagner filtering needs addition-compatible quotient "
                "projections; a prime-order cyclic group has no proper "
                "nontrivial subgroup quotient, and coordinate hash prefixes "
                "are not preserved by elliptic addition"
            ),
        },
        "summation_polynomial_route": {
            "equation": "S_11(x(A_1),...,x(A_5),x(C_1),...,x(C_5),x(R))=0",
            "standard_split_quotient_dimension_exponent_B": 5.0,
            "subcap_field_solver_supplied": False,
            "unproven_exception": (
                "a representation-specific FFE or elimination filtration "
                "with exact source provenance"
            ),
        },
    }


def hash_control_base(
    curve: dict[str, Any],
    size: int,
    salt: str,
) -> list[Point]:
    return list(itertools.islice(hash_point_candidates(curve, salt), size))


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
    control_count: int,
) -> dict[str, Any]:
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    atoms_a, atoms_c, factors, geometry = compact_factor_base(curve, offset)
    labels_a = verifier.labels(atoms_a)
    labels_c = verifier.labels(atoms_c)
    factor_labels = [
        (labels_a[left] + labels_c[right]) % curve["subgroup_order"]
        for left in range(len(atoms_a))
        for right in range(len(atoms_c))
    ]
    if not (
        labels_match_points(atoms_a, labels_a, generator, curve)
        and labels_match_points(atoms_c, labels_c, generator, curve)
        and labels_match_points(factors, factor_labels, generator, curve)
    ):
        raise AssertionError("verifier labels failed point replay")

    triple_a, triple_a_first = ordered_endpoint_map(
        atoms_a,
        TRIPLE_ARITY,
        curve,
    )
    triple_c, triple_c_first = ordered_endpoint_map(
        atoms_c,
        TRIPLE_ARITY,
        curve,
    )
    triple_support = convolution_support(triple_a, triple_c, curve)
    triple_total = sum(triple_a.values()) * sum(triple_c.values())
    if triple_total != len(factors) ** TRIPLE_ARITY:
        raise AssertionError("factorized triple count drifted")
    accepted_triple_targets = []
    for index in range(min(16, len(factors) ** TRIPLE_ARITY)):
        digest = hashlib.sha256(
            f"P1553-R82-triple|{curve['family_id']}|{offset}|{index}".encode()
        ).digest()
        factor_indices = [
            int.from_bytes(digest[2 * slot : 2 * slot + 2], "big")
            % len(factors)
            for slot in range(TRIPLE_ARITY)
        ]
        target = add_many((factors[item] for item in factor_indices), curve)
        source = sumset_source(
            target,
            triple_a_first,
            triple_c_first,
            curve,
        )
        accepted_triple_targets.append(
            source is not None
            and source_replay(
                target,
                source,
                atoms_a,
                atoms_c,
                curve,
            )
        )

    five_a_points, five_a_first = ordered_endpoint_map(
        atoms_a,
        RELATION_ARITY,
        curve,
    )
    five_c_points, five_c_first = ordered_endpoint_map(
        atoms_c,
        RELATION_ARITY,
        curve,
    )
    diagnostic_target = add_many(
        [
            *(atoms_a[index % len(atoms_a)] for index in range(RELATION_ARITY)),
            *(atoms_c[index % len(atoms_c)] for index in range(RELATION_ARITY)),
        ],
        curve,
    )
    diagnostic_source = sumset_source(
        diagnostic_target,
        five_a_first,
        five_c_first,
        curve,
    )
    full_source_replay_exact = (
        diagnostic_source is not None
        and source_replay(
            diagnostic_target,
            diagnostic_source,
            atoms_a,
            atoms_c,
            curve,
        )
    )

    modulus = curve["subgroup_order"]
    five_a_labels, five_a_label_first = weighted_multiset_histogram(
        labels_a,
        RELATION_ARITY,
        modulus,
    )
    five_c_labels, five_c_label_first = weighted_multiset_histogram(
        labels_c,
        RELATION_ARITY,
        modulus,
    )
    five_histogram, five_sources = convolve_label_histograms(
        five_a_labels,
        five_c_labels,
        five_a_label_first,
        five_c_label_first,
        modulus,
    )
    if sum(five_histogram.values()) != len(factors) ** RELATION_ARITY:
        raise AssertionError("candidate five-sum occurrence count drifted")

    selected = independent_source_rows(
        five_sources,
        len(atoms_a),
        len(atoms_c),
        modulus,
    )
    atom_rows = [
        atom_row(source, len(atoms_a), len(atoms_c))
        for _, source in selected
    ]
    selected_factor_rows = [
        factor_row(source, len(atoms_a), len(atoms_c))
        for _, source in selected
    ]
    selected_rhs = [endpoint for endpoint, _ in selected]
    if any(
        sum(value * label for value, label in zip(row, labels_a + labels_c))
        % modulus
        != rhs
        for row, rhs in zip(atom_rows, selected_rhs)
    ):
        raise AssertionError("atom row failed verifier RHS")
    if any(
        sum(value * label for value, label in zip(row, factor_labels))
        % modulus
        != rhs
        for row, rhs in zip(selected_factor_rows, selected_rhs)
    ):
        raise AssertionError("factor row failed verifier RHS")
    rectangles = rectangle_identity_rows(len(atoms_a), len(atoms_c))
    if any(
        sum(value * label for value, label in zip(row, factor_labels))
        % modulus
        for row in rectangles
    ):
        raise AssertionError("rectangle identity failed factor labels")
    rectangle_rank = R81.rank_mod(rectangles, modulus)
    projected_rank = R81.rank_mod(atom_rows, modulus)
    combined_factor_rank = R81.rank_mod(
        [*rectangles, *selected_factor_rows],
        modulus,
    )

    controls = []
    for control_index in range(control_count):
        control_points = hash_control_base(
            curve,
            len(factors),
            f"control|{offset}|{control_index}",
        )
        control_labels = verifier.labels(control_points)
        control_histogram, control_first = weighted_multiset_histogram(
            control_labels,
            RELATION_ARITY,
            modulus,
        )
        control_rows = [
            coefficient_row(source, len(control_points))
            for _, source in sorted(control_first.items())
        ]
        controls.append(
            {
                "control_index": control_index,
                "base_size": len(control_points),
                "five_sum_support": len(control_histogram),
                "five_sum_density": len(control_histogram) / modulus,
                "ordered_occurrence_count": sum(control_histogram.values()),
                "coefficient_row_rank": R81.rank_mod(
                    control_rows,
                    modulus,
                ),
                "full_prospective_rank": (
                    R81.rank_mod(control_rows, modulus)
                    == len(control_points)
                ),
                "verifier_only": True,
            }
        )

    base_size = len(factors)
    finite_atom_a_exponent = support_exponent(len(atoms_a), base_size)
    finite_atom_c_exponent = support_exponent(len(atoms_c), base_size)
    return {
        "family": curve,
        "family_validation": validate_family(curve, generator),
        "generator": point_json(generator),
        "geometry": geometry,
        "factor_base_preview": [point_json(point) for point in factors[:8]],
        "finite_scale": {
            "B": base_size,
            "subgroup_order": modulus,
            "subgroup_order_over_B5": modulus / base_size**5,
            "atom_a_exponent_B": finite_atom_a_exponent,
            "atom_c_exponent_B": finite_atom_c_exponent,
        },
        "triple_compiler": {
            "identity": "3F=3A+3C",
            "atom_a_ordered_occurrences": len(atoms_a) ** TRIPLE_ARITY,
            "atom_c_ordered_occurrences": len(atoms_c) ** TRIPLE_ARITY,
            "atom_a_endpoint_support": len(triple_a),
            "atom_c_endpoint_support": len(triple_c),
            "factor_triple_endpoint_support_verifier_only": len(
                triple_support
            ),
            "factor_triple_ordered_occurrences": triple_total,
            "stored_endpoint_words": len(triple_a) + len(triple_c),
            "source_query_scan_words": min(len(triple_a), len(triple_c)),
            "all_sampled_sources_exact": all(accepted_triple_targets),
            "sampled_source_count": len(accepted_triple_targets),
            "candidate_scalar_labels_consumed": False,
            "full_convolution_materialized_by_candidate": False,
            "asymptotic": {
                "atom_a_build_exponent_B": 3 * ATOM_A_EXPONENT,
                "atom_c_build_exponent_B": 3 * ATOM_C_EXPONENT,
                "retained_state_exponent_B": 3 * ATOM_C_EXPONENT,
                "fresh_endpoint_query_exponent_B": 3 * ATOM_A_EXPONENT,
                "state_inside_cap": (
                    3 * ATOM_C_EXPONENT <= SETUP_STATE_CAP_EXPONENT
                ),
                "query_inside_cap": (
                    3 * ATOM_A_EXPONENT <= ONLINE_CAP_EXPONENT
                ),
            },
        },
        "five_factor_source": {
            "identity": "5F=5A+5C",
            "atom_a_endpoint_support": len(five_a_points),
            "atom_c_endpoint_support": len(five_c_points),
            "diagnostic_full_source_replay_exact": full_source_replay_exact,
            "diagnostic_build_consumes_scalar_labels": False,
            "diagnostic_build_is_over_cap": True,
            "candidate_five_sum_support": len(five_histogram),
            "candidate_five_sum_density": len(five_histogram) / modulus,
            "candidate_ordered_occurrence_count": sum(
                five_histogram.values()
            ),
            "uniform_expected_ordered_count_per_target": (
                base_size**RELATION_ARITY / modulus
            ),
            "maximum_endpoint_occurrence_count": max(
                five_histogram.values(),
                default=0,
            ),
        },
        "rank_control": {
            "nominal_factor_column_count": base_size,
            "meaningful_atom_log_dimension": (
                len(atoms_a) + len(atoms_c) - 1
            ),
            "projected_known_rhs_row_rank_verifier_only": projected_rank,
            "rectangle_identity_count": len(rectangles),
            "rectangle_identity_rank": rectangle_rank,
            "expected_rectangle_identity_rank": (
                base_size - len(atoms_a) - len(atoms_c) + 1
            ),
            "combined_factor_row_rank_verifier_only": combined_factor_rank,
            "all_factor_columns_spanned_prospectively": (
                combined_factor_rank == base_size
            ),
            "selected_known_rhs_count": len(selected),
            "all_selected_rhs_rows_replay": True,
            "public_rectangle_identities_need_no_dlp": True,
            "known_rhs_values_obtained_by_candidate": False,
        },
        "matched_hash_controls": controls,
        "verifier_bsgs_receipt": verifier.receipt(),
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = FAMILIES,
    offsets: Sequence[int] = INSTANCE_OFFSETS,
    control_count: int = CONTROL_COUNT,
) -> dict[str, dict[str, Any]]:
    instances = [
        analyze_instance(dict(curve), offset, control_count)
        for curve in families
        for offset in offsets
    ]
    source_splits = source_split_ledger()
    all_family_checks = all(
        all(instance["family_validation"].values())
        for instance in instances
    )
    all_factor_bases_injective = all(
        instance["geometry"]["factor_base_injective"]
        for instance in instances
    )
    all_triple_sources_exact = all(
        instance["triple_compiler"]["all_sampled_sources_exact"]
        for instance in instances
    )
    all_triple_caps = all(
        instance["triple_compiler"]["asymptotic"]["state_inside_cap"]
        and instance["triple_compiler"]["asymptotic"]["query_inside_cap"]
        for instance in instances
    )
    all_full_source_replays = all(
        instance["five_factor_source"][
            "diagnostic_full_source_replay_exact"
        ]
        for instance in instances
    )
    all_rank_identities = all(
        instance["rank_control"]["rectangle_identity_rank"]
        == instance["rank_control"]["expected_rectangle_identity_rank"]
        and instance["rank_control"]["all_factor_columns_spanned_prospectively"]
        for instance in instances
    )
    candidate_density = [
        instance["five_factor_source"]["candidate_five_sum_density"]
        for instance in instances
    ]
    control_density = [
        control["five_sum_density"]
        for instance in instances
        for control in instance["matched_hash_controls"]
    ]

    geometry = {
        "schema": "p1553.frozen_compact_divisor_factor_base.r82.v1",
        "geometry_id": "cartesian_sum_addition_pushforward_u_B2o5_v_B3o5_v1",
        "curve_model": "y^2=x^3+1 with #E(F_p)=p+1=6q",
        "candidate_selection_after_outcomes": False,
        "asymptotic_atom_sizes": {
            "u": "B^(2/5+o(1))",
            "v": "B^(3/5+o(1))",
            "factor_base_size": "uv=B",
        },
        "scalar_labels_consumed": False,
        "instances": [
            {
                "family_id": instance["family"]["family_id"],
                "family_validation": instance["family_validation"],
                "geometry": instance["geometry"],
                "finite_scale": instance["finite_scale"],
                "factor_base_preview": instance["factor_base_preview"],
            }
            for instance in instances
        ],
    }
    endpoint_compiler = {
        "schema": "p1553.divisor_slp_to_s4_endpoint_compiler.r82.v1",
        "identity": "F=A+C implies 3F=3A+3C",
        "construction": {
            "store": "exact scalar-blind 3A and 3C endpoint dictionaries",
            "query": "scan smaller 3A support against hash table for 3C",
            "source": "return both atom triples and pair them into F columns",
            "retained_state_exponent_B": 3 * ATOM_C_EXPONENT,
            "fresh_query_exponent_B": 3 * ATOM_A_EXPONENT,
            "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_cap_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "instances": [
            {
                "family_id": instance["family"]["family_id"],
                "offset": instance["geometry"]["offset"],
                **instance["triple_compiler"],
            }
            for instance in instances
        ],
        "all_factor_bases_injective": all_factor_bases_injective,
        "all_sampled_sources_exact": all_triple_sources_exact,
        "all_asymptotic_triple_caps_pass": all_triple_caps,
        "local_s4_compiler_passes": (
            all_factor_bases_injective
            and all_triple_sources_exact
            and all_triple_caps
        ),
    }
    controls = {
        "schema": "p1553.prospective_density_matched_rank_controls.r82.v1",
        "verifier_only": True,
        "known_rhs_values_available_to_candidate": False,
        "instances": [
            {
                "family_id": instance["family"]["family_id"],
                "offset": instance["geometry"]["offset"],
                "finite_scale": instance["finite_scale"],
                "candidate": {
                    "five_factor_source": instance["five_factor_source"],
                    "rank_control": instance["rank_control"],
                },
                "matched_hash_controls": instance["matched_hash_controls"],
            }
            for instance in instances
        ],
        "candidate_density_range": [
            min(candidate_density),
            max(candidate_density),
        ],
        "matched_control_density_range": [
            min(control_density),
            max(control_density),
        ],
        "all_rectangle_and_prospective_rank_checks_pass": all_rank_identities,
        "interpretation": (
            "The B nominal columns split into a public rectangle kernel of "
            "dimension B-u-v+1 and u+v-1 meaningful atom-log directions. "
            "The verifier-only known-RHS rows span the latter on every toy."
        ),
    }
    cost_ledger = {
        "schema": "p1553.scalar_blind_query_source_cost_ledger.r82.v1",
        "caps": {
            "setup_state_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_work_exponent_B": ONLINE_CAP_EXPONENT,
            "online_workspace_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "local_triple_compiler": endpoint_compiler["construction"],
        "full_relation_identity": "5F=5A+5C",
        "full_relation_is_colored_ten_sum": True,
        "integer_split_ledger": source_splits,
        "best_explicit_join_work_exponent_B": source_splits[
            "best_explicit_equality_join"
        ]["maximum_exponent_B"],
        "best_explicit_join_state_exponent_B": source_splits[
            "best_explicit_equality_join"
        ]["minimum_exponent_B"],
        "generic_collision_work_exponent_B": 2.5,
        "generic_collision_work_exponent_N": 0.5,
        "full_source_inside_online_cap": False,
        "full_source_state_inside_setup_cap": False,
        "even_one_fresh_target_query_passes": False,
        "all_diagnostic_group_sources_exact": all_full_source_replays,
        "candidate_scalar_labels_consumed": False,
        "verifier_labels_excluded_from_credit": True,
        "failure_reasons": [
            "the passing 3A+3C compiler does not compose into a 5A+5C target router",
            "the best integer equality join has B^2.6 work and B^2.4 state",
            "rho-scale collision work is B^2.5=N^0.5",
            "Wagner filtering has no addition-compatible proper quotient in a prime-order group",
            "no subcap summation-polynomial or FFE source solver is supplied",
        ],
    }
    logs_descent = {
        "schema": "p1553.verified_factor_logs_identical_descent.r82.v1",
        "public_factor_identity": "log(F_ij)=log(A_i)+log(C_j)",
        "public_rectangle_relations": (
            "F_ij+F_i'j'-F_ij'-F_i'j=O"
        ),
        "meaningful_unknown_count": "u+v-1 after the additive gauge",
        "prospective_verifier_rank_complete_on_all_instances": (
            all_rank_identities
        ),
        "verifier_only_factor_labels_satisfy_all_rows": True,
        "algorithmic_known_rhs_relation_collection_complete": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "descent_obstruction": (
            "a fresh target is the same 5A+5C colored ten-sum that misses "
            "the online cap"
        ),
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }

    obligations = {
        "four_prime_order_relation_scale_families": (
            len(families) == 4 and len(instances) == 4 * len(offsets)
        ),
        "all_family_parameter_checks": all_family_checks,
        "factor_base_frozen_scalar_blind": True,
        "factor_base_addition_map_injective": all_factor_bases_injective,
        "compact_divisor_slp_supplied": True,
        "local_s4_triple_compiler_exact": all_triple_sources_exact,
        "local_s4_triple_state_inside_cap": all_triple_caps,
        "local_s4_triple_query_inside_cap": all_triple_caps,
        "prospective_density_controls_present": bool(candidate_density),
        "matched_hash_controls_present": bool(control_density),
        "public_rectangle_rank_exact": all_rank_identities,
        "full_scalar_blind_relation_source_inside_cap": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "candidate_factor_log_solve_without_verifier_dlp": False,
        "candidate_identical_fresh_target_descent": False,
        "generic_prime_family_theorem": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "CARTESIAN_SUM_COMPACT_S4_PASS__FULL_SOURCE_RHO_FAIL"
        ),
        "source_bindings": {
            "r81_complete_coset_screen": {
                "path": str(R81_REPORT),
                "sha256": R81_REPORT_SHA256,
            },
            "r14_tensor_trace_compiler": {
                "path": str(R14_GATE),
                "sha256": R14_GATE_SHA256,
            },
            "r11_cartesian_kummer_rigidity": {
                "path": str(R11_GATE),
                "sha256": R11_GATE_SHA256,
            },
            "r31_query2p1_registry": {
                "path": str(R31_REGISTRY),
                "sha256": R31_REGISTRY_SHA256,
                "bound_gate_sha256": R3_GATE_SHA256,
            },
        },
        "novelty_scope": (
            "R82 is not a one-dimensional field coset or scalar orbit. It "
            "freezes an elliptic addition pushforward factor base F=A+C and "
            "tests whether its exact factorized triple compiler survives "
            "the full known-RHS relation and identical descent."
        ),
        "instances": instances,
        "aggregate": {
            "instance_count": len(instances),
            "matched_control_count": len(control_density),
            "all_family_parameter_checks": all_family_checks,
            "all_factor_bases_injective": all_factor_bases_injective,
            "local_s4_compiler_passes": endpoint_compiler[
                "local_s4_compiler_passes"
            ],
            "all_diagnostic_full_sources_replay": all_full_source_replays,
            "all_rectangle_and_prospective_rank_checks_pass": (
                all_rank_identities
            ),
            "candidate_density_range": controls["candidate_density_range"],
            "matched_control_density_range": controls[
                "matched_control_density_range"
            ],
            "best_explicit_full_source_work_exponent_B": (
                cost_ledger["best_explicit_join_work_exponent_B"]
            ),
            "best_explicit_full_source_state_exponent_B": (
                cost_ledger["best_explicit_join_state_exponent_B"]
            ),
            "generic_collision_full_source_exponent_B": 2.5,
        },
        "side_artifacts": {
            "geometry": "frozen_compact_divisor_factor_base.json",
            "endpoint_compiler": "divisor_slp_to_s4_endpoint_compiler.json",
            "controls": "prospective_density_and_matched_rank_controls.json",
            "cost_ledger": "scalar_blind_query_source_cost_ledger.json",
            "logs_descent": "verified_factor_logs_and_identical_descent.json",
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
            "This closes only Cartesian addition-pushforward factor bases "
            "under exact equality joins, generic collision search, and "
            "standard quotient-free generalized-birthday accounting. It is "
            "not a lower bound against a representation-specific FFE or "
            "summation-polynomial filtration with exact provenance."
        ),
        "next_action": (
            "Derive or refute one addition-compatible field-coordinate "
            "filtration for the colored 5A+5C source. Each partial filter "
            "must compose under elliptic addition, retain exact source "
            "backpointers, avoid verifier logs, fit B^(9/4) setup and "
            "B^(5/4) fresh work, and pass a random-deck control."
        ),
        "disposition": (
            "REJECT_CARTESIAN_SUM_FULL_PIPELINE_ONLY__COMPACT_ADDITION_"
            "PUSHFORWARD_FACTOR_BASE__EXACT_B6O5_BY_B9O5_TRIPLE_COMPILER__"
            "B6O5_TRIPLE_QUERY_PASSES__PUBLIC_RECTANGLE_KERNEL_EXACT__"
            "PROSPECTIVE_MEANINGFUL_RANK_PASSES__FULL_RELATION_IS_5A_PLUS_"
            "5C__BEST_EXPLICIT_JOIN_B2P6_WORK_B2P4_STATE__GENERIC_COLLISION_"
            "B2P5_EQUALS_RHO__NO_QUOTIENT_WAGNER_CREDIT__NO_SUBCAP_FFE__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "geometry": geometry,
        "endpoint_compiler": endpoint_compiler,
        "controls": controls,
        "cost_ledger": cost_ledger,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
        ),
    )
    parser.add_argument(
        "--geometry-output",
        type=pathlib.Path,
        default=pathlib.Path("frozen_compact_divisor_factor_base.json"),
    )
    parser.add_argument(
        "--endpoint-output",
        type=pathlib.Path,
        default=pathlib.Path("divisor_slp_to_s4_endpoint_compiler.json"),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "prospective_density_and_matched_rank_controls.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("scalar_blind_query_source_cost_ledger.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "verified_factor_logs_and_identical_descent.json"
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
    write_json(args.geometry_output, bundle["geometry"])
    write_json(args.endpoint_output, bundle["endpoint_compiler"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.cost_output, bundle["cost_ledger"])
    write_json(args.logs_output, bundle["logs_descent"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"instances={aggregate['instance_count']} "
        f"s4_pass={aggregate['local_s4_compiler_passes']} "
        f"full_work_B={aggregate['best_explicit_full_source_work_exponent_B']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
