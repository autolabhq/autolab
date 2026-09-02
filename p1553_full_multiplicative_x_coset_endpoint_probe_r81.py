#!/usr/bin/env python3
"""Test complete multiplicative-x cosets as compressed S4 factor bases."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import statistics
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.full_multiplicative_x_coset_endpoint_probe.r81.v1"
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
COSET_OFFSETS = (0, 1, 2, 3)
CONTROL_COUNT = 2
TARGET_SAMPLE_COUNT = 32

FAMILIES = (
    {
        "family_id": "p98561_j0_q16427_h6_d16",
        "field_prime": 98_561,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 16_427,
        "cofactor": 6,
        "coordinate_subgroup_order": 16,
    },
    {
        "family_id": "p3148097_j0_q524683_h6_d32",
        "field_prime": 3_148_097,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 524_683,
        "cofactor": 6,
        "coordinate_subgroup_order": 32,
    },
    {
        "family_id": "p9603641_j0_q1600607_h6_d40",
        "field_prime": 9_603_641,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 1_600_607,
        "cofactor": 6,
        "coordinate_subgroup_order": 40,
    },
    {
        "family_id": "p100683137_j0_q16780523_h6_d64",
        "field_prime": 100_683_137,
        "curve_a": 0,
        "curve_b": 1,
        "subgroup_order": 16_780_523,
        "cofactor": 6,
        "coordinate_subgroup_order": 64,
    },
)

R80_REPORT = pathlib.Path(
    "p1553_batched_nested_norm_node_compiler_probe_report_r80.json"
)
R80_REPORT_SHA256 = (
    "936537fb78908dd2916bf6fa5b2091f336b9a47217a1ff787b068ae0491992c5"
)
R70_REPORT = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
)
R70_REPORT_SHA256 = (
    "9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89"
)
R31_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R31_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R3_GATE_SHA256 = (
    "b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e"
)

Point = tuple[int, int] | None


def load_r70() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_multiplicative_x_s3_closure_screen_r70.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r70_for_r81", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R70 controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R70 = load_r70()


def canonical_point(point: Point, curve: dict[str, Any]) -> Point:
    if point is None:
        return None
    negative = R70.negate(point, curve)
    assert negative is not None
    return min(point, negative, key=lambda value: (value[1], value[0]))


def point_key(point: Point) -> str | int:
    return "identity" if point is None else point[0]


def permutation_weight(indices: Sequence[int]) -> int:
    distinct = len(set(indices))
    if distinct == 1:
        return 1
    if distinct == 2:
        return 3
    return 6


def support_exponent(size: int, base_size: int) -> float:
    if size <= 1 or base_size <= 1:
        return 0.0
    return math.log(size) / math.log(base_size)


def power_law_slope(rows: Sequence[tuple[int, int]]) -> float | None:
    usable = [
        (math.log(base_size), math.log(support_size))
        for base_size, support_size in rows
        if base_size > 1 and support_size > 0
    ]
    if len(usable) < 2:
        return None
    mean_x = statistics.mean(row[0] for row in usable)
    mean_y = statistics.mean(row[1] for row in usable)
    denominator = sum((row[0] - mean_x) ** 2 for row in usable)
    if denominator == 0:
        return None
    return sum(
        (row[0] - mean_x) * (row[1] - mean_y)
        for row in usable
    ) / denominator


def curve_generator(curve: dict[str, Any]) -> Point:
    for x_coordinate in range(curve["field_prime"]):
        point = R70.point_from_x(x_coordinate, 0, curve)
        if point is None:
            continue
        point = canonical_point(point, curve)
        if point is not None:
            if R70.scalar_mul(curve["subgroup_order"], point, curve) is not None:
                raise AssertionError("generator candidate missed subgroup")
            return point
    raise AssertionError("prime-subgroup generator not found")


def validate_family(curve: dict[str, Any], generator: Point) -> dict[str, Any]:
    prime = curve["field_prime"]
    order = curve["subgroup_order"]
    coordinate_order = curve["coordinate_subgroup_order"]
    return {
        "field_prime_probable_prime": R70.is_prime(prime),
        "subgroup_order_probable_prime": R70.is_prime(order),
        "j_zero_supersingular_order_identity": (
            prime % 3 == 2
            and curve["curve_a"] == 0
            and prime + 1 == curve["cofactor"] * order
        ),
        "coordinate_subgroup_divides_field_group": (
            (prime - 1) % coordinate_order == 0
        ),
        "curve_discriminant_nonzero": R70.curve_discriminant(curve) != 0,
        "generator_nonidentity": generator is not None,
        "generator_has_stated_order": (
            generator is not None
            and R70.scalar_mul(order, generator, curve) is None
        ),
    }


def full_coset_factor_base(
    curve: dict[str, Any],
    offset: int,
) -> tuple[list[Point], dict[str, Any]]:
    prime = curve["field_prime"]
    order = curve["coordinate_subgroup_order"]
    primitive = R70.primitive_root(prime)
    step = pow(primitive, (prime - 1) // order, prime)
    multiplier = pow(primitive, offset, prime)
    coordinates = [
        multiplier * pow(step, index, prime) % prime
        for index in range(order)
    ]
    points_by_x: dict[int, Point] = {}
    liftable_count = 0
    cleared_identity_count = 0
    for x_coordinate in coordinates:
        point = R70.point_from_x(x_coordinate, 0, curve)
        if point is None:
            continue
        liftable_count += 1
        point = canonical_point(point, curve)
        if point is None:
            cleared_identity_count += 1
            continue
        points_by_x.setdefault(point[0], point)
    points = [points_by_x[x_coordinate] for x_coordinate in sorted(points_by_x)]
    coset_constant = pow(multiplier, order, prime)
    if any(
        (pow(x_coordinate, order, prime) - coset_constant) % prime
        for x_coordinate in coordinates
    ):
        raise AssertionError("coordinate escaped frozen multiplicative coset")
    if len(set(coordinates)) != order:
        raise AssertionError("coordinate coset did not have full order")
    return points, {
        "coset_offset": offset,
        "field_primitive_root": primitive,
        "coordinate_subgroup_order": order,
        "coordinate_subgroup_generator": step,
        "coset_multiplier": multiplier,
        "coset_constant": coset_constant,
        "sparse_domain_polynomial": {
            "degree": order,
            "nonzero_coefficient_count": 2,
            "formula": "X^d-c",
        },
        "enumerated_coordinate_count": len(coordinates),
        "liftable_coordinate_count": liftable_count,
        "cofactor_cleared_identity_count": cleared_identity_count,
        "deduplicated_kummer_factor_base_size": len(points),
        "scalar_labels_consumed": False,
    }


def hash_control_base(
    curve: dict[str, Any],
    base_size: int,
    salt: str,
) -> list[Point]:
    points_by_x: dict[int, Point] = {}
    counter = 0
    while len(points_by_x) < base_size:
        digest = hashlib.sha256(
            f"P1553-R81|{curve['family_id']}|{salt}|{counter}".encode()
        ).digest()
        counter += 1
        x_coordinate = int.from_bytes(digest, "big") % curve["field_prime"]
        point = R70.point_from_x(x_coordinate, digest[0] & 1, curve)
        if point is None:
            continue
        point = canonical_point(point, curve)
        if point is None:
            continue
        points_by_x.setdefault(point[0], point)
    return [points_by_x[x_coordinate] for x_coordinate in sorted(points_by_x)]


class BatchBsgsVerifier:
    def __init__(self, generator: Point, curve: dict[str, Any]) -> None:
        if generator is None:
            raise AssertionError("BSGS requires a nonidentity generator")
        self.generator = generator
        self.curve = curve
        self.order = curve["subgroup_order"]
        self.width = math.isqrt(self.order) + 1
        self.baby: dict[Point, int] = {}
        point: Point = None
        for index in range(self.width):
            self.baby.setdefault(point, index)
            point = R70.add(point, generator, curve)
        stride = R70.scalar_mul(self.width, generator, curve)
        self.negative_stride = R70.negate(stride, curve)
        self.giant_steps = 0

    def discrete_log(self, target: Point) -> int:
        probe = target
        for giant in range(self.width + 1):
            baby = self.baby.get(probe)
            if baby is not None or probe is None:
                value = (giant * self.width + int(baby or 0)) % self.order
                if R70.scalar_mul(value, self.generator, self.curve) != target:
                    raise AssertionError("BSGS verifier returned wrong label")
                self.giant_steps += giant
                return value
            probe = R70.add(probe, self.negative_stride, self.curve)
        raise AssertionError("BSGS verifier missed subgroup point")

    def labels(self, points: Sequence[Point]) -> list[int]:
        return [self.discrete_log(point) for point in points]

    def receipt(self) -> dict[str, Any]:
        return {
            "algorithm": "baby_step_giant_step_verifier_only",
            "baby_table_size": len(self.baby),
            "width": self.width,
            "total_giant_steps": self.giant_steps,
            "asymptotic_group_work_exponent_B": 2.5,
            "candidate_algorithm_may_consume_labels": False,
            "excluded_from_algorithmic_credit": True,
        }


def signed_point(
    point: Point,
    sign: int,
    curve: dict[str, Any],
) -> Point:
    return point if sign == 1 else R70.negate(point, curve)


def verify_endpoint_samples(
    points: Sequence[Point],
    labels: Sequence[int],
    generator: Point,
    curve: dict[str, Any],
    sample_count: int = 64,
) -> dict[str, Any]:
    combinations = list(
        itertools.combinations_with_replacement(range(len(points)), 3)
    )
    if len(combinations) > sample_count:
        chosen = [
            combinations[
                int.from_bytes(
                    hashlib.sha256(
                        f"{curve['family_id']}|endpoint|{index}".encode()
                    ).digest(),
                    "big",
                )
                % len(combinations)
            ]
            for index in range(sample_count)
        ]
    else:
        chosen = combinations
    failures = []
    check_count = 0
    signs = (
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    )
    for indices in chosen:
        for sign_row in signs:
            point: Point = None
            scalar = 0
            for index, sign in zip(indices, sign_row):
                point = R70.add(
                    point,
                    signed_point(points[index], sign, curve),
                    curve,
                )
                scalar = (
                    scalar + sign * labels[index]
                ) % curve["subgroup_order"]
            scalar_point = R70.scalar_mul(scalar, generator, curve)
            check_count += 1
            if point != scalar_point:
                failures.append(
                    {
                        "indices": list(indices),
                        "signs": list(sign_row),
                    }
                )
    return {
        "sampled_unordered_triples": len(chosen),
        "signed_endpoint_checks": check_count,
        "failure_count": len(failures),
        "all_group_endpoints_match_verifier_labels": not failures,
        "failures": failures[:8],
    }


def triple_endpoint_profile(
    labels: Sequence[int],
    subgroup_order: int,
) -> dict[str, Any]:
    base_size = len(labels)
    signs = (
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    )
    root_histogram: collections.Counter[int] = collections.Counter()
    endpoint_set_histogram: collections.Counter[
        tuple[int, ...]
    ] = collections.Counter()
    for indices in itertools.combinations_with_replacement(
        range(base_size),
        3,
    ):
        roots = set()
        for sign_row in signs:
            scalar = sum(
                sign * labels[index]
                for index, sign in zip(indices, sign_row)
            ) % subgroup_order
            roots.add(min(scalar, (-scalar) % subgroup_order))
        weight = permutation_weight(indices)
        endpoint_key = tuple(sorted(roots))
        endpoint_set_histogram[endpoint_key] += weight
        for root in roots:
            root_histogram[root] += weight
    if sum(endpoint_set_histogram.values()) != base_size**3:
        raise AssertionError("ordered triple occurrence count drifted")
    root_count = len(root_histogram)
    endpoint_set_count = len(endpoint_set_histogram)
    unordered_triple_count = math.comb(base_size + 2, 3)
    return {
        "base_size": base_size,
        "ordered_triple_occurrence_count": base_size**3,
        "unordered_triple_count": unordered_triple_count,
        "distinct_kummer_root_count": root_count,
        "distinct_endpoint_set_count": endpoint_set_count,
        "unordered_triple_to_endpoint_set_injective": (
            endpoint_set_count == unordered_triple_count
        ),
        "endpoint_set_fraction_of_unordered_triples": (
            endpoint_set_count / unordered_triple_count
        ),
        "root_support_fraction_of_four_per_unordered_triple": (
            root_count / (4 * unordered_triple_count)
        ),
        "root_support_exponent_B": support_exponent(root_count, base_size),
        "endpoint_set_exponent_B": support_exponent(
            endpoint_set_count,
            base_size,
        ),
        "root_support_inside_setup_cap": (
            root_count <= base_size**SETUP_STATE_CAP_EXPONENT
        ),
        "endpoint_sets_inside_setup_cap": (
            endpoint_set_count <= base_size**SETUP_STATE_CAP_EXPONENT
        ),
        "maximum_root_occurrence_multiplicity": max(
            root_histogram.values(),
            default=0,
        ),
        "root_collision_energy": sum(
            count * count for count in root_histogram.values()
        ),
    }


def coefficient_row(
    atoms: Iterable[tuple[int, int]],
    base_size: int,
) -> tuple[int, ...]:
    row = [0] * base_size
    for index, sign in atoms:
        row[index] += sign
    return tuple(row)


def rank_mod(rows: Iterable[Sequence[int]], modulus: int) -> int:
    basis: dict[int, list[int]] = {}
    for source_row in rows:
        row = [value % modulus for value in source_row]
        for pivot in sorted(basis):
            if not row[pivot]:
                continue
            factor = row[pivot]
            row = [
                (left - factor * right) % modulus
                for left, right in zip(row, basis[pivot])
            ]
        pivot = next((index for index, value in enumerate(row) if value), None)
        if pivot is None:
            continue
        inverse = pow(row[pivot], -1, modulus)
        row = [value * inverse % modulus for value in row]
        for old_pivot, old_row in list(basis.items()):
            if old_row[pivot]:
                factor = old_row[pivot]
                basis[old_pivot] = [
                    (left - factor * right) % modulus
                    for left, right in zip(old_row, row)
                ]
        basis[pivot] = row
    return len(basis)


def target_count(
    target: int,
    pair_counts: collections.Counter[int],
    triple_counts: collections.Counter[int],
    modulus: int,
) -> int:
    return sum(
        count
        * triple_counts.get((target - pair_sum) % modulus, 0)
        for pair_sum, count in pair_counts.items()
    )


def target_source(
    target: int,
    pair_first: dict[int, tuple[tuple[int, int], ...]],
    triple_first: dict[int, tuple[tuple[int, int], ...]],
    modulus: int,
    base_size: int,
) -> tuple[int, ...] | None:
    for pair_sum, pair_source in pair_first.items():
        triple_source = triple_first.get((target - pair_sum) % modulus)
        if triple_source is not None:
            return coefficient_row(
                (*pair_source, *triple_source),
                base_size,
            )
    return None


def relation_density_rank_profile(
    labels: Sequence[int],
    subgroup_order: int,
    sample_salt: str,
    sample_count: int,
) -> dict[str, Any]:
    base_size = len(labels)
    atoms = [
        (sign * label % subgroup_order, index, sign)
        for index, label in enumerate(labels)
        for sign in (1, -1)
    ]
    pair_counts: collections.Counter[int] = collections.Counter()
    pair_sources: dict[
        int,
        list[tuple[tuple[int, int], tuple[int, int]]],
    ] = collections.defaultdict(list)
    pair_first: dict[int, tuple[tuple[int, int], ...]] = {}
    for left in atoms:
        for right in atoms:
            value = (left[0] + right[0]) % subgroup_order
            source = ((left[1], left[2]), (right[1], right[2]))
            pair_counts[value] += 1
            pair_sources[value].append(source)
            pair_first.setdefault(value, source)

    triple_counts: collections.Counter[int] = collections.Counter()
    triple_first: dict[int, tuple[tuple[int, int], ...]] = {}
    for pair_sum, count in pair_counts.items():
        for atom in atoms:
            value = (pair_sum + atom[0]) % subgroup_order
            triple_counts[value] += count
            triple_first.setdefault(
                value,
                (*pair_first[pair_sum], (atom[1], atom[2])),
            )

    targets = [0, 1]
    counter = 0
    while len(targets) < sample_count + 2:
        digest = hashlib.sha256(
            f"P1553-R81-target|{sample_salt}|{counter}".encode()
        ).digest()
        counter += 1
        target = int.from_bytes(digest, "big") % subgroup_order
        if target not in targets:
            targets.append(target)
    counts = [
        target_count(
            target,
            pair_counts,
            triple_counts,
            subgroup_order,
        )
        for target in targets
    ]
    target_sources = [
        target_source(
            target,
            pair_first,
            triple_first,
            subgroup_order,
            base_size,
        )
        for target in targets
    ]
    inhomogeneous_rows = [
        source
        for source in target_sources
        if source is not None
    ]
    inhomogeneous_targets = [
        target
        for target, source in zip(targets, target_sources)
        if source is not None
    ]
    if any(
        sum(
            coefficient * label
            for coefficient, label in zip(source, labels)
        )
        % subgroup_order
        != target
        for source, target in zip(
            inhomogeneous_rows,
            inhomogeneous_targets,
        )
    ):
        raise AssertionError("inhomogeneous relation row failed verifier labels")
    inhomogeneous_rank = rank_mod(
        inhomogeneous_rows,
        subgroup_order,
    )

    relation_rows = set()
    enumerated_zero_occurrences = 0
    for first in atoms:
        for second in atoms:
            for third in atoms:
                triple_sum = (
                    first[0] + second[0] + third[0]
                ) % subgroup_order
                matches = pair_sources.get((-triple_sum) % subgroup_order, ())
                enumerated_zero_occurrences += len(matches)
                triple_source = (
                    (first[1], first[2]),
                    (second[1], second[2]),
                    (third[1], third[2]),
                )
                for pair_source in matches:
                    relation_rows.add(
                        coefficient_row(
                            (*pair_source, *triple_source),
                            base_size,
                        )
                    )
    if enumerated_zero_occurrences != counts[0]:
        raise AssertionError("zero relation occurrence replay drifted")
    if any(
        sum(coefficient * label for coefficient, label in zip(row, labels))
        % subgroup_order
        for row in relation_rows
    ):
        raise AssertionError("relation row failed verifier labels")

    relation_rank = rank_mod(relation_rows, subgroup_order)
    generator_source = target_source(
        1,
        pair_first,
        triple_first,
        subgroup_order,
        base_size,
    )
    generator_source_verified = (
        generator_source is not None
        and sum(
            coefficient * label
            for coefficient, label in zip(generator_source, labels)
        )
        % subgroup_order
        == 1
    )
    random_counts = counts[2:]
    return {
        "signed_atom_count": len(atoms),
        "pair_sum_support": len(pair_counts),
        "triple_sum_support": len(triple_counts),
        "five_sum_support_upper_bound": min(
            subgroup_order,
            len(pair_counts) * len(triple_counts),
        ),
        "uniform_expected_signed_tuple_count": (
            len(atoms) ** 5 / subgroup_order
        ),
        "zero_target_exact_occurrence_count": counts[0],
        "generator_target_exact_occurrence_count": counts[1],
        "sampled_random_target_count": len(random_counts),
        "sampled_random_positive_count": sum(
            count > 0 for count in random_counts
        ),
        "sampled_random_positive_rate": (
            sum(count > 0 for count in random_counts) / len(random_counts)
            if random_counts
            else 0.0
        ),
        "sampled_random_mean_occurrence_count": (
            statistics.mean(random_counts) if random_counts else 0.0
        ),
        "unique_zero_relation_row_count": len(relation_rows),
        "zero_relation_row_rank": relation_rank,
        "zero_relation_row_nullity": base_size - relation_rank,
        "full_homogeneous_factor_log_rank": relation_rank == base_size - 1,
        "sampled_inhomogeneous_relation_row_count": len(
            inhomogeneous_rows
        ),
        "sampled_inhomogeneous_relation_rank": inhomogeneous_rank,
        "full_inhomogeneous_factor_log_rank": (
            inhomogeneous_rank == base_size
        ),
        "verifier_only_factor_logs_satisfy_all_inhomogeneous_rows": True,
        "verifier_only_sampled_factor_logs_unique": (
            inhomogeneous_rank == base_size
        ),
        "generator_anchor_source_present": generator_source is not None,
        "generator_anchor_source_verified": generator_source_verified,
        "verifier_only_anchored_factor_logs_unique": (
            relation_rank == base_size - 1 and generator_source_verified
        ),
    }


def analyze_base(
    points: Sequence[Point],
    labels: Sequence[int],
    generator: Point,
    curve: dict[str, Any],
    sample_salt: str,
    target_count_value: int,
    group_sample_count: int = 64,
) -> dict[str, Any]:
    if len(points) != len(labels):
        raise AssertionError("point/label size mismatch")
    if any(
        R70.scalar_mul(label, generator, curve) != point
        for point, label in zip(points, labels)
    ):
        raise AssertionError("verifier label failed point replay")
    return {
        "base_size": len(points),
        "scale": {
            "subgroup_order_over_B5": (
                curve["subgroup_order"] / len(points) ** 5
            ),
            "subgroup_order_over_B3": (
                curve["subgroup_order"] / len(points) ** 3
            ),
            "triple_support_not_group_saturated": (
                curve["subgroup_order"] > 8 * len(points) ** 3
            ),
        },
        "scalar_labels_consumed_by_candidate": False,
        "verifier_labels_all_exact": True,
        "endpoint_profile": triple_endpoint_profile(
            labels,
            curve["subgroup_order"],
        ),
        "group_endpoint_sample_replay": verify_endpoint_samples(
            points,
            labels,
            generator,
            curve,
            sample_count=group_sample_count,
        ),
        "relation_density_rank": relation_density_rank_profile(
            labels,
            curve["subgroup_order"],
            sample_salt,
            max(target_count_value, 4 * len(points)),
        ),
    }


def analyze_family(
    curve: dict[str, Any],
    offsets: Sequence[int],
    control_count: int,
    target_count_value: int,
) -> dict[str, Any]:
    generator = curve_generator(curve)
    verifier = BatchBsgsVerifier(generator, curve)
    rows = []
    for offset in offsets:
        points, geometry = full_coset_factor_base(curve, offset)
        labels = verifier.labels(points)
        candidate = analyze_base(
            points,
            labels,
            generator,
            curve,
            f"{curve['family_id']}|coset|{offset}",
            target_count_value,
        )
        controls = []
        for control_index in range(control_count):
            control_points = hash_control_base(
                curve,
                len(points),
                f"{offset}|{control_index}",
            )
            control_labels = verifier.labels(control_points)
            controls.append(
                analyze_base(
                    control_points,
                    control_labels,
                    generator,
                    curve,
                    (
                        f"{curve['family_id']}|coset|{offset}|"
                        f"control|{control_index}"
                    ),
                    target_count_value,
                    group_sample_count=16,
                )
            )
        control_root_exponents = [
            control["endpoint_profile"]["root_support_exponent_B"]
            for control in controls
        ]
        control_density_rates = [
            control["relation_density_rank"][
                "sampled_random_positive_rate"
            ]
            for control in controls
        ]
        control_ranks = [
            control["relation_density_rank"][
                "sampled_inhomogeneous_relation_rank"
            ]
            for control in controls
        ]
        rows.append(
            {
                "geometry": geometry,
                "candidate": candidate,
                "matched_hash_controls": controls,
                "comparison": {
                    "candidate_root_exponent_minus_control_mean": (
                        candidate["endpoint_profile"][
                            "root_support_exponent_B"
                        ]
                        - statistics.mean(control_root_exponents)
                    ),
                    "candidate_root_exponent_below_every_control": (
                        candidate["endpoint_profile"][
                            "root_support_exponent_B"
                        ]
                        < min(control_root_exponents)
                    ),
                    "candidate_density_minus_control_mean": (
                        candidate["relation_density_rank"][
                            "sampled_random_positive_rate"
                        ]
                        - statistics.mean(control_density_rates)
                    ),
                    "candidate_relation_rank": candidate[
                        "relation_density_rank"
                    ]["sampled_inhomogeneous_relation_rank"],
                    "control_relation_ranks": control_ranks,
                },
            }
        )
    return {
        "family": curve,
        "family_validation": validate_family(curve, generator),
        "generator": list(generator) if generator is not None else None,
        "cosets": rows,
        "verifier_bsgs_receipt": verifier.receipt(),
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = FAMILIES,
    offsets: Sequence[int] = COSET_OFFSETS,
    control_count: int = CONTROL_COUNT,
    target_count_value: int = TARGET_SAMPLE_COUNT,
) -> dict[str, dict[str, Any]]:
    family_rows = [
        analyze_family(
            dict(curve),
            offsets,
            control_count,
            target_count_value,
        )
        for curve in families
    ]
    coset_rows = [
        row for family in family_rows for row in family["cosets"]
    ]
    candidate_rows = [row["candidate"] for row in coset_rows]
    control_rows = [
        control
        for row in coset_rows
        for control in row["matched_hash_controls"]
    ]
    endpoint_profiles = [
        row["endpoint_profile"] for row in candidate_rows
    ]
    maximum_root_exponent = max(
        row["root_support_exponent_B"] for row in endpoint_profiles
    )
    minimum_root_exponent = min(
        row["root_support_exponent_B"] for row in endpoint_profiles
    )
    maximum_endpoint_set_exponent = max(
        row["endpoint_set_exponent_B"] for row in endpoint_profiles
    )
    all_family_checks = all(
        all(family["family_validation"].values())
        for family in family_rows
    )
    all_group_replays = all(
        row["group_endpoint_sample_replay"][
            "all_group_endpoints_match_verifier_labels"
        ]
        for row in (*candidate_rows, *control_rows)
    )
    all_root_caps = all(
        row["root_support_inside_setup_cap"]
        for row in endpoint_profiles
    )
    all_endpoint_set_caps = all(
        row["endpoint_sets_inside_setup_cap"]
        for row in endpoint_profiles
    )
    any_root_cap = any(
        row["root_support_inside_setup_cap"]
        for row in endpoint_profiles
    )
    all_scalar_blind = all(
        not row["scalar_labels_consumed_by_candidate"]
        for row in candidate_rows
    )
    all_endpoint_set_injective = all(
        row["unordered_triple_to_endpoint_set_injective"]
        for row in endpoint_profiles
    )
    candidate_root_scaling_slope = power_law_slope(
        [
            (row["base_size"], row["distinct_kummer_root_count"])
            for row in endpoint_profiles
        ]
    )
    candidate_endpoint_set_scaling_slope = power_law_slope(
        [
            (row["base_size"], row["distinct_endpoint_set_count"])
            for row in endpoint_profiles
        ]
    )
    control_root_scaling_slope = power_law_slope(
        [
            (
                row["endpoint_profile"]["base_size"],
                row["endpoint_profile"]["distinct_kummer_root_count"],
            )
            for row in control_rows
        ]
    )
    root_exponent_deltas = [
        row["comparison"]["candidate_root_exponent_minus_control_mean"]
        for row in coset_rows
    ]
    density_deltas = [
        row["comparison"]["candidate_density_minus_control_mean"]
        for row in coset_rows
    ]
    candidate_full_rank_count = sum(
        row["relation_density_rank"]["full_inhomogeneous_factor_log_rank"]
        for row in candidate_rows
    )
    control_full_rank_count = sum(
        row["relation_density_rank"]["full_inhomogeneous_factor_log_rank"]
        for row in control_rows
    )

    geometry = {
        "schema": "p1553.frozen_full_multiplicative_x_coset_geometry.r81.v1",
        "geometry_id": "full_x_coset_lift_mask_cofactor_clear_v1",
        "curve_model": "y^2=x^3+1 with #E(F_p)=p+1=6q",
        "coset_offsets": list(offsets),
        "candidate_selection_after_outcomes": False,
        "scalar_labels_consumed": False,
        "families": [
            {
                "family": family["family"],
                "family_validation": family["family_validation"],
                "cosets": [row["geometry"] for row in family["cosets"]],
            }
            for family in family_rows
        ],
    }
    support_replay = {
        "schema": "p1553.triple_endpoint_support_lift_mask_replay.r81.v1",
        "families": [
            {
                "family_id": family["family"]["family_id"],
                "cosets": [
                    {
                        "coset_offset": row["geometry"]["coset_offset"],
                        "liftable_coordinate_count": row["geometry"][
                            "liftable_coordinate_count"
                        ],
                        "base_size": row["candidate"]["base_size"],
                        "scale": row["candidate"]["scale"],
                        "endpoint_profile": row["candidate"][
                            "endpoint_profile"
                        ],
                        "group_endpoint_sample_replay": row["candidate"][
                            "group_endpoint_sample_replay"
                        ],
                    }
                    for row in family["cosets"]
                ],
            }
            for family in family_rows
        ],
        "all_group_endpoint_samples_exact": all_group_replays,
        "any_candidate_root_support_inside_cap": any_root_cap,
        "all_candidate_root_support_inside_cap": all_root_caps,
        "all_candidate_endpoint_sets_inside_cap": all_endpoint_set_caps,
        "all_unordered_triple_endpoint_set_maps_injective": (
            all_endpoint_set_injective
        ),
        "candidate_root_support_log_log_slope": (
            candidate_root_scaling_slope
        ),
        "candidate_endpoint_set_log_log_slope": (
            candidate_endpoint_set_scaling_slope
        ),
        "matched_control_root_support_log_log_slope": (
            control_root_scaling_slope
        ),
    }
    matched_controls = {
        "schema": "p1553.matched_random_density_rank_controls.r81.v1",
        "control_count_per_coset": control_count,
        "rows": [
            {
                "family_id": family["family"]["family_id"],
                "cosets": [
                    {
                        "coset_offset": row["geometry"]["coset_offset"],
                        "candidate": {
                            "base_size": row["candidate"]["base_size"],
                            "endpoint_profile": row["candidate"][
                                "endpoint_profile"
                            ],
                            "relation_density_rank": row["candidate"][
                                "relation_density_rank"
                            ],
                        },
                        "controls": [
                            {
                                "base_size": control["base_size"],
                                "endpoint_profile": control[
                                    "endpoint_profile"
                                ],
                                "relation_density_rank": control[
                                    "relation_density_rank"
                                ],
                            }
                            for control in row["matched_hash_controls"]
                        ],
                        "comparison": row["comparison"],
                    }
                    for row in family["cosets"]
                ],
            }
            for family in family_rows
        ],
    }
    cost_ledger = {
        "schema": "p1553.scalar_blind_source_cost_ledger.r81.v1",
        "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
        "online_work_cap_exponent_B": ONLINE_CAP_EXPONENT,
        "online_workspace_cap_exponent_B": ONLINE_CAP_EXPONENT,
        "candidate_geometry": {
            "sparse_coordinate_domain_coefficients_exponent_B": 0.0,
            "factor_base_and_lift_mask_state_exponent_B": 1.0,
            "measured_minimum_triple_root_support_exponent_B": (
                minimum_root_exponent
            ),
            "measured_maximum_triple_root_support_exponent_B": (
                maximum_root_exponent
            ),
            "measured_maximum_endpoint_set_exponent_B": (
                maximum_endpoint_set_exponent
            ),
            "measured_root_support_log_log_slope": (
                candidate_root_scaling_slope
            ),
            "measured_endpoint_set_log_log_slope": (
                candidate_endpoint_set_scaling_slope
            ),
            "all_frozen_unordered_triple_endpoint_maps_injective": (
                all_endpoint_set_injective
            ),
            "standard_exact_triple_enumeration_work_exponent_B": 3.0,
            "standard_fresh_pair_work_exponent_B": 2.0,
            "standard_source_return_work_exponent_B": 3.0,
            "inside_direct_caps": (
                all_root_caps
                and all_endpoint_set_caps
                and 3.0 <= ONLINE_CAP_EXPONENT
            ),
        },
        "verifier_only": {
            "bsgs_exponent_B": 2.5,
            "labels_used_only_for_exact_support_rank_and_density_audit": True,
            "candidate_algorithm_may_consume_labels": False,
            "verifier_work_exceeds_setup_cap": True,
        },
        "failure_reasons": [
            "the sparse X^d-c domain does not imply compressed S4 endpoint support",
            "the exact lift mask and elliptic addition destroy one-dimensional coset closure",
            "standard scalar-blind source return remains B^3",
            "verifier BSGS labels are rho-scale evidence only, not an algorithm",
        ],
    }

    obligations = {
        "four_scale_matched_prime_order_families": len(family_rows) == 4,
        "all_family_parameter_checks": all_family_checks,
        "complete_cosets_not_prefixes": all(
            row["geometry"]["enumerated_coordinate_count"]
            == row["geometry"]["coordinate_subgroup_order"]
            for row in coset_rows
        ),
        "sparse_domain_polynomial_exact": all(
            row["geometry"]["sparse_domain_polynomial"][
                "nonzero_coefficient_count"
            ]
            == 2
            for row in coset_rows
        ),
        "scalar_blind_candidate_construction": all_scalar_blind,
        "group_endpoint_replay_exact": all_group_replays,
        "matched_hash_controls_present": (
            len(control_rows) == len(coset_rows) * control_count
        ),
        "root_support_inside_setup_cap": all_root_caps,
        "endpoint_sets_inside_setup_cap": all_endpoint_set_caps,
        "online_source_work_inside_cap": 3.0 <= ONLINE_CAP_EXPONENT,
        "candidate_factor_log_solve_without_verifier_dlp": False,
        "candidate_identical_fresh_target_descent": False,
        "generic_prime_family_theorem": False,
        "shoup_improvement_complete": False,
    }
    failures = [
        name for name, passed in obligations.items() if not passed
    ]
    passed_count = sum(obligations.values())

    report = {
        "schema": SCHEMA,
        "classification": (
            "FULL_MULTIPLICATIVE_X_COSET_LIFT_MASK_FAILS_ENDPOINT_CAP"
        ),
        "source_bindings": {
            "r80_batched_compiler": {
                "path": str(R80_REPORT),
                "sha256": R80_REPORT_SHA256,
            },
            "r70_multiplicative_x_prefix_screen": {
                "path": str(R70_REPORT),
                "sha256": R70_REPORT_SHA256,
            },
            "r31_query2p1_registry": {
                "path": str(R31_REGISTRY),
                "sha256": R31_REGISTRY_SHA256,
                "bound_gate_sha256": R3_GATE_SHA256,
            },
        },
        "novelty_scope": (
            "R70 tested fixed short prefixes of multiplicative-x sources. "
            "R81 freezes complete multiplicative coordinate cosets, retains "
            "their exact lift masks, and evaluates endpoint compression at "
            "q=Theta(B^5) before looking at outcomes."
        ),
        "families": family_rows,
        "aggregate": {
            "family_count": len(family_rows),
            "coset_instance_count": len(coset_rows),
            "matched_control_instance_count": len(control_rows),
            "all_family_parameter_checks": all_family_checks,
            "all_group_endpoint_samples_exact": all_group_replays,
            "minimum_candidate_root_support_exponent_B": minimum_root_exponent,
            "maximum_candidate_root_support_exponent_B": maximum_root_exponent,
            "maximum_candidate_endpoint_set_exponent_B": (
                maximum_endpoint_set_exponent
            ),
            "candidate_root_support_log_log_slope": (
                candidate_root_scaling_slope
            ),
            "candidate_endpoint_set_log_log_slope": (
                candidate_endpoint_set_scaling_slope
            ),
            "matched_control_root_support_log_log_slope": (
                control_root_scaling_slope
            ),
            "minimum_subgroup_order_over_B3": min(
                row["scale"]["subgroup_order_over_B3"]
                for row in candidate_rows
            ),
            "minimum_subgroup_order_over_B5": min(
                row["scale"]["subgroup_order_over_B5"]
                for row in candidate_rows
            ),
            "maximum_subgroup_order_over_B5": max(
                row["scale"]["subgroup_order_over_B5"]
                for row in candidate_rows
            ),
            "all_triple_supports_below_group_saturation": all(
                row["scale"]["triple_support_not_group_saturated"]
                for row in candidate_rows
            ),
            "all_unordered_triple_endpoint_set_maps_injective": (
                all_endpoint_set_injective
            ),
            "candidate_cosets_inside_root_support_cap": sum(
                row["root_support_inside_setup_cap"]
                for row in endpoint_profiles
            ),
            "candidate_cosets_inside_endpoint_set_cap": sum(
                row["endpoint_sets_inside_setup_cap"]
                for row in endpoint_profiles
            ),
            "candidate_cosets_with_full_sampled_inhomogeneous_rank": (
                candidate_full_rank_count
            ),
            "controls_with_full_sampled_inhomogeneous_rank": (
                control_full_rank_count
            ),
            "mean_candidate_root_exponent_minus_control_mean": (
                statistics.mean(root_exponent_deltas)
            ),
            "maximum_absolute_candidate_root_exponent_delta": max(
                abs(value) for value in root_exponent_deltas
            ),
            "mean_candidate_density_minus_control_mean": statistics.mean(
                density_deltas
            ),
            "candidate_cosets_with_generator_anchor": sum(
                row["relation_density_rank"][
                    "generator_anchor_source_verified"
                ]
                for row in candidate_rows
            ),
        },
        "side_artifacts": {
            "geometry": "frozen_full_multiplicative_x_coset_geometry.json",
            "support_replay": (
                "triple_endpoint_support_and_lift_mask_replay.json"
            ),
            "matched_controls": "matched_random_density_rank_controls.json",
            "cost_ledger": "scalar_blind_source_and_cost_ledger.json",
        },
        "cost_ledger": cost_ledger,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed_count,
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This rejects only complete one-dimensional multiplicative-x "
            "cosets with the canonical lift mask and cofactor map. It is not "
            "a lower bound against higher-dimensional compact divisors, "
            "non-Cartesian factor bases, or unknown structured circuits."
        ),
        "next_action": (
            "Leave one-dimensional field cosets and construct one "
            "scalar-blind compact divisor factor base whose triple S4 "
            "endpoint representation is proved at most B^(9/4+o(1)) before "
            "enumeration. Require a field-level source query at most "
            "B^(5/4+o(1)), prospective density, matched rank, verified logs, "
            "and identical target descent; do not use verifier DLP labels."
        ),
        "disposition": (
            "REJECT_FULL_MULTIPLICATIVE_X_COSET_LIFT_MASK_GEOMETRY_ONLY__"
            "FOUR_Q_THETA_B5_PRIME_ORDER_FAMILIES__SIXTEEN_COMPLETE_COSETS__"
            "THIRTY_TWO_MATCHED_HASH_CONTROLS__SPARSE_XD_MINUS_C_DOMAIN__"
            "EXACT_LIFT_MASK_AND_COFACTOR_MAP__EXACT_GROUP_ENDPOINT_REPLAY__"
            "TRIPLE_ROOT_AND_ENDPOINT_SET_CAP_FAILURE__VERIFIER_BSGS_B2P5_"
            "EXCLUDED__NO_SCALAR_BLIND_SUBCAP_SOURCE__NO_FACTOR_LOGS__NO_"
            "DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "geometry": geometry,
        "support_replay": support_replay,
        "matched_controls": matched_controls,
        "cost_ledger": cost_ledger,
    }


def build_report(
    families: Sequence[dict[str, Any]] = FAMILIES,
    offsets: Sequence[int] = COSET_OFFSETS,
    control_count: int = CONTROL_COUNT,
    target_count_value: int = TARGET_SAMPLE_COUNT,
) -> dict[str, Any]:
    return build_bundle(
        families=families,
        offsets=offsets,
        control_count=control_count,
        target_count_value=target_count_value,
    )["report"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_full_multiplicative_x_coset_endpoint_probe_report_r81.json"
        ),
    )
    parser.add_argument(
        "--geometry-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_full_multiplicative_x_coset_geometry.json"
        ),
    )
    parser.add_argument(
        "--support-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "triple_endpoint_support_and_lift_mask_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path("matched_random_density_rank_controls.json"),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("scalar_blind_source_and_cost_ledger.json"),
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
    write_json(args.support_output, bundle["support_replay"])
    write_json(args.controls_output, bundle["matched_controls"])
    write_json(args.cost_output, bundle["cost_ledger"])
    aggregate = bundle["report"]["aggregate"]
    print(
        f"families={aggregate['family_count']} "
        f"cosets={aggregate['coset_instance_count']} "
        f"root_cap={aggregate['candidate_cosets_inside_root_support_cap']} "
        f"endpoint_cap="
        f"{aggregate['candidate_cosets_inside_endpoint_set_cap']} "
        f"lane_admitted={bundle['report']['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
