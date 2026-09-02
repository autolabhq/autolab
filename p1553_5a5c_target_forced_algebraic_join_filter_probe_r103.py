#!/usr/bin/env python3
"""Audit the target-forced S3 filter and its standard FFE representations."""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_target_forced_algebraic_join_filter.r103.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
LEFT_SIDE_EXPONENT = Fraction(13, 5)
RIGHT_SIDE_EXPONENT = Fraction(12, 5)
RIGHT_LOCAL_QUERY_EXPONENT = Fraction(6, 5)
LEFT_LOCAL_QUERY_EXPONENT = Fraction(4, 5)
CANONICAL_PREFIX_STATE_EXPONENT = Fraction(11, 5)
CANONICAL_SUFFIX_QUERY_EXPONENT = Fraction(14, 5)
POINTWISE_LEFT_FIRST_EXPONENT = (
    LEFT_SIDE_EXPONENT + RIGHT_LOCAL_QUERY_EXPONENT
)
POINTWISE_RIGHT_FIRST_EXPONENT = (
    RIGHT_SIDE_EXPONENT + LEFT_LOCAL_QUERY_EXPONENT
)

R102_PRODUCER = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_r102.py"
)
R102_PRODUCER_SHA256 = (
    "3fae17c49683586178518a0df8d377c8f553bd131539f1bab579c0f55d5b0742"
)
R102_REPORT = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_report_r102.json"
)
R102_REPORT_SHA256 = (
    "5f16489b5f0535df6c9728edfed5f2f83641a4b9c30fd2609db52a73f46d40e7"
)
R102_GATE = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_gate_r102.md"
)
R102_GATE_SHA256 = (
    "1af82597762525110baf31cec7b4e7b96693a730270f7e1bd6556906e5c5f56c"
)
R102_PARENT = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_parent_report_r102.yaml"
)
R102_PARENT_SHA256 = (
    "9a965158bc96e51dd8daea1abb86f5552b75dac94998633ec97b22f8b9a3ad67"
)
R101_REPORT = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_"
    "merge_probe_report_r101.json"
)
R101_REPORT_SHA256 = (
    "050d1093fc34611f502a3f7b3ebf6fc632b3aeb9d03718f922e99baa2fe985c9"
)
R101_GATE = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_gate_r101.md"
)
R101_GATE_SHA256 = (
    "3e12e4ab70edba64076d27b68d39ab34307ca6d9ff08252432759673b74a9ac5"
)
R84_PRODUCER = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_r84.py"
)
R84_PRODUCER_SHA256 = (
    "80cd0887fd24bfc37f0568f03a8af4c98eb148e030a3112576c1a0940f4ad99d"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R84_GATE = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_gate_r84.md"
)
R84_GATE_SHA256 = (
    "4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661"
)
R82_PRODUCER = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_r82.py"
)
R82_PRODUCER_SHA256 = (
    "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R70_PRODUCER = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_r70.py"
)
R70_PRODUCER_SHA256 = (
    "4d06a22f56ca58fbd4f4ed13a120324ce630ce1b1576243ee2b82fcdd340643e"
)
R70_REPORT = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
)
R70_REPORT_SHA256 = (
    "9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89"
)
R70_GATE = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_gate_r70.md"
)
R70_GATE_SHA256 = (
    "dd37fc251816a2c6ee798525aec4e7dd3c69eec60952102b55f90f60c353faa8"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R102_PRODUCER: R102_PRODUCER_SHA256,
        R102_REPORT: R102_REPORT_SHA256,
        R102_GATE: R102_GATE_SHA256,
        R102_PARENT: R102_PARENT_SHA256,
        R101_REPORT: R101_REPORT_SHA256,
        R101_GATE: R101_GATE_SHA256,
        R84_PRODUCER: R84_PRODUCER_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R82_PRODUCER: R82_PRODUCER_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R70_PRODUCER: R70_PRODUCER_SHA256,
        R70_REPORT: R70_REPORT_SHA256,
        R70_GATE: R70_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R103 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R102 = load_module("p1553_r102_for_r103", R102_PRODUCER)
R82 = R102.R82
R84 = R102.R84
R70 = R82.R70


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def point_sort_key(point: Point) -> tuple[int, int, int]:
    return (-1, 0, 0) if point is None else (0, point[0], point[1])


def s3_coefficients(
    x_left: int,
    x_target: int,
    curve: dict[str, Any],
) -> tuple[int, int, int]:
    prime = curve["field_prime"]
    a = curve["curve_a"]
    b = curve["curve_b"]
    leading = (x_left - x_target) ** 2 % prime
    linear = (
        -2
        * (
            (x_left + x_target) * (x_left * x_target + a)
            + 2 * b
        )
    ) % prime
    constant = (
        (x_left * x_target - a) ** 2
        - 4 * b * (x_left + x_target)
    ) % prime
    return leading, linear, constant


def eval_quadratic(
    coefficients: tuple[int, int, int],
    value: int,
    prime: int,
) -> int:
    leading, linear, constant = coefficients
    return (leading * value * value + linear * value + constant) % prime


def translation_roots(
    left: Point,
    target: Point,
    curve: dict[str, Any],
) -> tuple[Point, Point]:
    if left is None or target is None:
        raise ValueError("affine target and left point required")
    minus = R70.add(
        target,
        R70.negate(left, curve),
        curve,
    )
    plus = R70.add(target, left, curve)
    return minus, plus


def root_factorization_receipt(
    left: Point,
    target: Point,
    curve: dict[str, Any],
) -> dict[str, Any]:
    if left is None or target is None:
        return {
            "regular": False,
            "reason": "identity_has_no_affine_x_coordinate",
        }
    minus, plus = translation_roots(left, target, curve)
    coefficients = s3_coefficients(left[0], target[0], curve)
    leading, linear, constant = coefficients
    if minus is None or plus is None or leading == 0:
        return {
            "regular": False,
            "reason": "translation_identity_or_quadratic_degree_drop",
            "left": point_json(left),
            "target": point_json(target),
            "minus": point_json(minus),
            "plus": point_json(plus),
            "leading": leading,
        }
    prime = curve["field_prime"]
    root_minus = minus[0]
    root_plus = plus[0]
    return {
        "regular": True,
        "left": point_json(left),
        "target": point_json(target),
        "root_x_t_minus_l": root_minus,
        "root_x_t_plus_l": root_plus,
        "both_roots_vanish": (
            eval_quadratic(coefficients, root_minus, prime) == 0
            and eval_quadratic(coefficients, root_plus, prime) == 0
        ),
        "factor_linear_coefficient_exact": (
            linear == (-leading * (root_minus + root_plus)) % prime
        ),
        "factor_constant_coefficient_exact": (
            constant == (leading * root_minus * root_plus) % prime
        ),
        "roots_distinct": root_minus != root_plus,
    }


def signed_branch_labels(
    left: Point,
    right: Point,
    target: Point,
    curve: dict[str, Any],
) -> list[str]:
    if left is None or right is None or target is None:
        return []
    labels = []
    minus, plus = translation_roots(left, target, curve)
    negative_right = R70.negate(right, curve)
    if right == minus:
        labels.append("R=T-L")
    if negative_right == minus:
        labels.append("-R=T-L")
    if right == plus:
        labels.append("R=T+L")
    if negative_right == plus:
        labels.append("-R=T+L")
    return labels


def source_pair_replays(
    target: Point,
    left_source: Source,
    right_source: Source,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    left = R84.source_endpoint(
        left_source,
        atoms_a,
        atoms_c,
        curve,
    )
    right = R84.source_endpoint(
        right_source,
        atoms_a,
        atoms_c,
        curve,
    )
    return R70.add(left, right, curve) == target


def analyze_target(
    target: Point,
    left_histogram: collections.Counter[Point],
    left_first: dict[Point, Source],
    right_histogram: collections.Counter[Point],
    right_first: dict[Point, Source],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    if target is None:
        raise ValueError("R103 regular target must be affine")
    true_pair_count = 0
    true_occurrence_count = 0
    s3_pair_count = 0
    s3_occurrence_count = 0
    false_pair_count = 0
    false_occurrence_count = 0
    all_true_survive = True
    all_s3_equal_translation_branches = True
    branch_counts: collections.Counter[str] = collections.Counter()
    first_true: tuple[Source, Source] | None = None
    first_s3: tuple[Point, Point, list[str]] | None = None
    affine_pair_count = 0
    for left, left_weight in left_histogram.items():
        if left is None:
            continue
        minus, plus = translation_roots(left, target, curve)
        expected_x = {
            point[0] for point in (minus, plus) if point is not None
        }
        for right, right_weight in right_histogram.items():
            if right is None:
                continue
            affine_pair_count += 1
            true = R70.add(left, right, curve) == target
            s3 = (
                R70.semaev_s3(left[0], right[0], target[0], curve)
                == 0
            )
            branch = right[0] in expected_x
            labels = signed_branch_labels(left, right, target, curve)
            all_true_survive &= not true or s3
            all_s3_equal_translation_branches &= s3 == branch
            weight = left_weight * right_weight
            if true:
                true_pair_count += 1
                true_occurrence_count += weight
                if first_true is None:
                    first_true = (left_first[left], right_first[right])
            if s3:
                s3_pair_count += 1
                s3_occurrence_count += weight
                if first_s3 is None:
                    first_s3 = (left, right, labels)
                for label in labels:
                    branch_counts[label] += 1
                if not true:
                    false_pair_count += 1
                    false_occurrence_count += weight
    root_receipts = [
        root_factorization_receipt(left, target, curve)
        for left in sorted(left_histogram, key=point_sort_key)
    ]
    regular_roots = [row for row in root_receipts if row["regular"]]
    first_true_replays = (
        first_true is None
        if true_occurrence_count == 0
        else source_pair_replays(
            target,
            first_true[0],
            first_true[1],
            atoms_a,
            atoms_c,
            curve,
        )
    )
    return {
        "target": point_json(target),
        "affine_pair_count": affine_pair_count,
        "true_pair_count": true_pair_count,
        "true_integer_occurrence_count": true_occurrence_count,
        "s3_pair_count": s3_pair_count,
        "s3_integer_occurrence_count": s3_occurrence_count,
        "false_sign_pair_count": false_pair_count,
        "false_sign_integer_occurrence_count": false_occurrence_count,
        "all_true_pairs_survive_s3": all_true_survive,
        "all_s3_pairs_equal_translation_x_branches": (
            all_s3_equal_translation_branches
        ),
        "signed_branch_pair_counts": dict(sorted(branch_counts.items())),
        "first_true_source_replays": first_true_replays,
        "first_s3_candidate": (
            None
            if first_s3 is None
            else {
                "left": point_json(first_s3[0]),
                "right": point_json(first_s3[1]),
                "signed_branches": first_s3[2],
            }
        ),
        "regular_root_receipt_count": len(regular_roots),
        "exceptional_root_receipt_count": (
            len(root_receipts) - len(regular_roots)
        ),
        "all_regular_translation_roots_vanish": all(
            row["both_roots_vanish"] for row in regular_roots
        ),
        "all_regular_quadratic_factorizations_exact": all(
            row["factor_linear_coefficient_exact"]
            and row["factor_constant_coefficient_exact"]
            for row in regular_roots
        ),
    }


def target_support(
    left_points: Iterable[Point],
    right_points: Iterable[Point],
    curve: dict[str, Any],
) -> set[Point]:
    return {
        R70.add(left, right, curve)
        for left in left_points
        for right in right_points
    }


def blind_target(
    support: Iterable[Point],
    curve: dict[str, Any],
    salt: str,
) -> Point:
    support_set = set(support)
    return next(
        point
        for point in R82.hash_point_candidates(curve, salt)
        if point not in support_set
    )


def analyze_instance(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    left_histogram, left_first, left_count = (
        R84.multiset_endpoint_section(
            atoms_a,
            atoms_c,
            2,
            3,
            curve,
        )
    )
    right_histogram, right_first, right_count = (
        R84.multiset_endpoint_section(
            atoms_a,
            atoms_c,
            3,
            2,
            curve,
        )
    )
    support = target_support(left_histogram, right_histogram, curve)
    positive_target = next(
        target
        for target in sorted(support, key=point_sort_key)
        if target is not None
    )
    negative_target = blind_target(
        support,
        curve,
        f"R103|{offset}|blind",
    )
    positive = analyze_target(
        positive_target,
        left_histogram,
        left_first,
        right_histogram,
        right_first,
        atoms_a,
        atoms_c,
        curve,
    )
    negative = analyze_target(
        negative_target,
        left_histogram,
        left_first,
        right_histogram,
        right_first,
        atoms_a,
        atoms_c,
        curve,
    )
    return {
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "scalar_labels_consumed": False,
        "left_source_occurrence_count": left_count,
        "left_distinct_endpoint_count": len(left_histogram),
        "right_source_occurrence_count": right_count,
        "right_distinct_endpoint_count": len(right_histogram),
        "full_target_support_count": len(support),
        "positive_target": positive,
        "blind_target": negative,
    }


def synthetic_sign_complete_control() -> dict[str, Any]:
    curve = dict(R82.FAMILIES[-1])
    generator = R82.R81.curve_generator(curve)
    left = R70.scalar_mul(2, generator, curve)
    target = R70.scalar_mul(5, generator, curve)
    if left is None or target is None:
        raise AssertionError("synthetic S3 control lost affine points")
    minus, plus = translation_roots(left, target, curve)
    if minus is None or plus is None:
        raise AssertionError("synthetic S3 branch became identity")
    candidates = (
        minus,
        R70.negate(minus, curve),
        plus,
        R70.negate(plus, curve),
    )
    rows = []
    for right in candidates:
        if right is None:
            raise AssertionError("synthetic sign branch became identity")
        rows.append(
            {
                "right": point_json(right),
                "s3_zero": (
                    R70.semaev_s3(
                        left[0],
                        right[0],
                        target[0],
                        curve,
                    )
                    == 0
                ),
                "true_join": R70.add(left, right, curve) == target,
                "signed_branches": signed_branch_labels(
                    left,
                    right,
                    target,
                    curve,
                ),
            }
        )
    root_receipt = root_factorization_receipt(left, target, curve)
    return {
        "family_id": curve["family_id"],
        "left": point_json(left),
        "target": point_json(target),
        "candidate_rows": rows,
        "all_four_signed_points_pass_s3": all(
            row["s3_zero"] for row in rows
        ),
        "exactly_one_signed_point_is_true_join": (
            sum(row["true_join"] for row in rows) == 1
        ),
        "two_distinct_x_roots": (
            len({row["right"][0] for row in rows}) == 2
        ),
        "root_factorization_exact": (
            root_receipt["regular"]
            and root_receipt["both_roots_vanish"]
            and root_receipt["factor_linear_coefficient_exact"]
            and root_receipt["factor_constant_coefficient_exact"]
        ),
        "scalar_labels_consumed": False,
    }


def asymptotic_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "target_forced_s3": {
            "degree_in_x_right": 2,
            "field_operations_per_point_pair": "O(1)",
            "every_true_join_survives": True,
            "regular_roots": "x(T-L) and x(T+L)",
            "sign_resolved": False,
            "source_resolved": False,
        },
        "pointwise_local_oracle_composition": {
            "enumerate_left_then_query_right_exponent_B": fraction_record(
                POINTWISE_LEFT_FIRST_EXPONENT
            ),
            "enumerate_right_then_query_left_exponent_B": fraction_record(
                POINTWISE_RIGHT_FIRST_EXPONENT
            ),
            "best_exponent_B": fraction_record(
                min(
                    POINTWISE_LEFT_FIRST_EXPONENT,
                    POINTWISE_RIGHT_FIRST_EXPONENT,
                )
            ),
            "inside_online_cap": (
                min(
                    POINTWISE_LEFT_FIRST_EXPONENT,
                    POINTWISE_RIGHT_FIRST_EXPONENT,
                )
                <= ONLINE_CAP
            ),
            "constant_sign_branches_change_exponent": False,
        },
        "canonical_prefix_suffix": {
            "prefix_state_exponent_B": fraction_record(
                CANONICAL_PREFIX_STATE_EXPONENT
            ),
            "suffix_query_exponent_B": fraction_record(
                CANONICAL_SUFFIX_QUERY_EXPONENT
            ),
            "s3_changes_prefix_state_exponent": False,
            "s3_changes_suffix_query_exponent": False,
            "inside_setup_cap": (
                CANONICAL_PREFIX_STATE_EXPONENT <= SETUP_CAP
            ),
            "inside_online_cap": (
                CANONICAL_SUFFIX_QUERY_EXPONENT <= ONLINE_CAP
            ),
        },
        "standard_aggregate_representations": {
            "right_endpoint_x_polynomial_degree_exponent_B": (
                fraction_record(RIGHT_SIDE_EXPONENT)
            ),
            "left_endpoint_x_polynomial_degree_exponent_B": (
                fraction_record(LEFT_SIDE_EXPONENT)
            ),
            "fresh_canonical_suffix_polynomial_degree_exponent_B": (
                fraction_record(CANONICAL_SUFFIX_QUERY_EXPONENT)
            ),
            "right_body_inside_setup_cap": (
                RIGHT_SIDE_EXPONENT <= SETUP_CAP
            ),
            "left_body_inside_setup_cap": (
                LEFT_SIDE_EXPONENT <= SETUP_CAP
            ),
            "fresh_suffix_body_inside_online_cap": (
                CANONICAL_SUFFIX_QUERY_EXPONENT <= ONLINE_CAP
            ),
            "s3_resultant_preserves_endpoint_degree_exponent": True,
        },
        "ffe_factorization": {
            "endpoint_x_roots_lie_in_base_field": True,
            "materialized_polynomial_splits_into_linear_factors": True,
            "total_linear_factor_count_exponent_B": fraction_record(
                RIGHT_SIDE_EXPONENT
            ),
            "factorization_reduces_total_degree_or_source_payload": False,
            "nonstandard_compact_preendpoint_pushdown_open": True,
        },
        "scope": (
            "Closes pointwise S3, explicit endpoint polynomials, standard "
            "resultants, and materialized FFE factor lists only. It does "
            "not close a nonstandard target-specialized S3/FFE pushdown "
            "acting before partial endpoint emission."
        ),
    }


@functools.lru_cache(maxsize=1)
def actual_controls() -> dict[str, Any]:
    instances = [
        analyze_instance(dict(family), offset)
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    target_rows = [
        instance[target_name]
        for instance in instances
        for target_name in ("positive_target", "blind_target")
    ]
    synthetic = synthetic_sign_complete_control()
    return {
        "instances": instances,
        "instance_count": len(instances),
        "target_control_count": len(target_rows),
        "all_true_pairs_survive_s3": all(
            row["all_true_pairs_survive_s3"] for row in target_rows
        ),
        "all_s3_pairs_equal_translation_x_branches": all(
            row["all_s3_pairs_equal_translation_x_branches"]
            for row in target_rows
        ),
        "all_regular_translation_roots_vanish": all(
            row["all_regular_translation_roots_vanish"]
            for row in target_rows
        ),
        "all_regular_quadratic_factorizations_exact": all(
            row["all_regular_quadratic_factorizations_exact"]
            for row in target_rows
        ),
        "all_positive_sources_replay": all(
            instance["positive_target"]["first_true_source_replays"]
            for instance in instances
        ),
        "all_blind_targets_have_no_true_join": all(
            instance["blind_target"]["true_integer_occurrence_count"] == 0
            for instance in instances
        ),
        "all_scalar_blind": all(
            not instance["scalar_labels_consumed"]
            for instance in instances
        ),
        "synthetic_sign_complete_control": synthetic,
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_controls()
    costs = asymptotic_control()
    synthetic = controls["synthetic_sign_complete_control"]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_target_forced_algebraic_join_filter.r103.v1"
        ),
        "invariant": "affine x-coordinate",
        "necessary_relation": "S3(x(L),x(R),x(T))=0",
        "regular_root_factorization": (
            "(x(R)-x(T-L))(x(R)-x(T+L)) up to "
            "(x(L)-x(T))^2"
        ),
        "sign_resolution": "point-level replay required",
        "caps": costs["caps"],
        "excluded_open_operation": (
            "compact target-specialized S3/FFE pushdown before partial "
            "endpoint or source-leaf emission"
        ),
    }
    invariant_ledger = {
        "schema": (
            "p1553.target_forced_invariant_and_ffe_ledger.r103.v1"
        ),
        "asymptotic_control": costs,
        "actual_root_and_branch_controls": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "positive_target": instance["positive_target"],
                "blind_target": instance["blind_target"],
            }
            for instance in controls["instances"]
        ],
    }
    source_replay = {
        "schema": (
            "p1553.target_forced_join_integer_source_replay.r103.v1"
        ),
        "all_true_pairs_survive_s3": controls[
            "all_true_pairs_survive_s3"
        ],
        "all_s3_pairs_equal_translation_x_branches": controls[
            "all_s3_pairs_equal_translation_x_branches"
        ],
        "all_positive_sources_replay": controls[
            "all_positive_sources_replay"
        ],
        "synthetic_sign_complete_control": synthetic,
        "exact_fresh_join_inside_caps": False,
    }
    exceptional = {
        "schema": (
            "p1553.target_forced_join_exceptional_controls.r103.v1"
        ),
        "blind_no_true_join_complete": controls[
            "all_blind_targets_have_no_true_join"
        ],
        "regular_affine_root_factorization_complete": (
            controls["all_regular_translation_roots_vanish"]
            and controls["all_regular_quadratic_factorizations_exact"]
        ),
        "synthetic_four_sign_branches_complete": (
            synthetic["all_four_signed_points_pass_s3"]
            and synthetic["exactly_one_signed_point_is_true_join"]
            and synthetic["two_distinct_x_roots"]
        ),
        "identity_target_chart_complete": False,
        "translation_identity_degree_drop_complete": False,
        "tangent_and_repeated_root_complete": False,
        "proper_subsum_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r103.v1",
        "target_forced_s3_relation_exact": True,
        "compact_preendpoint_pushdown_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "fourteen_source_bindings_verified": len(bindings) == 14,
        "eight_actual_instances_replayed": controls["instance_count"] == 8,
        "sixteen_actual_target_controls_replayed": (
            controls["target_control_count"] == 16
        ),
        "all_true_pairs_survive_s3": controls[
            "all_true_pairs_survive_s3"
        ],
        "all_s3_pairs_equal_two_translation_x_branches": controls[
            "all_s3_pairs_equal_translation_x_branches"
        ],
        "all_regular_translation_roots_vanish": controls[
            "all_regular_translation_roots_vanish"
        ],
        "all_regular_quadratic_factorizations_exact": controls[
            "all_regular_quadratic_factorizations_exact"
        ],
        "all_positive_joint_sources_replay": controls[
            "all_positive_sources_replay"
        ],
        "all_blind_targets_have_no_true_join": controls[
            "all_blind_targets_have_no_true_join"
        ],
        "scalar_blind_construction": controls["all_scalar_blind"],
        "synthetic_all_four_sign_points_pass_s3": synthetic[
            "all_four_signed_points_pass_s3"
        ],
        "synthetic_exactly_one_sign_point_is_true": synthetic[
            "exactly_one_signed_point_is_true_join"
        ],
        "synthetic_two_x_roots_exact": synthetic[
            "two_distinct_x_roots"
        ],
        "s3_degree_two_and_constant_point_cost": (
            costs["target_forced_s3"]["degree_in_x_right"] == 2
        ),
        "pointwise_best_exponent_B16O5": (
            costs["pointwise_local_oracle_composition"][
                "best_exponent_B"
            ]["exact"]
            == "16/5"
        ),
        "pointwise_composition_outside_online_cap": not costs[
            "pointwise_local_oracle_composition"
        ]["inside_online_cap"],
        "canonical_s3_query_remains_B14O5": (
            costs["canonical_prefix_suffix"][
                "suffix_query_exponent_B"
            ]["exact"]
            == "14/5"
        ),
        "explicit_left_and_right_x_bodies_over_setup_cap": (
            not costs["standard_aggregate_representations"][
                "right_body_inside_setup_cap"
            ]
            and not costs["standard_aggregate_representations"][
                "left_body_inside_setup_cap"
            ]
        ),
        "fresh_suffix_x_body_over_online_cap": not costs[
            "standard_aggregate_representations"
        ]["fresh_suffix_body_inside_online_cap"],
        "materialized_ffe_factorization_preserves_total_degree": not costs[
            "ffe_factorization"
        ]["factorization_reduces_total_degree_or_source_payload"],
        "target_forced_relation_supplied": True,
        "compact_preendpoint_s3_ffe_pushdown_inside_caps": False,
        "exact_integer_count_and_joint_source_inside_caps": False,
        "identity_and_degree_drop_charts_complete": False,
        "tangent_repeated_root_and_proper_subsum_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "full_pipeline_fresh_workspace_inside_cap": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "TARGET_FORCED_S3_FILTER_EXACT__REGULAR_ROOTS_X_T_MINUS_L_AND_"
            "X_T_PLUS_L__SIGN_BRANCH_CONSTANT_ONLY__POINTWISE_B16O5_OR_"
            "CANONICAL_B14O5__STANDARD_AGGREGATE_AND_FFE_BODIES_OVER_CAP__"
            "NONSTANDARD_COMPACT_PREENDPOINT_PUSHDOWN_OPEN"
        ),
        "source_bindings": {
            "r102_producer": {
                "path": str(R102_PRODUCER),
                "sha256": R102_PRODUCER_SHA256,
            },
            "r102_report": {
                "path": str(R102_REPORT),
                "sha256": R102_REPORT_SHA256,
            },
            "r102_gate": {
                "path": str(R102_GATE),
                "sha256": R102_GATE_SHA256,
            },
            "r102_parent": {
                "path": str(R102_PARENT),
                "sha256": R102_PARENT_SHA256,
            },
            "r101_report": {
                "path": str(R101_REPORT),
                "sha256": R101_REPORT_SHA256,
            },
            "r101_gate": {
                "path": str(R101_GATE),
                "sha256": R101_GATE_SHA256,
            },
            "r84_producer": {
                "path": str(R84_PRODUCER),
                "sha256": R84_PRODUCER_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "r82_producer": {
                "path": str(R82_PRODUCER),
                "sha256": R82_PRODUCER_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r70_producer": {
                "path": str(R70_PRODUCER),
                "sha256": R70_PRODUCER_SHA256,
            },
            "r70_report": {
                "path": str(R70_REPORT),
                "sha256": R70_REPORT_SHA256,
            },
            "r70_gate": {
                "path": str(R70_GATE),
                "sha256": R70_GATE_SHA256,
            },
        },
        "novelty_scope": (
            "R103 gives an exact campaign-level root factorization and sign "
            "accounting for the target-forced S3 join filter, then charges "
            "pointwise, aggregate-polynomial, and materialized FFE routes."
        ),
        "actual_controls": controls,
        "asymptotic_control": costs,
        "artifacts": {
            "frozen": (
                "frozen_5a5c_target_forced_algebraic_join_filter.json"
            ),
            "invariant_ledger": (
                "target_forced_invariant_and_ffe_ledger.json"
            ),
            "source_replay": (
                "target_forced_join_integer_source_replay.json"
            ),
            "exceptional": (
                "target_forced_join_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r103.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "target_forced_s3_identity_admitted": True,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": costs["scope"],
        "next_action": (
            "Construct or refute one sign-resolved target-specialized S3/FFE "
            "pushdown acting directly on D_A,D_C before 4A+1C or 1A+4C "
            "endpoint emission. Freeze the recurrence and projective charts "
            "before outcomes; require B^(9/4) setup/state, B^(5/4) fresh "
            "work/workspace, exact integer count and one coupled source, no "
            "root/verifier oracle, complete false-positive elimination, "
            "known-RHS rank, factor logs, and identical descent."
        ),
        "disposition": (
            "ADMIT_TARGET_FORCED_S3_IDENTITY_ONLY__EVERY_TRUE_JOIN_SURVIVES__"
            "TWO_REGULAR_X_ROOTS_ARE_T_MINUS_L_AND_T_PLUS_L__X_FILTER_LOSES_"
            "SIGN__POINTWISE_LOCAL_ORACLES_B16O5_BEST__CANONICAL_QUERY_"
            "B14O5_UNCHANGED__EXPLICIT_X_POLYNOMIAL_AND_FFE_FACTOR_BODIES_"
            "OVER_CAP__COMPACT_PREENDPOINT_PUSHDOWN_OPEN__NO_RANK__NO_"
            "FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "invariant_ledger": invariant_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_target_forced_algebraic_join_"
            "filter_probe_report_r103.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_target_forced_algebraic_join_filter.json"
        ),
    )
    parser.add_argument(
        "--invariant-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_forced_invariant_and_ffe_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_forced_join_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_forced_join_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r103.json"
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
    write_json(args.invariant_ledger_output, bundle["invariant_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
