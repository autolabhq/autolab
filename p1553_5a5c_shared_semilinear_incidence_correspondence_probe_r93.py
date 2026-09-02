#!/usr/bin/env python3
"""Audit a shared semilinear S3/S4 incidence operator for the 5A+5C lane."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = (
    "p1553.5a5c_shared_semilinear_incidence_correspondence.r93.v1"
)
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
FIELD_PRIME = 101
CURVE_A = 7
CURVE_B = 4
GROUP_ORDER = 97

R92_REPORT = pathlib.Path(
    "p1553_5a5c_compact_elliptic_subfunction_map_"
    "probe_report_r92.json"
)
R92_REPORT_SHA256 = (
    "6604e1ac8e15c9048d06262a4116596ff793d4805901df8be16d67c2f170c24c"
)
R92_GATE = pathlib.Path(
    "p1553_5a5c_compact_elliptic_subfunction_map_"
    "probe_gate_r92.md"
)
R92_GATE_SHA256 = (
    "f4e5cfe45f25721bd11e60fab31a49866f38fe690b65b832d0b2ba7e20e78ad6"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R84_GATE = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_gate_r84.md"
)
R84_GATE_SHA256 = (
    "4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661"
)
R78_REPORT = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
R78_REPORT_SHA256 = (
    "e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60"
)
R78_GATE = pathlib.Path(
    "p1553_actual_s6_fermat_tensor_train_probe_gate_r78.md"
)
R78_GATE_SHA256 = (
    "9149c3b903d91a2156125a2c61896dc1111eaeffafdf0536c89a50b7e0e84ff9"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R72_REPORT = pathlib.Path(
    "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
)
R72_REPORT_SHA256 = (
    "7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43"
)

Point = tuple[int, int] | None
Quadratic = tuple[int, int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R92_REPORT: R92_REPORT_SHA256,
        R92_GATE: R92_GATE_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R78_REPORT: R78_REPORT_SHA256,
        R78_GATE: R78_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R72_REPORT: R72_REPORT_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R93 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def curve_points() -> list[Point]:
    points: list[Point] = [None]
    for x_coord in range(FIELD_PRIME):
        rhs = (
            x_coord**3 + CURVE_A * x_coord + CURVE_B
        ) % FIELD_PRIME
        for y_coord in range(FIELD_PRIME):
            if y_coord * y_coord % FIELD_PRIME == rhs:
                points.append((x_coord, y_coord))
    return points


def canonical_affine_points() -> list[Point]:
    by_x: dict[int, list[int]] = {}
    for point in curve_points():
        if point is not None:
            by_x.setdefault(point[0], []).append(point[1])
    return [
        (x_coord, min(y_values))
        for x_coord, y_values in sorted(by_x.items())
    ]


def negate(point: Point) -> Point:
    if point is None:
        return None
    return (point[0], (-point[1]) % FIELD_PRIME)


def add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x_left, y_left = left
    x_right, y_right = right
    if x_left == x_right and (y_left + y_right) % FIELD_PRIME == 0:
        return None
    if left == right:
        if y_left == 0:
            return None
        slope = (
            (3 * x_left * x_left + CURVE_A)
            * pow(2 * y_left, -1, FIELD_PRIME)
        ) % FIELD_PRIME
    else:
        slope = (
            (y_right - y_left)
            * pow(x_right - x_left, -1, FIELD_PRIME)
        ) % FIELD_PRIME
    x_out = (slope * slope - x_left - x_right) % FIELD_PRIME
    y_out = (slope * (x_left - x_out) - y_left) % FIELD_PRIME
    return (x_out, y_out)


def add_many(points: Iterable[Point]) -> Point:
    total: Point = None
    for point in points:
        total = add(total, point)
    return total


def s3_quadratic(x_left: int, x_right: int) -> Quadratic:
    leading = (x_left - x_right) ** 2
    linear = -2 * (
        (x_left + x_right)
        * (x_left * x_right + CURVE_A)
        + 2 * CURVE_B
    )
    constant = (
        (x_left * x_right - CURVE_A) ** 2
        - 4 * CURVE_B * (x_left + x_right)
    )
    return tuple(
        coefficient % FIELD_PRIME
        for coefficient in (leading, linear, constant)
    )


def quadratic_resultant(left: Quadratic, right: Quadratic) -> int:
    a_coeff, b_coeff, c_coeff = left
    d_coeff, e_coeff, f_coeff = right
    return (
        a_coeff * a_coeff * f_coeff * f_coeff
        - a_coeff * b_coeff * e_coeff * f_coeff
        - 2 * a_coeff * c_coeff * d_coeff * f_coeff
        + a_coeff * c_coeff * e_coeff * e_coeff
        + b_coeff * b_coeff * d_coeff * f_coeff
        - b_coeff * c_coeff * d_coeff * e_coeff
        + c_coeff * c_coeff * d_coeff * d_coeff
    ) % FIELD_PRIME


def veronese(quadratic: Quadratic) -> list[int]:
    a_coeff, b_coeff, c_coeff = quadratic
    return [
        a_coeff * a_coeff % FIELD_PRIME,
        a_coeff * b_coeff % FIELD_PRIME,
        a_coeff * c_coeff % FIELD_PRIME,
        b_coeff * b_coeff % FIELD_PRIME,
        b_coeff * c_coeff % FIELD_PRIME,
        c_coeff * c_coeff % FIELD_PRIME,
    ]


RESULTANT_OPERATOR = [
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, -1, 0],
    [0, 0, -2, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
]


def bilinear_resultant(left: Quadratic, right: Quadratic) -> int:
    left_features = veronese(left)
    right_features = veronese(right)
    return sum(
        left_features[row]
        * RESULTANT_OPERATOR[row][column]
        * right_features[column]
        for row in range(6)
        for column in range(6)
    ) % FIELD_PRIME


def matrix_rank_mod(matrix: list[list[int]]) -> int:
    if not matrix:
        return 0
    work = [
        [value % FIELD_PRIME for value in row]
        for row in matrix
    ]
    row_index = 0
    for column in range(len(work[0])):
        pivot = next(
            (
                candidate
                for candidate in range(row_index, len(work))
                if work[candidate][column]
            ),
            None,
        )
        if pivot is None:
            continue
        work[row_index], work[pivot] = work[pivot], work[row_index]
        inverse = pow(work[row_index][column], -1, FIELD_PRIME)
        work[row_index] = [
            value * inverse % FIELD_PRIME
            for value in work[row_index]
        ]
        for other in range(len(work)):
            if other == row_index:
                continue
            multiplier = work[other][column]
            if multiplier:
                work[other] = [
                    (left - multiplier * right) % FIELD_PRIME
                    for left, right in zip(
                        work[other],
                        work[row_index],
                    )
                ]
        row_index += 1
        if row_index == len(work):
            break
    return row_index


def signed_relation(
    points: tuple[Point, Point, Point, Point],
) -> tuple[int, int, int, int] | None:
    for signs in itertools.product((-1, 1), repeat=4):
        signed_points = [
            point if sign == 1 else negate(point)
            for sign, point in zip(signs, points)
        ]
        if add_many(signed_points) is None:
            return signs
    return None


def has_proper_zero_subsum(
    points: tuple[Point, Point, Point, Point],
    signs: tuple[int, int, int, int],
) -> bool:
    signed_points = [
        point if sign == 1 else negate(point)
        for sign, point in zip(signs, points)
    ]
    for mask in range(1, (1 << 4) - 1):
        subset = [
            point
            for index, point in enumerate(signed_points)
            if mask & (1 << index)
        ]
        if add_many(subset) is None:
            return True
    return False


def chart_instance(deck_size: int) -> dict[str, Any]:
    points = canonical_affine_points()[:deck_size]
    pair_indices = list(
        itertools.combinations_with_replacement(range(deck_size), 2)
    )
    quadratics = [
        s3_quadratic(points[left][0], points[right][0])
        for left, right in pair_indices
    ]
    features = [veronese(quadratic) for quadratic in quadratics]
    raw_matrix = [
        [
            quadratic_resultant(left, right)
            for right in quadratics
        ]
        for left in quadratics
    ]
    zero_matrix = [
        [int(value == 0) for value in row]
        for row in raw_matrix
    ]

    zero_count = 0
    relation_count = 0
    proper_subsum_count = 0
    full_only_count = 0
    mismatch_count = 0
    for left_index, left_pair in enumerate(pair_indices):
        for right_index, right_pair in enumerate(pair_indices):
            source_points = (
                points[left_pair[0]],
                points[left_pair[1]],
                points[right_pair[0]],
                points[right_pair[1]],
            )
            relation = signed_relation(source_points)
            resultant_zero = zero_matrix[left_index][right_index] == 1
            relation_exists = relation is not None
            zero_count += int(resultant_zero)
            relation_count += int(relation_exists)
            mismatch_count += int(resultant_zero != relation_exists)
            if relation is not None:
                if has_proper_zero_subsum(source_points, relation):
                    proper_subsum_count += 1
                else:
                    full_only_count += 1

    return {
        "deck_size": deck_size,
        "pair_chart_count": len(pair_indices),
        "quadratic_feature_dimension": 6,
        "distinct_quadratic_count": len(set(quadratics)),
        "veronese_feature_rank": matrix_rank_mod(features),
        "shared_operator_rank": matrix_rank_mod(RESULTANT_OPERATOR),
        "raw_resultant_matrix_rank": matrix_rank_mod(raw_matrix),
        "zero_incidence_matrix_rank": matrix_rank_mod(zero_matrix),
        "zero_incidence_count": zero_count,
        "signed_relation_count": relation_count,
        "predicate_source_mismatch_count": mismatch_count,
        "predicate_source_biconditional_exact": mismatch_count == 0,
        "every_zero_has_signed_source": (
            zero_count == relation_count and mismatch_count == 0
        ),
        "proper_subsum_source_count": proper_subsum_count,
        "full_only_source_count": full_only_count,
        "affine_x_chart_only": True,
    }


def exact_chart_sweep() -> list[dict[str, Any]]:
    return [
        chart_instance(deck_size)
        for deck_size in (4, 6, 8, 10, 12)
    ]


def random_quadratic(seed: str, index: int) -> Quadratic:
    digest = hashlib.sha256(
        f"P1553-R93|{seed}|{index}".encode("ascii")
    ).digest()
    values = [
        int.from_bytes(digest[offset : offset + 4], "big")
        % FIELD_PRIME
        for offset in (0, 4, 8)
    ]
    if values == [0, 0, 0]:
        values[0] = 1
    return (values[0], values[1], values[2])


def random_quadratic_controls() -> list[dict[str, Any]]:
    rows = []
    for count in (10, 21, 36, 55, 78):
        quadratics = [
            random_quadratic(str(count), index)
            for index in range(count)
        ]
        raw = [
            [
                bilinear_resultant(left, right)
                for right in quadratics
            ]
            for left in quadratics
        ]
        zero = [
            [int(value == 0) for value in row]
            for row in raw
        ]
        rows.append(
            {
                "quadratic_count": count,
                "raw_resultant_matrix_rank": matrix_rank_mod(raw),
                "zero_incidence_matrix_rank": matrix_rank_mod(zero),
                "zero_incidence_count": sum(map(sum, zero)),
                "shared_operator_rank": matrix_rank_mod(
                    RESULTANT_OPERATOR
                ),
            }
        )
    return rows


def shared_operator_control() -> dict[str, Any]:
    sweep = exact_chart_sweep()
    random_controls = random_quadratic_controls()
    all_quadratics = [
        s3_quadratic(left[0], right[0])
        for left, right in itertools.combinations_with_replacement(
            canonical_affine_points()[:12],
            2,
        )
    ]
    identity_exact = all(
        quadratic_resultant(left, right)
        == bilinear_resultant(left, right)
        for left in all_quadratics
        for right in all_quadratics
    )
    largest = sweep[-1]
    return {
        "quadratic_resultant_identity": (
            "Res(a*z^2+b*z+c,d*z^2+e*z+f) "
            "= Ver2(a,b,c)^T*K*Ver2(d,e,f)"
        ),
        "veronese_dimension": 6,
        "frozen_operator": RESULTANT_OPERATOR,
        "operator_rank": matrix_rank_mod(RESULTANT_OPERATOR),
        "identity_exact_on_all_78_by_78_curve_charts": identity_exact,
        "curve_chart_sweep": sweep,
        "matched_random_quadratic_controls": random_controls,
        "largest_curve_chart": largest,
        "raw_kernel_rank_at_most_six_every_scale": all(
            row["raw_resultant_matrix_rank"] <= 6 for row in sweep
        ),
        "zero_mask_full_rank_at_largest_scale": (
            largest["zero_incidence_matrix_rank"]
            == largest["pair_chart_count"]
        ),
        "zero_projection_preserves_semilinear_rank": False,
    }


def root_source_feature_cost_control() -> dict[str, Any]:
    left = Fraction(12, 5)
    right = Fraction(13, 5)
    return {
        "r84_root_split": "(2A+3C) versus (3A+2C)",
        "left_source_feature_rows_exponent_B": fraction_record(left),
        "right_source_feature_rows_exponent_B": fraction_record(right),
        "smaller_feature_body_exponent_B": fraction_record(min(left, right)),
        "larger_feature_body_exponent_B": fraction_record(max(left, right)),
        "smaller_feature_body_inside_setup_cap": (
            min(left, right) <= SETUP_CAP
        ),
        "one_side_scan_inside_online_cap": (
            min(left, right) <= ONLINE_CAP
        ),
        "constant_operator_words_exponent_B": fraction_record(Fraction(0)),
        "per_source_feature_width_constant_in_B": True,
        "source_rows_removed_by_constant_operator": False,
        "finite_r84_side_endpoint_maps_injective": True,
        "fatal_obstruction": (
            "the fixed resultant operator compresses each comparison, not "
            "the B^(12/5) and B^(13/5) source-bearing feature rows"
        ),
        "scope_exception": (
            "an implicit Cartesian Veronese range index that evaluates "
            "zero hyperplane queries and returns a source before source "
            "feature rows are emitted"
        ),
    }


def projective_branch_controls() -> dict[str, Any]:
    sweep = exact_chart_sweep()
    return {
        "all_affine_predicate_source_biconditionals_exact": all(
            row["predicate_source_biconditional_exact"]
            for row in sweep
        ),
        "proper_subsum_branches_present": any(
            row["proper_subsum_source_count"] > 0 for row in sweep
        ),
        "full_only_branches_present": any(
            row["full_only_source_count"] > 0 for row in sweep
        ),
        "affine_infinity_chart_defined": False,
        "projective_infinity_source_replay_complete": False,
        "nonreduced_tangent_source_multiplicity_complete": False,
        "matched_random_controls": random_quadratic_controls(),
    }


def cost_ledger() -> dict[str, Any]:
    root = root_source_feature_cost_control()
    operator = shared_operator_control()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "shared_operator": {
            "operator_rank": operator["operator_rank"],
            "operator_state_exponent_B": fraction_record(Fraction(0)),
            "raw_resultant_rank_at_most_six": operator[
                "raw_kernel_rank_at_most_six_every_scale"
            ],
            "zero_mask_full_rank_finite_control": operator[
                "zero_mask_full_rank_at_largest_scale"
            ],
        },
        "root_source_features": root,
        "fatal_obstructions": [
            "the explicit smaller root feature body is B^(12/5), above setup",
            "a one-side feature scan is at least B^(12/5), above fresh work",
            "the nonlinear zero projector destroys the rank-six value bound on the largest finite control",
            "the affine x chart omits infinity and nonreduced source multiplicity",
        ],
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    operator = shared_operator_control()
    root_cost = root_source_feature_cost_control()
    branches = projective_branch_controls()
    costs = cost_ledger()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_shared_semilinear_"
            "incidence_correspondence.r93.v1"
        ),
        "curve": {
            "field_prime": FIELD_PRIME,
            "curve_a": CURVE_A,
            "curve_b": CURVE_B,
            "group_order": GROUP_ORDER,
        },
        "s3_quadratic": (
            "(x1-x2)^2*z^2 - "
            "2*((x1+x2)*(x1*x2+a)+2*b)*z + "
            "(x1*x2-a)^2-4*b*(x1+x2)"
        ),
        "veronese_feature_order": [
            "a^2",
            "a*b",
            "a*c",
            "b^2",
            "b*c",
            "c^2",
        ],
        "shared_resultant_operator": RESULTANT_OPERATOR,
        "operator_rank": operator["operator_rank"],
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
    }
    compression = {
        "schema": (
            "p1553.overlapping_chart_operator_compression_ledger.r93.v1"
        ),
        "frozen_candidate": frozen,
        "shared_operator_control": operator,
        "root_source_feature_cost": root_cost,
        "cost_ledger": costs,
    }
    source_replay = {
        "schema": "p1553.compact_source_unranking_replay.r93.v1",
        "affine_chart_sweep": operator["curve_chart_sweep"],
        "all_affine_zeroes_return_signed_source": all(
            row["every_zero_has_signed_source"]
            for row in operator["curve_chart_sweep"]
        ),
        "actual_five_a_five_c_source_unranking_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.projective_overlap_false_positive_controls.r93.v1"
        ),
        "projective_branch_controls": branches,
        "false_positive_count": sum(
            row["predicate_source_mismatch_count"]
            for row in operator["curve_chart_sweep"]
        ),
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r93.v1",
        "constant_shared_resultant_operator_found": True,
        "subcap_source_reporting_index_found": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "eight_source_bindings_verified": len(bindings) == 8,
        "quadratic_resultant_veronese_identity_exact": operator[
            "identity_exact_on_all_78_by_78_curve_charts"
        ],
        "shared_operator_rank_six_exact": operator["operator_rank"] == 6,
        "all_raw_resultant_ranks_at_most_six": operator[
            "raw_kernel_rank_at_most_six_every_scale"
        ],
        "affine_s4_predicate_source_biconditional_exact": branches[
            "all_affine_predicate_source_biconditionals_exact"
        ],
        "proper_subsum_branch_replayed": branches[
            "proper_subsum_branches_present"
        ],
        "full_only_branch_replayed": branches[
            "full_only_branches_present"
        ],
        "zero_mask_full_rank_finite_control": operator[
            "zero_mask_full_rank_at_largest_scale"
        ],
        "root_source_feature_exponents_charged": (
            root_cost["smaller_feature_body_exponent_B"]["exact"]
            == "12/5"
            and root_cost["larger_feature_body_exponent_B"]["exact"]
            == "13/5"
        ),
        "root_feature_body_inside_setup_cap": False,
        "root_feature_scan_inside_online_cap": False,
        "implicit_cartesian_veronese_range_index_supplied": False,
        "actual_five_a_five_c_source_unranking_complete": False,
        "projective_infinity_source_replay_complete": False,
        "nonreduced_signed_source_multiplicity_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "CONSTANT_S3_RESULTANT_OPERATOR_EXACT__"
            "ZERO_INCIDENCE_FULL_RANK_AT_78_CHARTS__"
            "ROOT_SOURCE_FEATURE_ROWS_B12O5__"
            "PROJECTIVE_SOURCE_UNRANKING_ABSENT"
        ),
        "source_bindings": {
            "r92_report": {
                "path": str(R92_REPORT),
                "sha256": R92_REPORT_SHA256,
            },
            "r92_gate": {
                "path": str(R92_GATE),
                "sha256": R92_GATE_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "r78_report": {
                "path": str(R78_REPORT),
                "sha256": R78_REPORT_SHA256,
            },
            "r78_gate": {
                "path": str(R78_GATE),
                "sha256": R78_GATE_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r72_report": {
                "path": str(R72_REPORT),
                "sha256": R72_REPORT_SHA256,
            },
        },
        "novelty_scope": (
            "R93 is the first campaign receipt to separate the exact "
            "rank-six Veronese resultant operator from the rank of its "
            "nonlinear zero-incidence matrix while retaining signed affine "
            "S4 source witnesses."
        ),
        "shared_semilinear_operator_control": operator,
        "root_source_feature_cost_control": root_cost,
        "projective_overlap_and_false_positive_controls": branches,
        "cost_ledger": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_shared_semilinear_"
                "incidence_correspondence.json"
            ),
            "compression": (
                "overlapping_chart_operator_compression_ledger.json"
            ),
            "source_replay": "compact_source_unranking_replay.json",
            "exceptional": (
                "projective_overlap_and_false_positive_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r93.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "This rejects only materializing one fixed-width feature row "
            "per R84 root source and treating a low-rank resultant value "
            "kernel as a low-rank zero projector. It is not a lower bound "
            "on an implicit Cartesian range index, a nonlinear source-"
            "returning circuit, or another FFE representation."
        ),
        "next_action": (
            "Construct or refute one implicit Cartesian Veronese "
            "hyperplane source index for the R84 root split. It must answer "
            "the exact bilinear zero query and return coupled 5A+5C source "
            "indices without emitting B^(12/5) feature rows, fit B^(9/4) "
            "setup and B^(5/4) fresh work/workspace, and replay infinity, "
            "proper-subsum, tangent, multiplicity, and blind-zero branches "
            "without DLP labels or verifier oracles."
        ),
        "disposition": (
            "REJECT_MATERIALIZED_SHARED_RESULTANT_FEATURE_ROWS_ONLY__"
            "RANK_SIX_VERONESE_OPERATOR_EXACT__AFFINE_S4_SIGNED_SOURCE_"
            "BICONDITIONAL_EXACT__ZERO_INCIDENCE_FULL_RANK_AT_78_CHARTS__"
            "ROOT_FEATURE_BODIES_B12O5_AND_B13O5__INFINITY_AND_FULL_5A5C_"
            "UNRANKING_ABSENT__IMPLICIT_CARTESIAN_RANGE_INDEX_OPEN__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "compression": compression,
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
            "p1553_5a5c_shared_semilinear_incidence_"
            "correspondence_probe_report_r93.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_shared_semilinear_"
            "incidence_correspondence.json"
        ),
    )
    parser.add_argument(
        "--compression-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "overlapping_chart_operator_compression_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path("compact_source_unranking_replay.json"),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_overlap_and_false_positive_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r93.json"
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
    write_json(args.compression_output, bundle["compression"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
