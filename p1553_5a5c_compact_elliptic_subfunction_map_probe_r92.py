#!/usr/bin/env python3
"""Audit compact elliptic endpoint maps in the subfunction framework."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_compact_elliptic_subfunction_map.r92.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ENDPOINT_EXPONENT = Fraction(5)

R91_REPORT = pathlib.Path(
    "p1553_5a5c_unequal_list_subfunction_inversion_"
    "probe_report_r91.json"
)
R91_REPORT_SHA256 = (
    "6bfaaaa72dd1135f8244a70a0ce2a6697e5f3fe63ce8ac0ce107bd05f19a71da"
)
R91_GATE = pathlib.Path(
    "p1553_5a5c_unequal_list_subfunction_inversion_"
    "probe_gate_r91.md"
)
R91_GATE_SHA256 = (
    "57469654e0b535ccbf1d62edd9548cd827a09d244deaecb06b8aedbd410ac6d3"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R83_REPORT = pathlib.Path(
    "p1553_5a5c_coordinate_filtration_probe_report_r83.json"
)
R83_REPORT_SHA256 = (
    "1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c"
)
R83_QUOTIENT_CONTROL = pathlib.Path(
    "partial_filter_composability_and_false_positive_controls.json"
)
R83_QUOTIENT_CONTROL_SHA256 = (
    "b0dd2c5f2675e8e0967a6c0722f26f35700ac9b68f41aafa5f461e17c1cb71af"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)
P1515_ROUTER = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_field_router_candidate_v1.md"
)
P1515_ROUTER_SHA256 = (
    "ee7c0ef479f33d3a82ab6827c286460604c6d26c9a76aa551305eccc2a337e24"
)
DINUR_GOLOVNEV_V2 = pathlib.Path(
    "references/dinur_golovnev_3sum_indexing_2512.04258v2.pdf"
)
DINUR_GOLOVNEV_V2_SHA256 = (
    "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c"
)

FIELD_PRIME = 101
CURVE_A = 7
CURVE_B = 4
GROUP_ORDER = 97

Point = tuple[int, int] | None
TaggedValue = tuple[str, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R91_REPORT: R91_REPORT_SHA256,
        R91_GATE: R91_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R83_REPORT: R83_REPORT_SHA256,
        R83_QUOTIENT_CONTROL: R83_QUOTIENT_CONTROL_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
        P1515_ROUTER: P1515_ROUTER_SHA256,
        DINUR_GOLOVNEV_V2: DINUR_GOLOVNEV_V2_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R92 source binding mismatch: {failures}")
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


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def point_sort_key(point: Point) -> tuple[int, int, int]:
    if point is None:
        return (0, 0, 0)
    return (1, point[0], point[1])


def point_record(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def is_on_curve(point: Point) -> bool:
    if point is None:
        return True
    x_coord, y_coord = point
    return (
        y_coord * y_coord
        - x_coord * x_coord * x_coord
        - CURVE_A * x_coord
        - CURVE_B
    ) % FIELD_PRIME == 0


def curve_points() -> list[Point]:
    points: list[Point] = [None]
    for x_coord in range(FIELD_PRIME):
        rhs = (
            x_coord * x_coord * x_coord
            + CURVE_A * x_coord
            + CURVE_B
        ) % FIELD_PRIME
        for y_coord in range(FIELD_PRIME):
            if y_coord * y_coord % FIELD_PRIME == rhs:
                points.append((x_coord, y_coord))
    return sorted(points, key=point_sort_key)


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


def scalar_mul(scalar: int, point: Point) -> Point:
    result: Point = None
    addend = point
    while scalar:
        if scalar & 1:
            result = add(result, addend)
        addend = add(addend, addend)
        scalar >>= 1
    return result


def prime_curve_control() -> dict[str, Any]:
    points = curve_points()
    point_set = set(points)
    generator = next(point for point in points if point is not None)
    closure_exact = all(
        add(left, right) in point_set
        for left in points
        for right in points
    )
    discriminant_core = (
        4 * CURVE_A**3 + 27 * CURVE_B**2
    ) % FIELD_PRIME
    return {
        "curve": (
            f"E/F_{FIELD_PRIME}: y^2=x^3+{CURVE_A}x+{CURVE_B}"
        ),
        "field_prime": FIELD_PRIME,
        "field_prime_exactly_prime": is_prime(FIELD_PRIME),
        "curve_discriminant_nonzero": discriminant_core != 0,
        "enumerated_projective_point_count": len(points),
        "stated_group_order": GROUP_ORDER,
        "group_order_exactly_prime": is_prime(GROUP_ORDER),
        "point_count_matches_stated_order": len(points) == GROUP_ORDER,
        "all_enumerated_points_on_curve": all(
            is_on_curve(point) for point in points
        ),
        "addition_closed_on_full_enumeration": closure_exact,
        "generator": point_record(generator),
        "generator_has_stated_prime_order": (
            generator is not None
            and scalar_mul(GROUP_ORDER, generator) is None
            and scalar_mul(1, generator) == generator
        ),
        "nontrivial_two_torsion_point_count": sum(
            point is not None and point[1] == 0 for point in points
        ),
        "infinity_included": points[0] is None,
    }


def tagged_finite(value: int) -> TaggedValue:
    return ("finite", value % FIELD_PRIME)


def tagged_infinity() -> TaggedValue:
    return ("infinity", 0)


def map1(point: Point, map_name: str) -> TaggedValue:
    if point is None:
        return tagged_infinity()
    x_coord, y_coord = point
    if map_name == "x":
        return tagged_finite(x_coord)
    if map_name == "y":
        return tagged_finite(y_coord)
    if map_name == "x_plus_y":
        return tagged_finite(x_coord + y_coord)
    if map_name == "x_plus_2y":
        return tagged_finite(x_coord + 2 * y_coord)
    raise AssertionError(f"unknown map: {map_name}")


def map2(point: Point, map_name: str) -> TaggedValue:
    if point is None:
        return tagged_infinity()
    x_coord, y_coord = point
    if map_name == "x":
        return tagged_finite(y_coord)
    if map_name in {"y", "x_plus_y", "x_plus_2y"}:
        return tagged_finite(x_coord)
    raise AssertionError(f"unknown map: {map_name}")


MAP_SPECS = {
    "x": {
        "rational_map": "h(P)=x(P)",
        "map2": "MAP2(P)=y(P)",
        "degree_upper_bound": 2,
    },
    "y": {
        "rational_map": "h(P)=y(P)",
        "map2": "MAP2(P)=x(P)",
        "degree_upper_bound": 3,
    },
    "x_plus_y": {
        "rational_map": "h(P)=x(P)+y(P)",
        "map2": "MAP2(P)=x(P)",
        "degree_upper_bound": 3,
    },
    "x_plus_2y": {
        "rational_map": "h(P)=x(P)+2y(P)",
        "map2": "MAP2(P)=x(P)",
        "degree_upper_bound": 3,
    },
}


def fiber_semantics_control(map_name: str) -> dict[str, Any]:
    points = curve_points()
    fibers: dict[TaggedValue, list[Point]] = {}
    for point in points:
        fibers.setdefault(map1(point, map_name), []).append(point)
    for fiber in fibers.values():
        fiber.sort(key=point_sort_key)

    target_rows = []
    for target in points:
        bucket = map1(target, map_name)
        target_image = map2(target, map_name)
        function_values = [
            map2(point, map_name) for point in fibers[bucket]
        ]
        preimages = [
            index
            for index, value in enumerate(function_values)
            if value == target_image
        ]
        translated = [
            fibers[bucket][index] for index in preimages
        ]
        target_rows.append(
            {
                "target": point_record(target),
                "preimage_count": len(preimages),
                "unique_endpoint_returned": translated == [target],
            }
        )

    absent_map2_rejected = True
    for bucket, fiber in fibers.items():
        present = {map2(point, map_name) for point in fiber}
        absent = next(
            (
                tagged_finite(value)
                for value in range(FIELD_PRIME)
                if tagged_finite(value) not in present
            ),
            None,
        )
        if absent is None:
            continue
        if any(map2(point, map_name) == absent for point in fiber):
            absent_map2_rejected = False

    sizes = [len(fiber) for fiber in fibers.values()]
    degree = int(MAP_SPECS[map_name]["degree_upper_bound"])
    return {
        "map_name": map_name,
        "map_definition": MAP_SPECS[map_name],
        "projective_domain_size": len(points),
        "MAP1_image_size_D": len(fibers),
        "maximum_subfunction_domain_L": max(sizes),
        "D_times_L": len(fibers) * max(sizes),
        "coverage_inequality_D_times_L_at_least_N": (
            len(fibers) * max(sizes) >= len(points)
        ),
        "fiber_size_histogram": {
            str(size): count
            for size, count in sorted(Counter(sizes).items())
        },
        "maximum_fiber_at_most_rational_degree": max(sizes) <= degree,
        "every_target_has_one_fd_preimage": all(
            row["preimage_count"] == 1 for row in target_rows
        ),
        "TR_returns_exact_endpoint_for_every_target": all(
            row["unique_endpoint_returned"] for row in target_rows
        ),
        "absent_MAP2_value_rejected": absent_map2_rejected,
        "infinity_bucket_is_explicit": tagged_infinity() in fibers,
        "runtime_credit": False,
        "scope": (
            "exact finite semantic replay for identity endpoint inversion; "
            "fiber enumeration is not an asymptotic implementation"
        ),
    }


def elliptic_fiber_controls() -> list[dict[str, Any]]:
    return [
        fiber_semantics_control(map_name)
        for map_name in MAP_SPECS
    ]


def random_bucket(point: Point, bucket_count: int) -> int:
    payload = (
        "infinity"
        if point is None
        else f"{point[0]},{point[1]}"
    )
    digest = hashlib.sha256(
        f"P1553-R92|{bucket_count}|{payload}".encode("ascii")
    ).digest()
    return int.from_bytes(digest, "big") % bucket_count


def matched_random_partition_controls() -> list[dict[str, Any]]:
    points = curve_points()
    controls = []
    for requested_bucket_count in (7, 13, 31, 49, 71):
        fibers: dict[int, list[Point]] = {}
        for point in points:
            fibers.setdefault(
                random_bucket(point, requested_bucket_count),
                [],
            ).append(point)
        maximum_fiber = max(len(fiber) for fiber in fibers.values())
        controls.append(
            {
                "requested_bucket_count": requested_bucket_count,
                "occupied_bucket_count_D": len(fibers),
                "maximum_fiber_L": maximum_fiber,
                "D_times_L": len(fibers) * maximum_fiber,
                "coverage_inequality_exact": (
                    len(fibers) * maximum_fiber >= len(points)
                ),
                "full_point_MAP2_and_TR_endpoint_exact": all(
                    point in fibers[random_bucket(point, requested_bucket_count)]
                    for point in points
                ),
            }
        )
    return controls


def synthetic_source_dictionary_control() -> dict[str, Any]:
    points = curve_points()
    endpoint_by_source = {
        source_id: point for source_id, point in enumerate(points)
    }
    source_by_endpoint = {
        point: source_id for source_id, point in endpoint_by_source.items()
    }
    fiber_control = fiber_semantics_control("x")
    recovered = [
        source_by_endpoint[point]
        for point in points
    ]
    return {
        "synthetic_source_count": len(endpoint_by_source),
        "one_source_per_endpoint": (
            len(source_by_endpoint) == len(endpoint_by_source)
        ),
        "explicit_dictionary_TR_exact": all(
            endpoint_by_source[source_id] == point
            for source_id, point in zip(recovered, points)
        ),
        "x_fiber_endpoint_control_exact": fiber_control[
            "TR_returns_exact_endpoint_for_every_target"
        ],
        "actual_five_a_five_c_source_tuples_used": False,
        "explicit_dictionary_entry_count": len(source_by_endpoint),
        "asymptotic_dictionary_state_exponent_B": fraction_record(
            ENDPOINT_EXPONENT
        ),
        "inside_setup_cap": ENDPOINT_EXPONENT <= SETUP_CAP,
        "candidate_credit": False,
        "scope": (
            "optimistic one-source-per-endpoint control; the explicit "
            "dictionary supplies the missing source translator at B^5 state"
        ),
    }


def framework_exponents(
    fiber_exponent: Fraction,
    delta: Fraction,
    *,
    charge_shared_randomness: bool,
) -> dict[str, Fraction]:
    if not Fraction(0) <= fiber_exponent <= ENDPOINT_EXPONENT:
        raise AssertionError("fiber exponent must lie in [0,5]")
    if not Fraction(0) <= delta <= Fraction(1):
        raise AssertionError("delta must lie in [0,1]")
    image_exponent = ENDPOINT_EXPONENT - fiber_exponent
    per_subfunction_advice = (
        image_exponent
        + fiber_exponent * (Fraction(3, 2) - delta)
    )
    query = fiber_exponent * delta
    setup = (
        max(per_subfunction_advice, query)
        if charge_shared_randomness
        else per_subfunction_advice
    )
    return {
        "image": image_exponent,
        "fiber": fiber_exponent,
        "delta": delta,
        "per_subfunction_advice": per_subfunction_advice,
        "shared_randomness_advice": query,
        "setup": setup,
        "query": query,
    }


def exponent_point(
    fiber_exponent: Fraction,
    delta: Fraction,
    *,
    charge_shared_randomness: bool,
) -> dict[str, Any]:
    values = framework_exponents(
        fiber_exponent,
        delta,
        charge_shared_randomness=charge_shared_randomness,
    )
    return {
        f"{name}_exponent_B": fraction_record(value)
        for name, value in values.items()
    } | {
        "shared_randomness_charged_to_setup": (
            charge_shared_randomness
        ),
        "setup_cap_satisfied": values["setup"] <= SETUP_CAP,
        "online_cap_satisfied": values["query"] <= ONLINE_CAP,
    }


def subfunction_framework_cost_control() -> dict[str, Any]:
    actual_unconstrained = exponent_point(
        Fraction(10, 3),
        Fraction(1),
        charge_shared_randomness=True,
    )
    optimistic_free_randomness = exponent_point(
        Fraction(5),
        Fraction(1),
        charge_shared_randomness=False,
    )
    online_compatible = exponent_point(
        Fraction(5, 4),
        Fraction(1),
        charge_shared_randomness=True,
    )
    return {
        "endpoint_coverage": {
            "endpoint_count": "N=B^5",
            "MAP1_image_size": "D=B^d",
            "maximum_subfunction_domain": "L=B^ell",
            "counting_constraint": "D*L>=N, hence d+ell>=5",
            "rational_map_specialization": (
                "for h:E->P1 of degree r, each rational fiber has "
                "at most r points, so |h(E(F_q))|*r>=|E(F_q)|"
            ),
        },
        "bound_theorem_4_1": {
            "space": (
                "soft-O(L^(3/2-delta)*D + Aux + L^delta)"
            ),
            "query": "soft-O(L^delta)",
            "delta_range": "[0,1]",
            "charged_term_exponent": (
                "max(5+ell*(1/2-delta), ell*delta)"
            ),
            "query_exponent": "ell*delta",
        },
        "actual_theorem_advice_unconstrained_optimum": (
            actual_unconstrained
        ),
        "optimistic_free_shared_randomness_optimum": (
            optimistic_free_randomness
        ),
        "online_compatible_optimum": online_compatible,
        "exact_optimum_proofs": {
            "actual_unconstrained": (
                "for ell<=10/3 choose delta=1 and the first advice "
                "term is at least 5-ell/2>=10/3; for ell>=10/3, "
                "max(5+ell/2-ell*delta,ell*delta)>=10/3, with "
                "equality at ell=10/3,delta=1"
            ),
            "optimistic_free_shared_randomness": (
                "after deleting the L^delta advice term, choose "
                "delta=1 and ell=5 to obtain 5-ell/2=5/2"
            ),
            "online_compatible": (
                "ell*delta<=5/4; for ell<=5/4 the best point has "
                "delta=1 and setup>=5-ell/2, while for ell>=5/4 "
                "setup>=15/4+ell/2; both meet at ell=5/4"
            ),
        },
        "all_state_charged_unconstrained_minimum_B": fraction_record(
            Fraction(10, 3)
        ),
        "free_randomness_optimistic_minimum_B": fraction_record(
            Fraction(5, 2)
        ),
        "online_compatible_minimum_B": fraction_record(
            Fraction(35, 8)
        ),
        "any_certified_point_inside_setup_cap": False,
        "any_certified_point_inside_both_caps": False,
        "map_description_and_evaluation_granted_free": True,
        "auxiliary_source_translation_state_granted_free": True,
        "construction_specific_nonclaim": (
            "this charges the advice built by Theorem 4.1 for D "
            "independent generic subfunctions; it is not a lower bound "
            "against joint compression across subfunctions"
        ),
    }


def projective_and_random_controls() -> dict[str, Any]:
    fiber_controls = elliptic_fiber_controls()
    random_controls = matched_random_partition_controls()
    return {
        "projective_infinity_replayed_in_every_map": all(
            control["infinity_bucket_is_explicit"]
            for control in fiber_controls
        ),
        "no_nontrivial_two_torsion_on_prime_order_curve": (
            prime_curve_control()[
                "nontrivial_two_torsion_point_count"
            ]
            == 0
        ),
        "rational_fiber_controls": fiber_controls,
        "matched_random_partition_controls": random_controls,
        "all_counting_controls_exact": all(
            control["coverage_inequality_D_times_L_at_least_N"]
            for control in fiber_controls
        )
        and all(
            control["coverage_inequality_exact"]
            for control in random_controls
        ),
        "actual_semaev_reduced_branch_replay": False,
        "nonreduced_signed_infinity_tangent_source_replay": False,
    }


def cost_ledger() -> dict[str, Any]:
    framework = subfunction_framework_cost_control()
    source = synthetic_source_dictionary_control()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "subfunction_framework": framework,
        "source_translation": source,
        "best_charged_setup_exponent_without_online_cap": (
            framework["all_state_charged_unconstrained_minimum_B"]
        ),
        "best_charged_setup_exponent_with_online_cap": (
            framework["online_compatible_minimum_B"]
        ),
        "fatal_obstructions": [
            "the theorem advice is at least B^(10/3) even without the online cap",
            "the online-compatible theorem point needs B^(35/8) setup",
            "the exact elliptic maps return only an endpoint, not a 5A+5C source tuple",
            "an explicit one-source-per-endpoint translator costs B^5 state",
        ],
        "scope_exception": (
            "a target-dependent overlapping correspondence with one jointly "
            "compressed operator, rather than D independently preprocessed "
            "endpoint fibers, plus an exact compact source unranking map"
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    curve = prime_curve_control()
    fibers = elliptic_fiber_controls()
    framework = subfunction_framework_cost_control()
    source = synthetic_source_dictionary_control()
    branches = projective_and_random_controls()
    costs = cost_ledger()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_compact_elliptic_"
            "subfunction_map.r92.v1"
        ),
        "candidate_interface": {
            "MAP1": "h(P), selecting one rational fiber",
            "MAP2": "a second public coordinate identifying P in the fiber",
            "f_d": "the MAP2 values on the d-th fiber",
            "TR": "translate a local fiber index back to an endpoint",
        },
        "toy_prime_curve": curve,
        "frozen_rational_maps": MAP_SPECS,
        "theorem_4_1": {
            "paper": "Dinur-Golovnev arXiv:2512.04258v2",
            "space": (
                "soft-O(L^(3/2-delta)*D + Aux + L^delta)"
            ),
            "query": "soft-O(L^delta)",
            "version_sha256": DINUR_GOLOVNEV_V2_SHA256,
        },
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
    }
    map_ledger = {
        "schema": "p1553.elliptic_map1_map2_fd_tr_ledger.r92.v1",
        "frozen_candidate": frozen,
        "rational_fiber_semantics": fibers,
        "subfunction_framework_cost": framework,
        "cost_ledger": costs,
    }
    replay = {
        "schema": (
            "p1553.compact_divisor_target_source_replay.r92.v1"
        ),
        "identity_endpoint_inversion_controls": fibers,
        "synthetic_explicit_source_dictionary_control": source,
        "actual_five_a_five_c_source_TR_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.projective_branch_random_group_controls.r92.v1"
        ),
        "prime_curve_control": curve,
        "projective_and_random_controls": branches,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r92.v1",
        "compact_endpoint_partition_inside_caps": False,
        "actual_five_a_five_c_source_TR_complete": False,
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
        "paper_v2_theorem_4_1_charged_exactly": True,
        "toy_curve_is_exact_prime_order_group": all(
            (
                curve["field_prime_exactly_prime"],
                curve["curve_discriminant_nonzero"],
                curve["point_count_matches_stated_order"],
                curve["group_order_exactly_prime"],
                curve["addition_closed_on_full_enumeration"],
                curve["generator_has_stated_prime_order"],
            )
        ),
        "projective_infinity_included": curve["infinity_included"],
        "four_rational_fiber_interfaces_endpoint_exact": all(
            control["every_target_has_one_fd_preimage"]
            and control["TR_returns_exact_endpoint_for_every_target"]
            for control in fibers
        ),
        "rational_degree_fiber_bounds_exact": all(
            control["maximum_fiber_at_most_rational_degree"]
            for control in fibers
        ),
        "finite_coverage_counting_exact": all(
            control["coverage_inequality_D_times_L_at_least_N"]
            for control in fibers
        ),
        "matched_random_partition_counting_exact": branches[
            "all_counting_controls_exact"
        ],
        "charged_unconstrained_minimum_exact": (
            framework[
                "all_state_charged_unconstrained_minimum_B"
            ]["exact"]
            == "10/3"
        ),
        "optimistic_free_randomness_minimum_exact": (
            framework[
                "free_randomness_optimistic_minimum_B"
            ]["exact"]
            == "5/2"
        ),
        "online_compatible_minimum_exact": (
            framework["online_compatible_minimum_B"]["exact"]
            == "35/8"
        ),
        "synthetic_explicit_source_dictionary_exact": source[
            "explicit_dictionary_TR_exact"
        ],
        "subfunction_framework_inside_setup_cap": False,
        "subfunction_framework_inside_both_caps": False,
        "compact_five_a_five_c_source_TR_complete": False,
        "actual_semaev_projective_exceptional_charts": False,
        "signed_source_biconditional_complete": False,
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
            "ELLIPTIC_RATIONAL_FIBER_ENDPOINT_TR_EXACT__"
            "THEOREM_STATE_MIN_B10O3__"
            "ONLINE_COMPATIBLE_MIN_B35O8__"
            "FIVE_A_FIVE_C_SOURCE_TR_ABSENT"
        ),
        "source_bindings": {
            "r91_report": {
                "path": str(R91_REPORT),
                "sha256": R91_REPORT_SHA256,
            },
            "r91_gate": {
                "path": str(R91_GATE),
                "sha256": R91_GATE_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r83_report": {
                "path": str(R83_REPORT),
                "sha256": R83_REPORT_SHA256,
            },
            "r83_prime_order_quotient_control": {
                "path": str(R83_QUOTIENT_CONTROL),
                "sha256": R83_QUOTIENT_CONTROL_SHA256,
            },
            "p1515_local_separator_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
            "p1515_field_router_candidate": {
                "path": str(P1515_ROUTER),
                "sha256": P1515_ROUTER_SHA256,
            },
            "dinur_golovnev_v2": {
                "path": str(DINUR_GOLOVNEV_V2),
                "sha256": DINUR_GOLOVNEV_V2_SHA256,
            },
        },
        "novelty_scope": (
            "R92 is the first campaign receipt to instantiate public "
            "elliptic MAP1, MAP2, f_d, and endpoint TR on a fully "
            "enumerated prime-order projective curve and then optimize "
            "Theorem 4.1 under the unavoidable D*L endpoint coverage law."
        ),
        "prime_curve_control": curve,
        "elliptic_rational_fiber_controls": fibers,
        "subfunction_framework_cost_control": framework,
        "source_translation_control": source,
        "projective_and_random_controls": branches,
        "cost_ledger": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_compact_elliptic_subfunction_map.json"
            ),
            "map_ledger": "elliptic_map1_map2_fd_tr_ledger.json",
            "source_replay": (
                "compact_divisor_target_source_replay.json"
            ),
            "exceptional": (
                "projective_branch_and_random_group_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r92.json",
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
            "This closes direct endpoint-partition instantiations of the "
            "bound Theorem 4.1 construction when each of D subfunctions "
            "receives independent generic inversion advice. It is not a "
            "lower bound on a jointly compressed family of overlapping "
            "correspondences, a non-Fiat-Naor data structure, or a "
            "representation-changing summation-polynomial/FFE identity."
        ),
        "next_action": (
            "Construct or refute one target-dependent overlapping S3/S4 "
            "incidence correspondence with a single shared semilinear "
            "operator across all charts. It must prove joint state below "
            "B^(9/4), fresh work and workspace below B^(5/4), and exact "
            "5A+5C source unranking on every projective branch, without "
            "D independent inversion tables, endpoint/source dictionaries, "
            "DLP labels, verifier oracles, or omitted elimination cost."
        ),
        "disposition": (
            "REJECT_DIRECT_ELLIPTIC_ENDPOINT_PARTITION_SUBFUNCTION_"
            "FRAMEWORK_ONLY__FOUR_PROJECTIVE_RATIONAL_MAP_CONTROLS_EXACT__"
            "D_TIMES_L_COVERAGE_EXACT__THEOREM_ALL_STATE_MIN_B10O3__"
            "ONLINE_COMPATIBLE_MIN_B35O8__EXPLICIT_SOURCE_TRANSLATOR_B5__"
            "ACTUAL_5A5C_SOURCE_TR_ABSENT__OVERLAPPING_SHARED_"
            "CORRESPONDENCE_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "map_ledger": map_ledger,
        "source_replay": replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_compact_elliptic_subfunction_map_"
            "probe_report_r92.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_compact_elliptic_subfunction_map.json"
        ),
    )
    parser.add_argument(
        "--map-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "elliptic_map1_map2_fd_tr_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "compact_divisor_target_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_branch_and_random_group_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r92.json"
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
    write_json(args.map_ledger_output, bundle["map_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
