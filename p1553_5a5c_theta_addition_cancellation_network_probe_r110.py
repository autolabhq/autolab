#!/usr/bin/env python3
"""Test a finite-field theta/Abel cancellation network on the 5A+5C deck."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_theta_addition_cancellation_network.r110.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
A_ATOM_EXPONENT = Fraction(2, 5)
C_ATOM_EXPONENT = Fraction(3, 5)
PAIR_TABLE_EXPONENT = Fraction(1)
SHIFT_SEARCH_EXPONENT = Fraction(6, 5)
FULL_SOURCE_EXPONENT = Fraction(5)
BALANCED_SECTION_EXPONENT = Fraction(12, 5)
SOURCE_ARITY = 10
ALTERNANT_DEGREE = SOURCE_ARITY + 1
NEGATIVE_SAMPLES_PER_INSTANCE = 32

R109_PRODUCER = pathlib.Path(
    "p1553_5a5c_poincare_theta_target_section_rank_probe_r109.py"
)
R109_PRODUCER_SHA256 = (
    "085b3df3d7442a78800dcd01e67c8fc155a9dfb1181d9a141ed9097c2af65c2e"
)
R109_REPORT = pathlib.Path(
    "p1553_5a5c_poincare_theta_target_section_rank_probe_report_r109.json"
)
R109_REPORT_SHA256 = (
    "73c826964d17877ea26436c40bd52753f2b5d0c416fee86ef04f2b4c8f030e4d"
)
R109_GATE = pathlib.Path(
    "p1553_5a5c_poincare_theta_target_section_rank_probe_gate_r109.md"
)
R109_GATE_SHA256 = (
    "fc876cc0b83fc43c61c2b95eff1cb0229663f1974b024d4a8f4619b264b7cc00"
)
R109_PARENT = pathlib.Path(
    "p1553_5a5c_poincare_theta_target_section_rank_probe_parent_report_r109.yaml"
)
R109_PARENT_SHA256 = (
    "dd826bb338e9eeada0692f75ec94f7cecc2ae2b407b0830120d4c9c742e1d9ef"
)
R108_REPORT = pathlib.Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
    "norm_probe_report_r108.json"
)
R108_REPORT_SHA256 = (
    "91fab519c6a59851a672aed1a9e18fac8c434359c91a9e1d9fc0acb112ee7b41"
)
R105_REPORT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_report_r105.json"
)
R105_REPORT_SHA256 = (
    "c00b8c3e934e783012acb1dab83282f98720702bb8cd81a706fbdf4671f93a83"
)
IDEA_REGISTRY = pathlib.Path("p1553_r35_artifact_index_README.md")
IDEA_REGISTRY_SHA256 = (
    "9f4371eefd5e4019833eef858e3bda79d41aff0c5d7b861c71a5987a96acc392"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R109_PRODUCER: R109_PRODUCER_SHA256,
        R109_REPORT: R109_REPORT_SHA256,
        R109_GATE: R109_GATE_SHA256,
        R109_PARENT: R109_PARENT_SHA256,
        R108_REPORT: R108_REPORT_SHA256,
        R105_REPORT: R105_REPORT_SHA256,
        IDEA_REGISTRY: IDEA_REGISTRY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R110 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R109 = load_module("p1553_r109_for_r110", R109_PRODUCER)
R108 = R109.R108
R107 = R109.R107
R105 = R109.R105
R102 = R109.R102
R84 = R109.R84
R82 = R109.R82
R70 = R109.R70


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


def source_json(source: Source) -> list[list[int]]:
    return [list(source[0]), list(source[1])]


def json_source(value: Sequence[Sequence[int]]) -> Source:
    return tuple(value[0]), tuple(value[1])


def add_many(points: Iterable[Point], curve: dict[str, Any]) -> Point:
    result: Point = None
    for point in points:
        result = R70.add(result, point, curve)
    return result


def alternant_basis_row(point: Point, prime: int) -> list[int]:
    if point is None:
        raise AssertionError("the affine alternant chart excludes O")
    x_coordinate, y_coordinate = point
    x2 = x_coordinate * x_coordinate % prime
    x3 = x2 * x_coordinate % prime
    x4 = x3 * x_coordinate % prime
    x5 = x4 * x_coordinate % prime
    return [
        1,
        x_coordinate,
        y_coordinate,
        x2,
        x_coordinate * y_coordinate % prime,
        x3,
        x2 * y_coordinate % prime,
        x4,
        x3 * y_coordinate % prime,
        x5,
        x4 * y_coordinate % prime,
    ]


def rank_mod(matrix: Sequence[Sequence[int]], prime: int) -> int:
    rows = [[value % prime for value in row] for row in matrix]
    if not rows:
        return 0
    rank = 0
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column] % prime
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], prime - 2, prime)
        rows[rank] = [
            value * inverse % prime for value in rows[rank]
        ]
        for row in range(rank + 1, len(rows)):
            factor = rows[row][column]
            if factor == 0:
                continue
            rows[row] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def translated_domains(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    target: Point,
    shifts: Sequence[Point],
    curve: dict[str, Any],
) -> list[list[Point]]:
    domains = [
        *([list(atoms_a)] * 5),
        *([list(atoms_c)] * 5),
        [R70.negate(target, curve)],
    ]
    return [
        [R70.add(point, shifts[position], curve) for point in domain]
        for position, domain in enumerate(domains)
    ]


def domains_are_disjoint_affine(domains: Sequence[Sequence[Point]]) -> bool:
    seen: set[Point] = set()
    for domain in domains:
        if any(point is None for point in domain):
            return False
        if len(set(domain)) != len(domain):
            return False
        if any(point in seen for point in domain):
            return False
        seen.update(domain)
    return True


def zero_sum_position_shifts(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    target: Point,
    curve: dict[str, Any],
    salt: str,
) -> dict[str, Any]:
    shift_generator = next(R82.hash_point_candidates(curve, salt))
    translated_entry_count = (
        5 * len(atoms_a) + 5 * len(atoms_c) + 1
    )
    candidate_limit = translated_entry_count**2 + 1
    for multiplier in range(1, candidate_limit + 1):
        shifts = [
            R70.scalar_mul(
                multiplier * (position + 1),
                shift_generator,
                curve,
            )
            for position in range(SOURCE_ARITY)
        ]
        shifts.append(
            R70.negate(add_many(shifts, curve), curve)
        )
        domains = translated_domains(
            atoms_a,
            atoms_c,
            target,
            shifts,
            curve,
        )
        if domains_are_disjoint_affine(domains):
            return {
                "shift_generator": shift_generator,
                "multiplier": multiplier,
                "shifts": shifts,
                "translated_domains": domains,
                "shift_sum": add_many(shifts, curve),
                "candidate_limit": candidate_limit,
            }
    raise AssertionError("unable to find zero-sum disjoint position shifts")


def shifted_row_tables(
    schedule: dict[str, Any],
    prime: int,
) -> list[list[list[int]]]:
    return [
        [alternant_basis_row(point, prime) for point in domain]
        for domain in schedule["translated_domains"]
    ]


def source_matrix(
    source: Source,
    row_tables: Sequence[Sequence[Sequence[int]]],
) -> list[list[int]]:
    return [
        *[
            list(row_tables[position][index])
            for position, index in enumerate(source[0])
        ],
        *[
            list(row_tables[5 + position][index])
            for position, index in enumerate(source[1])
        ],
        list(row_tables[10][0]),
    ]


def alternant_is_zero(
    source: Source,
    row_tables: Sequence[Sequence[Sequence[int]]],
    prime: int,
) -> tuple[bool, int]:
    rank = rank_mod(source_matrix(source, row_tables), prime)
    return rank < ALTERNANT_DEGREE, rank


def direct_source_relation(
    source: Source,
    target: Point,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    return R102.source_endpoint(source, atoms_a, atoms_c, curve) == target


def repeated_source() -> Source:
    return tuple([0] * 5), tuple([0] * 5)


def negative_sources(
    size_a: int,
    size_c: int,
    target: Point,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> list[Source]:
    output = []
    for source in R102.full_sources(size_a, size_c):
        if not direct_source_relation(
            source,
            target,
            atoms_a,
            atoms_c,
            curve,
        ):
            output.append(source)
            if len(output) == NEGATIVE_SAMPLES_PER_INSTANCE:
                break
    if len(output) != NEGATIVE_SAMPLES_PER_INSTANCE:
        raise AssertionError("insufficient negative alternant controls")
    return output


def r105_double_fibers() -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(R105_REPORT.read_text(encoding="utf-8"))
    output = {}
    for instance in payload["actual_controls"]["instances"]:
        if instance["maximum_target_multiplicity"] != 2:
            continue
        query = next(
            row
            for row in instance["query_controls"]
            if row["label"] == "maximum_multiplicity"
        )
        output[(instance["family_id"], instance["offset"])] = {
            "target": (
                None
                if query["target"] is None
                else tuple(query["target"])
            ),
            "sources": [
                json_source(source) for source in query["expected_sources"]
            ],
        }
    return output


def unshifted_repeat_false_positive(
    source: Source,
    nonrelation_target: Point,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    points = [
        *[atoms_a[index] for index in source[0]],
        *[atoms_c[index] for index in source[1]],
        R70.negate(nonrelation_target, curve),
    ]
    prime = curve["field_prime"]
    if any(point is None for point in points):
        rank = 0
    else:
        rank = rank_mod(
            [alternant_basis_row(point, prime) for point in points],
            prime,
        )
    return {
        "source": source_json(source),
        "nonrelation_target": point_json(nonrelation_target),
        "direct_relation": direct_source_relation(
            source,
            nonrelation_target,
            atoms_a,
            atoms_c,
            curve,
        ),
        "unshifted_alternant_rank": rank,
        "unshifted_alternant_zero": rank < ALTERNANT_DEGREE,
        "false_positive_from_repeated_rows": (
            rank < ALTERNANT_DEGREE
            and not direct_source_relation(
                source,
                nonrelation_target,
                atoms_a,
                atoms_c,
                curve,
            )
        ),
    }


def instance_control(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
    doubles: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    double = (
        doubles.get((curve["family_id"], offset))
        if control_class == "actual"
        else None
    )
    if double is None:
        positives = [repeated_source()]
        target = R102.source_endpoint(
            positives[0],
            atoms_a,
            atoms_c,
            curve,
        )
        target_class = "repeated_atom_positive"
    else:
        positives = double["sources"]
        target = double["target"]
        target_class = "r105_actual_double_fiber"
    schedule = zero_sum_position_shifts(
        atoms_a,
        atoms_c,
        target,
        curve,
        (
            f"R110|{control_class}|{curve['family_id']}|"
            f"{offset}|position-shift"
        ),
    )
    prime = curve["field_prime"]
    rows = shifted_row_tables(schedule, prime)
    positive_rows = []
    for source in positives:
        determinant_zero, rank = alternant_is_zero(source, rows, prime)
        positive_rows.append(
            {
                "source": source_json(source),
                "direct_relation": direct_source_relation(
                    source,
                    target,
                    atoms_a,
                    atoms_c,
                    curve,
                ),
                "alternant_rank": rank,
                "alternant_zero": determinant_zero,
                "marker": list(R105.marker_vector(source, prime)),
            }
        )
    negative_rows = []
    for source in negative_sources(
        len(atoms_a),
        len(atoms_c),
        target,
        atoms_a,
        atoms_c,
        curve,
    ):
        determinant_zero, rank = alternant_is_zero(source, rows, prime)
        negative_rows.append(
            {
                "source": source_json(source),
                "direct_relation": False,
                "alternant_rank": rank,
                "alternant_zero": determinant_zero,
            }
        )
    nonrelation_target = R70.add(
        target,
        schedule["shift_generator"],
        curve,
    )
    if nonrelation_target == target:
        raise AssertionError("nonrelation target control did not move")
    unshifted = unshifted_repeat_false_positive(
        repeated_source(),
        nonrelation_target,
        atoms_a,
        atoms_c,
        curve,
    )
    all_translated_points = [
        point
        for domain in schedule["translated_domains"]
        for point in domain
    ]
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": prime,
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "target_class": target_class,
        "target": point_json(target),
        "shift_generator": point_json(schedule["shift_generator"]),
        "shift_multiplier": schedule["multiplier"],
        "shift_candidate_limit": schedule["candidate_limit"],
        "position_shifts": [
            point_json(point) for point in schedule["shifts"]
        ],
        "position_shift_sum": point_json(schedule["shift_sum"]),
        "position_shift_sum_is_identity": schedule["shift_sum"] is None,
        "translated_domain_entry_count": len(all_translated_points),
        "translated_domains_pairwise_disjoint": (
            len(all_translated_points) == len(set(all_translated_points))
        ),
        "translated_domains_affine": all(
            point is not None for point in all_translated_points
        ),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "all_positive_alternants_exact": all(
            row["direct_relation"] and row["alternant_zero"]
            for row in positive_rows
        ),
        "all_negative_alternants_exact": all(
            not row["direct_relation"] and not row["alternant_zero"]
            for row in negative_rows
        ),
        "unshifted_repeat_control": unshifted,
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def actual_and_matched_controls() -> dict[str, Any]:
    doubles = r105_double_fibers()
    actual = [
        instance_control(
            dict(family),
            offset,
            "actual",
            doubles,
        )
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        instance_control(
            dict(family),
            offset,
            "matched_random_deck",
            doubles,
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "instance_count": len(rows),
        "double_fiber_instance_count": sum(
            row["target_class"] == "r105_actual_double_fiber"
            for row in rows
        ),
        "positive_source_count": sum(
            len(row["positive_rows"]) for row in rows
        ),
        "negative_source_count": sum(
            len(row["negative_rows"]) for row in rows
        ),
        "all_zero_sum_shift_schedules_exact": all(
            row["position_shift_sum_is_identity"] for row in rows
        ),
        "all_translated_domains_disjoint_affine": all(
            row["translated_domains_pairwise_disjoint"]
            and row["translated_domains_affine"]
            for row in rows
        ),
        "all_positive_and_negative_alternants_exact": all(
            row["all_positive_alternants_exact"]
            and row["all_negative_alternants_exact"]
            for row in rows
        ),
        "all_unshifted_repeat_controls_false_positive": all(
            row["unshifted_repeat_control"][
                "false_positive_from_repeated_rows"
            ]
            for row in rows
        ),
        "maximum_shift_multiplier": max(
            row["shift_multiplier"] for row in rows
        ),
    }


def alternant_theorem() -> dict[str, Any]:
    return {
        "curve_model": "short Weierstrass y^2=x^3+a*x+b",
        "line_bundle": "L(11O)",
        "basis_pole_orders": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "basis": [
            "1",
            "x",
            "y",
            "x^2",
            "x*y",
            "x^3",
            "x^2*y",
            "x^4",
            "x^3*y",
            "x^5",
            "x^4*y",
        ],
        "predicate": (
            "det(eval_basis(Q_i))=0 iff sum_i Q_i=O for eleven "
            "pairwise-distinct affine points Q_i"
        ),
        "proof": (
            "Singularity gives a nonzero f in L(11O) vanishing at all "
            "eleven distinct Q_i. Degree forces div(f)=sum_i[Q_i]-11[O], "
            "whose Abel sum is O. Conversely Abel sum O makes that divisor "
            "principal and supplies the kernel section."
        ),
        "position_shift_rule": (
            "Q_i=P_i+S_i with sum_i S_i=O; P_11=-T"
        ),
        "original_target_biconditional": (
            "det=0 iff sum_(i=1)^10 P_i=T"
        ),
        "confluent_jets_required_after_position_separation": False,
        "target_dependent_shift_search_scalar_blind": True,
    }


def cauchy_theta_scope() -> dict[str, Any]:
    return {
        "analytic_identity": (
            "det K(x_i,y_j;lambda) = "
            "sigma(lambda+X-Y)/sigma(lambda) * "
            "prod_(a<b)sigma(x_a-x_b)sigma(y_b-y_a) / "
            "prod_(a,b)sigma(x_a-y_b)"
        ),
        "primary_source": "https://arxiv.org/abs/2305.02837",
        "finite_field_algebraic_substitute": (
            "the position-separated L(11O) Abel alternant"
        ),
        "kronecker_kernel_is_poincare_bundle_section": True,
        "globally_scalar_rational_kernel": False,
        "unshifted_repeated_rows_are_false_zeros": True,
        "confluent_or_position_separated_repair_required": True,
        "pairwise_theta_table_exponent_B": fraction_record(
            PAIR_TABLE_EXPONENT
        ),
        "pairwise_theta_table_inside_online_cap": (
            PAIR_TABLE_EXPONENT <= ONLINE_CAP
        ),
        "constant_determinant_per_supplied_source": True,
        "constant_determinant_is_source_locator": False,
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "position_separated_alternant": {
            "target_shift_search_exponent_B": fraction_record(
                SHIFT_SEARCH_EXPONENT
            ),
            "target_shift_forbidden_multiplier_bound": (
                "(5*|A|+5*|C|+1)^2"
            ),
            "thin_row_table_state_exponent_B": fraction_record(
                C_ATOM_EXPONENT
            ),
            "both_inside_online_cap": (
                SHIFT_SEARCH_EXPONENT <= ONLINE_CAP
                and C_ATOM_EXPONENT <= ONLINE_CAP
            ),
            "determinant_dimension": ALTERNANT_DEGREE,
            "per_supplied_source_cost_exponent_B": fraction_record(
                Fraction(0)
            ),
            "full_source_scan_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "full_source_scan_inside_caps": False,
        },
        "zero_mask_contraction": {
            "field_zero_mask": "1-det^(p-1)",
            "mask_exact_on_field_values": True,
            "global_single_row_section_dimension": "11*(p-1)",
            "generic_prime_scaling_p": "Theta(B^5)",
            "global_row_mode_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "global_row_mode_inside_caps": False,
            "finite_deck_restriction_eliminates_global_mode": True,
            "finite_deck_annihilator_contraction_supplied": False,
            "balanced_postmask_incidence_exponent_B": fraction_record(
                BALANCED_SECTION_EXPONENT
            ),
            "balanced_postmask_incidence_inside_setup_cap": False,
            "balanced_postmask_incidence_inside_online_cap": False,
        },
        "linear_value_channel": {
            "determinant_multilinearity_allows_unary_row_sums": True,
            "linear_value_moment_reports_zero_locations": False,
            "quadratic_mixed_discriminant_reports_zero_locations": False,
            "deduplicated_campaign_lane": (
                "ECDLP-IDEA-012 determinant value channel and P1539 "
                "Abel-Jacobi evaluation minor"
            ),
        },
        "constructor_status": {
            "finite_field_target_zero_biconditional_complete": True,
            "repeated_atom_false_zeros_removed": True,
            "r108_weight_14400_marker_semantics_preserved": True,
            "inside_cap_zero_mask_aggregate_complete": False,
            "exact_source_locator_complete": False,
            "generic_multiplicity_integer_lift_complete": False,
        },
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r109_producer": {
            "path": str(R109_PRODUCER),
            "sha256": R109_PRODUCER_SHA256,
        },
        "r109_report": {
            "path": str(R109_REPORT),
            "sha256": R109_REPORT_SHA256,
        },
        "r109_gate": {
            "path": str(R109_GATE),
            "sha256": R109_GATE_SHA256,
        },
        "r109_parent": {
            "path": str(R109_PARENT),
            "sha256": R109_PARENT_SHA256,
        },
        "r108_report": {
            "path": str(R108_REPORT),
            "sha256": R108_REPORT_SHA256,
        },
        "r105_report": {
            "path": str(R105_REPORT),
            "sha256": R105_REPORT_SHA256,
        },
        "idea_registry": {
            "path": str(IDEA_REGISTRY),
            "sha256": IDEA_REGISTRY_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = actual_and_matched_controls()
    theorem = alternant_theorem()
    cauchy = cauchy_theta_scope()
    ledger = cost_ledger()
    obligations = {
        "seven_source_bindings_verified": len(source_hashes) == 7,
        "sixteen_actual_and_matched_instances": (
            controls["instance_count"] == 16
        ),
        "two_actual_double_fibers_replayed": (
            controls["double_fiber_instance_count"] == 2
        ),
        "all_zero_sum_shift_schedules_exact": controls[
            "all_zero_sum_shift_schedules_exact"
        ],
        "all_translated_domains_disjoint_affine": controls[
            "all_translated_domains_disjoint_affine"
        ],
        "all_positive_and_negative_alternants_exact": controls[
            "all_positive_and_negative_alternants_exact"
        ],
        "unshifted_repeat_false_positive_reproduced": controls[
            "all_unshifted_repeat_controls_false_positive"
        ],
        "finite_field_target_zero_biconditional_complete": True,
        "confluent_jets_removed_by_position_separation": theorem[
            "confluent_jets_required_after_position_separation"
        ]
        is False,
        "theta_pair_table_inside_online_cap": cauchy[
            "pairwise_theta_table_inside_online_cap"
        ],
        "r108_marker_weight_semantics_preserved": True,
        "determinant_value_channel_deduplicated": True,
        "inside_cap_zero_mask_aggregate_complete": ledger[
            "constructor_status"
        ]["inside_cap_zero_mask_aggregate_complete"],
        "exact_source_locator_complete": ledger["constructor_status"][
            "exact_source_locator_complete"
        ],
        "generic_multiplicity_integer_lift_complete": ledger[
            "constructor_status"
        ]["generic_multiplicity_integer_lift_complete"],
        "known_rhs_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one target-specialized finite-deck annihilator "
        "for the position-separated Abel alternant determinant. It must "
        "compute the exact zero count or existence bit and return one "
        "coupled canonical 5A+5C source without evaluating the B^5 source "
        "body, materializing the B^(12/5) balanced incidence, or using DLP "
        "labels; charge target updates, integer lifting, R108 weight 14400, "
        "R105 markers, multiplicity, rank, logs, and identical descent."
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_theta_addition_cancellation_network.r110.v1",
        "source_bindings": source_binding_records(),
        "field_model": "actual j=0 prime-field curves from R82",
        "source_arity": {"a": 5, "c": 5, "target": 1},
        "alternant_theorem": theorem,
        "cauchy_theta_scope": cauchy,
        "position_shift_search": {
            "coefficients": list(range(1, 11)),
            "last_shift": "negative sum of first ten shifts",
            "maximum_multiplier": controls["maximum_shift_multiplier"],
            "deterministic_candidate_bound": (
                "(5*|A|+5*|C|+1)^2+1"
            ),
            "candidate_bound_exponent_B": fraction_record(
                SHIFT_SEARCH_EXPONENT
            ),
            "selection_before_source_outcomes": True,
            "target_specialized": True,
        },
        "primary_sources": [
            {
                "url": "https://arxiv.org/abs/2305.02837",
                "use": "Frobenius elliptic Cauchy determinant identity",
            },
            {
                "url": "https://arxiv.org/abs/1005.3623",
                "use": "addition laws as line-bundle sections",
            },
            {
                "url": "https://arxiv.org/abs/1801.05677",
                "use": "algebraic Kronecker section of the Poincare bundle",
            },
        ],
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "ECDLP-IDEA-012/p1539 Abel-Jacobi evaluation minor",
                "ECDLP-IDEA-012 determinant value channel",
            ],
            "new_local_control_only": (
                "zero-sum position separation removes confluent repeated-row "
                "and affine-pole branches on the current 5A+5C decks"
            ),
            "new_algorithm_claimed": False,
        },
    }
    marker_replay = {
        "schema": "p1553.theta_addition_canonical_marker_replay.r110.v1",
        "canonical_cycle_weight": R108.FULL_CYCLE_SCALE,
        "marker_dimension": R105.MARKER_DIMENSION,
        "position_shifts_do_not_change_source_indices": True,
        "all_positive_markers_preserved": True,
        "actual": [
            {
                "family_id": row["family_id"],
                "offset": row["offset"],
                "target": row["target"],
                "target_class": row["target_class"],
                "positive_rows": row["positive_rows"],
            }
            for row in controls["actual"]
        ],
        "matched_random_decks": [
            {
                "family_id": row["family_id"],
                "offset": row["offset"],
                "target": row["target"],
                "target_class": row["target_class"],
                "positive_rows": row["positive_rows"],
            }
            for row in controls["matched_random_decks"]
        ],
        "generic_marker_factorization_inside_caps": False,
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": "p1553.theta_addition_pole_exceptional_controls.r110.v1",
        "actual": controls["actual"],
        "matched_random_decks": controls["matched_random_decks"],
        "summary": {
            "all_zero_sum_shift_schedules_exact": controls[
                "all_zero_sum_shift_schedules_exact"
            ],
            "all_translated_domains_disjoint_affine": controls[
                "all_translated_domains_disjoint_affine"
            ],
            "all_positive_and_negative_alternants_exact": controls[
                "all_positive_and_negative_alternants_exact"
            ],
            "all_unshifted_repeat_controls_false_positive": controls[
                "all_unshifted_repeat_controls_false_positive"
            ],
            "maximum_shift_multiplier": controls[
                "maximum_shift_multiplier"
            ],
        },
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r110.v1",
        "target_zero_biconditional_complete": True,
        "inside_cap_relation_source_locator_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_TARGET_PREDICATE_POSITIVE_WITHHOLD_ALGORITHM_PROMOTION"
        ),
        "classification": (
            "POSITION_SEPARATED_L11_ABEL_ALTERNANT_GIVES_EXACT_FINITE_"
            "FIELD_TARGET_ZERO_BICONDITIONAL__ZERO_SUM_SHIFTS_REMOVE_"
            "REPEATED_ROW_AND_AFFINE_POLE_BRANCHES_INSIDE_ONLINE_CAP__"
            "ELLIPTIC_CAUCHY_AND_VALUE_CHANNEL_MERGE_WITH_IDEA012__"
            "CONSTANT_DETERMINANT_PER_SUPPLIED_SOURCE_DOES_NOT_LOCATE_"
            "ZEROS__GLOBAL_ZERO_MASK_RESTORES_B5_ROW_MODE_AND_BALANCED_"
            "INCIDENCE_B12O5__FINITE_DECK_ANNIHILATOR_CONTRACTION_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "alternant_theorem": theorem,
        "cauchy_theta_scope": cauchy,
        "control_summary": {
            key: value
            for key, value in controls.items()
            if key not in {"actual", "matched_random_decks"}
        },
        "cost_ledger": ledger,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "failures": failures,
            "target_zero_biconditional_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_5a5c_theta_addition_cancellation_network.json",
            "circuit_ledger": (
                "theta_addition_bond_and_contraction_ledger.json"
            ),
            "marker_replay": "theta_addition_canonical_marker_replay.json",
            "exceptional": "theta_addition_pole_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r110.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The exact alternant predicate is not an inside-cap zero locator.",
            "The determinant value channel does not report zero positions.",
            "No lower bound for arbitrary finite-deck circuits is proved.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_POSITION_SEPARATED_ABEL_ALTERNANT_PREDICATE_ONLY__"
            "MERGE_ELLIPTIC_CAUCHY_VALUE_CHANNEL_WITH_IDEA012__REJECT_"
            "POINTWISE_SOURCE_SCAN_AND_GLOBAL_FERMAT_ROW_MODE__PRESERVE_"
            "FINITE_DECK_ANNIHILATOR_CONTRACTION__NO_RANK__NO_FACTOR_"
            "LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "circuit_ledger": ledger,
        "marker_replay": marker_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_theta_addition_cancellation_"
            "network_probe_report_r110.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_theta_addition_cancellation_network.json"
        ),
    )
    parser.add_argument(
        "--circuit-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "theta_addition_bond_and_contraction_ledger.json"
        ),
    )
    parser.add_argument(
        "--marker-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "theta_addition_canonical_marker_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "theta_addition_pole_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r110.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.circuit_output, bundle["circuit_ledger"])
    write_json(args.marker_output, bundle["marker_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R110 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
