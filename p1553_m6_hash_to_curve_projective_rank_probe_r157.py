#!/usr/bin/env python3
"""Transfer projective singleton rank controls to public curve points."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_hash_to_curve_projective_rank.r157.v1"

R156_PRODUCER = ROOT / "p1553_m6_a_diversity_projective_rank_probe_r156.py"
R156_REPORT = ROOT / (
    "p1553_m6_a_diversity_projective_rank_probe_report_r156.json"
)
R156_FROZEN = ROOT / "frozen_m6_a_diversity_projective_rank.json"
R156_COST = ROOT / "m6_a_diversity_projective_rank_cost_ledger.json"
R156_REPLAY = ROOT / "m6_a_diversity_projective_rank_replay.json"
R156_CONTROLS = ROOT / "m6_a_diversity_projective_rank_controls.json"
R156_LOGS = ROOT / "factor_logs_and_identical_descent_r156.json"
R156_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_a_diversity_projective_rank_probe_r156.py"
)
R156_GATE = ROOT / "p1553_m6_a_diversity_projective_rank_probe_gate_r156.md"
R156_PARENT = ROOT / (
    "p1553_m6_a_diversity_projective_rank_probe_parent_report_r156.yaml"
)

SOURCE_BINDINGS = (
    (
        "r156_producer",
        R156_PRODUCER,
        "b9caa2eb7bab40d3c3907111d1836130f3cc69394b0489445177996532f4a22a",
    ),
    (
        "r156_report",
        R156_REPORT,
        "8c04aa3e5640f661d9d2c30e24dcd769dfc4fa6b07c3947f57a83f75965414f3",
    ),
    (
        "r156_frozen",
        R156_FROZEN,
        "feb947fe5fcef1b7c016f11c1cd978882ab87358bbb855ac846f2934a4a41088",
    ),
    (
        "r156_cost",
        R156_COST,
        "db4ed01dace9c597b38686180406fc9392d7eedce1f09dd3b92e639e1d395f2a",
    ),
    (
        "r156_replay",
        R156_REPLAY,
        "eeb66740bcd2ad794082e780fcf0f80f1c23a4a40e15c0bc1a6dfe9ba99b4b4e",
    ),
    (
        "r156_controls",
        R156_CONTROLS,
        "82e745fb615edad055b00e24feb77f02ee45f33712e83ffe04e80e3f4c12d3ad",
    ),
    (
        "r156_logs",
        R156_LOGS,
        "fd84ab20b0d1fcee454e75c80f49fc7cf9d9b9dcf9f87e8eb6ece939ec18bc45",
    ),
    (
        "r156_test",
        R156_TEST,
        "f67774f461f02dea8b454467c8398e786991095fc3b1a5c8b995ff62fd58a99e",
    ),
    (
        "r156_gate",
        R156_GATE,
        "0d47ca3ee6534eb38287f8a611612d594d96f710c0b6cac675fdd2d7bc619bc1",
    ),
    (
        "r156_parent",
        R156_PARENT,
        "dbcc1e54ee782a39380b28dead9ae770c161be054f752ca687d79e295d4dde4d",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_hash_to_curve_projective_rank_probe_report_r157.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_hash_to_curve_projective_rank.json"
DEFAULT_COST = ROOT / "m6_hash_to_curve_projective_rank_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_hash_to_curve_projective_rank_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_hash_to_curve_projective_rank_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r157.json"

ARITY = 6
LOG_OVERSAMPLING_FACTORS = (2, 4, 8)
OFFSETS = (0, 1)


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R156 = load_module("p1553_r156_for_r157", R156_PRODUCER)
R155 = R156.R155
R154 = R156.R154
R153 = R156.R153
R82 = R153.R82
R81 = R153.R81
R70 = R81.R70
R144 = R153.R144
Point = tuple[int, int] | None


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R157 source binding mismatch: {failures}")
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


def point_sort_key(point: Point) -> tuple[int, int, int]:
    return (0, 0, 0) if point is None else (1, point[0], point[1])


def point_record(point: Point) -> str | list[int]:
    return "identity" if point is None else [point[0], point[1]]


def point_sum(
    points: Iterable[Point], curve: dict[str, Any]
) -> Point:
    result: Point = None
    for point in points:
        result = R70.add(result, point, curve)
    return result


def row_point(
    row: Iterable[int],
    representatives: tuple[Point, ...],
    curve: dict[str, Any],
) -> Point:
    order = int(curve["subgroup_order"])
    return point_sum(
        (
            R70.scalar_mul(coefficient % order, point, curve)
            for coefficient, point in zip(row, representatives)
        ),
        curve,
    )


def signed_vector(
    counts: tuple[int, ...], pair_count: int
) -> tuple[int, ...]:
    return tuple(
        counts[2 * index] - counts[2 * index + 1]
        for index in range(pair_count)
    )


def coefficient_fiber_size(
    vector: tuple[int, ...], pair_count: int
) -> int:
    weight = sum(abs(value) for value in vector)
    if weight > ARITY or (ARITY - weight) % 2:
        return 0
    cancellation_pairs = (ARITY - weight) // 2
    return math.comb(cancellation_pairs + pair_count - 1, pair_count - 1)


def projective_normalize_with_rhs(
    row: tuple[int, ...], rhs: int, modulus: int
) -> tuple[tuple[int, ...], int]:
    pivot = next(value for value in row if value % modulus)
    inverse = pow(pivot, -1, modulus)
    return (
        tuple(value * inverse % modulus for value in row),
        rhs * inverse % modulus,
    )


def choose_a_pair_count(
    c_pair_count: int,
    subgroup_order: int,
    log_factor: int,
) -> tuple[int, int, Fraction]:
    target = R156.occupancy_multiplier(c_pair_count, log_factor)
    max_c6 = R154.signed_coefficient_vector_count(
        c_pair_count, ARITY
    )
    pair_count = 1
    while True:
        max_a6 = R154.signed_coefficient_vector_count(
            pair_count, ARITY
        )
        nominal = Fraction(max_a6 * max_c6, subgroup_order)
        if nominal >= target:
            return pair_count, target, nominal
        pair_count += 1
        if pair_count > 32:
            raise AssertionError("unable to meet finite occupancy target")


def known_a_pairs(
    control_id: str,
    pair_count: int,
    generator: Point,
    curve: dict[str, Any],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[Point, Point], ...],
]:
    order = int(curve["subgroup_order"])
    scalar_pairs = R154.deterministic_signed_pairs(
        f"R157|{control_id}", pair_count, order
    )
    point_pairs = tuple(
        (
            R70.scalar_mul(positive, generator, curve),
            R70.scalar_mul(negative, generator, curve),
        )
        for positive, negative in scalar_pairs
    )
    if any(
        R70.negate(positive, curve) != negative
        for positive, negative in point_pairs
    ):
        raise AssertionError("known A point pairs lost inversion symmetry")
    return scalar_pairs, point_pairs


def c_point_pairs(
    curve: dict[str, Any], offset: int
) -> tuple[tuple[Point, Point], ...]:
    _, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    pairs = tuple(
        (point, R70.negate(point, curve)) for point in atoms_c
    )
    flat = [point for pair in pairs for point in pair]
    if None in flat or len(set(flat)) != len(flat):
        raise AssertionError("C inversion closure is not pairwise distinct")
    return pairs


def endpoint_structure(
    c_pairs: tuple[tuple[Point, Point], ...],
    curve: dict[str, Any],
) -> dict[str, Any]:
    labels = tuple(point for pair in c_pairs for point in pair)
    pair_count = len(c_pairs)
    endpoints: dict[
        Point, list[tuple[int, ...]]
    ] = collections.defaultdict(list)
    vectors: dict[
        tuple[int, ...], list[tuple[int, ...]]
    ] = collections.defaultdict(list)
    vector_points: dict[tuple[int, ...], Point] = {}
    vector_map_injective = True
    point_to_vector: dict[Point, tuple[int, ...]] = {}
    fiber_formula_exact = True

    for indices in itertools.combinations_with_replacement(
        range(len(labels)), ARITY
    ):
        counts = R155.multiplicity_vector(indices, len(labels))
        endpoint = point_sum((labels[index] for index in indices), curve)
        vector = signed_vector(counts, pair_count)
        endpoints[endpoint].append(counts)
        vectors[vector].append(counts)
        previous_point = vector_points.setdefault(vector, endpoint)
        if previous_point != endpoint:
            raise AssertionError("signed coefficient vector changed endpoint")
        previous_vector = point_to_vector.setdefault(endpoint, vector)
        vector_map_injective &= previous_vector == vector

    for vector, sources in vectors.items():
        fiber_formula_exact &= (
            len(sources) == coefficient_fiber_size(vector, pair_count)
        )

    singleton_endpoints = {
        endpoint: sources[0]
        for endpoint, sources in endpoints.items()
        if len(sources) == 1
    }
    singleton_l1_exact = all(
        sum(abs(value) for value in signed_vector(source, pair_count))
        == ARITY
        for source in singleton_endpoints.values()
    ) and all(
        (len(sources) == 1)
        == (
            sum(abs(value) for value in vector) == ARITY
            and vector_map_injective
        )
        for vector, sources in vectors.items()
    )

    return {
        "labels": labels,
        "endpoints": dict(endpoints),
        "vectors": dict(vectors),
        "singleton_endpoints": singleton_endpoints,
        "signed_vector_count": len(vectors),
        "endpoint_count": len(endpoints),
        "source_count": sum(len(sources) for sources in endpoints.values()),
        "signed_coefficient_map_injective": vector_map_injective,
        "coefficient_fiber_formula_exact": fiber_formula_exact,
        "singleton_iff_l1_six_exact": singleton_l1_exact,
        "singleton_endpoint_count": len(singleton_endpoints),
        "endpoint_multiplicity_histogram": dict(
            sorted(
                collections.Counter(
                    len(sources) for sources in endpoints.values()
                ).items()
            )
        ),
        "endpoint_sha256": sha256_json(
            [
                {
                    "point": point_record(endpoint),
                    "sources": sources,
                }
                for endpoint, sources in sorted(
                    endpoints.items(), key=lambda item: point_sort_key(item[0])
                )
            ]
        ),
    }


def a_shift_structure(
    control_id: str,
    pair_count: int,
    generator: Point,
    curve: dict[str, Any],
) -> dict[str, Any]:
    scalar_pairs, point_pairs = known_a_pairs(
        control_id, pair_count, generator, curve
    )
    scalars = tuple(value for pair in scalar_pairs for value in pair)
    points = tuple(point for pair in point_pairs for point in pair)
    order = int(curve["subgroup_order"])
    shifts: dict[Point, int] = {}
    multiplicities: collections.Counter[Point] = collections.Counter()
    scalar_consistent = True
    point_replay_exact = True

    for indices in itertools.combinations_with_replacement(
        range(len(points)), ARITY
    ):
        point = point_sum((points[index] for index in indices), curve)
        scalar = sum(scalars[index] for index in indices) % order
        previous = shifts.setdefault(point, scalar)
        scalar_consistent &= previous == scalar
        multiplicities[point] += 1
        point_replay_exact &= (
            R70.scalar_mul(scalar, generator, curve) == point
        )

    return {
        "scalar_pairs": scalar_pairs,
        "point_pairs": point_pairs,
        "shifts": shifts,
        "multiplicities": multiplicities,
        "shift_count": len(shifts),
        "source_count": sum(multiplicities.values()),
        "shift_scalars_consistent": scalar_consistent,
        "all_shift_points_replay_known_scalars": point_replay_exact,
        "shift_sha256": sha256_json(
            [
                {
                    "point": point_record(point),
                    "scalar": shifts[point],
                    "multiplicity": multiplicities[point],
                }
                for point in sorted(shifts, key=point_sort_key)
            ]
        ),
    }


def hash_to_curve_control(
    curve: dict[str, Any],
    offset: int,
    log_factor: int,
) -> dict[str, Any]:
    order = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    c_pairs = c_point_pairs(curve, offset)
    representatives = tuple(positive for positive, _ in c_pairs)
    c_pair_count = len(c_pairs)
    a_pair_count, target_multiplier, nominal_occupancy = (
        choose_a_pair_count(c_pair_count, order, log_factor)
    )
    control_id = (
        f"{curve['family_id']}_offset{offset}_"
        f"log{log_factor}_a{a_pair_count}"
    )
    c_data = endpoint_structure(c_pairs, curve)
    a_data = a_shift_structure(
        control_id, a_pair_count, generator, curve
    )
    c_labels = c_data["labels"]
    endpoint_sources = c_data["endpoints"]
    label_to_index = {
        point: index for index, point in enumerate(c_labels)
    }

    rows: list[tuple[int, ...]] = []
    rhs_values: list[int] = []
    supports: list[set[int]] = []
    relation_records: dict[
        tuple[Point, int], tuple[int, ...]
    ] = {}
    relation_identities_exact = True
    positive_rows = 0
    multi_source_rows = 0

    for shift in sorted(a_data["shifts"], key=point_sort_key):
        rhs = a_data["shifts"][shift]
        for target_index, target in enumerate(c_labels):
            endpoint = R70.add(shift, target, curve)
            sources = endpoint_sources.get(endpoint, [])
            if not sources:
                continue
            positive_rows += 1
            if len(sources) != 1:
                multi_source_rows += 1
                continue
            full = list(sources[0])
            full[target_index] -= 1
            row = tuple(
                full[2 * index] - full[2 * index + 1]
                for index in range(c_pair_count)
            )
            if not any(row):
                continue
            relation_identities_exact &= (
                row_point(row, representatives, curve) == shift
            )
            row_mod = tuple(value % order for value in row)
            rows.append(row_mod)
            rhs_values.append(rhs)
            supports.append(
                {index for index, value in enumerate(row) if value}
            )
            relation_records[(shift, target_index)] = row_mod

    opposite_rows_exact = True
    for (shift, target_index), row in relation_records.items():
        opposite_shift = R70.negate(shift, curve)
        opposite_target = R70.negate(c_labels[target_index], curve)
        opposite_target_index = label_to_index[opposite_target]
        opposite_rows_exact &= relation_records.get(
            (opposite_shift, opposite_target_index)
        ) == tuple((-value) % order for value in row)

    projective_rhs: dict[tuple[int, ...], int] = {}
    projective_rhs_consistent = True
    for row, rhs in zip(rows, rhs_values):
        normalized, normalized_rhs = projective_normalize_with_rhs(
            row, rhs, order
        )
        previous = projective_rhs.setdefault(normalized, normalized_rhs)
        projective_rhs_consistent &= previous == normalized_rhs

    projective_rows = sorted(projective_rhs)
    projective_values = [projective_rhs[row] for row in projective_rows]
    rank = R81.rank_mod(projective_rows, order)
    full_rank = rank == c_pair_count
    covered = set().union(*supports) if supports else set()
    selected_rows, selected_rhs = R153.select_independent_system(
        [list(row) for row in projective_rows],
        projective_values,
        c_pair_count,
        order,
    )
    recovered_logs: list[int] | None = None
    recovered_logs_verify = False
    if full_rank:
        recovered_logs = R144.solve_square_mod(
            selected_rows, selected_rhs, order
        )
        recovered_logs_verify = all(
            R70.scalar_mul(logarithm, generator, curve) == point
            for logarithm, point in zip(
                recovered_logs, representatives
            )
        )

    projective_count_deficit = max(
        0, c_pair_count - len(projective_rows)
    )
    dependency_nullity = max(
        0, min(c_pair_count, len(projective_rows)) - rank
    )
    realized_occupancy = Fraction(
        a_data["shift_count"] * c_data["endpoint_count"], order
    )
    actual_relation_supply = Fraction(len(projective_rows), c_pair_count)

    return {
        "control_id": control_id,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": order,
        "c_pair_count": c_pair_count,
        "signed_log_dimension": c_pair_count,
        "log_oversampling_factor": log_factor,
        "target_occupancy_multiplier": target_multiplier,
        "selected_a_pair_count": a_pair_count,
        "nominal_max_support_occupancy": fraction_record(
            nominal_occupancy
        ),
        "realized_support_occupancy": fraction_record(
            realized_occupancy
        ),
        "actual_relation_supply_per_dimension": fraction_record(
            actual_relation_supply
        ),
        "a6_shift_support_size": a_data["shift_count"],
        "a6_unordered_source_count": a_data["source_count"],
        "a6_shift_scalars_consistent": a_data[
            "shift_scalars_consistent"
        ],
        "all_a6_shift_points_replay_known_scalars": a_data[
            "all_shift_points_replay_known_scalars"
        ],
        "a6_shift_sha256": a_data["shift_sha256"],
        "c6_signed_coefficient_vector_count": c_data[
            "signed_vector_count"
        ],
        "c6_endpoint_support_size": c_data["endpoint_count"],
        "c6_unordered_source_count": c_data["source_count"],
        "c6_signed_coefficient_map_injective": c_data[
            "signed_coefficient_map_injective"
        ],
        "c6_coefficient_fiber_formula_exact": c_data[
            "coefficient_fiber_formula_exact"
        ],
        "c6_singleton_iff_l1_six_exact": c_data[
            "singleton_iff_l1_six_exact"
        ],
        "c6_singleton_endpoint_count": c_data[
            "singleton_endpoint_count"
        ],
        "c6_endpoint_multiplicity_histogram": c_data[
            "endpoint_multiplicity_histogram"
        ],
        "c6_endpoint_sha256": c_data["endpoint_sha256"],
        "positive_relation_row_count": positive_rows,
        "multi_source_relation_row_count": multi_source_rows,
        "singleton_nonzero_row_count": len(rows),
        "projectively_distinct_row_count": len(projective_rows),
        "covered_column_count": len(covered),
        "uncovered_column_count": c_pair_count - len(covered),
        "projective_count_deficit": projective_count_deficit,
        "dependency_nullity_after_projective_dedup": dependency_nullity,
        "signed_rank": rank,
        "signed_full_rank": full_rank,
        "all_public_group_relation_identities_exact": (
            relation_identities_exact
        ),
        "all_opposite_rows_exact": opposite_rows_exact,
        "projective_duplicate_rhs_consistent": projective_rhs_consistent,
        "selected_system_size": len(selected_rows),
        "factor_logs_computed_without_dlp_oracle": full_rank,
        "factor_logs_verified_by_public_scalar_multiplication": (
            recovered_logs_verify
        ),
        "recovered_factor_logs_sha256": (
            sha256_json(recovered_logs)
            if recovered_logs is not None
            else None
        ),
        "projective_rows_sha256": sha256_json(projective_rows),
        "projective_rhs_sha256": sha256_json(projective_values),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_count_marginal_rank_or_source_oracle_consumed": (
            False
        ),
        "verifier_bsgs_labels_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "coefficient_fiber_formula": (
            "For a signed C6 coefficient vector v, every unordered source "
            "with that vector is obtained by distributing "
            "k=(6-||v||_1)/2 cancellation pairs among d inversion pairs. "
            "Its exact fiber size is C(k+d-1,d-1)."
        ),
        "singleton_criterion": (
            "If the map v -> sum_j v_j C_j is injective on feasible signed "
            "C6 vectors, a C6 endpoint has exactly one unordered source if "
            "and only if ||v||_1=6."
        ),
        "public_group_relation": (
            "For singleton source v and signed target sigma*C_j, the row "
            "r=v-sigma*e_j is obtained from public multiplicities and "
            "satisfies sum_j r_j C_j=S exactly when the public endpoint "
            "equals S+sigma*C_j. No scalar label is needed to discover or "
            "form the row."
        ),
        "factor_log_boundary": (
            "Known-log A points supply the public right-hand side S. If the "
            "integer row matrix has full rank modulo the prime subgroup "
            "order, solving it yields C logs that are verified only by "
            "public scalar multiplication."
        ),
        "scope": (
            "The finite curve controls prove implementation-level transfer "
            "from scalar labels to public group equality. They do not prove "
            "asymptotic injectivity, relation concentration, or full rank "
            "for a generic hash-to-curve family."
        ),
        "novelty_status": (
            "public_group_l1_singleton_factor_log_transfer_novelty_unverified"
        ),
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_hash_to_curve_projective_rank.cost.r157.v1",
        "known_a_pair_count_exponent_B": fraction_record(Fraction(1, 12)),
        "a6_shift_support_exponent_B": fraction_record(Fraction(1, 2)),
        "signed_c_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "relation_count_exponent_B": fraction_record(Fraction(3, 4)),
        "relation_count_polylog_factor": "log(B)",
        "reverse_row_batch_exponent_B": fraction_record(Fraction(5, 4)),
        "reverse_row_batch_polylog_factor": "log(B)",
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "explicit_c6_endpoint_enumeration_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "explicit_control_total_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "explicit_control_exceeds_pollard_rho": True,
        "public_point_equality_changes_no_selected_exponent": True,
        "finite_explicit_enumeration_receives_attack_credit": False,
        "asymptotic_hash_to_curve_injectivity_supplied": False,
        "asymptotic_short_relation_rank_theorem_supplied": False,
        "reverse_only_signed_marker_operator_supplied": False,
        "identical_target_descent_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls_list = [
        hash_to_curve_control(curve, offset, log_factor)
        for curve in R82.FAMILIES
        for offset in OFFSETS
        for log_factor in LOG_OVERSAMPLING_FACTORS
    ]
    all_public_exact = all(
        row["all_public_group_relation_identities_exact"]
        and row["all_opposite_rows_exact"]
        and row["projective_duplicate_rhs_consistent"]
        for row in controls_list
    )
    all_singleton_theory = all(
        row["c6_signed_coefficient_map_injective"]
        and row["c6_coefficient_fiber_formula_exact"]
        and row["c6_singleton_iff_l1_six_exact"]
        for row in controls_list
    )
    all_no_oracles = all(
        not row["candidate_discrete_log_oracle_consumed"]
        and not row[
            "candidate_root_count_marginal_rank_or_source_oracle_consumed"
        ]
        and not row["verifier_bsgs_labels_consumed"]
        for row in controls_list
    )
    full_count = sum(row["signed_full_rank"] for row in controls_list)
    factor_log_count = sum(
        row["factor_logs_verified_by_public_scalar_multiplication"]
        for row in controls_list
    )
    full_by_log_factor = [
        sum(
            row["signed_full_rank"]
            for row in controls_list
            if row["log_oversampling_factor"] == factor
        )
        for factor in LOG_OVERSAMPLING_FACTORS
    ]
    full_by_c_pair_count = [
        {
            "c_pair_count": pair_count,
            "control_count": sum(
                row["c_pair_count"] == pair_count
                for row in controls_list
            ),
            "full_rank_control_count": sum(
                row["signed_full_rank"]
                for row in controls_list
                if row["c_pair_count"] == pair_count
            ),
        }
        for pair_count in sorted(
            {row["c_pair_count"] for row in controls_list}
        )
    ]
    controls = {
        "schema": (
            "p1553.m6_hash_to_curve_projective_rank.controls.r157.v1"
        ),
        "control_count": len(controls_list),
        "log_oversampling_factors": list(LOG_OVERSAMPLING_FACTORS),
        "offsets": list(OFFSETS),
        "all_c6_signed_coefficient_maps_injective": all(
            row["c6_signed_coefficient_map_injective"]
            for row in controls_list
        ),
        "all_coefficient_fiber_formulas_exact": all(
            row["c6_coefficient_fiber_formula_exact"]
            for row in controls_list
        ),
        "all_singleton_iff_l1_six_criteria_exact": all(
            row["c6_singleton_iff_l1_six_exact"]
            for row in controls_list
        ),
        "all_public_group_relations_and_opposites_exact": all_public_exact,
        "all_candidate_oracles_avoided": all_no_oracles,
        "full_rank_control_count": full_count,
        "factor_log_control_count": factor_log_count,
        "full_rank_counts_by_log_factor": full_by_log_factor,
        "full_rank_counts_by_c_pair_count": full_by_c_pair_count,
        "finite_controls_receive_asymptotic_credit": False,
        "controls": controls_list,
    }
    theorem = theorem_record()
    cost = cost_ledger()
    frozen = {
        "schema": (
            "p1553.m6_hash_to_curve_projective_rank.frozen.r157.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "arity": ARITY,
        "log_oversampling_factors": list(LOG_OVERSAMPLING_FACTORS),
        "offsets": list(OFFSETS),
        "occupancy_selection_rule": (
            "smallest A inversion-pair count whose max A6 support times "
            "max C6 support divided by subgroup order is at least "
            "ceil(log_factor*ln(C_pair_count))"
        ),
        "theorem": theorem,
        "cost": cost,
        "open_obligations": {
            "asymptotic_hash_to_curve_coefficient_map_injectivity": "open",
            "asymptotic_short_relation_rank_theorem": "open",
            "reverse_only_signed_marker_operator": "open",
            "signed_weight_separable_ffe_dag": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "pollard_rho_improvement": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
    }
    replay = {
        "schema": (
            "p1553.m6_hash_to_curve_projective_rank.replay.r157.v1"
        ),
        "controls": controls_list,
        "all_public_group_relations_and_opposites_exact": all_public_exact,
        "all_singleton_theory_exact_on_controls": all_singleton_theory,
        "all_candidate_oracles_avoided": all_no_oracles,
        "full_rank_control_count": full_count,
        "factor_log_control_count": factor_log_count,
        "breakthrough": False,
    }
    logs = {
        "schema": (
            "p1553.m6_hash_to_curve_projective_rank.logs.r157.v1"
        ),
        "factor_log_controls_computed_without_dlp_oracle": full_count,
        "factor_log_controls_verified_by_public_scalar_multiplication": (
            factor_log_count
        ),
        "all_full_rank_factor_log_controls_verified": (
            full_count == factor_log_count
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_bsgs_labels_consumed": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_transfer_complete": False,
        "finite_factor_logs_receive_attack_credit": False,
    }

    obligations = {
        "r156_projective_transition_inherited": True,
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "coefficient_fiber_formula_derived": True,
        "singleton_iff_l1_six_theorem_derived": True,
        "public_group_relation_formula_derived": True,
        "twenty_four_hash_to_curve_controls_complete": (
            len(controls_list) == 24
        ),
        "all_actual_signed_coefficient_maps_injective": all_singleton_theory,
        "all_public_group_relation_identities_exact": all_public_exact,
        "all_candidate_oracles_avoided": all_no_oracles,
        "finite_full_rank_transition_observed": full_count > 0,
        "factor_logs_computed_without_dlp_oracle": factor_log_count > 0,
        "all_full_rank_factor_logs_publicly_verified": (
            full_count == factor_log_count
        ),
        "finite_results_scoped_without_asymptotic_credit": True,
        "logarithmic_oversampling_charged": True,
        "explicit_c6_enumeration_B9O2_charged": True,
        "asymptotic_hash_to_curve_injectivity_complete": False,
        "asymptotic_short_relation_rank_theorem_complete": False,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(bool(value) for value in obligations.values())
    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "objective": (
            "Transfer the R156 projective singleton rank experiment from "
            "known scalar labels to public hash-to-curve group elements."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "classification": (
            "L1_SINGLETON_FIBER_THEOREM__PUBLIC_GROUP_POINT_EQUALITY_"
            "RELATION_DISCOVERY__HASH_TO_CURVE_FINITE_RANK_AND_FACTOR_LOG_"
            "TRANSFER__NO_DLP_OR_LABEL_ORACLE__ASYMPTOTIC_INJECTIVITY_RANK_"
            "REVERSE_FFE_AND_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH"
        ),
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "l1_singleton_theorem_admitted": True,
            "finite_public_group_rank_and_factor_logs_admitted": (
                full_count > 0 and full_count == factor_log_count
            ),
            "asymptotic_hash_to_curve_rank_admitted": False,
            "reverse_only_signed_marker_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": (
            "Prove a generic-prime high-probability injectivity and full-rank "
            "theorem for the public short-relation coefficient system, then "
            "instantiate the reverse signed FFE operator and identical "
            "target descent within the frozen caps."
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_count_marginal_rank_or_source_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    controls = bundle["controls"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"full={controls['full_rank_control_count']}/"
        f"{controls['control_count']} "
        f"logs={controls['factor_log_control_count']} "
        f"by_log={controls['full_rank_counts_by_log_factor']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
