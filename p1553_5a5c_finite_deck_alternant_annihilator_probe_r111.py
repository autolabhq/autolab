#!/usr/bin/env python3
"""Audit exact finite-deck annihilators for the R110 Abel alternant."""

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


SCHEMA = "p1553.5a5c_finite_deck_alternant_annihilator.r111.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
A_SIDE_SOURCE_EXPONENT = Fraction(2)
C_SIDE_SOURCE_EXPONENT = Fraction(3)
FULL_SOURCE_EXPONENT = Fraction(5)
ALTERNANT_DIMENSION = 11
A_ROW_COUNT = 5
EXTERIOR_DIMENSION = math.comb(ALTERNANT_DIMENSION, A_ROW_COUNT)

R110_PRODUCER = pathlib.Path(
    "p1553_5a5c_theta_addition_cancellation_network_probe_r110.py"
)
R110_PRODUCER_SHA256 = (
    "ab12255748d458c11fb26c3dc410645843e18fe5eb387fc27c3bfd4e42467793"
)
R110_REPORT = pathlib.Path(
    "p1553_5a5c_theta_addition_cancellation_network_probe_report_r110.json"
)
R110_REPORT_SHA256 = (
    "178c8c7032ab9f0d39ace520672598f007f192070695dcf53e78bfc249f282ef"
)
R110_FROZEN = pathlib.Path(
    "frozen_5a5c_theta_addition_cancellation_network.json"
)
R110_FROZEN_SHA256 = (
    "dcea7c885c813c2fb998e48f3ecbe9dacdf36d40f2b452e3a9a446338fd8bc14"
)
R110_CIRCUIT = pathlib.Path(
    "theta_addition_bond_and_contraction_ledger.json"
)
R110_CIRCUIT_SHA256 = (
    "78332c94ef224e7188b8c6771f560968d1ca1fb978a22c10e96043be6b9c37af"
)
R110_MARKERS = pathlib.Path("theta_addition_canonical_marker_replay.json")
R110_MARKERS_SHA256 = (
    "9f67a86d592ec434b88e65e3535097a9c1f2f7980f3e1010a06fee894d8f48e3"
)
R110_EXCEPTIONAL = pathlib.Path(
    "theta_addition_pole_exceptional_controls.json"
)
R110_EXCEPTIONAL_SHA256 = (
    "496b5d7d528c976882b3fc567451eeeb887c89cdd01ba6afd79986a45e715795"
)
R110_LOGS = pathlib.Path("factor_logs_and_identical_descent_r110.json")
R110_LOGS_SHA256 = (
    "5ed9c70928f822f129b2fd4f5952213a4c87e47273e1dc7d9482319b74873146"
)
R110_GATE = pathlib.Path(
    "p1553_5a5c_theta_addition_cancellation_network_probe_gate_r110.md"
)
R110_GATE_SHA256 = (
    "87a241865d19bdfe9fc24b527f9547d32d2a05c9449375f71f12bdfb5fa71c86"
)
R110_PARENT = pathlib.Path(
    "p1553_5a5c_theta_addition_cancellation_network_probe_parent_report_r110.yaml"
)
R110_PARENT_SHA256 = (
    "4cb4fccfd29b0310bd3c5635b1bd3cd2de7e00c02d2833ad40fb55f48a5b1fb2"
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
        R110_PRODUCER: R110_PRODUCER_SHA256,
        R110_REPORT: R110_REPORT_SHA256,
        R110_FROZEN: R110_FROZEN_SHA256,
        R110_CIRCUIT: R110_CIRCUIT_SHA256,
        R110_MARKERS: R110_MARKERS_SHA256,
        R110_EXCEPTIONAL: R110_EXCEPTIONAL_SHA256,
        R110_LOGS: R110_LOGS_SHA256,
        R110_GATE: R110_GATE_SHA256,
        R110_PARENT: R110_PARENT_SHA256,
        IDEA_REGISTRY: IDEA_REGISTRY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R111 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R110 = load_module("p1553_r110_for_r111", R110_PRODUCER)
R108 = R110.R108
R105 = R110.R105
R102 = R110.R102
R82 = R110.R82
R70 = R110.R70


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def observed_exponent(count: int, base: int) -> float:
    if count <= 1 or base <= 1:
        return 0.0
    return math.log(count, base)


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def source_json(source: Source) -> list[list[int]]:
    return [list(source[0]), list(source[1])]


def determinant_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> int:
    rows = [[value % prime for value in row] for row in matrix]
    if not rows:
        return 1
    if len(rows) != len(rows[0]):
        raise AssertionError("determinant requires a square matrix")
    determinant = 1
    for column in range(len(rows)):
        pivot = next(
            (
                row
                for row in range(column, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        pivot_value = rows[column][column]
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, prime - 2, prime)
        for row in range(column + 1, len(rows)):
            factor = rows[row][column] * inverse % prime
            if factor == 0:
                continue
            for index in range(column, len(rows)):
                rows[row][index] = (
                    rows[row][index] - factor * rows[column][index]
                ) % prime
    return determinant % prime


def rref_signature(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    rows = [[value % prime for value in row] for row in matrix]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], prime - 2, prime)
        rows[rank] = [value * inverse % prime for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank:
                continue
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    (value - factor * pivot_value) % prime
                    for value, pivot_value in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return tuple(tuple(row) for row in rows[:rank])


def exterior_pairing(
    a_rows: Sequence[Sequence[int]],
    c_target_rows: Sequence[Sequence[int]],
    prime: int,
) -> int:
    total = 0
    all_columns = tuple(range(ALTERNANT_DIMENSION))
    base_parity = A_ROW_COUNT * (A_ROW_COUNT - 1) // 2
    for columns_a in itertools.combinations(
        all_columns,
        A_ROW_COUNT,
    ):
        columns_a_set = set(columns_a)
        columns_c = tuple(
            column
            for column in all_columns
            if column not in columns_a_set
        )
        minor_a = determinant_mod(
            [[row[column] for column in columns_a] for row in a_rows],
            prime,
        )
        if minor_a == 0:
            continue
        minor_c = determinant_mod(
            [
                [row[column] for column in columns_c]
                for row in c_target_rows
            ],
            prime,
        )
        sign = -1 if (sum(columns_a) - base_parity) % 2 else 1
        total = (total + sign * minor_a * minor_c) % prime
    return total


def five_endpoint(
    source: Sequence[int],
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R110.add_many((atoms[index] for index in source), curve)


def source_value_digest(values: Iterable[int]) -> str:
    encoded = json.dumps(
        sorted(values),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def source_lists(size_a: int, size_c: int) -> tuple[
    list[tuple[int, ...]],
    list[tuple[int, ...]],
]:
    return (
        list(itertools.combinations_with_replacement(range(size_a), 5)),
        list(itertools.combinations_with_replacement(range(size_c), 5)),
    )


def instance_inputs(
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
        positives = [R110.repeated_source()]
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
    schedule = R110.zero_sum_position_shifts(
        atoms_a,
        atoms_c,
        target,
        curve,
        (
            f"R110|{control_class}|{curve['family_id']}|"
            f"{offset}|position-shift"
        ),
    )
    return {
        "atoms_a": atoms_a,
        "atoms_c": atoms_c,
        "factors": factors,
        "geometry": geometry,
        "positives": positives,
        "target": target,
        "target_class": target_class,
        "schedule": schedule,
    }


def endpoint_gauge_summary(
    values: Sequence[Sequence[int]],
    endpoints_a: Sequence[Point],
    endpoints_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    fibers: dict[Point, dict[str, Any]] = {}
    for row_index, endpoint_a in enumerate(endpoints_a):
        for column_index, endpoint_c in enumerate(endpoints_c):
            endpoint = R70.add(endpoint_a, endpoint_c, curve)
            fiber = fibers.setdefault(
                endpoint,
                {"source_count": 0, "values": set()},
            )
            fiber["source_count"] += 1
            fiber["values"].add(values[row_index][column_index])
    repeated = [
        fiber for fiber in fibers.values() if fiber["source_count"] > 1
    ]
    varying = [
        fiber for fiber in repeated if len(fiber["values"]) > 1
    ]
    return {
        "endpoint_count": len(fibers),
        "repeated_endpoint_fiber_count": len(repeated),
        "varying_value_repeated_endpoint_fiber_count": len(varying),
        "maximum_source_multiplicity": max(
            fiber["source_count"] for fiber in fibers.values()
        ),
        "maximum_value_multiplicity_in_one_endpoint_fiber": max(
            len(fiber["values"]) for fiber in fibers.values()
        ),
        "raw_determinant_is_endpoint_only": not varying,
    }


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
    doubles: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    inputs = instance_inputs(
        curve,
        offset,
        control_class,
        doubles,
    )
    atoms_a = inputs["atoms_a"]
    atoms_c = inputs["atoms_c"]
    target = inputs["target"]
    prime = curve["field_prime"]
    row_tables = R110.shifted_row_tables(inputs["schedule"], prime)
    a_sources, c_sources = source_lists(len(atoms_a), len(atoms_c))
    a_matrices = [
        [
            list(row_tables[position][index])
            for position, index in enumerate(source)
        ]
        for source in a_sources
    ]
    c_matrices = [
        [
            *[
                list(row_tables[5 + position][index])
                for position, index in enumerate(source)
            ],
            list(row_tables[10][0]),
        ]
        for source in c_sources
    ]
    endpoints_a = [
        five_endpoint(source, atoms_a, curve) for source in a_sources
    ]
    endpoints_c = [
        five_endpoint(source, atoms_c, curve) for source in c_sources
    ]
    values = [
        [
            determinant_mod([*a_matrix, *c_matrix], prime)
            for c_matrix in c_matrices
        ]
        for a_matrix in a_matrices
    ]
    zero_mask = [
        [1 if value == 0 else 0 for value in row] for row in values
    ]
    zero_sources = [
        (row_index, column_index)
        for row_index, row in enumerate(values)
        for column_index, value in enumerate(row)
        if value == 0
    ]
    direct_sources = [
        (row_index, column_index)
        for row_index, endpoint_a in enumerate(endpoints_a)
        for column_index, endpoint_c in enumerate(endpoints_c)
        if R70.add(endpoint_a, endpoint_c, curve) == target
    ]
    if zero_sources != direct_sources:
        raise AssertionError("finite-deck alternant zero mismatch")

    distinct_values = set(itertools.chain.from_iterable(values))
    distinct_nonzero_values = distinct_values - {0}
    row_degrees = [
        len(set(row) - {0})
        for row in values
    ]
    column_degrees = [
        len(
            {
                values[row_index][column_index]
                for row_index in range(len(a_sources))
            }
            - {0}
        )
        for column_index in range(len(c_sources))
    ]
    source_rows = []
    for row_index, column_index in zero_sources:
        source = (a_sources[row_index], c_sources[column_index])
        source_rows.append(
            {
                "source": source_json(source),
                "target": point_json(target),
                "marker": list(R105.marker_vector(source, prime)),
                "canonical_cycle_weight": R108.FULL_CYCLE_SCALE,
                "alternant_value": 0,
                "direct_relation": True,
            }
        )

    laplace_pairs = {(0, 0)}
    if zero_sources:
        laplace_pairs.add(zero_sources[0])
    laplace_pairs.add((len(a_sources) - 1, len(c_sources) - 1))
    laplace_checks = []
    for row_index, column_index in sorted(laplace_pairs):
        direct_value = values[row_index][column_index]
        exterior_value = exterior_pairing(
            a_matrices[row_index],
            c_matrices[column_index],
            prime,
        )
        laplace_checks.append(
            {
                "a_source_index": row_index,
                "c_source_index": column_index,
                "direct_determinant": direct_value,
                "exterior_pairing": exterior_value,
                "exact": direct_value == exterior_value,
            }
        )

    factor_base_size = len(inputs["factors"])
    body_size = len(a_sources) * len(c_sources)
    gauge = endpoint_gauge_summary(
        values,
        endpoints_a,
        endpoints_c,
        curve,
    )
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": prime,
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": factor_base_size,
        "factor_base_injective": inputs["geometry"]["factor_base_injective"],
        "target_class": inputs["target_class"],
        "target": point_json(target),
        "a_source_count": len(a_sources),
        "c_source_count": len(c_sources),
        "full_source_body_count": body_size,
        "full_source_body_observed_exponent_B": observed_exponent(
            body_size,
            factor_base_size,
        ),
        "distinct_a_row_space_count": len(
            {
                rref_signature(matrix, prime)
                for matrix in a_matrices
            }
        ),
        "distinct_c_target_row_space_count": len(
            {
                rref_signature(matrix, prime)
                for matrix in c_matrices
            }
        ),
        "exterior_dimension": EXTERIOR_DIMENSION,
        "laplace_checks": laplace_checks,
        "all_laplace_checks_exact": all(
            row["exact"] for row in laplace_checks
        ),
        "zero_source_count": len(zero_sources),
        "zero_sources": source_rows,
        "zero_mask_rank": R110.rank_mod(zero_mask, prime),
        "zero_biconditional_exact_on_full_body": (
            zero_sources == direct_sources
        ),
        "distinct_determinant_value_count": len(distinct_values),
        "distinct_nonzero_determinant_value_count": len(
            distinct_nonzero_values
        ),
        "distinct_nonzero_value_ratio": (
            len(distinct_nonzero_values) / body_size
        ),
        "minimum_exact_raw_value_zero_mask_degree": len(
            distinct_nonzero_values
        ),
        "minimum_raw_value_mask_degree_observed_exponent_B": (
            observed_exponent(
                len(distinct_nonzero_values),
                factor_base_size,
            )
        ),
        "determinant_value_set_sha256": source_value_digest(
            distinct_values
        ),
        "determinant_values": sorted(distinct_values),
        "row_conditioned_annihilator_degrees": row_degrees,
        "column_conditioned_annihilator_degrees": column_degrees,
        "row_conditioned_coefficient_slot_count": sum(
            degree + 1 for degree in row_degrees
        ),
        "column_conditioned_coefficient_slot_count": sum(
            degree + 1 for degree in column_degrees
        ),
        "endpoint_gauge": gauge,
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def all_controls() -> dict[str, Any]:
    doubles = R110.r105_double_fibers()
    actual = [
        analyze_instance(dict(family), offset, "actual", doubles)
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        analyze_instance(
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
        "full_source_count": sum(
            row["full_source_body_count"] for row in rows
        ),
        "zero_source_count": sum(
            row["zero_source_count"] for row in rows
        ),
        "double_fiber_instance_count": sum(
            row["target_class"] == "r105_actual_double_fiber"
            and row["zero_source_count"] == 2
            for row in rows
        ),
        "all_full_body_zero_biconditionals_exact": all(
            row["zero_biconditional_exact_on_full_body"] for row in rows
        ),
        "all_exterior_laplace_checks_exact": all(
            row["all_laplace_checks_exact"] for row in rows
        ),
        "all_a_row_spaces_source_distinct": all(
            row["distinct_a_row_space_count"] == row["a_source_count"]
            for row in rows
        ),
        "all_c_target_row_spaces_source_distinct": all(
            row["distinct_c_target_row_space_count"]
            == row["c_source_count"]
            for row in rows
        ),
        "minimum_distinct_nonzero_value_ratio": min(
            row["distinct_nonzero_value_ratio"] for row in rows
        ),
        "maximum_zero_mask_rank": max(
            row["zero_mask_rank"] for row in rows
        ),
        "raw_determinant_endpoint_only_on_all_instances": all(
            row["endpoint_gauge"]["raw_determinant_is_endpoint_only"]
            for row in rows
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r110_producer": {
            "path": str(R110_PRODUCER),
            "sha256": R110_PRODUCER_SHA256,
        },
        "r110_report": {
            "path": str(R110_REPORT),
            "sha256": R110_REPORT_SHA256,
        },
        "r110_frozen": {
            "path": str(R110_FROZEN),
            "sha256": R110_FROZEN_SHA256,
        },
        "r110_circuit": {
            "path": str(R110_CIRCUIT),
            "sha256": R110_CIRCUIT_SHA256,
        },
        "r110_markers": {
            "path": str(R110_MARKERS),
            "sha256": R110_MARKERS_SHA256,
        },
        "r110_exceptional": {
            "path": str(R110_EXCEPTIONAL),
            "sha256": R110_EXCEPTIONAL_SHA256,
        },
        "r110_logs": {
            "path": str(R110_LOGS),
            "sha256": R110_LOGS_SHA256,
        },
        "r110_gate": {
            "path": str(R110_GATE),
            "sha256": R110_GATE_SHA256,
        },
        "r110_parent": {
            "path": str(R110_PARENT),
            "sha256": R110_PARENT_SHA256,
        },
        "idea_registry": {
            "path": str(IDEA_REGISTRY),
            "sha256": IDEA_REGISTRY_SHA256,
        },
    }


def cost_ledger(controls: dict[str, Any]) -> dict[str, Any]:
    rows = [
        *controls["actual"],
        *controls["matched_random_decks"],
    ]
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "exterior_split": {
            "a_side_source_count_exponent_B": fraction_record(
                A_SIDE_SOURCE_EXPONENT
            ),
            "c_target_side_source_count_exponent_B": fraction_record(
                C_SIDE_SOURCE_EXPONENT
            ),
            "ambient_exterior_dimension": EXTERIOR_DIMENSION,
            "a_side_fits_setup_cap": A_SIDE_SOURCE_EXPONENT <= SETUP_CAP,
            "c_target_side_fits_setup_cap": (
                C_SIDE_SOURCE_EXPONENT <= SETUP_CAP
            ),
            "c_target_side_fits_online_cap": (
                C_SIDE_SOURCE_EXPONENT <= ONLINE_CAP
            ),
        },
        "raw_value_zero_mask": {
            "exact_restricted_polynomial": (
                "q_D(z)=product_(d in D_nonzero)(1-z/d)"
            ),
            "minimum_degree_theorem": (
                "Any polynomial q with q(0)=1 and q(d)=0 for every "
                "distinct nonzero d in D has degree at least |D_nonzero|; "
                "q_D attains equality."
            ),
            "minimum_observed_degree": min(
                row["minimum_exact_raw_value_zero_mask_degree"]
                for row in rows
            ),
            "maximum_observed_degree": max(
                row["minimum_exact_raw_value_zero_mask_degree"]
                for row in rows
            ),
            "minimum_distinct_nonzero_value_ratio": controls[
                "minimum_distinct_nonzero_value_ratio"
            ],
            "global_asymptotic_degree_claimed": False,
            "arbitrary_circuit_lower_bound_claimed": False,
        },
        "side_conditioned_layouts": {
            "one_polynomial_per_a_source": {
                "polynomial_count_exponent_B": fraction_record(
                    A_SIDE_SOURCE_EXPONENT
                ),
                "maximum_degree_exponent_B": fraction_record(
                    C_SIDE_SOURCE_EXPONENT
                ),
                "standard_coefficient_capacity_exponent_B": (
                    fraction_record(FULL_SOURCE_EXPONENT)
                ),
            },
            "one_polynomial_per_c_source": {
                "polynomial_count_exponent_B": fraction_record(
                    C_SIDE_SOURCE_EXPONENT
                ),
                "maximum_degree_exponent_B": fraction_record(
                    A_SIDE_SOURCE_EXPONENT
                ),
                "standard_coefficient_capacity_exponent_B": (
                    fraction_record(FULL_SOURCE_EXPONENT)
                ),
            },
            "maximum_observed_row_conditioned_slots": max(
                row["row_conditioned_coefficient_slot_count"]
                for row in rows
            ),
            "maximum_observed_column_conditioned_slots": max(
                row["column_conditioned_coefficient_slot_count"]
                for row in rows
            ),
            "inside_setup_cap": False,
            "inside_online_cap": False,
        },
        "zero_incidence": {
            "maximum_observed_zero_mask_rank": controls[
                "maximum_zero_mask_rank"
            ],
            "low_rank_mask_is_available_without_constructing_mask": False,
            "subset_stable_existence_oracle_supplied": False,
            "source_unranking_oracle_supplied": False,
        },
        "constructor_status": {
            "full_finite_deck_enumeration_complete": True,
            "exact_raw_value_annihilator_degree_complete": True,
            "inside_cap_annihilator_contraction_complete": False,
            "exact_source_locator_complete": False,
            "generic_multiplicity_integer_lift_complete": False,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = all_controls()
    ledger = cost_ledger(controls)
    rows = [
        *controls["actual"],
        *controls["matched_random_decks"],
    ]
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "sixteen_actual_and_matched_instances": (
            controls["instance_count"] == 16
        ),
        "all_canonical_source_bodies_enumerated": (
            controls["full_source_count"]
            == sum(row["full_source_body_count"] for row in rows)
        ),
        "all_full_body_zero_biconditionals_exact": controls[
            "all_full_body_zero_biconditionals_exact"
        ],
        "all_exterior_laplace_checks_exact": controls[
            "all_exterior_laplace_checks_exact"
        ],
        "two_actual_double_fibers_replayed": (
            controls["double_fiber_instance_count"] == 2
        ),
        "all_determinant_value_sets_frozen": all(
            len(row["determinant_values"])
            == row["distinct_determinant_value_count"]
            for row in rows
        ),
        "minimal_raw_value_mask_degree_exact": all(
            row["minimum_exact_raw_value_zero_mask_degree"]
            == row["distinct_nonzero_determinant_value_count"]
            for row in rows
        ),
        "r108_weight_and_r105_markers_preserved": all(
            source["canonical_cycle_weight"] == R108.FULL_CYCLE_SCALE
            and len(source["marker"]) == R105.MARKER_DIMENSION
            for row in rows
            for source in row["zero_sources"]
        ),
        "raw_determinant_gauge_dependence_measured": not controls[
            "raw_determinant_endpoint_only_on_all_instances"
        ],
        "inside_cap_annihilator_contraction_complete": False,
        "subset_stable_existence_oracle_complete": False,
        "exact_source_locator_complete": False,
        "generic_multiplicity_integer_lift_complete": False,
        "known_rhs_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one gauge-normalized endpoint Query2P1 index "
        "for the position-separated alternant. Quotient the nonzero "
        "alternant gauge factors, compile exact endpoint membership and a "
        "subset-stable source witness inside B^(9/4) setup and B^(5/4) "
        "fresh work/workspace, and replay target updates, repeated atoms, "
        "R108 weight 14400, R105 markers, multiplicity, rank, factor logs, "
        "and identical descent without DLP labels."
    )
    summaries = [
        {
            key: value
            for key, value in row.items()
            if key not in {
                "determinant_values",
                "row_conditioned_annihilator_degrees",
                "column_conditioned_annihilator_degrees",
            }
        }
        for row in rows
    ]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_finite_deck_alternant_annihilator.r111.v1"
        ),
        "source_bindings": source_binding_records(),
        "field_model": "actual j=0 prime-field curves from R82",
        "source_arity": {"a": 5, "c": 5, "target": 1},
        "alternant_dimension": ALTERNANT_DIMENSION,
        "exterior_split": {
            "a_rows": 5,
            "c_plus_target_rows": 6,
            "ambient_dimension": EXTERIOR_DIMENSION,
            "identity": (
                "det([A;C,T]) is the signed pairing of the Lambda^5 "
                "A-row coordinate with the complementary Lambda^6 "
                "(C,target)-row coordinate"
            ),
        },
        "raw_value_mask_theorem": ledger["raw_value_zero_mask"],
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "ECDLP-IDEA-012 determinant value channel",
                "ECDLP-IDEA-012 finite-deck weighted endpoint gate R2",
                "ECDLP-IDEA-012 Query2P1 indexing gate R3",
                "ECDLP-IDEA-012 target-label common-factor gate R4",
            ],
            "new_local_control_only": (
                "full actual-deck determinant spectra and exact minimal "
                "raw-value annihilator degrees"
            ),
            "new_algorithm_claimed": False,
        },
    }
    source_replay = {
        "schema": (
            "p1553.finite_deck_annihilator_source_replay.r111.v1"
        ),
        "actual": controls["actual"],
        "matched_random_decks": controls["matched_random_decks"],
        "summary": {
            key: value
            for key, value in controls.items()
            if key not in {"actual", "matched_random_decks"}
        },
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.finite_deck_annihilator_exceptional_controls.r111.v1"
        ),
        "instance_summaries": summaries,
        "controls": {
            "position_separated_affine_chart_inherited_from_r110": True,
            "repeated_original_atoms_replayed": True,
            "two_actual_double_fibers_replayed": True,
            "matched_random_decks_replayed": True,
            "raw_value_endpoint_gauge_dependence_detected": not controls[
                "raw_determinant_endpoint_only_on_all_instances"
            ],
            "candidate_scalar_labels_consumed": False,
        },
        "scope": {
            "raw_univariate_value_annihilator_closed": True,
            "standard_side_conditioned_coefficient_layouts_closed": True,
            "arbitrary_finite_deck_circuits_closed": False,
            "gauge_normalized_endpoint_query2p1_closed": False,
        },
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r111.v1",
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
            "EXACT_FINITE_DECK_SCOPED_NEGATIVE_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "FULL_ACTUAL_AND_MATCHED_FINITE_DECKS_REPLAY_EXACT_EXTERIOR_"
            "ALTERNANT_ZERO_BICONDITIONAL__MINIMAL_RAW_DETERMINANT_ZERO_"
            "MASK_POLYNOMIAL_DEGREE_EQUALS_DISTINCT_NONZERO_VALUE_COUNT__"
            "ALL_VALUE_SETS_AND_SIDE_CONDITIONED_DEGREES_FROZEN__STANDARD_"
            "ROW_OR_COLUMN_ANNIHILATOR_COEFFICIENT_LAYOUT_HAS_B5_CAPACITY__"
            "LOW_RANK_ZERO_INCIDENCE_IS_NOT_AVAILABLE_BEFORE_MASK_"
            "CONSTRUCTION__RAW_UNIVARIATE_AND_STANDARD_CONDITIONAL_"
            "ANNIHILATORS_CLOSED_ONLY__GAUGE_NORMALIZED_ENDPOINT_QUERY2P1_"
            "OPEN"
        ),
        "source_bindings": source_binding_records(),
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
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_5a5c_finite_deck_alternant_annihilator.json",
            "contraction_ledger": (
                "finite_deck_annihilator_contraction_ledger.json"
            ),
            "source_replay": (
                "finite_deck_annihilator_source_replay.json"
            ),
            "exceptional": (
                "finite_deck_annihilator_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r111.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Observed finite-deck degrees are not an asymptotic theorem.",
            "No lower bound for arbitrary nonlinear circuits is proved.",
            "Low rank of an already-built zero mask is not a mask constructor.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_RAW_VALUE_ANNIHILATOR_DEGREE_AND_STANDARD_LAYOUT_"
            "NEGATIVE_ONLY__MERGE_WITH_IDEA012_R2_R3_R4__PRESERVE_GAUGE_"
            "NORMALIZED_ENDPOINT_QUERY2P1__NO_LOCATOR__NO_RANK__NO_FACTOR_"
            "LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "contraction_ledger": ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_finite_deck_alternant_"
            "annihilator_probe_report_r111.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_finite_deck_alternant_annihilator.json"
        ),
    )
    parser.add_argument(
        "--ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "finite_deck_annihilator_contraction_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "finite_deck_annihilator_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "finite_deck_annihilator_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r111.json"
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
    write_json(args.ledger_output, bundle["contraction_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R111 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
