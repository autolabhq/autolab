#!/usr/bin/env python3
"""Audit the gauge-normalized endpoint Query2P1 route after R111."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_gauge_normalized_endpoint_query2p1.r112.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
A5_EXPONENT = Fraction(2)
C2_EXPONENT = Fraction(6, 5)
C3_EXPONENT = Fraction(9, 5)
C5_EXPONENT = Fraction(3)
A5_C2_EXPONENT = Fraction(16, 5)
A5_C3_EXPONENT = Fraction(19, 5)
FULL_SOURCE_EXPONENT = Fraction(5)

R111_PRODUCER = pathlib.Path(
    "p1553_5a5c_finite_deck_alternant_annihilator_probe_r111.py"
)
R111_PRODUCER_SHA256 = (
    "64c7d3ff92740945650af5149ecc7fb175567e06382a51bb732118c9fc8aa6c3"
)
R111_REPORT = pathlib.Path(
    "p1553_5a5c_finite_deck_alternant_annihilator_probe_report_r111.json"
)
R111_REPORT_SHA256 = (
    "15c15d8cf036222b8e21403838f7ba3a268959dec574e9da5be342425fae7efa"
)
R111_FROZEN = pathlib.Path(
    "frozen_5a5c_finite_deck_alternant_annihilator.json"
)
R111_FROZEN_SHA256 = (
    "65cd8da27c50c5f088875f5c476c0951672439772efba000bff0c1f3d155f3e8"
)
R111_LEDGER = pathlib.Path(
    "finite_deck_annihilator_contraction_ledger.json"
)
R111_LEDGER_SHA256 = (
    "f0619633f14489f30daf0a1eb05206ef67896d287dc9df96823c13b55a688437"
)
R111_REPLAY = pathlib.Path("finite_deck_annihilator_source_replay.json")
R111_REPLAY_SHA256 = (
    "8cc7d45b6bad43fe891b983fd8523f58e16c0638b9fcb28d0bf52a87a2d42cf3"
)
R111_EXCEPTIONAL = pathlib.Path(
    "finite_deck_annihilator_exceptional_controls.json"
)
R111_EXCEPTIONAL_SHA256 = (
    "f0d0e08605c0783882ca3673112d08c91a004105cd756ba7fd376eb89cb265ae"
)
R111_LOGS = pathlib.Path("factor_logs_and_identical_descent_r111.json")
R111_LOGS_SHA256 = (
    "40d5c676a172ef300d1de646c2876ff42d1d8389a94d41c9a8b36f4d2a8acf6b"
)
R111_GATE = pathlib.Path(
    "p1553_5a5c_finite_deck_alternant_annihilator_probe_gate_r111.md"
)
R111_GATE_SHA256 = (
    "464feb11977bdbbd760a31448d88b08f44e7e8513d8fa65f181076853331b62a"
)
R111_PARENT = pathlib.Path(
    "p1553_5a5c_finite_deck_alternant_annihilator_probe_parent_report_r111.yaml"
)
R111_PARENT_SHA256 = (
    "59289adda0a76ec0ba74ac1397a3a6dfedb0c7b2e58e7d226e84ebdf0519596c"
)
IDEA_REGISTRY = pathlib.Path("p1553_r35_artifact_index_README.md")
IDEA_REGISTRY_SHA256 = (
    "9f4371eefd5e4019833eef858e3bda79d41aff0c5d7b861c71a5987a96acc392"
)

Point = tuple[int, int] | None
IndexTuple = tuple[int, ...]
Source = tuple[IndexTuple, IndexTuple]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R111_PRODUCER: R111_PRODUCER_SHA256,
        R111_REPORT: R111_REPORT_SHA256,
        R111_FROZEN: R111_FROZEN_SHA256,
        R111_LEDGER: R111_LEDGER_SHA256,
        R111_REPLAY: R111_REPLAY_SHA256,
        R111_EXCEPTIONAL: R111_EXCEPTIONAL_SHA256,
        R111_LOGS: R111_LOGS_SHA256,
        R111_GATE: R111_GATE_SHA256,
        R111_PARENT: R111_PARENT_SHA256,
        IDEA_REGISTRY: IDEA_REGISTRY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R112 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R111 = load_module("p1553_r111_for_r112", R111_PRODUCER)
R110 = R111.R110
R108 = R111.R108
R105 = R111.R105
R102 = R111.R102
R82 = R111.R82
R70 = R111.R70


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


def json_source(value: Sequence[Sequence[int]]) -> Source:
    return tuple(value[0]), tuple(value[1])


def projective_key(point: Point) -> tuple[int, int, int]:
    if point is None:
        return 0, 1, 0
    return point[0], point[1], 1


def endpoint(
    source: Sequence[int],
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R110.add_many((atoms[index] for index in source), curve)


def source_list(size: int, arity: int) -> list[IndexTuple]:
    return list(
        itertools.combinations_with_replacement(range(size), arity)
    )


def a_endpoint_index(
    a_sources: Sequence[IndexTuple],
    atoms_a: Sequence[Point],
    curve: dict[str, Any],
) -> tuple[dict[tuple[int, int, int], list[IndexTuple]], list[Point]]:
    index: dict[tuple[int, int, int], list[IndexTuple]] = {}
    endpoints = []
    for source in a_sources:
        value = endpoint(source, atoms_a, curve)
        endpoints.append(value)
        index.setdefault(projective_key(value), []).append(source)
    return index, endpoints


def query_c_sources(
    target: Point,
    c_sources: Sequence[IndexTuple],
    c_endpoints: Sequence[Point],
    a_index: dict[tuple[int, int, int], list[IndexTuple]],
    curve: dict[str, Any],
    start: int = 0,
    stop: int | None = None,
) -> tuple[list[Source], int]:
    if stop is None:
        stop = len(c_sources)
    matches = []
    for index in range(start, stop):
        residual = R70.add(
            target,
            R70.negate(c_endpoints[index], curve),
            curve,
        )
        for source_a in a_index.get(projective_key(residual), []):
            matches.append((source_a, c_sources[index]))
    return matches, stop - start


def dyadic_source_recovery(
    target: Point,
    c_sources: Sequence[IndexTuple],
    c_endpoints: Sequence[Point],
    a_index: dict[tuple[int, int, int], list[IndexTuple]],
    curve: dict[str, Any],
) -> dict[str, Any]:
    full_matches, inspected = query_c_sources(
        target,
        c_sources,
        c_endpoints,
        a_index,
        curve,
    )
    if not full_matches:
        return {
            "exists": False,
            "returned_source": None,
            "subset_queries": 1,
            "inspected_c_sources": inspected,
            "full_c_source_count": len(c_sources),
        }
    low = 0
    high = len(c_sources)
    subset_queries = 1
    while high - low > 1:
        middle = (low + high) // 2
        left_matches, work = query_c_sources(
            target,
            c_sources,
            c_endpoints,
            a_index,
            curve,
            low,
            middle,
        )
        inspected += work
        subset_queries += 1
        if left_matches:
            high = middle
        else:
            low = middle
    final_matches, work = query_c_sources(
        target,
        c_sources,
        c_endpoints,
        a_index,
        curve,
        low,
        high,
    )
    inspected += work
    subset_queries += 1
    if not final_matches:
        raise AssertionError("dyadic source recovery lost the witness")
    return {
        "exists": True,
        "returned_source": source_json(final_matches[0]),
        "subset_queries": subset_queries,
        "inspected_c_sources": inspected,
        "full_c_source_count": len(c_sources),
        "inspection_ratio": inspected / len(c_sources),
    }


def canonical_pair_triple_query(
    target: Point,
    c2_sources: Sequence[IndexTuple],
    c2_endpoints: Sequence[Point],
    c3_sources: Sequence[IndexTuple],
    c3_endpoints: Sequence[Point],
    a_index: dict[tuple[int, int, int], list[IndexTuple]],
    curve: dict[str, Any],
) -> tuple[list[Source], int]:
    matches = []
    traffic = 0
    for source_c2, endpoint_c2 in zip(c2_sources, c2_endpoints):
        for source_c3, endpoint_c3 in zip(c3_sources, c3_endpoints):
            if source_c2[-1] > source_c3[0]:
                continue
            traffic += 1
            endpoint_c = R70.add(endpoint_c2, endpoint_c3, curve)
            residual = R70.add(
                target,
                R70.negate(endpoint_c, curve),
                curve,
            )
            for source_a in a_index.get(projective_key(residual), []):
                matches.append((source_a, source_c2 + source_c3))
    return matches, traffic


def expected_sources() -> dict[tuple[str, str, int], list[Source]]:
    payload = json.loads(R111_REPLAY.read_text(encoding="utf-8"))
    output = {}
    for control_class, key in (
        ("actual", "actual"),
        ("matched_random_deck", "matched_random_decks"),
    ):
        for row in payload[key]:
            output[
                control_class,
                row["family_id"],
                row["offset"],
            ] = [
                json_source(source["source"])
                for source in row["zero_sources"]
            ]
    return output


def point_multiplicity(points: Iterable[Point]) -> dict[str, int]:
    counts = collections.Counter(points)
    return {
        "distinct_endpoint_count": len(counts),
        "maximum_endpoint_multiplicity": max(counts.values()),
        "collision_excess": sum(value - 1 for value in counts.values()),
    }


def find_no_relation_target(
    target: Point,
    step: Point,
    c_sources: Sequence[IndexTuple],
    c_endpoints: Sequence[Point],
    a_index: dict[tuple[int, int, int], list[IndexTuple]],
    curve: dict[str, Any],
) -> dict[str, Any]:
    for multiplier in range(1, len(c_sources) + 2):
        candidate = R70.add(
            target,
            R70.scalar_mul(multiplier, step, curve),
            curve,
        )
        matches, work = query_c_sources(
            candidate,
            c_sources,
            c_endpoints,
            a_index,
            curve,
        )
        if not matches:
            return {
                "target": point_json(candidate),
                "step_multiplier": multiplier,
                "source_count": 0,
                "query_work": work,
            }
    raise AssertionError("unable to find no-relation target control")


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
    doubles: dict[tuple[str, int], dict[str, Any]],
    expected: dict[tuple[str, str, int], list[Source]],
) -> dict[str, Any]:
    inputs = R111.instance_inputs(
        curve,
        offset,
        control_class,
        doubles,
    )
    atoms_a = inputs["atoms_a"]
    atoms_c = inputs["atoms_c"]
    target = inputs["target"]
    a_sources = source_list(len(atoms_a), 5)
    c_sources = source_list(len(atoms_c), 5)
    c2_sources = source_list(len(atoms_c), 2)
    c3_sources = source_list(len(atoms_c), 3)
    a_index, a_endpoints = a_endpoint_index(
        a_sources,
        atoms_a,
        curve,
    )
    c_endpoints = [
        endpoint(source, atoms_c, curve) for source in c_sources
    ]
    c2_endpoints = [
        endpoint(source, atoms_c, curve) for source in c2_sources
    ]
    c3_endpoints = [
        endpoint(source, atoms_c, curve) for source in c3_sources
    ]
    direct_matches, direct_work = query_c_sources(
        target,
        c_sources,
        c_endpoints,
        a_index,
        curve,
    )
    split_matches, split_work = canonical_pair_triple_query(
        target,
        c2_sources,
        c2_endpoints,
        c3_sources,
        c3_endpoints,
        a_index,
        curve,
    )
    expected_rows = expected[
        control_class,
        curve["family_id"],
        offset,
    ]
    if sorted(direct_matches) != sorted(expected_rows):
        raise AssertionError("typed endpoint query disagrees with R111")
    if sorted(split_matches) != sorted(expected_rows):
        raise AssertionError("canonical pair/triple query disagrees")
    if split_work != len(c_sources):
        raise AssertionError("canonical C2+C3 split is not one-to-one")
    dyadic = dyadic_source_recovery(
        target,
        c_sources,
        c_endpoints,
        a_index,
        curve,
    )
    no_relation = find_no_relation_target(
        target,
        inputs["schedule"]["shift_generator"],
        c_sources,
        c_endpoints,
        a_index,
        curve,
    )
    factor_base_size = len(inputs["factors"])
    setup_entries = (
        len(a_sources) + len(c2_sources) + len(c3_sources)
    )
    source_rows = [
        {
            "source": source_json(source),
            "marker": list(
                R105.marker_vector(source, curve["field_prime"])
            ),
            "canonical_cycle_weight": R108.FULL_CYCLE_SCALE,
        }
        for source in direct_matches
    ]
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": factor_base_size,
        "target_class": inputs["target_class"],
        "target": point_json(target),
        "projective_key_dimension": 3,
        "projective_key_exact_on_all_endpoints": True,
        "a5_source_count": len(a_sources),
        "c2_source_count": len(c2_sources),
        "c3_source_count": len(c3_sources),
        "c5_source_count": len(c_sources),
        "stored_setup_entry_count": setup_entries,
        "stored_setup_observed_exponent_B": observed_exponent(
            setup_entries,
            factor_base_size,
        ),
        "a5_endpoint_multiplicity": point_multiplicity(a_endpoints),
        "c5_endpoint_multiplicity": point_multiplicity(c_endpoints),
        "direct_query_c5_traffic": direct_work,
        "canonical_pair_triple_query_traffic": split_work,
        "query_traffic_observed_exponent_B": observed_exponent(
            split_work,
            factor_base_size,
        ),
        "source_count": len(direct_matches),
        "sources": source_rows,
        "direct_query_matches_r111": (
            sorted(direct_matches) == sorted(expected_rows)
        ),
        "canonical_pair_triple_matches_r111": (
            sorted(split_matches) == sorted(expected_rows)
        ),
        "dyadic_source_recovery": dyadic,
        "no_relation_target_control": no_relation,
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def all_controls() -> dict[str, Any]:
    doubles = R110.r105_double_fibers()
    expected = expected_sources()
    actual = [
        analyze_instance(
            dict(family),
            offset,
            "actual",
            doubles,
            expected,
        )
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        analyze_instance(
            dict(family),
            offset,
            "matched_random_deck",
            doubles,
            expected,
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "instance_count": len(rows),
        "source_count": sum(row["source_count"] for row in rows),
        "double_fiber_instance_count": sum(
            row["target_class"] == "r105_actual_double_fiber"
            and row["source_count"] == 2
            for row in rows
        ),
        "all_direct_queries_match_r111": all(
            row["direct_query_matches_r111"] for row in rows
        ),
        "all_pair_triple_queries_match_r111": all(
            row["canonical_pair_triple_matches_r111"] for row in rows
        ),
        "all_no_relation_controls_exact": all(
            row["no_relation_target_control"]["source_count"] == 0
            for row in rows
        ),
        "all_dyadic_recoveries_exact": all(
            row["dyadic_source_recovery"]["exists"]
            and json_source(
                row["dyadic_source_recovery"]["returned_source"]
            )
            in expected[
                row["control_class"],
                row["family_id"],
                row["offset"],
            ]
            for row in rows
        ),
        "maximum_no_relation_step_multiplier": max(
            row["no_relation_target_control"]["step_multiplier"]
            for row in rows
        ),
        "maximum_dyadic_inspection_ratio": max(
            row["dyadic_source_recovery"]["inspection_ratio"]
            for row in rows
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r111_producer": {
            "path": str(R111_PRODUCER),
            "sha256": R111_PRODUCER_SHA256,
        },
        "r111_report": {
            "path": str(R111_REPORT),
            "sha256": R111_REPORT_SHA256,
        },
        "r111_frozen": {
            "path": str(R111_FROZEN),
            "sha256": R111_FROZEN_SHA256,
        },
        "r111_ledger": {
            "path": str(R111_LEDGER),
            "sha256": R111_LEDGER_SHA256,
        },
        "r111_replay": {
            "path": str(R111_REPLAY),
            "sha256": R111_REPLAY_SHA256,
        },
        "r111_exceptional": {
            "path": str(R111_EXCEPTIONAL),
            "sha256": R111_EXCEPTIONAL_SHA256,
        },
        "r111_logs": {
            "path": str(R111_LOGS),
            "sha256": R111_LOGS_SHA256,
        },
        "r111_gate": {
            "path": str(R111_GATE),
            "sha256": R111_GATE_SHA256,
        },
        "r111_parent": {
            "path": str(R111_PARENT),
            "sha256": R111_PARENT_SHA256,
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
        "typed_endpoint_normalization": {
            "key": "[X:Y:Z] with O=[0:1:0]",
            "key_dimension": 3,
            "exact_on_complete_projective_chart": True,
            "uses_dlp_label": False,
            "raw_alternant_gauge_quotiented_at_predicate_level": True,
            "globally_scalar_gauge_quotient_claimed": False,
        },
        "target_independent_setup": {
            "a5_endpoint_index_exponent_B": fraction_record(A5_EXPONENT),
            "c2_endpoint_table_exponent_B": fraction_record(C2_EXPONENT),
            "c3_endpoint_table_exponent_B": fraction_record(C3_EXPONENT),
            "combined_state_exponent_B": fraction_record(A5_EXPONENT),
            "inside_setup_cap": A5_EXPONENT <= SETUP_CAP,
            "maximum_observed_entry_count": max(
                row["stored_setup_entry_count"] for row in rows
            ),
        },
        "fresh_target_query": {
            "canonical_c2_c3_join_exponent_B": fraction_record(C5_EXPONENT),
            "direct_c5_scan_exponent_B": fraction_record(C5_EXPONENT),
            "inside_online_cap": C5_EXPONENT <= ONLINE_CAP,
            "target_update_exponent_B": fraction_record(C5_EXPONENT),
            "subset_source_recovery_exponent_B": fraction_record(C5_EXPONENT),
            "maximum_dyadic_inspection_ratio": controls[
                "maximum_dyadic_inspection_ratio"
            ],
        },
        "standard_pair_materializations": {
            "c2_plus_c3_exponent_B": fraction_record(C5_EXPONENT),
            "a5_plus_c2_exponent_B": fraction_record(A5_C2_EXPONENT),
            "a5_plus_c3_exponent_B": fraction_record(A5_C3_EXPONENT),
            "minimum_exponent_B": fraction_record(C5_EXPONENT),
            "minimum_inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "minimum_inside_online_cap": C5_EXPONENT <= ONLINE_CAP,
        },
        "constructor_status": {
            "exact_typed_endpoint_key_complete": True,
            "target_independent_thin_tables_inside_setup": True,
            "inside_cap_query2p1_decision_complete": False,
            "inside_cap_subset_source_recovery_complete": False,
            "generic_multiplicity_integer_lift_complete": False,
        },
        "scope": {
            "typed_hash_and_direct_pair_table_grammars_closed": True,
            "integer_3sum_transplants_closed_by_registry_r3": True,
            "standard_resultant_and_fitting_routes_closed_by_registry_r4": True,
            "nonlinear_orbit_product_recurrence_closed": False,
            "arbitrary_query_data_structure_lower_bound_claimed": False,
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
        "typed_projective_endpoint_key_exact": all(
            row["projective_key_exact_on_all_endpoints"] for row in rows
        ),
        "all_direct_queries_match_r111": controls[
            "all_direct_queries_match_r111"
        ],
        "all_pair_triple_queries_match_r111": controls[
            "all_pair_triple_queries_match_r111"
        ],
        "all_no_relation_controls_exact": controls[
            "all_no_relation_controls_exact"
        ],
        "all_dyadic_recoveries_exact": controls[
            "all_dyadic_recoveries_exact"
        ],
        "two_actual_double_fibers_replayed": (
            controls["double_fiber_instance_count"] == 2
        ),
        "r108_weight_and_r105_markers_preserved": all(
            source["canonical_cycle_weight"] == R108.FULL_CYCLE_SCALE
            and len(source["marker"]) == R105.MARKER_DIMENSION
            for row in rows
            for source in row["sources"]
        ),
        "target_independent_tables_inside_setup": ledger[
            "target_independent_setup"
        ]["inside_setup_cap"],
        "inside_cap_query2p1_decision_complete": False,
        "inside_cap_subset_source_recovery_complete": False,
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
        "Construct or refute one gauge-invariant nonlinear elliptic "
        "orbit-product recurrence for the canonical C2 x C3 endpoint join. "
        "It must answer the target-shifted A5 membership query and return a "
        "dyadic canonical source inside B^(9/4) setup and B^(5/4) fresh "
        "work/workspace, with explicit recurrence order, target update, "
        "exceptional charts, weight 14400, markers, multiplicity, rank, "
        "factor logs, and identical descent."
    )
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_gauge_normalized_endpoint_query2p1.r112.v1"
        ),
        "source_bindings": source_binding_records(),
        "field_model": "actual j=0 prime-field curves from R82",
        "typed_endpoint_key": ledger["typed_endpoint_normalization"],
        "source_split": {
            "a5_exponent_B": fraction_record(A5_EXPONENT),
            "c2_exponent_B": fraction_record(C2_EXPONENT),
            "c3_exponent_B": fraction_record(C3_EXPONENT),
            "canonical_boundary": "max(C2 indices) <= min(C3 indices)",
            "one_to_one_with_canonical_c5_sources": True,
        },
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "ECDLP-IDEA-012 finite-deck weighted endpoint gate R2",
                "ECDLP-IDEA-012 Query2P1 indexing gate R3",
                "ECDLP-IDEA-012 target-label common-factor gate R4",
            ],
            "new_local_control_only": (
                "full actual-deck typed endpoint, canonical C2+C3, "
                "no-relation, and dyadic source replay"
            ),
            "new_algorithm_claimed": False,
        },
    }
    source_replay = {
        "schema": "p1553.query2p1_subset_source_replay.r112.v1",
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
        "schema": "p1553.query2p1_exceptional_controls.r112.v1",
        "controls": {
            "complete_projective_key_includes_identity": True,
            "repeated_original_atoms_replayed": True,
            "two_actual_double_fibers_replayed": True,
            "matched_random_decks_replayed": True,
            "no_relation_targets_replayed": True,
            "maximum_no_relation_step_multiplier": controls[
                "maximum_no_relation_step_multiplier"
            ],
            "candidate_scalar_labels_consumed": False,
        },
        "scope": ledger["scope"],
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r112.v1",
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
            "EXACT_TYPED_ENDPOINT_INDEX_SCOPED_NEGATIVE_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "GAUGE_NORMALIZATION_TO_COMPLETE_PROJECTIVE_ENDPOINT_KEY_IS_"
            "EXACT_ON_ALL_ACTUAL_AND_MATCHED_SOURCES__A5_C2_C3_THIN_TABLES_"
            "FIT_SETUP__CANONICAL_C2_X_C3_QUERY_AND_TARGET_UPDATE_RETAIN_B3_"
            "TRAFFIC__DYADIC_SOURCE_RECOVERY_WITH_SCAN_BASED_SUBSET_ORACLE_"
            "RETAINS_B3_WORK__DOUBLE_FIBERS_MARKERS_AND_NO_RELATION_CONTROLS_"
            "REPLAY__TYPED_HASH_DIRECT_PAIR_AND_REGISTRY_R3_R4_STANDARD_"
            "GRAMMARS_CLOSED_ONLY__GAUGE_INVARIANT_NONLINEAR_ORBIT_PRODUCT_"
            "RECURRENCE_OPEN"
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
            "typed_endpoint_normalization_admitted": True,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_5a5c_gauge_normalized_endpoint_query2p1.json"
            ),
            "index_ledger": "gauge_normalized_endpoint_index_ledger.json",
            "source_replay": "query2p1_subset_source_replay.json",
            "exceptional": "query2p1_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r112.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The typed endpoint key is not a scalar DLP coordinate.",
            "No lower bound for arbitrary Query2P1 data structures is proved.",
            "The B^3 scan is a standard-grammar cost, not a universal bound.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_TYPED_ENDPOINT_NORMALIZATION_AND_THIN_SETUP_TABLES__"
            "REJECT_DIRECT_C5_AND_CANONICAL_C2XC3_B3_QUERY__MERGE_STANDARD_"
            "INDEXING_AND_RESULTANT_ROUTES_WITH_IDEA012_R3_R4__PRESERVE_"
            "NONLINEAR_ORBIT_PRODUCT_RECURRENCE__NO_LOCATOR__NO_RANK__NO_"
            "FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "index_ledger": ledger,
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
            "p1553_5a5c_gauge_normalized_endpoint_"
            "query2p1_probe_report_r112.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_gauge_normalized_endpoint_query2p1.json"
        ),
    )
    parser.add_argument(
        "--ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "gauge_normalized_endpoint_index_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path("query2p1_subset_source_replay.json"),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path("query2p1_exceptional_controls.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r112.json"
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
    write_json(args.ledger_output, bundle["index_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R112 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
