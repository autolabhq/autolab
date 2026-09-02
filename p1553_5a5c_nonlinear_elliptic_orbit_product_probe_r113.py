#!/usr/bin/env python3
"""Audit fixed-orbit and product-tree recurrences for the R112 join."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_nonlinear_elliptic_orbit_product.r113.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
C3_EXPONENT = Fraction(9, 5)
C5_EXPONENT = Fraction(3)
FULL_GROUP_EXPONENT = Fraction(5)
FULL_SOURCE_EXPONENT = Fraction(5)

R112_PRODUCER = pathlib.Path(
    "p1553_5a5c_gauge_normalized_endpoint_query2p1_probe_r112.py"
)
R112_PRODUCER_SHA256 = (
    "d556faf03c6c9ec5448b40754582e2f57c42cdb15a64a007e74581a49a3e8a34"
)
R112_REPORT = pathlib.Path(
    "p1553_5a5c_gauge_normalized_endpoint_query2p1_probe_report_r112.json"
)
R112_REPORT_SHA256 = (
    "3fa72cd2fd1ea45efc0db0946fac71ce42d325327d282a269439f61fe1719db2"
)
R112_FROZEN = pathlib.Path(
    "frozen_5a5c_gauge_normalized_endpoint_query2p1.json"
)
R112_FROZEN_SHA256 = (
    "0e817b083bb044ae719d70ed6db7d8e4bd328ab06d644d5c96a4f4c7e6ab2085"
)
R112_LEDGER = pathlib.Path("gauge_normalized_endpoint_index_ledger.json")
R112_LEDGER_SHA256 = (
    "98feb6e98a004219d16a15421d59f1f69afd414044a61bba09232b9af627973e"
)
R112_REPLAY = pathlib.Path("query2p1_subset_source_replay.json")
R112_REPLAY_SHA256 = (
    "b5bb8bf663f5b24bc74511b10c77cc6b693577266aba5b037c191f18dade51de"
)
R112_EXCEPTIONAL = pathlib.Path("query2p1_exceptional_controls.json")
R112_EXCEPTIONAL_SHA256 = (
    "110d294b56f23a917a2b15291bfa4eccc8c2fe9416aef84f8bf51b9e97ee1478"
)
R112_LOGS = pathlib.Path("factor_logs_and_identical_descent_r112.json")
R112_LOGS_SHA256 = (
    "05adbed7775de1ba91058bb320b861eda03fa683a13910c307a2b873d7373ff3"
)
R112_GATE = pathlib.Path(
    "p1553_5a5c_gauge_normalized_endpoint_query2p1_probe_gate_r112.md"
)
R112_GATE_SHA256 = (
    "d8de8e795feb08ebf967b9f10aa4b4f41919b4e808aa8f0bdaa88c206b7015ac"
)
R112_PARENT = pathlib.Path(
    "p1553_5a5c_gauge_normalized_endpoint_query2p1_probe_parent_report_r112.yaml"
)
R112_PARENT_SHA256 = (
    "82fda11f8c376fb127b3f7bd384fb52ec4c7fd70f6dc1cc4ece447e7647aaf19"
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
        R112_PRODUCER: R112_PRODUCER_SHA256,
        R112_REPORT: R112_REPORT_SHA256,
        R112_FROZEN: R112_FROZEN_SHA256,
        R112_LEDGER: R112_LEDGER_SHA256,
        R112_REPLAY: R112_REPLAY_SHA256,
        R112_EXCEPTIONAL: R112_EXCEPTIONAL_SHA256,
        R112_LOGS: R112_LOGS_SHA256,
        R112_GATE: R112_GATE_SHA256,
        R112_PARENT: R112_PARENT_SHA256,
        IDEA_REGISTRY: IDEA_REGISTRY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R113 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R112 = load_module("p1553_r112_for_r113", R112_PRODUCER)
R111 = R112.R111
R110 = R112.R110
R108 = R112.R108
R105 = R112.R105
R82 = R112.R82
R70 = R112.R70


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


def json_point(value: Sequence[int] | None) -> Point:
    return None if value is None else (value[0], value[1])


def source_json(source: Source) -> list[list[int]]:
    return [list(source[0]), list(source[1])]


def source_list(size: int, arity: int) -> list[IndexTuple]:
    return list(
        itertools.combinations_with_replacement(range(size), arity)
    )


def endpoint(
    source: Sequence[int],
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R110.add_many((atoms[index] for index in source), curve)


def product_tree(leaves: Sequence[int]) -> dict[str, Any]:
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            [
                current[index]
                * (current[index + 1] if index + 1 < len(current) else 1)
                for index in range(0, len(current), 2)
            ]
        )
    zero_indices = [
        index for index, value in enumerate(leaves) if value == 0
    ]
    return {
        "root": levels[-1][0],
        "level_sizes": [len(level) for level in levels],
        "leaf_count": len(leaves),
        "internal_node_count": sum(
            len(level) for level in levels[1:]
        ),
        "zero_leaf_indices": zero_indices,
        "zero_leaf_count": len(zero_indices),
        "prefix_recurrence_order": 1,
        "prefix_recurrence_state_dimension": 1,
        "leaf_generator_evaluation_count": len(leaves),
    }


def expected_rows() -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = json.loads(R112_REPLAY.read_text(encoding="utf-8"))
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
            ] = row
    return output


def leaf_data(
    target: Point,
    c_sources: Sequence[IndexTuple],
    c_endpoints: Sequence[Point],
    a_index: dict[tuple[int, int, int], list[IndexTuple]],
    curve: dict[str, Any],
) -> tuple[list[int], list[list[IndexTuple]]]:
    leaves = []
    witnesses = []
    for source_c, endpoint_c in zip(c_sources, c_endpoints):
        residual = R70.add(
            target,
            R70.negate(endpoint_c, curve),
            curve,
        )
        source_as = a_index.get(R112.projective_key(residual), [])
        leaves.append(0 if source_as else 1)
        witnesses.append(list(source_as))
    return leaves, witnesses


def successor_difference_summary(
    points: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    differences = [
        R70.add(
            points[index + 1],
            R70.negate(points[index], curve),
            curve,
        )
        for index in range(len(points) - 1)
    ]
    return {
        "transition_count": len(differences),
        "distinct_successor_difference_count": len(set(differences)),
        "constant_successor_difference": len(set(differences)) <= 1,
        "identity_successor_difference_count": sum(
            difference is None for difference in differences
        ),
    }


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
    doubles: dict[tuple[str, int], dict[str, Any]],
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    inputs = R111.instance_inputs(
        curve,
        offset,
        control_class,
        doubles,
    )
    atoms_a = inputs["atoms_a"]
    atoms_c = inputs["atoms_c"]
    a_sources = source_list(len(atoms_a), 5)
    c3_sources = source_list(len(atoms_c), 3)
    c5_sources = source_list(len(atoms_c), 5)
    a_index, a_endpoints = R112.a_endpoint_index(
        a_sources,
        atoms_a,
        curve,
    )
    c3_endpoints = [
        endpoint(source, atoms_c, curve) for source in c3_sources
    ]
    c5_endpoints = [
        endpoint(source, atoms_c, curve) for source in c5_sources
    ]
    expected_row = expected[
        control_class,
        curve["family_id"],
        offset,
    ]
    target = inputs["target"]
    no_relation_target = json_point(
        expected_row["no_relation_target_control"]["target"]
    )
    target_leaves, target_witnesses = leaf_data(
        target,
        c5_sources,
        c5_endpoints,
        a_index,
        curve,
    )
    no_relation_leaves, _ = leaf_data(
        no_relation_target,
        c5_sources,
        c5_endpoints,
        a_index,
        curve,
    )
    target_tree = product_tree(target_leaves)
    no_relation_tree = product_tree(no_relation_leaves)
    recovered_sources = []
    for c_index in target_tree["zero_leaf_indices"]:
        for source_a in target_witnesses[c_index]:
            recovered_sources.append((source_a, c5_sources[c_index]))
    expected_sources = [
        (tuple(source["source"][0]), tuple(source["source"][1]))
        for source in expected_row["sources"]
    ]
    if sorted(recovered_sources) != sorted(expected_sources):
        raise AssertionError("orbit-product leaves disagree with R112")

    endpoint_support = {
        R70.add(endpoint_a, endpoint_c, curve)
        for endpoint_a in a_endpoints
        for endpoint_c in c5_endpoints
    }
    body_count = len(a_sources) * len(c5_sources)
    successor = successor_difference_summary(c3_endpoints, curve)
    if successor["constant_successor_difference"]:
        raise AssertionError("C3 endpoint ordering unexpectedly forms an orbit")
    source_rows = [
        {
            "source": source_json(source),
            "marker": list(
                R105.marker_vector(source, curve["field_prime"])
            ),
            "canonical_cycle_weight": R108.FULL_CYCLE_SCALE,
        }
        for source in recovered_sources
    ]
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(inputs["factors"]),
        "target_class": inputs["target_class"],
        "target": point_json(target),
        "a5_source_count": len(a_sources),
        "c3_source_count": len(c3_sources),
        "c5_source_count": len(c5_sources),
        "target_product_tree": target_tree,
        "no_relation_product_tree": no_relation_tree,
        "target_root_reports_existence": target_tree["root"] == 0,
        "no_relation_root_reports_absence": no_relation_tree["root"] == 1,
        "sources": source_rows,
        "sources_match_r112": (
            sorted(recovered_sources) == sorted(expected_sources)
        ),
        "c3_successor_differences": successor,
        "c3_support_is_nonempty_proper_subset_of_prime_group": (
            0 < len(set(c3_endpoints)) < curve["subgroup_order"]
        ),
        "c3_translation_stabilizer_is_trivial": True,
        "nonzero_translation_orbit_length": curve["subgroup_order"],
        "endpoint_zero_divisor_occurrence_degree": body_count,
        "endpoint_zero_divisor_support_degree": len(endpoint_support),
        "endpoint_zero_divisor_support_ratio": (
            len(endpoint_support) / body_count
        ),
        "target_independent_endpoint_dictionary_entry_count": len(
            endpoint_support
        ),
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def all_controls() -> dict[str, Any]:
    doubles = R110.r105_double_fibers()
    expected = expected_rows()
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
        "source_count": sum(len(row["sources"]) for row in rows),
        "all_product_roots_exact": all(
            row["target_root_reports_existence"]
            and row["no_relation_root_reports_absence"]
            for row in rows
        ),
        "all_sources_match_r112": all(
            row["sources_match_r112"] for row in rows
        ),
        "all_c3_translation_stabilizers_trivial": all(
            row["c3_translation_stabilizer_is_trivial"] for row in rows
        ),
        "all_c3_orderings_nonorbit": all(
            not row["c3_successor_differences"][
                "constant_successor_difference"
            ]
            for row in rows
        ),
        "minimum_endpoint_zero_divisor_support_ratio": min(
            row["endpoint_zero_divisor_support_ratio"] for row in rows
        ),
        "maximum_prefix_leaf_evaluations": max(
            row["target_product_tree"]["leaf_generator_evaluation_count"]
            for row in rows
        ),
        "maximum_product_tree_nodes": max(
            row["target_product_tree"]["leaf_count"]
            + row["target_product_tree"]["internal_node_count"]
            for row in rows
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r112_producer": {
            "path": str(R112_PRODUCER),
            "sha256": R112_PRODUCER_SHA256,
        },
        "r112_report": {
            "path": str(R112_REPORT),
            "sha256": R112_REPORT_SHA256,
        },
        "r112_frozen": {
            "path": str(R112_FROZEN),
            "sha256": R112_FROZEN_SHA256,
        },
        "r112_ledger": {
            "path": str(R112_LEDGER),
            "sha256": R112_LEDGER_SHA256,
        },
        "r112_replay": {
            "path": str(R112_REPLAY),
            "sha256": R112_REPLAY_SHA256,
        },
        "r112_exceptional": {
            "path": str(R112_EXCEPTIONAL),
            "sha256": R112_EXCEPTIONAL_SHA256,
        },
        "r112_logs": {
            "path": str(R112_LOGS),
            "sha256": R112_LOGS_SHA256,
        },
        "r112_gate": {
            "path": str(R112_GATE),
            "sha256": R112_GATE_SHA256,
        },
        "r112_parent": {
            "path": str(R112_PARENT),
            "sha256": R112_PARENT_SHA256,
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
        "fixed_translation_orbit": {
            "prime_group_theorem": (
                "A nonzero translation in a prime-order group generates the "
                "full group; every nonempty proper subset has trivial "
                "translation stabilizer."
            ),
            "c3_endpoint_support_exponent_B": fraction_record(C3_EXPONENT),
            "nonzero_orbit_length_exponent_B": fraction_record(
                FULL_GROUP_EXPONENT
            ),
            "full_orbit_inside_setup_cap": (
                FULL_GROUP_EXPONENT <= SETUP_CAP
            ),
            "all_actual_c3_stabilizers_trivial": controls[
                "all_c3_translation_stabilizers_trivial"
            ],
            "all_canonical_orderings_nonorbit": controls[
                "all_c3_orderings_nonorbit"
            ],
        },
        "nonlinear_product_tree": {
            "prefix_recurrence_order": 1,
            "prefix_state_dimension": 1,
            "leaf_generator_exponent_B": fraction_record(C5_EXPONENT),
            "product_tree_state_exponent_B": fraction_record(C5_EXPONENT),
            "leaf_generation_inside_online_cap": C5_EXPONENT <= ONLINE_CAP,
            "tree_state_inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "maximum_observed_leaf_count": controls[
                "maximum_prefix_leaf_evaluations"
            ],
            "maximum_observed_tree_nodes": controls[
                "maximum_product_tree_nodes"
            ],
            "bounded_product_state_reduces_leaf_work": False,
        },
        "target_independent_compilation": {
            "zero_divisor_occurrence_degree_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "endpoint_dictionary_state_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "minimum_observed_support_ratio": controls[
                "minimum_endpoint_zero_divisor_support_ratio"
            ],
            "inside_setup_cap": FULL_SOURCE_EXPONENT <= SETUP_CAP,
        },
        "constructor_status": {
            "exact_product_root_and_source_replay_complete": True,
            "fixed_translation_orbit_recurrence_complete": False,
            "inside_cap_leaf_generator_complete": False,
            "inside_cap_source_adjoint_complete": False,
            "generic_multiplicity_integer_lift_complete": False,
        },
        "scope": {
            "fixed_translation_orbit_closed": True,
            "standard_product_tree_closed": True,
            "bounded_prefix_state_as_work_claim_closed": True,
            "transposed_nonuniform_leaf_generator_closed": False,
            "arbitrary_variable_coefficient_recurrence_lower_bound_claimed": (
                False
            ),
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
        "all_product_roots_exact": controls["all_product_roots_exact"],
        "all_sources_match_r112": controls["all_sources_match_r112"],
        "all_c3_translation_stabilizers_trivial": controls[
            "all_c3_translation_stabilizers_trivial"
        ],
        "all_canonical_c3_orderings_nonorbit": controls[
            "all_c3_orderings_nonorbit"
        ],
        "endpoint_zero_divisor_supports_frozen": all(
            row["endpoint_zero_divisor_support_degree"] > 0 for row in rows
        ),
        "r108_weight_and_r105_markers_preserved": all(
            source["canonical_cycle_weight"] == R108.FULL_CYCLE_SCALE
            and len(source["marker"]) == R105.MARKER_DIMENSION
            for row in rows
            for source in row["sources"]
        ),
        "inside_cap_leaf_generator_complete": False,
        "inside_cap_source_adjoint_complete": False,
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
        "Construct or refute one transposed nonuniform C5 leaf generator "
        "for the typed endpoint membership functional. It must evaluate an "
        "exact target existence bit and adjoint one canonical source without "
        "forming B^3 leaves or B^5 endpoint support, inside B^(9/4) setup "
        "and B^(5/4) fresh work/workspace; charge target updates, complete "
        "charts, weight 14400, markers, multiplicity, rank, factor logs, and "
        "identical descent."
    )
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_nonlinear_elliptic_orbit_product.r113.v1"
        ),
        "source_bindings": source_binding_records(),
        "field_model": "actual prime-order j=0 curves from R82",
        "fixed_translation_orbit_theorem": ledger[
            "fixed_translation_orbit"
        ],
        "product_recurrence": {
            "leaf": (
                "1 - indicator_A5(T - endpoint(C5_source))"
            ),
            "root": "product of all canonical C5 leaves",
            "root_zero_iff_relation_exists": True,
            "prefix_order": 1,
            "leaf_generation_exponent_B": fraction_record(C5_EXPONENT),
        },
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "ECDLP-IDEA-006 translated-pole annihilator",
                "ECDLP-IDEA-012 target-label common-factor gate R4",
                "R97 factored transposed projector trace",
                "R100 succinct aggregate digit trie",
            ],
            "new_local_control_only": (
                "actual C3 stabilizers, product trees, endpoint zero "
                "divisors, and no-relation roots"
            ),
            "new_algorithm_claimed": False,
        },
    }
    source_replay = {
        "schema": "p1553.orbit_product_subset_source_replay.r113.v1",
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
        "schema": "p1553.orbit_product_exceptional_controls.r113.v1",
        "controls": {
            "complete_projective_key_inherited_from_r112": True,
            "identity_chart_included": True,
            "repeated_atoms_replayed": True,
            "double_fibers_replayed": True,
            "no_relation_product_roots_replayed": True,
            "candidate_scalar_labels_consumed": False,
        },
        "scope": ledger["scope"],
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r113.v1",
        "target_existence_and_source_replay_complete": True,
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
            "EXACT_ORBIT_PRODUCT_SCOPED_NEGATIVE_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "CANONICAL_C3_ENDPOINT_SETS_HAVE_TRIVIAL_TRANSLATION_"
            "STABILIZER_IN_THE_PRIME_GROUP_AND_ARE_NOT_FIXED_TRANSLATION_"
            "ORBITS__ANY_NONZERO_TRANSLATION_ORBIT_HAS_FULL_B5_LENGTH__"
            "ORDER_ONE_PREFIX_PRODUCT_STILL_REQUIRES_B3_DISTINCT_LEAF_"
            "GENERATIONS__PRODUCT_TREE_RETURNS_EXACT_SOURCE_ONLY_AFTER_B3_"
            "LEAVES__TARGET_INDEPENDENT_ENDPOINT_ZERO_DIVISOR_HAS_B5_"
            "OCCURRENCE_DEGREE_AND_NEAR_SOURCE_BODY_SUPPORT__FIXED_ORBIT_"
            "STANDARD_PRODUCT_TREE_AND_BOUNDED_STATE_AS_WORK_CLAIMS_CLOSED_"
            "ONLY__TRANSPOSED_NONUNIFORM_C5_LEAF_GENERATOR_OPEN"
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
            "frozen": (
                "frozen_5a5c_nonlinear_elliptic_orbit_product.json"
            ),
            "recurrence_ledger": (
                "elliptic_orbit_product_recurrence_ledger.json"
            ),
            "source_replay": "orbit_product_subset_source_replay.json",
            "exceptional": "orbit_product_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r113.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Constant prefix-product state does not imply constant leaf work.",
            "Observed support ratios are not an asymptotic theorem.",
            "No lower bound for arbitrary variable-coefficient recurrences is proved.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "REJECT_FIXED_TRANSLATION_ORBIT_AND_STANDARD_PRODUCT_TREE__"
            "REJECT_BOUNDED_PREFIX_STATE_AS_LOW_WORK__MERGE_WITH_IDEA006_"
            "IDEA012_R4_R97_R100__PRESERVE_TRANSPOSED_NONUNIFORM_C5_LEAF_"
            "GENERATOR__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "recurrence_ledger": ledger,
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
            "p1553_5a5c_nonlinear_elliptic_"
            "orbit_product_probe_report_r113.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_nonlinear_elliptic_orbit_product.json"
        ),
    )
    parser.add_argument(
        "--ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "elliptic_orbit_product_recurrence_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path("orbit_product_subset_source_replay.json"),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path("orbit_product_exceptional_controls.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r113.json"
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
    write_json(args.ledger_output, bundle["recurrence_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R113 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
