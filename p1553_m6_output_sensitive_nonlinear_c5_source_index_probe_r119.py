#!/usr/bin/env python3
"""Audit output-sized C5 indexes and preserve implicit membership circuits."""

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


SCHEMA = "p1553.m6_output_sensitive_nonlinear_c5_source_index.r119.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
GROUP_ORDER_EXPONENT = Fraction(5)
C_ATOM_EXPONENT = Fraction(3, 4)
C3_EXPONENT = 3 * C_ATOM_EXPONENT
C5_EXPONENT = 5 * C_ATOM_EXPONENT
C5_PAIR_COLLISION_EXPONENT = 2 * C5_EXPONENT - GROUP_ORDER_EXPONENT
C5_SUPPORT_RATIO_EXPONENT = C5_EXPONENT - GROUP_ORDER_EXPONENT

R118_PRODUCER = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_r118.py"
)
R118_PRODUCER_SHA256 = (
    "250fe039b89eb898d7efe8c576417720e13604671942eec53fcd2392c3bc28fa"
)
R118_REPORT = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_"
    "source_locator_probe_report_r118.json"
)
R118_REPORT_SHA256 = (
    "465f8ec0b328580a3d7ad808ff9dcddc278402d64ea269f642f7996af681d036"
)
R118_FROZEN = pathlib.Path("frozen_m6_nonlinear_c6_source_locator.json")
R118_FROZEN_SHA256 = (
    "9634d37e048c5addc560adb9c00963b156e18eac0c81c22c88ddb0aa92ee5dd2"
)
R118_COST = pathlib.Path("m6_nonlinear_c6_branch_and_ffe_cost_ledger.json")
R118_COST_SHA256 = (
    "221ce9ee8b505bf59ed1f0296f6156378baa5e41cd69b769327c249060631822"
)
R118_REPLAY = pathlib.Path("m6_nonlinear_c6_source_replay.json")
R118_REPLAY_SHA256 = (
    "c21101df0fd83ff8d79438762c4fca872e192d6ab9454ead59813971aa0b395d"
)
R118_CONTROLS = pathlib.Path("m6_nonlinear_c6_exceptional_controls.json")
R118_CONTROLS_SHA256 = (
    "af35028330ccf255c150211f02e6bd62ed2069c98127bc9948ffc14fa33eb270"
)
R118_LOGS = pathlib.Path("factor_logs_and_identical_descent_r118.json")
R118_LOGS_SHA256 = (
    "00bf012cffbea3f97e9a37554fbbe78428ecca56cb95fb5eb0e7480b7d743cb8"
)
R118_GATE = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_gate_r118.md"
)
R118_GATE_SHA256 = (
    "9ed7f59fb94dd1ecff54d8844068502183a7928c64e991c9a98b270ab8645b9d"
)
R118_PARENT = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_"
    "source_locator_probe_parent_report_r118.yaml"
)
R118_PARENT_SHA256 = (
    "3b7ec8dd0690412e85ebf4fc9abe26a132dc91edde08e747658731c1e43d5213"
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
R82_GATE = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_gate_r82.md"
)
R82_GATE_SHA256 = (
    "7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e"
)

Point = tuple[int, int] | None


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R118 = load_module("p1553_r118_for_r119", R118_PRODUCER)
R117 = R118.R117
R82 = R118.R82
R70 = R118.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r118_producer", R118_PRODUCER, R118_PRODUCER_SHA256),
        ("r118_report", R118_REPORT, R118_REPORT_SHA256),
        ("r118_frozen", R118_FROZEN, R118_FROZEN_SHA256),
        ("r118_cost", R118_COST, R118_COST_SHA256),
        ("r118_replay", R118_REPLAY, R118_REPLAY_SHA256),
        ("r118_controls", R118_CONTROLS, R118_CONTROLS_SHA256),
        ("r118_logs", R118_LOGS, R118_LOGS_SHA256),
        ("r118_gate", R118_GATE, R118_GATE_SHA256),
        ("r118_parent", R118_PARENT, R118_PARENT_SHA256),
        ("r82_producer", R82_PRODUCER, R82_PRODUCER_SHA256),
        ("r82_report", R82_REPORT, R82_REPORT_SHA256),
        ("r82_gate", R82_GATE, R82_GATE_SHA256),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R119 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def canonical_c5_endpoint_map(
    deck: Sequence[Point],
    curve: dict[str, Any],
) -> tuple[collections.Counter[Point], dict[Point, tuple[int, ...]]]:
    histogram: collections.Counter[Point] = collections.Counter()
    first: dict[Point, tuple[int, ...]] = {}
    for source in itertools.combinations_with_replacement(
        range(len(deck)),
        5,
    ):
        endpoint = R82.add_many(
            (deck[index] for index in source),
            curve,
        )
        histogram[endpoint] += 1
        first.setdefault(endpoint, source)
    expected = math.comb(len(deck) + 4, 5)
    if sum(histogram.values()) != expected:
        raise AssertionError("canonical C5 source count drifted")
    return histogram, first


def source_replays(
    target: Point,
    source: Sequence[int],
    deck: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    return R82.add_many((deck[index] for index in source), curve) == target


def collision_pair_count(histogram: Iterable[int]) -> int:
    return sum(value * (value - 1) // 2 for value in histogram)


def empty_target(
    support: set[Point],
    curve: dict[str, Any],
) -> Point:
    if None not in support:
        return None
    generator = R117.R81.curve_generator(curve)
    for scalar in range(1, 1025):
        target = R70.scalar_mul(scalar, generator, curve)
        if target not in support:
            return target
    raise AssertionError("unable to find finite empty C5 target")


def actual_r82_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    generator = R117.R81.curve_generator(curve)
    validation = R82.validate_family(curve, generator)
    if not all(validation.values()):
        raise AssertionError("R119 finite curve validation failed")
    _, deck, _, construction = R82.compact_factor_base(curve, offset)
    histogram, first = canonical_c5_endpoint_map(deck, curve)
    source_count = sum(histogram.values())
    support_size = len(histogram)
    pair_collisions = collision_pair_count(histogram.values())
    deficiency = source_count - support_size
    if deficiency > pair_collisions:
        raise AssertionError("support deficiency exceeds collision pairs")
    sources_replay = all(
        source_replays(target, source, deck, curve)
        for target, source in first.items()
    )
    missing = empty_target(set(histogram), curve)
    return {
        "control_id": (
            f"{curve['family_id']}_offset{offset}"
        ),
        "curve": {
            "field_prime": curve["field_prime"],
            "subgroup_order": curve["subgroup_order"],
            "cofactor": curve["cofactor"],
        },
        "validation": validation,
        "factor_base_construction": construction["construction"],
        "deck_size": len(deck),
        "canonical_c5_source_count": source_count,
        "expected_canonical_source_count": math.comb(len(deck) + 4, 5),
        "distinct_c5_endpoint_count": support_size,
        "support_deficiency": deficiency,
        "endpoint_collision_pair_count": pair_collisions,
        "support_deficiency_at_most_collision_pairs": (
            deficiency <= pair_collisions
        ),
        "canonical_endpoint_map_injective": support_size == source_count,
        "all_selected_sources_replay": sources_replay,
        "empty_target": point_json(missing),
        "empty_target_rejected_exactly": missing not in histogram,
        "candidate_scalar_labels_consumed": False,
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def exact_iid_collision_control() -> dict[str, Any]:
    prime = 11
    deck_size = 3
    sources = list(
        itertools.combinations_with_replacement(range(deck_size), 5)
    )
    total_assignments = prime**deck_size
    total_collision_pairs = 0
    all_deficiencies_bounded = True
    for labels in itertools.product(range(prime), repeat=deck_size):
        histogram = collections.Counter(
            sum(labels[index] for index in source) % prime
            for source in sources
        )
        pair_collisions = collision_pair_count(histogram.values())
        deficiency = len(sources) - len(histogram)
        total_collision_pairs += pair_collisions
        all_deficiencies_bounded &= deficiency <= pair_collisions
    source_pairs = math.comb(len(sources), 2)
    expected_total = prime ** (deck_size - 1) * source_pairs
    return {
        "group_order": prime,
        "deck_size": deck_size,
        "canonical_source_count": len(sources),
        "distinct_source_pair_count": source_pairs,
        "deck_assignments_enumerated": total_assignments,
        "observed_total_collision_pairs": total_collision_pairs,
        "expected_total_collision_pairs": expected_total,
        "average_collision_pairs_exact": (
            f"{source_pairs}/{prime}"
        ),
        "every_distinct_source_pair_collision_probability": (
            f"1/{prime}"
        ),
        "total_matches_pairwise_uniform_theorem": (
            total_collision_pairs == expected_total
        ),
        "support_deficiency_bounded_on_every_assignment": (
            all_deficiencies_bounded
        ),
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    actual = [
        actual_r82_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    iid = exact_iid_collision_control()
    return {
        "schema": "p1553.m6_c5_membership_exceptional_controls.r119.v1",
        "actual_r82_controls": actual,
        "exact_iid_collision_control": iid,
        "actual_control_count": len(actual),
        "all_actual_c5_endpoint_maps_injective": all(
            row["canonical_endpoint_map_injective"] for row in actual
        ),
        "all_actual_sources_replay": all(
            row["all_selected_sources_replay"] for row in actual
        ),
        "all_actual_empty_targets_rejected": all(
            row["empty_target_rejected_exactly"] for row in actual
        ),
        "iid_pair_collision_theorem_exact_on_control": (
            iid["total_matches_pairwise_uniform_theorem"]
            and iid["support_deficiency_bounded_on_every_assignment"]
        ),
        "finite_enumeration_receives_asymptotic_credit": False,
    }


def random_deck_support_theorem() -> dict[str, Any]:
    return {
        "model": (
            "C_1,...,C_n are independent uniform elements of Z/qZ, "
            "q prime and q>5"
        ),
        "canonical_source_count": "M=binom(n+4,5)",
        "pair_collision_probability": {
            "value": "1/q",
            "proof": (
                "For distinct multiplicity vectors a,b of weight five, "
                "d=a-b is nonzero modulo q. Condition on every C_i except "
                "one with d_i nonzero; d_i*C_i is uniform."
            ),
        },
        "expected_collision_pair_count": "binom(M,2)/q",
        "support_deficiency_bound": (
            "M-|5C| <= sum_fibers binom(fiber_size,2)"
        ),
        "markov_bound": (
            "Pr[|5C|<(1-epsilon)M] <= (M-1)/(2*epsilon*q)"
        ),
        "campaign_substitution": {
            "n": "B^(3/4+o(1))",
            "q": "B^(5+o(1))",
            "M_exponent_B": fraction_record(C5_EXPONENT),
            "expected_collision_pair_exponent_B": fraction_record(
                C5_PAIR_COLLISION_EXPONENT
            ),
            "M_over_q_exponent_B": fraction_record(
                C5_SUPPORT_RATIO_EXPONENT
            ),
            "conclusion": (
                "|5C|=(1-o(1))M=B^(15/4+o(1)) with probability 1-o(1)"
            ),
        },
        "transfer_to_filtered_r82_hash_deck_claimed": False,
        "generic_deterministic_deck_theorem_claimed": False,
    }


def membership_cost_ledger() -> dict[str, Any]:
    theorem = random_deck_support_theorem()
    rows = [
        {
            "route_id": "explicit_endpoint_membership_hash_and_source",
            "state_exponent_B": fraction_record(C5_EXPONENT),
            "query_exponent_B": fraction_record(Fraction(0)),
            "inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "inside_query_cap": True,
            "random_deck_model": True,
        },
        {
            "route_id": "radical_endpoint_polynomial_and_source_selector",
            "coefficient_degree_exponent_B": fraction_record(C5_EXPONENT),
            "state_exponent_B": fraction_record(C5_EXPONENT),
            "inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "fixed_sign_projective_source_selector_required": True,
            "random_deck_model": True,
        },
        {
            "route_id": "output_linear_radical_image_or_trace_compiler",
            "output_exponent_B": fraction_record(C5_EXPONENT),
            "work_and_state_at_least_output_words": True,
            "inside_setup_cap": C5_EXPONENT <= SETUP_CAP,
            "random_deck_model": True,
        },
        {
            "route_id": "universal_characteristic_zero_linear_shift_index",
            "state_exponent_B": fraction_record(GROUP_ORDER_EXPONENT),
            "inside_setup_cap": False,
            "inherited_prime_cyclotomic_rank": True,
        },
        {
            "route_id": "single_regular_boolean_target_section",
            "pole_degree_exponent_B": fraction_record(
                GROUP_ORDER_EXPONENT
            ),
            "inside_setup_cap_if_materialized": False,
            "straight_line_circuit_lower_bound_claimed": False,
        },
        {
            "route_id": "current_dinur_golovnev_k6_zero_query_index",
            "state_exponent_B": fraction_record(Fraction(33, 8)),
            "query_exponent_B": fraction_record(Fraction(0)),
            "inside_setup_cap": False,
            "integer_residue_transfer_to_prime_order_ec": False,
        },
    ]
    return {
        "schema": "p1553.m6_c5_membership_ffe_cost_ledger.r119.v1",
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_c5_query_exponent_B": fraction_record(Fraction(0)),
        },
        "random_deck_support_theorem": theorem,
        "routes": rows,
        "any_output_linear_route_meets_setup_cap": any(
            row.get("inside_setup_cap", False) for row in rows[:3]
        ),
        "known_data_structure_lower_bound_boundary": {
            "polylog_query_with_n3_space_unconditionally_excluded": False,
            "reason": (
                "Super-logarithmic static data-structure query lower bounds "
                "remain a major open problem; current kSUM-indexing results "
                "used here are upper bounds or conjectural boundaries."
            ),
            "cell_probe_or_ram_lower_bound_claimed": False,
        },
        "preserved_interface": (
            "sub-output implicit nonlinear C5 membership/source circuit "
            "with n^3 state and polylogarithmic exact query"
        ),
        "general_arithmetic_circuit_lower_bound_claimed": False,
        "candidate_work_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "p1553.m6_c5_membership_source_replay.r119.v1",
        "actual_control_count": controls["actual_control_count"],
        "all_actual_endpoint_maps_injective": controls[
            "all_actual_c5_endpoint_maps_injective"
        ],
        "all_actual_selected_sources_replay": controls[
            "all_actual_sources_replay"
        ],
        "all_actual_empty_targets_rejected": controls[
            "all_actual_empty_targets_rejected"
        ],
        "explicit_finite_dictionary_complete": True,
        "explicit_finite_dictionary_receives_asymptotic_credit": False,
        "inside_cap_implicit_membership_circuit_constructed": False,
        "inside_cap_implicit_source_recovery_constructed": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R118_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R118 nonclaim boundary drifted")
    controls = finite_controls()
    cost = membership_cost_ledger()
    replay = source_replay(controls)
    theorem = cost["random_deck_support_theorem"]
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r118_one_atom_interface_inherited": (
            inherited["admission"]["exact_reduction_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "iid_random_deck_c5_support_theorem_complete": (
            theorem["pair_collision_probability"]["value"] == "1/q"
        ),
        "campaign_exponent_substitution_complete": (
            theorem["campaign_substitution"]["M_exponent_B"]["exact"]
            == "15/4"
            and theorem["campaign_substitution"][
                "expected_collision_pair_exponent_B"
            ]["exact"]
            == "5/2"
        ),
        "exact_iid_finite_collision_control_complete": controls[
            "iid_pair_collision_theorem_exact_on_control"
        ],
        "eight_actual_r82_c5_support_controls_complete": (
            controls["actual_control_count"] == 8
        ),
        "all_actual_r82_c5_endpoint_maps_injective": controls[
            "all_actual_c5_endpoint_maps_injective"
        ],
        "all_actual_sources_and_empty_queries_replay": (
            controls["all_actual_sources_replay"]
            and controls["all_actual_empty_targets_rejected"]
        ),
        "output_linear_c5_index_costs_charged": (
            not cost["any_output_linear_route_meets_setup_cap"]
        ),
        "scope_preserves_implicit_index_and_excludes_general_lower_bound": (
            not cost[
                "known_data_structure_lower_bound_boundary"
            ]["cell_probe_or_ram_lower_bound_claimed"]
            and not cost["general_arithmetic_circuit_lower_bound_claimed"]
        ),
        "inside_cap_implicit_c5_membership_circuit_complete": False,
        "inside_cap_implicit_c5_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one sub-output implicit nonlinear C5 "
        "membership/source circuit with B^(9/4+o(1)) state and "
        "polylogarithmic exact query. It must not enumerate the "
        "B^(15/4) random-deck endpoint support, emit its radical or source "
        "selector coefficients, use a B^5 translate representation, or "
        "assume a unit-cost kSUM, resultant, gcd, character, DLP, root, or "
        "source oracle. Freeze one target computation graph, every S6 or "
        "fixed-sign group-law branch and remainder dimension, an exact "
        "empty certificate, and reverse five-source recovery; then compose "
        "with R118 before rank, logs, identical descent, memory, and cost."
    )
    frozen = {
        "schema": "p1553.frozen_m6_output_sensitive_c5_source_index.r119.v1",
        "source_bindings": source_binding_records(),
        "r118_required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_c5_query_exponent_B": fraction_record(Fraction(0)),
            "exact_empty_rejection_required": True,
            "five_occurrence_backpointers_required": True,
        },
        "random_deck_support_boundary": theorem,
        "closed_scoped_grammars": [
            "explicit C5 endpoint membership dictionary",
            "explicit radical endpoint polynomial and source selector",
            "every compiler linear in represented C5 output support",
            "universal characteristic-zero linear shift index",
            "single regular rational Boolean target section",
            "current bound k=6 indexing route",
        ],
        "preserved_interface": cost["preserved_interface"],
        "random_model_only": True,
        "general_lower_bound_claimed": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r119.v1",
        "r118_one_atom_branch_reduction_complete": True,
        "r119_random_deck_output_support_audit_complete": True,
        "inside_cap_implicit_c5_source_index_complete": False,
        "relation_independence_theorem_complete": False,
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
            "RANDOM_DECK_OUTPUT_SUPPORT_THEOREM_AND_SCOPED_OUTPUT_LINEAR_"
            "NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "IID_RANDOM_PRIME_CYCLIC_C5_SUPPORT_IS_B15O4_WITH_HIGH_"
            "PROBABILITY_BY_PAIR_COLLISION_FIRST_MOMENT__ALL_EIGHT_R82_HASH_"
            "DECKS_HAVE_CANONICAL_C5_ENDPOINT_INJECTIVITY_AND_EXACT_SOURCE_"
            "EMPTY_REPLAY__OUTPUT_LINEAR_ENDPOINT_DICTIONARY_RADICAL_SELECTOR_"
            "AND_IMAGE_COMPILERS_EXCEED_B9O4_SETUP__CURRENT_KSUM_INDEXING_"
            "AND_KNOWN_LOWER_BOUNDS_DO_NOT_CLOSE_SUBOUTPUT_NONLINEAR_INDEX__"
            "IMPLICIT_POLYLOG_C5_MEMBERSHIP_SOURCE_CIRCUIT_OPEN__NO_RANK_"
            "LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "random_deck_support_theorem": theorem,
        "finite_evidence": {
            "actual_r82_control_count": controls["actual_control_count"],
            "all_actual_c5_endpoint_maps_injective": controls[
                "all_actual_c5_endpoint_maps_injective"
            ],
            "all_actual_sources_replay": controls[
                "all_actual_sources_replay"
            ],
            "all_actual_empty_targets_rejected": controls[
                "all_actual_empty_targets_rejected"
            ],
            "iid_pair_collision_theorem_exact_on_control": controls[
                "iid_pair_collision_theorem_exact_on_control"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "random_model_theorem_admitted": True,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_m6_output_sensitive_c5_source_index.json",
            "cost": "m6_c5_membership_ffe_cost_ledger.json",
            "source_replay": "m6_c5_membership_source_replay.json",
            "exceptional_controls": (
                "m6_c5_membership_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r119.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The asymptotic support theorem is for an iid random deck.",
            "Transfer to the filtered R82 hash process is not proved.",
            "Finite endpoint enumeration receives no asymptotic credit.",
            "Output support size is not an arithmetic-circuit lower bound.",
            "No cell-probe or RAM lower bound for the implicit index is claimed.",
            "No implicit C5 circuit, relation rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_IID_RANDOM_DECK_C5_SUPPORT_THEOREM_AND_FINITE_R82_SOURCE_"
            "CONTROLS_ONLY__REJECT_OUTPUT_LINEAR_ENDPOINT_DICTIONARY_RADICAL_"
            "SELECTOR_AND_IMAGE_COMPILERS_AT_FROZEN_SETUP__PRESERVE_SUBOUTPUT_"
            "IMPLICIT_NONLINEAR_C5_MEMBERSHIP_SOURCE_CIRCUIT__NO_LOCATOR__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_m6_output_sensitive_nonlinear_c5_"
            "source_index_probe_report_r119.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_m6_output_sensitive_c5_source_index.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path("m6_c5_membership_ffe_cost_ledger.json"),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path("m6_c5_membership_source_replay.json"),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "m6_c5_membership_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r119.json"
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
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R119 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
