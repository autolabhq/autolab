#!/usr/bin/env python3
"""Audit small-mode predicates against two-atom C5 progressions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.torus_c5_two_atom_geometric_progression.r134.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
DECK_EXPONENT_B = Fraction(3, 4)
C5_EXPONENT_B = Fraction(15, 4)
SOURCE_DEGREE = 5
COLOR_ACCEPTANCE_THRESHOLD = 2
PROGRESSION_LENGTH = SOURCE_DEGREE + 1

R133_PRODUCER = pathlib.Path(
    "p1553_torus_c5_sparse_monomial_root_bound_probe_r133.py"
)
R133_PRODUCER_SHA256 = (
    "b53985696fb7a6b270b7e3cdffa50f50b6801cdc0569a567abf1486064d52bc7"
)
R133_REPORT = pathlib.Path(
    "p1553_torus_c5_sparse_monomial_root_bound_"
    "probe_report_r133.json"
)
R133_REPORT_SHA256 = (
    "5b02982dfdce90645db52e36ba6063a40a6476307ec6a269d3c0c4311d64e8d3"
)
R133_FROZEN = pathlib.Path(
    "frozen_torus_c5_sparse_monomial_root_bound.json"
)
R133_FROZEN_SHA256 = (
    "28e2827b522efd71b2abaebca377b169b9744263541499bd9a262eaf9d4f911b"
)
R133_COST = pathlib.Path(
    "torus_c5_sparse_monomial_root_bound_cost_ledger.json"
)
R133_COST_SHA256 = (
    "7bc714f7c6b763c9347f78d094374fbf26442473bbdd58978e261de8494485c0"
)
R133_REPLAY = pathlib.Path(
    "torus_c5_sparse_monomial_root_bound_replay.json"
)
R133_REPLAY_SHA256 = (
    "958baefed1027c0bf58d6ff666af3fad2c6546d9624acdc5bef9b8aac9c09dda"
)
R133_CONTROLS = pathlib.Path(
    "torus_c5_sparse_monomial_root_bound_controls.json"
)
R133_CONTROLS_SHA256 = (
    "5bf39fb2ec030c9300f1913b09d0d28800d0330d2eca26478ad1a322d3503938"
)
R133_LOGS = pathlib.Path("factor_logs_and_identical_descent_r133.json")
R133_LOGS_SHA256 = (
    "757bb97a1adb867fb35455a17f822d825537cc59a9029459ad0c75f4e96a3e7a"
)
R133_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_sparse_monomial_root_bound_probe_r133.py"
)
R133_TEST_SHA256 = (
    "e5a350fa03fbf254f930ce9c52f1c36a648d234cba8617e7e6c8b533d6ab2891"
)
R133_GATE = pathlib.Path(
    "p1553_torus_c5_sparse_monomial_root_bound_probe_gate_r133.md"
)
R133_GATE_SHA256 = (
    "bfeb6cad6781ef2b63b36821b5ca4d02a7424353a29b99ed53aa974c580f5ffe"
)
R133_PARENT = pathlib.Path(
    "p1553_torus_c5_sparse_monomial_root_bound_"
    "probe_parent_report_r133.yaml"
)
R133_PARENT_SHA256 = (
    "236725e81e4a05451b7ea2ad5d0ef6ae51ed019d9916e7b8d0e9be5aaa4ebceb"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R133 = load_module("p1553_r133_for_r134", R133_PRODUCER)
R132 = R133.R132
R131 = R133.R131
R129 = R133.R129
R126 = R133.R126
R121 = R133.R121
R82 = R133.R82
Field = R131.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r133_producer", R133_PRODUCER, R133_PRODUCER_SHA256),
        ("r133_report", R133_REPORT, R133_REPORT_SHA256),
        ("r133_frozen", R133_FROZEN, R133_FROZEN_SHA256),
        ("r133_cost", R133_COST, R133_COST_SHA256),
        ("r133_replay", R133_REPLAY, R133_REPLAY_SHA256),
        ("r133_controls", R133_CONTROLS, R133_CONTROLS_SHA256),
        ("r133_logs", R133_LOGS, R133_LOGS_SHA256),
        ("r133_test", R133_TEST, R133_TEST_SHA256),
        ("r133_gate", R133_GATE, R133_GATE_SHA256),
        ("r133_parent", R133_PARENT, R133_PARENT_SHA256),
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
        raise AssertionError(f"R134 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def progression_sources(left: int, right: int) -> tuple[tuple[int, ...], ...]:
    if left == right:
        raise ValueError("progression atoms must be distinct")
    return tuple(
        tuple(sorted((left,) * (SOURCE_DEGREE - j) + (right,) * j))
        for j in range(PROGRESSION_LENGTH)
    )


def progression_targets(
    left: Fp2,
    right: Fp2,
    field: Field,
) -> tuple[Fp2, ...]:
    return tuple(
        field.mul(
            field.pow(left, SOURCE_DEGREE - j),
            field.pow(right, j),
        )
        for j in range(PROGRESSION_LENGTH)
    )


def sample_modes(subgroup_order: int) -> tuple[int, ...]:
    modes = (0, 1, 2, 3, subgroup_order - 2, subgroup_order - 1)
    if len(set(modes)) != PROGRESSION_LENGTH:
        raise AssertionError("sample modes collided")
    return modes


def progression_witness(
    field: Field,
    deck: tuple[Fp2, ...],
    c5_by_source: dict[tuple[int, ...], Fp2],
    color: int,
    part: tuple[int, ...],
    subgroup_order: int,
) -> dict[str, Any]:
    base = {
        "color": color,
        "part_indices": list(part),
        "part_size": len(part),
        "witness_available": len(part) >= 2,
        "finite_control_receives_asymptotic_credit": False,
    }
    if len(part) < 2:
        return {
            **base,
            "unavailable_reason": "finite_color_part_has_fewer_than_two_atoms",
        }
    left_index, right_index = part[:2]
    left = deck[left_index]
    right = deck[right_index]
    ratio = field.mul(right, field.inv(left))
    sources = progression_sources(left_index, right_index)
    targets = progression_targets(left, right, field)
    nodes = tuple(
        field.pow(ratio, exponent)
        for exponent in sample_modes(subgroup_order)
    )
    determinant = R131.vandermonde_determinant(nodes, field)
    return {
        **base,
        "left_index": left_index,
        "right_index": right_index,
        "source_count": len(sources),
        "target_count": len(targets),
        "sources": [list(source) for source in sources],
        "targets": [field.json(target) for target in targets],
        "ratio": field.json(ratio),
        "ratio_nonidentity": ratio != field.one,
        "ratio_to_q_is_identity": (
            field.pow(ratio, subgroup_order) == field.one
        ),
        "ratio_has_exact_prime_order_q": (
            ratio != field.one
            and field.pow(ratio, subgroup_order) == field.one
        ),
        "targets_distinct": len(set(targets)) == len(targets),
        "geometric_progression_recurrence_exact": all(
            targets[j + 1] == field.mul(targets[j], ratio)
            for j in range(SOURCE_DEGREE)
        ),
        "all_sources_in_c5_replay": all(
            source in c5_by_source for source in sources
        ),
        "all_source_values_match_progression_targets": all(
            c5_by_source[source] == target
            for source, target in zip(sources, targets)
        ),
        "all_sources_have_color_multiplicity_five": all(
            R131.source_color_multiplicity(source, color)
            == SOURCE_DEGREE
            for source in sources
        ),
        "all_sources_accepted_by_color": all(
            R131.source_color_multiplicity(source, color)
            >= COLOR_ACCEPTANCE_THRESHOLD
            for source in sources
        ),
        "sample_mode_count": len(nodes),
        "sample_mode_nodes_distinct": len(set(nodes)) == len(nodes),
        "sample_vandermonde_determinant": field.json(determinant),
        "sample_vandermonde_determinant_nonzero": (
            determinant != field.zero
        ),
    }


def finite_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, SOURCE_DEGREE)
    c5_by_source = {source: value for source, value in c5}
    witnesses = [
        progression_witness(
            field,
            deck,
            c5_by_source,
            color,
            part,
            curve["subgroup_order"],
        )
        for color, part in enumerate(R129.balanced_four_parts(len(deck)))
        if part
    ]
    available = [row for row in witnesses if row["witness_available"]]
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "subgroup_order_probable_prime": R82.R70.is_prime(
            curve["subgroup_order"]
        ),
        "deck_size": len(deck),
        "c5_source_count": len(c5),
        "active_color_count": len(witnesses),
        "available_progression_witness_count": len(available),
        "color_witnesses": witnesses,
        "all_available_ratios_have_exact_prime_order": all(
            row["ratio_has_exact_prime_order_q"] for row in available
        ),
        "all_available_targets_distinct": all(
            row["targets_distinct"] for row in available
        ),
        "all_available_progressions_exact": all(
            row["geometric_progression_recurrence_exact"]
            for row in available
        ),
        "all_available_sources_replay": all(
            row["all_sources_in_c5_replay"]
            and row["all_source_values_match_progression_targets"]
            for row in available
        ),
        "all_available_sources_accepted": all(
            row["all_sources_have_color_multiplicity_five"]
            and row["all_sources_accepted_by_color"]
            for row in available
        ),
        "all_available_sample_vandermonde_determinants_nonzero": all(
            row["sample_mode_nodes_distinct"]
            and row["sample_vandermonde_determinant_nonzero"]
            for row in available
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    witness_count = sum(
        row["available_progression_witness_count"] for row in controls
    )
    return {
        "schema": (
            "p1553.torus_c5_two_atom_geometric_progression_"
            "controls.r134.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "active_color_control_count": sum(
            row["active_color_count"] for row in controls
        ),
        "available_progression_witness_count": witness_count,
        "all_subgroup_orders_probable_prime": all(
            row["subgroup_order_probable_prime"] for row in controls
        ),
        "all_available_ratios_have_exact_prime_order": (
            witness_count > 0
            and all(
                row["all_available_ratios_have_exact_prime_order"]
                for row in controls
            )
        ),
        "all_available_targets_distinct": (
            witness_count > 0
            and all(
                row["all_available_targets_distinct"] for row in controls
            )
        ),
        "all_available_progressions_exact": (
            witness_count > 0
            and all(
                row["all_available_progressions_exact"] for row in controls
            )
        ),
        "all_available_sources_replay": (
            witness_count > 0
            and all(
                row["all_available_sources_replay"] for row in controls
            )
        ),
        "all_available_sources_accepted": (
            witness_count > 0
            and all(
                row["all_available_sources_accepted"] for row in controls
            )
        ),
        "all_available_sample_vandermonde_determinants_nonzero": (
            witness_count > 0
            and all(
                row[
                    "all_available_sample_vandermonde_determinants_nonzero"
                ]
                for row in controls
            )
        ),
        "all_active_finite_colors_have_witness": all(
            row["available_progression_witness_count"]
            == row["active_color_count"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "two_atom_progression": (
            "Let x and y be two distinct atoms in one color part. For "
            "j=0,...,5, the degree-five source x^(5-j)y^j is accepted by "
            "that color and has target z_j=x^5*(y/x)^j. These are six "
            "distinct points of a geometric progression because y/x has "
            "prime order q."
        ),
        "vandermonde_obstruction": (
            "Let f(z)=sum_{i=1}^t c_i z^(e_i) have 1<=t<=6 nonzero "
            "coefficients and distinct exponents modulo q. If f vanishes "
            "on z_j for j=0,...,t-1, then the coefficient vector lies in "
            "the kernel of the Vandermonde matrix with nodes "
            "(y/x)^(e_i). The nodes are distinct, so its determinant "
            "prod_{i<k}((y/x)^(e_k)-(y/x)^(e_i)) is nonzero. Thus all "
            "coefficients would be zero, a contradiction."
        ),
        "rational_predicate_consequence": (
            "The same obstruction applies separately to a represented "
            "at-most-six-mode numerator-zero or denominator-zero set. No "
            "such single zero or pole set can contain an entire color "
            "acceptance support."
        ),
        "field_independent": True,
        "random_deck_model_required": False,
        "candidate_discrete_log_required": False,
        "source_degree": SOURCE_DEGREE,
        "progression_length": PROGRESSION_LENGTH,
        "maximum_excluded_mode_count": PROGRESSION_LENGTH,
        "first_mode_count_not_excluded": PROGRESSION_LENGTH + 1,
        "structured_factor_base_application": (
            "At deck size n=B^(3/4+o(1)), every balanced color part has "
            "Theta(n) atoms and hence at least two. Therefore every "
            "asymptotic color acceptance support contains the required "
            "six-term progression deterministically."
        ),
        "scope_limits": [
            "one represented polynomial zero predicate",
            "one represented rational numerator-zero or denominator-zero set",
            "at most six distinct exponent modes modulo q",
            "the inherited degree-five color acceptance rule",
        ],
        "not_covered": [
            "seven or more represented modes",
            "multiple-predicate or adaptive decision DAGs",
            "nonzero-value tests and coordinate comparisons",
            "polynomials with large expansion but compact straight-line programs",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_two_atom_geometric_progression_"
            "cost_ledger.r134.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "one_to_six_mode_single_color_zero_predicate",
                "random_model_required": False,
                "structured_factor_base_theorem": True,
                "status": (
                    "rejected_deterministically_by_two_atom_progression"
                ),
            },
            {
                "route_id": "one_to_six_mode_single_color_pole_predicate",
                "random_model_required": False,
                "structured_factor_base_theorem": True,
                "status": (
                    "rejected_deterministically_by_two_atom_progression"
                ),
            },
            {
                "route_id": "seven_or_more_mode_extension_zero_predicate",
                "covered_by_six_point_vandermonde_witness": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "multiple_small_mode_predicate_dag",
                "single_predicate_obstruction_sufficient": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
            {
                "route_id": "low_slp_expanded_extension_predicate",
                "represented_mode_bound_applicable": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonzero_value_frobenius_coordinate_dag",
                "zero_set_obstruction_sufficient": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R133_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R133 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "r133_five_mode_interface_inherited": (
            inherited["admission"][
                "deterministic_one_to_four_mode_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "two_atom_progression_explicit": (
            theorem["progression_length"] == 6
        ),
        "prime_order_ratio_argument_explicit": (
            "prime order q" in theorem["two_atom_progression"]
        ),
        "vandermonde_kernel_argument_explicit": (
            "Vandermonde matrix" in theorem["vandermonde_obstruction"]
        ),
        "one_through_six_modes_excluded_deterministically": (
            theorem["maximum_excluded_mode_count"] == 6
        ),
        "seven_modes_first_not_excluded": (
            theorem["first_mode_count_not_excluded"] == 7
        ),
        "structured_factor_base_application_explicit": (
            "deterministically"
            in theorem["structured_factor_base_application"]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "twelve_actual_progression_witnesses_complete": (
            controls["available_progression_witness_count"] == 12
        ),
        "all_available_progression_replays_exact": (
            controls["all_available_ratios_have_exact_prime_order"]
            and controls["all_available_targets_distinct"]
            and controls["all_available_progressions_exact"]
            and controls["all_available_sources_replay"]
            and controls["all_available_sources_accepted"]
            and controls[
                "all_available_sample_vandermonde_determinants_nonzero"
            ]
        ),
        "finite_fixture_gaps_receive_no_credit": (
            not controls["all_active_finite_colors_have_witness"]
            and not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "seven_mode_multi_dag_and_slp_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "seven_or_more_mode_extension_zero_predicate",
                "multiple_small_mode_predicate_dag",
                "low_slp_expanded_extension_predicate",
                "nonzero_value_frobenius_coordinate_dag",
            )
        ),
        "inside_cap_asymmetric_predicate_complete": False,
        "inside_cap_five_source_index_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    next_action = (
        "Test the first surviving asymmetric classes after the deterministic "
        "six-mode obstruction: a seven-or-more-mode F_(p^2) predicate, a "
        "multiple-small-predicate decision DAG, a nonzero-value Frobenius-"
        "coordinate branch, or a high-expansion low-SLP predicate. Freeze "
        "all coefficients and nodes, replay exact positive/empty paths and "
        "C2+C3 sources, fit B^(9/4+o(1)) state and polylogarithmic arbitrary-"
        "target work, avoid field DLP, and charge rank, logs, identical "
        "descent, memory, field operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_two_atom_geometric_progression.r134.v1"
        ),
        "source_bindings": source_binding_records(),
        "required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
            "exact_empty_rejection_required": True,
            "five_projective_backpointers_required": True,
            "field_discrete_logarithms_allowed": False,
        },
        "closed_scoped_grammars": [
            "single represented color-zero predicates with at most six modes",
            "single represented color-pole predicates with at most six modes",
        ],
        "preserved_interface": (
            "seven-or-more-mode predicate, multiple-predicate DAG, nonzero-"
            "value Frobenius-coordinate branch, or high-expansion low-SLP "
            "predicate"
        ),
        "random_deck_model_required": False,
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_two_atom_geometric_progression_"
            "replay.r134.v1"
        ),
        "actual_control_count": controls["control_count"],
        "active_color_control_count": controls[
            "active_color_control_count"
        ],
        "available_progression_witness_count": controls[
            "available_progression_witness_count"
        ],
        "all_available_ratios_have_exact_prime_order": controls[
            "all_available_ratios_have_exact_prime_order"
        ],
        "all_available_progressions_exact": controls[
            "all_available_progressions_exact"
        ],
        "all_available_sources_replay": controls[
            "all_available_sources_replay"
        ],
        "all_available_sources_accepted": controls[
            "all_available_sources_accepted"
        ],
        "finite_fixture_gaps_receive_credit": False,
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r134.v1",
        "r133_sparse_monomial_root_bound_audit_complete": True,
        "r134_two_atom_progression_audit_complete": True,
        "inside_cap_target_specialized_source_index_complete": False,
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
    classification = (
        "DEGREE_FIVE_TWO_ATOM_SOURCES_FORCE_SIX_TERM_GEOMETRIC_"
        "PROGRESSIONS_IN_EVERY_ASYMPTOTIC_COLOR__PRIME_ORDER_RATIO_MAKES_"
        "ALL_AT_MOST_SIX_MODE_EVALUATION_MATRICES_VANDERMONDE__SINGLE_"
        "REPRESENTED_ZERO_OR_POLE_PREDICATES_THROUGH_SIX_MODES_REJECTED_"
        "DETERMINISTICALLY_ON_STRUCTURED_FACTOR_BASE__TWELVE_ACTUAL_"
        "PROGRESSION_WITNESSES_EXACT__SEVEN_MODE_MULTI_PREDICATE_NONZERO_"
        "VALUE_LOW_SLP_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "DETERMINISTIC_STRUCTURED_TWO_ATOM_PROGRESSION_OBSTRUCTION_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "active_color_control_count": controls[
                "active_color_control_count"
            ],
            "available_progression_witness_count": controls[
                "available_progression_witness_count"
            ],
            "all_available_progressions_exact": controls[
                "all_available_progressions_exact"
            ],
            "all_available_sources_replay": controls[
                "all_available_sources_replay"
            ],
            "all_active_finite_colors_have_witness": controls[
                "all_active_finite_colors_have_witness"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "deterministic_one_to_six_mode_negative_admitted": True,
            "structured_factor_base_progression_theorem_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_two_atom_geometric_progression.json"
            ),
            "cost": (
                "torus_c5_two_atom_geometric_progression_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_two_atom_geometric_progression_replay.json"
            ),
            "controls": (
                "torus_c5_two_atom_geometric_progression_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r134.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The Vandermonde obstruction is a scoped elementary theorem, not a general circuit lower bound.",
            "Finite colors with fewer than two atoms have no witness and receive no credit.",
            "Seven-or-more-mode represented predicates remain open.",
            "Multiple-predicate, nonzero-value, and low-SLP DAGs remain open.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_DETERMINISTIC_TWO_ATOM_SIX_POINT_PROGRESSION_THEOREM__"
            "REJECT_SINGLE_REPRESENTED_ZERO_OR_POLE_PREDICATES_THROUGH_SIX_"
            "MODES_ON_ASYMPTOTIC_STRUCTURED_COLORS__ADMIT_TWELVE_EXACT_"
            "FINITE_WITNESSES_WITHOUT_FIXTURE_GAP_CREDIT__PRESERVE_SEVEN_"
            "MODE_MULTI_PREDICATE_NONZERO_VALUE_AND_LOW_SLP_ROUTES__NO_"
            "LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_two_atom_geometric_progression_"
            "probe_report_r134.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_two_atom_geometric_progression.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_two_atom_geometric_progression_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_two_atom_geometric_progression_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_two_atom_geometric_progression_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r134.json"),
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
        f"R134 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
