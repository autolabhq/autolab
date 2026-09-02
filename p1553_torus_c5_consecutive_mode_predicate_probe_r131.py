#!/usr/bin/env python3
"""Audit consecutive-mode predicates in the actual order-two pairing fields."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_consecutive_mode_predicate.r131.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
DECK_EXPONENT = Fraction(3, 4)
C5_EXPONENT = Fraction(15, 4)

R130_PRODUCER = pathlib.Path(
    "p1553_torus_c5_sparse_fourier_predicate_transfer_probe_r130.py"
)
R130_PRODUCER_SHA256 = (
    "b990d602bc7b15a90bea8bbd1cc9bc600d7cbcab73caeaedecd7e666a95c55a1"
)
R130_REPORT = pathlib.Path(
    "p1553_torus_c5_sparse_fourier_predicate_transfer_"
    "probe_report_r130.json"
)
R130_REPORT_SHA256 = (
    "afcb0fd152a95535b5f004d9d07774a9a375ea548b79919f47a6d15682dfdeda"
)
R130_FROZEN = pathlib.Path(
    "frozen_torus_c5_sparse_fourier_predicate_transfer.json"
)
R130_FROZEN_SHA256 = (
    "1b4d4b5437b71d6ff9204296145bf88cf68a16d52155aa469bda394a8a61f204"
)
R130_COST = pathlib.Path(
    "torus_c5_sparse_fourier_predicate_transfer_cost_ledger.json"
)
R130_COST_SHA256 = (
    "552c3cc83c1b1635bd046845c2f4e8af6cd88c934918bfbb9303fc26eda67e8b"
)
R130_REPLAY = pathlib.Path(
    "torus_c5_sparse_fourier_predicate_transfer_replay.json"
)
R130_REPLAY_SHA256 = (
    "11eba087140a633fe286c35ecd41fa2eb36ed9ae2f12e02eb1dd6e5631b1ae1c"
)
R130_CONTROLS = pathlib.Path(
    "torus_c5_sparse_fourier_predicate_transfer_controls.json"
)
R130_CONTROLS_SHA256 = (
    "2c30b825eaf7a88b66be04dd29647dfc40aab2016db43c5c9c5957dbc8c95b1d"
)
R130_LOGS = pathlib.Path("factor_logs_and_identical_descent_r130.json")
R130_LOGS_SHA256 = (
    "b0c80b3af457cd080ae1cbf6e703e22fe831085e638fade844ef6d41fafc30a6"
)
R130_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_sparse_fourier_predicate_transfer_probe_r130.py"
)
R130_TEST_SHA256 = (
    "5d33f9e894264362d0c663a45b7bdbdf8edf2c854cc9bc13880adedc3ea4564f"
)
R130_GATE = pathlib.Path(
    "p1553_torus_c5_sparse_fourier_predicate_transfer_probe_gate_r130.md"
)
R130_GATE_SHA256 = (
    "5737c29c69068dfb01ff8bd9b6aab0f210b362e4060f2d19d23e91fe474de640"
)
R130_PARENT = pathlib.Path(
    "p1553_torus_c5_sparse_fourier_predicate_transfer_"
    "probe_parent_report_r130.yaml"
)
R130_PARENT_SHA256 = (
    "b14d2ac67d0d702f19f223509193e7b107866bede8eec975fa2c622189565824"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R130 = load_module("p1553_r130_for_r131", R130_PRODUCER)
R129 = R130.R129
R126 = R129.R126
R121 = R129.R121
R82 = R129.R82
Field = R129.Field
Fp2 = tuple[int, int]
Source = tuple[int, ...]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r130_producer", R130_PRODUCER, R130_PRODUCER_SHA256),
        ("r130_report", R130_REPORT, R130_REPORT_SHA256),
        ("r130_frozen", R130_FROZEN, R130_FROZEN_SHA256),
        ("r130_cost", R130_COST, R130_COST_SHA256),
        ("r130_replay", R130_REPLAY, R130_REPLAY_SHA256),
        ("r130_controls", R130_CONTROLS, R130_CONTROLS_SHA256),
        ("r130_logs", R130_LOGS, R130_LOGS_SHA256),
        ("r130_test", R130_TEST, R130_TEST_SHA256),
        ("r130_gate", R130_GATE, R130_GATE_SHA256),
        ("r130_parent", R130_PARENT, R130_PARENT_SHA256),
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
        raise AssertionError(f"R131 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def color_acceptance_count(deck_size: int, color_size: int) -> int:
    """Count degree-five multisets containing at least two colored atoms."""
    complement = deck_size - color_size
    total = math.comb(deck_size + 4, 5)
    no_colored_atom = math.comb(complement + 4, 5)
    one_colored_atom = (
        color_size * math.comb(complement + 3, 4)
    )
    return total - no_colored_atom - one_colored_atom


def source_color_multiplicity(
    source: Iterable[int],
    color: int,
) -> int:
    return sum(index % 4 == color for index in source)


def vandermonde_determinant(
    values: tuple[Fp2, ...],
    field: Field,
) -> Fp2:
    determinant = field.one
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            determinant = field.mul(
                determinant,
                field.sub(right, left),
            )
    return determinant


def dense_root_annihilator(
    roots: tuple[Fp2, ...],
    field: Field,
) -> tuple[Fp2, ...]:
    return R121.root_annihilator(roots, field)


def evaluate_dense_polynomial(
    coefficients: tuple[Fp2, ...],
    value: Fp2,
    field: Field,
) -> Fp2:
    return R121.evaluate_monic(coefficients, value, field)


def choose_colored_c2_source(source: Source) -> tuple[int, Source, Source]:
    counts = Counter(index % 4 for index in source)
    color = min(
        color for color, multiplicity in counts.items() if multiplicity >= 2
    )
    selected_positions = [
        position
        for position, index in enumerate(source)
        if index % 4 == color
    ][:2]
    pair = tuple(source[position] for position in selected_positions)
    triple = tuple(
        index
        for position, index in enumerate(source)
        if position not in selected_positions
    )
    return color, pair, triple


def consecutive_mode_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c2 = R126.source_products(field, deck, 2)
    c3 = R126.source_products(field, deck, 3)
    c5 = R126.source_products(field, deck, 5)
    c2_by_source = {source: value for source, value in c2}
    c3_by_source = {source: value for source, value in c3}
    c5_by_source = {source: value for source, value in c5}
    c5_by_value = {value: source for source, value in c5}
    supports_injective = (
        len(c2_by_source) == len(c2)
        and len(c3_by_source) == len(c3)
        and len(c5_by_source) == len(c5)
        and len(c5_by_value) == len(c5)
    )
    branch_set = set(R129.optimal_degree_two_branches(len(deck)))
    source_replays = []
    for source, target in c5:
        color, pair, triple = choose_colored_c2_source(source)
        pair = tuple(sorted(pair))
        triple = tuple(sorted(triple))
        replayed = field.mul(
            c2_by_source[pair],
            c3_by_source[triple],
        )
        source_replays.append(
            {
                "source": list(source),
                "target": field.json(target),
                "selected_color": color,
                "pair_source": list(pair),
                "triple_source": list(triple),
                "pair_is_optimal_branch": pair in branch_set,
                "product_replays": replayed == target,
            }
        )

    color_controls = []
    for color, part in enumerate(R129.balanced_four_parts(len(deck))):
        if not part:
            continue
        accepted = tuple(
            (source, target)
            for source, target in c5
            if source_color_multiplicity(source, color) >= 2
        )
        roots = tuple(target for _, target in accepted)
        expected_count = color_acceptance_count(len(deck), len(part))
        coefficients = dense_root_annihilator(roots, field)
        observed_zero_sources = tuple(
            source
            for source, target in c5
            if evaluate_dense_polynomial(
                coefficients,
                target,
                field,
            )
            == field.zero
        )
        expected_zero_sources = tuple(source for source, _ in accepted)
        determinant = vandermonde_determinant(roots, field)
        color_controls.append(
            {
                "color": color,
                "part_indices": list(part),
                "part_size": len(part),
                "accepted_target_count": len(roots),
                "accepted_target_count_formula": expected_count,
                "count_formula_exact": len(roots) == expected_count,
                "targets_distinct": len(set(roots)) == len(roots),
                "vandermonde_size": len(roots),
                "vandermonde_determinant": field.json(determinant),
                "vandermonde_determinant_nonzero": (
                    determinant != field.zero
                ),
                "dense_annihilator_serialized_coefficient_count": len(
                    coefficients
                ),
                "dense_annihilator_nonzero_coefficient_count": sum(
                    coefficient != field.zero
                    for coefficient in coefficients
                ),
                "annihilator_degree": len(coefficients) - 1,
                "annihilator_zero_sources": [
                    list(source) for source in observed_zero_sources
                ],
                "annihilator_zero_set_exact": (
                    observed_zero_sources == expected_zero_sources
                ),
                "zero_target_rejected": (
                    evaluate_dense_polynomial(
                        coefficients,
                        field.zero,
                        field,
                    )
                    != field.zero
                ),
            }
        )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "characteristic_mod_subgroup_order": (
            field.p % curve["subgroup_order"]
        ),
        "characteristic_order_two": (
            field.p % curve["subgroup_order"]
            == curve["subgroup_order"] - 1
        ),
        "deck_size": len(deck),
        "c2_source_count": len(c2),
        "c3_source_count": len(c3),
        "c5_source_count": len(c5),
        "all_c2_c3_c5_source_products_injective": supports_injective,
        "active_color_count": len(color_controls),
        "color_controls": color_controls,
        "all_color_count_formulas_exact": all(
            row["count_formula_exact"] for row in color_controls
        ),
        "all_color_targets_distinct": all(
            row["targets_distinct"] for row in color_controls
        ),
        "all_vandermonde_determinants_nonzero": all(
            row["vandermonde_determinant_nonzero"]
            for row in color_controls
        ),
        "all_dense_annihilator_zero_sets_exact": all(
            row["annihilator_zero_set_exact"] for row in color_controls
        ),
        "all_zero_targets_rejected": all(
            row["zero_target_rejected"] for row in color_controls
        ),
        "all_sources_have_repeated_color": (
            len(source_replays) == len(c5)
        ),
        "all_selected_pairs_are_optimal_branches": all(
            row["pair_is_optimal_branch"] for row in source_replays
        ),
        "all_c2_c3_source_products_replay": all(
            row["product_replays"] for row in source_replays
        ),
        "source_replays": source_replays,
        "candidate_discrete_logs_consumed": False,
        "verifier_source_labels_receive_algorithmic_credit": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        consecutive_mode_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    color_rows = [
        color
        for control in controls
        for color in control["color_controls"]
    ]
    return {
        "schema": (
            "p1553.torus_c5_consecutive_mode_predicate_"
            "controls.r131.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "active_color_control_count": len(color_rows),
        "all_fields_have_order_two_characteristic": all(
            row["characteristic_order_two"] for row in controls
        ),
        "all_supports_injective": all(
            row["all_c2_c3_c5_source_products_injective"]
            for row in controls
        ),
        "all_color_count_formulas_exact": all(
            row["all_color_count_formulas_exact"] for row in controls
        ),
        "all_color_targets_distinct": all(
            row["all_color_targets_distinct"] for row in controls
        ),
        "all_vandermonde_determinants_nonzero": all(
            row["all_vandermonde_determinants_nonzero"]
            for row in controls
        ),
        "all_dense_annihilator_zero_sets_exact": all(
            row["all_dense_annihilator_zero_sets_exact"]
            for row in controls
        ),
        "all_zero_targets_rejected": all(
            row["all_zero_targets_rejected"] for row in controls
        ),
        "all_selected_pairs_are_optimal_branches": all(
            row["all_selected_pairs_are_optimal_branches"]
            for row in controls
        ),
        "all_c2_c3_source_products_replay": all(
            row["all_c2_c3_source_products_replay"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "field_independent_root_bound": (
            "Over any field, a nonzero polynomial of degree d has at most "
            "d distinct roots."
        ),
        "consecutive_mode_consequence": (
            "On a nonzero cyclic subgroup, a Laurent polynomial with "
            "modes a,a+1,...,a+s-1 has at most s-1 distinct zeros unless "
            "it is zero: multiply by X^(-a) and apply the root bound."
        ),
        "vandermonde_equivalent": (
            "For s distinct subgroup points, the s-by-s matrix of modes "
            "0,...,s-1 has determinant product_{i<j}(x_j-x_i), which is "
            "nonzero over every coefficient field."
        ),
        "balanced_color_acceptance": (
            "For a balanced color class of m=Theta(n) among "
            "n=B^(3/4+o(1)) atoms, degree-five multisets containing at "
            "least two colored atoms number "
            "binom(n+4,5)-binom(n-m+4,5)-m*binom(n-m+3,4)"
            "=Theta(n^5)=B^(15/4+o(1))."
        ),
        "dense_consecutive_predicate_lower_bound": (
            "A single consecutive-mode zero predicate for one color "
            "therefore needs B^(15/4+o(1)) serialized modes."
        ),
        "dense_state_exponent_B": fraction_record(C5_EXPONENT),
        "sequential_horner_query_exponent_B": fraction_record(C5_EXPONENT),
        "scope_limits": [
            "single zero predicates with consecutive Fourier modes",
            "dense serialized consecutive coefficient blocks",
            "sequential Horner or explicit product-tree evaluation",
            "injective degree-five source-monomial color predicates",
        ],
        "not_covered": [
            "lacunary finite-field Fourier supports",
            "high-degree low-SLP polynomial or rational predicates",
            "multiple small predicates arranged in a shared decision DAG",
            "adaptive cell-probe selectors",
            "non-Fourier finite-field predicates",
            "general arithmetic-circuit or data-structure lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_consecutive_mode_predicate_"
            "cost_ledger.r131.v1"
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
                "route_id": "dense_consecutive_mode_color_zero_predicate",
                "minimum_serialized_mode_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "sequential_evaluation_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "inside_setup_cap": False,
                "inside_polylog_query_cap": False,
                "status": "rejected",
            },
            {
                "route_id": "explicit_color_root_product_tree",
                "stored_root_exponent_B": fraction_record(C5_EXPONENT),
                "single_target_leaf_evaluation_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "inside_setup_cap": False,
                "inside_polylog_query_cap": False,
                "status": "rejected",
            },
            {
                "route_id": "lacunary_order_two_finite_field_predicate",
                "field_specific_sparse_zero_theorem_supplied": False,
                "inside_cap_circuit_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonfourier_shared_predicate_decision_dag",
                "exact_dag_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R130_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R130 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "r130_order_two_interface_inherited": (
            inherited["admission"][
                "finite_field_transfer_rejection_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "field_independent_root_bound_explicit": (
            "any field" in theorem["field_independent_root_bound"]
        ),
        "consecutive_mode_reduction_explicit": (
            "X^(-a)" in theorem["consecutive_mode_consequence"]
        ),
        "eight_actual_order_two_controls_complete": (
            controls["control_count"] == 8
            and controls["all_fields_have_order_two_characteristic"]
        ),
        "all_actual_supports_injective": controls[
            "all_supports_injective"
        ],
        "all_color_acceptance_counts_exact": controls[
            "all_color_count_formulas_exact"
        ],
        "all_actual_vandermonde_determinants_nonzero": controls[
            "all_vandermonde_determinants_nonzero"
        ],
        "all_dense_annihilator_zero_sets_exact": controls[
            "all_dense_annihilator_zero_sets_exact"
        ],
        "all_zero_targets_rejected": controls[
            "all_zero_targets_rejected"
        ],
        "all_optimal_c2_c3_sources_replay": (
            controls["all_selected_pairs_are_optimal_branches"]
            and controls["all_c2_c3_source_products_replay"]
        ),
        "dense_B15O4_state_and_query_charged": (
            theorem["dense_state_exponent_B"]
            == fraction_record(C5_EXPONENT)
            and theorem["sequential_horner_query_exponent_B"]
            == fraction_record(C5_EXPONENT)
        ),
        "lacunary_or_low_slp_predicate_complete": False,
        "inside_cap_shared_predicate_dag_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute a lacunary predicate in the actual "
        "ord_q(characteristic)=2 fields, or a non-Fourier shared decision "
        "DAG. It must exploit more than a dense consecutive mode block, "
        "choose a valid C2 branch in polylogarithmic arbitrary-target work, "
        "return exact C2+C3 sources or an empty certificate, fit "
        "B^(9/4+o(1)) state, avoid field DLP, and include rank, logs, "
        "identical descent, memory, field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_consecutive_mode_"
            "predicate.r131.v1"
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
            "dense consecutive-mode finite-field color zero predicates",
            "explicit color-root product trees",
        ],
        "preserved_interface": (
            "lacunary order-two finite-field predicate, high-degree low-SLP "
            "predicate, or non-Fourier shared decision DAG"
        ),
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_consecutive_mode_predicate_"
            "replay.r131.v1"
        ),
        "actual_control_count": controls["control_count"],
        "active_color_control_count": controls[
            "active_color_control_count"
        ],
        "all_actual_vandermonde_determinants_nonzero": controls[
            "all_vandermonde_determinants_nonzero"
        ],
        "all_dense_annihilator_zero_sets_exact": controls[
            "all_dense_annihilator_zero_sets_exact"
        ],
        "all_zero_targets_rejected": controls[
            "all_zero_targets_rejected"
        ],
        "all_optimal_c2_c3_sources_replay": (
            controls["all_selected_pairs_are_optimal_branches"]
            and controls["all_c2_c3_source_products_replay"]
        ),
        "inside_cap_lacunary_or_nonfourier_dag_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r131.v1",
        "r130_fourier_transfer_audit_complete": True,
        "r131_consecutive_mode_actual_field_audit_complete": True,
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
        "FIELD_INDEPENDENT_ROOT_BOUND_CLOSES_DENSE_CONSECUTIVE_MODE_"
        "COLOR_PREDICATES__ALL_EIGHT_ACTUAL_ORDER2_CONTROLS_HAVE_NONZERO_"
        "VANDERMONDE_DETERMINANTS_AND_EXACT_DENSE_ANNIHILATOR_ZERO_SETS__"
        "B15O4_STATE_AND_SEQUENTIAL_QUERY_REJECTED__LACUNARY_LOW_SLP_OR_"
        "NONFOURIER_SHARED_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_"
        "BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_ACTUAL_FIELD_CONSECUTIVE_MODE_CONTROLS_AND_SCOPED_"
            "ROOT_BOUND_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "active_color_control_count": controls[
                "active_color_control_count"
            ],
            "all_actual_vandermonde_determinants_nonzero": controls[
                "all_vandermonde_determinants_nonzero"
            ],
            "all_dense_annihilator_zero_sets_exact": controls[
                "all_dense_annihilator_zero_sets_exact"
            ],
            "all_optimal_c2_c3_sources_replay": (
                controls["all_selected_pairs_are_optimal_branches"]
                and controls["all_c2_c3_source_products_replay"]
            ),
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "actual_field_consecutive_mode_negative_admitted": True,
            "dense_predicate_cost_rejection_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_consecutive_mode_predicate.json"
            ),
            "cost": (
                "torus_c5_consecutive_mode_predicate_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_consecutive_mode_predicate_replay.json"
            ),
            "controls": (
                "torus_c5_consecutive_mode_predicate_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r131.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The root bound does not constrain lacunary mode supports.",
            "Polynomial degree is not an arithmetic-circuit size bound.",
            "Verifier source labels are not an online source locator.",
            "Finite controls receive no asymptotic credit.",
            "No inside-cap shared selector DAG is supplied.",
            "No rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_FIELD_INDEPENDENT_CONSECUTIVE_MODE_ROOT_BOUND_AND_"
            "EXACT_ACTUAL_ORDER2_CONTROLS_ONLY__REJECT_DENSE_B15O4_"
            "PREDICATE_STATE_AND_QUERY__PRESERVE_LACUNARY_LOW_SLP_AND_"
            "NONFOURIER_SHARED_DAG__NO_LOCATOR__NO_RANK__NO_LOGS__NO_"
            "DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_consecutive_mode_predicate_"
            "probe_report_r131.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_consecutive_mode_predicate.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_consecutive_mode_predicate_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_consecutive_mode_predicate_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_consecutive_mode_predicate_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r131.json"),
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
        f"R131 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
