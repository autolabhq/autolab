#!/usr/bin/env python3
"""Audit piecewise-constant C2 selectors and compact decision-DAG escapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_piecewise_selector_decision_dag.r129.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
DECK_EXPONENT = Fraction(3, 4)
C2_EXPONENT = Fraction(3, 2)
C3_EXPONENT = Fraction(9, 4)
C5_EXPONENT = Fraction(15, 4)

R128_PRODUCER = pathlib.Path(
    "p1553_torus_c5_rational_selector_degree_probe_r128.py"
)
R128_PRODUCER_SHA256 = (
    "50ddb43a677d114b534ed55eb877f589ceadf481d29bf5c819323d2add6848c3"
)
R128_REPORT = pathlib.Path(
    "p1553_torus_c5_rational_selector_degree_probe_report_r128.json"
)
R128_REPORT_SHA256 = (
    "400968376068f669ee972aaed88303ce24ee12529fa69bfdc9e7a3890ccdef8f"
)
R128_FROZEN = pathlib.Path(
    "frozen_torus_c5_rational_selector_degree.json"
)
R128_FROZEN_SHA256 = (
    "6cd1af34f4e2c6c5f0539b08653927b51930fb1b22779f5d6f560e50e589e8bf"
)
R128_COST = pathlib.Path(
    "torus_c5_rational_selector_degree_cost_ledger.json"
)
R128_COST_SHA256 = (
    "e375179be8e4bcc39205e17a6ea57d20bb83d27da405f82a7f6a765d1536eddf"
)
R128_REPLAY = pathlib.Path(
    "torus_c5_rational_selector_degree_replay.json"
)
R128_REPLAY_SHA256 = (
    "96a1e44774730df55d4ae28cec543ad9eefba4d601a65e6fd7a39f95423a95c9"
)
R128_CONTROLS = pathlib.Path(
    "torus_c5_rational_selector_degree_controls.json"
)
R128_CONTROLS_SHA256 = (
    "ee1afcd3fe2aa702305da00bd447113b83c596043cb019d8fff52866f7261d02"
)
R128_LOGS = pathlib.Path("factor_logs_and_identical_descent_r128.json")
R128_LOGS_SHA256 = (
    "dd8702872a282ab88e93d70f8f8185db52e6623ed26be7bee9f442d609479bf3"
)
R128_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_rational_selector_degree_probe_r128.py"
)
R128_TEST_SHA256 = (
    "a78bd8ed02849a150ea7a30ffc24ccd345d12927c9178e246a2c65b30335d7d7"
)
R128_GATE = pathlib.Path(
    "p1553_torus_c5_rational_selector_degree_probe_gate_r128.md"
)
R128_GATE_SHA256 = (
    "6033e5f5bf8fa0b9934efde304d4036a01fddf8deede30255a936b406722977d"
)
R128_PARENT = pathlib.Path(
    "p1553_torus_c5_rational_selector_degree_probe_parent_report_r128.yaml"
)
R128_PARENT_SHA256 = (
    "1c12e3a51d749946801381a9519bc781288b4b8dde85e9d5e4a3c436224c9818"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R119_GATE = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_gate_r119.md"
)
R119_GATE_SHA256 = (
    "941e375918c4acd1be8293fcc40666879b4f5178b4454e28c21487cb9be9a9e9"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R128 = load_module("p1553_r128_for_r129", R128_PRODUCER)
R126 = R128.R126
R121 = R128.R121
R82 = R128.R82
Field = R128.Field
Fp2 = tuple[int, int]
Source = tuple[int, ...]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r128_producer", R128_PRODUCER, R128_PRODUCER_SHA256),
        ("r128_report", R128_REPORT, R128_REPORT_SHA256),
        ("r128_frozen", R128_FROZEN, R128_FROZEN_SHA256),
        ("r128_cost", R128_COST, R128_COST_SHA256),
        ("r128_replay", R128_REPLAY, R128_REPLAY_SHA256),
        ("r128_controls", R128_CONTROLS, R128_CONTROLS_SHA256),
        ("r128_logs", R128_LOGS, R128_LOGS_SHA256),
        ("r128_test", R128_TEST, R128_TEST_SHA256),
        ("r128_gate", R128_GATE, R128_GATE_SHA256),
        ("r128_parent", R128_PARENT, R128_PARENT_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r119_gate", R119_GATE, R119_GATE_SHA256),
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
        raise AssertionError(f"R129 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def balanced_four_parts(n: int) -> tuple[tuple[int, ...], ...]:
    if n < 0:
        raise ValueError("n must be nonnegative")
    return tuple(
        tuple(index for index in range(n) if index % 4 == color)
        for color in range(4)
    )


def turan_k5_edge_count(n: int) -> int:
    parts = balanced_four_parts(n)
    return sum(
        len(left) * len(right)
        for left_index, left in enumerate(parts)
        for right in parts[left_index + 1 :]
    )


def minimum_cross_branch_count(n: int) -> int:
    return n * (n - 1) // 2 - turan_k5_edge_count(n)


def optimal_degree_two_branches(n: int) -> tuple[Source, ...]:
    parts = balanced_four_parts(n)
    squares = tuple((index, index) for index in range(n))
    cross_terms = tuple(
        pair
        for part in parts
        for pair in itertools.combinations(part, 2)
    )
    return squares + cross_terms


def exhaustive_minimum_cross_branch_count(n: int) -> int:
    edges = tuple(itertools.combinations(range(n), 2))
    five_sets = tuple(
        frozenset(values)
        for values in itertools.combinations(range(n), 5)
    )
    if not five_sets:
        return 0
    expected = minimum_cross_branch_count(n)
    for size in range(expected + 1):
        for candidate in itertools.combinations(edges, size):
            if all(
                any(set(edge) <= values for edge in candidate)
                for values in five_sets
            ):
                return size
    raise AssertionError("no cross-branch cover found at Turan bound")


def source_contains(source: Iterable[int], divisor: Iterable[int]) -> bool:
    source_counts = Counter(source)
    divisor_counts = Counter(divisor)
    return all(
        source_counts[index] >= multiplicity
        for index, multiplicity in divisor_counts.items()
    )


def scan_router(
    field: Field,
    target: Fp2,
    branch_values: Iterable[tuple[Source, Fp2]],
    c3_by_value: dict[Fp2, Source],
) -> dict[str, Any]:
    probes = 0
    for pair_source, pair_value in branch_values:
        probes += 1
        triple_source = c3_by_value.get(field.div(target, pair_value))
        if triple_source is None:
            continue
        source = tuple(sorted(pair_source + triple_source))
        return {
            "found": True,
            "probes": probes,
            "pair_source": list(pair_source),
            "triple_source": list(triple_source),
            "source": list(source),
        }
    return {
        "found": False,
        "probes": probes,
        "pair_source": None,
        "triple_source": None,
        "source": None,
    }


def piecewise_selector_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    n = len(deck)
    c2 = R126.source_products(field, deck, 2)
    c3 = R126.source_products(field, deck, 3)
    c5 = R126.source_products(field, deck, 5)
    c2_by_source = {source: value for source, value in c2}
    c3_by_value = {value: source for source, value in c3}
    c5_by_value = {value: source for source, value in c5}
    injective_supports = (
        len(c2_by_source) == len(c2)
        and len(c3_by_value) == len(c3)
        and len(c5_by_value) == len(c5)
    )
    branches = optimal_degree_two_branches(n)
    branch_values = tuple(
        (source, c2_by_source[source]) for source in branches
    )
    source_cover_exact = all(
        any(source_contains(source, branch) for branch in branches)
        for source, _ in c5
    )
    positive_results = [
        (source, target, scan_router(field, target, branch_values, c3_by_value))
        for source, target in c5
    ]
    positive_sources_replay = all(
        result["found"]
        and tuple(result["source"]) in {source for source, _ in c5}
        and field.product(deck[index] for index in result["source"]) == target
        for _, target, result in positive_results
    )
    empty_result = scan_router(
        field,
        field.zero,
        branch_values,
        c3_by_value,
    )
    expected_cross = minimum_cross_branch_count(n)
    exhaustive_cross = exhaustive_minimum_cross_branch_count(n)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": n,
        "c2_source_count": len(c2),
        "c3_source_count": len(c3),
        "c5_source_count": len(c5),
        "all_c2_c3_c5_source_products_injective": injective_supports,
        "balanced_four_part_sizes": [
            len(part) for part in balanced_four_parts(n)
        ],
        "mandatory_square_branch_count": n,
        "minimum_cross_branch_count_by_turan": expected_cross,
        "minimum_cross_branch_count_by_exhaustion": exhaustive_cross,
        "selected_branch_count": len(branches),
        "selected_branch_count_is_turan_optimal": (
            len(branches) == n + expected_cross
            and expected_cross == exhaustive_cross
        ),
        "selected_branch_sources": [list(source) for source in branches],
        "source_monomial_cover_exact": source_cover_exact,
        "all_positive_targets_located": all(
            result["found"] for _, _, result in positive_results
        ),
        "all_returned_c2_c3_sources_replay": positive_sources_replay,
        "maximum_positive_query_branch_probes": max(
            result["probes"] for _, _, result in positive_results
        ),
        "empty_target": field.json(field.zero),
        "empty_target_rejected": not empty_result["found"],
        "empty_target_branch_probes": empty_result["probes"],
        "empty_target_scans_every_branch": (
            empty_result["probes"] == len(branches)
        ),
        "c3_dictionary_entry_count": len(c3_by_value),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        piecewise_selector_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    return {
        "schema": (
            "p1553.torus_c5_piecewise_selector_decision_dag_"
            "controls.r129.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_supports_injective": all(
            row["all_c2_c3_c5_source_products_injective"]
            for row in controls
        ),
        "all_selected_branch_sets_turan_optimal": all(
            row["selected_branch_count_is_turan_optimal"]
            for row in controls
        ),
        "all_source_monomial_covers_exact": all(
            row["source_monomial_cover_exact"] for row in controls
        ),
        "all_positive_targets_located": all(
            row["all_positive_targets_located"] for row in controls
        ),
        "all_returned_sources_replay": all(
            row["all_returned_c2_c3_sources_replay"]
            for row in controls
        ),
        "all_empty_targets_rejected_after_full_scan": all(
            row["empty_target_rejected"]
            and row["empty_target_scans_every_branch"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "source_model": (
            "The Ck supports are injective degree-k commutative monomials "
            "in n deck atoms, with n=B^(3/4+o(1)). A constant C2 branch "
            "covers targets divisible by its degree-two monomial."
        ),
        "mandatory_squares": (
            "Each pure fifth power Xi^5 forces the square branch Xi^2."
        ),
        "squarefree_reduction": (
            "After squares cover repeated variables, cross branches are "
            "edges that must hit every square-free five-subset."
        ),
        "turan_reduction": (
            "The unselected edge graph is K5-free. By Turan's theorem it "
            "has at most ex(n,K5)=|E(T4(n))| edges."
        ),
        "exact_minimum_branch_count": (
            "Lmin=n+binom(n,2)-ex(n,K5)"
        ),
        "matching_construction": (
            "Partition the n atoms into four balanced parts, select all "
            "squares and all within-part pairs, and apply pigeonhole to "
            "every square-free five-subset."
        ),
        "iid_exponent_lower_bound": (
            "Lmin=Theta(n^2)=B^(3/2+o(1))"
        ),
        "sequential_scan_query_exponent_B": fraction_record(C2_EXPONENT),
        "shared_c3_dictionary_state_exponent_B": fraction_record(
            C3_EXPONENT
        ),
        "explicit_target_router_state_exponent_B": fraction_record(
            C5_EXPONENT
        ),
        "slp_degree_boundary": (
            "An arithmetic SLP with t multiplication layers can have "
            "degree up to 2^t, so the inherited degree B^(9/4) forces only "
            "t=Omega(log B), which is compatible with polylogarithmic "
            "query work."
        ),
        "scope_limits": [
            "injective commutative source-monomial supports",
            "piecewise-constant C2 translate branches",
            "sequential branch scans with a shared exact C3 dictionary",
            "explicit target-to-branch routing tables",
        ],
        "not_covered": [
            "compact shared-predicate branch decision DAGs",
            "high-degree low-SLP rational selectors",
            "adaptive cell-probe selectors",
            "filtered decks with exploitable algebraic collisions",
            "general arithmetic-circuit or data-structure lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_piecewise_selector_decision_dag_"
            "cost_ledger.r129.v1"
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
                "route_id": (
                    "optimal_piecewise_constant_branches_sequential_scan"
                ),
                "branch_count_exponent_B": fraction_record(C2_EXPONENT),
                "shared_c3_dictionary_state_exponent_B": fraction_record(
                    C3_EXPONENT
                ),
                "query_exponent_B": fraction_record(C2_EXPONENT),
                "inside_setup_cap": True,
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "explicit_target_to_branch_router",
                "state_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
            },
            {
                "route_id": "balanced_shared_predicate_decision_dag",
                "ideal_depth": "O(log B)",
                "compact_predicates_constructed": False,
                "exact_source_backpointers_constructed": False,
                "status": "open",
            },
            {
                "route_id": "high_degree_low_slp_rational_selector",
                "degree_lower_bound_exponent_B": fraction_record(
                    C3_EXPONENT
                ),
                "multiplication_lower_bound": "Omega(log B)",
                "polylog_query_excluded": False,
                "exact_selector_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R128_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R128 nonclaim boundary drifted")
    controls = finite_controls()
    theorem = theorem_record()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r128_low_slp_piecewise_interface_inherited": (
            inherited["admission"][
                "scoped_rational_degree_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "eight_actual_piecewise_controls_complete": (
            controls["control_count"] == 8
        ),
        "all_c2_c3_c5_supports_injective": controls[
            "all_supports_injective"
        ],
        "turan_branch_count_theorem_explicit": (
            theorem["exact_minimum_branch_count"]
            == "Lmin=n+binom(n,2)-ex(n,K5)"
        ),
        "finite_branch_minima_verified_by_exhaustion": controls[
            "all_selected_branch_sets_turan_optimal"
        ],
        "all_source_monomial_covers_exact": controls[
            "all_source_monomial_covers_exact"
        ],
        "all_positive_targets_located_with_sources": (
            controls["all_positive_targets_located"]
            and controls["all_returned_sources_replay"]
        ),
        "all_empty_targets_rejected_after_full_scan": controls[
            "all_empty_targets_rejected_after_full_scan"
        ],
        "B3O2_sequential_scan_cost_charged": (
            theorem["sequential_scan_query_exponent_B"]
            == fraction_record(C2_EXPONENT)
        ),
        "explicit_B15O4_router_rejected": (
            theorem["explicit_target_router_state_exponent_B"]
            == fraction_record(C5_EXPONENT)
        ),
        "low_slp_degree_caveat_preserved": (
            "Omega(log B)" in theorem["slp_degree_boundary"]
            and "compact shared-predicate branch decision DAGs"
            in theorem["not_covered"]
        ),
        "compact_shared_predicate_dag_complete": False,
        "asymptotic_five_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute a compact shared-predicate decision DAG for "
        "the B^(3/2) optimal C2 translate branches, or a high-degree "
        "low-SLP selector. It must choose a valid branch in polylogarithmic "
        "arbitrary-target work without an explicit B^(15/4) target table, "
        "return exact C2+C3 sources or an empty certificate, fit "
        "B^(9/4+o(1)) total state, avoid field DLP, and include rank, logs, "
        "identical descent, memory, field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_piecewise_selector_"
            "decision_dag.r129.v1"
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
            "sequential scans of piecewise-constant C2 translate branches",
            "explicit target-to-branch routing tables",
            "dense single rational C2 selectors",
        ],
        "preserved_interface": (
            "compact shared-predicate branch decision DAG or high-degree "
            "low-SLP rational selector"
        ),
        "general_circuit_or_data_structure_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_piecewise_selector_decision_dag_"
            "replay.r129.v1"
        ),
        "actual_control_count": controls["control_count"],
        "all_selected_branch_sets_turan_optimal": controls[
            "all_selected_branch_sets_turan_optimal"
        ],
        "all_positive_targets_located": controls[
            "all_positive_targets_located"
        ],
        "all_c2_c3_sources_replay": controls[
            "all_returned_sources_replay"
        ],
        "all_empty_targets_rejected_after_full_scan": controls[
            "all_empty_targets_rejected_after_full_scan"
        ],
        "inside_cap_compact_decision_dag_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r129.v1",
        "r128_rational_selector_degree_audit_complete": True,
        "r129_piecewise_selector_branch_audit_complete": True,
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
        "INJECTIVE_SOURCE_MONOMIAL_PIECEWISE_C2_SELECTOR_NEEDS_TURAN_"
        "OPTIMAL_B3O2_BRANCHES__ALL_EIGHT_ACTUAL_OPTIMAL_BRANCH_COVERS_"
        "RETURN_EXACT_SOURCES_AND_EMPTY_CERTIFICATES__SEQUENTIAL_SCAN_"
        "B3O2_QUERY_AND_EXPLICIT_ROUTER_B15O4_STATE_REJECTED__DEGREE_ONLY_"
        "FORCES_LOGARITHMIC_SLP_MULTIPLICATIONS__COMPACT_SHARED_PREDICATE_"
        "DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_PIECEWISE_SELECTOR_BRANCH_CONTROLS_AND_SCOPED_TURAN_"
            "BOUND_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_selected_branch_sets_turan_optimal": controls[
                "all_selected_branch_sets_turan_optimal"
            ],
            "all_positive_targets_located": controls[
                "all_positive_targets_located"
            ],
            "all_returned_sources_replay": controls[
                "all_returned_sources_replay"
            ],
            "all_empty_targets_rejected_after_full_scan": controls[
                "all_empty_targets_rejected_after_full_scan"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "piecewise_selector_semantics_admitted": True,
            "scoped_turan_branch_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_piecewise_selector_decision_dag.json"
            ),
            "cost": (
                "torus_c5_piecewise_selector_decision_dag_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_piecewise_selector_decision_dag_replay.json"
            ),
            "controls": (
                "torus_c5_piecewise_selector_decision_dag_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r129.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The Turan result is scoped to injective source monomials.",
            "Branch count is not a decision-DAG query lower bound.",
            "Degree is not an arithmetic-circuit size lower bound.",
            "Compact shared predicates and low-SLP selectors remain open.",
            "Finite controls receive no asymptotic credit.",
            "No rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_ACTUAL_OPTIMAL_PIECEWISE_BRANCH_COVERS_AND_SOURCE_"
            "SEMANTICS_ONLY__REJECT_SEQUENTIAL_BRANCH_SCAN_AND_EXPLICIT_"
            "TARGET_ROUTER_AT_FROZEN_CAPS__PRESERVE_SHARED_PREDICATE_DAG_"
            "AND_HIGH_DEGREE_LOW_SLP__NO_LOCATOR__NO_RANK__NO_LOGS__NO_"
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
            "p1553_torus_c5_piecewise_selector_decision_dag_"
            "probe_report_r129.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_piecewise_selector_decision_dag.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_piecewise_selector_decision_dag_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_piecewise_selector_decision_dag_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_piecewise_selector_decision_dag_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r129.json"),
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
        f"R129 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
