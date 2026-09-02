#!/usr/bin/env python3
"""Audit sparse extension-field zero predicates on the prime-order torus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.torus_c5_sparse_monomial_root_bound.r133.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
DECK_EXPONENT_B = Fraction(3, 4)
C5_EXPONENT_B = Fraction(15, 4)
GROUP_ORDER_EXPONENT_B = Fraction(5)
COLOR_ROOT_EXPONENT_Q = Fraction(3, 4)
PURE_FIFTH_EXPONENT_Q = Fraction(3, 20)
ROOT_DENSITY_DENOMINATOR = 8

R132_PRODUCER = pathlib.Path(
    "p1553_torus_c5_base_field_frobenius_predicate_dag_probe_r132.py"
)
R132_PRODUCER_SHA256 = (
    "9235668bed218df40c60ea774673b03bc2d55236cd1c04ab3ef813ad61e0baad"
)
R132_REPORT = pathlib.Path(
    "p1553_torus_c5_base_field_frobenius_predicate_dag_"
    "probe_report_r132.json"
)
R132_REPORT_SHA256 = (
    "54e0396f8fcbed3fb41893af1c2d3612f81f26b5ff44dcbc0784299a7ff96f17"
)
R132_FROZEN = pathlib.Path(
    "frozen_torus_c5_base_field_frobenius_predicate_dag.json"
)
R132_FROZEN_SHA256 = (
    "efc22a83ab20a27b490204acdb67a8bda351d774caf2c186cda9f04cdd0ae56f"
)
R132_COST = pathlib.Path(
    "torus_c5_base_field_frobenius_predicate_dag_cost_ledger.json"
)
R132_COST_SHA256 = (
    "437e3a1165d17d6f85caa4e246ea9769012df2915113675249bfcc68502fbe4f"
)
R132_REPLAY = pathlib.Path(
    "torus_c5_base_field_frobenius_predicate_dag_replay.json"
)
R132_REPLAY_SHA256 = (
    "0e7f17f5c0c04cbdcf1a430f36114343b2b786cf3cb85a9ca586afeb549471e9"
)
R132_CONTROLS = pathlib.Path(
    "torus_c5_base_field_frobenius_predicate_dag_controls.json"
)
R132_CONTROLS_SHA256 = (
    "f0ee93f6ddfef5395abc88a8f03b31a325504a14bf04b7d7a246d9efec3f53ab"
)
R132_LOGS = pathlib.Path("factor_logs_and_identical_descent_r132.json")
R132_LOGS_SHA256 = (
    "f85f4135fe44d2aebbe61191dff04e2e40aff4a07d45118c752758c8425dfed3"
)
R132_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_base_field_frobenius_predicate_dag_probe_r132.py"
)
R132_TEST_SHA256 = (
    "296daa244226d7d3a41da492f97cf1ea291280af6bb9e0374e0b366df0e4d099"
)
R132_GATE = pathlib.Path(
    "p1553_torus_c5_base_field_frobenius_predicate_dag_probe_gate_r132.md"
)
R132_GATE_SHA256 = (
    "dd20f3188596712929c9715d1dbf27782a41a5be8900f727f948def2a7947332"
)
R132_PARENT = pathlib.Path(
    "p1553_torus_c5_base_field_frobenius_predicate_dag_"
    "probe_parent_report_r132.yaml"
)
R132_PARENT_SHA256 = (
    "958b63522081adcba52bdfbdbaa982356d00d67098ea4c298e53227af1b6e447"
)
KELLEY_PDF = pathlib.Path(
    "references/kelley_sparse_polynomial_roots_1602.00208.pdf"
)
KELLEY_PDF_SHA256 = (
    "250daacc1b157043cd9c7345036eb332fc69fee8e48a12ba81eddcc2b91995b7"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R132 = load_module("p1553_r132_for_r133", R132_PRODUCER)
R131 = R132.R131
R129 = R131.R129
R126 = R131.R126
R121 = R131.R121
R82 = R131.R82


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r132_producer", R132_PRODUCER, R132_PRODUCER_SHA256),
        ("r132_report", R132_REPORT, R132_REPORT_SHA256),
        ("r132_frozen", R132_FROZEN, R132_FROZEN_SHA256),
        ("r132_cost", R132_COST, R132_COST_SHA256),
        ("r132_replay", R132_REPLAY, R132_REPLAY_SHA256),
        ("r132_controls", R132_CONTROLS, R132_CONTROLS_SHA256),
        ("r132_logs", R132_LOGS, R132_LOGS_SHA256),
        ("r132_test", R132_TEST, R132_TEST_SHA256),
        ("r132_gate", R132_GATE, R132_GATE_SHA256),
        ("r132_parent", R132_PARENT, R132_PARENT_SHA256),
        ("kelley_sparse_roots_pdf", KELLEY_PDF, KELLEY_PDF_SHA256),
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
        raise AssertionError(f"R133 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def sparse_root_exponent(term_count: int) -> Fraction:
    if term_count < 1:
        raise ValueError("term count must be positive")
    if term_count == 1:
        return Fraction(0)
    return Fraction(term_count - 2, term_count - 1)


def logarithmic_mode_threshold(subgroup_order: int) -> int:
    if subgroup_order <= 1:
        raise ValueError("subgroup order must exceed one")
    return 1 + int(math.log2(subgroup_order) // 4)


def random_deck_log2_union_bound(
    subgroup_order: int,
    minimum_color_size: int,
) -> dict[str, Any]:
    threshold = logarithmic_mode_threshold(subgroup_order)
    log2_bound = (
        math.log2(4 * threshold)
        + threshold * math.log2(36)
        + 3 * threshold * math.log2(subgroup_order)
        - 3 * minimum_color_size
    )
    return {
        "maximum_term_count": threshold,
        "minimum_color_pure_fifth_count": minimum_color_size,
        "log2_union_bound": log2_bound,
        "union_bound_below_one": log2_bound < 0,
        "actual_control_receives_probability_credit": False,
    }


def sparse_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, 5)
    c5_by_source = {source: value for source, value in c5}
    color_controls = []
    for color, part in enumerate(R129.balanced_four_parts(len(deck))):
        if not part:
            continue
        accepted = {
            value
            for source, value in c5
            if R131.source_color_multiplicity(source, color) >= 2
        }
        pure_sources = tuple((index,) * 5 for index in part)
        pure_targets = tuple(
            field.pow(deck[index], 5)
            for index in part
        )
        color_controls.append(
            {
                "color": color,
                "part_size": len(part),
                "accepted_target_count": len(accepted),
                "pure_fifth_source_count": len(pure_sources),
                "pure_fifth_target_count": len(pure_targets),
                "pure_fifth_targets_distinct": (
                    len(set(pure_targets)) == len(pure_targets)
                ),
                "all_pure_sources_in_c5_replay": all(
                    source in c5_by_source for source in pure_sources
                ),
                "all_pure_source_values_match_direct_fifth_powers": all(
                    c5_by_source[source] == target
                    for source, target in zip(pure_sources, pure_targets)
                ),
                "all_pure_fifth_targets_in_color_acceptance": all(
                    target in accepted for target in pure_targets
                ),
            }
        )
    q = curve["subgroup_order"]
    minimum_color_size = min(
        row["pure_fifth_target_count"] for row in color_controls
    )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": q,
        "subgroup_order_probable_prime": R82.R70.is_prime(q),
        "field_characteristic_equals_six_q_minus_one": (
            field.p == 6 * q - 1
        ),
        "coefficient_field_size": field.p * field.p,
        "coefficient_field_size_below_36_q_squared": (
            field.p * field.p < 36 * q * q
        ),
        "fifth_power_is_subgroup_permutation": math.gcd(5, q) == 1,
        "deck_size": len(deck),
        "c5_source_count": len(c5),
        "color_controls": color_controls,
        "all_pure_fifth_targets_distinct": all(
            row["pure_fifth_targets_distinct"] for row in color_controls
        ),
        "all_pure_fifth_sources_replay": all(
            row["all_pure_sources_in_c5_replay"]
            and row["all_pure_source_values_match_direct_fifth_powers"]
            for row in color_controls
        ),
        "all_pure_fifth_targets_in_color_acceptance": all(
            row["all_pure_fifth_targets_in_color_acceptance"]
            for row in color_controls
        ),
        "random_deck_union_bound_diagnostic": random_deck_log2_union_bound(
            q,
            minimum_color_size,
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        sparse_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    return {
        "schema": (
            "p1553.torus_c5_sparse_monomial_root_bound_"
            "controls.r133.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_subgroup_orders_probable_prime": all(
            row["subgroup_order_probable_prime"] for row in controls
        ),
        "all_fields_satisfy_p_equals_six_q_minus_one": all(
            row["field_characteristic_equals_six_q_minus_one"]
            for row in controls
        ),
        "all_coefficient_fields_below_36_q_squared": all(
            row["coefficient_field_size_below_36_q_squared"]
            for row in controls
        ),
        "all_fifth_power_maps_are_permutations": all(
            row["fifth_power_is_subgroup_permutation"] for row in controls
        ),
        "all_pure_fifth_targets_distinct": all(
            row["all_pure_fifth_targets_distinct"] for row in controls
        ),
        "all_pure_fifth_sources_replay": all(
            row["all_pure_fifth_sources_replay"] for row in controls
        ),
        "all_pure_fifth_targets_in_color_acceptance": all(
            row["all_pure_fifth_targets_in_color_acceptance"]
            for row in controls
        ),
        "all_actual_random_deck_union_bounds_below_one": all(
            row["random_deck_union_bound_diagnostic"][
                "union_bound_below_one"
            ]
            for row in controls
        ),
        "actual_random_deck_union_bounds_receive_credit": False,
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    term_rows = [
        {
            "term_count": term_count,
            "root_exponent_q": fraction_record(
                sparse_root_exponent(term_count)
            ),
            "excluded_for_q_to_three_quarters_roots": term_count <= 4,
            "first_not_excluded_by_bound": term_count == 5,
        }
        for term_count in range(1, 6)
    ]
    return {
        "primary_source": {
            "author": "Zander Kelley",
            "title": "Roots of Sparse Polynomials over a Finite Field",
            "arxiv": "1602.00208",
            "url": "https://arxiv.org/abs/1602.00208",
            "pinned_artifact": str(KELLEY_PDF),
            "pinned_sha256": KELLEY_PDF_SHA256,
            "source_result": "Theorem 2.3 and its proof",
        },
        "subgroup_adaptation": (
            "Let H be a cyclic subgroup of prime order q in a field K of "
            "characteristic not q. For a nonzero t-mode function "
            "f(z)=sum_i a_i z^(e_i) with nonzero coefficients and distinct "
            "e_i modulo q, the degree-reduction proof of Kelley Theorem 2.3 "
            "applies with cyclic domain order q. Since H has prime order and "
            "distinct characters are linearly independent, the largest root "
            "coset has size C=1. Thus for t>=2, "
            "|{z in H:f(z)=0}| <= 2 q^(1-1/(t-1))."
        ),
        "adaptation_proof_dependencies": [
            "exponent reduction modulo the cyclic domain order q",
            "the geometry-of-numbers degree reduction in t-1 exponents",
            "power maps and coset decomposition in a cyclic group",
            "the ordinary degree bound over the coefficient field K",
            "linear independence of distinct characters because char(K) does not divide q",
        ],
        "prime_order_coset_parameter": 1,
        "required_color_root_exponent_q": fraction_record(
            COLOR_ROOT_EXPONENT_Q
        ),
        "term_threshold_rows": term_rows,
        "deterministic_consequence": (
            "A represented extension-field zero predicate with at most four "
            "distinct modes cannot vanish on q^(3/4+o(1)) color targets. "
            "Five modes are the first count not excluded by this bound; the "
            "bound does not construct such a predicate."
        ),
        "random_deck_model": {
            "assumptions": [
                "q is prime and q is not 5",
                "the deck is an ordered uniform sample without replacement from H",
                "the four balanced color parts are fixed independently of deck values",
                "the coefficient field has size p^2 with p<6q",
                "predicates are represented sparse sums with distinct exponents modulo q",
            ],
            "pure_fifth_subset": (
                "Every atom in a color supplies its pure fifth power to that "
                "color's C5 acceptance set. Since gcd(5,q)=1, fifth powering "
                "permutes H, so a color's pure fifth powers form a uniform "
                "sample without replacement."
            ),
            "logarithmic_threshold": (
                "If t-1 <= log2(q)/4, the subgroup root bound gives root "
                "density at most 2*q^(-1/(t-1)) <= 1/8."
            ),
            "projective_polynomial_count": (
                "For exact term count t, support choices are at most q^t. "
                "After projective coefficient normalization, coefficient "
                "choices are at most (p^2)^(t-1) < 36^t q^(2t). Hence the "
                "represented predicate count is below 36^t q^(3t)."
            ),
            "union_bound": (
                "Writing T=1+floor(log2(q)/4) and m for the smallest color "
                "part, the probability that any represented predicate with "
                "at most T modes vanishes on every pure fifth power of any "
                "color is at most 4*T*36^T*q^(3T)*8^(-m)."
            ),
            "asymptotic_consequence": (
                "At deck size q^(3/20+o(1)), m=q^(3/20+o(1))/4 and the "
                "displayed failure bound tends to zero because its positive "
                "logarithm is O((log q)^2) while 3m grows polynomially."
            ),
            "model_bound": True,
            "transfers_to_structured_factor_base": False,
            "receives_candidate_credit": False,
        },
        "scope_limits": [
            "one represented sparse polynomial zero predicate",
            "distinct exponent modes modulo the prime subgroup order",
            "one through four modes deterministically",
            "up to one quarter log2(q) modes only in the frozen random-deck model",
        ],
        "not_covered": [
            "five-mode existence or nonexistence at the q^(3/4) threshold",
            "larger polylogarithmic represented support",
            "polynomials with large expansion but compact straight-line programs",
            "multiple-predicate or adaptive decision DAGs",
            "nonzero-value tests and coordinate comparisons",
            "deterministic structured factor bases",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_sparse_monomial_root_bound_"
            "cost_ledger.r133.v1"
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
                "route_id": "one_to_four_mode_extension_zero_predicate",
                "model_required": False,
                "required_root_exponent_q": fraction_record(
                    COLOR_ROOT_EXPONENT_Q
                ),
                "status": "rejected_asymptotically_by_subgroup_root_bound",
            },
            {
                "route_id": "small_log_mode_random_deck_zero_predicate",
                "maximum_term_count": "1+floor(log2(q)/4)",
                "model_required": True,
                "structured_factor_base_transfer_proved": False,
                "status": (
                    "rejected_with_overwhelming_probability_under_"
                    "uniform_random_deck_model_only"
                ),
            },
            {
                "route_id": "five_mode_extension_zero_predicate",
                "first_not_excluded_by_root_exponent": True,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "larger_polylog_sparse_extension_predicate",
                "covered_by_random_deck_union_bound": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "low_slp_expanded_extension_predicate",
                "represented_sparse_support_bound_applicable": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "multi_predicate_frobenius_coordinate_dag",
                "single_zero_predicate_bound_applicable": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R132_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R132 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "eleven_source_bindings_verified": len(source_hashes) == 11,
        "r132_asymmetric_interface_inherited": (
            inherited["admission"][
                "base_field_frobenius_dag_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "primary_source_pinned": (
            theorem["primary_source"]["pinned_sha256"]
            == KELLEY_PDF_SHA256
        ),
        "cyclic_subgroup_adaptation_explicit": (
            len(theorem["adaptation_proof_dependencies"]) == 5
        ),
        "prime_order_coset_parameter_is_one": (
            theorem["prime_order_coset_parameter"] == 1
        ),
        "one_through_four_modes_excluded_deterministically": all(
            row["excluded_for_q_to_three_quarters_roots"]
            for row in theorem["term_threshold_rows"][:4]
        ),
        "five_modes_first_not_excluded": (
            theorem["term_threshold_rows"][4][
                "first_not_excluded_by_bound"
            ]
        ),
        "random_deck_assumptions_explicit": (
            len(theorem["random_deck_model"]["assumptions"]) == 5
        ),
        "small_log_union_bound_explicit": (
            "4*T*36^T*q^(3T)*8^(-m)"
            in theorem["random_deck_model"]["union_bound"]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "pure_fifth_source_witnesses_exact": (
            controls["all_fifth_power_maps_are_permutations"]
            and controls["all_pure_fifth_targets_distinct"]
            and controls["all_pure_fifth_sources_replay"]
            and controls["all_pure_fifth_targets_in_color_acceptance"]
        ),
        "model_and_finite_evidence_receive_no_credit": (
            not theorem["random_deck_model"]["receives_candidate_credit"]
            and not controls[
                "actual_random_deck_union_bounds_receive_credit"
            ]
            and not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "larger_sparse_slp_and_multi_dag_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "larger_polylog_sparse_extension_predicate",
                "low_slp_expanded_extension_predicate",
                "multi_predicate_frobenius_coordinate_dag",
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
        "Test the first surviving asymmetric selector classes: a five-mode "
        "F_(p^2) zero predicate at the q^(3/4) threshold, a represented "
        "predicate above the small-log regime, a high-expansion low-SLP "
        "predicate, or a multi-predicate Frobenius-coordinate DAG. Freeze "
        "every coefficient and node, replay exact positive/empty paths and "
        "C2+C3 sources, fit B^(9/4+o(1)) state and polylogarithmic arbitrary-"
        "target work, avoid field DLP, and charge rank, logs, identical "
        "descent, memory, field operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_sparse_monomial_root_bound.r133.v1"
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
            "single represented F_(p^2) zero predicates with at most four modes",
            (
                "single represented predicates through one quarter log2(q) "
                "modes under the uniform-random-deck model only"
            ),
        ],
        "preserved_interface": (
            "five-mode or larger-polylog sparse predicate, high-expansion "
            "low-SLP predicate, or multi-predicate Frobenius-coordinate DAG"
        ),
        "random_deck_model_transferred_to_structured_factor_base": False,
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_sparse_monomial_root_bound_replay.r133.v1"
        ),
        "actual_control_count": controls["control_count"],
        "all_actual_subgroup_orders_prime": controls[
            "all_subgroup_orders_probable_prime"
        ],
        "all_fifth_power_maps_are_permutations": controls[
            "all_fifth_power_maps_are_permutations"
        ],
        "all_pure_fifth_sources_replay": controls[
            "all_pure_fifth_sources_replay"
        ],
        "all_pure_fifth_targets_in_color_acceptance": controls[
            "all_pure_fifth_targets_in_color_acceptance"
        ],
        "all_actual_random_deck_union_bounds_below_one": controls[
            "all_actual_random_deck_union_bounds_below_one"
        ],
        "actual_union_bounds_receive_credit": False,
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r133.v1",
        "r132_base_field_frobenius_dag_audit_complete": True,
        "r133_sparse_monomial_root_bound_audit_complete": True,
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
        "PRIME_ORDER_SUBGROUP_ADAPTATION_OF_KELLEY_SPARSE_ROOT_BOUND__"
        "ONE_TO_FOUR_EXTENSION_MODES_CANNOT_COVER_Q_TO_THREE_QUARTERS__"
        "FIVE_MODES_FIRST_NOT_EXCLUDED__UNIFORM_RANDOM_DECK_PURE_FIFTH_"
        "UNION_BOUND_EXCLUDES_UP_TO_QUARTER_LOG2_Q_MODES_WITH_OVERWHELMING_"
        "PROBABILITY_ONLY__NO_STRUCTURED_FACTOR_BASE_TRANSFER__LARGER_"
        "POLYLOG_LOW_SLP_MULTI_PREDICATE_DAG_OPEN__NO_RANK_LOGS_DESCENT_"
        "SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "PINNED_SPARSE_ROOT_THEOREM_ADAPTATION_AND_RANDOM_DECK_MODEL_"
            "NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_pure_fifth_sources_replay": controls[
                "all_pure_fifth_sources_replay"
            ],
            "all_pure_fifth_targets_in_color_acceptance": controls[
                "all_pure_fifth_targets_in_color_acceptance"
            ],
            "all_actual_random_deck_union_bounds_below_one": controls[
                "all_actual_random_deck_union_bounds_below_one"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "deterministic_one_to_four_mode_negative_admitted": True,
            "random_deck_small_log_negative_admitted_model_only": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_sparse_monomial_root_bound.json",
            "cost": (
                "torus_c5_sparse_monomial_root_bound_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_sparse_monomial_root_bound_replay.json"
            ),
            "controls": (
                "torus_c5_sparse_monomial_root_bound_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r133.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The subgroup result adapts a pinned proof; it is not a new sparse-root theorem.",
            "The random-deck union bound does not transfer to the structured factor base.",
            "The actual small controls receive no probability or asymptotic credit.",
            "Five modes and larger polylogarithmic represented support remain open.",
            "High-expansion low-SLP and multi-predicate DAGs remain open.",
            "No general circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_PINNED_SUBGROUP_SPARSE_ROOT_BOUND_AND_ONE_TO_FOUR_MODE_"
            "NEGATIVE__ADMIT_SMALL_LOG_NEGATIVE_UNDER_RANDOM_DECK_MODEL_ONLY__"
            "WITHHOLD_STRUCTURED_TRANSFER__PRESERVE_FIVE_MODE_LARGER_POLYLOG_"
            "LOW_SLP_AND_MULTI_PREDICATE_DAGS__NO_LOCATOR__NO_RANK__NO_LOGS__"
            "NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_sparse_monomial_root_bound_"
            "probe_report_r133.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_sparse_monomial_root_bound.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_monomial_root_bound_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_monomial_root_bound_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_sparse_monomial_root_bound_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r133.json"),
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
        f"R133 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
