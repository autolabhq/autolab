#!/usr/bin/env python3
"""Solve the exact exponent envelope for the post-R114 relation redesign."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = (
    "p1553.relation_arity_factor_base_transposed_interface_rebalance."
    "r115.v1"
)
LOG_B_GROUP_ORDER = Fraction(5)
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
RHO_EXPONENT = Fraction(5, 2)
LOCAL_HALF_ARITY = 3
MIN_ARITY = 3
MAX_ARITY = 32

R114_PRODUCER = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_leaf_generator_probe_r114.py"
)
R114_PRODUCER_SHA256 = (
    "b477e0162d7cd89d3341f77521dc0b004879b8a916b484ca40b6787746089d83"
)
R114_REPORT = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_"
    "leaf_generator_probe_report_r114.json"
)
R114_REPORT_SHA256 = (
    "b0759fd0e0fe14e0ea44f908b802e24b611802a64663e138337465cf36ef69a5"
)
R114_FROZEN = pathlib.Path(
    "frozen_5a5c_transposed_nonuniform_c5_leaf_generator.json"
)
R114_FROZEN_SHA256 = (
    "4708bc972a2df5dab75d208329f8f00e519d0496200ce6a59afb455d9f092b3d"
)
R114_LEDGER = pathlib.Path("transposed_c5_leaf_generator_ledger.json")
R114_LEDGER_SHA256 = (
    "6552df6c72229abdb1bf10d4cc05dc44a5ff2e4138f7413ac96c9a29ad3b2dd9"
)
R114_REPLAY = pathlib.Path("transposed_c5_source_adjoint_replay.json")
R114_REPLAY_SHA256 = (
    "031e639ab8903ea8f4f7f9f3f7bec384a2940aa40473562419494cb27efee825"
)
R114_EXCEPTIONAL = pathlib.Path("transposed_c5_exceptional_controls.json")
R114_EXCEPTIONAL_SHA256 = (
    "d8e30dfc7975dd885543211188654dda14d3808d7e3aa21bc2e2aa9b5b3ab0e7"
)
R114_LOGS = pathlib.Path("factor_logs_and_identical_descent_r114.json")
R114_LOGS_SHA256 = (
    "a9392bf49a42c8acacb40b5ebc118e525fefee1adabac158201dff470c7a8118"
)
R114_GATE = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_leaf_generator_probe_gate_r114.md"
)
R114_GATE_SHA256 = (
    "6574145a6d67ec4e55a8bf0cf90c3a0c9f858aa4cddbfd7e9a733c1033b3a3da"
)
R114_PARENT = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_"
    "leaf_generator_probe_parent_report_r114.yaml"
)
R114_PARENT_SHA256 = (
    "4acc68f562bbed3ec506cc0cbf4f1223fb24d151b602c92494ae75a74f081287"
)
IDEA_REGISTRY = pathlib.Path("p1553_r35_artifact_index_README.md")
IDEA_REGISTRY_SHA256 = (
    "9f4371eefd5e4019833eef858e3bda79d41aff0c5d7b861c71a5987a96acc392"
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


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r114_producer", R114_PRODUCER, R114_PRODUCER_SHA256),
        ("r114_report", R114_REPORT, R114_REPORT_SHA256),
        ("r114_frozen", R114_FROZEN, R114_FROZEN_SHA256),
        ("r114_ledger", R114_LEDGER, R114_LEDGER_SHA256),
        ("r114_replay", R114_REPLAY, R114_REPLAY_SHA256),
        ("r114_exceptional", R114_EXCEPTIONAL, R114_EXCEPTIONAL_SHA256),
        ("r114_logs", R114_LOGS, R114_LOGS_SHA256),
        ("r114_gate", R114_GATE, R114_GATE_SHA256),
        ("r114_parent", R114_PARENT, R114_PARENT_SHA256),
        ("idea_registry", IDEA_REGISTRY, IDEA_REGISTRY_SHA256),
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
        raise AssertionError(f"R115 source binding mismatch: {failures}")
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


def semaev_degree(arity: int) -> dict[str, Any]:
    if arity < 2:
        raise ValueError("Semaev arity must be at least two")
    per_variable = 2 ** (arity - 2)
    total = (arity - 1) * per_variable
    return {
        "arity": arity,
        "degree_per_variable": per_variable,
        "total_degree": total,
    }


def best_explicit_split(
    relation_arity: int,
    alpha: Fraction,
    beta: Fraction,
) -> dict[str, Any]:
    source_exponent = relation_arity * (alpha + beta)
    candidates = []
    for a_count in range(relation_arity + 1):
        for c_count in range(relation_arity + 1):
            if (a_count, c_count) in {
                (0, 0),
                (relation_arity, relation_arity),
            }:
                continue
            left = a_count * alpha + c_count * beta
            right = source_exponent - left
            candidates.append(
                (
                    max(left, right),
                    min(left, right),
                    abs(left - right),
                    a_count,
                    c_count,
                    left,
                    right,
                )
            )
    best = min(candidates)
    return {
        "left_a_count": best[3],
        "left_c_count": best[4],
        "right_a_count": relation_arity - best[3],
        "right_c_count": relation_arity - best[4],
        "left_exponent_B": fraction_record(best[5]),
        "right_exponent_B": fraction_record(best[6]),
        "work_exponent_B": fraction_record(best[0]),
        "state_exponent_B": fraction_record(best[1]),
        "below_rho": best[0] < RHO_EXPONENT,
        "inside_setup_cap": best[1] <= SETUP_CAP,
        "inside_online_cap": best[0] <= ONLINE_CAP,
    }


def transposed_prefix_profile(
    relation_arity: int,
    alpha: Fraction,
    beta: Fraction,
) -> list[dict[str, Any]]:
    output = []
    for c_depth in range(relation_arity + 1):
        exponent = relation_arity * alpha + c_depth * beta
        output.append(
            {
                "c_prefix_depth": c_depth,
                "typed_occurrence_exponent_B": fraction_record(exponent),
                "inside_setup_cap": exponent <= SETUP_CAP,
                "inside_online_cap": exponent <= ONLINE_CAP,
            }
        )
    return output


def evaluate_regime(
    relation_arity: int,
    alpha: Fraction,
    beta: Fraction,
    *,
    regime_id: str,
) -> dict[str, Any]:
    if relation_arity < 2:
        raise ValueError("relation arity must be at least two")
    if alpha < 0 or beta < 0:
        raise ValueError("deck exponents must be nonnegative")
    if alpha > beta:
        raise ValueError("regimes use the canonical alpha <= beta orientation")

    factor_base_exponent = alpha + beta
    source_exponent = relation_arity * factor_base_exponent
    retry_exponent = max(Fraction(0), LOG_B_GROUP_ORDER - source_exponent)
    meaningful_rank_exponent = beta
    terminal_a_exponent = relation_arity * alpha
    first_interface_exponent = terminal_a_exponent + beta
    fresh_total_exponent = retry_exponent + first_interface_exponent
    relation_phase_exponent = meaningful_rank_exponent + fresh_total_exponent
    sparse_linear_algebra_exponent = 2 * meaningful_rank_exponent
    local_a3_query_exponent = LOCAL_HALF_ARITY * alpha
    local_c3_state_exponent = LOCAL_HALF_ARITY * beta
    conditional_total_exponent = max(
        factor_base_exponent,
        local_c3_state_exponent,
        fresh_total_exponent,
        relation_phase_exponent,
        sparse_linear_algebra_exponent,
    )
    profile = transposed_prefix_profile(relation_arity, alpha, beta)
    first_online_overflow = next(
        (
            row["c_prefix_depth"]
            for row in profile
            if not row["inside_online_cap"]
        ),
        None,
    )
    first_setup_overflow = next(
        (
            row["c_prefix_depth"]
            for row in profile
            if not row["inside_setup_cap"]
        ),
        None,
    )
    expanded_arity = 2 * relation_arity + 1
    factor_arity = relation_arity + 1
    necessary_checks = {
        "both_decks_polynomially_nonempty": alpha > 0 and beta > 0,
        "dense_source_body": source_exponent >= LOG_B_GROUP_ORDER,
        "factor_base_materialization_inside_setup": (
            factor_base_exponent <= SETUP_CAP
        ),
        "r82_local_a3_query_inside_online": (
            local_a3_query_exponent <= ONLINE_CAP
        ),
        "r82_local_c3_state_inside_setup": (
            local_c3_state_exponent <= SETUP_CAP
        ),
        "first_transposed_interface_inside_online": (
            first_interface_exponent <= ONLINE_CAP
        ),
        "density_adjusted_fresh_work_inside_online": (
            fresh_total_exponent <= ONLINE_CAP
        ),
        "conditional_relation_collection_inside_setup": (
            relation_phase_exponent <= SETUP_CAP
        ),
        "conditional_sparse_linear_algebra_inside_setup": (
            sparse_linear_algebra_exponent <= SETUP_CAP
        ),
        "conditional_total_below_rho": (
            conditional_total_exponent < RHO_EXPONENT
        ),
    }
    return {
        "regime_id": regime_id,
        "relation_arity_m": relation_arity,
        "alpha_A_exponent_B": fraction_record(alpha),
        "beta_C_exponent_B": fraction_record(beta),
        "factor_base_exponent_B": fraction_record(factor_base_exponent),
        "full_source_body_exponent_B": fraction_record(source_exponent),
        "density_retry_exponent_B": fraction_record(retry_exponent),
        "meaningful_log_rank_exponent_B": fraction_record(
            meaningful_rank_exponent
        ),
        "terminal_a_m_support_exponent_B": fraction_record(
            terminal_a_exponent
        ),
        "first_transposed_interface_exponent_B": fraction_record(
            first_interface_exponent
        ),
        "density_adjusted_fresh_work_exponent_B": fraction_record(
            fresh_total_exponent
        ),
        "conditional_relation_collection_exponent_B": fraction_record(
            relation_phase_exponent
        ),
        "conditional_sparse_linear_algebra_exponent_B": fraction_record(
            sparse_linear_algebra_exponent
        ),
        "r82_local_a3_query_exponent_B": fraction_record(
            local_a3_query_exponent
        ),
        "r82_local_c3_state_exponent_B": fraction_record(
            local_c3_state_exponent
        ),
        "conditional_total_exponent_B": fraction_record(
            conditional_total_exponent
        ),
        "conditional_headroom_below_rho_B": fraction_record(
            RHO_EXPONENT - conditional_total_exponent
        ),
        "best_explicit_split": best_explicit_split(
            relation_arity,
            alpha,
            beta,
        ),
        "transposed_prefix_profile": profile,
        "first_online_overflow_depth": first_online_overflow,
        "first_setup_overflow_depth": first_setup_overflow,
        "summation_polynomials": {
            "factor_level": semaev_degree(factor_arity),
            "expanded_atom_level": semaev_degree(expanded_arity),
            "fixed_arity_degree_has_B_exponent_zero": True,
            "fixed_arity_degree_constant_is_not_free": True,
        },
        "necessary_checks": necessary_checks,
        "necessary_exponent_envelope_pass": all(necessary_checks.values()),
        "implicit_source_locator_constructed": False,
        "known_rhs_rank_theorem_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "candidate_work_credit": False,
    }


def dense_online_boundary(relation_arity: int) -> dict[str, Any]:
    factor_base_exponent = LOG_B_GROUP_ORDER / relation_arity
    if relation_arity <= 4:
        return {
            "relation_arity_m": relation_arity,
            "positive_boundary_exists": False,
            "reason": (
                "dense source requires alpha+beta>=5/m, while a positive "
                "alpha makes m*alpha+beta>5/m>=5/4"
            ),
        }
    alpha = (
        ONLINE_CAP - factor_base_exponent
    ) / (relation_arity - 1)
    beta = factor_base_exponent - alpha
    row = evaluate_regime(
        relation_arity,
        alpha,
        beta,
        regime_id=f"dense_online_boundary_m{relation_arity}",
    )
    row["positive_boundary_exists"] = True
    return row


def exact_controls() -> list[dict[str, Any]]:
    return [
        evaluate_regime(
            5,
            Fraction(2, 5),
            Fraction(3, 5),
            regime_id="r82_r114_original_m5",
        ),
        evaluate_regime(
            5,
            Fraction(1, 16),
            Fraction(15, 16),
            regime_id="m5_online_boundary_local_c3_overflow",
        ),
        evaluate_regime(
            5,
            Fraction(1, 4),
            Fraction(3, 4),
            regime_id="m5_local_c3_boundary_online_overflow",
        ),
        evaluate_regime(
            6,
            Fraction(5, 12),
            Fraction(5, 12),
            regime_id="m6_balanced_control",
        ),
        evaluate_regime(
            6,
            Fraction(1, 12),
            Fraction(2, 3),
            regime_id="m6_sparse_source_retry_control",
        ),
    ]


def feasibility_ledger() -> dict[str, Any]:
    boundary_rows = [
        dense_online_boundary(arity)
        for arity in range(MIN_ARITY, MAX_ARITY + 1)
    ]
    passing = [
        row
        for row in boundary_rows
        if row.get("necessary_exponent_envelope_pass")
    ]
    if not passing:
        raise AssertionError("no necessary exponent regime found")
    selected = passing[0]
    if (
        selected["relation_arity_m"] != 6
        or selected["alpha_A_exponent_B"]["exact"] != "1/12"
        or selected["beta_C_exponent_B"]["exact"] != "3/4"
    ):
        raise AssertionError("unexpected minimal R115 boundary vertex")
    return {
        "schema": (
            "p1553.relation_arity_factor_base_feasibility_ledger.r115.v1"
        ),
        "normalization": {
            "group_order_N": "B^(5+o(1))",
            "factor_base": "F=A+C",
            "deck_sizes": "|A|=B^(alpha+o(1)), |C|=B^(beta+o(1))",
            "relation": "sum of m factor points equals a public target",
            "expanded_relation": "m A atoms plus m C atoms",
        },
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
            "rho_exponent_B": fraction_record(RHO_EXPONENT),
        },
        "exact_inequalities": {
            "dense_relation_supply": "m*(alpha+beta)>=5",
            "density_retry": "delta=max(0,5-m*(alpha+beta))",
            "meaningful_log_rank": "d=max(alpha,beta)=beta",
            "first_transposed_interface": "w1=m*alpha+beta",
            "fresh_target": "delta+w1<=5/4",
            "relation_collection": "beta+delta+w1<=9/4",
            "sparse_linear_algebra": "2*beta<=9/4",
            "r82_local_3f_compiler": "3*alpha<=5/4 and 3*beta<=9/4",
        },
        "minimal_arity_proof": {
            "m_at_most_4": (
                "dense supply and alpha>0 force m*alpha+beta>5/m>=5/4"
            ),
            "m_equal_5": (
                "w1<=5/4 forces alpha<=1/16 and beta>=15/16, so "
                "3*beta>=45/16>9/4"
            ),
            "m_equal_6_witness": (
                "alpha=1/12, beta=3/4 meets dense supply, w1=5/4, "
                "3*beta=9/4, and 3*alpha=1/4"
            ),
            "minimal_passing_arity": selected["relation_arity_m"],
        },
        "boundary_family": boundary_rows,
        "selected_vertex": selected,
        "controls": exact_controls(),
        "arity_scope": {
            "fixed_integer_arities_only": True,
            "growing_arity_credit": False,
            "reason": (
                "Semaev degree per variable is 2^(n-2); if arity grows "
                "with log B its bit and polynomial costs acquire a nonzero "
                "B exponent and require a separate model"
            ),
        },
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    ledger = feasibility_ledger()
    selected = ledger["selected_vertex"]
    split = selected["best_explicit_split"]
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "exact_rational_inequalities_frozen": True,
        "arity_three_through_thirty_two_enumerated": (
            len(ledger["boundary_family"]) == MAX_ARITY - MIN_ARITY + 1
        ),
        "minimal_preserved_compiler_arity_is_six": (
            ledger["minimal_arity_proof"]["minimal_passing_arity"] == 6
        ),
        "m6_alpha_1o12_beta_3o4_vertex_exact": (
            selected["alpha_A_exponent_B"]["exact"] == "1/12"
            and selected["beta_C_exponent_B"]["exact"] == "3/4"
        ),
        "m6_dense_source_body_exact": (
            selected["full_source_body_exponent_B"]["exact"] == "5"
        ),
        "m6_first_interface_and_local_c3_caps_exact": (
            selected["first_transposed_interface_exponent_B"]["exact"]
            == "5/4"
            and selected["r82_local_c3_state_exponent_B"]["exact"]
            == "9/4"
        ),
        "conditional_relation_and_linear_algebra_charged": (
            selected["conditional_relation_collection_exponent_B"]["exact"]
            == "2"
            and selected[
                "conditional_sparse_linear_algebra_exponent_B"
            ]["exact"]
            == "3/2"
        ),
        "factor_and_atom_semaev_degrees_charged": (
            selected["summation_polynomials"]["factor_level"][
                "degree_per_variable"
            ]
            == 32
            and selected["summation_polynomials"]["expanded_atom_level"][
                "degree_per_variable"
            ]
            == 2048
        ),
        "explicit_three_f_join_remains_rho": (
            split["work_exponent_B"]["exact"] == "5/2"
            and not split["below_rho"]
        ),
        "implicit_3f_self_convolution_locator_complete": False,
        "known_rhs_rank_theorem_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "At the frozen m=6, alpha=1/12, beta=3/4 vertex, construct or "
        "refute an exact implicit 3F self-convolution/source locator for "
        "R=X+(R-X). It may retain the B^(9/4) C3 dictionary and B^(1/4) "
        "single-membership query, but may not enumerate the B^(5/2) 3F "
        "occurrences. Realize and charge either the factor-level S7 or "
        "expanded-atom S13 summation-polynomial/FFE contraction, return six "
        "coupled factor sources, and preserve known-RHS rank, factor logs, "
        "and identical descent."
    )
    frozen = {
        "schema": (
            "p1553.frozen_relation_arity_factor_base_exponent_model.r115.v1"
        ),
        "source_bindings": source_binding_records(),
        "normalization": ledger["normalization"],
        "caps": ledger["caps"],
        "inequalities": ledger["exact_inequalities"],
        "selected_vertex": selected,
        "literature_boundary": {
            "semaev_original": "https://eprint.iacr.org/2004/031",
            "summation_polynomial_degree": (
                "S_n has degree 2^(n-2) in each variable"
            ),
            "shoup_generic_lower_bound": (
                "https://www.shoup.net/papers/dlbounds1.pdf"
            ),
            "representation_specific_route_only": True,
        },
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "R82 Cartesian addition-pushforward factor base",
                "R83-R114 colored sum, FFE, endpoint, and transpose screens",
            ],
            "new_local_control_only": (
                "minimal fixed arity preserving the R82 local compiler and "
                "the R114 first transposed-interface cap"
            ),
            "new_algorithm_claimed": False,
        },
    }
    cap_table = {
        "schema": "p1553.transposed_interface_cap_table.r115.v1",
        "boundary_family": ledger["boundary_family"],
        "controls": ledger["controls"],
        "selected_vertex_id": selected["regime_id"],
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r115.v1",
        "necessary_exponent_envelope_complete": True,
        "public_rectangle_kernel_available": True,
        "meaningful_rank_exponent_B": selected[
            "meaningful_log_rank_exponent_B"
        ],
        "conditional_relation_collection_exponent_B": selected[
            "conditional_relation_collection_exponent_B"
        ],
        "conditional_sparse_linear_algebra_exponent_B": selected[
            "conditional_sparse_linear_algebra_exponent_B"
        ],
        "implicit_source_locator_complete": False,
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
            "NECESSARY_EXPONENT_VERTEX_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "MINIMAL_DENSE_FIXED_ARITY_PRESERVING_R82_LOCAL_3F_COMPILER_"
            "AND_R114_FIRST_TRANSPOSED_INTERFACE_IS_M6_ALPHA1O12_BETA3O4__"
            "FACTOR_BASE_B5O6__MEANINGFUL_RANK_B3O4__CONDITIONAL_RELATION_"
            "COLLECTION_B2__SPARSE_LINEAR_ALGEBRA_B3O2__FACTOR_S7_DEGREE32_"
            "ATOM_S13_DEGREE2048__DIRECT_3F_SELF_JOIN_REMAINS_B5O2_RHO__"
            "IMPLICIT_SELF_CONVOLUTION_LOCATOR_RANK_LOGS_DESCENT_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "selected_vertex": selected,
        "cost_ledger": {
            "caps": ledger["caps"],
            "conditional_total_exponent_B": selected[
                "conditional_total_exponent_B"
            ],
            "conditional_headroom_below_rho_B": selected[
                "conditional_headroom_below_rho_B"
            ],
            "explicit_split": split,
            "conditional_only": True,
            "inside_cap_algorithm_constructed": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "necessary_exponent_envelope_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen_model": (
                "frozen_relation_arity_factor_base_exponent_model.json"
            ),
            "feasibility_ledger": (
                "relation_arity_factor_base_feasibility_ledger.json"
            ),
            "cap_table": "transposed_interface_cap_table.json",
            "logs_descent": "factor_logs_and_identical_descent_r115.json",
        },
        "next_action": next_action,
        "non_claims": [
            "A feasible exponent vertex is not a source-locator algorithm.",
            "Conditional relation counts do not prove relation independence.",
            "Fixed-arity Semaev degree has exponent zero but is not free.",
            "The direct implicit-half join remains Pollard-rho scale.",
            "No factor logs or identical target descent are supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_M6_ALPHA1O12_BETA3O4_NECESSARY_EXPONENT_VERTEX_ONLY__"
            "REJECT_M5_WHEN_R82_LOCAL_3F_COMPILER_AND_R114_ONLINE_INTERFACE_"
            "ARE_BOTH_REQUIRED__DIRECT_3F_SELF_JOIN_EQUALS_RHO__PRESERVE_"
            "IMPLICIT_S7_OR_S13_SELF_CONVOLUTION_SOURCE_LOCATOR__NO_LOCATOR__"
            "NO_RANK_THEOREM__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__"
            "NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "ledger": ledger,
        "cap_table": cap_table,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_relation_arity_factor_base_transposed_"
            "interface_rebalance_probe_report_r115.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_relation_arity_factor_base_exponent_model.json"
        ),
    )
    parser.add_argument(
        "--ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "relation_arity_factor_base_feasibility_ledger.json"
        ),
    )
    parser.add_argument(
        "--cap-table-output",
        type=pathlib.Path,
        default=pathlib.Path("transposed_interface_cap_table.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r115.json"
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
    write_json(args.ledger_output, bundle["ledger"])
    write_json(args.cap_table_output, bundle["cap_table"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R115 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
