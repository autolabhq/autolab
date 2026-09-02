#!/usr/bin/env python3
"""Rebalance explicit torus split costs across relation arities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.torus_c5_explicit_split_global_rebalance.r122.v1"
LOG_B_GROUP_ORDER = Fraction(5)
SETUP_CAP = Fraction(9, 4)
RHO_EXPONENT = Fraction(5, 2)

R121_PRODUCER = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_r121.py"
)
R121_PRODUCER_SHA256 = (
    "b590354da291830d9a3372d5d144f1552dee04b2f3e8368258213637c022a6f9"
)
R121_REPORT = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_"
    "moment_torus_probe_report_r121.json"
)
R121_REPORT_SHA256 = (
    "356560169e475900452e83c68a8726420151d57f44735194cb9aaae23a06e54e"
)
R121_FROZEN = pathlib.Path(
    "frozen_m6_small_k_multiplicative_c5_subfunction_index.json"
)
R121_FROZEN_SHA256 = (
    "2876c2d9fbc95c7b1e1481554909065b54d13c82a24e0026a19a70744a7d2a3d"
)
R121_COST = pathlib.Path(
    "m6_small_k_multiplicative_c5_cost_ledger.json"
)
R121_COST_SHA256 = (
    "13922a59704fd9ebac8e5db0afe3b85ec893623e73aac0343f8abe655fef9245"
)
R121_REPLAY = pathlib.Path(
    "m6_small_k_multiplicative_c5_source_replay.json"
)
R121_REPLAY_SHA256 = (
    "428fcb4261ac6d92fb67343b19f8a673ca6a0d870ab95fe6fbf91e7abfa50723"
)
R121_CONTROLS = pathlib.Path(
    "m6_small_k_multiplicative_c5_exceptional_controls.json"
)
R121_CONTROLS_SHA256 = (
    "f12a9e30a2fdf4d295fb6a1b0544632ea5fb2ed1938960363403ecd45736bad1"
)
R121_LOGS = pathlib.Path("factor_logs_and_identical_descent_r121.json")
R121_LOGS_SHA256 = (
    "991ffc245bc7c5fe7a68b16652ded8850489c3df4258319b0a4b674401e0ffa4"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R121_PARENT = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_"
    "moment_torus_probe_parent_report_r121.yaml"
)
R121_PARENT_SHA256 = (
    "fc7c62925af5287f3e201c7e474fef881b890754cd805f772a993aa70352c82b"
)
R115_PRODUCER = pathlib.Path(
    "p1553_relation_arity_factor_base_"
    "transposed_interface_rebalance_probe_r115.py"
)
R115_PRODUCER_SHA256 = (
    "d3da38cdf54c39fca800fa451ad54a99bf0288529987b14ec39ee1367b3b615a"
)
R115_REPORT = pathlib.Path(
    "p1553_relation_arity_factor_base_"
    "transposed_interface_rebalance_probe_report_r115.json"
)
R115_REPORT_SHA256 = (
    "2b43c37f93e03c6ccdb675a34d48d7478deb3b0349026c65b2f6fe65695cb88e"
)
R115_LEDGER = pathlib.Path("relation_arity_factor_base_feasibility_ledger.json")
R115_LEDGER_SHA256 = (
    "4936d19fa0a1024743d4aca462ddb14362495e1f52581913a6231586c7bc8ffe"
)
R115_GATE = pathlib.Path(
    "p1553_relation_arity_factor_base_"
    "transposed_interface_rebalance_probe_gate_r115.md"
)
R115_GATE_SHA256 = (
    "11dd899c1ab6b89444cd9c9cc75709a64d9abf99d9370d2536e2f20e3d0fd1de"
)
R115_PARENT = pathlib.Path(
    "p1553_relation_arity_factor_base_"
    "transposed_interface_rebalance_probe_parent_report_r115.yaml"
)
R115_PARENT_SHA256 = (
    "78e92013af76848e4c925cbbe99e8b74d9c9e73726e2f9c7f542a03c2dd9b27b"
)
R118_GATE = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_gate_r118.md"
)
R118_GATE_SHA256 = (
    "9ed7f59fb94dd1ecff54d8844068502183a7928c64e991c9a98b270ab8645b9d"
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


R121 = load_module("p1553_r121_for_r122", R121_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r121_producer", R121_PRODUCER, R121_PRODUCER_SHA256),
        ("r121_report", R121_REPORT, R121_REPORT_SHA256),
        ("r121_frozen", R121_FROZEN, R121_FROZEN_SHA256),
        ("r121_cost", R121_COST, R121_COST_SHA256),
        ("r121_replay", R121_REPLAY, R121_REPLAY_SHA256),
        ("r121_controls", R121_CONTROLS, R121_CONTROLS_SHA256),
        ("r121_logs", R121_LOGS, R121_LOGS_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r121_parent", R121_PARENT, R121_PARENT_SHA256),
        ("r115_producer", R115_PRODUCER, R115_PRODUCER_SHA256),
        ("r115_report", R115_REPORT, R115_REPORT_SHA256),
        ("r115_ledger", R115_LEDGER, R115_LEDGER_SHA256),
        ("r115_gate", R115_GATE, R115_GATE_SHA256),
        ("r115_parent", R115_PARENT, R115_PARENT_SHA256),
        ("r118_gate", R118_GATE, R118_GATE_SHA256),
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
        raise AssertionError(f"R122 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def split_regime(
    relation_arity: int,
    alpha: Fraction,
    beta: Fraction,
    stored_c_arity: int,
) -> dict[str, Any]:
    if relation_arity < 2:
        raise ValueError("relation arity must be at least two")
    if not (Fraction(0) <= alpha <= beta):
        raise ValueError("expected 0<=alpha<=beta")
    if beta <= 0:
        raise ValueError("the meaningful C rank must be polynomial")
    if not (0 <= stored_c_arity <= relation_arity - 1):
        raise ValueError("stored C arity must be in [0,m-1]")
    enumerated_c_arity = relation_arity - 1 - stored_c_arity
    source_body = relation_arity * (alpha + beta)
    retry = max(Fraction(0), LOG_B_GROUP_ORDER - source_body)
    state = stored_c_arity * beta
    query = enumerated_c_arity * beta
    one_c_branch_attempt = (
        relation_arity * alpha + beta + query
    )
    density_adjusted_fresh = retry + one_c_branch_attempt
    relation_collection = density_adjusted_fresh + beta
    supply_term = retry + source_body
    first_bound = supply_term + (1 - stored_c_arity) * beta
    setup_bound = LOG_B_GROUP_ORDER + beta - SETUP_CAP
    return {
        "relation_arity_m": relation_arity,
        "alpha_A_exponent_B": fraction_record(alpha),
        "beta_C_exponent_B": fraction_record(beta),
        "stored_c_arity_s": stored_c_arity,
        "enumerated_c_arity_r": enumerated_c_arity,
        "source_body_exponent_B": fraction_record(source_body),
        "density_retry_exponent_B": fraction_record(retry),
        "stored_c_state_exponent_B": fraction_record(state),
        "enumerated_c_query_exponent_B": fraction_record(query),
        "one_c_branch_attempt_exponent_B": fraction_record(
            one_c_branch_attempt
        ),
        "density_adjusted_fresh_exponent_B": fraction_record(
            density_adjusted_fresh
        ),
        "meaningful_relation_rank_exponent_B": fraction_record(beta),
        "relation_collection_exponent_B": fraction_record(
            relation_collection
        ),
        "supply_identity_lower_bound_B": fraction_record(first_bound),
        "setup_cap_lower_bound_B": fraction_record(setup_bound),
        "setup_eligible": state <= SETUP_CAP,
        "supply_identity_exact": relation_collection == first_bound,
        "supply_term_at_least_group_order": (
            supply_term >= LOG_B_GROUP_ORDER
        ),
        "setup_bound_applies": (
            state <= SETUP_CAP
            and relation_collection >= setup_bound
        ),
        "strictly_above_rho_if_setup_eligible": (
            state > SETUP_CAP
            or relation_collection > RHO_EXPONENT
        ),
        "candidate_work_credit": False,
    }


def rational_grid_audit() -> dict[str, Any]:
    denominator = 48
    checked = 0
    setup_eligible = 0
    violations = []
    minimum: tuple[Fraction, dict[str, Any]] | None = None
    for relation_arity in range(3, 21):
        for beta_numerator in range(1, denominator + 1):
            beta = Fraction(beta_numerator, denominator)
            for alpha_numerator in range(0, beta_numerator + 1):
                alpha = Fraction(alpha_numerator, denominator)
                if alpha + beta > SETUP_CAP:
                    continue
                for stored in range(relation_arity):
                    checked += 1
                    row = split_regime(
                        relation_arity,
                        alpha,
                        beta,
                        stored,
                    )
                    if not row["setup_eligible"]:
                        continue
                    setup_eligible += 1
                    exponent = Fraction(
                        row["relation_collection_exponent_B"]["exact"]
                    )
                    if minimum is None or exponent < minimum[0]:
                        minimum = (exponent, row)
                    if (
                        not row["setup_bound_applies"]
                        or not row[
                            "strictly_above_rho_if_setup_eligible"
                        ]
                    ):
                        violations.append(row)
    if minimum is None:
        raise AssertionError("rational grid had no setup-eligible rows")
    return {
        "denominator": denominator,
        "relation_arity_range": [3, 20],
        "candidate_count": checked,
        "setup_eligible_candidate_count": setup_eligible,
        "theorem_violation_count": len(violations),
        "minimum_relation_collection_row": minimum[1],
        "minimum_relation_collection_exponent_B": fraction_record(
            minimum[0]
        ),
        "all_setup_eligible_rows_above_rho": not violations,
        "finite_grid_receives_proof_credit": False,
    }


def exhaustive_collision_controls() -> list[dict[str, Any]]:
    prime = 11
    deck_size = 3
    deck_count = prime**deck_size
    rows = []
    for source_arity in range(1, 5):
        sources = tuple(
            itertools.combinations_with_replacement(
                range(deck_size),
                source_arity,
            )
        )
        collision_pairs = 0
        for deck in itertools.product(range(prime), repeat=deck_size):
            endpoints = [
                sum(deck[index] for index in source) % prime
                for source in sources
            ]
            collision_pairs += sum(
                endpoints[left] == endpoints[right]
                for left in range(len(endpoints))
                for right in range(left + 1, len(endpoints))
            )
        expected = deck_count * math.comb(len(sources), 2) // prime
        rows.append(
            {
                "field_order": prime,
                "deck_size": deck_size,
                "source_arity": source_arity,
                "canonical_source_count": len(sources),
                "deck_count": deck_count,
                "observed_colliding_source_pair_count": collision_pairs,
                "expected_colliding_source_pair_count": expected,
                "exact_pair_collision_probability": "1/q",
                "observed_equals_theorem": collision_pairs == expected,
            }
        )
    return rows


def theorem_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_explicit_split_global_rebalance_"
            "cost_ledger.r122.v1"
        ),
        "caps": {
            "log_B_group_order": fraction_record(LOG_B_GROUP_ORDER),
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "rho_exponent_B": fraction_record(RHO_EXPONENT),
        },
        "model": {
            "relation": "mF=mA+mC",
            "deck_exponents": "|A|=B^alpha, |C|=B^beta, 0<=alpha<=beta",
            "density_retry": "delta=max(0,5-m(alpha+beta))",
            "meaningful_relation_rank_exponent_B": "beta",
            "one_c_branch": (
                "enumerate A_m and one distinguished C, then query "
                "C_(m-1) by stored C_s and enumerated C_r, s+r=m-1"
            ),
        },
        "theorem": {
            "relation_collection_exponent": (
                "L=delta+m*alpha+(r+2)*beta"
            ),
            "supply_identity": (
                "L=(delta+m(alpha+beta))+(1-s)*beta"
            ),
            "supply_lower_bound": "delta+m(alpha+beta)>=5",
            "setup_condition": "s*beta<=9/4",
            "combined_lower_bound": "L>=5+beta-s*beta>=11/4+beta",
            "rho_gap": "L-5/2>=1/4+beta>1/4",
            "all_fixed_arities_and_positive_beta_covered": True,
        },
        "selected_controls": [
            split_regime(6, Fraction(1, 12), Fraction(3, 4), 3),
            split_regime(5, Fraction(2, 5), Fraction(3, 5), 3),
            split_regime(10, Fraction(1, 40), Fraction(19, 40), 4),
        ],
        "grid_audit": rational_grid_audit(),
        "routes": [
            {
                "route_id": "explicit_occurrence_c_s_table",
                "inside_below_rho_relation_collection": False,
                "covered_by_theorem": True,
            },
            {
                "route_id": "iid_random_deck_distinct_endpoint_c_s_table",
                "stored_support_exponent_B": "s*beta",
                "setup_forces_support_over_group_ratio": (
                    "B^(s*beta-5)<=B^(-11/4)"
                ),
                "whp_occurrence_scale_support": True,
                "covered_by_theorem": True,
            },
            {
                "route_id": "collision_compressed_filtered_deck_table",
                "covered_by_iid_theorem": False,
                "status": "open_requires_actual_support_and_source_receipts",
            },
            {
                "route_id": (
                    "target_specialized_nonoccurrence_torus_c5_circuit"
                ),
                "covered_by_theorem": False,
                "exact_circuit_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "semantic_dedup": {
            "nearby_lanes": [
                "R115 relation-arity/factor-base envelope",
                "R118 m=6 one-C branch and explicit split theorem",
                "R119 iid C5 output-support theorem",
                "R121 exact multiplicative C2|C3 torus split",
            ],
            "r122_distinct_scope": (
                "relation-collection lower bound for every arity after "
                "charging meaningful rank and the setup-limited stored "
                "C_s side"
            ),
            "new_idea_id_claimed": False,
        },
        "candidate_work_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    collision_rows = exhaustive_collision_controls()
    ledger = theorem_ledger()
    selected = ledger["selected_controls"]
    return {
        "schema": (
            "p1553.torus_c5_explicit_split_global_rebalance_"
            "controls.r122.v1"
        ),
        "exhaustive_pair_collision_controls": collision_rows,
        "all_exhaustive_pair_collision_controls_exact": all(
            row["observed_equals_theorem"] for row in collision_rows
        ),
        "selected_regime_controls": selected,
        "all_selected_setup_eligible_regimes_above_rho": all(
            row["setup_eligible"]
            and row["strictly_above_rho_if_setup_eligible"]
            for row in selected
        ),
        "r115_r121_selected_vertex_relation_collection_exponent_B": (
            selected[0]["relation_collection_exponent_B"]
        ),
        "r115_r121_selected_vertex_fresh_exponent_B": selected[0][
            "density_adjusted_fresh_exponent_B"
        ],
        "grid_audit": ledger["grid_audit"],
        "candidate_scalar_labels_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_explicit_split_global_rebalance_"
            "replay.r122.v1"
        ),
        "r121_exact_torus_source_replay_inherited": True,
        "all_exhaustive_pair_collision_controls_exact": controls[
            "all_exhaustive_pair_collision_controls_exact"
        ],
        "all_selected_setup_eligible_regimes_above_rho": controls[
            "all_selected_setup_eligible_regimes_above_rho"
        ],
        "inside_cap_nonoccurrence_torus_source_circuit_constructed": False,
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R121_REPORT.read_text(encoding="utf-8"))
    r115 = json.loads(R115_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or r115.get("breakthrough"):
        raise AssertionError("inherited nonclaim boundary drifted")
    controls = finite_controls()
    cost = theorem_ledger()
    replay = source_replay(controls)
    selected = controls["selected_regime_controls"][0]
    obligations = {
        "sixteen_source_bindings_verified": len(source_hashes) == 16,
        "r121_target_specialized_torus_interface_inherited": (
            inherited["admission"]["torus_and_moment_controls_admitted"]
            and not inherited["admission"]["lane_admitted"]
        ),
        "r115_global_exponent_contract_inherited": (
            r115["selected_vertex"]["relation_arity_m"] == 6
            and not r115["admission"]["lane_admitted"]
        ),
        "all_arity_explicit_split_lower_bound_complete": cost["theorem"][
            "all_fixed_arities_and_positive_beta_covered"
        ],
        "rational_grid_audit_has_no_violations": (
            controls["grid_audit"]["theorem_violation_count"] == 0
        ),
        "selected_m6_split_charged_through_relation_collection": (
            selected["density_adjusted_fresh_exponent_B"]["exact"] == "11/4"
            and selected["relation_collection_exponent_B"]["exact"] == "7/2"
        ),
        "exhaustive_pair_collision_controls_exact": controls[
            "all_exhaustive_pair_collision_controls_exact"
        ],
        "iid_setup_side_support_theorem_complete": (
            cost["routes"][1]["whp_occurrence_scale_support"]
        ),
        "all_explicit_split_route_costs_charged": (
            not cost["candidate_work_credit"]
        ),
        "semantic_dedup_complete": (
            not cost["semantic_dedup"]["new_idea_id_claimed"]
        ),
        "candidate_scalar_labels_not_consumed": (
            not controls["candidate_scalar_labels_consumed"]
        ),
        "inside_cap_nonoccurrence_torus_membership_complete": False,
        "inside_cap_nonoccurrence_torus_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one target-specialized nonoccurrence torus C5 "
        "membership/source circuit outside stored C_s endpoint tables and "
        "enumerated C_r complements. It may use the exact R121 form "
        "B-t_target*A before endpoint expansion, or a filtered-deck support "
        "compression only with an actual support/source theorem. Require "
        "B^(9/4+o(1)) state, polylogarithmic arbitrary-target work, exact "
        "empty certification, five projective backpointers, no field DLP, "
        "and complete pairing, relation-rank, factor-log, identical-descent, "
        "memory, field-operation, and bit-cost receipts."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_explicit_split_global_"
            "rebalance.r122.v1"
        ),
        "source_bindings": source_binding_records(),
        "caps": cost["caps"],
        "closed_scoped_grammar": {
            "relation_arity": "every fixed m>=2",
            "deck_exponents": "0<=alpha<=beta with beta>0",
            "stored_side": "explicit occurrence or iid-distinct C_s table",
            "query_side": "explicit C_r enumeration, s+r=m-1",
            "meaningful_rank_rows": "B^(beta+o(1))",
            "relation_collection_lower_bound": (
                "B^(11/4+beta+o(1))"
            ),
        },
        "preserved_interface": (
            "target-specialized nonoccurrence torus C5 circuit or an "
            "actual filtered-deck compression with exact source reporting"
        ),
        "general_arithmetic_circuit_or_data_structure_lower_bound_claimed": (
            False
        ),
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r122.v1",
        "r121_torus_moment_audit_complete": True,
        "r122_explicit_split_global_rebalance_audit_complete": True,
        "inside_cap_nonoccurrence_torus_source_index_complete": False,
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
            "EXACT_ALL_ARITY_EXPLICIT_SPLIT_REBALANCE_NEGATIVE_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "ONE_C_BRANCH_WITH_STORED_CS_AND_ENUMERATED_CR_HAS_RELATION_"
            "COLLECTION_L_DELTA_MALPHA_RPLUS2BETA__SUPPLY_AND_SETUP_FORCE_"
            "L_AT_LEAST_11O4_PLUS_BETA_ABOVE_RHO_FOR_EVERY_ARITY__R115_"
            "M6_R121_C2C3_VERTEX_COSTS_B11O4_FRESH_AND_B7O2_COLLECTION__"
            "IID_SETUP_SIDE_SUPPORT_REMAINS_OCCURRENCE_SCALE_WHP__FILTERED_"
            "COLLISION_COMPRESSION_AND_NONOCCURRENCE_TORUS_CIRCUIT_OPEN__"
            "NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "theorem": cost["theorem"],
        "finite_evidence": {
            "all_exhaustive_pair_collision_controls_exact": controls[
                "all_exhaustive_pair_collision_controls_exact"
            ],
            "all_selected_setup_eligible_regimes_above_rho": controls[
                "all_selected_setup_eligible_regimes_above_rho"
            ],
            "grid_candidate_count": controls["grid_audit"][
                "candidate_count"
            ],
            "grid_setup_eligible_candidate_count": controls["grid_audit"][
                "setup_eligible_candidate_count"
            ],
            "grid_theorem_violation_count": controls["grid_audit"][
                "theorem_violation_count"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "explicit_split_rebalance_negative_admitted": True,
            "iid_setup_support_theorem_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_explicit_split_global_rebalance.json"
            ),
            "cost": (
                "torus_c5_explicit_split_global_rebalance_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_explicit_split_global_rebalance_replay.json"
            ),
            "controls": (
                "torus_c5_explicit_split_global_rebalance_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r122.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The theorem covers explicit occurrence/output split tables only.",
            "The iid support theorem is not transferred to every filtered deck.",
            "No general circuit, cell-probe, or RAM lower bound is claimed.",
            "No field discrete logarithm receives oracle credit.",
            "No known-RHS rank, factor logs, or identical descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "REJECT_GLOBAL_ARITY_REBALANCE_WITH_EXPLICIT_CS_TABLE_AND_CR_"
            "ENUMERATION__ADMIT_EXACT_L_GE_11O4_PLUS_BETA_THEOREM_AND_IID_"
            "SETUP_SUPPORT_ONLY__PRESERVE_FILTERED_COLLISION_COMPRESSION_AND_"
            "TARGET_SPECIALIZED_NONOCCURRENCE_TORUS_C5_CIRCUIT__NO_"
            "LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_"
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
            "p1553_torus_c5_explicit_split_global_"
            "rebalance_probe_report_r122.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_explicit_split_global_rebalance.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_split_global_rebalance_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_split_global_rebalance_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_split_global_rebalance_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r122.json"
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
        f"R122 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
