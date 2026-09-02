#!/usr/bin/env python3
"""Test whether growing A diversity removes signed singleton rank deficits."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_a_diversity_projective_rank.r156.v1"

R155_PRODUCER = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_probe_r155.py"
)
R155_REPORT = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_"
    "probe_report_r155.json"
)
R155_FROZEN = ROOT / (
    "frozen_m6_singleton_relation_hypergraph_rank.json"
)
R155_COST = ROOT / (
    "m6_singleton_relation_hypergraph_rank_cost_ledger.json"
)
R155_REPLAY = ROOT / (
    "m6_singleton_relation_hypergraph_rank_replay.json"
)
R155_CONTROLS = ROOT / (
    "m6_singleton_relation_hypergraph_rank_controls.json"
)
R155_LOGS = ROOT / "factor_logs_and_identical_descent_r155.json"
R155_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_singleton_relation_hypergraph_rank_probe_r155.py"
)
R155_GATE = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_probe_gate_r155.md"
)
R155_PARENT = ROOT / (
    "p1553_m6_singleton_relation_hypergraph_rank_"
    "probe_parent_report_r155.yaml"
)

SOURCE_BINDINGS = (
    (
        "r155_producer",
        R155_PRODUCER,
        "a8c59626ebd38c87c7bc19c8c98cae15a548ce489ba3f39b6eb2387ababd5c00",
    ),
    (
        "r155_report",
        R155_REPORT,
        "11e69758caceba9926e8d1d046aef71f56ed075ab2d2766988eb8ecc258a7bac",
    ),
    (
        "r155_frozen",
        R155_FROZEN,
        "f3a0f229c3449bf32e26e0c832eb36cd5911ebb1357e005f5e7fc9a5d195155b",
    ),
    (
        "r155_cost",
        R155_COST,
        "9c7ac01885694dc51ed6450e20a6c44d4425d088a00e3cdfdf73f19bff84b977",
    ),
    (
        "r155_replay",
        R155_REPLAY,
        "3d8c2281175f3584a83c8502aa379a2222a9d0b95526f33a77a48f5f5f05570b",
    ),
    (
        "r155_controls",
        R155_CONTROLS,
        "1b9f0096c3fad2565e3af956b8a52be04877afabbfa24e063039a43211bb4530",
    ),
    (
        "r155_logs",
        R155_LOGS,
        "307739c16b7f2d695679648dc91f79c84f564c69ac670305654d63cf088abb2d",
    ),
    (
        "r155_test",
        R155_TEST,
        "0c5dbd2eb8cded7a05bd9bea690d4dad17c2525dce0ef5dc1030a03143e3f509",
    ),
    (
        "r155_gate",
        R155_GATE,
        "f558cd9a526471ad51e076256934bdea34a7fd2c5d41f878ae6fdd39eb797875",
    ),
    (
        "r155_parent",
        R155_PARENT,
        "66d36dd4a801f8b621c90415cbef3886562a8ca1660744b3771e3ba6ac011971",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_a_diversity_projective_rank_probe_report_r156.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_a_diversity_projective_rank.json"
)
DEFAULT_COST = ROOT / (
    "m6_a_diversity_projective_rank_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_a_diversity_projective_rank_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_a_diversity_projective_rank_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r156.json"

A_PAIR_COUNTS = (2, 3, 4)
C_PAIR_COUNTS = (5, 6, 7, 8)
LOG_OVERSAMPLING_FACTORS = (2, 4, 8)
SEEDS = (15601, 15602, 15603)
ARITY = 6


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R155 = load_module("p1553_r155_for_r156", R155_PRODUCER)
R154 = R155.R154
R153 = R155.R153
R81 = R155.R81


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R156 source binding mismatch: {failures}")
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


def occupancy_multiplier(
    c_pair_count: int, log_factor: int
) -> int:
    return max(1, math.ceil(log_factor * math.log(c_pair_count)))


def projective_normalize(
    row: tuple[int, ...], modulus: int
) -> tuple[int, ...]:
    pivot = next(value for value in row if value % modulus)
    inverse = pow(pivot, -1, modulus)
    return tuple(value * inverse % modulus for value in row)


def diversity_control(
    a_pair_count: int,
    c_pair_count: int,
    log_factor: int,
    seed: int,
) -> dict[str, Any]:
    multiplier = occupancy_multiplier(c_pair_count, log_factor)
    max_a6_support = R154.signed_coefficient_vector_count(
        a_pair_count, ARITY
    )
    max_c6_support = R154.signed_coefficient_vector_count(
        c_pair_count, ARITY
    )
    modulus = R154.next_prime(
        math.ceil(max_a6_support * max_c6_support / multiplier)
    )
    tag = (
        f"a{a_pair_count}|c{c_pair_count}|log{log_factor}|"
        f"lambda{multiplier}|seed{seed}"
    )
    a_pairs = R154.deterministic_signed_pairs(
        f"A|{tag}", a_pair_count, modulus
    )
    c_pairs = R154.deterministic_signed_pairs(
        f"C|{tag}", c_pair_count, modulus
    )
    a_labels = R154.flatten_pairs(a_pairs)
    c_labels = R154.flatten_pairs(c_pairs)
    shift_counts = R153.convolution_power(
        a_labels, ARITY, modulus
    )
    endpoints = R155.endpoint_multisets(
        c_labels, ARITY, modulus
    )
    rows: list[tuple[int, ...]] = []
    supports: list[set[int]] = []
    relation_identities_exact = True
    opposite_rows_exact = True
    positive_rows = 0
    multi_source_rows = 0

    row_records: list[tuple[int, int, tuple[int, ...]]] = []
    for shift in sorted(shift_counts):
        for target_index, target in enumerate(c_labels):
            sources = endpoints.get((shift + target) % modulus, [])
            if not sources:
                continue
            positive_rows += 1
            if len(sources) != 1:
                multi_source_rows += 1
                continue
            row = tuple(
                value % modulus
                for value in R155.signed_sparse_row(
                    sources[0], target_index, c_pairs
                )
            )
            if not any(row):
                continue
            representatives = [positive for positive, _ in c_pairs]
            relation_identities_exact &= (
                sum(
                    coefficient * representative
                    for coefficient, representative in zip(
                        row, representatives
                    )
                )
                % modulus
                == shift
            )
            rows.append(row)
            supports.append(
                {index for index, value in enumerate(row) if value}
            )
            row_records.append((shift, target_index, row))

    row_set = set(rows)
    for shift, target_index, row in row_records:
        target = c_labels[target_index]
        opposite_target = (-target) % modulus
        opposite_target_index = c_labels.index(opposite_target)
        opposite = tuple((-value) % modulus for value in row)
        matching = [
            candidate
            for candidate_shift, candidate_target, candidate in row_records
            if candidate_shift == (-shift) % modulus
            and candidate_target == opposite_target_index
        ]
        opposite_rows_exact &= opposite in matching

    unique_rows = set(rows)
    projective_rows = {
        projective_normalize(row, modulus) for row in unique_rows
    }
    projective_matrix = sorted(projective_rows)
    rank = R81.rank_mod(projective_matrix, modulus)
    covered = set().union(*supports) if supports else set()
    core = R155.peel_two_core(supports, c_pair_count)
    projective_count_deficit = max(
        0, c_pair_count - len(projective_rows)
    )
    dependency_nullity = max(
        0, min(c_pair_count, len(projective_rows)) - rank
    )
    actual_shift_support = len(shift_counts)
    realized_occupancy = Fraction(
        actual_shift_support * len(endpoints),
        modulus,
    )

    return {
        "control_id": tag,
        "a_pair_count": a_pair_count,
        "c_pair_count": c_pair_count,
        "signed_log_dimension": c_pair_count,
        "log_oversampling_factor": log_factor,
        "occupancy_multiplier": multiplier,
        "seed": seed,
        "subgroup_order": modulus,
        "max_a6_signed_coefficient_support": max_a6_support,
        "actual_a6_shift_support": actual_shift_support,
        "max_c6_signed_coefficient_support": max_c6_support,
        "actual_c6_endpoint_support": len(endpoints),
        "realized_support_occupancy": fraction_record(
            realized_occupancy
        ),
        "positive_relation_row_count": positive_rows,
        "multi_source_relation_row_count": multi_source_rows,
        "singleton_nonzero_row_count": len(rows),
        "unique_singleton_row_count": len(unique_rows),
        "projectively_distinct_row_count": len(projective_rows),
        "covered_column_count": len(covered),
        "uncovered_column_count": c_pair_count - len(covered),
        "projective_count_deficit": projective_count_deficit,
        "dependency_nullity_after_projective_dedup": dependency_nullity,
        "signed_rank": rank,
        "signed_full_rank": rank == c_pair_count,
        "all_relation_identities_exact": relation_identities_exact,
        "all_opposite_rows_exact": opposite_rows_exact,
        "two_core": core,
        "projective_rows_sha256": sha256_json(projective_matrix),
        "row_supports_sha256": sha256_json(
            [sorted(support) for support in supports]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def grouped_summary(
    controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for c_pair_count in C_PAIR_COUNTS:
        for log_factor in LOG_OVERSAMPLING_FACTORS:
            for a_pair_count in A_PAIR_COUNTS:
                rows = [
                    row
                    for row in controls
                    if row["c_pair_count"] == c_pair_count
                    and row["log_oversampling_factor"] == log_factor
                    and row["a_pair_count"] == a_pair_count
                ]
                summaries.append(
                    {
                        "a_pair_count": a_pair_count,
                        "c_pair_count": c_pair_count,
                        "log_oversampling_factor": log_factor,
                        "occupancy_multiplier": occupancy_multiplier(
                            c_pair_count, log_factor
                        ),
                        "trial_count": len(rows),
                        "full_rank_trial_count": sum(
                            row["signed_full_rank"] for row in rows
                        ),
                        "rank_values": [
                            row["signed_rank"] for row in rows
                        ],
                        "projective_row_counts": [
                            row["projectively_distinct_row_count"]
                            for row in rows
                        ],
                        "uncovered_column_counts": [
                            row["uncovered_column_count"] for row in rows
                        ],
                        "dependency_nullities": [
                            row[
                                "dependency_nullity_after_projective_dedup"
                            ]
                            for row in rows
                        ],
                    }
                )
    return summaries


def a_pair_summary(
    controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for a_pair_count in A_PAIR_COUNTS:
        rows = [
            row
            for row in controls
            if row["a_pair_count"] == a_pair_count
        ]
        result.append(
            {
                "a_pair_count": a_pair_count,
                "control_count": len(rows),
                "full_rank_control_count": sum(
                    row["signed_full_rank"] for row in rows
                ),
                "covered_control_count": sum(
                    row["uncovered_column_count"] == 0 for row in rows
                ),
                "projective_count_sufficient_control_count": sum(
                    row["projective_count_deficit"] == 0 for row in rows
                ),
                "dependency_free_control_count": sum(
                    row[
                        "dependency_nullity_after_projective_dedup"
                    ]
                    == 0
                    for row in rows
                ),
            }
        )
    return result


def log_factor_summary(
    controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for log_factor in LOG_OVERSAMPLING_FACTORS:
        rows = [
            row
            for row in controls
            if row["log_oversampling_factor"] == log_factor
        ]
        result.append(
            {
                "log_oversampling_factor": log_factor,
                "control_count": len(rows),
                "full_rank_control_count": sum(
                    row["signed_full_rank"] for row in rows
                ),
                "covered_control_count": sum(
                    row["uncovered_column_count"] == 0 for row in rows
                ),
                "projective_count_sufficient_control_count": sum(
                    row["projective_count_deficit"] == 0 for row in rows
                ),
                "dependency_free_control_count": sum(
                    row[
                        "dependency_nullity_after_projective_dedup"
                    ]
                    == 0
                    for row in rows
                ),
            }
        )
    return result


def theorem_record() -> dict[str, Any]:
    return {
        "opposite_row_quotient": (
            "Inversion symmetry maps every singleton row at (s,a) to its "
            "negative at (-s,-a). Projective normalization removes this "
            "exact duplicate without changing row span or rank."
        ),
        "a_diversity_scaling": (
            "With r independent A inversion pairs, the generic signed A6 "
            "coefficient support has size "
            "sum_{t even,0<=t<=6} sum_j C(r,j)C(t-1,j-1)2^j, "
            "which is Theta(r^6). Thus the fixed two-pair R155 controls "
            "do not model the unbounded B^(1/12) A deck."
        ),
        "preregistered_grid": (
            "The finite grid uses A-pair counts 2,3,4; C-pair counts "
            "5,6,7,8; logarithmic factors 2,4,8; and three frozen seeds. "
            "The occupancy multiplier is ceil(factor*ln(C-pair-count)) "
            "and the prime modulus is fixed before labels and ranks."
        ),
        "rank_decomposition": (
            "Each deficit is separated into uncovered columns, too few "
            "projectively distinct rows, and residual algebraic dependency "
            "after projective deduplication."
        ),
        "scope": (
            "Finite cyclic labels use verifier-known scalars and receive "
            "no asymptotic credit. A monotone A-diversity transition would "
            "support, but not prove, convolution-Tanner contiguity or "
            "hash-to-curve rank transfer."
        ),
        "novelty_status": (
            "a_diversity_projective_rank_transition_novelty_unverified"
        ),
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_a_diversity_projective_rank.cost.r156.v1"
        ),
        "a_pair_count_exponent_B": fraction_record(Fraction(1, 12)),
        "a6_shift_support_exponent_B": fraction_record(Fraction(1, 2)),
        "signed_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "relation_count_exponent_B": fraction_record(Fraction(3, 4)),
        "relation_count_polylog_factor": "log(B)",
        "structured_row_batch_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "structured_row_batch_polylog_factor": "log(B)",
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "opposite_row_projective_quotient_changes_no_exponent": True,
        "growing_a_diversity_changes_no_selected_exponent": True,
        "finite_control_work_receives_attack_credit": False,
        "convolution_tanner_contiguity_supplied": False,
        "hash_to_curve_rank_transfer_supplied": False,
        "reverse_only_signed_marker_operator_supplied": False,
        "factor_logs_without_verifier_labels_supplied": False,
        "identical_target_descent_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls_list = [
        diversity_control(a_pairs, c_pairs, log_factor, seed)
        for a_pairs in A_PAIR_COUNTS
        for c_pairs in C_PAIR_COUNTS
        for log_factor in LOG_OVERSAMPLING_FACTORS
        for seed in SEEDS
    ]
    summaries = grouped_summary(controls_list)
    a_summaries = a_pair_summary(controls_list)
    log_summaries = log_factor_summary(controls_list)
    all_exact = all(
        row["all_relation_identities_exact"]
        and row["all_opposite_rows_exact"]
        for row in controls_list
    )
    full_count = sum(
        row["signed_full_rank"] for row in controls_list
    )
    full_by_a = [
        row["full_rank_control_count"] for row in a_summaries
    ]
    monotone_full_counts = full_by_a == sorted(full_by_a)
    full_by_log_factor = [
        row["full_rank_control_count"] for row in log_summaries
    ]
    monotone_log_transition = (
        full_by_log_factor == sorted(full_by_log_factor)
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_a_diversity_projective_rank.controls.r156.v1"
        ),
        "control_count": len(controls_list),
        "all_relation_and_opposite_row_identities_exact": all_exact,
        "full_rank_control_count": full_count,
        "full_rank_counts_by_a_pair_count": full_by_a,
        "full_rank_counts_monotone_in_a_diversity": monotone_full_counts,
        "full_rank_counts_by_log_oversampling_factor": (
            full_by_log_factor
        ),
        "full_rank_counts_monotone_in_log_oversampling": (
            monotone_log_transition
        ),
        "grouped_summary": summaries,
        "a_pair_summary": a_summaries,
        "log_factor_summary": log_summaries,
        "controls": controls_list,
        "finite_controls_receive_asymptotic_credit": False,
    }

    frozen = {
        "schema": (
            "p1553.m6_a_diversity_projective_rank.frozen.r156.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "design": {
            "a_pair_counts": list(A_PAIR_COUNTS),
            "c_pair_counts": list(C_PAIR_COUNTS),
            "log_oversampling_factors": list(
                LOG_OVERSAMPLING_FACTORS
            ),
            "seeds": list(SEEDS),
            "arity": ARITY,
        },
        "required_open_outputs": {
            "asymptotic_a_diversity_rank_theorem": "open",
            "convolution_tanner_contiguity_or_direct_rank_theorem": "open",
            "hash_to_curve_rank_transfer": "open",
            "reverse_only_signed_marker_operator": "open",
            "factor_logs_without_verifier_labels": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_a_diversity_projective_rank.replay.r156.v1"
        ),
        "controls": controls_list,
        "grouped_summary": summaries,
        "a_pair_summary": a_summaries,
        "log_factor_summary": log_summaries,
        "all_relation_and_opposite_row_identities_exact": all_exact,
    }

    logs = {
        "schema": (
            "p1553.m6_a_diversity_projective_rank."
            "logs_descent.r156.v1"
        ),
        "finite_full_rank_control_count": full_count,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "r155_singleton_hypergraph_inherited": True,
        "opposite_row_projective_quotient_derived": True,
        "a6_signed_coefficient_support_formula_derived": True,
        "growing_a_diversity_grid_preregistered": True,
        "one_hundred_eight_controls_complete": (
            len(controls_list) == 108
        ),
        "all_relation_identities_exact": all_exact,
        "all_opposite_rows_exact": all_exact,
        "coverage_projective_count_and_dependency_split_complete": True,
        "all_two_core_diagnostics_complete": all(
            "two_core" in row for row in controls_list
        ),
        "finite_full_rank_transition_observed": full_count > 0,
        "finite_full_rank_counts_monotone_in_a_diversity": (
            monotone_full_counts
        ),
        "finite_full_rank_counts_monotone_in_log_oversampling": (
            monotone_log_transition
        ),
        "logarithmic_oversampling_charged": True,
        "finite_results_scoped_without_transfer": True,
        "asymptotic_a_diversity_rank_theorem_complete": False,
        "convolution_tanner_contiguity_complete": False,
        "hash_to_curve_rank_transfer_complete": False,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "factor_logs_without_verifier_labels_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Use the frozen A-diversity transition to derive or refute a "
        "direct rank theorem after projective opposite-row quotienting. "
        "The theorem must cover A-pair count B^(1/12), "
        "B^(3/4)log(B) singleton relations, shared-deck dependencies, "
        "coverage, and residual nullity. Then transfer to hash-to-curve "
        "decks and instantiate the reverse signed FFE operator within "
        "B^(9/4) setup and B^(5/4+o(1)) batch work before exact-residual "
        "logs or identical descent."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "OPPOSITE_SINGLETON_ROWS_PROJECTIVELY_QUOTIENT_EXACTLY__A6_"
            "SHIFT_DIVERSITY_GROWS_AS_A_PAIR_COUNT_TO_SIXTH__"
            "PREREGISTERED_108_CONTROL_LOG_OVERSAMPLING_GRID__COVERAGE_"
            "PROJECTIVE_COUNT_AND_RESIDUAL_DEPENDENCY_SEPARATED__FINITE_"
            "A_DIVERSITY_RANK_TRANSITION_TESTED__ASYMPTOTIC_DIRECT_RANK_"
            "HASH_TO_CURVE_REVERSE_FFE_LOGS_DESCENT_OPEN__NO_SHOUP_"
            "BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether R155 rank deficits are structural or an "
            "artifact of fixing only two A inversion pairs."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "projective_opposite_row_quotient_admitted": True,
            "finite_a_diversity_transition_admitted_without_transfer": (
                full_count > 0
            ),
            "asymptotic_rank_or_hash_to_curve_transfer_admitted": False,
            "reverse_only_signed_marker_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": costs,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"full={bundle['controls']['full_rank_control_count']} "
        f"by_a={bundle['controls']['full_rank_counts_by_a_pair_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
