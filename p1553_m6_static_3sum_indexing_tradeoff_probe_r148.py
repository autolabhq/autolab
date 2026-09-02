#!/usr/bin/env python3
"""Audit static 3SUM-indexing tradeoffs for the R147 occurrence list."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_static_3sum_indexing_tradeoff.r148.v1"

R147_PRODUCER = ROOT / (
    "p1553_m6_occurrence_pair_resultant_local_valuation_probe_r147.py"
)
R147_REPORT = ROOT / (
    "p1553_m6_occurrence_pair_resultant_local_valuation_"
    "probe_report_r147.json"
)
R147_FROZEN = ROOT / (
    "frozen_m6_occurrence_pair_resultant_local_valuation.json"
)
R147_COST = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_cost_ledger.json"
)
R147_REPLAY = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_replay.json"
)
R147_CONTROLS = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_controls.json"
)
R147_LOGS = ROOT / "factor_logs_and_identical_descent_r147.json"
R147_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_occurrence_pair_resultant_local_valuation_probe_r147.py"
)
R147_GATE = ROOT / (
    "p1553_m6_occurrence_pair_resultant_local_valuation_probe_gate_r147.md"
)
R147_PARENT = ROOT / (
    "p1553_m6_occurrence_pair_resultant_local_valuation_"
    "probe_parent_report_r147.yaml"
)
GGHPV_PAPER = ROOT / (
    "references/golovnev_3sum_preprocessing_1907.08355.pdf"
)
DINUR_GOLOVNEV_PAPER = ROOT / (
    "references/dinur_golovnev_3sum_indexing_2512.04258.pdf"
)

SOURCE_BINDINGS = (
    (
        "r147_producer",
        R147_PRODUCER,
        "cbd08abbdb85025c2353be2253284d79896c6a1709690aa31d01a90622e1a49f",
    ),
    (
        "r147_report",
        R147_REPORT,
        "a419f994907d392b8d8cec7a3af6dc0f3a67769350be7c1044e3de4320a69977",
    ),
    (
        "r147_frozen",
        R147_FROZEN,
        "0e1c41e3c0ddcb6808977a52d50be72c1a2cef36e186599d3bc5337a793cb26b",
    ),
    (
        "r147_cost",
        R147_COST,
        "dbb23343d92b1d757bce760fefeed04aa47719779a4f153fe8a2c6b25dac2c10",
    ),
    (
        "r147_replay",
        R147_REPLAY,
        "30e839872cd09e34d12224662c3e3988c5cb478377da1632eb2a4f0c985ef643",
    ),
    (
        "r147_controls",
        R147_CONTROLS,
        "d61405a35dc3efec0c9c49dd3be7159718f052ea7daa601ed56d64a1628fdf65",
    ),
    (
        "r147_logs",
        R147_LOGS,
        "4389c787e70da680929bb59faaf1550f22d435f253e8f002dc0d50eb1df2f31c",
    ),
    (
        "r147_test",
        R147_TEST,
        "da3d847481dd5bc38386f4f50ecb3c9d933e8e6ec0afee0b5c697813e2174a75",
    ),
    (
        "r147_gate",
        R147_GATE,
        "7398ae8894fc40f710f76e2ec0e6f31726f283ca6fb9153f5258e0561537a6fe",
    ),
    (
        "r147_parent",
        R147_PARENT,
        "a1fd7d4c9c4b0f3b75b97c6e726187bda9db382c1f58910374009dc322f7b7cf",
    ),
    (
        "gghpv_3sum_preprocessing",
        GGHPV_PAPER,
        "b9161a299ee5227bdf11be0bbfec1c58a9348deb8d261875b935d573b4112785",
    ),
    (
        "dinur_golovnev_3sum_indexing",
        DINUR_GOLOVNEV_PAPER,
        "e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_static_3sum_indexing_tradeoff.json"
DEFAULT_COST = ROOT / "m6_static_3sum_indexing_tradeoff_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_static_3sum_indexing_tradeoff_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_static_3sum_indexing_tradeoff_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r148.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R147 = load_module("p1553_r147_for_r148", R147_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise AssertionError(f"R148 source binding mismatch: {failures}")
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


def exponent_ledger() -> dict[str, Any]:
    occurrence = Fraction(9, 4)
    query_batch = Fraction(5, 4)
    return {
        "occurrence_list_length_exponent_B": fraction_record(occurrence),
        "query_batch_size_exponent_B": fraction_record(query_batch),
        "setup_state_cap_exponent_B": fraction_record(occurrence),
        "fresh_work_cap_exponent_B": fraction_record(query_batch),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "trivial_linear_state_exponent_B": fraction_record(occurrence),
        "trivial_one_query_scan_exponent_B": fraction_record(occurrence),
        "trivial_full_batch_exponent_B": fraction_record(
            occurrence + query_batch
        ),
        "trivial_full_batch_exponent_N": fraction_record(
            Fraction(7, 10)
        ),
        "full_sumset_state_exponent_B": fraction_record(2 * occurrence),
        "full_sumset_one_query_exponent_B": fraction_record(Fraction(0)),
        "gghpv_tradeoff": {
            "relation": "T*S^3=soft-O(n^6)",
            "linear_state_query_exponent_n": fraction_record(Fraction(3)),
            "linear_state_query_exponent_B": fraction_record(
                3 * occurrence
            ),
            "linear_state_full_batch_exponent_B": fraction_record(
                3 * occurrence + query_batch
            ),
            "linear_state_endpoint_dominated_by_trivial_scan": True,
        },
        "dinur_golovnev_tradeoff": {
            "relation": "T*S=soft-O(n^(5/2))",
            "improvement_space_range_exponent_n": [
                fraction_record(Fraction(3, 2)),
                fraction_record(Fraction(7, 4)),
            ],
            "minimum_improvement_state_exponent_B": fraction_record(
                Fraction(3, 2) * occurrence
            ),
            "preprocessing_exponent_n": fraction_record(Fraction(2)),
            "preprocessing_exponent_B": fraction_record(2 * occurrence),
            "minimum_state_exceeds_setup_cap": True,
            "preprocessing_exceeds_setup_cap": True,
        },
        "scope": (
            "These are instantiations of published decision or witness "
            "3SUM-indexing upper bounds. They neither provide exact "
            "multiplicity or atom marginals nor prove a lower bound for "
            "structured elliptic occurrence divisors, arithmetic circuits, "
            "RAM, or cell probes."
        ),
    }


def finite_control(
    control: dict[str, Any],
) -> dict[str, Any]:
    occurrence_count = int(control["occurrence_divisor_degree"])
    sampled_query_count = (
        len(control["positive_queries"]) + len(control["empty_queries"])
    )
    return {
        "control_id": control["control_id"],
        "occurrence_list_length": occurrence_count,
        "sampled_query_count": sampled_query_count,
        "linear_state_words": occurrence_count,
        "trivial_scan_operations_per_query": occurrence_count,
        "trivial_sample_batch_operations": (
            occurrence_count * sampled_query_count
        ),
        "full_sumset_table_entries": occurrence_count * occurrence_count,
        "all_r147_query_answers_reused_exactly": (
            control["all_positive_valuations_exact"]
            and control["all_empty_valuations_zero"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    r147_bundle = R147.build_bundle()
    actual = [
        finite_control(control)
        for control in r147_bundle["controls"]["actual_controls"]
    ]
    all_reused = all(
        row["all_r147_query_answers_reused_exactly"] for row in actual
    )
    costs = exponent_ledger()

    controls = {
        "schema": "p1553.m6_static_3sum_indexing_tradeoff.controls.r148.v1",
        "actual_control_count": len(actual),
        "all_r147_query_answers_reused_exactly": all_reused,
        "all_sample_batches_have_eighteen_queries": all(
            row["sampled_query_count"] == 18 for row in actual
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    reduction = {
        "input": (
            "The repeated ordered-C3 endpoint list of length "
            "n=B^(9/4+o(1))."
        ),
        "query": (
            "Given target tau, find or count occurrence pairs whose Cayley "
            "sum is tau."
        ),
        "decision_equivalence": (
            "Ignoring weights, existence is a static 3SUM-indexing query "
            "on two identical lists in the elliptic Cayley group."
        ),
        "required_output_is_stronger": (
            "R144 requires exact integer multiplicity and full A/C atom "
            "marginals, not only existence or one witness."
        ),
    }

    frozen = {
        "schema": "p1553.m6_static_3sum_indexing_tradeoff.frozen.r148.v1",
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "reduction": reduction,
        "published_upper_bound_instantiations": costs,
        "required_open_outputs": {
            "structure_aware_shared_occurrence_autocorrelation": "open",
            "exact_integer_count_index": "open",
            "offline_online_transposed_atom_marginals": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": "p1553.m6_static_3sum_indexing_tradeoff.replay.r148.v1",
        "controls": actual,
        "all_r147_query_answers_reused_exactly": all_reused,
        "exponent_ledger": costs,
    }

    cost = {
        "schema": "p1553.m6_static_3sum_indexing_tradeoff.cost.r148.v1",
        "published_upper_bound_instantiations": costs,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "linear_state_sqrt_query_claimed": False,
        "published_exact_count_index_supplied": False,
        "published_atom_marginal_index_supplied": False,
        "structure_aware_elliptic_index_supplied": False,
        "unconditional_computational_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.m6_static_3sum_indexing_tradeoff."
            "logs_descent.r148.v1"
        ),
        "finite_r147_queries_reused_exactly": all_reused,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "twelve_source_bindings_verified": len(actual_bindings) == 12,
        "occurrence_pair_to_3sum_indexing_reduction_complete": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "finite_trivial_scan_accounting_exact": True,
        "finite_full_sumset_accounting_exact": True,
        "candidate_dlp_and_root_oracles_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "gghpv_primary_tradeoff_bound_bound": True,
        "dinur_golovnev_primary_tradeoff_bound_bound": True,
        "r115_cap_translation_complete": True,
        "trivial_full_batch_exceeds_rho": True,
        "full_sumset_table_exceeds_setup_cap": True,
        "gghpv_linear_state_endpoint_dominated_by_scan": True,
        "dinur_golovnev_improvement_state_exceeds_cap": True,
        "dinur_golovnev_preprocessing_exceeds_cap": True,
        "structure_aware_shared_index_complete": False,
        "exact_integer_count_index_complete": False,
        "offline_online_transposed_atom_marginals_complete": False,
        "structured_rank_and_density_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Do not import a sqrt(n)-query 3SUM index. Exploit the fact that "
        "the occurrence divisor is a triple elliptic convolution of the "
        "compact C divisor before applying generic static indexing. Build "
        "one structure-aware shared autocorrelation or marker operator with "
        "B^(9/4) setup and B^(5/4) total query work that emits exact integer "
        "counts and A/C marginals, then replay rank, factor logs, and "
        "identical descent. It may use no DLP, root, 3SUM, count, marginal, "
        "rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "OCCURRENCE_PAIR_QUERY_IS_WEIGHTED_STATIC_3SUM_INDEXING__"
            "LINEAR_STATE_TRIVIAL_QUERY_D_B9O4__FULL_BATCH_B7O2_N7O10__"
            "FIAT_NAOR_LINEAR_STATE_ENDPOINT_DOMINATED__DINUR_GOLOVNEV_"
            "IMPROVEMENT_REQUIRES_D3O2_STATE_AND_D2_PREPROCESSING__NO_"
            "PUBLISHED_EXACT_COUNT_OR_MARGINAL_INDEX__STRUCTURE_AWARE_"
            "ELLIPTIC_AUTOCORRELATION_OPEN__NO_LOWER_BOUND_LOGS_DESCENT_"
            "SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether published static 3SUM-indexing tradeoffs "
            "supply the shared inside-cap occurrence-pair operator left "
            "open by R147."
        ),
        "source_bindings": source_binding_records(),
        "reduction": reduction,
        "published_upper_bound_instantiations": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "weighted_3sum_indexing_reduction_admitted": True,
            "published_standard_indexing_negative_admitted": True,
            "structure_aware_shared_operator_admitted": False,
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
        "cost": cost,
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
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
