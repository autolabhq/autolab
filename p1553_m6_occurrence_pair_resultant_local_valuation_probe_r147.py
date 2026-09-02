#!/usr/bin/env python3
"""Audit C6 counts as local valuations of an occurrence pair-resultant."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_occurrence_pair_resultant_local_valuation.r147.v1"

R146_PRODUCER = ROOT / (
    "p1553_m6_aggregate_marginal_singleton_source_equivalence_"
    "probe_r146.py"
)
R146_REPORT = ROOT / (
    "p1553_m6_aggregate_marginal_singleton_source_equivalence_"
    "probe_report_r146.json"
)
R146_FROZEN = ROOT / (
    "frozen_m6_aggregate_marginal_singleton_source_equivalence.json"
)
R146_COST = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_cost_ledger.json"
)
R146_REPLAY = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_replay.json"
)
R146_CONTROLS = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_controls.json"
)
R146_LOGS = ROOT / "factor_logs_and_identical_descent_r146.json"
R146_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_aggregate_marginal_singleton_"
    "source_equivalence_probe_r146.py"
)
R146_GATE = ROOT / (
    "p1553_m6_aggregate_marginal_singleton_"
    "source_equivalence_probe_gate_r146.md"
)
R146_PARENT = ROOT / (
    "p1553_m6_aggregate_marginal_singleton_"
    "source_equivalence_probe_parent_report_r146.yaml"
)
TRUNCATED_RESULTANT_PAPER = ROOT / (
    "references/moroz_schost_truncated_resultant_1609.04259.pdf"
)

SOURCE_BINDINGS = (
    (
        "r146_producer",
        R146_PRODUCER,
        "088ece9b3359cd80b510ccb6bd2aba7fbc2ff037978e1e54a9d8159d902034a3",
    ),
    (
        "r146_report",
        R146_REPORT,
        "6b7c1eefcb4d062bf5eaee84e88ab2f5e1fd65ba060f16c1ce4adaf0dbfb64f9",
    ),
    (
        "r146_frozen",
        R146_FROZEN,
        "0570199c52a34c19819852c003870f6962148b1f8ff80e13326a3553665b527b",
    ),
    (
        "r146_cost",
        R146_COST,
        "8f40955365b4fc074ed83a98dd168f8f61b747a968d7e71682e3fdaeb9e90711",
    ),
    (
        "r146_replay",
        R146_REPLAY,
        "b33844c11cb0b60562292be41d9f7e6738bc290650c64d17cd6f16dc96cb2d24",
    ),
    (
        "r146_controls",
        R146_CONTROLS,
        "9a2a729ce1169ca979b52047452149e551cffc2e68ea875ef819def7c322412d",
    ),
    (
        "r146_logs",
        R146_LOGS,
        "cc960f475d3029958cd71bdc22b1c7c4411fe30fca31a08580eb1d6c3f8d9037",
    ),
    (
        "r146_test",
        R146_TEST,
        "6b23ddbffe6a3ef455c90b15628b117a4ea74aa2c9f51c390846d7bfbbd11af2",
    ),
    (
        "r146_gate",
        R146_GATE,
        "8b0dc8312273770d5f7f44a26b89a1ce06873b20994d7445fbf8b3d55ba02216",
    ),
    (
        "r146_parent",
        R146_PARENT,
        "8c6b9f0505a3b26b3443a40f1cb38cc4e0a36f58515d5db3c510b3004e6232bb",
    ),
    (
        "moroz_schost_truncated_resultant",
        TRUNCATED_RESULTANT_PAPER,
        "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_occurrence_pair_resultant_local_valuation_"
    "probe_report_r147.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_occurrence_pair_resultant_local_valuation.json"
)
DEFAULT_COST = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_occurrence_pair_resultant_local_valuation_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r147.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R146 = load_module("p1553_r146_for_r147", R146_PRODUCER)
R145 = R146.R145
R141 = R145.R141
R82 = R146.R82


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
        raise AssertionError(f"R147 source binding mismatch: {failures}")
    return actual


def root_multiplicity(
    polynomial: list[int],
    root: int,
    prime: int,
) -> int:
    divisor = [(-root) % prime, 1]
    multiplicity = 0
    remainder = polynomial
    while remainder and R145.poly_evaluate(remainder, root, prime) == 0:
        quotient, residual = R145.poly_divmod(
            remainder,
            divisor,
            prime,
        )
        if residual:
            break
        multiplicity += 1
        remainder = quotient
    return multiplicity


def occurrence_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck = R141.R121.pairing_deck(curve, offset)
    c3_occurrences: collections.Counter[Any] = collections.Counter(
        field.product(source)
        for source in itertools.product(deck, repeat=3)
    )
    minus_one = field.neg(field.one)
    if minus_one in c3_occurrences:
        raise AssertionError("C3 occurrence support reaches omitted chart")
    parameter_weights = {
        R141.R121.torus_parameter(value, field): count
        for value, count in c3_occurrences.items()
    }
    repeated_parameters = [
        parameter
        for parameter, weight in sorted(parameter_weights.items())
        for _ in range(weight)
    ]
    occurrence_polynomial = R145.root_polynomial(
        repeated_parameters,
        field.p,
    )
    multiplicities_exact = all(
        root_multiplicity(occurrence_polynomial, parameter, field.p)
        == weight
        for parameter, weight in parameter_weights.items()
    )

    c6_occurrences: collections.Counter[Any] = collections.Counter()
    for left, left_count in c3_occurrences.items():
        for right, right_count in c3_occurrences.items():
            c6_occurrences[field.mul(left, right)] += (
                left_count * right_count
            )
    positive_parameters = sorted(
        R141.R121.torus_parameter(value, field)
        for value in c6_occurrences
        if value != minus_one
    )
    positive_parameters = positive_parameters[: min(12, len(positive_parameters))]
    positive_values = {
        R141.R121.torus_parameter(value, field): value
        for value in c6_occurrences
        if value != minus_one
    }
    positive_queries = []
    for target in positive_parameters:
        valuation = 0
        for parameter, weight in parameter_weights.items():
            partner = R145.mobius_parameter(
                parameter,
                target,
                field.nonsquare,
                field.p,
            )
            if partner in parameter_weights:
                valuation += weight * parameter_weights[partner]
        direct_count = c6_occurrences[positive_values[target]]
        positive_queries.append(
            {
                "target_parameter": target,
                "pair_resultant_local_valuation": valuation,
                "direct_ordered_c6_count": direct_count,
                "valuation_equals_ordered_c6_count": (
                    valuation == direct_count
                ),
                "required_truncation_order_for_first_nonzero_term": (
                    valuation + 1
                ),
            }
        )

    positive_set = set(
        R141.R121.torus_parameter(value, field)
        for value in c6_occurrences
        if value != minus_one
    )
    empty_parameters = []
    candidate = 0
    while len(empty_parameters) < 6:
        if candidate not in positive_set:
            empty_parameters.append(candidate)
        candidate += 1
    empty_queries = []
    for target in empty_parameters:
        valuation = 0
        for parameter, weight in parameter_weights.items():
            partner = R145.mobius_parameter(
                parameter,
                target,
                field.nonsquare,
                field.p,
            )
            if partner in parameter_weights:
                valuation += weight * parameter_weights[partner]
        empty_queries.append(
            {
                "target_parameter": target,
                "pair_resultant_local_valuation": valuation,
                "direct_ordered_c6_count": 0,
                "valuation_equals_ordered_c6_count": valuation == 0,
                "required_truncation_order_for_first_nonzero_term": 1,
            }
        )

    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "deck_size": len(deck),
        "ordered_c3_occurrence_count": len(deck) ** 3,
        "occurrence_divisor_degree": len(repeated_parameters),
        "squarefree_c3_support_degree": len(parameter_weights),
        "occurrence_polynomial_degree": len(occurrence_polynomial) - 1,
        "all_occurrence_root_multiplicities_exact": multiplicities_exact,
        "implicit_pair_resultant_degree": len(repeated_parameters) ** 2,
        "positive_queries": positive_queries,
        "empty_queries": empty_queries,
        "all_positive_valuations_exact": all(
            row["valuation_equals_ordered_c6_count"]
            for row in positive_queries
        ),
        "all_empty_valuations_zero": all(
            row["valuation_equals_ordered_c6_count"]
            for row in empty_queries
        ),
        "maximum_sample_positive_valuation": max(
            row["pair_resultant_local_valuation"]
            for row in positive_queries
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def standard_truncated_resultant_cost() -> dict[str, Any]:
    return {
        "occurrence_divisor_degree_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "full_pair_resultant_degree_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "bounded_local_truncation_order_exponent_B": fraction_record(
            Fraction(0)
        ),
        "one_local_valuation_exponent_B": fraction_record(Fraction(9, 4)),
        "a6_targets_per_known_target_exponent_B": fraction_record(
            Fraction(1, 2)
        ),
        "one_known_target_a6_batch_exponent_B": fraction_record(
            Fraction(11, 4)
        ),
        "known_target_row_count_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "full_relation_query_batch_size_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "componentwise_full_relation_batch_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "componentwise_full_relation_batch_exponent_N": fraction_record(
            Fraction(7, 10)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "fresh_work_cap_exponent_B": fraction_record(Fraction(5, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "paper_bound": (
            "Moroz-Schost computes a degree-d bivariate resultant "
            "truncated to order k in soft-O(d*k) base-field operations, "
            "including singular expansion points under its characteristic "
            "hypothesis."
        ),
        "paper_characteristic_condition": (
            "characteristic zero or at least the truncation order k"
        ),
        "standard_occurrence_divisor_fits_setup": True,
        "standard_one_local_valuation_inside_fresh_cap": False,
        "standard_one_known_target_batch_inside_rho": False,
        "standard_full_relation_batch_inside_rho": False,
        "scope": (
            "This charges componentwise or direct-product applications of "
            "the bound truncated-resultant algorithm to the occurrence "
            "divisor. It is not a lower bound for shared transposed "
            "multi-target circuits, modular-composition data structures, "
            "RAM, or cell probes."
        ),
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        occurrence_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    all_multiplicities = all(
        row["all_occurrence_root_multiplicities_exact"] for row in actual
    )
    all_positive = all(
        row["all_positive_valuations_exact"] for row in actual
    )
    all_empty = all(row["all_empty_valuations_zero"] for row in actual)
    cost_boundary = standard_truncated_resultant_cost()

    controls = {
        "schema": (
            "p1553.m6_occurrence_pair_resultant_local_valuation."
            "controls.r147.v1"
        ),
        "actual_control_count": len(actual),
        "all_occurrence_root_multiplicities_exact": all_multiplicities,
        "all_positive_pair_resultant_valuations_exact": all_positive,
        "all_empty_pair_resultant_valuations_zero": all_empty,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_occurrence_pair_resultant_local_valuation."
            "frozen.r147.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "occurrence_divisor": {
            "definition": (
                "P_occ(X)=product over ordered C3 occurrences "
                "(X-cayley_parameter(sum occurrence)); repeated endpoints "
                "remain repeated roots."
            ),
            "degree": "|C|^3=B^(9/4+o(1))",
        },
        "pair_resultant": {
            "definition": (
                "R(T)=product over ordered C3 occurrence pairs "
                "(T-(x oplus y)), up to nonzero Cayley chart units."
            ),
            "degree": "|C|^6=B^(9/2+o(1))",
            "local_count_identity": (
                "ord_(T=tau) R(T) equals the ordered C6 fiber count "
                "when every local Cayley factor is simple in T."
            ),
            "root_extraction_required": False,
        },
        "standard_truncated_resultant_cost": cost_boundary,
        "required_open_outputs": {
            "shared_sublinear_multi_target_valuation_index": "open",
            "offline_online_transposed_atom_marginals": "open",
            "generic_integer_lift": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_occurrence_pair_resultant_local_valuation."
            "replay.r147.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "occurrence_divisor_degree": row[
                    "occurrence_divisor_degree"
                ],
                "implicit_pair_resultant_degree": row[
                    "implicit_pair_resultant_degree"
                ],
                "positive_queries": row["positive_queries"],
                "empty_queries": row["empty_queries"],
            }
            for row in actual
        ],
        "all_positive_pair_resultant_valuations_exact": all_positive,
        "all_empty_pair_resultant_valuations_zero": all_empty,
    }

    cost = {
        "schema": (
            "p1553.m6_occurrence_pair_resultant_local_valuation."
            "cost.r147.v1"
        ),
        "standard_truncated_resultant_cost": cost_boundary,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "finite_pair_enumeration_charged_to_candidate": False,
        "full_pair_resultant_materialized": False,
        "inside_cap_multi_target_valuation_index_supplied": False,
        "offline_online_transposed_marginal_index_supplied": False,
        "unconditional_computational_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.m6_occurrence_pair_resultant_local_valuation."
            "logs_descent.r147.v1"
        ),
        "finite_occurrence_valuations_exact": all_positive and all_empty,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "eleven_source_bindings_verified": len(actual_bindings) == 11,
        "r146_source_equivalence_boundary_inherited": True,
        "occurrence_divisor_definition_complete": True,
        "pair_resultant_local_valuation_identity_derived": True,
        "root_extraction_not_required": True,
        "moroz_schost_primary_algorithm_bound_bound": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_occurrence_root_multiplicities_exact": all_multiplicities,
        "all_actual_positive_valuations_exact": all_positive,
        "all_actual_empty_valuations_zero": all_empty,
        "candidate_dlp_and_root_oracles_avoided": True,
        "standard_local_truncation_cost_charged": True,
        "standard_a6_batch_exceeds_rho": True,
        "standard_full_relation_batch_exceeds_rho": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "shared_sublinear_multi_target_valuation_index_complete": False,
        "offline_online_transposed_atom_marginals_complete": False,
        "generic_integer_lift_complete": False,
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
        "Do not apply local truncated resultants independently. Construct "
        "one genuinely shared transposed multi-target valuation-and-marker "
        "operator for the occurrence divisor. It must process the complete "
        "B^(5/4) target stream after B^(9/4) setup without a B^(9/4) factor "
        "per target, emit exact integer counts and B^(3/4) A/C marginals, "
        "and replay R144 rank, factor logs, and shifted descent. It may use "
        "no DLP, root, resultant, valuation, count, marginal, rank, or "
        "source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "ORDERED_C6_COUNT_IS_LOCAL_VALUATION_OF_OCCURRENCE_PAIR_"
            "RESULTANT__REPEATED_C3_DIVISOR_DEGREE_B9O4__FULL_PAIR_"
            "RESULTANT_NEVER_MATERIALIZED__MOROZ_SCHOST_LOCAL_TRUNCATION_"
            "SOFT_O_DK__ONE_LOCAL_QUERY_B9O4__A6_BATCH_B11O4__FULL_"
            "RELATION_STREAM_B7O2__SHARED_TRANSPOSED_VALUATION_MARKER_"
            "OPERATOR_OPEN__NO_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether replacing dense squarefree gcd traces by "
            "local valuations of a repeated occurrence pair-resultant "
            "meets the source-equivalent R146 count-and-marginal caps."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "occurrence_divisor": frozen["occurrence_divisor"],
            "pair_resultant": frozen["pair_resultant"],
            "standard_truncated_resultant_cost": cost_boundary,
            "finite_actual_result": (
                "Every sampled positive and empty target on all eight "
                "actual controls has local pair-resultant valuation equal "
                "to its direct ordered C6 count."
            ),
            "scope": cost_boundary["scope"],
            "literature_novelty": "unverified",
        },
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "local_valuation_count_identity_admitted": True,
            "standard_componentwise_truncation_negative_admitted": True,
            "shared_multi_target_operator_admitted": False,
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
