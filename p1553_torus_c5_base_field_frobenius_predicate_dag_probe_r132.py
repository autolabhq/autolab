#!/usr/bin/env python3
"""Audit base-field predicate DAGs under order-two Frobenius inversion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_base_field_frobenius_predicate_dag.r132.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
DECK_EXPONENT = Fraction(3, 4)
C5_EXPONENT = Fraction(15, 4)
GROUP_ORDER_EXPONENT = Fraction(5)
RANDOM_INVERSE_OVERLAP_EXPONENT = Fraction(5, 2)

R131_PRODUCER = pathlib.Path(
    "p1553_torus_c5_consecutive_mode_predicate_probe_r131.py"
)
R131_PRODUCER_SHA256 = (
    "d312f28885677339632ec9517c8529e327a63fa7e8aca5d6a67d6a4f71e3adf9"
)
R131_REPORT = pathlib.Path(
    "p1553_torus_c5_consecutive_mode_predicate_probe_report_r131.json"
)
R131_REPORT_SHA256 = (
    "a2ffed2c599fd1003dce13a3a360d7ac49f983ab92a439f1023a8abe0cb2efcb"
)
R131_FROZEN = pathlib.Path(
    "frozen_torus_c5_consecutive_mode_predicate.json"
)
R131_FROZEN_SHA256 = (
    "46775815377c1cadbc095735b4f1fe9cd64167ea7b933664ae5892e5bc1b8ded"
)
R131_COST = pathlib.Path(
    "torus_c5_consecutive_mode_predicate_cost_ledger.json"
)
R131_COST_SHA256 = (
    "d18d3317aa2952b604c880d44447cef0d7bbc2c01b33a28b4131889fdfea8965"
)
R131_REPLAY = pathlib.Path(
    "torus_c5_consecutive_mode_predicate_replay.json"
)
R131_REPLAY_SHA256 = (
    "80c0a9d57247a981a55ec104cfb8f933e44b2bc610db945aaa89a7b473068f9b"
)
R131_CONTROLS = pathlib.Path(
    "torus_c5_consecutive_mode_predicate_controls.json"
)
R131_CONTROLS_SHA256 = (
    "c9b874db6c10f6108b1179bcddd7e2e5dcc398cec6eaa871649912f07e20fb2a"
)
R131_LOGS = pathlib.Path("factor_logs_and_identical_descent_r131.json")
R131_LOGS_SHA256 = (
    "9388946d26f2e311fd712feaaed9e6905e4e5ecbe5ceed554084d6e57fd39b36"
)
R131_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_consecutive_mode_predicate_probe_r131.py"
)
R131_TEST_SHA256 = (
    "50c4fb705854bb11de19096716295d05ba54b035e6dc8ef675c112219481cb3a"
)
R131_GATE = pathlib.Path(
    "p1553_torus_c5_consecutive_mode_predicate_probe_gate_r131.md"
)
R131_GATE_SHA256 = (
    "e2b714358fa386b23f8c9bdf75ab6d6971d080209d0e45885b4ba4fbb623f11d"
)
R131_PARENT = pathlib.Path(
    "p1553_torus_c5_consecutive_mode_predicate_"
    "probe_parent_report_r131.yaml"
)
R131_PARENT_SHA256 = (
    "ca65e6e86c3aeae074b13d80774b08c18743932776e8b546409417c40dca6e46"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R131 = load_module("p1553_r131_for_r132", R131_PRODUCER)
R129 = R131.R129
R126 = R131.R126
R121 = R131.R121
R82 = R131.R82
Field = R131.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r131_producer", R131_PRODUCER, R131_PRODUCER_SHA256),
        ("r131_report", R131_REPORT, R131_REPORT_SHA256),
        ("r131_frozen", R131_FROZEN, R131_FROZEN_SHA256),
        ("r131_cost", R131_COST, R131_COST_SHA256),
        ("r131_replay", R131_REPLAY, R131_REPLAY_SHA256),
        ("r131_controls", R131_CONTROLS, R131_CONTROLS_SHA256),
        ("r131_logs", R131_LOGS, R131_LOGS_SHA256),
        ("r131_test", R131_TEST, R131_TEST_SHA256),
        ("r131_gate", R131_GATE, R131_GATE_SHA256),
        ("r131_parent", R131_PARENT, R131_PARENT_SHA256),
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
        raise AssertionError(f"R132 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def frobenius(value: Fp2, field: Field) -> Fp2:
    return value[0], (-value[1]) % field.p


def evaluate_base_field_polynomial(
    coefficients: Iterable[int],
    value: Fp2,
    field: Field,
) -> Fp2:
    result = field.zero
    for coefficient in coefficients:
        result = field.add(
            field.mul(result, value),
            field.elt(coefficient),
        )
    return result


def coefficients_in_base_field(coefficients: Iterable[Fp2]) -> bool:
    return all(value[1] == 0 for value in coefficients)


def frobenius_predicate_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, 5)
    c5_values = tuple(value for _, value in c5)
    c5_set = set(c5_values)
    inverse_values = tuple(field.inv(value) for value in c5_values)
    base_polynomials = (
        (1,),
        (1, 1),
        (1, 2, 3),
        (5, 0, 7, 9),
        (1, 0, 0, 0, 11),
    )
    polynomial_checks = []
    for coefficients in base_polynomials:
        identities = []
        zero_invariance = []
        for value, inverse in zip(c5_values, inverse_values):
            evaluated = evaluate_base_field_polynomial(
                coefficients,
                value,
                field,
            )
            evaluated_inverse = evaluate_base_field_polynomial(
                coefficients,
                inverse,
                field,
            )
            identities.append(
                evaluated_inverse == frobenius(evaluated, field)
            )
            zero_invariance.append(
                (evaluated == field.zero)
                == (evaluated_inverse == field.zero)
            )
        polynomial_checks.append(
            {
                "coefficients_high_to_low": list(coefficients),
                "frobenius_evaluation_identity_exact": all(identities),
                "zero_outcome_inversion_invariant": all(zero_invariance),
            }
        )

    support_annihilator = R121.root_annihilator(c5_values, field)
    color_controls = []
    for color, part in enumerate(R129.balanced_four_parts(len(deck))):
        if not part:
            continue
        accepted = tuple(
            value
            for source, value in c5
            if R131.source_color_multiplicity(source, color) >= 2
        )
        accepted_set = set(accepted)
        inverse_accepted = tuple(field.inv(value) for value in accepted)
        annihilator = R121.root_annihilator(accepted, field)
        color_controls.append(
            {
                "color": color,
                "part_size": len(part),
                "accepted_target_count": len(accepted),
                "accepted_inverse_in_same_color_count": sum(
                    value in accepted_set for value in inverse_accepted
                ),
                "accepted_inverse_in_any_c5_count": sum(
                    value in c5_set for value in inverse_accepted
                ),
                "accepted_inverse_empty_count": sum(
                    value not in c5_set for value in inverse_accepted
                ),
                "accepted_set_inversion_invariant": (
                    set(inverse_accepted) == accepted_set
                ),
                "exact_annihilator_coefficient_count": len(annihilator),
                "exact_annihilator_in_base_field": (
                    coefficients_in_base_field(annihilator)
                ),
                "exact_annihilator_uses_extension_coefficient": (
                    not coefficients_in_base_field(annihilator)
                ),
            }
        )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "characteristic_is_minus_one_mod_subgroup_order": (
            field.p % curve["subgroup_order"]
            == curve["subgroup_order"] - 1
        ),
        "deck_size": len(deck),
        "c5_source_count": len(c5),
        "c5_targets_distinct": len(c5_set) == len(c5_values),
        "all_c5_targets_nonzero": all(value != field.zero for value in c5_values),
        "all_target_frobenius_values_equal_inverses": all(
            frobenius(value, field) == inverse
            for value, inverse in zip(c5_values, inverse_values)
        ),
        "positive_inverse_in_c5_count": sum(
            value in c5_set for value in inverse_values
        ),
        "positive_inverse_empty_count": sum(
            value not in c5_set for value in inverse_values
        ),
        "positive_support_inversion_disjoint": (
            not c5_set.intersection(inverse_values)
        ),
        "positive_support_inversion_invariant": (
            c5_set == set(inverse_values)
        ),
        "support_annihilator_coefficient_count": len(support_annihilator),
        "support_annihilator_in_base_field": (
            coefficients_in_base_field(support_annihilator)
        ),
        "support_annihilator_uses_extension_coefficient": (
            not coefficients_in_base_field(support_annihilator)
        ),
        "base_polynomial_checks": polynomial_checks,
        "all_sample_base_polynomial_frobenius_identities_exact": all(
            row["frobenius_evaluation_identity_exact"]
            for row in polynomial_checks
        ),
        "all_sample_base_polynomial_zero_outcomes_invariant": all(
            row["zero_outcome_inversion_invariant"]
            for row in polynomial_checks
        ),
        "color_controls": color_controls,
        "all_color_supports_inversion_disjoint": all(
            row["accepted_inverse_in_same_color_count"] == 0
            for row in color_controls
        ),
        "all_color_inverses_empty": all(
            row["accepted_inverse_empty_count"]
            == row["accepted_target_count"]
            for row in color_controls
        ),
        "all_color_annihilators_use_extension_coefficients": all(
            row["exact_annihilator_uses_extension_coefficient"]
            for row in color_controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        frobenius_predicate_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    return {
        "schema": (
            "p1553.torus_c5_base_field_frobenius_predicate_dag_"
            "controls.r132.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_fields_have_characteristic_minus_one_mod_q": all(
            row["characteristic_is_minus_one_mod_subgroup_order"]
            for row in controls
        ),
        "all_c5_supports_injective": all(
            row["c5_targets_distinct"] for row in controls
        ),
        "all_frobenius_maps_equal_inversion": all(
            row["all_target_frobenius_values_equal_inverses"]
            for row in controls
        ),
        "all_positive_supports_inversion_disjoint": all(
            row["positive_support_inversion_disjoint"]
            for row in controls
        ),
        "all_positive_inverses_are_empty": all(
            row["positive_inverse_empty_count"] == row["c5_source_count"]
            for row in controls
        ),
        "all_sample_base_polynomial_frobenius_identities_exact": all(
            row["all_sample_base_polynomial_frobenius_identities_exact"]
            for row in controls
        ),
        "all_sample_base_polynomial_zero_outcomes_invariant": all(
            row["all_sample_base_polynomial_zero_outcomes_invariant"]
            for row in controls
        ),
        "all_support_annihilators_use_extension_coefficients": all(
            row["support_annihilator_uses_extension_coefficient"]
            for row in controls
        ),
        "all_color_supports_inversion_disjoint": all(
            row["all_color_supports_inversion_disjoint"]
            for row in controls
        ),
        "all_color_inverses_are_empty": all(
            row["all_color_inverses_empty"] for row in controls
        ),
        "all_color_annihilators_use_extension_coefficients": all(
            row["all_color_annihilators_use_extension_coefficients"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "order_two_frobenius": (
            "For z in the order-q subgroup and characteristic p=-1 mod q, "
            "z^p=z^(-1)."
        ),
        "base_field_polynomial_identity": (
            "For f in F_p[X], f(z^(-1))=f(z^p)=f(z)^p. Hence "
            "f(z)=0 if and only if f(z^(-1))=0."
        ),
        "base_field_rational_identity": (
            "The numerator-zero and denominator-zero outcomes of every "
            "univariate F_p rational predicate are inversion invariant on "
            "the subgroup."
        ),
        "decision_dag_consequence": (
            "A deterministic DAG branching only on zero or definedness "
            "outcomes of univariate base-field polynomial or rational "
            "predicates follows the same path on z and z^(-1)."
        ),
        "actual_control_contradiction": (
            "In every actual R82 pairing deck, each positive C5 target has "
            "an empty inverse. Such a base-field predicate DAG cannot be an "
            "exact membership or source locator on those controls."
        ),
        "uniform_random_subset_comparator": (
            "For a uniformly random M-subset S of an odd prime-order "
            "group, E|S intersect S^(-1)|=M^2/q. At "
            "M=B^(15/4+o(1)), q=B^(5+o(1)), this is "
            "B^(5/2+o(1)), an o(1) fraction B^(-5/4+o(1)) of S."
        ),
        "random_subset_overlap_exponent_B": fraction_record(
            RANDOM_INVERSE_OVERLAP_EXPONENT
        ),
        "random_subset_comparator_is_model_bound": True,
        "random_subset_comparator_receives_candidate_credit": False,
        "scope_limits": [
            "univariate target polynomials with coefficients in F_p",
            "univariate base-field rational zero and definedness tests",
            "deterministic DAGs using only those Boolean outcomes",
            "actual inversion-disjoint R82 C5 supports",
        ],
        "not_covered": [
            "extension-field coefficients",
            "Frobenius-aware coordinate predicates in (z,z^p)",
            "tests that consume nonzero predicate values",
            "lacunary F_(p^2) polynomials",
            "adaptive cell-probe selectors",
            "general arithmetic-circuit or data-structure lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_base_field_frobenius_predicate_dag_"
            "cost_ledger.r132.v1"
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
                "route_id": "base_field_univariate_zero_predicate_dag",
                "inversion_invariant": True,
                "exact_on_actual_inversion_disjoint_supports": False,
                "status": "rejected_on_actual_controls",
            },
            {
                "route_id": "base_field_trace_or_dickson_predicate_dag",
                "inversion_invariant": True,
                "exact_on_actual_inversion_disjoint_supports": False,
                "status": "rejected_on_actual_controls",
            },
            {
                "route_id": "extension_field_lacunary_predicate",
                "inversion_invariant_forced": False,
                "inside_cap_exact_circuit_constructed": False,
                "status": "open",
            },
            {
                "route_id": "frobenius_aware_coordinate_predicate_dag",
                "may_distinguish_z_from_inverse": True,
                "inside_cap_exact_circuit_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R131_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R131 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "r131_lacunary_interface_inherited": (
            inherited["admission"][
                "actual_field_consecutive_mode_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "order_two_frobenius_identity_explicit": (
            "z^p=z^(-1)" in theorem["order_two_frobenius"]
        ),
        "base_field_zero_invariance_theorem_explicit": (
            "if and only if"
            in theorem["base_field_polynomial_identity"]
        ),
        "base_field_zero_predicate_dag_scope_explicit": (
            "same path" in theorem["decision_dag_consequence"]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "all_actual_frobenius_maps_equal_inversion": controls[
            "all_frobenius_maps_equal_inversion"
        ],
        "all_actual_positive_inverses_empty": controls[
            "all_positive_inverses_are_empty"
        ],
        "all_sample_base_polynomial_identities_exact": (
            controls[
                "all_sample_base_polynomial_frobenius_identities_exact"
            ]
            and controls[
                "all_sample_base_polynomial_zero_outcomes_invariant"
            ]
        ),
        "all_support_and_color_annihilators_need_extension": (
            controls[
                "all_support_annihilators_use_extension_coefficients"
            ]
            and controls[
                "all_color_annihilators_use_extension_coefficients"
            ]
        ),
        "random_subset_comparator_scoped_as_model_bound": (
            theorem["random_subset_comparator_is_model_bound"]
            and not theorem[
                "random_subset_comparator_receives_candidate_credit"
            ]
        ),
        "base_field_univariate_dag_rejected_on_actual_controls": (
            cost["routes"][0]["status"] == "rejected_on_actual_controls"
            and cost["routes"][1]["status"]
            == "rejected_on_actual_controls"
        ),
        "extension_or_frobenius_aware_predicate_complete": False,
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
        "Construct an asymmetric lacunary predicate over F_(p^2), or a "
        "Frobenius-aware coordinate DAG using both z and z^p. It must "
        "distinguish the observed positive/empty inverse pairs, choose a "
        "valid C2 branch in polylogarithmic arbitrary-target work, return "
        "exact C2+C3 sources or an empty certificate, fit "
        "B^(9/4+o(1)) state, avoid field DLP, and include rank, logs, "
        "identical descent, memory, field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_base_field_frobenius_"
            "predicate_dag.r132.v1"
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
            "univariate base-field polynomial zero-test DAGs",
            "univariate base-field rational zero/definedness DAGs",
            "base-field trace or Dickson zero-test DAGs",
        ],
        "preserved_interface": (
            "asymmetric extension-field lacunary predicate or "
            "Frobenius-aware coordinate decision DAG"
        ),
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_base_field_frobenius_predicate_dag_"
            "replay.r132.v1"
        ),
        "actual_control_count": controls["control_count"],
        "all_actual_frobenius_maps_equal_inversion": controls[
            "all_frobenius_maps_equal_inversion"
        ],
        "all_actual_positive_inverses_empty": controls[
            "all_positive_inverses_are_empty"
        ],
        "all_base_polynomial_zero_outcomes_invariant": controls[
            "all_sample_base_polynomial_zero_outcomes_invariant"
        ],
        "all_exact_annihilators_need_extension_coefficients": (
            controls[
                "all_support_annihilators_use_extension_coefficients"
            ]
            and controls[
                "all_color_annihilators_use_extension_coefficients"
            ]
        ),
        "inside_cap_asymmetric_or_frobenius_aware_dag_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r132.v1",
        "r131_consecutive_mode_actual_field_audit_complete": True,
        "r132_base_field_frobenius_dag_audit_complete": True,
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
        "ORDER2_FROBENIUS_EQUALS_INVERSION__UNIVARIATE_BASE_FIELD_"
        "POLYNOMIAL_RATIONAL_ZERO_DAGS_ARE_INVERSION_INVARIANT__ALL_EIGHT_"
        "ACTUAL_C5_SUPPORTS_HAVE_EVERY_POSITIVE_INVERSE_EMPTY_AND_REQUIRE_"
        "EXTENSION_COEFFICIENT_ANNIHILATORS__BASE_FIELD_DAG_REJECTED_ON_"
        "ACTUAL_CONTROLS__ASYMMETRIC_EXTENSION_OR_FROBENIUS_AWARE_DAG_"
        "OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_ACTUAL_FIELD_INVERSE_WITNESSES_AND_SCOPED_BASE_FIELD_"
            "FROBENIUS_DAG_OBSTRUCTION_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_actual_positive_inverses_empty": controls[
                "all_positive_inverses_are_empty"
            ],
            "all_base_polynomial_zero_outcomes_invariant": controls[
                "all_sample_base_polynomial_zero_outcomes_invariant"
            ],
            "all_exact_annihilators_need_extension_coefficients": (
                controls[
                    "all_support_annihilators_use_extension_coefficients"
                ]
                and controls[
                    "all_color_annihilators_use_extension_coefficients"
                ]
            ),
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "base_field_frobenius_dag_negative_admitted": True,
            "actual_inverse_witnesses_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_base_field_frobenius_predicate_dag.json"
            ),
            "cost": (
                "torus_c5_base_field_frobenius_predicate_dag_"
                "cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_base_field_frobenius_predicate_dag_replay.json"
            ),
            "controls": (
                "torus_c5_base_field_frobenius_predicate_dag_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r132.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The actual inverse witnesses receive no asymptotic credit.",
            "The random-subset comparator is model-bound.",
            "Extension-field and Frobenius-aware predicates remain open.",
            "Nonzero-value and coordinate tests are outside the theorem.",
            "No general circuit or data-structure lower bound is claimed.",
            "No rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_ORDER2_FROBENIUS_INVERSION_THEOREM_AND_EXACT_ACTUAL_"
            "POSITIVE_EMPTY_INVERSE_WITNESSES_ONLY__REJECT_UNIVARIATE_"
            "BASE_FIELD_ZERO_TEST_DAGS_ON_ACTUAL_CONTROLS__PRESERVE_"
            "ASYMMETRIC_EXTENSION_AND_FROBENIUS_AWARE_COORDINATE_DAGS__"
            "NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_"
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
            "p1553_torus_c5_base_field_frobenius_predicate_dag_"
            "probe_report_r132.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_base_field_frobenius_predicate_dag.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_base_field_frobenius_predicate_dag_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_base_field_frobenius_predicate_dag_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_base_field_frobenius_predicate_dag_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r132.json"),
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
        f"R132 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
