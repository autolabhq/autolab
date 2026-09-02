#!/usr/bin/env python3
"""Audit label-only Mobius-character composition and explicit-row costs."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.torus_c5_label_congruence_correction.r143.v1"

R142_PRODUCER = ROOT / (
    "p1553_torus_c5_adaptive_character_decision_router_probe_r142.py"
)
R142_REPORT = ROOT / (
    "p1553_torus_c5_adaptive_character_decision_router_"
    "probe_report_r142.json"
)
R142_FROZEN = ROOT / (
    "frozen_torus_c5_adaptive_character_decision_router.json"
)
R142_COST = ROOT / (
    "torus_c5_adaptive_character_decision_router_cost_ledger.json"
)
R142_REPLAY = ROOT / (
    "torus_c5_adaptive_character_decision_router_replay.json"
)
R142_CONTROLS = ROOT / (
    "torus_c5_adaptive_character_decision_router_controls.json"
)
R142_LOGS = ROOT / "factor_logs_and_identical_descent_r142.json"
R142_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_adaptive_character_decision_router_probe_r142.py"
)
R142_GATE = ROOT / (
    "p1553_torus_c5_adaptive_character_decision_router_probe_gate_r142.md"
)
R142_PARENT = ROOT / (
    "p1553_torus_c5_adaptive_character_decision_router_"
    "probe_parent_report_r142.yaml"
)

R142_BINDINGS = (
    (
        "r142_producer",
        R142_PRODUCER,
        "d99409f519e96a9c62644bddc8262d9e535ec55e8fdd4a70ee828a5486af2eef",
    ),
    (
        "r142_report",
        R142_REPORT,
        "2a6b812448fcf1e24aac0a7d85f56e2fc0cc9a26de5aaf73b42a1d8ae58b8122",
    ),
    (
        "r142_frozen",
        R142_FROZEN,
        "b3b1a96ec46f8617b249f366b3fb77bc64bdde4d5188b47e293033a63a82a51e",
    ),
    (
        "r142_cost",
        R142_COST,
        "71ee44314fd109bf2854a659d4a80356d5b084602f1f2e5a7113e2018ba706d4",
    ),
    (
        "r142_replay",
        R142_REPLAY,
        "b285ea1f2883f16b53f30b00aa6793d84f55b8397200297ec7c227c5e3083a10",
    ),
    (
        "r142_controls",
        R142_CONTROLS,
        "bf452fbfc523fa2469c70da68e25439d8fe79fe7a550a12f8cc6736d6b73d59a",
    ),
    (
        "r142_logs",
        R142_LOGS,
        "d5a35abfa58dc93a8a5bd6d2d315a186c53a8e435d637084abe02d106f63b81f",
    ),
    (
        "r142_test",
        R142_TEST,
        "43723e93867533406eb45ab41cdfedf9f1476f8879018b32e8a106be7935dcb0",
    ),
    (
        "r142_gate",
        R142_GATE,
        "c4663f680cd2816d721b0d8020c4c5ced9479fc40b4b417701ebc2e060cbdae4",
    ),
    (
        "r142_parent",
        R142_PARENT,
        "4ab045100abb9537baa6c2e89cbfbd82809f6adff420837dcadce1224dc3d6d7",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_torus_c5_label_congruence_correction_"
    "probe_report_r143.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_torus_c5_label_congruence_correction.json"
)
DEFAULT_COST = ROOT / (
    "torus_c5_label_congruence_correction_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "torus_c5_label_congruence_correction_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "torus_c5_label_congruence_correction_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r143.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R142 = load_module("p1553_r142_for_r143", R142_PRODUCER)
R141 = R142.R141


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in R142_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in R142_BINDINGS
    }
    failures = [
        name
        for name, _, expected in R142_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R143 source binding mismatch: {failures}")
    return actual


def unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def signature(
    value: Any,
    parameters: tuple[Any, ...],
    indexes: tuple[int, ...],
    subgroup_order: int,
    field: Any,
) -> tuple[Any, ...]:
    return tuple(
        R141.sextic_character(
            value,
            parameters[index],
            subgroup_order,
            field,
        )
        for index in indexes
    )


def cross_ratio_defect_raw(
    left: Any,
    target: Any,
    parameter: Any,
    field: Any,
) -> Any:
    xz = field.mul(left, parameter)
    tz = field.mul(target, parameter)
    numerator = field.neg(
        field.product(
            (
                field.add(field.add(target, xz), left),
                field.add(field.add(left, parameter), field.one),
                field.add(field.add(tz, target), parameter),
            )
        )
    )
    denominator = field.product(
        (
            field.add(field.add(target, parameter), field.one),
            field.add(field.add(tz, target), xz),
            field.add(field.add(xz, left), parameter),
        )
    )
    return field.div(numerator, denominator)


def cross_ratio_defect_control(
    left: Any,
    right: Any,
    parameter: Any,
    subgroup_order: int,
    field: Any,
) -> dict[str, Any]:
    target = field.mul(left, right)
    target_character = R141.sextic_character(
        target,
        parameter,
        subgroup_order,
        field,
    )
    operand_character = field.mul(
        R141.sextic_character(
            left,
            parameter,
            subgroup_order,
            field,
        ),
        R141.sextic_character(
            right,
            parameter,
            subgroup_order,
            field,
        ),
    )
    direct_defect = field.div(target_character, operand_character)
    raw = cross_ratio_defect_raw(left, target, parameter, field)
    cross_ratio_defect = field.pow(raw, subgroup_order)
    return {
        "left": field.json(left),
        "right": field.json(right),
        "target": field.json(target),
        "parameter": field.json(parameter),
        "direct_defect": field.json(direct_defect),
        "cross_ratio_defect": field.json(cross_ratio_defect),
        "identity_exact": direct_defect == cross_ratio_defect,
    }


def composition_subset_control(
    c2: tuple[Any, ...],
    c3: tuple[Any, ...],
    parameters: tuple[Any, ...],
    indexes: tuple[int, ...],
    subgroup_order: int,
    field: Any,
) -> dict[str, Any]:
    value_signatures: dict[Any, tuple[Any, ...]] = {}

    def sig(value: Any) -> tuple[Any, ...]:
        if value not in value_signatures:
            value_signatures[value] = signature(
                value,
                parameters,
                indexes,
                subgroup_order,
                field,
            )
        return value_signatures[value]

    outputs: dict[
        tuple[tuple[Any, ...], tuple[Any, ...]],
        set[tuple[Any, ...]],
    ] = {}
    for left in c2:
        left_signature = sig(left)
        for right in c3:
            key = (left_signature, sig(right))
            outputs.setdefault(key, set()).add(
                sig(field.mul(left, right))
            )
    variant_counts = [len(values) for values in outputs.values()]
    deterministic = all(count == 1 for count in variant_counts)
    return {
        "parameter_indexes": list(indexes),
        "parameter_count": len(indexes),
        "input_signature_pair_count": len(outputs),
        "input_output_table_entry_count": sum(variant_counts),
        "ambiguous_input_signature_pair_count": sum(
            count > 1 for count in variant_counts
        ),
        "maximum_output_variant_count": max(variant_counts),
        "composition_deterministic": deterministic,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck = R141.R121.pairing_deck(curve, offset)
    subgroup_order = curve["subgroup_order"]
    c2 = R141.product_deck(deck, 2, field)
    c3 = R141.product_deck(deck, 3, field)
    parameters = unique(value for value in deck if value != field.one)
    if not parameters:
        raise AssertionError("actual control has no Mobius parameter")
    subset_controls = [
        composition_subset_control(
            c2,
            c3,
            parameters,
            tuple(
                index
                for index in range(len(parameters))
                if mask & (1 << index)
            ),
            subgroup_order,
            field,
        )
        for mask in range(1, 1 << len(parameters))
    ]
    deterministic = [
        row
        for row in subset_controls
        if row["composition_deterministic"]
    ]
    best = (
        min(
            deterministic,
            key=lambda row: (
                row["input_output_table_entry_count"],
                row["parameter_count"],
                row["parameter_indexes"],
            ),
        )
        if deterministic
        else None
    )
    pair_count = len(c2) * len(c3)
    finite_cap = math.floor(len(deck) ** (9 / 4))
    defect_rows = [
        cross_ratio_defect_control(
            left,
            right,
            parameter,
            subgroup_order,
            field,
        )
        for parameter in parameters
        for left in c2
        for right in c3
    ]
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "deck_size": len(deck),
        "c2_size": len(c2),
        "c3_size": len(c3),
        "c2_by_c3_operand_pair_count": pair_count,
        "parameter_count": len(parameters),
        "parameter_subset_count": len(subset_controls),
        "subset_controls": subset_controls,
        "deterministic_subset_count": len(deterministic),
        "best_deterministic_composition": best,
        "deterministic_composition_exists": best is not None,
        "best_table_recreates_full_c2_by_c3_body": (
            best is not None
            and best["input_output_table_entry_count"] == pair_count
        ),
        "finite_B9_over_4_table_cap_floor": finite_cap,
        "best_deterministic_table_inside_finite_cap": (
            best is not None
            and best["input_output_table_entry_count"] <= finite_cap
        ),
        "cross_ratio_defect_row_count": len(defect_rows),
        "all_cross_ratio_defect_identities_exact": all(
            row["identity_exact"] for row in defect_rows
        ),
        "cross_ratio_defect_rows": defect_rows,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def generalized_birthday_boundary() -> dict[str, Any]:
    beta = Fraction(9, 20)
    minimum_merge = 1 - beta
    return {
        "model": (
            "Four-list explicit relation generation with factor-base size "
            "n=N^beta and an ideal compatible projection of size N^mu."
        ),
        "filtered_pair_list_exponent_N": "2*beta-mu",
        "four_source_output_exponent_N": "4*beta-mu-1",
        "explicit_rank_requirement": (
            "At least N^beta explicit relation rows are required for "
            "N^beta factor-log unknowns."
        ),
        "rank_supply_constraint": "mu<=3*beta-1",
        "merge_work_lower_envelope": "1-beta",
        "explicit_row_output_exponent_N": "beta",
        "total_lower_envelope": "max(beta,1-beta)",
        "minimum_over_beta": {
            "beta": fraction_record(Fraction(1, 2)),
            "total_exponent_N": fraction_record(Fraction(1, 2)),
        },
        "campaign_beta": fraction_record(beta),
        "campaign_merge_exponent_N": fraction_record(minimum_merge),
        "campaign_explicit_output_exponent_N": fraction_record(beta),
        "campaign_below_rho": False,
        "scope": (
            "Exponent boundary for one ideal projected two-pair merge that "
            "must emit enough explicit rows. It is not a lower bound for "
            "implicit relation-span operators, compressed linear algebra, "
            "nonuniform structured factor bases, or general ECDLP "
            "algorithms."
        ),
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        actual_control(curve, offset)
        for curve in R141.R82.FAMILIES
        for offset in (0, 1)
    ]
    all_defects_exact = all(
        row["all_cross_ratio_defect_identities_exact"]
        for row in actual
    )
    no_inside_cap_table = all(
        not row["best_deterministic_table_inside_finite_cap"]
        for row in actual
    )
    full_body_when_deterministic = all(
        not row["deterministic_composition_exists"]
        or row["best_table_recreates_full_c2_by_c3_body"]
        for row in actual
    )
    deterministic_count = sum(
        row["deterministic_composition_exists"] for row in actual
    )
    impossible_count = len(actual) - deterministic_count
    birthday = generalized_birthday_boundary()

    controls = {
        "schema": (
            "p1553.torus_c5_label_congruence_correction."
            "controls.r143.v1"
        ),
        "actual_control_count": len(actual),
        "all_cross_ratio_defect_identities_exact": all_defects_exact,
        "deterministic_composition_control_count": deterministic_count,
        "composition_impossible_control_count": impossible_count,
        "all_deterministic_tables_recreate_full_c2_by_c3_body": (
            full_body_when_deterministic
        ),
        "all_controls_fail_finite_B9_over_4_table_cap": no_inside_cap_table,
        "actual_controls": actual,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }

    frozen = {
        "schema": (
            "p1553.torus_c5_label_congruence_correction."
            "frozen.r143.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "cross_ratio_defect": {
            "definition": (
                "delta_z(x,y)=chi_z(x*y)/(chi_z(x)*chi_z(y))"
            ),
            "target_split_formula": (
                "For t=x*y, delta_z(x,t/x) is the qth power of "
                "-(t+x*z+x)*(x+z+1)*(t*z+t+z) / "
                "((t+z+1)*(t*z+t+x*z)*(x*z+x+z))."
            ),
            "branch_point_degree": "two numerator and two denominator roots in x",
            "candidate_dependence": (
                "The fixed-degree formula still requires the unknown C2 "
                "operand x."
            ),
        },
        "label_congruence_theorem": {
            "hypothesis": (
                "sigma:G->A and F:A*A->A satisfy "
                "sigma(x*y)=F(sigma(x),sigma(y)) for every x,y in G."
            ),
            "proof": (
                "Equality of sigma values is a multiplication congruence. "
                "The identity class is a subgroup and every class is its "
                "coset. For prime-order G the subgroup is G or {1}."
            ),
            "conclusion": (
                "Every total label-only composition is constant or "
                "injective."
            ),
            "sextic_tuple_consequence": (
                "A nonconstant k-label Mobius signature with 6^k<q cannot "
                "have a total label-only product law."
            ),
        },
        "generalized_birthday_boundary": birthday,
        "required_open_outputs": {
            "implicit_relation_span_operator": "open",
            "compressed_known_rhs_rank": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.torus_c5_label_congruence_correction."
            "replay.r143.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "cross_ratio_defect_rows": row[
                    "cross_ratio_defect_rows"
                ],
                "best_deterministic_composition": row[
                    "best_deterministic_composition"
                ],
                "best_table_recreates_full_c2_by_c3_body": row[
                    "best_table_recreates_full_c2_by_c3_body"
                ],
            }
            for row in actual
        ],
        "all_cross_ratio_defect_identities_exact": all_defects_exact,
    }

    cost = {
        "schema": (
            "p1553.torus_c5_label_congruence_correction."
            "cost.r143.v1"
        ),
        "single_cross_ratio_defect_evaluation": (
            "O(log q) field operations after the candidate C2 operand is known"
        ),
        "candidate_C2_scan_exponent_B": {
            "exact": "2",
            "decimal": 2.0,
        },
        "label_only_fixed_k_composition": (
            "impossible when nonconstant and 6^k<q"
        ),
        "injective_label_threshold": "k>=ceil(log_6(q))",
        "explicit_restricted_correction_table_exponent_B": {
            "exact": "5",
            "decimal": 5.0,
        },
        "explicit_restricted_table_inside_setup_cap": False,
        "generalized_birthday_boundary": birthday,
        "implicit_relation_span_operator_cost_supplied": False,
        "candidate_field_dlp_used": False,
        "rank_cost_supplied": False,
        "factor_log_cost_supplied": False,
        "identical_descent_cost_supplied": False,
        "total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.torus_c5_label_congruence_correction."
            "logs_descent.r143.v1"
        ),
        "known_rhs_relation_rank_computed": False,
        "factor_logs_computed": False,
        "identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "r142_adaptive_tree_boundary_inherited": True,
        "cross_ratio_defect_formula_derived": True,
        "all_actual_cross_ratio_defects_replay": all_defects_exact,
        "prime_group_label_congruence_theorem_proved": True,
        "compressed_nonconstant_total_label_law_rejected": True,
        "all_parameter_subsets_enumerated_on_eight_controls": (
            len(actual) == 8
        ),
        "two_restricted_controls_have_no_deterministic_composition": (
            impossible_count == 2
        ),
        "six_deterministic_controls_recreate_full_pair_body": (
            deterministic_count == 6 and full_body_when_deterministic
        ),
        "all_controls_fail_finite_table_cap": no_inside_cap_table,
        "explicit_four_list_birthday_envelope_derived": True,
        "explicit_row_envelope_minimizes_at_rho": (
            birthday["minimum_over_beta"]["total_exponent_N"]["exact"]
            == "1/2"
        ),
        "scope_excludes_implicit_relation_span_operators": True,
        "candidate_discrete_log_oracle_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "implicit_relation_span_operator_complete": False,
        "compressed_known_rhs_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Pivot from character landmarks and explicit Wagner rows to one "
        "target-batched transposed summation-polynomial/FFE relation-span "
        "operator. At factor-base exponent beta=9/20, it must compile the "
        "implicit high-density relation family, expose enough independent "
        "known-RHS row action to solve factor logs, and return identical "
        "target descent in N^(1/2-epsilon) total field and bit work with "
        "N^(9/20+o(1)) state. It may not enumerate the N^(11/20) pair "
        "merge, emit N^(9/20) rows without charging them, consume a DLP or "
        "root oracle, or infer rank from finite fixtures."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "LABEL_ONLY_COMPOSITION_REJECTED__"
            "EXPLICIT_ROW_BIRTHDAY_AT_RHO__"
            "IMPLICIT_RELATION_SPAN_OPEN"
        ),
        "objective": (
            "Determine whether the R141 Mobius characters admit a compact "
            "label-only C2-by-C3 product correction, and whether an ideal "
            "Wagner projection with explicit rows could cross rho."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "cross_ratio_defect": frozen["cross_ratio_defect"],
            "label_congruence": frozen["label_congruence_theorem"],
            "actual_correction_table_result": (
                "Two controls have no deterministic composition for any "
                "nonempty deck-parameter subset. In the other six, the "
                "smallest deterministic table has one entry for every "
                "C2-by-C3 operand pair."
            ),
            "generalized_birthday_explicit_row_boundary": birthday,
            "literature_novelty": "unverified",
            "scope": (
                "The congruence theorem covers total product laws computed "
                "only from the operand signatures. The finite table result "
                "covers frozen deck parameters. The birthday envelope "
                "covers one ideal projected two-pair merge that emits "
                "explicit rank rows. None is a lower bound for implicit "
                "summation-polynomial/FFE relation-span operators, circuits, "
                "RAM, cell probes, or all generic-prime ECDLP algorithms."
            ),
        },
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "label_only_composition_negative_admitted": True,
            "explicit_row_birthday_boundary_admitted": True,
            "implicit_relation_span_operator_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
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
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=DEFAULT_CONTROLS,
    )
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
        "obligations="
        f"{admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
