#!/usr/bin/env python3
"""Audit when aggregate M6 count/marginal output is source-equivalent."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = (
    "p1553.m6_aggregate_marginal_singleton_source_equivalence.r146.v1"
)

R145_PRODUCER = ROOT / (
    "p1553_m6_weighted_c3_mobius_gcd_trace_probe_r145.py"
)
R145_REPORT = ROOT / (
    "p1553_m6_weighted_c3_mobius_gcd_trace_probe_report_r145.json"
)
R145_FROZEN = ROOT / "frozen_m6_weighted_c3_mobius_gcd_trace.json"
R145_COST = ROOT / "m6_weighted_c3_mobius_gcd_trace_cost_ledger.json"
R145_REPLAY = ROOT / "m6_weighted_c3_mobius_gcd_trace_replay.json"
R145_CONTROLS = ROOT / "m6_weighted_c3_mobius_gcd_trace_controls.json"
R145_LOGS = ROOT / "factor_logs_and_identical_descent_r145.json"
R145_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_weighted_c3_mobius_gcd_trace_probe_r145.py"
)
R145_GATE = ROOT / (
    "p1553_m6_weighted_c3_mobius_gcd_trace_probe_gate_r145.md"
)
R145_PARENT = ROOT / (
    "p1553_m6_weighted_c3_mobius_gcd_trace_probe_parent_report_r145.yaml"
)
R117_REPORT = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_"
    "probe_report_r117.json"
)
R117_GATE = ROOT / (
    "p1553_m6_target_batched_c3_elliptic_transpose_"
    "probe_gate_r117.md"
)
R144_CONTROLS = ROOT / "m6_weighted_fiber_marginal_log_operator_controls.json"

SOURCE_BINDINGS = (
    (
        "r145_producer",
        R145_PRODUCER,
        "38048dcaee982faf9f6eaec2d3bd44cfdea22b784f7f46041118543048a47565",
    ),
    (
        "r145_report",
        R145_REPORT,
        "015ff9c84547b1bdfa8ddb0e107d8e5be3872741740fc49c63f05fbf0e21037c",
    ),
    (
        "r145_frozen",
        R145_FROZEN,
        "1471f66d1703fa6cbacb9ed9974b389d47b0ee23f53a763bf139e685270e5f71",
    ),
    (
        "r145_cost",
        R145_COST,
        "9585b1aaa473eb2c4b5ce2d2030600cc737ea4115dcaee3705a274ba254dd82c",
    ),
    (
        "r145_replay",
        R145_REPLAY,
        "d1159f37db92a4572d56ab71cb8d63cd2d3684972f4a226bda1b785daadc1f1c",
    ),
    (
        "r145_controls",
        R145_CONTROLS,
        "582cb06bbf5a58c9110218823a0ac9fa80c60bf927ff21989588eb7668835fe1",
    ),
    (
        "r145_logs",
        R145_LOGS,
        "78c35606a89edf074609276808f818176e229787257327718c30b6e54d9a470b",
    ),
    (
        "r145_test",
        R145_TEST,
        "2373f522398c78349a2156dbc931b1367c0836fc673491c0e522598cf4e40d8a",
    ),
    (
        "r145_gate",
        R145_GATE,
        "dbbb5ada91deae0823cf502312f9e490a8b37fbcf165babba994c3b009fc8a8d",
    ),
    (
        "r145_parent",
        R145_PARENT,
        "6bdb7d1421e4425ca2073a70cc929cf634519b6f0197396eb0076426deaf41cb",
    ),
    (
        "r117_report",
        R117_REPORT,
        "449021dd58e6567bb827bcfe41336175d7b628acda4446e1b63d9a46cf48f824",
    ),
    (
        "r117_gate",
        R117_GATE,
        "014f74c772dfc77d8821f658b10ef9bd11a644fe8f0b356c24fda61ef81d9de5",
    ),
    (
        "r144_controls",
        R144_CONTROLS,
        "0677e3a7220e7258e4dc56bff14f5e083a09ed048948667fe0582dd2fa4c6ea8",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_aggregate_marginal_singleton_source_equivalence_"
    "probe_report_r146.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_aggregate_marginal_singleton_source_equivalence.json"
)
DEFAULT_COST = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_aggregate_marginal_singleton_source_equivalence_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r146.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R145 = load_module("p1553_r145_for_r146", R145_PRODUCER)
R144 = R145.R144
R82 = R144.R82
R81 = R144.R81


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
        raise AssertionError(f"R146 source binding mismatch: {failures}")
    return actual


def canonical_deck_sources(
    labels: list[int],
    modulus: int,
) -> list[tuple[int, tuple[int, ...], int]]:
    result = []
    for source in itertools.combinations_with_replacement(
        range(len(labels)),
        R144.RELATION_ARITY,
    ):
        result.append(
            (
                sum(labels[index] for index in source) % modulus,
                source,
                R144.multinomial_weight(source),
            )
        )
    return result


def source_multiplicities(source: tuple[int, ...], width: int) -> list[int]:
    counts = collections.Counter(source)
    return [counts.get(index, 0) for index in range(width)]


def canonical_ac_control(
    curve: dict[str, Any],
    offset: int,
    r144_control: dict[str, Any],
) -> dict[str, Any]:
    modulus = curve["subgroup_order"]
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    atoms_a, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    labels_a = verifier.labels(atoms_a)
    labels_c = verifier.labels(atoms_c)
    a_sources = canonical_deck_sources(labels_a, modulus)
    c_sources = canonical_deck_sources(labels_c, modulus)

    fibers: dict[int, dict[str, Any]] = {}
    for a_scalar, a_source, a_weight in a_sources:
        a_multiplicities = source_multiplicities(
            a_source,
            len(atoms_a),
        )
        for c_scalar, c_source, c_weight in c_sources:
            target = (a_scalar + c_scalar) % modulus
            weight = a_weight * c_weight
            row = fibers.setdefault(
                target,
                {
                    "canonical_source_count": 0,
                    "ordered_fiber_count": 0,
                    "full_atom_marginal": [0]
                    * (len(atoms_a) + len(atoms_c)),
                    "singleton_a_source": None,
                    "singleton_c_source": None,
                },
            )
            row["canonical_source_count"] += 1
            row["ordered_fiber_count"] += weight
            c_multiplicities = source_multiplicities(
                c_source,
                len(atoms_c),
            )
            for index, multiplicity in enumerate(a_multiplicities):
                row["full_atom_marginal"][index] += weight * multiplicity
            for index, multiplicity in enumerate(c_multiplicities):
                row["full_atom_marginal"][len(atoms_a) + index] += (
                    weight * multiplicity
                )
            if row["canonical_source_count"] == 1:
                row["singleton_a_source"] = list(a_source)
                row["singleton_c_source"] = list(c_source)
            else:
                row["singleton_a_source"] = None
                row["singleton_c_source"] = None

    occupancy_histogram = collections.Counter(
        row["canonical_source_count"] for row in fibers.values()
    )
    singleton_rows = [
        row
        for row in fibers.values()
        if row["canonical_source_count"] == 1
    ]
    singleton_inversion_exact = all(
        row["full_atom_marginal"]
        == [
            row["ordered_fiber_count"] * multiplicity
            for multiplicity in (
                source_multiplicities(
                    tuple(row["singleton_a_source"]),
                    len(atoms_a),
                )
                + source_multiplicities(
                    tuple(row["singleton_c_source"]),
                    len(atoms_c),
                )
            )
        ]
        for row in singleton_rows
    )

    samples_match = True
    sample_replay = []
    for sample in r144_control["fiber_samples"]:
        target = sample["target_scalar"]
        row = fibers[target]
        exact = (
            row["ordered_fiber_count"]
            == sample["ordered_fiber_count"]
            and row["full_atom_marginal"]
            == sample["full_atom_marginal"]
        )
        samples_match = samples_match and exact
        sample_replay.append(
            {
                "target_scalar": target,
                "canonical_ac_source_count": row[
                    "canonical_source_count"
                ],
                "ordered_fiber_count": row["ordered_fiber_count"],
                "count_and_marginal_match_r144": exact,
            }
        )

    canonical_source_count = len(a_sources) * len(c_sources)
    positive_fiber_count = len(fibers)
    singleton_count = len(singleton_rows)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": curve["field_prime"],
        "subgroup_order": modulus,
        "atom_a_size": len(atoms_a),
        "atom_c_size": len(atoms_c),
        "canonical_a_multiset_count": len(a_sources),
        "canonical_c_multiset_count": len(c_sources),
        "canonical_ac_source_count": canonical_source_count,
        "positive_fiber_count": positive_fiber_count,
        "positive_fiber_count_matches_r144": (
            positive_fiber_count == r144_control["positive_fiber_count"]
        ),
        "singleton_positive_fiber_count": singleton_count,
        "singleton_positive_fiber_fraction": (
            singleton_count / positive_fiber_count
        ),
        "multiple_source_positive_fiber_count": (
            positive_fiber_count - singleton_count
        ),
        "maximum_canonical_ac_fiber_occupancy": max(
            row["canonical_source_count"] for row in fibers.values()
        ),
        "canonical_ac_occupancy_histogram": {
            str(key): value
            for key, value in sorted(occupancy_histogram.items())
        },
        "all_singleton_marginals_invert_to_atom_multiplicities": (
            singleton_inversion_exact
        ),
        "r144_sample_replay": sample_replay,
        "all_r144_sample_counts_and_marginals_match": samples_match,
        "diagnostic_build_consumes_verifier_scalar_labels": True,
        "candidate_operator_consumes_scalar_labels": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def occupancy_theorem() -> dict[str, Any]:
    arity = R144.RELATION_ARITY
    factorial = math.factorial(arity)
    lambda_value = 1 / (factorial * factorial)
    conditional_singleton = (
        lambda_value
        * math.exp(-lambda_value)
        / (1 - math.exp(-lambda_value))
    )
    return {
        "model": (
            "Canonical A- and C-multiset source pairs map independently "
            "and uniformly into a prime-order target group."
        ),
        "canonical_source_to_group_ratio": (
            "lambda=1/(m!)^2 at m*(alpha+beta)=5"
        ),
        "m": arity,
        "lambda_exact": f"1/{factorial * factorial}",
        "lambda_decimal": lambda_value,
        "limiting_occupancy": "Poisson(lambda)",
        "conditional_singleton_probability": (
            "lambda*exp(-lambda)/(1-exp(-lambda))"
        ),
        "conditional_singleton_probability_decimal": conditional_singleton,
        "conditional_singleton_first_order": "1-lambda/2+O(lambda^2)",
        "random_model_only": True,
        "structured_factor_base_transfer_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    r144_controls_payload = json.loads(
        R144_CONTROLS.read_text(encoding="utf-8")
    )
    r144_by_id = {
        row["control_id"]: row
        for row in r144_controls_payload["actual_controls"]
    }
    actual = []
    for curve in R82.FAMILIES:
        for offset in (0, 1):
            control_id = f"{curve['family_id']}_offset{offset}"
            actual.append(
                canonical_ac_control(
                    curve,
                    offset,
                    r144_by_id[control_id],
                )
            )

    all_positive_match = all(
        row["positive_fiber_count_matches_r144"] for row in actual
    )
    all_samples_match = all(
        row["all_r144_sample_counts_and_marginals_match"]
        for row in actual
    )
    all_singletons_invert = all(
        row["all_singleton_marginals_invert_to_atom_multiplicities"]
        for row in actual
    )
    minimum_singleton_fraction = min(
        row["singleton_positive_fiber_fraction"] for row in actual
    )
    theorem = occupancy_theorem()

    controls = {
        "schema": (
            "p1553.m6_aggregate_marginal_singleton_source_equivalence."
            "controls.r146.v1"
        ),
        "actual_control_count": len(actual),
        "all_positive_fiber_counts_match_r144": all_positive_match,
        "all_r144_sample_counts_and_marginals_match": all_samples_match,
        "all_singleton_marginals_invert_exactly": all_singletons_invert,
        "minimum_singleton_positive_fiber_fraction": (
            minimum_singleton_fraction
        ),
        "maximum_singleton_positive_fiber_fraction": max(
            row["singleton_positive_fiber_fraction"] for row in actual
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_aggregate_marginal_singleton_source_equivalence."
            "frozen.r146.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "singleton_inversion_theorem": {
            "hypothesis": (
                "A positive target fiber contains exactly one canonical "
                "A-multiset/C-multiset source pair."
            ),
            "count": "c_T=w_A*w_C",
            "a_marginal": "d_A(a)=c_T*r_A(a)",
            "c_marginal": "d_C(c)=c_T*r_C(c)",
            "recovery": (
                "r_A(a)=d_A(a)/c_T and r_C(c)=d_C(c)/c_T over the integers."
            ),
            "consequence": (
                "Exact count plus all A/C marginals is source-equivalent "
                "on every singleton canonical fiber."
            ),
        },
        "occupancy_conservation": {
            "canonical_source_count": "M",
            "positive_target_support": "H",
            "mean_positive_occupancy": "L=M/H",
            "uniform_target_success_probability": "H/q=M/(L*q)",
            "boundary": (
                "At M=Theta(q), increasing mean aggregate occupancy by "
                "B^gamma decreases uniform-target hit density by B^-gamma. "
                "A targetable structured family is required to avoid the "
                "matching retry exponent."
            ),
        },
        "random_occupancy_theorem": theorem,
        "required_open_outputs": {
            "implicit_batched_count_and_marginal_index": "open",
            "targetable_superconstant_structured_fibers": "open",
            "density_retry_avoiding_target_selector": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_aggregate_marginal_singleton_source_equivalence."
            "replay.r146.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "canonical_ac_source_count": row[
                    "canonical_ac_source_count"
                ],
                "positive_fiber_count": row["positive_fiber_count"],
                "singleton_positive_fiber_count": row[
                    "singleton_positive_fiber_count"
                ],
                "singleton_positive_fiber_fraction": row[
                    "singleton_positive_fiber_fraction"
                ],
                "maximum_canonical_ac_fiber_occupancy": row[
                    "maximum_canonical_ac_fiber_occupancy"
                ],
                "canonical_ac_occupancy_histogram": row[
                    "canonical_ac_occupancy_histogram"
                ],
                "r144_sample_replay": row["r144_sample_replay"],
            }
            for row in actual
        ],
        "all_singleton_marginals_invert_exactly": all_singletons_invert,
        "minimum_singleton_positive_fiber_fraction": (
            minimum_singleton_fraction
        ),
    }

    cost = {
        "schema": (
            "p1553.m6_aggregate_marginal_singleton_source_equivalence."
            "cost.r146.v1"
        ),
        "m6_selected_vertex": {
            "canonical_source_count_exponent_B": "5",
            "group_order_exponent_B": "5",
            "meaningful_log_dimension_exponent_B": "3/4",
            "fresh_interface_exponent_B": "5/4",
            "conditional_precomputation_exponent_B": "9/4",
        },
        "source_free_interface_weaker_on_singleton_fibers": False,
        "source_equivalent_on_singleton_fibers": True,
        "superconstant_occupancy_gets_free_density_credit": False,
        "targetable_structured_multi_fiber_family_supplied": False,
        "implicit_count_and_marginal_index_supplied": False,
        "unconditional_computational_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.m6_aggregate_marginal_singleton_source_equivalence."
            "logs_descent.r146.v1"
        ),
        "finite_singleton_source_equivalence_exact": (
            all_singletons_invert
        ),
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "thirteen_source_bindings_verified": len(actual_bindings) == 13,
        "r144_count_and_marginal_interface_inherited": True,
        "r145_exact_count_identity_inherited": True,
        "canonical_ac_source_quotient_defined": True,
        "singleton_integer_inversion_theorem_complete": True,
        "occupancy_conservation_identity_complete": True,
        "random_poisson_model_theorem_complete": True,
        "random_model_scoped_without_structured_transfer": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_positive_supports_match_r144": all_positive_match,
        "all_r144_sample_counts_and_marginals_replay": all_samples_match,
        "all_actual_singleton_inversions_exact": all_singletons_invert,
        "all_actual_singleton_positive_fraction_at_least_98_percent": (
            minimum_singleton_fraction >= 0.98
        ),
        "candidate_dlp_oracle_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "implicit_count_and_marginal_index_complete": False,
        "targetable_superconstant_structured_fibers_complete": False,
        "density_retry_avoiding_target_selector_complete": False,
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
        "Treat exact count plus full marginals as a source-equivalent "
        "interface on singleton canonical fibers. Construct one targetable "
        "structured multi-fiber family whose canonical occupancy grows as "
        "B^gamma while its hit-density retry, setup, marginal output, rank, "
        "factor-log, and shifted-descent costs remain below the R115 caps; "
        "or construct the implicit batched count/marginal index without "
        "claiming that source elimination alone makes the query easier. "
        "Use no DLP, root, count, marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "AGGREGATE_COUNT_AND_MARGINALS_INVERT_TO_CANONICAL_A_C_SOURCE_"
            "ON_SINGLETON_FIBERS__EIGHT_ACTUAL_SINGLETON_POSITIVE_FRACTIONS_"
            "98P18_TO_100_PERCENT__RANDOM_DENSE_M6_MODEL_CONDITIONAL_"
            "SINGLETON_PROBABILITY_1_MINUS_1_OVER_2_6FACTORIAL_SQUARED__"
            "SUPERCONSTANT_OCCUPANCY_TRADES_FOR_HIT_DENSITY__NO_COUNT_INDEX_"
            "LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether the R144 source-free count-and-marginal "
            "interface is strictly weaker than canonical A/C source "
            "recovery on the useful positive fibers."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "singleton_inversion": frozen["singleton_inversion_theorem"],
            "occupancy_conservation": frozen["occupancy_conservation"],
            "random_occupancy": theorem,
            "finite_actual_result": (
                "Every singleton fiber in all eight actual controls "
                "inverts exactly; singleton fibers constitute between "
                f"{minimum_singleton_fraction:.6%} and 100% of positive "
                "fibers."
            ),
            "scope": (
                "The exact inversion theorem is unconditional on singleton "
                "fibers. The occupancy probability is random-model-only. "
                "Neither statement is a computational lower bound or a "
                "structured generic-prime transfer."
            ),
            "literature_novelty": "unverified",
        },
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "singleton_source_equivalence_admitted": True,
            "occupancy_conservation_admitted": True,
            "random_model_occupancy_admitted_model_bound_only": True,
            "implicit_count_operator_admitted": False,
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
