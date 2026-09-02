#!/usr/bin/env python3
"""Audit shift-Krylov compression of the actual sixfold deck convolution."""

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
SCHEMA = "p1553.m6_actual_c6_shift_krylov_rank.r149.v1"

R148_PRODUCER = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_r148.py"
R148_REPORT = ROOT / (
    "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
)
R148_FROZEN = ROOT / "frozen_m6_static_3sum_indexing_tradeoff.json"
R148_COST = ROOT / "m6_static_3sum_indexing_tradeoff_cost_ledger.json"
R148_REPLAY = ROOT / "m6_static_3sum_indexing_tradeoff_replay.json"
R148_CONTROLS = ROOT / "m6_static_3sum_indexing_tradeoff_controls.json"
R148_LOGS = ROOT / "factor_logs_and_identical_descent_r148.json"
R148_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_static_3sum_indexing_tradeoff_probe_r148.py"
)
R148_GATE = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_gate_r148.md"
R148_PARENT = ROOT / (
    "p1553_m6_static_3sum_indexing_tradeoff_probe_parent_report_r148.yaml"
)
R124_PRODUCER = ROOT / "p1553_torus_c5_linear_sketch_circulant_probe_r124.py"
R124_GATE = ROOT / "p1553_torus_c5_linear_sketch_circulant_probe_gate_r124.md"
R124_PARENT = ROOT / (
    "p1553_torus_c5_linear_sketch_circulant_probe_parent_report_r124.yaml"
)

SOURCE_BINDINGS = (
    (
        "r148_producer",
        R148_PRODUCER,
        "415bbb0d00b426855886607d6cb9409dff21f5730dc605079264e0b36c87141f",
    ),
    (
        "r148_report",
        R148_REPORT,
        "b01496572b919ffd15406ee83bcd185675b96669d0cd40f51972ddf56f2caed7",
    ),
    (
        "r148_frozen",
        R148_FROZEN,
        "071b8e2364b6ba0da5bf949b7f0051af25d558b4ea244c6a0aa31bf4f6d1b133",
    ),
    (
        "r148_cost",
        R148_COST,
        "3161ea7199f1200b1a9a6643b2cad20f082e343f9a85ca36235a218df8418e7d",
    ),
    (
        "r148_replay",
        R148_REPLAY,
        "190281a3930976fe162232be997201694955015eea27387691aeb6aab3792511",
    ),
    (
        "r148_controls",
        R148_CONTROLS,
        "3bee12463c7c4966421832f1edb89673aac3f9139151489faf180298ed5db5b0",
    ),
    (
        "r148_logs",
        R148_LOGS,
        "8197c71bfe22bec14b3db5ff5e594d1602229d3b02ad87520a357e5932ec6a85",
    ),
    (
        "r148_test",
        R148_TEST,
        "319b2b006e35a7aa9abf942980527d5172e0ca26ab6e04aab039b7b4612e28dc",
    ),
    (
        "r148_gate",
        R148_GATE,
        "50e465e70d3e57457a64185cd9b86fc9b08bd9ebc56cfe6969f8c1f04508d9ca",
    ),
    (
        "r148_parent",
        R148_PARENT,
        "6cd74b061465a2037edb64872907866d4bad2c1a247b97a9c4e01f6332c26fda",
    ),
    (
        "r124_producer",
        R124_PRODUCER,
        "c3aabfe9d2ce8553c3facdc7fcf2f8b669b09ec48a6144772068f264e4ab012d",
    ),
    (
        "r124_gate",
        R124_GATE,
        "30a3ce3036bab96731430a662cde3ded8e63b9fe1326942a5091148b3e939938",
    ),
    (
        "r124_parent",
        R124_PARENT,
        "eb31ae1bccf81431d27bcef8d6fe27ac3269f077bcf115eecab61526bc7a752a",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_actual_c6_shift_krylov_rank_probe_report_r149.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_actual_c6_shift_krylov_rank.json"
DEFAULT_COST = ROOT / "m6_actual_c6_shift_krylov_rank_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_actual_c6_shift_krylov_rank_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_actual_c6_shift_krylov_rank_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r149.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R148 = load_module("p1553_r148_for_r149", R148_PRODUCER)
R147 = R148.R147
R146 = R147.R146
R82 = R146.R82
R81 = R146.R81
R121 = R147.R141.R121


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


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
        raise AssertionError(f"R149 source binding mismatch: {failures}")
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


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    divisor = 41
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    subgroup_order = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    _, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    labels = verifier.labels(atoms_c)
    field, _, torus_deck = R121.pairing_deck(curve, offset)
    if len(labels) != len(torus_deck):
        raise AssertionError("C deck and torus deck widths differ")

    torus_to_scalar: dict[Any, int] = {}
    torus_counts: collections.Counter[Any] = collections.Counter()
    scalar_counts: collections.Counter[int] = collections.Counter()
    map_consistent = True
    for source in itertools.product(range(len(labels)), repeat=6):
        torus_target = field.product(torus_deck[index] for index in source)
        scalar_target = sum(labels[index] for index in source) % subgroup_order
        previous = torus_to_scalar.setdefault(torus_target, scalar_target)
        map_consistent = map_consistent and previous == scalar_target
        torus_counts[torus_target] += 1
        scalar_counts[scalar_target] += 1

    mapped_counts = collections.Counter(
        {
            torus_to_scalar[target]: multiplicity
            for target, multiplicity in torus_counts.items()
        }
    )
    binary_proper_deck = (
        0 < len(labels) < subgroup_order
        and len(set(labels)) == len(labels)
    )
    count_replay_exact = map_consistent and mapped_counts == scalar_counts
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": int(curve["field_prime"]),
        "subgroup_order": subgroup_order,
        "subgroup_order_is_prime": is_prime(subgroup_order),
        "deck_size": len(labels),
        "verifier_labels_sha256": sha256_json(labels),
        "labels_are_distinct": len(set(labels)) == len(labels),
        "proper_nonempty_binary_deck": binary_proper_deck,
        "ordered_c6_total_multiplicity": sum(scalar_counts.values()),
        "expected_ordered_c6_total_multiplicity": len(labels) ** 6,
        "positive_c6_support_size": len(scalar_counts),
        "torus_target_to_scalar_label_is_consistent": map_consistent,
        "torus_and_scalar_c6_multiplicities_match": count_replay_exact,
        "characteristic_zero_count_fourier_support_theorem_applies": (
            is_prime(subgroup_order) and binary_proper_deck
        ),
        "characteristic_zero_c_atom_marginal_fourier_support_theorem_applies": (
            is_prime(subgroup_order) and binary_proper_deck
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_bsgs_labels_receive_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "deck_polynomial": (
            "U(X)=sum_(a in C) X^a in Q[X]/(X^q-1), with C a proper "
            "nonempty subset of the prime-order cyclic group."
        ),
        "cyclotomic_nonvanishing": (
            "For every qth root zeta^j, U(zeta^j) is nonzero. At j=0 it "
            "equals |C|. At j!=0, a zero would force Phi_q to divide the "
            "proper binary polynomial U, possible only for U=0 or U=Phi_q."
        ),
        "ordered_c6_sequence": "c=u^(*6)",
        "ordered_c6_fourier_transform": (
            "DFT(c)_j=DFT(u)_j^6, hence all q modes are nonzero."
        ),
        "ordered_c6_shift_krylov_dimension": "q",
        "ordered_c6_minimum_cyclic_linear_recurrence_order": "q",
        "atom_marginal_identity": (
            "For atom a, d_a(y)=6*u^(*5)(y-a)."
        ),
        "atom_marginal_fourier_transform": (
            "DFT(d_a)_j=6*zeta^(-aj)*DFT(u)_j^5, hence all q modes are "
            "nonzero in characteristic zero."
        ),
        "atom_marginal_shift_krylov_dimension": "q",
        "proof_model": (
            "Characteristic-zero cyclic shifts, constant-coefficient linear "
            "recurrences, and linear shift-Krylov state."
        ),
        "scope": (
            "This is a full-rank theorem for the actual fixed count and "
            "atom-marginal sequences, not merely a universal input sketch. "
            "It is not a lower bound for nonlinear preprocessing, "
            "target-specialized arithmetic circuits, adaptive RAM or "
            "cell-probe structures, bounded-error algorithms, or implicit "
            "summation-polynomial/FFE operators."
        ),
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_actual_c6_shift_krylov_rank.cost.r149.v1",
        "group_order_exponent_B": fraction_record(Fraction(5)),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "full_query_stream_cap_exponent_B": fraction_record(Fraction(5, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "count_linear_recurrence_state_exponent_B": fraction_record(
            Fraction(5)
        ),
        "shared_atom_marginal_linear_recurrence_state_exponent_B": (
            fraction_record(Fraction(5))
        ),
        "count_recurrence_state_inside_setup_cap": False,
        "marginal_recurrence_state_inside_setup_cap": False,
        "full_target_generation_inside_rho": False,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "nonlinear_compact_divisor_count_circuit_supplied": False,
        "unconditional_computational_lower_bound_claimed": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        actual_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    all_conditions = all(
        row["subgroup_order_is_prime"]
        and row["proper_nonempty_binary_deck"]
        and row["labels_are_distinct"]
        for row in actual
    )
    all_replays = all(
        row["torus_target_to_scalar_label_is_consistent"]
        and row["torus_and_scalar_c6_multiplicities_match"]
        and row["ordered_c6_total_multiplicity"]
        == row["expected_ordered_c6_total_multiplicity"]
        for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": "p1553.m6_actual_c6_shift_krylov_rank.controls.r149.v1",
        "actual_control_count": len(actual),
        "all_prime_order_proper_binary_deck_conditions_hold": all_conditions,
        "all_torus_and_scalar_c6_replays_exact": all_replays,
        "all_count_and_marginal_full_support_theorems_apply": all(
            row[
                "characteristic_zero_count_fourier_support_theorem_applies"
            ]
            and row[
                "characteristic_zero_c_atom_marginal_fourier_support_theorem_applies"
            ]
            for row in actual
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": "p1553.m6_actual_c6_shift_krylov_rank.frozen.r149.v1",
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "nonlinear_compact_divisor_count_circuit": "open",
            "offline_online_transposed_atom_marginals": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": "p1553.m6_actual_c6_shift_krylov_rank.replay.r149.v1",
        "controls": actual,
        "all_torus_and_scalar_c6_replays_exact": all_replays,
        "theorem": theorem,
    }

    logs = {
        "schema": (
            "p1553.m6_actual_c6_shift_krylov_rank."
            "logs_descent.r149.v1"
        ),
        "finite_c6_replays_exact": all_replays,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "thirteen_source_bindings_verified": len(actual_bindings) == 13,
        "r148_structure_aware_exception_inherited": True,
        "r124_linear_scope_inherited_without_overclaim": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_labels_distinct_and_decks_proper": all_conditions,
        "all_torus_to_scalar_maps_consistent": all_replays,
        "all_actual_c6_multiplicities_exact": all_replays,
        "count_fourier_nonvanishing_theorem_complete": True,
        "count_shift_krylov_full_rank_theorem_complete": True,
        "atom_marginal_derivative_identity_complete": True,
        "atom_marginal_shift_krylov_full_rank_theorem_complete": True,
        "group_order_B5_state_charged": True,
        "count_and_marginal_linear_recurrences_exceed_cap": True,
        "candidate_dlp_and_root_oracles_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "nonlinear_compact_divisor_count_circuit_complete": False,
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
        "Do not use a constant-coefficient shift recurrence or linear "
        "Krylov state for u^(*6) or its atom marginals. Construct one "
        "nonlinear target-specialized compact-divisor circuit that evaluates "
        "only the B^(5/4) required markers after B^(9/4) setup, without "
        "forming q modes, q target values, or the C3 occurrence list. It "
        "must emit exact integer counts and A/C marginals and replay rank, "
        "factor logs, and identical descent. It may use no DLP, root, "
        "Fourier, recurrence, count, marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "ACTUAL_FIXED_C6_SEQUENCE_U6_HAS_ALL_Q_RATIONAL_FOURIER_MODES__"
            "SHIFT_KRYLOV_DIMENSION_Q_B5__EVERY_C_ATOM_MARGINAL_6_SHIFT_U5_"
            "ALSO_FULL_Q__CONSTANT_COEFFICIENT_LINEAR_RECURRENCE_STATE_B5_"
            "OVER_CAP__NONLINEAR_TARGET_SPECIALIZED_COMPACT_DIVISOR_CIRCUIT_"
            "OPEN__NO_GENERAL_LOWER_BOUND_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether the actual fixed sixfold deck convolution "
            "and its atom marginals admit a low-order shift recurrence that "
            "implements the structure-aware R148 exception."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "actual_count_and_marginal_full_krylov_rank_admitted": True,
            "linear_shift_recurrence_negative_admitted": True,
            "nonlinear_compact_divisor_circuit_admitted": False,
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
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
