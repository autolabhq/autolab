#!/usr/bin/env python3
"""Audit rational convolution-algebra compression of the actual C deck."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import json
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_rational_convolution_subalgebra_rigidity.r150.v1"

R149_PRODUCER = ROOT / "p1553_m6_actual_c6_shift_krylov_rank_probe_r149.py"
R149_REPORT = ROOT / (
    "p1553_m6_actual_c6_shift_krylov_rank_probe_report_r149.json"
)
R149_FROZEN = ROOT / "frozen_m6_actual_c6_shift_krylov_rank.json"
R149_COST = ROOT / "m6_actual_c6_shift_krylov_rank_cost_ledger.json"
R149_REPLAY = ROOT / "m6_actual_c6_shift_krylov_rank_replay.json"
R149_CONTROLS = ROOT / "m6_actual_c6_shift_krylov_rank_controls.json"
R149_LOGS = ROOT / "factor_logs_and_identical_descent_r149.json"
R149_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_actual_c6_shift_krylov_rank_probe_r149.py"
)
R149_GATE = ROOT / "p1553_m6_actual_c6_shift_krylov_rank_probe_gate_r149.md"
R149_PARENT = ROOT / (
    "p1553_m6_actual_c6_shift_krylov_rank_probe_parent_report_r149.yaml"
)

SOURCE_BINDINGS = (
    (
        "r149_producer",
        R149_PRODUCER,
        "0faee973ce1bdc8deb0410eef0b5de429b526210b017fccea1e6ca1b67784ed8",
    ),
    (
        "r149_report",
        R149_REPORT,
        "cadfbb8804e79fb718c00f13bf58970482fe106e2a6fb14d6e8638b0a382d44f",
    ),
    (
        "r149_frozen",
        R149_FROZEN,
        "6edac765d2d426f0345080418bafbda91e3d5c149763e5cfac0cd604bb8ae8e8",
    ),
    (
        "r149_cost",
        R149_COST,
        "9f647991bc492784c62cb07e5ee9bc66ff9516fe8fc602db3f6d3414f7ef4d1e",
    ),
    (
        "r149_replay",
        R149_REPLAY,
        "685f38f0683b6a6767b7c7b4a46055cf60537a9b567f884dd078aa4ee387989e",
    ),
    (
        "r149_controls",
        R149_CONTROLS,
        "ca13a7a2dd3137dfc1bf2c69a48bcc5e58df5951e81943f02244de9c84cb7b00",
    ),
    (
        "r149_logs",
        R149_LOGS,
        "2155cd38959d7117a762d5a4ef90b1fa55e708af536d252dd799cdba8441b4d3",
    ),
    (
        "r149_test",
        R149_TEST,
        "765c319bbacf0c062af77f2ba6677a3a9db862aed9d7a324016fd82866682051",
    ),
    (
        "r149_gate",
        R149_GATE,
        "cee5fccdb2e59bd355e10168bad08e267b29b6a855e0c3ac112d934b1be98398",
    ),
    (
        "r149_parent",
        R149_PARENT,
        "b7240146539bd663851398d3ac21fc293658018baa4162790b229b95ed726d56",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_"
    "probe_report_r150.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_rational_convolution_subalgebra_rigidity.json"
)
DEFAULT_COST = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r150.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R149 = load_module("p1553_r149_for_r150", R149_PRODUCER)
R82 = R149.R82
R81 = R149.R81


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
        raise AssertionError(f"R150 source binding mismatch: {failures}")
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


def cyclic_convolution(
    left: Iterable[int], right: Iterable[int]
) -> tuple[int, ...]:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("cyclic convolution widths differ")
    width = len(left_values)
    output = [0] * width
    for left_index, left_value in enumerate(left_values):
        if left_value == 0:
            continue
        for right_index, right_value in enumerate(right_values):
            if right_value:
                output[(left_index + right_index) % width] += (
                    left_value * right_value
                )
    return tuple(output)


def convolution_power(vector: Iterable[int], exponent: int) -> tuple[int, ...]:
    values = tuple(vector)
    output = (1,) + (0,) * (len(values) - 1)
    base = values
    remaining = exponent
    while remaining:
        if remaining & 1:
            output = cyclic_convolution(output, base)
        remaining >>= 1
        if remaining:
            base = cyclic_convolution(base, base)
    return output


def multiplier_stabilizer(labels: Iterable[int], prime: int) -> tuple[int, ...]:
    deck = frozenset(int(label) % prime for label in labels)
    anchor = next((label for label in deck if label != 0), None)
    if anchor is None:
        return tuple(range(1, prime))
    inverse = pow(anchor, -1, prime)
    candidates = {
        (image * inverse) % prime
        for image in deck
        if image != 0
    }
    return tuple(
        multiplier
        for multiplier in sorted(candidates)
        if frozenset(
            (multiplier * label) % prime for label in deck
        )
        == deck
    )


def shifted_deck_augmentation_witness(
    labels: Iterable[int], prime: int
) -> dict[str, Any]:
    deck = tuple(sorted(int(label) % prime for label in labels))
    counts: collections.Counter[int] = collections.Counter({0: 1})
    atoms = collections.Counter(deck)
    for _ in range(6):
        product: collections.Counter[int] = collections.Counter()
        for left, left_count in counts.items():
            for right, right_count in atoms.items():
                product[(left + right) % prime] += (
                    left_count * right_count
                )
        counts = product
    shift = 1
    displacement = 6 % prime
    candidate_targets = set(counts)
    candidate_targets.update(
        (target + displacement) % prime for target in counts
    )
    for target in sorted(candidate_targets):
        count = counts[target]
        shifted_count = counts[(target - displacement) % prime]
        if count != shifted_count:
            shifted = tuple(
                sorted((label + shift) % prime for label in deck)
            )
            return {
                "shift": shift,
                "sixfold_shift_displacement": displacement,
                "target": target,
                "original_count": count,
                "shifted_deck_count": shifted_count,
                "original_deck_sha256": sha256_json(deck),
                "shifted_deck_sha256": sha256_json(shifted),
                "same_augmentation": len(deck) == len(shifted),
                "different_selected_c6_count": count != shifted_count,
            }
    raise AssertionError("proper deck C6 sequence unexpectedly shift invariant")


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    subgroup_order = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    _, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    labels = tuple(verifier.labels(atoms_c))
    stabilizer = multiplier_stabilizer(labels, subgroup_order)
    stabilizer_order = len(stabilizer)
    orbit_degree = (subgroup_order - 1) // stabilizer_order
    lower_bound_floor = (subgroup_order - 1) // len(labels)
    witness = shifted_deck_augmentation_witness(labels, subgroup_order)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "subgroup_order": subgroup_order,
        "deck_size": len(labels),
        "verifier_labels_sha256": sha256_json(labels),
        "labels_are_distinct": len(set(labels)) == len(labels),
        "proper_nonempty_binary_deck": (
            0 < len(labels) < subgroup_order
            and len(set(labels)) == len(labels)
        ),
        "multiplier_stabilizer": list(stabilizer),
        "multiplier_stabilizer_order": stabilizer_order,
        "stabilizer_order_divides_q_minus_one": (
            (subgroup_order - 1) % stabilizer_order == 0
        ),
        "stabilizer_orbits_cover_nonzero_deck": all(
            (
                frozenset(
                    (multiplier * label) % subgroup_order
                    for multiplier in stabilizer
                )
                <= frozenset(labels)
            )
            for label in labels
            if label != 0
        ),
        "cyclotomic_conjugate_orbit_degree": orbit_degree,
        "orbit_degree_lower_bound_floor_q_minus_one_over_deck_size": (
            lower_bound_floor
        ),
        "orbit_degree_meets_deck_size_bound": (
            Fraction(subgroup_order - 1, stabilizer_order)
            >= Fraction(subgroup_order - 1, len(labels))
        ),
        "actual_generated_subalgebra_dimension_lower_bound": orbit_degree,
        "augmentation_only_collision_witness": witness,
        "augmentation_only_cannot_recover_selected_c6_counts": (
            witness["same_augmentation"]
            and witness["different_selected_c6_count"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_bsgs_labels_receive_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "ambient_algebra": (
            "A=Q[C_q]=Q[X]/(X^q-1), for prime q, with cyclic "
            "convolution as multiplication."
        ),
        "ambient_crt_decomposition": (
            "A is isomorphic to Q direct-product Q(zeta_q), because "
            "X^q-1=(X-1)Phi_q and Phi_q is irreducible over Q."
        ),
        "ambient_quotient_rigidity": (
            "The only unital rational algebra quotient dimensions are "
            "1, q-1, and q. The one-dimensional quotient is augmentation "
            "and discards every nonconstant cyclic component."
        ),
        "deck_element": (
            "U=sum_(a in C) X^a for a proper nonempty binary deck C."
        ),
        "multiplier_stabilizer": (
            "H_C={k in F_q^*: kC=C}."
        ),
        "cyclotomic_conjugate_identity": (
            "U(zeta_q^k)=U(zeta_q^l) iff kC=lC. The forward direction "
            "follows because the reduced binary coefficient difference has "
            "degree below q, is divisible by Phi_q, and has coefficient "
            "sum zero, so it is the zero polynomial."
        ),
        "cyclotomic_degree": (
            "[Q(U(zeta_q)):Q]=(q-1)/|H_C|."
        ),
        "stabilizer_size_bound": (
            "If C contains a nonzero atom, every nonzero H_C orbit in C "
            "has size |H_C|. Hence |H_C|<=|C| and the deck-generated "
            "convolution subalgebra has dimension at least (q-1)/|C|. "
            "The exceptional constant deck C={0} is outside the campaign "
            "factor-base families."
        ),
        "actual_subalgebra_scope": (
            "The bound applies to a reusable characteristic-zero "
            "multiplication-closed representation containing the actual "
            "deck element U. It permits nonlinear convolution products and "
            "arbitrary powers inside that representation."
        ),
        "atom_marginal_scope": (
            "The same representation must carry U^5 and its cyclic shifts "
            "to emit d_a(y)=6[U^5]_(y-a); no smaller augmentation-only "
            "quotient recovers varying selected counts."
        ),
        "excluded_models": (
            "This is not a time lower bound for a bounded-depth circuit "
            "specialized only to U^6 and the fixed marker batch. It does "
            "not cover adaptive RAM or cell-probe structures, bounded "
            "error, nonhomomorphic encodings, or implicit "
            "summation-polynomial/FFE elimination."
        ),
        "novelty_status": "elementary_derivation_novelty_unverified",
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_rational_convolution_subalgebra_rigidity."
            "cost.r150.v1"
        ),
        "group_order_exponent_B": fraction_record(Fraction(5)),
        "c_atom_deck_exponent_B": fraction_record(Fraction(3, 4)),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "full_query_stream_cap_exponent_B": fraction_record(Fraction(5, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "universal_deck_generated_subalgebra_lower_bound_exponent_B": (
            fraction_record(Fraction(17, 4))
        ),
        "trivial_multiplier_stabilizer_dimension_exponent_B": (
            fraction_record(Fraction(5))
        ),
        "universal_subalgebra_bound_inside_setup_cap": False,
        "universal_subalgebra_bound_inside_pollard_rho": False,
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "finite_depth_u6_marker_circuit_supplied": False,
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
    all_decks_valid = all(
        row["labels_are_distinct"]
        and row["proper_nonempty_binary_deck"]
        for row in actual
    )
    all_orbits_exact = all(
        row["stabilizer_order_divides_q_minus_one"]
        and row["stabilizer_orbits_cover_nonzero_deck"]
        and row["orbit_degree_meets_deck_size_bound"]
        for row in actual
    )
    all_augmentation_witnesses = all(
        row["augmentation_only_cannot_recover_selected_c6_counts"]
        for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_rational_convolution_subalgebra_rigidity."
            "controls.r150.v1"
        ),
        "actual_control_count": len(actual),
        "all_actual_decks_valid": all_decks_valid,
        "all_multiplier_orbit_checks_exact": all_orbits_exact,
        "all_augmentation_collision_witnesses_exact": (
            all_augmentation_witnesses
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_rational_convolution_subalgebra_rigidity."
            "frozen.r150.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "finite_depth_target_specialized_u6_circuit": "open",
            "offline_online_transposed_atom_marginals": "open",
            "structured_rank_and_density": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_rational_convolution_subalgebra_rigidity."
            "replay.r150.v1"
        ),
        "controls": actual,
        "all_multiplier_orbit_checks_exact": all_orbits_exact,
        "all_augmentation_collision_witnesses_exact": (
            all_augmentation_witnesses
        ),
        "theorem": theorem,
    }

    logs = {
        "schema": (
            "p1553.m6_rational_convolution_subalgebra_rigidity."
            "logs_descent.r150.v1"
        ),
        "finite_orbit_and_augmentation_controls_exact": (
            all_orbits_exact and all_augmentation_witnesses
        ),
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "r149_linear_scope_inherited_without_overclaim": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_decks_valid": all_decks_valid,
        "all_multiplier_stabilizers_enumerated_exactly": all_orbits_exact,
        "all_cyclotomic_orbit_degrees_meet_bound": all_orbits_exact,
        "all_augmentation_collision_witnesses_exact": (
            all_augmentation_witnesses
        ),
        "prime_cyclic_rational_crt_decomposition_complete": True,
        "ambient_quotient_dimension_rigidity_complete": True,
        "actual_deck_galois_stabilizer_derivation_complete": True,
        "deck_generated_subalgebra_B17_over_4_bound_charged": True,
        "subalgebra_state_exceeds_setup_and_rho_caps": True,
        "candidate_dlp_and_root_oracles_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "finite_depth_target_specialized_u6_circuit_complete": False,
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
        "Do not retain a reusable rational cyclic-convolution quotient or "
        "multiplication-closed subalgebra for U. Construct one bounded-depth "
        "nonhomomorphic circuit specialized to the single power U^6 and the "
        "B^(5/4) fixed marker batch. It must avoid q modes, q target values, "
        "and the B^(9/4) C3 occurrence list, emit exact integer counts and "
        "A/C marginals, and replay rank, factor logs, and identical descent "
        "without a DLP, root, Fourier, recurrence, algebra-state, count, "
        "marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "PRIME_CYCLIC_RATIONAL_CONVOLUTION_ALGEBRA_HAS_NO_SMALL_"
            "NONTRIVIAL_QUOTIENT__ACTUAL_DECK_GENERATED_SUBALGEBRA_"
            "DIMENSION_AT_LEAST_Q_MINUS_1_OVER_C_EQUALS_B17_OVER_4__"
            "REUSABLE_MULTIPLICATION_CLOSED_STATE_OVER_SETUP_AND_RHO__"
            "FINITE_DEPTH_TARGET_SPECIALIZED_U6_CIRCUIT_OPEN__NO_GENERAL_"
            "LOWER_BOUND_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether nonlinear powering in a reusable rational "
            "cyclic-convolution quotient or deck-generated subalgebra "
            "implements the R149 compact-divisor exception."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "rational_convolution_subalgebra_rigidity_admitted": True,
            "reusable_multiplication_closed_route_negative_admitted": True,
            "finite_depth_target_specialized_u6_circuit_admitted": False,
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
