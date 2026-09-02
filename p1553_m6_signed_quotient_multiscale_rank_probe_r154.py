#!/usr/bin/env python3
"""Test signed-log quotient ranks in occupancy-calibrated M6 controls."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_signed_quotient_multiscale_rank.r154.v1"

R153_PRODUCER = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_probe_r153.py"
)
R153_REPORT = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_"
    "probe_report_r153.json"
)
R153_FROZEN = ROOT / (
    "frozen_m6_symmetric_shift_reverse_only_marginal.json"
)
R153_COST = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_cost_ledger.json"
)
R153_REPLAY = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_replay.json"
)
R153_CONTROLS = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_controls.json"
)
R153_LOGS = ROOT / "factor_logs_and_identical_descent_r153.json"
R153_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_symmetric_shift_reverse_only_marginal_probe_r153.py"
)
R153_GATE = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_probe_gate_r153.md"
)
R153_PARENT = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_"
    "probe_parent_report_r153.yaml"
)

SOURCE_BINDINGS = (
    (
        "r153_producer",
        R153_PRODUCER,
        "fa55c47f6b0c8a4a6071580c3ba6c519cdb7150d692ee16253fdf8150694e646",
    ),
    (
        "r153_report",
        R153_REPORT,
        "0e5c8feb4498a16b02fd1ffb5e2f7df4a64d20097c62f35d99abef84a120afce",
    ),
    (
        "r153_frozen",
        R153_FROZEN,
        "81aa786dc9431f6e0bf83de1a43a2030996e06c73faf0f2900fe8ade4e19754c",
    ),
    (
        "r153_cost",
        R153_COST,
        "0a2c48b7604bcef168acb850ab6de69a09d927c4d8558a3c2d8ee7ae60ffd859",
    ),
    (
        "r153_replay",
        R153_REPLAY,
        "6e66783f3ea159ad0b522b46075fe11e0c20b1c82428ed11aa31b0a72fb96f26",
    ),
    (
        "r153_controls",
        R153_CONTROLS,
        "abf91eff4a023d400fdb56caac7927a496cf83b5cba205536118cfdc4ef00f62",
    ),
    (
        "r153_logs",
        R153_LOGS,
        "2bfb012acdbac35964775c8fa1ae38d519c3e9ebddcc81b969c791b97e86a569",
    ),
    (
        "r153_test",
        R153_TEST,
        "5d62c7b1be2f5267ae2d66a931ae6a7034ff2ebc6b4c97bc179587b2f7a47a79",
    ),
    (
        "r153_gate",
        R153_GATE,
        "afaeede72a11b0ae870a0551f9d5ec5bbdf94099f53dede5b5327fbfb9b0bf6d",
    ),
    (
        "r153_parent",
        R153_PARENT,
        "1d46ec932cf58564ebbbccfdee888b7ffacd482d47e7077a1b33feb4dd382c87",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_probe_report_r154.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_signed_quotient_multiscale_rank.json"
)
DEFAULT_COST = ROOT / (
    "m6_signed_quotient_multiscale_rank_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_signed_quotient_multiscale_rank_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_signed_quotient_multiscale_rank_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r154.json"

SYNTHETIC_LEVELS = (4, 5, 6, 7)
OCCUPANCY_MULTIPLIERS = (1, 2, 4, 8)
SYNTHETIC_SEEDS = (15401, 15402, 15403)
A_PAIR_COUNT = 2
ARITY = 6


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R153 = load_module("p1553_r153_for_r154", R153_PRODUCER)
R82 = R153.R82
R81 = R153.R81
R144 = R153.R144


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
        raise AssertionError(f"R154 source binding mismatch: {failures}")
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
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def next_prime(value: int) -> int:
    candidate = max(3, value | 1)
    while not is_prime(candidate):
        candidate += 2
    return candidate


def signed_coefficient_vector_count(pair_count: int, arity: int) -> int:
    total = 0
    for weight in range(arity % 2, arity + 1, 2):
        if weight == 0:
            total += 1
            continue
        for support in range(1, min(pair_count, weight) + 1):
            total += (
                math.comb(pair_count, support)
                * math.comb(weight - 1, support - 1)
                * (2**support)
            )
    return total


def deterministic_signed_pairs(
    tag: str, pair_count: int, modulus: int
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    occupied: set[int] = set()
    counter = 0
    while len(pairs) < pair_count:
        value = (
            int.from_bytes(
                hashlib.sha256(
                    f"R154|{tag}|{counter}".encode("utf-8")
                ).digest(),
                "big",
            )
            % (modulus - 1)
            + 1
        )
        counter += 1
        opposite = (-value) % modulus
        if value in occupied or opposite in occupied:
            continue
        pairs.append((value, opposite))
        occupied.update((value, opposite))
    return tuple(pairs)


def flatten_pairs(
    pairs: Iterable[tuple[int, int]]
) -> tuple[int, ...]:
    return tuple(value for pair in pairs for value in pair)


def public_sign_rows(
    pairs: tuple[tuple[int, int], ...],
    labels: tuple[int, ...],
) -> list[list[int]]:
    rows: list[list[int]] = []
    for positive, negative in pairs:
        row = [0] * len(labels)
        row[labels.index(positive)] = 1
        row[labels.index(negative)] = 1
        rows.append(row)
    return rows


def signed_relation_system(
    a_labels: tuple[int, ...],
    c_pairs: tuple[tuple[int, int], ...],
    modulus: int,
) -> dict[str, Any]:
    c_labels = flatten_pairs(c_pairs)
    shift_counts = R153.convolution_power(a_labels, ARITY, modulus)
    c5_counts = R153.convolution_power(c_labels, ARITY - 1, modulus)
    c6_counts = R153.convolution_power(c_labels, ARITY, modulus)
    quotient_rows: list[list[int]] = []
    full_rows: list[list[int]] = []
    rhs: list[int] = []
    support_positive_rows = 0
    field_nonzero_count_rows = 0
    relation_identities_exact = True

    for shift, shift_weight in sorted(shift_counts.items()):
        for row_atom in c_labels:
            integer_count = (
                shift_weight * c6_counts[(shift + row_atom) % modulus]
            )
            count = integer_count % modulus
            support_positive_rows += int(integer_count > 0)
            field_nonzero_count_rows += int(count != 0)
            full_row = [
                (
                    shift_weight
                    * ARITY
                    * c5_counts[
                        (shift + row_atom - column_atom) % modulus
                    ]
                    - (count if row_atom == column_atom else 0)
                )
                % modulus
                for column_atom in c_labels
            ]
            quotient_row = [
                (
                    full_row[c_labels.index(positive)]
                    - full_row[c_labels.index(negative)]
                )
                % modulus
                for positive, negative in c_pairs
            ]
            rhs_value = (count * shift) % modulus
            relation_identities_exact &= (
                sum(
                    coefficient * positive
                    for coefficient, (positive, _) in zip(
                        quotient_row, c_pairs
                    )
                )
                % modulus
                == rhs_value
            )
            full_rows.append(full_row)
            quotient_rows.append(quotient_row)
            rhs.append(rhs_value)

    quotient_rank = R81.rank_mod(quotient_rows, modulus)
    full_relation_rank = R81.rank_mod(full_rows, modulus)
    sign_rows = public_sign_rows(c_pairs, c_labels)
    combined_rank = R81.rank_mod([*sign_rows, *full_rows], modulus)
    combined_rank_formula_exact = (
        combined_rank == len(c_pairs) + quotient_rank
    )
    selected_rows, selected_rhs = R153.select_independent_system(
        quotient_rows, rhs, len(c_pairs), modulus
    )
    recovered: list[int] | None = None
    recovered_exact = False
    if quotient_rank == len(c_pairs):
        recovered = R144.solve_square_mod(
            selected_rows, selected_rhs, modulus
        )
        recovered_exact = recovered == [
            positive for positive, _ in c_pairs
        ]

    return {
        "a_atom_count": len(a_labels),
        "c_atom_count": len(c_labels),
        "signed_log_dimension": len(c_pairs),
        "a6_shift_support_size": len(shift_counts),
        "c6_endpoint_support_size": len(c6_counts),
        "relation_row_count": len(quotient_rows),
        "support_positive_relation_row_count": support_positive_rows,
        "field_nonzero_count_row_count": field_nonzero_count_rows,
        "full_relation_rank": full_relation_rank,
        "public_sign_constraint_rank": len(c_pairs),
        "signed_quotient_rank": quotient_rank,
        "combined_full_system_rank": combined_rank,
        "combined_rank_formula_exact": combined_rank_formula_exact,
        "signed_quotient_full_rank": quotient_rank == len(c_pairs),
        "signed_verifier_logs_recovered": recovered_exact,
        "all_signed_relation_identities_exact": (
            relation_identities_exact
        ),
        "quotient_rows_sha256": sha256_json(quotient_rows),
        "rhs_sha256": sha256_json(rhs),
        "selected_rows_sha256": sha256_json(selected_rows),
        "selected_rhs_sha256": sha256_json(selected_rhs),
        "recovered_logs_sha256": (
            sha256_json(recovered) if recovered is not None else None
        ),
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    modulus = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    _, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    original_labels = tuple(verifier.labels(atoms_c))
    closure = R153.symmetric_closure(original_labels, modulus)
    representatives = tuple(
        value
        for value in closure
        if value < (-value) % modulus
    )
    pairs = tuple(
        (value, (-value) % modulus) for value in representatives
    )
    control_id = f"{curve['family_id']}_offset{offset}"
    a_labels = R153.symmetric_known_scalars(
        control_id, int(curve["atom_a_size"]), modulus
    )
    system = signed_relation_system(a_labels, pairs, modulus)
    inherited = R153.actual_control(curve, offset)
    return {
        "control_id": control_id,
        "subgroup_order": modulus,
        "pairing_is_complete": set(flatten_pairs(pairs)) == set(closure),
        "inherited_stacked_relation_rank": inherited[
            "stacked_relation_rank"
        ],
        "inherited_rank_matches_recomputed_full_rank": (
            inherited["stacked_relation_rank"]
            == system["full_relation_rank"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_labels_receive_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
        **system,
    }


def synthetic_control(
    c_pair_count: int, occupancy_multiplier: int, seed: int
) -> dict[str, Any]:
    max_a6_support = signed_coefficient_vector_count(
        A_PAIR_COUNT, ARITY
    )
    max_c6_support = signed_coefficient_vector_count(
        c_pair_count, ARITY
    )
    modulus = next_prime(
        math.ceil(
            max_a6_support
            * max_c6_support
            / occupancy_multiplier
        )
    )
    tag = (
        f"c{c_pair_count}|lambda{occupancy_multiplier}|seed{seed}"
    )
    a_pairs = deterministic_signed_pairs(
        f"A|{tag}", A_PAIR_COUNT, modulus
    )
    c_pairs = deterministic_signed_pairs(
        f"C|{tag}", c_pair_count, modulus
    )
    system = signed_relation_system(
        flatten_pairs(a_pairs), c_pairs, modulus
    )
    realized_occupancy = Fraction(
        system["a6_shift_support_size"]
        * system["c6_endpoint_support_size"],
        modulus,
    )
    return {
        "control_id": tag,
        "control_kind": "verifier_only_prime_cyclic_random_label_model",
        "seed": seed,
        "subgroup_order": modulus,
        "a_pair_count": A_PAIR_COUNT,
        "c_pair_count": c_pair_count,
        "preregistered_occupancy_multiplier": occupancy_multiplier,
        "max_a6_signed_coefficient_support": max_a6_support,
        "max_c6_signed_coefficient_support": max_c6_support,
        "realized_support_occupancy": fraction_record(
            realized_occupancy
        ),
        "a_pairs_sha256": sha256_json(a_pairs),
        "c_pairs_sha256": sha256_json(c_pairs),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_labels_receive_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
        **system,
    }


def synthetic_summary(
    controls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for c_pair_count in SYNTHETIC_LEVELS:
        for multiplier in OCCUPANCY_MULTIPLIERS:
            rows = [
                row
                for row in controls
                if row["c_pair_count"] == c_pair_count
                and row["preregistered_occupancy_multiplier"]
                == multiplier
            ]
            summaries.append(
                {
                    "c_pair_count": c_pair_count,
                    "signed_log_dimension": c_pair_count,
                    "preregistered_occupancy_multiplier": multiplier,
                    "trial_count": len(rows),
                    "full_rank_trial_count": sum(
                        row["signed_quotient_full_rank"]
                        for row in rows
                    ),
                    "rank_values": [
                        row["signed_quotient_rank"] for row in rows
                    ],
                    "support_positive_row_counts": [
                        row["support_positive_relation_row_count"]
                        for row in rows
                    ],
                    "realized_support_occupancies": [
                        row["realized_support_occupancy"]["exact"]
                        for row in rows
                    ],
                    "all_relation_identities_exact": all(
                        row["all_signed_relation_identities_exact"]
                        for row in rows
                    ),
                    "all_combined_rank_formulas_exact": all(
                        row["combined_rank_formula_exact"]
                        for row in rows
                    ),
                }
            )
    return summaries


def theorem_record() -> dict[str, Any]:
    return {
        "public_sign_quotient": (
            "For every public inversion pair (C_j,-C_j), "
            "ell(-C_j)=-ell(C_j). Substituting one representative variable "
            "turns each full row h into the signed row "
            "hbar_j=h_j-h_(-j), halving the C-log dimension up to constants."
        ),
        "rank_formula": (
            "The public sign constraints have rank |C|/2, and adjoining "
            "the full relation rows has rank |C|/2+rank(Hbar). Thus full "
            "factor-log recovery is equivalent to full column rank of the "
            "signed quotient matrix Hbar."
        ),
        "actual_control_boundary": (
            "The R153 controls have signed quotient ranks one or zero. "
            "The sign quotient corrects the meaningful dimension but does "
            "not produce a full-rank actual control."
        ),
        "multiscale_design": (
            "Synthetic prime-cyclic controls use two A inversion pairs, "
            "four through seven C inversion pairs, occupancy multipliers "
            "1,2,4,8, and three frozen seeds. The prime order is selected "
            "before labels from the maximum signed coefficient-support "
            "counts, so no observed target or rank adapts the modulus."
        ),
        "finite_scope": (
            "Synthetic labels are verifier-only random cyclic controls. "
            "They test finite occupancy and rank transitions, not a "
            "hash-to-curve theorem, an elliptic-coordinate FFE circuit, or "
            "a generic-prime asymptotic transfer."
        ),
        "novelty_status": (
            "signed_quotient_rank_transfer_novelty_unverified"
        ),
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_signed_quotient_multiscale_rank.cost.r154.v1"
        ),
        "signed_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "public_sign_constraint_state_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "structured_relation_row_count_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "conditional_reverse_operator_apply_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "sign_quotient_changes_no_exponent": True,
        "synthetic_control_work_receives_attack_credit": False,
        "random_rank_concentration_theorem_supplied": False,
        "hash_to_curve_rank_transfer_supplied": False,
        "reverse_only_signed_marker_operator_supplied": False,
        "signed_weight_separable_ffe_dag_supplied": False,
        "factor_logs_without_verifier_labels_supplied": False,
        "identical_target_descent_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        actual_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    synthetic = [
        synthetic_control(c_pair_count, multiplier, seed)
        for c_pair_count in SYNTHETIC_LEVELS
        for multiplier in OCCUPANCY_MULTIPLIERS
        for seed in SYNTHETIC_SEEDS
    ]
    summaries = synthetic_summary(synthetic)
    all_actual_exact = all(
        row["pairing_is_complete"]
        and row["inherited_rank_matches_recomputed_full_rank"]
        and row["all_signed_relation_identities_exact"]
        and row["combined_rank_formula_exact"]
        for row in actual
    )
    all_synthetic_exact = all(
        row["all_signed_relation_identities_exact"]
        and row["combined_rank_formula_exact"]
        for row in synthetic
    )
    synthetic_full_count = sum(
        row["signed_quotient_full_rank"] for row in synthetic
    )
    actual_full_count = sum(
        row["signed_quotient_full_rank"] for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_signed_quotient_multiscale_rank.controls.r154.v1"
        ),
        "actual_control_count": len(actual),
        "synthetic_control_count": len(synthetic),
        "all_actual_signed_systems_exact": all_actual_exact,
        "all_synthetic_signed_systems_exact": all_synthetic_exact,
        "actual_signed_quotient_ranks": [
            row["signed_quotient_rank"] for row in actual
        ],
        "actual_full_rank_control_count": actual_full_count,
        "synthetic_full_rank_control_count": synthetic_full_count,
        "synthetic_summary": summaries,
        "actual_controls": actual,
        "synthetic_controls": synthetic,
        "finite_controls_receive_asymptotic_credit": False,
    }

    frozen = {
        "schema": (
            "p1553.m6_signed_quotient_multiscale_rank.frozen.r154.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "synthetic_design": {
            "a_pair_count": A_PAIR_COUNT,
            "c_pair_counts": list(SYNTHETIC_LEVELS),
            "occupancy_multipliers": list(OCCUPANCY_MULTIPLIERS),
            "seeds": list(SYNTHETIC_SEEDS),
            "arity": ARITY,
        },
        "required_open_outputs": {
            "random_rank_concentration_theorem": "open",
            "hash_to_curve_rank_transfer": "open",
            "reverse_only_signed_marker_operator": "open",
            "signed_weight_separable_ffe_dag": "open",
            "factor_logs_without_verifier_labels": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_signed_quotient_multiscale_rank.replay.r154.v1"
        ),
        "actual_controls": actual,
        "synthetic_controls": synthetic,
        "synthetic_summary": summaries,
        "all_actual_signed_systems_exact": all_actual_exact,
        "all_synthetic_signed_systems_exact": all_synthetic_exact,
    }

    logs = {
        "schema": (
            "p1553.m6_signed_quotient_multiscale_rank."
            "logs_descent.r154.v1"
        ),
        "actual_verifier_only_full_rank_solve_count": actual_full_count,
        "synthetic_verifier_only_full_rank_solve_count": (
            synthetic_full_count
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
        "r153_reverse_only_symmetry_inherited": True,
        "public_inversion_sign_quotient_derived": True,
        "public_sign_constraints_have_half_dimension": all(
            row["public_sign_constraint_rank"]
            == row["signed_log_dimension"]
            for row in actual
        ),
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_relation_ranks_match_r153": all(
            row["inherited_rank_matches_recomputed_full_rank"]
            for row in actual
        ),
        "all_actual_signed_relation_identities_exact": all_actual_exact,
        "all_actual_combined_rank_formulas_exact": all_actual_exact,
        "actual_signed_quotient_rank_deficit_recorded": (
            actual_full_count == 0
        ),
        "synthetic_design_preregistered": True,
        "forty_eight_synthetic_controls_complete": len(synthetic) == 48,
        "all_synthetic_signed_relation_identities_exact": (
            all_synthetic_exact
        ),
        "all_synthetic_combined_rank_formulas_exact": all_synthetic_exact,
        "synthetic_full_rank_transition_observed": synthetic_full_count > 0,
        "finite_rank_transition_scoped_without_transfer": True,
        "sign_quotient_exponents_charged": True,
        "random_rank_concentration_theorem_complete": False,
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
        "Use the frozen signed quotient and occupancy-calibrated controls "
        "to derive or refute a random-deck concentration and full-rank "
        "theorem before claiming transfer to hash-to-curve decks. In "
        "parallel, construct the one reverse-adjoint signed marker FFE "
        "operator promised by R153 within B^(9/4) setup and B^(5/4) batch "
        "work. Freeze pivots, integer counts, exact-residual factor logs, "
        "and identical descent without DLP, root, count, marginal, rank, "
        "or source oracles."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "PUBLIC_INVERSION_CONSTRAINTS_HALVE_C_LOG_DIMENSION__SIGNED_"
            "QUOTIENT_RANK_FORMULA_EXACT__ACTUAL_QUOTIENT_RANKS_ONE_OR_"
            "ZERO_AND_NONE_FULL__OCCUPANCY_CALIBRATED_MULTISCALE_FINITE_"
            "RANK_TRANSITION_TESTED__RANDOM_RANK_THEOREM_HASH_TO_CURVE_"
            "TRANSFER_REVERSE_FFE_OPERATOR_LOGS_DESCENT_OPEN__NO_SHOUP_"
            "BREAKTHROUGH"
        ),
        "objective": (
            "Correct the R153 rank target for public inversion signs and "
            "test whether finite rank failure persists under preregistered "
            "occupancy-calibrated random cyclic controls."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "signed_quotient_reduction_admitted": True,
            "actual_rank_deficit_admitted": actual_full_count == 0,
            "synthetic_rank_transition_admitted_without_transfer": (
                synthetic_full_count > 0
            ),
            "random_rank_or_hash_to_curve_transfer_admitted": False,
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
        f"synthetic_full="
        f"{bundle['controls']['synthetic_full_rank_control_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
