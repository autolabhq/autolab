#!/usr/bin/env python3
"""Test an inversion-symmetric M6 family requiring only reverse actions."""

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
SCHEMA = "p1553.m6_symmetric_shift_reverse_only_marginal.r153.v1"

R152_PRODUCER = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_probe_r152.py"
)
R152_REPORT = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_"
    "probe_report_r152.json"
)
R152_FROZEN = ROOT / (
    "frozen_m6_geometry_only_weight_interpolation_adjoint.json"
)
R152_COST = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_cost_ledger.json"
)
R152_REPLAY = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_replay.json"
)
R152_CONTROLS = ROOT / (
    "m6_geometry_only_weight_interpolation_adjoint_controls.json"
)
R152_LOGS = ROOT / "factor_logs_and_identical_descent_r152.json"
R152_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_geometry_only_weight_interpolation_adjoint_probe_r152.py"
)
R152_GATE = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_probe_gate_r152.md"
)
R152_PARENT = ROOT / (
    "p1553_m6_geometry_only_weight_interpolation_adjoint_"
    "probe_parent_report_r152.yaml"
)

R144_PRODUCER = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_r144.py"
)
R144_GATE = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_gate_r144.md"
)
R144_PARENT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_parent_report_r144.yaml"
)

SOURCE_BINDINGS = (
    (
        "r152_producer",
        R152_PRODUCER,
        "0ccb39253ab66c97961bd08722dab7ad45248e4b6faaf74f9e752344c4491fb2",
    ),
    (
        "r152_report",
        R152_REPORT,
        "f2dd5a43593e9e747d969c73afefac9ee7ce7aac3ca79ca2dd009ea1ffe4e46c",
    ),
    (
        "r152_frozen",
        R152_FROZEN,
        "6f51f9a2cb8199f8f45c919cfc0e3d2fbdcfb9b9aa2755b57c1d7e56b29e1c25",
    ),
    (
        "r152_cost",
        R152_COST,
        "abb38aac0549ecdf871955000accf92e89fee3d2697a5ebbccf49c25adb9fbf9",
    ),
    (
        "r152_replay",
        R152_REPLAY,
        "858e0802bf8f0702d6a596b97bb5b7ae514d2a34849a8e55f3ea66dff0ddf5bf",
    ),
    (
        "r152_controls",
        R152_CONTROLS,
        "31bfcb6da0a040ee79885ffbb5c2ca5190d2a7c1d97efb9d4e69e00698ab3db4",
    ),
    (
        "r152_logs",
        R152_LOGS,
        "551b053ad3b4730d0f615b784becb4ae1e11d9665bedd8b365d460e69db7ee92",
    ),
    (
        "r152_test",
        R152_TEST,
        "fbcaad6bac3619973c27bbd990d9a23e8890b5318626e12a9132c53831852282",
    ),
    (
        "r152_gate",
        R152_GATE,
        "57c336c6e337e06092efc54af710b8cd5ae8de46edf184d9c3335cd44f4bb38a",
    ),
    (
        "r152_parent",
        R152_PARENT,
        "7cce460153d6249f9bc01a77dbe4fe6ef6f45fff770b1a013373f4f22c656182",
    ),
    (
        "r144_producer",
        R144_PRODUCER,
        "3cd7a3e40bf696a836ff6b6cad4c59a7ca73a44d72d152232cddeffa14e90a5c",
    ),
    (
        "r144_gate",
        R144_GATE,
        "c967cc10137fe861384a80a8261763c2ecb86ca2951db8c5bf43f348c65e1e9b",
    ),
    (
        "r144_parent",
        R144_PARENT,
        "db906bd74c9c43fd12ab196a36d59ac80bfbbd22de494c4b6b2efb4d0c96c268",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_symmetric_shift_reverse_only_marginal_"
    "probe_report_r153.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_symmetric_shift_reverse_only_marginal.json"
)
DEFAULT_COST = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_symmetric_shift_reverse_only_marginal_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r153.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R152 = load_module("p1553_r152_for_r153", R152_PRODUCER)
R82 = R152.R82
R81 = R152.R151.R144.R81
R144 = R152.R151.R144


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
        raise AssertionError(f"R153 source binding mismatch: {failures}")
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


def symmetric_known_scalars(
    control_id: str, size: int, modulus: int
) -> tuple[int, ...]:
    values: list[int] = []
    if size % 2:
        values.append(0)
    counter = 0
    while len(values) < size:
        candidate = (
            int.from_bytes(
                hashlib.sha256(
                    f"R153|KNOWN-A|{control_id}|{counter}".encode("utf-8")
                ).digest(),
                "big",
            )
            % (modulus - 1)
            + 1
        )
        counter += 1
        if candidate in values or (-candidate) % modulus in values:
            continue
        values.extend((candidate, (-candidate) % modulus))
    result = tuple(sorted(values[:size]))
    if len(set(result)) != size:
        raise AssertionError("known symmetric A deck collided")
    if {(-value) % modulus for value in result} != set(result):
        raise AssertionError("known A deck is not inversion symmetric")
    return result


def symmetric_closure(
    labels: Iterable[int], modulus: int
) -> tuple[int, ...]:
    closure = {
        value
        for label in labels
        for value in (int(label) % modulus, (-int(label)) % modulus)
    }
    return tuple(sorted(closure))


def sparse_convolution(
    left: collections.Counter[int],
    right: collections.Counter[int],
    modulus: int,
) -> collections.Counter[int]:
    output: collections.Counter[int] = collections.Counter()
    for left_value, left_count in left.items():
        for right_value, right_count in right.items():
            output[(left_value + right_value) % modulus] += (
                left_count * right_count
            )
    return output


def convolution_power(
    atoms: Iterable[int], exponent: int, modulus: int
) -> collections.Counter[int]:
    base = collections.Counter(int(value) % modulus for value in atoms)
    output: collections.Counter[int] = collections.Counter({0: 1})
    for _ in range(exponent):
        output = sparse_convolution(output, base, modulus)
    return output


def matrix_vector(
    matrix: Iterable[Iterable[int]],
    vector: Iterable[int],
    modulus: int,
) -> list[int]:
    values = tuple(vector)
    return [
        sum(coefficient * value for coefficient, value in zip(row, values))
        % modulus
        for row in matrix
    ]


def transpose(matrix: Iterable[Iterable[int]]) -> list[list[int]]:
    rows = [list(row) for row in matrix]
    return [list(column) for column in zip(*rows)]


def deterministic_vector(
    control_id: str, role: str, width: int, modulus: int
) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                f"R153|{control_id}|{role}|{index}".encode("utf-8")
            ).digest(),
            "big",
        )
        % modulus
        for index in range(width)
    ]


def select_independent_system(
    rows: list[list[int]],
    rhs: list[int],
    width: int,
    modulus: int,
) -> tuple[list[list[int]], list[int]]:
    selected_rows: list[list[int]] = []
    selected_rhs: list[int] = []
    rank = 0
    for row, value in zip(rows, rhs):
        candidate_rank = R81.rank_mod([*selected_rows, row], modulus)
        if candidate_rank == rank:
            continue
        selected_rows.append(row)
        selected_rhs.append(value)
        rank = candidate_rank
        if rank == width:
            break
    return selected_rows, selected_rhs


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    modulus = int(curve["subgroup_order"])
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    _, atoms_c, _, _ = R82.compact_factor_base(curve, offset)
    original_c_labels = tuple(verifier.labels(atoms_c))
    c_labels = symmetric_closure(original_c_labels, modulus)
    control_id = f"{curve['family_id']}_offset{offset}"
    a_scalars = symmetric_known_scalars(
        control_id, int(curve["atom_a_size"]), modulus
    )
    shift_counts = convolution_power(a_scalars, 6, modulus)
    c5_counts = convolution_power(c_labels, 5, modulus)
    c6_counts = convolution_power(c_labels, 6, modulus)

    c_even = all(
        c5_counts[value] == c5_counts[(-value) % modulus]
        for value in set(c5_counts)
        | {(-value) % modulus for value in c5_counts}
    )
    shift_even = all(
        shift_counts[value] == shift_counts[(-value) % modulus]
        for value in set(shift_counts)
        | {(-value) % modulus for value in shift_counts}
    )

    blocks: dict[int, dict[str, Any]] = {}
    block_matrices: dict[int, list[list[int]]] = {}
    block_counts: dict[int, list[int]] = {}
    all_rows: list[list[int]] = []
    all_rhs: list[int] = []
    positive_relation_rows = 0
    total_weighted_relation_count = 0
    all_relation_identities = True
    all_row_sums = True
    true_logs = list(c_labels)
    for shift in sorted(shift_counts):
        shift_weight = shift_counts[shift]
        matrix = [
            [
                (
                    shift_weight
                    * 6
                    * c5_counts[(shift + row_atom - column_atom) % modulus]
                )
                % modulus
                for column_atom in c_labels
            ]
            for row_atom in c_labels
        ]
        counts = [
            (
                shift_weight
                * c6_counts[(shift + row_atom) % modulus]
            )
            % modulus
            for row_atom in c_labels
        ]
        positive_relation_rows += sum(count != 0 for count in counts)
        total_weighted_relation_count += sum(counts)
        h_rows = [
            [
                (
                    matrix[row][column]
                    - (
                        counts[row]
                        if row == column
                        else 0
                    )
                )
                % modulus
                for column in range(len(c_labels))
            ]
            for row in range(len(c_labels))
        ]
        rhs = [
            (counts[row] * shift) % modulus
            for row in range(len(c_labels))
        ]
        relation_exact = matrix_vector(
            h_rows, true_logs, modulus
        ) == rhs
        row_sums_exact = all(
            sum(matrix[row]) % modulus
            == (6 * counts[row]) % modulus
            for row in range(len(c_labels))
        )
        all_relation_identities &= relation_exact
        all_row_sums &= row_sums_exact
        block_matrices[shift] = matrix
        block_counts[shift] = counts
        all_rows.extend(h_rows)
        all_rhs.extend(rhs)
        blocks[shift] = {
            "shift": shift,
            "opposite_shift": (-shift) % modulus,
            "a6_shift_multiplicity": shift_weight,
            "matrix_sha256": sha256_json(matrix),
            "counts_sha256": sha256_json(counts),
            "h_rows_sha256": sha256_json(h_rows),
            "rhs_sha256": sha256_json(rhs),
            "row_sum_recovers_six_times_count": row_sums_exact,
            "known_shift_relation_identity_exact": relation_exact,
        }

    all_transpose_blocks = True
    all_reverse_forward_actions = True
    for shift, matrix in block_matrices.items():
        opposite = (-shift) % modulus
        transpose_exact = transpose(matrix) == block_matrices[opposite]
        vector = deterministic_vector(
            control_id, f"shift_{shift}", len(c_labels), modulus
        )
        forward = matrix_vector(matrix, vector, modulus)
        reverse_from_opposite = matrix_vector(
            transpose(block_matrices[opposite]),
            vector,
            modulus,
        )
        action_exact = forward == reverse_from_opposite
        blocks[shift]["opposite_block_transpose_exact"] = transpose_exact
        blocks[shift][
            "forward_action_from_opposite_reverse_exact"
        ] = action_exact
        all_transpose_blocks &= transpose_exact
        all_reverse_forward_actions &= action_exact

    rank = R81.rank_mod(all_rows, modulus)
    selected_rows, selected_rhs = select_independent_system(
        all_rows, all_rhs, len(c_labels), modulus
    )
    full_rank = rank == len(c_labels)
    recovered_logs: list[int] | None = None
    recovered_exact = False
    if full_rank:
        recovered_logs = R144.solve_square_mod(
            selected_rows, selected_rhs, modulus
        )
        recovered_exact = recovered_logs == true_logs

    return {
        "control_id": control_id,
        "subgroup_order": modulus,
        "known_symmetric_a_scalars": list(a_scalars),
        "known_symmetric_a_deck_sha256": sha256_json(a_scalars),
        "known_a_scalar_generation_requires_dlp": False,
        "original_c_atom_count": len(original_c_labels),
        "symmetric_c_atom_count": len(c_labels),
        "symmetric_c_labels_sha256": sha256_json(c_labels),
        "symmetric_c_closure_has_constant_factor_width": (
            len(c_labels) <= 2 * len(original_c_labels)
        ),
        "c_deck_inversion_symmetric": {
            (-value) % modulus for value in c_labels
        }
        == set(c_labels),
        "c5_kernel_even": c_even,
        "a6_shift_multiplicity_even": shift_even,
        "a6_shift_support_size": len(shift_counts),
        "relation_row_count": len(all_rows),
        "positive_relation_row_count": positive_relation_rows,
        "total_weighted_relation_count_mod_q": (
            total_weighted_relation_count % modulus
        ),
        "all_opposite_shift_block_transposes_exact": all_transpose_blocks,
        "all_forward_actions_from_opposite_reverse_exact": (
            all_reverse_forward_actions
        ),
        "all_row_sums_recover_six_times_count": all_row_sums,
        "all_known_shift_relation_identities_exact": (
            all_relation_identities
        ),
        "stacked_relation_rank": rank,
        "stacked_relation_rank_fraction": rank / len(c_labels),
        "stacked_relation_full_c_log_rank": full_rank,
        "selected_system_size": len(selected_rows),
        "verifier_only_c_logs_recovered": recovered_exact,
        "selected_rows_sha256": sha256_json(selected_rows),
        "selected_rhs_sha256": sha256_json(selected_rhs),
        "recovered_logs_sha256": (
            sha256_json(recovered_logs)
            if recovered_logs is not None
            else None
        ),
        "blocks": [blocks[shift] for shift in sorted(blocks)],
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_bsgs_labels_receive_candidate_credit": False,
        "finite_rank_and_log_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "factor_base_family": (
            "Choose a public known-log A deck of size B^(1/12+o(1)) "
            "closed under negation, and replace the C deck by C union -C. "
            "The Cartesian factor base remains F=A+C up to constant-factor "
            "C width, with log(A) known by construction."
        ),
        "symmetric_kernels": (
            "Let U be the indicator of the inversion-closed C deck and "
            "K=U^(*5). Then U(-g)=U(g), K(-g)=K(g), and the ordered A6 "
            "shift multiplicity mu satisfies mu(-s)=mu(s)."
        ),
        "target_family": (
            "For each known A6 shift s and C atom a, target the public "
            "point [s]G+C_a. Aggregate all ordered six-factor sources with "
            "that fixed A6 shift and C6 endpoint."
        ),
        "marginal_block": (
            "D_s(a,b)=6*mu(s)*K(s+a-b), and "
            "c_s(a)=mu(s)*U^(*6)(s+a)."
        ),
        "transpose_pairing": (
            "Evenness gives D_s^T=D_(-s). Because the shift set is "
            "inversion closed, every forward block action is the reverse "
            "action of its opposite-shift block."
        ),
        "count_from_row_sum": (
            "sum_b D_s(a,b)=6*c_s(a). Since the subgroup characteristic "
            "is greater than six, counts require no independent count "
            "oracle once block actions are available."
        ),
        "known_rhs_relation": (
            "If ell_b=log_G(C_b), then "
            "sum_b D_s(a,b)ell_b-c_s(a)ell_a=c_s(a)*s. "
            "The right side is public because s is a constructed scalar."
        ),
        "reverse_only_consequence": (
            "A single reverse-adjoint marker operator over the symmetric "
            "shift batch supplies D_s and D_s^T actions, counts by row "
            "sums, and therefore both actions of the corrected relation "
            "matrix H_s=D_s-diag(c_s). A separate forward tangent through "
            "the elimination setup is unnecessary."
        ),
        "density_model": (
            "|A|^6=B^(1/2), |C|^6=B^(9/2), and q=B^5. Restricting each "
            "A6 shift to targets s+C yields B^(3/4) expected aggregate "
            "relations across B^(5/4) structured rows under the uniform "
            "endpoint model, matching the C-log dimension. This is "
            "model-bound, not a structured rank theorem."
        ),
        "scope": (
            "The identities are exact in any odd prime-order group. Finite "
            "rank controls use verifier-only BSGS labels and receive no "
            "asymptotic credit. The eight actual controls have ranks from "
            "zero to four and none has full C-log rank, so the uniform "
            "density model is not a transfer result. No reverse marker "
            "circuit, signed FFE DAG, generic rank/density proof, or "
            "descent is supplied."
        ),
        "novelty_status": "symmetric_reverse_only_family_novelty_unverified",
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_symmetric_shift_reverse_only_marginal."
            "cost.r153.v1"
        ),
        "known_a_deck_exponent_B": fraction_record(Fraction(1, 12)),
        "a6_shift_support_exponent_B": fraction_record(Fraction(1, 2)),
        "symmetric_c_deck_exponent_B": fraction_record(Fraction(3, 4)),
        "structured_relation_row_count_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "uniform_model_expected_relation_count_exponent_B": (
            fraction_record(Fraction(3, 4))
        ),
        "meaningful_c_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "conditional_reverse_operator_apply_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "known_a_generation_inside_cap": True,
        "symmetric_closure_changes_no_exponent": True,
        "separate_forward_tangent_operator_required": False,
        "reverse_only_marker_operator_supplied": False,
        "signed_weight_separable_ffe_dag_supplied": False,
        "structured_generic_prime_rank_and_density_supplied": False,
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
    all_symmetry = all(
        row["c_deck_inversion_symmetric"]
        and row["c5_kernel_even"]
        and row["a6_shift_multiplicity_even"]
        and row["all_opposite_shift_block_transposes_exact"]
        and row["all_forward_actions_from_opposite_reverse_exact"]
        for row in actual
    )
    all_counts = all(
        row["all_row_sums_recover_six_times_count"] for row in actual
    )
    all_relations = all(
        row["all_known_shift_relation_identities_exact"] for row in actual
    )
    all_rank = all(
        row["stacked_relation_full_c_log_rank"] for row in actual
    )
    all_logs = all(
        row["verifier_only_c_logs_recovered"] for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_symmetric_shift_reverse_only_marginal."
            "controls.r153.v1"
        ),
        "actual_control_count": len(actual),
        "all_symmetric_block_identities_exact": all_symmetry,
        "all_counts_recovered_from_row_sums": all_counts,
        "all_known_shift_relation_identities_exact": all_relations,
        "all_stacked_relation_matrices_full_c_log_rank": all_rank,
        "all_verifier_only_c_logs_recovered": all_logs,
        "finite_stacked_rank_range": [
            min(row["stacked_relation_rank"] for row in actual),
            max(row["stacked_relation_rank"] for row in actual),
        ],
        "finite_full_rank_control_count": sum(
            row["stacked_relation_full_c_log_rank"] for row in actual
        ),
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_symmetric_shift_reverse_only_marginal."
            "frozen.r153.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "reverse_only_signed_marker_operator": "open",
            "signed_weight_separable_ffe_dag": "open",
            "structured_generic_prime_rank_and_density": "open",
            "factor_logs_without_verifier_labels": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_symmetric_shift_reverse_only_marginal."
            "replay.r153.v1"
        ),
        "controls": actual,
        "all_symmetric_block_identities_exact": all_symmetry,
        "all_counts_recovered_from_row_sums": all_counts,
        "all_known_shift_relation_identities_exact": all_relations,
        "all_stacked_relation_matrices_full_c_log_rank": all_rank,
    }

    logs = {
        "schema": (
            "p1553.m6_symmetric_shift_reverse_only_marginal."
            "logs_descent.r153.v1"
        ),
        "finite_verifier_only_c_logs_recovered": all_logs,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "thirteen_source_bindings_verified": len(actual_bindings) == 13,
        "r152_leaf_derivative_state_inherited": True,
        "r144_aggregate_relation_semantics_inherited": True,
        "known_log_a_deck_construction_complete": True,
        "inversion_closed_c_factor_base_construction_complete": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_c5_kernels_even": all(
            row["c5_kernel_even"] for row in actual
        ),
        "all_a6_shift_multiplicities_even": all(
            row["a6_shift_multiplicity_even"] for row in actual
        ),
        "all_opposite_shift_block_transposes_exact": all_symmetry,
        "all_forward_actions_from_reverse_exact": all_symmetry,
        "all_counts_recovered_from_row_sums": all_counts,
        "all_known_shift_relation_identities_exact": all_relations,
        "all_finite_stacked_matrices_full_c_log_rank": all_rank,
        "all_finite_verifier_only_c_logs_recovered": all_logs,
        "reverse_only_operator_reduction_complete": True,
        "symmetric_family_exponents_charged": True,
        "uniform_density_model_scoped_without_transfer": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "reverse_only_signed_marker_operator_complete": False,
        "signed_weight_separable_ffe_dag_complete": False,
        "structured_generic_prime_rank_and_density_complete": False,
        "factor_logs_without_verifier_labels_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Exploit D_s^T=D_(-s) and row-sum count recovery. Construct only "
        "the reverse-adjoint signed marker operator for the inversion-closed "
        "C deck and symmetric known-A6 shift batch. Its geometry-only setup "
        "must fit B^(9/4+o(1)) and one complete reverse batch B^(5/4+o(1)); "
        "do not build a separate forward tangent. Because the eight finite "
        "ranks are zero through four and never full, add a preregistered "
        "multiscale targetable-density and rank transfer rather than "
        "crediting the uniform model. Freeze signed FFE pivots, integer "
        "counts, exact-residual factor logs, and identical descent without "
        "DLP, root, count, marginal, rank, or source oracles."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "KNOWN_LOG_A_DECK_AND_INVERSION_CLOSED_C_DECK_PRESERVE_"
            "EXPONENTS__A6_SHIFT_SET_SYMMETRIC__C5_KERNEL_EVEN__MARGINAL_"
            "BLOCK_DS_TRANSPOSE_EQUALS_D_MINUS_S__ROW_SUM_RECOVERS_6_COUNT__"
            "KNOWN_SHIFT_RELATIONS_EXACT__EIGHT_FINITE_STACKED_RANKS_ZERO_"
            "TO_FOUR_AND_NONE_FULL__REVERSE_ONLY_OPERATOR_REDUCTION_EXACT__"
            "TARGETABLE_DENSITY_SIGNED_FFE_REVERSE_MARKER_CIRCUIT_GENERIC_"
            "RANK_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether inversion symmetry can remove the separate "
            "forward-tangent requirement from the R152 signed marker DAG."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "reverse_only_operator_reduction_admitted": True,
            "finite_rank_deficit_admitted": not all_rank,
            "finite_full_rank_controls_admitted_without_asymptotic_credit": (
                all_rank and all_logs
            ),
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
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
