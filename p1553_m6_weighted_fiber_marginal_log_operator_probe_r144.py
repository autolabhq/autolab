#!/usr/bin/env python3
"""Test source-free relation aggregation at the R115 six-factor vertex."""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_weighted_fiber_marginal_log_operator.r144.v1"

R143_PRODUCER = ROOT / (
    "p1553_torus_c5_label_congruence_correction_probe_r143.py"
)
R143_REPORT = ROOT / (
    "p1553_torus_c5_label_congruence_correction_"
    "probe_report_r143.json"
)
R143_FROZEN = ROOT / "frozen_torus_c5_label_congruence_correction.json"
R143_COST = ROOT / "torus_c5_label_congruence_correction_cost_ledger.json"
R143_REPLAY = ROOT / "torus_c5_label_congruence_correction_replay.json"
R143_CONTROLS = ROOT / "torus_c5_label_congruence_correction_controls.json"
R143_LOGS = ROOT / "factor_logs_and_identical_descent_r143.json"
R143_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_label_congruence_correction_probe_r143.py"
)
R143_GATE = ROOT / (
    "p1553_torus_c5_label_congruence_correction_probe_gate_r143.md"
)
R143_PARENT = ROOT / (
    "p1553_torus_c5_label_congruence_correction_"
    "probe_parent_report_r143.yaml"
)
R115_REPORT = ROOT / (
    "p1553_relation_arity_factor_base_transposed_interface_"
    "rebalance_probe_report_r115.json"
)
R115_GATE = ROOT / (
    "p1553_relation_arity_factor_base_transposed_interface_"
    "rebalance_probe_gate_r115.md"
)

SOURCE_BINDINGS = (
    (
        "r143_producer",
        R143_PRODUCER,
        "715c34ecaf4fd839a8c30d8b43886bad734d96f9113d0c1bed3cde31ed08b00d",
    ),
    (
        "r143_report",
        R143_REPORT,
        "d7d2ef76a1a0a8579ca543152e5d621e66a8e46723987896b45a8cc47feb950b",
    ),
    (
        "r143_frozen",
        R143_FROZEN,
        "ad81801828ee4e88e7117bc3bd06d62ce237cc54bb24e2a90537f0857aebc588",
    ),
    (
        "r143_cost",
        R143_COST,
        "2229360cdcd9d625b9c93746f53c973393d4bd005637296c3b3558f8c62bc7f5",
    ),
    (
        "r143_replay",
        R143_REPLAY,
        "e793128490b03e22b8d532f3cbbc298583a206fe0a8a206b41615fb99d20402a",
    ),
    (
        "r143_controls",
        R143_CONTROLS,
        "82f41b2215f9261e38533d65128b6cdc2e107b9aa496cc01086514a1d3d94e9e",
    ),
    (
        "r143_logs",
        R143_LOGS,
        "abe05d9d06e23008f46ac9c040affa57e5bb488886ec925bec0de81143c3c271",
    ),
    (
        "r143_test",
        R143_TEST,
        "4d7573f630148b687a59a27fec9cb95f1e902e36d0e8ac5dc9d24bf09481a5d6",
    ),
    (
        "r143_gate",
        R143_GATE,
        "e17719b1f8ef38c13c02c682f4a51b072ef3f47f23633cf29f3499a8e4ec6653",
    ),
    (
        "r143_parent",
        R143_PARENT,
        "91d76c1b26396e28a2f7199d196ae1bdd294641491dbdfed418d2840e60cc6c9",
    ),
    (
        "r115_report",
        R115_REPORT,
        "2b43c37f93e03c6ccdb675a34d48d7478deb3b0349026c65b2f6fe65695cb88e",
    ),
    (
        "r115_gate",
        R115_GATE,
        "11dd899c1ab6b89444cd9c9cc75709a64d9abf99d9370d2536e2f20e3d0fd1de",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_report_r144.json"
)
DEFAULT_FROZEN = ROOT / "frozen_m6_weighted_fiber_marginal_log_operator.json"
DEFAULT_COST = ROOT / "m6_weighted_fiber_marginal_log_operator_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_weighted_fiber_marginal_log_operator_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_weighted_fiber_marginal_log_operator_controls.json"
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r144.json"

RELATION_ARITY = 6


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R143 = load_module("p1553_r143_for_r144", R143_PRODUCER)
R141 = R143.R141
R82 = R141.R82
R81 = R82.R81
R70 = R82.R70


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
        raise AssertionError(f"R144 source binding mismatch: {failures}")
    return actual


def multinomial_weight(source: tuple[int, ...]) -> int:
    weight = math.factorial(len(source))
    for multiplicity in collections.Counter(source).values():
        weight //= math.factorial(multiplicity)
    return weight


def meaningful_log_vector(
    labels_a: list[int],
    labels_c: list[int],
    modulus: int,
) -> list[int]:
    return [
        (labels_a[0] + labels_c[0]) % modulus,
        *((value - labels_a[0]) % modulus for value in labels_a[1:]),
        *((value - labels_c[0]) % modulus for value in labels_c[1:]),
    ]


def reduced_marginal_row(
    full_marginal: list[int],
    count: int,
    size_a: int,
    size_c: int,
) -> list[int]:
    return [
        RELATION_ARITY * count,
        *full_marginal[1:size_a],
        *full_marginal[size_a + 1 : size_a + size_c],
    ]


def solve_square_mod(
    matrix: list[list[int]],
    rhs: list[int],
    modulus: int,
) -> list[int]:
    width = len(matrix)
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("square nonempty matrix required")
    work = [
        [*(value % modulus for value in row), value % modulus]
        for row, value in zip(matrix, rhs)
    ]
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(pivot_row, width)
                if work[row][column] % modulus
            ),
            None,
        )
        if pivot is None:
            raise ValueError("singular matrix")
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column], modulus - 2, modulus)
        work[pivot_row] = [
            value * inverse % modulus
            for value in work[pivot_row]
        ]
        for row in range(width):
            if row == pivot_row:
                continue
            scale = work[row][column] % modulus
            if not scale:
                continue
            work[row] = [
                (left - scale * right) % modulus
                for left, right in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
    return [work[row][-1] for row in range(width)]


def aggregate_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    modulus = curve["subgroup_order"]
    generator = R81.curve_generator(curve)
    verifier = R81.BatchBsgsVerifier(generator, curve)
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    labels_a = verifier.labels(atoms_a)
    labels_c = verifier.labels(atoms_c)
    factor_labels = [
        (labels_a[left] + labels_c[right]) % modulus
        for left in range(len(atoms_a))
        for right in range(len(atoms_c))
    ]
    if not (
        R82.labels_match_points(atoms_a, labels_a, generator, curve)
        and R82.labels_match_points(atoms_c, labels_c, generator, curve)
        and R82.labels_match_points(
            factors,
            factor_labels,
            generator,
            curve,
        )
    ):
        raise AssertionError("verifier labels failed projective replay")

    size_a = len(atoms_a)
    size_c = len(atoms_c)
    factor_count = len(factors)
    meaningful_width = size_a + size_c - 1
    counts: dict[int, int] = {}
    full_marginals: dict[int, list[int]] = {}
    first_source: dict[int, tuple[int, ...]] = {}
    for source in itertools.combinations_with_replacement(
        range(factor_count),
        RELATION_ARITY,
    ):
        multiplicities = collections.Counter(source)
        weight = multinomial_weight(source)
        target_scalar = sum(factor_labels[index] for index in source) % modulus
        counts[target_scalar] = counts.get(target_scalar, 0) + weight
        marginal = full_marginals.setdefault(
            target_scalar,
            [0] * (size_a + size_c),
        )
        for factor_index, multiplicity in multiplicities.items():
            left, right = divmod(factor_index, size_c)
            marginal[left] += weight * multiplicity
            marginal[size_a + right] += weight * multiplicity
        first_source.setdefault(target_scalar, source)

    rows = {
        scalar: reduced_marginal_row(
            full_marginals[scalar],
            count,
            size_a,
            size_c,
        )
        for scalar, count in counts.items()
    }
    true_logs = meaningful_log_vector(labels_a, labels_c, modulus)
    all_rhs_identities_exact = all(
        sum(
            coefficient * value
            for coefficient, value in zip(rows[scalar], true_logs)
        )
        % modulus
        == scalar * counts[scalar] % modulus
        for scalar in rows
    )
    all_a_marginal_sums_exact = all(
        sum(full_marginals[scalar][:size_a])
        == RELATION_ARITY * counts[scalar]
        for scalar in rows
    )
    all_c_marginal_sums_exact = all(
        sum(full_marginals[scalar][size_a:])
        == RELATION_ARITY * counts[scalar]
        for scalar in rows
    )
    aggregate_rank = R81.rank_mod(list(rows.values()), modulus)

    selected: list[dict[str, Any]] = []
    basis: list[list[int]] = []
    for scalar in sorted(rows):
        row = rows[scalar]
        if R81.rank_mod([*basis, row], modulus) == len(basis):
            continue
        basis.append(row)
        selected.append(
            {
                "target_scalar": scalar,
                "ordered_fiber_count": counts[scalar],
                "reduced_marginal_row": row,
                "known_rhs": scalar * counts[scalar] % modulus,
            }
        )
        if len(selected) == meaningful_width:
            break
    if len(selected) != meaningful_width:
        raise AssertionError("aggregate fibers failed meaningful rank")
    recovered_logs = solve_square_mod(
        [row["reduced_marginal_row"] for row in selected],
        [row["known_rhs"] for row in selected],
        modulus,
    )
    factor_logs_recovered = recovered_logs == true_logs
    reconstructed_factor_logs = [
        (
            recovered_logs[0]
            + (recovered_logs[left] if left else 0)
            + (
                recovered_logs[size_a - 1 + right]
                if right
                else 0
            )
        )
        % modulus
        for left in range(size_a)
        for right in range(size_c)
    ]
    all_factor_logs_replay = (
        reconstructed_factor_logs == factor_labels
    )
    all_positive_target_logs_recovered = all(
        (
            sum(
                coefficient * value
                for coefficient, value in zip(
                    rows[scalar],
                    recovered_logs,
                )
            )
            * pow(counts[scalar] % modulus, modulus - 2, modulus)
        )
        % modulus
        == scalar
        for scalar in rows
        if counts[scalar] % modulus
    )
    all_counts_invertible = all(
        count % modulus for count in counts.values()
    )

    control_id = f"{curve['family_id']}_offset{offset}"
    shift = (
        int.from_bytes(
            hashlib.sha256(f"R144|{control_id}".encode()).digest(),
            "big",
        )
        % (modulus - 1)
        + 1
    )
    shifted_samples = []
    for scalar in sorted(rows)[: min(12, len(rows))]:
        recovered_shifted = (
            sum(
                coefficient * value
                for coefficient, value in zip(
                    rows[scalar],
                    recovered_logs,
                )
            )
            * pow(counts[scalar] % modulus, modulus - 2, modulus)
            - shift
        ) % modulus
        shifted_target = R70.scalar_mul(
            (scalar - shift) % modulus,
            generator,
            curve,
        )
        shifted_samples.append(
            {
                "fiber_target_scalar": scalar,
                "known_shift": shift,
                "recovered_unknown_target_scalar": recovered_shifted,
                "unknown_target": R70.point_to_json(shifted_target),
                "projective_replay_exact": (
                    R70.scalar_mul(recovered_shifted, generator, curve)
                    == shifted_target
                ),
            }
        )
    empty_scalar = next(
        scalar for scalar in range(modulus) if scalar not in counts
    )
    empty_target = R70.scalar_mul(empty_scalar, generator, curve)

    source_samples = []
    for scalar in sorted(rows)[: min(12, len(rows))]:
        source = first_source[scalar]
        endpoint = R82.add_many(
            (factors[index] for index in source),
            curve,
        )
        source_samples.append(
            {
                "target_scalar": scalar,
                "source": list(source),
                "source_endpoint": R70.point_to_json(endpoint),
                "known_target": R70.point_to_json(
                    R70.scalar_mul(scalar, generator, curve)
                ),
                "projective_source_replay_exact": (
                    endpoint
                    == R70.scalar_mul(scalar, generator, curve)
                ),
            }
        )

    sample_scalars = sorted(rows)[: min(12, len(rows))]
    fiber_samples = [
        {
            "target_scalar": scalar,
            "ordered_fiber_count": counts[scalar],
            "full_atom_marginal": full_marginals[scalar],
            "reduced_marginal_row": rows[scalar],
            "known_rhs": scalar * counts[scalar] % modulus,
            "identity_exact": (
                sum(
                    coefficient * value
                    for coefficient, value in zip(
                        rows[scalar],
                        true_logs,
                    )
                )
                % modulus
                == scalar * counts[scalar] % modulus
            ),
        }
        for scalar in sample_scalars
    ]

    return {
        "control_id": control_id,
        "field_prime": curve["field_prime"],
        "subgroup_order": modulus,
        "offset": offset,
        "atom_a_size": size_a,
        "atom_c_size": size_c,
        "factor_base_size": factor_count,
        "meaningful_log_dimension": meaningful_width,
        "canonical_six_factor_source_count": math.comb(
            factor_count + RELATION_ARITY - 1,
            RELATION_ARITY,
        ),
        "ordered_six_factor_occurrence_count": factor_count**RELATION_ARITY,
        "positive_fiber_count": len(counts),
        "positive_fiber_density": len(counts) / modulus,
        "maximum_ordered_fiber_count": max(counts.values()),
        "all_fiber_counts_invertible_mod_subgroup_order": (
            all_counts_invertible
        ),
        "all_a_marginal_sums_exact": all_a_marginal_sums_exact,
        "all_c_marginal_sums_exact": all_c_marginal_sums_exact,
        "all_known_rhs_fiber_identities_exact": all_rhs_identities_exact,
        "aggregate_marginal_rank": aggregate_rank,
        "aggregate_marginal_full_meaningful_rank": (
            aggregate_rank == meaningful_width
        ),
        "selected_independent_fibers": selected,
        "known_target_prefix_length_for_selected_fibers": (
            max(row["target_scalar"] for row in selected) + 1
        ),
        "meaningful_factor_logs_recovered": factor_logs_recovered,
        "all_cartesian_factor_logs_replay": all_factor_logs_replay,
        "all_positive_target_logs_recovered": (
            all_positive_target_logs_recovered
        ),
        "shifted_target_samples": shifted_samples,
        "all_shifted_target_samples_replay": all(
            row["projective_replay_exact"] for row in shifted_samples
        ),
        "empty_target_scalar": empty_scalar,
        "empty_target": R70.point_to_json(empty_target),
        "empty_target_count": 0,
        "empty_target_rejected": empty_scalar not in counts,
        "fiber_samples": fiber_samples,
        "source_semantic_samples": source_samples,
        "all_source_semantic_samples_replay": all(
            row["projective_source_replay_exact"]
            for row in source_samples
        ),
        "verifier": verifier.receipt(),
        "diagnostic_build_consumes_verifier_scalar_labels": True,
        "candidate_operator_consumes_scalar_labels": False,
        "finite_control_receives_asymptotic_credit": False,
        "geometry": geometry,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def conditional_exponent_envelope() -> dict[str, Any]:
    return {
        "group_order_N_exponent_B": fraction_record(Fraction(5)),
        "atom_a_size_exponent_B": fraction_record(Fraction(1, 12)),
        "atom_c_size_exponent_B": fraction_record(Fraction(3, 4)),
        "factor_base_size_exponent_B": fraction_record(Fraction(5, 6)),
        "meaningful_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "six_factor_occurrence_exponent_B": fraction_record(Fraction(5)),
        "known_target_hit_probability_exponent_B": fraction_record(
            Fraction(0)
        ),
        "positive_marginal_matrix_output_exponent_B": fraction_record(
            Fraction(3, 2)
        ),
        "dense_log_elimination_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "conditional_precomputation_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "conditional_precomputation_exponent_N": fraction_record(
            Fraction(9, 20)
        ),
        "conditional_shifted_descent_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "conditional_shifted_descent_exponent_N": fraction_record(
            Fraction(3, 20)
        ),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "pollard_rho_exponent_N": fraction_record(Fraction(1, 2)),
        "conditions": [
            (
                "A scalar-blind weighted six-factor fiber-count circuit "
                "builds in B^(9/4+o(1)) work and state."
            ),
            (
                "A fresh target count query costs polylog(B), and reverse "
                "transposed state returns A/C marginals in "
                "B^(3/4+o(1)) output work without replaying setup."
            ),
            (
                "A B^(3/4+o(1)) known-scalar target stream supplies full "
                "meaningful aggregate rank with invertible fiber counts."
            ),
            (
                "The same count circuit accepts arbitrary shifted targets "
                "and its successful-fiber density is B^(0+o(1))."
            ),
            (
                "All extension-field, chart, integer-lift, memory, and bit "
                "costs remain within the stated exponents."
            ),
        ],
        "credit": "conditional_reduction_only",
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    actual = [
        aggregate_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    all_rhs = all(
        row["all_known_rhs_fiber_identities_exact"] for row in actual
    )
    all_rank = all(
        row["aggregate_marginal_full_meaningful_rank"] for row in actual
    )
    all_logs = all(
        row["meaningful_factor_logs_recovered"]
        and row["all_cartesian_factor_logs_replay"]
        for row in actual
    )
    all_descent = all(
        row["all_positive_target_logs_recovered"]
        and row["all_shifted_target_samples_replay"]
        and row["empty_target_rejected"]
        for row in actual
    )
    all_derivatives = all(
        row["all_a_marginal_sums_exact"]
        and row["all_c_marginal_sums_exact"]
        for row in actual
    )
    envelope = conditional_exponent_envelope()

    controls = {
        "schema": (
            "p1553.m6_weighted_fiber_marginal_log_operator."
            "controls.r144.v1"
        ),
        "actual_control_count": len(actual),
        "all_weight_derivative_marginal_sums_exact": all_derivatives,
        "all_known_rhs_fiber_identities_exact": all_rhs,
        "all_aggregate_marginal_matrices_full_meaningful_rank": all_rank,
        "all_finite_factor_log_solves_exact": all_logs,
        "all_positive_and_shifted_target_descents_exact": all_descent,
        "all_empty_targets_rejected": all(
            row["empty_target_rejected"] for row in actual
        ),
        "all_source_semantic_samples_replay": all(
            row["all_source_semantic_samples_replay"] for row in actual
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_weighted_fiber_marginal_log_operator."
            "frozen.r144.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "weighted_fiber_polynomial": {
            "definition": (
                "Z_T(u,v)=sum_{(a_j,c_j): sum_j(A_a_j+C_c_j)=T} "
                "prod_j u_a_j v_c_j over ordered six-factor sources."
            ),
            "count": "c_T=Z_T(1,1)",
            "a_marginal": (
                "d^A_T(a)=(u_a partial Z_T/partial u_a) at u=v=1"
            ),
            "c_marginal": (
                "d^C_T(c)=(v_c partial Z_T/partial v_c) at u=v=1"
            ),
            "reverse_differentiation": (
                "Baur-Strassen gives constant-factor nonscalar overhead "
                "for differentiating one complete division-safe circuit "
                "for Z_T, plus explicit marginal output."
            ),
            "offline_online_boundary": (
                "Baur-Strassen does not by itself preserve an offline "
                "setup/online query split. The candidate must compile "
                "reusable derivative-capable state whose fresh marginal "
                "query avoids replaying B^(9/4) preprocessing."
            ),
            "primary_reference": (
                "Baur and Strassen, The Complexity of Partial Derivatives, "
                "Theoretical Computer Science 22 (1983), 317-330, "
                "doi:10.1016/0304-3975(83)90110-X."
            ),
        },
        "meaningful_log_coordinates": {
            "x0": "log(A_0+C_0)",
            "delta_a": "log(A_a)-log(A_0), a>0",
            "delta_c": "log(C_c)-log(C_0), c>0",
            "dimension": "|A|+|C|-1",
            "factor_log": (
                "log(A_a+C_c)=x0+delta_a(a)+delta_c(c)"
            ),
        },
        "known_rhs_identity": (
            "For T=[k]G, 6*c_T*x0 + sum_{a>0}d^A_T(a)*delta_a "
            "+ sum_{c>0}d^C_T(c)*delta_c = k*c_T mod q."
        ),
        "source_free_descent_identity": (
            "If c_T is nonzero mod q, log_G(T)=c_T^(-1) times the "
            "aggregate marginal dot product. For T=Q+[r]G this returns "
            "log_G(Q) after subtracting r."
        ),
        "source_requirement": (
            "No individual relation source is needed for factor-log solve "
            "or target recovery once exact count and marginals are supplied."
        ),
        "conditional_exponent_envelope": envelope,
        "required_open_outputs": {
            "weighted_s13_or_group_law_count_circuit": "open",
            "offline_online_transposed_derivative_index": "open",
            "structured_generic_family_rank_and_density": "open",
            "complete_projective_chart_and_integer_lift": "open",
            "factor_logs_without_verifier_labels": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_weighted_fiber_marginal_log_operator."
            "replay.r144.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "selected_independent_fibers": row[
                    "selected_independent_fibers"
                ],
                "fiber_samples": row["fiber_samples"],
                "shifted_target_samples": row["shifted_target_samples"],
                "source_semantic_samples": row[
                    "source_semantic_samples"
                ],
                "empty_target_scalar": row["empty_target_scalar"],
                "empty_target_count": row["empty_target_count"],
                "empty_target_rejected": row["empty_target_rejected"],
            }
            for row in actual
        ],
        "all_known_rhs_fiber_identities_exact": all_rhs,
        "all_finite_factor_log_solves_exact": all_logs,
        "all_positive_and_shifted_target_descents_exact": all_descent,
    }

    cost = {
        "schema": (
            "p1553.m6_weighted_fiber_marginal_log_operator."
            "cost.r144.v1"
        ),
        "conditional_exponent_envelope": envelope,
        "finite_verifier_six_factor_enumeration_charged_to_candidate": False,
        "finite_verifier_bsgs_labels_charged_to_candidate": False,
        "candidate_field_dlp_used": False,
        "weighted_count_circuit_supplied": False,
        "offline_online_transposed_derivative_index_supplied": False,
        "structured_rank_density_theorem_supplied": False,
        "complete_projective_integer_lift_supplied": False,
        "rank_cost_supplied_conditionally": True,
        "factor_log_cost_supplied_conditionally": True,
        "identical_descent_cost_supplied_conditionally": True,
        "unconditional_total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.m6_weighted_fiber_marginal_log_operator."
            "logs_descent.r144.v1"
        ),
        "finite_verifier_factor_logs_recovered": all_logs,
        "finite_verifier_positive_target_logs_recovered": all_descent,
        "candidate_factor_logs_computed_without_verifier_labels": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "twelve_source_bindings_verified": len(actual_bindings) == 12,
        "r115_six_factor_vertex_inherited": True,
        "weighted_fiber_polynomial_frozen": True,
        "count_as_weighted_specialization_derived": True,
        "marginals_as_log_weight_derivatives_derived": True,
        "baur_strassen_offline_online_separation_not_inferred": True,
        "known_rhs_aggregate_identity_derived": True,
        "source_free_shifted_descent_identity_derived": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_marginal_sum_identities_exact": all_derivatives,
        "all_actual_known_rhs_identities_exact": all_rhs,
        "all_actual_aggregate_matrices_full_meaningful_rank": all_rank,
        "all_finite_factor_log_solves_exact": all_logs,
        "all_finite_positive_and_shifted_descents_exact": all_descent,
        "all_finite_empty_targets_rejected": controls[
            "all_empty_targets_rejected"
        ],
        "all_source_semantic_samples_replay": controls[
            "all_source_semantic_samples_replay"
        ],
        "candidate_discrete_log_oracle_avoided": True,
        "conditional_n9o20_cost_envelope_derived": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "weighted_s13_or_group_law_count_circuit_complete": False,
        "offline_online_transposed_derivative_index_complete": False,
        "structured_generic_family_rank_density_complete": False,
        "complete_projective_integer_lift_complete": False,
        "factor_logs_without_verifier_labels_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Construct the scalar-blind weighted six-factor fiber-count "
        "circuit Z_T(u,v) directly from the compact A/C divisors and the "
        "S7 or expanded S13 relation. It must build in B^(9/4+o(1)) work "
        "and state, answer a fresh target count in polylog(B) work, and "
        "compile reusable division-safe transposed state that emits the "
        "B^(3/4+o(1)) A/C marginal vector without replaying setup. Freeze "
        "projective charts, "
        "multiplicity and integer lifting, known-target rank and density, "
        "factor-log solve, shifted identical descent, memory, field "
        "operations, and bit cost. Do not consume a DLP, root, count, "
        "marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "ORDERED_M6_RELATIONS_COMPRESS_TO_WEIGHTED_FIBER_COUNT_AND_"
            "ATOM_MARGINALS__KNOWN_RHS_LOG_IDENTITY_AND_SOURCE_FREE_SHIFTED_"
            "DESCENT_EXACT__EIGHT_ACTUAL_AGGREGATE_MATRICES_FULL_MEANINGFUL_"
            "RANK__CONDITIONAL_PRECOMPUTATION_N9O20__WEIGHTED_S13_COUNT_"
            "CIRCUIT_RANK_DENSITY_AND_GENERIC_TRANSFER_OPEN__NO_SHOUP_"
            "BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether summing all six-factor relations in each "
            "known-target fiber removes the individual-source requirement "
            "and reduces factor logs plus identical descent to one weighted "
            "summation-polynomial/FFE count circuit."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "weighted_fiber_operator": frozen[
                "weighted_fiber_polynomial"
            ],
            "known_rhs_identity": frozen["known_rhs_identity"],
            "source_free_descent_identity": frozen[
                "source_free_descent_identity"
            ],
            "source_requirement": frozen["source_requirement"],
            "conditional_exponent_envelope": envelope,
            "finite_actual_result": (
                "All eight actual Cartesian controls have full meaningful "
                "aggregate marginal rank; the selected known-RHS fibers "
                "recover every Cartesian factor log, every checked positive "
                "and shifted target replays, and every empty target is "
                "rejected."
            ),
            "literature_novelty": "unverified",
            "scope": (
                "The identities are exact in any prime-order group. The "
                "finite rank and descent controls use verifier-only BSGS "
                "labels and receive no asymptotic credit. The N^(9/20) "
                "envelope is conditional on a scalar-blind weighted count "
                "index, an offline/online transposed derivative index, "
                "structured rank/density, and complete projective and "
                "bit-cost gates. Baur-Strassen alone does not preserve the "
                "offline/online split. This is not an "
                "ECDLP algorithm or a Shoup-bound improvement."
            ),
        },
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "aggregate_identity_reduction_admitted": True,
            "source_free_log_descent_reduction_admitted": True,
            "conditional_exponent_envelope_admitted": True,
            "weighted_count_circuit_admitted": False,
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
