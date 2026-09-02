#!/usr/bin/env python3
"""Audit universal linear sketches for translated torus C5 queries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_linear_sketch_circulant.r124.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C_ATOM_EXPONENT = Fraction(3, 4)
C2_EXPONENT = 2 * C_ATOM_EXPONENT
C3_EXPONENT = 3 * C_ATOM_EXPONENT
GROUP_ORDER_EXPONENT = Fraction(5)

R123_PRODUCER = pathlib.Path(
    "p1553_torus_c5_fourier_product_resultant_probe_r123.py"
)
R123_PRODUCER_SHA256 = (
    "fb18e752bac3b39bef9d7aa235decdb5913d7de8ff39c6c05c661a54951ba326"
)
R123_REPORT = pathlib.Path(
    "p1553_torus_c5_fourier_product_resultant_probe_report_r123.json"
)
R123_REPORT_SHA256 = (
    "9d7c4bf836e106202526233038409bdbbae704632df16b4da84453cc05eb2870"
)
R123_FROZEN = pathlib.Path(
    "frozen_torus_c5_fourier_product_resultant.json"
)
R123_FROZEN_SHA256 = (
    "f836bb2301cb472197aa26705e2bc8edd6f67d98447a718dffc2f4f935caf9f4"
)
R123_COST = pathlib.Path(
    "torus_c5_fourier_product_resultant_cost_ledger.json"
)
R123_COST_SHA256 = (
    "cd53e6948d00d5befa7598e6f5b4f1fdbc1295c6182f979e3a68ec2652dce72a"
)
R123_REPLAY = pathlib.Path(
    "torus_c5_fourier_product_resultant_replay.json"
)
R123_REPLAY_SHA256 = (
    "efb3425ffada1ceab062a3667688833e9f05c06c6dbba011b44afc9b43dff06b"
)
R123_CONTROLS = pathlib.Path(
    "torus_c5_fourier_product_resultant_controls.json"
)
R123_CONTROLS_SHA256 = (
    "c3ab2d7ed113d231df174f689ac7efca353e0016c0e091bbc67e5d79e97912d4"
)
R123_LOGS = pathlib.Path("factor_logs_and_identical_descent_r123.json")
R123_LOGS_SHA256 = (
    "1a967d9b7fb6b9e10563bf242d390cff83b6ee5f238b7bd453ab9a2f4b1623a7"
)
R123_GATE = pathlib.Path(
    "p1553_torus_c5_fourier_product_resultant_probe_gate_r123.md"
)
R123_GATE_SHA256 = (
    "1b0711d92aeee1a1a4bb2208cdbc772d8a07e1dff14b423803e75f1bd859733f"
)
R123_PARENT = pathlib.Path(
    "p1553_torus_c5_fourier_product_resultant_probe_parent_report_r123.yaml"
)
R123_PARENT_SHA256 = (
    "23bb3ca35ea47189f1e5436d8caaec09cb5ace3c6ced214db9046103b1068bea"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R114_GATE = pathlib.Path(
    "p1553_5a5c_transposed_nonuniform_c5_leaf_generator_probe_gate_r114.md"
)
R114_GATE_SHA256 = (
    "6574145a6d67ec4e55a8bf0cf90c3a0c9f858aa4cddbfd7e9a733c1033b3a3da"
)
R90_GATE = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_probe_gate_r90.md"
)
R90_GATE_SHA256 = (
    "b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R123 = load_module("p1553_r123_for_r124", R123_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r123_producer", R123_PRODUCER, R123_PRODUCER_SHA256),
        ("r123_report", R123_REPORT, R123_REPORT_SHA256),
        ("r123_frozen", R123_FROZEN, R123_FROZEN_SHA256),
        ("r123_cost", R123_COST, R123_COST_SHA256),
        ("r123_replay", R123_REPLAY, R123_REPLAY_SHA256),
        ("r123_controls", R123_CONTROLS, R123_CONTROLS_SHA256),
        ("r123_logs", R123_LOGS, R123_LOGS_SHA256),
        ("r123_gate", R123_GATE, R123_GATE_SHA256),
        ("r123_parent", R123_PARENT, R123_PARENT_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r114_gate", R114_GATE, R114_GATE_SHA256),
        ("r90_gate", R90_GATE, R90_GATE_SHA256),
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
        raise AssertionError(f"R124 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def cyclic_convolution(
    left: Iterable[int],
    right: Iterable[int],
) -> tuple[int, ...]:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("cyclic convolution lengths differ")
    order = len(left_values)
    return tuple(
        sum(
            left_values[index] * right_values[(target - index) % order]
            for index in range(order)
        )
        for target in range(order)
    )


def convolution_power(vector: Iterable[int], exponent: int) -> tuple[int, ...]:
    values = tuple(vector)
    result = (1,) + (0,) * (len(values) - 1)
    for _ in range(exponent):
        result = cyclic_convolution(result, values)
    return result


def translated_query_matrix(kernel: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(kernel)
    order = len(values)
    return tuple(
        tuple(values[(target - source) % order] for source in range(order))
        for target in range(order)
    )


def matrix_vector_product(
    matrix: Iterable[Iterable[int]],
    vector: Iterable[int],
) -> tuple[int, ...]:
    values = tuple(vector)
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, values))
        for row in matrix
    )


def rational_rank(matrix: Iterable[Iterable[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, row_count)
                if rows[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(row_count):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def element_of_order(prime: int, order: int) -> int:
    if (prime - 1) % order:
        raise ValueError("order must divide prime-1")
    exponent = (prime - 1) // order
    for candidate in range(2, prime):
        value = pow(candidate, exponent, prime)
        if value != 1 and pow(value, order, prime) == 1:
            return value
    raise AssertionError("no element of requested order")


def synthetic_control(
    order: int,
    coefficient_prime: int,
    support: tuple[int, ...],
) -> dict[str, Any]:
    if not support or len(support) == order:
        raise ValueError("support must be nonempty and proper")
    deck = tuple(int(index in support) for index in range(order))
    c2 = convolution_power(deck, 2)
    c3 = convolution_power(deck, 3)
    c5 = convolution_power(deck, 5)
    query_matrix = translated_query_matrix(c2)
    queried = matrix_vector_product(query_matrix, c3)
    rank = rational_rank(query_matrix)
    root = element_of_order(coefficient_prime, order)
    deck_spectrum = tuple(
        sum(
            deck[index] * pow(root, mode * index, coefficient_prime)
            for index in range(order)
        )
        % coefficient_prime
        for mode in range(order)
    )
    kernel_spectrum = tuple(
        value * value % coefficient_prime for value in deck_spectrum
    )
    direct_counts = tuple(
        sum(
            sum(indices) % order == target
            for indices in itertools.product(support, repeat=5)
        )
        for target in range(order)
    )
    return {
        "group_order": order,
        "coefficient_prime": coefficient_prime,
        "deck_support": list(support),
        "deck_size": len(support),
        "proper_nonempty_binary_deck": (
            0 < len(support) < order
            and all(value in (0, 1) for value in deck)
        ),
        "c2_total_multiplicity": sum(c2),
        "c3_total_multiplicity": sum(c3),
        "c5_total_multiplicity": sum(c5),
        "translated_inner_products_equal_c5_convolution": queried == c5,
        "direct_five_tuple_counts_equal_convolution": direct_counts == c5,
        "rational_circulant_rank": rank,
        "rational_circulant_rank_is_full": rank == order,
        "finite_field_nonzero_deck_fourier_mode_count": sum(
            value != 0 for value in deck_spectrum
        ),
        "finite_field_nonzero_c2_kernel_mode_count": sum(
            value != 0 for value in kernel_spectrum
        ),
        "finite_field_spectral_rank_is_full": all(kernel_spectrum),
        "candidate_discrete_logs_consumed": False,
        "verifier_support_labels_receive_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
        "c2_sha256": sha256_json(c2),
        "c3_sha256": sha256_json(c3),
        "c5_sha256": sha256_json(c5),
    }


def finite_controls() -> dict[str, Any]:
    cases = (
        (5, 101, (0,)),
        (5, 101, (0, 1)),
        (7, 29, (0, 2, 4)),
        (11, 23, (0, 1)),
        (13, 53, (0, 1, 3, 4)),
    )
    controls = [
        synthetic_control(order, prime, support)
        for order, prime, support in cases
    ]
    return {
        "schema": "p1553.torus_c5_linear_sketch_circulant_controls.r124.v1",
        "controls": controls,
        "control_count": len(controls),
        "all_translated_inner_product_identities_exact": all(
            row["translated_inner_products_equal_c5_convolution"]
            and row["direct_five_tuple_counts_equal_convolution"]
            for row in controls
        ),
        "all_rational_circulant_ranks_full": all(
            row["rational_circulant_rank_is_full"] for row in controls
        ),
        "all_selected_finite_field_spectral_ranks_full": all(
            row["finite_field_spectral_rank_is_full"] for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "translated_inner_product": (
            "u^(*5)(y)=sum_x u^(*3)(x)*u^(*2)(y-x)"
        ),
        "universal_linear_sketch_model": (
            "For fixed C2 kernel w=u^(*2), preprocess every possible "
            "C3 vector v by s linear measurements Sv; recover every exact "
            "count <v,tau_y w> by a linear decoder depending on y."
        ),
        "row_space_argument": (
            "Exact linear decoding for every v forces every row of the "
            "circulant C_w into rowspace(S), so s>=rank(C_w)."
        ),
        "circulant_rank_identity": (
            "Over a splitting field of characteristic zero, "
            "rank(C_w)=#{j:DFT(w)_j!=0}=#{j:DFT(u)_j!=0}."
        ),
        "prime_order_binary_deck_criterion": (
            "For prime q and a proper nonempty binary deck polynomial U, "
            "U(zeta_q^j) is nonzero for every j. For j!=0, a zero would "
            "force the cyclotomic polynomial Phi_q=1+...+X^(q-1) to "
            "divide U over Q, which is possible for a binary polynomial "
            "of degree at most q-1 only when U is empty or all of G."
        ),
        "minimum_linear_sketch_dimension": "q",
        "dimension_exponent_B": fraction_record(GROUP_ORDER_EXPONENT),
        "inside_setup_cap": False,
        "scope_limits": [
            "universal over all C3 occurrence vectors for the fixed kernel",
            "linear preprocessing measurements",
            "linear exact-count decoders followed optionally by a zero test",
            "characteristic-zero rank argument",
        ],
        "not_covered": [
            "a sketch specialized nonlinearly to the coupled pair (u^2,u^3)",
            "nonlinear decoding that returns only membership",
            "adaptive cell-probe or RAM data structures",
            "approximate or bounded-error sketches",
            "general arithmetic circuits",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_linear_sketch_circulant_cost_ledger.r124.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "deck_and_group_exponents": {
            "C_atom": fraction_record(C_ATOM_EXPONENT),
            "C2_occurrences": fraction_record(C2_EXPONENT),
            "C3_occurrences": fraction_record(C3_EXPONENT),
            "group_order_q": fraction_record(GROUP_ORDER_EXPONENT),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "full_translated_query_table",
                "state_exponent_B": fraction_record(GROUP_ORDER_EXPONENT),
                "inside_setup_cap": False,
            },
            {
                "route_id": "universal_linear_C3_sketch_linear_count_decoder",
                "minimum_state_exponent_B": fraction_record(
                    GROUP_ORDER_EXPONENT
                ),
                "inside_setup_cap": False,
                "scoped_lower_bound_proved": True,
            },
            {
                "route_id": "represented_C3_hash_then_C2_scan",
                "setup_exponent_B": fraction_record(C3_EXPONENT),
                "query_exponent_B": fraction_record(C2_EXPONENT),
                "inside_setup_cap": True,
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "coupled_nonlinear_membership_data_structure",
                "scoped_lower_bound_proved": False,
                "exact_structure_constructed": False,
                "status": "open",
            },
        ],
        "source_recovery": {
            "linear_count_sketch_returns_five_backpointers": False,
            "source_augmentation_can_reduce_base_rank_requirement": False,
            "general_source_locator_lower_bound_claimed": False,
        },
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R123_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R123 nonclaim boundary drifted")
    controls = finite_controls()
    theorem = theorem_record()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r123_nonrepresented_circuit_interface_inherited": (
            inherited["admission"][
                "fourier_and_product_resultant_semantics_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "translated_c3_c2_inner_product_identity_exact": controls[
            "all_translated_inner_product_identities_exact"
        ],
        "five_finite_rank_controls_complete": (
            controls["control_count"] == 5
        ),
        "all_rational_circulant_ranks_full": controls[
            "all_rational_circulant_ranks_full"
        ],
        "selected_finite_field_spectral_controls_full": controls[
            "all_selected_finite_field_spectral_ranks_full"
        ],
        "row_space_lower_bound_explicit": (
            theorem["minimum_linear_sketch_dimension"] == "q"
        ),
        "prime_order_cyclotomic_nonvanishing_explicit": (
            "Phi_q" in theorem["prime_order_binary_deck_criterion"]
        ),
        "group_order_B5_state_charged": (
            theorem["dimension_exponent_B"]
            == fraction_record(GROUP_ORDER_EXPONENT)
            and not theorem["inside_setup_cap"]
        ),
        "linear_sketch_scope_frozen": bool(theorem["scope_limits"]),
        "nonlinear_and_coupled_routes_preserved": bool(
            theorem["not_covered"]
        ),
        "candidate_dlogs_not_consumed": (
            not controls["candidate_discrete_logs_consumed"]
        ),
        "inside_cap_nonlinear_membership_complete": False,
        "inside_cap_five_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Freeze one target-specialized nonlinear zero-test for the coupled "
        "pair (u^(*2),u^(*3)) rather than a universal linear sketch. Test "
        "whether a multilinear, rational-Krylov, adaptive probe, or "
        "nonlinear fingerprint circuit can answer exact C5 membership and "
        "return five projective sources with B^(9/4+o(1)) state and "
        "polylogarithmic arbitrary-target work. Charge noncancellation, "
        "all preprocessing, source adjoints, rank, logs, identical descent, "
        "memory, field operations, and bit complexity."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_linear_sketch_circulant.r124.v1"
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
        "closed_scoped_grammar": (
            "universal target-independent linear measurements of C3 with "
            "linear exact-count decoding against every C2 translate"
        ),
        "preserved_interface": (
            "target-specialized nonlinear data structure for the coupled "
            "deck powers u^2 and u^3"
        ),
        "general_data_structure_or_arithmetic_circuit_lower_bound_claimed": (
            False
        ),
    }
    replay = {
        "schema": (
            "p1553.torus_c5_linear_sketch_circulant_replay.r124.v1"
        ),
        "translated_query_identity_exact": controls[
            "all_translated_inner_product_identities_exact"
        ],
        "rational_full_rank_controls_exact": controls[
            "all_rational_circulant_ranks_full"
        ],
        "scoped_universal_linear_sketch_negative_admitted": True,
        "inside_cap_nonlinear_membership_constructed": False,
        "inside_cap_five_source_recovery_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r124.v1",
        "r123_fourier_resultant_audit_complete": True,
        "r124_universal_linear_sketch_audit_complete": True,
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
        "C5_TRANSLATED_C3_C2_INNER_PRODUCT_EXACT__UNIVERSAL_LINEAR_SKETCH_"
        "DIMENSION_EQUALS_CIRCULANT_RANK__PRIME_ORDER_PROPER_BINARY_DECK_"
        "C2_KERNEL_HAS_FULL_RATIONAL_FOURIER_SUPPORT_Q_B5__SETUP_OVER_CAP__"
        "COUPLED_NONLINEAR_TARGET_DATA_STRUCTURE_OPEN__NO_SOURCE_RANK_LOGS_"
        "DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_TRANSLATED_INNER_PRODUCT_AND_SCOPED_UNIVERSAL_LINEAR_"
            "SKETCH_NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_translated_inner_product_identities_exact": controls[
                "all_translated_inner_product_identities_exact"
            ],
            "all_rational_circulant_ranks_full": controls[
                "all_rational_circulant_ranks_full"
            ],
            "all_selected_finite_field_spectral_ranks_full": controls[
                "all_selected_finite_field_spectral_ranks_full"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "translated_inner_product_semantics_admitted": True,
            "scoped_universal_linear_sketch_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_linear_sketch_circulant.json",
            "cost": "torus_c5_linear_sketch_circulant_cost_ledger.json",
            "source_replay": "torus_c5_linear_sketch_circulant_replay.json",
            "controls": "torus_c5_linear_sketch_circulant_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r124.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The rank theorem is universal over C3 vectors for one kernel.",
            "It does not cover nonlinear preprocessing specialized to u^3.",
            "It does not cover nonlinear membership-only decoding.",
            "Finite controls receive no asymptotic credit.",
            "No five-source locator, rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_TRANSLATED_C3C2_INNER_PRODUCT_AND_FULL_CIRCULANT_"
            "RANK_THEOREM_FOR_UNIVERSAL_LINEAR_COUNT_SKETCHES_ONLY__REJECT_"
            "THAT_GRAMMAR_AT_B5_STATE__PRESERVE_COUPLED_NONLINEAR_TARGET_"
            "DATA_STRUCTURE__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_"
            "SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_linear_sketch_circulant_probe_report_r124.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_linear_sketch_circulant.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_linear_sketch_circulant_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_linear_sketch_circulant_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_linear_sketch_circulant_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r124.json"),
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
        f"R124 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
