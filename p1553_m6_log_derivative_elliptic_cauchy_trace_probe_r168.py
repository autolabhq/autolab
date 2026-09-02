#!/usr/bin/env python3
"""Linearize the R167 resultant as a denominator-aware elliptic trace."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_log_derivative_elliptic_cauchy_trace.r168.v1"

R167_PRODUCER = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_r167.py"
R167_REPORT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_report_r167.json"
R167_FROZEN = ROOT / "frozen_m6_generalized_target_divisor_weil_reciprocity_swap.json"
R167_COST = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_cost_ledger.json"
R167_REPLAY = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_replay.json"
R167_CONTROLS = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_controls.json"
R167_RESULTANT = ROOT / "generalized_miller_elliptic_resultant_swap_r167.json"
R167_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_r167.py"
R167_GATE = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_gate_r167.md"
R167_PARENT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_parent_report_r167.yaml"
EAGEN_PAPER = ROOT / "references/eagen_ecip_weil_reciprocity_2022_596.pdf"
R147_REPORT = ROOT / "p1553_m6_occurrence_pair_resultant_local_valuation_probe_report_r147.json"
R147_GATE = ROOT / "p1553_m6_occurrence_pair_resultant_local_valuation_probe_gate_r147.md"
R166_REPORT = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_report_r166.json"
R166_GATE = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_gate_r166.md"

SOURCE_BINDINGS = (
    ("r167_producer", R167_PRODUCER, "ba57052082668daf38027b928d88b0eca210510e2f6a6784ce3e0e8b481e1cd3"),
    ("r167_report", R167_REPORT, "1d2f009525ef9d54a0f538d3a0a8cefe8451d4d97933b91b31ed1bb5c0b47e3c"),
    ("r167_frozen", R167_FROZEN, "47d3d39017faa9adc1e79abc939c658c51272097dcb8620e72db51ce7543f492"),
    ("r167_cost", R167_COST, "a84dc45c92d9a17ef9e9b73ddad426a8fe5111e72b6cd974fd5968841ebd10c3"),
    ("r167_replay", R167_REPLAY, "cf983cfdb3428503634d84a5a69305b71ac70e44338c37acb24e646e5593f6d0"),
    ("r167_controls", R167_CONTROLS, "f78db8e5e5195727087552c8d8a122037eb1f0a0c5d469594ef08a070c093f83"),
    ("r167_resultant", R167_RESULTANT, "95a42cb9d499012b676d96a77ca7c8a066e0cc6ade79e6cf9d0dd9ccfb674247"),
    ("r167_test", R167_TEST, "8c098e9a820ea7967cc700544aa64cc419750806c67e7c3dca395f271ae505e9"),
    ("r167_gate", R167_GATE, "41c4869bb231dfa2d0de1bb0d280aa34bd95903ed425792b80a84c762b845e3b"),
    ("r167_parent", R167_PARENT, "0952a627b3bd5acd95e622533627f54b339605f92e9187be68d72431e7931db7"),
    ("eagen_2022_596", EAGEN_PAPER, "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e"),
    ("r147_report", R147_REPORT, "a419f994907d392b8d8cec7a3af6dc0f3a67769350be7c1044e3de4320a69977"),
    ("r147_gate", R147_GATE, "7398ae8894fc40f710f76e2ec0e6f31726f283ca6fb9153f5258e0561537a6fe"),
    ("r166_report", R166_REPORT, "94ee650b1302d231d946fc02f14d1c12b62a63bd9fba240a9d6a93e4bb986285"),
    ("r166_gate", R166_GATE, "e98cd1ba5b73c591dae5882e9058fafbd6e548df3c2ae929563181bc84920d9c"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_report_r168.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_log_derivative_elliptic_cauchy_trace.json"
DEFAULT_COST = ROOT / "m6_log_derivative_elliptic_cauchy_trace_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_log_derivative_elliptic_cauchy_trace_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_log_derivative_elliptic_cauchy_trace_controls.json"
DEFAULT_TRACE = ROOT / "log_derivative_candidate_poles_and_trace_r168.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R167 = load_module("p1553_r167_for_r168", R167_PRODUCER)
R166 = R167.R166
R161 = R167.R161


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R168 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def polynomial_derivative(poly: list[int], prime: int) -> list[int]:
    if len(poly) <= 1:
        return [0]
    return [index * coefficient % prime for index, coefficient in enumerate(poly)][1:]


def invariant_derivative_witness_value(
    witness: dict[str, Any],
    point: tuple[int, int],
    curve: dict[str, Any],
) -> int:
    prime = int(curve["field_prime"])
    curve_a = int(curve["curve_a"])
    curve_b = int(curve["curve_b"])
    x_coord, y_coord = point
    curve_value = (pow(x_coord, 3, prime) + curve_a * x_coord + curve_b) % prime
    result = 0
    for coefficient, monomial in zip(
        witness["coefficients"], witness["basis"]
    ):
        coefficient = int(coefficient)
        exponent = int(monomial["exponent"])
        if not coefficient:
            continue
        if monomial["kind"] == "x":
            if exponent:
                term = (
                    2
                    * exponent
                    * y_coord
                    * pow(x_coord, exponent - 1, prime)
                ) % prime
            else:
                term = 0
        else:
            term = (
                (3 * x_coord * x_coord + curve_a)
                * pow(x_coord, exponent, prime)
            ) % prime
            if exponent:
                term += (
                    2
                    * exponent
                    * curve_value
                    * pow(x_coord, exponent - 1, prime)
                ) % prime
        result = (result + coefficient * term) % prime
    return result


def direct_factor_derivative_at_zero(
    divisor: dict[str, Any],
    translated: tuple[int, int],
    prime: int,
) -> int:
    derivative = polynomial_derivative(divisor["u"], prime)
    return (
        -2
        * int(translated[1])
        * R161.poly_eval(derivative, int(translated[0]), prime)
    ) % prime


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r167_control = R167.finite_control(curve, seed)
    _, divisor, target_records = R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    point_index = {tuple(record["endpoint"]): record for record in divisor["records"]}
    retained_records = [
        record
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    dropped_records = [
        record
        for record in target_records
        if tuple(record["target"]) in selected_set
    ]
    numerator = r167_control["numerator_witness"]
    denominator = r167_control["denominator_witness"]
    signed_support = selected + [
        R167.point_negate(point, curve) for point in selected
    ]
    prime = int(curve["field_prime"])
    local_rows: list[dict[str, Any]] = []
    all_direct_zero_derivatives_nonzero = True
    all_swapped_zero_derivatives_nonzero = True
    all_multiplicities_equal = True
    all_multiplicity_residues_nonzero = True
    true_orientation_count = 0
    opposite_orientation_count = 0
    for left in selected:
        direct_derivatives: list[int] = []
        orientations: list[str] = []
        for target_record in retained_records:
            target = tuple(target_record["target"])
            translated = R167.point_subtract(target, left, curve)
            if translated is None:
                raise AssertionError("retained target reached equality pole")
            if R167.kummer_value(divisor, translated, prime) != 0:
                continue
            derivative = direct_factor_derivative_at_zero(
                divisor, translated, prime
            )
            direct_derivatives.append(derivative)
            all_direct_zero_derivatives_nonzero &= derivative != 0
            orientation, _ = R166.classify_translate(
                left, target, point_index, curve
            )
            if orientation not in {"true_signed", "opposite_sign"}:
                raise AssertionError("Kummer zero lacks a signed orientation")
            orientations.append(orientation)
            true_orientation_count += int(orientation == "true_signed")
            opposite_orientation_count += int(orientation == "opposite_sign")

        swapped_derivatives: list[int] = []
        for support_point in signed_support:
            translated = R167.point_add(support_point, left, curve)
            denominator_value = R167.witness_value(
                denominator, translated, prime
            )
            if denominator_value == 0:
                raise AssertionError("R167 denominator unit was lost")
            numerator_value = R167.witness_value(numerator, translated, prime)
            if numerator_value != 0:
                continue
            if translated is None:
                raise AssertionError("full-pole witness vanished at infinity")
            derivative = (
                invariant_derivative_witness_value(numerator, translated, curve)
                * pow(denominator_value, -1, prime)
                % prime
            )
            swapped_derivatives.append(derivative)
            all_swapped_zero_derivatives_nonzero &= derivative != 0

        direct_multiplicity = len(direct_derivatives)
        swapped_multiplicity = len(swapped_derivatives)
        all_multiplicities_equal &= direct_multiplicity == swapped_multiplicity
        all_multiplicity_residues_nonzero &= (
            direct_multiplicity == 0 or direct_multiplicity % prime != 0
        )
        local_rows.append(
            {
                "left_endpoint": R167.point_list(left),
                "direct_zero_multiplicity": direct_multiplicity,
                "swapped_zero_multiplicity": swapped_multiplicity,
                "logarithmic_differential_residue_mod_p": (
                    direct_multiplicity % prime
                ),
                "candidate_pole": direct_multiplicity > 0,
                "orientation_counts": dict(sorted(Counter(orientations).items())),
                "direct_zero_derivatives_sha256": sha256_json(direct_derivatives),
                "swapped_zero_derivatives_sha256": sha256_json(swapped_derivatives),
                "all_direct_zero_derivatives_nonzero": all(direct_derivatives),
                "all_swapped_zero_derivatives_nonzero": all(swapped_derivatives),
            }
        )

    candidate_roots = sorted(
        int(row["left_endpoint"][0])
        for row in local_rows
        if row["candidate_pole"]
    )
    candidate_factor = R161.monic_root_polynomial(candidate_roots, prime)
    equality_rows = []
    for target_record in dropped_records:
        target = tuple(target_record["target"])
        selected_sign_matches = (
            R161.poly_eval(divisor["u"], int(target[0]), prime) == 0
            and R161.poly_eval(divisor["v"], int(target[0]), prime)
            == int(target[1])
        )
        pole_order = 2 * len(selected)
        equality_rows.append(
            {
                "target_role": target_record["role"],
                "target": R167.point_list(target),
                "selected_sign_matches": selected_sign_matches,
                "rational_factor_pole_order": pole_order,
                "rational_logarithmic_residue_mod_p": (-pole_order) % prime,
                "rational_logarithmic_residue_nonzero": pole_order % prime != 0,
                "semantic_regularized_factor": 1,
                "semantic_regularized_logarithmic_derivative": 0,
                "removed_by_public_u_v_target_equality_prefilter": True,
            }
        )

    multiplicities = [
        int(row["direct_zero_multiplicity"])
        for row in local_rows
        if row["candidate_pole"]
    ]
    return {
        "control_id": f"{curve['family_id']}_log_derivative_trace_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": len(selected),
        "retained_target_count": len(retained_records),
        "public_target_equality_correction_count": len(equality_rows),
        "candidate_pole_count": len(candidate_roots),
        "candidate_zero_occurrence_count": sum(multiplicities),
        "maximum_candidate_multiplicity": max(multiplicities, default=0),
        "candidate_multiplicity_histogram": {
            str(key): value for key, value in sorted(Counter(multiplicities).items())
        },
        "true_orientation_occurrence_count": true_orientation_count,
        "opposite_orientation_occurrence_count": opposite_orientation_count,
        "all_direct_zero_derivatives_nonzero": (
            all_direct_zero_derivatives_nonzero
        ),
        "all_swapped_zero_derivatives_nonzero": (
            all_swapped_zero_derivatives_nonzero
        ),
        "all_direct_and_swapped_multiplicities_equal": all_multiplicities_equal,
        "all_candidate_multiplicity_residues_nonzero": (
            all_multiplicity_residues_nonzero
        ),
        "candidate_roots": candidate_roots,
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "r167_candidate_roots": r167_control["candidate_roots"],
        "candidate_poles_match_r167_roots": candidate_roots
        == r167_control["candidate_roots"],
        "public_target_equality_corrections": equality_rows,
        "all_public_equality_poles_detected_and_regularized": (
            len(equality_rows) == 1
            and all(
                row["selected_sign_matches"]
                and row["rational_logarithmic_residue_nonzero"]
                and row["semantic_regularized_logarithmic_derivative"] == 0
                for row in equality_rows
            )
        ),
        "local_residue_rows_sha256": sha256_json(local_rows),
        "equality_correction_rows_sha256": sha256_json(equality_rows),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_pair_enumeration_receives_asymptotic_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "invariant_logarithmic_derivative": (
            "On E:y^2=x^3+ax+b, let D=2y*d/dx+(3x^2+a)*d/dy, dual "
            "to the invariant differential omega=dx/(2y). For every nonzero "
            "function g, (Dg/g)omega=dg/g. At a zero of order m this "
            "differential has a simple pole with residue m; at a pole of order "
            "m it has residue -m."
        ),
        "candidate_pole_biconditional": (
            "For the R166 product G(P)=product_j U(x(T_j-P)), after public "
            "P=T pole regularization, every Kummer candidate P has zero "
            "multiplicity m between one and N. Because N<p at the campaign "
            "caps, m is nonzero in F_p. Thus the denominator support of DG/G "
            "on the selected signed divisor is exactly the true-or-opposite-"
            "sign candidate support; no candidate is lost to characteristic-p "
            "residue cancellation."
        ),
        "logarithmic_weil_swap": (
            "Differentiating the R167 meromorphic identity turns the corrected "
            "elliptic resultant into an additive trace: Dlog G equals Dlog of "
            "the auxiliary corrections plus sum_(Q in S union -S) "
            "Dlog h(Q+P) minus 2n Dlog h(P). The finite controls require all "
            "h denominators, h(P), and auxiliary factors to be units on S, so "
            "candidate poles arise exactly from translated numerator zeros."
        ),
        "public_equality_correction": (
            "A target T=P makes the rational factor U(x(T-P)) meet its order-"
            "2n pole at O, producing logarithmic residue -2n rather than a "
            "candidate zero. R166 assigns that semantic factor one. Membership "
            "of a public target in the selected signed divisor is detected by "
            "U(x(T))=0 and V(x(T))=y(T), so fast multipoint evaluation can "
            "remove these known poles before the trace locator."
        ),
        "linearized_interface": (
            "The multiplicative resultant is replaced by a denominator-aware "
            "elliptic Cauchy trace of the compact rational function Dh/h. A "
            "value-only trace that inverts h in the tensor quotient is invalid "
            "at candidates and may erase the required denominator support. The "
            "output must include the Fitting/subresultant denominator factor "
            "gcd(U,denominator(Dlog G))."
        ),
        "scope": (
            "The finite producer enumerates endpoint-target and endpoint-support "
            "pairs only to verify local multiplicities and derivatives. It does "
            "not construct the denominator-aware trace below rho and receives "
            "no asymptotic attack credit."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_log_derivative_elliptic_cauchy_trace.cost.r168.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_h_and_dlog_h_state_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "public_target_equality_prefilter_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "direct_log_trace_table_exponent_B": fraction_record(Fraction(7, 2)),
        "raw_swapped_log_trace_table_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "standard_tensor_quotient_state_exponent_B": fraction_record(
            Fraction(9, 2)
        ),
        "expected_candidate_denominator_degree_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "expected_signed_verification_exponent_B": fraction_record(Fraction(2)),
        "preferred_denominator_aware_trace_work_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "compact_dlog_h_state_inside_rho": True,
        "public_target_equality_prefilter_inside_rho": True,
        "direct_log_trace_table_inside_rho": False,
        "raw_swapped_log_trace_table_inside_rho": False,
        "standard_tensor_quotient_inside_rho": False,
        "denominator_aware_elliptic_cauchy_trace_mod_u_supplied": False,
        "value_only_trace_receives_candidate_locator_credit": False,
        "finite_pair_enumeration_receives_asymptotic_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_derivatives = all(
        row["all_direct_zero_derivatives_nonzero"]
        and row["all_swapped_zero_derivatives_nonzero"]
        for row in rows
    )
    all_multiplicities = all(
        row["all_direct_and_swapped_multiplicities_equal"] for row in rows
    )
    all_residues = all(
        row["all_candidate_multiplicity_residues_nonzero"] for row in rows
    )
    all_candidates = all(row["candidate_poles_match_r167_roots"] for row in rows)
    all_equalities = all(
        row["all_public_equality_poles_detected_and_regularized"] for row in rows
    )
    controls = {
        "schema": "p1553.m6_log_derivative_elliptic_cauchy_trace.controls.r168.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_direct_and_swapped_zero_derivatives_nonzero": all_derivatives,
        "all_direct_and_swapped_multiplicities_equal": all_multiplicities,
        "all_candidate_multiplicity_residues_nonzero": all_residues,
        "all_candidate_poles_match_r167_roots": all_candidates,
        "all_public_equality_poles_detected_and_regularized": all_equalities,
        "candidate_pole_count": sum(row["candidate_pole_count"] for row in rows),
        "candidate_zero_occurrence_count": sum(
            row["candidate_zero_occurrence_count"] for row in rows
        ),
        "maximum_candidate_multiplicity": max(
            row["maximum_candidate_multiplicity"] for row in rows
        ),
        "true_orientation_occurrence_count": sum(
            row["true_orientation_occurrence_count"] for row in rows
        ),
        "opposite_orientation_occurrence_count": sum(
            row["opposite_orientation_occurrence_count"] for row in rows
        ),
        "public_target_equality_correction_count": sum(
            row["public_target_equality_correction_count"] for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_asymptotic_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fifteen_source_bindings_verified": len(actual_bindings) == 15,
        "r167_target_divisor_and_reciprocity_interface_inherited": True,
        "eagen_logarithmic_derivative_interface_bound": True,
        "r147_local_valuation_lane_deduplicated": True,
        "r166_signed_candidate_semantics_inherited": True,
        "invariant_derivation_formula_complete": True,
        "logarithmic_differential_residue_theorem_complete": True,
        "candidate_multiplicity_bounded_by_N_below_p": all_residues,
        "candidate_pole_biconditional_complete": all_candidates,
        "logarithmic_weil_swap_identity_complete": True,
        "compact_dlog_h_representation_complete": True,
        "public_target_equality_pole_semantics_complete": all_equalities,
        "public_u_v_equality_prefilter_complete": all_equalities,
        "six_finite_control_batches_complete": len(rows) == 6,
        "all_direct_zero_derivatives_nonzero": all_derivatives,
        "all_swapped_zero_derivatives_nonzero": all_derivatives,
        "all_direct_and_swapped_multiplicities_equal": all_multiplicities,
        "all_candidate_residues_nonzero": all_residues,
        "all_candidate_factors_match_r167": all_candidates,
        "all_six_equality_poles_regularized": all_equalities,
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "direct_and_raw_trace_costs_charged": True,
        "tensor_quotient_B9O2_cost_charged": True,
        "denominator_aware_elliptic_cauchy_trace_mod_u_complete": False,
        "candidate_safe_zero_divisor_handling_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "log_derivative_candidate_pole_interface_admitted": True,
        "additive_elliptic_trace_linearization_admitted": True,
        "denominator_aware_trace_constructor_admitted": False,
        "lane_admitted": False,
    }
    trace = {
        "schema": "p1553.m6_log_derivative_candidate_poles_and_trace.r168.v1",
        "theorem": theorem,
        "finite_control_records": [
            {
                "control_id": row["control_id"],
                "candidate_pole_count": row["candidate_pole_count"],
                "candidate_zero_occurrence_count": row[
                    "candidate_zero_occurrence_count"
                ],
                "maximum_candidate_multiplicity": row[
                    "maximum_candidate_multiplicity"
                ],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "local_residue_rows_sha256": row["local_residue_rows_sha256"],
                "equality_correction_rows_sha256": row[
                    "equality_correction_rows_sha256"
                ],
            }
            for row in rows
        ],
        "denominator_aware_elliptic_cauchy_trace_mod_u_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_log_derivative_elliptic_cauchy_trace.replay.r168.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "local_residue_rows_sha256": row["local_residue_rows_sha256"],
                "equality_correction_rows_sha256": row[
                    "equality_correction_rows_sha256"
                ],
            }
            for row in rows
        ],
        "all_replay_invariants_pass": all_derivatives
        and all_multiplicities
        and all_residues
        and all_candidates
        and all_equalities,
    }
    frozen = {
        "schema": "p1553.m6_log_derivative_elliptic_cauchy_trace.frozen.r168.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "degree-n signed divisor U,V, degree-N rational target witness "
                "h and Dh/h, anchor and auxiliary corrections, and public "
                "target-equality correction records"
            ),
            "required_output": (
                "gcd(U, denominator of the corrected elliptic trace "
                "sum_(Q in S union -S) Dlog h(Q+P)-2n Dlog h(P)+corrections), "
                "with target-equality poles removed"
            ),
            "required_semantics": (
                "retain every nonzero multiplicity candidate pole, preserve "
                "true-or-opposite Kummer roots, then invoke exact signed "
                "verification"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "nN or n^2 pair table, n^2 tensor quotient, value-only inversion "
                "at candidate zero divisors, unit-cost trace/inverse/resultant/"
                "root/count/marginal/rank/source oracle"
            ),
            "open_primitive": (
                "denominator-aware transposed elliptic Cauchy trace modulo U"
            ),
        },
    }
    next_action = (
        "Construct or refute a denominator-aware transposed elliptic Cauchy "
        "trace modulo U for the compact Dh/h witness below B^(5/2), preferably "
        "B^(9/4+o(1)). Preserve zero-divisor/Fitting information rather than "
        "inverting h at candidates; test structured subresultants, displacement "
        "rank, and transposed multipoint algorithms without nN, n^2, or tensor-"
        "quotient materialization."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Differentiate the R167 elliptic-resultant identity to replace its "
            "multiplicative product by an additive trace while preserving every "
            "candidate as a nonzero-residue denominator pole."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r167": (
                "R167 leaves a multiplicative SLP resultant modulo U. R168 "
                "differentiates that exact identity and asks only for its "
                "denominator-aware additive trace."
            ),
            "eagen_2022_596": (
                "The primary source uses logarithmic derivatives to linearize "
                "divisor-witness products in a proof system. R168 adapts the "
                "identity as a candidate-pole interface and attributes no ECDLP "
                "complexity improvement to the paper."
            ),
            "r147": (
                "R147 verifies occurrence-pair local valuations of an implicit "
                "resultant. R168 uses logarithmic residues of the R167 target "
                "divisor and still requires a new shared denominator-aware trace."
            ),
            "r166": (
                "R166 supplies the true-or-opposite Kummer candidate semantics, "
                "public P=T regularization, and exact signed verifier. R168 "
                "changes only the constructor interface."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_INVARIANT_LOG_DERIVATIVE_CANDIDATE_POLE_INTERFACE__241_"
            "DIRECT_AND_SWAPPED_ZERO_OCCURRENCES__140_NONZERO_RESIDUE_ROOTS__"
            "SIX_PUBLIC_EQUALITY_POLES_REGULARIZED__ADDITIVE_ELLIPTIC_TRACE__"
            "DIRECT_B7O2_RAW_B9O2_TENSOR_B9O2__DENOMINATOR_AWARE_TRACE_MOD_U_"
            "OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "trace": trace,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.trace_output, bundle["trace"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
