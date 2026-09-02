#!/usr/bin/env python3
"""Audit character-diagonal scalar count circuits for the 5A+5C fiber."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_scalar_target_norm_count_circuit.r106.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
GROUP_EXPONENT = Fraction(5, 1)

R105_PRODUCER = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_r105.py"
)
R105_PRODUCER_SHA256 = (
    "d7215def9fa82b01ec72ea43f3d297fd1343d2ceae9f5d364a351326c9f9c6b8"
)
R105_REPORT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_"
    "target_pullback_probe_report_r105.json"
)
R105_REPORT_SHA256 = (
    "c00b8c3e934e783012acb1dab83282f98720702bb8cd81a706fbdf4671f93a83"
)
R105_GATE = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_gate_r105.md"
)
R105_GATE_SHA256 = (
    "e349dc695975b3d5ce923ed334c61c6329884bbc30228f7bf52fb1c8413ea3ef"
)
R105_PARENT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_"
    "target_pullback_probe_parent_report_r105.yaml"
)
R105_PARENT_SHA256 = (
    "34ed83a9fbb47048734f9a14f0311fc272761b747da9b12fa66e110ab8daa3ad"
)
R77_REPORT = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_report_r77.json"
)
R77_REPORT_SHA256 = (
    "73f66184fa53a2d43397a915c4249a41c5687cbcd1d5d16cc3e4ff47cf254787"
)
R77_GATE = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_gate_r77.md"
)
R77_GATE_SHA256 = (
    "45324f816cc159032cb6c2ac97c0a2f52a618c409e276616521b4e894cb46b55"
)
R91_REPORT = pathlib.Path(
    "p1553_5a5c_unequal_list_subfunction_"
    "inversion_probe_report_r91.json"
)
R91_REPORT_SHA256 = (
    "6bfaaaa72dd1135f8244a70a0ce2a6697e5f3fe63ce8ac0ce107bd05f19a71da"
)
R91_GATE = pathlib.Path(
    "p1553_5a5c_unequal_list_subfunction_inversion_probe_gate_r91.md"
)
R91_GATE_SHA256 = (
    "57469654e0b535ccbf1d62edd9548cd827a09d244deaecb06b8aedbd410ac6d3"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R105_PRODUCER: R105_PRODUCER_SHA256,
        R105_REPORT: R105_REPORT_SHA256,
        R105_GATE: R105_GATE_SHA256,
        R105_PARENT: R105_PARENT_SHA256,
        R77_REPORT: R77_REPORT_SHA256,
        R77_GATE: R77_GATE_SHA256,
        R91_REPORT: R91_REPORT_SHA256,
        R91_GATE: R91_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R106 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R105 = load_module("p1553_r105_for_r106", R105_PRODUCER)
R104 = R105.R104
R102 = R105.R102
R82 = R105.R82
R70 = R105.R70


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def tao_uncertainty_control(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    a5, _a_stages = R104.multiset_group_algebra(
        atoms_a,
        R102.FULL_A_COUNT,
        curve,
    )
    c5, _c_stages = R104.multiset_group_algebra(
        atoms_c,
        R102.FULL_C_COUNT,
        curve,
    )
    order = curve["subgroup_order"]
    support_a = len(a5)
    support_c = len(c5)
    fourier_a_lower = order + 1 - support_a
    fourier_c_lower = order + 1 - support_c
    full_spectrum_lower = order - support_a - support_c + 2
    if full_spectrum_lower <= 0:
        raise AssertionError("uncertainty lower bound became vacuous")
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "subgroup_order_q": order,
        "subgroup_order_prime": R70.is_prime(order),
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "a5_endpoint_support": support_a,
        "c5_endpoint_support": support_c,
        "a5_fourier_support_lower_bound": fourier_a_lower,
        "c5_fourier_support_lower_bound": fourier_c_lower,
        "full_count_fourier_support_lower_bound": full_spectrum_lower,
        "full_count_live_mode_fraction_lower_bound": (
            full_spectrum_lower / order
        ),
        "zero_mode_union_upper_bound": support_a + support_c - 2,
        "bound_identity": (
            "supp(hat(H_A*H_C)) >= "
            "q-|supp(H_A)|-|supp(H_C)|+2"
        ),
        "scalar_labels_consumed_by_candidate": False,
        "verifier_dlp_labels_needed_for_mode_coordinates": True,
        "candidate_mode_constructor_supplied": False,
    }


@functools.lru_cache(maxsize=1)
def actual_and_matched_controls() -> dict[str, Any]:
    actual = [
        tao_uncertainty_control(dict(family), offset, "actual")
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        tao_uncertainty_control(dict(family), offset, "matched_random_deck")
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "actual_count": len(actual),
        "matched_random_deck_count": len(matched),
        "all_subgroup_orders_prime": all(
            row["subgroup_order_prime"] for row in rows
        ),
        "all_uncertainty_bounds_positive": all(
            row["full_count_fourier_support_lower_bound"] > 0
            for row in rows
        ),
        "all_live_mode_fractions_above_99_percent": all(
            row["full_count_live_mode_fraction_lower_bound"] > 0.99
            for row in rows
        ),
        "minimum_live_mode_fraction_lower_bound": min(
            row["full_count_live_mode_fraction_lower_bound"]
            for row in rows
        ),
        "all_scalar_blind_candidate_inputs": all(
            not row["scalar_labels_consumed_by_candidate"]
            for row in rows
        ),
    }


def composite_order_positive_control() -> dict[str, Any]:
    return {
        "group": "Z/16Z",
        "function": "indicator of the even subgroup 2Z/16Z",
        "primal_support_size": 8,
        "fourier_support": [0, 8],
        "fourier_support_size": 2,
        "support_sum": 10,
        "prime_order_sharp_threshold_if_misapplied": 17,
        "violates_prime_order_additive_bound": 10 < 17,
        "role": (
            "Confirms that the q+1 support theorem depends on prime cyclic "
            "order and does not reject genuine subgroup-sparse spectra."
        ),
    }


def character_density_theorem() -> dict[str, Any]:
    return {
        "primary_source": {
            "title": (
                "An uncertainty principle for cyclic groups of prime order"
            ),
            "author": "Terence Tao",
            "arxiv": "math/0308286",
            "url": "https://arxiv.org/abs/math/0308286",
            "published": "Mathematical Research Letters 12 (2005), no. 1",
        },
        "theorem": (
            "For nonzero f on Z/qZ with q prime, "
            "|supp(f)|+|supp(hat(f))| >= q+1 over the complex Fourier field."
        ),
        "side_functions": (
            "H_A and H_C are the canonical five-multiset endpoint "
            "histograms on the prime cyclic subgroup"
        ),
        "full_count": "H=H_A*H_C",
        "fourier_identity": "hat(H)=hat(H_A)*hat(H_C) pointwise",
        "zero_union_argument": (
            "hat(H_A) has at most |supp(H_A)|-1 zero modes and hat(H_C) "
            "has at most |supp(H_C)|-1; the product zero set is their union"
        ),
        "lower_bound": (
            "|supp(hat(H))| >= "
            "q-|supp(H_A)|-|supp(H_C)|+2"
        ),
        "asymptotic_inputs": {
            "q_exponent_B": fraction_record(GROUP_EXPONENT),
            "a5_support_upper_exponent_B": fraction_record(Fraction(2)),
            "c5_support_upper_exponent_B": fraction_record(Fraction(3)),
        },
        "live_mode_exponent_B": fraction_record(GROUP_EXPONENT),
        "character_state_inside_setup_cap": GROUP_EXPONENT <= SETUP_CAP,
        "dense_character_sum_inside_online_cap": (
            GROUP_EXPONENT <= ONLINE_CAP
        ),
        "scope": (
            "Exact for explicit complex-character diagonalizations and "
            "equivalent mode tables on the prime cyclic subgroup. It is not "
            "an arithmetic-circuit lower bound, does not cover finite-field "
            "representations where the required Fourier minors vanish, and "
            "does not reject target-injected nonlinear resultants or norms "
            "that never expose character modes."
        ),
    }


def cost_ledger() -> dict[str, Any]:
    theorem = character_density_theorem()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "explicit_character_diagonalization": {
            "live_mode_exponent_B": theorem["live_mode_exponent_B"],
            "mode_state_inside_setup_cap": theorem[
                "character_state_inside_setup_cap"
            ],
            "per_target_dense_character_sum_inside_online_cap": theorem[
                "dense_character_sum_inside_online_cap"
            ],
            "all_target_fft_exponent_B": fraction_record(GROUP_EXPONENT),
            "requires_endpoint_discrete_log_coordinates": True,
            "candidate_dlp_coordinates_available": False,
        },
        "marker_channels": {
            "r105_unit_and_power_sum_channel_count": 11,
            "constant_channel_count_changes_mode_exponent": False,
        },
        "constructor_status": {
            "character_diagonal_scalar_count_inside_caps": False,
            "noncharacter_resultant_or_norm_inside_caps": False,
            "whole_deck_scalar_target_norm_complete": False,
        },
        "fatal_obstruction": (
            "the canonical actual side supports force B^(5-o(1)) live "
            "complex-character modes before source markers are considered"
        ),
        "scope_exception": theorem["scope"],
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r105_producer": {
            "path": str(R105_PRODUCER),
            "sha256": R105_PRODUCER_SHA256,
        },
        "r105_report": {
            "path": str(R105_REPORT),
            "sha256": R105_REPORT_SHA256,
        },
        "r105_gate": {
            "path": str(R105_GATE),
            "sha256": R105_GATE_SHA256,
        },
        "r105_parent": {
            "path": str(R105_PARENT),
            "sha256": R105_PARENT_SHA256,
        },
        "r77_report": {
            "path": str(R77_REPORT),
            "sha256": R77_REPORT_SHA256,
        },
        "r77_gate": {
            "path": str(R77_GATE),
            "sha256": R77_GATE_SHA256,
        },
        "r91_report": {
            "path": str(R91_REPORT),
            "sha256": R91_REPORT_SHA256,
        },
        "r91_gate": {
            "path": str(R91_GATE),
            "sha256": R91_GATE_SHA256,
        },
        "r82_report": {
            "path": str(R82_REPORT),
            "sha256": R82_REPORT_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_and_matched_controls()
    positive = composite_order_positive_control()
    theorem = character_density_theorem()
    costs = cost_ledger()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_scalar_target_norm_count_circuit.r106.v1"
        ),
        "candidate_grammar": (
            "exact character-diagonal evaluation of the canonical H_A*H_C "
            "target-count function, including equivalent explicit mode tables"
        ),
        "target_injection": "character phase chi(-T)",
        "marker_channels": "R105 unit plus ten power-sum deformations",
        "prime_cyclic_group_required": True,
        "caps": costs["caps"],
        "excluded_open_operation": (
            "a target-injected nonlinear resultant, norm, or rational "
            "straight-line program that never exposes character modes"
        ),
    }
    circuit_ledger = {
        "schema": "p1553.scalar_target_norm_gate_cost_ledger.r106.v1",
        "frozen_candidate": frozen,
        "character_density_theorem": theorem,
        "actual_and_matched_controls": controls,
        "composite_order_positive_control": positive,
        "costs": costs,
    }
    marker_replay = {
        "schema": "p1553.scalar_target_norm_marker_jet_replay.r106.v1",
        "r105_marker_adjoint_reused_conditionally": True,
        "unit_count_channel_live_mode_exponent_B": theorem[
            "live_mode_exponent_B"
        ],
        "constant_marker_channels_change_exponent": False,
        "character_constructor_inside_caps": False,
        "no_source_replay_credit": True,
    }
    exceptional = {
        "schema": "p1553.scalar_target_norm_exceptional_controls.r106.v1",
        "all_actual_and_matched_subgroups_prime": controls[
            "all_subgroup_orders_prime"
        ],
        "composite_order_sparse_spectrum_control": positive,
        "complex_fourier_field_scope_explicit": True,
        "finite_field_fourier_minor_scope_complete": False,
        "noncharacter_projective_resultant_charts_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r106.v1",
        "character_diagonal_scalar_constructor_inside_caps": False,
        "noncharacter_scalar_constructor_supplied": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "nine_source_bindings_verified": len(bindings) == 9,
        "eight_actual_decks_replayed": controls["actual_count"] == 8,
        "eight_matched_random_decks_replayed": (
            controls["matched_random_deck_count"] == 8
        ),
        "all_subgroup_orders_prime": controls["all_subgroup_orders_prime"],
        "tao_prime_cyclic_uncertainty_theorem_bound": True,
        "canonical_side_histogram_supports_exact": True,
        "fourier_zero_union_bound_exact": True,
        "all_finite_live_mode_bounds_positive": controls[
            "all_uncertainty_bounds_positive"
        ],
        "all_finite_live_mode_fractions_above_99_percent": controls[
            "all_live_mode_fractions_above_99_percent"
        ],
        "composite_order_sparse_spectrum_positive_control": positive[
            "violates_prime_order_additive_bound"
        ],
        "live_mode_exponent_B5": (
            theorem["live_mode_exponent_B"]["exact"] == "5"
        ),
        "character_mode_state_inside_setup_cap": theorem[
            "character_state_inside_setup_cap"
        ],
        "dense_character_sum_inside_online_cap": theorem[
            "dense_character_sum_inside_online_cap"
        ],
        "candidate_dlp_coordinates_available": False,
        "eleven_marker_channels_preserve_mode_exponent": True,
        "character_diagonal_scalar_count_inside_caps": False,
        "noncharacter_scalar_target_norm_supplied": False,
        "finite_field_fourier_minor_scope_complete": False,
        "generic_multiplicity_and_integer_lift_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "full_pipeline_fresh_workspace_inside_cap": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "TAO_PRIME_CYCLIC_UNCERTAINTY_FORCES_B5_LIVE_CHARACTER_MODES__"
            "ACTUAL_AND_MATCHED_SIDE_SUPPORTS_EXACT__COMPOSITE_SUBGROUP_"
            "SPARSE_SPECTRUM_CONTROL_PASSES__CHARACTER_DIAGONAL_SCALAR_"
            "CONSTRUCTOR_OVER_CAP__NONCHARACTER_RESULTANT_NORM_CIRCUIT_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "novelty_scope": (
            "R106 is the first campaign receipt to apply the sharp prime-"
            "cyclic additive uncertainty theorem directly to the canonical "
            "five-multiset A/C endpoint histograms, proving an actual-family "
            "B^5 live-mode bound rather than an ambient arbitrary-vector rank."
        ),
        "character_density_theorem": theorem,
        "actual_and_matched_controls": controls,
        "composite_order_positive_control": positive,
        "cost_ledger": costs,
        "artifacts": {
            "frozen": "frozen_5a5c_scalar_target_norm_count_circuit.json",
            "circuit_ledger": "scalar_target_norm_gate_and_cost_ledger.json",
            "marker_replay": "scalar_target_norm_marker_jet_replay.json",
            "exceptional": "scalar_target_norm_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r106.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "character_diagonal_constructor_closed": True,
        "noncharacter_scalar_constructor_open": True,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": theorem["scope"],
        "next_action": (
            "Construct or refute one non-character target-injected algebraic "
            "norm/resultant circuit for the compact atom divisors. Freeze "
            "the allowed nested resultants, quotient-free transposition, "
            "integer lift, and R105 marker deformations; forbid character/"
            "DLP coordinates, B^(14/5) residuals, B^(11/5) per-target "
            "coefficient reads, B^5 Macaulay/source bodies, and unit-cost "
            "determinants; require both caps and full exceptional replay."
        ),
        "disposition": (
            "REJECT_CHARACTER_DIAGONAL_SCALAR_TARGET_COUNT_ONLY__TAO_"
            "UNCERTAINTY_GIVES_Q_MINUS_A5_MINUS_C5_PLUS_2_LIVE_MODES__"
            "ASYMPTOTIC_B5_MODE_BODY__ACTUAL_AND_MATCHED_CONTROLS_EXACT__"
            "COMPOSITE_POSITIVE_CONTROL_PASSES__NONCHARACTER_RESULTANT_"
            "NORM_CIRCUIT_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_"
            "SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "circuit_ledger": circuit_ledger,
        "marker_replay": marker_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_scalar_target_norm_count_"
            "circuit_probe_report_r106.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_scalar_target_norm_count_circuit.json"
        ),
    )
    parser.add_argument(
        "--circuit-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "scalar_target_norm_gate_and_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--marker-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "scalar_target_norm_marker_jet_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "scalar_target_norm_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r106.json"
        ),
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
    write_json(args.circuit_output, bundle["circuit_ledger"])
    write_json(args.marker_output, bundle["marker_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        "R106 "
        f"classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
