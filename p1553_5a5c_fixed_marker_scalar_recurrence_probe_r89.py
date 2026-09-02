#!/usr/bin/env python3
"""Test target-local fixed-marker scalar recurrences after R88."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_fixed_marker_scalar_recurrence.r89.v1"
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4

R88_REPORT = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_"
    "probe_report_r88.json"
)
R88_REPORT_SHA256 = (
    "d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1"
)
R88_GATE = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_probe_gate_r88.md"
)
R88_GATE_SHA256 = (
    "90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56"
)
R87_GATE = pathlib.Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_gate_r87.md"
)
R87_GATE_SHA256 = (
    "16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565"
)
P1536_AUDIT = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-133/p1536_frobenius_projector_norm_jet_audit.md"
)
P1536_AUDIT_SHA256 = (
    "81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393"
)
P1515_TRICHOTOMY = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-098/recursive_s3_local_separator_trichotomy_v1.md"
)
P1515_TRICHOTOMY_SHA256 = (
    "dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a"
)

COLLISION_DECKS_P = [
    [8, 14],
    [0, 12],
    [16, 5],
    [15, 14],
    [14, 10],
]
COLLISION_DECKS_Q = [
    [16, 5],
    [13, 8],
    [12, 1],
    [7, 6],
    [5, 16],
]


def load_r87() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_5a5c_jet_preserving_addition_pushforward_probe_r87.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r87_for_r89", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R87 polynomial controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R87 = load_r87()


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R88_REPORT: R88_REPORT_SHA256,
        R88_GATE: R88_GATE_SHA256,
        R87_GATE: R87_GATE_SHA256,
        P1536_AUDIT: P1536_AUDIT_SHA256,
        P1515_TRICHOTOMY: P1515_TRICHOTOMY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R89 source binding mismatch: {failures}")
    return actual


def endpoint_rows(
    decks: Sequence[Sequence[int]],
    modulus: int,
) -> list[tuple[int, tuple[int, ...]]]:
    rows = []
    for source in itertools.product(
        *(range(len(deck)) for deck in decks)
    ):
        endpoint = sum(
            decks[index][source[index]]
            for index in range(len(decks))
        ) % modulus
        marker_weights = tuple(choice + 1 for choice in source)
        rows.append((endpoint, marker_weights))
    return rows


def fixed_marker_first_jet(
    decks: Sequence[Sequence[int]],
    target: int,
    modulus: int,
) -> tuple[int, ...]:
    rows = endpoint_rows(decks, modulus)
    values = [(target - endpoint) % modulus for endpoint, _ in rows]
    zero_indices = [
        index for index, value in enumerate(values) if value == 0
    ]
    if len(zero_indices) >= 2:
        return (0, 0, 0, 0, 0, 0, 0)
    if len(zero_indices) == 1:
        zero_index = zero_indices[0]
        product = 1
        for index, value in enumerate(values):
            if index != zero_index:
                product = product * value % modulus
        weights = rows[zero_index][1]
        return (
            0,
            product,
            *[
                (-weight * product) % modulus
                for weight in weights
            ],
        )
    norm = 1
    for value in values:
        norm = norm * value % modulus
    inverses = [pow(value, modulus - 2, modulus) for value in values]
    derivative = norm * sum(inverses) % modulus
    markers = [
        (
            -norm
            * sum(
                rows[index][1][slot] * inverses[index]
                for index in range(len(rows))
            )
        ) % modulus
        for slot in range(5)
    ]
    return (norm, derivative, *markers)


def fixed_marker_collision_witness() -> dict[str, Any]:
    modulus = 17
    target = 0
    translated_target = 1
    local_p = fixed_marker_first_jet(
        COLLISION_DECKS_P,
        target,
        modulus,
    )
    local_q = fixed_marker_first_jet(
        COLLISION_DECKS_Q,
        target,
        modulus,
    )
    translated_p = fixed_marker_first_jet(
        COLLISION_DECKS_P,
        translated_target,
        modulus,
    )
    translated_q = fixed_marker_first_jet(
        COLLISION_DECKS_Q,
        translated_target,
        modulus,
    )
    return {
        "field_prime": modulus,
        "deck_sizes": [2, 2, 2, 2, 2],
        "target": target,
        "translated_target": translated_target,
        "decks_p": COLLISION_DECKS_P,
        "decks_q": COLLISION_DECKS_Q,
        "fixed_marker_rule": "source choice index plus one in each slot",
        "local_jet_p": list(local_p),
        "local_jet_q": list(local_q),
        "local_jets_equal": local_p == local_q,
        "local_norm_nonzero": local_p[0] != 0,
        "translated_jet_p": list(translated_p),
        "translated_jet_q": list(translated_q),
        "translated_jets_differ": translated_p != translated_q,
        "theorem": (
            "The fixed seven-scalar local jet is not a congruence for "
            "target translation, even for five equal two-element decks and "
            "a nonzero local norm."
        ),
    }


def marker_polynomials(
    decks: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[list[int], list[list[int]]]:
    rows = endpoint_rows(decks, modulus)
    roots = [endpoint for endpoint, _ in rows]
    norm_poly = R87.poly_from_roots(roots, modulus)
    markers = []
    for slot in range(5):
        weights = [row[1][slot] for row in rows]
        markers.append(
            R87.marker_deformation_poly(roots, weights, modulus)
        )
    return norm_poly, markers


def translation_orbit_controls() -> dict[str, Any]:
    modulus = 1009
    decks = R87.deterministic_decks("R89-C", 5, 2, modulus)
    norm_poly, markers = marker_polynomials(decks, modulus)
    quotient_degree = 16
    quotient_poly = R87.poly_from_roots(
        range(400, 400 + quotient_degree),
        modulus,
    )

    def rank_for(poly: Sequence[int]) -> int:
        degree = len(poly) - 1
        rows = []
        for target in range(degree + 1):
            remainder = R87.poly_mod(
                R87.translated_poly(poly, target, modulus),
                quotient_poly,
                modulus,
            )
            rows.append(
                remainder + [0] * (quotient_degree - len(remainder))
            )
        return R87.matrix_rank(rows, modulus)

    norm_rank = rank_for(norm_poly)
    marker_ranks = [rank_for(marker) for marker in markers]
    return {
        "field_prime": modulus,
        "deck_sizes": [2, 2, 2, 2, 2],
        "norm_polynomial_degree": len(norm_poly) - 1,
        "marker_polynomial_degrees": [
            len(marker) - 1 for marker in markers
        ],
        "quotient_degree": quotient_degree,
        "norm_translation_remainder_rank": norm_rank,
        "marker_translation_remainder_ranks": marker_ranks,
        "all_channels_full_quotient_rank": (
            norm_rank == quotient_degree
            and all(rank == quotient_degree for rank in marker_ranks)
        ),
        "fixed_marker_leading_coefficients_nonzero": all(
            marker[-1] % modulus != 0 for marker in markers
        ),
        "theorem": (
            "Each nonzero degree-d fixed-marker polynomial has a full "
            "translation span in characteristic greater than d; reduction "
            "modulo a lower-degree polynomial therefore fills that quotient."
        ),
    }


def prefix_support_control(
    deck_size: int,
    modulus: int = 1_000_003,
) -> dict[str, Any]:
    decks = R87.deterministic_decks(
        f"R89-SUPPORT-{deck_size}",
        5,
        deck_size,
        modulus,
    )
    supports = {0}
    sizes = []
    occurrence_sizes = []
    for deck in decks:
        supports = {
            (left + right) % modulus
            for left in supports
            for right in deck
        }
        sizes.append(len(supports))
        occurrence_sizes.append(deck_size ** len(sizes))
    return {
        "field_prime": modulus,
        "deck_size": deck_size,
        "prefix_distinct_support_sizes": sizes,
        "prefix_occurrence_sizes": occurrence_sizes,
        "prefix_occupancy_fractions": [
            support / occurrences
            for support, occurrences in zip(sizes, occurrence_sizes)
        ],
        "minimum_prefix_occupancy_fraction": min(
            support / occurrences
            for support, occurrences in zip(sizes, occurrence_sizes)
        ),
        "collision_free_every_prefix": sizes == occurrence_sizes,
    }


def recurrence_cost_ledger() -> dict[str, Any]:
    return {
        "c_deck_size_exponent_B": 3 / 5,
        "five_slot_shift_support_exponents_B": [
            3 / 5,
            6 / 5,
            9 / 5,
            12 / 5,
            15 / 5,
        ],
        "first_setup_cap_crossing_slot": 4,
        "four_slot_support_exponent_B": 12 / 5,
        "five_slot_support_exponent_B": 3.0,
        "target_local_fixed_marker_state_exponent_B": 0.0,
        "target_local_fixed_marker_state_exact_under_translation": False,
        "explicit_shift_recurrence_inside_setup_cap": False,
        "explicit_shift_recurrence_inside_online_cap": False,
        "scope_exception": (
            "a nonlocal nonlinear target-independent translation sketch "
            "whose update and source inverse never expose shift evaluations, "
            "polynomial coefficients, quotient vectors, or endpoint tables"
        ),
    }


def exceptional_controls() -> dict[str, Any]:
    witness = fixed_marker_collision_witness()
    simple_at_shift = (
        witness["translated_jet_p"][0] == 0
        and witness["translated_jet_p"][1] != 0
    )
    multiple_at_shift = all(
        value == 0 for value in witness["translated_jet_q"]
    )
    return {
        "nonzero_local_state_collision": (
            witness["local_jets_equal"]
            and witness["local_norm_nonzero"]
        ),
        "one_translation_is_simple": simple_at_shift,
        "other_translation_is_multiple_or_nonreduced": multiple_at_shift,
        "local_state_cannot_predict_multiplicity_branch": (
            simple_at_shift and multiple_at_shift
        ),
        "actual_projective_semaev_charts_supplied": False,
        "signed_infinity_tangent_replay_complete": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    collision = fixed_marker_collision_witness()
    orbit = translation_orbit_controls()
    supports = [
        prefix_support_control(deck_size)
        for deck_size in (2, 3, 4)
    ]
    costs = recurrence_cost_ledger()
    exceptional = exceptional_controls()
    frozen = {
        "schema": "p1553.frozen_fixed_marker_scalar_recurrence.r89.v1",
        "state": [
            "N(T)",
            "dN/dT",
            "dN/ds_1",
            "dN/ds_2",
            "dN/ds_3",
            "dN/ds_4",
            "dN/ds_5",
        ],
        "marker_rule": "source choice index plus one in each C slot",
        "target_local_state_is_translation_congruence": False,
        "forbidden_materializations": [
            "translated shift-evaluation support",
            "norm or marker polynomial coefficients",
            "B^2 quotient vectors",
            "C endpoint/source tables",
        ],
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
    }
    transition = {
        "schema": (
            "p1553.coefficient_free_marker_transition_ledger.r89.v1"
        ),
        "fixed_marker_collision": collision,
        "translation_orbit": orbit,
        "prefix_support_controls": supports,
        "cost_ledger": costs,
    }
    replay = {
        "schema": "p1553.scalar_norm_marker_source_replay.r89.v1",
        "collision": collision,
        "all_fixed_marker_values_exact": True,
        "source_recovered_without_endpoint_dictionary": False,
        "coefficient_free_recurrence_inside_caps": False,
        "candidate_credit": False,
    }
    exceptional_receipt = {
        "schema": (
            "p1553.projective_multiplicity_exceptional_controls.r89.v1"
        ),
        "controls": exceptional,
        "fixed_local_state_predicts_translation_branch": False,
        "actual_elliptic_source_biconditional_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r89.v1",
        "target_local_fixed_marker_recurrence_found": False,
        "nonlocal_compressed_translation_state_found": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "five_source_bindings_verified": len(bindings) == 5,
        "equal_size_fixed_marker_collision_exact": (
            collision["local_jets_equal"]
            and collision["local_norm_nonzero"]
            and collision["translated_jets_differ"]
        ),
        "all_marker_translation_orbits_full_rank": orbit[
            "all_channels_full_quotient_rank"
        ],
        "three_prefix_support_controls": len(supports) == 3,
        "multiplicity_branch_collision_exact": exceptional[
            "local_state_cannot_predict_multiplicity_branch"
        ],
        "cost_ledger_complete": True,
        "nonlocal_compressed_translation_state_inside_caps": False,
        "coefficient_free_fixed_marker_recurrence_inside_caps": False,
        "source_inverse_without_endpoint_dictionary": False,
        "actual_semaev_projective_exceptional_charts": False,
        "signed_source_biconditional_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "FIXED_MARKER_LOCAL_JET_NONFUNCTORIAL__"
            "EXPLICIT_SHIFT_RECURRENCE_REACHES_C4_B2P4"
        ),
        "source_bindings": {
            "r88_report": {
                "path": str(R88_REPORT),
                "sha256": R88_REPORT_SHA256,
            },
            "r88_gate": {
                "path": str(R88_GATE),
                "sha256": R88_GATE_SHA256,
            },
            "r87_gate": {
                "path": str(R87_GATE),
                "sha256": R87_GATE_SHA256,
            },
            "p1536_norm_jet_audit": {
                "path": str(P1536_AUDIT),
                "sha256": P1536_AUDIT_SHA256,
            },
            "p1515_local_separator_trichotomy": {
                "path": str(P1515_TRICHOTOMY),
                "sha256": P1515_TRICHOTOMY_SHA256,
            },
        },
        "novelty_scope": (
            "R89 removes R87's fixed-marker caveat for target-local state "
            "using an equal-size, nonzero-norm collision. It also verifies "
            "that all six norm/marker translation channels fill the tested "
            "quotient and charges the exact slotwise shift recurrence."
        ),
        "fixed_marker_collision": collision,
        "translation_orbit_control": orbit,
        "prefix_support_controls": supports,
        "recurrence_cost_ledger": costs,
        "exceptional_controls": exceptional,
        "side_artifacts": {
            "frozen": "frozen_5a5c_fixed_marker_scalar_recurrence.json",
            "transition": "coefficient_free_marker_transition_ledger.json",
            "replay": "scalar_norm_marker_and_source_replay.json",
            "exceptional": (
                "projective_multiplicity_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r89.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "This closes the seven-scalar target-local fixed-marker state "
            "and explicit shifted-evaluation recurrence only. It does not "
            "lower-bound a nonlocal nonlinear translation sketch, a "
            "non-Krylov arithmetic circuit, or a representation-changing "
            "elliptic/FFE identity."
        ),
        "next_action": (
            "Construct or refute one nonlocal nonlinear translation sketch "
            "for the fixed norm/marker family. Freeze its target-independent "
            "state and deck-update law; prove state at most B^(9/4), fresh "
            "translation and source return at most B^(5/4), no shift-value "
            "or coefficient/quotient table, and exact reduced, multiple, "
            "nonreduced, signed, infinity, tangent, and exceptional replay."
        ),
        "disposition": (
            "REJECT_TARGET_LOCAL_FIXED_MARKER_AND_EXPLICIT_SHIFT_RECURRENCE_"
            "ONLY__EQUAL_SIZE_NONZERO_JET_COLLISION__TRANSLATED_SIMPLE_"
            "VERSUS_MULTIPLE_BRANCH__NORM_AND_FIVE_MARKER_TRANSLATION_"
            "ORBITS_FULL_QUOTIENT_RANK__C4_SHIFT_SUPPORT_B2P4__C5_B3__"
            "NONLOCAL_NONLINEAR_TRANSLATION_SKETCH_OPEN__NO_RANK__NO_FACTOR_"
            "LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "transition": transition,
        "replay": replay,
        "exceptional": exceptional_receipt,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_fixed_marker_scalar_recurrence_probe_report_r89.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_fixed_marker_scalar_recurrence.json"
        ),
    )
    parser.add_argument(
        "--transition-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "coefficient_free_marker_transition_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "scalar_norm_marker_and_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "projective_multiplicity_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r89.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.transition_output, bundle["transition"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
