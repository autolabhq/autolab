#!/usr/bin/env python3
"""Audit black-box translated-resultant localization after R87."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from collections import Counter
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_black_box_resultant_localizer.r88.v1"
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4

R87_REPORT = pathlib.Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_report_r87.json"
)
R87_REPORT_SHA256 = (
    "f10ba663867815c9ee0b1234f4d9dee698d450a3a7171336d36f3e328ea2333a"
)
R87_GATE = pathlib.Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_gate_r87.md"
)
R87_GATE_SHA256 = (
    "16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565"
)
R80_GATE = pathlib.Path(
    "p1553_batched_nested_norm_node_compiler_probe_gate_r80.md"
)
R80_GATE_SHA256 = (
    "1ff3641688f4f0e13fd64f83aa540ea429a4164e0d0741b64b38acd804d7fb01"
)
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)
P1536_AUDIT = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-133/p1536_frobenius_projector_norm_jet_audit.md"
)
P1536_AUDIT_SHA256 = (
    "81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393"
)


def load_r87() -> Any:
    path = pathlib.Path(__file__).with_name(
        "p1553_5a5c_jet_preserving_addition_pushforward_probe_r87.py"
    )
    spec = importlib.util.spec_from_file_location("p1553_r87_for_r88", path)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load R87 controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R87 = load_r87()


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R87_REPORT: R87_REPORT_SHA256,
        R87_GATE: R87_GATE_SHA256,
        R80_GATE: R80_GATE_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
        P1536_AUDIT: P1536_AUDIT_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R88 source binding mismatch: {failures}")
    return actual


def frozen_scalar_instance() -> dict[str, Any]:
    prime = 1_000_003
    a_decks = R87.deterministic_decks("A", 5, 2, prime)
    c_decks = R87.deterministic_decks("C", 5, 3, prime)
    a_endpoints, a_histogram, a_sources = R87.endpoint_table(
        a_decks,
        prime,
    )
    c_endpoints, c_histogram, c_sources = R87.endpoint_table(
        c_decks,
        prime,
    )
    target_histogram: Counter[int] = Counter()
    target_first: dict[int, tuple[int, int]] = {}
    for a_endpoint, a_multiplicity in a_histogram.items():
        for c_endpoint, c_multiplicity in c_histogram.items():
            target = (a_endpoint + c_endpoint) % prime
            target_histogram[target] += a_multiplicity * c_multiplicity
            target_first.setdefault(target, (a_endpoint, c_endpoint))
    target = next(
        value
        for value in sorted(target_histogram)
        if target_histogram[value] == 1
    )
    expected_a, expected_c = target_first[target]
    return {
        "prime": prime,
        "a_decks": a_decks,
        "c_decks": c_decks,
        "a_endpoints": a_endpoints,
        "a_histogram": a_histogram,
        "a_sources": a_sources,
        "c_endpoints": c_endpoints,
        "c_histogram": c_histogram,
        "c_sources": c_sources,
        "target": target,
        "target_multiplicity": target_histogram[target],
        "expected_a": expected_a,
        "expected_c": expected_c,
        "poly_a": R87.poly_from_roots(a_endpoints, prime),
        "poly_c": R87.poly_from_roots(c_endpoints, prime),
    }


def subset_zero_test(
    a_subset: Sequence[int],
    poly_c: Sequence[int],
    target: int,
    prime: int,
) -> bool:
    product = 1
    for endpoint in a_subset:
        product = (
            product * R87.poly_eval(poly_c, target - endpoint, prime)
        ) % prime
    return product == 0


def binary_oracle_localize(
    a_endpoints: Sequence[int],
    poly_c: Sequence[int],
    target: int,
    prime: int,
) -> dict[str, Any]:
    current = list(a_endpoints)
    calls = 1
    root_positive = subset_zero_test(
        current,
        poly_c,
        target,
        prime,
    )
    path = [
        {
            "subset_size": len(current),
            "zero": root_positive,
            "role": "acceptance",
        }
    ]
    if not root_positive:
        return {
            "accepted": False,
            "oracle_calls": calls,
            "localized_endpoint": None,
            "path": path,
        }
    while len(current) > 1:
        midpoint = (len(current) + 1) // 2
        left = current[:midpoint]
        right = current[midpoint:]
        left_positive = subset_zero_test(
            left,
            poly_c,
            target,
            prime,
        )
        calls += 1
        path.append(
            {
                "subset_size": len(left),
                "zero": left_positive,
                "role": "left_child",
            }
        )
        current = left if left_positive else right
    return {
        "accepted": True,
        "oracle_calls": calls,
        "localized_endpoint": current[0],
        "path": path,
    }


def conditional_oracle_source_replay() -> dict[str, Any]:
    instance = frozen_scalar_instance()
    prime = instance["prime"]
    localization = binary_oracle_localize(
        instance["a_endpoints"],
        instance["poly_c"],
        instance["target"],
        prime,
    )
    recovered_a = localization["localized_endpoint"]
    recovered_c = (instance["target"] - recovered_a) % prime
    a_source = instance["a_sources"][recovered_a]
    c_source = instance["c_sources"][recovered_c]
    replay = (
        sum(
            instance["a_decks"][index][choice]
            for index, choice in enumerate(a_source)
        )
        + sum(
            instance["c_decks"][index][choice]
            for index, choice in enumerate(c_source)
        )
    ) % prime
    expected_calls = 1 + math.ceil(
        math.log2(len(instance["a_endpoints"]))
    )
    return {
        "field_prime": prime,
        "a_endpoint_occurrences": len(instance["a_endpoints"]),
        "c_endpoint_occurrences": len(instance["c_endpoints"]),
        "target": instance["target"],
        "target_multiplicity": instance["target_multiplicity"],
        "oracle_localization": localization,
        "expected_acceptance_plus_binary_calls": expected_calls,
        "logarithmic_call_count_exact": (
            localization["oracle_calls"] == expected_calls
        ),
        "expected_a_endpoint": instance["expected_a"],
        "recovered_a_endpoint": recovered_a,
        "recovered_c_endpoint": recovered_c,
        "a_source": list(a_source),
        "c_source": list(c_source),
        "joint_source_replay": replay == instance["target"],
        "a_subproduct_source_dictionary_words": len(
            instance["a_endpoints"]
        ),
        "c_source_dictionary_words": len(instance["c_endpoints"]),
        "zero_test_reads_materialized_c_polynomial": True,
        "c_source_dictionary_enumerated": True,
        "oracle_constructor_supplied": False,
        "candidate_credit": False,
    }


def quotient_krylov_control() -> dict[str, Any]:
    instance = frozen_scalar_instance()
    prime = instance["prime"]
    eigenvalues = [
        R87.poly_eval(
            instance["poly_c"],
            instance["target"] - endpoint,
            prime,
        )
        for endpoint in instance["a_endpoints"]
    ]
    dimension = len(eigenvalues)
    sequence = [
        sum(pow(value, power, prime) for value in eigenvalues) % prime
        for power in range(2 * dimension)
    ]
    hankel = [
        sequence[row + column]
        for row in range(dimension)
        for column in range(dimension)
    ]
    hankel_rows = [
        hankel[index * dimension : (index + 1) * dimension]
        for index in range(dimension)
    ]
    rank = R87.matrix_rank(hankel_rows, prime)
    return {
        "field_prime": prime,
        "quotient_dimension_m": dimension,
        "zero_eigenvalue_count": eigenvalues.count(0),
        "distinct_eigenvalue_count": len(set(eigenvalues)),
        "multiplication_operator_nullity": eigenvalues.count(0),
        "scalar_krylov_hankel_rank": rank,
        "scalar_krylov_full_linear_complexity": rank == dimension,
        "berlekamp_massey_sequence_terms": 2 * dimension,
        "sequence_sha256": hashlib.sha256(
            json.dumps(sequence, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "verifier_diagonal_eigenvalues_consumed": True,
        "candidate_credit": False,
    }


def block_krylov_tradeoff() -> dict[str, Any]:
    points = []
    for width_exponent in (0.0, 0.25, 0.5, 0.75, 1.0):
        iteration_exponent = 2.0 - width_exponent
        state_exponent = 2.0 + width_exponent
        points.append(
            {
                "block_width_exponent_B": width_exponent,
                "optimistic_iteration_exponent_B": iteration_exponent,
                "stored_block_state_exponent_B": state_exponent,
                "inside_setup_cap": (
                    state_exponent <= SETUP_CAP_EXPONENT
                ),
                "inside_online_iteration_cap": (
                    iteration_exponent <= ONLINE_CAP_EXPONENT
                ),
                "both_caps": (
                    state_exponent <= SETUP_CAP_EXPONENT
                    and iteration_exponent <= ONLINE_CAP_EXPONENT
                ),
            }
        )
    return {
        "quotient_dimension_exponent_B": 2.0,
        "model": (
            "block width B^alpha, optimistic iterations B^(2-alpha), "
            "stored quotient blocks B^(2+alpha)"
        ),
        "points": points,
        "minimum_alpha_for_online_cap": 0.75,
        "maximum_alpha_for_setup_cap": 0.25,
        "feasible_alpha_interval_empty": True,
        "scope": (
            "standard scalar/block Krylov with explicit quotient blocks; "
            "matvec cost is optimistically treated as unit"
        ),
    }


def operation_ledger() -> dict[str, Any]:
    return {
        "a_characteristic_degree_exponent_B": 2.0,
        "c_characteristic_degree_exponent_B": 3.0,
        "translated_remainder_exponent_B": 2.0,
        "standard_routes": {
            "materialized_c_polynomial": {
                "setup_state_exponent_B": 3.0,
                "inside_setup_cap": False,
                "owner": "R80/P1513/R87",
            },
            "coefficient_half_gcd": {
                "c_input_exponent_B": 3.0,
                "remainder_exponent_B": 2.0,
                "inside_caps": False,
                "owner": "R80/P1513/R87",
            },
            "scalar_krylov": {
                "quotient_dimension_exponent_B": 2.0,
                "optimistic_scalar_sample_exponent_B": 2.0,
                "inside_online_cap": False,
            },
            "explicit_nested_c_norms": {
                "slot_support_exponents_B": [0.6, 1.2, 1.8, 2.4, 3.0],
                "first_setup_cap_crossing_slot": 4,
                "inside_setup_cap": False,
            },
            "conditional_subproduct_localizer": {
                "oracle_call_exponent_B": 0.0,
                "oracle_calls": "1+ceil(log2(B^2))",
                "source_output_exact_if_unique": True,
                "zero_test_constructor_omitted": True,
            },
        },
        "standard_black_box_and_half_gcd_inside_caps": False,
        "scope_exception": (
            "a coefficient-free structured marked-resultant identity that "
            "returns zero/multiplicity and the five C source markers as "
            "scalars without quotient vectors, Krylov blocks, or a C table"
        ),
    }


def exceptional_controls() -> dict[str, Any]:
    prime = 101
    poly_a = R87.poly_from_roots([1, 2], prime)
    poly_c = R87.poly_from_roots([4, 5], prime)
    multiple_target = 6
    multiple_norm_jet = R87.convolution_first_jet(
        poly_c,
        [1, 2],
        multiple_target,
        prime,
    )
    localized_multiple = binary_oracle_localize(
        [1, 2],
        poly_c,
        multiple_target,
        prime,
    )
    nonreduced_c = R87.poly_from_roots([5, 5], prime)
    nonreduced_norm_jet = R87.convolution_first_jet(
        nonreduced_c,
        [1],
        6,
        prime,
    )
    empty_target = 20
    empty = binary_oracle_localize(
        [1, 2],
        poly_c,
        empty_target,
        prime,
    )
    return {
        "empty": {
            "target": empty_target,
            "accepted": empty["accepted"],
            "correctly_rejected": not empty["accepted"],
        },
        "multiple_distinct": {
            "target": multiple_target,
            "scalar_norm_first_jet": list(multiple_norm_jet),
            "binary_localizer_returns_one_endpoint": (
                localized_multiple["localized_endpoint"]
            ),
            "scalar_zero_oracle_alone_rejects_multiple": False,
            "first_jet_rejects_unique_branch": multiple_norm_jet == (0, 0),
        },
        "nonreduced": {
            "target": 6,
            "scalar_norm_first_jet": list(nonreduced_norm_jet),
            "scalar_zero_oracle_alone_rejects_nonreduced": False,
            "first_jet_rejects_unique_branch": (
                nonreduced_norm_jet == (0, 0)
            ),
        },
        "marked_multiplicity_oracle_supplied": False,
        "actual_projective_semaev_charts_supplied": False,
        "signed_and_infinity_replay_complete": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    replay = conditional_oracle_source_replay()
    krylov = quotient_krylov_control()
    block = block_krylov_tradeoff()
    ledger = operation_ledger()
    exceptional = exceptional_controls()
    frozen = {
        "schema": "p1553.frozen_5a5c_black_box_resultant.r88.v1",
        "left_input": "P_A(X), degree B^2, source dictionary allowed",
        "right_input": "implicit five-slot C divisor, degree B^3 eliminant",
        "query": "fresh T",
        "required_scalar_outputs": [
            "zero or nonzero",
            "simple versus multiple or nonreduced",
            "five fixed source markers",
        ],
        "forbidden_materializations": [
            "P_C coefficient vector",
            "P_C(T-X) mod P_A(X)",
            "translated C evaluation vector on A endpoints",
            "B^2 quotient Krylov block",
            "C endpoint/source dictionary",
        ],
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
    }
    operation_receipt = {
        "schema": (
            "p1553.implicit_resultant_half_gcd_operation_ledger.r88.v1"
        ),
        "ledger": ledger,
        "quotient_krylov_control": krylov,
        "block_krylov_tradeoff": block,
    }
    source_receipt = {
        "schema": (
            "p1553.target_specialization_source_localization.r88.v1"
        ),
        "conditional_oracle_replay": replay,
        "logarithmic_source_localization_exact": (
            replay["logarithmic_call_count_exact"]
            and replay["joint_source_replay"]
        ),
        "public_zero_test_and_marker_constructor_inside_caps": False,
        "candidate_credit": False,
    }
    exceptional_receipt = {
        "schema": (
            "p1553.multiplicity_projective_exceptional_controls.r88.v1"
        ),
        "controls": exceptional,
        "scalar_empty_control_exact": exceptional["empty"][
            "correctly_rejected"
        ],
        "multiple_and_nonreduced_require_first_jet": (
            exceptional["multiple_distinct"][
                "first_jet_rejects_unique_branch"
            ]
            and exceptional["nonreduced"][
                "first_jet_rejects_unique_branch"
            ]
        ),
        "actual_elliptic_source_biconditional_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r88.v1",
        "conditional_oracle_source_localizer_exact": True,
        "public_black_box_resultant_constructor_inside_caps": False,
        "fixed_marker_source_output_inside_caps": False,
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
        "conditional_logarithmic_oracle_localizer_exact": (
            replay["logarithmic_call_count_exact"]
            and replay["joint_source_replay"]
        ),
        "scalar_krylov_full_linear_complexity": (
            krylov["scalar_krylov_full_linear_complexity"]
        ),
        "block_krylov_cap_interval_empty": block[
            "feasible_alpha_interval_empty"
        ],
        "standard_operation_ledger_complete": True,
        "empty_multiple_nonreduced_scalar_controls": (
            exceptional["empty"]["correctly_rejected"]
            and exceptional["multiple_distinct"][
                "first_jet_rejects_unique_branch"
            ]
            and exceptional["nonreduced"][
                "first_jet_rejects_unique_branch"
            ]
        ),
        "public_coefficient_free_zero_test_inside_caps": False,
        "fixed_marker_source_output_inside_caps": False,
        "standard_black_box_or_half_gcd_inside_caps": False,
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
            "ORACLE_SOURCE_LOCALIZER_LOGARITHMIC__"
            "SCALAR_BLOCK_KRYLOV_OR_HALF_GCD_OVER_CAP"
        ),
        "source_bindings": {
            "r87_report": {
                "path": str(R87_REPORT),
                "sha256": R87_REPORT_SHA256,
            },
            "r87_gate": {
                "path": str(R87_GATE),
                "sha256": R87_GATE_SHA256,
            },
            "r80_batched_gcd_gate": {
                "path": str(R80_GATE),
                "sha256": R80_GATE_SHA256,
            },
            "p1513_direct_ku_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
            "p1536_norm_jet_audit": {
                "path": str(P1536_AUDIT),
                "sha256": P1536_AUDIT_SHA256,
            },
        },
        "novelty_scope": (
            "R88 separates source localization from zero-test construction "
            "on the R87 split. It adds an exact logarithmic conditional "
            "localizer and a scalar/block-Krylov cap tradeoff without "
            "reclaiming R80's materialized gcd or P1513's standard KU/norm "
            "routes."
        ),
        "conditional_oracle_source_replay": replay,
        "quotient_krylov_control": krylov,
        "block_krylov_tradeoff": block,
        "operation_ledger": ledger,
        "exceptional_controls": exceptional,
        "side_artifacts": {
            "frozen": "frozen_5a5c_black_box_resultant_localizer.json",
            "operations": "implicit_resultant_half_gcd_operation_ledger.json",
            "source_replay": (
                "target_specialization_and_source_localization_replay.json"
            ),
            "exceptional": (
                "multiplicity_and_projective_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r88.json",
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
            "This closes materialized half-gcd and the frozen scalar/block "
            "Krylov quotient grammar only. It does not lower-bound a "
            "coefficient-free structured scalar marked-resultant identity, "
            "a non-Krylov determinant/kernel algorithm, or an unrestricted "
            "arithmetic circuit over the compact elliptic divisors."
        ),
        "next_action": (
            "Construct or refute one coefficient-free fixed-marker scalar "
            "resultant recurrence for the five C decks modulo P_A. It must "
            "return zero, multiplicity, and all five C source markers "
            "without P_C coefficients, a B^2 quotient vector or Krylov "
            "block, a C endpoint dictionary, or a unit-cost determinant "
            "oracle; fit B^(9/4) setup and B^(5/4) fresh work and replay "
            "every projective exceptional chart."
        ),
        "disposition": (
            "REJECT_MATERIALIZED_HALF_GCD_AND_SCALAR_BLOCK_KRYLOV_ONLY__"
            "CONDITIONAL_BINARY_SOURCE_LOCALIZER_LOGARITHMIC__ZERO_TEST_"
            "CONSTRUCTOR_OMITTED__KRYLOV_HANKEL_RANK_B2__BLOCK_WIDTH_CAP_"
            "INTERVAL_EMPTY__C_COEFFICIENTS_B3__NESTED_C_NORM_CROSSES_CAP_"
            "AT_SLOT4__FIXED_MARKER_SCALAR_RECURRENCE_OPEN__NO_RANK__NO_"
            "FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "operations": operation_receipt,
        "source_replay": source_receipt,
        "exceptional": exceptional_receipt,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_black_box_translated_resultant_localizer_"
            "probe_report_r88.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_black_box_resultant_localizer.json"
        ),
    )
    parser.add_argument(
        "--operations-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "implicit_resultant_half_gcd_operation_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_specialization_and_source_localization_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "multiplicity_and_projective_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r88.json"
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
    write_json(args.operations_output, bundle["operations"])
    write_json(args.source_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
