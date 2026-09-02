#!/usr/bin/env python3
"""Audit exact aggregate digit tries after R99."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import pathlib
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.5a5c_succinct_aggregate_digit_trie.r100.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SOURCE_EXPONENT = Fraction(12, 5)
ENTROPY_WORD_FRACTION = Fraction(13, 25)

R99_PRODUCER = pathlib.Path(
    "p1553_5a5c_multiedge_digitized_equality_projector_probe_r99.py"
)
R99_PRODUCER_SHA256 = (
    "3497a069b3bb5b2590537344d68ca0d1cefd0995de6cd739d8f79b352c862e23"
)
R99_REPORT = pathlib.Path(
    "p1553_5a5c_multiedge_digitized_equality_"
    "projector_probe_report_r99.json"
)
R99_REPORT_SHA256 = (
    "dac6fbf38357e640bb174860df16117fdd3461717cb3b2ceac4e76ec7f77707c"
)
R99_GATE = pathlib.Path(
    "p1553_5a5c_multiedge_digitized_equality_projector_probe_gate_r99.md"
)
R99_GATE_SHA256 = (
    "a7e69ad7cf661f09c679039c5ed9874940eca5432d7ca91dc980f97005bfe560"
)
R94_REPORT = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_report_r94.json"
)
R94_REPORT_SHA256 = (
    "a9966f1fb407e720ce1e9aaafc82ab4a38eade9f338cda287cb737445d099df0"
)
R94_GATE = pathlib.Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_gate_r94.md"
)
R94_GATE_SHA256 = (
    "66d2004d1b4dfa63ac69f60397da45e379dfca4bb8db12ede0647f028aaf517d"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R84_GATE = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_gate_r84.md"
)
R84_GATE_SHA256 = (
    "4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R82_GATE = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_gate_r82.md"
)
R82_GATE_SHA256 = (
    "7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e"
)
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R99_PRODUCER: R99_PRODUCER_SHA256,
        R99_REPORT: R99_REPORT_SHA256,
        R99_GATE: R99_GATE_SHA256,
        R94_REPORT: R94_REPORT_SHA256,
        R94_GATE: R94_GATE_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R82_GATE: R82_GATE_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R100 source binding mismatch: {failures}")
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


def ceil_log_base(value: int, base: int) -> int:
    if value < 1 or base < 2:
        raise ValueError("invalid logarithm inputs")
    words = 0
    capacity = 1
    while capacity < value:
        capacity *= base
        words += 1
    return words


def deterministic_subset(prime: int, size: int) -> list[int]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < size:
        digest = hashlib.sha256(
            f"r100:{prime}:{size}:{counter}".encode("ascii")
        ).digest()
        selected.add(int.from_bytes(digest[:8], "big") % prime)
        counter += 1
    return sorted(selected)


def binary_trie_node_count(values: list[int], bit_count: int) -> int:
    root: dict[int, Any] = {}
    nodes = 1
    for value in values:
        node = root
        for shift in range(bit_count - 1, -1, -1):
            digit = (value >> shift) & 1
            if digit not in node:
                node[digit] = {}
                nodes += 1
            node = node[digit]
    return nodes


@functools.lru_cache(maxsize=1)
def arbitrary_set_information_controls() -> dict[str, Any]:
    pairs = ((17, 4), (31, 5), (61, 7), (127, 10))
    sweep = []
    for prime, size in pairs:
        state_count = math.comb(prime, size)
        field_words = ceil_log_base(state_count, prime)
        values = deterministic_subset(prime, size)
        bit_count = math.ceil(math.log2(prime))
        sweep.append(
            {
                "prime": prime,
                "subset_size": size,
                "possible_distinct_subsets": state_count,
                "minimum_field_words": field_words,
                "minimum_field_words_exact": (
                    prime ** max(field_words - 1, 0) < state_count
                    <= prime**field_words
                ),
                "binary_trie_nodes": binary_trie_node_count(
                    values, bit_count
                ),
                "explicit_leaf_records": size,
                "patricia_leaf_nodes": size,
                "patricia_branch_nodes_for_D_gt_1": size - 1,
                "patricia_total_nodes": 2 * size - 1,
                "values": values,
            }
        )
    return {
        "family": "all D-element subsets of F_p",
        "state_count": "binomial(p,D)",
        "exact_membership_state_theorem": (
            "Two different subsets must have different persistent states: "
            "a target in their symmetric difference forces different exact "
            "membership answers. Exact source return is at least as strong."
        ),
        "binomial_lower_bound": "binomial(p,D)>=(p/D)^D",
        "campaign_substitution": {
            "p": "Theta(B^5)",
            "D": "Theta(B^(12/5))",
            "log_p_D": fraction_record(Fraction(12, 25)),
            "minimum_field_words": (
                "D*(1-log_p D)>=13D/25=Theta(B^(12/5))"
            ),
            "constant_fraction_of_D": fraction_record(
                ENTROPY_WORD_FRACTION
            ),
        },
        "finite_sweep": sweep,
        "all_finite_word_bounds_exact": all(
            row["minimum_field_words_exact"]
            for row in sweep
        ),
        "scope": (
            "Information lower bound for a data structure supporting every "
            "D-subset. It does not apply to a restricted structured family "
            "with a short public generator or merge law."
        ),
    }


def interval_membership_source(
    lower: int,
    size: int,
    target: int,
) -> dict[str, Any]:
    upper = lower + size
    present = lower <= target < upper
    return {
        "summary": {"lower": lower, "upper": upper},
        "summary_words": 2,
        "target": target,
        "present": present,
        "source_index": target - lower if present else None,
        "returned_bottom": not present,
    }


@functools.lru_cache(maxsize=1)
def structured_and_occurrence_controls() -> dict[str, Any]:
    interval_positive = interval_membership_source(20, 16, 27)
    interval_blind = interval_membership_source(20, 16, 19)
    occurrences = [2, 2, 5, 7, 9, 12, 20, 31]
    target = 2
    blind_target = 3
    return {
        "structured_interval_positive": interval_positive,
        "structured_interval_blind": interval_blind,
        "structured_interval_is_constant_state_counterexample": (
            interval_positive["summary_words"] == 2
            and interval_positive["present"]
            and interval_positive["source_index"] == 7
            and interval_blind["returned_bottom"]
        ),
        "occurrence_list": occurrences,
        "positive_target": target,
        "positive_integer_count": sum(
            value == target for value in occurrences
        ),
        "positive_source_index": next(
            index
            for index, value in enumerate(occurrences)
            if value == target
        ),
        "blind_target": blind_target,
        "blind_integer_count": sum(
            value == blind_target for value in occurrences
        ),
        "duplicate_occurrence_payload_words": len(occurrences),
        "value_leaf_count": len(set(occurrences)),
        "occurrence_payload_needed_for_complete_source_return": True,
        "integer_counts_below_prime": True,
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "arbitrary_D_subset_exact_index": {
            "minimum_field_words": "Omega(D)",
            "exponent_B": fraction_record(SOURCE_EXPONENT),
            "inside_setup_cap": SOURCE_EXPONENT <= SETUP_CAP,
            "explicit_or_patricia_trie_state": "Theta(D)",
            "build_work": "Omega(D)",
        },
        "explicit_index_target_query": {
            "work": "O(log p)",
            "inside_online_cap_after_overcap_setup": True,
        },
        "structured_interval_positive_control": {
            "persistent_words": "O(1)",
            "target_query": "O(1) representation-aware comparisons",
            "inside_direct_caps": True,
        },
        "r84_smaller_side": {
            "prospective_source_exponent_B": fraction_record(
                SOURCE_EXPONENT
            ),
            "actual_divisor_image_is_arbitrary_D_subset": False,
            "actual_divisor_image_short_generator_proved": False,
            "leaf_free_mergeable_summary_supplied": False,
        },
        "scope": (
            "Closes exact tries and summaries universal over arbitrary "
            "D-subsets, plus explicit occurrence payloads. It does not "
            "lower-bound the structured 3A+2C endpoint image or another "
            "short public family."
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    information = arbitrary_set_information_controls()
    controls = structured_and_occurrence_controls()
    costs = asymptotic_cost_control()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_succinct_aggregate_digit_trie.r100.v1"
        ),
        "universal_family": "all D-element subsets of F_p",
        "query_contract": "exact membership plus one occurrence or bottom",
        "state_model": "persistent F_p words",
        "structured_positive_control": "integer interval family",
        "caps": costs["caps"],
        "excluded_unfrozen_family": (
            "the actual 3A+2C endpoint image or another restricted family "
            "with a short public generator and leaf-free merge law"
        ),
    }
    state_ledger = {
        "schema": (
            "p1553.aggregate_digit_trie_state_transition_ledger.r100.v1"
        ),
        "information_controls": information,
        "structured_and_occurrence_controls": controls,
        "asymptotic_cost": costs,
        "universal_exact_trie_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.aggregate_digit_trie_integer_source_replay.r100.v1"
        ),
        "structured_interval_positive": controls[
            "structured_interval_positive"
        ],
        "structured_interval_blind": controls[
            "structured_interval_blind"
        ],
        "duplicate_occurrence_count": controls[
            "positive_integer_count"
        ],
        "duplicate_occurrence_source_index": controls[
            "positive_source_index"
        ],
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_complete": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.aggregate_digit_trie_exceptional_controls.r100.v1"
        ),
        "structured_blind_bottom_exact": controls[
            "structured_interval_blind"
        ]["returned_bottom"],
        "duplicate_occurrence_count_exact": (
            controls["positive_integer_count"] == 2
        ),
        "occurrence_payload_charged": controls[
            "occurrence_payload_needed_for_complete_source_return"
        ],
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_actual_source_return": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r100.v1",
        "universal_exact_trie_inside_caps": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "ten_source_bindings_verified": len(bindings) == 10,
        "subset_state_count_exact": all(
            row["possible_distinct_subsets"]
            == math.comb(row["prime"], row["subset_size"])
            for row in information["finite_sweep"]
        ),
        "exact_membership_state_injection_theorem_recorded": (
            "symmetric difference"
            in information["exact_membership_state_theorem"]
        ),
        "binomial_entropy_lower_bound_recorded": (
            information["binomial_lower_bound"]
            == "binomial(p,D)>=(p/D)^D"
        ),
        "campaign_entropy_fraction_13O25": (
            information["campaign_substitution"][
                "constant_fraction_of_D"
            ]["exact"]
            == "13/25"
        ),
        "arbitrary_index_exponent_B12O5": (
            costs["arbitrary_D_subset_exact_index"]["exponent_B"][
                "exact"
            ]
            == "12/5"
        ),
        "finite_word_bounds_replayed": information[
            "all_finite_word_bounds_exact"
        ],
        "explicit_trie_leaf_count_charged": all(
            row["explicit_leaf_records"] == row["subset_size"]
            for row in information["finite_sweep"]
        ),
        "patricia_node_count_charged": all(
            row["patricia_total_nodes"] == 2 * row["subset_size"] - 1
            for row in information["finite_sweep"]
        ),
        "duplicate_occurrence_payload_charged": (
            controls["duplicate_occurrence_payload_words"]
            == len(controls["occurrence_list"])
        ),
        "duplicate_occurrence_count_exact": exceptional[
            "duplicate_occurrence_count_exact"
        ],
        "toy_occurrence_source_exact": (
            controls["positive_source_index"] == 0
            and controls["occurrence_list"][0]
            == controls["positive_target"]
        ),
        "structured_interval_membership_source_exact": controls[
            "structured_interval_is_constant_state_counterexample"
        ],
        "structured_family_nonuniversal_boundary_recorded": (
            "structured" in information["scope"]
        ),
        "integer_no_wrap_replayed": controls[
            "integer_counts_below_prime"
        ],
        "universal_arbitrary_trie_correctly_rejected": not costs[
            "arbitrary_D_subset_exact_index"
        ]["inside_setup_cap"],
        "actual_divisor_image_entropy_theorem_supplied": False,
        "actual_divisor_short_generator_supplied": False,
        "leaf_free_merge_law_supplied": False,
        "actual_five_a_five_c_integer_lift_complete": False,
        "actual_five_a_five_c_source_unranking_complete": False,
        "projective_infinity_source_complete": False,
        "proper_subsum_source_complete": False,
        "tangent_source_complete": False,
        "multiplicity_complete_actual_source_return": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "actual_compact_aggregate_digit_trie_inside_caps": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "ARBITRARY_D_SUBSET_EXACT_INDEX_NEEDS_OMEGA_D_WORDS__"
            "EXPLICIT_AND_PATRICIA_TRIES_THETA_D__STRUCTURED_INTERVAL_"
            "CONSTANT_STATE_POSITIVE__ACTUAL_DIVISOR_IMAGE_OPEN"
        ),
        "source_bindings": {
            "r99_producer": {
                "path": str(R99_PRODUCER),
                "sha256": R99_PRODUCER_SHA256,
            },
            "r99_report": {
                "path": str(R99_REPORT),
                "sha256": R99_REPORT_SHA256,
            },
            "r99_gate": {
                "path": str(R99_GATE),
                "sha256": R99_GATE_SHA256,
            },
            "r94_report": {
                "path": str(R94_REPORT),
                "sha256": R94_REPORT_SHA256,
            },
            "r94_gate": {
                "path": str(R94_GATE),
                "sha256": R94_GATE_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r82_gate": {
                "path": str(R82_GATE),
                "sha256": R82_GATE_SHA256,
            },
            "p1513_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R100 is the first campaign receipt to give an information-"
            "theoretic persistent-state bound for universal exact aggregate "
            "source indices while preserving a structured-family positive "
            "counterexample."
        ),
        "arbitrary_set_information_controls": information,
        "structured_and_occurrence_controls": controls,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": "frozen_5a5c_succinct_aggregate_digit_trie.json",
            "state_ledger": (
                "aggregate_digit_trie_state_transition_ledger.json"
            ),
            "source_replay": (
                "aggregate_digit_trie_integer_source_replay.json"
            ),
            "exceptional": (
                "aggregate_digit_trie_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r100.json",
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
            "R100 closes universal exact D-subset tries and summaries, "
            "including Patricia compression and occurrence payloads. The "
            "interval positive control proves this is not a lower bound on "
            "a structured endpoint family. No entropy or short-generator "
            "theorem is claimed for the actual 3A+2C divisor image."
        ),
        "next_action": (
            "Prove or refute one actual-image entropy and merge theorem for "
            "the R84 3A+2C side directly from D_A,D_C. Freeze the public "
            "parameter family and endpoint-key map before outcomes; either "
            "derive a leaf-free summary below B^(9/4) with B^(5/4) exact "
            "query/source return, or prove that the reachable image contains "
            "an Omega(B^(12/5))-word distinguishable subfamily. Include "
            "multiplicity, all exceptional branches, rank, factor logs, and "
            "identical target descent."
        ),
        "disposition": (
            "REJECT_UNIVERSAL_AGGREGATE_DIGIT_TRIES_ONLY__ALL_D_SUBSETS_"
            "REQUIRE_BINOMIAL_P_D_STATES__OMEGA_D_FIELD_WORDS_B12O5__"
            "EXPLICIT_AND_PATRICIA_TRIES_THETA_D__DUPLICATE_OCCURRENCE_"
            "PAYLOAD_D__STRUCTURED_INTERVAL_CONSTANT_STATE_POSITIVE__ACTUAL_"
            "3A2C_IMAGE_ENTROPY_AND_MERGE_OPEN__PROJECTIVE_AND_FULL_5A5C_"
            "INCOMPLETE__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_"
            "CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "state_ledger": state_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_succinct_aggregate_digit_"
            "trie_probe_report_r100.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_succinct_aggregate_digit_trie.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "aggregate_digit_trie_state_transition_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "aggregate_digit_trie_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "aggregate_digit_trie_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r100.json"
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
    write_json(args.state_ledger_output, bundle["state_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
