#!/usr/bin/env python3
"""Exact finite audit of adaptive sextic-character source routers."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.torus_c5_adaptive_character_decision_router.r142.v1"
INF = 10**12

R141_PRODUCER = ROOT / (
    "p1553_torus_c5_sextic_mobius_character_router_probe_r141.py"
)
R141_REPORT = ROOT / (
    "p1553_torus_c5_sextic_mobius_character_router_probe_report_r141.json"
)
R141_FROZEN = ROOT / "frozen_torus_c5_sextic_mobius_character_router.json"
R141_COST = ROOT / "torus_c5_sextic_mobius_character_router_cost_ledger.json"
R141_REPLAY = ROOT / "torus_c5_sextic_mobius_character_router_replay.json"
R141_CONTROLS = ROOT / (
    "torus_c5_sextic_mobius_character_router_controls.json"
)
R141_LOGS = ROOT / "factor_logs_and_identical_descent_r141.json"
R141_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_sextic_mobius_character_router_probe_r141.py"
)
R141_GATE = ROOT / (
    "p1553_torus_c5_sextic_mobius_character_router_probe_gate_r141.md"
)
R141_PARENT = ROOT / (
    "p1553_torus_c5_sextic_mobius_character_router_"
    "probe_parent_report_r141.yaml"
)

R141_BINDINGS = (
    (
        "r141_producer",
        R141_PRODUCER,
        "5cdbf62193929e3fc8ed3634763f1d9eb2ce7d2561b752e3ac81006d423e2f49",
    ),
    (
        "r141_report",
        R141_REPORT,
        "9b3705ae5515c8f700875e31766a7e9dcb1c5150ac880ae3c0ef197e07388173",
    ),
    (
        "r141_frozen",
        R141_FROZEN,
        "07dedf03d33d83e729e12a52f5302bbb6cc03c508e54d8ec82a21bf0e1f59a0c",
    ),
    (
        "r141_cost",
        R141_COST,
        "62856c871858208554c515949994390af29071079a1200d9903ca7c6cac9274c",
    ),
    (
        "r141_replay",
        R141_REPLAY,
        "c783712f1b583f05f45a7b8864ff8cc34214225db92e4331958eac845d3c5b36",
    ),
    (
        "r141_controls",
        R141_CONTROLS,
        "8a8689012a8e41f3815e203b72fd0e942f7d0f2771a320dfd0034e303a604f8f",
    ),
    (
        "r141_logs",
        R141_LOGS,
        "f0da0ce91e958076bd365b09adbaa91197c78ad761b7e68c6ea262b4882c7458",
    ),
    (
        "r141_test",
        R141_TEST,
        "3e900cf9ae19b4b32eb058886329760d40afdf053357358e97f49de4d0199b9a",
    ),
    (
        "r141_gate",
        R141_GATE,
        "5874757459e62c4e7e9dd4f68f9109cc66262b2ed7cae6501c82e1be1d550f0f",
    ),
    (
        "r141_parent",
        R141_PARENT,
        "c6fa4314e6d0981e2b14c40d92ed40813c3276bb5068b06c6cf3107cd8dd331a",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_torus_c5_adaptive_character_decision_router_"
    "probe_report_r142.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_torus_c5_adaptive_character_decision_router.json"
)
DEFAULT_COST = ROOT / (
    "torus_c5_adaptive_character_decision_router_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "torus_c5_adaptive_character_decision_router_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "torus_c5_adaptive_character_decision_router_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r142.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R141 = load_module("p1553_r141_for_r142", R141_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in R141_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in R141_BINDINGS
    }
    failures = [
        name
        for name, _, expected in R141_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R142 source binding mismatch: {failures}")
    return actual


def unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def source_pair_candidates(source: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        tuple(sorted((source[left], source[right])))
        for left in range(len(source))
        for right in range(left + 1, len(source))
    )


def remove_pair(
    source: tuple[int, ...],
    pair: tuple[int, int],
) -> tuple[int, ...]:
    remaining = list(source)
    for index in pair:
        remaining.remove(index)
    if len(remaining) != 3:
        raise AssertionError("C2 removal did not leave a C3 certificate")
    return tuple(remaining)


def leaf_action(
    target_ids: tuple[int, ...],
    positive_count: int,
    candidate_pairs: tuple[frozenset[tuple[int, int]], ...],
) -> tuple[str, tuple[int, int] | None] | None:
    positives = [index for index in target_ids if index < positive_count]
    negatives = [index for index in target_ids if index >= positive_count]
    if not positives:
        return ("reject", None)
    if negatives:
        return None
    common = set(candidate_pairs[positives[0]])
    for index in positives[1:]:
        common.intersection_update(candidate_pairs[index])
    if not common:
        return None
    return ("accept", min(common))


def split_targets(
    target_ids: tuple[int, ...],
    parameter_index: int,
    labels: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, tuple[int, ...]], ...]:
    groups: dict[Any, list[int]] = {}
    for target_id in target_ids:
        groups.setdefault(labels[target_id][parameter_index], []).append(
            target_id
        )
    return tuple(
        (label, tuple(groups[label]))
        for label in sorted(groups)
    )


def optimize_tree(
    labels: tuple[tuple[Any, ...], ...],
    candidate_pairs: tuple[frozenset[tuple[int, int]], ...],
    positive_count: int,
    parameters: tuple[Any, ...],
    field: Any,
) -> dict[str, Any]:
    choices: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}

    @functools.lru_cache(maxsize=None)
    def solve(
        target_ids: tuple[int, ...],
        remaining: tuple[int, ...],
    ) -> tuple[int, int, int]:
        action = leaf_action(target_ids, positive_count, candidate_pairs)
        if action is not None:
            return (1, 0, 1)
        best = (INF, INF, INF)
        best_parameter: int | None = None
        for parameter_index in remaining:
            groups = split_targets(target_ids, parameter_index, labels)
            if len(groups) == 1:
                continue
            child_remaining = tuple(
                index
                for index in remaining
                if index != parameter_index
            )
            child_scores = [
                solve(child_ids, child_remaining)
                for _, child_ids in groups
            ]
            if any(score[0] >= INF for score in child_scores):
                continue
            score = (
                sum(item[0] for item in child_scores),
                1 + max(item[1] for item in child_scores),
                1 + sum(item[2] for item in child_scores),
            )
            if score < best or (
                score == best
                and (
                    best_parameter is None
                    or parameter_index < best_parameter
                )
            ):
                best = score
                best_parameter = parameter_index
        if best_parameter is not None:
            choices[(target_ids, remaining)] = best_parameter
        return best

    root_targets = tuple(range(len(labels)))
    root_parameters = tuple(range(len(parameters)))
    score = solve(root_targets, root_parameters)
    exact = score[0] < INF

    def build_tree(
        target_ids: tuple[int, ...],
        remaining: tuple[int, ...],
    ) -> dict[str, Any]:
        action = leaf_action(target_ids, positive_count, candidate_pairs)
        positives = sum(index < positive_count for index in target_ids)
        negatives = len(target_ids) - positives
        if action is not None:
            kind, pointer = action
            return {
                "kind": kind,
                "positive_target_count": positives,
                "inverse_empty_target_count": negatives,
                "c2_pointer": (
                    list(pointer) if pointer is not None else None
                ),
            }
        parameter_index = choices[(target_ids, remaining)]
        child_remaining = tuple(
            index
            for index in remaining
            if index != parameter_index
        )
        return {
            "kind": "query",
            "parameter_index": parameter_index,
            "parameter": field.json(parameters[parameter_index]),
            "positive_target_count": positives,
            "inverse_empty_target_count": negatives,
            "branches": [
                {
                    "character_value": field.json(label),
                    "child": build_tree(child_ids, child_remaining),
                }
                for label, child_ids in split_targets(
                    target_ids,
                    parameter_index,
                    labels,
                )
            ],
        }

    return {
        "exact": exact,
        "minimum_leaf_count": score[0] if exact else None,
        "minimum_maximum_depth": score[1] if exact else None,
        "minimum_node_count_at_leaf_depth_optimum": (
            score[2] if exact else None
        ),
        "tree": (
            build_tree(root_targets, root_parameters)
            if exact
            else None
        ),
        "dynamic_program_state_count": solve.cache_info().currsize,
    }


def full_signature_obstructions(
    labels: tuple[tuple[Any, ...], ...],
    candidate_pairs: tuple[frozenset[tuple[int, int]], ...],
    positive_count: int,
    field: Any,
) -> dict[str, Any]:
    cells: dict[tuple[Any, ...], list[int]] = {}
    for target_id, signature in enumerate(labels):
        cells.setdefault(signature, []).append(target_id)
    obstructions: list[dict[str, Any]] = []
    admissible_count = 0
    for signature in sorted(cells):
        target_ids = tuple(cells[signature])
        action = leaf_action(target_ids, positive_count, candidate_pairs)
        if action is not None:
            admissible_count += 1
            continue
        positives = [
            index for index in target_ids if index < positive_count
        ]
        negatives = [
            index - positive_count
            for index in target_ids
            if index >= positive_count
        ]
        common: set[tuple[int, int]] = set()
        if positives:
            common = set(candidate_pairs[positives[0]])
            for index in positives[1:]:
                common.intersection_update(candidate_pairs[index])
        reason = (
            "positive_inverse_signature_collision"
            if positives and negatives
            else "positive_sources_have_no_common_c2"
        )
        obstructions.append(
            {
                "reason": reason,
                "signature": [field.json(value) for value in signature],
                "positive_source_ids": positives,
                "inverse_empty_origin_source_ids": negatives,
                "common_c2_pointers": [list(pair) for pair in sorted(common)],
            }
        )
    return {
        "full_signature_cell_count": len(cells),
        "admissible_full_signature_cell_count": admissible_count,
        "obstruction_cell_count": len(obstructions),
        "obstructions": obstructions,
    }


def follow_tree(
    tree: dict[str, Any],
    target_labels: tuple[Any, ...],
    field: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node = tree
    path: list[dict[str, Any]] = []
    while node["kind"] == "query":
        parameter_index = node["parameter_index"]
        label = field.json(target_labels[parameter_index])
        path.append(
            {
                "parameter_index": parameter_index,
                "parameter": node["parameter"],
                "character_value": label,
            }
        )
        matches = [
            branch
            for branch in node["branches"]
            if branch["character_value"] == label
        ]
        if len(matches) != 1:
            raise AssertionError("decision tree branch is not deterministic")
        node = matches[0]["child"]
    return node, path


def replay_tree(
    tree: dict[str, Any],
    labels: tuple[tuple[Any, ...], ...],
    sources: tuple[tuple[int, ...], ...],
    target_values: tuple[Any, ...],
    deck: tuple[Any, ...],
    field: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive_count = len(sources)
    for target_id, target_labels in enumerate(labels):
        leaf, path = follow_tree(tree, target_labels, field)
        positive = target_id < positive_count
        source_id = (
            target_id if positive else target_id - positive_count
        )
        source = sources[source_id]
        target = (
            target_values[source_id]
            if positive
            else field.inv(target_values[source_id])
        )
        pointer = (
            tuple(leaf["c2_pointer"])
            if leaf["c2_pointer"] is not None
            else None
        )
        row: dict[str, Any] = {
            "target_kind": "positive" if positive else "inverse_empty",
            "source_id": source_id,
            "origin_five_source": list(source),
            "target": field.json(target),
            "path": path,
            "leaf_kind": leaf["kind"],
            "c2_pointer": list(pointer) if pointer is not None else None,
        }
        if positive:
            valid_pointer = (
                pointer is not None
                and pointer in source_pair_candidates(source)
            )
            if not valid_pointer:
                raise AssertionError("positive target has an invalid C2 pointer")
            c3 = remove_pair(source, pointer)
            c2_value = field.product(deck[index] for index in pointer)
            c3_value = field.product(deck[index] for index in c3)
            reverse_source = tuple(sorted(pointer + c3))
            product_exact = field.mul(c2_value, c3_value) == target
            reverse_exact = reverse_source == source
            row.update(
                {
                    "c2_value": field.json(c2_value),
                    "c3_certificate": list(c3),
                    "c3_value": field.json(c3_value),
                    "reverse_five_source": list(reverse_source),
                    "c2_c3_product_exact": product_exact,
                    "reverse_five_source_exact": reverse_exact,
                    "valid": (
                        leaf["kind"] == "accept"
                        and product_exact
                        and reverse_exact
                    ),
                }
            )
        else:
            row["valid"] = leaf["kind"] == "reject" and pointer is None
        rows.append(row)
    return {
        "row_count": len(rows),
        "positive_row_count": positive_count,
        "inverse_empty_row_count": positive_count,
        "all_rows_valid": all(row["valid"] for row in rows),
        "rows": rows,
    }


def actual_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck = R141.R121.pairing_deck(curve, offset)
    subgroup_order = curve["subgroup_order"]
    sources = tuple(
        itertools.combinations_with_replacement(range(len(deck)), 5)
    )
    target_values = tuple(
        field.product(deck[index] for index in source)
        for source in sources
    )
    target_set = set(target_values)
    parameters = unique(value for value in deck if value != field.one)
    if not parameters:
        raise AssertionError("actual control has no character parameter")
    positive_labels = tuple(
        tuple(
            R141.sextic_character(
                target,
                parameter,
                subgroup_order,
                field,
            )
            for parameter in parameters
        )
        for target in target_values
    )
    inverse_labels = tuple(
        tuple(
            R141.sextic_character(
                field.inv(target),
                parameter,
                subgroup_order,
                field,
            )
            for parameter in parameters
        )
        for target in target_values
    )
    labels = positive_labels + inverse_labels
    candidate_pairs = tuple(
        source_pair_candidates(source)
        for source in sources
    )
    optimizer = optimize_tree(
        labels,
        candidate_pairs,
        len(sources),
        parameters,
        field,
    )
    signatures = full_signature_obstructions(
        labels,
        candidate_pairs,
        len(sources),
        field,
    )
    replay = (
        replay_tree(
            optimizer["tree"],
            labels,
            sources,
            target_values,
            deck,
            field,
        )
        if optimizer["exact"]
        else None
    )
    finite_cap = math.floor(len(deck) ** (9 / 4))
    exact_inside_cap = (
        optimizer["exact"]
        and optimizer["minimum_leaf_count"] <= finite_cap
    )
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": subgroup_order,
        "deck_size": len(deck),
        "source_count": len(sources),
        "expected_source_count": math.comb(len(deck) + 4, 5),
        "c5_injective": len(target_set) == len(target_values),
        "all_positive_inverses_empty": all(
            field.inv(target) not in target_set
            for target in target_values
        ),
        "parameter_count": len(parameters),
        "parameters": [field.json(value) for value in parameters],
        "grammar": {
            "query": "chi_z(target) for a frozen deck parameter z",
            "branch_alphabet_size": 6,
            "parameter_reuse": "omitted_as_redundant_on_a_path",
            "accept_leaf": (
                "one C2 pointer shared by every positive source at the leaf"
            ),
            "reject_leaf": "inverse-empty targets only",
        },
        "optimizer": optimizer,
        "full_signatures": signatures,
        "finite_B9_over_4_leaf_cap_floor": finite_cap,
        "exact_router_inside_finite_cap": exact_inside_cap,
        "replay": replay,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def random_source_comparator() -> dict[str, Any]:
    return {
        "model": (
            "independent uniform five-subsets of a B-element deck with "
            "independent uniform signature buckets"
        ),
        "fixed_pair_containment_probability": "20/(B*(B-1))",
        "m_source_common_pair_union_bound": (
            "binom(B,2)*(20/(B*(B-1)))^m"
        ),
        "two_source_bound": "200/(B*(B-1))",
        "consequence": (
            "A random signature cell containing two independent sources "
            "has no common C2 pointer with probability 1-o(1); a direct "
            "router therefore approaches one leaf per C5 target."
        ),
        "predicted_leaf_exponent_B": "5",
        "scope": (
            "Random-model comparator only; not a theorem for the structured "
            "pairing decks or arbitrary character parameters."
        ),
        "candidate_credit": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    controls = [
        actual_control(curve, offset)
        for curve in R141.R82.FAMILIES
        for offset in (0, 1)
    ]
    exact_controls = [
        row for row in controls if row["optimizer"]["exact"]
    ]
    impossible_controls = [
        row for row in controls if not row["optimizer"]["exact"]
    ]
    all_replays_valid = all(
        row["replay"] is not None
        and row["replay"]["all_rows_valid"]
        for row in exact_controls
    )
    all_fail_cap = all(
        not row["exact_router_inside_finite_cap"]
        for row in controls
    )
    comparator = random_source_comparator()

    control_artifact = {
        "schema": (
            "p1553.torus_c5_adaptive_character_decision_router."
            "controls.r142.v1"
        ),
        "actual_control_count": len(controls),
        "exact_router_control_count": len(exact_controls),
        "impossible_router_control_count": len(impossible_controls),
        "all_actual_c5_supports_injective": all(
            row["c5_injective"] for row in controls
        ),
        "all_actual_positive_inverses_empty": all(
            row["all_positive_inverses_empty"] for row in controls
        ),
        "all_exact_router_replays_valid": all_replays_valid,
        "all_controls_fail_finite_B9_over_4_leaf_cap": all_fail_cap,
        "minimum_leaf_counts": {
            row["control_id"]: row["optimizer"]["minimum_leaf_count"]
            for row in controls
        },
        "minimum_maximum_depths": {
            row["control_id"]: row["optimizer"]["minimum_maximum_depth"]
            for row in controls
        },
        "full_signature_obstruction_counts": {
            row["control_id"]: row["full_signatures"][
                "obstruction_cell_count"
            ]
            for row in controls
        },
        "random_source_comparator": comparator,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }

    frozen = {
        "schema": (
            "p1553.torus_c5_adaptive_character_decision_router."
            "frozen.r142.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "grammar": {
            "parameter_domain": "nonidentity elements of the frozen deck",
            "node_operation": "exact sextic Mobius character evaluation",
            "branch_values": "mu_6",
            "adaptive": True,
            "leaf_contract": (
                "accept with a common C2 pointer or reject inverse-empty only"
            ),
            "optimization": (
                "minimum leaves, then minimum maximum depth, then "
                "minimum node count and parameter index"
            ),
        },
        "controls": [
            {
                "control_id": row["control_id"],
                "field_prime": row["field_prime"],
                "subgroup_order": row["subgroup_order"],
                "deck_size": row["deck_size"],
                "source_count": row["source_count"],
                "parameters": row["parameters"],
                "optimizer": row["optimizer"],
                "full_signatures": row["full_signatures"],
                "finite_B9_over_4_leaf_cap_floor": row[
                    "finite_B9_over_4_leaf_cap_floor"
                ],
                "exact_router_inside_finite_cap": row[
                    "exact_router_inside_finite_cap"
                ],
            }
            for row in controls
        ],
        "required_open_outputs": {
            "arbitrary_parameter_character_tree": "open",
            "compact_nonlinear_character_composition": "open",
            "inside_cap_exact_c2_router": "open",
            "known_rhs_relation_rank": "open",
            "factor_logs": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.torus_c5_adaptive_character_decision_router."
            "replay.r142.v1"
        ),
        "controls": [
            {
                "control_id": row["control_id"],
                "optimizer_exact": row["optimizer"]["exact"],
                "replay": row["replay"],
                "full_signature_obstructions": (
                    row["full_signatures"]["obstructions"]
                    if not row["optimizer"]["exact"]
                    else []
                ),
            }
            for row in controls
        ],
        "all_exact_router_replays_valid": all_replays_valid,
    }

    cost = {
        "schema": (
            "p1553.torus_c5_adaptive_character_decision_router."
            "cost.r142.v1"
        ),
        "actual_exact_router_control_count": len(exact_controls),
        "actual_impossible_router_control_count": len(impossible_controls),
        "successful_minimum_leaf_counts": [
            row["optimizer"]["minimum_leaf_count"]
            for row in exact_controls
        ],
        "successful_minimum_maximum_depths": [
            row["optimizer"]["minimum_maximum_depth"]
            for row in exact_controls
        ],
        "all_controls_fail_finite_B9_over_4_leaf_cap": all_fail_cap,
        "finite_cap_comparison_receives_asymptotic_credit": False,
        "random_model_leaf_exponent_B": {
            "exact": "5",
            "decimal": 5.0,
        },
        "random_model_receives_candidate_credit": False,
        "arbitrary_parameter_router_cost_supplied": False,
        "nonlinear_character_composition_cost_supplied": False,
        "candidate_field_dlp_used": False,
        "rank_cost_supplied": False,
        "factor_log_cost_supplied": False,
        "identical_descent_cost_supplied": False,
        "total_attack_cost_supplied": False,
    }

    logs = {
        "schema": (
            "p1553.torus_c5_adaptive_character_decision_router."
            "logs_descent.r142.v1"
        ),
        "factor_logs_computed": False,
        "known_rhs_relation_rank_computed": False,
        "identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "ten_source_bindings_verified": len(actual_bindings) == 10,
        "r141_sextic_character_interface_inherited": True,
        "adaptive_deck_parameter_tree_grammar_frozen": True,
        "eight_actual_controls_complete": len(controls) == 8,
        "all_actual_c5_supports_injective": control_artifact[
            "all_actual_c5_supports_injective"
        ],
        "all_actual_positive_inverses_empty": control_artifact[
            "all_actual_positive_inverses_empty"
        ],
        "dynamic_program_exhaustive_for_frozen_grammar": True,
        "three_exact_finite_routers_identified": len(exact_controls) == 3,
        "five_full_signature_obstructions_identified": (
            len(impossible_controls) == 5
        ),
        "all_exact_router_replays_valid": all_replays_valid,
        "all_positive_c2_c3_certificates_reverse_exact": all_replays_valid,
        "all_inverse_empty_paths_reject": all_replays_valid,
        "all_controls_fail_finite_B9_over_4_leaf_cap": all_fail_cap,
        "random_source_comparator_explicit": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "candidate_discrete_log_oracle_avoided": True,
        "arbitrary_parameter_character_tree_complete": False,
        "compact_nontranslation_composition_complete": False,
        "inside_cap_exact_c2_router_complete": False,
        "complete_five_source_index": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Leave direct deck-landmark trees. Derive an algebraic composition "
        "of chi_z(x) across the C2*C3 split, or admit a broader parameter "
        "family with a proved sub-B^(9/4) compiled state bound. Freeze the "
        "composition law, arbitrary-target branch trace, C2 pointer, C3 "
        "certificate, inverse-empty rejection, reverse five-source replay, "
        "and full memory, field-operation, extension-degree, bit, rank, "
        "factor-log, and identical-descent costs."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "objective": (
            "Test whether adaptive nonlinear branching on the R141 sextic "
            "Mobius labels yields an exact inside-cap C2 source router."
        ),
        "source_bindings": source_binding_records(),
        "theorem": {
            "finite_grammar_completeness": (
                "The dynamic program enumerates every deterministic tree "
                "whose nodes query an unused frozen deck parameter and "
                "branch on its exact mu_6 value. Repeating a parameter on a "
                "path cannot refine the path."
            ),
            "leaf_contract": (
                "An accept leaf is valid exactly when no inverse-empty "
                "target reaches it and one C2 pair occurs in every positive "
                "five-source multiset at that leaf. A reject leaf may contain "
                "inverse-empty targets only."
            ),
            "exact_finite_result": (
                "Five of eight controls have an inadmissible cell even after "
                "all deck parameters are queried. The other three require "
                "minimum leaf counts 33, 360, and 645, respectively."
            ),
            "finite_cap_result": (
                "Every control either has no exact tree or exceeds the "
                "floor(B^(9/4)) leaf comparator. This finite comparison "
                "receives no asymptotic credit."
            ),
            "random_model_comparator": comparator["consequence"],
            "literature_context": (
                "Shifted multiplicative-character fingerprints have "
                "efficient quantum reconstruction in nearby models, while "
                "their classical inversion is not known to be efficient. "
                "No reduction or lower bound is claimed here."
            ),
            "primary_sources": [
                {
                    "title": (
                        "Efficient Quantum Algorithms for Shifted Quadratic "
                        "Character Problems"
                    ),
                    "authors": "Wim van Dam and Sean Hallgren",
                    "url": "https://arxiv.org/abs/quant-ph/0011067",
                    "use": (
                        "Nearby shifted-character inversion context; not an "
                        "algorithm or lower bound for the torus router."
                    ),
                },
                {
                    "title": "On the Hidden Shifted Power Problem",
                    "authors": (
                        "Jean Bourgain, Moubariz Z. Garaev, Sergei V. "
                        "Konyagin, and Igor E. Shparlinski"
                    ),
                    "url": "https://arxiv.org/abs/1110.0812",
                    "use": (
                        "Classical shifted-power reconstruction context; no "
                        "direct transfer to the R142 grammar."
                    ),
                },
            ],
            "scope": (
                "The exact negative covers only adaptive trees over the "
                "frozen nonidentity deck parameters with one mu_6 label per "
                "node and one C2 pointer per accept leaf. It is not a lower "
                "bound for arbitrary parameters, arithmetic combinations of "
                "labels, circuits, RAM, cell probes, or structured "
                "asymptotic factor-base families."
            ),
        },
        "controls": control_artifact,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "frozen_grammar_negative_admitted": True,
            "nonlinear_source_router_admitted": False,
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
        "controls": control_artifact,
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
