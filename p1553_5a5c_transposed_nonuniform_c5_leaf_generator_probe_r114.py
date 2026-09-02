#!/usr/bin/env python3
"""Audit forward and adjoint transposition of the R113 C5 leaf product."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_transposed_nonuniform_c5_leaf_generator.r114.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
A5_EXPONENT = Fraction(2)
C_ATOM_EXPONENT = Fraction(3, 5)
FIRST_TRANSPOSED_STATE_EXPONENT = A5_EXPONENT + C_ATOM_EXPONENT
C5_EXPONENT = Fraction(3)
FULL_SOURCE_EXPONENT = Fraction(5)

R113_PRODUCER = pathlib.Path(
    "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_r113.py"
)
R113_PRODUCER_SHA256 = (
    "126a2405ae94aa01143ecdf88927f7f4fedb95922aeb7539b467516c97979e1c"
)
R113_REPORT = pathlib.Path(
    "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_report_r113.json"
)
R113_REPORT_SHA256 = (
    "5b961649fa71ced6b049a3500316114b3c9f3806b5b61639471d9a6f86b01102"
)
R113_FROZEN = pathlib.Path(
    "frozen_5a5c_nonlinear_elliptic_orbit_product.json"
)
R113_FROZEN_SHA256 = (
    "1ffc99132f13f640b170bd2325bec13a196982bb26e377ce8230d2944428f907"
)
R113_LEDGER = pathlib.Path(
    "elliptic_orbit_product_recurrence_ledger.json"
)
R113_LEDGER_SHA256 = (
    "6dbbe7b38e4a2e67584c80e70cbfc3be4745ab7895eccad19192785965edc754"
)
R113_REPLAY = pathlib.Path("orbit_product_subset_source_replay.json")
R113_REPLAY_SHA256 = (
    "ef6bcec03bde76982453f5d508f6cdf9b6f32ba9ea1bca36ce143715b9dbb3df"
)
R113_EXCEPTIONAL = pathlib.Path("orbit_product_exceptional_controls.json")
R113_EXCEPTIONAL_SHA256 = (
    "716969c248a88870109d92caf821e9cd26b1f92ccd719f6b2dea5b76553ba3b4"
)
R113_LOGS = pathlib.Path("factor_logs_and_identical_descent_r113.json")
R113_LOGS_SHA256 = (
    "f9f2e0ae9da91f59c3844e9139ef72f768ad2c56f44a8c1dc45229781c1f7ecf"
)
R113_GATE = pathlib.Path(
    "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_gate_r113.md"
)
R113_GATE_SHA256 = (
    "d1939734367f92798ec40b89f9954ff0c9fe3bf07c6ad0bdaeb33e9a5fccad94"
)
R113_PARENT = pathlib.Path(
    "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_parent_report_r113.yaml"
)
R113_PARENT_SHA256 = (
    "9945ff193a3fd2cc7022e27e68ddf5cf6e40d0ab5301a9450433fe8c0128f129"
)
IDEA_REGISTRY = pathlib.Path("p1553_r35_artifact_index_README.md")
IDEA_REGISTRY_SHA256 = (
    "9f4371eefd5e4019833eef858e3bda79d41aff0c5d7b861c71a5987a96acc392"
)

Point = tuple[int, int] | None
IndexTuple = tuple[int, ...]
Source = tuple[IndexTuple, IndexTuple]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R113_PRODUCER: R113_PRODUCER_SHA256,
        R113_REPORT: R113_REPORT_SHA256,
        R113_FROZEN: R113_FROZEN_SHA256,
        R113_LEDGER: R113_LEDGER_SHA256,
        R113_REPLAY: R113_REPLAY_SHA256,
        R113_EXCEPTIONAL: R113_EXCEPTIONAL_SHA256,
        R113_LOGS: R113_LOGS_SHA256,
        R113_GATE: R113_GATE_SHA256,
        R113_PARENT: R113_PARENT_SHA256,
        IDEA_REGISTRY: IDEA_REGISTRY_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R114 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R113 = load_module("p1553_r113_for_r114", R113_PRODUCER)
R112 = R113.R112
R111 = R113.R111
R110 = R113.R110
R108 = R113.R108
R105 = R113.R105
R82 = R113.R82
R70 = R113.R70


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def source_json(source: Source) -> list[list[int]]:
    return [list(source[0]), list(source[1])]


def source_list(size: int, arity: int) -> list[IndexTuple]:
    return list(
        itertools.combinations_with_replacement(range(size), arity)
    )


def endpoint(
    source: Sequence[int],
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R110.add_many((atoms[index] for index in source), curve)


def product_gradient(leaves: Sequence[int]) -> list[int]:
    if any(leaf not in {0, 1} for leaf in leaves):
        raise AssertionError("product leaves must be binary")
    zero_indices = [index for index, leaf in enumerate(leaves) if leaf == 0]
    output = [0] * len(leaves)
    if not zero_indices:
        return [1] * len(leaves)
    if len(zero_indices) == 1:
        output[zero_indices[0]] = 1
    return output


def product_hessian_ones(leaves: Sequence[int]) -> list[int]:
    if any(leaf not in {0, 1} for leaf in leaves):
        raise AssertionError("product leaves must be binary")
    zero_indices = [index for index, leaf in enumerate(leaves) if leaf == 0]
    size = len(leaves)
    if not zero_indices:
        return [size - 1] * size
    if len(zero_indices) == 1:
        output = [1] * size
        output[zero_indices[0]] = size - 1
        return output
    output = [0] * size
    if len(zero_indices) == 2:
        for index in zero_indices:
            output[index] = 1
    return output


def typed_transposed_profile(
    target: Point,
    a_endpoints: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for depth in range(6):
        prefixes = source_list(len(atoms_c), depth)
        prefix_endpoints = [
            endpoint(prefix, atoms_c, curve) for prefix in prefixes
        ]
        occurrences = len(a_endpoints) * len(prefixes)
        typed_states = set()
        residuals = set()
        for prefix, prefix_endpoint in zip(prefixes, prefix_endpoints):
            boundary = prefix[-1] if prefix else 0
            for endpoint_a in a_endpoints:
                residual = R70.add(
                    target,
                    R70.negate(
                        R70.add(endpoint_a, prefix_endpoint, curve),
                        curve,
                    ),
                    curve,
                )
                key = R112.projective_key(residual)
                residuals.add(key)
                typed_states.add((boundary, key))
        output.append(
            {
                "c_prefix_depth": depth,
                "a_times_prefix_occurrence_count": occurrences,
                "distinct_residual_count": len(residuals),
                "distinct_boundary_typed_state_count": len(typed_states),
                "typed_state_collision_excess": occurrences
                - len(typed_states),
            }
        )
    return output


def expected_rows() -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = json.loads(R113_REPLAY.read_text(encoding="utf-8"))
    output = {}
    for control_class, key in (
        ("actual", "actual"),
        ("matched_random_deck", "matched_random_decks"),
    ):
        for row in payload[key]:
            output[
                control_class,
                row["family_id"],
                row["offset"],
            ] = row
    return output


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
    doubles: dict[tuple[str, int], dict[str, Any]],
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    inputs = R111.instance_inputs(
        curve,
        offset,
        control_class,
        doubles,
    )
    atoms_a = inputs["atoms_a"]
    atoms_c = inputs["atoms_c"]
    target = inputs["target"]
    a_sources = source_list(len(atoms_a), 5)
    c5_sources = source_list(len(atoms_c), 5)
    a_index, a_endpoints = R112.a_endpoint_index(
        a_sources,
        atoms_a,
        curve,
    )
    c5_endpoints = [
        endpoint(source, atoms_c, curve) for source in c5_sources
    ]
    leaves, witnesses = R113.leaf_data(
        target,
        c5_sources,
        c5_endpoints,
        a_index,
        curve,
    )
    zero_indices = [
        index for index, leaf in enumerate(leaves) if leaf == 0
    ]
    gradient = product_gradient(leaves)
    gradient_support = [
        index for index, value in enumerate(gradient) if value
    ]
    hessian_ones = product_hessian_ones(leaves)
    hessian_support = [
        index for index, value in enumerate(hessian_ones) if value
    ]
    expected_support = (
        zero_indices if len(zero_indices) == 1 else []
    )
    if gradient_support != expected_support:
        raise AssertionError("product gradient support mismatch")
    if len(zero_indices) == 2 and hessian_support != zero_indices:
        raise AssertionError("product Hessian-vector support mismatch")

    recovered = []
    adjoint_support = (
        gradient_support if gradient_support else hessian_support
    )
    for c_index in adjoint_support:
        for source_a in witnesses[c_index]:
            recovered.append((source_a, c5_sources[c_index]))
    expected_row = expected[
        control_class,
        curve["family_id"],
        offset,
    ]
    expected_sources = [
        (tuple(source["source"][0]), tuple(source["source"][1]))
        for source in expected_row["sources"]
    ]
    if sorted(recovered) != sorted(expected_sources):
        raise AssertionError("product adjoint source mismatch")
    profile = typed_transposed_profile(
        target,
        a_endpoints,
        atoms_c,
        curve,
    )
    source_rows = [
        {
            "source": source_json(source),
            "marker": list(
                R105.marker_vector(source, curve["field_prime"])
            ),
            "canonical_cycle_weight": R108.FULL_CYCLE_SCALE,
        }
        for source in recovered
    ]
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(inputs["factors"]),
        "target_class": inputs["target_class"],
        "target": point_json(target),
        "a5_source_count": len(a_sources),
        "c5_source_count": len(c5_sources),
        "zero_leaf_count": len(zero_indices),
        "zero_leaf_indices": zero_indices,
        "gradient_support": gradient_support,
        "hessian_ones_support": hessian_support,
        "adjoint_derivative_order": (
            1 if gradient_support else 2
        ),
        "sources": source_rows,
        "sources_match_r113": sorted(recovered) == sorted(expected_sources),
        "primal_leaf_evaluation_count": len(leaves),
        "reverse_product_operation_exponent_B": fraction_record(C5_EXPONENT),
        "checkpointed_reverse_state_exponent_B": fraction_record(
            Fraction(0)
        ),
        "checkpointed_reverse_work_exponent_B": fraction_record(C5_EXPONENT),
        "typed_transposed_state_profile": profile,
        "first_transposed_state_count": profile[1][
            "distinct_boundary_typed_state_count"
        ],
        "candidate_scalar_labels_consumed": False,
        "candidate_work_credit": False,
    }


def all_controls() -> dict[str, Any]:
    doubles = R110.r105_double_fibers()
    expected = expected_rows()
    actual = [
        analyze_instance(
            dict(family),
            offset,
            "actual",
            doubles,
            expected,
        )
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        analyze_instance(
            dict(family),
            offset,
            "matched_random_deck",
            doubles,
            expected,
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "instance_count": len(rows),
        "source_count": sum(len(row["sources"]) for row in rows),
        "unique_zero_instance_count": sum(
            row["zero_leaf_count"] == 1 for row in rows
        ),
        "double_zero_instance_count": sum(
            row["zero_leaf_count"] == 2 for row in rows
        ),
        "all_sources_match_r113": all(
            row["sources_match_r113"] for row in rows
        ),
        "all_unique_gradients_exact": all(
            row["gradient_support"] == row["zero_leaf_indices"]
            for row in rows
            if row["zero_leaf_count"] == 1
        ),
        "all_double_hessian_vectors_exact": all(
            row["hessian_ones_support"] == row["zero_leaf_indices"]
            for row in rows
            if row["zero_leaf_count"] == 2
        ),
        "minimum_first_transposed_state_count": min(
            row["first_transposed_state_count"] for row in rows
        ),
        "maximum_first_transposed_state_count": max(
            row["first_transposed_state_count"] for row in rows
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r113_producer": {
            "path": str(R113_PRODUCER),
            "sha256": R113_PRODUCER_SHA256,
        },
        "r113_report": {
            "path": str(R113_REPORT),
            "sha256": R113_REPORT_SHA256,
        },
        "r113_frozen": {
            "path": str(R113_FROZEN),
            "sha256": R113_FROZEN_SHA256,
        },
        "r113_ledger": {
            "path": str(R113_LEDGER),
            "sha256": R113_LEDGER_SHA256,
        },
        "r113_replay": {
            "path": str(R113_REPLAY),
            "sha256": R113_REPLAY_SHA256,
        },
        "r113_exceptional": {
            "path": str(R113_EXCEPTIONAL),
            "sha256": R113_EXCEPTIONAL_SHA256,
        },
        "r113_logs": {
            "path": str(R113_LOGS),
            "sha256": R113_LOGS_SHA256,
        },
        "r113_gate": {
            "path": str(R113_GATE),
            "sha256": R113_GATE_SHA256,
        },
        "r113_parent": {
            "path": str(R113_PARENT),
            "sha256": R113_PARENT_SHA256,
        },
        "idea_registry": {
            "path": str(IDEA_REGISTRY),
            "sha256": IDEA_REGISTRY_SHA256,
        },
    }


def cost_ledger(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "product_forward_and_adjoint": {
            "primal_leaf_exponent_B": fraction_record(C5_EXPONENT),
            "reverse_product_work_exponent_B": fraction_record(C5_EXPONENT),
            "checkpointed_reverse_state_exponent_B": fraction_record(
                Fraction(0)
            ),
            "checkpointed_reverse_work_exponent_B": fraction_record(
                C5_EXPONENT
            ),
            "inside_online_cap": C5_EXPONENT <= ONLINE_CAP,
            "unique_zero_first_adjoint_exact": True,
            "double_zero_first_adjoint_collapses": True,
            "double_zero_hessian_vector_exact": True,
            "higher_adjoint_order_reduces_primal_leaf_work": False,
        },
        "transpose_before_leaf_formation": {
            "terminal_a5_support_exponent_B": fraction_record(A5_EXPONENT),
            "one_c_atom_expansion_exponent_B": fraction_record(
                FIRST_TRANSPOSED_STATE_EXPONENT
            ),
            "first_state_inside_setup_cap": (
                FIRST_TRANSPOSED_STATE_EXPONENT <= SETUP_CAP
            ),
            "first_state_inside_online_cap": (
                FIRST_TRANSPOSED_STATE_EXPONENT <= ONLINE_CAP
            ),
            "five_c_atom_occurrence_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "minimum_observed_first_state_count": controls[
                "minimum_first_transposed_state_count"
            ],
            "maximum_observed_first_state_count": controls[
                "maximum_first_transposed_state_count"
            ],
        },
        "constructor_status": {
            "exact_forward_existence_complete": True,
            "exact_first_second_source_adjoint_complete": True,
            "inside_cap_primal_leaf_generator_complete": False,
            "inside_cap_preleaf_transpose_complete": False,
            "generic_multiplicity_integer_lift_complete": False,
        },
        "scope": {
            "standard_product_reverse_ad_closed": True,
            "first_and_second_adjoint_localization_closed": True,
            "typed_preleaf_support_propagation_closed": True,
            "arbitrary_black_box_transposed_group_algebra_closed": False,
            "general_arithmetic_circuit_lower_bound_claimed": False,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = all_controls()
    ledger = cost_ledger(controls)
    rows = [
        *controls["actual"],
        *controls["matched_random_decks"],
    ]
    obligations = {
        "ten_source_bindings_verified": len(source_hashes) == 10,
        "sixteen_actual_and_matched_instances": (
            controls["instance_count"] == 16
        ),
        "all_sources_match_r113": controls["all_sources_match_r113"],
        "fourteen_unique_zero_gradients_exact": (
            controls["unique_zero_instance_count"] == 14
        ),
        "two_double_zero_hessian_vectors_exact": (
            controls["double_zero_instance_count"] == 2
            and controls["all_double_hessian_vectors_exact"]
        ),
        "typed_transposed_profiles_frozen": all(
            len(row["typed_transposed_state_profile"]) == 6 for row in rows
        ),
        "r108_weight_and_r105_markers_preserved": all(
            source["canonical_cycle_weight"] == R108.FULL_CYCLE_SCALE
            and len(source["marker"]) == R105.MARKER_DIMENSION
            for row in rows
            for source in row["sources"]
        ),
        "inside_cap_primal_leaf_generator_complete": False,
        "inside_cap_preleaf_transpose_complete": False,
        "generic_multiplicity_integer_lift_complete": False,
        "known_rhs_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Search the relation-arity and asymmetric factor-base exponent space "
        "for a new summation-polynomial/FFE regime whose first transposed "
        "source-return interface fits B^(9/4) setup and B^(5/4) fresh work "
        "while preserving relation supply, known-RHS rank, factor logs, and "
        "identical descent. Freeze the feasibility inequalities before "
        "running any new toy instance."
    )
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_transposed_nonuniform_c5_leaf_generator.r114.v1"
        ),
        "source_bindings": source_binding_records(),
        "field_model": "actual prime-order j=0 curves from R82",
        "forward_functional": (
            "product over canonical C5 leaves "
            "(1-indicator_A5(T-endpoint(C5)))"
        ),
        "adjoint_controls": {
            "unique_zero": "first product gradient",
            "double_zero": "product Hessian times all-ones vector",
            "primal_leaf_trace_required": True,
        },
        "preleaf_transpose": {
            "terminal_support_exponent_B": fraction_record(A5_EXPONENT),
            "first_c_atom_state_exponent_B": fraction_record(
                FIRST_TRANSPOSED_STATE_EXPONENT
            ),
            "canonical_boundary_is_typed": True,
        },
        "novelty_deduplication": {
            "registry": str(IDEA_REGISTRY),
            "merged_lanes": [
                "R97 factored transposed projector trace",
                "R104 compact preendpoint S3/FFE pushdown",
                "R108 lambda-ring complete-homogeneous expansion",
                "R113 nonlinear orbit product",
            ],
            "new_local_control_only": (
                "actual first/second product adjoints and typed preleaf "
                "support profiles"
            ),
            "new_algorithm_claimed": False,
        },
    }
    source_replay = {
        "schema": "p1553.transposed_c5_source_adjoint_replay.r114.v1",
        "actual": controls["actual"],
        "matched_random_decks": controls["matched_random_decks"],
        "summary": {
            key: value
            for key, value in controls.items()
            if key not in {"actual", "matched_random_decks"}
        },
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": "p1553.transposed_c5_exceptional_controls.r114.v1",
        "controls": {
            "complete_projective_key_inherited": True,
            "unique_zero_first_adjoint_replayed": True,
            "double_zero_first_adjoint_collapse_replayed": True,
            "double_zero_second_adjoint_replayed": True,
            "repeated_atoms_and_no_relation_inherited": True,
            "candidate_scalar_labels_consumed": False,
        },
        "scope": ledger["scope"],
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r114.v1",
        "target_existence_and_source_adjoint_replay_complete": True,
        "inside_cap_relation_source_locator_complete": False,
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
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_TRANSPOSED_ADJOINT_SCOPED_NEGATIVE_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "FIRST_PRODUCT_ADJOINT_LOCALIZES_EVERY_UNIQUE_ZERO_AND_"
            "COLLAPSES_ON_DOUBLE_ZEROS__SECOND_HESSIAN_VECTOR_LOCALIZES_"
            "BOTH_DOUBLE_ZERO_LEAVES__ALL_ADJOINTS_REQUIRE_THE_B3_PRIMAL_"
            "LEAF_TRACE__CHECKPOINTING_REDUCES_STATE_NOT_B3_WORK__"
            "TRANSPOSING_BEFORE_LEAF_FORMATION_EXPANDS_A5_BY_ONE_C_ATOM_"
            "TO_B13O5_TYPED_STATE_ALREADY_OVER_SETUP__STANDARD_PRODUCT_AD_"
            "AND_TYPED_PRELEAF_TRANSPOSE_CLOSED_ONLY__RELATION_ARITY_AND_"
            "FACTOR_BASE_REBALANCE_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "control_summary": {
            key: value
            for key, value in controls.items()
            if key not in {"actual", "matched_random_decks"}
        },
        "cost_ledger": ledger,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "failures": failures,
            "scoped_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_5a5c_transposed_nonuniform_c5_leaf_generator.json"
            ),
            "ledger": "transposed_c5_leaf_generator_ledger.json",
            "source_replay": "transposed_c5_source_adjoint_replay.json",
            "exceptional": "transposed_c5_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r114.json",
        },
        "next_action": next_action,
        "non_claims": [
            "Sparse adjoint output does not imply cheap primal construction.",
            "Checkpointed reverse mode reduces memory, not leaf work.",
            "No lower bound for arbitrary black-box transposed multiplication is proved.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_FIRST_SECOND_PRODUCT_ADJOINTS_ONLY__REJECT_"
            "STANDARD_REVERSE_AD_AND_TYPED_PRELEAF_TRANSPOSE_AS_INSIDE_CAP_"
            "LOCATORS__MERGE_WITH_R97_R104_R108_R113__PIVOT_TO_RELATION_"
            "ARITY_FACTOR_BASE_REBALANCE__NO_LOCATOR__NO_RANK__NO_FACTOR_"
            "LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "ledger": ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_transposed_nonuniform_c5_"
            "leaf_generator_probe_report_r114.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_transposed_nonuniform_c5_leaf_generator.json"
        ),
    )
    parser.add_argument(
        "--ledger-output",
        type=pathlib.Path,
        default=pathlib.Path("transposed_c5_leaf_generator_ledger.json"),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path("transposed_c5_source_adjoint_replay.json"),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path("transposed_c5_exceptional_controls.json"),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r114.json"
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
    write_json(args.ledger_output, bundle["ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R114 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
