#!/usr/bin/env python3
"""Test a coloured norm-jet moment recurrence for R82's compact factor base."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_sparse_multihomogeneous_moment_recurrence.r86.v1"
COLOR_COUNT = 5
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
ATOM_A_EXPONENT = 2 / 5
ATOM_C_EXPONENT = 3 / 5

R85_REPORT = pathlib.Path(
    "p1553_5a5c_target_uniform_precoefficient_circuit_probe_report_r85.json"
)
R85_REPORT_SHA256 = (
    "2a94e5b2807a327cc2de5a35ec8c30c5961065cdc374a7ffd3090a3626e67f78"
)
R85_GATE = pathlib.Path(
    "p1553_5a5c_target_uniform_precoefficient_circuit_probe_gate_r85.md"
)
R85_GATE_SHA256 = (
    "bb052ad21fde0f8f35dd56e014a6ab3a1ba1e148fd697fb80af230e4d3c4029a"
)
P1536_AUDIT = pathlib.Path(
    "/Volumes/Volume/crypto-autoresearcher/ideas/artifacts/"
    "ECDLP-IDEA-133/p1536_frobenius_projector_norm_jet_audit.md"
)
P1536_AUDIT_SHA256 = (
    "81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393"
)
R14_GATE = pathlib.Path("p1553_tensor_trace_minpoly_compiler_gate_r14.md")
R14_GATE_SHA256 = (
    "da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac"
)
P1514_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1514_idea133_apolar_moment_constructor_handoff_v1_20260717.md"
)
P1514_HANDOFF_SHA256 = (
    "16edd92f80a515f645d29577cea951859c4a56b45c65cd4931cf3874f83e48c7"
)

Point = tuple[int, int] | None


def load_module(filename: str, name: str) -> Any:
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R85 = load_module(
    "p1553_5a5c_target_uniform_precoefficient_circuit_probe_r85.py",
    "p1553_r85_for_r86",
)
R82 = R85.R82
R70 = R82.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R85_REPORT: R85_REPORT_SHA256,
        R85_GATE: R85_GATE_SHA256,
        P1536_AUDIT: P1536_AUDIT_SHA256,
        R14_GATE: R14_GATE_SHA256,
        P1514_HANDOFF: P1514_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R86 source binding mismatch: {failures}")
    return actual


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def row_digest(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(compact_json(row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def support_exponent(size: int, base_size: int) -> float:
    if size <= 1 or base_size <= 1:
        return 0.0
    return math.log(size) / math.log(base_size)


def colored_factor_indices(base_size: int) -> list[list[int]]:
    colors = [
        [index for index in range(base_size) if index % COLOR_COUNT == color]
        for color in range(COLOR_COUNT)
    ]
    if any(not color for color in colors):
        raise AssertionError("every frozen factor color must be nonempty")
    return colors


def rectangular_color_description(
    size_a: int,
    size_c: int,
) -> list[dict[str, Any]]:
    colors = []
    for color in range(COLOR_COUNT):
        rectangles = []
        for a_residue in range(COLOR_COUNT):
            a_indices = [
                index
                for index in range(size_a)
                if index % COLOR_COUNT == a_residue
            ]
            if not a_indices:
                continue
            for c_residue in range(COLOR_COUNT):
                c_indices = [
                    index
                    for index in range(size_c)
                    if index % COLOR_COUNT == c_residue
                ]
                if not c_indices:
                    continue
                flattened_residue = (
                    a_residue * size_c + c_residue
                ) % COLOR_COUNT
                if flattened_residue == color:
                    rectangles.append(
                        {
                            "a_residue": a_residue,
                            "c_residue": c_residue,
                            "a_indices": a_indices,
                            "c_indices": c_indices,
                        }
                    )
        colors.append(
            {
                "color": color,
                "rectangle_count": len(rectangles),
                "rectangles": rectangles,
            }
        )
    return colors


def add_factor_tuple(
    source: Sequence[int],
    factors: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R82.add_many((factors[index] for index in source), curve)


def tuple_table(
    colors: Sequence[Sequence[int]],
    factor_labels: Sequence[int],
    modulus: int,
) -> tuple[
    list[tuple[tuple[int, ...], int]],
    collections.Counter[int],
]:
    rows = []
    histogram: collections.Counter[int] = collections.Counter()
    for source in itertools.product(*colors):
        endpoint = sum(factor_labels[index] for index in source) % modulus
        rows.append((source, endpoint))
        histogram[endpoint] += 1
    expected = math.prod(len(color) for color in colors)
    if len(rows) != expected or sum(histogram.values()) != expected:
        raise AssertionError("colored tuple table lost occurrences")
    return rows, histogram


def simple_norm_jet(
    rows: Sequence[tuple[tuple[int, ...], int]],
    target: int,
    modulus: int,
) -> dict[str, Any]:
    zero_rows = [source for source, endpoint in rows if endpoint == target]
    if len(zero_rows) != 1:
        raise AssertionError("simple norm-jet control requires one zero")
    source = zero_rows[0]
    derivative_t = 1
    for _, endpoint in rows:
        value = (endpoint - target) % modulus
        if value:
            derivative_t = derivative_t * value % modulus
    if derivative_t == 0:
        raise AssertionError("simple norm derivative vanished")
    derivatives = [
        derivative_t * (factor_index + 1) % modulus
        for factor_index in source
    ]
    inverse = pow(derivative_t, modulus - 2, modulus)
    recovered = tuple(
        (derivative * inverse % modulus) - 1
        for derivative in derivatives
    )
    return {
        "target_label_verifier_only": target,
        "support_size": 1,
        "norm_constant": 0,
        "derivative_t": derivative_t,
        "marker_derivatives": derivatives,
        "recovered_factor_indices": list(recovered),
        "expected_factor_indices": list(source),
        "source_recovered_exactly": recovered == source,
        "first_jet_branch": "simple_colored_fiber",
    }


def empty_norm_control(
    rows: Sequence[tuple[tuple[int, ...], int]],
    histogram: collections.Counter[int],
    modulus: int,
) -> dict[str, Any]:
    target = next(value for value in range(modulus) if value not in histogram)
    norm = 1
    for _, endpoint in rows:
        norm = norm * ((endpoint - target) % modulus) % modulus
    return {
        "target_label_verifier_only": target,
        "support_size": 0,
        "norm_constant": norm,
        "norm_nonzero": norm != 0,
        "first_jet_branch": "empty_fiber",
    }


def multiple_norm_control(
    histogram: collections.Counter[int],
) -> dict[str, Any]:
    target, multiplicity = max(
        histogram.items(),
        key=lambda item: (item[1], -item[0]),
    )
    multiple_fiber_present = multiplicity >= 2
    all_first_derivatives_zero = multiple_fiber_present
    return {
        "target_label_verifier_only": target,
        "support_size": multiplicity,
        "norm_constant": 0,
        "multiple_fiber_present": multiple_fiber_present,
        "all_first_derivatives_zero": all_first_derivatives_zero,
        "branch_exact_or_vacuous": (
            not multiple_fiber_present or all_first_derivatives_zero
        ),
        "first_jet_branch": (
            "multiple_or_nonreduced_reject"
            if multiple_fiber_present
            else "simple_colored_fiber"
        ),
    }


def synthetic_nonreduced_control(modulus: int) -> dict[str, Any]:
    values = [0, 0, 1, 2]
    norm = math.prod(values) % modulus
    derivative = sum(
        math.prod(values[:index] + values[index + 1 :])
        for index in range(len(values))
    ) % modulus
    return {
        "local_factor_values": values,
        "norm_constant": norm,
        "first_derivative": derivative,
        "norm_and_first_jet_vanish": norm == 0 and derivative == 0,
        "branch": "nonreduced_or_multiple_reject",
    }


def multihomogeneous_constructor_ledger() -> dict[str, Any]:
    return {
        "colored_factor_sizes": "|F_i|=Theta(B)",
        "membership_multidegrees": [
            "(B,0,0,0,0)",
            "(0,B,0,0,0)",
            "(0,0,B,0,0)",
            "(0,0,0,B,0)",
            "(0,0,0,0,B)",
        ],
        "source_quotient_dimension_exponent_B": 5.0,
        "norm_degree_exponent_B": 5.0,
        "projector_value_vector_exponent_B": 5.0,
        "sparse_companion_tensor_matvec_exponent_B": 5.0,
        "low_order_jet_output_word_exponent_B": 0.0,
        "output_small_only_after_constructor": True,
        "r82_addition_pushforward_refinement": {
            "five_colored_a_choices_exponent_B": (
                COLOR_COUNT * ATOM_A_EXPONENT
            ),
            "five_c_choices_exponent_B": (
                COLOR_COUNT * ATOM_C_EXPONENT
            ),
            "standard_atom_split": "B^2 by B^3",
            "smaller_side_inside_setup_cap": (
                COLOR_COUNT * ATOM_A_EXPONENT
                <= SETUP_CAP_EXPONENT
            ),
            "larger_side_inside_setup_cap": (
                COLOR_COUNT * ATOM_C_EXPONENT
                <= SETUP_CAP_EXPONENT
            ),
            "fresh_query_inside_online_cap": False,
        },
        "standard_sparse_multihomogeneous_constructor_inside_caps": False,
        "unproved_exception": (
            "a jet-preserving addition-pushforward intertwiner that contracts "
            "before the C^5 or full colored source product is formed"
        ),
    }


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    generator = R82.R81.curve_generator(curve)
    verifier = R82.R81.BatchBsgsVerifier(generator, curve)
    factor_labels = verifier.labels(factors)
    modulus = curve["subgroup_order"]
    colors = colored_factor_indices(len(factors))
    rows, histogram = tuple_table(colors, factor_labels, modulus)
    frozen_source = tuple(color[0] for color in colors)
    target_label = sum(
        factor_labels[index] for index in frozen_source
    ) % modulus
    if histogram[target_label] != 1:
        raise AssertionError("frozen colored target is not simple")
    target_point = add_factor_tuple(frozen_source, factors, curve)
    if R70.scalar_mul(target_label, generator, curve) != target_point:
        raise AssertionError("verifier target label failed group replay")
    simple = simple_norm_jet(rows, target_label, modulus)
    recovered = tuple(simple["recovered_factor_indices"])
    recovered_point = add_factor_tuple(recovered, factors, curve)
    simple["target_point"] = point_json(target_point)
    simple["recovered_source_group_replay"] = recovered_point == target_point
    simple["factor_to_atom_sources"] = [
        {
            "factor_index": factor_index,
            "atom_a_index": factor_index // len(atoms_c),
            "atom_c_index": factor_index % len(atoms_c),
        }
        for factor_index in recovered
    ]
    rectangular = rectangular_color_description(
        len(atoms_a),
        len(atoms_c),
    )
    return {
        "family_id": curve["family_id"],
        "offset": offset,
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "color_sizes": [len(color) for color in colors],
        "color_index_rule": "flattened factor index modulo 5",
        "coloring_target_independent": True,
        "rectangular_union_description": rectangular,
        "maximum_rectangles_per_color": max(
            color["rectangle_count"] for color in rectangular
        ),
        "colored_quotient_dimension": len(rows),
        "colored_quotient_finite_exponent_B": support_exponent(
            len(rows),
            len(factors),
        ),
        "distinct_target_labels_verifier_only": len(histogram),
        "maximum_target_multiplicity_verifier_only": max(
            histogram.values()
        ),
        "simple_norm_jet": simple,
        "empty_norm_control": empty_norm_control(
            rows,
            histogram,
            modulus,
        ),
        "multiple_norm_control": multiple_norm_control(histogram),
        "synthetic_nonreduced_control": synthetic_nonreduced_control(modulus),
        "tuple_table_sha256_verifier_only": row_digest(
            (list(source), endpoint) for source, endpoint in rows
        ),
        "candidate_scalar_labels_consumed": True,
        "candidate_credit": False,
        "verifier_bsgs_receipt": verifier.receipt(),
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = R82.FAMILIES,
    offsets: Sequence[int] = R82.INSTANCE_OFFSETS,
) -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    ledger = multihomogeneous_constructor_ledger()
    instances = [
        analyze_instance(dict(curve), offset)
        for curve in families
        for offset in offsets
    ]
    all_simple_sources = all(
        instance["simple_norm_jet"]["source_recovered_exactly"]
        and instance["simple_norm_jet"]["recovered_source_group_replay"]
        for instance in instances
    )
    all_empty = all(
        instance["empty_norm_control"]["norm_nonzero"]
        for instance in instances
    )
    all_multiple = all(
        instance["multiple_norm_control"]["branch_exact_or_vacuous"]
        for instance in instances
    )
    multiple_control_count = sum(
        instance["multiple_norm_control"]["multiple_fiber_present"]
        for instance in instances
    )
    all_nonreduced = all(
        instance["synthetic_nonreduced_control"][
            "norm_and_first_jet_vanish"
        ]
        for instance in instances
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_sparse_moment_recurrence.r86.v1",
        "factor_base": "R82 F_(i,j)=A_i+C_j",
        "public_coloring": "flattened factor index modulo 5",
        "color_count": COLOR_COUNT,
        "color_representation": (
            "constant union of A-residue by C-residue rectangles"
        ),
        "target_functional": (
            "first-order norm jet over one factor from each public color"
        ),
        "semantic_simple_fiber_decoder": (
            "dNorm/ds_i divided by dNorm/dt returns factor index i"
        ),
        "candidate_dlp_labels_forbidden": True,
        "verifier_label_control_only": True,
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
    }
    support = {
        "schema": (
            "p1553.multihomogeneous_support_regular_degree_receipts.r86.v1"
        ),
        "constructor_ledger": ledger,
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "factor_base_size_B": instance["factor_base_size_B"],
                "color_sizes": instance["color_sizes"],
                "maximum_rectangles_per_color": instance[
                    "maximum_rectangles_per_color"
                ],
                "colored_quotient_dimension": instance[
                    "colored_quotient_dimension"
                ],
                "colored_quotient_finite_exponent_B": instance[
                    "colored_quotient_finite_exponent_B"
                ],
                "distinct_target_labels_verifier_only": instance[
                    "distinct_target_labels_verifier_only"
                ],
            }
            for instance in instances
        ],
        "compact_input_description_inside_setup_cap": True,
        "standard_moment_constructor_inside_caps": False,
    }
    replay = {
        "schema": (
            "p1553.target_moment_constructor_flat_extension_replay.r86.v1"
        ),
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "simple_norm_jet": instance["simple_norm_jet"],
                "tuple_table_sha256_verifier_only": instance[
                    "tuple_table_sha256_verifier_only"
                ],
                "candidate_scalar_labels_consumed": True,
                "candidate_credit": False,
            }
            for instance in instances
        ],
        "all_supplied_simple_jets_recover_source": all_simple_sources,
        "rank_one_flat_extension_semantic_control": True,
        "public_input_moment_constructor_inside_caps": False,
        "failure_reason": (
            "the verifier enumerates the full colored quotient and consumes "
            "DLP labels before the constant-size jet exists"
        ),
    }
    source_boundary = {
        "schema": (
            "p1553.source_biconditional_exceptional_fibers.r86.v1"
        ),
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "empty": instance["empty_norm_control"],
                "multiple": instance["multiple_norm_control"],
                "nonreduced": instance["synthetic_nonreduced_control"],
                "factor_to_atom_sources": instance["simple_norm_jet"][
                    "factor_to_atom_sources"
                ],
            }
            for instance in instances
        ],
        "all_empty_fibers_detected": all_empty,
        "all_multiple_fibers_rejected_by_first_jet": all_multiple,
        "multiple_fiber_control_count": multiple_control_count,
        "all_synthetic_nonreduced_fibers_rejected": all_nonreduced,
        "all_simple_sources_jointly_coupled": all_simple_sources,
        "actual_semaev_projective_chart_constructor_supplied": False,
        "signed_and_infinity_replay_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r86.v1",
        "supplied_colored_norm_jet_source_exact": all_simple_sources,
        "public_input_jet_constructor_inside_caps": False,
        "algorithmic_known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "five_source_bindings_verified": len(bindings) == 5,
        "four_prime_order_relation_scale_families": (
            len(families) == 4
            and len(instances) == 4 * len(offsets)
        ),
        "eight_frozen_instances": len(instances) == 8,
        "compact_rectangular_coloring": all(
            instance["maximum_rectangles_per_color"] <= COLOR_COUNT
            for instance in instances
        ),
        "all_supplied_simple_norm_jets_exact": all_simple_sources,
        "empty_fiber_branch_exact": all_empty,
        "multiple_fiber_branch_exact": all_multiple,
        "synthetic_nonreduced_branch_exact": all_nonreduced,
        "public_input_moment_constructor_without_dlp": False,
        "standard_multihomogeneous_constructor_inside_setup_cap": False,
        "fresh_target_jet_inside_online_cap": False,
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
            "SUPPLIED_COLORED_NORM_JET_EXACT__"
            "STANDARD_MULTIGRADED_CONSTRUCTOR_PRODUCT_DIMENSION"
        ),
        "source_bindings": {
            "r85_report": {
                "path": str(R85_REPORT),
                "sha256": R85_REPORT_SHA256,
            },
            "r85_gate": {
                "path": str(R85_GATE),
                "sha256": R85_GATE_SHA256,
            },
            "p1536_norm_jet_audit": {
                "path": str(P1536_AUDIT),
                "sha256": P1536_AUDIT_SHA256,
            },
            "r14_trace_minpoly_compiler": {
                "path": str(R14_GATE),
                "sha256": R14_GATE_SHA256,
            },
            "p1514_apolar_handoff": {
                "path": str(P1514_HANDOFF),
                "sha256": P1514_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R86 instantiates P1536's exact coloured first norm jet on the "
            "R82 addition-pushforward factor base using a compact constant-"
            "rectangle public coloring, then charges the multigraded "
            "constructor down to its A^5 and C^5 atom supports."
        ),
        "instances": instances,
        "aggregate": {
            "instance_count": len(instances),
            "all_supplied_simple_jets_recover_source": all_simple_sources,
            "all_empty_fibers_detected": all_empty,
            "all_multiple_fibers_rejected": all_multiple,
            "multiple_fiber_control_count": multiple_control_count,
            "all_nonreduced_controls_rejected": all_nonreduced,
            "maximum_colored_quotient_dimension": max(
                instance["colored_quotient_dimension"]
                for instance in instances
            ),
            "standard_source_quotient_exponent_B": 5.0,
            "r82_colored_a_support_exponent_B": 2.0,
            "r82_c5_support_exponent_B": 3.0,
        },
        "side_artifacts": {
            "frozen": "frozen_5a5c_sparse_moment_recurrence.json",
            "support": (
                "multihomogeneous_support_and_regular_degree_receipts.json"
            ),
            "replay": (
                "target_moment_constructor_and_flat_extension_replay.json"
            ),
            "source_boundary": (
                "source_biconditional_and_exceptional_fibers.json"
            ),
            "logs_descent": (
                "factor_logs_and_identical_descent_r86.json"
            ),
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
            "This closes supplied-jet decoding as a constructor and the "
            "standard full multigraded quotient, companion-tensor, Fermat "
            "projector, norm/resultant, and explicit B^2-by-B^3 atom-split "
            "routes. It does not reject a jet-preserving compositional "
            "intertwiner through the R82 addition pushforward or an "
            "unrestricted target-specialized circuit."
        ),
        "next_action": (
            "Construct or refute one jet-preserving addition-pushforward "
            "intertwiner for F=A+C. It must propagate the coloured first norm "
            "jet from compact D_A,D_C through five factor slots without "
            "forming the B^3 C^5 support or B^5 source quotient, specialize "
            "a fresh target inside B^(5/4), and return the jointly coupled "
            "factor and atom source with complete exceptional-chart replay."
        ),
        "disposition": (
            "REJECT_STANDARD_SPARSE_MULTIHOMOGENEOUS_MOMENT_CONSTRUCTOR_ONLY__"
            "COMPACT_CONSTANT_RECTANGLE_FIVE_COLORING__SUPPLIED_SIMPLE_NORM_"
            "JET_RECOVERS_ALL_FACTOR_AND_ATOM_INDICES__EMPTY_MULTIPLE_AND_"
            "NONREDUCED_BRANCH_CONTROLS_EXACT__VERIFIER_DLP_AND_FULL_COLORED_"
            "QUOTIENT_CONSUMED__QUOTIENT_AND_NORM_DEGREE_B5__R82_ATOM_SPLIT_"
            "B2_BY_B3__C5_SUPPORT_EXCEEDS_SETUP_AND_QUERY__JET_PRESERVING_"
            "ADDITION_PUSHFORWARD_INTERTWINER_OPEN__NO_RANK__NO_FACTOR_LOGS__"
            "NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "support": support,
        "replay": replay,
        "source_boundary": source_boundary,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_sparse_multihomogeneous_moment_recurrence_"
            "probe_report_r86.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_sparse_moment_recurrence.json"
        ),
    )
    parser.add_argument(
        "--support-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "multihomogeneous_support_and_regular_degree_receipts.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "target_moment_constructor_and_flat_extension_replay.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "source_biconditional_and_exceptional_fibers.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r86.json"
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
    write_json(args.support_output, bundle["support"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.source_output, bundle["source_boundary"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
