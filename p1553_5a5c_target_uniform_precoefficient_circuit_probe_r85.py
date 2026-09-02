#!/usr/bin/env python3
"""Screen fixed target-uniform pre-coefficient circuits for R84's join."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Callable, Iterable, Sequence


SCHEMA = "p1553.5a5c_target_uniform_precoefficient_circuit.r85.v1"
SETUP_CAP_EXPONENT = 9 / 4
ONLINE_CAP_EXPONENT = 5 / 4
ATOM_A_EXPONENT = 2 / 5
ATOM_C_EXPONENT = 3 / 5

R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_report_r84.json"
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
P1512_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1512_linear_chow_negative_handoff_20260717.md"
)
P1512_HANDOFF_SHA256 = (
    "d725027f381770686eb972b1acfcf9c370f17254725f05526d338dea2f213f25"
)
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)
P1514_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1514_idea133_apolar_moment_constructor_handoff_v1_20260717.md"
)
P1514_HANDOFF_SHA256 = (
    "16edd92f80a515f645d29577cea951859c4a56b45c65cd4931cf3874f83e48c7"
)

Point = tuple[int, int] | None
LabelMap = Callable[[Point], Any]


def load_module(filename: str, name: str) -> Any:
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R84 = load_module(
    "p1553_5a5c_marked_resultant_source_section_probe_r84.py",
    "p1553_r84_for_r85",
)
R82 = R84.R82
R70 = R82.R70


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        P1512_HANDOFF: P1512_HANDOFF_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
        P1514_HANDOFF: P1514_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R85 source binding mismatch: {failures}")
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


def target_equivariant_fiber_theorem() -> dict[str, Any]:
    return {
        "statement": (
            "Let pi:G->Y be surjective. If every translation t has a map "
            "tau_t:Y->Y with pi(t+g)=tau_t(pi(g)) for all g, then the fibers "
            "of pi are cosets of a subgroup H of G."
        ),
        "proof": [
            "pi(g)=pi(h) defines a translation-invariant equivalence relation",
            "the identity class H is closed under differences",
            "every equivalence class is one coset g+H",
            "if G has prime order then H is {0} or G",
        ],
        "prime_order_consequence": (
            "Every exact fixed target-equivariant label map is injective or "
            "constant; there is no proper compressing quotient."
        ),
        "requires_pi_to_be_a_homomorphism": False,
        "covers_arbitrary_fixed_labels_with_exact_all_target_action": True,
        "does_not_cover": [
            "target-specialized labels",
            "labels with false positives followed by charged verification",
            "partial actions defined only on a restricted target set",
            "nonlinear circuits that never expose a label quotient",
        ],
    }


def composite_order_positive_control(
    order: int = 808,
    quotient_order: int = 8,
) -> dict[str, Any]:
    failures = 0
    for target in range(order):
        for point in range(order):
            left = (target - point) % quotient_order
            right = (
                target % quotient_order - point % quotient_order
            ) % quotient_order
            failures += left != right
    return {
        "group": f"Z/{order}Z",
        "label_map": f"Z/{order}Z -> Z/{quotient_order}Z",
        "proper_kernel_size": order // quotient_order,
        "checked_actions": order * order,
        "failure_count": failures,
        "exact_target_action": failures == 0,
        "proper_compression": quotient_order < order,
    }


def subgroup_sample(
    generator: Point,
    curve: dict[str, Any],
    radius: int = 24,
) -> tuple[list[Point], list[Point]]:
    positives = [
        R70.scalar_mul(index, generator, curve)
        for index in range(radius + 1)
    ]
    points = list(positives)
    points.extend(
        R70.negate(point, curve)
        for point in positives[1:]
    )
    targets = positives[: min(12, len(positives))]
    if len(set(points)) != len(points):
        raise AssertionError("subgroup control sample collided")
    return points, targets


def hash_label(point: Point, modulus: int) -> int:
    encoded = (
        b"O"
        if point is None
        else f"{point[0]}:{point[1]}".encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big") % modulus


def induced_target_action_control(
    points: Sequence[Point],
    targets: Sequence[Point],
    label_map: LabelMap,
    curve: dict[str, Any],
) -> dict[str, Any]:
    labels = [label_map(point) for point in points]
    buckets: dict[Any, list[int]] = collections.defaultdict(list)
    for index, label in enumerate(labels):
        buckets[label].append(index)
    collision_pairs = sum(
        len(indices) * (len(indices) - 1) // 2
        for indices in buckets.values()
    )
    conflicts = 0
    witness = None
    for target in targets:
        action: dict[Any, Any] = {}
        for point, label in zip(points, labels):
            translated = R70.add(
                target,
                R70.negate(point, curve),
                curve,
            )
            output_label = label_map(translated)
            previous = action.setdefault(label, output_label)
            if previous != output_label:
                conflicts += 1
                if witness is None:
                    witness = {
                        "target": point_json(target),
                        "input_label": str(label),
                        "first_output_label": str(previous),
                        "second_output_label": str(output_label),
                    }
    unique = len(set(labels))
    constant = unique == 1
    injective = unique == len(points)
    source_biconditional = injective
    return {
        "sample_point_count": len(points),
        "sample_target_count": len(targets),
        "distinct_label_count": unique,
        "collision_pair_count": collision_pairs,
        "injective_on_sample": injective,
        "constant_on_sample": constant,
        "induced_target_action_conflict_count": conflicts,
        "sampled_exact_target_action": conflicts == 0,
        "sampled_source_biconditional": source_biconditional,
        "witness": witness,
    }


def label_controls(
    curve: dict[str, Any],
    right_points: Sequence[Point],
) -> dict[str, Any]:
    generator = R82.R81.curve_generator(curve)
    points, targets = subgroup_sample(generator, curve)
    maps: list[tuple[str, LabelMap]] = [
        ("full_point_key", lambda point: point),
        (
            "x_coordinate",
            lambda point: ("O",) if point is None else point[0],
        ),
        ("constant", lambda point: 0),
        ("encoding_hash_mod_2", lambda point: hash_label(point, 2)),
        ("encoding_hash_mod_4", lambda point: hash_label(point, 4)),
        ("encoding_hash_mod_8", lambda point: hash_label(point, 8)),
    ]
    rows = []
    for name, label_map in maps:
        control = induced_target_action_control(
            points,
            targets,
            label_map,
            curve,
        )
        right_labels = [label_map(point) for point in right_points]
        rows.append(
            {
                "map": name,
                **control,
                "right_endpoint_count": len(right_points),
                "right_attained_label_count": len(set(right_labels)),
                "right_compression_ratio": (
                    len(set(right_labels)) / len(right_points)
                ),
            }
        )
    return {
        "generator": point_json(generator),
        "maps": rows,
        "full_key_exact_but_uncompressed": next(
            row["sampled_exact_target_action"]
            and row["sampled_source_biconditional"]
            and row["right_attained_label_count"]
            == row["right_endpoint_count"]
            for row in rows
            if row["map"] == "full_point_key"
        ),
        "constant_action_exact_but_not_source_biconditional": next(
            row["sampled_exact_target_action"]
            and not row["sampled_source_biconditional"]
            for row in rows
            if row["map"] == "constant"
        ),
        "all_tested_nontrivial_compressing_maps_fail_exact_action_or_source": all(
            (
                row["right_attained_label_count"]
                == row["right_endpoint_count"]
            )
            or not row["sampled_exact_target_action"]
            or not row["sampled_source_biconditional"]
            for row in rows
            if row["map"] != "full_point_key"
        ),
    }


def split_payload_ledger() -> dict[str, Any]:
    rows = []
    for count_a in range(6):
        for count_c in range(6):
            if (count_a, count_c) in ((0, 0), (5, 5)):
                continue
            left = (
                count_a * ATOM_A_EXPONENT
                + count_c * ATOM_C_EXPONENT
            )
            right = 5.0 - left
            rows.append(
                {
                    "left_a_count": count_a,
                    "left_c_count": count_c,
                    "left_payload_exponent_B": left,
                    "right_payload_exponent_B": right,
                    "larger_payload_exponent_B": max(left, right),
                    "smaller_payload_exponent_B": min(left, right),
                }
            )
    best = min(
        rows,
        key=lambda row: (
            row["larger_payload_exponent_B"],
            -row["smaller_payload_exponent_B"],
            row["left_a_count"],
            row["left_c_count"],
        ),
    )
    return {
        "all_binary_root_partitions": rows,
        "best_balanced_partition": best,
        "minimum_larger_payload_exponent_B": best[
            "larger_payload_exponent_B"
        ],
        "corresponding_smaller_payload_exponent_B": best[
            "smaller_payload_exponent_B"
        ],
        "smaller_payload_inside_setup_cap": (
            best["smaller_payload_exponent_B"] <= SETUP_CAP_EXPONENT
        ),
        "smaller_payload_inside_online_cap": (
            best["smaller_payload_exponent_B"] <= ONLINE_CAP_EXPONENT
        ),
        "scope": (
            "standard explicit sparse-convolution nodes and source-bearing "
            "binary contractions only"
        ),
    }


def prior_circuit_route_ledger() -> dict[str, Any]:
    return {
        "scalar_linear_chow_tate_atomizer": {
            "binding_sha256": P1512_HANDOFF_SHA256,
            "charged_payload_exponent_B": 5.0,
            "status": "scoped_negative_full_source_cycle_length",
            "nonlinear_target_specialized_circuit_open": True,
        },
        "shared_norm_resultant_and_standard_ku": {
            "binding_sha256": P1513_HANDOFF_SHA256,
            "standard_dense_coordinate_exponent_B": 3.0,
            "status": "scoped_negative_standard_representations",
            "supplied_common_factor_decoder_only": True,
        },
        "apolar_flat_extension_and_dense_macaulay": {
            "binding_sha256": P1514_HANDOFF_SHA256,
            "direct_source_exponent_B": 5.0,
            "materialized_two_plus_three_exponent_B": 3.0,
            "dense_safe_degree_macaulay_exponent_B": 5.0,
            "status": "scoped_negative_standard_constructors",
            "sparse_multihomogeneous_constructor_open": True,
        },
        "fixed_target_equivariant_label_quotient": {
            "status": "prime_order_triviality_theorem",
            "proper_compressing_exact_quotient_exists": False,
        },
        "not_a_universal_circuit_lower_bound": True,
    }


def analyze_instance(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    _, right_first, right_source_count = R84.multiset_endpoint_section(
        atoms_a,
        atoms_c,
        R84.RIGHT_A_COUNT,
        R84.RIGHT_C_COUNT,
        curve,
    )
    controls = label_controls(curve, list(right_first))
    return {
        "family_id": curve["family_id"],
        "offset": offset,
        "factor_base_size_B": len(factors),
        "right_multiset_source_count": right_source_count,
        "right_distinct_endpoint_count": len(right_first),
        "right_endpoint_map_injective": (
            right_source_count == len(right_first)
        ),
        "factor_base_injective": geometry["factor_base_injective"],
        "label_controls": controls,
    }


def build_bundle(
    families: Sequence[dict[str, Any]] = R82.FAMILIES,
    offsets: Sequence[int] = R82.INSTANCE_OFFSETS,
) -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    theorem = target_equivariant_fiber_theorem()
    composite_control = composite_order_positive_control()
    split_ledger = split_payload_ledger()
    route_ledger = prior_circuit_route_ledger()
    instances = [
        analyze_instance(dict(curve), offset)
        for curve in families
        for offset in offsets
    ]
    all_endpoint_maps_injective = all(
        instance["right_endpoint_map_injective"]
        for instance in instances
    )
    all_label_controls = all(
        instance["label_controls"][
            "full_key_exact_but_uncompressed"
        ]
        and instance["label_controls"][
            "constant_action_exact_but_not_source_biconditional"
        ]
        and instance["label_controls"][
            "all_tested_nontrivial_compressing_maps_fail_exact_action_or_source"
        ]
        for instance in instances
    )

    frozen = {
        "schema": "p1553.frozen_5a5c_precoefficient_circuit.r85.v1",
        "input": "compact R82 divisors D_A,D_C",
        "target_join": "coefficient of A^5*C^5 at a fresh group target",
        "admitted_fixed_label_interface": (
            "pi:G->Y with exact induced action for every translation"
        ),
        "source_requirement": "one exact jointly coupled 5A+5C atom source",
        "caps": {
            "setup_state_exponent_B": SETUP_CAP_EXPONENT,
            "fresh_work_exponent_B": ONLINE_CAP_EXPONENT,
        },
        "forbidden_shortcuts": [
            "candidate DLP labels",
            "hidden resultant or root oracle",
            "B^2.4 endpoint coefficients or provenance leaves",
            "supplied moments or common factor",
        ],
    }
    node_ledger = {
        "schema": "p1553.circuit_node_degree_payload_ledger.r85.v1",
        "binary_sparse_convolution": split_ledger,
        "prior_circuit_routes": route_ledger,
        "target_equivariant_fiber_theorem": theorem,
        "standard_fixed_precoefficient_circuit_inside_caps": False,
        "scope_boundary": (
            "fixed quotient labels, explicit sparse convolution nodes, "
            "scalar-linear atomizers, standard shared norms/KU encodings, "
            "and standard supplied-moment or dense Macaulay constructors"
        ),
    }
    fresh_replay = {
        "schema": (
            "p1553.fresh_target_specialization_source_replay.r85.v1"
        ),
        "instances": instances,
        "all_right_endpoint_maps_injective": all_endpoint_maps_injective,
        "all_fixed_label_controls_pass_declared_outcomes": all_label_controls,
        "fixed_exact_compressing_target_action_found": False,
        "fresh_target_source_inside_online_cap": False,
        "reason": (
            "the full key is exact but uncompressed; every proper fixed "
            "compression is constant or loses exact target action/source"
        ),
    }
    controls = {
        "schema": (
            "p1553.matched_explicit_coefficient_random_controls.r85.v1"
        ),
        "prime_order_theorem": theorem,
        "composite_order_positive_control": composite_control,
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "label_controls": instance["label_controls"],
            }
            for instance in instances
        ],
        "p1512_p1513_p1514_routes": route_ledger,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r85.v1",
        "fixed_target_equivariant_quotient_supplied": False,
        "standard_precoefficient_circuit_inside_caps": False,
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
        "prime_order_equivariant_fiber_theorem": True,
        "composite_order_positive_control": composite_control[
            "exact_target_action"
        ],
        "right_endpoint_maps_injective": all_endpoint_maps_injective,
        "full_key_and_constant_controls_exact": all_label_controls,
        "proper_fixed_compressing_target_action": False,
        "binary_source_payload_inside_caps": False,
        "scalar_linear_chow_inside_caps": False,
        "standard_shared_norm_or_ku_inside_caps": False,
        "standard_apolar_constructor_inside_caps": False,
        "target_specialized_sparse_moment_constructor": False,
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
            "FIXED_TARGET_EQUIVARIANT_QUOTIENT_TRIVIAL__"
            "STANDARD_PRECOEFFICIENT_GRAMMARS_OVER_CAP"
        ),
        "source_bindings": {
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "p1512_linear_chow_handoff": {
                "path": str(P1512_HANDOFF),
                "sha256": P1512_HANDOFF_SHA256,
            },
            "p1513_direct_ku_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
            "p1514_apolar_handoff": {
                "path": str(P1514_HANDOFF),
                "sha256": P1514_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R85 generalizes R83's homomorphic-quotient obstruction to any "
            "fixed label map admitting exact actions for all translations. "
            "It then maps R84's circuit residual onto the independently "
            "audited P1512-P1514 standard-route boundaries."
        ),
        "target_equivariant_fiber_theorem": theorem,
        "instances": instances,
        "aggregate": {
            "instance_count": len(instances),
            "all_right_endpoint_maps_injective": all_endpoint_maps_injective,
            "all_fixed_label_controls_match_theorem": all_label_controls,
            "best_binary_larger_payload_exponent_B": split_ledger[
                "minimum_larger_payload_exponent_B"
            ],
            "best_binary_smaller_payload_exponent_B": split_ledger[
                "corresponding_smaller_payload_exponent_B"
            ],
            "proper_fixed_compressing_target_action_found": False,
        },
        "side_artifacts": {
            "frozen": "frozen_5a5c_precoefficient_circuit.json",
            "node_ledger": "circuit_node_degree_and_payload_ledger.json",
            "fresh_replay": (
                "fresh_target_specialization_and_source_replay.json"
            ),
            "controls": (
                "matched_explicit_coefficient_and_random_controls.json"
            ),
            "logs_descent": (
                "factor_logs_and_identical_descent_r85.json"
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
            "This closes fixed exact target-equivariant quotient labels and "
            "the bound standard circuit grammars only. It does not reject a "
            "target-specialized nonlinear circuit, sparse multihomogeneous "
            "moment recurrence, false-positive filter with charged replay, "
            "or unrestricted arithmetic circuit."
        ),
        "next_action": (
            "Construct or refute one target-specialized sparse "
            "multihomogeneous moment recurrence for the 5A+5C fiber. Derive "
            "its moments directly from compact D_A,D_C without a fixed label "
            "quotient or supplied oracle, fit B^(9/4) setup and B^(5/4) fresh "
            "work, and recover one jointly coupled source on every accepted "
            "fiber including nonreduced and exceptional strata."
        ),
        "disposition": (
            "REJECT_FIXED_TARGET_EQUIVARIANT_QUOTIENT_AND_STANDARD_"
            "PRECOEFFICIENT_GRAMMARS_ONLY__TRANSLATION_INVARIANT_FIBERS_ARE_"
            "COSETS__PRIME_ORDER_MAP_INJECTIVE_OR_CONSTANT__FULL_KEY_EXACT_"
            "BUT_UNCOMPRESSED__CONSTANT_NOT_SOURCE_BICONDITIONAL__HASH_AND_X_"
            "CONTROLS_FAIL_ACTION_OR_COMPRESSION__COMPOSITE_QUOTIENT_POSITIVE_"
            "CONTROL__BEST_BINARY_PAYLOAD_B2P6_BY_B2P4__P1512_LINEAR_CHOW_"
            "BOUND__P1513_STANDARD_NORM_KU_BOUND__P1514_STANDARD_MOMENT_"
            "BOUND__SPARSE_TARGET_SPECIALIZED_MOMENT_RECURRENCE_OPEN__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "node_ledger": node_ledger,
        "fresh_replay": fresh_replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_target_uniform_precoefficient_circuit_"
            "probe_report_r85.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path("frozen_5a5c_precoefficient_circuit.json"),
    )
    parser.add_argument(
        "--node-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "circuit_node_degree_and_payload_ledger.json"
        ),
    )
    parser.add_argument(
        "--fresh-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "fresh_target_specialization_and_source_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "matched_explicit_coefficient_and_random_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r85.json"
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
    write_json(args.node_output, bundle["node_ledger"])
    write_json(args.fresh_output, bundle["fresh_replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
