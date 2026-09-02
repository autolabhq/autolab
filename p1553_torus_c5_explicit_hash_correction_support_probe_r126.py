#!/usr/bin/env python3
"""Audit explicit correction lists for hashed torus C5 products."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from collections import defaultdict
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_explicit_hash_correction_support.r126.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C_ATOM_EXPONENT = Fraction(3, 4)
C2_EXPONENT = 2 * C_ATOM_EXPONENT
C3_EXPONENT = 3 * C_ATOM_EXPONENT
C5_EXPONENT = 5 * C_ATOM_EXPONENT

R125_PRODUCER = pathlib.Path(
    "p1553_torus_c5_prime_order_homomorphic_fingerprint_probe_r125.py"
)
R125_PRODUCER_SHA256 = (
    "28a74fc7699f34569a30c3b378ef80985b8ff09bd01aed497df41a98c80160e4"
)
R125_REPORT = pathlib.Path(
    "p1553_torus_c5_prime_order_homomorphic_"
    "fingerprint_probe_report_r125.json"
)
R125_REPORT_SHA256 = (
    "c6f2bd31d83d2b959bd07477d0697f2e91180a5f65e02236a3c32cffd6c43a9b"
)
R125_FROZEN = pathlib.Path(
    "frozen_torus_c5_prime_order_homomorphic_fingerprint.json"
)
R125_FROZEN_SHA256 = (
    "be8bc523f1970c07bcc1e2285070051061c1e6dd189da717d879b539012e88ff"
)
R125_COST = pathlib.Path(
    "torus_c5_prime_order_homomorphic_fingerprint_cost_ledger.json"
)
R125_COST_SHA256 = (
    "095ae32e7b927c2639de8cd02ff6818229a5026065d49407a0f42e3b0fea828e"
)
R125_REPLAY = pathlib.Path(
    "torus_c5_prime_order_homomorphic_fingerprint_replay.json"
)
R125_REPLAY_SHA256 = (
    "63c51d0728f52b0618c6afdf73a3ffda070282d7fb79663b79fceb85effcc8c0"
)
R125_CONTROLS = pathlib.Path(
    "torus_c5_prime_order_homomorphic_fingerprint_controls.json"
)
R125_CONTROLS_SHA256 = (
    "28da34281db3b0ea782b9b5bb8b89c39eaabb902f22cec499113d4b3cc1ba2b9"
)
R125_LOGS = pathlib.Path("factor_logs_and_identical_descent_r125.json")
R125_LOGS_SHA256 = (
    "750c96ae91a642278df899d845db00d26e70858d74af22dc7b51260584303c1a"
)
R125_GATE = pathlib.Path(
    "p1553_torus_c5_prime_order_homomorphic_fingerprint_probe_gate_r125.md"
)
R125_GATE_SHA256 = (
    "b4d19d79a68076f8850afd7c079c7f25bb108ba9aee853330aecb508d9d11155"
)
R125_PARENT = pathlib.Path(
    "p1553_torus_c5_prime_order_homomorphic_"
    "fingerprint_probe_parent_report_r125.yaml"
)
R125_PARENT_SHA256 = (
    "29d29a6e507839ee60964fd49cae14ed3ee9dab348c961a99586acd2ea48298d"
)
R124_GATE = pathlib.Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_gate_r124.md"
)
R124_GATE_SHA256 = (
    "30a3ce3036bab96731430a662cde3ded8e63b9fe1326942a5091148b3e939938"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R119_GATE = pathlib.Path(
    "p1553_m6_output_sensitive_nonlinear_c5_source_index_probe_gate_r119.md"
)
R119_GATE_SHA256 = (
    "941e375918c4acd1be8293fcc40666879b4f5178b4454e28c21487cb9be9a9e9"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R125 = load_module("p1553_r125_for_r126", R125_PRODUCER)
R121 = R125.R124.R123.R121
R82 = R121.R82
Field = R121.Field
Fp2 = tuple[int, int]


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
        ("r125_producer", R125_PRODUCER, R125_PRODUCER_SHA256),
        ("r125_report", R125_REPORT, R125_REPORT_SHA256),
        ("r125_frozen", R125_FROZEN, R125_FROZEN_SHA256),
        ("r125_cost", R125_COST, R125_COST_SHA256),
        ("r125_replay", R125_REPLAY, R125_REPLAY_SHA256),
        ("r125_controls", R125_CONTROLS, R125_CONTROLS_SHA256),
        ("r125_logs", R125_LOGS, R125_LOGS_SHA256),
        ("r125_gate", R125_GATE, R125_GATE_SHA256),
        ("r125_parent", R125_PARENT, R125_PARENT_SHA256),
        ("r124_gate", R124_GATE, R124_GATE_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r119_gate", R119_GATE, R119_GATE_SHA256),
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
        raise AssertionError(f"R126 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def coordinate_hash(point: Fp2, modulus: int, salt: int) -> int:
    if modulus <= 0:
        raise ValueError("hash modulus must be positive")
    return (point[0] + salt * point[1]) % modulus


def source_products(
    field: Field,
    deck: tuple[Fp2, ...],
    arity: int,
) -> tuple[tuple[tuple[int, ...], Fp2], ...]:
    return tuple(
        (
            source,
            field.product(deck[index] for index in source),
        )
        for source in itertools.combinations_with_replacement(
            range(len(deck)),
            arity,
        )
    )


def explicit_correction_control(
    curve: dict[str, Any],
    offset: int,
    hash_modulus: int,
    salt: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c2 = source_products(field, deck, 2)
    c3 = source_products(field, deck, 3)
    c5 = source_products(field, deck, 5)
    c5_products = {product for _, product in c5}
    correction_sets: dict[
        tuple[int, int],
        set[Fp2],
    ] = defaultdict(set)
    backpointers: dict[
        tuple[tuple[int, int], Fp2],
        tuple[tuple[int, ...], tuple[int, ...]],
    ] = {}
    for left_source, left_product in c2:
        left_bucket = coordinate_hash(left_product, hash_modulus, salt)
        for right_source, right_product in c3:
            right_bucket = coordinate_hash(
                right_product,
                hash_modulus,
                salt,
            )
            bucket_pair = (left_bucket, right_bucket)
            product = field.mul(left_product, right_product)
            correction_sets[bucket_pair].add(product)
            backpointers.setdefault(
                (bucket_pair, product),
                (left_source, right_source),
            )
    correction_union = set().union(*correction_sets.values())
    correction_entry_count = sum(
        len(products) for products in correction_sets.values()
    )
    source_replay_exact = True
    for (bucket_pair, product), sources in backpointers.items():
        merged = tuple(sorted(sources[0] + sources[1]))
        source_replay_exact &= (
            merged
            in {
                source
                for source, c5_product in c5
                if c5_product == product
            }
        )
        source_replay_exact &= (
            coordinate_hash(
                field.product(deck[index] for index in sources[0]),
                hash_modulus,
                salt,
            )
            == bucket_pair[0]
        )
        source_replay_exact &= (
            coordinate_hash(
                field.product(deck[index] for index in sources[1]),
                hash_modulus,
                salt,
            )
            == bucket_pair[1]
        )
    return {
        "control_id": (
            f"{curve['family_id']}_offset{offset}_"
            f"m{hash_modulus}_s{salt}"
        ),
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": len(deck),
        "hash_modulus": hash_modulus,
        "hash_salt": salt,
        "hash_uses_field_coordinates_only": True,
        "candidate_discrete_logs_consumed": False,
        "c2_source_count": len(c2),
        "c3_source_count": len(c3),
        "c5_source_count": len(c5),
        "distinct_c5_product_count": len(c5_products),
        "occupied_bucket_pair_count": len(correction_sets),
        "explicit_correction_entry_count": correction_entry_count,
        "correction_entries_at_least_distinct_c5_support": (
            correction_entry_count >= len(c5_products)
        ),
        "correction_union_equals_c5_support": (
            correction_union == c5_products
        ),
        "all_correction_entries_have_source_backpointers": (
            len(backpointers) == correction_entry_count
        ),
        "all_source_backpointers_replay": source_replay_exact,
        "correction_duplication_ratio": (
            correction_entry_count / len(c5_products)
        ),
        "correction_sets_sha256": sha256_json(
            [
                {
                    "bucket_pair": list(bucket_pair),
                    "products": sorted(
                        [field.json(value) for value in products]
                    ),
                }
                for bucket_pair, products in sorted(
                    correction_sets.items()
                )
            ]
        ),
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        explicit_correction_control(
            curve,
            offset,
            hash_modulus,
            salt,
        )
        for curve in R82.FAMILIES
        for offset in (0, 1)
        for hash_modulus, salt in ((2, 1), (3, 2), (5, 3))
    ]
    return {
        "schema": (
            "p1553.torus_c5_explicit_hash_correction_support_"
            "controls.r126.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_hashes_use_field_coordinates_without_dlp": all(
            row["hash_uses_field_coordinates_only"]
            and not row["candidate_discrete_logs_consumed"]
            for row in controls
        ),
        "all_correction_unions_equal_c5_support": all(
            row["correction_union_equals_c5_support"]
            for row in controls
        ),
        "all_correction_lists_cover_at_least_c5_support": all(
            row["correction_entries_at_least_distinct_c5_support"]
            for row in controls
        ),
        "all_source_backpointers_complete_and_exact": all(
            row["all_correction_entries_have_source_backpointers"]
            and row["all_source_backpointers_replay"]
            for row in controls
        ),
        "minimum_correction_duplication_ratio": min(
            row["correction_duplication_ratio"] for row in controls
        ),
        "maximum_correction_duplication_ratio": max(
            row["correction_duplication_ratio"] for row in controls
        ),
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "model": (
            "Arbitrary hashes h2:C2->A and h3:C3->B, plus an explicit "
            "correction list L_(a,b) containing every distinct product xy "
            "for x in C2 bucket a and y in C3 bucket b."
        ),
        "coverage_identity": (
            "union_(a,b) L_(a,b)=C2*C3=C5"
        ),
        "entry_lower_bound": (
            "sum_(a,b)|L_(a,b)| >= |union_(a,b)L_(a,b)| = |C5|"
        ),
        "global_deduplication_boundary": (
            "A globally deduplicated exact product dictionary still has "
            "exactly |C5| target entries."
        ),
        "iid_support_inheritance": (
            "Under the R119 iid-distinct-support theorem at "
            "|C|=B^(3/4+o(1)), |C5|=B^(15/4+o(1)) with high probability."
        ),
        "minimum_explicit_correction_state_exponent_B": fraction_record(
            C5_EXPONENT
        ),
        "inside_setup_cap": False,
        "source_backpointer_consequence": (
            "Attaching one C2 and one C3 source to each represented product "
            "does not reduce the number of represented product entries."
        ),
        "scope_limits": [
            "explicit per-bucket distinct-product correction lists",
            "globally deduplicated explicit exact product dictionaries",
            "arbitrary target-independent hashes",
            "exact membership with represented product keys",
        ],
        "not_covered": [
            "implicit correction circuits",
            "adaptive probes into compressed correction data",
            "nonlinear algebraic certificates not listing product targets",
            "bounded-error fingerprints",
            "general arithmetic circuits or data structures",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_explicit_hash_correction_support_"
            "cost_ledger.r126.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "split_exponents": {
            "C2": fraction_record(C2_EXPONENT),
            "C3": fraction_record(C3_EXPONENT),
            "C5_support": fraction_record(C5_EXPONENT),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "explicit_per_bucket_pair_product_corrections",
                "minimum_state_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
                "scoped_lower_bound_proved": True,
            },
            {
                "route_id": "global_deduplicated_product_dictionary",
                "minimum_state_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
                "scoped_lower_bound_proved": True,
            },
            {
                "route_id": "coordinate_hash_without_corrections",
                "exact_product_composition_proved": False,
                "admitted": False,
            },
            {
                "route_id": "implicit_or_adaptive_correction_circuit",
                "scoped_lower_bound_proved": False,
                "exact_structure_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R125_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R125 nonclaim boundary drifted")
    controls = finite_controls()
    theorem = theorem_record()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r125_nonhomomorphic_correction_interface_inherited": (
            inherited["admission"][
                "scoped_homomorphic_fingerprint_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "twenty_four_coordinate_hash_controls_complete": (
            controls["control_count"] == 24
        ),
        "coordinate_hashes_use_no_dlp": controls[
            "all_hashes_use_field_coordinates_without_dlp"
        ],
        "all_correction_unions_equal_c5_support": controls[
            "all_correction_unions_equal_c5_support"
        ],
        "all_explicit_correction_lists_at_least_c5_support": controls[
            "all_correction_lists_cover_at_least_c5_support"
        ],
        "all_correction_sources_replay": controls[
            "all_source_backpointers_complete_and_exact"
        ],
        "coverage_lower_bound_explicit": (
            "|C5|" in theorem["entry_lower_bound"]
        ),
        "global_deduplication_boundary_explicit": (
            "exactly |C5|"
            in theorem["global_deduplication_boundary"]
        ),
        "iid_B15O4_support_cost_charged": (
            theorem["minimum_explicit_correction_state_exponent_B"]
            == fraction_record(C5_EXPONENT)
            and not theorem["inside_setup_cap"]
        ),
        "implicit_correction_circuits_preserved": (
            "implicit correction circuits" in theorem["not_covered"]
        ),
        "finite_controls_receive_no_asymptotic_credit": (
            not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "inside_cap_implicit_membership_complete": False,
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
        "Freeze one implicit correction circuit or adaptive probe scheme "
        "for a nonhomomorphic coordinate hash. It must evaluate exact "
        "target membership without listing per-bucket or globally "
        "deduplicated C5 products, prove deterministic noncancellation and "
        "empty semantics, recover five projective sources, fit "
        "B^(9/4+o(1)) total setup and polylogarithmic arbitrary-target work, "
        "avoid field DLP, and charge rank, logs, identical descent, memory, "
        "field operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_explicit_hash_correction_"
            "support.r126.v1"
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
        "closed_scoped_grammars": [
            "explicit per-bucket-pair distinct-product correction lists",
            "globally deduplicated explicit exact C5 product dictionaries",
        ],
        "preserved_interface": (
            "implicit correction circuit or adaptive probe over "
            "nonhomomorphic coordinate hashes"
        ),
        "general_data_structure_or_arithmetic_circuit_lower_bound_claimed": (
            False
        ),
    }
    replay = {
        "schema": (
            "p1553.torus_c5_explicit_hash_correction_support_"
            "replay.r126.v1"
        ),
        "coordinate_hash_controls_complete": (
            controls["control_count"] == 24
        ),
        "correction_union_semantics_exact": controls[
            "all_correction_unions_equal_c5_support"
        ],
        "explicit_correction_lower_bound_replayed": controls[
            "all_correction_lists_cover_at_least_c5_support"
        ],
        "source_backpointers_replayed": controls[
            "all_source_backpointers_complete_and_exact"
        ],
        "inside_cap_implicit_membership_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r126.v1",
        "r125_homomorphic_fingerprint_audit_complete": True,
        "r126_explicit_hash_correction_audit_complete": True,
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
        "ARBITRARY_COORDINATE_HASH_C2C3_EXPLICIT_CORRECTION_UNION_EQUALS_"
        "C5_SUPPORT__SUM_BUCKET_PAIR_PRODUCT_ENTRIES_AT_LEAST_C5__GLOBAL_"
        "DEDUP_DICTIONARY_STILL_C5__IID_STATE_B15O4_OVER_CAP__ACTUAL_R82_"
        "NO_DLP_HASH_AND_SOURCE_CONTROLS_EXACT__IMPLICIT_ADAPTIVE_CORRECTION_"
        "CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_COORDINATE_HASH_CORRECTION_CONTROLS_AND_SCOPED_EXPLICIT_"
            "SUPPORT_NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_hashes_use_field_coordinates_without_dlp": controls[
                "all_hashes_use_field_coordinates_without_dlp"
            ],
            "all_correction_unions_equal_c5_support": controls[
                "all_correction_unions_equal_c5_support"
            ],
            "all_correction_sources_replay": controls[
                "all_source_backpointers_complete_and_exact"
            ],
            "minimum_correction_duplication_ratio": controls[
                "minimum_correction_duplication_ratio"
            ],
            "maximum_correction_duplication_ratio": controls[
                "maximum_correction_duplication_ratio"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "coordinate_hash_correction_semantics_admitted": True,
            "scoped_explicit_correction_support_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_explicit_hash_correction_support.json"
            ),
            "cost": (
                "torus_c5_explicit_hash_correction_support_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_explicit_hash_correction_support_replay.json"
            ),
            "controls": (
                "torus_c5_explicit_hash_correction_support_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r126.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The lower bound covers represented correction product entries.",
            "It does not cover implicit or adaptive correction circuits.",
            "Finite controls receive no asymptotic credit.",
            "No complete source index, rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_COORDINATE_HASH_EXACT_CORRECTION_AND_SOURCE_CONTROLS_"
            "ONLY__REJECT_EXPLICIT_PER_BUCKET_AND_GLOBAL_DEDUP_PRODUCT_"
            "DICTIONARIES_AT_C5_B15O4_STATE__PRESERVE_IMPLICIT_ADAPTIVE_"
            "CORRECTION_CIRCUIT__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__"
            "NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_explicit_hash_correction_"
            "support_probe_report_r126.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_explicit_hash_correction_support.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_hash_correction_support_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_hash_correction_support_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_explicit_hash_correction_support_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r126.json"),
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
        f"R126 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
