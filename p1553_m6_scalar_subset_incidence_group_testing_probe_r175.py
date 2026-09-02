#!/usr/bin/env python3
"""Reduce the R174 signed locator to scalar subset group testing."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_scalar_subset_incidence_group_testing.r175.v1"

R174_PRODUCER = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_r174.py"
R174_REPORT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_report_r174.json"
R174_FROZEN = ROOT / "frozen_m6_confluent_signed_dual_chow_pushforward.json"
R174_COST = ROOT / "m6_confluent_signed_dual_chow_pushforward_cost_ledger.json"
R174_REPLAY = ROOT / "m6_confluent_signed_dual_chow_pushforward_replay.json"
R174_CONTROLS = ROOT / "m6_confluent_signed_dual_chow_pushforward_controls.json"
R174_CHOW = ROOT / "confluent_signed_dual_chow_pushforward_r174.json"
R174_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_confluent_signed_dual_chow_pushforward_probe_r174.py"
R174_GATE = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_gate_r174.md"
R174_PARENT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_parent_report_r174.yaml"
R163_REPORT = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_report_r163.json"
R163_GATE = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_gate_r163.md"
R160_REPORT = ROOT / "p1553_m6_positive_c6_generic_locator_reduction_probe_report_r160.json"
R160_GATE = ROOT / "p1553_m6_positive_c6_generic_locator_reduction_probe_gate_r160.md"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r174_producer", R174_PRODUCER, "9b30baf9bea77492816bd81bcbd6cfae01ae05b467b8793562101da8afe9765c"),
    ("r174_report", R174_REPORT, "a8b1d1d5fd4ffeaef17726325ebd85c343285ef61d16ca4dd1ce91dfcce28496"),
    ("r174_frozen", R174_FROZEN, "4673a2fd8dea6f212d4761b8ac60f99aadd99ddb43656db4e64992fb7cbd5d3b"),
    ("r174_cost", R174_COST, "dee4664a442779b2c531714bcfe8af33deff1af1b903c99801c363438a3693b1"),
    ("r174_replay", R174_REPLAY, "fbac601d12b26ce98237c3481af4a7c8b956df658585bb085acf16c3afb8e3aa"),
    ("r174_controls", R174_CONTROLS, "6d69f7146857cfd017281550e7a44b99aa4c53623b767c852da40b81e8bc9d40"),
    ("r174_chow", R174_CHOW, "a87d2bb7b4b68d874c1f74867cd3d34ba5d69b2c60c7f58903647579c1cfd3f8"),
    ("r174_test", R174_TEST, "747064ba9bc417572ab40ca6bac9169c35fc130bb112739d893ff76c06d274a7"),
    ("r174_gate", R174_GATE, "46597f7afd91ada661b5822031489b7d08445b041dfd4399b4343199457afebb"),
    ("r174_parent", R174_PARENT, "cac550192f9da4090c1e5459a5baed1f0ea3437c89ec012436995fcfe5bb4911"),
    ("r163_report", R163_REPORT, "0a35b4fee30c7abf4ba69232e0f7af65d48ec785aef832c934bf03d5366645ab"),
    ("r163_gate", R163_GATE, "3064a108bf063ab59a0991be6910b70bf3a2e498974add780f6a57e157a63212"),
    ("r160_report", R160_REPORT, "8b061717b7822ce969433b17003f00fb2e8e4644305cda007b70ca2213f4f48c"),
    ("r160_gate", R160_GATE, "e4d13e79feb831e0d65e734170def4b5e79a6f53edaeb8c49c9bead4e739ec92"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_scalar_subset_incidence_group_testing_probe_report_r175.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_scalar_subset_incidence_group_testing.json"
DEFAULT_COST = ROOT / "m6_scalar_subset_incidence_group_testing_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_scalar_subset_incidence_group_testing_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_scalar_subset_incidence_group_testing_controls.json"
DEFAULT_TREE = ROOT / "scalar_subset_incidence_group_testing_r175.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R174 = load_module("p1553_r174_for_r175", R174_PRODUCER)
R164 = R174.R164
R161 = R174.R161


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
        raise AssertionError(f"R175 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def r174_candidate_roots(family_id: str, seed: int) -> list[int]:
    report = json.loads(R174_REPORT.read_text())
    matches = [
        row["candidate_roots"]
        for row in report["controls"]["controls"]
        if row["family_id"] == family_id and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one R174 control for family={family_id} seed={seed}"
        )
    return [int(root) for root in matches[0]]


def signed_aggregate_values(
    curve: dict[str, Any], seed: int
) -> tuple[list[tuple[int, int]], list[int], list[int], int]:
    _, divisor, target_records = R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    target_chow = R174.dual_chow(targets, prime)
    values: list[int] = []
    for left in selected:
        x_value, y_value = left
        divided = R174.divided_difference_coefficients(
            divisor["v"], x_value, prime
        )
        aggregate = 1
        for right in selected:
            z_value = int(right[0])
            if z_value == x_value:
                curve_derivative = (
                    3 * x_value * x_value + int(curve["curve_a"])
                ) % prime
                alpha = (-curve_derivative) % prime
                beta = (-2 * y_value) % prime
                gamma = (
                    x_value * curve_derivative - 2 * y_value * y_value
                ) % prime
            else:
                divided_value = R161.poly_eval(divided, z_value, prime)
                alpha = (-divided_value) % prime
                beta = prime - 1
                gamma = (x_value * divided_value - y_value) % prime
            aggregate = aggregate * R174.homogeneous_eval(
                target_chow, alpha, beta, gamma, prime
            ) % prime
        values.append(aggregate)
    return selected, divisor["v"], values, len(targets)


def node_descriptor(
    selected: list[tuple[int, int]],
    v_poly: list[int],
    start: int,
    end: int,
    prime: int,
) -> dict[str, Any]:
    points = selected[start:end]
    u_subset = R161.monic_root_polynomial(
        (int(point[0]) for point in points), prime
    )
    v_subset = R161.poly_mod(v_poly, u_subset, prime)
    if any(
        R161.poly_eval(v_subset, int(x_value), prime) != int(y_value) % prime
        for x_value, y_value in points
    ):
        raise AssertionError("subset V remainder does not interpolate its leaves")
    return {
        "u_degree": R161.poly_degree(u_subset),
        "v_degree": R161.poly_degree(v_subset),
        "coefficient_slot_count": len(u_subset) + len(v_subset),
        "u_sha256": sha256_json(u_subset),
        "v_sha256": sha256_json(v_subset),
        "endpoint_sha256": sha256_json(points),
    }


def recover_zero_leaves(
    selected: list[tuple[int, int]],
    v_poly: list[int],
    values: list[int],
    expected_roots: list[int],
    prime: int,
) -> dict[str, Any]:
    expected = set(expected_roots)
    queries: list[dict[str, Any]] = []
    recovered: list[int] = []

    def query(path: str, start: int, end: int, depth: int) -> None:
        product = math.prod(values[start:end]) % prime
        roots = [
            int(selected[index][0])
            for index in range(start, end)
            if int(selected[index][0]) in expected
        ]
        if (product == 0) != bool(roots):
            raise AssertionError("subset product zero biconditional failed")
        descriptor = node_descriptor(selected, v_poly, start, end, prime)
        queries.append(
            {
                "path": path,
                "depth": depth,
                "start": start,
                "end": end,
                "subset_size": end - start,
                "scalar_product": product,
                "zero": product == 0,
                "candidate_count_in_subset": len(roots),
                "descriptor": descriptor,
            }
        )
        if product != 0:
            return
        if end - start == 1:
            recovered.append(int(selected[start][0]))
            return
        midpoint = (start + end) // 2
        query(path + "0", start, midpoint, depth + 1)
        query(path + "1", midpoint, end, depth + 1)

    query("r", 0, len(selected), 0)
    recovered.sort()
    height = (len(selected) - 1).bit_length()
    candidate_count = len(expected_roots)
    query_bound = 1 + 2 * candidate_count * height
    subset_size_bound = len(selected) * (height + 1)
    queried_subset_size_sum = sum(row["subset_size"] for row in queries)
    if len(queries) > query_bound:
        raise AssertionError("balanced-tree query bound failed")
    if queried_subset_size_sum > subset_size_bound:
        raise AssertionError("balanced-tree subset-size bound failed")
    if recovered != sorted(expected_roots):
        raise AssertionError("balanced-tree recovery lost R174 roots")
    return {
        "tree_height": height,
        "query_count": len(queries),
        "query_count_bound": query_bound,
        "zero_query_count": sum(row["zero"] for row in queries),
        "nonzero_query_count": sum(not row["zero"] for row in queries),
        "leaf_query_count": sum(row["subset_size"] == 1 for row in queries),
        "queried_subset_size_sum": queried_subset_size_sum,
        "queried_subset_size_bound": subset_size_bound,
        "queried_descriptor_slot_count": sum(
            row["descriptor"]["coefficient_slot_count"] for row in queries
        ),
        "all_query_zero_biconditionals_exact": True,
        "recovered_roots": recovered,
        "query_transcript_sha256": sha256_json(queries),
        "queries": queries,
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    selected, v_poly, values, target_count = signed_aggregate_values(curve, seed)
    prime = int(curve["field_prime"])
    expected_roots = r174_candidate_roots(curve["family_id"], seed)
    scalar_roots = sorted(
        int(point[0])
        for point, value in zip(selected, values)
        if value == 0
    )
    if scalar_roots != expected_roots:
        raise AssertionError("scalar leaves differ from R174 candidate roots")
    tree = recover_zero_leaves(
        selected, v_poly, values, expected_roots, prime
    )
    root_u = R161.monic_root_polynomial(
        (int(point[0]) for point in selected), prime
    )
    root_v = R161.poly_mod(v_poly, root_u, prime)
    return {
        "control_id": f"{curve['family_id']}_scalar_subset_tree_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "c3_divisor_degree": len(selected),
        "retained_target_count": target_count,
        "candidate_root_count": len(expected_roots),
        "scalar_leaf_roots": scalar_roots,
        "r174_candidate_roots": expected_roots,
        "scalar_leaf_roots_match_r174": True,
        "leaf_value_sha256": sha256_json(values),
        "selected_endpoint_sha256": sha256_json(selected),
        "root_u_sha256": sha256_json(root_u),
        "root_v_sha256": sha256_json(root_v),
        "tree": tree,
        "full_leaf_enumeration_performed_for_finite_control": True,
        "finite_tree_counts_receive_asymptotic_credit": False,
        "scalar_subset_oracle_supplied": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "scalar_subset_incidence": (
            "Let A(P) be the exact R174 signed aggregate for a selected leaf P. "
            "For every nonempty selected subset S, define Sigma(S)=product_{P in S} "
            "A(P) in F_p. Because F_p is an integral domain, Sigma(S)=0 if and "
            "only if S contains at least one R174 candidate leaf."
        ),
        "balanced_zero_product_recovery": (
            "Query Sigma at the root of a balanced selected-leaf tree. Stop below "
            "a nonzero node and query both children of every zero internal node. "
            "The zero leaves are exactly all K candidates, with at most "
            "1+2K*ceil(log2(n)) scalar queries."
        ),
        "amortized_subset_volume": (
            "At each tree depth the queried child subsets are disjoint, so their "
            "total cardinality is at most n. Across the tree the queried subset "
            "volume is at most n*(1+ceil(log2(n))). Thus an oracle costing softly "
            "O(|S|+N) per queried node gives softly O(n+KN) total work."
        ),
        "coordinate_specific_oracle_contract": (
            "The missing oracle may preprocess the compact selected divisor U,V "
            "and N target points once, then accept a balanced-node subset through "
            "its compact U_S and V mod U_S descriptor and return its exact R174 "
            "signed scalar product in softly O(|S|+N) work. It must preserve the "
            "secant divided-difference and geometric-tangent charts, be reusable "
            "across subsets, and avoid candidate-dependent inversions."
        ),
        "generic_boundary": (
            "R160 and Shoup exclude credit for an encoding-invariant generic "
            "locator at these caps. Any admissible implementation must exploit "
            "the explicit prime-field coordinate representation and fully charge "
            "conversion and preprocessing; the scalar oracle is not supplied here."
        ),
        "scope": (
            "This is an exact output-sensitive reduction and oracle interface, "
            "not a construction of the oracle. Directly multiplying the known "
            "R174 leaf factors over the tree costs n^2*N, while represented target "
            "Chow state already costs N^2 at rho. No unconditional ECDLP, Pollard-"
            "rho, or Shoup improvement follows."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_scalar_subset_incidence_group_testing.cost.r175.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "r163_charged_candidate_output_exponent_B": fraction_record(Fraction(3, 4)),
        "one_time_compact_geometry_preprocessing_exponent_B": fraction_record(Fraction(9, 4)),
        "conditional_tree_subset_volume_exponent_B": fraction_record(Fraction(9, 4)),
        "conditional_tree_target_overhead_KN_exponent_B": fraction_record(Fraction(2)),
        "conditional_scalar_oracle_total_exponent_B": fraction_record(Fraction(9, 4)),
        "r163_target_label_and_backpointer_postprocessing_exponent_B": fraction_record(Fraction(2)),
        "direct_expanded_tree_factor_work_exponent_B": fraction_record(Fraction(23, 4)),
        "represented_target_dual_chow_body_exponent_B": fraction_record(Fraction(5, 2)),
        "represented_selected_pair_query_exponent_B": fraction_record(Fraction(9, 2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "conditional_scalar_oracle_total_strictly_inside_rho": True,
        "candidate_output_exponent_is_r163_charged_contract_not_finite_fit": True,
        "scalar_subset_oracle_supplied": False,
        "standard_direct_route_inside_rho": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["scalar_leaf_roots_match_r174"]
        and row["tree"]["all_query_zero_biconditionals_exact"]
        and row["tree"]["recovered_roots"] == row["r174_candidate_roots"]
        and row["tree"]["query_count"] <= row["tree"]["query_count_bound"]
        and row["tree"]["queried_subset_size_sum"]
        <= row["tree"]["queried_subset_size_bound"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_scalar_subset_incidence_group_testing.controls.r175.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_scalar_leaf_roots_match_r174": all(
            row["scalar_leaf_roots_match_r174"] for row in rows
        ),
        "all_query_zero_biconditionals_exact": all(
            row["tree"]["all_query_zero_biconditionals_exact"] for row in rows
        ),
        "all_balanced_trees_recover_r174_roots": all(
            row["tree"]["recovered_roots"] == row["r174_candidate_roots"]
            for row in rows
        ),
        "all_query_count_bounds_hold": all(
            row["tree"]["query_count"] <= row["tree"]["query_count_bound"]
            for row in rows
        ),
        "all_subset_volume_bounds_hold": all(
            row["tree"]["queried_subset_size_sum"]
            <= row["tree"]["queried_subset_size_bound"]
            for row in rows
        ),
        "candidate_root_count": sum(row["candidate_root_count"] for row in rows),
        "query_count": sum(row["tree"]["query_count"] for row in rows),
        "query_count_bound": sum(row["tree"]["query_count_bound"] for row in rows),
        "zero_query_count": sum(row["tree"]["zero_query_count"] for row in rows),
        "nonzero_query_count": sum(row["tree"]["nonzero_query_count"] for row in rows),
        "leaf_query_count": sum(row["tree"]["leaf_query_count"] for row in rows),
        "queried_subset_size_sum": sum(
            row["tree"]["queried_subset_size_sum"] for row in rows
        ),
        "queried_subset_size_bound": sum(
            row["tree"]["queried_subset_size_bound"] for row in rows
        ),
        "queried_descriptor_slot_count": sum(
            row["tree"]["queried_descriptor_slot_count"] for row in rows
        ),
        "full_leaf_enumeration_performed_for_finite_controls": True,
        "finite_tree_counts_receive_asymptotic_credit": False,
        "scalar_subset_oracle_supplied": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r174_r163_r160_shoup_source_bindings_exact": True,
        "six_controls_replayed": len(rows) == 6,
        "scalar_subset_zero_biconditional_complete": controls[
            "all_query_zero_biconditionals_exact"
        ],
        "r174_signed_leaf_replay_complete": controls[
            "all_scalar_leaf_roots_match_r174"
        ],
        "balanced_tree_exact_recovery_complete": controls[
            "all_balanced_trees_recover_r174_roots"
        ],
        "query_count_bound_complete": controls["all_query_count_bounds_hold"],
        "subset_volume_bound_complete": controls["all_subset_volume_bounds_hold"],
        "compact_subset_uv_descriptors_replayed": all_exact,
        "r163_candidate_output_contract_charged": True,
        "r163_label_backpointer_postprocessing_charged": True,
        "direct_n2N_route_charged": True,
        "represented_target_N2_route_charged": True,
        "represented_selected_pair_route_charged": True,
        "finite_tree_counts_scoped_without_asymptotic_credit": True,
        "candidate_oracles_avoided": True,
        "reusable_scalar_subset_oracle_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    classification = (
        "ADMIT_SCALAR_SUBSET_ZERO_PRODUCT_BICONDITIONAL__BALANCED_GROUP_TESTING_"
        "RECOVERS_ALL_R174_ROOTS__QUERY_BOUND_1_PLUS_2KLOGN__SOFT_SUBSET_VOLUME_"
        "N__CONDITIONAL_REUSABLE_ORACLE_ENVELOPE_B9O4_BELOW_RHO__ORACLE_"
        "UNSUPPLIED__DIRECT_N2N_B23O4__GENERIC_INVARIANT_LOCATOR_EXCLUDED__NO_"
        "UNCONDITIONAL_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute the reusable scalar subset-incidence oracle: after "
        "one softly O(n+N) coordinate-specific preprocessing of U,V and the N "
        "target points, accept every balanced-node U_S,V mod U_S and return the "
        "exact tangent-aware R174 subset product in softly O(|S|+N) work. Reject "
        "leaf enumeration, N^2 or n^2 represented Chow bodies, nN per-node factor "
        "expansion, target-dependent preprocessing, candidate inversions, and "
        "unit-cost multipoint, norm, resultant, root, or generic locator oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Reduce the R174 vector locator to output-sensitive scalar subset "
            "queries and isolate the exact reusable-oracle contract needed for "
            "a below-rho implementation."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "shoup_generic_lower_bound": {
                "title": "Lower Bounds for Discrete Logarithms and Related Problems",
                "source": "EUROCRYPT 1997",
                "fit": (
                    "Rules out assigning below-rho credit to an encoding-invariant "
                    "generic locator; R175's open contract is explicitly coordinate-specific."
                ),
            },
            "r174_factored_chow_boundary": {
                "fit": (
                    "Fast represented multivariate evaluation does not supply the "
                    "factored, tangent-aware subset product; the represented target "
                    "Chow input itself has Theta(N^2) coefficients at rho."
                )
            },
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "scalar_subset_reduction_admitted": all_exact,
            "conditional_below_rho_envelope_admitted": True,
            "reusable_scalar_subset_oracle_admitted": False,
            "lane_admitted": False,
        },
        "classification": classification,
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_scalar_subset_incidence_group_testing.frozen.r175.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "The tangent-aware R174 subset product admits reusable scalar "
                "queries in softly O(|S|+N) work after one softly O(n+N) preprocessing."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "Every implementation enumerates the selected leaves, expands all "
                "N target factors for each leaf or tree node, represents N^2 or n^2 "
                "Chow coefficients, drops the tangent chart, or relies on target- or "
                "candidate-dependent uncharged preprocessing."
            ),
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_scalar_subset_incidence_group_testing.replay.r175.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "leaf_value_sha256": row["leaf_value_sha256"],
                "selected_endpoint_sha256": row["selected_endpoint_sha256"],
                "root_u_sha256": row["root_u_sha256"],
                "root_v_sha256": row["root_v_sha256"],
                "query_transcript_sha256": row["tree"]["query_transcript_sha256"],
                "candidate_roots": row["r174_candidate_roots"],
                "recovered_roots": row["tree"]["recovered_roots"],
            }
            for row in rows
        ],
    }
    tree = {
        "schema": "p1553.m6_scalar_subset_incidence_group_testing.tree.r175.v1",
        "scalar_subset_incidence": theorem["scalar_subset_incidence"],
        "balanced_zero_product_recovery": theorem[
            "balanced_zero_product_recovery"
        ],
        "amortized_subset_volume": theorem["amortized_subset_volume"],
        "coordinate_specific_oracle_contract": theorem[
            "coordinate_specific_oracle_contract"
        ],
        "controls": [
            {
                "control_id": row["control_id"],
                "c3_divisor_degree": row["c3_divisor_degree"],
                "retained_target_count": row["retained_target_count"],
                "candidate_root_count": row["candidate_root_count"],
                "r174_candidate_roots": row["r174_candidate_roots"],
                "tree": row["tree"],
            }
            for row in rows
        ],
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "tree": tree,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--tree-output", type=Path, default=DEFAULT_TREE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    outputs = (
        (args.report_output, bundle["report"]),
        (args.frozen_output, bundle["frozen"]),
        (args.cost_output, bundle["cost"]),
        (args.replay_output, bundle["replay"]),
        (args.controls_output, bundle["controls"]),
        (args.tree_output, bundle["tree"]),
    )
    for path, value in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
