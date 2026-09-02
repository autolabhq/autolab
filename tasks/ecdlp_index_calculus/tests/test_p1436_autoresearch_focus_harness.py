from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "p1436_autoresearch_focus_harness.py"
SPEC = importlib.util.spec_from_file_location("p1436_focus", MODULE_PATH)
assert SPEC and SPEC.loader
FOCUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOCUS)


def config(
    rank: int,
    *,
    unknowns: int = 4,
    collisions: int = 8,
    cross_shift: int = 4,
    rows: int = 4,
    logs: bool = False,
    cost_ratio: float = 2.0,
    total_operations: int | None = None,
) -> dict:
    return {
        "attempts": 100,
        "accepted_residual_events": 100,
        "collision_edge_count": collisions,
        "cross_shift_collision_count": cross_shift,
        "within_shift_collision_count": collisions - cross_shift,
        "relation_row_count": rows,
        "relation_rank": rank,
        "augmented_rank": rank,
        "unknown_factor_count": unknowns,
        "full_rank": rank == unknowns,
        "rhs_compatible": True,
        "factor_logs_available": logs,
        "factor_log_verification_failures": 0,
        "total_field_ratio_vs_11x_rho": cost_ratio,
        "total_field_operation_estimate": (
            total_operations
            if total_operations is not None
            else int(cost_ratio * 1000)
        ),
        "validation": {
            "sources_exact": True,
            "residual_equalities_exact": True,
            "factor_logs_verify": True,
            "anchor_is_generator": True,
            "rank_boundary_explicit": True,
        },
    }


def exact_payload(schema: str) -> dict:
    return {
        "schema": schema,
        "source_sha256": "a" * 64,
    }


def admitted_discovery_contract(
    *,
    measured_source_operations: int = 40,
    direct_pair_complement_operations: int = 100,
) -> dict:
    return {
        "source_enumerator_id": "scalar-blind-new-factor-row-v1",
        "scalar_blind": True,
        "new_factor_row_count": 3,
        "independent_new_factor_row_count": 2,
        "measured_source_operations": measured_source_operations,
        "direct_pair_complement_operations": direct_pair_complement_operations,
        "replay_artifact_sha256": "b" * 64,
    }


def successful_descent(*, operations: int = 10) -> dict:
    return {
        "recovered": True,
        "invalid_candidate_count": 0,
        "total_field_operation_estimate": operations,
    }


def payload(configurations: dict, *, descents: list | None = None, breakthrough: bool = False) -> dict:
    return {
        "schema": "ecdlp.p1436_large_prime_residual_collision_collector.v1",
        "claim_status": "TEST",
        "curve_records": [
            {
                "split": "prospective",
                "bits": 24,
                "seed": 1432401,
                "order": 1009,
                "policies": {
                    "two_map_union": {
                        "full": {
                            "factor_base_size_B": 12,
                            "configurations": configurations,
                            "target_descents": descents or [],
                        }
                    }
                },
            }
        ],
        "summary": {"large_prime_breakthrough": breakthrough},
    }


def payload_with_curves(curve_records: list[dict], *, breakthrough: bool = False) -> dict:
    return {
        "schema": "ecdlp.p1436_large_prime_residual_collision_collector.v1",
        "claim_status": "TEST",
        "curve_records": curve_records,
        "summary": {"large_prime_breakthrough": breakthrough},
    }


def curve_record(
    *,
    seed: int,
    bits: int,
    split: str,
    configurations: dict,
    descents: list | None = None,
    order: int = 1009,
) -> dict:
    return {
        "split": split,
        "bits": bits,
        "seed": seed,
        "order": order,
        "policies": {
            "two_map_union": {
                "full": {
                    "factor_base_size_B": 12,
                    "configurations": configurations,
                    "target_descents": descents or [],
                }
            }
        },
    }


class FocusHarnessTests(unittest.TestCase):
    def write_payload(self, root: Path, value: dict) -> Path:
        path = root / "probe.json"
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_closed_slice_preflight_becomes_single_top_next_action(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        slice_preflight = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "slice_quadratic_public_source"
            ],
            "classification": "SLICE_QUADRATIC_ORACLE_DIAGNOSTIC_ONLY",
            "source_bindings": {"probe": {"sha256": "a" * 64}},
            "admission": {
                "source_lane_admitted": False,
                "passed_obligation_count": 0,
                "obligation_count": 6,
            },
            "next_action": "Emit the frozen label-separated public corpus.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            preflight_path = root / "slice-preflight.json"
            preflight_path.write_text(
                json.dumps(slice_preflight, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "slice_quadratic_public_source": (
                        slice_preflight,
                        preflight_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"], "public_slice_source_corpus"
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Emit the frozen label-separated public corpus.",
        )
        frontier = report["frontier_lane_preflights"]
        self.assertEqual(
            frontier["closed_lanes"], ["slice_quadratic_public_source"]
        )
        self.assertEqual(report["summary"]["frontier_closed_lane_count"], 1)

    def test_presurface_full_charge_preflight_supersedes_slice_routing(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        slice_preflight = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "slice_quadratic_public_source"
            ],
            "classification": "SLICE_QUADRATIC_ORACLE_DIAGNOSTIC_ONLY",
            "source_bindings": {"probe": {"sha256": "a" * 64}},
            "admission": {
                "source_lane_admitted": False,
                "passed_obligation_count": 0,
                "obligation_count": 6,
            },
            "next_action": "Emit the public slice corpus.",
        }
        presurface_preflight = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "presurface_full_charge"
            ],
            "classification": "PRESURFACE_PUBLIC_COMPONENT_ONLY",
            "source_bindings": {"backfill": {"sha256": "b" * 64}},
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 3,
                "obligation_count": 8,
            },
            "next_action": "Run the frozen prospective target transfer.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            slice_path = root / "slice-preflight.json"
            slice_path.write_text(
                json.dumps(slice_preflight, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            presurface_path = root / "presurface-preflight.json"
            presurface_path.write_text(
                json.dumps(presurface_preflight, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "slice_quadratic_public_source": (
                        slice_preflight,
                        slice_path,
                    ),
                    "presurface_full_charge": (
                        presurface_preflight,
                        presurface_path,
                    ),
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "presurface_full_charge_target_transfer",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Run the frozen prospective target transfer.",
        )
        self.assertNotIn(
            "public_slice_source_corpus",
            [row["id"] for row in report["focus_queue"]],
        )

    def test_factor_line_closure_moves_focus_upstream(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        factor_line_probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "factor_line_direct_root_equivalence"
            ],
            "classification": (
                "FFE_FACTORS_ARE_PUBLIC_ROOT_LINES_WITH_PARTIAL_DIRECT_REPLAY"
            ),
            "source_bindings": {"backfill": {"sha256": "c" * 64}},
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 2,
                "obligation_count": 6,
            },
            "next_action": "Generate rows before selected leaves exist.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "factor-line-probe.json"
            probe_path.write_text(
                json.dumps(factor_line_probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "factor_line_direct_root_equivalence": (
                        factor_line_probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "scalar_blind_fixed_sum_source_generator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Generate rows before selected leaves exist.",
        )

    def test_constructive_closure_gate_moves_focus_to_collision_locator(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        closure_gate = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "constructive_closure_collision"
            ],
            "classification": (
                "SCALAR_BLIND_CLOSURE_COLLISIONS_RECOVER_TOY_LOGS_BUT_COST_ABOVE_RHO"
            ),
            "source_bindings": {
                "r67": {"path": "r67.json", "sha256": "d" * 64},
                "r68": {"path": "r68.json", "sha256": "e" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 4,
                "obligation_count": 8,
            },
            "next_action": "Locate structured collisions before pair materialization.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            gate_path = root / "closure-gate.json"
            gate_path.write_text(
                json.dumps(closure_gate, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "constructive_closure_collision": (
                        closure_gate,
                        gate_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "structured_closure_collision_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Locate structured collisions before pair materialization.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["constructive_closure_collision"],
        )

    def test_multiplicative_x_screen_keeps_focus_on_real_locator(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        screen = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "multiplicative_x_s3_closure"
            ],
            "classification": (
                "MULTIPLICATIVE_X_S3_STRUCTURE_SCREENED_WITHOUT_UNIFORM_RANK_EXCESS"
            ),
            "source_bindings": {
                "r69": {"path": "r69.json", "sha256": "f" * 64}
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 3,
                "obligation_count": 7,
            },
            "next_action": "Require an explicit sub-pair collision locator.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            screen_path = root / "multiplicative-x-screen.json"
            screen_path.write_text(
                json.dumps(screen, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "multiplicative_x_s3_closure": (
                        screen,
                        screen_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "structured_closure_collision_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Require an explicit sub-pair collision locator.",
        )

    def test_s4_carry_probe_routes_to_s6_unit_interface(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "s4_centered_carry_rank"
            ],
            "classification": (
                "S4_CENTERED_ELLIPTIC_CARRY_HAS_FULL_MODE_RANK_ON_ALL_"
                "FROZEN_LARGEST_GRIDS"
            ),
            "source_bindings": {
                "r8": {"path": "r8.md", "sha256": "1" * 64},
                "r70": {"path": "r70.json", "sha256": "2" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 3,
                "obligation_count": 7,
            },
            "next_action": (
                "Specify an exact branch-complete S6 unit-or-zero-divisor router."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "s4-carry-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "s4_centered_carry_rank": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_unit_zero_divisor_source_router",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Specify an exact branch-complete S6 unit-or-zero-divisor router.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["s4_centered_carry_rank"],
        )

    def test_s6_carry_probe_routes_to_noncp_trace_contraction(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "s6_centered_carry_rank_minor"
            ],
            "classification": (
                "S6_CANONICAL_AND_CENTERED_COEFFICIENT_CARRIES_HAVE_"
                "FULL_B18_MODE_RANK"
            ),
            "source_bindings": {
                "r71": {"path": "r71.json", "sha256": "3" * 64},
                "r9": {"path": "r9.md", "sha256": "4" * 64},
                "r10": {"path": "r10.md", "sha256": "5" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 3,
                "obligation_count": 7,
            },
            "next_action": (
                "Construct an exact non-CP balanced projector-trace contraction."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "s6-carry-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "s6_centered_carry_rank_minor": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_noncp_balanced_trace_contraction",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct an exact non-CP balanced projector-trace contraction.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["s6_centered_carry_rank_minor"],
        )

    def test_resultant_valuation_gate_routes_to_quotient_transducer(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "resultant_valuation_trace_grammar"
            ],
            "classification": (
                "RESULTANT_VALUATION_GRAMMAR_EXACT_BUT_OVER_CAP"
            ),
            "source_bindings": {
                "r9": {"path": "r9.md", "sha256": "4" * 64},
                "r10": {"path": "r10.md", "sha256": "5" * 64},
                "r72": {"path": "r72.json", "sha256": "6" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 5,
                "obligation_count": 9,
            },
            "next_action": (
                "Construct an exact quotient-algebra trace transducer."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "resultant-valuation-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "resultant_valuation_trace_grammar": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_quotient_algebra_trace_transducer",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct an exact quotient-algebra trace transducer.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["resultant_valuation_trace_grammar"],
        )

    def test_residual_diagram_gate_routes_to_support_adaptive_incidence(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "s6_residual_decision_diagram"
            ],
            "classification": (
                "S6_RESIDUAL_RADICAL_MEMOIZATION_HAS_B3_ALGEBRAIC_STATES"
            ),
            "source_bindings": {
                "r72": {"path": "r72.json", "sha256": "6" * 64},
                "r73": {"path": "r73.json", "sha256": "7" * 64},
                "r14": {"path": "r14.md", "sha256": "8" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 6,
                "obligation_count": 10,
            },
            "next_action": (
                "Construct a support-adaptive transposed incidence algorithm."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "residual-diagram-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "s6_residual_decision_diagram": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_support_adaptive_transposed_incidence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct a support-adaptive transposed incidence algorithm.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["s6_residual_decision_diagram"],
        )

    def test_iterated_norm_gate_routes_to_transposed_functional(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "s6_iterated_norm_support"
            ],
            "classification": (
                "S6_FIRST_ITERATED_NORM_HAS_FULL_THREE_DIMENSIONAL_SUPPORT"
            ),
            "source_bindings": {
                "r74": {"path": "r74.json", "sha256": "9" * 64},
                "r73": {"path": "r73.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 4,
                "obligation_count": 9,
            },
            "next_action": (
                "Construct a transposed norm scalar functional."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "iterated-norm-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "s6_iterated_norm_support": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_transposed_norm_scalar_functional",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct a transposed norm scalar functional.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["s6_iterated_norm_support"],
        )

    def test_subset_incidence_gate_routes_to_target_frequency_oracle(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "s6_subset_incidence_mobius"
            ],
            "classification": (
                "EXACT_S6_SUBSET_INCIDENCE_COUNT_BUT_B3_PREFIX_STATE"
            ),
            "source_bindings": {
                "r74": {"path": "r74.json", "sha256": "b" * 64},
                "r73": {"path": "r73.json", "sha256": "c" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 8,
                "obligation_count": 12,
            },
            "next_action": (
                "Construct a target-translated subset-frequency oracle."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "subset-incidence-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "s6_subset_incidence_mobius": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_target_translated_subset_frequency_oracle",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct a target-translated subset-frequency oracle.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["s6_subset_incidence_mobius"],
        )

    def test_frequency_orbit_gate_routes_to_nonlinear_resultant(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "target_translated_frequency_orbit"
            ],
            "classification": (
                "TARGET_TRANSLATED_LINEAR_FREQUENCY_ORACLE_HAS_FULL_GROUP_ORBIT"
            ),
            "source_bindings": {
                "r76": {"path": "r76.json", "sha256": "d" * 64},
                "r3": {"path": "r3.md", "sha256": "e" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 5,
                "obligation_count": 9,
            },
            "next_action": (
                "Freeze one nonlinear target-specialized nested-resultant "
                "scalar functional."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "frequency-orbit-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "target_translated_frequency_orbit": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_nonlinear_target_specialized_nested_resultant",
        )
        self.assertEqual(
            report["next_action"]["action"],
            (
                "Freeze one nonlinear target-specialized nested-resultant "
                "scalar functional."
            ),
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["target_translated_frequency_orbit"],
        )

    def test_fermat_tt_gate_routes_to_scalar_only_nested_norm(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "actual_s6_fermat_tensor_train"
            ],
            "classification": (
                "ACTUAL_S6_VALUE_FIRST_FERMAT_TT_HAS_B5_CENTER_CORE"
            ),
            "source_bindings": {
                "r76": {"path": "r76.json", "sha256": "f" * 64},
                "r75": {"path": "r75.json", "sha256": "1" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 7,
                "obligation_count": 11,
            },
            "next_action": (
                "Construct a straight-line black-box nested S4 norm "
                "functional."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "fermat-tt-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "actual_s6_fermat_tensor_train": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_scalar_only_black_box_nested_norm",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct a straight-line black-box nested S4 norm functional.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["actual_s6_fermat_tensor_train"],
        )

    def test_scalar_norm_gate_routes_to_batched_node_compiler(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "scalar_only_nested_norm_slp"
            ],
            "classification": (
                "SCALAR_ONLY_LEAF_NESTED_NORM_HAS_B5_FAILED_ZERO_WORK"
            ),
            "source_bindings": {
                "r78": {"path": "r78.json", "sha256": "2" * 64},
                "r76": {"path": "r76.json", "sha256": "3" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 8,
                "obligation_count": 12,
            },
            "next_action": (
                "Freeze one batched nested-norm node compiler."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "scalar-norm-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "scalar_only_nested_norm_slp": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_batched_nested_norm_node_compiler",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Freeze one batched nested-norm node compiler.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["scalar_only_nested_norm_slp"],
        )

    def test_batched_norm_gate_routes_to_structured_factor_base(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "batched_nested_norm_node_compiler"
            ],
            "classification": (
                "BATCHED_S4_PRODUCT_GCD_IMPROVES_B5_TO_B3_BUT_MISSES_CAPS"
            ),
            "source_bindings": {
                "r79": {"path": "r79.json", "sha256": "4" * 64},
                "r76": {"path": "r76.json", "sha256": "5" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 9,
                "obligation_count": 14,
            },
            "next_action": (
                "Construct one scalar-blind structured factor-base geometry."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "batched-norm-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "batched_nested_norm_node_compiler": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_structured_factor_base_endpoint_compression",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one scalar-blind structured factor-base geometry.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["batched_nested_norm_node_compiler"],
        )

    def test_full_coset_gate_routes_to_compact_divisor_compiler(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "full_multiplicative_x_coset_endpoint"
            ],
            "classification": (
                "FULL_MULTIPLICATIVE_X_COSET_LIFT_MASK_FAILS_ENDPOINT_CAP"
            ),
            "source_bindings": {
                "r80": {"path": "r80.json", "sha256": "6" * 64},
                "r70": {"path": "r70.json", "sha256": "7" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 7,
                "obligation_count": 14,
            },
            "next_action": (
                "Construct one scalar-blind compact divisor factor base."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "full-coset-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "full_multiplicative_x_coset_endpoint": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_compact_divisor_factor_base_endpoint_compiler",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one scalar-blind compact divisor factor base.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["full_multiplicative_x_coset_endpoint"],
        )

    def test_cartesian_sum_gate_routes_to_field_filtration(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "cartesian_sum_compact_divisor"
            ],
            "classification": (
                "CARTESIAN_SUM_COMPACT_S4_PASS__FULL_SOURCE_RHO_FAIL"
            ),
            "source_bindings": {
                "r81": {"path": "r81.json", "sha256": "8" * 64},
                "r14": {"path": "r14.md", "sha256": "9" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 11,
                "obligation_count": 17,
            },
            "next_action": (
                "Test one addition-compatible field-coordinate filtration."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "cartesian-sum-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "cartesian_sum_compact_divisor": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_addition_compatible_5a5c_field_filtration",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one addition-compatible field-coordinate filtration.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["cartesian_sum_compact_divisor"],
        )

    def test_coordinate_filtration_routes_to_marked_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_coordinate_filtration"
            ],
            "classification": (
                "COORDINATE_BUCKET_FILTRATION_ENTROPY_REPLAY_FAIL"
            ),
            "source_bindings": {
                "r82": {"path": "r82.json", "sha256": "a" * 64},
                "registry": {"path": "registry.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 11,
                "obligation_count": 16,
            },
            "next_action": "Test one marked-resultant source section.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "coordinate-filtration-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_coordinate_filtration": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_marked_resultant_source_section",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one marked-resultant source section.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_coordinate_filtration"],
        )

    def test_marked_resultant_routes_to_precoefficient_circuit(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_marked_resultant_source_section"
            ],
            "classification": (
                "EXPLICIT_MARKED_SOURCE_SECTION_EXACT__"
                "COEFFICIENT_BODY_OVER_CAP"
            ),
            "source_bindings": {
                "r83": {"path": "r83.json", "sha256": "a" * 64},
                "p1510": {"path": "p1510.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 9,
                "obligation_count": 17,
            },
            "next_action": "Test one target-uniform pre-coefficient circuit.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "marked-resultant-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_marked_resultant_source_section": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_target_uniform_precoefficient_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one target-uniform pre-coefficient circuit.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_marked_resultant_source_section"],
        )

    def test_precoefficient_circuit_routes_to_sparse_moments(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_target_uniform_precoefficient_circuit"
            ],
            "classification": (
                "FIXED_TARGET_EQUIVARIANT_QUOTIENT_TRIVIAL__"
                "STANDARD_PRECOEFFICIENT_GRAMMARS_OVER_CAP"
            ),
            "source_bindings": {
                "r84": {"path": "r84.json", "sha256": "a" * 64},
                "p1514": {"path": "p1514.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 7,
                "obligation_count": 18,
            },
            "next_action": "Test one sparse multihomogeneous moment recurrence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "precoefficient-circuit-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_target_uniform_precoefficient_circuit": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_sparse_multihomogeneous_moment_recurrence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one sparse multihomogeneous moment recurrence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_target_uniform_precoefficient_circuit"],
        )

    def test_sparse_moments_route_to_jet_pushforward_intertwiner(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_sparse_multihomogeneous_moment_recurrence"
            ],
            "classification": (
                "SUPPLIED_COLORED_NORM_JET_EXACT__"
                "STANDARD_MULTIGRADED_CONSTRUCTOR_PRODUCT_DIMENSION"
            ),
            "source_bindings": {
                "r85": {"path": "r85.json", "sha256": "a" * 64},
                "p1536": {"path": "p1536.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 8,
                "obligation_count": 18,
            },
            "next_action": "Test one jet-preserving A+C intertwiner.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "sparse-moment-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_sparse_multihomogeneous_moment_recurrence": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_jet_preserving_addition_pushforward_intertwiner",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one jet-preserving A+C intertwiner.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_sparse_multihomogeneous_moment_recurrence"],
        )

    def test_jet_pushforward_routes_to_black_box_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_jet_preserving_addition_pushforward"
            ],
            "classification": (
                "TARGET_LOCAL_FIRST_NORM_JET_NONFUNCTORIAL__"
                "EXPLICIT_TRANSLATED_REMAINDER_FULL_B2"
            ),
            "source_bindings": {
                "r86": {"path": "r86.json", "sha256": "a" * 64},
                "p1515": {"path": "p1515.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 6,
                "obligation_count": 16,
            },
            "next_action": "Test one black-box translated resultant.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "jet-pushforward-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_jet_preserving_addition_pushforward": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_black_box_translated_resultant_gcd_localizer",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one black-box translated resultant.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_jet_preserving_addition_pushforward"],
        )

    def test_black_box_resultant_routes_to_fixed_marker_recurrence(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_black_box_resultant_localizer"
            ],
            "classification": (
                "ORACLE_SOURCE_LOCALIZER_LOGARITHMIC__"
                "SCALAR_BLOCK_KRYLOV_OR_HALF_GCD_OVER_CAP"
            ),
            "source_bindings": {
                "r87": {"path": "r87.json", "sha256": "a" * 64},
                "p1513": {"path": "p1513.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 6,
                "obligation_count": 16,
            },
            "next_action": "Test one fixed-marker scalar recurrence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "black-box-resultant-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_black_box_resultant_localizer": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_coefficient_free_fixed_marker_resultant_recurrence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one fixed-marker scalar recurrence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_black_box_resultant_localizer"],
        )

    def test_fixed_marker_recurrence_routes_to_nonlocal_sketch(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_fixed_marker_scalar_recurrence"
            ],
            "classification": (
                "FIXED_MARKER_LOCAL_JET_NONFUNCTORIAL__"
                "EXPLICIT_SHIFT_RECURRENCE_REACHES_C4_B2P4"
            ),
            "source_bindings": {
                "r88": {"path": "r88.json", "sha256": "a" * 64},
                "p1514": {"path": "p1514.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 6,
                "obligation_count": 16,
            },
            "next_action": "Test one nonlocal nonlinear translation sketch.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "fixed-marker-recurrence-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_fixed_marker_scalar_recurrence": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_nonlocal_nonlinear_translation_sketch",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one nonlocal nonlinear translation sketch.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_fixed_marker_scalar_recurrence"],
        )

    def test_nonlocal_moment_hankel_routes_to_unequal_list_index(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_nonlocal_moment_hankel_translation"
            ],
            "classification": (
                "NONLOCAL_MOMENT_UPDATE_EXACT__"
                "HANKEL_PADE_ORDER_C5_B3_OVER_CAP"
            ),
            "source_bindings": {
                "r89": {"path": "r89.json", "sha256": "a" * 64},
                "p1515": {"path": "p1515.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 8,
                "obligation_count": 17,
            },
            "next_action": "Test one unequal-list subfunction index.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "nonlocal-moment-hankel-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_nonlocal_moment_hankel_translation": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_unequal_list_subfunction_inversion_index",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one unequal-list subfunction index.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_nonlocal_moment_hankel_translation"],
        )

    def test_unequal_list_index_routes_to_compact_elliptic_map(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_unequal_list_subfunction_inversion"
            ],
            "classification": (
                "UNEQUAL_LIST_SUBFUNCTION_THEOREM_EXACT__"
                "BEST_ONLINE_COMPATIBLE_SETUP_B4P4__"
                "INTEGER_RESIDUE_MAP_NO_GENERIC_GROUP_TRANSFER"
            ),
            "source_bindings": {
                "r90": {"path": "r90.json", "sha256": "a" * 64},
                "paper": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 9,
                "obligation_count": 20,
            },
            "next_action": "Test one compact elliptic subfunction map.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "unequal-list-subfunction-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_unequal_list_subfunction_inversion": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_compact_elliptic_subfunction_map",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one compact elliptic subfunction map.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_unequal_list_subfunction_inversion"],
        )

    def test_compact_elliptic_map_routes_to_shared_correspondence(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_compact_elliptic_subfunction_map"
            ],
            "classification": (
                "ELLIPTIC_RATIONAL_FIBER_ENDPOINT_TR_EXACT__"
                "THEOREM_STATE_MIN_B10O3__"
                "ONLINE_COMPATIBLE_MIN_B35O8__"
                "FIVE_A_FIVE_C_SOURCE_TR_ABSENT"
            ),
            "source_bindings": {
                "r91": {"path": "r91.json", "sha256": "a" * 64},
                "paper": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 22,
            },
            "next_action": "Test one shared semilinear correspondence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "compact-elliptic-subfunction-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_compact_elliptic_subfunction_map": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_shared_semilinear_incidence_correspondence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one shared semilinear correspondence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_compact_elliptic_subfunction_map"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_compact_elliptic_subfunction_map_scope",
            ambiguity_ids,
        )

    def test_shared_correspondence_routes_to_implicit_range_index(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_shared_semilinear_incidence_correspondence"
            ],
            "classification": (
                "CONSTANT_S3_RESULTANT_OPERATOR_EXACT__"
                "ZERO_INCIDENCE_FULL_RANK_AT_78_CHARTS__"
                "ROOT_SOURCE_FEATURE_ROWS_B12O5__"
                "PROJECTIVE_SOURCE_UNRANKING_ABSENT"
            ),
            "source_bindings": {
                "r92": {"path": "r92.json", "sha256": "a" * 64},
                "r84": {"path": "r84.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 9,
                "obligation_count": 20,
            },
            "next_action": "Test one implicit Veronese range index.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "shared-semilinear-incidence-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_shared_semilinear_incidence_correspondence": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_implicit_veronese_hyperplane_source_index",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one implicit Veronese range index.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_shared_semilinear_incidence_correspondence"],
        )

    def test_implicit_veronese_index_routes_to_aggregate_recurrence(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_implicit_veronese_hyperplane_source_index"
            ],
            "classification": (
                "IMPLICIT_VERONESE_QUERY_EQUALS_FERMAT_PROJECTOR__"
                "DIRECT_DYADIC_SOURCE_EXACT__STANDARD_MULTIPOINT_B12O5__"
                "CAP_SIZED_CONTRACTION_UNSUPPLIED"
            ),
            "source_bindings": {
                "r93": {"path": "r93.json", "sha256": "a" * 64},
                "r9": {"path": "r9.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 23,
            },
            "next_action": "Test one aggregate projector recurrence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "implicit-veronese-index-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_implicit_veronese_hyperplane_source_index": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_aggregate_veronese_projector_recurrence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one aggregate projector recurrence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_implicit_veronese_hyperplane_source_index"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_implicit_veronese_hyperplane_index_scope",
            ambiguity_ids,
        )

    def test_aggregate_projector_routes_to_frobenius_trace(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_aggregate_veronese_projector_recurrence"
            ],
            "classification": (
                "AGGREGATE_FERMAT_MOMENT_IDENTITY_EXACT__"
                "VERONESE_QUOTIENT_PAIRING_FULL_RANK_P3_TO_P29__"
                "CANONICAL_MOMENT_STATE_B10__NONMOMENT_RECURRENCE_OPEN"
            ),
            "source_bindings": {
                "r94": {"path": "r94.json", "sha256": "a" * 64},
                "r78": {"path": "r78.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 11,
                "obligation_count": 25,
            },
            "next_action": "Test one modular Frobenius trace.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "aggregate-projector-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_aggregate_veronese_projector_recurrence": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_modular_frobenius_trace_recurrence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one modular Frobenius trace.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_aggregate_veronese_projector_recurrence"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_aggregate_veronese_projector_scope",
            ambiguity_ids,
        )

    def test_frobenius_trace_routes_to_factored_transpose(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_modular_frobenius_trace_recurrence"
            ],
            "classification": (
                "SPLIT_QUOTIENT_PROJECTOR_TRACE_EXACT__"
                "REDUCED_FROBENIUS_IDENTITY__NONREDUCED_FROBENIUS_"
                "LOSES_NILPOTENT_SOURCE_STATE__STANDARD_QUOTIENT_B12O5"
            ),
            "source_bindings": {
                "r95": {"path": "r95.json", "sha256": "a" * 64},
                "r84": {"path": "r84.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 27,
            },
            "next_action": "Test one factored transposed trace.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "frobenius-trace-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_modular_frobenius_trace_recurrence": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_factored_transposed_projector_trace",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one factored transposed trace.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_modular_frobenius_trace_recurrence"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_modular_frobenius_trace_scope",
            ambiguity_ids,
        )

    def test_factored_transpose_routes_to_nonlinear_tensor_tower(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_factored_transposed_projector_trace"
            ],
            "classification": (
                "POINTWISE_PROJECTOR_JACOBIAN_AND_DYADIC_ADJOINT_"
                "FULL_RANK__UNIQUE_PRODUCT_GRADIENT_LOCALIZER__"
                "MULTIZERO_GRADIENT_COLLAPSE__STANDARD_TRANSPOSE_B12O5"
            ),
            "source_bindings": {
                "r96": {"path": "r96.json", "sha256": "a" * 64},
                "r88": {"path": "r88.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 28,
            },
            "next_action": "Test one nonlinear tensor-tower trace.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "factored-transpose-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_factored_transposed_projector_trace": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_nonlinear_tensor_tower_trace",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one nonlinear tensor-tower trace.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_factored_transposed_projector_trace"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_factored_transposed_projector_scope",
            ambiguity_ids,
        )

    def test_nonlinear_tensor_tower_routes_to_digitized_projector(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_nonlinear_tensor_tower_trace"
            ],
            "classification": (
                "MONIC_QUADRATIC_RESULTANT_PROJECTOR_IS_EQUALITY_"
                "KERNEL__ONE_BOND_NONLINEAR_ENCODER_WIDTH_P__"
                "RESTRICTED_DISTINCT_WIDTH_D__DIGITIZED_MULTI_EDGE_OPEN"
            ),
            "source_bindings": {
                "r97": {"path": "r97.json", "sha256": "a" * 64},
                "r95": {"path": "r95.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 14,
                "obligation_count": 30,
            },
            "next_action": "Test one multi-edge digitized projector.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "nonlinear-tensor-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_nonlinear_tensor_tower_trace": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_multiedge_digitized_equality_projector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one multi-edge digitized projector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_nonlinear_tensor_tower_trace"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_nonlinear_tensor_tower_scope",
            ambiguity_ids,
        )

    def test_digitized_projector_routes_to_aggregate_digit_trie(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_multiedge_digitized_equality_projector"
            ],
            "classification": (
                "SUPPLIED_RADIX_DIGITS_DECOMPOSE_EQUALITY_EXACTLY__"
                "FLATTENED_CAPACITY_STILL_P__STANDARD_DIGIT_TABLE_B5__"
                "SOURCEWISE_DIGIT_TRAFFIC_B12O5__AGGREGATE_DIGIT_INDEX_OPEN"
            ),
            "source_bindings": {
                "r98": {"path": "r98.json", "sha256": "a" * 64},
                "r94": {"path": "r94.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 31,
            },
            "next_action": "Test one succinct aggregate digit trie.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "digitized-projector-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_multiedge_digitized_equality_projector": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_succinct_aggregate_digit_trie",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one succinct aggregate digit trie.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_multiedge_digitized_equality_projector"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_multiedge_digitized_projector_scope",
            ambiguity_ids,
        )

    def test_aggregate_digit_trie_routes_to_actual_image_theorem(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_succinct_aggregate_digit_trie"
            ],
            "classification": (
                "ARBITRARY_D_SUBSET_EXACT_INDEX_NEEDS_OMEGA_D_WORDS__"
                "EXPLICIT_AND_PATRICIA_TRIES_THETA_D__STRUCTURED_INTERVAL_"
                "CONSTANT_STATE_POSITIVE__ACTUAL_DIVISOR_IMAGE_OPEN"
            ),
            "source_bindings": {
                "r99": {"path": "r99.json", "sha256": "a" * 64},
                "r84": {"path": "r84.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 31,
            },
            "next_action": "Test one actual divisor-image theorem.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "aggregate-digit-trie-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_succinct_aggregate_digit_trie": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_actual_divisor_image_entropy_merge",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one actual divisor-image theorem.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_succinct_aggregate_digit_trie"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_succinct_aggregate_digit_trie_scope",
            ambiguity_ids,
        )

    def test_actual_divisor_image_oracles_route_to_two_sided_join(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_actual_divisor_image_entropy_merge"
            ],
            "classification": (
                "ACTUAL_3A2C_AND_2A3C_LOCAL_COUNT_SOURCE_ORACLES_PASS_CAPS__"
                "WEIGHTED_C_SUMMARIES_LEAF_FREE__FULL_TWO_SIDED_JOIN_OPEN"
            ),
            "source_bindings": {
                "r100": {"path": "r100.json", "sha256": "a" * 64},
                "r84": {"path": "r84.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 19,
                "obligation_count": 32,
            },
            "next_action": "Test one two-sided implicit join.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "actual-divisor-image-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_actual_divisor_image_entropy_merge": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_two_sided_implicit_join",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one two-sided implicit join.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_actual_divisor_image_entropy_merge"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_actual_divisor_image_entropy_merge_scope",
            ambiguity_ids,
        )

    def test_two_sided_join_routes_to_target_forced_algebraic_filter(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_two_sided_implicit_join"
            ],
            "classification": (
                "CANONICAL_4A1C_PREFIX_JOIN_EXACT_WITH_B11O5_STATE__"
                "B14O5_FRESH_QUERY_OVER_CAP__DENSITY_ONE_DISJOINT_FILTER_"
                "PRUNING_PAID_BACK_BY_REPETITIONS__TARGET_FORCED_"
                "ALGEBRAIC_JOIN_OPEN"
            ),
            "source_bindings": {
                "r101": {"path": "r101.json", "sha256": "a" * 64},
                "r84": {"path": "r84.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 34,
            },
            "next_action": "Test one target-forced algebraic filter.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "two-sided-join-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_two_sided_implicit_join": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_target_forced_algebraic_join_filter",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one target-forced algebraic filter.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_two_sided_implicit_join"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_two_sided_implicit_join_scope",
            ambiguity_ids,
        )

    def test_target_forced_filter_routes_to_compact_preendpoint_pushdown(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_target_forced_algebraic_join_filter"
            ],
            "classification": (
                "TARGET_FORCED_S3_FILTER_EXACT__REGULAR_ROOTS_X_T_MINUS_L_"
                "AND_X_T_PLUS_L__SIGN_BRANCH_CONSTANT_ONLY__POINTWISE_"
                "B16O5_OR_CANONICAL_B14O5__STANDARD_AGGREGATE_AND_FFE_"
                "BODIES_OVER_CAP__NONSTANDARD_COMPACT_PREENDPOINT_"
                "PUSHDOWN_OPEN"
            ),
            "source_bindings": {
                "r102": {"path": "r102.json", "sha256": "a" * 64},
                "r70": {"path": "r70.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 21,
                "obligation_count": 32,
            },
            "next_action": "Test one compact pre-endpoint S3 pushdown.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "target-forced-filter-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_target_forced_algebraic_join_filter": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_compact_preendpoint_s3_ffe_pushdown",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one compact pre-endpoint S3 pushdown.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_target_forced_algebraic_join_filter"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_target_forced_algebraic_join_filter_scope",
            ambiguity_ids,
        )

    def test_preendpoint_pushdown_routes_to_actual_deck_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_compact_preendpoint_s3_ffe_pushdown"
            ],
            "classification": (
                "SIGN_RESOLVED_S3_IS_GROUP_COEFFICIENT__EXPLICIT_REVERSE_"
                "RESIDUAL_RECURRENCE_EXACT__B14O5_RESIDUAL_BODY__ACTUAL_"
                "DECK_SPECIFIC_NONMERGEABLE_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r103": {"path": "r103.json", "sha256": "a" * 64},
                "r102": {"path": "r102.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 22,
                "obligation_count": 33,
            },
            "next_action": "Test one actual-deck non-mergeable circuit.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "preendpoint-pushdown-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_compact_preendpoint_s3_ffe_pushdown": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_actual_deck_nonmergeable_target_pullback_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one actual-deck non-mergeable circuit.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_compact_preendpoint_s3_ffe_pushdown"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_compact_preendpoint_s3_ffe_pushdown_scope",
            ambiguity_ids,
        )

    def test_actual_deck_pullback_routes_to_scalar_norm_constructor(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_actual_deck_nonmergeable_target_pullback"
            ],
            "classification": (
                "WHOLE_DECK_NORM_JET_SOURCE_ADJOINT_EXACT__"
                "SCALAR_TARGET_NORM_CONSTRUCTOR_UNSUPPLIED"
            ),
            "source_bindings": {
                "r104": {"path": "r104.json", "sha256": "a" * 64},
                "r89": {"path": "r89.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 18,
                "obligation_count": 29,
            },
            "next_action": "Test the scalar target norm constructor.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "actual-deck-pullback-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_actual_deck_nonmergeable_target_pullback": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_scalar_target_norm_count_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test the scalar target norm constructor.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_actual_deck_nonmergeable_target_pullback"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_actual_deck_nonmergeable_target_pullback_scope",
            ambiguity_ids,
        )

    def test_scalar_norm_lane_routes_to_noncharacter_algebraic_constructor(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_scalar_target_norm_count_circuit"
            ],
            "classification": (
                "TAO_PRIME_CYCLIC_UNCERTAINTY_FORCES_B5_LIVE_CHARACTER_"
                "MODES__NONCHARACTER_RESULTANT_NORM_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r105": {"path": "r105.json", "sha256": "a" * 64},
                "r104": {"path": "r104.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 26,
            },
            "next_action": "Test one non-character algebraic norm circuit.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "scalar-target-norm-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_scalar_target_norm_count_circuit": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_noncharacter_algebraic_target_norm_resultant_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one non-character algebraic norm circuit.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_scalar_target_norm_count_circuit"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_scalar_target_norm_count_circuit_scope",
            ambiguity_ids,
        )

    def test_noncharacter_resultant_routes_to_factored_chow_recurrence(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_noncharacter_algebraic_target_norm_resultant"
            ],
            "classification": (
                "BALANCED_NONCHARACTER_ROOT_RESULTANT_REQUIRES_B13O5_"
                "INTERFACE__FACTORED_ELLIPTIC_LAMBDA_RING_CHOW_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r106": {"path": "r106.json", "sha256": "a" * 64},
                "r105": {"path": "r105.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 27,
            },
            "next_action": "Test one factored elliptic Chow recurrence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "noncharacter-resultant-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_noncharacter_algebraic_target_norm_resultant": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_factored_elliptic_lambda_ring_chow_norm_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one factored elliptic Chow recurrence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_noncharacter_algebraic_target_norm_resultant"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_noncharacter_algebraic_target_norm_resultant_scope",
            ambiguity_ids,
        )

    def test_lambda_ring_chow_routes_to_poincare_section_rank(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_factored_elliptic_lambda_ring_chow_norm"
            ],
            "classification": (
                "DEGREE_FIVE_CYCLE_INDEX_GIVES_EXACT_CANONICAL_DIVISOR__"
                "POINCARE_THETA_SECTION_FACTORIZATION_OPEN"
            ),
            "source_bindings": {
                "r107": {"path": "r107.json", "sha256": "a" * 64},
                "r105": {"path": "r105.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 17,
                "obligation_count": 30,
            },
            "next_action": "Test one Poincare target-section factorization.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "lambda-ring-chow-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_factored_elliptic_lambda_ring_chow_norm": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_poincare_theta_target_section_rank",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one Poincare target-section factorization.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_factored_elliptic_lambda_ring_chow_norm"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_factored_elliptic_lambda_ring_chow_norm_scope",
            ambiguity_ids,
        )

    def test_poincare_section_rank_routes_to_theta_addition_network(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_poincare_theta_target_section_rank"
            ],
            "classification": (
                "TARGET_EQUALITY_FIBER_HAS_NO_UNARY_OR_PAIRWISE_ZERO_"
                "CYLINDER__TRANSLATED_SIGNED_SECTIONS_HAVE_B12O5_UNIFORM_"
                "FLATTENING_RANK__RATIONAL_THETA_CANCELLATION_NETWORK_OPEN"
            ),
            "source_bindings": {
                "r108": {"path": "r108.json", "sha256": "a" * 64},
                "r107": {"path": "r107.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 26,
            },
            "next_action": "Test one exact theta-addition cancellation network.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "poincare-section-rank-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_poincare_theta_target_section_rank": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_theta_addition_cancellation_network",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one exact theta-addition cancellation network.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_poincare_theta_target_section_rank"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_poincare_theta_target_section_rank_scope",
            ambiguity_ids,
        )

    def test_theta_addition_routes_to_finite_deck_annihilator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_theta_addition_cancellation_network"
            ],
            "classification": (
                "POSITION_SEPARATED_L11_ABEL_ALTERNANT_GIVES_EXACT_FINITE_"
                "FIELD_TARGET_ZERO_BICONDITIONAL__FINITE_DECK_ANNIHILATOR_"
                "CONTRACTION_OPEN"
            ),
            "source_bindings": {
                "r109": {"path": "r109.json", "sha256": "a" * 64},
                "r108": {"path": "r108.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": "Test one finite-deck alternant annihilator.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "theta-addition-network-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_theta_addition_cancellation_network": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_finite_deck_alternant_annihilator_contraction",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one finite-deck alternant annihilator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_theta_addition_cancellation_network"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_theta_addition_cancellation_network_scope",
            ambiguity_ids,
        )

    def test_finite_deck_annihilator_routes_to_query2p1_index(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_finite_deck_alternant_annihilator"
            ],
            "classification": (
                "MINIMAL_RAW_VALUE_ZERO_MASK_DEGREE_EQUALS_DISTINCT_"
                "NONZERO_VALUE_COUNT__GAUGE_NORMALIZED_QUERY2P1_OPEN"
            ),
            "source_bindings": {
                "r110": {"path": "r110.json", "sha256": "a" * 64},
                "registry": {"path": "registry.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 19,
            },
            "next_action": (
                "Test one gauge-normalized endpoint Query2P1 index."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "finite-deck-annihilator-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_finite_deck_alternant_annihilator": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_gauge_normalized_endpoint_query2p1_index",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one gauge-normalized endpoint Query2P1 index.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_finite_deck_alternant_annihilator"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_finite_deck_alternant_annihilator_scope",
            ambiguity_ids,
        )

    def test_query2p1_routes_to_nonlinear_orbit_product(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_gauge_normalized_endpoint_query2p1"
            ],
            "classification": (
                "TYPED_ENDPOINT_NORMALIZATION_EXACT__CANONICAL_C2XC3_"
                "QUERY_RETAINS_B3__NONLINEAR_ORBIT_PRODUCT_OPEN"
            ),
            "source_bindings": {
                "r111": {"path": "r111.json", "sha256": "a" * 64},
                "registry": {"path": "registry.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 18,
            },
            "next_action": (
                "Test one nonlinear elliptic orbit-product recurrence."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "endpoint-query2p1-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_gauge_normalized_endpoint_query2p1": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_nonlinear_elliptic_orbit_product_recurrence",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one nonlinear elliptic orbit-product recurrence.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_gauge_normalized_endpoint_query2p1"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_gauge_normalized_endpoint_query2p1_scope",
            ambiguity_ids,
        )

    def test_orbit_product_routes_to_transposed_leaf_generator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_nonlinear_elliptic_orbit_product"
            ],
            "classification": (
                "C3_FIXED_TRANSLATION_ORBIT_FALSE__ORDER_ONE_PRODUCT_"
                "RETAINS_B3_LEAVES__TRANSPOSED_C5_GENERATOR_OPEN"
            ),
            "source_bindings": {
                "r112": {"path": "r112.json", "sha256": "a" * 64},
                "registry": {"path": "registry.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 8,
                "obligation_count": 16,
            },
            "next_action": (
                "Test one transposed nonuniform C5 leaf generator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "orbit-product-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_nonlinear_elliptic_orbit_product": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_5a5c_transposed_nonuniform_c5_leaf_generator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one transposed nonuniform C5 leaf generator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_nonlinear_elliptic_orbit_product"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_nonlinear_elliptic_orbit_product_scope",
            ambiguity_ids,
        )

    def test_transposed_leaf_routes_to_exponent_rebalance(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "5a5c_transposed_nonuniform_c5_leaf_generator"
            ],
            "classification": (
                "PRODUCT_ADJOINT_RETAINS_B3_LEAF_TRACE__PRELEAF_"
                "TRANSPOSE_REACHES_B13O5__EXPONENT_REBALANCE_OPEN"
            ),
            "source_bindings": {
                "r113": {"path": "r113.json", "sha256": "a" * 64},
                "registry": {"path": "registry.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 7,
                "obligation_count": 15,
            },
            "next_action": (
                "Freeze the relation-arity and asymmetric factor-base "
                "exponent feasibility inequalities."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "transposed-leaf-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "5a5c_transposed_nonuniform_c5_leaf_generator": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s6_relation_arity_factor_base_transposed_interface_rebalance",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Freeze the relation-arity and asymmetric factor-base "
            "exponent feasibility inequalities.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["5a5c_transposed_nonuniform_c5_leaf_generator"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "5a5c_transposed_nonuniform_c5_leaf_generator_scope",
            ambiguity_ids,
        )

    def test_exponent_rebalance_routes_to_implicit_3f_locator(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "relation_arity_factor_base_transposed_interface_rebalance"
            ],
            "classification": (
                "M6_ALPHA1O12_BETA3O4_NECESSARY_VERTEX__DIRECT_"
                "3F_SELF_JOIN_REMAINS_B5O2__IMPLICIT_LOCATOR_OPEN"
            ),
            "source_bindings": {
                "r114": {"path": "r114.json", "sha256": "a" * 64},
                "r82": {"path": "r82.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 16,
            },
            "next_action": (
                "Construct one exact implicit 3F self-convolution/source "
                "locator at the frozen m=6 vertex."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "exponent-rebalance-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    (
                        "relation_arity_factor_base_transposed_"
                        "interface_rebalance"
                    ): (probe, probe_path)
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s7_m6_implicit_3f_self_convolution_ffe_source_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one exact implicit 3F self-convolution/source "
            "locator at the frozen m=6 vertex.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["relation_arity_factor_base_transposed_interface_rebalance"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            (
                "relation_arity_factor_base_transposed_interface_"
                "rebalance_scope"
            ),
            ambiguity_ids,
        )

    def test_m6_pair_sum_routes_to_target_batched_elliptic_transpose(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_a6_batched_c3_pair_sum_source_locator"
            ],
            "classification": (
                "A6_BATCHED_C3_PAIR_SUM_EXACT__CURRENT_INDEXING_AND_"
                "EXPLICIT_DIVISOR_ROUTES_OVER_CAP__ELLIPTIC_TRANSPOSE_OPEN"
            ),
            "source_bindings": {
                "r115": {"path": "r115.json", "sha256": "a" * 64},
                "r82": {"path": "r82.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 9,
                "obligation_count": 16,
            },
            "next_action": (
                "Construct one exact target-batched elliptic coefficient "
                "functional and source adjoint."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-pair-sum-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_a6_batched_c3_pair_sum_source_locator": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s8_m6_target_batched_c3_pair_sum_elliptic_transpose",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one exact target-batched elliptic coefficient "
            "functional and source adjoint.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_a6_batched_c3_pair_sum_source_locator"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_a6_batched_c3_pair_sum_source_locator_scope",
            ambiguity_ids,
        )

    def test_m6_elliptic_transpose_routes_to_nonlinear_c6_locator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_target_batched_c3_elliptic_transpose"
            ],
            "classification": (
                "FULL_CHARACTERISTIC_ZERO_TRANSLATION_RANK__NONLINEAR_"
                "VALUE_SENSITIVE_SOURCE_LOCATOR_OPEN"
            ),
            "source_bindings": {
                "r116": {"path": "r116.json", "sha256": "a" * 64},
                "r77": {"path": "r77.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 16,
            },
            "next_action": (
                "Construct one nonlinear value-sensitive six-C source "
                "locator with explicit branch and source costs."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-elliptic-transpose-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_target_batched_c3_elliptic_transpose": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s9_m6_nonlinear_value_sensitive_c6_source_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one nonlinear value-sensitive six-C source "
            "locator with explicit branch and source costs.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_target_batched_c3_elliptic_transpose"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_target_batched_c3_elliptic_transpose_scope",
            ambiguity_ids,
        )

    def test_m6_nonlinear_c6_routes_to_output_sensitive_c5_index(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_nonlinear_value_sensitive_c6_source_locator"
            ],
            "classification": (
                "ONE_C_BRANCH_EXACT__OUTPUT_SENSITIVE_C5_INDEX_OPEN"
            ),
            "source_bindings": {
                "r117": {"path": "r117.json", "sha256": "a" * 64},
                "multipoint": {
                    "path": "multipoint.pdf",
                    "sha256": "b" * 64,
                },
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 17,
            },
            "next_action": (
                "Construct one output-sensitive nonlinear C5 membership "
                "and source index with polylogarithmic exact query."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-nonlinear-c6-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_nonlinear_value_sensitive_c6_source_locator": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s10_m6_output_sensitive_nonlinear_c5_source_index",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one output-sensitive nonlinear C5 membership "
            "and source index with polylogarithmic exact query.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_nonlinear_value_sensitive_c6_source_locator"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_nonlinear_value_sensitive_c6_source_locator_scope",
            ambiguity_ids,
        )

    def test_m6_output_sensitive_c5_routes_to_implicit_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_output_sensitive_nonlinear_c5_source_index"
            ],
            "classification": (
                "RANDOM_C5_SUPPORT_LARGE__IMPLICIT_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r118": {"path": "r118.json", "sha256": "a" * 64},
                "r82": {"path": "r82.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 10,
                "obligation_count": 17,
            },
            "next_action": (
                "Construct one sub-output implicit nonlinear C5 "
                "membership/source circuit with exact empty rejection."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-output-sensitive-c5-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_output_sensitive_nonlinear_c5_source_index": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s11_m6_suboutput_implicit_c5_membership_source_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one sub-output implicit nonlinear C5 "
            "membership/source circuit with exact empty rejection.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_output_sensitive_nonlinear_c5_source_index"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_output_sensitive_nonlinear_c5_source_index_scope",
            ambiguity_ids,
        )

    def test_m6_character_pairing_routes_to_small_k_product_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_suboutput_implicit_c5_character_pairing"
            ],
            "classification": (
                "PAIRING_REENCODES_C5__SMALL_K_PRODUCT_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r119": {"path": "r119.json", "sha256": "a" * 64},
                "pairing": {"path": "pairing.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 19,
            },
            "next_action": (
                "Construct one exact small-k multiplicative C5 "
                "membership/source circuit and fail closed otherwise."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-character-pairing-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_suboutput_implicit_c5_character_pairing": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s12_m6_small_k_multiplicative_c5_membership_source_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one exact small-k multiplicative C5 "
            "membership/source circuit and fail closed otherwise.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_suboutput_implicit_c5_character_pairing"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_suboutput_implicit_c5_character_pairing_scope",
            ambiguity_ids,
        )

    def test_m6_moment_torus_routes_to_target_specialized_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "m6_small_k_multiplicative_c5_moment_torus"
            ],
            "classification": (
                "TORUS_FORM_EXACT__FULL_MOMENT_OVER_CAP__"
                "TARGET_SPECIALIZED_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r120": {"path": "r120.json", "sha256": "a" * 64},
                "paper": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one target-injected nonlinear torus C5 "
                "membership/source circuit outside full moments."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-moment-torus-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "m6_small_k_multiplicative_c5_moment_torus": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s13_m6_target_specialized_nonlinear_torus_c5_source_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one target-injected nonlinear torus C5 "
            "membership/source circuit outside full moments.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["m6_small_k_multiplicative_c5_moment_torus"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_small_k_multiplicative_c5_moment_torus_scope",
            ambiguity_ids,
        )

    def test_torus_split_rebalance_routes_to_nonoccurrence_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "torus_c5_explicit_split_global_rebalance"
            ],
            "classification": (
                "EXPLICIT_SPLIT_ALL_ARITIES_ABOVE_RHO__"
                "NONOCCURRENCE_TORUS_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r121": {"path": "r121.json", "sha256": "a" * 64},
                "r115": {"path": "r115.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 11,
                "obligation_count": 18,
            },
            "next_action": (
                "Construct one target-specialized nonoccurrence torus C5 "
                "circuit outside all explicit split tables."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-split-rebalance-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "torus_c5_explicit_split_global_rebalance": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s14_target_specialized_nonoccurrence_torus_c5_source_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one target-specialized nonoccurrence torus C5 "
            "circuit outside all explicit split tables.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["torus_c5_explicit_split_global_rebalance"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_explicit_split_global_rebalance_scope",
            ambiguity_ids,
        )

    def test_torus_fourier_resultant_routes_to_nonrepresented_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "torus_c5_fourier_product_resultant"
            ],
            "classification": (
                "FOURIER_AND_PRODUCT_RESULTANT_EXACT__"
                "NONREPRESENTED_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r122": {"path": "r122.json", "sha256": "a" * 64},
                "paper": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 19,
            },
            "next_action": (
                "Construct one nonrepresented Fourier/resultant torus C5 "
                "source circuit outside all represented bodies."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-fourier-resultant-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "torus_c5_fourier_product_resultant": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s15_nonrepresented_fourier_resultant_torus_c5_source_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one nonrepresented Fourier/resultant torus C5 "
            "source circuit outside all represented bodies.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["torus_c5_fourier_product_resultant"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_fourier_product_resultant_scope",
            ambiguity_ids,
        )

    def test_torus_linear_sketch_routes_to_coupled_nonlinear_zero_test(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[
                "torus_c5_linear_sketch_circulant"
            ],
            "classification": (
                "UNIVERSAL_LINEAR_SKETCH_FULL_RANK__"
                "COUPLED_NONLINEAR_DATA_STRUCTURE_OPEN"
            ),
            "source_bindings": {
                "r123": {"path": "r123.json", "sha256": "a" * 64},
                "r114": {"path": "r114.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one coupled nonlinear torus C5 zero-test "
                "outside universal linear sketches."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-linear-sketch-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    "torus_c5_linear_sketch_circulant": (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s16_coupled_nonlinear_torus_c5_zero_test",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one coupled nonlinear torus C5 zero-test "
            "outside universal linear sketches.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            ["torus_c5_linear_sketch_circulant"],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_linear_sketch_circulant_scope",
            ambiguity_ids,
        )

    def test_torus_homomorphic_fingerprint_routes_to_nonhomomorphic(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_prime_order_homomorphic_fingerprint"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "PRIME_HOMOMORPHISMS_TRIVIAL_OR_INJECTIVE__"
                "NONHOMOMORPHIC_ADAPTIVE_OPEN"
            ),
            "source_bindings": {
                "r124": {"path": "r124.json", "sha256": "a" * 64},
                "r120": {"path": "r120.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one nonhomomorphic adaptive torus C5 "
                "fingerprint with charged corrections."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-homomorphic-fingerprint-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s17_nonhomomorphic_adaptive_torus_c5_fingerprint",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one nonhomomorphic adaptive torus C5 "
            "fingerprint with charged corrections.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_prime_order_homomorphic_fingerprint_scope",
            ambiguity_ids,
        )

    def test_torus_explicit_hash_corrections_route_to_implicit_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_explicit_hash_correction_support"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "EXPLICIT_CORRECTIONS_AT_LEAST_C5__"
                "IMPLICIT_ADAPTIVE_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r125": {"path": "r125.json", "sha256": "a" * 64},
                "r119": {"path": "r119.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one implicit adaptive torus C5 correction "
                "circuit outside represented product lists."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-explicit-correction-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s18_implicit_adaptive_torus_c5_hash_correction_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one implicit adaptive torus C5 correction "
            "circuit outside represented product lists.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_explicit_hash_correction_support_scope",
            ambiguity_ids,
        )

    def test_torus_bucket_tradeoff_routes_to_singleton_c3_router(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_bucket_resultant_routing_tradeoff"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "BUCKET_RESULTANT_TRADEOFF__"
                "SINGLETON_C3_CONSTANT_ROUTER_OPEN"
            ),
            "source_bindings": {
                "r126": {"path": "r126.json", "sha256": "a" * 64},
                "paper": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one cap-tight singleton-C3 constant-pair "
                "target router with source return."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-bucket-tradeoff-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s19_cap_tight_singleton_c3_target_router",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one cap-tight singleton-C3 constant-pair "
            "target router with source return.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_bucket_resultant_routing_tradeoff_scope",
            ambiguity_ids,
        )

    def test_torus_rational_selector_routes_to_low_slp_piecewise(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_rational_selector_degree"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "RATIONAL_SELECTOR_DEGREE_B9O4__"
                "LOW_SLP_PIECEWISE_OPEN"
            ),
            "source_bindings": {
                "r127": {"path": "r127.json", "sha256": "a" * 64},
                "r114": {"path": "r114.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one high-degree low-SLP or compact piecewise "
                "torus C5 selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-rational-selector-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s20_low_slp_piecewise_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one high-degree low-SLP or compact piecewise "
            "torus C5 selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_rational_selector_degree_scope",
            ambiguity_ids,
        )

    def test_torus_piecewise_selector_routes_to_shared_predicate_dag(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_piecewise_selector_decision_dag"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "TURAN_B3O2_PIECEWISE_BRANCHES__"
                "SHARED_PREDICATE_DAG_OPEN"
            ),
            "source_bindings": {
                "r128": {"path": "r128.json", "sha256": "a" * 64},
                "r121": {"path": "r121.md", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one compact shared-predicate torus C5 "
                "selector DAG."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-piecewise-selector-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s21_shared_predicate_torus_c5_selector_dag",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one compact shared-predicate torus C5 selector DAG.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_piecewise_selector_decision_dag_scope",
            ambiguity_ids,
        )

    def test_torus_fourier_transfer_routes_to_order_two_predicate(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_sparse_fourier_predicate_transfer"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "COMPLEX_UNCERTAINTY_ONLY__"
                "ORDER2_FINITE_FIELD_PREDICATE_OPEN"
            ),
            "source_bindings": {
                "r129": {"path": "r129.json", "sha256": "a" * 64},
                "tao": {"path": "tao.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one actual-field order-two torus C5 "
                "selector predicate."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-fourier-transfer-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s22_order2_finite_field_torus_c5_selector_predicate",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one actual-field order-two torus C5 "
            "selector predicate.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_sparse_fourier_predicate_transfer_scope",
            ambiguity_ids,
        )

    def test_torus_consecutive_modes_route_to_lacunary_predicate(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_consecutive_mode_predicate"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "DENSE_CONSECUTIVE_MODES_CLOSED__"
                "LACUNARY_ORDER2_PREDICATE_OPEN"
            ),
            "source_bindings": {
                "r130": {"path": "r130.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one lacunary order-two torus C5 "
                "selector predicate."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-consecutive-mode-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s23_lacunary_order2_torus_c5_selector_predicate",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one lacunary order-two torus C5 "
            "selector predicate.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_consecutive_mode_predicate_scope",
            ambiguity_ids,
        )

    def test_torus_base_field_frobenius_routes_to_asymmetric_predicate(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_base_field_frobenius_predicate_dag"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "BASE_FIELD_FROBENIUS_DAG_CLOSED__"
                "ASYMMETRIC_EXTENSION_PREDICATE_OPEN"
            ),
            "source_bindings": {
                "r131": {"path": "r131.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 12,
                "obligation_count": 20,
            },
            "next_action": (
                "Construct one asymmetric Frobenius-aware torus C5 "
                "selector predicate."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-frobenius-dag-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s24_asymmetric_frobenius_torus_c5_selector_predicate",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct one asymmetric Frobenius-aware torus C5 "
            "selector predicate.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_base_field_frobenius_predicate_dag_scope",
            ambiguity_ids,
        )

    def test_torus_sparse_root_bound_routes_to_surviving_selectors(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_sparse_monomial_root_bound"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SMALL_REPRESENTED_SPARSE_PREDICATES_CLOSED__"
                "FIVE_MODE_LOW_SLP_AND_MULTI_DAG_OPEN"
            ),
            "source_bindings": {
                "r132": {"path": "r132.json", "sha256": "a" * 64},
                "kelley": {"path": "kelley.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 21,
            },
            "next_action": (
                "Test one five-mode, low-SLP, or multi-predicate "
                "Frobenius-aware torus C5 selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-sparse-root-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s25_five_mode_low_slp_frobenius_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one five-mode, low-SLP, or multi-predicate "
            "Frobenius-aware torus C5 selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_sparse_monomial_root_bound_scope",
            ambiguity_ids,
        )

    def test_torus_two_atom_progression_routes_to_seven_mode_dag(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_two_atom_geometric_progression"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SINGLE_PREDICATES_THROUGH_SIX_MODES_CLOSED__"
                "SEVEN_MODE_MULTI_DAG_NONZERO_VALUE_OPEN"
            ),
            "source_bindings": {
                "r133": {"path": "r133.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 21,
            },
            "next_action": (
                "Test one seven-mode, multi-predicate, nonzero-value, "
                "or low-SLP torus C5 selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-two-atom-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s26_seven_mode_multi_predicate_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test one seven-mode, multi-predicate, nonzero-value, "
            "or low-SLP torus C5 selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_two_atom_geometric_progression_scope",
            ambiguity_ids,
        )

    def test_torus_khatri_rao_routes_to_composed_selector(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_khatri_rao_kruskal_amplification"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "DETERMINISTIC_SIX_MODE_AND_RANDOM_NEAR_FIVE_LOG_CLOSED__"
                "COMPOSED_NONZERO_LOW_SLP_OPEN"
            ),
            "source_bindings": {
                "r134": {"path": "r134.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 14,
                "obligation_count": 22,
            },
            "next_action": (
                "Test a composed, nonzero-value, or low-SLP torus C5 "
                "selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-khatri-rao-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s27_composed_nonzero_low_slp_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test a composed, nonzero-value, or low-SLP torus C5 "
            "selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_khatri_rao_kruskal_amplification_scope",
            ambiguity_ids,
        )

    def test_torus_path_product_routes_to_growing_support_selector(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_all_nonzero_path_product"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "BOUNDED_PATH_PRODUCT_CLOSED__"
                "GROWING_SUPPORT_LOW_SLP_NONZERO_VALUE_OPEN"
            ),
            "source_bindings": {
                "r135": {"path": "r135.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 13,
                "obligation_count": 21,
            },
            "next_action": (
                "Test a growing-support low-SLP or nonzero-value torus C5 "
                "selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-path-product-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s28_growing_support_low_slp_nonzero_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test a growing-support low-SLP or nonzero-value torus C5 "
            "selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_all_nonzero_path_product_scope",
            ambiguity_ids,
        )

    def test_torus_binomial_depth_routes_to_three_plus_modes(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_binomial_node_union_depth"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "STRUCTURED_BINOMIAL_AND_RANDOM_FOUR_MODE_CLOSED__"
                "STRUCTURED_THREE_PLUS_NONZERO_OPEN"
            ),
            "source_bindings": {
                "r136": {"path": "r136.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 23,
            },
            "next_action": (
                "Test structured three-plus-mode, five-plus-mode low-SLP, "
                "or nonzero-value torus C5 selectors."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-binomial-depth-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s29_three_plus_five_plus_nonzero_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test structured three-plus-mode, five-plus-mode low-SLP, "
            "or nonzero-value torus C5 selectors.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_binomial_node_union_depth_scope",
            ambiguity_ids,
        )

    def test_torus_chebotarev_fiber_cover_routes_to_actual_spark(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_chebotarev_fiber_cover"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "CONDITIONAL_TRINOMIAL_DEPTH__PRIMITIVE_ORDER_TRANSFER_"
                "REJECTED__CHARACTERISTIC_SPECIFIC_SPARK_OPEN"
            ),
            "source_bindings": {
                "r137": {"path": "r137.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 23,
            },
            "next_action": (
                "Prove actual atom-restricted full spark or test a "
                "four-plus-mode or nonzero-value selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-chebotarev-fiber-probe.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s30_characteristic_specific_spark_or_nonzero_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove actual atom-restricted full spark or test a "
            "four-plus-mode or nonzero-value selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_chebotarev_fiber_cover_scope",
            ambiguity_ids,
        )

    def test_torus_order_two_three_minor_routes_to_four_plus(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_order_two_three_minor_rigidity"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "ORDER_TWO_THREE_MINOR_RIGIDITY__TRINOMIALS_CLOSED__"
                "FOUR_PLUS_NONZERO_OPEN"
            ),
            "source_bindings": {
                "r138": {"path": "r138.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 23,
            },
            "next_action": (
                "Test order-two four-plus-mode or nonzero-value torus "
                "C5 selectors."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-order-two-three-minor.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s31_four_plus_or_nonzero_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Test order-two four-plus-mode or nonzero-value torus "
            "C5 selectors.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_order_two_three_minor_rigidity_scope",
            ambiguity_ids,
        )

    def test_torus_order_two_four_minor_claw_routes_to_subcap_claw(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_order_two_four_minor_claw"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "FOUR_MINOR_MOBIUS_CLAW__FULL_SPARK_FALSE__"
                "SUBCAP_CLAW_NONZERO_OPEN"
            ),
            "source_bindings": {
                "r139": {"path": "r139.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 26,
            },
            "next_action": (
                "Build a sub-q^(9/20) known-mode claw or nonzero-value "
                "torus C5 selector."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-order-two-four-minor-claw.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s32_subcap_mobius_claw_or_nonzero_torus_c5_selector",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build a sub-q^(9/20) known-mode claw or nonzero-value "
            "torus C5 selector.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_order_two_four_minor_claw_scope",
            ambiguity_ids,
        )

    def test_torus_sextic_character_routes_to_nonlinear_source_router(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_sextic_mobius_character_router"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "POLYLOG_INVERSE_SEPARATOR__LINEAR_B5__"
                "NONLINEAR_SOURCE_ROUTER_OPEN"
            ),
            "source_bindings": {
                "r140": {"path": "r140.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 26,
            },
            "next_action": (
                "Build a nonlinear nontranslation sextic-character "
                "C2 source router."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-sextic-character-router.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s33_nonlinear_sextic_mobius_source_router",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build a nonlinear nontranslation sextic-character "
            "C2 source router.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_sextic_mobius_character_router_scope",
            ambiguity_ids,
        )

    def test_adaptive_character_tree_routes_to_algebraic_composition(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_adaptive_character_decision_router"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "DECK_PARAMETER_TREE_EXHAUSTED__"
                "ALGEBRAIC_COMPOSITION_OPEN"
            ),
            "source_bindings": {
                "r141": {"path": "r141.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 27,
            },
            "next_action": (
                "Derive a compact algebraic sextic-character composition "
                "across C2 by C3."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-adaptive-character-router.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s34_algebraic_sextic_character_composition",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Derive a compact algebraic sextic-character composition "
            "across C2 by C3.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_adaptive_character_decision_router_scope",
            ambiguity_ids,
        )

    def test_label_congruence_routes_to_transposed_ffe_relation_span(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "torus_c5_label_congruence_correction"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "LABEL_ONLY_COMPOSITION_REJECTED__"
                "EXPLICIT_ROW_BIRTHDAY_AT_RHO__"
                "IMPLICIT_RELATION_SPAN_OPEN"
            ),
            "source_bindings": {
                "r142": {"path": "r142.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 23,
            },
            "next_action": (
                "Build a target-batched transposed summation-polynomial/FFE "
                "relation-span operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "torus-label-congruence.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s35_transposed_ffe_relation_span",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build a target-batched transposed summation-polynomial/FFE "
            "relation-span operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "torus_c5_label_congruence_correction_scope",
            ambiguity_ids,
        )

    def test_weighted_fiber_reduction_routes_to_s13_count_transpose(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_weighted_fiber_marginal_log_operator"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "M6_WEIGHTED_FIBER_LOG_REDUCTION__"
                "CONDITIONAL_N9O20__COUNT_TRANSPOSE_OPEN"
            ),
            "source_bindings": {
                "r143": {"path": "r143.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 19,
                "obligation_count": 29,
            },
            "next_action": (
                "Construct the weighted S7/S13 fiber-count circuit and "
                "reusable transposed marginal index."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-weighted-fiber.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s36_weighted_s13_fiber_count_transpose",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct the weighted S7/S13 fiber-count circuit and "
            "reusable transposed marginal index.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_weighted_fiber_marginal_log_operator_scope",
            ambiguity_ids,
        )

    def test_weighted_gcd_trace_routes_to_implicit_batched_resultant(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_weighted_c3_mobius_gcd_trace"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "WEIGHTED_C6_GCD_TRACE_EXACT__DENSE_BATCH_N11O20__"
                "IMPLICIT_BATCH_OPEN"
            ),
            "source_bindings": {
                "r144": {"path": "r144.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 17,
                "obligation_count": 28,
            },
            "next_action": (
                "Construct an implicit target-batched modular resultant "
                "without dense per-target pullbacks."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-weighted-gcd-trace.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s37_implicit_batched_mobius_resultant",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Construct an implicit target-batched modular resultant "
            "without dense per-target pullbacks.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_weighted_c3_mobius_gcd_trace_scope",
            ambiguity_ids,
        )

    def test_singleton_equivalence_routes_to_source_charged_index(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_aggregate_marginal_singleton_source_equivalence"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SINGLETON_MARGINALS_SOURCE_EQUIVALENT__"
                "BATCHED_INDEX_OPEN"
            ),
            "source_bindings": {
                "r145": {"path": "r145.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 25,
            },
            "next_action": (
                "Build the batched count-and-marginal index while charging "
                "singleton output as source-equivalent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-singleton-equivalence.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s38_source_equivalent_batched_count_marginal_index",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build the batched count-and-marginal index while charging "
            "singleton output as source-equivalent.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_aggregate_marginal_singleton_source_equivalence_scope",
            ambiguity_ids,
        )

    def test_local_valuation_routes_to_shared_transposed_operator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_occurrence_pair_resultant_local_valuation"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "LOCAL_VALUATION_COUNT_EXACT__STANDARD_BATCH_N7O10__"
                "SHARED_TRANSPOSE_OPEN"
            ),
            "source_bindings": {
                "r146": {"path": "r146.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 25,
            },
            "next_action": (
                "Build one shared transposed multi-target "
                "valuation-and-marker operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-local-valuation.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s39_shared_transposed_multi_target_valuation_marker",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one shared transposed multi-target "
            "valuation-and-marker operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_occurrence_pair_resultant_local_valuation_scope",
            ambiguity_ids,
        )

    def test_static_indexing_routes_to_structure_aware_autocorrelation(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_static_3sum_indexing_tradeoff"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "WEIGHTED_STATIC_3SUM_REDUCTION__STANDARD_INDEXES_OVER_"
                "CAP__STRUCTURE_AWARE_AUTOCORRELATION_OPEN"
            ),
            "source_bindings": {
                "r147": {"path": "r147.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 25,
            },
            "next_action": (
                "Build one structure-aware occurrence autocorrelation "
                "directly from the compact C divisor."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-static-indexing.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s40_structure_aware_occurrence_autocorrelation",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one structure-aware occurrence autocorrelation "
            "directly from the compact C divisor.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_static_3sum_indexing_tradeoff_scope",
            ambiguity_ids,
        )

    def test_actual_full_krylov_rank_routes_to_nonlinear_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_actual_c6_shift_krylov_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "ACTUAL_C6_AND_MARGINAL_FULL_SHIFT_RANK__"
                "NONLINEAR_COMPACT_DIVISOR_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r148": {"path": "r148.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 25,
            },
            "next_action": (
                "Build one nonlinear target-specialized compact-divisor "
                "count-and-marginal circuit."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-actual-krylov.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s41_nonlinear_target_specialized_compact_divisor_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one nonlinear target-specialized compact-divisor "
            "count-and-marginal circuit.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_actual_c6_shift_krylov_rank_scope",
            ambiguity_ids,
        )

    def test_rational_subalgebra_rigidity_routes_to_finite_depth_circuit(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_rational_convolution_subalgebra_rigidity"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "RATIONAL_CONVOLUTION_SUBALGEBRA_B17_OVER_4__"
                "FINITE_DEPTH_U6_CIRCUIT_OPEN"
            ),
            "source_bindings": {
                "r149": {"path": "r149.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 14,
                "obligation_count": 24,
            },
            "next_action": (
                "Build one finite-depth nonhomomorphic U6 marker circuit."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-rational-subalgebra.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s42_finite_depth_nonhomomorphic_u6_marker_circuit",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one finite-depth nonhomomorphic U6 marker circuit.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_rational_convolution_subalgebra_rigidity_scope",
            ambiguity_ids,
        )

    def test_matrix_free_jacobian_routes_to_bidirectional_operator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_matrix_free_marginal_jacobian_krylov"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "MARGINAL_JACOBIAN_MATRIX_FREE_B2_CONDITIONAL__"
                "BIDIRECTIONAL_OPERATOR_OPEN"
            ),
            "source_bindings": {
                "r150": {"path": "r150.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 25,
            },
            "next_action": (
                "Build one reusable weight-parametric bidirectional "
                "marker operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-matrix-free-jacobian.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s43_weight_parametric_bidirectional_marker_operator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one reusable weight-parametric bidirectional "
            "marker operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_matrix_free_marginal_jacobian_krylov_scope",
            ambiguity_ids,
        )

    def test_geometry_only_leaf_state_routes_to_signed_ffe_dag(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_geometry_only_weight_interpolation_adjoint"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "GEOMETRY_ONLY_LEAF_TANGENT_ADJOINT_B3O4__"
                "SIGNED_INTERNAL_FFE_DAG_OPEN"
            ),
            "source_bindings": {
                "r151": {"path": "r151.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 25,
            },
            "next_action": (
                "Build one signed weight-separable internal FFE "
                "elimination DAG."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-geometry-only-leaves.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s44_signed_weight_separable_ffe_elimination_dag",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one signed weight-separable internal FFE "
            "elimination DAG.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_geometry_only_weight_interpolation_adjoint_scope",
            ambiguity_ids,
        )

    def test_symmetric_reverse_only_route_requires_density_transfer(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_symmetric_shift_reverse_only_marginal"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "REVERSE_ONLY_SYMMETRY_EXACT__FINITE_RANK_DEFICIT"
            ),
            "source_bindings": {
                "r152": {"path": "r152.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 27,
            },
            "next_action": (
                "Build one signed reverse marker operator with "
                "nonadaptive multiscale density and rank transfer."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-symmetric-reverse-only.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s45_reverse_only_signed_marker_density_transfer",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build one signed reverse marker operator with "
            "nonadaptive multiscale density and rank transfer.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_symmetric_shift_reverse_only_marginal_scope",
            ambiguity_ids,
        )

    def test_signed_quotient_route_requires_rank_theorem_and_reverse_ffe(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_signed_quotient_multiscale_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SIGNED_QUOTIENT_EXACT__FINITE_RANK_TRANSITION"
            ),
            "source_bindings": {
                "r153": {"path": "r153.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 26,
            },
            "next_action": (
                "Prove random-rank and hash-to-curve transfer, then build "
                "the reverse signed FFE operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-signed-quotient-rank.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s46_signed_quotient_random_rank_reverse_ffe_transfer",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove random-rank and hash-to-curve transfer, then build "
            "the reverse signed FFE operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_signed_quotient_multiscale_rank_scope",
            ambiguity_ids,
        )

    def test_singleton_hypergraph_route_requires_contiguity_and_ffe(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_singleton_relation_hypergraph_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SINGLETON_ROWS_EXACT__SPARSE_RANK_TRANSFER_GAP"
            ),
            "source_bindings": {
                "r154": {"path": "r154.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 17,
                "obligation_count": 28,
            },
            "next_action": (
                "Prove convolution-Tanner contiguity under logarithmic "
                "oversampling, then build the reverse signed FFE operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-singleton-hypergraph.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s47_convolution_tanner_contiguity_reverse_ffe",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove convolution-Tanner contiguity under logarithmic "
            "oversampling, then build the reverse signed FFE operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_singleton_relation_hypergraph_rank_scope",
            ambiguity_ids,
        )

    def test_a_diversity_route_requires_direct_rank_and_reverse_ffe(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_a_diversity_projective_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "PROJECTIVE_ROWS_EXACT__FINITE_LOG_RANK_TRANSITION"
            ),
            "source_bindings": {
                "r155": {"path": "r155.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 26,
            },
            "next_action": (
                "Prove the direct asymptotic projective singleton rank "
                "bound, then build the reverse signed FFE operator."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-a-diversity-rank.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s48_projective_singleton_direct_rank_reverse_ffe",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove the direct asymptotic projective singleton rank "
            "bound, then build the reverse signed FFE operator.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_a_diversity_projective_rank_scope",
            ambiguity_ids,
        )

    def test_public_group_factor_logs_route_to_rank_ffe_and_descent(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_hash_to_curve_projective_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "PUBLIC_GROUP_ROWS__FINITE_FACTOR_LOG_TRANSFER"
            ),
            "source_bindings": {
                "r156": {"path": "r156.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 24,
            },
            "next_action": (
                "Prove asymptotic public short-relation rank, replace "
                "explicit C6 enumeration by reverse FFE, and perform "
                "identical target descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-public-group-rank.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s49_public_short_relation_rank_reverse_ffe_descent",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove asymptotic public short-relation rank, replace "
            "explicit C6 enumeration by reverse FFE, and perform "
            "identical target descent.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_hash_to_curve_projective_rank_scope",
            ambiguity_ids,
        )

    def test_near_injectivity_supply_routes_to_conditioned_rank_ffe_and_descent(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_short_relation_near_injectivity_supply"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "NEAR_INJECTIVITY__PAIRWISE_RELATION_SUPPLY"
            ),
            "source_bindings": {
                "r157": {"path": "r157.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 26,
            },
            "next_action": (
                "Prove full rank for the conditioned sampler, then build "
                "the reverse signed FFE operator and identical descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-near-injectivity-supply.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s50_conditioned_short_relation_full_rank_reverse_ffe_descent",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Prove full rank for the conditioned sampler, then build "
            "the reverse signed FFE operator and identical descent.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_short_relation_near_injectivity_supply_scope",
            ambiguity_ids,
        )

    def test_random_diagonal_rank_routes_to_batched_positive_c6_locator(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_random_diagonal_known_target_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "DIRECT_TARGET_COVERAGE__RANDOM_DIAGONAL_FULL_RANK"
            ),
            "source_bindings": {
                "r158": {"path": "r158.json", "sha256": "a" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 20,
                "obligation_count": 29,
            },
            "next_action": (
                "Build the batched positive-C6 reverse FFE source locator "
                "and replay logs and identical descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-random-diagonal-rank.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={
                    lane: (
                        probe,
                        probe_path,
                    )
                },
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s51_batched_positive_c6_reverse_ffe_source_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build the batched positive-C6 reverse FFE source locator "
            "and replay logs and identical descent.",
        )
        self.assertEqual(
            report["summary"]["frontier_closed_lanes"],
            [lane],
        )
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_random_diagonal_known_target_rank_scope",
            ambiguity_ids,
        )

    def test_generic_locator_reduction_routes_to_coordinate_specific_s7(
        self,
    ) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_positive_c6_generic_locator_reduction"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "R159_TO_GENERIC_DLP_Q9O20__COORDINATE_ESCAPE_OPEN"
            ),
            "source_bindings": {
                "r159": {"path": "r159.json", "sha256": "a" * 64},
                "shoup": {"path": "shoup.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 17,
                "obligation_count": 24,
            },
            "next_action": (
                "Build a coordinate-specific S7/resultant/FFE source "
                "locator and replay logs and identical descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-generic-locator-reduction.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s52_coordinate_specific_s7_reverse_ffe_source_locator",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build a coordinate-specific S7/resultant/FFE source "
            "locator and replay logs and identical descent.",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_positive_c6_generic_locator_reduction_scope",
            ambiguity_ids,
        )

    def test_signed_divisor_gcd_routes_to_many_target_composition(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_signed_c3_divisor_translation_gcd"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SIGNED_C3_DIVISOR_GCD_EXACT__INDEPENDENT_BATCH_B7O2__"
                "MANY_TARGET_COMPOSITION_OPEN"
            ),
            "source_bindings": {
                "r160": {"path": "r160.json", "sha256": "a" * 64},
                "semaev": {"path": "semaev.ps", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 18,
                "obligation_count": 26,
            },
            "next_action": (
                "Build a target-batched many-inner modular-composition and "
                "gcd source adjoint, then replay logs and identical descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-signed-divisor-gcd.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s53_target_batched_signed_divisor_modular_composition_gcd",
        )
        self.assertEqual(
            report["next_action"]["action"],
            "Build a target-batched many-inner modular-composition and "
            "gcd source adjoint, then replay logs and identical descent.",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_signed_c3_divisor_translation_gcd_scope",
            ambiguity_ids,
        )

    def test_batch_inverse_fit_routes_to_global_below_rho_aggregate(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_batch_inverse_transpose_modcomp_fit"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": "BATCH_INVERSE_EXACT__NONLINEAR_AGGREGATE_OPEN",
            "source_bindings": {
                "r161": {"path": "r161.json", "sha256": "a" * 64},
                "modcomp": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 18,
                "obligation_count": 27,
            },
            "next_action": (
                "Build an aggregate nonlinear signed-divisor operator below "
                "B^(5/2) and replay logs and descent."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-batch-inverse-fit.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s54_aggregate_nonlinear_signed_divisor_below_rho",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_batch_inverse_transpose_modcomp_fit_scope",
            ambiguity_ids,
        )

    def test_aggregate_union_reduction_routes_to_unlabeled_constructor(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_aggregate_union_factor_label_recovery"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": "UNION_SEMANTICS_EXACT__CONSTRUCTOR_OPEN",
            "source_bindings": {
                "r162": {"path": "r162.json", "sha256": "a" * 64},
                "r88": {"path": "r88.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 20,
                "obligation_count": 27,
            },
            "next_action": (
                "Build the unlabeled aggregate union factor below B^(5/2)."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-aggregate-union.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s55_unlabeled_aggregate_union_factor_constructor",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_aggregate_union_factor_label_recovery_scope",
            ambiguity_ids,
        )

    def test_randomized_norm_routes_to_elliptic_translation_constructor(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_randomized_target_divisor_norm_union"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "RANDOM_LINEAR_NORM_EXACT_AFTER_VERIFICATION__"
                "ELLIPTIC_TRANSLATION_CONSTRUCTOR_OPEN"
            ),
            "source_bindings": {
                "r163": {"path": "r163.json", "sha256": "a" * 64},
                "dahan": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 20,
                "obligation_count": 27,
            },
            "next_action": (
                "Build the regular elliptic-translation target norm below "
                "B^(5/2) without the n-by-N residual table."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-randomized-target-norm.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s56_output_sensitive_elliptic_translation_target_norm",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_randomized_target_divisor_norm_union_scope",
            ambiguity_ids,
        )

    def test_global_randomizer_routes_to_single_function_translate_product(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_global_randomizer_elliptic_translate_product"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "ONE_GLOBAL_RANDOMIZER__FIXED_FUNCTION_TRANSLATE_PRODUCT_OPEN"
            ),
            "source_bindings": {
                "r164": {"path": "r164.json", "sha256": "a" * 64},
                "miller": {"path": "miller.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 21,
                "obligation_count": 28,
            },
            "next_action": (
                "Build the arbitrary-target translate product of one fixed "
                "elliptic function modulo U below B^(5/2)."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-global-randomizer-translate-product.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s57_single_function_arbitrary_translate_product_remainder",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_global_randomizer_elliptic_translate_product_scope",
            ambiguity_ids,
        )

    def test_kummer_x_routes_to_output_sensitive_translate_remainder(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_kummer_x_translate_signed_verification"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "KUMMER_X_RELAXATION__SIGNED_VERIFICATION__REMAINDER_OPEN"
            ),
            "source_bindings": {
                "r165": {"path": "r165.json", "sha256": "a" * 64},
                "r161": {"path": "r161.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 31,
            },
            "next_action": (
                "Build the arbitrary-target Kummer translate product modulo "
                "U below B^(5/2) and prove deterministic hash transfer."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-kummer-x-translate.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s58_output_sensitive_kummer_translate_product_remainder",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_kummer_x_translate_signed_verification_scope",
            ambiguity_ids,
        )

    def test_target_divisor_swap_routes_to_slp_elliptic_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_generalized_target_divisor_weil_reciprocity_swap"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "TARGET_DIVISOR_WITNESS__WEIL_SWAP__SLP_RESULTANT_OPEN"
            ),
            "source_bindings": {
                "r166": {"path": "r166.json", "sha256": "a" * 64},
                "eagen": {"path": "eagen.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 32,
            },
            "next_action": (
                "Build an SLP elliptic-resultant remainder modulo U below "
                "B^(5/2) without nN, n^2, or represented degree-nN state."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-target-divisor-weil-swap.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s59_slp_elliptic_resultant_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_generalized_target_divisor_weil_reciprocity_swap_scope",
            ambiguity_ids,
        )

    def test_log_derivative_routes_to_denominator_aware_trace(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_log_derivative_elliptic_cauchy_trace"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "LOG_DERIVATIVE__CANDIDATE_POLES__DENOMINATOR_TRACE_OPEN"
            ),
            "source_bindings": {
                "r167": {"path": "r167.json", "sha256": "a" * 64},
                "eagen": {"path": "eagen.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 32,
            },
            "next_action": (
                "Build a denominator-aware transposed elliptic Cauchy trace "
                "modulo U below B^(5/2) without candidate inversion."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-log-derivative-trace.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s60_denominator_aware_elliptic_cauchy_trace_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_log_derivative_elliptic_cauchy_trace_scope",
            ambiguity_ids,
        )

    def test_regularized_displacement_routes_to_fraction_free_fitting(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_regularized_log_trace_displacement_rank"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SCALAR_RESOLVENT__DIAGONAL_DISPLACEMENT_FULL_RANK__"
                "FRACTION_FREE_FITTING_OPEN"
            ),
            "source_bindings": {
                "r168": {"path": "r168.json", "sha256": "a" * 64},
                "bostan": {"path": "bostan.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 21,
                "obligation_count": 29,
            },
            "next_action": (
                "Build a fraction-free elliptic Fitting/subresultant modulo U "
                "below B^(5/2) without generic lambda interpolation."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-regularized-displacement.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s61_fraction_free_elliptic_fitting_subresultant_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_regularized_log_trace_displacement_rank_scope",
            ambiguity_ids,
        )

    def test_lambda_zero_dedup_routes_to_slp_streaming_norm(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_lambda_zero_fitting_target_norm_dedup"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": "LAMBDA_ZERO_NORM_EQUIVALENT__SLP_STREAMING_OPEN",
            "source_bindings": {
                "r169": {"path": "r169.json", "sha256": "a" * 64},
                "r167": {"path": "r167.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 24,
            },
            "next_action": (
                "Build an SLP-streaming output-sensitive target norm modulo U "
                "below B^(5/2) without nN or n^2 state."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-lambda-zero-dedup.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s62_slp_streaming_output_sensitive_target_norm_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_lambda_zero_fitting_target_norm_dedup_scope",
            ambiguity_ids,
        )

    def test_balanced_miller_streaming_routes_to_nonlocal_leaf_batch(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_balanced_miller_tree_norm_streaming"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": "BALANCED_MILLER_NN_CLOSED__NONLOCAL_BATCH_OPEN",
            "source_bindings": {
                "r170": {"path": "r170.json", "sha256": "a" * 64},
                "miller": {"path": "miller.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 22,
                "obligation_count": 29,
            },
            "next_action": (
                "Build a nonlocal batched elliptic leaf-translate product "
                "below B^(5/2) without the n-by-N pair grid."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-balanced-miller-streaming.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s63_nonlocal_batched_elliptic_leaf_translate_product",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_balanced_miller_tree_norm_streaming_scope",
            ambiguity_ids,
        )

    def test_target_sign_s3_routes_to_factored_self_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_target_sign_conjugate_s3_self_resultant"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": "TARGET_SIGN_S3_EXACT__FACTORED_SELF_RESULTANT_OPEN",
            "source_bindings": {
                "r171": {"path": "r171.json", "sha256": "a" * 64},
                "semaev": {"path": "semaev.ps", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 31,
            },
            "next_action": (
                "Build a factored self-S3 resultant modulo U below B^(5/2) "
                "without the N^2 reverse body or nN point/factor grid."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-target-sign-s3.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s64_factored_self_s3_resultant_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_target_sign_conjugate_s3_self_resultant_scope",
            ambiguity_ids,
        )

    def test_s3_determinantal_transfer_routes_to_commutative_pushforward(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_s3_determinantal_transfer_noncommutativity"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "S3_DETERMINANT_EXACT__NAIVE_2X2_TRANSFER_CLOSED__"
                "COMMUTATIVE_PUSHFORWARD_OPEN"
            ),
            "source_bindings": {
                "r172": {"path": "r172.json", "sha256": "a" * 64},
                "semaev": {"path": "semaev.ps", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 14,
                "obligation_count": 23,
            },
            "next_action": (
                "Use 4*V(X)*V_T(u) for a commutative divisor pushforward "
                "modulo arbitrary squarefree U below B^(5/2)."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-s3-determinantal-transfer.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s65_commutative_target_sign_divisor_pushforward_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_s3_determinantal_transfer_noncommutativity_scope",
            ambiguity_ids,
        )

    def test_confluent_dual_chow_routes_to_fused_outer_norm(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_confluent_signed_dual_chow_pushforward"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "CONFLUENT_SIGNED_DUAL_CHOW_EXACT__"
                "FACTORED_OUTER_NORM_OPEN"
            ),
            "source_bindings": {
                "r173": {"path": "r173.json", "sha256": "a" * 64},
                "multipoint": {"path": "mpe.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 23,
            },
            "next_action": (
                "Fuse the factored target dual-Chow form with the tangent-aware "
                "outer norm modulo U below B^(5/2)."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-confluent-dual-chow.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_confluent_signed_dual_chow_pushforward_scope",
            ambiguity_ids,
        )

    def test_scalar_subset_tree_routes_to_reusable_incidence_oracle(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_scalar_subset_incidence_group_testing"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "SCALAR_SUBSET_GROUP_TESTING_EXACT__"
                "REUSABLE_INCIDENCE_ORACLE_OPEN"
            ),
            "source_bindings": {
                "r174": {"path": "r174.json", "sha256": "a" * 64},
                "shoup": {"path": "shoup.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 22,
            },
            "next_action": (
                "Build a reusable tangent-aware scalar subset-incidence oracle "
                "with softly O(n+N) preprocessing and O(|S|+N) query work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-scalar-subset-incidence.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s67_reusable_scalar_subset_incidence_oracle",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_scalar_subset_incidence_group_testing_scope",
            ambiguity_ids,
        )

    def test_principal_pontryagin_routes_to_factored_trilinear_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_principal_target_pontryagin_resultant"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "PRINCIPAL_PONTRYAGIN_RECIPROCITY_EXACT__"
                "FACTORED_TRILINEAR_RESULTANT_OPEN"
            ),
            "source_bindings": {
                "r175": {"path": "r175.json", "sha256": "a" * 64},
                "weil": {"path": "weil.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 16,
                "obligation_count": 23,
            },
            "next_action": (
                "Build a factored trilinear elliptic resultant on compact "
                "U_A,V_A, U_D,V_D, and h inputs without mn pair sums."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-principal-pontryagin.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s68_factored_trilinear_elliptic_resultant",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn(
            "m6_principal_target_pontryagin_resultant_scope",
            ambiguity_ids,
        )

    def test_global_marked_locator_routes_to_output_sensitive_fitting(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_global_marked_fitting_locator"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "GLOBAL_MARKED_LOCATOR_EXACT__"
                "OUTPUT_SENSITIVE_MARKED_FITTING_OPEN"
            ),
            "source_bindings": {
                "r176": {"path": "r176.json", "sha256": "a" * 64},
                "shoup": {"path": "shoup.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 22,
            },
            "next_action": (
                "Build a fraction-free output-sensitive marked Fitting operator "
                "that emits det(AI-X_1|ker K) from compact U,V,h."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-global-marked-fitting.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s69_output_sensitive_marked_fitting_locator",
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn("m6_global_marked_fitting_locator_scope", ambiguity_ids)

    def test_marked_fitting_dedup_returns_to_signed_outer_norm(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_marked_fitting_signed_norm_dedup"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "MARKED_FITTING_NOT_DISTINCT_ALGORITHMIC_LANE__"
                "UNIFIED_NONLOCAL_SIGNED_TRANSLATE_PRODUCT_OPEN"
            ),
            "source_bindings": {
                "r177": {"path": "r177.json", "sha256": "a" * 64},
                "r174": {"path": "r174.json", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 15,
                "obligation_count": 22,
            },
            "next_action": (
                "Build the unified nonlocal signed elliptic translate product "
                "modulo U in softly O(n+N) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-marked-fitting-dedup.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 318)
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"][
                "ambiguity_resolutions"
            ]
        }
        self.assertIn("m6_marked_fitting_signed_norm_dedup_scope", ambiguity_ids)

    def test_squarefree_truncated_resultant_routes_factored_dynamic_evaluation(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_squarefree_truncated_resultant_applicability"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "DIRECT_TRUNCATED_RESULTANT_ROUTE_CLOSED__"
                "FACTORED_SQUAREFREE_DYNAMIC_EVALUATION_OPEN"
            ),
            "source_bindings": {
                "r178": {"path": "r178.json", "sha256": "a" * 64},
                "moroz_schost": {"path": "paper.pdf", "sha256": "b" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 11,
                "obligation_count": 20,
            },
            "next_action": (
                "Build a factored arbitrary-squarefree dynamic-evaluation "
                "operator in softly O(n+N) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-squarefree-truncated-resultant.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 320)
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertIn(
            "m6_squarefree_truncated_resultant_applicability_scope",
            ambiguity_ids,
        )

    def test_d5_survivor_routes_one_shot_monogenic_compiler(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_d5_directed_evaluation_survivor"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "LITERAL_FACTOR_STREAM_AND_DIRECTED_EVALUATION_B7O2__"
                "ONE_SHOT_MONOGENIC_FINITE_FIELD_FOLD_OPEN"
            ),
            "source_bindings": {
                "r179": {"path": "r179.json", "sha256": "a" * 64},
                "d5": {"path": "d5.pdf", "sha256": "b" * 64},
                "directed": {"path": "directed.pdf", "sha256": "c" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 18,
                "obligation_count": 28,
            },
            "next_action": (
                "Build a one-shot monogenic finite-field modular-composition "
                "compiler in softly O(n+N) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-d5-directed-survivor.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 322)
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertIn("m6_d5_directed_evaluation_survivor_scope", ambiguity_ids)

    def test_monogenic_kernel_routes_gcd_equivalent_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_monogenic_kernel_bidegree"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "EXACT_BOUNDED_SOURCE_DEGREE_FOLD_CLOSED__"
                "GCD_EQUIVALENT_OUTPUT_SENSITIVE_RESULTANT_OPEN"
            ),
            "source_bindings": {
                "r180": {"path": "r180.json", "sha256": "a" * 64},
                "triangular": {"path": "triangular.pdf", "sha256": "b" * 64},
                "modcomp": {"path": "modcomp.pdf", "sha256": "c" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 21,
                "obligation_count": 33,
            },
            "next_action": (
                "Build a gcd-equivalent output-sensitive elliptic composed "
                "resultant modulo U in softly O(n+N) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-monogenic-kernel.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 324)
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertIn("m6_monogenic_kernel_bidegree_scope", ambiguity_ids)

    def test_target_subresultant_routes_slp_direct_resultant(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_gcd_equivalent_target_subresultant"
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": (
                "TARGET_SUBRESULTANT_IDENTITY_ADMITTED__"
                "SLP_DIRECT_OUTPUT_SENSITIVE_RESULTANT_MOD_U_OPEN"
            ),
            "source_bindings": {
                "r181": {"path": "r181.json", "sha256": "a" * 64},
                "triangular": {"path": "triangular.pdf", "sha256": "b" * 64},
                "resultant": {"path": "resultant.pdf", "sha256": "c" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 24,
                "obligation_count": 35,
            },
            "next_action": (
                "Build an SLP-direct determinant-zero or resultant-mod-U "
                "algorithm emitting only G_1 in softly O(n+N) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-target-subresultant.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 326)
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertIn(
            "m6_gcd_equivalent_target_subresultant_scope", ambiguity_ids
        )

    def test_sparse_projector_routes_slp_direct_nested_moments(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(4, logs=True),
                "mixed_balanced_stride_mask1": config(4, logs=True),
            },
            descents=[successful_descent()],
        )
        lane = "m6_sparse_projector_prony_locator"
        classification = (
            "ADMIT_NESTED_PROJECTOR_AND_WEIGHTED_PRONY__"
            "SLP_DIRECT_NESTED_MOMENT_CONSTRUCTOR_OPEN"
        )
        probe = {
            "schema": FOCUS.FRONTIER_PREFLIGHT_SCHEMAS[lane],
            "classification": classification,
            "source_bindings": {
                "r182": {"path": "r182.json", "sha256": "a" * 64},
                "wiedemann": {"path": "wiedemann.pdf", "sha256": "b" * 64},
                "shoup": {"path": "shoup.pdf", "sha256": "c" * 64},
            },
            "admission": {
                "lane_admitted": False,
                "passed_obligation_count": 19,
                "obligation_count": 28,
            },
            "next_action": (
                "Build an SLP-direct nested projector-moment operator emitting "
                "2c scalars in softly O(n+N+c) work."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            probe_path = root / "m6-sparse-projector-prony.json"
            probe_path.write_text(
                json.dumps(probe, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = FOCUS.build_report(
                value,
                source,
                frontier_preflights={lane: (probe, probe_path)},
            )

        self.assertEqual(
            report["next_action"]["focus_id"],
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
        )
        selected = [
            row
            for row in report["focus_queue"]
            if row["id"] == "s66_fused_factored_dual_chow_outer_norm_mod_u"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["priority_score"], 327)
        self.assertTrue(
            any(classification in item for item in selected[0]["evidence"])
        )
        self.assertEqual(report["summary"]["frontier_closed_lanes"], [lane])
        ambiguity_ids = {
            item["id"]
            for item in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertIn("m6_sparse_projector_prony_locator_scope", ambiguity_ids)

    def test_exposes_stored_but_unusable_and_routing_headroom(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(1),
                "mixed_balanced_stride_mask1": config(3),
                "consecutive_canonical_mask1": config(4, logs=True, cost_ratio=0.8),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        summary = report["summary"]
        self.assertEqual(summary["natural_stored_but_unusable_full_cell_count"], 1)
        self.assertEqual(summary["oracle_rank_headroom_cell_count"], 1)
        self.assertEqual(summary["fixed_rank_headroom_cell_count"], 1)
        self.assertFalse(summary["promotion_allowed"])
        cell = report["cell_reports"][0]
        self.assertEqual(cell["natural_route"]["bottleneck"], "relation_rank")
        matrix = cell["natural_route"]["major_result_replication"]
        self.assertEqual(matrix[0]["stage"], "replay_and_exact_residual")
        self.assertEqual(matrix[0]["status"], "passed")
        self.assertAlmostEqual(
            cell["routing_gap"]["fixed_fraction_of_oracle_rank_headroom"],
            2.0 / 3.0,
        )
        self.assertEqual(
            cell["routing_gap"]["fixed_delta"]["first_positive_stage"],
            "relation_rank",
        )
        self.assertFalse(cell["routing_gap"]["causal_self_patch_equivalent"])
        focus_ids = {row["id"] for row in report["focus_queue"]}
        self.assertIn("collision_to_rank_routing_ablation", focus_ids)
        self.assertIn("routing_intervention_generalization", focus_ids)
        focus = report["autoresearch_steering"]
        self.assertEqual(focus["critical_experiment_budget"], 3)
        self.assertGreaterEqual(focus["candidate_count"], len(report["focus_queue"]))
        self.assertIn("hypothesis", report["focus_queue"][0]["experiment"])
        self.assertIn("falsifier", report["focus_queue"][0]["experiment"])
        lineage = focus["experiment_lineage"]
        self.assertEqual(lineage["lineage_mode"], "logical_plan_only")
        self.assertEqual(lineage["queued_count"], len(report["focus_queue"]))
        root = lineage["nodes"][0]
        self.assertEqual(root["node_type"], "immutable_source_baseline")
        self.assertEqual(root["source_sha256"], report["source"]["sha256"])
        self.assertTrue(
            all(node["parent_id"] == root["id"] for node in lineage["nodes"][1:])
        )
        self.assertTrue(
            all(not node["run_materialized"] for node in lineage["nodes"])
        )
        uniform = cell["synthetic_uniform_occupancy_control"]
        self.assertTrue(uniform["available"])
        self.assertEqual(uniform["accepted_residual_events"], 100)
        self.assertEqual(uniform["scope"], "occupancy_only_no_relation_or_log_labels")

    def test_summation_ffe_markers_are_detected_in_stage_record(self) -> None:
        stage = config(4, logs=True, cost_ratio=0.5)
        stage["summation_poly_degree"] = 8
        stage["ffe_profile"] = "experimental"
        stage["pipeline_note"] = "Summation polynomial and FFE probe."
        record = FOCUS.stage_record(stage)
        self.assertTrue(record["has_summation_ffe_evidence"])
        self.assertEqual(record["summation_ffe_evidence_count"], 3)
        self.assertIn("summation_poly_degree", record["summation_ffe_markers"])
        self.assertIn("ffe_profile", record["summation_ffe_markers"])
        self.assertIn("pipeline_note::Summation polynomial and FFE probe.", record["summation_ffe_markers"])
        readiness = record["summation_ffe_readiness"]
        self.assertTrue(readiness["has_markers"])
        self.assertTrue(readiness["needs_inputs"])
        self.assertEqual(readiness["replay_readiness"], "missing_exact_inputs")
        self.assertEqual(readiness["replay_readiness_class"], "marker_only")
        self.assertIn("summation", readiness["needs"])
        self.assertIn("ffe", readiness["needs"])
        self.assertIn("exact_summation_source_payload", readiness["required_inputs"])
        self.assertIn("exact_ffe_source_payload", readiness["required_inputs"])
        self.assertEqual(readiness["present_inputs"], [])
        gate = record["summation_ffe_new_factor_row_discovery_gate"]
        self.assertFalse(gate["lane_admitted"])
        self.assertEqual(gate["status"], "missing_discovery_contract")
        self.assertEqual(gate["product_quotient_information_credit"], 0)

    def test_uniform_occupancy_control_is_deterministic(self) -> None:
        source = config(1)
        first = FOCUS.synthetic_uniform_occupancy(
            source, order=1009, shift_count=48, label="fixed"
        )
        second = FOCUS.synthetic_uniform_occupancy(
            source, order=1009, shift_count=48, label="fixed"
        )
        self.assertEqual(first, second)
        self.assertGreater(first["collision_edge_count"], 0)

    def test_summation_ffe_readiness_enters_focus_queue_and_summary(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.05)
        natural["summation_polynomial_order"] = 4
        natural["ffe_route_id"] = "summation-branch-1"
        value = payload(
            {"random_hash_mask1": natural},
            descents=[
                {"recovered": True, "invalid_candidate_count": 0},
            ],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={
                    "audit_passed": True,
                    "source_sha256": FOCUS.sha256_file(source),
                },
            )
        self.assertIn(
            "summation_ffe_new_factor_row_cost_gate",
            {row["id"] for row in report["focus_queue"]},
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_evidence_cell_count"],
            1,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_evidence_marker_count"],
            2,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_evidence_ready_cell_count"],
            0,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_evidence_missing_cell_count"],
            1,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_evidence_marker_only_count"],
            1,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_lane_admitted_cell_count"],
            0,
        )
        self.assertEqual(
            report["summary"]["natural_summation_ffe_lane_blocked_cell_count"],
            1,
        )
        self.assertFalse(report["summary"]["promotion_allowed"])

    def test_requires_a_source_bound_independent_audit_for_promotion(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.05)
        value = payload(
            {"random_hash_mask1": natural},
            descents=[
                successful_descent(),
                successful_descent(),
            ],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            without_audit = FOCUS.build_report(value, source)
            self.assertFalse(without_audit["summary"]["promotion_allowed"])

            audit = {
                "audit_passed": True,
                "source_sha256": FOCUS.sha256_file(source),
            }
            with_audit = FOCUS.build_report(value, source, audit=audit)

        self.assertTrue(with_audit["summary"]["natural_pipeline_complete"])
        self.assertTrue(with_audit["independent_audit"]["promotion_binding_valid"])
        self.assertTrue(with_audit["summary"]["promotion_allowed"])
        self.assertEqual(with_audit["claim_status"], "AUDIT_BOUND_PROMOTION_CANDIDATE")

    def test_failed_audit_hash_cannot_bind(self) -> None:
        value = payload({"random_hash_mask1": config(4, logs=True, cost_ratio=0.5)}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={"audit_passed": True, "source_sha256": "0" * 64},
            )

        self.assertFalse(report["independent_audit"]["source_hash_matches"])
        self.assertFalse(report["summary"]["promotion_allowed"])

    def test_missing_target_descent_is_untested_and_blocks_promotion(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={"audit_passed": True, "source_sha256": FOCUS.sha256_file(source)},
            )

        self.assertFalse(report["summary"]["natural_pipeline_complete"])
        self.assertFalse(report["summary"]["promotion_allowed"])
        self.assertEqual(
            report["cell_reports"][0]["natural_route"]["bottleneck"],
            "target_descent_untested",
        )
        self.assertIn(
            "factor_logs_to_target_descent_probe",
            {row["id"] for row in report["focus_queue"]},
        )
        resolutions = {
            row["id"]: row for row in report["autoresearch_steering"]["ambiguity_resolutions"]
        }
        self.assertTrue(resolutions["target_descent_absence"]["blocks_promotion"])
        self.assertEqual(
            resolutions["target_descent_absence"]["uncertainty_class"],
            "promotion_blocking",
        )
        self.assertTrue(resolutions["target_descent_absence"]["operator_interrupt_required"])
        self.assertEqual(
            resolutions["target_descent_absence"]["resolution"],
            "Treat absent descent as untested, never as success.",
        )

    def test_descent_failure_count_is_linear_in_full_cells(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        tested_descent = {"recovered": True, "invalid_candidate_count": 0}
        value = payload(
            {"random_hash_mask1": natural},
            descents=[tested_descent],
            breakthrough=True,
        )
        value["curve_records"].append(
            {
                "split": "dev",
                "bits": 24,
                "seed": 1432402,
                "order": 1013,
                "policies": {
                    "two_map_union": {
                        "full": {
                            "factor_base_size_B": 12,
                            "configurations": {
                                "random_hash_mask1": config(1),
                                "mixed_balanced_stride_mask1": config(1),
                            },
                            "target_descents": [],
                        }
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={
                    "audit_passed": True,
                    "source_sha256": FOCUS.sha256_file(source),
                },
                focus_budget=10,
            )

        descent_queue = [
            row
            for row in report["focus_queue"]
            if row["id"] == "factor_logs_to_target_descent_probe"
        ]
        self.assertEqual(len(descent_queue), 1)
        self.assertEqual(descent_queue[0]["evidence"][0], "1 full cells omit or fail the log-to-descent transition.")

    def test_focus_budget_defers_lower_priority_experiments(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(1, logs=False, cost_ratio=2.0),
                "mixed_balanced_stride_mask1": config(3, logs=False, cost_ratio=1.5),
                "consecutive_canonical_mask1": config(4, logs=True, cost_ratio=0.8),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source, focus_budget=1)

        steering = report["autoresearch_steering"]
        self.assertEqual(len(report["focus_queue"]), 1)
        self.assertGreater(steering["focus_accounting"]["deferred_count"], 0)
        self.assertTrue(steering["deferred_experiments"])
        self.assertEqual(
            steering["deferred_experiments"][0]["reason"],
            "outside_current_critical_experiment_budget",
        )

    def test_methodology_binds_the_source_post(self) -> None:
        self.assertEqual(
            FOCUS.SCHEMA,
            "ecdlp.p1436_autoresearch_focus_report.v119",
        )
        self.assertEqual(
            FOCUS.METHODOLOGY["source_post_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )
        self.assertEqual(
            FOCUS.METHODOLOGY["source_post_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )
        self.assertTrue(FOCUS.METHODOLOGY["paper_headroom_range_is_not_an_ecdlp_gate"])
        self.assertEqual(
            FOCUS.METHODOLOGY["openresearch_cli_url"],
            "https://github.com/alphaXiv/openresearch-cli",
        )
        guidance = FOCUS.METHODOLOGY["tweet_guidance"]
        self.assertTrue(guidance["bounded_critical_set"])
        self.assertTrue(guidance["non_blocking_ambiguity_is_deterministic"])
        self.assertTrue(guidance["peripheral_scope_defer"])
        self.assertIn("bounded_critical_set_note", guidance)
        self.assertIn("non_blocking_ambiguity_note", guidance)
        self.assertIn("peripheral_scope_defer_note", guidance)
        self.assertEqual(guidance["source_post_id"], "2076737985559822734")
        self.assertEqual(guidance["source_author"], "askalphaxiv")
        self.assertEqual(guidance["source_query"], "?s=46")
        self.assertEqual(guidance["source"], FOCUS.METHODOLOGY["source_post_url"])

    def test_allows_note_url_override_in_report_metadata(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload(
            {"random_hash_mask1": natural},
            descents=[
                {"recovered": True, "invalid_candidate_count": 0},
                {"recovered": True, "invalid_candidate_count": 0},
            ],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={
                    "audit_passed": True,
                    "source_sha256": FOCUS.sha256_file(source),
                },
                note_url="https://x.com/altstatus/status/9999999999999999999?s=46",
            )

        self.assertEqual(
            report["note_url"], "https://x.com/altstatus/status/9999999999999999999?s=46"
        )
        self.assertEqual(
            report["methodology"]["source_post_url"],
            "https://x.com/altstatus/status/9999999999999999999",
        )
        self.assertEqual(
            report["methodology"]["source_post_url_with_query"],
            "https://x.com/altstatus/status/9999999999999999999?s=46",
        )
        self.assertEqual(
            report["source_intake"]["tweet_source"]["tweet_author"], "altstatus"
        )
        self.assertEqual(
            report["source_intake"]["tweet_source"]["tweet_post_id"],
            "9999999999999999999",
        )
        self.assertEqual(
            report["source_intake"]["tweet_source"]["tweet_query"], "?s=46"
        )
        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(source_intake["tweet_source_title"], "")
        self.assertEqual(source_intake["tweet_referenced_paper_title"], "")
        self.assertEqual(source_intake["tweet_source_url"], "")
        self.assertEqual(source_intake["tweet_media_urls"], [])
        self.assertEqual(source_intake["tweet_media_types"], [])
        self.assertEqual(source_intake["tweet_media_count"], 0)
        self.assertFalse(source_intake["tweet_has_media"])
        self.assertEqual(source_intake["tweet_hashtags"], [])
        guidance = report["methodology"]["tweet_guidance"]
        self.assertFalse(guidance["tweet_text_included"])
        self.assertEqual(report["source_intake"]["tweet_source"]["tweet_text"], "")
        self.assertEqual(report["source_intake"]["tweet_source"]["tweet_text_sha256"], "")
        self.assertTrue(guidance["source_summary_is_verbatim"] is False)
        self.assertEqual(guidance["source_intake_mode"], "public_snapshot_summary")
        self.assertTrue(guidance["operator_interrupt_only_for_blocking_uncertainty"])
        self.assertIn("required_resolution_ids", guidance)

    def test_known_tweet_text_included_for_web_status_path(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                note_url="https://x.com/i/web/status/2076737985559822734?s=46",
            )

        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(source_intake["tweet_author"], "askalphaxiv")
        self.assertEqual(source_intake["tweet_post_id"], "2076737985559822734")
        self.assertEqual(source_intake["tweet_query"], "?s=46")
        self.assertTrue(source_intake["tweet_text_included"])
        self.assertEqual(source_intake["tweet_text"], source_intake["tweet_summary"])
        self.assertIn("GPT-5.6 stayed more focused", source_intake["tweet_summary"])
        self.assertEqual(
            source_intake["tweet_posted_at"],
            "2026-07-13T18:38:14.340000+00:00",
        )
        self.assertEqual(
            source_intake["tweet_source_title"],
            (
                "Towards Mechanistically Understanding Why Memorized Knowledge "
                "Fails to Generalize in Large Language Model Finetuning"
            ),
        )
        self.assertEqual(
            source_intake["tweet_referenced_paper_title"],
            (
                "Towards Mechanistically Understanding Why Memorized Knowledge "
                "Fails to Generalize in LLM Finetuning"
            ),
        )
        self.assertEqual(
            source_intake["tweet_source_url"],
            "https://arxiv.org/abs/2607.08393",
        )
        self.assertEqual(
            source_intake["tweet_media_urls"],
            [
                "https://x.com/askalphaxiv/status/2076737985559822734/photo/1"
            ],
        )
        self.assertEqual(source_intake["tweet_media_types"], ["video"])
        self.assertEqual(source_intake["tweet_media_count"], 1)
        self.assertTrue(source_intake["tweet_has_media"])
        self.assertEqual(source_intake["tweet_hashtags"], [])

    def test_status_only_tweet_paths_are_canonicalized_to_authorized_x_post(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        aliases = [
            "https://x.com/status/2076737985559822734?s=46",
            "https://x.com/i/status/2076737985559822734?s=46",
            "https://x.com/i/web/status/2076737985559822734?s=46",
        ]
        for alias in aliases:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                source = self.write_payload(Path(directory), value)
                report = FOCUS.build_report(value, source, note_url=alias)
            source_intake = report["source_intake"]["tweet_source"]
            self.assertEqual(
                source_intake["tweet_url"],
                "https://x.com/askalphaxiv/status/2076737985559822734",
            )
            self.assertEqual(
                source_intake["tweet_url_with_query"],
                "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
            )
            self.assertTrue(source_intake["tweet_text_included"])
            self.assertIn(
                "GPT-5.6 stayed more focused",
                source_intake["tweet_summary"],
            )

    def test_note_url_scheme_relative_input_is_normalized_and_parsed(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                note_url="x.com/askalphaxiv/status/2076737985559822734?s=46",
            )

        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(
            source_intake["tweet_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )
        self.assertEqual(
            source_intake["tweet_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )
        self.assertTrue(source_intake["tweet_text_included"])

    def test_http_scheme_input_is_normalized_to_https(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                note_url="http://x.com/askalphaxiv/status/2076737985559822734?s=46",
            )

        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(
            source_intake["tweet_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )
        self.assertEqual(
            source_intake["tweet_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )

    def test_host_aliases_are_normalized_to_x(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                note_url="www.x.com/askalphaxiv/status/2076737985559822734",
            )

        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(
            source_intake["tweet_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )

    def test_mobile_host_aliases_are_normalized(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                note_url="m.x.com/askalphaxiv/status/2076737985559822734?s=46",
            )

        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(
            source_intake["tweet_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )
        self.assertEqual(
            source_intake["tweet_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )

    def test_other_tweet_host_aliases_are_normalized(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        aliases = [
            "mobile.x.com/askalphaxiv/status/2076737985559822734?s=46",
            "www.twitter.com/askalphaxiv/status/2076737985559822734?s=46",
            "twitter.com/askalphaxiv/status/2076737985559822734?s=46",
        ]
        for alias in aliases:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                source = self.write_payload(Path(directory), value)
                report = FOCUS.build_report(value, source, note_url=alias)
                source_intake = report["source_intake"]["tweet_source"]
                self.assertEqual(
                    source_intake["tweet_url"],
                    "https://x.com/askalphaxiv/status/2076737985559822734",
                )
                self.assertEqual(
                    source_intake["tweet_url_with_query"],
                    "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
                )

    def test_guidance_compliance_is_recorded(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(1),
                "mixed_balanced_stride_mask1": config(3),
                "consecutive_canonical_mask1": config(4, logs=True),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        compliance = report["autoresearch_steering"]["guidance_compliance"]
        self.assertTrue(compliance["bounded_critical_set_enforced"])
        self.assertTrue(compliance["peripheral_scope_defer_enforced"])
        self.assertTrue(compliance["focus_selection_matches_budget"])
        self.assertTrue(compliance["selected_candidates_have_full_spec"])
        self.assertTrue(compliance["non_blocking_ambiguity_resolutions_recorded"])
        self.assertTrue(compliance["operator_interrupt_alignment"])
        self.assertFalse(compliance["missing_required_resolution_ids"])
        self.assertTrue(compliance["overall_compliant"])
        source_intake = report["source_intake"]["tweet_source"]
        self.assertEqual(
            source_intake["tweet_url"],
            "https://x.com/askalphaxiv/status/2076737985559822734",
        )
        self.assertEqual(source_intake["tweet_post_id"], "2076737985559822734")
        self.assertEqual(source_intake["tweet_author"], "askalphaxiv")
        self.assertEqual(
            source_intake["tweet_intake_mode"], "public_snapshot_summary"
        )
        self.assertEqual(len(source_intake["tweet_text_sha256"]), 64)
        self.assertTrue(source_intake["tweet_summary_is_verbatim"])
        self.assertTrue(source_intake["tweet_text_included"])
        self.assertIn("GPT-5.6 stayed more focused", source_intake["tweet_summary"])
        self.assertIn("@OpenAI pushing the boundaries", source_intake["tweet_text"])
        rep = report["summary"]["major_result_replication"]
        self.assertIn("cell_count", rep)
        self.assertIn("stage_status_counts", rep)
        self.assertEqual(rep["cell_count"], 1)
        self.assertIn("target_descent", rep["stage_status_counts"])

    def test_next_action_maps_to_top_focus_candidate(self) -> None:
        value = payload({"random_hash_mask1": config(1), "mixed_balanced_stride_mask1": config(3)})
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source, focus_budget=2)

        next_action = report["next_action"]
        self.assertEqual(next_action["position"], 1)
        self.assertEqual(next_action["focus_id"], report["focus_queue"][0]["id"])
        self.assertIn("action", next_action)
        self.assertIn("decisive_test", next_action)
        self.assertIn("required_artifacts", next_action)
        self.assertTrue(next_action["required_artifacts"])

    def test_reports_prospective_headroom_and_matched_hash_specificity(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(1),
                "mixed_balanced_stride_mask1": config(3),
                "consecutive_canonical_mask1": config(4, logs=True),
            }
        )
        policies = value["curve_records"][0]["policies"]
        for name, fixed_rank in (
            ("hash_control_0", 2),
            ("hash_control_1", 1),
            ("hash_control_2", 1),
        ):
            configurations = {"random_hash_mask1": config(1)}
            if name == "hash_control_0":
                configurations["mixed_balanced_stride_mask1"] = config(fixed_rank)
            policies[name] = {
                "full": {
                    "factor_base_size_B": 12,
                    "configurations": configurations,
                    "target_descents": [],
                }
            }
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        transfer = report["routing_generalization"]["prospective_transfer"]
        self.assertEqual(transfer["status"], "positive_rank_transfer")
        self.assertAlmostEqual(transfer["fixed_fraction_of_oracle_rank_headroom"], 2.0 / 3.0)
        specificity = report["routing_specificity_control"]
        self.assertEqual(specificity["matched_curve_count"], 1)
        self.assertEqual(specificity["comparisons"][0]["hash_control_count"], 1)
        self.assertTrue(specificity["target_gain_exceeds_hash_controls_on_every_matched_curve"])
        self.assertEqual(
            specificity["conclusion"],
            "coordinate_specific_routing_signal_diagnostic_only",
        )

    def test_matched_hash_gain_can_falsify_coordinate_specificity(self) -> None:
        value = payload(
            {
                "random_hash_mask1": config(1),
                "mixed_balanced_stride_mask1": config(2),
                "consecutive_canonical_mask1": config(4, logs=True),
            }
        )
        policies = value["curve_records"][0]["policies"]
        for name in FOCUS.HASH_POLICIES:
            policies[name] = {
                "full": {
                    "factor_base_size_B": 12,
                    "configurations": {
                        "random_hash_mask1": config(1),
                        "mixed_balanced_stride_mask1": config(3),
                    },
                    "target_descents": [],
                }
            }
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        specificity = report["routing_specificity_control"]
        self.assertFalse(specificity["target_gain_exceeds_hash_controls_on_every_matched_curve"])
        self.assertEqual(specificity["conclusion"], "generic_routing_effect_not_ruled_out")

    def test_shoup_pressure_summary_passes_with_multiscale_fit(self) -> None:
        curves = [
            curve_record(
                seed=1432401,
                bits=24,
                split="prospective",
                order=2**24 - 3,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.05,
                        total_operations=int((2**24 - 3) ** 0.4),
                    )
                },
                descents=[successful_descent()],
            ),
            curve_record(
                seed=1432402,
                bits=28,
                split="prospective",
                order=2**28 - 5,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.05,
                        total_operations=int((2**28 - 5) ** 0.4),
                    )
                },
                descents=[successful_descent()],
            ),
            curve_record(
                seed=1432403,
                bits=32,
                split="prospective",
                order=2**32 - 5,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.05,
                        total_operations=int((2**32 - 5) ** 0.4),
                    )
                },
                descents=[successful_descent()],
            ),
        ]
        value = payload_with_curves(curves, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        shoup = report["summary"]["shoup_pressure"]
        self.assertEqual(shoup["status"], "pass")
        self.assertTrue(shoup["meets_shoup_pressure_gate"])
        self.assertIsNotNone(shoup["fit"])
        self.assertLess(shoup["fit"]["exponent_in_group_order"], 0.5)
        focus_ids = {row["id"] for row in report["focus_queue"]}
        self.assertNotIn("shoup_pressure_scaling_probe", focus_ids)

    def test_shoup_pressure_candidate_added_when_scaling_is_inadequate(self) -> None:
        value = payload(
            {"random_hash_mask1": config(4, logs=True, cost_ratio=0.55)},
            descents=[{"recovered": True, "invalid_candidate_count": 0}],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source, focus_budget=2)

        shoup = report["summary"]["shoup_pressure"]
        self.assertFalse(shoup["meets_shoup_pressure_gate"])
        self.assertEqual(shoup["status"], "insufficient_comparable_cells")
        self.assertIn(
            "shoup_pressure_scaling_probe",
            {row["id"] for row in report["focus_queue"]},
        )

    def test_shoup_pressure_candidate_includes_nonpass_fit_gate(self) -> None:
        curves = [
            curve_record(
                seed=1432401,
                bits=24,
                split="prospective",
                order=2**24 - 3,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.2,
                        total_operations=int((2**24 - 3) ** 0.7),
                    )
                },
                descents=[successful_descent()],
            ),
            curve_record(
                seed=1432402,
                bits=28,
                split="prospective",
                order=2**28 - 5,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.45,
                        total_operations=int((2**28 - 5) ** 0.7),
                    )
                },
                descents=[successful_descent()],
            ),
            curve_record(
                seed=1432403,
                bits=32,
                split="prospective",
                order=2**32 - 5,
                configurations={
                    "random_hash_mask1": config(
                        4,
                        logs=True,
                        cost_ratio=0.9,
                        total_operations=int((2**32 - 5) ** 0.7),
                    )
                },
                descents=[successful_descent()],
            ),
        ]
        value = payload_with_curves(curves, breakthrough=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(value, source)

        shoup = report["summary"]["shoup_pressure"]
        self.assertEqual(shoup["status"], "fit_gate_failed")
        self.assertFalse(shoup["meets_shoup_pressure_gate"])
        self.assertGreater(shoup["fit"]["exponent_in_group_order"], 0.5)
        focus_ids = [row["id"] for row in report["focus_queue"]]
        self.assertEqual(focus_ids[0], "shoup_pressure_scaling_probe")

    def test_cli_writes_json_and_human_focus_note(self) -> None:
        value = payload({"random_hash_mask1": config(1)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            output = root / "focus.json"
            note = root / "focus.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(source),
                    "--output",
                    str(output),
                    "--note",
                    str(note),
                    "--focus-budget",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            note_text = note.read_text(encoding="utf-8")

        self.assertIn("claim=DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION", completed.stdout)
        self.assertLessEqual(len(report["focus_queue"]), 2)
        self.assertIn("## Focus Queue", note_text)
        self.assertIn("## Deferred Experiments", note_text)
        self.assertIn("## Ambiguity Resolutions", note_text)
        self.assertIn("## Autoresearch Guidance", note_text)
        self.assertIn("## Guidance Compliance", note_text)
        self.assertIn("## Major Result Replication", note_text)
        self.assertIn("## Experiment Lineage", note_text)
        self.assertIn("## Shoup Pressure", note_text)
        self.assertIn("Shoup pressure status:", note_text)
        self.assertIn(
            "Source: https://x.com/askalphaxiv/status/2076737985559822734?s=46",
            note_text,
        )
        self.assertIn("Bounded critical set:", note_text)
        self.assertIn("Deterministic non-blocking ambiguity handling:", note_text)
        self.assertIn("Peripheral scope deference:", note_text)
        self.assertIn("Operator interrupt policy:", note_text)
        self.assertIn("Tweet source captured:", note_text)
        self.assertIn("Tweet source summary:", note_text)
        self.assertIn(
            "Tweet source title: Towards Mechanistically Understanding Why Memorized Knowledge "
            "Fails to Generalize in Large Language Model Finetuning",
            note_text,
        )
        self.assertIn(
            "Paper title as written in post: Towards Mechanistically Understanding Why Memorized "
            "Knowledge Fails to Generalize in LLM Finetuning",
            note_text,
        )
        self.assertIn("Tweet media: 1 items, has_media=True (video)", note_text)
        self.assertIn("## Next Action", note_text)
        self.assertIn("Required artifacts:", note_text)
        self.assertEqual(
            report["source_intake"]["tweet_source"]["tweet_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )

    def test_cli_writes_summation_ffe_artifacts(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        natural["summation_poly_degree"] = 4
        natural["ffe_profile"] = "summation-replay-probe"
        natural["summation_polynomial_payload"] = exact_payload(
            "ecdlp.summation_polynomial_payload.v1"
        )
        natural["ffe_profile_payload"] = exact_payload("ecdlp.ffe_profile_payload.v1")
        natural["summation_ffe_new_factor_row_discovery"] = admitted_discovery_contract()
        value = payload(
            {"random_hash_mask1": natural},
            descents=[{"recovered": True, "invalid_candidate_count": 0}],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_payload(root, value)
            output = root / "focus.json"
            note = root / "focus.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(source),
                    "--output",
                    str(output),
                    "--note",
                    str(note),
                    "--focus-budget",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            inventory_path = root / "summation_ffe_evidence_inventory.json"
            replay_plan_path = root / "summation_ffe_evidence_replay_plan.json"
            self.assertTrue(inventory_path.exists())
            self.assertTrue(replay_plan_path.exists())
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            replay_plan = json.loads(replay_plan_path.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0)
        self.assertIn(
            "claim=DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION",
            completed.stdout,
        )
        self.assertEqual(
            inventory["schema"],
            FOCUS.SUMMATION_FFE_EVIDENCE_INVENTORY_SCHEMA,
        )
        self.assertEqual(
            replay_plan["schema"],
            FOCUS.SUMMATION_FFE_REPLAY_PLAN_SCHEMA,
        )
        self.assertEqual(len(inventory["records"]), 1)
        self.assertEqual(len(replay_plan["replay_rows"]), 1)
        self.assertEqual(inventory["execution_status"]["records_ready"], 1)
        self.assertEqual(inventory["execution_status"]["records_payload_ready"], 1)
        self.assertEqual(inventory["execution_status"]["records_missing_inputs"], 0)
        self.assertEqual(inventory["execution_status"]["records_lane_admitted"], 1)
        self.assertEqual(replay_plan["execution_status"]["replay_ready_count"], 1)
        self.assertEqual(replay_plan["execution_status"]["replay_missing_count"], 0)
        self.assertEqual(
            replay_plan["execution_status"]["status"],
            "ready",
        )
        source = inventory["source"]
        replay_source = replay_plan["source"]
        self.assertEqual(
            source["tweet_url_with_query"],
            "https://x.com/askalphaxiv/status/2076737985559822734?s=46",
        )
        self.assertEqual(
            replay_source,
            source,
        )
        self.assertEqual(source["tweet_media_count"], 1)
        self.assertTrue(source["tweet_has_media"])
        self.assertEqual(source["tweet_media_types"], ["video"])
        self.assertEqual(
            source["tweet_text_sha256"],
            FOCUS.KNOWN_TWEET_SOURCE_TEXT_SHA256,
        )
        self.assertEqual(
            source["tweet_source_title"],
            "Towards Mechanistically Understanding Why Memorized Knowledge "
            "Fails to Generalize in Large Language Model Finetuning",
        )
        self.assertEqual(
            replay_plan["replay_rows"][0]["replay_readiness"],
            "exact_replay_inputs_ready",
        )
        self.assertTrue(
            replay_plan["replay_rows"][0]["new_factor_row_discovery_gate"][
                "lane_admitted"
            ]
        )

    def test_summation_ffe_artifacts_require_exact_payloads(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        natural["summation_poly_degree"] = 8
        natural["ffe_profile"] = "summation-replay-probe"
        value = payload({"random_hash_mask1": natural}, breakthrough=True)
        inventory, replay_plan = FOCUS.build_summation_ffe_artifacts(value)

        self.assertEqual(inventory["execution_status"]["status"], "missing_exact_inputs")
        self.assertEqual(inventory["execution_status"]["records_ready"], 0)
        self.assertEqual(inventory["execution_status"]["records_missing_inputs"], 1)
        self.assertEqual(replay_plan["replay_rows"][0]["replay_readiness"], "missing_exact_inputs")
        self.assertEqual(
            replay_plan["replay_rows"][0]["summation_ffe_replay_readiness_class"],
            "marker_only",
        )

    def test_opaque_payload_labels_do_not_count_as_exact_replay(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        natural["summation_poly_degree"] = 8
        natural["ffe_profile"] = "summation-replay-probe"
        natural["summation_polynomial_payload"] = "unbound-summation-label"
        natural["ffe_profile_payload"] = "unbound-ffe-label"
        natural["summation_ffe_new_factor_row_discovery"] = admitted_discovery_contract()
        record = FOCUS.stage_record(natural)

        readiness = record["summation_ffe_readiness"]
        self.assertEqual(readiness["replay_readiness"], "missing_exact_inputs")
        self.assertEqual(readiness["replay_readiness_class"], "partial_payload")
        self.assertEqual(
            readiness["payload_validation"]["summation"]["invalid_fields"][0]["reason"],
            "external_or_opaque_payload_requires_companion_sha256",
        )
        gate = record["summation_ffe_new_factor_row_discovery_gate"]
        self.assertEqual(gate["status"], "blocked_missing_exact_replay")
        self.assertFalse(gate["lane_admitted"])

    def test_new_factor_row_contract_must_beat_direct_pair_complement(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        natural["summation_poly_degree"] = 8
        natural["ffe_profile"] = "summation-replay-probe"
        natural["summation_polynomial_payload"] = exact_payload(
            "ecdlp.summation_polynomial_payload.v1"
        )
        natural["ffe_profile_payload"] = exact_payload("ecdlp.ffe_profile_payload.v1")
        natural["summation_ffe_new_factor_row_discovery"] = admitted_discovery_contract(
            measured_source_operations=100,
            direct_pair_complement_operations=100,
        )
        record = FOCUS.stage_record(natural)

        gate = record["summation_ffe_new_factor_row_discovery_gate"]
        self.assertEqual(gate["status"], "fails_direct_pair_complement_cost_gate")
        self.assertFalse(gate["lane_admitted"])
        self.assertEqual(gate["cost_ratio_vs_direct_pair_complement"], 1.0)
        self.assertIn(
            "source_cost_not_below_direct_pair_complement",
            gate["failures"],
        )

    def test_summation_ffe_gate_blocks_otherwise_complete_promotion(self) -> None:
        natural = config(4, logs=True, cost_ratio=0.5)
        natural["summation_poly_degree"] = 8
        natural["ffe_profile"] = "summation-replay-probe"
        value = payload(
            {"random_hash_mask1": natural},
            descents=[{"recovered": True, "invalid_candidate_count": 0}],
            breakthrough=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_payload(Path(directory), value)
            report = FOCUS.build_report(
                value,
                source,
                audit={
                    "audit_passed": True,
                    "source_sha256": FOCUS.sha256_file(source),
                },
            )

        self.assertFalse(report["summary"]["natural_pipeline_complete"])
        self.assertFalse(report["summary"]["natural_summation_ffe_admission_complete"])
        self.assertFalse(report["summary"]["promotion_allowed"])
        self.assertEqual(
            report["next_action"]["focus_id"],
            "summation_ffe_new_factor_row_cost_gate",
        )


if __name__ == "__main__":
    unittest.main()
