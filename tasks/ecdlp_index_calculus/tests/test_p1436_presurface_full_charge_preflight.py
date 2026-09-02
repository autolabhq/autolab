from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "p1436_presurface_full_charge_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("p1436_presurface_charge", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def route(*, passing: bool) -> dict:
    ratio = 0.8 if passing else 1.1
    return {
        "all_splits_full_recall": passing,
        "all_splits_positive": passing,
        "split_count": 4,
        "test_strict_recall": 1.0,
        "test_strict_surface_count": 8,
        "test_recovered_strict_surface_count": 8,
        "strict_public_quadratic_root_ops": 40,
        "all_selected_evaluated_factor_count": 40 if passing else 70,
        "all_selected_selector_eval_ops": 40 if passing else 90,
        "sum_generic_rho_steps_for_all_test_strict": 100,
        "root_plus_evaluated_factor_proxy_over_all_test_strict_rho": ratio,
        "root_plus_selector_eval_ops_over_all_test_strict_rho": ratio
        if passing
        else 1.3,
    }


def fixtures(*, passing: bool) -> dict:
    proposal_count = 20
    strict_count = 8
    guard_summary = {
        "strict_recall_on_materialized": 1.0,
        "missed_strict_surface_count": 0,
        "false_positive_materialized_surface_count": 0,
        "max_charged_ops_over_rho": 0.8,
        "materialized_strict_surface_count": strict_count,
    }
    heldout_route = route(passing=passing)
    historical = {
        "proposal_count": 10,
        "strict_surface_count": 4,
        "strict_public_quadratic_root_ops": 60,
        "all_proposal_evaluated_factor_count": 41,
        "all_proposal_selector_eval_ops": 90,
        "sum_generic_rho_steps_for_strict": 100,
        "root_plus_evaluated_factor_proxy_over_strict_rho": 1.01,
        "root_plus_selector_eval_ops_over_strict_rho": 1.5,
    }
    return {
        "generator": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["generator"],
            "parameters": {
                "selection_labels_forbidden": sorted(
                    PREFLIGHT.REQUIRED_FORBIDDEN_SELECTION_LABELS
                )
            },
            "summary": {
                "promoted_pre_surface_profile": {
                    "profile": "low_term_support_total2",
                    "summary": {
                        "proposal_surface_count": proposal_count,
                        "materialized_strict_surface_count": strict_count,
                        "missed_materialized_surface_count": 0,
                        "row_leaf_recall": 1.0,
                        "proposal_to_materialized_precision": 0.4,
                    },
                }
            },
        },
        "pruning": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["pruning"],
            "summary": {
                "status": "NEGATIVE PUBLIC PRUNING RESULT",
                "best_complete_recall_family": None,
                "best_recall_family": {"summary": {"row_leaf_recall": 0.8}},
            },
        },
        "stage_guard": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["stage_guard"],
            "summary": {
                "charge_ready_for_all_proposals": False,
                "materialization_coverage": 0.25,
            },
        },
        "backfill": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["backfill"],
            "summary": {
                "charge_ready_for_all_proposals": True,
                "materialization_coverage": 1.0,
                "unmaterialized_proposal_surface_count": 0,
                "proposal_surface_count": proposal_count,
                "best_full_materialized_guard": {
                    "guard": "factor_zero_equals_selected",
                    "summary": guard_summary,
                },
                "new_independent_fixed_sum_row_count": 2 if passing else 0,
                "fresh_rank_delta": 2 if passing else 0,
            },
        },
        "prefactor": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["prefactor"],
            "method": "factor_stream_charge",
            "parameters": {
                "charge_proxy_note": "complete",
                "full_sage_factorization_cost_charged": passing,
                "all_rejected_candidates_charged": passing,
            },
            "summary": {"status": "TEST"},
        },
        "screen_holdout": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["screen_holdout"],
            "summary": {
                "proposal_surface_count": proposal_count,
                "strict_surface_count": strict_count,
                "target_count": 4 if passing else 2,
                "window_count": 8,
                "status": "TEST",
                "leave_one_target_route": {
                    "baseline_policy_only": heldout_route,
                    "trained_screen_policy": heldout_route,
                },
            },
        },
        "historical_prefactor": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["historical_prefactor"],
            "summary": {
                "status": "MIXED",
                "best_single_policy_stream_proxy": historical,
            },
        },
        "p1324": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1324"],
            "summary": {
                "target": "third-target",
                "p1323_strict_slice_quadratic_gate_passed": passing,
                "exact_route_join_gate_passed": passing,
                "additional_or_target_independent_gate_passed": passing,
            },
        },
        "r68": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["r68"],
            "result": {
                "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": True
            },
        },
    }


class PresurfaceFullChargePreflightTests(unittest.TestCase):
    def test_rejects_component_only_branch(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=False))

        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertEqual(report["admission"]["passed_obligation_count"], 3)
        self.assertEqual(
            report["classification"], "PRESURFACE_PUBLIC_COMPONENT_ONLY"
        )
        self.assertFalse(report["claim_boundary"]["algorithm_breakthrough"])

    def test_admits_complete_source_without_promoting_full_attack(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=True))

        self.assertTrue(report["admission"]["lane_admitted"])
        self.assertEqual(report["admission"]["failed_obligations"], [])
        self.assertFalse(report["claim_boundary"]["algorithm_breakthrough"])

    def test_records_historical_proxy_gap_separately(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=False))
        near_miss = report["historical_two_operation_proxy_near_miss"]

        self.assertEqual(near_miss["evaluated_factor_proxy_excess_over_rho"], 1)
        self.assertEqual(near_miss["selector_eval_excess_over_rho"], 50)
        self.assertFalse(near_miss["factorization_cost_included"])

    def test_fails_closed_on_schema_drift(self) -> None:
        values = fixtures(passing=True)
        values["backfill"]["schema"] = "unexpected"

        with self.assertRaisesRegex(ValueError, "input schema mismatch"):
            PREFLIGHT.evaluate(values)


if __name__ == "__main__":
    unittest.main()
