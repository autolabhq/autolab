from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "p1436_slice_quadratic_public_source_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("p1436_slice_source", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


ORACLE_SOURCE = """
def selected_coordinates(surface_record):
    return surface_record["selected_leaf_indices"]

def choose_slice_factors(surface_record, axis):
    coords = selected_coordinates(surface_record)
    return coords
"""

PUBLIC_SOURCE = """
def choose_slice_factors(surface_record, axis):
    return surface_record["public_candidate_schedule"]
"""


def fixtures(*, passing: bool) -> dict:
    targets = ("a", "b", "c", "d") if passing else ("a", "b")
    surfaces = []
    for index, target in enumerate(targets):
        surfaces.append(
            {
                "surface_id": f"surface-{target}",
                "target": target,
                "row_key": f"row-{target}",
                "challenge_seed": f"seed-{target}",
                "p": 101,
                "split": (
                    "train"
                    if passing and index < 2
                    else "heldout" if passing else None
                ),
                "preserving_candidate_count": 2,
                "candidates": [
                    {
                        "axis": "b",
                        "generic_rho_steps": 100,
                        "factors": [{"fixed_value": index + 1}],
                    },
                    {
                        "axis": "c",
                        "generic_rho_steps": 100,
                        "factors": [{"fixed_value": index + 2}],
                    },
                ],
            }
        )
    return {
        "slice_probe": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["slice_probe"],
            "parameters": {
                "selector_mode": "public_pre_choice" if passing else None
            },
            "summary": {
                "surface_candidate_count": 2 * len(surfaces),
                "surface_count": len(surfaces),
                "surfaces_with_preserving_slice_quadratic": len(surfaces),
                "slice_quadratic_all_hit_below_rho_count": len(surfaces),
                "slice_quadratic_selected_hit_below_rho_count": len(surfaces),
                "min_slice_quadratic_selected_hit_ops_over_rho": 0.4,
                "min_slice_quadratic_all_hit_ops_over_rho": 0.5,
                "all_public_candidates_charged": passing,
                "public_pre_choice_candidate_count": (
                    2 * len(surfaces) if passing else 0
                ),
                "public_source_total_ops_over_rho_max": 0.8 if passing else None,
                "new_independent_fixed_sum_row_count": 2 if passing else 0,
                "fresh_rank_delta": 2 if passing else 0,
            },
            "surfaces": surfaces,
        },
        "signature": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["signature"],
            "positive_cases": [{"target": "a"}],
            "negative_cases": [{"target": "d"}] if passing else [],
        },
        "p1228": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1228"],
            "leave_one_target_out_structural_rule": {
                "covered_positive_count": 2,
                "covered_negative_count": 2 if passing else 0,
            },
            "promotion_gate": {"passed": passing},
        },
        "p1324": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1324"],
            "summary": {
                "exact_route_join_gate_passed": passing,
                "additional_or_target_independent_gate_passed": passing,
            },
        },
        "r68": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["r68"],
            "pass": True,
            "result": {
                "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": True
            },
        },
    }


class SliceQuadraticPublicSourcePreflightTests(unittest.TestCase):
    def test_detects_selected_leaf_oracle_dependency(self) -> None:
        audit = PREFLIGHT.source_dependency_audit(ORACLE_SOURCE)

        self.assertTrue(audit["oracle_label_dependency"])
        self.assertTrue(audit["chooser_calls_selected_coordinates"])
        self.assertTrue(audit["selected_coordinates_reads_selected_leaf_indices"])

    def test_rejects_oracle_positive_only_diagnostic(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=False), ORACLE_SOURCE)

        self.assertFalse(report["admission"]["source_lane_admitted"])
        self.assertEqual(report["admission"]["passed_obligation_count"], 0)
        self.assertEqual(
            report["classification"], "SLICE_QUADRATIC_ORACLE_DIAGNOSTIC_ONLY"
        )
        blind = report["obligations"]["public_source_below_rho"][
            "blind_uniform_slice_audit"
        ]
        self.assertGreater(
            blind["minimum_optimistic_expected_draws_over_rho"], 0.5
        )
        self.assertFalse(report["claim_boundary"]["algorithm_breakthrough"])

    def test_admits_complete_public_source_but_not_full_attack(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=True), PUBLIC_SOURCE)

        self.assertTrue(report["admission"]["source_lane_admitted"])
        self.assertEqual(report["admission"]["failed_obligations"], [])
        self.assertFalse(report["claim_boundary"]["algorithm_breakthrough"])

    def test_fails_closed_on_schema_drift(self) -> None:
        values = fixtures(passing=True)
        values["signature"]["schema"] = "unexpected"

        with self.assertRaisesRegex(ValueError, "input schema mismatch"):
            PREFLIGHT.evaluate(values, PUBLIC_SOURCE)


if __name__ == "__main__":
    unittest.main()
