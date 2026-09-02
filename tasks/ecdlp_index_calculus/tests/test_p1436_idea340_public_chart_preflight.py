from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "p1436_idea340_public_chart_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("p1436_idea340", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


POLICIES = ("coordinate_public", "map_public")


def fixtures(*, passing: bool) -> dict:
    policy_audit = {
        name: {
            "additive_energy_gate": passing,
            "exactness_rank_descent_gate": True,
            "symbolic_B2_5_gate": passing,
        }
        for name in POLICIES
    }
    ratios = {
        name: {
            "costs": {
                "total_tested_batch_field_units_over_sign_rho": (
                    0.75 if passing else 22.0
                )
            }
        }
        for name in POLICIES
    }
    ratios["hash_control_0"] = {
        "costs": {"total_tested_batch_field_units_over_sign_rho": 0.1}
    }
    return {
        "p1407": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1407"],
            "forbidden_selector_inputs_used": [],
            "summary": {
                "evaluated_target_policy_count": 8,
                "public_recovery_success_count": 8,
                "invalid_advice_witness_count": 0,
            },
        },
        "p1408": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1408"],
            "forbidden_selector_inputs_used": [],
            "summary": {
                "evaluated_target_policy_count": 8,
                "public_recovery_success_count": 8,
                "invalid_advice_witness_count": 0,
            },
        },
        "p1416": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1416"],
            "summary": {
                "promotion_audit": {"policies": policy_audit},
                "invalid_witness_count": 0,
                "full_rank_policy_curve_count": 2,
                "policy_curve_count": 2,
                "target_success_count": 16,
                "target_total": 16,
            },
            "curve_records": [
                {
                    "split": "heldout",
                    "policies": ratios,
                }
            ],
        },
        "p1432": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["p1432"],
            "summary": {
                "full_factor_rank_cell_count_in_fixed_512_target_batch": (
                    2 if passing else 1
                ),
                "policy_curve_count": 2,
                "fresh_curve_persistence_gate": passing,
                "explicit_pair_triple_join_symbolic_basis_exponent_in_n": (
                    0.4 if passing else 0.6
                ),
                "projected_selected_basis_work_exponent_in_n": (
                    0.45 if passing else 0.675
                ),
                "prospective_generation_started_after_training_freeze": True,
            },
        },
        "r68": {
            "schema": PREFLIGHT.EXPECTED_SCHEMAS["r68"],
            "pass": True,
            "result": {
                "ffe_product_relations_add_information_beyond_fixed_sum_rows": False,
                "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": True,
            },
        },
    }


class Idea340PublicChartPreflightTests(unittest.TestCase):
    def test_rejects_exact_but_high_cost_unstable_public_chart(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=False))

        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertEqual(report["admission"]["passed_obligation_count"], 2)
        self.assertTrue(
            report["obligations"]["public_dlp_free_factor_base_construction"]["pass"]
        )
        self.assertTrue(
            report["obligations"]["exact_relation_rank_and_target_descent"]["pass"]
        )
        self.assertFalse(
            report["obligations"]["superuniform_heldout_additive_energy"]["pass"]
        )
        rho = report["obligations"]["heldout_end_to_end_below_rho"]
        self.assertEqual(rho["minimum_ratio_over_all_public_policy_cells"], 22.0)
        self.assertEqual(rho["tested_public_policy_count"], 2)
        self.assertNotIn("hash_control_0", rho["per_policy_maximum_ratio"])
        self.assertFalse(report["claim_boundary"]["algorithm_breakthrough"])

    def test_admits_only_when_every_obligation_passes(self) -> None:
        report = PREFLIGHT.evaluate(fixtures(passing=True))

        self.assertTrue(report["admission"]["lane_admitted"])
        self.assertEqual(report["admission"]["failed_obligations"], [])
        self.assertTrue(
            all(
                obligation["pass"]
                for obligation in report["obligations"].values()
            )
        )

    def test_fails_closed_on_schema_drift(self) -> None:
        values = fixtures(passing=True)
        values["p1416"]["schema"] = "unexpected"

        with self.assertRaisesRegex(ValueError, "input schema mismatch"):
            PREFLIGHT.evaluate(values)


if __name__ == "__main__":
    unittest.main()
