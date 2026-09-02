from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_black_box_translated_resultant_localizer_probe_r88.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r88", MODULE_PATH)
assert SPEC and SPEC.loader
R88 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R88)


class BlackBoxTranslatedResultantLocalizerTests(unittest.TestCase):
    def test_conditional_oracle_localizer_is_logarithmic(self) -> None:
        replay = R88.conditional_oracle_source_replay()
        self.assertEqual(replay["target_multiplicity"], 1)
        self.assertTrue(replay["logarithmic_call_count_exact"])
        self.assertEqual(replay["oracle_localization"]["oracle_calls"], 6)
        self.assertTrue(replay["joint_source_replay"])
        self.assertFalse(replay["oracle_constructor_supplied"])
        self.assertFalse(replay["candidate_credit"])

    def test_scalar_krylov_has_full_quotient_complexity(self) -> None:
        control = R88.quotient_krylov_control()
        self.assertEqual(control["quotient_dimension_m"], 32)
        self.assertEqual(control["zero_eigenvalue_count"], 1)
        self.assertEqual(control["distinct_eigenvalue_count"], 32)
        self.assertEqual(control["scalar_krylov_hankel_rank"], 32)
        self.assertTrue(control["scalar_krylov_full_linear_complexity"])

    def test_block_krylov_cap_interval_is_empty(self) -> None:
        tradeoff = R88.block_krylov_tradeoff()
        self.assertEqual(tradeoff["minimum_alpha_for_online_cap"], 0.75)
        self.assertEqual(tradeoff["maximum_alpha_for_setup_cap"], 0.25)
        self.assertTrue(tradeoff["feasible_alpha_interval_empty"])
        self.assertFalse(any(row["both_caps"] for row in tradeoff["points"]))

    def test_scalar_oracle_needs_multiplicity_jet(self) -> None:
        controls = R88.exceptional_controls()
        self.assertTrue(controls["empty"]["correctly_rejected"])
        self.assertFalse(
            controls["multiple_distinct"][
                "scalar_zero_oracle_alone_rejects_multiple"
            ]
        )
        self.assertTrue(
            controls["multiple_distinct"][
                "first_jet_rejects_unique_branch"
            ]
        )
        self.assertTrue(
            controls["nonreduced"]["first_jet_rejects_unique_branch"]
        )

    def test_bundle_preserves_fixed_marker_recurrence_exception(self) -> None:
        bundle = R88.build_bundle()
        report = bundle["report"]
        self.assertEqual(
            report["classification"],
            (
                "ORACLE_SOURCE_LOCALIZER_LOGARITHMIC__"
                "SCALAR_BLOCK_KRYLOV_OR_HALF_GCD_OVER_CAP"
            ),
        )
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertIn("fixed-marker", report["next_action"])
        self.assertFalse(report["factor_log_solve_complete"])
        self.assertFalse(report["fresh_target_descent_complete"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertFalse(report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
