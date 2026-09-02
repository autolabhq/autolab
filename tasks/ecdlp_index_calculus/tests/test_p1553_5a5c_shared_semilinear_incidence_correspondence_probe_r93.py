from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_shared_semilinear_incidence_correspondence_probe_r93.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r93", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R93 probe")
R93 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R93)


class SharedSemilinearIncidenceCorrespondenceTests(unittest.TestCase):
    def test_resultant_veronese_identity_is_exact(self) -> None:
        control = R93.shared_operator_control()
        self.assertTrue(
            control["identity_exact_on_all_78_by_78_curve_charts"]
        )
        self.assertEqual(control["operator_rank"], 6)

    def test_raw_resultant_rank_stays_six(self) -> None:
        sweep = R93.exact_chart_sweep()
        self.assertEqual(
            [row["raw_resultant_matrix_rank"] for row in sweep],
            [6, 6, 6, 6, 6],
        )
        self.assertEqual(
            [row["veronese_feature_rank"] for row in sweep],
            [6, 6, 6, 6, 6],
        )

    def test_zero_incidence_rank_reaches_full_pair_count(self) -> None:
        sweep = R93.exact_chart_sweep()
        self.assertEqual(
            [row["zero_incidence_matrix_rank"] for row in sweep],
            [7, 18, 29, 53, 78],
        )
        self.assertEqual(sweep[-1]["pair_chart_count"], 78)
        self.assertEqual(sweep[-1]["zero_incidence_matrix_rank"], 78)

    def test_every_affine_zero_has_exact_signed_source(self) -> None:
        for row in R93.exact_chart_sweep():
            self.assertTrue(row["predicate_source_biconditional_exact"])
            self.assertTrue(row["every_zero_has_signed_source"])
            self.assertEqual(row["predicate_source_mismatch_count"], 0)
            self.assertGreater(row["proper_subsum_source_count"], 0)
            self.assertGreater(row["full_only_source_count"], 0)

    def test_matched_random_controls_separate_value_and_zero_rank(
        self,
    ) -> None:
        for row in R93.random_quadratic_controls():
            self.assertLessEqual(row["raw_resultant_matrix_rank"], 6)
            self.assertEqual(row["shared_operator_rank"], 6)
            self.assertGreaterEqual(
                row["zero_incidence_matrix_rank"],
                row["raw_resultant_matrix_rank"],
            )

    def test_root_feature_rows_miss_both_caps(self) -> None:
        control = R93.root_source_feature_cost_control()
        self.assertEqual(
            control["smaller_feature_body_exponent_B"]["exact"],
            "12/5",
        )
        self.assertEqual(
            control["larger_feature_body_exponent_B"]["exact"],
            "13/5",
        )
        self.assertFalse(
            control["smaller_feature_body_inside_setup_cap"]
        )
        self.assertFalse(control["one_side_scan_inside_online_cap"])

    def test_bundle_preserves_implicit_range_index(self) -> None:
        report = R93.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"],
            9,
        )
        self.assertEqual(report["admission"]["obligation_count"], 20)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("not a lower bound", report["scope_boundary"])
        self.assertIn(
            "implicit Cartesian Veronese hyperplane source index",
            report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
