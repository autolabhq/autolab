import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_torus_c5_label_congruence_correction_probe_r143.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r143_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R143 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R143)


class TorusC5LabelCongruenceCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R143.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R143.verify_source_bindings()), 10)

    def test_all_parameter_subsets_are_enumerated(self):
        self.assertEqual(self.controls["actual_control_count"], 8)
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["parameter_subset_count"],
                (1 << control["parameter_count"]) - 1,
            )

    def test_cross_ratio_defect_replays_exactly(self):
        self.assertTrue(
            self.controls["all_cross_ratio_defect_identities_exact"]
        )
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["cross_ratio_defect_row_count"],
                control["parameter_count"]
                * control["c2_by_c3_operand_pair_count"],
            )
            self.assertTrue(
                all(
                    row["identity_exact"]
                    for row in control["cross_ratio_defect_rows"]
                )
            )

    def test_deterministic_and_impossible_counts_are_stable(self):
        self.assertEqual(
            self.controls["deterministic_composition_control_count"],
            6,
        )
        self.assertEqual(
            self.controls["composition_impossible_control_count"],
            2,
        )

    def test_every_deterministic_table_recreates_full_operand_body(self):
        self.assertTrue(
            self.controls[
                "all_deterministic_tables_recreate_full_c2_by_c3_body"
            ]
        )
        for control in self.controls["actual_controls"]:
            best = control["best_deterministic_composition"]
            if best is None:
                continue
            self.assertEqual(
                best["input_output_table_entry_count"],
                control["c2_by_c3_operand_pair_count"],
            )

    def test_every_control_fails_the_finite_table_cap(self):
        self.assertTrue(
            self.controls[
                "all_controls_fail_finite_B9_over_4_table_cap"
            ]
        )

    def test_label_congruence_scope_is_explicit(self):
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["label_congruence"]["conclusion"],
            "Every total label-only composition is constant or injective.",
        )
        self.assertIn("total product laws", theorem["scope"])
        self.assertIn(
            "not a lower bound",
            theorem["generalized_birthday_explicit_row_boundary"][
                "scope"
            ],
        )
        self.assertIn("circuits", theorem["scope"])

    def test_explicit_row_birthday_envelope_is_exact(self):
        boundary = R143.generalized_birthday_boundary()
        self.assertEqual(boundary["campaign_beta"]["exact"], "9/20")
        self.assertEqual(
            boundary["campaign_merge_exponent_N"]["exact"],
            "11/20",
        )
        self.assertEqual(
            boundary["minimum_over_beta"]["total_exponent_N"]["exact"],
            "1/2",
        )
        self.assertFalse(boundary["campaign_below_rho"])

    def test_no_discrete_log_is_consumed(self):
        self.assertFalse(
            self.controls["candidate_discrete_log_oracle_consumed"]
        )
        self.assertFalse(
            self.bundle["cost"]["candidate_field_dlp_used"]
        )

    def test_rank_logs_and_identical_descent_remain_open(self):
        logs = self.bundle["logs"]
        self.assertFalse(logs["known_rhs_relation_rank_computed"])
        self.assertFalse(logs["factor_logs_computed"])
        self.assertFalse(logs["identical_target_descent_computed"])

    def test_gate_does_not_promote_operator_or_algorithm(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 23)
        self.assertTrue(
            admission["label_only_composition_negative_admitted"]
        )
        self.assertTrue(
            admission["explicit_row_birthday_boundary_admitted"]
        )
        self.assertFalse(
            admission["implicit_relation_span_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
