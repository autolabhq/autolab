from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "p1553_5a5c_implicit_veronese_hyperplane_source_index_probe_r94.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r94", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R94 probe")
R94 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R94)


class ImplicitVeroneseHyperplaneSourceIndexTests(unittest.TestCase):
    def test_fermat_projector_is_exact_on_entire_field(self) -> None:
        control = R94.minimal_projector_degree_control()
        self.assertEqual(control["projector"], "1-X^(p-1)")
        self.assertEqual(control["degree"], 100)
        self.assertTrue(control["all_field_values_exact"])
        self.assertTrue(control["minimal_degree_exact"])

    def test_projector_matches_resultant_zero_predicate(self) -> None:
        control = R94.implicit_index_semantics_control()
        self.assertTrue(control["all_projector_predicates_exact"])
        self.assertTrue(control["all_dyadic_source_routes_exact"])

    def test_projector_rank_reaches_full_pair_chart_rank(self) -> None:
        rows = R94.implicit_index_semantics_control()["chart_rows"]
        self.assertEqual(
            [row["projector_matrix_rank"] for row in rows],
            [7, 18, 29, 53, 78],
        )
        self.assertEqual(rows[-1]["pair_chart_count"], 78)
        self.assertEqual(rows[-1]["projector_matrix_rank"], 78)

    def test_direct_dyadic_source_is_exact_but_scan_sized(self) -> None:
        rows = R94.implicit_index_semantics_control()["chart_rows"]
        for row in rows:
            self.assertTrue(row["all_dyadic_counts_exact"])
            self.assertTrue(row["all_nonempty_rows_return_zero"])
            self.assertTrue(
                row["all_returned_zeroes_have_signed_source"]
            )
            self.assertTrue(row["less_than_twice_one_side_scan"])
            self.assertGreaterEqual(
                row["maximum_direct_dyadic_field_evaluations"],
                row["pair_chart_count"],
            )

    def test_materialized_and_all_output_routes_miss_caps(self) -> None:
        control = R94.asymptotic_cost_control()
        self.assertEqual(
            control["r84_root_source_sides"][
                "smaller_exponent_B"
            ]["exact"],
            "12/5",
        )
        self.assertFalse(
            control["materialized_smaller_feature_index"][
                "inside_setup_cap"
            ]
        )
        self.assertFalse(
            control["direct_implicit_projector_scan"][
                "inside_online_cap"
            ]
        )
        self.assertFalse(
            control["all-output_multipoint_evaluation"][
                "inside_online_cap"
            ]
        )

    def test_multipoint_receipt_preserves_non_lower_bound_scope(
        self,
    ) -> None:
        control = R94.multipoint_literature_control()
        self.assertIn("N output points", control["bound"])
        self.assertIn("upper-bound control", control["scope"])
        self.assertFalse(
            control["source_reporting_without_all_outputs_supplied"]
        )

    def test_bundle_routes_only_to_aggregate_recurrence(self) -> None:
        report = R94.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"],
            10,
        )
        self.assertEqual(report["admission"]["obligation_count"], 23)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn("does not lower-bound", report["scope_boundary"])
        self.assertIn("aggregate recurrence", report["next_action"])


if __name__ == "__main__":
    unittest.main()
