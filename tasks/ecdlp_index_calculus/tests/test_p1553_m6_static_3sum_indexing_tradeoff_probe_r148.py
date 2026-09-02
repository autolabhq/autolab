import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "p1553_m6_static_3sum_indexing_tradeoff_probe_r148.py"
SPEC = importlib.util.spec_from_file_location("p1553_r148_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R148 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R148)


class M6Static3SUMIndexingTradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R148.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]
        cls.costs = cls.bundle["cost"][
            "published_upper_bound_instantiations"
        ]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R148.verify_source_bindings()), 12)

    def test_occurrence_pair_query_reduces_to_weighted_indexing(self):
        reduction = self.report["reduction"]
        self.assertIn("static 3SUM-indexing", reduction["decision_equivalence"])
        self.assertIn(
            "exact integer multiplicity",
            reduction["required_output_is_stronger"],
        )
        self.assertIn(
            "A/C atom marginals",
            reduction["required_output_is_stronger"],
        )

    def test_all_actual_controls_and_lengths_are_frozen(self):
        actual = self.controls["actual_controls"]
        self.assertEqual(len(actual), 8)
        self.assertEqual(
            sorted(row["occurrence_list_length"] for row in actual),
            [27, 27, 125, 125, 216, 216, 343, 343],
        )
        self.assertTrue(
            self.controls["all_sample_batches_have_eighteen_queries"]
        )

    def test_finite_scan_accounting_is_exact(self):
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["trivial_scan_operations_per_query"],
                control["occurrence_list_length"],
            )
            self.assertEqual(
                control["trivial_sample_batch_operations"],
                control["occurrence_list_length"]
                * control["sampled_query_count"],
            )
            self.assertTrue(
                control["all_r147_query_answers_reused_exactly"]
            )

    def test_finite_full_sumset_accounting_is_exact(self):
        for control in self.controls["actual_controls"]:
            self.assertEqual(
                control["full_sumset_table_entries"],
                control["occurrence_list_length"] ** 2,
            )
            self.assertEqual(
                control["linear_state_words"],
                control["occurrence_list_length"],
            )

    def test_r115_exponents_are_frozen(self):
        self.assertEqual(
            self.costs["occurrence_list_length_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            self.costs["query_batch_size_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            self.costs["trivial_full_batch_exponent_B"]["exact"],
            "7/2",
        )
        self.assertEqual(
            self.costs["trivial_full_batch_exponent_N"]["exact"],
            "7/10",
        )
        self.assertEqual(
            self.costs["full_sumset_state_exponent_B"]["exact"],
            "9/2",
        )

    def test_fiat_naor_linear_state_endpoint_is_dominated(self):
        tradeoff = self.costs["gghpv_tradeoff"]
        self.assertEqual(tradeoff["relation"], "T*S^3=soft-O(n^6)")
        self.assertEqual(
            tradeoff["linear_state_query_exponent_n"]["exact"],
            "3",
        )
        self.assertEqual(
            tradeoff["linear_state_full_batch_exponent_B"]["exact"],
            "8",
        )
        self.assertTrue(
            tradeoff["linear_state_endpoint_dominated_by_trivial_scan"]
        )

    def test_dinur_golovnev_range_exceeds_both_caps(self):
        tradeoff = self.costs["dinur_golovnev_tradeoff"]
        self.assertEqual(tradeoff["relation"], "T*S=soft-O(n^(5/2))")
        self.assertEqual(
            tradeoff["minimum_improvement_state_exponent_B"]["exact"],
            "27/8",
        )
        self.assertEqual(
            tradeoff["preprocessing_exponent_B"]["exact"],
            "9/2",
        )
        self.assertTrue(tradeoff["minimum_state_exceeds_setup_cap"])
        self.assertTrue(tradeoff["preprocessing_exceeds_setup_cap"])

    def test_no_sqrt_query_count_or_marginal_index_is_imported(self):
        cost = self.bundle["cost"]
        self.assertFalse(cost["linear_state_sqrt_query_claimed"])
        self.assertFalse(cost["published_exact_count_index_supplied"])
        self.assertFalse(cost["published_atom_marginal_index_supplied"])
        self.assertFalse(cost["structure_aware_elliptic_index_supplied"])

    def test_no_computational_lower_bound_or_candidate_oracle(self):
        cost = self.bundle["cost"]
        self.assertIn("neither", self.costs["scope"].lower())
        self.assertIn("lower bound", self.costs["scope"])
        self.assertFalse(
            cost["unconditional_computational_lower_bound_claimed"]
        )
        self.assertFalse(cost["candidate_field_dlp_used"])
        self.assertFalse(cost["candidate_root_oracle_used"])

    def test_gate_admits_only_reduction_and_standard_index_negative(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(admission["weighted_3sum_indexing_reduction_admitted"])
        self.assertTrue(
            admission["published_standard_indexing_negative_admitted"]
        )
        self.assertFalse(
            admission["structure_aware_shared_operator_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
