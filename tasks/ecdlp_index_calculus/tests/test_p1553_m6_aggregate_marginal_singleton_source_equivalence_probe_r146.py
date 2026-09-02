import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_m6_aggregate_marginal_singleton_source_equivalence_probe_r146.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r146_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PROBE_PATH}")
R146 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R146)


class M6AggregateMarginalSingletonSourceEquivalenceTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R146.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R146.verify_source_bindings()), 13)

    def test_all_actual_controls_are_present(self):
        self.assertEqual(self.controls["actual_control_count"], 8)

    def test_positive_support_and_r144_samples_replay(self):
        self.assertTrue(
            self.controls["all_positive_fiber_counts_match_r144"]
        )
        self.assertTrue(
            self.controls[
                "all_r144_sample_counts_and_marginals_match"
            ]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(control["positive_fiber_count_matches_r144"])
            self.assertTrue(
                control["all_r144_sample_counts_and_marginals_match"]
            )

    def test_every_singleton_marginal_inverts_exactly(self):
        self.assertTrue(
            self.controls["all_singleton_marginals_invert_exactly"]
        )
        for control in self.controls["actual_controls"]:
            self.assertTrue(
                control[
                    "all_singleton_marginals_invert_to_atom_multiplicities"
                ]
            )

    def test_finite_singleton_fractions_are_frozen(self):
        fractions = sorted(
            control["singleton_positive_fiber_fraction"]
            for control in self.controls["actual_controls"]
        )
        self.assertAlmostEqual(fractions[0], 54 / 55)
        self.assertEqual(fractions[-1], 1.0)
        self.assertGreaterEqual(
            self.controls["minimum_singleton_positive_fiber_fraction"],
            0.98,
        )

    def test_finite_occupancy_histograms_conserve_sources(self):
        for control in self.controls["actual_controls"]:
            total = sum(
                int(occupancy) * fiber_count
                for occupancy, fiber_count in control[
                    "canonical_ac_occupancy_histogram"
                ].items()
            )
            self.assertEqual(
                total,
                control["canonical_ac_source_count"],
            )

    def test_random_occupancy_formula_is_exactly_scoped(self):
        theorem = R146.occupancy_theorem()
        self.assertEqual(theorem["lambda_exact"], "1/518400")
        expected = 1 / math.factorial(6) ** 2
        self.assertAlmostEqual(theorem["lambda_decimal"], expected)
        self.assertGreater(
            theorem["conditional_singleton_probability_decimal"],
            0.999999,
        )
        self.assertTrue(theorem["random_model_only"])
        self.assertFalse(
            theorem["structured_factor_base_transfer_supplied"]
        )

    def test_occupancy_conservation_charges_density_retry(self):
        conservation = self.report["theorem"]["occupancy_conservation"]
        self.assertEqual(
            conservation["mean_positive_occupancy"],
            "L=M/H",
        )
        self.assertIn("B^-gamma", conservation["boundary"])
        self.assertIn(
            "targetable structured family",
            conservation["boundary"],
        )

    def test_no_computational_lower_bound_is_claimed(self):
        scope = self.report["theorem"]["scope"]
        self.assertIn("Neither statement is a computational lower bound", scope)
        self.assertIn("random-model-only", scope)
        self.assertFalse(
            self.bundle["cost"][
                "unconditional_computational_lower_bound_claimed"
            ]
        )

    def test_count_index_logs_and_descent_remain_open(self):
        cost = self.bundle["cost"]
        logs = self.bundle["logs"]
        self.assertFalse(
            cost["implicit_count_and_marginal_index_supplied"]
        )
        self.assertFalse(
            cost["targetable_structured_multi_fiber_family_supplied"]
        )
        self.assertFalse(logs["candidate_factor_logs_computed"])
        self.assertFalse(
            logs["candidate_identical_target_descent_computed"]
        )
        self.assertFalse(logs["generic_prime_family_transfer_supplied"])

    def test_gate_admits_only_source_equivalence_boundary(self):
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 15)
        self.assertEqual(admission["obligation_count"], 25)
        self.assertTrue(
            admission["singleton_source_equivalence_admitted"]
        )
        self.assertTrue(admission["occupancy_conservation_admitted"])
        self.assertTrue(
            admission[
                "random_model_occupancy_admitted_model_bound_only"
            ]
        )
        self.assertFalse(admission["implicit_count_operator_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
