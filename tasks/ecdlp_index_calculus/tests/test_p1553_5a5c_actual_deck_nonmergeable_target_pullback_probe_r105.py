from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_r105.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r105_test", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load R105 probe")
R105 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R105)


class ActualDeckNonmergeableTargetPullbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R105.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.report["actual_controls"]
        cls.theorem = cls.report["marker_adjoint_theorem"]

    def test_actual_norm_jets_recover_counts_and_sources(self) -> None:
        self.assertEqual(self.controls["instance_count"], 8)
        self.assertEqual(self.controls["query_control_count"], 32)
        self.assertTrue(self.controls["all_query_counts_and_sources_exact"])
        self.assertTrue(self.controls["all_unique_positive_sources_exact"])
        self.assertTrue(self.controls["all_blind_queries_bottom"])
        self.assertTrue(self.controls["all_projective_identity_queries_exact"])

    def test_actual_double_fibers_are_factored(self) -> None:
        self.assertTrue(self.controls["all_actual_fibers_at_most_two"])
        self.assertEqual(
            self.controls["multiplicity_two_instance_count"],
            2,
        )
        repeated = [
            query
            for instance in self.controls["instances"]
            if instance["maximum_target_multiplicity"] == 2
            for query in instance["query_controls"]
            if query["label"] == "maximum_multiplicity"
        ]
        self.assertEqual(len(repeated), 2)
        self.assertTrue(
            all(query["lowest_nonzero_jet_order"] == 2 for query in repeated)
        )
        self.assertTrue(
            all(query["all_marker_factors_recovered"] for query in repeated)
        )

    def test_power_sum_markers_invert_multisets(self) -> None:
        for size in (2, 3, 4, 5, 6, 7):
            source = tuple(sorted((0, 0, size - 1, size - 1, size // 2)))
            powers = tuple(
                sum(pow(index, degree, 100_683_137) for index in source)
                % 100_683_137
                for degree in range(1, 6)
            )
            self.assertEqual(
                R105.multiset_from_power_sums(
                    powers,
                    size,
                    100_683_137,
                ),
                source,
            )

    def test_quadratic_marker_factor_alignment(self) -> None:
        prime = 100_683_137
        markers = (
            tuple(range(1, 11)),
            tuple(range(11, 21)),
        )
        nonsquare = R105.R84.least_nonsquare(prime)
        jet = R105.normalized_marker_jet(
            markers,
            (7, 9),
            prime,
            nonsquare,
        )
        self.assertEqual(
            R105.marker_factors_from_jet(jet, prime),
            sorted(markers),
        )

    def test_marker_adjoint_cost_is_below_online_cap(self) -> None:
        self.assertEqual(
            self.theorem["source_recovery_exponent_B"]["exact"],
            "3/5",
        )
        self.assertTrue(self.theorem["source_recovery_inside_online_cap"])
        self.assertTrue(self.theorem["finite_actual_jet_overhead_constant"])
        self.assertEqual(
            self.theorem["marker_channels"]["channel_count"],
            10,
        )

    def test_bundle_admits_adjoint_not_scalar_constructor(self) -> None:
        admission = self.report["admission"]
        self.assertFalse(admission["lane_admitted"])
        self.assertTrue(self.report["conditional_source_adjoint_admitted"])
        self.assertFalse(self.report["scalar_target_norm_constructor_complete"])
        self.assertFalse(self.report["generic_multiplicity_bound_complete"])
        self.assertFalse(self.report["factor_log_solve_complete"])
        self.assertFalse(self.report["fresh_target_descent_complete"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])


if __name__ == "__main__":
    unittest.main()
