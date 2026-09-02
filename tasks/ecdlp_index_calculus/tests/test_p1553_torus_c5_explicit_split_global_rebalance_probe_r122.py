from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_torus_c5_explicit_split_global_rebalance_probe_r122.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r122", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R122 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R122)


class TorusC5ExplicitSplitGlobalRebalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R122.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R122.verify_source_bindings()), 16)

    def test_symbolic_split_identity_and_setup_bound(self) -> None:
        rows = (
            R122.split_regime(6, Fraction(1, 12), Fraction(3, 4), 3),
            R122.split_regime(9, Fraction(1, 9), Fraction(4, 9), 5),
            R122.split_regime(20, Fraction(0), Fraction(1, 8), 18),
        )
        for row in rows:
            self.assertTrue(row["supply_identity_exact"])
            self.assertTrue(row["supply_term_at_least_group_order"])
            if row["setup_eligible"]:
                self.assertTrue(row["setup_bound_applies"])
                self.assertTrue(
                    row["strictly_above_rho_if_setup_eligible"]
                )

    def test_all_arity_theorem_has_strict_rho_gap(self) -> None:
        theorem = self.cost["theorem"]
        self.assertEqual(
            theorem["combined_lower_bound"],
            "L>=5+beta-s*beta>=11/4+beta",
        )
        self.assertEqual(
            theorem["rho_gap"],
            "L-5/2>=1/4+beta>1/4",
        )
        self.assertTrue(
            theorem["all_fixed_arities_and_positive_beta_covered"]
        )

    def test_selected_m6_vertex_charges_fresh_and_collection(self) -> None:
        row = self.controls["selected_regime_controls"][0]
        self.assertEqual(row["relation_arity_m"], 6)
        self.assertEqual(row["stored_c_arity_s"], 3)
        self.assertEqual(row["enumerated_c_arity_r"], 2)
        self.assertEqual(
            row["density_adjusted_fresh_exponent_B"]["exact"],
            "11/4",
        )
        self.assertEqual(
            row["relation_collection_exponent_B"]["exact"],
            "7/2",
        )
        self.assertTrue(row["setup_eligible"])

    def test_rational_grid_has_no_counterexample(self) -> None:
        grid = self.controls["grid_audit"]
        self.assertGreater(grid["candidate_count"], 250_000)
        self.assertGreater(
            grid["setup_eligible_candidate_count"],
            90_000,
        )
        self.assertEqual(grid["theorem_violation_count"], 0)
        self.assertTrue(grid["all_setup_eligible_rows_above_rho"])
        self.assertFalse(grid["finite_grid_receives_proof_credit"])

    def test_exhaustive_pair_collision_theorem_is_exact(self) -> None:
        self.assertTrue(
            self.controls["all_exhaustive_pair_collision_controls_exact"]
        )
        rows = self.controls["exhaustive_pair_collision_controls"]
        self.assertEqual([row["source_arity"] for row in rows], [1, 2, 3, 4])
        for row in rows:
            self.assertEqual(
                row["observed_colliding_source_pair_count"],
                row["expected_colliding_source_pair_count"],
            )
            self.assertEqual(
                row["exact_pair_collision_probability"],
                "1/q",
            )

    def test_iid_support_scope_does_not_transfer_to_filtered_decks(
        self,
    ) -> None:
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        iid = routes["iid_random_deck_distinct_endpoint_c_s_table"]
        filtered = routes["collision_compressed_filtered_deck_table"]
        self.assertTrue(iid["whp_occurrence_scale_support"])
        self.assertFalse(filtered["covered_by_iid_theorem"])
        self.assertIn("open", filtered["status"])

    def test_nonoccurrence_torus_circuit_remains_outside_theorem(
        self,
    ) -> None:
        routes = {row["route_id"]: row for row in self.cost["routes"]}
        residual = routes[
            "target_specialized_nonoccurrence_torus_c5_circuit"
        ]
        self.assertFalse(residual["covered_by_theorem"])
        self.assertFalse(residual["exact_circuit_constructed"])
        self.assertFalse(residual["general_lower_bound_claimed"])
        self.assertIn(
            "nonoccurrence torus C5",
            self.frozen["preserved_interface"],
        )

    def test_scoped_rebalance_negative_does_not_promote(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 11)
        self.assertEqual(admission["obligation_count"], 18)
        self.assertTrue(
            admission["explicit_split_rebalance_negative_admitted"]
        )
        self.assertTrue(admission["iid_setup_support_theorem_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-specialized nonoccurrence torus C5",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
