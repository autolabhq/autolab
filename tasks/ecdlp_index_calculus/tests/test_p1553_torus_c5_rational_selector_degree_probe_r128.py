from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_torus_c5_rational_selector_degree_probe_r128.py"
SPEC = importlib.util.spec_from_file_location("p1553_r128", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R128 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R128)


class TorusC5RationalSelectorDegreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R128.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R128.verify_source_bindings()), 12)

    def test_newton_interpolation_rejects_bad_domains(self) -> None:
        field, _, _ = R128.R121.pairing_deck(
            R128.R82.FAMILIES[0],
            0,
        )
        with self.assertRaises(ValueError):
            R128.newton_interpolate(
                field,
                (field.one,),
                (),
            )
        with self.assertRaises(ValueError):
            R128.newton_interpolate(
                field,
                (field.one, field.one),
                (field.zero, field.one),
            )

    def test_all_eight_selector_controls_complete(self) -> None:
        self.assertEqual(self.controls["control_count"], 8)
        self.assertTrue(
            self.controls[
                "all_selector_domains_equal_distinct_c5_support"
            ]
        )
        self.assertFalse(self.controls["candidate_discrete_logs_consumed"])

    def test_all_interpolations_are_exact(self) -> None:
        self.assertTrue(self.controls["all_interpolations_exact"])
        for row in self.controls["controls"]:
            self.assertTrue(
                row["interpolation_exact_on_all_positive_targets"]
            )

    def test_actual_interpolants_have_full_domain_degree(self) -> None:
        self.assertTrue(
            self.controls[
                "all_interpolation_polynomials_have_full_degree"
            ]
        )
        for row in self.controls["controls"]:
            self.assertEqual(
                row["unique_interpolation_polynomial_degree"],
                row["selector_domain_size"] - 1,
            )

    def test_selected_c2_c3_sources_replay(self) -> None:
        self.assertTrue(self.controls["all_sources_replay"])
        self.assertTrue(
            self.controls["all_fibers_respect_c3_bound"]
        )
        for row in self.controls["controls"]:
            self.assertLessEqual(
                row["maximum_selected_c2_fiber_size"],
                row["maximum_possible_fixed_c2_fiber_size"],
            )

    def test_rational_selector_degree_counting_bound(self) -> None:
        theorem = self.report["theorem"]
        self.assertEqual(
            theorem["degree_lower_bound"],
            "d>=ceil(|C5|/|C2|)",
        )
        self.assertIn("B^(9/4", theorem["iid_exponent_lower_bound"])
        self.assertIn("degree at most d", theorem["fiber_bound"])

    def test_dense_rational_selector_misses_query_cap(self) -> None:
        route = self.routes["dense_single_rational_c2_selector"]
        self.assertEqual(
            route["minimum_degree_exponent_B"]["exact"],
            "9/4",
        )
        self.assertTrue(route["inside_setup_cap"])
        self.assertFalse(route["inside_polylog_query_cap"])
        table = self.routes["explicit_target_to_c2_selector_table"]
        self.assertEqual(table["state_exponent_B"]["exact"], "15/4")
        self.assertFalse(table["inside_setup_cap"])

    def test_low_slp_and_piecewise_selectors_remain_open(self) -> None:
        low_slp = self.routes["high_degree_low_slp_rational_selector"]
        piecewise = self.routes["compact_piecewise_rational_selector"]
        self.assertTrue(low_slp["degree_lower_bound_applies"])
        self.assertFalse(low_slp["circuit_size_lower_bound_proved"])
        self.assertFalse(low_slp["exact_structure_constructed"])
        self.assertEqual(low_slp["status"], "open")
        self.assertFalse(piecewise["compact_branch_router_constructed"])
        self.assertEqual(piecewise["status"], "open")
        self.assertIn("low-SLP", self.frozen["preserved_interface"])

    def test_scoped_degree_bound_does_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 20)
        self.assertTrue(admission["rational_selector_semantics_admitted"])
        self.assertTrue(
            admission["scoped_rational_degree_negative_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "high-degree low-SLP",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
