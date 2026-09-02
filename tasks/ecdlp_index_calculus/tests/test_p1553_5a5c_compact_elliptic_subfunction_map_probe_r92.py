from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "p1553_5a5c_compact_elliptic_subfunction_map_probe_r92.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r92", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise AssertionError("unable to load R92 probe")
R92 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R92)


class CompactEllipticSubfunctionMapTests(unittest.TestCase):
    def test_toy_curve_is_exact_prime_order_group(self) -> None:
        control = R92.prime_curve_control()
        self.assertEqual(control["enumerated_projective_point_count"], 97)
        self.assertTrue(control["field_prime_exactly_prime"])
        self.assertTrue(control["curve_discriminant_nonzero"])
        self.assertTrue(control["group_order_exactly_prime"])
        self.assertTrue(control["addition_closed_on_full_enumeration"])
        self.assertTrue(control["generator_has_stated_prime_order"])
        self.assertTrue(control["infinity_included"])

    def test_four_rational_fiber_interfaces_return_endpoints(self) -> None:
        controls = {
            control["map_name"]: control
            for control in R92.elliptic_fiber_controls()
        }
        self.assertEqual(
            {
                name: (
                    control["MAP1_image_size_D"],
                    control["maximum_subfunction_domain_L"],
                )
                for name, control in controls.items()
            },
            {
                "x": (49, 2),
                "y": (63, 3),
                "x_plus_y": (71, 3),
                "x_plus_2y": (60, 3),
            },
        )
        for control in controls.values():
            self.assertTrue(control["every_target_has_one_fd_preimage"])
            self.assertTrue(
                control["TR_returns_exact_endpoint_for_every_target"]
            )
            self.assertTrue(control["absent_MAP2_value_rejected"])
            self.assertTrue(control["infinity_bucket_is_explicit"])

    def test_degree_and_coverage_bounds_hold(self) -> None:
        for control in R92.elliptic_fiber_controls():
            self.assertTrue(
                control["maximum_fiber_at_most_rational_degree"]
            )
            self.assertGreaterEqual(
                control["D_times_L"],
                control["projective_domain_size"],
            )

    def test_framework_formula_charges_shared_randomness(self) -> None:
        exponents = R92.framework_exponents(
            Fraction(10, 3),
            Fraction(1),
            charge_shared_randomness=True,
        )
        self.assertEqual(
            exponents["per_subfunction_advice"],
            Fraction(10, 3),
        )
        self.assertEqual(
            exponents["shared_randomness_advice"],
            Fraction(10, 3),
        )
        self.assertEqual(exponents["setup"], Fraction(10, 3))

    def test_optimistic_and_online_optima_are_exact(self) -> None:
        control = R92.subfunction_framework_cost_control()
        self.assertEqual(
            control[
                "all_state_charged_unconstrained_minimum_B"
            ]["exact"],
            "10/3",
        )
        self.assertEqual(
            control["free_randomness_optimistic_minimum_B"]["exact"],
            "5/2",
        )
        self.assertEqual(
            control["online_compatible_minimum_B"]["exact"],
            "35/8",
        )
        online = control["online_compatible_optimum"]
        self.assertEqual(online["fiber_exponent_B"]["exact"], "5/4")
        self.assertEqual(online["query_exponent_B"]["exact"], "5/4")
        self.assertFalse(online["setup_cap_satisfied"])
        self.assertTrue(online["online_cap_satisfied"])

    def test_matched_random_partitions_preserve_counting(self) -> None:
        controls = R92.matched_random_partition_controls()
        self.assertEqual(len(controls), 5)
        for control in controls:
            self.assertTrue(control["coverage_inequality_exact"])
            self.assertTrue(
                control["full_point_MAP2_and_TR_endpoint_exact"]
            )

    def test_source_dictionary_is_exact_but_over_cap(self) -> None:
        control = R92.synthetic_source_dictionary_control()
        self.assertTrue(control["explicit_dictionary_TR_exact"])
        self.assertFalse(
            control["actual_five_a_five_c_source_tuples_used"]
        )
        self.assertEqual(
            control["asymptotic_dictionary_state_exponent_B"]["exact"],
            "5",
        )
        self.assertFalse(control["inside_setup_cap"])
        self.assertFalse(control["candidate_credit"])

    def test_bundle_preserves_overlapping_correspondence(self) -> None:
        report = R92.build_bundle()["report"]
        self.assertEqual(
            report["admission"]["passed_obligation_count"],
            12,
        )
        self.assertEqual(report["admission"]["obligation_count"], 22)
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(report["breakthrough"])
        self.assertIn("not a lower bound", report["scope_boundary"])
        self.assertIn(
            "overlapping S3/S4 incidence correspondence",
            report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
