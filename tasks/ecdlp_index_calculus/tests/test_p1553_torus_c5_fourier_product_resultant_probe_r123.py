from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_torus_c5_fourier_product_resultant_probe_r123.py"
SPEC = importlib.util.spec_from_file_location("p1553_r123", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R123 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R123)


class TorusC5FourierProductResultantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R123.build_bundle()
        cls.report = cls.bundle["report"]
        cls.frozen = cls.bundle["frozen"]
        cls.cost = cls.bundle["cost"]
        cls.replay = cls.bundle["replay"]
        cls.controls = cls.bundle["controls"]
        cls.routes = {
            row["route_id"]: row for row in cls.cost["routes"]
        }

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R123.verify_source_bindings()), 19)

    def test_multiplicative_fourier_identity_uses_no_dlp(self) -> None:
        identity = self.cost["fourier_identity"]
        self.assertEqual(
            identity["characters_computable_without_dlp"],
            "chi_j(z)=z^j",
        )
        self.assertFalse(
            self.controls[
                "candidate_scalar_labels_or_discrete_logs_consumed"
            ]
        )

    def test_all_r82_ordered_moments_match_weighted_support(self) -> None:
        self.assertEqual(self.controls["r82_control_count"], 8)
        self.assertTrue(
            self.controls[
                "all_r82_ordered_fourier_moment_identities_exact"
            ]
        )
        for row in self.controls["r82_ordered_fourier_controls"]:
            self.assertEqual(
                row["ordered_moment_sha256"],
                row["weighted_support_moment_sha256"],
            )
            self.assertTrue(
                row["all_ordered_multiplicity_weights_nonzero"]
            )

    def test_ordered_fourier_sequence_has_full_support_order(self) -> None:
        self.assertTrue(
            self.controls["all_r82_bm_orders_equal_distinct_support"]
        )
        for row in self.controls["r82_ordered_fourier_controls"]:
            self.assertEqual(
                row["berlekamp_massey_order"],
                row["distinct_product_count"],
            )
            self.assertTrue(
                row["recurrence_equals_support_annihilator"]
            )

    def test_synthetic_full_fourier_inversion_is_exact(self) -> None:
        row = self.controls[
            "synthetic_full_fourier_and_resultant_control"
        ]
        self.assertTrue(
            row["field_characteristic_exceeds_total_ordered_source_count"]
        )
        self.assertTrue(
            row["full_q_mode_fourier_counts_equal_direct_counts"]
        )
        self.assertTrue(row["fourier_nonzero_iff_membership"])
        self.assertGreater(row["positive_target_count"], 0)
        self.assertGreater(row["empty_target_count"], 0)

    def test_synthetic_p2_p3_resultant_is_exact(self) -> None:
        row = self.controls[
            "synthetic_full_fourier_and_resultant_control"
        ]
        self.assertEqual(row["p2_root_count_with_multiplicity"], 4)
        self.assertEqual(row["p3_root_count_with_multiplicity"], 8)
        self.assertTrue(
            row[
                "p2_p3_product_resultant_equals_ordered_p5_evaluation"
            ]
        )
        self.assertTrue(row["resultant_zero_iff_membership"])

    def test_standard_fourier_routes_miss_caps(self) -> None:
        full = self.routes["full_multiplicative_fourier_inversion"]
        sparse = self.routes["sparse_fourier_prony_bm_reconstruction"]
        self.assertEqual(full["mode_count_exponent_B"]["exact"], "5")
        self.assertFalse(full["inside_setup_or_query_cap"])
        self.assertEqual(
            sparse["linear_complexity_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(sparse["inside_setup_cap"])
        self.assertIn("Prony", sparse["scope"])

    def test_standard_product_resultants_miss_query_or_setup(self) -> None:
        p3 = self.routes["ordered_p3_product_polynomial"]
        target = self.routes["target_scaled_p2_p3_fast_resultant"]
        p5 = self.routes["symbolic_ordered_p5_product_resultant"]
        self.assertEqual(
            p3["degree_and_coefficient_exponent_B"]["exact"],
            "9/4",
        )
        self.assertTrue(p3["inside_setup_cap"])
        self.assertEqual(
            target["optimistic_query_exponent_B"]["exact"],
            "9/4",
        )
        self.assertFalse(target["inside_polylog_query_cap"])
        self.assertEqual(
            p5["degree_and_output_exponent_B"]["exact"],
            "15/4",
        )
        self.assertFalse(p5["inside_setup_cap"])

    def test_nonrepresented_circuit_remains_open_without_lower_bound(
        self,
    ) -> None:
        residual = self.routes[
            "target_specialized_nonrepresented_fourier_resultant_"
            "torus_circuit"
        ]
        self.assertFalse(residual["covered_by_scoped_costs"])
        self.assertFalse(residual["exact_circuit_constructed"])
        self.assertFalse(residual["general_lower_bound_claimed"])
        self.assertIn(
            "nonrepresented Fourier/resultant torus circuit",
            self.frozen["preserved_interface"],
        )

    def test_scoped_semantics_do_not_promote_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertEqual(admission["passed_obligation_count"], 12)
        self.assertEqual(admission["obligation_count"], 19)
        self.assertTrue(
            admission["fourier_and_product_resultant_semantics_admitted"]
        )
        self.assertTrue(
            admission["scoped_representation_negatives_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "target-specialized nonrepresented torus C5",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
