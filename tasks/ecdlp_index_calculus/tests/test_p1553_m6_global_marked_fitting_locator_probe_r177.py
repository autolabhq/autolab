import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_global_marked_fitting_locator_probe_r177.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R177 = load_module("p1553_r177_test", PRODUCER)


class GlobalMarkedFittingLocatorProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R177.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 23)
        self.assertEqual(len(R177.verify_source_bindings()), 23)

    def test_global_marked_norm_is_bound(self) -> None:
        theorem = self.report["theorem"]["marked_global_norm"]
        self.assertIn("F(A,lambda)=det", theorem)
        self.assertIn("h(P+Q)+lambda*(A-x(P))", theorem)
        self.assertEqual(self.report["controls"]["pair_algebra_dimension_sum"], 8922)

    def test_kernel_nullity_equals_signed_incidence_count(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_kernel_dimensions_match_r176_incidences"])
        self.assertEqual(controls["kernel_dimension_sum"], 241)
        for row in controls["controls"]:
            self.assertEqual(
                row["kernel_dimension"], row["r176_principal_pair_incidence_count"]
            )

    def test_lowest_lambda_degree_is_kernel_dimension(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(
            controls["all_lowest_lambda_degrees_equal_kernel_dimensions"]
        )
        self.assertTrue(controls["all_lower_lambda_coefficients_zero"])
        theorem = self.report["theorem"]["lowest_lambda_coefficient"]
        self.assertIn("ord_lambda F=M", theorem)
        self.assertIn("pdet(K)", theorem)

    def test_interpolated_lowest_coefficient_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_interpolated_lowest_coefficients_exact"])
        self.assertEqual(controls["marker_interpolation_sample_count"], 247)
        for row in controls["controls"]:
            self.assertEqual(
                row["expected_lowest_coefficient_sha256"],
                row["interpolated_lowest_coefficient_sha256"],
            )

    def test_restricted_kernel_characteristic_polynomial_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_recovered_marker_polynomials_exact"])
        self.assertEqual(controls["marker_polynomial_degree_sum"], 241)
        self.assertEqual(controls["maximum_candidate_incidence_multiplicity"], 4)
        theorem = self.report["theorem"][
            "restricted_kernel_characteristic_polynomial"
        ]
        self.assertIn("det(A*I-X_1|ker(K))", theorem)
        self.assertIn("incidence multiplicity", theorem)

    def test_candidate_gcd_returns_all_r176_roots(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_candidate_factor_roots_match_r176"])
        self.assertTrue(controls["all_candidate_roots_match_r176"])
        self.assertEqual(controls["candidate_root_count"], 140)
        self.assertEqual(controls["candidate_factor_degree_sum"], 140)
        for row in controls["controls"]:
            self.assertEqual(row["candidate_factor_roots"], row["r176_candidate_roots"])

    def test_candidate_locator_requires_no_subset_queries(self) -> None:
        theorem = self.report["theorem"]["candidate_locator"]
        self.assertIn("gcd(U,monic([lambda^M]F))", theorem)
        self.assertIn("distinct R176 candidate roots", theorem)
        self.assertIn("No subset query", theorem)

    def test_finite_truncated_interpolation_is_fully_charged(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["truncated_lambda_update_count"], 28788282)
        self.assertFalse(
            controls[
                "finite_pair_scan_and_marker_interpolation_receive_asymptotic_credit"
            ]
        )

    def test_standard_pair_algebra_is_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(cost["pair_algebra_dimension_exponent_B"]["exact"], "9/2")
        self.assertEqual(cost["global_pollard_rho_exponent_B"]["exact"], "5/2")
        self.assertFalse(cost["standard_pair_algebra_route_inside_rho"])
        self.assertEqual(
            cost["generic_pair_algebra_element_state_exponent_B"]["exact"], "9/2"
        )

    def test_full_body_and_explicit_truncation_are_above_rho(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["full_bivariate_marked_norm_body_exponent_B"]["exact"], "9"
        )
        self.assertEqual(
            cost["explicit_marker_interpolation_work_exponent_B"]["exact"], "6"
        )
        self.assertFalse(cost["explicit_truncated_marker_route_inside_rho"])

    def test_low_degree_output_state_itself_fits(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["candidate_marker_polynomial_degree_exponent_B"]["exact"],
            "3/4",
        )
        self.assertEqual(
            cost["explicit_symbolic_truncated_marker_state_exponent_B"]["exact"],
            "3/2",
        )

    def test_output_sensitive_fitting_constructor_remains_open(self) -> None:
        cost = self.report["cost"]
        admission = self.report["admission"]
        self.assertEqual(
            cost["conditional_output_sensitive_fitting_total_exponent_B"]["exact"],
            "9/4",
        )
        self.assertTrue(
            cost["conditional_output_sensitive_fitting_strictly_inside_rho"]
        )
        self.assertFalse(cost["output_sensitive_marked_fitting_constructor_supplied"])
        self.assertFalse(
            admission["output_sensitive_marked_fitting_constructor_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])

    def test_standard_negative_is_not_a_circuit_lower_bound(self) -> None:
        cost = self.report["cost"]
        self.assertFalse(
            cost["standard_route_negative_claimed_as_circuit_lower_bound"]
        )
        self.assertIn("not arithmetic-circuit lower bounds", self.report["theorem"]["scope"])

    def test_no_oracle_or_finite_attack_credit_is_consumed(self) -> None:
        controls = self.report["controls"]
        self.assertFalse(controls["candidate_oracle_consumed"])
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])

    def test_no_breakthrough_flags(self) -> None:
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn("OUTPUT_SENSITIVE_MARKED_FITTING_OPEN", self.report["classification"])

    def test_next_action_preserves_the_marked_fitting_contract(self) -> None:
        action = self.report["next_action"]
        self.assertIn("output-sensitive marked Fitting", action)
        self.assertIn("det(AI-X_1|ker K)", action)
        self.assertIn("O(n+N+M)", action)
        self.assertIn("Reject n^2 pair enumeration", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R177.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "marker"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
