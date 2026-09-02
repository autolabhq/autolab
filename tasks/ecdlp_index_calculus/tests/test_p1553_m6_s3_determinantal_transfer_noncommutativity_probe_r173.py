import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_s3_determinantal_transfer_noncommutativity_probe_r173.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R173 = load_module("p1553_r173_test", PRODUCER)


class S3DeterminantalTransferNoncommutativityProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R173.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 12)
        self.assertEqual(len(R173.verify_source_bindings()), 12)

    def test_elementary_symmetric_determinant_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(
            controls["all_elementary_symmetric_determinant_identities_exact"]
        )
        self.assertEqual(controls["determinant_identity_count"], 68326)
        self.assertIn("(s2-a)^2", self.report["theorem"]["elementary_symmetric_s3"])

    def test_separable_discriminant_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_s3_z_discriminant_identities_exact"])
        self.assertEqual(controls["discriminant_identity_count"], 1486)
        self.assertIn("4 V(X) V_T(u)", self.report["theorem"]["separable_discriminant"])

    def test_pencil_order_identity_exposes_noncommutativity(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["exceptional_commutator_evaluation_count"], 0)
        self.assertEqual(controls["pencil_pair_order_identity_count"], 1675890)
        self.assertEqual(
            controls["pencil_pair_noncommuting_count"],
            controls["pencil_pair_order_identity_count"],
        )
        self.assertIn(
            "delta=4(b-Xu(X+u))", self.report["theorem"]["commutator"]
        )

    def test_target_factor_products_are_order_sensitive(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_target_factor_pairs_noncommuting"])
        self.assertEqual(controls["target_factor_pair_count"], 118)
        self.assertEqual(controls["target_factor_noncommuting_pair_count"], 118)
        self.assertTrue(controls["all_forward_reverse_target_products_differ"])
        for row in controls["controls"]:
            self.assertEqual(
                row["forward_reverse_target_product_changed_entry_count"], 2
            )

    def test_transfer_determinants_replay_r172(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(
            controls["all_transfer_determinants_equal_r172_reverse_resultants"]
        )
        for row in controls["controls"]:
            self.assertTrue(
                row[
                    "forward_transfer_determinant_equals_r172_reverse_resultant"
                ]
            )
            self.assertTrue(
                row[
                    "reverse_transfer_determinant_equals_r172_reverse_resultant"
                ]
            )

    def test_represented_matrix_body_is_fully_charged(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(
            controls["represented_transfer_coefficient_slot_count"], 1448
        )
        self.assertEqual(
            controls["represented_transfer_nonzero_coefficient_count"], 1418
        )
        self.assertEqual(controls["represented_transfer_coefficient_rank_sum"], 184)
        for row in controls["controls"]:
            target_count = row["retained_target_count"]
            self.assertEqual(
                row["forward_transfer"]["coefficient_slot_count"],
                4 * (target_count + 1) ** 2,
            )

    def test_costs_reject_naive_transfer_without_general_lower_bound(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["factored_pencil_input_state_exponent_B"]["exact"], "5/4"
        )
        self.assertEqual(
            cost["represented_ordered_matrix_transfer_body_exponent_B"]["exact"],
            "5/2",
        )
        self.assertEqual(
            cost["standard_outer_root_or_factor_grid_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(cost["represented_ordered_transfer_strictly_inside_rho"])
        self.assertFalse(cost["custom_implicit_resultant_lower_bound_claimed"])
        self.assertIn("not a resultant lower bound", self.report["theorem"]["scope"])

    def test_diagonalization_returns_r172_sign_split(self) -> None:
        theorem = self.report["theorem"]
        self.assertIn("two target-sign branches", theorem["separable_discriminant"])
        self.assertFalse(
            self.report["cost"][
                "commutative_diagonalization_beyond_r172_sign_split_supplied"
            ]
        )

    def test_no_oracle_or_asymptotic_credit_is_consumed(self) -> None:
        controls = self.report["controls"]
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(controls["candidate_oracle_consumed"])
        self.assertFalse(
            controls["finite_noncommutativity_receives_general_lower_bound_credit"]
        )
        self.assertFalse(controls["finite_transfer_density_receives_asymptotic_credit"])

    def test_no_breakthrough_flags(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["determinantal_identity_admitted"])
        self.assertFalse(
            admission["naive_order_independent_two_by_two_transfer_admitted"]
        )
        self.assertFalse(admission["factored_self_s3_resultant_mod_u_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_next_action_preserves_implicit_squarefree_route(self) -> None:
        action = self.report["next_action"]
        self.assertIn("4*V(X)*V_T(u)", action)
        self.assertIn("arbitrary squarefree U", action)
        self.assertIn("strictly below B^(5/2)", action)
        self.assertIn("target-dependent setup", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R173.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "transfer"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
