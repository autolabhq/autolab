import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_r172.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R172 = load_module("p1553_r172_test", PRODUCER)


class TargetSignConjugateS3SelfResultantProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R172.build_bundle()
        cls.report = cls.bundle["report"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(self.report["source_bindings"]), 15)
        self.assertEqual(len(R172.verify_source_bindings()), 15)

    def test_compact_target_divisors_are_valid(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["control_count"], 6)
        self.assertTrue(controls["all_target_x_coordinates_distinct"])
        self.assertTrue(controls["all_target_selected_x_supports_disjoint"])
        self.assertTrue(controls["all_compact_target_divisors_exact"])
        self.assertEqual(controls["target_divisor_coefficient_slot_count"], 86)

    def test_homogenized_translation_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["homogenized_translate_identity_count"], 2972)
        self.assertTrue(controls["all_target_conjugate_s3_identities_exact"])
        self.assertEqual(controls["target_conjugate_s3_identity_count"], 1486)

    def test_iterated_resultant_identity_is_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_reverse_resultant_row_identities_exact"])
        self.assertEqual(controls["reverse_resultant_row_identity_count"], 8922)
        self.assertTrue(controls["all_batched_denominators_exact_and_units"])

    def test_reverse_resultant_degree_and_body_are_recorded(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["reverse_resultant_coefficient_slot_count"], 1270)
        self.assertEqual(controls["reverse_resultant_nonzero_coefficient_count"], 1270)
        self.assertEqual(controls["reverse_resultant_coefficient_rank_sum"], 86)
        self.assertTrue(controls["all_reverse_resultants_full_grid_dense"])
        self.assertTrue(
            controls["all_reverse_resultant_coefficient_matrices_full_rank"]
        )
        for row in controls["controls"]:
            reverse = row["reverse_target_resultant"]
            expected_degree = 2 * row["retained_target_count"]
            self.assertEqual(reverse["degree_x"], expected_degree)
            self.assertEqual(reverse["degree_z"], expected_degree)

    def test_plus_branch_replays_r171(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_plus_candidate_roots_match_r171"])
        self.assertEqual(controls["plus_candidate_root_count"], 140)
        for row in controls["controls"]:
            self.assertEqual(row["plus_candidate_roots"], row["r171_candidate_roots"])

    def test_conjugate_branch_union_is_explicit(self) -> None:
        controls = self.report["controls"]
        self.assertEqual(controls["conjugate_candidate_root_count"], 1)
        self.assertEqual(controls["extra_conjugate_candidate_root_count"], 1)
        self.assertEqual(controls["overlap_candidate_root_count"], 0)
        self.assertEqual(controls["symmetric_candidate_root_count"], 141)
        for row in controls["controls"]:
            self.assertEqual(
                row["symmetric_candidate_roots"],
                sorted(
                    set(row["plus_candidate_roots"])
                    | set(row["conjugate_candidate_roots"])
                ),
            )

    def test_costs_charge_represented_and_local_routes(self) -> None:
        cost = self.report["cost"]
        self.assertEqual(
            cost["factored_reverse_s3_resultant_state_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            cost["batched_denominator_work_exponent_B"]["exact"], "9/4"
        )
        self.assertEqual(
            cost["represented_reverse_resultant_body_exponent_B"]["exact"],
            "5/2",
        )
        self.assertEqual(
            cost["standard_pair_or_factor_local_work_exponent_B"]["exact"],
            "7/2",
        )
        self.assertFalse(cost["represented_reverse_resultant_strictly_inside_rho"])
        self.assertFalse(cost["standard_pair_or_factor_local_work_inside_rho"])

    def test_density_and_candidate_counts_receive_no_asymptotic_credit(self) -> None:
        controls = self.report["controls"]
        self.assertFalse(
            controls["finite_controls_receive_asymptotic_attack_or_lower_bound_credit"]
        )
        self.assertFalse(self.report["finite_controls_receive_asymptotic_credit"])
        self.assertFalse(
            self.report["cost"]["finite_reverse_density_receives_lower_bound_credit"]
        )
        self.assertFalse(
            self.report["cost"]["finite_conjugate_candidate_count_receives_attack_credit"]
        )

    def test_literature_contracts_do_not_supply_missing_operator(self) -> None:
        literature = self.report["literature"]
        self.assertIn(
            "does not accept",
            literature["moroz_schost_truncated_resultant"]["fit"],
        )
        self.assertIn(
            "does not avoid",
            literature["hyun_neiger_schost_bivariate_resultants"]["fit"],
        )
        self.assertIn(
            "does not supply",
            literature["semaev_summation_polynomials"]["fit"],
        )

    def test_surviving_interface_is_explicitly_unproved(self) -> None:
        interface = self.report["theorem"]["surviving_interface"]
        self.assertIn("unproved interface", interface)
        self.assertIn("softly O(n+N)", interface)
        self.assertFalse(
            self.report["cost"]["factored_self_s3_resultant_mod_u_supplied"]
        )
        self.assertFalse(
            self.report["admission"]["factored_self_s3_resultant_mod_u_admitted"]
        )

    def test_no_oracle_is_consumed(self) -> None:
        self.assertFalse(self.report["candidate_discrete_log_oracle_consumed"])
        self.assertFalse(self.report["controls"]["candidate_oracle_consumed"])
        for row in self.report["controls"]["controls"]:
            self.assertFalse(row["candidate_discrete_log_oracle_consumed"])
            self.assertFalse(
                row[
                    "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed"
                ]
            )

    def test_no_breakthrough_flags(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(admission["target_sign_conjugate_s3_identity_admitted"])
        self.assertTrue(admission["single_denominator_batch_admitted"])
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(self.report["pollard_rho_improvement"])
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])

    def test_next_action_rejects_hidden_pair_or_oracle_cost(self) -> None:
        action = self.report["next_action"]
        self.assertIn("N^2 reverse-resultant coefficient body", action)
        self.assertIn("nN point/factor grid", action)
        self.assertIn("unit-cost resultant", action)

    def test_replay_is_deterministic(self) -> None:
        self.assertEqual(self.bundle, R172.build_bundle())

    def test_all_output_schemas_are_bound(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"report", "frozen", "cost", "replay", "controls", "resultant"},
        )
        for value in self.bundle.values():
            self.assertIn("schema", value)


if __name__ == "__main__":
    unittest.main()
