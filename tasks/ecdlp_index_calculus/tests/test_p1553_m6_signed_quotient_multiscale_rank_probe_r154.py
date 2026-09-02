from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "p1553_m6_signed_quotient_multiscale_rank_probe_r154.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("p1553_r154_test", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R154 = load_module()


class M6SignedQuotientMultiscaleRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = R154.build_bundle()
        cls.report = cls.bundle["report"]
        cls.controls = cls.bundle["controls"]

    def test_source_bindings_are_exact(self):
        self.assertEqual(len(R154.verify_source_bindings()), 10)

    def test_signed_coefficient_support_counts_are_exact(self):
        self.assertEqual(
            R154.signed_coefficient_vector_count(2, 6),
            49,
        )
        self.assertEqual(
            [
                R154.signed_coefficient_vector_count(pair_count, 6)
                for pair_count in R154.SYNTHETIC_LEVELS
            ],
            [833, 2471, 6321, 14407],
        )

    def test_actual_signed_quotient_ranks_are_frozen(self):
        self.assertEqual(
            self.controls["actual_signed_quotient_ranks"],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(
            self.controls["actual_full_rank_control_count"],
            0,
        )

    def test_public_sign_constraints_and_rank_formula_are_exact(self):
        for row in self.controls["actual_controls"]:
            self.assertEqual(
                row["public_sign_constraint_rank"],
                row["signed_log_dimension"],
            )
            self.assertTrue(row["combined_rank_formula_exact"])
            self.assertEqual(
                row["combined_full_system_rank"],
                row["public_sign_constraint_rank"]
                + row["signed_quotient_rank"],
            )

    def test_actual_full_ranks_match_r153(self):
        for row in self.controls["actual_controls"]:
            self.assertTrue(
                row["inherited_rank_matches_recomputed_full_rank"]
            )
            self.assertTrue(row["all_signed_relation_identities_exact"])

    def test_synthetic_design_is_preregistered(self):
        design = self.bundle["frozen"]["synthetic_design"]
        self.assertEqual(design["a_pair_count"], 2)
        self.assertEqual(design["c_pair_counts"], [4, 5, 6, 7])
        self.assertEqual(
            design["occupancy_multipliers"],
            [1, 2, 4, 8],
        )
        self.assertEqual(design["seeds"], [15401, 15402, 15403])
        self.assertEqual(design["arity"], 6)

    def test_synthetic_moduli_are_selected_before_labels(self):
        for row in self.controls["synthetic_controls"]:
            expected = R154.next_prime(
                math.ceil(
                    row["max_a6_signed_coefficient_support"]
                    * row["max_c6_signed_coefficient_support"]
                    / row["preregistered_occupancy_multiplier"]
                )
            )
            self.assertEqual(row["subgroup_order"], expected)

    def test_all_forty_eight_synthetic_systems_are_exact(self):
        synthetic = self.controls["synthetic_controls"]
        self.assertEqual(len(synthetic), 48)
        for row in synthetic:
            self.assertTrue(row["all_signed_relation_identities_exact"])
            self.assertTrue(row["combined_rank_formula_exact"])

    def test_finite_full_rank_transition_is_frozen(self):
        self.assertEqual(
            self.controls["synthetic_full_rank_control_count"],
            11,
        )
        pattern = [
            row["full_rank_trial_count"]
            for row in self.controls["synthetic_summary"]
        ]
        self.assertEqual(
            pattern,
            [0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2],
        )

    def test_full_rank_synthetic_solves_are_exact(self):
        full_rows = [
            row
            for row in self.controls["synthetic_controls"]
            if row["signed_quotient_full_rank"]
        ]
        self.assertEqual(len(full_rows), 11)
        self.assertTrue(
            all(row["signed_verifier_logs_recovered"] for row in full_rows)
        )

    def test_finite_controls_receive_no_attack_credit(self):
        self.assertFalse(
            self.report["finite_controls_receive_asymptotic_credit"]
        )
        self.assertFalse(
            self.report["admission"][
                "random_rank_or_hash_to_curve_transfer_admitted"
            ]
        )
        self.assertFalse(self.report["admission"]["lane_admitted"])
        self.assertFalse(self.report["breakthrough"])

    def test_open_algorithmic_obligations_remain_explicit(self):
        required = self.bundle["frozen"]["required_open_outputs"]
        self.assertEqual(
            required["random_rank_concentration_theorem"],
            "open",
        )
        self.assertEqual(
            required["reverse_only_signed_marker_operator"],
            "open",
        )
        self.assertEqual(
            required["factor_logs_without_verifier_labels"],
            "open",
        )
        self.assertEqual(required["shoup_bound_improvement"], "open")


if __name__ == "__main__":
    unittest.main()
