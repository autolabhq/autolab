from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "p1553_5a5c_theta_addition_cancellation_network_probe_r110.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r110", MODULE_PATH)
assert SPEC and SPEC.loader
R110 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R110)


class ThetaAdditionCancellationNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R110.build_bundle()
        cls.exceptional = cls.bundle["exceptional"]
        cls.instances = [
            *cls.exceptional["actual"],
            *cls.exceptional["matched_random_decks"],
        ]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R110.verify_source_bindings()), 7)

    def test_l11_basis_has_the_expected_pole_orders(self) -> None:
        theorem = R110.alternant_theorem()
        self.assertEqual(theorem["line_bundle"], "L(11O)")
        self.assertEqual(
            theorem["basis_pole_orders"],
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )
        self.assertEqual(len(theorem["basis"]), R110.ALTERNANT_DEGREE)

    def test_zero_sum_shifts_separate_every_position_domain(self) -> None:
        self.assertEqual(len(self.instances), 16)
        self.assertTrue(
            all(row["position_shift_sum_is_identity"] for row in self.instances)
        )
        self.assertTrue(
            all(
                row["translated_domains_pairwise_disjoint"]
                and row["translated_domains_affine"]
                for row in self.instances
            )
        )
        self.assertEqual(
            max(row["shift_multiplier"] for row in self.instances),
            1,
        )
        self.assertTrue(
            all(
                row["shift_multiplier"] <= row["shift_candidate_limit"]
                for row in self.instances
            )
        )

    def test_shifted_alternant_is_exact_on_positive_and_negative_rows(
        self,
    ) -> None:
        self.assertTrue(
            all(
                all(
                    query["direct_relation"]
                    and query["alternant_zero"]
                    and query["alternant_rank"] == 10
                    for query in row["positive_rows"]
                )
                and all(
                    not query["direct_relation"]
                    and not query["alternant_zero"]
                    and query["alternant_rank"] == 11
                    for query in row["negative_rows"]
                )
                for row in self.instances
            )
        )

    def test_both_known_double_fibers_replay(self) -> None:
        doubles = [
            row
            for row in self.exceptional["actual"]
            if row["target_class"] == "r105_actual_double_fiber"
        ]
        self.assertEqual(len(doubles), 2)
        self.assertTrue(all(len(row["positive_rows"]) == 2 for row in doubles))
        self.assertTrue(
            all(
                query["alternant_rank"] == 10
                for row in doubles
                for query in row["positive_rows"]
            )
        )

    def test_unshifted_repeated_rows_are_false_zeros(self) -> None:
        self.assertTrue(
            all(
                row["unshifted_repeat_control"][
                    "false_positive_from_repeated_rows"
                ]
                and not row["unshifted_repeat_control"]["direct_relation"]
                and row["unshifted_repeat_control"][
                    "unshifted_alternant_zero"
                ]
                for row in self.instances
            )
        )

    def test_bundle_admits_predicate_not_locator(self) -> None:
        report = self.bundle["report"]
        ledger = report["cost_ledger"]
        self.assertTrue(
            report["admission"]["target_zero_biconditional_admitted"]
        )
        self.assertFalse(report["admission"]["lane_admitted"])
        self.assertFalse(
            ledger["zero_mask_contraction"][
                "finite_deck_annihilator_contraction_supplied"
            ]
        )
        self.assertEqual(
            ledger["zero_mask_contraction"][
                "global_row_mode_exponent_B"
            ]["exact"],
            "5",
        )
        self.assertEqual(
            ledger["position_separated_alternant"][
                "target_shift_search_exponent_B"
            ]["exact"],
            "6/5",
        )
        self.assertTrue(
            ledger["position_separated_alternant"][
                "both_inside_online_cap"
            ]
        )
        self.assertFalse(report["breakthrough"])
        self.assertFalse(report["shoup_bound_improvement"])
        self.assertIn(
            "finite-deck annihilator",
            report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
