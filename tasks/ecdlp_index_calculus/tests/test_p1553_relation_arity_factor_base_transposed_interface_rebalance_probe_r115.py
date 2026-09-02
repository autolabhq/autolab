from __future__ import annotations

import importlib.util
import pathlib
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[3]
PRODUCER = (
    ROOT
    / "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_r115.py"
)
SPEC = importlib.util.spec_from_file_location("p1553_r115", PRODUCER)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"unable to import {PRODUCER}")
R115 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R115)


class RelationArityFactorBaseRebalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = R115.build_bundle()
        cls.report = cls.bundle["report"]
        cls.ledger = cls.bundle["ledger"]
        cls.vertex = cls.report["selected_vertex"]

    def test_source_bindings_are_exact(self) -> None:
        self.assertEqual(len(R115.verify_source_bindings()), 12)

    def test_semaev_degree_formula_is_exact(self) -> None:
        self.assertEqual(
            R115.semaev_degree(7),
            {
                "arity": 7,
                "degree_per_variable": 32,
                "total_degree": 192,
            },
        )
        self.assertEqual(
            R115.semaev_degree(13),
            {
                "arity": 13,
                "degree_per_variable": 2048,
                "total_degree": 24576,
            },
        )

    def test_minimal_preserved_compiler_arity_is_six(self) -> None:
        proof = self.ledger["minimal_arity_proof"]
        self.assertEqual(proof["minimal_passing_arity"], 6)
        rows = {
            row["relation_arity_m"]: row
            for row in self.ledger["boundary_family"]
        }
        self.assertFalse(rows[3]["positive_boundary_exists"])
        self.assertFalse(rows[4]["positive_boundary_exists"])
        self.assertFalse(rows[5]["necessary_exponent_envelope_pass"])
        self.assertTrue(rows[6]["necessary_exponent_envelope_pass"])

    def test_selected_vertex_costs_are_exact(self) -> None:
        expected = {
            "alpha_A_exponent_B": "1/12",
            "beta_C_exponent_B": "3/4",
            "factor_base_exponent_B": "5/6",
            "full_source_body_exponent_B": "5",
            "meaningful_log_rank_exponent_B": "3/4",
            "first_transposed_interface_exponent_B": "5/4",
            "r82_local_a3_query_exponent_B": "1/4",
            "r82_local_c3_state_exponent_B": "9/4",
            "conditional_relation_collection_exponent_B": "2",
            "conditional_sparse_linear_algebra_exponent_B": "3/2",
        }
        self.assertEqual(self.vertex["relation_arity_m"], 6)
        for key, value in expected.items():
            self.assertEqual(self.vertex[key]["exact"], value)
        self.assertTrue(self.vertex["necessary_exponent_envelope_pass"])

    def test_direct_split_remains_rho_and_prefixes_overflow(self) -> None:
        split = self.vertex["best_explicit_split"]
        self.assertEqual(split["left_a_count"], 3)
        self.assertEqual(split["left_c_count"], 3)
        self.assertEqual(split["work_exponent_B"]["exact"], "5/2")
        self.assertFalse(split["below_rho"])
        self.assertEqual(self.vertex["first_online_overflow_depth"], 2)
        self.assertEqual(self.vertex["first_setup_overflow_depth"], 3)

    def test_density_retry_control_restores_over_cap_work(self) -> None:
        sparse = next(
            row
            for row in self.ledger["controls"]
            if row["regime_id"] == "m6_sparse_source_retry_control"
        )
        self.assertEqual(sparse["full_source_body_exponent_B"]["exact"], "9/2")
        self.assertEqual(sparse["density_retry_exponent_B"]["exact"], "1/2")
        self.assertEqual(
            sparse["density_adjusted_fresh_work_exponent_B"]["exact"],
            "5/3",
        )
        self.assertFalse(sparse["necessary_exponent_envelope_pass"])

    def test_m5_online_and_local_compiler_caps_are_incompatible(self) -> None:
        online = R115.evaluate_regime(
            5,
            Fraction(1, 16),
            Fraction(15, 16),
            regime_id="test_m5_online",
        )
        local = R115.evaluate_regime(
            5,
            Fraction(1, 4),
            Fraction(3, 4),
            regime_id="test_m5_local",
        )
        self.assertEqual(
            online["first_transposed_interface_exponent_B"]["exact"],
            "5/4",
        )
        self.assertEqual(
            online["r82_local_c3_state_exponent_B"]["exact"],
            "45/16",
        )
        self.assertEqual(
            local["r82_local_c3_state_exponent_B"]["exact"],
            "9/4",
        )
        self.assertEqual(
            local["first_transposed_interface_exponent_B"]["exact"],
            "2",
        )

    def test_conditional_envelope_never_promotes_algorithm(self) -> None:
        admission = self.report["admission"]
        self.assertTrue(
            admission["necessary_exponent_envelope_admitted"]
        )
        self.assertFalse(admission["lane_admitted"])
        self.assertFalse(
            self.vertex["implicit_source_locator_constructed"]
        )
        self.assertFalse(self.report["shoup_bound_improvement"])
        self.assertFalse(self.report["breakthrough"])
        self.assertIn(
            "implicit 3F self-convolution/source locator",
            self.report["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
