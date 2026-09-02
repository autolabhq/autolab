#!/usr/bin/env python3
"""Add focus, routing, and generalization diagnostics to a P1436 probe.

This is a post-processing harness.  It never changes the preregistered P1436
collector or promotes a result by itself.  It turns the collector's existing
configuration matrix into a staged collision-to-descent report, uses the
non-baseline configurations as diagnostic routing interventions, and requires
an independently bound audit before forwarding any promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import copy
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Iterable


SCHEMA = "ecdlp.p1436_autoresearch_focus_report.v119"
SUMMATION_FFE_EVIDENCE_INVENTORY_SCHEMA = "ecdlp.p1436_summation_ffe_evidence_inventory.v3"
SUMMATION_FFE_REPLAY_PLAN_SCHEMA = "ecdlp.p1436_summation_ffe_replay_plan.v3"
SUMMATION_PAYLOAD_FIELDS = (
    "summation_polynomial",
    "summation_polynomial_artifact",
    "summation_polynomial_artifact_path",
    "summation_polynomial_payload",
    "summation_polynomial_payload_path",
    "summation_poly_payload",
    "summation_poly_payload_path",
    "summation_poly_source",
)
FFE_PAYLOAD_FIELDS = (
    "ffe_factorization_profile",
    "ffe_payload",
    "ffe_payload_path",
    "ffe_profile_payload",
    "ffe_profile_payload_path",
    "ffe_system_payload",
    "ffe_system_profile_payload",
)
SUMMATION_FFE_DISCOVERY_CONTRACT_FIELDS = (
    "summation_ffe_new_factor_row_discovery",
    "summation_new_factor_row_discovery",
    "ffe_new_factor_row_discovery",
)
SUMMATION_FFE_DISCOVERY_REQUIRED_FIELDS = (
    "source_enumerator_id",
    "scalar_blind",
    "new_factor_row_count",
    "independent_new_factor_row_count",
    "measured_source_operations",
    "direct_pair_complement_operations",
    "replay_artifact_sha256",
)
TWEET_POST_URL = "https://x.com/askalphaxiv/status/2076737985559822734"
TWEET_POST_QUERY = "?s=46"
DEFAULT_NOTE_URL = f"{TWEET_POST_URL}{TWEET_POST_QUERY}"
KNOWN_TWEET_SOURCE_TEXT = {
    "askalphaxiv:2076737985559822734": (
        "Introducing autoresearch with GPT 5.6\n\n"
        "We had GPT 5.6 Sol reproduce the key findings from \u201cTowards Mechanistically Understanding "
        "Why Memorized Knowledge Fails to Generalize in LLM Finetuning\u201d\n\n"
        "Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, "
        "critical experiments and spent less time on peripheral details. It also asked fewer "
        "\u201cclarification\u201d questions and independently resolved ambiguities instead of "
        "pushing them back to us\n\n"
        "@OpenAI pushing the boundaries of the automated research loop with models that don\u2019t "
        "have handcuffs"
    ),
    "2076737985559822734": (
        "Introducing autoresearch with GPT 5.6\n\n"
        "We had GPT 5.6 Sol reproduce the key findings from \u201cTowards Mechanistically Understanding "
        "Why Memorized Knowledge Fails to Generalize in LLM Finetuning\u201d\n\n"
        "Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, "
        "critical experiments and spent less time on peripheral details. It also asked fewer "
        "\u201cclarification\u201d questions and independently resolved ambiguities instead of "
        "pushing them back to us\n\n"
        "@OpenAI pushing the boundaries of the automated research loop with models that don\u2019t "
        "have handcuffs"
    ),
}
KNOWN_TWEET_SOURCE_META = {
    "askalphaxiv:2076737985559822734": {
        "tweet_published_at": "2026-07-13T18:38:14.340000+00:00",
        "referenced_paper_title_in_post": (
            "Towards Mechanistically Understanding Why Memorized Knowledge Fails to "
            "Generalize in LLM Finetuning"
        ),
        "paper_title": (
            "Towards Mechanistically Understanding Why Memorized Knowledge Fails to "
            "Generalize in Large Language Model Finetuning"
        ),
        "paper_url": "https://arxiv.org/abs/2607.08393",
        "media_urls": [
            "https://x.com/askalphaxiv/status/2076737985559822734/photo/1"
        ],
        "media_types": ["video"],
        "hashtags": [],
    },
    "2076737985559822734": {
        "tweet_published_at": "2026-07-13T18:38:14.340000+00:00",
        "referenced_paper_title_in_post": (
            "Towards Mechanistically Understanding Why Memorized Knowledge Fails to "
            "Generalize in LLM Finetuning"
        ),
        "paper_title": (
            "Towards Mechanistically Understanding Why Memorized Knowledge Fails to "
            "Generalize in Large Language Model Finetuning"
        ),
        "paper_url": "https://arxiv.org/abs/2607.08393",
        "media_urls": [
            "https://x.com/askalphaxiv/status/2076737985559822734/photo/1"
        ],
        "media_types": ["video"],
        "hashtags": [],
    },
}
KNOWN_TWEET_SOURCE_TEXT_SHA256 = hashlib.sha256(
    KNOWN_TWEET_SOURCE_TEXT["2076737985559822734"].encode("utf-8")
).hexdigest()
KNOWN_TWEET_SOURCE_SUMMARY = KNOWN_TWEET_SOURCE_TEXT["2076737985559822734"]
KNOWN_TWEET_SOURCE_AUTHOR = {
    "2076737985559822734": "askalphaxiv",
}
DEFAULT_INPUT = Path(
    "ecdlp_index_calculus_state/"
    "p1436_large_prime_residual_collision_collector_after_p1435_probe_for_harness_v4_smoke.json"
)
DEFAULT_OUTPUT = Path(
    "ecdlp_index_calculus_state/"
    "p1436_autoresearch_focus_report_after_p1435.json"
)
DEFAULT_SUMMATION_FFE_EVIDENCE_INVENTORY = (
    DEFAULT_OUTPUT.with_name("summation_ffe_evidence_inventory.json")
)
DEFAULT_SUMMATION_FFE_REPLAY_PLAN = (
    DEFAULT_OUTPUT.with_name("summation_ffe_evidence_replay_plan.json")
)
DEFAULT_NOTE = Path(
    "ecdlp_index_calculus_state/"
    "p1436_autoresearch_focus_report_after_p1435.md"
)
DEFAULT_IDEA340_PREFLIGHT = Path(
    "ecdlp_index_calculus_state/p1436_idea340_public_chart_preflight.json"
)
DEFAULT_SLICE_SOURCE_PREFLIGHT = Path(
    "ecdlp_index_calculus_state/p1436_slice_quadratic_public_source_preflight.json"
)
DEFAULT_PRESURFACE_FULL_CHARGE_PREFLIGHT = Path(
    "ecdlp_index_calculus_state/p1436_presurface_full_charge_preflight.json"
)
DEFAULT_FACTOR_LINE_EQUIVALENCE_PROBE = Path(
    "ecdlp_index_calculus_state/"
    "p1436_factor_line_direct_root_equivalence_probe.json"
)
DEFAULT_CONSTRUCTIVE_CLOSURE_COLLISION_GATE = Path(
    "p1553_constructive_closure_collision_gate_report_r69.json"
)
DEFAULT_MULTIPLICATIVE_X_S3_CLOSURE_SCREEN = Path(
    "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
)
DEFAULT_S4_CENTERED_CARRY_RANK_PROBE = Path(
    "p1553_s4_centered_carry_rank_probe_report_r71.json"
)
DEFAULT_S6_CENTERED_CARRY_RANK_MINOR_PROBE = Path(
    "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
)
DEFAULT_RESULTANT_VALUATION_TRACE_GRAMMAR = Path(
    "p1553_resultant_valuation_trace_grammar_report_r73.json"
)
DEFAULT_S6_RESIDUAL_DECISION_DIAGRAM_PROBE = Path(
    "p1553_s6_residual_decision_diagram_probe_report_r74.json"
)
DEFAULT_S6_ITERATED_NORM_SUPPORT_PROBE = Path(
    "p1553_s6_iterated_norm_support_probe_report_r75.json"
)
DEFAULT_S6_SUBSET_INCIDENCE_MOBIUS_PROBE = Path(
    "p1553_s6_subset_incidence_mobius_probe_report_r76.json"
)
DEFAULT_TARGET_TRANSLATED_FREQUENCY_ORBIT_PROBE = Path(
    "p1553_target_translated_frequency_orbit_probe_report_r77.json"
)
DEFAULT_ACTUAL_S6_FERMAT_TT_PROBE = Path(
    "p1553_actual_s6_fermat_tensor_train_probe_report_r78.json"
)
DEFAULT_SCALAR_ONLY_NESTED_NORM_SLP_PROBE = Path(
    "p1553_scalar_only_nested_norm_slp_probe_report_r79.json"
)
DEFAULT_BATCHED_NESTED_NORM_NODE_COMPILER_PROBE = Path(
    "p1553_batched_nested_norm_node_compiler_probe_report_r80.json"
)
DEFAULT_FULL_MULTIPLICATIVE_X_COSET_ENDPOINT_PROBE = Path(
    "p1553_full_multiplicative_x_coset_endpoint_probe_report_r81.json"
)
DEFAULT_CARTESIAN_SUM_COMPACT_DIVISOR_PROBE = Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
DEFAULT_5A5C_COORDINATE_FILTRATION_PROBE = Path(
    "p1553_5a5c_coordinate_filtration_probe_report_r83.json"
)
DEFAULT_5A5C_MARKED_RESULTANT_SOURCE_SECTION_PROBE = Path(
    "p1553_5a5c_marked_resultant_source_section_probe_report_r84.json"
)
DEFAULT_5A5C_TARGET_UNIFORM_PRECOEFFICIENT_CIRCUIT_PROBE = Path(
    "p1553_5a5c_target_uniform_precoefficient_circuit_probe_report_r85.json"
)
DEFAULT_5A5C_SPARSE_MULTIHOMOGENEOUS_MOMENT_RECURRENCE_PROBE = Path(
    "p1553_5a5c_sparse_multihomogeneous_moment_recurrence_probe_report_r86.json"
)
DEFAULT_5A5C_JET_PRESERVING_ADDITION_PUSHFORWARD_PROBE = Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_report_r87.json"
)
DEFAULT_5A5C_BLACK_BOX_RESULTANT_LOCALIZER_PROBE = Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_"
    "probe_report_r88.json"
)
DEFAULT_5A5C_FIXED_MARKER_SCALAR_RECURRENCE_PROBE = Path(
    "p1553_5a5c_fixed_marker_scalar_recurrence_probe_report_r89.json"
)
DEFAULT_5A5C_NONLOCAL_MOMENT_HANKEL_TRANSLATION_PROBE = Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_"
    "probe_report_r90.json"
)
DEFAULT_5A5C_UNEQUAL_LIST_SUBFUNCTION_INVERSION_PROBE = Path(
    "p1553_5a5c_unequal_list_subfunction_inversion_"
    "probe_report_r91.json"
)
DEFAULT_5A5C_COMPACT_ELLIPTIC_SUBFUNCTION_MAP_PROBE = Path(
    "p1553_5a5c_compact_elliptic_subfunction_map_"
    "probe_report_r92.json"
)
DEFAULT_5A5C_SHARED_SEMILINEAR_INCIDENCE_PROBE = Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_report_r93.json"
)
DEFAULT_5A5C_IMPLICIT_VERONESE_HYPERPLANE_SOURCE_INDEX_PROBE = Path(
    "p1553_5a5c_implicit_veronese_hyperplane_"
    "source_index_probe_report_r94.json"
)
DEFAULT_5A5C_AGGREGATE_VERONESE_PROJECTOR_RECURRENCE_PROBE = Path(
    "p1553_5a5c_aggregate_veronese_projector_"
    "recurrence_probe_report_r95.json"
)
DEFAULT_5A5C_MODULAR_FROBENIUS_TRACE_RECURRENCE_PROBE = Path(
    "p1553_5a5c_modular_frobenius_trace_"
    "recurrence_probe_report_r96.json"
)
DEFAULT_5A5C_FACTORED_TRANSPOSED_PROJECTOR_TRACE_PROBE = Path(
    "p1553_5a5c_factored_transposed_projector_"
    "trace_probe_report_r97.json"
)
DEFAULT_5A5C_NONLINEAR_TENSOR_TOWER_TRACE_PROBE = Path(
    "p1553_5a5c_nonlinear_tensor_tower_"
    "trace_probe_report_r98.json"
)
DEFAULT_5A5C_MULTIEDGE_DIGITIZED_EQUALITY_PROJECTOR_PROBE = Path(
    "p1553_5a5c_multiedge_digitized_equality_"
    "projector_probe_report_r99.json"
)
DEFAULT_5A5C_SUCCINCT_AGGREGATE_DIGIT_TRIE_PROBE = Path(
    "p1553_5a5c_succinct_aggregate_digit_"
    "trie_probe_report_r100.json"
)
DEFAULT_5A5C_ACTUAL_DIVISOR_IMAGE_ENTROPY_MERGE_PROBE = Path(
    "p1553_5a5c_actual_divisor_image_entropy_"
    "merge_probe_report_r101.json"
)
DEFAULT_5A5C_TWO_SIDED_IMPLICIT_JOIN_PROBE = Path(
    "p1553_5a5c_two_sided_implicit_join_probe_report_r102.json"
)
DEFAULT_5A5C_TARGET_FORCED_ALGEBRAIC_JOIN_FILTER_PROBE = Path(
    "p1553_5a5c_target_forced_algebraic_join_"
    "filter_probe_report_r103.json"
)
DEFAULT_5A5C_COMPACT_PREENDPOINT_S3_FFE_PUSHDOWN_PROBE = Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_"
    "pushdown_probe_report_r104.json"
)
DEFAULT_5A5C_ACTUAL_DECK_NONMERGEABLE_TARGET_PULLBACK_PROBE = Path(
    "p1553_5a5c_actual_deck_nonmergeable_"
    "target_pullback_probe_report_r105.json"
)
DEFAULT_5A5C_SCALAR_TARGET_NORM_COUNT_CIRCUIT_PROBE = Path(
    "p1553_5a5c_scalar_target_norm_count_"
    "circuit_probe_report_r106.json"
)
DEFAULT_5A5C_NONCHARACTER_ALGEBRAIC_TARGET_NORM_RESULTANT_PROBE = Path(
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_report_r107.json"
)
DEFAULT_5A5C_FACTORED_ELLIPTIC_LAMBDA_RING_CHOW_NORM_PROBE = Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
    "norm_probe_report_r108.json"
)
DEFAULT_5A5C_POINCARE_THETA_TARGET_SECTION_RANK_PROBE = Path(
    "p1553_5a5c_poincare_theta_target_section_"
    "rank_probe_report_r109.json"
)
DEFAULT_5A5C_THETA_ADDITION_CANCELLATION_NETWORK_PROBE = Path(
    "p1553_5a5c_theta_addition_cancellation_"
    "network_probe_report_r110.json"
)
DEFAULT_5A5C_FINITE_DECK_ALTERNANT_ANNIHILATOR_PROBE = Path(
    "p1553_5a5c_finite_deck_alternant_"
    "annihilator_probe_report_r111.json"
)
DEFAULT_5A5C_GAUGE_NORMALIZED_ENDPOINT_QUERY2P1_PROBE = Path(
    "p1553_5a5c_gauge_normalized_endpoint_"
    "query2p1_probe_report_r112.json"
)
DEFAULT_5A5C_NONLINEAR_ELLIPTIC_ORBIT_PRODUCT_PROBE = Path(
    "p1553_5a5c_nonlinear_elliptic_"
    "orbit_product_probe_report_r113.json"
)
DEFAULT_5A5C_TRANSPOSED_NONUNIFORM_C5_LEAF_GENERATOR_PROBE = Path(
    "p1553_5a5c_transposed_nonuniform_c5_"
    "leaf_generator_probe_report_r114.json"
)
DEFAULT_RELATION_ARITY_FACTOR_BASE_TRANSPOSED_INTERFACE_PROBE = Path(
    "p1553_relation_arity_factor_base_transposed_"
    "interface_rebalance_probe_report_r115.json"
)
DEFAULT_M6_A6_BATCHED_C3_PAIR_SUM_SOURCE_LOCATOR_PROBE = Path(
    "p1553_m6_a6_batched_c3_pair_sum_source_"
    "locator_probe_report_r116.json"
)
DEFAULT_M6_TARGET_BATCHED_C3_ELLIPTIC_TRANSPOSE_PROBE = Path(
    "p1553_m6_target_batched_c3_elliptic_"
    "transpose_probe_report_r117.json"
)
DEFAULT_M6_NONLINEAR_VALUE_SENSITIVE_C6_SOURCE_LOCATOR_PROBE = Path(
    "p1553_m6_nonlinear_value_sensitive_c6_"
    "source_locator_probe_report_r118.json"
)
DEFAULT_M6_OUTPUT_SENSITIVE_NONLINEAR_C5_SOURCE_INDEX_PROBE = Path(
    "p1553_m6_output_sensitive_nonlinear_c5_"
    "source_index_probe_report_r119.json"
)
DEFAULT_M6_SUBOUTPUT_IMPLICIT_C5_CHARACTER_PAIRING_PROBE = Path(
    "p1553_m6_suboutput_implicit_c5_character_"
    "pairing_probe_report_r120.json"
)
DEFAULT_M6_SMALL_K_MULTIPLICATIVE_C5_MOMENT_TORUS_PROBE = Path(
    "p1553_m6_small_k_multiplicative_c5_"
    "moment_torus_probe_report_r121.json"
)
DEFAULT_TORUS_C5_EXPLICIT_SPLIT_GLOBAL_REBALANCE_PROBE = Path(
    "p1553_torus_c5_explicit_split_global_"
    "rebalance_probe_report_r122.json"
)
DEFAULT_TORUS_C5_FOURIER_PRODUCT_RESULTANT_PROBE = Path(
    "p1553_torus_c5_fourier_product_resultant_probe_report_r123.json"
)
DEFAULT_TORUS_C5_LINEAR_SKETCH_CIRCULANT_PROBE = Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_report_r124.json"
)
DEFAULT_TORUS_C5_PRIME_ORDER_HOMOMORPHIC_FINGERPRINT_PROBE = Path(
    "p1553_torus_c5_prime_order_homomorphic_"
    "fingerprint_probe_report_r125.json"
)
DEFAULT_TORUS_C5_EXPLICIT_HASH_CORRECTION_SUPPORT_PROBE = Path(
    "p1553_torus_c5_explicit_hash_correction_"
    "support_probe_report_r126.json"
)
DEFAULT_TORUS_C5_BUCKET_RESULTANT_ROUTING_TRADEOFF_PROBE = Path(
    "p1553_torus_c5_bucket_resultant_routing_"
    "tradeoff_probe_report_r127.json"
)
DEFAULT_TORUS_C5_RATIONAL_SELECTOR_DEGREE_PROBE = Path(
    "p1553_torus_c5_rational_selector_degree_probe_report_r128.json"
)
DEFAULT_TORUS_C5_PIECEWISE_SELECTOR_DECISION_DAG_PROBE = Path(
    "p1553_torus_c5_piecewise_selector_decision_dag_"
    "probe_report_r129.json"
)
DEFAULT_TORUS_C5_SPARSE_FOURIER_PREDICATE_TRANSFER_PROBE = Path(
    "p1553_torus_c5_sparse_fourier_predicate_transfer_"
    "probe_report_r130.json"
)
DEFAULT_TORUS_C5_CONSECUTIVE_MODE_PREDICATE_PROBE = Path(
    "p1553_torus_c5_consecutive_mode_predicate_"
    "probe_report_r131.json"
)
DEFAULT_TORUS_C5_BASE_FIELD_FROBENIUS_PREDICATE_DAG_PROBE = Path(
    "p1553_torus_c5_base_field_frobenius_predicate_dag_"
    "probe_report_r132.json"
)
DEFAULT_TORUS_C5_SPARSE_MONOMIAL_ROOT_BOUND_PROBE = Path(
    "p1553_torus_c5_sparse_monomial_root_bound_"
    "probe_report_r133.json"
)
DEFAULT_TORUS_C5_TWO_ATOM_GEOMETRIC_PROGRESSION_PROBE = Path(
    "p1553_torus_c5_two_atom_geometric_progression_"
    "probe_report_r134.json"
)
DEFAULT_TORUS_C5_KHATRI_RAO_KRUSKAL_AMPLIFICATION_PROBE = Path(
    "p1553_torus_c5_khatri_rao_kruskal_amplification_"
    "probe_report_r135.json"
)
DEFAULT_TORUS_C5_ALL_NONZERO_PATH_PRODUCT_PROBE = Path(
    "p1553_torus_c5_all_nonzero_path_product_probe_report_r136.json"
)
DEFAULT_TORUS_C5_BINOMIAL_NODE_UNION_DEPTH_PROBE = Path(
    "p1553_torus_c5_binomial_node_union_depth_probe_report_r137.json"
)
DEFAULT_TORUS_C5_CHEBOTAREV_FIBER_COVER_PROBE = Path(
    "p1553_torus_c5_chebotarev_fiber_cover_probe_report_r138.json"
)
DEFAULT_TORUS_C5_ORDER_TWO_THREE_MINOR_RIGIDITY_PROBE = Path(
    "p1553_torus_c5_order_two_three_minor_"
    "rigidity_probe_report_r139.json"
)
DEFAULT_TORUS_C5_ORDER_TWO_FOUR_MINOR_CLAW_PROBE = Path(
    "p1553_torus_c5_order_two_four_minor_"
    "claw_probe_report_r140.json"
)
DEFAULT_TORUS_C5_SEXTIC_MOBIUS_CHARACTER_ROUTER_PROBE = Path(
    "p1553_torus_c5_sextic_mobius_character_"
    "router_probe_report_r141.json"
)
DEFAULT_TORUS_C5_ADAPTIVE_CHARACTER_DECISION_ROUTER_PROBE = Path(
    "p1553_torus_c5_adaptive_character_decision_"
    "router_probe_report_r142.json"
)
DEFAULT_TORUS_C5_LABEL_CONGRUENCE_CORRECTION_PROBE = Path(
    "p1553_torus_c5_label_congruence_correction_"
    "probe_report_r143.json"
)
DEFAULT_M6_WEIGHTED_FIBER_MARGINAL_LOG_OPERATOR_PROBE = Path(
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_report_r144.json"
)
DEFAULT_M6_WEIGHTED_C3_MOBIUS_GCD_TRACE_PROBE = Path(
    "p1553_m6_weighted_c3_mobius_gcd_trace_"
    "probe_report_r145.json"
)
DEFAULT_M6_AGGREGATE_MARGINAL_SINGLETON_SOURCE_EQUIVALENCE_PROBE = Path(
    "p1553_m6_aggregate_marginal_singleton_source_equivalence_"
    "probe_report_r146.json"
)
DEFAULT_M6_OCCURRENCE_PAIR_RESULTANT_LOCAL_VALUATION_PROBE = Path(
    "p1553_m6_occurrence_pair_resultant_local_valuation_"
    "probe_report_r147.json"
)
DEFAULT_M6_STATIC_3SUM_INDEXING_TRADEOFF_PROBE = Path(
    "p1553_m6_static_3sum_indexing_tradeoff_probe_report_r148.json"
)
DEFAULT_M6_ACTUAL_C6_SHIFT_KRYLOV_RANK_PROBE = Path(
    "p1553_m6_actual_c6_shift_krylov_rank_probe_report_r149.json"
)
DEFAULT_M6_RATIONAL_CONVOLUTION_SUBALGEBRA_RIGIDITY_PROBE = Path(
    "p1553_m6_rational_convolution_subalgebra_rigidity_"
    "probe_report_r150.json"
)
DEFAULT_M6_MATRIX_FREE_MARGINAL_JACOBIAN_KRYLOV_PROBE = Path(
    "p1553_m6_matrix_free_marginal_jacobian_krylov_"
    "probe_report_r151.json"
)
DEFAULT_M6_GEOMETRY_ONLY_WEIGHT_INTERPOLATION_ADJOINT_PROBE = Path(
    "p1553_m6_geometry_only_weight_interpolation_adjoint_"
    "probe_report_r152.json"
)
DEFAULT_M6_SYMMETRIC_SHIFT_REVERSE_ONLY_MARGINAL_PROBE = Path(
    "p1553_m6_symmetric_shift_reverse_only_marginal_"
    "probe_report_r153.json"
)
DEFAULT_M6_SIGNED_QUOTIENT_MULTISCALE_RANK_PROBE = Path(
    "p1553_m6_signed_quotient_multiscale_rank_"
    "probe_report_r154.json"
)
DEFAULT_M6_SINGLETON_RELATION_HYPERGRAPH_RANK_PROBE = Path(
    "p1553_m6_singleton_relation_hypergraph_rank_"
    "probe_report_r155.json"
)
DEFAULT_M6_A_DIVERSITY_PROJECTIVE_RANK_PROBE = Path(
    "p1553_m6_a_diversity_projective_rank_"
    "probe_report_r156.json"
)
DEFAULT_M6_HASH_TO_CURVE_PROJECTIVE_RANK_PROBE = Path(
    "p1553_m6_hash_to_curve_projective_rank_"
    "probe_report_r157.json"
)
DEFAULT_M6_SHORT_RELATION_NEAR_INJECTIVITY_SUPPLY_PROBE = Path(
    "p1553_m6_short_relation_near_injectivity_supply_"
    "probe_report_r158.json"
)
DEFAULT_M6_RANDOM_DIAGONAL_KNOWN_TARGET_RANK_PROBE = Path(
    "p1553_m6_random_diagonal_known_target_rank_"
    "probe_report_r159.json"
)
DEFAULT_M6_POSITIVE_C6_GENERIC_LOCATOR_REDUCTION_PROBE = Path(
    "p1553_m6_positive_c6_generic_locator_reduction_"
    "probe_report_r160.json"
)
DEFAULT_M6_SIGNED_C3_DIVISOR_TRANSLATION_GCD_PROBE = Path(
    "p1553_m6_signed_c3_divisor_translation_gcd_"
    "probe_report_r161.json"
)
DEFAULT_M6_BATCH_INVERSE_TRANSPOSE_MODCOMP_FIT_PROBE = Path(
    "p1553_m6_batch_inverse_transpose_modcomp_fit_"
    "probe_report_r162.json"
)
DEFAULT_M6_AGGREGATE_UNION_FACTOR_LABEL_RECOVERY_PROBE = Path(
    "p1553_m6_aggregate_union_factor_label_recovery_"
    "probe_report_r163.json"
)
DEFAULT_M6_RANDOMIZED_TARGET_DIVISOR_NORM_UNION_PROBE = Path(
    "p1553_m6_randomized_target_divisor_norm_union_"
    "probe_report_r164.json"
)
DEFAULT_M6_GLOBAL_RANDOMIZER_ELLIPTIC_TRANSLATE_PRODUCT_PROBE = Path(
    "p1553_m6_global_randomizer_elliptic_translate_product_"
    "probe_report_r165.json"
)
DEFAULT_M6_KUMMER_X_TRANSLATE_SIGNED_VERIFICATION_PROBE = Path(
    "p1553_m6_kummer_x_translate_signed_verification_"
    "probe_report_r166.json"
)
DEFAULT_M6_GENERALIZED_TARGET_DIVISOR_WEIL_RECIPROCITY_SWAP_PROBE = Path(
    "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_"
    "probe_report_r167.json"
)
DEFAULT_M6_LOG_DERIVATIVE_ELLIPTIC_CAUCHY_TRACE_PROBE = Path(
    "p1553_m6_log_derivative_elliptic_cauchy_trace_"
    "probe_report_r168.json"
)
DEFAULT_M6_REGULARIZED_LOG_TRACE_DISPLACEMENT_RANK_PROBE = Path(
    "p1553_m6_regularized_log_trace_displacement_rank_"
    "probe_report_r169.json"
)
DEFAULT_M6_LAMBDA_ZERO_FITTING_TARGET_NORM_DEDUP_PROBE = Path(
    "p1553_m6_lambda_zero_fitting_target_norm_dedup_"
    "probe_report_r170.json"
)
DEFAULT_M6_BALANCED_MILLER_TREE_NORM_STREAMING_PROBE = Path(
    "p1553_m6_balanced_miller_tree_norm_streaming_"
    "probe_report_r171.json"
)
DEFAULT_M6_TARGET_SIGN_CONJUGATE_S3_SELF_RESULTANT_PROBE = Path(
    "p1553_m6_target_sign_conjugate_s3_self_resultant_"
    "probe_report_r172.json"
)
DEFAULT_M6_S3_DETERMINANTAL_TRANSFER_NONCOMMUTATIVITY_PROBE = Path(
    "p1553_m6_s3_determinantal_transfer_noncommutativity_"
    "probe_report_r173.json"
)
DEFAULT_M6_CONFLUENT_SIGNED_DUAL_CHOW_PUSHFORWARD_PROBE = Path(
    "p1553_m6_confluent_signed_dual_chow_pushforward_"
    "probe_report_r174.json"
)
DEFAULT_M6_SCALAR_SUBSET_INCIDENCE_GROUP_TESTING_PROBE = Path(
    "p1553_m6_scalar_subset_incidence_group_testing_"
    "probe_report_r175.json"
)
DEFAULT_M6_PRINCIPAL_TARGET_PONTRYAGIN_RESULTANT_PROBE = Path(
    "p1553_m6_principal_target_pontryagin_resultant_"
    "probe_report_r176.json"
)
DEFAULT_M6_GLOBAL_MARKED_FITTING_LOCATOR_PROBE = Path(
    "p1553_m6_global_marked_fitting_locator_probe_report_r177.json"
)
DEFAULT_M6_MARKED_FITTING_SIGNED_NORM_DEDUP_PROBE = Path(
    "p1553_m6_marked_fitting_signed_norm_dedup_probe_report_r178.json"
)
DEFAULT_M6_SQUAREFREE_TRUNCATED_RESULTANT_APPLICABILITY_PROBE = Path(
    "p1553_m6_squarefree_truncated_resultant_applicability_"
    "probe_report_r179.json"
)
DEFAULT_M6_D5_DIRECTED_EVALUATION_SURVIVOR_PROBE = Path(
    "p1553_m6_d5_directed_evaluation_survivor_probe_report_r180.json"
)
DEFAULT_M6_MONOGENIC_KERNEL_BIDEGREE_PROBE = Path(
    "p1553_m6_monogenic_kernel_bidegree_probe_report_r181.json"
)
DEFAULT_M6_GCD_EQUIVALENT_TARGET_SUBRESULTANT_PROBE = Path(
    "p1553_m6_gcd_equivalent_target_subresultant_probe_report_r182.json"
)
DEFAULT_M6_SPARSE_PROJECTOR_PRONY_LOCATOR_PROBE = Path(
    "p1553_m6_sparse_projector_prony_locator_probe_report_r183.json"
)
FRONTIER_PREFLIGHT_SCHEMAS = {
    "idea340_public_chart": "ecdlp.p1436_idea340_public_chart_preflight.v1",
    "slice_quadratic_public_source": (
        "ecdlp.p1436_slice_quadratic_public_source_preflight.v1"
    ),
    "presurface_full_charge": (
        "ecdlp.p1436_presurface_full_charge_preflight.v1"
    ),
    "factor_line_direct_root_equivalence": (
        "ecdlp.p1436_factor_line_direct_root_equivalence_probe.v1"
    ),
    "constructive_closure_collision": (
        "p1553.constructive_closure_collision_gate.r69.v1"
    ),
    "multiplicative_x_s3_closure": (
        "p1553.multiplicative_x_s3_closure_screen.r70.v1"
    ),
    "s4_centered_carry_rank": (
        "p1553.s4_centered_carry_rank_probe.r71.v1"
    ),
    "s6_centered_carry_rank_minor": (
        "p1553.s6_centered_carry_rank_minor_probe.r72.v1"
    ),
    "resultant_valuation_trace_grammar": (
        "p1553.resultant_valuation_trace_grammar.r73.v1"
    ),
    "s6_residual_decision_diagram": (
        "p1553.s6_residual_decision_diagram_probe.r74.v1"
    ),
    "s6_iterated_norm_support": (
        "p1553.s6_iterated_norm_support_probe.r75.v1"
    ),
    "s6_subset_incidence_mobius": (
        "p1553.s6_subset_incidence_mobius_probe.r76.v1"
    ),
    "target_translated_frequency_orbit": (
        "p1553.target_translated_frequency_orbit_probe.r77.v1"
    ),
    "actual_s6_fermat_tensor_train": (
        "p1553.actual_s6_fermat_tensor_train_probe.r78.v1"
    ),
    "scalar_only_nested_norm_slp": (
        "p1553.scalar_only_nested_norm_slp_probe.r79.v1"
    ),
    "batched_nested_norm_node_compiler": (
        "p1553.batched_nested_norm_node_compiler_probe.r80.v1"
    ),
    "full_multiplicative_x_coset_endpoint": (
        "p1553.full_multiplicative_x_coset_endpoint_probe.r81.v1"
    ),
    "cartesian_sum_compact_divisor": (
        "p1553.cartesian_sum_compact_divisor_probe.r82.v1"
    ),
    "5a5c_coordinate_filtration": (
        "p1553.5a5c_coordinate_filtration_probe.r83.v1"
    ),
    "5a5c_marked_resultant_source_section": (
        "p1553.5a5c_marked_resultant_source_section.r84.v1"
    ),
    "5a5c_target_uniform_precoefficient_circuit": (
        "p1553.5a5c_target_uniform_precoefficient_circuit.r85.v1"
    ),
    "5a5c_sparse_multihomogeneous_moment_recurrence": (
        "p1553.5a5c_sparse_multihomogeneous_moment_recurrence.r86.v1"
    ),
    "5a5c_jet_preserving_addition_pushforward": (
        "p1553.5a5c_jet_preserving_addition_pushforward.r87.v1"
    ),
    "5a5c_black_box_resultant_localizer": (
        "p1553.5a5c_black_box_resultant_localizer.r88.v1"
    ),
    "5a5c_fixed_marker_scalar_recurrence": (
        "p1553.5a5c_fixed_marker_scalar_recurrence.r89.v1"
    ),
    "5a5c_nonlocal_moment_hankel_translation": (
        "p1553.5a5c_nonlocal_moment_hankel_translation.r90.v1"
    ),
    "5a5c_unequal_list_subfunction_inversion": (
        "p1553.5a5c_unequal_list_subfunction_inversion.r91.v1"
    ),
    "5a5c_compact_elliptic_subfunction_map": (
        "p1553.5a5c_compact_elliptic_subfunction_map.r92.v1"
    ),
    "5a5c_shared_semilinear_incidence_correspondence": (
        "p1553.5a5c_shared_semilinear_incidence_correspondence.r93.v1"
    ),
    "5a5c_implicit_veronese_hyperplane_source_index": (
        "p1553.5a5c_implicit_veronese_hyperplane_index.r94.v1"
    ),
    "5a5c_aggregate_veronese_projector_recurrence": (
        "p1553.5a5c_aggregate_veronese_projector_recurrence.r95.v1"
    ),
    "5a5c_modular_frobenius_trace_recurrence": (
        "p1553.5a5c_modular_frobenius_trace_recurrence.r96.v1"
    ),
    "5a5c_factored_transposed_projector_trace": (
        "p1553.5a5c_factored_transposed_projector_trace.r97.v1"
    ),
    "5a5c_nonlinear_tensor_tower_trace": (
        "p1553.5a5c_nonlinear_tensor_tower_trace.r98.v1"
    ),
    "5a5c_multiedge_digitized_equality_projector": (
        "p1553.5a5c_multiedge_digitized_equality_projector.r99.v1"
    ),
    "5a5c_succinct_aggregate_digit_trie": (
        "p1553.5a5c_succinct_aggregate_digit_trie.r100.v1"
    ),
    "5a5c_actual_divisor_image_entropy_merge": (
        "p1553.5a5c_actual_divisor_image_entropy_merge.r101.v1"
    ),
    "5a5c_two_sided_implicit_join": (
        "p1553.5a5c_two_sided_implicit_join.r102.v1"
    ),
    "5a5c_target_forced_algebraic_join_filter": (
        "p1553.5a5c_target_forced_algebraic_join_filter.r103.v1"
    ),
    "5a5c_compact_preendpoint_s3_ffe_pushdown": (
        "p1553.5a5c_compact_preendpoint_s3_ffe_pushdown.r104.v1"
    ),
    "5a5c_actual_deck_nonmergeable_target_pullback": (
        "p1553.5a5c_actual_deck_nonmergeable_target_pullback.r105.v1"
    ),
    "5a5c_scalar_target_norm_count_circuit": (
        "p1553.5a5c_scalar_target_norm_count_circuit.r106.v1"
    ),
    "5a5c_noncharacter_algebraic_target_norm_resultant": (
        "p1553.5a5c_noncharacter_algebraic_target_norm_resultant.r107.v1"
    ),
    "5a5c_factored_elliptic_lambda_ring_chow_norm": (
        "p1553.5a5c_factored_elliptic_lambda_ring_chow_norm.r108.v1"
    ),
    "5a5c_poincare_theta_target_section_rank": (
        "p1553.5a5c_poincare_theta_target_section_rank.r109.v1"
    ),
    "5a5c_theta_addition_cancellation_network": (
        "p1553.5a5c_theta_addition_cancellation_network.r110.v1"
    ),
    "5a5c_finite_deck_alternant_annihilator": (
        "p1553.5a5c_finite_deck_alternant_annihilator.r111.v1"
    ),
    "5a5c_gauge_normalized_endpoint_query2p1": (
        "p1553.5a5c_gauge_normalized_endpoint_query2p1.r112.v1"
    ),
    "5a5c_nonlinear_elliptic_orbit_product": (
        "p1553.5a5c_nonlinear_elliptic_orbit_product.r113.v1"
    ),
    "5a5c_transposed_nonuniform_c5_leaf_generator": (
        "p1553.5a5c_transposed_nonuniform_c5_leaf_generator.r114.v1"
    ),
    "relation_arity_factor_base_transposed_interface_rebalance": (
        "p1553.relation_arity_factor_base_transposed_interface_rebalance."
        "r115.v1"
    ),
    "m6_a6_batched_c3_pair_sum_source_locator": (
        "p1553.m6_a6_batched_c3_pair_sum_source_locator.r116.v1"
    ),
    "m6_target_batched_c3_elliptic_transpose": (
        "p1553.m6_target_batched_c3_elliptic_transpose.r117.v1"
    ),
    "m6_nonlinear_value_sensitive_c6_source_locator": (
        "p1553.m6_nonlinear_value_sensitive_c6_source_locator.r118.v1"
    ),
    "m6_output_sensitive_nonlinear_c5_source_index": (
        "p1553.m6_output_sensitive_nonlinear_c5_source_index.r119.v1"
    ),
    "m6_suboutput_implicit_c5_character_pairing": (
        "p1553.m6_suboutput_implicit_c5_character_pairing.r120.v1"
    ),
    "m6_small_k_multiplicative_c5_moment_torus": (
        "p1553.m6_small_k_multiplicative_c5_moment_torus.r121.v1"
    ),
    "torus_c5_explicit_split_global_rebalance": (
        "p1553.torus_c5_explicit_split_global_rebalance.r122.v1"
    ),
    "torus_c5_fourier_product_resultant": (
        "p1553.torus_c5_fourier_product_resultant.r123.v1"
    ),
    "torus_c5_linear_sketch_circulant": (
        "p1553.torus_c5_linear_sketch_circulant.r124.v1"
    ),
    "torus_c5_prime_order_homomorphic_fingerprint": (
        "p1553.torus_c5_prime_order_homomorphic_fingerprint.r125.v1"
    ),
    "torus_c5_explicit_hash_correction_support": (
        "p1553.torus_c5_explicit_hash_correction_support.r126.v1"
    ),
    "torus_c5_bucket_resultant_routing_tradeoff": (
        "p1553.torus_c5_bucket_resultant_routing_tradeoff.r127.v1"
    ),
    "torus_c5_rational_selector_degree": (
        "p1553.torus_c5_rational_selector_degree.r128.v1"
    ),
    "torus_c5_piecewise_selector_decision_dag": (
        "p1553.torus_c5_piecewise_selector_decision_dag.r129.v1"
    ),
    "torus_c5_sparse_fourier_predicate_transfer": (
        "p1553.torus_c5_sparse_fourier_predicate_transfer.r130.v1"
    ),
    "torus_c5_consecutive_mode_predicate": (
        "p1553.torus_c5_consecutive_mode_predicate.r131.v1"
    ),
    "torus_c5_base_field_frobenius_predicate_dag": (
        "p1553.torus_c5_base_field_frobenius_predicate_dag.r132.v1"
    ),
    "torus_c5_sparse_monomial_root_bound": (
        "p1553.torus_c5_sparse_monomial_root_bound.r133.v1"
    ),
    "torus_c5_two_atom_geometric_progression": (
        "p1553.torus_c5_two_atom_geometric_progression.r134.v1"
    ),
    "torus_c5_khatri_rao_kruskal_amplification": (
        "p1553.torus_c5_khatri_rao_kruskal_amplification.r135.v1"
    ),
    "torus_c5_all_nonzero_path_product": (
        "p1553.torus_c5_all_nonzero_path_product.r136.v1"
    ),
    "torus_c5_binomial_node_union_depth": (
        "p1553.torus_c5_binomial_node_union_depth.r137.v1"
    ),
    "torus_c5_chebotarev_fiber_cover": (
        "p1553.torus_c5_chebotarev_fiber_cover.r138.v1"
    ),
    "torus_c5_order_two_three_minor_rigidity": (
        "p1553.torus_c5_order_two_three_minor_rigidity.r139.v1"
    ),
    "torus_c5_order_two_four_minor_claw": (
        "p1553.torus_c5_order_two_four_minor_claw.r140.v1"
    ),
    "torus_c5_sextic_mobius_character_router": (
        "p1553.torus_c5_sextic_mobius_character_router.r141.v1"
    ),
    "torus_c5_adaptive_character_decision_router": (
        "p1553.torus_c5_adaptive_character_decision_router.r142.v1"
    ),
    "torus_c5_label_congruence_correction": (
        "p1553.torus_c5_label_congruence_correction.r143.v1"
    ),
    "m6_weighted_fiber_marginal_log_operator": (
        "p1553.m6_weighted_fiber_marginal_log_operator.r144.v1"
    ),
    "m6_weighted_c3_mobius_gcd_trace": (
        "p1553.m6_weighted_c3_mobius_gcd_trace.r145.v1"
    ),
    "m6_aggregate_marginal_singleton_source_equivalence": (
        "p1553.m6_aggregate_marginal_singleton_source_equivalence.r146.v1"
    ),
    "m6_occurrence_pair_resultant_local_valuation": (
        "p1553.m6_occurrence_pair_resultant_local_valuation.r147.v1"
    ),
    "m6_static_3sum_indexing_tradeoff": (
        "p1553.m6_static_3sum_indexing_tradeoff.r148.v1"
    ),
    "m6_actual_c6_shift_krylov_rank": (
        "p1553.m6_actual_c6_shift_krylov_rank.r149.v1"
    ),
    "m6_rational_convolution_subalgebra_rigidity": (
        "p1553.m6_rational_convolution_subalgebra_rigidity.r150.v1"
    ),
    "m6_matrix_free_marginal_jacobian_krylov": (
        "p1553.m6_matrix_free_marginal_jacobian_krylov.r151.v1"
    ),
    "m6_geometry_only_weight_interpolation_adjoint": (
        "p1553.m6_geometry_only_weight_interpolation_adjoint.r152.v1"
    ),
    "m6_symmetric_shift_reverse_only_marginal": (
        "p1553.m6_symmetric_shift_reverse_only_marginal.r153.v1"
    ),
    "m6_signed_quotient_multiscale_rank": (
        "p1553.m6_signed_quotient_multiscale_rank.r154.v1"
    ),
    "m6_singleton_relation_hypergraph_rank": (
        "p1553.m6_singleton_relation_hypergraph_rank.r155.v1"
    ),
    "m6_a_diversity_projective_rank": (
        "p1553.m6_a_diversity_projective_rank.r156.v1"
    ),
    "m6_hash_to_curve_projective_rank": (
        "p1553.m6_hash_to_curve_projective_rank.r157.v1"
    ),
    "m6_short_relation_near_injectivity_supply": (
        "p1553.m6_short_relation_near_injectivity_supply.r158.v1"
    ),
    "m6_random_diagonal_known_target_rank": (
        "p1553.m6_random_diagonal_known_target_rank.r159.v1"
    ),
    "m6_positive_c6_generic_locator_reduction": (
        "p1553.m6_positive_c6_generic_locator_reduction.r160.v1"
    ),
    "m6_signed_c3_divisor_translation_gcd": (
        "p1553.m6_signed_c3_divisor_translation_gcd.r161.v1"
    ),
    "m6_batch_inverse_transpose_modcomp_fit": (
        "p1553.m6_batch_inverse_transpose_modcomp_fit.r162.v1"
    ),
    "m6_aggregate_union_factor_label_recovery": (
        "p1553.m6_aggregate_union_factor_label_recovery.r163.v1"
    ),
    "m6_randomized_target_divisor_norm_union": (
        "p1553.m6_randomized_target_divisor_norm_union.r164.v1"
    ),
    "m6_global_randomizer_elliptic_translate_product": (
        "p1553.m6_global_randomizer_elliptic_translate_product.r165.v1"
    ),
    "m6_kummer_x_translate_signed_verification": (
        "p1553.m6_kummer_x_translate_signed_verification.r166.v1"
    ),
    "m6_generalized_target_divisor_weil_reciprocity_swap": (
        "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.r167.v1"
    ),
    "m6_log_derivative_elliptic_cauchy_trace": (
        "p1553.m6_log_derivative_elliptic_cauchy_trace.r168.v1"
    ),
    "m6_regularized_log_trace_displacement_rank": (
        "p1553.m6_regularized_log_trace_displacement_rank.r169.v1"
    ),
    "m6_lambda_zero_fitting_target_norm_dedup": (
        "p1553.m6_lambda_zero_fitting_target_norm_dedup.r170.v1"
    ),
    "m6_balanced_miller_tree_norm_streaming": (
        "p1553.m6_balanced_miller_tree_norm_streaming.r171.v1"
    ),
    "m6_target_sign_conjugate_s3_self_resultant": (
        "p1553.m6_target_sign_conjugate_s3_self_resultant.r172.v1"
    ),
    "m6_s3_determinantal_transfer_noncommutativity": (
        "p1553.m6_s3_determinantal_transfer_noncommutativity.r173.v1"
    ),
    "m6_confluent_signed_dual_chow_pushforward": (
        "p1553.m6_confluent_signed_dual_chow_pushforward.r174.v1"
    ),
    "m6_scalar_subset_incidence_group_testing": (
        "p1553.m6_scalar_subset_incidence_group_testing.r175.v1"
    ),
    "m6_principal_target_pontryagin_resultant": (
        "p1553.m6_principal_target_pontryagin_resultant.r176.v1"
    ),
    "m6_global_marked_fitting_locator": (
        "p1553.m6_global_marked_fitting_locator.r177.v1"
    ),
    "m6_marked_fitting_signed_norm_dedup": (
        "p1553.m6_marked_fitting_signed_norm_dedup.r178.v1"
    ),
    "m6_squarefree_truncated_resultant_applicability": (
        "p1553.m6_squarefree_truncated_resultant_applicability.r179.v1"
    ),
    "m6_d5_directed_evaluation_survivor": (
        "p1553.m6_d5_directed_evaluation_survivor.r180.v1"
    ),
    "m6_monogenic_kernel_bidegree": (
        "p1553.m6_monogenic_kernel_bidegree.r181.v1"
    ),
    "m6_gcd_equivalent_target_subresultant": (
        "p1553.m6_gcd_equivalent_target_subresultant.r182.v1"
    ),
    "m6_sparse_projector_prony_locator": (
        "p1553.m6_sparse_projector_prony_locator.r183.v1"
    ),
}
DEFAULT_FIXED_CONFIG = "mixed_balanced_stride_mask1"
HASH_POLICIES = ("hash_control_0", "hash_control_1", "hash_control_2")
MAJOR_RESULT_STAGES = (
    "replay_and_exact_residual",
    "collision_supply",
    "cross_shift_routing",
    "source_row_elimination",
    "relation_rank",
    "rhs_compatibility",
    "factor_log_verification",
    "target_descent",
    "total_cost",
)
SHOUP_RHO_SCALE = 11
SHOUP_SCALING_RESIDUAL_TOLERANCE = 1.0
SHOUP_MIN_SCALE_COUNT = 2
SUMMATION_FFE_HINT_TERMS = ("summation", "ffe")
METHODOLOGY = {
    "source_post_url": TWEET_POST_URL,
    "source_post_url_with_query": f"{TWEET_POST_URL}{TWEET_POST_QUERY}",
    "autoresearch_url": "https://www.alphaxiv.org/replicate/2607.08393",
    "openresearch_cli_url": "https://github.com/alphaXiv/openresearch-cli",
    "paper_url": "https://arxiv.org/abs/2607.08393",
    "tweet_guidance": {
        "source_intake_mode": "public_snapshot_summary",
        "bounded_critical_set": True,
        "non_blocking_ambiguity_is_deterministic": True,
        "peripheral_scope_defer": True,
        "bounded_critical_set_note": "Keep critical experiments bounded to the smallest unresolved bottlenecks.",
        "source_summary": KNOWN_TWEET_SOURCE_SUMMARY,
        "source_summary_is_verbatim": True,
        "non_blocking_ambiguity_note": "Record non-blocking ambiguities as deterministic resolutions and keep them diagnostic-only.",
        "peripheral_scope_defer_note": "Defer peripheral branches until bounded critical queue is exhausted or falsified.",
        "source": "https://x.com/askalphaxiv/status/2076737985559822734",
        "source_post_id": "2076737985559822734",
        "source_author": "askalphaxiv",
        "source_query": "?s=46",
        "tweet_text_included": True,
        "tweet_text": KNOWN_TWEET_SOURCE_SUMMARY,
        "tweet_text_sha256": KNOWN_TWEET_SOURCE_TEXT_SHA256,
        "tweet_posted_at": "",
        "tweet_source_title": "",
        "tweet_referenced_paper_title": "",
        "tweet_source_url": "",
        "tweet_media_urls": [],
        "tweet_media_types": [],
        "tweet_hashtags": [],
        "tweet_has_media": False,
        "tweet_media_count": 0,
        "tweet_intake_status": "Exact tweet text snapshot captured for this known source.",
        "tweet_text_source_note": (
            "Exact tweet text is embedded for this known source "
            "as an auditable source snapshot."
        ),
        "source_guidance_version": "askalphaxiv-2076737985559822734-v1",
        "experiment_lineage_required": True,
        "baseline_parent_required": True,
        "run_evidence_required": True,
        "operator_interrupt_only_for_blocking_uncertainty": True,
        "required_resolution_ids": [
            "self_patching_fidelity",
            "paper_headroom_scope",
            "posthoc_oracle_scope",
            "synthetic_null_scope",
            "fixed_route_availability",
            "experiment_lineage_materialization",
            "ffe_information_conservation_scope",
            "resultant_valuation_trace_scope",
            "s6_residual_decision_diagram_scope",
            "s6_iterated_norm_support_scope",
            "s6_subset_incidence_mobius_scope",
            "target_translated_frequency_orbit_scope",
            "actual_s6_fermat_tensor_train_scope",
            "scalar_only_nested_norm_slp_scope",
            "batched_nested_norm_node_compiler_scope",
            "full_multiplicative_x_coset_endpoint_scope",
            "target_descent_absence",
            "independent_audit_binding",
            "source_claim_status",
        ],
    },
    "adaptation": (
        "Separate stored collision evidence from usable rank, verified logs, "
        "and held-out target descent; use interventions only as diagnostics."
    ),
    "process_adaptation": (
        "Concentrate each continuation on a bounded set of decisive experiments, "
        "defer lower-priority branches explicitly, and record deterministic "
        "resolutions for non-blocking ambiguities."
    ),
    "openresearch_harness_adaptation": (
        "Represent the immutable source as a baseline root and every selected or "
        "deferred probe as an explicitly parented experiment node. This harness "
        "emits logical lineage only; it does not claim that a git branch or run "
        "was materialized."
    ),
    "paper_self_patching_method": (
        "The paper copies one anchor representation from a source layer into a "
        "target layer, scans layer pairs, and compares a fixed heuristic with a "
        "posthoc oracle on held-out use cases."
    ),
    "fidelity_boundary": (
        "P1436 configurations rerun the collector under different frozen routing "
        "rules; they do not patch an identical intermediate EC state. Their deltas "
        "are matched configuration ablations, not causal self-patching evidence."
    ),
    "paper_headroom_range_is_not_an_ecdlp_gate": True,
    "ffe_information_conservation_boundary": (
        "Product-section and summation-polynomial quotient rows add no factor-log "
        "information beyond their fixed-sum factor rows. Admit a new FFE/summation "
        "lane only when it supplies a scalar-blind, independently replayable "
        "new-factor-row enumerator below direct pair-complement cost."
    ),
}

CRITICAL_EXPERIMENTS = {
    "repair_exactness_before_search": {
        "hypothesis": "The apparent bottleneck is caused by a replay or validation defect.",
        "decisive_test": "Rebuild every source equation and residual equality independently.",
        "falsifier": "All source, residual, sign, and validation checks replay exactly.",
        "required_artifacts": ["source_replay.json", "validation_failures.json"],
    },
    "residual_supply_and_cross_shift_probe": {
        "hypothesis": "The natural route underproduces useful cross-shift residual collisions.",
        "decisive_test": "Compare exact collision arrival and shift labels with the matched uniform stream.",
        "falsifier": "Natural collision supply and cross-shift fraction match or exceed the control.",
        "required_artifacts": ["occupancy_comparison.json", "cross_shift_arrivals.jsonl"],
    },
    "collision_to_rank_routing_ablation": {
        "hypothesis": "Stored collision edges are lost when compiled into independent factor rows.",
        "decisive_test": "Replay all-edge, cross-shift-only, and within-shift-only matrices with exact sources.",
        "falsifier": "The natural all-edge matrix reaches full RHS-compatible rank.",
        "required_artifacts": ["routing_ablation.json", "relation_matrices.json"],
    },
    "summation_ffe_evidence_readiness": {
        "hypothesis": "Summation-polynomial and FFE-labeled configuration state is not yet normalized into the exact replay contract.",
        "decisive_test": (
            "Normalize every summation/FFE-labeled config into exact replay artifacts, "
            "then verify collision provenance for each."
        ),
        "falsifier": "Any labeled summation/FFE route is missing source provenance or replay artifacts.",
        "required_artifacts": [
            "summation_ffe_evidence_inventory.json",
            "summation_ffe_evidence_replay_plan.json",
        ],
    },
    "summation_ffe_new_factor_row_cost_gate": {
        "hypothesis": (
            "A scalar-blind summation/FFE source enumerator discovers new independent "
            "fixed-sum factor rows below direct pair-complement cost."
        ),
        "decisive_test": (
            "Replay the bound source enumerator, verify each claimed row is new and "
            "independent, charge all source operations, and compare the measured cost "
            "with direct pair-complement enumeration."
        ),
        "falsifier": (
            "The route uses scalar labels, lacks a hash-bound replay, finds no new "
            "independent factor row, or costs at least direct pair-complement enumeration."
        ),
        "required_artifacts": [
            "summation_ffe_evidence_inventory.json",
            "summation_ffe_evidence_replay_plan.json",
            "new_factor_row_replay.json",
            "direct_pair_complement_cost.json",
        ],
    },
    "public_slice_source_corpus": {
        "hypothesis": (
            "Public pre-choice surface features concentrate preserving "
            "slice-quadratic candidates enough to overcome the blind slice floor."
        ),
        "decisive_test": (
            "Emit one label-separated corpus over at least four target families, "
            "freeze deterministic public candidates before outcomes are revealed, "
            "charge every rejection, and run leave-one-target-family-out."
        ),
        "falsifier": (
            "No public candidate schedule supplies fresh independent fixed-sum rows "
            "below rho on every held-out target family."
        ),
        "required_artifacts": [
            "public_slice_candidate_corpus.json",
            "heldout_public_slice_source_replay.json",
            "fresh_fixed_sum_rank_delta.json",
        ],
    },
    "presurface_full_charge_target_transfer": {
        "hypothesis": (
            "One frozen label-free algebraic factor-stage prefilter preserves every "
            "strict pre-surface relation and reduces fully charged source cost below "
            "rho on unseen generic-prime target families."
        ),
        "decisive_test": (
            "Freeze the prefilter and exact operation ledger on the current two-target "
            "development corpus, materialize it unchanged on at least two new target "
            "families, and require every split positive, full strict recall, full Sage "
            "factorization and rejection charges below rho, and positive fresh rank."
        ),
        "falsifier": (
            "Any held-out family loses a strict relation, has no positive surface, "
            "costs at least rho after full source charging, or adds no fresh rank."
        ),
        "required_artifacts": [
            "frozen_factor_stage_prefilter.json",
            "prospective_target_family_materialization.json",
            "full_source_operation_ledger.json",
            "fresh_fixed_sum_rank_delta.json",
        ],
    },
    "scalar_blind_fixed_sum_source_generator": {
        "hypothesis": (
            "A scalar-blind source acting before selected-leaf materialization can "
            "emit genuinely new independent fixed-sum rows below direct "
            "pair-complement enumeration."
        ),
        "decisive_test": (
            "Freeze a public candidate generator before verifier rows or selected "
            "leaves exist, replay it on at least four target families, deduplicate "
            "every verified row against the complete source ledger, and charge the "
            "full source, rank, and descent path."
        ),
        "falsifier": (
            "The source consumes selected-leaf or verifier labels, emits no fresh "
            "independent row, or costs at least direct pair-complement enumeration "
            "or rho on any held-out family."
        ),
        "required_artifacts": [
            "scalar_blind_fixed_sum_candidate_stream.json",
            "source_ledger_dedup_and_rank.json",
            "full_source_cost_vs_pair_complement.json",
            "prospective_four_family_descent.json",
        ],
    },
    "structured_closure_collision_locator": {
        "hypothesis": (
            "A public coordinate-defined source concentrates independently "
            "rank-reducing closure collisions and locates them before pair-graph "
            "materialization."
        ),
        "decisive_test": (
            "Freeze the coordinate set and collision locator before outcomes, "
            "replay it on at least four generic-prime target families, and require "
            "positive independent collision rank plus complete source, linear "
            "algebra, and target-descent cost below rho on every family."
        ),
        "falsifier": (
            "Fresh constructive rows merely introduce new atoms, collision density "
            "matches the uniform residual model, the locator materializes the pair "
            "graph, or any fully charged family costs at least rho."
        ),
        "required_artifacts": [
            "frozen_coordinate_closure_source.json",
            "closure_collision_locator_replay.json",
            "collision_rank_vs_fresh_residual_rank.json",
            "four_family_full_cost_and_descent.json",
        ],
    },
    "s6_unit_zero_divisor_source_router": {
        "hypothesis": (
            "The five-label S6 predicate has an exact target-symbolic "
            "unit-or-zero-divisor interface that avoids the full centered-carry "
            "and CRT state while preserving one occurrence-labelled source."
        ),
        "decisive_test": (
            "Freeze one branch-complete representation before target outcomes, "
            "certify empty fibers, preserve a positive dyadic child, recover one "
            "verified signed source, and charge target-independent state, target "
            "specialization, every failed child, exact zero testing, integer "
            "heights, and setup/online/workspace costs."
        ),
        "falsifier": (
            "The interface materializes the canonical centered carry, a B^3 "
            "balanced source table, the B^5 CRT quotient, presupposes the root, "
            "or exceeds B^(9/4) setup/state or B^(5/4) online/workspace."
        ),
        "required_artifacts": [
            "s6_unit_or_zero_divisor_representation.json",
            "empty_and_positive_fiber_certificates.json",
            "dyadic_source_recovery_replay.json",
            "integer_height_and_direct_cost_ledger.json",
        ],
    },
    "s6_noncp_balanced_trace_contraction": {
        "hypothesis": (
            "The exact restricted projector trace can be contracted on the "
            "actual S6 pair/triple signature images without representing a "
            "centered carry, dense character table, or B^3 triple incidence set."
        ),
        "decisive_test": (
            "Freeze one non-CP factorized-circuit grammar and compute exact "
            "full-box and adaptive-child counts for blind and positive targets, "
            "including zero signatures, integer multiplicities, one occurrence "
            "backpointer, and R10's rank-two sparse multiplicative-convolution "
            "control, inside the direct setup/state and online/workspace caps."
        ),
        "falsifier": (
            "The grammar emits a B^3 source table, a B^5 quotient, N character "
            "modes, either centered carry, a root-presupposing projector factor, "
            "or loses rectangle identity, exact counts, or a joint source."
        ),
        "required_artifacts": [
            "frozen_noncp_trace_circuit_grammar.json",
            "restricted_projector_trace_replay.json",
            "rank_two_sparse_convolution_control.json",
            "dyadic_joint_source_and_direct_cost_ledger.json",
        ],
    },
    "s6_quotient_algebra_trace_transducer": {
        "hypothesis": (
            "A quotient-algebra trace transducer built from dyadic unary "
            "subproduct trees evaluates the actual S6 restricted trace without "
            "forming a degree-B^2 target polynomial, a degree-B^4 parametric "
            "resultant, or a B^3 triple-signature deck."
        ),
        "decisive_test": (
            "Freeze the transducer grammar before target outcomes and replay the "
            "R73 duplicate-multiplicity rank-two batch, zero signatures, blind "
            "and positive targets, adaptive children, and one joint source; then "
            "run the actual S6 pair/triple pullback inside B^(9/4) setup/state "
            "and B^(5/4) online/workspace."
        ),
        "falsifier": (
            "Any transposed or quotient operation first represents the "
            "degree-B^2 specialized polynomial, degree-B^4 product resultant, "
            "B^3 triple deck, loses exact valuation multiplicity or rectangle "
            "identity, or presupposes a positive source."
        ),
        "required_artifacts": [
            "frozen_quotient_trace_transducer_grammar.json",
            "rank_two_valuation_batch_replay.json",
            "actual_s6_quotient_trace_replay.json",
            "transposed_state_source_and_cost_ledger.json",
        ],
    },
    "s6_support_adaptive_transposed_incidence": {
        "hypothesis": (
            "The sparse nonempty intersections between S4 prefix and suffix "
            "supports can be constructed directly, without enumerating the B^3 "
            "prefix radicals, the B^5 incidence grid, or an outcome-conditioned "
            "oracle diagram."
        ),
        "decisive_test": (
            "Freeze one support-adaptive transposed incidence grammar before "
            "targets, certify every blind zero, recover every forced source, "
            "preserve R73 duplicate multiplicities and adaptive children, and "
            "charge all setup, failed searches, source selection, online work, "
            "and workspace against the B^(9/4)/B^(5/4) caps."
        ),
        "falsifier": (
            "The constructor emits B^3 prefix residuals, scans B^5 incidences, "
            "uses measured positive signatures as advice, loses an empty-fiber "
            "certificate or occurrence multiplicity, or exceeds either direct cap."
        ),
        "required_artifacts": [
            "frozen_support_adaptive_incidence_grammar.json",
            "blind_zero_and_forced_source_replay.json",
            "r73_multiplicity_and_dyadic_control.json",
            "failed_search_source_and_direct_cost_ledger.json",
        ],
    },
    "s6_transposed_norm_scalar_functional": {
        "hypothesis": (
            "A transposed scalar functional can compute the exact suffix gcd "
            "or zero certificate directly from the factored unary S4 norm "
            "without representing the full trivariate coefficient body or a "
            "B^3 evaluation vector."
        ),
        "decisive_test": (
            "Freeze the scalar-functional grammar before targets, replay blind "
            "zero and forced-positive instances, and require R73 duplicate "
            "multiplicity, zero strata, adaptive children, and one joint source "
            "inside B^(9/4) setup/state and B^(5/4) online/workspace."
        ),
        "falsifier": (
            "Transposition constructs the (4B+1)^3 coefficient cube, a B^3 "
            "value vector, the degree-B^2 specialized target polynomial, or "
            "returns only a nonconstructive scalar without exact source replay."
        ),
        "required_artifacts": [
            "frozen_transposed_norm_functional_grammar.json",
            "blind_zero_and_positive_gcd_replay.json",
            "r73_source_multiplicity_and_child_replay.json",
            "transposition_input_state_and_direct_cost_ledger.json",
        ],
    },
    "s6_target_translated_subset_frequency_oracle": {
        "hypothesis": (
            "A target-independent compressed frequency oracle for signed "
            "three-point endpoint subsets can answer the target-translated "
            "pair query without enumerating B^3 prefix triples or B^2 suffix "
            "pairs."
        ),
        "decisive_test": (
            "Freeze the advice and target-update grammar before outcomes, bind "
            "the Query2P1 gates, and replay R76 exact occurrence counts, blind "
            "zero, a forced source, duplicate/multiple-root multiplicity, and "
            "every dyadic child inside B^(9/4) setup/state and B^(5/4) "
            "fresh-target work/workspace."
        ),
        "falsifier": (
            "The oracle materializes B^3 prefix occurrences, enumerates B^2 "
            "target suffixes, assumes a positive endpoint or source, loses the "
            "Mobius multiple-root correction or occurrence backpointer, or "
            "exceeds either direct cap."
        ),
        "required_artifacts": [
            "frozen_target_translated_subset_oracle_grammar.json",
            "compressed_prefix_advice_replay.json",
            "blind_zero_forced_source_and_dyadic_replay.json",
            "query2p1_target_update_and_direct_cost_ledger.json",
        ],
    },
    "s6_nonlinear_target_specialized_nested_resultant": {
        "hypothesis": (
            "A nonlinear scalar functional can partially evaluate the nested "
            "actual-S4 deck resultants at a fresh target without constructing "
            "group characters, B^3 prefix values, or B^2 suffix roots."
        ),
        "decisive_test": (
            "Freeze the nested-resultant functional before target outcomes and "
            "run it on the actual S4 factors. Replay R76 exact counts, "
            "multiple-root Mobius correction, blind zero, one forced source, "
            "and every dyadic child inside B^(9/4) setup/state and B^(5/4) "
            "fresh-target work/workspace."
        ),
        "falsifier": (
            "The functional expands a B^3 prefix body, forms a degree-B^2 "
            "target polynomial or its roots, invokes any scalar character, "
            "returns only an outcome bit without source/children, loses R76 "
            "multiplicity, or exceeds either direct cap."
        ),
        "required_artifacts": [
            "frozen_nonlinear_nested_resultant_functional_grammar.json",
            "actual_s4_target_specialization_replay.json",
            "r76_mobius_zero_source_and_child_replay.json",
            "nonlinear_functional_state_and_direct_cost_ledger.json",
        ],
    },
    "s6_scalar_only_black_box_nested_norm": {
        "hypothesis": (
            "A straight-line scalar nested norm of the actual S4 factors can "
            "return the exact target zero count and source transcript without "
            "ever representing tuple values, a tensor core, a coefficient "
            "body, a quotient algebra, or a suffix polynomial."
        ),
        "decisive_test": (
            "Freeze every scalar/product/norm node and its input/output type "
            "before target outcomes. Replay blind zero, forced count/source, "
            "R76 duplicate and multiple-root semantics, and all dyadic "
            "children while charging every failed branch and intermediate "
            "against B^(9/4) setup/state and B^(5/4) fresh-target work/state."
        ),
        "falsifier": (
            "Any node hides B^3 tuple values, a B^5 TT/value body, a dense "
            "norm coefficient cube, a degree-B^2 suffix object, an uncharged "
            "oracle zero bit, or lacks exact count/source/child replay."
        ),
        "required_artifacts": [
            "frozen_scalar_only_nested_norm_slp.json",
            "actual_s4_scalar_zero_count_replay.json",
            "rectangle_source_and_dyadic_transcript.json",
            "per_node_state_work_and_failed_zero_ledger.json",
        ],
    },
    "s6_batched_nested_norm_node_compiler": {
        "hypothesis": (
            "A batched compiler for nested actual-S4 norm nodes can use unary "
            "deck polynomials, subproducts, remainders, or transposition to "
            "beat B^5 leaf work without materializing any forbidden body."
        ),
        "decisive_test": (
            "Freeze one compiler and expand every resultant, subproduct, "
            "remainder, modular-composition, transposed, count, source, and "
            "child operation into field-operation and state receipts. First "
            "require a strict asymptotic improvement over B^5 on blind and "
            "positive paths, then enforce B^(9/4) setup/state and B^(5/4) "
            "fresh-target work/state."
        ),
        "falsifier": (
            "The compiler retains B^5 leaf traffic, creates a B^3 value or "
            "coefficient body, creates a degree-B^2 target object, treats a "
            "norm/resultant as unit cost, loses exact count/source/children, "
            "or exceeds either direct cap."
        ),
        "required_artifacts": [
            "frozen_batched_norm_node_compiler.json",
            "resultant_subproduct_remainder_receipts.json",
            "r76_batched_zero_count_source_replay.json",
            "compiled_node_field_state_cost_ledger.json",
        ],
    },
    "s6_structured_factor_base_endpoint_compression": {
        "hypothesis": (
            "A scalar-blind structured factor-base geometry can compress its "
            "B^3 triple endpoint multiset below B^(9/4+o(1)) and answer a "
            "fresh target-plus-pair query with source return below "
            "B^(5/4+o(1)) without destroying relation density or rank."
        ),
        "decisive_test": (
            "Freeze the factor-base generator, endpoint representation, and "
            "query algorithm before outcomes. Prove the setup and online "
            "bounds, then run matched random controls and require prospective "
            "relation density, full RHS-compatible rank, verified factor "
            "logs, and identical scalar-blind fresh-target descent."
        ),
        "falsifier": (
            "The geometry consumes scalar labels, materializes B^3 prefix or "
            "B^2 target occurrences, exceeds B^(9/4) setup or B^(5/4) online "
            "cost, loses exact source replay, underproduces independent "
            "relations versus matched random controls, or cannot complete "
            "factor logs and the identical target descent."
        ),
        "required_artifacts": [
            "frozen_structured_factor_base_geometry.json",
            "triple_endpoint_compression_and_density_theorem.json",
            "source_query_and_full_cost_ledger.json",
            "matched_random_relation_rank_controls.json",
            "factor_logs_and_identical_target_descent.json",
        ],
    },
    "s6_compact_divisor_factor_base_endpoint_compiler": {
        "hypothesis": (
            "A scalar-blind factor base given by a compact divisor or "
            "straight-line function can compile its triple S4 endpoints "
            "without expanding the cubic endpoint image."
        ),
        "decisive_test": (
            "Freeze the divisor/function circuit, factor-base extraction, "
            "and endpoint compiler before outcomes. Prove at most B^(9/4) "
            "setup/state and B^(5/4) fresh query/source work, then require "
            "prospective density, matched full rank, verified factor logs, "
            "and identical scalar-blind target descent."
        ),
        "falsifier": (
            "The circuit is a renamed one-dimensional field coset or scalar "
            "orbit, consumes DLP labels, materializes C(B+2,3) endpoint "
            "sets, invokes an uncharged root/resultant oracle, exceeds either "
            "cap, loses source provenance, or fails density, rank, logs, or "
            "identical descent."
        ),
        "required_artifacts": [
            "frozen_compact_divisor_factor_base.json",
            "divisor_slp_to_s4_endpoint_compiler.json",
            "prospective_density_and_matched_rank_controls.json",
            "scalar_blind_query_source_cost_ledger.json",
            "verified_factor_logs_and_identical_descent.json",
        ],
    },
    "s6_addition_compatible_5a5c_field_filtration": {
        "hypothesis": (
            "A representation-specific field-coordinate filtration can "
            "compose partial constraints for the colored 5A+5C source and "
            "beat the rho-scale equality join without losing provenance."
        ),
        "decisive_test": (
            "Freeze every coordinate projection, partial constraint, merge "
            "schedule, and source backpointer before outcomes. Require each "
            "filter to compose under elliptic addition, prove B^(9/4) "
            "setup/state and B^(5/4) fresh work/workspace, replay exact "
            "Semaev or FFE sources, and pass matched random-deck controls."
        ),
        "falsifier": (
            "Any filter is only an encoding hash, assumes a nonexistent "
            "prime-order quotient, admits uncharged false positives, loses "
            "a jointly coupled source, exposes B^(5/2) collision work, "
            "exceeds either cap, or fails factor logs and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_coordinate_filtration.json",
            "partial_filter_composability_and_false_positive_controls.json",
            "summation_polynomial_ffe_source_replay.json",
            "full_query_state_cost_ledger.json",
            "factor_logs_and_identical_descent.json",
        ],
    },
    "s6_5a5c_marked_resultant_source_section": {
        "hypothesis": (
            "A P1510-style marked resultant can compile a complete source "
            "section for the R82 2A+3C versus 3A+2C split without explicit "
            "bucket replay or rho-scale endpoint enumeration."
        ),
        "decisive_test": (
            "Freeze the split algebras, marked variables, elimination order, "
            "source selector, and completeness certificate. Expand every "
            "coefficient and operation, require one jointly coupled source "
            "for every accepted target inside B^(9/4) setup and B^(5/4) "
            "fresh work, then complete rank, logs, and identical descent."
        ),
        "falsifier": (
            "The resultant has source degree B^5, a split side above B^(9/4), "
            "an over-cap coefficient body, only aggregate marker moments, "
            "an unproved image-containment claim, multiple uncoupled source "
            "coordinates, rho-scale target work, or incomplete logs/descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_marked_resultant_section.json",
            "resultant_coefficient_and_containment_receipts.json",
            "joint_source_section_replay.json",
            "fresh_query_state_cost_ledger.json",
            "factor_logs_and_identical_descent_r84.json",
        ],
    },
    "s6_5a5c_target_uniform_precoefficient_circuit": {
        "hypothesis": (
            "A target-uniform circuit built directly from the compact "
            "D_A,D_C inputs can specialize the 2A+3C versus 3A+2C join "
            "without emitting either explicit side coefficient body or "
            "provenance-leaf family."
        ),
        "decisive_test": (
            "Freeze every circuit node, represented payload, elimination "
            "map, target-specialization operation, zero test, and source "
            "inverse before outcomes. Charge every field operation and live "
            "word; require exact containment and one jointly coupled source "
            "inside B^(9/4) setup and B^(5/4) fresh work/workspace, then "
            "complete rank, logs, and identical descent."
        ),
        "falsifier": (
            "The circuit emits a B^(2.4) side polynomial, B^(2.4) source "
            "leaves, a hidden resultant/root oracle, an uncharged "
            "circuit-valued coefficient payload, aggregate marker moments "
            "without one source, incomplete exceptional charts, rho-scale "
            "work, or incomplete rank/logs/descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_precoefficient_circuit.json",
            "circuit_node_degree_and_payload_ledger.json",
            "fresh_target_specialization_and_source_replay.json",
            "matched_explicit_coefficient_and_random_controls.json",
            "factor_logs_and_identical_descent_r85.json",
        ],
    },
    "s6_5a5c_sparse_multihomogeneous_moment_recurrence": {
        "hypothesis": (
            "The colored 5A+5C fiber has a target-specialized sparse "
            "multihomogeneous moment recurrence whose public-input "
            "constructor avoids fixed quotient labels and dense Macaulay "
            "or source enumeration."
        ),
        "decisive_test": (
            "Freeze the multigrading, monomial support, recurrence, target "
            "specialization, moment functional, flat-extension test, and "
            "source inverse. Derive every moment from compact D_A,D_C, "
            "charge all represented coordinates and failed targets, fit "
            "B^(9/4) setup and B^(5/4) fresh work/workspace, and replay "
            "reduced, nonreduced, signed, infinity, and exceptional fibers."
        ),
        "falsifier": (
            "The constructor assumes supplied moments, reaches B^3 through "
            "a 2+3 deck, reaches B^5 through source or dense Macaulay "
            "coordinates, hides a common factor/root oracle, proves "
            "flatness without a source biconditional, omits exceptional "
            "fibers, or fails rank, logs, and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_sparse_moment_recurrence.json",
            "multihomogeneous_support_and_regular_degree_receipts.json",
            "target_moment_constructor_and_flat_extension_replay.json",
            "source_biconditional_and_exceptional_fibers.json",
            "factor_logs_and_identical_descent_r86.json",
        ],
    },
    "s6_5a5c_jet_preserving_addition_pushforward_intertwiner": {
        "hypothesis": (
            "The first coloured norm jet can be propagated through the "
            "R82 addition map F=A+C by a compositional intertwiner whose "
            "state never materializes the C^5 or full coloured source "
            "product."
        ),
        "decisive_test": (
            "Freeze the five-slot jet algebra, the A+C pushforward map, all "
            "truncations, target-specialization operations, and source "
            "inverse. Derive the jet from public compact D_A,D_C without "
            "DLP labels, charge every intermediate word and failed target, "
            "fit B^(9/4) setup and B^(5/4) fresh work/workspace, and replay "
            "reduced, nonreduced, signed, infinity, and exceptional fibers."
        ),
        "falsifier": (
            "Any step forms the B^3 five-C support or B^5 coloured quotient, "
            "assumes supplied moments or endpoint labels, hides a root or "
            "norm oracle, loses the jointly coupled factor and atom source, "
            "omits an exceptional chart, or fails rank, logs, and identical "
            "descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_jet_pushforward_intertwiner.json",
            "slotwise_jet_composition_and_truncation_receipts.json",
            "public_input_target_jet_and_source_replay.json",
            "exceptional_chart_and_false_positive_controls.json",
            "factor_logs_and_identical_descent_r87.json",
        ],
    },
    "s6_5a5c_black_box_translated_resultant_gcd_localizer": {
        "hypothesis": (
            "A scalar black-box resultant or implicit half-gcd can test and "
            "localize the unique common endpoint of P_A(X) and implicit "
            "P_C(T-X) without emitting either characteristic polynomial or "
            "the translated remainder orbit."
        ),
        "decisive_test": (
            "Freeze the compact five-slot A,C divisor inputs, scalar "
            "resultant interface, source-localization recursion, target "
            "specialization, multiplicity test, and every intermediate "
            "representation. Charge construction, failed targets, and live "
            "state; fit B^(9/4) setup and B^(5/4) fresh work/workspace, then "
            "replay one jointly coupled source through every projective "
            "exceptional stratum."
        ),
        "falsifier": (
            "The algorithm materializes the B^3 C characteristic polynomial, "
            "a B^2 translated remainder or evaluation vector, assumes a "
            "resultant/root/gcd oracle, scans B^2 A endpoints, loses source "
            "provenance, misses multiplicity or exceptional charts, or "
            "fails rank, logs, and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_black_box_resultant_localizer.json",
            "implicit_resultant_half_gcd_operation_ledger.json",
            "target_specialization_and_source_localization_replay.json",
            "multiplicity_and_projective_exceptional_controls.json",
            "factor_logs_and_identical_descent_r88.json",
        ],
    },
    "s6_5a5c_coefficient_free_fixed_marker_resultant_recurrence": {
        "hypothesis": (
            "The five C decks admit a coefficient-free scalar recurrence "
            "that evaluates the translated norm and all fixed source-marker "
            "derivatives modulo the compact five-A divisor without quotient "
            "vectors, Krylov blocks, or endpoint dictionaries."
        ),
        "decisive_test": (
            "Freeze every scalar recurrence state, deck transition, marker "
            "update, target specialization, multiplicity branch, and source "
            "inverse. Derive all values from compact D_A,D_C, charge every "
            "field operation and live word, fit B^(9/4) setup and B^(5/4) "
            "fresh work/workspace, and replay reduced, nonreduced, signed, "
            "infinity, tangent, and exceptional fibers."
        ),
        "falsifier": (
            "Any transition emits P_C coefficients, a B^2 quotient/value "
            "vector or Krylov block, a C endpoint/source dictionary, assumes "
            "a determinant/resultant/root oracle, loses fixed-marker source "
            "recovery, omits multiplicity or exceptional charts, or fails "
            "rank, logs, and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_fixed_marker_scalar_recurrence.json",
            "coefficient_free_marker_transition_ledger.json",
            "scalar_norm_marker_and_source_replay.json",
            "projective_multiplicity_exceptional_controls.json",
            "factor_logs_and_identical_descent_r89.json",
        ],
    },
    "s6_5a5c_nonlocal_nonlinear_translation_sketch": {
        "hypothesis": (
            "The fixed five-C norm and marker family admits a target-"
            "independent nonlinear sketch whose deck updates and target "
            "translations avoid both the explicit shift orbit and the "
            "target-local first-jet obstruction."
        ),
        "decisive_test": (
            "Freeze the nonlinear state, public deck-update law, target-"
            "translation law, zero and multiplicity tests, and source inverse "
            "before outcomes. Prove setup/state at most B^(9/4), fresh "
            "translation and source return at most B^(5/4), and no shift-value, "
            "coefficient, quotient, endpoint, or source table; replay exact "
            "reduced, multiple, nonreduced, signed, infinity, tangent, and "
            "exceptional branches."
        ),
        "falsifier": (
            "The state is target-local, has an R89 equal-state/different-shift "
            "collision, linearizes to a full translated-remainder orbit, "
            "materializes C^4 or larger slot support, assumes a root/resultant "
            "oracle, loses a jointly coupled source, exceeds either cap, or "
            "fails rank, logs, and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_nonlocal_translation_sketch.json",
            "nonlinear_translation_state_and_update_ledger.json",
            "target_translation_marker_and_source_replay.json",
            "projective_exceptional_branch_controls.json",
            "factor_logs_and_identical_descent_r90.json",
        ],
    },
    "s6_5a5c_unequal_list_subfunction_inversion_index": {
        "hypothesis": (
            "An unequal-list specialization of subfunction inversion can "
            "index the B^2 five-A endpoints against the B^3 five-C "
            "endpoints with exact finite-field source reporting inside the "
            "P1515 setup and fresh-query rectangle."
        ),
        "decisive_test": (
            "Freeze the five-A/five-C list decomposition, subfunctions, "
            "preprocessed state, random coins or deterministic replacement, "
            "query algorithm, source reporter, and every memory word. Derive "
            "the setup, query, reporting, and success-amplification exponents "
            "from the explicit construction; require B^(9/4) setup/state and "
            "B^(5/4) fresh work/workspace with exact signed source and "
            "exceptional-branch replay."
        ),
        "falsifier": (
            "No parameter point meets both caps; preprocessing stores B^3 C "
            "endpoints or B^(5/2) pair data; the query is membership-only, "
            "assumes random-oracle or verifier labels, omits reporting or "
            "amplification cost, is not valid over the finite-field addition "
            "law, loses exceptional branches, or fails rank, logs, and "
            "identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_unequal_list_subfunction_index.json",
            "subfunction_inversion_exponent_ledger.json",
            "finite_field_source_reporting_replay.json",
            "exceptional_branch_and_matched_random_controls.json",
            "factor_logs_and_identical_descent_r91.json",
        ],
    },
    "s6_5a5c_compact_elliptic_subfunction_map": {
        "hypothesis": (
            "The compact A/C divisor circuits admit public elliptic "
            "MAP1, MAP2, f_d, and source-translation operations that reduce "
            "one 5A+5C target to a single subfunction without integer DLP "
            "labels or explicit endpoint lists."
        ),
        "decisive_test": (
            "Freeze the projective coordinate maps, subfunction index set, "
            "compact f_d circuit, target maps, source translator, random "
            "coins or deterministic replacement, and every exceptional "
            "chart before outcomes. Prove each operation directly from "
            "D_A,D_C and one target, fit B^(9/4) setup/state and B^(5/4) "
            "fresh work/workspace, and replay one exact signed source or "
            "bottom on every reduced and nonreduced branch."
        ),
        "falsifier": (
            "Any map is a fixed coordinate bucket or proper quotient, uses "
            "DLP labels, materializes an endpoint/source list, norm/moment "
            "body, or target-offset table, requires more than one "
            "subfunction, hides an elimination/root oracle, exceeds either "
            "cap, loses projective source provenance, or fails rank, logs, "
            "and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_compact_elliptic_subfunction_map.json",
            "elliptic_map1_map2_fd_tr_ledger.json",
            "compact_divisor_target_source_replay.json",
            "projective_branch_and_random_group_controls.json",
            "factor_logs_and_identical_descent_r92.json",
        ],
    },
    "s6_5a5c_shared_semilinear_incidence_correspondence": {
        "hypothesis": (
            "A target-dependent family of overlapping S3/S4 incidence "
            "charts admits one shared semilinear operator whose joint state "
            "compresses across charts and returns an exact 5A+5C source "
            "without partitioning all endpoints into independent fibers."
        ),
        "decisive_test": (
            "Freeze the incidence correspondence, chart overlaps, shared "
            "operator, target specialization, source unranking map, and "
            "every projective exceptional branch before outcomes. Prove "
            "joint setup/state at most B^(9/4), fresh work/workspace at most "
            "B^(5/4), and exact one-source-or-bottom replay without charging "
            "D independent function-inversion tables."
        ),
        "falsifier": (
            "The charts are disjoint endpoint buckets in disguise, the "
            "operator decomposes into D independent advice tables, an "
            "overlap multiplicity or elimination step restores B^(5/2) or "
            "larger state, source unranking uses an endpoint/source "
            "dictionary or verifier oracle, any projective branch is lost, "
            "or rank, logs, and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_shared_semilinear_incidence_correspondence.json",
            "overlapping_chart_operator_compression_ledger.json",
            "compact_source_unranking_replay.json",
            "projective_overlap_and_false_positive_controls.json",
            "factor_logs_and_identical_descent_r93.json",
        ],
    },
    "s6_5a5c_implicit_veronese_hyperplane_source_index": {
        "hypothesis": (
            "The Cartesian A/C source circuits support an implicit algebraic "
            "range index for the rank-six Veronese resultant kernel, so an "
            "exact zero hyperplane and coupled source can be found before "
            "the B^(12/5) root feature rows are emitted."
        ),
        "decisive_test": (
            "Freeze the implicit source parameterization, Veronese query, "
            "preprocessed index, source reporter, and all exceptional charts "
            "before outcomes. Charge construction, field operations, "
            "workspace, false-positive replay, and source unranking; require "
            "B^(9/4) setup/state and B^(5/4) fresh work/workspace."
        ),
        "falsifier": (
            "The index materializes B^(12/5) source features, scans a full "
            "root side, expands a hyperplane into an over-cap polynomial or "
            "Macaulay system, uses a verifier/root oracle, loses coupled "
            "source provenance or any projective branch, or fails rank, "
            "logs, and identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_implicit_veronese_hyperplane_index.json",
            "implicit_cartesian_range_index_cost_ledger.json",
            "coupled_source_unranking_and_false_positive_replay.json",
            "projective_hyperplane_exceptional_controls.json",
            "factor_logs_and_identical_descent_r94.json",
        ],
    },
    "s6_5a5c_aggregate_veronese_projector_recurrence": {
        "hypothesis": (
            "The Cartesian A/C source circuits admit an exact aggregate "
            "recurrence for the Fermat projector trace "
            "sum_box(1-H^(p-1)), allowing zero counting and one coupled "
            "source to be recovered without evaluating every B^(12/5) "
            "root-side value."
        ),
        "decisive_test": (
            "Freeze the projector recurrence grammar, compact A/C input "
            "states, every intermediate tensor or trace state, dyadic range "
            "restriction, source reporter, and projective exceptional "
            "branches before outcomes. Prove exact count and one-source-or-"
            "bottom semantics, B^(9/4) setup/state, and B^(5/4) fresh "
            "work/workspace without individual root-side evaluations."
        ),
        "falsifier": (
            "Expansion of H^(p-1), polarization, tensor contraction, "
            "character sums, or range restriction creates B^(12/5) values, "
            "over-cap support or state, source replay rescans a root side or "
            "uses a verifier/root oracle, any blind-zero, infinity, proper-"
            "subsum, tangent, or multiplicity branch is lost, or rank, logs, "
            "and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_aggregate_veronese_projector_recurrence.json",
            "projector_trace_recurrence_state_ledger.json",
            "dyadic_projector_count_and_source_replay.json",
            "projective_trace_exceptional_controls.json",
            "factor_logs_and_identical_descent_r95.json",
        ],
    },
    "s6_5a5c_modular_frobenius_trace_recurrence": {
        "hypothesis": (
            "A modular trace recurrence can use H^p=H on finite-field "
            "values to aggregate the Fermat zero projector over compact "
            "Cartesian A/C source circuits without storing the B^10 "
            "degree-2(p-1) moment vectors."
        ),
        "decisive_test": (
            "Freeze the quotient or trace algebra, Frobenius transition, "
            "compact source-state constructor, target update, range "
            "restriction, integer count lift, source reporter, and every "
            "exceptional branch before outcomes. Prove each transition "
            "symbolically and require B^(9/4) setup/state and B^(5/4) "
            "fresh work/workspace with exact one-source-or-bottom replay."
        ),
        "falsifier": (
            "The trace is only pointwise Fermat powering, materializes "
            "degree-2(p-1) moments, a quotient algebra of over-cap "
            "dimension, B^(12/5) root values, or an endpoint/source table; "
            "range restriction or integer lifting rescans a side, source "
            "replay uses a root/verifier oracle, any exceptional branch is "
            "lost, or rank, logs, and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_modular_frobenius_trace_recurrence.json",
            "frobenius_trace_state_and_transition_ledger.json",
            "integer_lift_dyadic_source_replay.json",
            "quotient_trace_exceptional_controls.json",
            "factor_logs_and_identical_descent_r96.json",
        ],
    },
    "s6_5a5c_factored_transposed_projector_trace": {
        "hypothesis": (
            "A factored transposed trace can evaluate the Fermat projector "
            "and one range-restricted child directly on the compact A/C "
            "divisor circuits without materializing the B^(12/5) split "
            "quotient basis or losing source occurrence multiplicity."
        ),
        "decisive_test": (
            "Freeze the factored divisor circuit, transposed trace identity, "
            "target update, range idempotent or restriction operator, "
            "integer lift, source reporter, and exceptional branches before "
            "outcomes. Derive every state and operation count from the "
            "circuit; require B^(9/4) setup/state and B^(5/4) fresh "
            "work/workspace with exact one-source-or-bottom replay."
        ),
        "falsifier": (
            "Transposition emits the root polynomial, quotient basis, "
            "degree-2(p-1) moments, endpoint/source rows, or B^(12/5) "
            "adjoints; range restriction needs source-sized idempotents or "
            "a side scan; multiplicity is radicalized away; any branch, "
            "integer lift, rank, logs, or identical descent fails."
        ),
        "required_artifacts": [
            "frozen_5a5c_factored_transposed_projector_trace.json",
            "factored_trace_state_and_adjoint_ledger.json",
            "range_idempotent_integer_source_replay.json",
            "transposed_trace_exceptional_controls.json",
            "factor_logs_and_identical_descent_r97.json",
        ],
    },
    "s6_5a5c_nonlinear_tensor_tower_trace": {
        "hypothesis": (
            "A nonlinear tensor-tower trace can construct the Fermat "
            "projector and one range-restricted child directly from compact "
            "A/C divisor-circuit node states without exposing a "
            "B^(12/5) source, quotient, moment, or adjoint body."
        ),
        "decisive_test": (
            "Freeze the tensor algebra, compact node-state constructor, "
            "contraction and target transition, range restriction, integer "
            "lift, source reporter, and exceptional branches before "
            "outcomes. Derive every state and operation count and require "
            "B^(9/4) setup/state and B^(5/4) fresh work/workspace with exact "
            "one-source-or-bottom replay."
        ),
        "falsifier": (
            "Any node emits source leaves, endpoint rows, quotient bases, "
            "degree-2(p-1) moments, linearized dyadic adjoints, or an "
            "equivalent B^(12/5) body; tensor rank or transition state "
            "exceeds a cap; source replay scans a side or uses a verifier; "
            "multiplicity or another exceptional branch is lost; or rank, "
            "logs, and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_nonlinear_tensor_tower_trace.json",
            "nonlinear_tensor_state_and_transition_ledger.json",
            "tensor_tower_integer_source_replay.json",
            "tensor_tower_exceptional_controls.json",
            "factor_logs_and_identical_descent_r98.json",
        ],
    },
    "s6_5a5c_multiedge_digitized_equality_projector": {
        "hypothesis": (
            "Several small algebraic channels can encode each prime-field "
            "bridge value so equality is a product of low-width channel "
            "equalities, while the channel extractor acts directly on "
            "compact A/C divisor circuits below both caps."
        ),
        "decisive_test": (
            "Freeze the channel alphabet, algebraic extractor, reconstruction "
            "or injectivity proof, per-edge projector, total cut capacity, "
            "target transition, integer lift, source reporter, and every "
            "exceptional branch before outcomes. Charge all lookup, "
            "interpolation, bit/digit extraction, and composition state."
        ),
        "falsifier": (
            "The extractor needs a p-entry table, degree-p interpolation, "
            "source scan, nonuniform advice, or over-cap circuit/state; the "
            "combined channels are not injective; total cut capacity or "
            "fresh work exceeds a cap; occurrence multiplicity or another "
            "branch is lost; or rank, logs, and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_multiedge_digitized_equality_projector.json",
            "digit_extractor_state_and_cut_capacity_ledger.json",
            "digitized_projector_integer_source_replay.json",
            "digitized_projector_exceptional_controls.json",
            "factor_logs_and_identical_descent_r99.json",
        ],
    },
    "s6_5a5c_succinct_aggregate_digit_trie": {
        "hypothesis": (
            "The compact A/C divisor circuits admit a recursively composable "
            "digit-fiber summary whose total node state is below B^(9/4) "
            "and whose fresh target traversal returns exact count and one "
            "occurrence below B^(5/4)."
        ),
        "decisive_test": (
            "Freeze the digit/fiber arithmetic circuit, leaf-free summary "
            "state, merge law, target traversal, integer lift, occurrence "
            "reporter, and exceptional branches before outcomes. Prove "
            "summary sufficiency and charge every coefficient, lookup, "
            "node, source restriction, and failed-target traversal."
        ),
        "falsifier": (
            "Summary construction or merge emits D source digits, a p-size "
            "fiber table, endpoint rows, quotient/moment bodies, or an "
            "equivalent over-cap state; target traversal scans a side or "
            "uses a verifier; occurrence multiplicity or another branch is "
            "lost; or rank, logs, and identical descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_succinct_aggregate_digit_trie.json",
            "aggregate_digit_trie_state_transition_ledger.json",
            "aggregate_digit_trie_integer_source_replay.json",
            "aggregate_digit_trie_exceptional_controls.json",
            "factor_logs_and_identical_descent_r100.json",
        ],
    },
    "s6_5a5c_actual_divisor_image_entropy_merge": {
        "hypothesis": (
            "The R84 3A+2C endpoint image is either compressible by a "
            "leaf-free merge law acting on D_A,D_C or contains a reachable "
            "distinguishable family large enough to force B^(12/5) "
            "persistent state."
        ),
        "decisive_test": (
            "Freeze the public A,C parameter family, endpoint-key map, "
            "allowed divisor operations, summary state, merge law, target "
            "query, occurrence reporter, and exceptional branches before "
            "outcomes. Prove the summary on all reachable instances or "
            "construct an explicit injective reachable subfamily and charge "
            "its parameter and output entropy."
        ),
        "falsifier": (
            "The compression uses endpoint/source leaves, p-size advice, "
            "verifier labels, or over-cap coefficients; the entropy family "
            "counts arbitrary sets not reachable from D_A,D_C, charges "
            "input parameters as free, or lacks injectivity; query/source "
            "replay scans a side; or rank, logs, and descent fail."
        ),
        "required_artifacts": [
            "frozen_5a5c_actual_divisor_image_entropy_merge.json",
            "actual_divisor_image_parameter_entropy_ledger.json",
            "actual_divisor_leaf_free_merge_source_replay.json",
            "actual_divisor_image_exceptional_controls.json",
            "factor_logs_and_identical_descent_r101.json",
        ],
    },
    "s6_5a5c_two_sided_implicit_join": {
        "hypothesis": (
            "The exact local 2A+3C and 3A+2C source oracles admit a shared "
            "target-translated intersection summary below B^(9/4) state "
            "whose fresh query returns exact 5A+5C multiplicity and one "
            "jointly coupled source below B^(5/4) work and workspace."
        ),
        "decisive_test": (
            "Freeze the shared state, target transition, intersection "
            "operator, integer lift, coupled source reporter, and every "
            "exceptional branch before outcomes. Replay against direct "
            "two-sided enumeration and charge every local-oracle query, "
            "translated key, collision bucket, and failed target."
        ),
        "falsifier": (
            "The join enumerates either B^(13/5) or B^(12/5) side, stores "
            "endpoint/source leaves, uses target-specific preprocessing or "
            "a verifier/root oracle, loses joint multiplicity or source "
            "coupling, exceeds either cap, or fails rank, factor logs, and "
            "identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_two_sided_implicit_join.json",
            "two_sided_join_state_and_transition_ledger.json",
            "two_sided_join_integer_source_replay.json",
            "two_sided_join_exceptional_controls.json",
            "factor_logs_and_identical_descent_r102.json",
        ],
    },
    "s6_5a5c_target_forced_algebraic_join_filter": {
        "hypothesis": (
            "A public rational invariant of L,R,T and a target-forced "
            "Semaev/FFE relation preserve every true L+R=T join while "
            "pruning false pairs compositionally from compact D_A,D_C "
            "state below both direct caps."
        ),
        "decisive_test": (
            "Freeze the invariant, necessary algebraic relation, compact "
            "pushforward state, target specialization, false-positive "
            "eliminator, integer lift, coupled source reporter, and all "
            "exceptional charts before outcomes. Prove every true source "
            "survives and charge all polynomial, FFE, resultant, and "
            "source-unranking operations."
        ),
        "falsifier": (
            "The filter thins any true source, materializes B^(11/5) or "
            "larger endpoint rows and still scans B^(14/5), emits an "
            "over-cap polynomial/FFE body, uses target-specific setup or a "
            "root/verifier oracle, retains over-cap false positives, loses "
            "joint multiplicity/source, or fails rank, logs, and descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_target_forced_algebraic_join_filter.json",
            "target_forced_invariant_and_ffe_ledger.json",
            "target_forced_join_integer_source_replay.json",
            "target_forced_join_exceptional_controls.json",
            "factor_logs_and_identical_descent_r103.json",
        ],
    },
    "s6_5a5c_compact_preendpoint_s3_ffe_pushdown": {
        "hypothesis": (
            "A sign-resolved target-specialized S3/FFE recurrence can be "
            "pushed through the compact D_A,D_C addition circuit before "
            "partial endpoint emission, preserving exact integer count and "
            "one coupled source below both direct caps."
        ),
        "decisive_test": (
            "Freeze the recurrence grammar, sign marker, target transition, "
            "FFE state, source-unranking adjoint, and projective charts "
            "before outcomes. Expand every node into field operations and "
            "compare exact count/source output with direct enumeration on "
            "positive, blind, repeated-root, and infinity controls."
        ),
        "falsifier": (
            "Any node emits B^(11/5) provenance leaves followed by a "
            "B^(14/5) scan, a B^(12/5) endpoint polynomial/factor list, "
            "an over-cap tensor or quotient body, an uncharged root/"
            "resultant oracle, or loses sign, multiplicity, source coupling, "
            "an exceptional chart, rank, logs, or identical descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_compact_preendpoint_s3_ffe_pushdown.json",
            "preendpoint_pushdown_recurrence_and_cost_ledger.json",
            "preendpoint_s3_ffe_integer_source_replay.json",
            "preendpoint_pushdown_exceptional_controls.json",
            "factor_logs_and_identical_descent_r104.json",
        ],
    },
    "s6_5a5c_actual_deck_nonmergeable_target_pullback_circuit": {
        "hypothesis": (
            "The frozen actual D_A,D_C straight-line program admits a "
            "target-specialized nonlinear circuit that consumes the whole "
            "deck program without exposing a mergeable endpoint/residual "
            "child state and returns exact count and one source below both "
            "direct caps."
        ),
        "decisive_test": (
            "Freeze the allowed arithmetic/FFE gate grammar, inject T at "
            "the root, expand every gate and adjoint operation, and compare "
            "exact integer count/source output against direct enumeration "
            "on all actual and matched random decks plus every projective "
            "and multiplicity control."
        ),
        "falsifier": (
            "The circuit factors through a universal mergeable child state, "
            "emits B^(14/5) residuals or B^(12/5) endpoint factors, uses an "
            "uncharged root/resultant/source oracle, is fixture-specific, "
            "loses count/source/chart exactness, or fails rank, logs, "
            "identical descent, and the full charged workspace bound."
        ),
        "required_artifacts": [
            "frozen_5a5c_actual_deck_nonmergeable_target_pullback.json",
            "actual_deck_target_pullback_circuit_ledger.json",
            "actual_deck_target_pullback_integer_source_replay.json",
            "actual_deck_target_pullback_exceptional_controls.json",
            "factor_logs_and_identical_descent_r105.json",
        ],
    },
    "s6_5a5c_scalar_target_norm_count_circuit": {
        "hypothesis": (
            "A target-injected scalar norm/count straight-line program "
            "acts on compact D_A,D_C, supports the eleven R105 marker "
            "deformations, and returns exact target-fiber multiplicity "
            "inside both caps without materialized source-domain state."
        ),
        "decisive_test": (
            "Freeze the scalar gate grammar, target injection, norm or "
            "resultant identity, integer lift, marker channels, and every "
            "intermediate dimension before outcomes. Expand the circuit on "
            "actual and matched random decks and replay the R105 marker jet "
            "without source-factor enumeration."
        ),
        "falsifier": (
            "The scalar circuit emits B^(14/5) residuals, reads or translates "
            "B^(11/5) coefficients per target, expands B^5 source factors, "
            "uses a unit-cost determinant/root oracle, cannot support the "
            "marker deformations, lacks a generic multiplicity/no-wrap gate, "
            "or fails rank, logs, identical descent, and charged workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_scalar_target_norm_count_circuit.json",
            "scalar_target_norm_gate_and_cost_ledger.json",
            "scalar_target_norm_marker_jet_replay.json",
            "scalar_target_norm_exceptional_controls.json",
            "factor_logs_and_identical_descent_r106.json",
        ],
    },
    "s6_5a5c_noncharacter_algebraic_target_norm_resultant_circuit": {
        "hypothesis": (
            "A non-character target-injected algebraic norm/resultant "
            "straight-line program evaluates the canonical scalar count "
            "and the R105 marker jets directly from compact atom divisors "
            "inside both caps."
        ),
        "decisive_test": (
            "Freeze every nested resultant/norm gate, target occurrence, "
            "quotient-free transpose, marker deformation, integer lift, "
            "and intermediate degree before outcomes. Expand the exact "
            "circuit on actual and matched random decks and compare against "
            "R105 counts, marker factors, and projective branches."
        ),
        "falsifier": (
            "Any route exposes B^(14/5) residuals, B^(11/5) per-target "
            "coefficient reads, B^5 Macaulay/source factors, B^2 or larger "
            "fresh quotient/Krylov state, character/DLP coordinates, or an "
            "uncharged determinant/root oracle; or it loses marker, "
            "multiplicity, integer lift, rank, logs, descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_noncharacter_algebraic_target_norm.json",
            "noncharacter_norm_resultant_gate_ledger.json",
            "noncharacter_norm_marker_jet_replay.json",
            "noncharacter_norm_exceptional_controls.json",
            "factor_logs_and_identical_descent_r107.json",
        ],
    },
    "s6_5a5c_factored_elliptic_lambda_ring_chow_norm_circuit": {
        "hypothesis": (
            "The canonical 5A+5C target norm has a factored elliptic "
            "lambda-ring or Chow-form recurrence whose Adams operations "
            "act directly on compact D_A,D_C and preserve canonical "
            "multiplicity without explicit endpoint coefficient bodies."
        ),
        "decisive_test": (
            "Freeze the lambda/Adams identities, Chow evaluations, target "
            "injection, canonical partition-weight correction, marker "
            "deformations, projective charts, and every intermediate "
            "dimension before outcomes. Expand on actual and matched decks "
            "and compare exact canonical counts and R105 source factors."
        ),
        "falsifier": (
            "Any recurrence materializes B^(13/5) side coefficients, "
            "subresultant/cofactor vectors, B^5 source or Macaulay bodies, "
            "uses character/DLP coordinates or unit-cost norms, returns "
            "partition-weighted rather than canonical count, or loses "
            "markers, multiplicity, integer lift, charts, rank, logs, "
            "descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_factored_elliptic_lambda_ring_chow_norm.json",
            "lambda_ring_chow_gate_and_cost_ledger.json",
            "canonical_weight_marker_jet_replay.json",
            "lambda_ring_chow_exceptional_controls.json",
            "factor_logs_and_identical_descent_r108.json",
        ],
    },
    "s6_5a5c_poincare_theta_target_section_rank": {
        "hypothesis": (
            "After a frozen Poincare trivialization, the target section of "
            "the ten-input elliptic sum pullback has bounded one-body/"
            "pairwise tensor rank, so contractions over the compact A and C "
            "decks fit both caps."
        ),
        "decisive_test": (
            "Freeze a concrete theta basis, line-bundle trivializations, "
            "addition formulas, section tensor decomposition, and all "
            "exceptional charts. Compute exact flattening ranks on actual "
            "and matched deck restrictions, then charge every one-body and "
            "pairwise contraction and the 14400 canonical marker lift."
        ),
        "falsifier": (
            "The proposed identity factors only the line-bundle class, "
            "requires section rank growing beyond the caps, materializes "
            "B^(13/5) or B^5 state, uses analytic or DLP coordinates without "
            "exact finite-field transfer, or loses canonical weights, "
            "markers, multiplicity, charts, rank, logs, or descent."
        ),
        "required_artifacts": [
            "frozen_5a5c_poincare_theta_target_section.json",
            "poincare_theta_section_rank_ledger.json",
            "poincare_theta_canonical_marker_replay.json",
            "poincare_theta_exceptional_controls.json",
            "factor_logs_and_identical_descent_r109.json",
        ],
    },
    "s6_5a5c_theta_addition_cancellation_network": {
        "hypothesis": (
            "An exact finite-field theta-addition circuit generates the "
            "high-rank target section implicitly with bounded intermediate "
            "bond state, and its all-source norm contraction factors inside "
            "both caps after rational pole cancellation."
        ),
        "decisive_test": (
            "Freeze a finite-field theta basis, every rational addition "
            "gate, pole divisor, projective trivialization, bond dimension, "
            "and contraction order. Expand exact actual and matched values, "
            "audit cancellations symbolically, and compare zero order and "
            "R108 markers against direct canonical fibers."
        ),
        "falsifier": (
            "Any gate uses analytic-only coordinates, outcome-dependent "
            "poles, uncharged division by zero, B^(12/5) separated state, "
            "B^(13/5) resultant state, B^5 source expansion, or a norm/"
            "determinant oracle; or it loses zero biconditional, weight "
            "14400, markers, multiplicity, charts, rank, logs, descent, or "
            "workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_theta_addition_cancellation_network.json",
            "theta_addition_bond_and_contraction_ledger.json",
            "theta_addition_canonical_marker_replay.json",
            "theta_addition_pole_exceptional_controls.json",
            "factor_logs_and_identical_descent_r110.json",
        ],
    },
    "s6_5a5c_finite_deck_alternant_annihilator_contraction": {
        "hypothesis": (
            "The target-specialized determinant values on the finite "
            "position-separated decks admit a compact annihilator or "
            "subset-stable existence recurrence whose contractions avoid "
            "both the global B^5 row mode and the B^(12/5) incidence."
        ),
        "decisive_test": (
            "Freeze every finite-deck determinant value set, annihilator "
            "polynomial or recurrence, target update, contraction state, "
            "integer lift, and dyadic source-unranking step. Replay the two "
            "R105 double fibers, blind targets, repeated atoms, and matched "
            "decks against direct canonical sources."
        ),
        "falsifier": (
            "The construction evaluates the B^5 source body, materializes "
            "B^(12/5) endpoint or zero-incidence state, restores an "
            "11(p-1) global row mode, uses a supplied determinant value or "
            "DLP label, or loses the zero biconditional, weight 14400, "
            "markers, multiplicity, rank, logs, descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_finite_deck_alternant_annihilator.json",
            "finite_deck_annihilator_contraction_ledger.json",
            "finite_deck_annihilator_source_replay.json",
            "finite_deck_annihilator_exceptional_controls.json",
            "factor_logs_and_identical_descent_r111.json",
        ],
    },
    "s6_5a5c_gauge_normalized_endpoint_query2p1_index": {
        "hypothesis": (
            "Quotienting the nonzero alternant gauge factors collapses the "
            "raw determinant channel to an endpoint predicate, and a typed "
            "Query2P1 index answers target-specialized membership and "
            "returns a subset-stable canonical source inside both caps."
        ),
        "decisive_test": (
            "Freeze the gauge quotient on every complete chart, the "
            "target-update state, endpoint membership representation, "
            "subset existence bit, and dyadic source-unranking transcript. "
            "Replay all actual and matched targets, both R105 double fibers, "
            "repeated atoms, and no-relation controls against direct sources."
        ),
        "falsifier": (
            "The quotient loses the target zero biconditional or requires an "
            "outcome-dependent pole; the index stores or scans B^3 C-side "
            "endpoints, restores B^3/B^4 query traffic, uses a target DLP "
            "label, or loses weight 14400, markers, multiplicity, rank, "
            "logs, identical descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_gauge_normalized_endpoint_query2p1.json",
            "gauge_normalized_endpoint_index_ledger.json",
            "query2p1_subset_source_replay.json",
            "query2p1_exceptional_controls.json",
            "factor_logs_and_identical_descent_r112.json",
        ],
    },
    "s6_5a5c_nonlinear_elliptic_orbit_product_recurrence": {
        "hypothesis": (
            "A gauge-invariant nonlinear recurrence evaluates the "
            "target-shifted A5 membership product over the canonical "
            "C2 x C3 endpoint family with bounded order and state, and its "
            "adjoint returns a subset-stable canonical source."
        ),
        "decisive_test": (
            "Freeze the orbit-product section, divisor, recurrence "
            "coefficients and order, target-update map, exceptional charts, "
            "and source adjoint. Expand it on every actual and matched deck "
            "and compare existence, multiplicity, and dyadic source output "
            "against the direct typed endpoint query."
        ),
        "falsifier": (
            "The recurrence is only a renamed C2 x C3 product, has order or "
            "state B^3, requires B^3 target specialization, uses a target "
            "DLP label or supplied common factor, or loses complete charts, "
            "weight 14400, markers, multiplicity, rank, logs, identical "
            "descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_nonlinear_elliptic_orbit_product.json",
            "elliptic_orbit_product_recurrence_ledger.json",
            "orbit_product_subset_source_replay.json",
            "orbit_product_exceptional_controls.json",
            "factor_logs_and_identical_descent_r113.json",
        ],
    },
    "s6_5a5c_transposed_nonuniform_c5_leaf_generator": {
        "hypothesis": (
            "A transposed complete-homogeneous evaluator contracts the "
            "nonuniform C5 endpoint membership leaves directly to an exact "
            "existence bit, and its adjoint returns one canonical source "
            "without forming the C5 list or full endpoint support."
        ),
        "decisive_test": (
            "Freeze the forward functional, transposed program, target "
            "specialization, intermediate dimensions, and source adjoint. "
            "Compare forward existence and backward source outputs on every "
            "actual, matched, double-fiber, repeated-atom, and no-relation "
            "control against R113."
        ),
        "falsifier": (
            "The forward or adjoint pass materializes B^3 leaves, B^3 "
            "product-tree state, or B^5 endpoint support; assumes a supplied "
            "zero/common factor or DLP label; or loses complete charts, "
            "weight 14400, markers, multiplicity, rank, logs, identical "
            "descent, or workspace."
        ),
        "required_artifacts": [
            "frozen_5a5c_transposed_nonuniform_c5_leaf_generator.json",
            "transposed_c5_leaf_generator_ledger.json",
            "transposed_c5_source_adjoint_replay.json",
            "transposed_c5_exceptional_controls.json",
            "factor_logs_and_identical_descent_r114.json",
        ],
    },
    "s6_relation_arity_factor_base_transposed_interface_rebalance": {
        "hypothesis": (
            "A higher relation arity with asymmetric factor-base exponents "
            "reduces the first typed transposed source-return interface below "
            "B^(9/4) setup and B^(5/4) fresh work while retaining sufficient "
            "relations for known-RHS rank and identical target descent."
        ),
        "decisive_test": (
            "Freeze the relation-supply, factor-base, summation-polynomial/"
            "FFE degree, source-interface, rank, log, descent, and total-cost "
            "inequalities. Enumerate the rational exponent polytope before "
            "running toys and independently replay every feasible vertex."
        ),
        "falsifier": (
            "No exponent assignment satisfies all setup, fresh-work, "
            "relation-supply, rank, factor-log, identical-descent, and "
            "source-to-target cost inequalities; or a feasible assignment "
            "relies on an uncharged summation-polynomial degree, target DLP "
            "label, supplied source, or verifier-only witness."
        ),
        "required_artifacts": [
            "frozen_relation_arity_factor_base_exponent_model.json",
            "relation_arity_factor_base_feasibility_ledger.json",
            "transposed_interface_cap_table.json",
            "factor_logs_and_identical_descent_r115.json",
        ],
    },
    "s7_m6_implicit_3f_self_convolution_ffe_source_locator": {
        "hypothesis": (
            "At the m=6, alpha=1/12, beta=3/4 vertex, a transposed "
            "summation-polynomial/FFE autocorrelation finds X in 3F with "
            "R-X in 3F and returns all six coupled factor sources without "
            "enumerating the B^(5/2) half occurrences."
        ),
        "decisive_test": (
            "Freeze the implicit 3F membership section, target translation, "
            "S7 and expanded S13 contractions, intermediate dimensions, "
            "multiplicity handling, and source adjoint. Replay exact "
            "positive, no-relation, repeated-source, and matched controls "
            "with complete projective charts and field-level costs."
        ),
        "falsifier": (
            "The evaluator materializes B^(5/2) 3F occurrences, scans an "
            "equivalent half list, treats degree-32 S7 or degree-2048 S13 "
            "arithmetic as free, consumes a target DLP label or supplied "
            "source, or fails six-source replay, rank, logs, identical "
            "descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_c3_pair_sum_batch_interface.json",
            "c3_pair_sum_indexing_and_algebraic_cost_ledger.json",
            "m6_a6_c3_self_convolution_source_replay.json",
            "factor_logs_and_identical_descent_r116.json",
        ],
    },
    "s8_m6_target_batched_c3_pair_sum_elliptic_transpose": {
        "hypothesis": (
            "A jointly transposed elliptic coefficient functional evaluates "
            "the A6-weighted C3 self-convolution at one fresh target and its "
            "source adjoint returns one A6 plus two C3 backpointers without "
            "materializing pair sums or target-by-target divisor translates."
        ),
        "decisive_test": (
            "Freeze the C3 endpoint-divisor representation, A6 target batch, "
            "forward coefficient functional, target specialization, all "
            "intermediate dimensions, and source adjoint. Compare exact "
            "counts and sources against R116 on positive, empty, repeated, "
            "and complete-projective controls, then charge relation and "
            "identical-descent use of the same operator."
        ),
        "falsifier": (
            "The operator emits B^(9/2) C3 pair sums, performs B^(11/4) "
            "target-by-target divisor translation, uses B^5 dense characters, "
            "requires DLP/residue labels, a determinant oracle, or a supplied "
            "source, or fails B^(9/4) setup, B^(5/4) fresh work, exact source "
            "replay, known-RHS rank, factor logs, identical descent, memory, "
            "or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_target_batched_c3_elliptic_transpose.json",
            "m6_target_batched_c3_transpose_cost_ledger.json",
            "m6_target_batched_c3_source_adjoint_replay.json",
            "m6_target_batched_c3_exceptional_controls.json",
            "factor_logs_and_identical_descent_r117.json",
        ],
    },
    "s9_m6_nonlinear_value_sensitive_c6_source_locator": {
        "hypothesis": (
            "A target-specialized nonlinear branch network over the original "
            "C deck uses the persistent C3 representation to find one six-C "
            "source in B^(3/4+o(1)) work per A6 target without materializing "
            "a universal translate table or regular Boolean target section."
        ),
        "decisive_test": (
            "Freeze the target-dependent branches, every S7/FFE remainder "
            "or subresultant dimension, exceptional projective charts, "
            "false-positive verification, and reverse source path. Replay "
            "one exact six-C source on positive, empty, repeated, and "
            "adversarial finite controls, compose it with A6, and charge the "
            "identical relation and target-descent operator."
        ),
        "falsifier": (
            "The network hides a B^5 linear translate state, a B^5 regular "
            "target section, B^(33/8) k=7 index, B^(11/4) translated-divisor "
            "batch, DLP labels, supplied sources, or uncharged branching/"
            "subresultant work; or it fails B^(9/4) setup, B^(5/4) batch "
            "work, exact backpointers, known-RHS rank, factor logs, identical "
            "descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_nonlinear_c6_source_locator.json",
            "m6_nonlinear_c6_branch_and_ffe_cost_ledger.json",
            "m6_nonlinear_c6_source_replay.json",
            "m6_nonlinear_c6_exceptional_controls.json",
            "factor_logs_and_identical_descent_r118.json",
        ],
    },
    "s10_m6_output_sensitive_nonlinear_c5_source_index": {
        "hypothesis": (
            "The scalar-blind C deck admits a nonlinear endpoint-compressed "
            "five-sum membership/source index with B^(9/4+o(1)) setup and "
            "polylogarithmic exact query, enabling the R118 one-C branch "
            "without materializing C5 occurrences."
        ),
        "decisive_test": (
            "Freeze D_C, S6 or the chosen fixed-sign group-law system, every "
            "quotient/remainder/subresultant dimension, target-dependent "
            "branch, sign and projective exception, empty-answer certificate, "
            "and reverse source path. Replay exact positive, repeated, empty, "
            "identity, and adversarial controls before composing with the "
            "R118 one-C branch and A6 batch."
        ),
        "falsifier": (
            "The index stores B^(15/4) C5 occurrences or an equivalent grid, "
            "spends B^(3/2) on a C2 scan or B^(9/4) on an explicit resultant "
            "per target, assumes x-only S6 is fixed-sign exact, consumes DLP "
            "labels or supplied sources, or fails setup, polylog query, exact "
            "empty rejection, source replay, rank, logs, identical descent, "
            "memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_output_sensitive_c5_source_index.json",
            "m6_c5_membership_ffe_cost_ledger.json",
            "m6_c5_membership_source_replay.json",
            "m6_c5_membership_exceptional_controls.json",
            "factor_logs_and_identical_descent_r119.json",
        ],
    },
    "s11_m6_suboutput_implicit_c5_membership_source_circuit": {
        "hypothesis": (
            "A target-specialized nonlinear circuit over the scalar-blind C "
            "deck answers exact five-sum membership and returns five "
            "occurrence backpointers using B^(9/4+o(1)) persistent state and "
            "polylogarithmic field work, without emitting the "
            "B^(15/4+o(1)) endpoint support."
        ),
        "decisive_test": (
            "Freeze one target computation graph, all S6 or fixed-sign "
            "group-law branches, quotient/remainder dimensions, character "
            "or pairing evaluations, exceptional projective charts, exact "
            "empty certificate, and reverse five-source path. Replay "
            "positive, repeated, empty, identity, and adversarial controls, "
            "then compose unchanged with the R118 one-C branch and A6 batch."
        ),
        "falsifier": (
            "The circuit emits or scans B^(15/4) endpoints, materializes a "
            "B^5 translate section, stores more than B^(9/4) state, spends "
            "polynomial work per target, assumes a unit-cost character, "
            "pairing, DLP, resultant, gcd, root, or source oracle, or fails "
            "exact empty rejection, fixed-sign source replay, rank, logs, "
            "identical descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_suboutput_implicit_c5_membership_circuit.json",
            "m6_c5_implicit_circuit_cost_ledger.json",
            "m6_c5_implicit_circuit_source_replay.json",
            "m6_c5_implicit_circuit_exceptional_controls.json",
            "factor_logs_and_identical_descent_r120.json",
        ],
    },
    "s12_m6_small_k_multiplicative_c5_membership_source_circuit": {
        "hypothesis": (
            "For k=ord_q(p)=B^(o(1)), pairing images of the scalar-blind C "
            "deck admit an exact sub-output five-product membership/source "
            "circuit with B^(9/4+o(1)) state and polylogarithmic query, "
            "without finite-field discrete logarithms or product-support "
            "materialization."
        ),
        "decisive_test": (
            "Freeze the extension field, independent q-torsion construction, "
            "Miller and final-exponentiation graph, multiplicative deck "
            "polynomial or FFE circuit, all target branches, exact empty "
            "certificate, and five-source adjoint. Replay positive, repeated, "
            "empty, identity, and adversarial product controls, compose with "
            "R118, and separately fail closed on pairing-unfriendly inputs."
        ),
        "falsifier": (
            "The circuit takes pairing-image discrete logs, emits or scans "
            "B^(15/4) five-product endpoints, uses B^(33/8) current-index "
            "state, hides extension/torsion/pairing/root/resultant work, "
            "generalizes k=2 fixtures to arbitrary inputs, or fails exact "
            "empty answers, five backpointers, rank, logs, identical descent, "
            "memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_small_k_multiplicative_c5_subfunction_index.json",
            "m6_small_k_multiplicative_c5_cost_ledger.json",
            "m6_small_k_multiplicative_c5_source_replay.json",
            "m6_small_k_multiplicative_c5_exceptional_controls.json",
            "factor_logs_and_identical_descent_r121.json",
        ],
    },
    "s13_m6_target_specialized_nonlinear_torus_c5_source_circuit": {
        "hypothesis": (
            "The exact norm-one-torus form B-t_target*A admits a "
            "target-specialized nonlinear membership/source circuit with "
            "B^(9/4+o(1)) state and polylogarithmic arbitrary-target work, "
            "without full endpoint moments, product support, or field "
            "discrete logarithms."
        ),
        "decisive_test": (
            "Freeze the Cayley chart, target-injected degree-five gate graph, "
            "every denominator and false-positive branch, exact empty "
            "certificate, and reverse five-source path. Replay positive, "
            "repeated, empty, identity, and adversarial torus controls, then "
            "compose unchanged with R120 pairings and the R118 outer batch."
        ),
        "falsifier": (
            "The circuit reconstructs B^(15/4) moments or endpoints, scans "
            "B^(3/2) C2 pairs, invokes the integer-residue subfunction "
            "theorem outside its hypotheses, takes pairing-image discrete "
            "logs, hides denominator or extension work, or fails exact empty "
            "answers, five projective backpointers, rank, logs, identical "
            "descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_m6_small_k_multiplicative_c5_subfunction_index.json",
            "m6_small_k_multiplicative_c5_cost_ledger.json",
            "m6_small_k_multiplicative_c5_source_replay.json",
            "m6_small_k_multiplicative_c5_exceptional_controls.json",
            "factor_logs_and_identical_descent_r121.json",
        ],
    },
    "s14_target_specialized_nonoccurrence_torus_c5_source_circuit": {
        "hypothesis": (
            "The exact R121 target form admits a nonoccurrence "
            "membership/source circuit that avoids every stored C_s "
            "endpoint table and enumerated C_r complement, while using "
            "B^(9/4+o(1)) state and polylogarithmic arbitrary-target work."
        ),
        "decisive_test": (
            "Freeze the target-injected torus gate graph, all denominator "
            "and false-positive branches, any filtered-deck compression "
            "theorem, exact empty certificate, and reverse five-source "
            "path. Compose it with R120 and R118, then charge meaningful "
            "relation rank and identical descent without an explicit split "
            "at any arity."
        ),
        "falsifier": (
            "The route stores or emits an occurrence-scale C_s support, "
            "enumerates a C_r complement, relies on global arity rebalance "
            "despite the R122 L>=11/4+beta theorem, transfers iid support "
            "claims to a filtered deck without proof, takes field DLP, or "
            "fails exact empty answers, five backpointers, rank, logs, "
            "identical descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_torus_c5_explicit_split_global_rebalance.json",
            "torus_c5_explicit_split_global_rebalance_cost_ledger.json",
            "torus_c5_explicit_split_global_rebalance_replay.json",
            "torus_c5_explicit_split_global_rebalance_controls.json",
            "factor_logs_and_identical_descent_r122.json",
        ],
    },
    "s15_nonrepresented_fourier_resultant_torus_c5_source_circuit": {
        "hypothesis": (
            "The R121 target form admits an exact nonrepresented circuit "
            "that exploits multiplicative characters or product-resultant "
            "composition without materializing q modes, full C5 recurrence "
            "state, target-scaled P2|P3 coefficient bodies, or symbolic P5."
        ),
        "decisive_test": (
            "Freeze the actual target circuit or data structure, "
            "noncancellation proof, all field branches, exact empty "
            "certificate, and reverse five-source adjoint. Charge its state "
            "and query gates before composing with R120 pairings, R118 "
            "relation collection, known-RHS rank, logs, and descent."
        ),
        "falsifier": (
            "The route consumes q modes, B^(15/4) Prony/BM or P5 state, "
            "B^(9/4) represented target resultant work, an explicit endpoint "
            "split, field DLP, or a unit-cost resultant/root oracle, or fails "
            "exact empty answers, five backpointers, rank, logs, identical "
            "descent, memory, or complete source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_torus_c5_fourier_product_resultant.json",
            "torus_c5_fourier_product_resultant_cost_ledger.json",
            "torus_c5_fourier_product_resultant_replay.json",
            "torus_c5_fourier_product_resultant_controls.json",
            "factor_logs_and_identical_descent_r123.json",
        ],
    },
    "s16_coupled_nonlinear_torus_c5_zero_test": {
        "hypothesis": (
            "The coupled deck powers (u^(*2),u^(*3)) admit an exact "
            "target-specialized nonlinear membership/source data structure "
            "that is unavailable to a universal linear C3 sketch."
        ),
        "decisive_test": (
            "Freeze one multilinear, rational-Krylov, adaptive-probe, or "
            "nonlinear-fingerprint circuit specialized jointly to u^(*2) "
            "and u^(*3). Prove exact noncancellation and empty semantics, "
            "expose five source backpointers, and charge every setup, query, "
            "memory, field, bit, rank, log, and descent operation."
        ),
        "falsifier": (
            "The route is a universal linear sketch in disguise, stores q "
            "translated counts, scans C2, materializes C5, consumes field "
            "DLP, assumes a unit-cost nonlinear fingerprint, or fails exact "
            "empty answers, five sources, rank, logs, identical descent, "
            "memory, or complete source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_torus_c5_linear_sketch_circulant.json",
            "torus_c5_linear_sketch_circulant_cost_ledger.json",
            "torus_c5_linear_sketch_circulant_replay.json",
            "torus_c5_linear_sketch_circulant_controls.json",
            "factor_logs_and_identical_descent_r124.json",
        ],
    },
    "s17_nonhomomorphic_adaptive_torus_c5_fingerprint": {
        "hypothesis": (
            "A nonhomomorphic or adaptive target fingerprint can exploit "
            "the coupled deck powers while representing all product-law "
            "corrections below the frozen setup and query caps."
        ),
        "decisive_test": (
            "Freeze the hash/probe map, every collision and product-law "
            "correction class, deterministic noncancellation proof, exact "
            "empty certificate, and reverse five-source path. Charge the "
            "complete correction data before rank, logs, and descent."
        ),
        "falsifier": (
            "The route silently assumes h(xy)=h(x)h(y), requires an "
            "injective q-sized image, stores or scans occurrence-scale "
            "correction classes, uses field DLP, permits probabilistic "
            "false answers, or fails five sources, rank, logs, identical "
            "descent, memory, or full source-to-target cost."
        ),
        "required_artifacts": [
            "frozen_torus_c5_prime_order_homomorphic_fingerprint.json",
            "torus_c5_prime_order_homomorphic_fingerprint_cost_ledger.json",
            "torus_c5_prime_order_homomorphic_fingerprint_replay.json",
            "torus_c5_prime_order_homomorphic_fingerprint_controls.json",
            "factor_logs_and_identical_descent_r125.json",
        ],
    },
    "s18_implicit_adaptive_torus_c5_hash_correction_circuit": {
        "hypothesis": (
            "A nonhomomorphic coordinate hash admits an implicit or "
            "adaptive correction circuit that recognizes exact C2-C3 "
            "products without representing the B^(15/4) C5 support."
        ),
        "decisive_test": (
            "Freeze the correction circuit or probe graph, all accessed "
            "state cells, target branches, deterministic noncancellation "
            "and empty certificates, and reverse C2+C3 source path. Charge "
            "its full preprocessing and query complexity before rank, logs, "
            "and identical descent."
        ),
        "falsifier": (
            "The route lists per-bucket or globally deduplicated C5 "
            "products, hides occurrence-scale correction state behind an "
            "oracle, scans C2 or C3, takes field DLP, permits probabilistic "
            "false answers, or fails five sources, rank, logs, descent, "
            "memory, field operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_explicit_hash_correction_support.json",
            "torus_c5_explicit_hash_correction_support_cost_ledger.json",
            "torus_c5_explicit_hash_correction_support_replay.json",
            "torus_c5_explicit_hash_correction_support_controls.json",
            "factor_logs_and_identical_descent_r126.json",
        ],
    },
    "s19_cap_tight_singleton_c3_target_router": {
        "hypothesis": (
            "A B^(9/4+o(1))-state singleton-C3 index admits an exact "
            "implicit O(1)-pair arbitrary-target router that returns one "
            "matching C2 and C3 source without field DLP."
        ),
        "decisive_test": (
            "Freeze the C3 perfect-hash layout and target-to-pair routing "
            "circuit, including all empty branches and reverse source "
            "pointers. Prove O(1) routed pairs for every arbitrary target "
            "and charge construction, memory, field operations, bits, rank, "
            "logs, and identical descent."
        ),
        "falsifier": (
            "The router stores C5 targets, scans C2/C3, evaluates H or H^2 "
            "bucket resultants, hides a field DLP or target dictionary, "
            "permits false answers, or fails five sources, rank, logs, "
            "identical descent, memory, field operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_bucket_resultant_routing_tradeoff.json",
            "torus_c5_bucket_resultant_routing_tradeoff_cost_ledger.json",
            "torus_c5_bucket_resultant_routing_tradeoff_replay.json",
            "torus_c5_bucket_resultant_routing_tradeoff_controls.json",
            "factor_logs_and_identical_descent_r127.json",
        ],
    },
    "s20_low_slp_piecewise_torus_c5_selector": {
        "hypothesis": (
            "The cap-tight singleton-C3 index admits a high-degree "
            "low-SLP or compact piecewise rational C2 selector with "
            "polylogarithmic arbitrary-target evaluation."
        ),
        "decisive_test": (
            "Freeze the selector SLP or branch DAG, every coefficient and "
            "branch predicate, exact empty semantics, and reverse C2+C3 "
            "source path. Measure online field operations and total state "
            "before composing with rank, logs, and identical descent."
        ),
        "falsifier": (
            "The selector is densely evaluated, stores target-to-source "
            "entries, hides a branch dictionary or DLP, scans C2/C3, "
            "permits false answers, or fails five sources, rank, logs, "
            "identical descent, memory, field operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_rational_selector_degree.json",
            "torus_c5_rational_selector_degree_cost_ledger.json",
            "torus_c5_rational_selector_degree_replay.json",
            "torus_c5_rational_selector_degree_controls.json",
            "factor_logs_and_identical_descent_r128.json",
        ],
    },
    "s21_shared_predicate_torus_c5_selector_dag": {
        "hypothesis": (
            "The B^(3/2+o(1)) optimal C2 translate branches admit a "
            "compact shared-predicate decision DAG, or an equivalent "
            "high-degree low-SLP selector, with polylogarithmic "
            "arbitrary-target evaluation."
        ),
        "decisive_test": (
            "Freeze every DAG node, shared branch predicate, selector SLP, "
            "exact empty path, and reverse C2+C3 source pointer. Measure "
            "online field operations and total state before composing with "
            "rank, logs, and identical descent."
        ),
        "falsifier": (
            "The router scans B^(3/2) branches, stores B^(15/4) target "
            "entries, hides source exponents or a field DLP, permits false "
            "answers, or fails five sources, rank, logs, identical descent, "
            "memory, field operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_piecewise_selector_decision_dag.json",
            "torus_c5_piecewise_selector_decision_dag_cost_ledger.json",
            "torus_c5_piecewise_selector_decision_dag_replay.json",
            "torus_c5_piecewise_selector_decision_dag_controls.json",
            "factor_logs_and_identical_descent_r129.json",
        ],
    },
    "s22_order2_finite_field_torus_c5_selector_predicate": {
        "hypothesis": (
            "The actual ord_q(characteristic)=2 pairing fields admit either "
            "a field-specific sparse predicate theorem or a direct "
            "finite-field/non-Fourier selector DAG with polylogarithmic "
            "arbitrary-target evaluation."
        ),
        "decisive_test": (
            "Freeze the actual coefficient field, every Fourier minor or "
            "non-Fourier DAG node, exact empty path, and reverse C2+C3 "
            "source pointer. Prove the claimed field-specific theorem or "
            "replay the circuit before composing with rank, logs, and "
            "identical descent."
        ),
        "falsifier": (
            "The argument imports a complex uncertainty theorem without "
            "finite-field transfer, scans B^(3/2) branches, stores "
            "B^(15/4) targets, hides a DLP, permits false answers, or fails "
            "five sources, rank, logs, descent, memory, operations, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_sparse_fourier_predicate_transfer.json",
            "torus_c5_sparse_fourier_predicate_transfer_cost_ledger.json",
            "torus_c5_sparse_fourier_predicate_transfer_replay.json",
            "torus_c5_sparse_fourier_predicate_transfer_controls.json",
            "factor_logs_and_identical_descent_r130.json",
        ],
    },
    "s23_lacunary_order2_torus_c5_selector_predicate": {
        "hypothesis": (
            "The actual ord_q(characteristic)=2 pairing fields admit a "
            "lacunary high-degree low-SLP predicate or a non-Fourier "
            "shared decision DAG with polylogarithmic arbitrary-target "
            "evaluation."
        ),
        "decisive_test": (
            "Freeze every lacunary mode, SLP node, or non-Fourier DAG node, "
            "the exact empty path, and reverse C2+C3 source pointer. Replay "
            "the circuit in each actual field before composing with rank, "
            "logs, and identical descent."
        ),
        "falsifier": (
            "The representation expands to a dense B^(15/4) mode block, "
            "scans B^(3/2) branches, stores B^(15/4) targets, hides a DLP, "
            "permits false answers, or fails five sources, rank, logs, "
            "descent, memory, operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_consecutive_mode_predicate.json",
            "torus_c5_consecutive_mode_predicate_cost_ledger.json",
            "torus_c5_consecutive_mode_predicate_replay.json",
            "torus_c5_consecutive_mode_predicate_controls.json",
            "factor_logs_and_identical_descent_r131.json",
        ],
    },
    "s24_asymmetric_frobenius_torus_c5_selector_predicate": {
        "hypothesis": (
            "An asymmetric F_(p^2) lacunary predicate or a coordinate DAG "
            "that explicitly consumes z and z^p distinguishes the actual "
            "positive/empty inverse pairs in polylogarithmic target work."
        ),
        "decisive_test": (
            "Freeze every extension coefficient, lacunary mode, Frobenius "
            "coordinate node, exact empty path, and reverse C2+C3 source "
            "pointer. Replay inverse-pair separation and the full circuit "
            "in each actual field before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The predicate is inversion invariant, expands to B^(15/4) "
            "state or query work, scans B^(3/2) branches, hides a DLP, "
            "permits false answers, or fails five sources, rank, logs, "
            "descent, memory, operations, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_base_field_frobenius_predicate_dag.json",
            "torus_c5_base_field_frobenius_predicate_dag_cost_ledger.json",
            "torus_c5_base_field_frobenius_predicate_dag_replay.json",
            "torus_c5_base_field_frobenius_predicate_dag_controls.json",
            "factor_logs_and_identical_descent_r132.json",
        ],
    },
    "s25_five_mode_low_slp_frobenius_torus_c5_selector": {
        "hypothesis": (
            "A five-mode or larger-polylog F_(p^2) predicate, a compact "
            "high-expansion SLP, or a multi-predicate Frobenius-coordinate "
            "DAG selects one valid C2 branch in polylogarithmic target work."
        ),
        "decisive_test": (
            "Freeze every extension coefficient, exponent, SLP operation, "
            "coordinate predicate, branch, exact empty path, and reverse "
            "C2+C3 source pointer. Replay the full selector in every actual "
            "field before composing with rank, logs, and identical descent."
        ),
        "falsifier": (
            "The predicate misses a positive, accepts an empty target, "
            "expands to B^(15/4) represented state, scans B^(3/2) branches, "
            "hides a DLP, or fails five sources, rank, logs, descent, memory, "
            "field-operation, or bit costs."
        ),
        "required_artifacts": [
            "frozen_torus_c5_sparse_monomial_root_bound.json",
            "torus_c5_sparse_monomial_root_bound_cost_ledger.json",
            "torus_c5_sparse_monomial_root_bound_replay.json",
            "torus_c5_sparse_monomial_root_bound_controls.json",
            "factor_logs_and_identical_descent_r133.json",
        ],
    },
    "s26_seven_mode_multi_predicate_torus_c5_selector": {
        "hypothesis": (
            "A seven-or-more-mode F_(p^2) predicate, a decision DAG of "
            "multiple small predicates, a nonzero-value Frobenius branch, "
            "or a high-expansion low-SLP circuit selects a valid C2 branch "
            "in polylogarithmic target work."
        ),
        "decisive_test": (
            "Freeze every coefficient, exponent, SLP operation, nonzero-"
            "value comparison, coordinate node, branch, exact empty path, "
            "and reverse C2+C3 source pointer. Replay the full selector in "
            "every actual field before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The construction reduces to one at-most-six-mode zero or pole "
            "set, misses a positive, accepts an empty target, expands to "
            "B^(15/4) state, scans B^(3/2) branches, hides a DLP, or fails "
            "five sources, rank, logs, descent, memory, operations, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_two_atom_geometric_progression.json",
            "torus_c5_two_atom_geometric_progression_cost_ledger.json",
            "torus_c5_two_atom_geometric_progression_replay.json",
            "torus_c5_two_atom_geometric_progression_controls.json",
            "factor_logs_and_identical_descent_r134.json",
        ],
    },
    "s27_composed_nonzero_low_slp_torus_c5_selector": {
        "hypothesis": (
            "A structured-deck-specific composition of small predicates, "
            "a nonzero-value Frobenius-coordinate branch, or a compact "
            "high-expansion SLP selects a valid C2 branch in "
            "polylogarithmic arbitrary-target work."
        ),
        "decisive_test": (
            "Freeze every predicate, coefficient, SLP operation, nonzero-"
            "value comparison, coordinate node, branch, exact empty path, "
            "and reverse C2+C3 source pointer. Replay the complete composed "
            "selector in every actual field before rank, logs, and "
            "identical descent."
        ),
        "falsifier": (
            "The construction reduces to one represented sparse zero or "
            "pole set covered by the rank bound, misses a positive, accepts "
            "an empty target, expands to B^(15/4) state, scans B^(3/2) "
            "branches, hides a DLP, or fails five sources, rank, logs, "
            "descent, memory, operations, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_khatri_rao_kruskal_amplification.json",
            "torus_c5_khatri_rao_kruskal_amplification_cost_ledger.json",
            "torus_c5_khatri_rao_kruskal_amplification_replay.json",
            "torus_c5_khatri_rao_kruskal_amplification_controls.json",
            "factor_logs_and_identical_descent_r135.json",
        ],
    },
    "s28_growing_support_low_slp_nonzero_torus_c5_selector": {
        "hypothesis": (
            "A growing-support but compact straight-line program, or a "
            "nonzero-value Frobenius-coordinate circuit, selects one valid "
            "C2 branch with polylogarithmic arbitrary-target work."
        ),
        "decisive_test": (
            "Freeze every circuit node, expanded-support checkpoint, "
            "nonzero-value operation, coordinate branch, exact empty path, "
            "and reverse C2+C3 source pointer. Replay one positive and its "
            "inverse empty through the full circuit in every actual field "
            "before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The circuit reduces to a bounded-support all-nonzero zero-test "
            "path, misses a positive, accepts an empty target, materializes "
            "B^(15/4) state, scans B^(3/2) branches, hides a DLP, or fails "
            "five sources, rank, logs, descent, memory, operations, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_all_nonzero_path_product.json",
            "torus_c5_all_nonzero_path_product_cost_ledger.json",
            "torus_c5_all_nonzero_path_product_replay.json",
            "torus_c5_all_nonzero_path_product_controls.json",
            "factor_logs_and_identical_descent_r136.json",
        ],
    },
    "s29_three_plus_five_plus_nonzero_torus_c5_selector": {
        "hypothesis": (
            "A structured three-plus-mode zero-test circuit, a five-plus-"
            "mode low-SLP circuit, or a nonzero-value Frobenius-coordinate "
            "branch selects one valid C2 source in polylogarithmic target "
            "work."
        ),
        "decisive_test": (
            "Freeze every mode, coefficient, circuit node, value operation, "
            "coordinate branch, exact empty path, and reverse C2+C3 source "
            "pointer. Replay the full circuit on positives and inverse "
            "empties in every actual field before rank, logs, and identical "
            "descent."
        ),
        "falsifier": (
            "The construction reduces to a binomial zero-test tree, misses "
            "a positive, accepts an empty target, materializes B^(15/4) "
            "state, scans B^(3/2) branches, hides a DLP, or fails five "
            "sources, rank, logs, descent, memory, operations, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_binomial_node_union_depth.json",
            "torus_c5_binomial_node_union_depth_cost_ledger.json",
            "torus_c5_binomial_node_union_depth_replay.json",
            "torus_c5_binomial_node_union_depth_controls.json",
            "factor_logs_and_identical_descent_r137.json",
        ],
    },
    "s30_characteristic_specific_spark_or_nonzero_torus_c5_selector": {
        "hypothesis": (
            "A characteristic-specific restricted-minor theorem for the "
            "actual norm-one atoms, a four-plus-mode low-SLP circuit, or "
            "a nonzero-value Frobenius-coordinate branch yields an exact "
            "polylogarithmic-target selector."
        ),
        "decisive_test": (
            "Prove the restricted-minor property in the actual order-two "
            "characteristic regime or freeze every surviving circuit mode, "
            "coefficient, node, value operation, branch, exact empty path, "
            "and reverse C2+C3 source pointer. Replay positives and inverse "
            "empties in every actual field before rank, logs, and identical "
            "descent."
        ),
        "falsifier": (
            "The proof assumes primitive characteristic order, the "
            "representation needs extension degree q-1, the circuit reduces "
            "to a closed sparse zero-test class, misses a positive, accepts "
            "an empty target, hides a DLP, or fails five sources, rank, logs, "
            "descent, memory, operations, extension degree, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_chebotarev_fiber_cover.json",
            "torus_c5_chebotarev_fiber_cover_cost_ledger.json",
            "torus_c5_chebotarev_fiber_cover_replay.json",
            "torus_c5_chebotarev_fiber_cover_controls.json",
            "factor_logs_and_identical_descent_r138.json",
        ],
    },
    "s31_four_plus_or_nonzero_torus_c5_selector": {
        "hypothesis": (
            "Order-two four-plus-mode structure, a compact low-SLP "
            "circuit, or a nonzero-value Frobenius-coordinate branch "
            "selects an exact C2 source with polylogarithmic target work."
        ),
        "decisive_test": (
            "Analyze order-two four-column minors and root fibers or freeze "
            "every surviving circuit mode, coefficient, node, value "
            "operation, branch, exact empty path, and reverse C2+C3 source "
            "pointer. Replay positives and inverse empties in every actual "
            "field before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The construction reduces to a closed at-most-three-mode tree, "
            "misses a positive, accepts an empty target, materializes "
            "B^(15/4) state, scans polynomially many branches, hides a DLP, "
            "or fails five sources, rank, logs, descent, memory, operations, "
            "extension degree, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_order_two_three_minor_rigidity.json",
            "torus_c5_order_two_three_minor_rigidity_cost_ledger.json",
            "torus_c5_order_two_three_minor_rigidity_replay.json",
            "torus_c5_order_two_three_minor_rigidity_controls.json",
            "factor_logs_and_identical_descent_r139.json",
        ],
    },
    "s32_subcap_mobius_claw_or_nonzero_torus_c5_selector": {
        "hypothesis": (
            "The order-two Mobius claw admits a known-mode construction "
            "below q^(9/20), its hidden image mode can be evaluated without "
            "a field DLP, or a nonzero-value Frobenius-coordinate branch "
            "selects an exact C2 source with polylogarithmic target work."
        ),
        "decisive_test": (
            "Freeze a sub-q^(9/20) known-mode claw, a direct hidden-mode "
            "evaluation circuit, or every node and branch of a nonzero-"
            "value selector. Replay positives and inverse empties in every "
            "actual field before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The route performs a field DLP, costs q^(1/2), covers only a "
            "constant progression subset, misses a positive, accepts an "
            "empty target, materializes B^(15/4) state, scans polynomially "
            "many branches, or fails five sources, rank, logs, descent, "
            "memory, operations, extension degree, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_order_two_four_minor_claw.json",
            "torus_c5_order_two_four_minor_claw_cost_ledger.json",
            "torus_c5_order_two_four_minor_claw_replay.json",
            "torus_c5_order_two_four_minor_claw_controls.json",
            "factor_logs_and_identical_descent_r140.json",
        ],
    },
    "s33_nonlinear_sextic_mobius_source_router": {
        "hypothesis": (
            "A nonlinear, nontranslation data structure composes the "
            "polylogarithmic sextic Mobius labels into an exact C2 source "
            "router without materializing the Omega(q) linear orbit."
        ),
        "decisive_test": (
            "Freeze every character parameter, label, nonlinear state "
            "transition, branch, C2 pointer, C3 certificate, inverse-empty "
            "path, and reverse five-source pointer. Replay arbitrary "
            "positives and empties before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The route reduces to translated linear convolution, stores "
            "Omega(q), scans C2, admits an ambiguous label bucket, misses a "
            "positive, accepts an empty target, hides a DLP, or fails five "
            "sources, rank, logs, descent, memory, operations, extension "
            "degree, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_sextic_mobius_character_router.json",
            "torus_c5_sextic_mobius_character_router_cost_ledger.json",
            "torus_c5_sextic_mobius_character_router_replay.json",
            "torus_c5_sextic_mobius_character_router_controls.json",
            "factor_logs_and_identical_descent_r141.json",
        ],
    },
    "s34_algebraic_sextic_character_composition": {
        "hypothesis": (
            "An algebraic composition law combines sextic Mobius labels "
            "across the C2-by-C3 split into an exact source router without "
            "a direct deck-landmark tree or a materialized translation "
            "orbit."
        ),
        "decisive_test": (
            "Freeze the composition law or broader parameter compiler, "
            "every arbitrary-target value and branch, the C2 pointer, C3 "
            "certificate, inverse-empty rejection, and reverse five-source "
            "replay. Prove compiled state below B^(9/4+o(1)) and charge "
            "all query work before rank, logs, and identical descent."
        ),
        "falsifier": (
            "The route is only a deck-parameter decision tree, needs a "
            "full-signature table, stores Omega(q), scans C2 or C3, admits "
            "an ambiguous positive/empty cell, hides a DLP, or fails five "
            "sources, rank, logs, descent, memory, operations, extension "
            "degree, or bits."
        ),
        "required_artifacts": [
            "frozen_torus_c5_adaptive_character_decision_router.json",
            "torus_c5_adaptive_character_decision_router_cost_ledger.json",
            "torus_c5_adaptive_character_decision_router_replay.json",
            "torus_c5_adaptive_character_decision_router_controls.json",
            "factor_logs_and_identical_descent_r142.json",
        ],
    },
    "s35_transposed_ffe_relation_span": {
        "hypothesis": (
            "A target-batched transposed summation-polynomial/FFE operator "
            "applies the high-density relation family and its transpose "
            "without enumerating pair merges or explicit relation rows."
        ),
        "decisive_test": (
            "Freeze the implicit operator circuit, forward and transposed "
            "actions, known-RHS rank certificate, factor-log solve, and "
            "identical target descent at beta=9/20. Replay all actions and "
            "charge field, bit, memory, rank, log, and descent costs below "
            "N^(1/2-epsilon)."
        ),
        "falsifier": (
            "The route enumerates the N^(11/20) pair merge, emits "
            "N^(9/20) explicit rows without charging them, consumes a DLP "
            "or root oracle, infers asymptotic rank from finite fixtures, "
            "or fails known-RHS rank, factor logs, identical descent, "
            "memory, field operations, or bit complexity."
        ),
        "required_artifacts": [
            "frozen_torus_c5_label_congruence_correction.json",
            "torus_c5_label_congruence_correction_cost_ledger.json",
            "torus_c5_label_congruence_correction_replay.json",
            "torus_c5_label_congruence_correction_controls.json",
            "factor_logs_and_identical_descent_r143.json",
        ],
    },
    "s36_weighted_s13_fiber_count_transpose": {
        "hypothesis": (
            "A scalar-blind weighted S7 or expanded S13 circuit counts "
            "ordered six-factor target fibers and compiles reusable "
            "transposed state for all A/C occurrence marginals."
        ),
        "decisive_test": (
            "Freeze compact A/C divisor inputs, every weighted count gate, "
            "projective chart, multiplicity lift, setup state, online target "
            "trace, and offline/online reverse action. Build in B^(9/4+o(1)), "
            "answer counts in polylogarithmic work, and emit positive "
            "B^(3/4+o(1)) marginals without replaying setup; then replay "
            "known-target rank, factor logs, and shifted identical descent."
        ),
        "falsifier": (
            "The route enumerates six-factor sources or C3+C3 pairs, emits "
            "the endpoint body, replays B^(9/4) setup per target, consumes "
            "a DLP, root, count, marginal, rank, or source oracle, loses an "
            "exceptional chart or integer multiplicity, or fails structured "
            "rank, factor logs, shifted descent, memory, field, or bit cost."
        ),
        "required_artifacts": [
            "frozen_m6_weighted_fiber_marginal_log_operator.json",
            "m6_weighted_fiber_marginal_log_operator_cost_ledger.json",
            "m6_weighted_fiber_marginal_log_operator_replay.json",
            "m6_weighted_fiber_marginal_log_operator_controls.json",
            "factor_logs_and_identical_descent_r144.json",
        ],
    },
    "s37_implicit_batched_mobius_resultant": {
        "hypothesis": (
            "The frozen weighted C3 divisor admits one shared target-batched "
            "modular resultant or remainder operator that computes exact "
            "C6 counts without materializing a dense Mobius pullback for "
            "each A6-shifted target."
        ),
        "decisive_test": (
            "Freeze the compact C divisor, the complete A6 target batch, "
            "all product/remainder-tree state, target-dependent modular "
            "actions, exact integer lifts, and reusable transposed weight "
            "actions. Build and store at most B^(9/4+o(1)), spend at most "
            "B^(5/4+o(1)) fresh work, and replay R144 rank, factor logs, "
            "and shifted identical descent."
        ),
        "falsifier": (
            "The route emits a degree-B^(9/4) transformed polynomial per "
            "target, performs B^(9/4) independent work per target, "
            "materializes C3+C3 pairs or six-factor sources, consumes a "
            "DLP, root, gcd, count, marginal, rank, or source oracle, or "
            "fails chart, multiplicity, integer-lift, transpose, rank, "
            "factor-log, descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_weighted_c3_mobius_gcd_trace.json",
            "m6_weighted_c3_mobius_gcd_trace_cost_ledger.json",
            "m6_weighted_c3_mobius_gcd_trace_replay.json",
            "m6_weighted_c3_mobius_gcd_trace_controls.json",
            "factor_logs_and_identical_descent_r145.json",
        ],
    },
    "s38_source_equivalent_batched_count_marginal_index": {
        "hypothesis": (
            "The frozen weighted C3 divisor admits one target-batched "
            "count-and-marginal index inside the R115 caps even when its "
            "singleton-fiber output is charged as canonical-source-"
            "equivalent information."
        ),
        "decisive_test": (
            "Freeze the full known-target/A6 query batch, compact C3 "
            "divisor state, every implicit modular or FFE action, integer "
            "count lift, and reusable marginal transpose. Build and store "
            "at most B^(9/4+o(1)), spend at most B^(5/4+o(1)) fresh work, "
            "emit all required marginals, and replay rank, factor logs, and "
            "shifted descent without using source-freeness as cost credit."
        ),
        "falsifier": (
            "The route emits dense per-target pullbacks or C3+C3 pairs, "
            "replays preprocessing per target, hides singleton source "
            "information, consumes a DLP, root, gcd, count, marginal, rank, "
            "or source oracle, loses chart or integer multiplicity, or "
            "fails rank, factor-log, descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_aggregate_marginal_singleton_source_equivalence.json",
            "m6_aggregate_marginal_singleton_source_equivalence_cost_ledger.json",
            "m6_aggregate_marginal_singleton_source_equivalence_replay.json",
            "m6_aggregate_marginal_singleton_source_equivalence_controls.json",
            "factor_logs_and_identical_descent_r146.json",
        ],
    },
    "s39_shared_transposed_multi_target_valuation_marker": {
        "hypothesis": (
            "The repeated C3 occurrence divisor admits one genuinely shared "
            "transposed multi-target valuation-and-marker operator that "
            "answers the complete relation stream without independently "
            "replaying degree-B^(9/4) truncated-resultant work."
        ),
        "decisive_test": (
            "Freeze at most B^(9/4+o(1)) occurrence-divisor and transposed "
            "setup state. Process all B^(5/4+o(1)) targets within the fresh "
            "work cap, emit exact integer counts and B^(3/4+o(1)) A/C "
            "marginals, and replay R144 rank, factor logs, and shifted "
            "identical descent."
        ),
        "falsifier": (
            "The route invokes the truncated-resultant algorithm once per "
            "target, materializes the B^(9/2) pair resultant or C3+C3 "
            "pairs, emits dense endpoint bodies, consumes a DLP, root, "
            "resultant, valuation, count, marginal, rank, or source oracle, "
            "or fails chart, multiplicity, integer-lift, rank, factor-log, "
            "descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_occurrence_pair_resultant_local_valuation.json",
            "m6_occurrence_pair_resultant_local_valuation_cost_ledger.json",
            "m6_occurrence_pair_resultant_local_valuation_replay.json",
            "m6_occurrence_pair_resultant_local_valuation_controls.json",
            "factor_logs_and_identical_descent_r147.json",
        ],
    },
    "s40_structure_aware_occurrence_autocorrelation": {
        "hypothesis": (
            "The occurrence divisor's origin as a triple elliptic "
            "convolution of the compact C divisor permits a shared "
            "autocorrelation or marker operator that is unavailable to "
            "generic static 3SUM-indexing."
        ),
        "decisive_test": (
            "Construct the operator directly from the compact C divisor "
            "without materializing the B^(9/4) occurrence list as an "
            "unstructured index. Use at most B^(9/4+o(1)) setup and state "
            "and B^(5/4+o(1)) total query work, emit exact integer counts "
            "and A/C marginals, and replay rank, factor logs, and shifted "
            "identical descent."
        ),
        "falsifier": (
            "The route imports an unsupported square-root 3SUM query, "
            "uses superlinear advice or quadratic occurrence preprocessing, "
            "scans the occurrence list per target, materializes pair sums, "
            "consumes a DLP, root, 3SUM, count, marginal, rank, or source "
            "oracle, or fails chart, multiplicity, integer-lift, rank, "
            "factor-log, descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_static_3sum_indexing_tradeoff.json",
            "m6_static_3sum_indexing_tradeoff_cost_ledger.json",
            "m6_static_3sum_indexing_tradeoff_replay.json",
            "m6_static_3sum_indexing_tradeoff_controls.json",
            "factor_logs_and_identical_descent_r148.json",
        ],
    },
    "s41_nonlinear_target_specialized_compact_divisor_circuit": {
        "hypothesis": (
            "A nonlinear target-specialized circuit can act directly on "
            "the compact C divisor and the required marker batch without "
            "representing the full count sequence, its q Fourier modes, or "
            "the B^(9/4) C3 occurrence body."
        ),
        "decisive_test": (
            "Freeze a circuit with at most B^(9/4+o(1)) setup and state "
            "that evaluates all B^(5/4+o(1)) requested markers within the "
            "fresh-work cap. It must emit exact integer counts and A/C "
            "marginals, preserve every chart and multiplicity, and replay "
            "rank, factor logs, and shifted identical descent."
        ),
        "falsifier": (
            "The route forms q modes or q target values, uses a full-order "
            "linear recurrence, materializes C3 occurrences or pair sums, "
            "replays compact-divisor elimination per target, consumes a "
            "DLP, root, Fourier, recurrence, count, marginal, rank, or "
            "source oracle, or fails chart, integer-lift, rank, factor-log, "
            "descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_actual_c6_shift_krylov_rank.json",
            "m6_actual_c6_shift_krylov_rank_cost_ledger.json",
            "m6_actual_c6_shift_krylov_rank_replay.json",
            "m6_actual_c6_shift_krylov_rank_controls.json",
            "factor_logs_and_identical_descent_r149.json",
        ],
    },
    "s42_finite_depth_nonhomomorphic_u6_marker_circuit": {
        "hypothesis": (
            "A bounded-depth nonhomomorphic circuit can specialize jointly "
            "to the compact C divisor, the single power U^6, and the fixed "
            "marker batch without retaining a reusable cyclic-convolution "
            "algebra or materializing the C3 occurrence body."
        ),
        "decisive_test": (
            "Freeze a finite-depth circuit with at most B^(9/4+o(1)) setup "
            "and state that evaluates all B^(5/4+o(1)) fixed markers within "
            "the fresh-work cap. It must emit exact integer counts and A/C "
            "marginals, preserve every chart and multiplicity, and replay "
            "rank, factor logs, and shifted identical descent."
        ),
        "falsifier": (
            "The route retains a multiplication-closed convolution algebra, "
            "forms q modes or q target values, materializes C3 occurrences "
            "or pair sums, replays compact-divisor elimination per target, "
            "consumes a DLP, root, Fourier, recurrence, algebra-state, "
            "count, marginal, rank, or source oracle, or fails chart, "
            "integer-lift, rank, factor-log, descent, memory, field, or bit "
            "costs."
        ),
        "required_artifacts": [
            "frozen_m6_rational_convolution_subalgebra_rigidity.json",
            "m6_rational_convolution_subalgebra_rigidity_cost_ledger.json",
            "m6_rational_convolution_subalgebra_rigidity_replay.json",
            "m6_rational_convolution_subalgebra_rigidity_controls.json",
            "factor_logs_and_identical_descent_r150.json",
        ],
    },
    "s43_weight_parametric_bidirectional_marker_operator": {
        "hypothesis": (
            "A scalar-blind fixed-depth marker circuit can freeze setup "
            "independently of atom weights, or retain compact reusable "
            "tangent and adjoint state, so that both aggregate marginal "
            "Jacobian actions are available without materializing rows."
        ),
        "decisive_test": (
            "Freeze one circuit and derivative-state compiler with at most "
            "B^(9/4+o(1)) setup and state that applies both M and M^T in "
            "B^(5/4+o(1)) work. Run a matrix-free solve with exact residual, "
            "factor-log verification, and shifted identical descent."
        ),
        "falsifier": (
            "The route differentiates or replays setup for each vector, "
            "retains a multiplication-closed convolution algebra, forms "
            "marginal rows, q modes, q targets, C3 occurrences, or pair "
            "sums, consumes a DLP, root, Fourier, recurrence, algebra-state, "
            "count, marginal, rank, or source oracle, or fails chart, "
            "integer-lift, rank, residual, factor-log, descent, memory, "
            "field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_matrix_free_marginal_jacobian_krylov.json",
            "m6_matrix_free_marginal_jacobian_krylov_cost_ledger.json",
            "m6_matrix_free_marginal_jacobian_krylov_replay.json",
            "m6_matrix_free_marginal_jacobian_krylov_controls.json",
            "factor_logs_and_identical_descent_r151.json",
        ],
    },
    "s44_signed_weight_separable_ffe_elimination_dag": {
        "hypothesis": (
            "A signed summation-polynomial/FFE elimination DAG can freeze "
            "all topology and pivots from public A/C geometry while "
            "propagating arbitrary tangent and adjoint payloads from the "
            "R152 interpolation leaves."
        ),
        "decisive_test": (
            "Freeze one division-safe signed elimination DAG with at most "
            "B^(9/4+o(1)) setup and state. Its forward and reverse complete "
            "marker batches must cost B^(5/4+o(1)), agree with exact integer "
            "counts and Jacobian actions, and support an exact-residual "
            "factor-log solve plus shifted identical descent."
        ),
        "falsifier": (
            "Any pivot, branch, topology, or retained coefficient state "
            "depends on tangent values or target outcomes; the route loses "
            "signed-point exactness, exceeds B^(9/4) setup or B^(5/4) "
            "application cost, forms q modes, q targets, C3 occurrences, "
            "or pair sums, consumes a DLP, root, count, marginal, rank, or "
            "source oracle, or fails integer-lift, residual, factor-log, "
            "descent, memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_geometry_only_weight_interpolation_adjoint.json",
            "m6_geometry_only_weight_interpolation_adjoint_cost_ledger.json",
            "m6_geometry_only_weight_interpolation_adjoint_replay.json",
            "m6_geometry_only_weight_interpolation_adjoint_controls.json",
            "factor_logs_and_identical_descent_r152.json",
        ],
    },
    "s45_reverse_only_signed_marker_density_transfer": {
        "hypothesis": (
            "The inversion-closed C and symmetric known-A6 family admits "
            "one signed reverse-adjoint marker circuit, and a targetable "
            "multiscale shift family supplies full C-log rank despite the "
            "zero-to-four ranks in current finite controls."
        ),
        "decisive_test": (
            "Freeze one geometry-only signed reverse circuit with at most "
            "B^(9/4+o(1)) setup and B^(5/4+o(1)) complete-batch work. "
            "Preregister at least three comparable scales, measure positive "
            "row density and rank without outcome-adaptive shifts, and "
            "require exact-residual factor logs and shifted descent."
        ),
        "falsifier": (
            "Any shift is selected after relation outcomes; finite or "
            "multiscale rank remains deficient; the circuit loses "
            "opposite-shift transpose, row-sum counts, or signed exactness; "
            "setup or batch work exceeds its cap; a separate forward "
            "tangent is built; or the route consumes DLP, root, count, "
            "marginal, rank, or source oracles or fails residual, descent, "
            "memory, field, or bit costs."
        ),
        "required_artifacts": [
            "frozen_m6_symmetric_shift_reverse_only_marginal.json",
            "m6_symmetric_shift_reverse_only_marginal_cost_ledger.json",
            "m6_symmetric_shift_reverse_only_marginal_replay.json",
            "m6_symmetric_shift_reverse_only_marginal_controls.json",
            "factor_logs_and_identical_descent_r153.json",
        ],
    },
    "s46_signed_quotient_random_rank_reverse_ffe_transfer": {
        "hypothesis": (
            "The R154 signed quotient has a high-probability full-rank "
            "transition for independently generated known-A and "
            "hash-to-curve C decks, and one reverse-adjoint signed FFE "
            "operator realizes its rows within the frozen caps."
        ),
        "decisive_test": (
            "Prove a nonadaptive random-deck concentration and signed "
            "full-rank theorem with explicit failure probability, transfer "
            "it to the exact hash-to-curve sampler, and freeze one "
            "geometry-only reverse signed marker circuit with at most "
            "B^(9/4+o(1)) setup and B^(5/4+o(1)) batch work. Require "
            "integer counts, exact-residual factor logs, and identical "
            "shifted descent."
        ),
        "falsifier": (
            "The rank transition depends on verifier labels, adaptive "
            "moduli, shifts, or targets; concentration or hash-to-curve "
            "transfer fails; signed rank remains deficient; the reverse "
            "circuit exceeds either cap or loses opposite-shift, count, "
            "pivot, or signed exactness; or any DLP, root, count, marginal, "
            "rank, or source oracle is consumed."
        ),
        "required_artifacts": [
            "frozen_m6_signed_quotient_multiscale_rank.json",
            "m6_signed_quotient_multiscale_rank_cost_ledger.json",
            "m6_signed_quotient_multiscale_rank_replay.json",
            "m6_signed_quotient_multiscale_rank_controls.json",
            "factor_logs_and_identical_descent_r154.json",
        ],
    },
    "s47_convolution_tanner_contiguity_reverse_ffe": {
        "hypothesis": (
            "After logarithmic nonadaptive oversampling, the singleton M6 "
            "Tanner graph is contiguous to a prescribed-degree sparse "
            "random-matrix model after quotienting exact opposite-row "
            "dependencies, and the reverse signed FFE operator realizes "
            "the resulting rows within the frozen caps."
        ),
        "decisive_test": (
            "Prove a coupling or direct rank theorem controlling "
            "shared-deck endpoint dependencies, repeated and opposite "
            "rows, cancellations, coverage, and the 2-core through "
            "B^(3/4)log(B) relations. Transfer it to the exact "
            "hash-to-curve sampler and freeze one reverse signed marker "
            "circuit with B^(9/4) setup and B^(5/4+o(1)) batch work, "
            "followed by exact-residual logs and identical descent."
        ),
        "falsifier": (
            "No contiguity or direct rank bound survives shared-deck "
            "correlations; projectively distinct rows remain deficient "
            "after logarithmic oversampling; hash-to-curve transfer fails; "
            "the circuit exceeds either cap or loses signed, pivot, count, "
            "or opposite-shift exactness; or a DLP, root, count, marginal, "
            "rank, or source oracle is consumed."
        ),
        "required_artifacts": [
            "frozen_m6_singleton_relation_hypergraph_rank.json",
            "m6_singleton_relation_hypergraph_rank_cost_ledger.json",
            "m6_singleton_relation_hypergraph_rank_replay.json",
            "m6_singleton_relation_hypergraph_rank_controls.json",
            "factor_logs_and_identical_descent_r155.json",
        ],
    },
    "s48_projective_singleton_direct_rank_reverse_ffe": {
        "hypothesis": (
            "For A-pair count B^(1/12) and logarithmically oversampled "
            "singleton relations, projective opposite-row quotienting "
            "leaves enough distinct rows for full signed C-log rank with "
            "high probability, and the reverse signed FFE operator "
            "realizes those rows within the frozen caps."
        ),
        "decisive_test": (
            "Prove a direct rank bound for B^(3/4)log(B) singleton rows "
            "that controls coverage, projective row count, shared-deck "
            "dependencies, and residual nullity as A diversity grows. "
            "Transfer it to the exact hash-to-curve sampler and freeze one "
            "reverse signed marker circuit with B^(9/4) setup and "
            "B^(5/4+o(1)) batch work, followed by exact-residual logs and "
            "identical descent."
        ),
        "falsifier": (
            "Projectively distinct row supply or rank remains deficient "
            "with nonnegligible probability after logarithmic "
            "oversampling; the proof assumes independent labels, shifts, "
            "or rows not present in the convolution family; hash-to-curve "
            "transfer fails; the reverse circuit exceeds a cap or loses "
            "signed, pivot, count, or opposite-shift exactness; or any "
            "DLP, root, count, marginal, rank, or source oracle is used."
        ),
        "required_artifacts": [
            "frozen_m6_a_diversity_projective_rank.json",
            "m6_a_diversity_projective_rank_cost_ledger.json",
            "m6_a_diversity_projective_rank_replay.json",
            "m6_a_diversity_projective_rank_controls.json",
            "factor_logs_and_identical_descent_r156.json",
        ],
    },
    "s49_public_short_relation_rank_reverse_ffe_descent": {
        "hypothesis": (
            "The public hash-to-curve short-relation coefficient system has "
            "injective feasible C6 coefficient maps and full projective "
            "signed rank with high probability after logarithmic "
            "oversampling, and a reverse signed FFE operator can replace "
            "explicit C6 enumeration within the frozen caps."
        ),
        "decisive_test": (
            "Prove generic-prime high-probability coefficient-map "
            "injectivity, projective row supply, and full signed rank for "
            "the public short-relation system. Freeze a reverse signed FFE "
            "operator with B^(9/4+o(1)) setup and B^(5/4+o(1)) batch work, "
            "then recover exact-residual factor logs and perform identical "
            "target descent without explicit C6 endpoint enumeration."
        ),
        "falsifier": (
            "Coefficient-map collisions or rank failure remain "
            "nonnegligible; the proof imports independent rows absent from "
            "the shared A/C deck process; the reverse operator materializes "
            "C6 endpoints, exceeds either cap, loses signed or opposite-row "
            "exactness, or uses a DLP, root, count, marginal, rank, or "
            "source oracle; or target descent does not use the identical "
            "public relation mechanism."
        ),
        "required_artifacts": [
            "frozen_m6_hash_to_curve_projective_rank.json",
            "m6_hash_to_curve_projective_rank_cost_ledger.json",
            "m6_hash_to_curve_projective_rank_replay.json",
            "m6_hash_to_curve_projective_rank_controls.json",
            "factor_logs_and_identical_descent_r157.json",
        ],
    },
    "s50_conditioned_short_relation_full_rank_reverse_ffe_descent": {
        "hypothesis": (
            "After discarding the asymptotically vanishing fraction of "
            "collision-ambiguous C6 sources, the conditioned public "
            "hash-to-curve short-relation process covers every factor-base "
            "column and has full projective signed rank with high "
            "probability, and one reverse signed FFE operator realizes the "
            "relations within the frozen caps."
        ),
        "decisive_test": (
            "Upgrade the R158 pairwise relation-event theorem to zero "
            "uncovered columns and full projective rank for the exact "
            "conditioned hash-to-curve sampler, using a higher-moment, "
            "dependency-graph, or direct sparse-rank argument. Freeze a "
            "reverse signed FFE operator with B^(9/4+o(1)) setup and "
            "B^(5/4+o(1)) batch work, then recover exact-residual factor "
            "logs and perform identical target descent without explicit "
            "C6 endpoint enumeration."
        ),
        "falsifier": (
            "The full-coverage or rank argument assumes global coefficient-"
            "map injectivity, independent rows absent from the shared A/C "
            "deck process, or an unconditioned sampler; residual coverage "
            "or nullity failure remains nonnegligible; the reverse operator "
            "materializes C6 endpoints, exceeds either cap, loses signed or "
            "opposite-row exactness, or uses a DLP, root, count, marginal, "
            "rank, or source oracle; or target descent does not use the "
            "identical public relation mechanism."
        ),
        "required_artifacts": [
            "frozen_m6_short_relation_near_injectivity_supply.json",
            "m6_short_relation_near_injectivity_supply_cost_ledger.json",
            "m6_short_relation_near_injectivity_supply_replay.json",
            "m6_short_relation_near_injectivity_supply_controls.json",
            "factor_logs_and_identical_descent_r158.json",
        ],
    },
    "s51_batched_positive_c6_reverse_ffe_source_locator": {
        "hypothesis": (
            "A public reverse FFE operator can locate unique positive-C6 "
            "sources for the R159 batch of B^(5/4)log(B) random-diagonal "
            "known targets using at most B^(9/4+o(1)) setup and "
            "B^(5/4+o(1)) work, without materializing the B^(9/2) "
            "positive-C6 endpoint universe."
        ),
        "decisive_test": (
            "Freeze a signed-weight-separable reverse FFE DAG for positive "
            "C6 source location. Charge all field operations, target "
            "specialization, state, memory, output, and false positives; "
            "require B^(9/4+o(1)) setup and B^(5/4+o(1)) total work for "
            "B^(5/4)log(B) arbitrary public targets. Replay the admitted "
            "random-diagonal factor-log solve and identical Q+tG descent "
            "using only returned sources."
        ),
        "falsifier": (
            "The operator materializes C6 endpoints, source tuples, "
            "B^(9/2) state, q modes, or q targets; its target "
            "specialization, false-positive filtering, output, memory, "
            "field, or bit cost exceeds a cap; it consumes a DLP, root, "
            "count, marginal, rank, or source oracle; it relies on "
            "deterministic hash-to-curve pseudorandomness without a "
            "stated model; or logs and descent do not use the identical "
            "returned-source mechanism."
        ),
        "required_artifacts": [
            "frozen_m6_random_diagonal_known_target_rank.json",
            "m6_random_diagonal_known_target_rank_cost_ledger.json",
            "m6_random_diagonal_known_target_rank_replay.json",
            "m6_random_diagonal_known_target_rank_controls.json",
            "factor_logs_and_identical_descent_r159.json",
        ],
    },
    "s52_coordinate_specific_s7_reverse_ffe_source_locator": {
        "hypothesis": (
            "A coordinate-aware S7 summation-polynomial, resultant, or FFE "
            "circuit can return positive-C6 sources for the complete R159 "
            "random-diagonal target batch within B^(9/4+o(1)) setup and "
            "B^(5/4+o(1)) work by exploiting finite-field structure that "
            "is unavailable from opaque generic encodings."
        ),
        "decisive_test": (
            "Freeze a compact factor-base x-polynomial and a source-returning "
            "S7/resultant/FFE circuit. Identify every coordinate operation "
            "that prevents generic-group simulation; charge polynomial "
            "construction, target specialization, field operations, memory, "
            "false positives, and source extraction. Require the R159 caps, "
            "then replay the admitted random-diagonal factor-log solve and "
            "identical Q+tG descent without C3+C3 materialization."
        ),
        "falsifier": (
            "The circuit is encoding-invariant, materializes B^(9/2) C3+C3 "
            "or positive-C6 endpoints, performs B^(7/2) target scans, exceeds "
            "either cap, returns only a decision/count without source "
            "backpointers, consumes a DLP/root/count/rank/source oracle, relies "
            "on an unproved factor-base distribution transfer, or fails the "
            "factor-log and identical-descent replay."
        ),
        "required_artifacts": [
            "frozen_m6_positive_c6_generic_locator_reduction.json",
            "m6_positive_c6_generic_locator_reduction_cost_ledger.json",
            "m6_positive_c6_generic_locator_reduction_replay.json",
            "m6_positive_c6_generic_locator_reduction_controls.json",
            "factor_logs_and_identical_descent_r160.json",
        ],
    },
    "s53_target_batched_signed_divisor_modular_composition_gcd": {
        "hypothesis": (
            "One frozen signed C3 divisor (U,V) admits a many-inner "
            "target-batched modular-composition and gcd source adjoint that "
            "returns all degree-at-most-20 positive-C6 source factors for "
            "the R159 target batch in B^(5/4+o(1)) work after "
            "B^(9/4+o(1)) setup."
        ),
        "decisive_test": (
            "Freeze U,V and the R159 public targets. Specify an arithmetic "
            "DAG or transposed operator for all target-dependent inner maps "
            "phi_j and psi_j; charge preprocessing, field operations, gcds, "
            "memory, exceptional u_j-X roots, source factorization, and "
            "backpointers. Require B^(5/4+o(1)) complete-batch work without "
            "materializing a degree-B^(9/4) state per target, then replay "
            "factor logs and identical descent from the returned sources."
        ),
        "falsifier": (
            "The operator performs independent degree-B^(9/4) composition "
            "or gcd per target, materializes B^(7/2) aggregate state/work, "
            "returns only decisions or counts, omits exceptional roots, "
            "consumes a DLP/root/count/rank/source oracle, assumes an "
            "unstated distribution transfer, or fails the identical "
            "returned-source factor-log and descent replay."
        ),
        "required_artifacts": [
            "frozen_m6_signed_c3_divisor_translation_gcd.json",
            "m6_signed_c3_divisor_translation_gcd_cost_ledger.json",
            "m6_signed_c3_divisor_translation_gcd_replay.json",
            "m6_signed_c3_divisor_translation_gcd_controls.json",
            "factor_logs_and_identical_descent_r161.json",
        ],
    },
    "s54_aggregate_nonlinear_signed_divisor_below_rho": {
        "hypothesis": (
            "The target-varying lambda-square, composition, and source-gcd "
            "layer can be aggregated across the R159 targets in total "
            "B^(5/2-delta) work for some delta>0, even if it misses the "
            "stricter B^(5/4) batch phase cap."
        ),
        "decisive_test": (
            "Freeze one aggregate arithmetic DAG starting after R162's "
            "batched denominator inverse. Charge every nonlinear operation, "
            "state, target-label assignment, exceptional root, gcd factor, "
            "and backpointer. Require total exponent strictly below B^(5/2), "
            "then replay factor logs and identical descent using only the "
            "target-labeled returned sources."
        ),
        "falsifier": (
            "The construction performs independent degree-B^(9/4) "
            "composition per target, merely returns linear functionals, "
            "decisions, or unlabeled union factors, reaches B^(5/2), consumes "
            "a DLP/root/count/rank/source oracle, omits exceptional roots, or "
            "fails the returned-source log and descent replay."
        ),
        "required_artifacts": [
            "frozen_m6_batch_inverse_transpose_modcomp_fit.json",
            "m6_batch_inverse_transpose_modcomp_fit_cost_ledger.json",
            "m6_batch_inverse_transpose_modcomp_fit_replay.json",
            "m6_batch_inverse_transpose_modcomp_fit_controls.json",
            "batch_inverse_functional_transpose_r162.json",
        ],
    },
    "s55_unlabeled_aggregate_union_factor_constructor": {
        "hypothesis": (
            "The target-coupled signed membership projectors can be "
            "aggregated modulo the frozen C3 divisor into one unlabeled "
            "degree-O(B^(3/4)) union factor in total work below B^(5/2), "
            "without independent modular composition per target."
        ),
        "decisive_test": (
            "Freeze an arithmetic DAG for the product of same-target x/y "
            "membership projectors modulo U. Charge composition, Fermat "
            "projection, product/remainder trees, gcd, state, and exceptional "
            "roots. Require total work below B^(5/2), preferably B^(9/4), "
            "then use R163's B^2 postprocessor to recover labels, sources, "
            "factor logs, and identical descent."
        ),
        "falsifier": (
            "The constructor forms every U(phi_j) or V(phi_j), separates x "
            "and y products across targets, materializes B^(7/2) work/state, "
            "reaches B^(5/2), consumes an oracle, omits exceptional roots, "
            "or the R163 postprocessor fails to recover and verify all "
            "required sources."
        ),
        "required_artifacts": [
            "frozen_m6_aggregate_union_factor_label_recovery.json",
            "m6_aggregate_union_factor_label_recovery_cost_ledger.json",
            "m6_aggregate_union_factor_label_recovery_replay.json",
            "m6_aggregate_union_factor_label_recovery_controls.json",
            "aggregate_union_target_labels_and_backpointers_r163.json",
        ],
    },
    "s56_output_sensitive_elliptic_translation_target_norm": {
        "hypothesis": (
            "The regular-branch norm gcd(U,Norm_target(a+R*b)) can be "
            "constructed from the shared elliptic-translation action in "
            "total work below B^(5/2), without materializing the n-by-N "
            "residual table or a standard coefficient-ring resultant."
        ),
        "decisive_test": (
            "Freeze U,V, the collision-safe target-label algebra W, and one "
            "randomizer polynomial R. Specify an output-sensitive arithmetic "
            "DAG for the regular translation norm and charge translation "
            "actions, coefficient-ring state, products/resultants, gcd, "
            "factorization, expected false positives, and verification. "
            "Require total work strictly below B^(5/2), preferably B^(9/4), "
            "then invoke R164's exact verifier and R163's B^2 labels and "
            "source-backpointer replay."
        ),
        "falsifier": (
            "The construction forms independent a_j,b_j tables, uses the "
            "standard B^(7/2) coefficient-ring norm, assigns unit cost to a "
            "D5/subresultant or norm oracle, reaches B^(5/2), loses true "
            "roots, omits the R164 incidence split, consumes a DLP/root/count/"
            "rank/source oracle, or fails exact source replay."
        ),
        "required_artifacts": [
            "frozen_m6_randomized_target_divisor_norm_union.json",
            "m6_randomized_target_divisor_norm_union_cost_ledger.json",
            "m6_randomized_target_divisor_norm_union_replay.json",
            "m6_randomized_target_divisor_norm_union_controls.json",
            "randomized_target_label_algebra_and_false_positive_r164.json",
        ],
    },
    "s57_single_function_arbitrary_translate_product_remainder": {
        "hypothesis": (
            "For the one fixed function f_r=U(x)+r(y-V(x)), the factor "
            "gcd(U,product_j f_r(T_j-P)) can be constructed in total work "
            "below B^(5/2) by an output-sensitive elliptic translation or "
            "remainder algorithm, without expanding its degree-2nN divisor."
        ),
        "decisive_test": (
            "Freeze U,V, one global scalar r, f_r, and the N public targets. "
            "Specify an arithmetic DAG for the arbitrary-target translate "
            "product modulo U; charge target-divisor preprocessing, function "
            "translations, divisor or Riemann-Roch state, products, remainders, "
            "gcd, factorization, pole-equality correction, false positives, "
            "and verification. Require total work strictly below B^(5/2), "
            "preferably B^(9/4), then replay R165 verification and R163 source "
            "labels and backpointers."
        ),
        "falsifier": (
            "The construction expands a 2nN divisor or n-by-N value table, "
            "uses a standard B^(7/2) product/resultant, assumes the arbitrary "
            "target divisor has a Miller scalar chain, assigns unit cost to a "
            "translation/norm/3SUM oracle, reaches B^(5/2), mishandles P=T "
            "poles or P=-T tangents, consumes a DLP/root/count/rank/source "
            "oracle, or fails exact source replay."
        ),
        "required_artifacts": [
            "frozen_m6_global_randomizer_elliptic_translate_product.json",
            "m6_global_randomizer_elliptic_translate_product_cost_ledger.json",
            "m6_global_randomizer_elliptic_translate_product_replay.json",
            "m6_global_randomizer_elliptic_translate_product_controls.json",
            "global_randomizer_fixed_function_translate_product_r165.json",
        ],
    },
    "s58_output_sensitive_kummer_translate_product_remainder": {
        "hypothesis": (
            "For the deterministic Kummer function f_0=U(x), the factor "
            "gcd(U,product_j U(x(T_j-P))) can be constructed in total work "
            "below B^(5/2), preferably B^(9/4), without expanding its "
            "degree-2nN divisor or the n-by-N value table."
        ),
        "decisive_test": (
            "Freeze U,V (or U with its equivalent signed y side table) and "
            "the N public targets. "
            "Specify an arithmetic DAG for the arbitrary-target Kummer "
            "translate product modulo U; charge target preprocessing, "
            "translation actions, divisor or quotient-ring state, products, "
            "remainders, gcd, factorization, pole-equality correction, all "
            "true and opposite-sign candidates, and signed verification. "
            "Require total work strictly below B^(5/2), preferably B^(9/4), "
            "then replay R166 verification and R163 labels and backpointers."
        ),
        "falsifier": (
            "The construction expands a 2nN divisor or n-by-N value table, "
            "uses a standard B^(7/2) product/resultant, applies the signed "
            "filter to all nN pairs, assumes iid cyclic-label bounds prove "
            "deterministic hash-to-curve transfer, assigns unit cost to a "
            "Kummer-translation/norm/3SUM oracle, reaches B^(5/2), mishandles "
            "P=T poles, consumes a DLP/root/count/rank/source oracle, or fails "
            "exact signed source replay."
        ),
        "required_artifacts": [
            "frozen_m6_kummer_x_translate_signed_verification.json",
            "m6_kummer_x_translate_signed_verification_cost_ledger.json",
            "m6_kummer_x_translate_signed_verification_replay.json",
            "m6_kummer_x_translate_signed_verification_controls.json",
            "kummer_x_candidate_and_signed_false_branch_r166.json",
        ],
    },
    "s59_slp_elliptic_resultant_mod_u": {
        "hypothesis": (
            "For a degree-N generalized Miller target-divisor SLP h and the "
            "degree-n signed C3 divisor U,V, the denominator-cleared elliptic "
            "resultant Res_E(U(x(Q)),h(Q+P)) can be restricted modulo U in "
            "total work below B^(5/2), preferably B^(9/4), without an nN or "
            "n^2 value table or a degree-Theta(nN) represented resultant."
        ),
        "decisive_test": (
            "Freeze U,V (or U with its signed y side table), the degree-N "
            "generalized Miller line SLP h, its anchor and auxiliary poles, "
            "and every denominator-clearing correction. Specify an arithmetic "
            "DAG for the elliptic-resultant or tame-symbol remainder modulo U; "
            "charge SLP construction, quotient-ring or transposed state, all "
            "line restrictions, modular composition or half-GCD operations, "
            "correction units, gcd, factorization, candidate output, and signed "
            "verification. Require total work strictly below B^(5/2), "
            "preferably B^(9/4), then replay R167 reciprocity transcripts and "
            "R166 signed source recovery."
        ),
        "falsifier": (
            "The construction expands an n-by-N target table, a 2n-by-n "
            "swapped table, or a degree-Theta(nN) represented resultant; uses "
            "the standard B^(7/2) resultant or raw B^(9/2) swap; treats a "
            "generalized Miller SLP as unit-cost evaluation on all U roots; "
            "drops auxiliary corrections or candidate-zero tame-symbol "
            "semantics; reaches B^(5/2); consumes a DLP/resultant/root/count/"
            "marginal/rank/source oracle; or fails exact R167 and R166 replay."
        ),
        "required_artifacts": [
            "frozen_m6_generalized_target_divisor_weil_reciprocity_swap.json",
            "m6_generalized_target_divisor_weil_reciprocity_swap_cost_ledger.json",
            "m6_generalized_target_divisor_weil_reciprocity_swap_replay.json",
            "m6_generalized_target_divisor_weil_reciprocity_swap_controls.json",
            "generalized_miller_elliptic_resultant_swap_r167.json",
        ],
    },
    "s60_denominator_aware_elliptic_cauchy_trace_mod_u": {
        "hypothesis": (
            "For the compact degree-N rational witness Dh/h, the denominator "
            "factor of its corrected elliptic Cauchy trace on the degree-n "
            "signed divisor can be recovered modulo U below B^(5/2), preferably "
            "B^(9/4), while preserving candidate zero-divisor information."
        ),
        "decisive_test": (
            "Freeze U,V, h, Dh/h, all auxiliary logarithmic corrections, and "
            "the public target-equality removals. Specify a denominator-aware "
            "transposed trace DAG that emits gcd(U,denominator(Dlog G)); charge "
            "structured subresultants or Fitting state, displacement generators, "
            "transposed multipoint operations, zero-divisor branches, output "
            "factorization, and signed candidate verification. Require total "
            "work strictly below B^(5/2), preferably B^(9/4), then replay all "
            "R168 residues, R167 reciprocity values, and R166 signed sources."
        ),
        "falsifier": (
            "The construction forms an nN or n^2 pair table, materializes the "
            "n^2 tensor quotient, inverts h or its denominator at candidate "
            "nonunits, returns only generic-point trace values, uses the standard "
            "B^(7/2) direct or B^(9/2) swapped route, drops P=T corrections or "
            "multiplicity residues, reaches B^(5/2), consumes a unit-cost trace/"
            "inverse/resultant/root/count/marginal/rank/source oracle, or fails "
            "exact R168/R167/R166 replay."
        ),
        "required_artifacts": [
            "frozen_m6_log_derivative_elliptic_cauchy_trace.json",
            "m6_log_derivative_elliptic_cauchy_trace_cost_ledger.json",
            "m6_log_derivative_elliptic_cauchy_trace_replay.json",
            "m6_log_derivative_elliptic_cauchy_trace_controls.json",
            "log_derivative_candidate_poles_and_trace_r168.json",
        ],
    },
    "s61_fraction_free_elliptic_fitting_subresultant_mod_u": {
        "hypothesis": (
            "The zero specialization of the compact scalar resolvent can be "
            "computed as a fraction-free elliptic Fitting or subresultant "
            "denominator modulo U below B^(5/2), preferably B^(9/4), without "
            "interpolating its degree-2n lambda pencil."
        ),
        "decisive_test": (
            "Freeze U,V,F_num,F_den, their invariant derivatives, every R168 "
            "public equality correction, and the exact R169 scalar-resolvent "
            "specialization. Specify a fraction-free arithmetic DAG that emits "
            "the lambda-zero Fitting denominator or equivalent subresultant "
            "factor modulo U. Charge pseudo-division, modular composition, "
            "quotient state, custom displacement operators and generators, "
            "zero-divisor branches, factorization, candidate output, and signed "
            "verification. Require total work strictly below B^(5/2), preferably "
            "B^(9/4), then replay R169 multiplicities and R168/R167/R166 sources."
        ),
        "falsifier": (
            "The construction samples or stores 2n+1 lambda values per P, forms "
            "an nN or n^2 table or tensor quotient, uses any tested full-rank "
            "diagonal x/y displacement generator as compressed state, inverts a "
            "candidate nonunit, returns only generic-lambda values, drops the "
            "zero specialization or public corrections, reaches B^(5/2), "
            "consumes a DLP/Fitting/subresultant/root/count/marginal/rank/source "
            "oracle, or fails exact R169/R168/R167/R166 replay."
        ),
        "required_artifacts": [
            "frozen_m6_regularized_log_trace_displacement_rank.json",
            "m6_regularized_log_trace_displacement_rank_cost_ledger.json",
            "m6_regularized_log_trace_displacement_rank_replay.json",
            "m6_regularized_log_trace_displacement_rank_controls.json",
            "regularized_trace_pencil_and_displacement_r169.json",
        ],
    },
    "s62_slp_streaming_output_sensitive_target_norm_mod_u": {
        "hypothesis": (
            "The compact generalized Miller target-divisor SLP can stream the "
            "corrected lambda-zero norm directly into F_p[X]/U in total work "
            "below B^(5/2), preferably B^(9/4), without representing each "
            "target factor in the quotient ring."
        ),
        "decisive_test": (
            "Freeze U,V, the generalized Miller SLP for F_num/F_den, every "
            "R167 correction unit, and the R170 target-norm identity. Specify "
            "an arithmetic DAG that emits the aggregate element or gcd(U,C); "
            "charge every line merge, quotient restriction, norm update, "
            "pseudo-division, zero-divisor split, output factorization, and "
            "signed verification. Require total work strictly below B^(5/2), "
            "preferably B^(9/4), then replay all R170/R169/R167/R166 sources."
        ),
        "falsifier": (
            "The route materializes N dense elements of F_p[X]/U, an nN "
            "coefficient body, an n^2 Fitting matrix, or N independent modular "
            "compositions; uses finite coefficient density as a circuit lower "
            "bound; invokes complex sigma-function factorization without a "
            "finite-field arithmetic DAG; inverts candidate nonunits; reaches "
            "B^(5/2); consumes a DLP/norm/Fitting/subresultant/root/count/"
            "marginal/rank/source oracle; or fails exact replay."
        ),
        "required_artifacts": [
            "frozen_m6_lambda_zero_fitting_target_norm_dedup.json",
            "m6_lambda_zero_fitting_target_norm_dedup_cost_ledger.json",
            "m6_lambda_zero_fitting_target_norm_dedup_replay.json",
            "m6_lambda_zero_fitting_target_norm_dedup_controls.json",
            "lambda_zero_fitting_norm_and_density_r170.json",
        ],
    },
    "s63_nonlocal_batched_elliptic_leaf_translate_product": {
        "hypothesis": (
            "A nonlocal elliptic batch operator can fuse the N surviving target "
            "leaves against the degree-n selected divisor and emit the aggregate "
            "target norm modulo U below B^(5/2), preferably B^(9/4), without "
            "node-local or pair-local expansion."
        ),
        "decisive_test": (
            "Freeze U,V, the compact target and selected divisors, the exact R171 "
            "signed leaf ledger, and all R167 correction semantics. Specify a "
            "point-list-to-product arithmetic DAG that emits product_j "
            "U(x(T_j-P)) modulo U or gcd(U,C). Charge divisor compilation, "
            "elliptic addition charts, any bivariate resultant or transposed "
            "multipoint input body, quotient reduction, zero-divisor branches, "
            "factorization, candidate output, and signed verification. Require "
            "softly O(n+N) work and total cost below B^(5/2), then replay all "
            "R171/R170/R167/R166 controls."
        ),
        "falsifier": (
            "The route visits the n-by-N point/leaf grid, materializes N dense "
            "elements of F_p[X]/U or a degree-nN body, invokes generic "
            "multipoint or truncated-resultant theorems without constructing "
            "their represented input inside the budget, specializes individual "
            "line ratios at f0(-P)=0, inverts candidate nonunits, reaches "
            "B^(5/2), consumes a DLP/norm/resultant/root/count/marginal/rank/"
            "source oracle, or fails exact replay."
        ),
        "required_artifacts": [
            "frozen_m6_balanced_miller_tree_norm_streaming.json",
            "m6_balanced_miller_tree_norm_streaming_cost_ledger.json",
            "m6_balanced_miller_tree_norm_streaming_replay.json",
            "m6_balanced_miller_tree_norm_streaming_controls.json",
            "balanced_miller_tree_and_leaf_cancellation_r171.json",
        ],
    },
    "s64_factored_self_s3_resultant_mod_u": {
        "hypothesis": (
            "The product of N quadratic Semaev S3 kernels admits a factored "
            "self-resultant remainder modulo the degree-n selected divisor U "
            "in softly O(n+N) work, without expanding its N^2 coefficient body."
        ),
        "decisive_test": (
            "Freeze U, the compact target-x divisor W, the N explicit S3 "
            "factors, and the exact R172 target-sign conjugate identity. Specify "
            "an arithmetic DAG for Res_Z(U(Z),product_j S3(X,Z,u_j)) modulo "
            "U(X). Charge factor ingestion, product or subproduct trees, all "
            "polynomial-matrix or transposed inputs, quotient reduction, "
            "zero-divisor branches, output factorization, candidate output, and "
            "signed verification. Require softly O(n+N) work and total cost "
            "strictly below B^(5/2), then replay all R172/R171/R167/R166 controls."
        ),
        "falsifier": (
            "The route materializes the (2N+1)^2 reverse-resultant grid, visits "
            "nN point/factor pairs, builds an nN or n^2 polynomial matrix, "
            "performs N independent modular compositions, inverts candidate "
            "nonunits, reaches B^(5/2), assumes a local x^k truncation is an "
            "arbitrary squarefree-U remainder, consumes a DLP/norm/resultant/"
            "root/count/marginal/rank/source oracle, or fails exact replay."
        ),
        "required_artifacts": [
            "frozen_m6_target_sign_conjugate_s3_self_resultant.json",
            "m6_target_sign_conjugate_s3_self_resultant_cost_ledger.json",
            "m6_target_sign_conjugate_s3_self_resultant_replay.json",
            "m6_target_sign_conjugate_s3_self_resultant_controls.json",
            "target_conjugate_s3_factored_self_resultant_r172.json",
        ],
    },
    "s65_commutative_target_sign_divisor_pushforward_mod_u": {
        "hypothesis": (
            "The separable S3 discriminant square root 4*V(X)*V_T(u) lets the "
            "two target-sign branches be compiled as compact commutative "
            "divisor pushforwards modulo arbitrary squarefree U in total work "
            "strictly below B^(5/2)."
        ),
        "decisive_test": (
            "Freeze U,V, the compact target divisor W,V_T, the exact R172 sign "
            "branches, and the R173 determinant/discriminant identities. "
            "Specify a transposed modular-composition, multipoint, or equivalent "
            "commutative arithmetic DAG that emits both aggregate branches or "
            "their product modulo U. Charge all target-dependent preprocessing, "
            "squarefree-modulus transforms, divisor pushforwards, quotient "
            "reductions, zero-divisor branches, factorization, candidate output, "
            "and signed verification. Require total work strictly below B^(5/2), "
            "preferably B^(9/4), then replay all R173/R172/R171/R167/R166 controls."
        ),
        "falsifier": (
            "The route materializes either ordered 2 by 2 matrix product or an "
            "N^2 coefficient body, visits nN point/factor pairs, performs N "
            "target-specific modular compositions, uses target-dependent setup "
            "without charging it, merely restates the R172 sign split, reaches "
            "B^(5/2), inverts candidate nonunits, assumes a unit-cost norm, "
            "resultant, root, or multipoint oracle, or fails exact replay."
        ),
        "required_artifacts": [
            "frozen_m6_s3_determinantal_transfer_noncommutativity.json",
            "m6_s3_determinantal_transfer_noncommutativity_cost_ledger.json",
            "m6_s3_determinantal_transfer_noncommutativity_replay.json",
            "m6_s3_determinantal_transfer_noncommutativity_controls.json",
            "s3_determinantal_transfer_noncommutativity_r173.json",
        ],
    },
    "s66_fused_factored_dual_chow_outer_norm_mod_u": {
        "hypothesis": (
            "The compact signed line-product SLP and target charts admit a "
            "nested Fermat-projector moment operator that emits the 2c weighted "
            "source moments in softly O(n+N+c) work without represented nN state."
        ),
        "decisive_test": (
            "Freeze U,V, the R181 kernel hashes, the R182 target charts, and the "
            "R183 weighted-moment/Prony output contract. Apply one SLP-direct "
            "nested projector-moment operator that emits 2c scalar moments without "
            "representing R, the Fermat projector, an A-valued target Krylov "
            "sequence, or any nN tensor body. Charge every operation, require "
            "softly O(n+N+c), and test seed 18308 plus a second strided family."
        ),
        "falsifier": (
            "The route emits or visits an nN remainder, projector, or trace body; "
            "uses a source-blind target trace that loses source labels; represents "
            "noncandidate inverses or a triangular norm; drops either elliptic "
            "target chart; or assumes a candidate, determinant, resultant, norm, "
            "root, count, rank, or power-projection oracle."
        ),
        "required_artifacts": [
            "frozen_m6_sparse_projector_prony_locator.json",
            "m6_sparse_projector_prony_locator_cost_ledger.json",
            "m6_sparse_projector_prony_locator_replay.json",
            "m6_sparse_projector_prony_locator_controls.json",
            "sparse_projector_prony_locator_applicability_r183.json",
        ],
    },
    "s67_reusable_scalar_subset_incidence_oracle": {
        "hypothesis": (
            "After one softly O(n+N) coordinate-specific preprocessing, the "
            "exact R174 signed product over any balanced selected subset S can "
            "be returned in softly O(|S|+N) work, making R175 balanced group "
            "testing output-sensitive and strictly below B^(5/2)."
        ),
        "decisive_test": (
            "Freeze U,V, the N target points, all R174 secant and geometric-"
            "tangent charts, and the R175 balanced-node U_S,V mod U_S "
            "descriptors. Specify one reusable arithmetic DAG with softly "
            "O(n+N) preprocessing and softly O(|S|+N) exact scalar-query work. "
            "Charge subproduct and remainder trees, target-factor ingestion, "
            "all transformed state, quotient reductions, zero-divisor branches, "
            "candidate output, R163 label/backpointer recovery, and signed "
            "verification. Replay all R175/R174/R166 controls and require total "
            "cost strictly below B^(5/2)."
        ),
        "falsifier": (
            "The route enumerates leaves, expands N targets for every selected "
            "pair or tree node, represents Theta(N^2) target-Chow or Theta(n^2) "
            "selected-Chow coefficients, performs target-dependent preprocessing "
            "per query, drops the tangent chart, inverts candidate nonunits, "
            "reaches B^(5/2), or assumes a unit-cost multipoint, norm, resultant, "
            "root, count, marginal, rank, source, or generic locator oracle."
        ),
        "required_artifacts": [
            "frozen_m6_scalar_subset_incidence_group_testing.json",
            "m6_scalar_subset_incidence_group_testing_cost_ledger.json",
            "m6_scalar_subset_incidence_group_testing_replay.json",
            "m6_scalar_subset_incidence_group_testing_controls.json",
            "scalar_subset_incidence_group_testing_r175.json",
        ],
    },
    "s68_factored_trilinear_elliptic_resultant": {
        "hypothesis": (
            "The principal target scalar h(A*D) can be evaluated directly from "
            "compact signed-divisor descriptors for A and D and the degree-N "
            "target witness h, without representing the degree-mn Pontryagin cycle."
        ),
        "decisive_test": (
            "Freeze U_A,V_A, fixed U_D,V_D, the R167 principal target witness h, "
            "the R176 completion and reciprocity identities, and the R175 query "
            "tree. Specify a factored trilinear elliptic-resultant arithmetic DAG "
            "with one softly O(n+N) reusable setup and softly O(m+N) work per "
            "balanced node. Charge divisor ingestion, group-law charts, target "
            "witness construction, all transformed state, zero specializations, "
            "candidate output, R163 labels/backpointers, and signed verification. "
            "Replay all R176/R175/R167 controls and require total work strictly "
            "below B^(5/2)."
        ),
        "falsifier": (
            "The route emits mn pair sums, builds a degree-mn principal function "
            "or Miller program, represents an n^2 tensor or displacement kernel, "
            "uses target-dependent per-node setup, drops candidate-zero Fitting "
            "information, inverts candidate nonunits, reaches B^(5/2), or assumes "
            "a unit-cost resultant, norm, root, count, marginal, rank, source, or "
            "generic locator oracle."
        ),
        "required_artifacts": [
            "frozen_m6_principal_target_pontryagin_resultant.json",
            "m6_principal_target_pontryagin_resultant_cost_ledger.json",
            "m6_principal_target_pontryagin_resultant_replay.json",
            "m6_principal_target_pontryagin_resultant_controls.json",
            "principal_target_pontryagin_resultant_r176.json",
        ],
    },
    "s69_output_sensitive_marked_fitting_locator": {
        "hypothesis": (
            "The first nonzero lambda coefficient of the global marked norm "
            "can be emitted from compact U,V,h in time proportional to the "
            "input and signed-incidence output, without constructing the pair "
            "algebra or adaptive subset queries."
        ),
        "decisive_test": (
            "Freeze compact U,V, the R176 principal target witness h, the R177 "
            "kernel dimension M, restricted X1 characteristic polynomial, and "
            "candidate gcd. Specify a fraction-free marked Fitting or "
            "subresultant arithmetic DAG that computes M and "
            "det(AI-X_1|ker K) in softly O(n+N+M) total work. Charge divisor "
            "and target ingestion, all transformed state, coefficient "
            "extraction, candidate output, R163 labels/backpointers, and signed "
            "verification. Replay all R177/R176 controls and require total work "
            "strictly below B^(5/2)."
        ),
        "falsifier": (
            "The route enumerates n^2 pairs, represents an n^2 tensor or "
            "generic matrix pencil, computes the full lambda/A determinant body, "
            "uses M^2 marker interpolation, inverts candidate nonunits, reaches "
            "B^(5/2), or assumes a unit-cost Fitting, kernel, resultant, root, "
            "count, marginal, rank, source, or generic locator oracle."
        ),
        "required_artifacts": [
            "frozen_m6_global_marked_fitting_locator.json",
            "m6_global_marked_fitting_locator_cost_ledger.json",
            "m6_global_marked_fitting_locator_replay.json",
            "m6_global_marked_fitting_locator_controls.json",
            "global_marked_fitting_locator_r177.json",
        ],
    },
    "routing_intervention_generalization": {
        "hypothesis": "One fixed routing rule recovers natural-route rank on unseen curves.",
        "decisive_test": "Freeze the rule on development cells and replay it unchanged on every prospective cell.",
        "falsifier": "The fixed route fails to improve prospective rank or violates exact replay.",
        "required_artifacts": ["frozen_routing_rule.json", "prospective_transfer.json"],
    },
    "routing_specificity_control": {
        "hypothesis": "Fixed-route rank recovery is specific to the coordinate policy rather than a generic scheduler effect.",
        "decisive_test": "Compare paired fixed-minus-natural rank gains for two_map_union against all matched hash policies.",
        "falsifier": "The coordinate-policy gain does not exceed the mean matched hash-control gain.",
        "required_artifacts": ["matched_hash_routing_control.json", "routing_specificity_summary.json"],
    },
    "rank_to_verified_log_probe": {
        "hypothesis": "Full relation rank survives anchored elimination and yields correct factor logs.",
        "decisive_test": "Independently solve the anchored system and verify every recovered logarithm on-curve.",
        "falsifier": "Any recovered factor logarithm fails or the augmented rank is incompatible.",
        "required_artifacts": ["anchored_elimination.json", "factor_log_replay.json"],
    },
    "factor_logs_to_target_descent_probe": {
        "hypothesis": "Frozen verified factor logs support independent descent of unseen public targets.",
        "decisive_test": "Run separate target descent without changing the relation or factor-log set.",
        "falsifier": "Any prospective target fails recovery or admits an invalid candidate.",
        "required_artifacts": ["target_descent.json", "target_replay.json"],
    },
    "end_to_end_cost_reduction": {
        "hypothesis": "The verified natural pipeline can cross the charged rho gate without changing its evidence set.",
        "decisive_test": "Measure source generation, retained traffic, matrix work, and descent at three sizes.",
        "falsifier": "The fitted total exponent is not below the preregistered gate or any cell exceeds rho.",
        "required_artifacts": ["charged_costs.json", "scaling_fit.json"],
    },
    "independent_claim_audit": {
        "hypothesis": "The complete natural-route claim survives an implementation-independent replay.",
        "decisive_test": "Rebuild collisions, matrices, logs, descents, and costs from the bound source JSON.",
        "falsifier": "Any claim-critical field or source hash fails to match.",
        "required_artifacts": ["independent_audit.json", "audit_transcript.md"],
    },
    "shoup_pressure_scaling_probe": {
        "hypothesis": (
            "Single-scale experiments can hide generic-bound pressure. A stable multiscale cost fit is required "
            "before claiming any asymptotic Shoup-bound progress."
        ),
        "decisive_test": (
            "Collect exact, verified, multiscale run points and fit charged total-field "
            "operations against group order; require exponent strictly below 0.5 with "
            "explicit scaling residual checks."
        ),
        "falsifier": (
            "Any fitted exponent is >= 0.5, or multiscale points are not exactly comparable, or scaling residuals invalidate the fit."
        ),
        "required_artifacts": [
            "multiscale_cost_points.json",
            "fitted_exponent_with_ci.json",
            "scaling_residual_audit.md",
        ],
    },
}


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 10)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:16], "big")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified_logs(config: dict[str, Any]) -> bool:
    return bool(
        config.get("factor_logs_available")
        and int(config.get("factor_log_verification_failures") or 0) == 0
    )


def exact_config(config: dict[str, Any]) -> bool:
    validation = config.get("validation") or {}
    return bool(validation) and all(bool(value) for value in validation.values())


def summation_ffe_markers(config: dict[str, Any]) -> list[str]:
    """Collect config field markers indicating summation-polynomial / FFE signal."""
    markers: list[str] = []
    for key, value in (config or {}).items():
        if not isinstance(key, str):
            continue
        key_lower = key.lower()
        if any(term in key_lower for term in SUMMATION_FFE_HINT_TERMS):
            markers.append(key)
            continue
        if isinstance(value, str) and any(term in value.lower() for term in SUMMATION_FFE_HINT_TERMS):
            markers.append(f"{key}::{value}")
    return sorted(set(markers))


def parse_note_url(url_text: str) -> dict[str, str]:
    """Parse a tweet-like source URL into stable metadata fields."""
    raw = (url_text or "").strip()
    if not raw:
        return {
            "source_post_url": TWEET_POST_URL,
            "source_post_url_with_query": DEFAULT_NOTE_URL,
            "source_query": TWEET_POST_QUERY,
            "source_post_id": "2076737985559822734",
            "source_author": KNOWN_TWEET_SOURCE_AUTHOR.get("2076737985559822734", ""),
        }

    parsed = urlsplit(raw)
    if not parsed.netloc and not parsed.scheme:
        fallback = urlsplit(f"//{raw}")
        parsed = fallback if fallback.netloc else parsed
    path = (parsed.path or "").strip("/")
    segments = [segment for segment in path.split("/") if segment]

    source_query = f"?{parsed.query}" if parsed.query else ""
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    if host in {"m.x.com", "mobile.x.com", "m.twitter.com", "www.twitter.com"}:
        host = "x.com"
    if host == "twitter.com":
        host = "x.com"
    source_scheme = (parsed.scheme or "").lower()
    if source_scheme == "http":
        source_scheme = "https"

    post_id = ""
    if "status" in segments:
        idx = segments.index("status")
        if idx + 1 < len(segments):
            post_id = segments[idx + 1]
    if "status" not in segments and len(segments) > 1 and segments[0].isdigit():
        post_id = segments[0]

    author = ""
    if segments:
        if segments[0] not in {"status", "i", "web"}:
            author = segments[0]
        elif len(segments) >= 2 and segments[0] == "i":
            if segments[1] in {"status", "web"}:
                author = ""
            else:
                author = segments[1]
    if not author and post_id:
        author = KNOWN_TWEET_SOURCE_AUTHOR.get(post_id, "")

    is_twitter_host = host in {"x.com", "twitter.com"}
    if source_scheme:
        canonical_scheme = source_scheme
    elif source_query or path:
        canonical_scheme = "https"
    else:
        canonical_scheme = "https"

    if is_twitter_host and post_id:
        if author:
            source_path = f"/{author}/status/{post_id}"
        else:
            source_path = f"/status/{post_id}"
        source_url = f"{canonical_scheme}://x.com{source_path}"
    else:
        canonical_path = f"/{path}" if path else ""
        source_url = (
            f"{canonical_scheme}://{host}{canonical_path}"
            if host
            else raw
        )
    source_url_with_query = f"{source_url}{source_query}" if source_query else source_url

    return {
        "source_post_url": source_url,
        "source_post_url_with_query": source_url_with_query,
        "source_query": source_query,
        "source_post_id": post_id,
        "source_author": author,
    }


def tweet_text_for(parsed: dict[str, str]) -> str | None:
    """Return exact tweet text for known source links."""
    post_id = parsed["source_post_id"]
    candidates = []
    if parsed.get("source_author") and post_id:
        candidates.append(f"{parsed['source_author']}:{post_id}")
    if post_id:
        candidates.append(post_id)
    for key in candidates:
        text = KNOWN_TWEET_SOURCE_TEXT.get(key)
        if text:
            return text
    return None


def tweet_meta_for(parsed: dict[str, str]) -> dict[str, Any]:
    """Return deterministic tweet metadata for known source links."""
    post_id = parsed["source_post_id"]
    candidates: list[str] = []
    if parsed.get("source_author") and post_id:
        candidates.append(f"{parsed['source_author']}:{post_id}")
    if post_id:
        candidates.append(post_id)
    for key in candidates:
        meta = KNOWN_TWEET_SOURCE_META.get(key)
        if meta:
            return meta
    return {}


def build_methodology(note_url: str) -> dict[str, Any]:
    parsed = parse_note_url(note_url)
    tweet_text = tweet_text_for(parsed)
    tweet_meta = tweet_meta_for(parsed)
    method = copy.deepcopy(METHODOLOGY)
    method["source_post_url"] = parsed["source_post_url"]
    method["source_post_url_with_query"] = parsed["source_post_url_with_query"]
    source_author = (
        parsed["source_author"]
        or KNOWN_TWEET_SOURCE_AUTHOR.get(parsed["source_post_id"], "")
    )
    method["tweet_guidance"].update(
        {
            "source": parsed["source_post_url"],
            "source_post_id": parsed["source_post_id"],
            "source_author": source_author,
            "source_query": parsed["source_query"],
        }
    )
    if tweet_text:
        method["tweet_guidance"].update(
            {
                "tweet_text_included": True,
                "source_summary": tweet_text,
                "tweet_text": tweet_text,
                "tweet_posted_at": tweet_meta.get("tweet_published_at", ""),
                "tweet_source_title": tweet_meta.get("paper_title", ""),
                "tweet_referenced_paper_title": tweet_meta.get(
                    "referenced_paper_title_in_post",
                    "",
                ),
                "tweet_source_url": tweet_meta.get("paper_url", ""),
                "tweet_media_urls": list(tweet_meta.get("media_urls", [])),
                "tweet_media_types": list(tweet_meta.get("media_types", [])),
                "tweet_has_media": bool(tweet_meta.get("media_urls") or tweet_meta.get("media_types")),
                "tweet_media_count": len(tweet_meta.get("media_urls", [])),
                "tweet_hashtags": list(tweet_meta.get("hashtags", [])),
                "tweet_intake_status": "Exact tweet text snapshot captured for this known source.",
                "tweet_text_source_note": (
                    "Exact tweet text is embedded for this known source "
                    "as an auditable source snapshot."
                ),
                "source_summary_is_verbatim": True,
                "tweet_text_sha256": hashlib.sha256(tweet_text.encode("utf-8")).hexdigest(),
            }
        )
    else:
        method["tweet_guidance"].update(
            {
                "source_summary": (
                    "The linked post summarized focused autoresearch behavior and "
                    "critical-experiment selection; exact text was not embedded in "
                    "this runtime context."
                ),
                "source_summary_is_verbatim": False,
                "tweet_text_included": False,
                "tweet_text": "",
                "tweet_text_sha256": "",
                "tweet_intake_status": (
                    "Source URL supplied by user; exact tweet text is not embedded in this harness."
                ),
                "tweet_text_source_note": (
                    "Exact tweet text was not retrievable from this harness context; "
                    "guidance was embedded as a non-verbatim, source-linked summary."
                ),
            }
        )
    return method


def build_tweet_source_payload(methodology: dict[str, Any]) -> dict[str, Any]:
    """Build a consistent tweet-source payload used across all report artifacts."""
    tweet_guidance = methodology["tweet_guidance"]
    return {
        "tweet_url": tweet_guidance["source"],
        "tweet_url_with_query": methodology["source_post_url_with_query"],
        "tweet_query": tweet_guidance["source_query"],
        "tweet_author": tweet_guidance["source_author"],
        "tweet_post_id": tweet_guidance["source_post_id"],
        "tweet_posted_at": tweet_guidance.get("tweet_posted_at", ""),
        "tweet_text_included": tweet_guidance["tweet_text_included"],
        "tweet_intake_mode": tweet_guidance["source_intake_mode"],
        "tweet_intake_status": tweet_guidance["tweet_intake_status"],
        "tweet_summary": tweet_guidance["source_summary"],
        "tweet_summary_is_verbatim": tweet_guidance["source_summary_is_verbatim"],
        "tweet_text_source_note": tweet_guidance["tweet_text_source_note"],
        "tweet_text": tweet_guidance.get("tweet_text", ""),
        "tweet_text_sha256": tweet_guidance.get("tweet_text_sha256", ""),
        "tweet_source_title": tweet_guidance.get("tweet_source_title", ""),
        "tweet_referenced_paper_title": tweet_guidance.get(
            "tweet_referenced_paper_title",
            "",
        ),
        "tweet_source_url": tweet_guidance.get("tweet_source_url", ""),
        "tweet_media_urls": list(tweet_guidance.get("tweet_media_urls", [])),
        "tweet_media_types": list(tweet_guidance.get("tweet_media_types", [])),
        "tweet_has_media": bool(tweet_guidance.get("tweet_has_media", False)),
        "tweet_media_count": int(tweet_guidance.get("tweet_media_count", 0)),
        "tweet_hashtags": list(tweet_guidance.get("tweet_hashtags", [])),
    }


def stage_record(
    config: dict[str, Any],
    descents: Iterable[dict[str, Any]] = (),
    *,
    require_descent: bool = False,
) -> dict[str, Any]:
    descent_rows = list(descents)
    collisions = int(config.get("collision_edge_count") or 0)
    cross_shift = int(config.get("cross_shift_collision_count") or 0)
    within_shift = int(config.get("within_shift_collision_count") or 0)
    rows = int(config.get("relation_row_count") or 0)
    rank = int(config.get("relation_rank") or 0)
    unknowns = int(config.get("unknown_factor_count") or 0)
    descent_successes = sum(bool(row.get("recovered")) for row in descent_rows)
    descent_invalid = sum(int(row.get("invalid_candidate_count") or 0) for row in descent_rows)
    full_rank = bool(config.get("full_rank")) and unknowns > 0 and rank == unknowns
    rhs_compatible = bool(config.get("rhs_compatible"))
    logs_ok = verified_logs(config)
    summation_ffe_indicators = summation_ffe_markers(config)
    summation_ffe_evidence_count = len(summation_ffe_indicators)
    preprocessing_operations = _positive_float(
        config.get("total_field_operation_estimate")
    )
    preprocessing_cost_ratio = _positive_float(
        config.get("total_field_ratio_vs_11x_rho")
    )
    descent_operation_estimates = [
        _positive_float(row.get("total_field_operation_estimate"))
        for row in descent_rows
    ]
    descent_costs_complete = bool(descent_rows) and all(
        value is not None for value in descent_operation_estimates
    )
    single_target_operations = preprocessing_operations
    if preprocessing_operations is not None and descent_costs_complete:
        single_target_operations += max(
            float(value) for value in descent_operation_estimates if value is not None
        )
    rho_field_baseline = (
        preprocessing_operations / preprocessing_cost_ratio
        if preprocessing_operations is not None and preprocessing_cost_ratio is not None
        else None
    )
    cost_ratio = (
        single_target_operations / rho_field_baseline
        if single_target_operations is not None and rho_field_baseline
        else preprocessing_cost_ratio
    )
    below_11x_rho = cost_ratio is not None and float(cost_ratio) < 1.0
    total_field_ratio_vs_rho = (
        ratio(float(cost_ratio) * SHOUP_RHO_SCALE, 1.0) if cost_ratio is not None else None
    )
    below_rho = (
        total_field_ratio_vs_rho is not None
        and float(total_field_ratio_vs_rho) < 1.0
    )

    if not exact_config(config):
        bottleneck = "exactness"
    elif collisions == 0:
        bottleneck = "residual_collision_supply"
    elif cross_shift == 0:
        bottleneck = "cross_shift_routing"
    elif rows == 0:
        bottleneck = "source_row_elimination"
    elif rank < unknowns:
        bottleneck = "relation_rank"
    elif not rhs_compatible:
        bottleneck = "rhs_compatibility"
    elif not logs_ok:
        bottleneck = "factor_log_verification"
    elif require_descent and not descent_rows:
        bottleneck = "target_descent_untested"
    elif descent_rows and descent_successes != len(descent_rows):
        bottleneck = "target_descent"
    elif require_descent and descent_rows and not descent_costs_complete:
        bottleneck = "target_descent_cost_unavailable"
    elif not below_rho:
        bottleneck = "total_cost"
    else:
        bottleneck = "none"
    summation_ffe_readiness = _infer_replay_readiness(
        summation_ffe_indicators,
        config,
    )
    summation_ffe_discovery_gate = _infer_new_factor_row_discovery_gate(
        summation_ffe_indicators,
        config,
        summation_ffe_readiness,
    )

    return {
        "attempts": int(config.get("attempts") or 0),
        "accepted_residual_events": int(config.get("accepted_residual_events") or 0),
        "collision_edges": collisions,
        "cross_shift_collision_edges": cross_shift,
        "within_shift_collision_edges": within_shift,
        "usable_relation_rows": rows,
        "relation_rank": rank,
        "unknown_factor_count": unknowns,
        "rank_fraction": ratio(rank, unknowns),
        "collision_to_row_fraction": ratio(rows, collisions),
        "row_to_independent_rank_fraction": ratio(rank, min(rows, unknowns)),
        "full_rank": full_rank,
        "rhs_compatible": rhs_compatible,
        "verified_factor_logs": logs_ok,
        "target_descent_successes": descent_successes,
        "target_descent_count": len(descent_rows),
        "target_descent_invalid_count": descent_invalid,
        "target_descent_fraction": ratio(descent_successes, len(descent_rows)),
        "preprocessing_field_operation_estimate": preprocessing_operations,
        "preprocessing_field_ratio_vs_11x_rho": preprocessing_cost_ratio,
        "target_descent_costs_complete": descent_costs_complete,
        "target_descent_field_operation_estimates": descent_operation_estimates,
        "total_field_operation_estimate": single_target_operations,
        "total_field_ratio_vs_11x_rho": cost_ratio,
        "total_field_ratio_vs_rho": total_field_ratio_vs_rho,
        "below_11x_rho_proxy": below_11x_rho,
        "below_rho_proxy": below_rho,
        "below_generic_rho_proxy": below_rho,
        "summation_ffe_markers": summation_ffe_indicators,
        "summation_ffe_evidence_count": summation_ffe_evidence_count,
        "has_summation_ffe_evidence": bool(summation_ffe_evidence_count),
        "summation_ffe_readiness": summation_ffe_readiness,
        "summation_ffe_new_factor_row_discovery_gate": summation_ffe_discovery_gate,
        "exact": exact_config(config) and descent_invalid == 0,
        "stored_but_unusable": collisions > 0 and rank < unknowns,
        "stored_but_not_log_usable": collisions > 0 and not logs_ok,
        "bottleneck": bottleneck,
    }


def synthetic_uniform_occupancy(
    config: dict[str, Any],
    *,
    order: int,
    shift_count: int,
    label: str,
) -> dict[str, Any]:
    """Execute a deterministic uniform-residual null with matched event count.

    The control intentionally stops at occupancy.  Synthetic labels have no EC
    source equation, so assigning them relation rows or logarithms would create
    an invalid cryptanalytic control.
    """
    accepted = int(config.get("accepted_residual_events") or 0)
    if order <= 1 or shift_count <= 0:
        return {"available": False, "reason": "missing curve order or shift count"}

    state = stable_seed("p1436-uniform", label)
    mask = (1 << 128) - 1
    first_shift: dict[int, int] = {}
    multiplicities: Counter[int] = Counter()
    collisions = cross_shift = within_shift = 0
    for _ in range(accepted):
        state = (state * 47026247687942121848144207491837523525 + 1) & mask
        residual = state % order
        shift = (state >> 64) % shift_count
        multiplicities[residual] += 1
        previous_shift = first_shift.get(residual)
        if previous_shift is None:
            first_shift[residual] = shift
            continue
        collisions += 1
        cross_shift += int(previous_shift != shift)
        within_shift += int(previous_shift == shift)

    observed = int(config.get("collision_edge_count") or 0)
    return {
        "available": True,
        "accepted_residual_events": accepted,
        "distinct_residual_count": len(first_shift),
        "collision_edge_count": collisions,
        "cross_shift_collision_count": cross_shift,
        "within_shift_collision_count": within_shift,
        "maximum_residual_multiplicity": max(multiplicities.values()) if multiplicities else 0,
        "observed_collision_ratio_vs_executed_uniform": ratio(observed, collisions),
        "label_sha256": hashlib.sha256(label.encode("ascii")).hexdigest(),
        "scope": "occupancy_only_no_relation_or_log_labels",
    }


def intervention_score(config: dict[str, Any]) -> tuple[Any, ...]:
    stage = stage_record(config)
    return (
        stage["exact"],
        stage["verified_factor_logs"],
        stage["rhs_compatible"],
        stage["full_rank"],
        stage["relation_rank"],
        stage["cross_shift_collision_edges"],
        -int(config.get("total_field_operation_estimate") or 0),
    )


def best_intervention(configurations: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not configurations:
        raise ValueError("P1436 cell has no configurations")
    return max(
        configurations.items(),
        key=lambda item: (intervention_score(item[1]), item[0]),
    )


def recovered_headroom(natural: float, fixed: float, oracle: float) -> float | None:
    headroom = oracle - natural
    if headroom <= 0:
        return None
    return round(max(0.0, min(1.0, (fixed - natural) / headroom)), 10)


def average(values: Iterable[float]) -> float | None:
    rows = list(values)
    if not rows:
        return None
    return round(sum(rows) / len(rows), 10)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _log_scale_cost_fit(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len({x for x, _ in points}) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    x_var = sum((x - mean_x) ** 2 for x, _ in points)
    if x_var == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / x_var
    intercept = mean_y - (slope * mean_x)
    residuals = [y - (intercept + slope * x) for x, y in points]
    sum_sq = sum(value * value for value in residuals)
    mse = sum_sq / len(points)
    y_var = sum((y - mean_y) ** 2 for _, y in points)
    r2 = 1 - (sum_sq / y_var) if y_var else None
    return {
        "exponent_in_group_order": round(slope, 10),
        "intercept_log_operations": round(intercept, 10),
        "residual_abs_max": round(max(abs(value) for value in residuals), 10),
        "residual_rmse": round(math.sqrt(mse), 10),
        "residuals": [round(value, 10) for value in residuals],
        "r2": round(r2, 10) if r2 is not None else None,
    }


def _residual_tolerance_passed(fit: dict[str, Any] | None) -> bool | None:
    if not fit:
        return None
    return bool(fit["residual_abs_max"] <= SHOUP_SCALING_RESIDUAL_TOLERANCE)


def shoup_pressure_summary(full_cells: list[dict[str, Any]]) -> dict[str, Any]:
    comparable_cells = []
    points: list[tuple[float, float]] = []
    for cell in full_cells:
        route = cell["natural_route"]
        if not route.get("exact") or not route.get("verified_factor_logs"):
            continue
        if route.get("target_descent_count") and not route.get(
            "target_descent_costs_complete"
        ):
            continue
        bits = _positive_float(cell["curve"].get("bits"))
        order = _positive_float(cell["curve"].get("order"))
        operations = _positive_float(route.get("total_field_operation_estimate"))
        cost_ratio = _positive_float(route.get("total_field_ratio_vs_11x_rho"))
        if bits is None or order is None or operations is None or cost_ratio is None:
            continue
        comparable_cells.append(
            {
                "split": cell["curve"].get("split"),
                "bits": bits,
                "seed": cell["curve"].get("seed"),
                "group_order": int(order),
                "policy": cell["policy"],
                "total_field_operation_estimate": int(operations),
                "total_field_ratio_vs_11x_rho": round(cost_ratio, 10),
                "total_field_ratio_vs_rho": route.get("total_field_ratio_vs_rho"),
            }
        )
        points.append((math.log(order), math.log(operations)))

    scale_counts = Counter(str(int(round(row["bits"]))) for row in comparable_cells)
    if len(comparable_cells) < SHOUP_MIN_SCALE_COUNT:
        status = "insufficient_comparable_cells"
    elif len(scale_counts) < SHOUP_MIN_SCALE_COUNT:
        status = "insufficient_scale_coverage"
    else:
        status = "fit_attempted"

    fit = _log_scale_cost_fit(points) if status == "fit_attempted" else None
    residual_ok = _residual_tolerance_passed(fit)
    exponent = fit.get("exponent_in_group_order") if fit else None
    exponent_below_threshold = exponent is not None and exponent < 0.5
    all_points_below_rho = bool(comparable_cells) and all(
        float(row["total_field_ratio_vs_rho"]) < 1.0
        for row in comparable_cells
        if row["total_field_ratio_vs_rho"] is not None
    )
    meets_gate = (
        fit is not None
        and len(scale_counts) >= SHOUP_MIN_SCALE_COUNT
        and residual_ok
        and exponent_below_threshold
        and all_points_below_rho
    )
    if status == "fit_attempted" and fit is None:
        status = "fit_not_available"
    elif (
        fit is not None
        and residual_ok
        and exponent_below_threshold
        and not all_points_below_rho
    ):
        status = "finite_scale_cost_gate_failed"
    elif not meets_gate and fit is not None:
        status = "fit_gate_failed"

    return {
        "eligible_cell_count": len(comparable_cells),
        "distinct_scale_count": len(scale_counts),
        "required_scale_count": SHOUP_MIN_SCALE_COUNT,
        "scale_counts": dict(sorted(scale_counts.items())),
        "fit": fit,
        "residual_tolerance": SHOUP_SCALING_RESIDUAL_TOLERANCE,
        "exponent_threshold": 0.5,
        "exponent_below_threshold": exponent_below_threshold,
        "residual_within_tolerance": residual_ok,
        "all_points_below_rho": all_points_below_rho,
        "meets_shoup_pressure_gate": meets_gate,
        "status": "pass" if meets_gate else status,
        "comparable_cells": comparable_cells,
        "evidence": (
            f"selected {len(comparable_cells)} exact+verified cells across "
            f"{len(scale_counts)} scales"
        ),
    }


def major_result_replication(route: dict[str, Any]) -> list[dict[str, Any]]:
    checks = (
        (
            "replay_and_exact_residual",
            "Residual replay and exactness gates pass before any later stage.",
            lambda row: bool(row.get("exact")),
        ),
        (
            "collision_supply",
            "At least one collision edge must be present for rank generation.",
            lambda row: int(row.get("collision_edges") or 0) > 0,
        ),
        (
            "cross_shift_routing",
            "Cross-shift collisions must exist before routing effects are evaluable.",
            lambda row: int(row.get("cross_shift_collision_edges") or 0) > 0,
        ),
        (
            "source_row_elimination",
            "Usable relation rows are required before rank checks.",
            lambda row: int(row.get("usable_relation_rows") or 0) > 0,
        ),
        (
            "relation_rank",
            "Full relation rank is needed before logarithm recovery.",
            lambda row: bool(row.get("full_rank")),
        ),
        (
            "rhs_compatibility",
            "RHS compatibility is required before trusted factor-log interpretation.",
            lambda row: bool(row.get("rhs_compatible")),
        ),
        (
            "factor_log_verification",
            "Factor logarithm verification must be complete for trusted logs.",
            lambda row: bool(row.get("verified_factor_logs")),
        ),
        (
            "target_descent",
            "Prospective target descent must be complete before cost gates.",
            lambda row: int(row.get("target_descent_count") or 0) > 0
            and float(row.get("target_descent_fraction") or 0.0) == 1.0,
        ),
        (
            "total_cost",
            "All recorded work must be below the P1436 rho proxy.",
            lambda row: bool(row.get("below_rho_proxy")),
        ),
    )
    matrix: list[dict[str, Any]] = []
    blocked_state: str | None = None
    for stage, rationale, condition in checks:
        if blocked_state == "blocked":
            status = "blocked"
            blocks = True
            interrupt = True
        elif blocked_state == "untested":
            status = "untested"
            blocks = True
            interrupt = True
        elif condition(route):
            status = "passed"
            blocks = False
            interrupt = False
        elif stage == "target_descent" and int(route.get("target_descent_count") or 0) == 0:
            status = "untested"
            blocks = True
            interrupt = True
            blocked_state = "untested"
        else:
            status = "blocked"
            blocks = True
            interrupt = True
            blocked_state = "blocked"

        matrix.append(
            {
                "stage": stage,
                "status": status,
                "blocks_promotion": blocks,
                "operator_interrupt_required": interrupt,
                "evidence": rationale,
            }
        )
    return matrix


def major_result_replication_summary(full_cells: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = {stage: Counter() for stage in MAJOR_RESULT_STAGES}
    fully_replicated = 0
    for cell in full_cells:
        route = cell.get("natural_route", {})
        rows = route.get("major_result_replication") if isinstance(route, dict) else None
        if not rows:
            continue

        all_passed = True
        for row in rows:
            stage_counts[row["stage"]][row["status"]] += 1
            if row["status"] != "passed":
                all_passed = False
        fully_replicated += int(all_passed)
    return {
        "cell_count": len(full_cells),
        "fully_replicated_cells": fully_replicated,
        "stage_status_counts": {
            stage: dict(sorted(counts.items())) for stage, counts in stage_counts.items()
        },
    }


def routing_delta(
    natural: dict[str, Any], intervention: dict[str, Any]
) -> dict[str, Any]:
    count_fields = (
        "collision_edges",
        "cross_shift_collision_edges",
        "usable_relation_rows",
        "relation_rank",
    )
    deltas = {
        f"{name}_delta": int(intervention[name]) - int(natural[name])
        for name in count_fields
    }
    deltas.update(
        {
            "rank_fraction_delta": round(
                float(intervention.get("rank_fraction") or 0.0)
                - float(natural.get("rank_fraction") or 0.0),
                10,
            ),
            "verified_factor_logs_changed": (
                bool(intervention["verified_factor_logs"])
                != bool(natural["verified_factor_logs"])
            ),
            "total_field_ratio_delta": (
                round(
                    float(intervention["total_field_ratio_vs_11x_rho"])
                    - float(natural["total_field_ratio_vs_11x_rho"]),
                    10,
                )
                if intervention.get("total_field_ratio_vs_11x_rho") is not None
                and natural.get("total_field_ratio_vs_11x_rho") is not None
                else None
            ),
        }
    )
    positive_stages = [
        stage
        for stage, value in (
            ("collision_supply", deltas["collision_edges_delta"]),
            ("cross_shift_routing", deltas["cross_shift_collision_edges_delta"]),
            ("source_row_elimination", deltas["usable_relation_rows_delta"]),
            ("relation_rank", deltas["rank_fraction_delta"]),
            (
                "factor_log_verification",
                int(intervention["verified_factor_logs"])
                - int(natural["verified_factor_logs"]),
            ),
        )
        if value > 0
    ]
    deltas["first_positive_stage"] = positive_stages[0] if positive_stages else None
    return deltas


def iter_cells(payload: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str, str, dict[str, Any]]]:
    for curve in payload.get("curve_records") or []:
        for policy, prefixes in (curve.get("policies") or {}).items():
            for prefix, cell in prefixes.items():
                yield curve, policy, prefix, cell


def iter_full_cells(payload: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str, str, dict[str, Any]]]:
    for curve, policy, prefix, cell in iter_cells(payload):
        if prefix != "full":
            continue
        yield curve, policy, prefix, cell


def _has_payload_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _payload_field_names(config: dict[str, Any], aliases: Iterable[str]) -> list[str]:
    return [
        name
        for name in aliases
        if name in config and _has_payload_value(config.get(name))
    ]


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _payload_sha_fields(term: str, field: str) -> list[str]:
    candidates = [f"{field}_sha256"]
    if field.endswith("_path"):
        candidates.append(f"{field[:-5]}_sha256")
    if term == "summation":
        candidates.extend(
            [
                "summation_polynomial_payload_sha256",
                "summation_poly_payload_sha256",
                "summation_polynomial_artifact_sha256",
                "summation_poly_source_sha256",
            ]
        )
    else:
        candidates.extend(
            [
                "ffe_payload_sha256",
                "ffe_profile_payload_sha256",
                "ffe_system_payload_sha256",
                "ffe_system_profile_payload_sha256",
            ]
        )
    return list(dict.fromkeys(candidates))


def _validate_payload_fields(
    term: str,
    config: dict[str, Any],
    aliases: Iterable[str],
) -> dict[str, Any]:
    observed = sorted(_payload_field_names(config, aliases))
    valid: list[str] = []
    invalid: list[dict[str, str]] = []
    for field in observed:
        value = config[field]
        if isinstance(value, dict):
            schema = value.get("schema")
            payload_sha = (
                value.get("source_sha256")
                or value.get("artifact_sha256")
                or value.get("payload_sha256")
            )
            if isinstance(schema, str) and schema.strip() and _is_sha256(payload_sha):
                valid.append(field)
            else:
                invalid.append(
                    {
                        "field": field,
                        "reason": "embedded_payload_requires_schema_and_sha256",
                    }
                )
            continue

        sha_field = next(
            (
                name
                for name in _payload_sha_fields(term, field)
                if _is_sha256(config.get(name))
            ),
            "",
        )
        if isinstance(value, str) and value.strip() and sha_field:
            valid.append(field)
        else:
            invalid.append(
                {
                    "field": field,
                    "reason": "external_or_opaque_payload_requires_companion_sha256",
                }
            )
    return {
        "observed_fields": observed,
        "valid_fields": sorted(valid),
        "invalid_fields": invalid,
    }


def _infer_replay_readiness(markers: list[str], config: dict[str, Any]) -> dict[str, Any]:
    marker_terms = {"summation": False, "ffe": False}
    for marker in markers:
        marker_lower = marker.lower()
        if "summation" in marker_lower:
            marker_terms["summation"] = True
        if "ffe" in marker_lower:
            marker_terms["ffe"] = True

    present_inputs: list[str] = []
    missing_inputs: list[str] = []
    present_input_fields: dict[str, list[str]] = {}
    payload_validation: dict[str, dict[str, Any]] = {}
    for term, prefix in marker_terms.items():
        if not prefix:
            continue
        aliases = SUMMATION_PAYLOAD_FIELDS if term == "summation" else FFE_PAYLOAD_FIELDS
        validation = _validate_payload_fields(term, config, aliases)
        payload_validation[term] = validation
        present = validation["valid_fields"]
        present_inputs.extend(present)
        present_input_fields[term] = present

        if not present:
            missing_inputs.append(f"exact_{term}_source_payload")

    required_count = len([name for name in marker_terms.values() if name])
    present_count = sum(1 for term in marker_terms if marker_terms[term] and present_input_fields.get(term))

    if required_count == 0:
        return {
            "has_markers": False,
            "needs_inputs": False,
            "needs": [],
            "required_inputs": [],
            "present_inputs": [],
            "present_input_fields": {},
            "payload_validation": {},
            "replay_readiness": "not_applicable",
            "replay_readiness_class": "not_applicable",
        }

    readiness = "missing_exact_inputs"
    readiness_class = "marker_only"
    if required_count and present_count == required_count:
        readiness = "exact_replay_inputs_ready"
        readiness_class = "ready"
    elif any(
        validation["observed_fields"]
        for validation in payload_validation.values()
    ):
        readiness_class = "partial_payload"

    return {
        "has_markers": True,
        "needs_inputs": bool(missing_inputs),
        "needs": [
            name
            for name, needed in marker_terms.items()
            if needed and not present_input_fields.get(name)
        ],
        "required_inputs": sorted(set(missing_inputs)),
        "present_inputs": sorted(set(present_inputs)),
        "present_input_fields": present_input_fields,
        "payload_validation": payload_validation,
        "replay_readiness": readiness,
        "replay_readiness_class": readiness_class,
    }


def _infer_new_factor_row_discovery_gate(
    markers: list[str],
    config: dict[str, Any],
    replay_readiness: dict[str, Any],
) -> dict[str, Any]:
    if not markers:
        return {
            "applicable": False,
            "lane_admitted": False,
            "status": "not_applicable",
            "contract_fields": [],
            "selected_contract_field": "",
            "missing_fields": [],
            "failures": [],
            "cost_ratio_vs_direct_pair_complement": None,
        }

    contract_fields = [
        field
        for field in SUMMATION_FFE_DISCOVERY_CONTRACT_FIELDS
        if _has_payload_value(config.get(field))
    ]
    selected_field = contract_fields[0] if contract_fields else ""
    contract = config.get(selected_field) if selected_field else None
    failures: list[str] = []
    missing_fields: list[str] = []
    measured_cost: float | None = None
    direct_cost: float | None = None

    if len(contract_fields) > 1:
        failures.append("multiple_discovery_contracts")
    if not selected_field:
        missing_fields = list(SUMMATION_FFE_DISCOVERY_REQUIRED_FIELDS)
        status = "missing_discovery_contract"
    elif not isinstance(contract, dict):
        failures.append("discovery_contract_must_be_object")
        status = "invalid_discovery_contract"
    else:
        missing_fields = [
            field
            for field in SUMMATION_FFE_DISCOVERY_REQUIRED_FIELDS
            if field not in contract
        ]
        if missing_fields:
            failures.append("missing_required_contract_fields")
        if not isinstance(contract.get("source_enumerator_id"), str) or not contract[
            "source_enumerator_id"
        ].strip():
            failures.append("source_enumerator_id_missing")
        if contract.get("scalar_blind") is not True:
            failures.append("source_enumerator_not_scalar_blind")

        new_rows = contract.get("new_factor_row_count")
        independent_rows = contract.get("independent_new_factor_row_count")
        if not isinstance(new_rows, int) or isinstance(new_rows, bool) or new_rows <= 0:
            failures.append("no_new_factor_rows")
        if (
            not isinstance(independent_rows, int)
            or isinstance(independent_rows, bool)
            or independent_rows <= 0
        ):
            failures.append("no_independent_new_factor_rows")
        elif isinstance(new_rows, int) and independent_rows > new_rows:
            failures.append("independent_row_count_exceeds_new_row_count")

        measured_raw = contract.get("measured_source_operations")
        direct_raw = contract.get("direct_pair_complement_operations")
        if isinstance(measured_raw, (int, float)) and not isinstance(measured_raw, bool):
            measured_cost = float(measured_raw)
        if isinstance(direct_raw, (int, float)) and not isinstance(direct_raw, bool):
            direct_cost = float(direct_raw)
        if measured_cost is None or measured_cost < 0:
            failures.append("measured_source_operations_invalid")
        if direct_cost is None or direct_cost <= 0:
            failures.append("direct_pair_complement_operations_invalid")
        if not _is_sha256(contract.get("replay_artifact_sha256")):
            failures.append("replay_artifact_sha256_invalid")

        structural_failures = bool(failures or missing_fields)
        if structural_failures:
            status = "invalid_discovery_contract"
        elif replay_readiness["replay_readiness"] != "exact_replay_inputs_ready":
            status = "blocked_missing_exact_replay"
        elif measured_cost is not None and direct_cost is not None and measured_cost >= direct_cost:
            status = "fails_direct_pair_complement_cost_gate"
            failures.append("source_cost_not_below_direct_pair_complement")
        else:
            status = "admitted_for_exact_replay"

    cost_ratio = (
        ratio(measured_cost, direct_cost)
        if measured_cost is not None and direct_cost is not None
        else None
    )
    return {
        "applicable": True,
        "gate_basis": "p1553_ffe_fixed_sum_information_conservation_r68",
        "product_quotient_information_credit": 0,
        "target_uniform_evidence_required_for_promotion": True,
        "lane_admitted": status == "admitted_for_exact_replay",
        "status": status,
        "contract_fields": contract_fields,
        "selected_contract_field": selected_field,
        "required_fields": list(SUMMATION_FFE_DISCOVERY_REQUIRED_FIELDS),
        "missing_fields": sorted(set(missing_fields)),
        "failures": sorted(set(failures)),
        "measured_source_operations": measured_cost,
        "direct_pair_complement_operations": direct_cost,
        "cost_ratio_vs_direct_pair_complement": cost_ratio,
    }


def _build_summation_ffe_inventory_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    full_config_rows: list[dict[str, Any]] = []
    checked = 0
    marker_total = 0
    for curve, policy, prefix, cell in iter_full_cells(payload):
        configurations = cell.get("configurations") or {}
        factor_base_size = int(cell.get("factor_base_size_B") or 0)
        for config_name in sorted(configurations):
            config = configurations[config_name]
            checked += 1
            markers = summation_ffe_markers(config)
            if not markers:
                continue
            marker_count = len(markers)
            marker_total += marker_count
            replay_readiness = _infer_replay_readiness(markers, config)
            discovery_gate = _infer_new_factor_row_discovery_gate(
                markers,
                config,
                replay_readiness,
            )
            config_snapshot = {
                key: config.get(key)
                for key in (
                    "collision_edge_count",
                    "cross_shift_collision_count",
                    "within_shift_collision_count",
                    "relation_row_count",
                    "relation_rank",
                    "unknown_factor_count",
                    "total_field_ratio_vs_11x_rho",
                    "attempts",
                )
                if key in config
            }
            full_mark = [
                marker
                for marker in markers
                if marker and marker.startswith("pipeline_note::") is False
            ]
            full_config_rows.append(
                {
                    "route_id": f"{curve.get('seed')}|{policy}|{prefix}|{config_name}",
                    "curve": {
                        "split": curve.get("split"),
                        "bits": curve.get("bits"),
                        "seed": curve.get("seed"),
                    },
                    "policy": policy,
                    "prefix": prefix,
                    "factor_base_size_B": factor_base_size,
                    "configuration": config_name,
                    "summation_ffe_marker_count": marker_count,
                    "summation_ffe_markers": markers,
                    "summation_ffe_key_markers": sorted(full_mark),
                    "summation_ffe_readiness": replay_readiness,
                    "summation_ffe_new_factor_row_discovery_gate": discovery_gate,
                    "summation_ffe_replay_inputs": {
                        "required_inputs": replay_readiness["required_inputs"],
                        "present_inputs": replay_readiness["present_inputs"],
                        "present_input_fields": replay_readiness["present_input_fields"],
                        "payload_validation": replay_readiness["payload_validation"],
                        "replay_readiness": replay_readiness["replay_readiness"],
                        "replay_readiness_class": replay_readiness["replay_readiness_class"],
                    },
                    "config_observed_fields": sorted(config.keys()),
                    "config_summary": config_snapshot,
                }
            )
    inventory_summary = {
        "checked_configs": checked,
        "summation_ffe_marked_configs": len(full_config_rows),
        "summation_ffe_markers": marker_total,
    }
    return full_config_rows, inventory_summary, checked, marker_total


def build_summation_ffe_artifacts(
    payload: dict[str, Any], note_url: str = DEFAULT_NOTE_URL
) -> tuple[dict[str, Any], dict[str, Any]]:
    methodology = build_methodology(note_url)
    inventory_rows, inventory_summary, checked, marker_total = _build_summation_ffe_inventory_rows(
        payload,
    )
    replay_rows = []
    partial_payload_count = 0
    payload_ready_count = 0
    missing_count = 0
    marker_only_count = 0
    lane_admitted_count = 0
    discovery_contract_missing_count = 0
    discovery_contract_invalid_count = 0
    discovery_cost_gate_failed_count = 0

    for record in inventory_rows:
        markers = list(record["summation_ffe_markers"])
        config_name = record["configuration"]
        replay_inputs = record.get("summation_ffe_replay_inputs") or record["summation_ffe_readiness"]
        discovery_gate = record["summation_ffe_new_factor_row_discovery_gate"]
        readiness = replay_inputs.get("replay_readiness", "missing_exact_inputs")
        readiness_class = replay_inputs.get(
            "replay_readiness_class",
            "marker_only" if readiness == "missing_exact_inputs" else readiness,
        )
        required_inputs = list(replay_inputs.get("required_inputs", []))
        payload_ready = readiness == "exact_replay_inputs_ready"
        lane_admitted = bool(discovery_gate["lane_admitted"])
        if payload_ready:
            payload_ready_count += 1
        else:
            missing_count += 1
            if readiness_class == "marker_only":
                marker_only_count += 1
            elif readiness_class == "partial_payload":
                partial_payload_count += 1
            else:
                # Keep defensive fallback for future status values.
                marker_only_count += 1
        if lane_admitted:
            lane_admitted_count += 1
        if discovery_gate["status"] == "missing_discovery_contract":
            discovery_contract_missing_count += 1
        elif discovery_gate["status"] == "invalid_discovery_contract":
            discovery_contract_invalid_count += 1
        elif discovery_gate["status"] == "fails_direct_pair_complement_cost_gate":
            discovery_cost_gate_failed_count += 1

        replay_rows.append(
            {
                "route_id": record["route_id"],
                "curve": record["curve"],
                "policy": record["policy"],
                "prefix": record["prefix"],
                "configuration": config_name,
                "summation_ffe_markers": markers,
                "replay_readiness": readiness,
                "summation_ffe_replay_readiness_class": readiness_class,
                "required_inputs": required_inputs,
                "present_inputs": replay_inputs.get("present_inputs", []),
                "summation_ffe_present_input_fields": replay_inputs.get(
                    "present_input_fields",
                    {},
                ),
                "summation_ffe_payload_validation": replay_inputs.get(
                    "payload_validation",
                    {},
                ),
                "new_factor_row_discovery_gate": discovery_gate,
                "lane_admitted_for_exact_replay": lane_admitted,
                "observed_inputs": sorted(set(record["config_observed_fields"])),
                "next_step": (
                    "Collect exact summation/FFE route payloads and replay each route exact"
                    if not payload_ready
                    else (
                        "Supply or repair the scalar-blind new-factor-row discovery contract."
                        if not lane_admitted
                        else "Replay the admitted source enumerator and charge it against direct pair-complement enumeration."
                    )
                ),
            }
        )

    if not replay_rows:
        execution_status = "not_applicable"
    elif missing_count:
        execution_status = "missing_exact_inputs"
    elif lane_admitted_count != len(replay_rows):
        execution_status = "discovery_contract_blocked"
    else:
        execution_status = "ready"

    inventory = {
        "schema": SUMMATION_FFE_EVIDENCE_INVENTORY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note_url": methodology["source_post_url_with_query"],
        "source": build_tweet_source_payload(methodology),
        "records": inventory_rows,
        "execution_status": {
            **inventory_summary,
            "replay_rows": len(replay_rows),
            "records_ready": lane_admitted_count,
            "records_payload_ready": payload_ready_count,
            "records_missing_inputs": missing_count,
            "records_partial_payload_count": partial_payload_count,
            "records_marker_only_count": marker_only_count,
            "records_discovery_contract_missing": discovery_contract_missing_count,
            "records_discovery_contract_invalid": discovery_contract_invalid_count,
            "records_discovery_cost_gate_failed": discovery_cost_gate_failed_count,
            "records_lane_admitted": lane_admitted_count,
            "status": execution_status,
        },
    }

    replay_plan = {
        "schema": SUMMATION_FFE_REPLAY_PLAN_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note_url": methodology["source_post_url_with_query"],
        "source": build_tweet_source_payload(methodology),
        "replay_rows": replay_rows,
        "execution_status": {
            "replay_record_count": len(replay_rows),
            "replay_ready_count": lane_admitted_count,
            "replay_payload_ready_count": payload_ready_count,
            "replay_missing_count": missing_count,
            "replay_partial_payload_count": partial_payload_count,
            "replay_marker_only_count": marker_only_count,
            "discovery_contract_missing_count": discovery_contract_missing_count,
            "discovery_contract_invalid_count": discovery_contract_invalid_count,
            "discovery_cost_gate_failed_count": discovery_cost_gate_failed_count,
            "lane_admitted_count": lane_admitted_count,
            "status": execution_status,
            "next_step": (
                "No summation/FFE-labeled route is present."
                if not replay_rows
                else (
                    "Collect missing summation and/or FFE payloads before replaying."
                    if missing_count
                    else (
                        "Supply a valid scalar-blind new-factor-row discovery contract below direct pair-complement cost."
                        if lane_admitted_count != len(replay_rows)
                        else "Replay all admitted new-factor-row source enumerators."
                    )
                )
            ),
        },
    }

    return inventory, replay_plan


def cell_report(
    curve: dict[str, Any],
    policy: str,
    prefix: str,
    cell: dict[str, Any],
    fixed_config_name: str,
) -> dict[str, Any]:
    configurations = cell.get("configurations") or {}
    if "random_hash_mask1" not in configurations:
        raise ValueError(f"missing random_hash_mask1 for {curve.get('seed')}:{policy}:{prefix}")

    natural = stage_record(
        configurations["random_hash_mask1"],
        cell.get("target_descents") or [],
        require_descent=True,
    )
    oracle_name, oracle_config = best_intervention(configurations)
    oracle = stage_record(oracle_config)
    fixed_name = fixed_config_name if fixed_config_name in configurations else "random_hash_mask1"
    fixed = stage_record(configurations[fixed_name])
    natural_rank = float(natural.get("rank_fraction") or 0.0)
    fixed_rank = float(fixed.get("rank_fraction") or 0.0)
    oracle_rank = float(oracle.get("rank_fraction") or 0.0)

    factor_base_size = int(cell.get("factor_base_size_B") or 0)
    natural_uniform = synthetic_uniform_occupancy(
        configurations["random_hash_mask1"],
        order=int(curve.get("order") or 0),
        shift_count=4 * factor_base_size,
        label=f"{curve.get('seed')}:{policy}:{prefix}:random_hash_mask1",
    )

    return {
        "curve": {
            "split": curve.get("split"),
            "bits": curve.get("bits"),
            "seed": curve.get("seed"),
            "order": curve.get("order"),
        },
        "policy": policy,
        "prefix": prefix,
        "factor_base_size_B": factor_base_size,
        "natural_route": {
            "configuration": "random_hash_mask1",
            **natural,
            "major_result_replication": major_result_replication(natural),
        },
        "fixed_routing_intervention": {
            "configuration": fixed_name,
            "available": fixed_name == fixed_config_name,
            **fixed,
        },
        "oracle_routing_intervention": {
            "configuration": oracle_name,
            "posthoc_diagnostic_only": True,
            **oracle,
        },
        "routing_gap": {
            "oracle_rank_headroom": round(oracle_rank - natural_rank, 10),
            "fixed_rank_headroom": round(fixed_rank - natural_rank, 10),
            "fixed_fraction_of_oracle_rank_headroom": recovered_headroom(
                natural_rank, fixed_rank, oracle_rank
            ),
            "natural_collisions_reach_verified_logs": bool(
                natural["collision_edges"] and natural["verified_factor_logs"]
            ),
            "fixed_delta": routing_delta(natural, fixed),
            "oracle_delta": routing_delta(natural, oracle),
            "intervention_design": "matched_configuration_ablation",
            "causal_self_patch_equivalent": False,
        },
        "synthetic_uniform_occupancy_control": natural_uniform,
    }


def headroom_summary(cells: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(cells)
    comparable = [
        row
        for row in rows
        if row["fixed_routing_intervention"]["available"]
        and row["natural_route"]["exact"]
        and row["fixed_routing_intervention"]["exact"]
        and row["oracle_routing_intervention"]["exact"]
    ]
    natural_scores = [float(row["natural_route"].get("rank_fraction") or 0.0) for row in comparable]
    fixed_scores = [
        float(row["fixed_routing_intervention"].get("rank_fraction") or 0.0)
        for row in comparable
    ]
    oracle_scores = [
        float(row["oracle_routing_intervention"].get("rank_fraction") or 0.0)
        for row in comparable
    ]
    natural_mean = average(natural_scores)
    fixed_mean = average(fixed_scores)
    oracle_mean = average(oracle_scores)
    fixed_gains = [fixed - natural for natural, fixed in zip(natural_scores, fixed_scores)]
    oracle_gains = [oracle - natural for natural, oracle in zip(natural_scores, oracle_scores)]
    return {
        "cell_count": len(rows),
        "fixed_available_cell_count": sum(
            row["fixed_routing_intervention"]["available"] for row in rows
        ),
        "exact_comparable_cell_count": len(comparable),
        "mean_natural_rank_fraction": natural_mean,
        "mean_fixed_rank_fraction": fixed_mean,
        "mean_oracle_rank_fraction": oracle_mean,
        "fixed_fraction_of_oracle_rank_headroom": (
            recovered_headroom(natural_mean, fixed_mean, oracle_mean)
            if natural_mean is not None and fixed_mean is not None and oracle_mean is not None
            else None
        ),
        "fixed_improvement_cell_count": sum(gain > 0 for gain in fixed_gains),
        "fixed_regression_cell_count": sum(gain < 0 for gain in fixed_gains),
        "oracle_improvement_cell_count": sum(gain > 0 for gain in oracle_gains),
        "scope": "rank_only_matched_configuration_ablation",
    }


def routing_generalization_summary(
    full_cells: list[dict[str, Any]], fixed_config_name: str
) -> dict[str, Any]:
    target_cells = [cell for cell in full_cells if cell["policy"] == "two_map_union"]
    split_names = sorted({str(cell["curve"].get("split") or "unspecified") for cell in target_cells})
    by_split = {
        split: headroom_summary(
            cell
            for cell in target_cells
            if str(cell["curve"].get("split") or "unspecified") == split
        )
        for split in split_names
    }
    prospective_cells = [
        cell for cell in target_cells if cell["curve"].get("split") == "prospective"
    ]
    prospective = headroom_summary(prospective_cells)
    if not prospective_cells:
        transfer_status = "not_available"
    elif prospective["exact_comparable_cell_count"] != len(prospective_cells):
        transfer_status = "invalid_exactness_or_missing_fixed_route"
    elif prospective["fixed_regression_cell_count"]:
        transfer_status = "rank_regression"
    elif prospective["fixed_improvement_cell_count"]:
        transfer_status = "positive_rank_transfer"
    else:
        transfer_status = "no_rank_headroom_or_no_effect"
    return {
        "fixed_rule": fixed_config_name,
        "target_policy": "two_map_union",
        "all_target_cells": headroom_summary(target_cells),
        "target_cells_by_split": by_split,
        "prospective_transfer": {
            **prospective,
            "status": transfer_status,
            "promotion_eligible": False,
        },
        "interpretation": (
            "A positive fixed-route transfer is a non-oracle diagnostic. It does not replace the "
            "natural-route correctness, descent, cost, or independent-audit gates."
        ),
    }


def routing_specificity_summary(full_cells: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for cell in full_cells:
        key = (
            cell["curve"].get("split"),
            cell["curve"].get("bits"),
            cell["curve"].get("seed"),
            cell["prefix"],
        )
        grouped.setdefault(key, {})[cell["policy"]] = cell

    comparisons = []
    for key, policies in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        target = policies.get("two_map_union")
        controls = [
            policies[name]
            for name in HASH_POLICIES
            if name in policies and policies[name]["fixed_routing_intervention"]["available"]
        ]
        if target is None or not controls:
            continue
        candidates = [target, *controls]
        if any(
            not row["fixed_routing_intervention"]["available"]
            or not row["natural_route"]["exact"]
            or not row["fixed_routing_intervention"]["exact"]
            for row in candidates
        ):
            continue
        target_gain = float(target["routing_gap"]["fixed_rank_headroom"])
        control_gains = [float(row["routing_gap"]["fixed_rank_headroom"]) for row in controls]
        control_mean = average(control_gains)
        assert control_mean is not None
        comparisons.append(
            {
                "curve": {
                    "split": key[0],
                    "bits": key[1],
                    "seed": key[2],
                },
                "target_policy": "two_map_union",
                "hash_control_count": len(controls),
                "target_fixed_rank_headroom": target_gain,
                "mean_hash_fixed_rank_headroom": control_mean,
                "coordinate_specific_excess": round(target_gain - control_mean, 10),
            }
        )
    positive = sum(row["coordinate_specific_excess"] > 0 for row in comparisons)
    all_positive = bool(comparisons) and positive == len(comparisons)
    return {
        "matched_curve_count": len(comparisons),
        "positive_coordinate_specific_excess_count": positive,
        "nonpositive_coordinate_specific_excess_count": len(comparisons) - positive,
        "target_gain_exceeds_hash_controls_on_every_matched_curve": all_positive,
        "comparisons": comparisons,
        "conclusion": (
            "coordinate_specific_routing_signal_diagnostic_only"
            if all_positive
            else (
                "generic_routing_effect_not_ruled_out"
                if comparisons
                else "matched_hash_controls_unavailable"
            )
        ),
        "control_role": (
            "Matched hash policies play the role of an irrelevant-route perturbation control; "
            "they do not create promotion evidence."
        ),
    }


def audit_binding(audit: dict[str, Any] | None, source_sha256: str) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {
            "provided": False,
            "audit_passed": False,
            "source_hash_matches": False,
            "promotion_binding_valid": False,
        }
    hashes = {
        str(audit.get(name) or "")
        for name in ("source_sha256", "probe_sha256", "p1436_probe_sha256")
    }
    source_matches = source_sha256 in hashes
    passed = audit.get("audit_passed") is True
    return {
        "provided": True,
        "audit_passed": passed,
        "source_hash_matches": source_matches,
        "promotion_binding_valid": passed and source_matches,
    }


def frontier_lane_bindings(
    preflights: dict[str, tuple[dict[str, Any], Path]] | None,
) -> dict[str, Any]:
    rows = {}
    for name, expected_schema in FRONTIER_PREFLIGHT_SCHEMAS.items():
        entry = (preflights or {}).get(name)
        if entry is None:
            rows[name] = {
                "provided": False,
                "valid_schema": False,
                "hash_bound": False,
                "lane_admitted": False,
                "closed_by_current_evidence": False,
            }
            continue
        payload, path = entry
        bindings = payload.get("source_bindings") or {}
        hash_bound = bool(bindings) and all(
            isinstance(binding, dict) and _is_sha256(binding.get("sha256"))
            for binding in bindings.values()
        )
        if name == "slice_quadratic_public_source":
            lane_admitted = bool(
                (payload.get("admission") or {}).get("source_lane_admitted")
            )
        else:
            lane_admitted = bool(
                (payload.get("admission") or {}).get("lane_admitted")
            )
        valid_schema = payload.get("schema") == expected_schema
        rows[name] = {
            "provided": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "expected_schema": expected_schema,
            "valid_schema": valid_schema,
            "hash_bound": hash_bound,
            "classification": payload.get("classification"),
            "lane_admitted": lane_admitted,
            "closed_by_current_evidence": (
                valid_schema and hash_bound and not lane_admitted
            ),
            "admission": payload.get("admission"),
            "next_action": payload.get("next_action"),
        }
    closed = sorted(
        name for name, row in rows.items() if row["closed_by_current_evidence"]
    )
    return {
        "lanes": rows,
        "provided_count": sum(row["provided"] for row in rows.values()),
        "closed_lane_count": len(closed),
        "closed_lanes": closed,
    }


def add_focus_candidate(
    candidates: dict[str, dict[str, Any]],
    identifier: str,
    score: int,
    action: str,
    evidence: str,
) -> None:
    row = candidates.setdefault(
        identifier,
        {"id": identifier, "priority_score": score, "action": action, "evidence": []},
    )
    row["priority_score"] = max(int(row["priority_score"]), score)
    if evidence not in row["evidence"]:
        row["evidence"].append(evidence)


def ranked_focus_candidates(
    full_cells: list[dict[str, Any]],
    frontier_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    natural = [cell["natural_route"] for cell in full_cells]
    bottlenecks = Counter(row["bottleneck"] for row in natural)
    if bottlenecks.get("exactness"):
        add_focus_candidate(
            candidates,
            "repair_exactness_before_search",
            100,
            "Repair source replay, residual equality, or validation failures before interpreting performance.",
            f"{bottlenecks['exactness']} full cells fail exactness.",
        )
    if bottlenecks.get("residual_collision_supply") or bottlenecks.get("cross_shift_routing"):
        count = bottlenecks.get("residual_collision_supply", 0) + bottlenecks.get("cross_shift_routing", 0)
        add_focus_candidate(
            candidates,
            "residual_supply_and_cross_shift_probe",
            90,
            "Compare exact occupancy and cross-shift collision arrival against the synthetic uniform stream.",
            f"{count} full cells fail before usable cross-shift routing.",
        )
    stored_unusable = sum(row["stored_but_unusable"] for row in natural)
    if stored_unusable:
        add_focus_candidate(
            candidates,
            "collision_to_rank_routing_ablation",
            85,
            "Replay all-edge, cross-shift-only, and within-shift-only matrices from raw "
            "collision records; synthetic previews remain diagnostic-only.",
            f"{stored_unusable} full cells contain collisions without full usable rank.",
        )
    slice_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "slice_quadratic_public_source"
        )
        or {}
    )
    presurface_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "presurface_full_charge"
        )
        or {}
    )
    factor_line_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "factor_line_direct_root_equivalence"
        )
        or {}
    )
    closure_collision_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "constructive_closure_collision"
        )
        or {}
    )
    multiplicative_x_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "multiplicative_x_s3_closure"
        )
        or {}
    )
    s4_carry_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "s4_centered_carry_rank"
        )
        or {}
    )
    s6_carry_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "s6_centered_carry_rank_minor"
        )
        or {}
    )
    resultant_valuation_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "resultant_valuation_trace_grammar"
        )
        or {}
    )
    residual_decision_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "s6_residual_decision_diagram"
        )
        or {}
    )
    iterated_norm_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "s6_iterated_norm_support"
        )
        or {}
    )
    subset_incidence_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "s6_subset_incidence_mobius"
        )
        or {}
    )
    frequency_orbit_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "target_translated_frequency_orbit"
        )
        or {}
    )
    fermat_tt_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "actual_s6_fermat_tensor_train"
        )
        or {}
    )
    scalar_norm_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "scalar_only_nested_norm_slp"
        )
        or {}
    )
    batched_norm_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "batched_nested_norm_node_compiler"
        )
        or {}
    )
    full_coset_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "full_multiplicative_x_coset_endpoint"
        )
        or {}
    )
    cartesian_sum_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "cartesian_sum_compact_divisor"
        )
        or {}
    )
    coordinate_filtration_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_coordinate_filtration"
        )
        or {}
    )
    marked_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_marked_resultant_source_section"
        )
        or {}
    )
    precoefficient_circuit_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_target_uniform_precoefficient_circuit"
        )
        or {}
    )
    sparse_moment_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_sparse_multihomogeneous_moment_recurrence"
        )
        or {}
    )
    jet_pushforward_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_jet_preserving_addition_pushforward"
        )
        or {}
    )
    black_box_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_black_box_resultant_localizer"
        )
        or {}
    )
    fixed_marker_recurrence_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_fixed_marker_scalar_recurrence"
        )
        or {}
    )
    nonlocal_moment_hankel_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_nonlocal_moment_hankel_translation"
        )
        or {}
    )
    unequal_list_subfunction_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_unequal_list_subfunction_inversion"
        )
        or {}
    )
    compact_elliptic_subfunction_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_compact_elliptic_subfunction_map"
        )
        or {}
    )
    shared_semilinear_incidence_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_shared_semilinear_incidence_correspondence"
        )
        or {}
    )
    implicit_veronese_index_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_implicit_veronese_hyperplane_source_index"
        )
        or {}
    )
    aggregate_veronese_recurrence_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_aggregate_veronese_projector_recurrence"
        )
        or {}
    )
    modular_frobenius_trace_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_modular_frobenius_trace_recurrence"
        )
        or {}
    )
    factored_transposed_trace_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_factored_transposed_projector_trace"
        )
        or {}
    )
    nonlinear_tensor_tower_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_nonlinear_tensor_tower_trace"
        )
        or {}
    )
    multiedge_digitized_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_multiedge_digitized_equality_projector"
        )
        or {}
    )
    aggregate_digit_trie_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_succinct_aggregate_digit_trie"
        )
        or {}
    )
    actual_divisor_image_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_actual_divisor_image_entropy_merge"
        )
        or {}
    )
    two_sided_join_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_two_sided_implicit_join"
        )
        or {}
    )
    target_forced_filter_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_target_forced_algebraic_join_filter"
        )
        or {}
    )
    preendpoint_pushdown_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_compact_preendpoint_s3_ffe_pushdown"
        )
        or {}
    )
    actual_deck_pullback_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_actual_deck_nonmergeable_target_pullback"
        )
        or {}
    )
    scalar_target_norm_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_scalar_target_norm_count_circuit"
        )
        or {}
    )
    noncharacter_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_noncharacter_algebraic_target_norm_resultant"
        )
        or {}
    )
    lambda_ring_chow_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_factored_elliptic_lambda_ring_chow_norm"
        )
        or {}
    )
    poincare_section_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_poincare_theta_target_section_rank"
        )
        or {}
    )
    theta_addition_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_theta_addition_cancellation_network"
        )
        or {}
    )
    finite_deck_annihilator_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_finite_deck_alternant_annihilator"
        )
        or {}
    )
    endpoint_query2p1_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_gauge_normalized_endpoint_query2p1"
        )
        or {}
    )
    orbit_product_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_nonlinear_elliptic_orbit_product"
        )
        or {}
    )
    transposed_leaf_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "5a5c_transposed_nonuniform_c5_leaf_generator"
        )
        or {}
    )
    exponent_rebalance_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "relation_arity_factor_base_transposed_interface_rebalance"
        )
        or {}
    )
    m6_pair_sum_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_a6_batched_c3_pair_sum_source_locator"
        )
        or {}
    )
    m6_elliptic_transpose_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_target_batched_c3_elliptic_transpose"
        )
        or {}
    )
    m6_nonlinear_c6_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_nonlinear_value_sensitive_c6_source_locator"
        )
        or {}
    )
    m6_output_sensitive_c5_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_output_sensitive_nonlinear_c5_source_index"
        )
        or {}
    )
    m6_character_pairing_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_suboutput_implicit_c5_character_pairing"
        )
        or {}
    )
    m6_moment_torus_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_small_k_multiplicative_c5_moment_torus"
        )
        or {}
    )
    torus_split_rebalance_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_explicit_split_global_rebalance"
        )
        or {}
    )
    torus_fourier_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_fourier_product_resultant"
        )
        or {}
    )
    torus_linear_sketch_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_linear_sketch_circulant"
        )
        or {}
    )
    torus_homomorphic_fingerprint_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_prime_order_homomorphic_fingerprint"
        )
        or {}
    )
    torus_explicit_hash_correction_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_explicit_hash_correction_support"
        )
        or {}
    )
    torus_bucket_resultant_tradeoff_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_bucket_resultant_routing_tradeoff"
        )
        or {}
    )
    torus_rational_selector_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_rational_selector_degree"
        )
        or {}
    )
    torus_piecewise_selector_dag_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_piecewise_selector_decision_dag"
        )
        or {}
    )
    torus_sparse_fourier_transfer_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_sparse_fourier_predicate_transfer"
        )
        or {}
    )
    torus_consecutive_mode_predicate_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_consecutive_mode_predicate"
        )
        or {}
    )
    torus_base_field_frobenius_dag_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_base_field_frobenius_predicate_dag"
        )
        or {}
    )
    torus_sparse_monomial_root_bound_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_sparse_monomial_root_bound"
        )
        or {}
    )
    torus_two_atom_progression_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_two_atom_geometric_progression"
        )
        or {}
    )
    torus_khatri_rao_amplification_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_khatri_rao_kruskal_amplification"
        )
        or {}
    )
    torus_all_nonzero_path_product_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_all_nonzero_path_product"
        )
        or {}
    )
    torus_binomial_node_union_depth_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_binomial_node_union_depth"
        )
        or {}
    )
    torus_chebotarev_fiber_cover_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_chebotarev_fiber_cover"
        )
        or {}
    )
    torus_order_two_three_minor_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_order_two_three_minor_rigidity"
        )
        or {}
    )
    torus_order_two_four_minor_claw_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_order_two_four_minor_claw"
        )
        or {}
    )
    torus_sextic_mobius_character_router_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_sextic_mobius_character_router"
        )
        or {}
    )
    torus_adaptive_character_decision_router_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_adaptive_character_decision_router"
        )
        or {}
    )
    torus_label_congruence_correction_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "torus_c5_label_congruence_correction"
        )
        or {}
    )
    m6_weighted_fiber_marginal_log_operator_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_weighted_fiber_marginal_log_operator"
        )
        or {}
    )
    m6_weighted_c3_mobius_gcd_trace_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_weighted_c3_mobius_gcd_trace"
        )
        or {}
    )
    m6_aggregate_marginal_singleton_source_equivalence_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_aggregate_marginal_singleton_source_equivalence"
        )
        or {}
    )
    m6_occurrence_pair_resultant_local_valuation_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_occurrence_pair_resultant_local_valuation"
        )
        or {}
    )
    m6_static_3sum_indexing_tradeoff_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_static_3sum_indexing_tradeoff"
        )
        or {}
    )
    m6_actual_c6_shift_krylov_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_actual_c6_shift_krylov_rank"
        )
        or {}
    )
    m6_rational_convolution_subalgebra_rigidity_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_rational_convolution_subalgebra_rigidity"
        )
        or {}
    )
    m6_matrix_free_marginal_jacobian_krylov_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_matrix_free_marginal_jacobian_krylov"
        )
        or {}
    )
    m6_geometry_only_weight_interpolation_adjoint_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_geometry_only_weight_interpolation_adjoint"
        )
        or {}
    )
    m6_symmetric_shift_reverse_only_marginal_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_symmetric_shift_reverse_only_marginal"
        )
        or {}
    )
    m6_signed_quotient_multiscale_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_signed_quotient_multiscale_rank"
        )
        or {}
    )
    m6_singleton_relation_hypergraph_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_singleton_relation_hypergraph_rank"
        )
        or {}
    )
    m6_a_diversity_projective_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_a_diversity_projective_rank"
        )
        or {}
    )
    m6_hash_to_curve_projective_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_hash_to_curve_projective_rank"
        )
        or {}
    )
    m6_short_relation_near_injectivity_supply_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_short_relation_near_injectivity_supply"
        )
        or {}
    )
    m6_random_diagonal_known_target_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_random_diagonal_known_target_rank"
        )
        or {}
    )
    m6_positive_c6_generic_locator_reduction_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_positive_c6_generic_locator_reduction"
        )
        or {}
    )
    m6_signed_c3_divisor_translation_gcd_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_signed_c3_divisor_translation_gcd"
        )
        or {}
    )
    m6_batch_inverse_transpose_modcomp_fit_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_batch_inverse_transpose_modcomp_fit"
        )
        or {}
    )
    m6_aggregate_union_factor_label_recovery_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_aggregate_union_factor_label_recovery"
        )
        or {}
    )
    m6_randomized_target_divisor_norm_union_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_randomized_target_divisor_norm_union"
        )
        or {}
    )
    m6_global_randomizer_elliptic_translate_product_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_global_randomizer_elliptic_translate_product"
        )
        or {}
    )
    m6_kummer_x_translate_signed_verification_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_kummer_x_translate_signed_verification"
        )
        or {}
    )
    m6_generalized_target_divisor_weil_reciprocity_swap_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_generalized_target_divisor_weil_reciprocity_swap"
        )
        or {}
    )
    m6_log_derivative_elliptic_cauchy_trace_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_log_derivative_elliptic_cauchy_trace"
        )
        or {}
    )
    m6_regularized_log_trace_displacement_rank_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_regularized_log_trace_displacement_rank"
        )
        or {}
    )
    m6_lambda_zero_fitting_target_norm_dedup_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_lambda_zero_fitting_target_norm_dedup"
        )
        or {}
    )
    m6_balanced_miller_tree_norm_streaming_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_balanced_miller_tree_norm_streaming"
        )
        or {}
    )
    m6_target_sign_conjugate_s3_self_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_target_sign_conjugate_s3_self_resultant"
        )
        or {}
    )
    m6_s3_determinantal_transfer_noncommutativity_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_s3_determinantal_transfer_noncommutativity"
        )
        or {}
    )
    m6_confluent_signed_dual_chow_pushforward_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_confluent_signed_dual_chow_pushforward"
        )
        or {}
    )
    m6_scalar_subset_incidence_group_testing_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_scalar_subset_incidence_group_testing"
        )
        or {}
    )
    m6_principal_target_pontryagin_resultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_principal_target_pontryagin_resultant"
        )
        or {}
    )
    m6_global_marked_fitting_locator_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_global_marked_fitting_locator"
        )
        or {}
    )
    m6_marked_fitting_signed_norm_dedup_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_marked_fitting_signed_norm_dedup"
        )
        or {}
    )
    m6_squarefree_truncated_resultant_applicability_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_squarefree_truncated_resultant_applicability"
        )
        or {}
    )
    m6_d5_directed_evaluation_survivor_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_d5_directed_evaluation_survivor"
        )
        or {}
    )
    m6_monogenic_kernel_bidegree_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_monogenic_kernel_bidegree"
        )
        or {}
    )
    m6_gcd_equivalent_target_subresultant_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_gcd_equivalent_target_subresultant"
        )
        or {}
    )
    m6_sparse_projector_prony_locator_lane = (
        ((frontier_status or {}).get("lanes") or {}).get(
            "m6_sparse_projector_prony_locator"
        )
        or {}
    )
    if m6_sparse_projector_prony_locator_lane.get("closed_by_current_evidence"):
        admission = m6_sparse_projector_prony_locator_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            327,
            str(
                m6_sparse_projector_prony_locator_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_sparse_projector_prony_locator_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} sparse-projector "
                "obligations pass. Exact target-gcd multiplicities and 2c "
                "weighted source moments recover G_1 by Prony, while a source-"
                "blind trace loses labels. Represented projectors and A-linear "
                "target Krylov output retain B^(7/2); an SLP-direct nested-moment "
                "operator without represented nN state remains open."
            ),
        )
    elif m6_gcd_equivalent_target_subresultant_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_gcd_equivalent_target_subresultant_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            326,
            str(
                m6_gcd_equivalent_target_subresultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_gcd_equivalent_target_subresultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} target-subresultant "
                "obligations pass. Exact remainders, Newton traces, unit "
                "inverses, represented norms, and target half-GCD retain "
                "B^(7/2); the opposite-point control validates both target "
                "charts. An SLP-direct candidate-only resultant modulo U "
                "without represented nN state remains open."
            ),
        )
    elif m6_monogenic_kernel_bidegree_lane.get("closed_by_current_evidence"):
        admission = m6_monogenic_kernel_bidegree_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            324,
            str(
                m6_monogenic_kernel_bidegree_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_monogenic_kernel_bidegree_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} monogenic-kernel "
                "obligations pass. The exact canonical kernel has a fully "
                "dense 3n^2 body and full finite source rank, while flattened "
                "norms and complete composed resultants retain B^(7/2). A "
                "gcd-equivalent output-sensitive elliptic resultant modulo U "
                "that discards target-dependent units remains open."
            ),
        )
    elif m6_d5_directed_evaluation_survivor_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_d5_directed_evaluation_survivor_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            322,
            str(
                m6_d5_directed_evaluation_survivor_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_d5_directed_evaluation_survivor_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} D5/directed-evaluation "
                "obligations pass. Every noncandidate CRT component survives "
                "all N target factors, so literal factor streaming and directed "
                "evaluation remain B^(7/2), while standard D5 half-GCD is "
                "B^(9/2). A one-shot monogenic finite-field modular-composition "
                "compiler remains open."
            ),
        )
    elif m6_squarefree_truncated_resultant_applicability_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_squarefree_truncated_resultant_applicability_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            320,
            str(
                m6_squarefree_truncated_resultant_applicability_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_squarefree_truncated_resultant_applicability_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} truncated-resultant "
                "applicability obligations pass. One order-n x-adic expansion "
                "does not determine reduction modulo squarefree U, while n "
                "order-one CRT calls and the single expansion both charge "
                "B^(9/2). The factored arbitrary-squarefree dynamic-evaluation "
                "operator remains open."
            ),
        )
    elif m6_marked_fitting_signed_norm_dedup_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_marked_fitting_signed_norm_dedup_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            318,
            str(
                m6_marked_fitting_signed_norm_dedup_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_marked_fitting_signed_norm_dedup_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Fitting-filtration "
                "obligations pass. R177's first layer is exactly the R174 signed "
                "aggregate-norm factor; higher layers add multiplicities but no "
                "ECDLP roots. The distinct marked-Fitting lane is closed, leaving "
                "the unified nonlocal signed translate-product primitive."
            ),
        )
    elif m6_global_marked_fitting_locator_lane.get("closed_by_current_evidence"):
        admission = m6_global_marked_fitting_locator_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s69_output_sensitive_marked_fitting_locator",
            316,
            str(
                m6_global_marked_fitting_locator_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s69_output_sensitive_marked_fitting_locator"
                ]["decisive_test"]
            ),
            (
                f"{m6_global_marked_fitting_locator_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} global marked-locator "
                "obligations pass. The first nonzero lambda coefficient exactly "
                "emits all 140 R176 roots without subset queries, but the standard "
                "pair algebra is B^(9/2), the full body is B^9, and explicit "
                "interpolation is B^6. Only an output-sensitive marked Fitting "
                "constructor remains."
            ),
        )
    elif m6_principal_target_pontryagin_resultant_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_principal_target_pontryagin_resultant_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s68_factored_trilinear_elliptic_resultant",
            314,
            str(
                m6_principal_target_pontryagin_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s68_factored_trilinear_elliptic_resultant"
                ]["decisive_test"]
            ),
            (
                f"{m6_principal_target_pontryagin_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} principal Pontryagin "
                "obligations pass. The signed principal function and all completed "
                "Weil identities exactly replay R175, but represented pair cycles "
                "and Miller state are B^(9/2). Only the factored trilinear "
                "elliptic resultant remains."
            ),
        )
    elif m6_scalar_subset_incidence_group_testing_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_scalar_subset_incidence_group_testing_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s67_reusable_scalar_subset_incidence_oracle",
            312,
            str(
                m6_scalar_subset_incidence_group_testing_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s67_reusable_scalar_subset_incidence_oracle"
                ]["decisive_test"]
            ),
            (
                f"{m6_scalar_subset_incidence_group_testing_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} scalar subset obligations "
                "pass. Balanced zero-product queries exactly replay all R174 roots "
                "and conditionally cost B^(9/4), but the reusable coordinate-"
                "specific scalar oracle is unsupplied; direct expansion is B^(23/4)."
            ),
        )
    elif m6_confluent_signed_dual_chow_pushforward_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_confluent_signed_dual_chow_pushforward_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s66_fused_factored_dual_chow_outer_norm_mod_u",
            310,
            str(
                m6_confluent_signed_dual_chow_pushforward_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s66_fused_factored_dual_chow_outer_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_confluent_signed_dual_chow_pushforward_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} confluent dual-Chow "
                "obligations pass. The signed two-chart pushforward exactly "
                "replays R166 roots, but represented target Chow is B^(5/2), "
                "represented selected Chow is B^(9/2), and standard query grids "
                "are above rho. Only the fused factored outer norm remains."
            ),
        )
    elif m6_s3_determinantal_transfer_noncommutativity_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_s3_determinantal_transfer_noncommutativity_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s65_commutative_target_sign_divisor_pushforward_mod_u",
            308,
            str(
                m6_s3_determinantal_transfer_noncommutativity_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s65_commutative_target_sign_divisor_pushforward_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_s3_determinantal_transfer_noncommutativity_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} determinantal-transfer "
                "obligations pass. The exact 2 by 2 determinant replays R172, "
                "but the pencil and target factors are noncommutative and its "
                "represented product is B^(5/2). Only a custom commutative or "
                "transposed target-sign pushforward modulo squarefree U remains."
            ),
        )
    elif m6_target_sign_conjugate_s3_self_resultant_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_target_sign_conjugate_s3_self_resultant_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s64_factored_self_s3_resultant_mod_u",
            306,
            str(
                m6_target_sign_conjugate_s3_self_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s64_factored_self_s3_resultant_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_target_sign_conjugate_s3_self_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} target-sign/S3 "
                "obligations pass. The exact conjugate identity and single "
                "denominator batch close the target-y formulation, but the "
                "represented reverse resultant is B^(5/2) and the standard "
                "local route is B^(7/2). Only the factored self-S3 resultant "
                "modulo U remains."
            ),
        )
    elif m6_balanced_miller_tree_norm_streaming_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_balanced_miller_tree_norm_streaming_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s63_nonlocal_batched_elliptic_leaf_translate_product",
            304,
            str(
                m6_balanced_miller_tree_norm_streaming_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s63_nonlocal_batched_elliptic_leaf_translate_product"
                ]["decisive_test"]
            ),
            (
                f"{m6_balanced_miller_tree_norm_streaming_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} balanced-Miller "
                "obligations pass. Exact tree and line-norm semantics close "
                "node-local streaming at B^(7/2); symbolic cancellation leaves "
                "exactly the original target leaves. Only a nonlocal batched "
                "leaf-translate product remains."
            ),
        )
    elif m6_lambda_zero_fitting_target_norm_dedup_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_lambda_zero_fitting_target_norm_dedup_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s62_slp_streaming_output_sensitive_target_norm_mod_u",
            302,
            str(
                m6_lambda_zero_fitting_target_norm_dedup_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s62_slp_streaming_output_sensitive_target_norm_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_lambda_zero_fitting_target_norm_dedup_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} lambda-zero norm "
                "obligations pass. The Fitting specialization is the R167 "
                "target norm up to units; represented factors are dense and "
                "cost B^(7/2). Only an SLP-streaming constructor remains."
            ),
        )
    elif m6_regularized_log_trace_displacement_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_regularized_log_trace_displacement_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s61_fraction_free_elliptic_fitting_subresultant_mod_u",
            300,
            str(
                m6_regularized_log_trace_displacement_rank_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s61_fraction_free_elliptic_fitting_subresultant_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_regularized_log_trace_displacement_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} regularized-trace "
                "obligations pass; scalar-resolvent multiplicities are exact, "
                "while the tested diagonal displacements and generic lambda "
                "interpolation require full B^(9/2) state. The surviving route "
                "is a fraction-free elliptic Fitting/subresultant modulo U."
            ),
        )
    elif m6_log_derivative_elliptic_cauchy_trace_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_log_derivative_elliptic_cauchy_trace_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s60_denominator_aware_elliptic_cauchy_trace_mod_u",
            298,
            str(
                m6_log_derivative_elliptic_cauchy_trace_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s60_denominator_aware_elliptic_cauchy_trace_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_log_derivative_elliptic_cauchy_trace_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} logarithmic-trace "
                "obligations pass; the additive linearization, nonzero-residue "
                "candidate poles, public equality corrections, and finite local "
                "semantics are closed. Only a denominator-aware transposed "
                "trace modulo U and deterministic hash transfer remain."
            ),
        )
    elif m6_generalized_target_divisor_weil_reciprocity_swap_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_generalized_target_divisor_weil_reciprocity_swap_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s59_slp_elliptic_resultant_mod_u",
            296,
            str(
                m6_generalized_target_divisor_weil_reciprocity_swap_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s59_slp_elliptic_resultant_mod_u"
                ]["decisive_test"]
            ),
            (
                f"{m6_generalized_target_divisor_weil_reciprocity_swap_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} target-divisor and "
                "reciprocity obligations pass; compact generalized Miller "
                "state, exact corrected swap semantics, and finite controls "
                "are closed. Only an output-sensitive SLP elliptic resultant "
                "modulo U and deterministic hash transfer remain."
            ),
        )
    elif m6_kummer_x_translate_signed_verification_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_kummer_x_translate_signed_verification_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s58_output_sensitive_kummer_translate_product_remainder",
            294,
            str(
                m6_kummer_x_translate_signed_verification_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s58_output_sensitive_kummer_translate_product_remainder"
                ]["decisive_test"]
            ),
            (
                f"{m6_kummer_x_translate_signed_verification_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Kummer-x obligations "
                "pass; the deterministic zero divisor, opposite-sign density, "
                "B^2 signed verifier, and exact finite candidate semantics are "
                "closed. Only an output-sensitive arbitrary-target Kummer "
                "remainder and deterministic hash transfer remain."
            ),
        )
    elif m6_global_randomizer_elliptic_translate_product_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_global_randomizer_elliptic_translate_product_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s57_single_function_arbitrary_translate_product_remainder",
            292,
            str(
                m6_global_randomizer_elliptic_translate_product_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s57_single_function_arbitrary_translate_product_remainder"
                ]["decisive_test"]
            ),
            (
                f"{m6_global_randomizer_elliptic_translate_product_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} global-randomizer "
                "obligations pass; one fixed function, tangent/pole semantics, "
                "one-sided error, and exact verification are closed. Only an "
                "output-sensitive arbitrary-translate remainder remains."
            ),
        )
    elif m6_randomized_target_divisor_norm_union_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_randomized_target_divisor_norm_union_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s56_output_sensitive_elliptic_translation_target_norm",
            290,
            str(
                m6_randomized_target_divisor_norm_union_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s56_output_sensitive_elliptic_translation_target_norm"
                ]["decisive_test"]
            ),
            (
                f"{m6_randomized_target_divisor_norm_union_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} randomized-norm "
                "obligations pass; target labels, one-sided error, exact B^2 "
                "verification, and the B^(9/4) incidence branch are closed. "
                "Only the sub-rho regular elliptic-translation norm remains."
            ),
        )
    elif m6_aggregate_union_factor_label_recovery_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_aggregate_union_factor_label_recovery_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s55_unlabeled_aggregate_union_factor_constructor",
            288,
            str(
                m6_aggregate_union_factor_label_recovery_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s55_unlabeled_aggregate_union_factor_constructor"
                ]["decisive_test"]
            ),
            (
                f"{m6_aggregate_union_factor_label_recovery_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} aggregate-union "
                "obligations pass; the union factor has output degree "
                "B^(3/4), labels and backpointers cost B^2, and only the "
                "below-rho unlabeled constructor remains open."
            ),
        )
    elif m6_batch_inverse_transpose_modcomp_fit_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_batch_inverse_transpose_modcomp_fit_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s54_aggregate_nonlinear_signed_divisor_below_rho",
            286,
            str(
                m6_batch_inverse_transpose_modcomp_fit_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s54_aggregate_nonlinear_signed_divisor_below_rho"
                ]["decisive_test"]
            ),
            (
                f"{m6_batch_inverse_transpose_modcomp_fit_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} batch-inverse and "
                "fit obligations pass; linear denominator access is exact, "
                "an aggregate B^(9/4) pass remains below rho, and the "
                "nonlinear target-labeled source layer is open."
            ),
        )
    elif m6_signed_c3_divisor_translation_gcd_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_signed_c3_divisor_translation_gcd_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s53_target_batched_signed_divisor_modular_composition_gcd",
            284,
            str(
                m6_signed_c3_divisor_translation_gcd_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s53_target_batched_signed_divisor_modular_composition_gcd"
                ]["decisive_test"]
            ),
            (
                f"{m6_signed_c3_divisor_translation_gcd_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} signed-divisor "
                "obligations pass; the coordinate translation and "
                "degree-at-most-20 source gcd are exact, while independent "
                "target composition costs B^(7/2) and the many-target "
                "transpose remains open."
            ),
        )
    elif m6_positive_c6_generic_locator_reduction_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_positive_c6_generic_locator_reduction_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s52_coordinate_specific_s7_reverse_ffe_source_locator",
            282,
            str(
                m6_positive_c6_generic_locator_reduction_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s52_coordinate_specific_s7_reverse_ffe_source_locator"
                ]["decisive_test"]
            ),
            (
                f"{m6_positive_c6_generic_locator_reduction_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} generic-reduction "
                "obligations pass; Shoup excludes an encoding-invariant "
                "locator at the requested caps, while a coordinate-specific "
                "S7/resultant/FFE source circuit remains open."
            ),
        )
    elif m6_random_diagonal_known_target_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_random_diagonal_known_target_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s51_batched_positive_c6_reverse_ffe_source_locator",
            280,
            str(
                m6_random_diagonal_known_target_rank_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s51_batched_positive_c6_reverse_ffe_source_locator"
                ]["decisive_test"]
            ),
            (
                f"{m6_random_diagonal_known_target_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} direct-target "
                "coverage, rank, and descent obligations pass; the batched "
                "reverse FFE source locator and unconditional cost remain "
                "open."
            ),
        )
    elif m6_short_relation_near_injectivity_supply_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_short_relation_near_injectivity_supply_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s50_conditioned_short_relation_full_rank_reverse_ffe_descent",
            278,
            str(
                m6_short_relation_near_injectivity_supply_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s50_conditioned_short_relation_full_rank_reverse_ffe_descent"
                ]["decisive_test"]
            ),
            (
                f"{m6_short_relation_near_injectivity_supply_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} near-injectivity "
                "and relation-supply obligations pass; exact conditioned "
                "transfer, full coverage/rank, reverse FFE, and identical "
                "descent remain open."
            ),
        )
    elif m6_hash_to_curve_projective_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_hash_to_curve_projective_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s49_public_short_relation_rank_reverse_ffe_descent",
            276,
            str(
                m6_hash_to_curve_projective_rank_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s49_public_short_relation_rank_reverse_ffe_descent"
                ]["decisive_test"]
            ),
            (
                f"{m6_hash_to_curve_projective_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} public-group "
                "transfer obligations pass; asymptotic rank, reverse FFE, "
                "and identical descent remain open."
            ),
        )
    elif m6_a_diversity_projective_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_a_diversity_projective_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s48_projective_singleton_direct_rank_reverse_ffe",
            274,
            str(
                m6_a_diversity_projective_rank_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s48_projective_singleton_direct_rank_reverse_ffe"
                ]["decisive_test"]
            ),
            (
                f"{m6_a_diversity_projective_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} projective "
                "A-diversity obligations pass; a direct asymptotic rank "
                "theorem and the reverse FFE operator remain open."
            ),
        )
    elif m6_singleton_relation_hypergraph_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_singleton_relation_hypergraph_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s47_convolution_tanner_contiguity_reverse_ffe",
            272,
            str(
                m6_singleton_relation_hypergraph_rank_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s47_convolution_tanner_contiguity_reverse_ffe"
                ]["decisive_test"]
            ),
            (
                f"{m6_singleton_relation_hypergraph_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} singleton "
                "hypergraph obligations pass; convolution-Tanner "
                "contiguity and the reverse FFE operator remain open."
            ),
        )
    elif m6_signed_quotient_multiscale_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_signed_quotient_multiscale_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s46_signed_quotient_random_rank_reverse_ffe_transfer",
            270,
            str(
                m6_signed_quotient_multiscale_rank_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s46_signed_quotient_random_rank_reverse_ffe_transfer"
                ]["decisive_test"]
            ),
            (
                f"{m6_signed_quotient_multiscale_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} signed quotient "
                "and multiscale obligations pass; random-rank transfer and "
                "the reverse FFE operator remain open."
            ),
        )
    elif m6_symmetric_shift_reverse_only_marginal_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_symmetric_shift_reverse_only_marginal_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s45_reverse_only_signed_marker_density_transfer",
            268,
            str(
                m6_symmetric_shift_reverse_only_marginal_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s45_reverse_only_signed_marker_density_transfer"
                ]["decisive_test"]
            ),
            (
                f"{m6_symmetric_shift_reverse_only_marginal_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} symmetric "
                "reverse-only obligations pass; finite rank deficit remains."
            ),
        )
    elif m6_geometry_only_weight_interpolation_adjoint_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_geometry_only_weight_interpolation_adjoint_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s44_signed_weight_separable_ffe_elimination_dag",
            266,
            str(
                m6_geometry_only_weight_interpolation_adjoint_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s44_signed_weight_separable_ffe_elimination_dag"
                ]["decisive_test"]
            ),
            (
                f"{m6_geometry_only_weight_interpolation_adjoint_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} geometry-only "
                "leaf derivative obligations pass."
            ),
        )
    elif m6_matrix_free_marginal_jacobian_krylov_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_matrix_free_marginal_jacobian_krylov_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s43_weight_parametric_bidirectional_marker_operator",
            264,
            str(
                m6_matrix_free_marginal_jacobian_krylov_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s43_weight_parametric_bidirectional_marker_operator"
                ]["decisive_test"]
            ),
            (
                f"{m6_matrix_free_marginal_jacobian_krylov_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} matrix-free "
                "Jacobian obligations pass."
            ),
        )
    elif m6_rational_convolution_subalgebra_rigidity_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_rational_convolution_subalgebra_rigidity_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s42_finite_depth_nonhomomorphic_u6_marker_circuit",
            262,
            str(
                m6_rational_convolution_subalgebra_rigidity_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s42_finite_depth_nonhomomorphic_u6_marker_circuit"
                ]["decisive_test"]
            ),
            (
                f"{m6_rational_convolution_subalgebra_rigidity_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} rational "
                "convolution-subalgebra obligations pass."
            ),
        )
    elif m6_actual_c6_shift_krylov_rank_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_actual_c6_shift_krylov_rank_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s41_nonlinear_target_specialized_compact_divisor_circuit",
            260,
            str(
                m6_actual_c6_shift_krylov_rank_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s41_nonlinear_target_specialized_compact_divisor_circuit"
                ]["decisive_test"]
            ),
            (
                f"{m6_actual_c6_shift_krylov_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} actual "
                "shift-Krylov obligations pass."
            ),
        )
    elif m6_static_3sum_indexing_tradeoff_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_static_3sum_indexing_tradeoff_lane.get("admission")
            or {}
        )
        add_focus_candidate(
            candidates,
            "s40_structure_aware_occurrence_autocorrelation",
            258,
            str(
                m6_static_3sum_indexing_tradeoff_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s40_structure_aware_occurrence_autocorrelation"
                ]["decisive_test"]
            ),
            (
                f"{m6_static_3sum_indexing_tradeoff_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} static-indexing "
                "tradeoff obligations pass."
            ),
        )
    elif m6_occurrence_pair_resultant_local_valuation_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_occurrence_pair_resultant_local_valuation_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s39_shared_transposed_multi_target_valuation_marker",
            256,
            str(
                m6_occurrence_pair_resultant_local_valuation_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s39_shared_transposed_multi_target_valuation_marker"
                ]["decisive_test"]
            ),
            (
                f"{m6_occurrence_pair_resultant_local_valuation_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} local-valuation "
                "obligations pass."
            ),
        )
    elif m6_aggregate_marginal_singleton_source_equivalence_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            m6_aggregate_marginal_singleton_source_equivalence_lane.get(
                "admission"
            )
            or {}
        )
        add_focus_candidate(
            candidates,
            "s38_source_equivalent_batched_count_marginal_index",
            254,
            str(
                m6_aggregate_marginal_singleton_source_equivalence_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s38_source_equivalent_batched_count_marginal_index"
                ]["decisive_test"]
            ),
            (
                f"{m6_aggregate_marginal_singleton_source_equivalence_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} singleton "
                "source-equivalence obligations pass."
            ),
        )
    elif m6_weighted_c3_mobius_gcd_trace_lane.get(
        "closed_by_current_evidence"
    ):
        admission = m6_weighted_c3_mobius_gcd_trace_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s37_implicit_batched_mobius_resultant",
            252,
            str(
                m6_weighted_c3_mobius_gcd_trace_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s37_implicit_batched_mobius_resultant"
                ]["decisive_test"]
            ),
            (
                f"{m6_weighted_c3_mobius_gcd_trace_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} weighted "
                "Mobius-gcd trace obligations pass."
            ),
        )
    elif m6_weighted_fiber_marginal_log_operator_lane.get(
        "closed_by_current_evidence"
    ):
        admission = m6_weighted_fiber_marginal_log_operator_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s36_weighted_s13_fiber_count_transpose",
            250,
            str(
                m6_weighted_fiber_marginal_log_operator_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s36_weighted_s13_fiber_count_transpose"
                ]["decisive_test"]
            ),
            (
                f"{m6_weighted_fiber_marginal_log_operator_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} weighted "
                "fiber-marginal obligations pass."
            ),
        )
    elif torus_label_congruence_correction_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_label_congruence_correction_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s35_transposed_ffe_relation_span",
            248,
            str(
                torus_label_congruence_correction_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s35_transposed_ffe_relation_span"
                ]["decisive_test"]
            ),
            (
                f"{torus_label_congruence_correction_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} label-congruence "
                "and explicit-row obligations pass."
            ),
        )
    elif torus_adaptive_character_decision_router_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_adaptive_character_decision_router_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s34_algebraic_sextic_character_composition",
            246,
            str(
                torus_adaptive_character_decision_router_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s34_algebraic_sextic_character_composition"
                ]["decisive_test"]
            ),
            (
                f"{torus_adaptive_character_decision_router_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} adaptive "
                "character-router obligations pass."
            ),
        )
    elif torus_sextic_mobius_character_router_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_sextic_mobius_character_router_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s33_nonlinear_sextic_mobius_source_router",
            244,
            str(
                torus_sextic_mobius_character_router_lane.get(
                    "next_action"
                )
                or CRITICAL_EXPERIMENTS[
                    "s33_nonlinear_sextic_mobius_source_router"
                ]["decisive_test"]
            ),
            (
                f"{torus_sextic_mobius_character_router_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} sextic "
                "Mobius-character obligations pass."
            ),
        )
    elif torus_order_two_four_minor_claw_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_order_two_four_minor_claw_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s32_subcap_mobius_claw_or_nonzero_torus_c5_selector",
            242,
            str(
                torus_order_two_four_minor_claw_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s32_subcap_mobius_claw_or_nonzero_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_order_two_four_minor_claw_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} order-two "
                "four-minor claw obligations pass."
            ),
        )
    elif torus_order_two_three_minor_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_order_two_three_minor_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s31_four_plus_or_nonzero_torus_c5_selector",
            240,
            str(
                torus_order_two_three_minor_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s31_four_plus_or_nonzero_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_order_two_three_minor_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} order-two "
                "three-minor obligations pass."
            ),
        )
    elif torus_chebotarev_fiber_cover_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_chebotarev_fiber_cover_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s30_characteristic_specific_spark_or_nonzero_torus_c5_selector",
            238,
            str(
                torus_chebotarev_fiber_cover_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s30_characteristic_specific_spark_or_nonzero_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_chebotarev_fiber_cover_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} fiber-cover "
                "and theorem-transfer obligations pass."
            ),
        )
    elif torus_binomial_node_union_depth_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_binomial_node_union_depth_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s29_three_plus_five_plus_nonzero_torus_c5_selector",
            236,
            str(
                torus_binomial_node_union_depth_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s29_three_plus_five_plus_nonzero_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_binomial_node_union_depth_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} nodewise "
                "root-union obligations pass."
            ),
        )
    elif torus_all_nonzero_path_product_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_all_nonzero_path_product_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s28_growing_support_low_slp_nonzero_torus_c5_selector",
            234,
            str(
                torus_all_nonzero_path_product_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s28_growing_support_low_slp_nonzero_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_all_nonzero_path_product_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} all-nonzero "
                "path-product obligations pass."
            ),
        )
    elif torus_khatri_rao_amplification_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_khatri_rao_amplification_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s27_composed_nonzero_low_slp_torus_c5_selector",
            232,
            str(
                torus_khatri_rao_amplification_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s27_composed_nonzero_low_slp_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_khatri_rao_amplification_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Khatri-Rao "
                "amplification obligations pass."
            ),
        )
    elif torus_two_atom_progression_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_two_atom_progression_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s26_seven_mode_multi_predicate_torus_c5_selector",
            230,
            str(
                torus_two_atom_progression_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s26_seven_mode_multi_predicate_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_two_atom_progression_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} two-atom "
                "progression obligations pass."
            ),
        )
    elif torus_sparse_monomial_root_bound_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_sparse_monomial_root_bound_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s25_five_mode_low_slp_frobenius_torus_c5_selector",
            228,
            str(
                torus_sparse_monomial_root_bound_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s25_five_mode_low_slp_frobenius_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_sparse_monomial_root_bound_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} sparse-root "
                "obligations pass."
            ),
        )
    elif torus_base_field_frobenius_dag_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_base_field_frobenius_dag_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s24_asymmetric_frobenius_torus_c5_selector_predicate",
            226,
            str(
                torus_base_field_frobenius_dag_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s24_asymmetric_frobenius_torus_c5_selector_predicate"
                ]["decisive_test"]
            ),
            (
                f"{torus_base_field_frobenius_dag_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Frobenius-DAG "
                "obligations pass."
            ),
        )
    elif torus_consecutive_mode_predicate_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_consecutive_mode_predicate_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s23_lacunary_order2_torus_c5_selector_predicate",
            224,
            str(
                torus_consecutive_mode_predicate_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s23_lacunary_order2_torus_c5_selector_predicate"
                ]["decisive_test"]
            ),
            (
                f"{torus_consecutive_mode_predicate_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} consecutive-mode "
                "obligations pass."
            ),
        )
    elif torus_sparse_fourier_transfer_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_sparse_fourier_transfer_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s22_order2_finite_field_torus_c5_selector_predicate",
            222,
            str(
                torus_sparse_fourier_transfer_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s22_order2_finite_field_torus_c5_selector_predicate"
                ]["decisive_test"]
            ),
            (
                f"{torus_sparse_fourier_transfer_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Fourier-transfer "
                "obligations pass."
            ),
        )
    elif torus_piecewise_selector_dag_lane.get(
        "closed_by_current_evidence"
    ):
        admission = torus_piecewise_selector_dag_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s21_shared_predicate_torus_c5_selector_dag",
            220,
            str(
                torus_piecewise_selector_dag_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s21_shared_predicate_torus_c5_selector_dag"
                ]["decisive_test"]
            ),
            (
                f"{torus_piecewise_selector_dag_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} piecewise-selector "
                "obligations pass."
            ),
        )
    elif torus_rational_selector_lane.get("closed_by_current_evidence"):
        admission = torus_rational_selector_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s20_low_slp_piecewise_torus_c5_selector",
            218,
            str(
                torus_rational_selector_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s20_low_slp_piecewise_torus_c5_selector"
                ]["decisive_test"]
            ),
            (
                f"{torus_rational_selector_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} rational-selector "
                "obligations pass."
            ),
        )
    elif torus_bucket_resultant_tradeoff_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            torus_bucket_resultant_tradeoff_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s19_cap_tight_singleton_c3_target_router",
            216,
            str(
                torus_bucket_resultant_tradeoff_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s19_cap_tight_singleton_c3_target_router"
                ]["decisive_test"]
            ),
            (
                f"{torus_bucket_resultant_tradeoff_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} bucket-resultant "
                "tradeoff obligations pass."
            ),
        )
    elif torus_explicit_hash_correction_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            torus_explicit_hash_correction_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s18_implicit_adaptive_torus_c5_hash_correction_circuit",
            214,
            str(
                torus_explicit_hash_correction_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s18_implicit_adaptive_torus_c5_hash_correction_circuit"
                ]["decisive_test"]
            ),
            (
                f"{torus_explicit_hash_correction_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} explicit-"
                "correction obligations pass."
            ),
        )
    elif torus_homomorphic_fingerprint_lane.get(
        "closed_by_current_evidence"
    ):
        admission = (
            torus_homomorphic_fingerprint_lane.get("admission") or {}
        )
        add_focus_candidate(
            candidates,
            "s17_nonhomomorphic_adaptive_torus_c5_fingerprint",
            212,
            str(
                torus_homomorphic_fingerprint_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s17_nonhomomorphic_adaptive_torus_c5_fingerprint"
                ]["decisive_test"]
            ),
            (
                f"{torus_homomorphic_fingerprint_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} homomorphic-"
                "fingerprint obligations pass."
            ),
        )
    elif torus_linear_sketch_lane.get("closed_by_current_evidence"):
        admission = torus_linear_sketch_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s16_coupled_nonlinear_torus_c5_zero_test",
            210,
            str(
                torus_linear_sketch_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s16_coupled_nonlinear_torus_c5_zero_test"
                ]["decisive_test"]
            ),
            (
                f"{torus_linear_sketch_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} linear-sketch "
                "obligations pass."
            ),
        )
    elif torus_fourier_resultant_lane.get("closed_by_current_evidence"):
        admission = torus_fourier_resultant_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s15_nonrepresented_fourier_resultant_torus_c5_source_circuit",
            208,
            str(
                torus_fourier_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s15_nonrepresented_fourier_resultant_torus_c5_"
                    "source_circuit"
                ]["decisive_test"]
            ),
            (
                f"{torus_fourier_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Fourier/resultant "
                "obligations pass."
            ),
        )
    elif torus_split_rebalance_lane.get("closed_by_current_evidence"):
        admission = torus_split_rebalance_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s14_target_specialized_nonoccurrence_torus_c5_source_circuit",
            206,
            str(
                torus_split_rebalance_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s14_target_specialized_nonoccurrence_torus_c5_"
                    "source_circuit"
                ]["decisive_test"]
            ),
            (
                f"{torus_split_rebalance_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} global-rebalance "
                "obligations pass."
            ),
        )
    elif m6_moment_torus_lane.get("closed_by_current_evidence"):
        admission = m6_moment_torus_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s13_m6_target_specialized_nonlinear_torus_c5_source_circuit",
            204,
            str(
                m6_moment_torus_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s13_m6_target_specialized_nonlinear_torus_c5_"
                    "source_circuit"
                ]["decisive_test"]
            ),
            (
                f"{m6_moment_torus_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} torus/moment "
                "obligations pass."
            ),
        )
    elif m6_character_pairing_lane.get("closed_by_current_evidence"):
        admission = m6_character_pairing_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s12_m6_small_k_multiplicative_c5_membership_source_circuit",
            202,
            str(
                m6_character_pairing_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s12_m6_small_k_multiplicative_c5_membership_source_"
                    "circuit"
                ]["decisive_test"]
            ),
            (
                f"{m6_character_pairing_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} character/pairing "
                "obligations pass."
            ),
        )
    elif m6_output_sensitive_c5_lane.get("closed_by_current_evidence"):
        admission = m6_output_sensitive_c5_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s11_m6_suboutput_implicit_c5_membership_source_circuit",
            200,
            str(
                m6_output_sensitive_c5_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s11_m6_suboutput_implicit_c5_membership_source_circuit"
                ]["decisive_test"]
            ),
            (
                f"{m6_output_sensitive_c5_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} output-sensitive "
                "C5 source-index obligations pass."
            ),
        )
    elif m6_nonlinear_c6_lane.get("closed_by_current_evidence"):
        admission = m6_nonlinear_c6_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s10_m6_output_sensitive_nonlinear_c5_source_index",
            198,
            str(
                m6_nonlinear_c6_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s10_m6_output_sensitive_nonlinear_c5_source_index"
                ]["decisive_test"]
            ),
            (
                f"{m6_nonlinear_c6_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} nonlinear C6 "
                "source-locator obligations pass."
            ),
        )
    elif m6_elliptic_transpose_lane.get("closed_by_current_evidence"):
        admission = m6_elliptic_transpose_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s9_m6_nonlinear_value_sensitive_c6_source_locator",
            196,
            str(
                m6_elliptic_transpose_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s9_m6_nonlinear_value_sensitive_c6_source_locator"
                ]["decisive_test"]
            ),
            (
                f"{m6_elliptic_transpose_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} target-batched "
                "elliptic-transpose obligations pass."
            ),
        )
    elif m6_pair_sum_lane.get("closed_by_current_evidence"):
        admission = m6_pair_sum_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s8_m6_target_batched_c3_pair_sum_elliptic_transpose",
            194,
            str(
                m6_pair_sum_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s8_m6_target_batched_c3_pair_sum_elliptic_transpose"
                ]["decisive_test"]
            ),
            (
                f"{m6_pair_sum_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} A6-batched C3 "
                "pair-sum obligations pass."
            ),
        )
    elif exponent_rebalance_lane.get("closed_by_current_evidence"):
        admission = exponent_rebalance_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s7_m6_implicit_3f_self_convolution_ffe_source_locator",
            192,
            str(
                exponent_rebalance_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s7_m6_implicit_3f_self_convolution_ffe_source_locator"
                ]["decisive_test"]
            ),
            (
                f"{exponent_rebalance_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} exponent-rebalance "
                "obligations pass."
            ),
        )
    elif transposed_leaf_lane.get("closed_by_current_evidence"):
        admission = transposed_leaf_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_relation_arity_factor_base_transposed_interface_rebalance",
            190,
            str(
                transposed_leaf_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_relation_arity_factor_base_transposed_interface_rebalance"
                ]["decisive_test"]
            ),
            (
                f"{transposed_leaf_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} transposed-leaf "
                "obligations pass."
            ),
        )
    elif orbit_product_lane.get("closed_by_current_evidence"):
        admission = orbit_product_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_transposed_nonuniform_c5_leaf_generator",
            188,
            str(
                orbit_product_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_transposed_nonuniform_c5_leaf_generator"
                ]["decisive_test"]
            ),
            (
                f"{orbit_product_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} orbit-product "
                "obligations pass."
            ),
        )
    elif endpoint_query2p1_lane.get("closed_by_current_evidence"):
        admission = endpoint_query2p1_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_nonlinear_elliptic_orbit_product_recurrence",
            186,
            str(
                endpoint_query2p1_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_nonlinear_elliptic_orbit_product_recurrence"
                ]["decisive_test"]
            ),
            (
                f"{endpoint_query2p1_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} endpoint "
                "Query2P1 obligations pass."
            ),
        )
    elif finite_deck_annihilator_lane.get("closed_by_current_evidence"):
        admission = finite_deck_annihilator_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_gauge_normalized_endpoint_query2p1_index",
            184,
            str(
                finite_deck_annihilator_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_gauge_normalized_endpoint_query2p1_index"
                ]["decisive_test"]
            ),
            (
                f"{finite_deck_annihilator_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} finite-deck "
                "annihilator obligations pass."
            ),
        )
    elif theta_addition_lane.get("closed_by_current_evidence"):
        admission = theta_addition_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_finite_deck_alternant_annihilator_contraction",
            182,
            str(
                theta_addition_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_finite_deck_alternant_annihilator_contraction"
                ]["decisive_test"]
            ),
            (
                f"{theta_addition_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} theta-addition "
                "network obligations pass."
            ),
        )
    elif poincare_section_rank_lane.get("closed_by_current_evidence"):
        admission = poincare_section_rank_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_theta_addition_cancellation_network",
            180,
            str(
                poincare_section_rank_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_theta_addition_cancellation_network"
                ]["decisive_test"]
            ),
            (
                f"{poincare_section_rank_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Poincare/theta "
                "section obligations pass."
            ),
        )
    elif lambda_ring_chow_lane.get("closed_by_current_evidence"):
        admission = lambda_ring_chow_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_poincare_theta_target_section_rank",
            178,
            str(
                lambda_ring_chow_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_poincare_theta_target_section_rank"
                ]["decisive_test"]
            ),
            (
                f"{lambda_ring_chow_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} lambda-ring/Chow "
                "obligations pass."
            ),
        )
    elif noncharacter_resultant_lane.get("closed_by_current_evidence"):
        admission = noncharacter_resultant_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_factored_elliptic_lambda_ring_chow_norm_circuit",
            176,
            str(
                noncharacter_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_factored_elliptic_lambda_ring_chow_norm_circuit"
                ]["decisive_test"]
            ),
            (
                f"{noncharacter_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} non-character "
                "resultant obligations pass."
            ),
        )
    elif scalar_target_norm_lane.get("closed_by_current_evidence"):
        admission = scalar_target_norm_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_noncharacter_algebraic_target_norm_resultant_circuit",
            174,
            str(
                scalar_target_norm_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_noncharacter_algebraic_target_norm_resultant_circuit"
                ]["decisive_test"]
            ),
            (
                f"{scalar_target_norm_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} scalar target "
                "norm/count obligations pass."
            ),
        )
    elif actual_deck_pullback_lane.get("closed_by_current_evidence"):
        admission = actual_deck_pullback_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_scalar_target_norm_count_circuit",
            172,
            str(
                actual_deck_pullback_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_scalar_target_norm_count_circuit"
                ]["decisive_test"]
            ),
            (
                f"{actual_deck_pullback_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} actual-deck "
                "pullback obligations pass."
            ),
        )
    elif preendpoint_pushdown_lane.get("closed_by_current_evidence"):
        admission = preendpoint_pushdown_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_actual_deck_nonmergeable_target_pullback_circuit",
            170,
            str(
                preendpoint_pushdown_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_actual_deck_nonmergeable_target_pullback_circuit"
                ]["decisive_test"]
            ),
            (
                f"{preendpoint_pushdown_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} compact "
                "pre-endpoint pushdown obligations pass."
            ),
        )
    elif target_forced_filter_lane.get("closed_by_current_evidence"):
        admission = target_forced_filter_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_compact_preendpoint_s3_ffe_pushdown",
            168,
            str(
                target_forced_filter_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_compact_preendpoint_s3_ffe_pushdown"
                ]["decisive_test"]
            ),
            (
                f"{target_forced_filter_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} target-forced "
                "filter obligations pass."
            ),
        )
    elif two_sided_join_lane.get("closed_by_current_evidence"):
        admission = two_sided_join_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_target_forced_algebraic_join_filter",
            166,
            str(
                two_sided_join_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_target_forced_algebraic_join_filter"
                ]["decisive_test"]
            ),
            (
                f"{two_sided_join_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} two-sided "
                "join obligations pass."
            ),
        )
    elif actual_divisor_image_lane.get("closed_by_current_evidence"):
        admission = actual_divisor_image_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_two_sided_implicit_join",
            164,
            str(
                actual_divisor_image_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_two_sided_implicit_join"
                ]["decisive_test"]
            ),
            (
                f"{actual_divisor_image_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} actual-image "
                "oracle obligations pass."
            ),
        )
    elif aggregate_digit_trie_lane.get("closed_by_current_evidence"):
        admission = aggregate_digit_trie_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_actual_divisor_image_entropy_merge",
            162,
            str(
                aggregate_digit_trie_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_actual_divisor_image_entropy_merge"
                ]["decisive_test"]
            ),
            (
                f"{aggregate_digit_trie_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} aggregate "
                "digit-trie obligations pass."
            ),
        )
    elif multiedge_digitized_lane.get("closed_by_current_evidence"):
        admission = multiedge_digitized_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_succinct_aggregate_digit_trie",
            160,
            str(
                multiedge_digitized_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_succinct_aggregate_digit_trie"
                ]["decisive_test"]
            ),
            (
                f"{multiedge_digitized_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} multi-edge "
                "digitized-projector obligations pass."
            ),
        )
    elif nonlinear_tensor_tower_lane.get("closed_by_current_evidence"):
        admission = nonlinear_tensor_tower_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_multiedge_digitized_equality_projector",
            158,
            str(
                nonlinear_tensor_tower_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_multiedge_digitized_equality_projector"
                ]["decisive_test"]
            ),
            (
                f"{nonlinear_tensor_tower_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} nonlinear "
                "tensor-tower obligations pass."
            ),
        )
    elif factored_transposed_trace_lane.get(
        "closed_by_current_evidence"
    ):
        admission = factored_transposed_trace_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_nonlinear_tensor_tower_trace",
            156,
            str(
                factored_transposed_trace_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_nonlinear_tensor_tower_trace"
                ]["decisive_test"]
            ),
            (
                f"{factored_transposed_trace_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} factored "
                "transposed-trace obligations pass."
            ),
        )
    elif modular_frobenius_trace_lane.get("closed_by_current_evidence"):
        admission = modular_frobenius_trace_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_factored_transposed_projector_trace",
            154,
            str(
                modular_frobenius_trace_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_factored_transposed_projector_trace"
                ]["decisive_test"]
            ),
            (
                f"{modular_frobenius_trace_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} modular "
                "Frobenius-trace obligations pass."
            ),
        )
    elif aggregate_veronese_recurrence_lane.get(
        "closed_by_current_evidence"
    ):
        admission = aggregate_veronese_recurrence_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_modular_frobenius_trace_recurrence",
            152,
            str(
                aggregate_veronese_recurrence_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_modular_frobenius_trace_recurrence"
                ]["decisive_test"]
            ),
            (
                f"{aggregate_veronese_recurrence_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} aggregate "
                "projector-recurrence obligations pass."
            ),
        )
    elif implicit_veronese_index_lane.get("closed_by_current_evidence"):
        admission = implicit_veronese_index_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_aggregate_veronese_projector_recurrence",
            150,
            str(
                implicit_veronese_index_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_aggregate_veronese_projector_recurrence"
                ]["decisive_test"]
            ),
            (
                f"{implicit_veronese_index_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} implicit "
                "Veronese source-index obligations pass."
            ),
        )
    elif shared_semilinear_incidence_lane.get(
        "closed_by_current_evidence"
    ):
        admission = shared_semilinear_incidence_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_implicit_veronese_hyperplane_source_index",
            148,
            str(
                shared_semilinear_incidence_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_implicit_veronese_hyperplane_source_index"
                ]["decisive_test"]
            ),
            (
                f"{shared_semilinear_incidence_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} shared semilinear "
                "incidence obligations pass."
            ),
        )
    elif compact_elliptic_subfunction_lane.get(
        "closed_by_current_evidence"
    ):
        admission = compact_elliptic_subfunction_lane.get(
            "admission"
        ) or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_shared_semilinear_incidence_correspondence",
            146,
            str(
                compact_elliptic_subfunction_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_shared_semilinear_incidence_correspondence"
                ]["decisive_test"]
            ),
            (
                f"{compact_elliptic_subfunction_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} compact elliptic "
                "subfunction obligations pass."
            ),
        )
    elif unequal_list_subfunction_lane.get("closed_by_current_evidence"):
        admission = unequal_list_subfunction_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_compact_elliptic_subfunction_map",
            144,
            str(
                unequal_list_subfunction_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_compact_elliptic_subfunction_map"
                ]["decisive_test"]
            ),
            (
                f"{unequal_list_subfunction_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} unequal-list "
                "subfunction obligations pass."
            ),
        )
    elif nonlocal_moment_hankel_lane.get("closed_by_current_evidence"):
        admission = nonlocal_moment_hankel_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_unequal_list_subfunction_inversion_index",
            142,
            str(
                nonlocal_moment_hankel_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_unequal_list_subfunction_inversion_index"
                ]["decisive_test"]
            ),
            (
                f"{nonlocal_moment_hankel_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} nonlocal "
                "moment/Hankel obligations pass."
            ),
        )
    elif fixed_marker_recurrence_lane.get("closed_by_current_evidence"):
        admission = fixed_marker_recurrence_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_nonlocal_nonlinear_translation_sketch",
            140,
            str(
                fixed_marker_recurrence_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_nonlocal_nonlinear_translation_sketch"
                ]["decisive_test"]
            ),
            (
                f"{fixed_marker_recurrence_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} fixed-marker "
                "recurrence obligations pass."
            ),
        )
    elif black_box_resultant_lane.get("closed_by_current_evidence"):
        admission = black_box_resultant_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_coefficient_free_fixed_marker_resultant_recurrence",
            138,
            str(
                black_box_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_coefficient_free_fixed_marker_resultant_recurrence"
                ]["decisive_test"]
            ),
            (
                f"{black_box_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} black-box "
                "resultant obligations pass."
            ),
        )
    elif jet_pushforward_lane.get("closed_by_current_evidence"):
        admission = jet_pushforward_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_black_box_translated_resultant_gcd_localizer",
            136,
            str(
                jet_pushforward_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_black_box_translated_resultant_gcd_localizer"
                ]["decisive_test"]
            ),
            (
                f"{jet_pushforward_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} jet-pushforward "
                "obligations pass."
            ),
        )
    elif sparse_moment_lane.get("closed_by_current_evidence"):
        admission = sparse_moment_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_jet_preserving_addition_pushforward_intertwiner",
            134,
            str(
                sparse_moment_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_jet_preserving_addition_pushforward_intertwiner"
                ]["decisive_test"]
            ),
            (
                f"{sparse_moment_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} sparse-moment "
                "constructor obligations pass."
            ),
        )
    elif precoefficient_circuit_lane.get("closed_by_current_evidence"):
        admission = precoefficient_circuit_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_sparse_multihomogeneous_moment_recurrence",
            132,
            str(
                precoefficient_circuit_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_sparse_multihomogeneous_moment_recurrence"
                ]["decisive_test"]
            ),
            (
                f"{precoefficient_circuit_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} pre-coefficient "
                "circuit obligations pass."
            ),
        )
    elif marked_resultant_lane.get("closed_by_current_evidence"):
        admission = marked_resultant_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_target_uniform_precoefficient_circuit",
            130,
            str(
                marked_resultant_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_target_uniform_precoefficient_circuit"
                ]["decisive_test"]
            ),
            (
                f"{marked_resultant_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} marked-resultant "
                "source-section obligations pass."
            ),
        )
    elif coordinate_filtration_lane.get("closed_by_current_evidence"):
        admission = coordinate_filtration_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_5a5c_marked_resultant_source_section",
            128,
            str(
                coordinate_filtration_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_5a5c_marked_resultant_source_section"
                ]["decisive_test"]
            ),
            (
                f"{coordinate_filtration_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} coordinate-"
                "filtration obligations pass."
            ),
        )
    elif cartesian_sum_lane.get("closed_by_current_evidence"):
        admission = cartesian_sum_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_addition_compatible_5a5c_field_filtration",
            126,
            str(
                cartesian_sum_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_addition_compatible_5a5c_field_filtration"
                ]["decisive_test"]
            ),
            (
                f"{cartesian_sum_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} Cartesian-sum "
                "compact-divisor obligations pass."
            ),
        )
    elif full_coset_lane.get("closed_by_current_evidence"):
        admission = full_coset_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_compact_divisor_factor_base_endpoint_compiler",
            124,
            str(
                full_coset_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_compact_divisor_factor_base_endpoint_compiler"
                ]["decisive_test"]
            ),
            (
                f"{full_coset_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} full-coset "
                "endpoint-compression obligations pass."
            ),
        )
    elif batched_norm_lane.get("closed_by_current_evidence"):
        admission = batched_norm_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_structured_factor_base_endpoint_compression",
            122,
            str(
                batched_norm_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_structured_factor_base_endpoint_compression"
                ]["decisive_test"]
            ),
            (
                f"{batched_norm_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} batched "
                "nested-norm compiler obligations pass."
            ),
        )
    elif scalar_norm_lane.get("closed_by_current_evidence"):
        admission = scalar_norm_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_batched_nested_norm_node_compiler",
            120,
            str(
                scalar_norm_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_batched_nested_norm_node_compiler"
                ]["decisive_test"]
            ),
            (
                f"{scalar_norm_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} scalar-only "
                "nested-norm obligations pass."
            ),
        )
    elif fermat_tt_lane.get("closed_by_current_evidence"):
        admission = fermat_tt_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_scalar_only_black_box_nested_norm",
            118,
            str(
                fermat_tt_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_scalar_only_black_box_nested_norm"
                ]["decisive_test"]
            ),
            (
                f"{fermat_tt_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} value-first "
                "nonlinear-functional obligations pass."
            ),
        )
    elif frequency_orbit_lane.get("closed_by_current_evidence"):
        admission = frequency_orbit_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_nonlinear_target_specialized_nested_resultant",
            116,
            str(
                frequency_orbit_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_nonlinear_target_specialized_nested_resultant"
                ]["decisive_test"]
            ),
            (
                f"{frequency_orbit_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} linear "
                "translation-orbit obligations pass."
            ),
        )
    elif subset_incidence_lane.get("closed_by_current_evidence"):
        admission = subset_incidence_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_target_translated_subset_frequency_oracle",
            114,
            str(
                subset_incidence_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_target_translated_subset_frequency_oracle"
                ]["decisive_test"]
            ),
            (
                f"{subset_incidence_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} exact "
                "subset-incidence obligations pass."
            ),
        )
    elif iterated_norm_lane.get("closed_by_current_evidence"):
        admission = iterated_norm_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_transposed_norm_scalar_functional",
            112,
            str(
                iterated_norm_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_transposed_norm_scalar_functional"
                ]["decisive_test"]
            ),
            (
                f"{iterated_norm_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} iterated-norm "
                "grammar obligations pass."
            ),
        )
    elif residual_decision_lane.get("closed_by_current_evidence"):
        admission = residual_decision_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_support_adaptive_transposed_incidence",
            110,
            str(
                residual_decision_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_support_adaptive_transposed_incidence"
                ]["decisive_test"]
            ),
            (
                f"{residual_decision_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} residual-decision "
                "grammar obligations pass."
            ),
        )
    elif resultant_valuation_lane.get("closed_by_current_evidence"):
        admission = resultant_valuation_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_quotient_algebra_trace_transducer",
            108,
            str(
                resultant_valuation_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_quotient_algebra_trace_transducer"
                ]["decisive_test"]
            ),
            (
                f"{resultant_valuation_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} resultant-valuation "
                "grammar obligations pass."
            ),
        )
    elif s6_carry_lane.get("closed_by_current_evidence"):
        admission = s6_carry_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_noncp_balanced_trace_contraction",
            106,
            str(
                s6_carry_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_noncp_balanced_trace_contraction"
                ]["decisive_test"]
            ),
            (
                f"{s6_carry_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} S6 carry "
                "obligations pass."
            ),
        )
    elif s4_carry_lane.get("closed_by_current_evidence"):
        admission = s4_carry_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "s6_unit_zero_divisor_source_router",
            104,
            str(
                s4_carry_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "s6_unit_zero_divisor_source_router"
                ]["decisive_test"]
            ),
            (
                f"{s4_carry_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} centered-carry "
                "obligations pass."
            ),
        )
    elif multiplicative_x_lane.get("closed_by_current_evidence"):
        admission = multiplicative_x_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "structured_closure_collision_locator",
            102,
            str(
                multiplicative_x_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "structured_closure_collision_locator"
                ]["decisive_test"]
            ),
            (
                f"{multiplicative_x_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} multiplicative-x "
                "source obligations pass."
            ),
        )
    elif closure_collision_lane.get("closed_by_current_evidence"):
        admission = closure_collision_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "structured_closure_collision_locator",
            101,
            str(
                closure_collision_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "structured_closure_collision_locator"
                ]["decisive_test"]
            ),
            (
                f"{closure_collision_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} closure-source "
                "obligations pass."
            ),
        )
    elif factor_line_lane.get("closed_by_current_evidence"):
        admission = factor_line_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "scalar_blind_fixed_sum_source_generator",
            100,
            str(
                factor_line_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "scalar_blind_fixed_sum_source_generator"
                ]["decisive_test"]
            ),
            (
                f"{factor_line_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} source obligations pass."
            ),
        )
    elif presurface_lane.get("closed_by_current_evidence"):
        admission = presurface_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "presurface_full_charge_target_transfer",
            99,
            str(
                presurface_lane.get("next_action")
                or CRITICAL_EXPERIMENTS[
                    "presurface_full_charge_target_transfer"
                ]["decisive_test"]
            ),
            (
                f"{presurface_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} full-source "
                "obligations pass."
            ),
        )
    elif slice_lane.get("closed_by_current_evidence"):
        admission = slice_lane.get("admission") or {}
        add_focus_candidate(
            candidates,
            "public_slice_source_corpus",
            97,
            str(slice_lane.get("next_action") or CRITICAL_EXPERIMENTS[
                "public_slice_source_corpus"
            ]["decisive_test"]),
            (
                f"{slice_lane.get('classification')}: "
                f"{admission.get('passed_obligation_count', 0)}/"
                f"{admission.get('obligation_count', 0)} source obligations pass."
            ),
        )
    summation_ffe_cells = sum(row["has_summation_ffe_evidence"] for row in natural)
    if summation_ffe_cells:
        replay_ready = sum(
            row["summation_ffe_readiness"]["replay_readiness"]
            == "exact_replay_inputs_ready"
            for row in natural
            if row["has_summation_ffe_evidence"]
        )
        lane_admitted = sum(
            row["summation_ffe_new_factor_row_discovery_gate"]["lane_admitted"]
            for row in natural
            if row["has_summation_ffe_evidence"]
        )
        add_focus_candidate(
            candidates,
            "summation_ffe_new_factor_row_cost_gate",
            96 if lane_admitted < summation_ffe_cells else 82,
            (
                "Require an exact, scalar-blind new-factor-row source enumerator "
                "below direct pair-complement cost before opening this lane."
            ),
            (
                f"{summation_ffe_cells} full cells include summation/FFE evidence markers, "
                f"{replay_ready} have exact replay payloads, and "
                f"{lane_admitted} pass the R68 new-factor-row cost gate."
            ),
        )
    target_cells = [cell for cell in full_cells if cell["policy"] == "two_map_union"]
    oracle_help = sum(
        float(cell["routing_gap"]["oracle_rank_headroom"] or 0.0) > 0.0
        for cell in target_cells
    )
    fixed_help = sum(
        float(cell["routing_gap"]["fixed_rank_headroom"] or 0.0) > 0.0
        for cell in target_cells
    )
    if oracle_help:
        add_focus_candidate(
            candidates,
            "routing_intervention_generalization",
            80,
            "Freeze one routing intervention on development cells and require transfer to every prospective curve.",
            f"Posthoc routing improves rank in {oracle_help} cells; the fixed route improves {fixed_help}.",
        )
    specificity = routing_specificity_summary(full_cells)
    if fixed_help and not specificity["matched_curve_count"]:
        add_focus_candidate(
            candidates,
            "routing_specificity_control",
            79,
            "Run the fixed route on every matched hash policy before interpreting a coordinate-policy gain.",
            "The fixed route improves target-policy rank, but no exact matched hash comparison is available.",
        )
    missing_logs = sum(row["full_rank"] and not row["verified_factor_logs"] for row in natural)
    if missing_logs:
        add_focus_candidate(
            candidates,
            "rank_to_verified_log_probe",
            75,
            "Audit anchored elimination, RHS signs, and every recovered factor logarithm.",
            f"{missing_logs} full-rank cells do not produce verified factor logs.",
        )
    descent_failures = sum(
        row["target_descent_count"] == 0
        or row["target_descent_successes"] != row["target_descent_count"]
        for row in natural
    )
    if descent_failures:
        add_focus_candidate(
            candidates,
            "factor_logs_to_target_descent_probe",
            70,
            "Keep factor-log collection frozen and test separate target descent on unseen public targets.",
            f"{descent_failures} full cells omit or fail the log-to-descent transition.",
        )
    above_rho = sum(
        row["verified_factor_logs"] and not row["below_rho_proxy"] for row in natural
    )
    if above_rho:
        add_focus_candidate(
            candidates,
            "end_to_end_cost_reduction",
            60,
            "Reduce charged source generation, retained residual traffic, or matrix work without changing the frozen evidence set.",
            f"{above_rho} verified-log cells remain above the rho proxy.",
        )
    shoup = shoup_pressure_summary(full_cells)
    summation_ffe_cells = sum(row["has_summation_ffe_evidence"] for row in natural)
    if (
        above_rho
        and summation_ffe_cells == 0
        and shoup["status"] == "finite_scale_cost_gate_failed"
    ):
        add_focus_candidate(
            candidates,
            "summation_ffe_new_factor_row_cost_gate",
            62,
            "Test one scalar-blind summation/FFE new-factor-row enumerator against direct pair-complement cost.",
            (
                "The exact multiscale collision route is functionally complete but every "
                "measured single-target point remains above rho; no admissible "
                "summation/FFE source has been measured."
            ),
        )
    if not shoup["meets_shoup_pressure_gate"]:
        shoup_priority = 94 if shoup["status"] == "fit_gate_failed" else 58
        add_focus_candidate(
            candidates,
            "shoup_pressure_scaling_probe",
            shoup_priority,
            "Fit charged natural-route operations versus group order and check explicit Shoup-pressure residuals.",
            f"{shoup['status']}: {shoup['evidence']}.",
        )
    if not candidates:
        add_focus_candidate(
            candidates,
            "independent_claim_audit",
            50,
            "Rebuild claim-critical collisions, matrices, factor logs, target descents, and total cost independently.",
            "No earlier stage bottleneck remains in the natural route.",
        )

    ranked = sorted(
        candidates.values(),
        key=lambda row: (-int(row["priority_score"]), row["id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["experiment"] = CRITICAL_EXPERIMENTS[row["id"]]
    return ranked


def focus_plan(
    full_cells: list[dict[str, Any]],
    budget: int,
    frontier_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = ranked_focus_candidates(full_cells, frontier_status)
    selected = ranked[: max(1, budget)]
    deferred = [
        {
            "id": row["id"],
            "rank": row["rank"],
            "priority_score": row["priority_score"],
            "reason": "outside_current_critical_experiment_budget",
        }
        for row in ranked[len(selected):]
    ]
    total_priority = sum(int(row["priority_score"]) for row in ranked)
    selected_priority = sum(int(row["priority_score"]) for row in selected)
    return {
        "budget": max(1, budget),
        "candidate_count": len(ranked),
        "selected": selected,
        "deferred": deferred,
        "selected_count": len(selected),
        "deferred_count": len(deferred),
        "focus_accounting": {
            "selected_count": len(selected),
            "deferred_count": len(deferred),
            "selected_priority_mass_fraction": ratio(selected_priority, total_priority),
            "priority_gap_after_selection": (
                int(selected[-1]["priority_score"]) - int(ranked[len(selected)]["priority_score"])
                if selected and len(selected) < len(ranked)
                else None
            ),
        },
        "selection_policy": (
            "Run selected experiments in rank order. Do not open a deferred branch until a selected "
            "experiment is falsified, completed, or made impossible by an exactness failure."
        ),
    }


def guidance_compliance(
    steering: dict[str, Any],
    resolutions: list[dict[str, Any]],
) -> dict[str, Any]:
    required_ids = set(METHODOLOGY["tweet_guidance"]["required_resolution_ids"])
    observed_ids = {row["id"] for row in resolutions}
    missing_ids = sorted(required_ids - observed_ids)
    selected_candidates = steering["selected"]
    selected_full_spec = all(
        row.get("id")
        and row.get("evidence")
        and row.get("experiment")
        and row["experiment"].get("hypothesis")
        and row["experiment"].get("decisive_test")
        and row["experiment"].get("falsifier")
        and row["experiment"].get("required_artifacts")
        for row in selected_candidates
    )
    deferred = steering["deferred"]
    deferred_reasoned = all(
        item.get("reason") == "outside_current_critical_experiment_budget"
        for item in deferred
    )
    candidate_count = steering["candidate_count"]
    budget = steering["budget"]
    selected_count = steering["selected_count"]
    deferred_count = steering["deferred_count"]
    bounded = selected_count <= budget
    peripheral_ok = deferred_count == max(0, candidate_count - budget) and deferred_reasoned
    selected_mass_complete = selected_count == min(candidate_count, budget)
    non_blocking_ok = not missing_ids
    operator_interrupt_alignment = all(
        (
            not row["blocks_promotion"]
            and row["uncertainty_class"] == "non_blocking"
            and row["operator_interrupt_required"] is False
        )
        or (
            row["blocks_promotion"]
            and row["uncertainty_class"] == "promotion_blocking"
            and row["operator_interrupt_required"] is True
        )
        for row in resolutions
    )
    overall = (
        bounded
        and peripheral_ok
        and selected_mass_complete
        and non_blocking_ok
        and selected_full_spec
        and operator_interrupt_alignment
    )
    return {
        "bounded_critical_set_enforced": bounded,
        "peripheral_scope_defer_enforced": peripheral_ok,
        "focus_selection_matches_budget": selected_mass_complete,
        "non_blocking_ambiguity_resolutions_recorded": non_blocking_ok,
        "operator_interrupt_alignment": operator_interrupt_alignment,
        "selected_candidates_have_full_spec": selected_full_spec,
        "missing_required_resolution_ids": missing_ids,
        "overall_compliant": overall,
    }


def focus_queue(
    full_cells: list[dict[str, Any]],
    budget: int,
    frontier_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return focus_plan(full_cells, budget, frontier_status)["selected"]


def experiment_lineage(source_sha256: str, steering: dict[str, Any]) -> dict[str, Any]:
    """Build a logical baseline/variant tree without claiming run materialization."""
    root_id = f"baseline:{source_sha256[:16]}"
    nodes: list[dict[str, Any]] = [
        {
            "id": root_id,
            "parent_id": None,
            "node_type": "immutable_source_baseline",
            "status": "observed",
            "source_sha256": source_sha256,
            "git_branch_created": False,
            "run_materialized": False,
        }
    ]
    for status, rows in (
        ("queued", steering["selected"]),
        ("deferred", steering["deferred"]),
    ):
        for row in rows:
            experiment = row.get("experiment") or CRITICAL_EXPERIMENTS[row["id"]]
            nodes.append(
                {
                    "id": f"experiment:{row['id']}",
                    "parent_id": root_id,
                    "node_type": "diagnostic_variant",
                    "status": status,
                    "rank": row["rank"],
                    "priority_score": row["priority_score"],
                    "hypothesis": experiment["hypothesis"],
                    "decisive_test": experiment["decisive_test"],
                    "falsifier": experiment["falsifier"],
                    "required_artifacts": list(experiment["required_artifacts"]),
                    "evidence": list(row.get("evidence", [])),
                    "git_branch_created": False,
                    "run_materialized": False,
                }
            )
    return {
        "schema": "ecdlp.p1436_autoresearch_experiment_lineage.v1",
        "source": {
            "post_url": TWEET_POST_URL,
            "openresearch_cli_url": METHODOLOGY["openresearch_cli_url"],
        },
        "lineage_mode": "logical_plan_only",
        "baseline_root_id": root_id,
        "node_count": len(nodes),
        "queued_count": len(steering["selected"]),
        "deferred_count": len(steering["deferred"]),
        "nodes": nodes,
        "materialization_boundary": (
            "This report records parentage and evidence contracts only. A node becomes "
            "executed evidence only after a separate branch/run receipt binds its artifacts "
            "to the immutable source hash."
        ),
    }


def next_action(steering: dict[str, Any]) -> dict[str, Any]:
    """Return the single next action from the ranked critical experiment set."""
    if not steering.get("selected"):
        return {
            "position": 1,
            "focus_id": "independent_claim_audit",
            "action": "No ranked critical candidates were available; rerun an independent claim audit with exact source binding.",
            "evidence": ["No ranked critical candidate selected."],
            "decisive_test": "Rebuild claim-critical artifacts against the bound source hash.",
            "falsifier": "Any claim-critical item or source hash differs from the bound source.",
            "required_artifacts": ["independent_audit.json", "audit_transcript.md"],
        }
    row = steering["selected"][0]
    experiment = row.get("experiment") or {}
    return {
        "position": 1,
        "focus_id": row["id"],
        "action": row["action"],
        "evidence": row.get("evidence", []),
        "decisive_test": experiment.get("decisive_test", ""),
        "falsifier": experiment.get("falsifier", ""),
        "required_artifacts": experiment.get("required_artifacts", []),
    }


def ambiguity_resolutions(
    full_cells: list[dict[str, Any]],
    audit_status: dict[str, Any],
    fixed_config_name: str,
    source_breakthrough: bool,
) -> list[dict[str, Any]]:
    missing_fixed = sum(
        not cell["fixed_routing_intervention"]["available"] for cell in full_cells
    )
    missing_descents = sum(
        cell["natural_route"]["target_descent_count"] == 0 for cell in full_cells
    )
    resolutions = [
        {
            "id": "self_patching_fidelity",
            "observed": "P1436 interventions rerun configurations rather than replacing one identical intermediate state.",
            "resolution": "Label every delta a matched configuration ablation and prohibit causal self-patching language.",
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "paper_headroom_scope",
            "observed": "The source paper reports an LLM-specific fixed-versus-oracle recovery range.",
            "resolution": "Recompute headroom from P1436 rank fractions; never import the paper's numeric range as an ECDLP gate.",
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "posthoc_oracle_scope",
            "observed": "Multiple frozen configurations may tie or outperform the natural route.",
            "resolution": "Break score ties by configuration name and keep every oracle choice diagnostic-only.",
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "synthetic_null_scope",
            "observed": "Synthetic residual labels have no elliptic-curve source equation.",
            "resolution": "Use the synthetic stream for occupancy only; assign no rows, logs, or descent credit.",
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "fixed_route_availability",
            "observed": f"{missing_fixed} full cells omit `{fixed_config_name}`.",
            "resolution": "Fall back to the natural route and mark the fixed intervention unavailable.",
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "experiment_lineage_materialization",
            "observed": (
                "The focus report can define baseline/variant parentage but does not "
                "materialize OpenResearch git branches or experiment runs."
            ),
            "resolution": (
                "Label lineage logical-plan-only until a separate branch/run receipt "
                "binds artifacts to the source hash."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "ffe_information_conservation_scope",
            "observed": (
                "R68 proves product-quotient rows are sums of fixed-sum factor-row "
                "differences and add no separate factor-log information."
            ),
            "resolution": (
                "Assign zero information credit to product/summation quotient rows and "
                "require a scalar-blind new-factor-row source below direct pair-complement cost."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "constructive_closure_collision_scope",
            "observed": (
                "R69 proves that a constructive relation introducing one fresh "
                "residual preserves unresolved-log nullity; only an independent "
                "closure collision can reduce it."
            ),
            "resolution": (
                "Assign zero rank-reduction credit to fresh residual rows and "
                "require a prospective coordinate-defined collision locator below "
                "both pair materialization and rho."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "multiplicative_x_s3_scope",
            "observed": (
                "R70 finds no transferred independent collision-rank excess for "
                "the frozen multiplicative-subgroup x prefixes, despite exact S3 "
                "replay on four prime-field families."
            ),
            "resolution": (
                "Close that frozen prefix candidate and do not treat FFT-compatible "
                "coordinates as a source locator without a prospective density "
                "theorem and explicit sub-pair algorithm."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "s4_centered_carry_scope",
            "observed": (
                "R71 finds full mode rank for the canonical target-specific S4 "
                "centered remainder and carry on every frozen B=3..8 prefix, "
                "across four prime-field families and two target types."
            ),
            "resolution": (
                "Close only the named k=1 S4 canonical-carry CP precursor. "
                "Preserve S6-specific, noncanonical, and non-CP interfaces, and "
                "require an exact branch-complete source router with direct costs."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "s6_centered_carry_scope",
            "observed": (
                "R72 certifies raw S6 mode rank 17 but centered-remainder and "
                "carry mode rank 18 for both natural coefficient lifts, every "
                "mode, four large prime-order curves, and two target types."
            ),
            "resolution": (
                "Close those two explicit S6 carry lifts. Route work to R9/R10's "
                "non-CP exact trace contraction, preserving other lifts only "
                "when accompanied by a new cancellation and source theorem."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "resultant_valuation_trace_scope",
            "observed": (
                "R73 exactly realizes R10's rank-two multiplicative-convolution "
                "coefficient as a product-resultant root valuation, including "
                "zero strata, duplicate multiplicity, dyadic containment, and "
                "one occurrence-labelled source."
            ),
            "resolution": (
                "Close only the frozen resultant-valuation grammar: its expanded "
                "state is B^4, specialized queries cost B^2 each, and its actual "
                "S6 extension emits B^3 triple occurrences. Preserve structured "
                "quotient-algebra transducers and unrestricted circuits."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "s6_residual_decision_diagram_scope",
            "observed": (
                "R74 finds B^3 distinct target-independent squarefree S4 "
                "residual keys on every frozen standardized-curve prefix, while "
                "the outcome-aware Boolean diagram has at most two states."
            ),
            "resolution": (
                "Close only squarefree residual-radical memoization. Assign zero "
                "constructive credit to the tiny oracle diagram because building "
                "its signatures presupposes relation incidences; route work to a "
                "support-adaptive constructor with blind zero certificates."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "s6_iterated_norm_support_scope",
            "observed": (
                "R75 finds full (4B+1)^3 coefficient support after the first "
                "unary S4 norm on all frozen curves and prefixes; for B>4 the "
                "suffix modulus cannot yet reduce the Z degree."
            ),
            "resolution": (
                "Close only expanded iterated-norm/remainder representations. "
                "Preserve a transposed scalar functional that never materializes "
                "the coefficient cube or B^3 values and still returns exact "
                "multiplicity, zero certificates, children, and one source."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "s6_subset_incidence_mobius_scope",
            "observed": (
                "R76 replaces the literal B^5 endpoint-incidence grid by an "
                "exact Mobius subset-histogram count with blind zero, a "
                "verified source, duplicate multiplicity, and dyadic children, "
                "but uses B^3 prefix state and B^2 target work."
            ),
            "resolution": (
                "Close only the explicit subset-histogram representation and "
                "do not promote its exact finite counts. Route the next bounded "
                "test to a target-translated frequency oracle that avoids both "
                "prefix-triple and target-pair enumeration under direct caps."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "target_translated_frequency_orbit_scope",
            "observed": (
                "R77 grants scalar labels and exact characters on four toy "
                "prime-order groups, yet every one of 40 multiplicative/hash "
                "pair kernels has full ambient translation-orbit rank."
            ),
            "resolution": (
                "Close only universal exact linear shift-equivariant frequency "
                "sketches. Keep nonlinear target-specialized nested resultants "
                "open, and require them to replay R76 multiplicity, zero, source, "
                "children, and direct costs without character coordinates."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "actual_s6_fermat_tensor_train_scope",
            "observed": (
                "R78 finds exact actual-S6 TT ranks [B,B^2,B^2,B] on "
                "all 32 standardized-curve instances, forcing a B^5 center "
                "core before the Fermat projector reaches its tiny final mask."
            ),
            "resolution": (
                "Close only the value-first exact TT and binary-Hadamard "
                "Fermat grammar. Keep scalar-only straight-line nested norms "
                "open, but expose and charge every node and require R76 exact "
                "count, source, multiplicity, blind zero, and child replay."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "scalar_only_nested_norm_slp_scope",
            "observed": (
                "R79 uses only scalar nodes and six live streaming scalars, "
                "but blind zero certification and exact positive counting "
                "still evaluate B^5 actual-S6 leaves; caching uses B^5 state."
            ),
            "resolution": (
                "Close only the explicit scalar-leaf product/norm SLP. Route "
                "the remaining possibility to a batched norm-node compiler "
                "whose resultants, subproducts, remainders, transposition, "
                "source, and child operations all receive field-level receipts."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "batched_nested_norm_node_compiler_scope",
            "observed": (
                "R80 compiles the B^5 scalar leaves into exact prefix/suffix "
                "product trees and gcd-guided source return with B^3 dominant "
                "work, but uses B^3 setup state, B^3 fresh work, and B^2 "
                "fresh workspace."
            ),
            "resolution": (
                "Close only the standard materialized S4 subproduct/gcd "
                "compiler. Route the residual to one scalar-blind structured "
                "factor-base geometry with subcap triple-endpoint compression, "
                "prospective density, matched rank controls, factor logs, and "
                "identical target descent."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "full_multiplicative_x_coset_endpoint_scope",
            "observed": (
                "R81 freezes 16 complete multiplicative x-coordinate cosets "
                "on four relation-scale prime-order toys. Every unordered "
                "triple has a distinct endpoint-set key, candidate root "
                "support has slope 3.0175, and matched controls have slope "
                "2.9825."
            ),
            "resolution": (
                "Close only complete one-dimensional multiplicative-x cosets "
                "with the canonical lift mask and cofactor map. Route the "
                "remaining test to a compact divisor or straight-line "
                "factor-base description whose S4 endpoint compiler, source "
                "query, density, rank, logs, and identical descent are all "
                "prospectively charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "cartesian_sum_compact_divisor_scope",
            "observed": (
                "R82 freezes eight scalar-blind addition-pushforward factor "
                "bases. The exact 3F=3A+3C compiler uses B^(9/5) state and "
                "B^(6/5) query work, but the full 5F=5A+5C source has a "
                "B^(5/2)=N^(1/2) generic collision baseline."
            ),
            "resolution": (
                "Preserve the passing local S4 compiler, close only the "
                "Cartesian addition-pushforward full pipeline under explicit "
                "equality joins and quotient-free generic collisions, and "
                "route the residual to one addition-compatible field "
                "filtration with exact FFE or summation-polynomial sources."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_coordinate_filtration_scope",
            "observed": (
                "R83 tests 72 frozen x-, y-, and encoding-hash bucket "
                "profiles on one canonical 5A+5C source per attained R82 "
                "target. No single bucket or target offset is complete; "
                "replaying every offset restores the discarded exponent."
            ),
            "resolution": (
                "Close only explicit coordinate buckets and target-coupled "
                "offset replay. Preserve target-forced algebraic selectors "
                "and route the next test to a marked resultant with exact "
                "containment and one jointly coupled source section."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_marked_resultant_source_section_scope",
            "observed": (
                "R84 builds exact scalar-blind Fp2 side polynomials and "
                "packed source interpolants on eight R82 instances. Every "
                "finite attained target has a jointly coupled source, but "
                "the smaller explicit side has prospective B^(2.4) "
                "coefficient size and a fresh translation has the same cost."
            ),
            "resolution": (
                "Close only explicit dense endpoint-polynomial and packed-"
                "interpolant source sections. Preserve P1510's output-"
                "sensitive positive control and route the residual to one "
                "target-uniform circuit before coefficient or provenance-"
                "leaf emission."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_target_uniform_precoefficient_circuit_scope",
            "observed": (
                "R85 proves that any fixed label map with exact action for "
                "all translations has subgroup-coset fibers, hence is "
                "injective or constant on a prime-order group. It also "
                "reconciles the P1512-P1514 standard circuit negatives."
            ),
            "resolution": (
                "Close fixed exact target-equivariant quotients and the "
                "bound standard circuit grammars only. Route the surviving "
                "target-specialized mechanism to P1514's explicit sparse "
                "multihomogeneous moment-constructor exception."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_sparse_multihomogeneous_moment_scope",
            "observed": (
                "R86 gives a compact five-colour description and exact "
                "supplied first-norm-jet source recovery on eight R82 "
                "instances. Standard constructors still materialize the "
                "B^5 coloured quotient or the B^3 five-C support, and the "
                "finite controls consume verifier DLP labels."
            ),
            "resolution": (
                "Close supplied-jet decoding and the charged standard "
                "multigraded constructors only. Preserve the compositional "
                "exception and route it to one jet-preserving A+C "
                "addition-pushforward intertwiner with full source and "
                "exceptional-chart replay."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_jet_pushforward_scope",
            "observed": (
                "R87 proves that even a full marked first norm jet at the "
                "final target is not a universal addition-pushforward state. "
                "Exact standard composition uses shifted jets, while the "
                "translated remainder orbit has full B^2 dimension and the "
                "five-C characteristic polynomial has B^3 degree."
            ),
            "resolution": (
                "Close target-local jet propagation and explicit translated "
                "remainders only. Preserve the fixed-marker and implicit "
                "scalar exceptions, and route one black-box translated "
                "resultant/gcd source localizer with every representation "
                "and exceptional chart charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_black_box_resultant_scope",
            "observed": (
                "R88 shows that conditional binary source localization is "
                "logarithmic, but the tested zero test reads B^3 C state. "
                "The accepted quotient has full B^2 scalar-Krylov rank, and "
                "the optimistic block-width setup/query intervals are "
                "disjoint."
            ),
            "resolution": (
                "Close materialized half-gcd and the frozen scalar/block "
                "Krylov grammar only. Preserve non-Krylov arithmetic "
                "circuits and route the residual to one coefficient-free "
                "fixed-marker scalar resultant recurrence with exact "
                "multiplicity and projective source replay."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_fixed_marker_scalar_recurrence_scope",
            "observed": (
                "R89 gives equal-size five-by-two decks with the same nonzero "
                "fixed-marker seven-scalar jet at one target but different "
                "translated jets, including simple versus multiple branches. "
                "The frozen translated norm and marker channels have full "
                "B^2 quotient rank, while explicit slot support reaches "
                "C^4=B^(2.4)."
            ),
            "resolution": (
                "Close target-local fixed-marker jets and explicit shift or "
                "slot-support recurrences only. Preserve target-independent "
                "nonlocal nonlinear states, and require one frozen deck-update "
                "and translation law below both caps with exact source and "
                "exceptional-branch replay."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_nonlocal_moment_hankel_translation_scope",
            "observed": (
                "R90 gives exact target-independent nonlinear deck updates, "
                "exact target translation, and exact full-state marker source "
                "recovery. On distinct radix supports of sizes 32, 243, and "
                "1024, the norm and all five marker channels have Hankel "
                "complexity equal to the full C^5 endpoint count."
            ),
            "resolution": (
                "Close only exponential-moment, Newton, and Hankel/Padé "
                "translation states, whose exact order is C^5=B^3. Preserve "
                "non-moment source-reporting indices and route one unequal-"
                "list subfunction-inversion construction with setup, query, "
                "reporting, memory, and exceptional branches fully charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_unequal_list_subfunction_inversion_scope",
            "observed": (
                "R91 applies the bound 2026 unequal-list theorem to the "
                "B^2/B^3 endpoint split and all eleven ten-deck partitions. "
                "The intended online-cap point needs B^(19/4) space; the "
                "best partition still needs B^(22/5), and the explicit "
                "large-list auxiliary is over cap. Integer source reporting "
                "is exact, but residue maps do not transfer to public "
                "encodings of a generic prime-order group."
            ),
            "resolution": (
                "Close direct applications of the bound unequal-list and "
                "balanced kSUM theorems to explicit endpoint lists only. "
                "Preserve a compact elliptic subfunction decomposition and "
                "require public MAP1, MAP2, f_d, and TR circuits acting "
                "directly on D_A,D_C with all costs and projective branches "
                "charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_compact_elliptic_subfunction_map_scope",
            "observed": (
                "R92 gives exact public MAP1, MAP2, f_d, and endpoint TR "
                "semantics for four rational maps on a fully enumerated "
                "prime-order projective curve. Endpoint coverage forces "
                "D*L at least B^5; the bound theorem's charged advice has "
                "minimum B^(10/3), or B^(35/8) under the online cap. An "
                "explicit exact source translator costs B^5, while compact "
                "5A+5C source TR remains absent."
            ),
            "resolution": (
                "Close direct endpoint partitions with D independently "
                "preprocessed generic subfunctions only. Preserve "
                "target-dependent overlapping incidence correspondences "
                "with a jointly compressed shared operator, and require "
                "exact projective source unranking with all overlap, "
                "elimination, state, and fresh-work costs charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_shared_semilinear_incidence_scope",
            "observed": (
                "R93 factors the quadratic S3 resultant through an exact "
                "rank-six Veronese operator and returns signed affine S4 "
                "sources with zero predicate mismatches. The raw resultant "
                "matrix remains rank six, but the zero-incidence matrix "
                "reaches full rank 78. At the full R84 split, explicit "
                "source-feature rows still cost B^(12/5) and B^(13/5)."
            ),
            "resolution": (
                "Close materialized per-source Veronese rows and low-rank "
                "value-kernel arguments only. Preserve an implicit Cartesian "
                "algebraic range index that acts before feature emission, "
                "and require exact coupled source return with all projective "
                "and nonreduced branches charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_implicit_veronese_hyperplane_index_scope",
            "observed": (
                "R94 identifies the exact query as the R9 Fermat projector "
                "1-H^(p-1), verifies direct dyadic affine source recovery, "
                "and finds projector ranks 7, 18, 29, 53, and 78. "
                "Materialization, direct scanning, and the bound all-output "
                "multipoint route retain the B^(12/5) smaller-side cost."
            ),
            "resolution": (
                "Close those three standard routes only. Full finite "
                "projector rank does not refute a nonlinear aggregate trace, "
                "so route one exact cap-sized projector-count recurrence "
                "with dyadic coupled-source and complete exceptional-branch "
                "replay; withhold every rank, log, descent, and Shoup claim."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_aggregate_veronese_projector_scope",
            "observed": (
                "R95 constructs the exact coefficient-moment contraction "
                "for Res^(p-1), verifies count and dyadic toy-source "
                "semantics, and finds full coefficient-pairing rank at "
                "p=3,5,7,11,13,17,19,29. After all Veronese relations, "
                "the canonical state still has p(2p-1)=Theta(B^10) "
                "coordinates."
            ),
            "resolution": (
                "Close only the explicit all-monomial moment contraction. "
                "Treat the finite full-rank sweep as diagnostic rather than "
                "an asymptotic lower bound, and route one modular Frobenius "
                "trace with explicit state transitions, integer lifting, "
                "dyadic source return, and all costs and branches charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_modular_frobenius_trace_scope",
            "observed": (
                "R96 verifies Tr_A(1-M_h^(p-1)) on reduced, duplicate, "
                "blind, and dyadic F_11 controls. Frobenius is the identity "
                "on the reduced split quotient; on a duplicate nonreduced "
                "quotient its rank drops from five to four and loses the "
                "nilpotent occurrence direction. The standard quotient "
                "basis costs B^(12/5) and explicit matrices B^(24/5)."
            ),
            "resolution": (
                "Close only explicit quotient bases, matrices, and dyadic "
                "range quotients. Preserve a factored transposed trace that "
                "never emits those bodies, but require an explicit adjoint "
                "identity, integer lift, range restriction, multiplicity-"
                "complete source return, and full cost ledger."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_factored_transposed_projector_scope",
            "observed": (
                "R97 verifies exact blind, unique-zero, two-zero, and "
                "dyadic F_101 controls. The pointwise projector Jacobian "
                "and complete dyadic adjoint family have full source rank; "
                "a product gradient localizes one zero but collapses to "
                "zero with two occurrences. Standard source values, "
                "product trees, and adjoints cost B^(12/5)."
            ),
            "resolution": (
                "Close pointwise Fermat powering, source-valued product "
                "trees, reverse adjoints, and linearized dyadic masks only. "
                "The finite full-rank sweep is not an asymptotic lower "
                "bound; route one nonlinear tensor-tower trace whose node "
                "states arise directly from compact A/C divisor circuits."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_nonlinear_tensor_tower_scope",
            "observed": (
                "R98 proves that the resultant projector on "
                "A_u=z^2+u and C_v=z^2+v is the F_p equality kernel. Its "
                "one-bond separation rank is exactly p, and a restriction "
                "to m distinct messages has rank m. Under p=Theta(B^5), "
                "the full channel costs B^5; D distinct source messages "
                "would still cost B^(12/5)."
            ),
            "resolution": (
                "Close one-bond contractions even with arbitrary nonlinear "
                "local encoders. Preserve multi-edge digitized encodings "
                "and actual-image restrictions, but require a frozen "
                "algebraic extractor, injectivity, total cut capacity, no "
                "p-size advice, and complete occurrence-source replay."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_multiedge_digitized_projector_scope",
            "observed": (
                "R99 positively verifies exact binary-radix equality with "
                "ceil(log2 p) width-two edges when digits are supplied. "
                "Every nonempty digit-fiber indicator has interpolation "
                "degree p-1. Materialized digit/fiber tables cost B^5 and "
                "sourcewise digit traffic costs B^(12/5)."
            ),
            "resolution": (
                "Credit the small-edge equality representation but close "
                "full-field tables, fiber root/coefficient tables, and "
                "per-source extraction. Route only a leaf-free aggregate "
                "digit trie compiled directly from compact A/C divisor "
                "circuits, with all merge, query, and source costs charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_succinct_aggregate_digit_trie_scope",
            "observed": (
                "R100 proves that an exact index universal over all "
                "D-subsets needs at least binomial(p,D) states and "
                "Omega(D)=B^(12/5) field words. Explicit and Patricia tries "
                "retain Theta(D) state. A structured interval family has "
                "an exact two-word summary and source reporter."
            ),
            "resolution": (
                "Close only universal arbitrary-set indices and explicit "
                "occurrence tries. Do not transfer that bound to R84's "
                "structured 3A+2C image; route an actual-image theorem that "
                "either supplies a leaf-free mergeable summary or an "
                "injective reachable high-entropy subfamily."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_actual_divisor_image_entropy_merge_scope",
            "observed": (
                "R101 constructs exact scalar-blind local count/source "
                "oracles for both actual side images. The 2A+3C oracle uses "
                "B^(9/5) state and B^(4/5) query work; the 3A+2C oracle "
                "uses B^(6/5) state and B^(6/5) query work. It does not "
                "select a common endpoint for a fresh target."
            ),
            "resolution": (
                "Credit both structured local oracles inside the direct "
                "caps, but do not credit a 5A+5C relation source. Route one "
                "two-sided implicit join requiring exact joint multiplicity "
                "and one coupled source without enumerating either side."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_two_sided_implicit_join_scope",
            "observed": (
                "R102 constructs an exact canonical 4A+1C prefix join "
                "with B^(11/5) state and B^(14/5) fresh query work. "
                "Exhausting all atom-count splits proves this is the best "
                "direct setup-eligible split. The source and group "
                "exponents are both B^5, so B^(31/20) independent pruning "
                "requires B^(31/20) repetitions for constant success."
            ),
            "resolution": (
                "Close direct meet-in-the-middle splits and extra disjoint "
                "or independent filters only. Preserve target-forced "
                "algebraic identities that every true L+R=T pair satisfies, "
                "but require compact pre-endpoint evaluation, exact "
                "multiplicity/source, and full false-positive costs."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_target_forced_algebraic_join_filter_scope",
            "observed": (
                "R103 proves the regular S3 filter has roots x(T-L) and "
                "x(T+L), preserves every true pair, and carries four signed "
                "point branches. Pointwise local-oracle composition costs "
                "B^(16/5) best; the canonical query stays B^(14/5); "
                "explicit endpoint polynomials and materialized base-field "
                "FFE factors retain B^(12/5) or larger bodies."
            ),
            "resolution": (
                "Close pointwise S3, explicit endpoint-polynomial "
                "resultants, and materialized FFE factor lists only. Route "
                "one sign-resolved target-specialized pushdown through the "
                "compact divisor circuit before endpoint or provenance-leaf "
                "emission, with field-level state and source receipts."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_compact_preendpoint_s3_ffe_pushdown_scope",
            "observed": (
                "R104 proves that fully sign-resolved S3 acceptance is the "
                "group-coefficient predicate L+R=T. Its exact reverse "
                "recurrence reproduces all eight actual histograms and 32 "
                "count/source queries, but emits B^(14/5) suffix residuals. "
                "Fixed-target singleton contexts force the same exponent "
                "for every universal mergeable nonlinear or FFE summary."
            ),
            "resolution": (
                "Close explicit residual sets and universal mergeable child "
                "summaries only. Preserve one actual-deck-specific, "
                "non-mergeable target circuit that injects T before any "
                "child summary, while requiring fully expanded gate, "
                "source-adjoint, matched-random, and exceptional receipts."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_actual_deck_nonmergeable_target_pullback_scope",
            "observed": (
                "R105 proves conditionally that the lowest homogeneous jet "
                "of one target-injected source norm gives exact fiber "
                "multiplicity and factors into one coupled marker form per "
                "source. Ten power sums recover the unordered 5A+5C source "
                "in B^(3/5); all actual simple and double fibers replay."
            ),
            "resolution": (
                "Credit the source adjoint only after a scalar circuit is "
                "supplied, and assign no cost credit to the verifier's B^5 "
                "source product. Route the remaining work solely to a "
                "compact scalar target norm/count constructor with generic "
                "multiplicity and integer-lift gates."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_scalar_target_norm_count_circuit_scope",
            "observed": (
                "R106 applies Tao's sharp prime-cyclic uncertainty theorem "
                "to the actual canonical five-multiset side histograms. "
                "Their full count spectrum has at least "
                "q-|supp(H_A)|-|supp(H_C)|+2 live modes, which is B^5; "
                "all actual and matched finite bounds retain over 99 percent "
                "of character modes."
            ),
            "resolution": (
                "Close explicit complex-character diagonalizations and "
                "equivalent mode tables only. Preserve a non-character "
                "target-injected algebraic norm/resultant circuit, while "
                "forbidding DLP coordinates and requiring every degree, "
                "marker, integer-lift, and exceptional cost receipt."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_noncharacter_algebraic_target_norm_resultant_scope",
            "observed": (
                "R107 exhausts all 34 root partitions and finds a minimum "
                "B^(13/5) explicit resultant interface. It also proves that "
                "the unrestricted balanced resultant counts source-dependent "
                "partition weights: the two actual double fibers have "
                "canonical count two but orders 27 and 43."
            ),
            "resolution": (
                "Close explicit coefficient, Sylvester/subresultant, "
                "quotient-free cofactor, and sparse Macaulay grammars only. "
                "Preserve a factored elliptic lambda-ring/Chow recurrence "
                "that computes the canonical norm directly, with no general "
                "arithmetic-circuit lower-bound claim."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_factored_elliptic_lambda_ring_chow_norm_scope",
            "observed": (
                "R108 proves the exact seven-term degree-five cycle identity "
                "and 49-term two-deck identity. Every canonical source has "
                "uniform weight 14400, but the coefficient-one identity "
                "cycle retains a B^5 body and B^(13/5) root interface."
            ),
            "resolution": (
                "Admit the canonical cycle-weight correction and reject "
                "termwise norm evaluation. Preserve only a Poincare/theta "
                "factorization of the pulled-back target section; the "
                "theorem-of-the-cube line-bundle identity alone receives no "
                "scalar-evaluator credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_poincare_theta_target_section_rank_scope",
            "observed": (
                "R109 proves that the target equality fiber has no unary or "
                "pairwise zero cylinder and that uniform signed-section "
                "separation rank is at least B^(12/5). All actual and "
                "matched balanced images are injective and sampled ranks "
                "are full."
            ),
            "resolution": (
                "Close regular pure pairwise products and uniform separated "
                "section state only. Preserve an exact rational finite-field "
                "theta-addition circuit that generates high rank implicitly; "
                "require explicit pole cancellation and contraction costs."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_theta_addition_cancellation_network_scope",
            "observed": (
                "R110 constructs an exact finite-field L(11O) alternant "
                "whose zero set is the target relation. Zero-sum position "
                "shifts remove repeated-row and affine exceptional branches "
                "with deterministic B^(6/5) target work, but one determinant "
                "still requires a supplied source."
            ),
            "resolution": (
                "Admit the exact target predicate and close analytic-only, "
                "unshifted-repeat, and uncharged exceptional-chart concerns. "
                "Preserve only a finite-deck annihilator which returns an "
                "existence bit and coupled source without the B^5 source "
                "body, B^(12/5) incidence, or global Fermat row mode."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_finite_deck_alternant_annihilator_scope",
            "observed": (
                "R111 enumerates all 135744 canonical sources on sixteen "
                "actual and matched instances. The exact raw determinant "
                "zero-mask degree equals the distinct nonzero value count, "
                "which is at least 98.4 percent of every toy source body; "
                "standard side-conditioned coefficient layouts have B^5 "
                "capacity. Already-built zero masks have rank at most two."
            ),
            "resolution": (
                "Close raw univariate and standard row- or column-conditioned "
                "annihilators only. Give no constructor credit to low rank "
                "measured after forming the mask, and preserve the gauge-"
                "normalized endpoint Query2P1 index as a representation-"
                "sensitive exception with full source-unranking charges."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_gauge_normalized_endpoint_query2p1_scope",
            "observed": (
                "R112 gives an exact complete-projective endpoint key and "
                "stores A5, C2, and C3 tables in B^2 dominant setup state. "
                "The unique canonical C2 x C3 join, fresh-target update, "
                "no-relation decision, and scan-backed dyadic recovery all "
                "retain B^3 C-side traffic on every actual and matched deck."
            ),
            "resolution": (
                "Admit typed endpoint normalization and the thin setup tables. "
                "Close direct hashing, pair materialization, and the standard "
                "IDEA-012 R3/R4 indexing/resultant grammars only. Preserve "
                "one gauge-invariant nonlinear orbit-product recurrence with "
                "explicit order, target update, and source adjoint."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_nonlinear_elliptic_orbit_product_scope",
            "observed": (
                "R113 proves that every C3 endpoint set has trivial "
                "translation stabilizer in the prime group and that its "
                "canonical order is not a fixed translation orbit. An "
                "order-one prefix product and exact product tree still "
                "generate B^3 leaves, while target-independent zero support "
                "has B^5 occurrence degree and at least 99.4 percent toy "
                "source-body support."
            ),
            "resolution": (
                "Close fixed translation orbits, standard product trees, and "
                "bounded prefix state as a work claim. Preserve only a "
                "transposed nonuniform C5 leaf generator with a charged "
                "target specialization and exact source adjoint."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "5a5c_transposed_nonuniform_c5_leaf_generator_scope",
            "observed": (
                "R114 replays exact first-product adjoints on fourteen "
                "unique-zero instances and second Hessian-vector adjoints on "
                "both double-zero instances. Standard reverse AD still "
                "requires the B^3 primal leaf trace, while canonical preleaf "
                "transposition reaches B^(13/5) typed state after one C atom."
            ),
            "resolution": (
                "Close standard product reverse AD, first/second derivative "
                "localization as a cheap-constructor claim, and canonical "
                "typed preleaf propagation only. Preserve a broader "
                "relation-arity and asymmetric factor-base exponent "
                "rebalance with all source-to-target costs charged."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": (
                "relation_arity_factor_base_transposed_interface_"
                "rebalance_scope"
            ),
            "observed": (
                "R115 solves the exact rational exponent envelope and finds "
                "m=6, alpha=1/12, beta=3/4 as the first dense fixed-arity "
                "vertex preserving both the R82 local 3F compiler and R114 "
                "first-interface cap. Conditional total cost is B^(9/4), "
                "but the explicit balanced 3F self-join remains B^(5/2)."
            ),
            "resolution": (
                "Admit the vertex as a necessary planning envelope only. "
                "Give no algorithm, relation-rank, log, descent, or Shoup "
                "credit until an implicit S7/S13 3F self-convolution returns "
                "six coupled sources without half-occurrence enumeration."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_a6_batched_c3_pair_sum_source_locator_scope",
            "observed": (
                "R116 proves that the m=6 3F self-convolution is exactly an "
                "A6 batch of C3 pair-sum queries. C3 saturates B^(9/4) setup, "
                "the B^(1/2) target batch leaves B^(3/4) average query work, "
                "and finite projective controls return exact six-factor "
                "sources. Current indexing needs B^(39/8) setup at that "
                "online point; explicit divisor batches cost B^(11/4)."
            ),
            "resolution": (
                "Admit the exact reduction and source semantics only. Close "
                "the bound integer-indexing, explicit translated-divisor, "
                "materialized resultant, and group-algebra routes at the "
                "frozen caps. Preserve one jointly transposed elliptic "
                "coefficient functional with a source adjoint."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_target_batched_c3_elliptic_transpose_scope",
            "observed": (
                "R117 proves full q-dimensional characteristic-zero "
                "translation rank for the six-A plus six-C count and an "
                "Omega(q) pole-degree boundary for one regular rational "
                "Boolean target section. Sparse finite controls retain full "
                "auxiliary-field rank. Current original-deck k=7 indexing "
                "needs B^(33/8) state at the online-compatible endpoint."
            ),
            "resolution": (
                "Close universal characteristic-zero linear shift sketches, "
                "single regular rational target sections, and the charged "
                "current k=7 routes only. Do not infer a base-field nonlinear "
                "circuit or data-structure lower bound; preserve a "
                "target-specialized value-sensitive six-C source locator "
                "with explicit branching and reverse backpointers."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_nonlinear_value_sensitive_c6_source_locator_scope",
            "observed": (
                "R118 reduces the exact six-C source query to one C atom "
                "plus a five-C membership/source query. Enumerating that "
                "atom over the A6 batch consumes B^(5/4), leaving only "
                "polylogarithmic query slack. No occurrence-materialized "
                "split table meets both caps; current k=6 indexing needs "
                "B^(33/8) state and the C2-against-C3 scan costs B^(11/4) "
                "for the batch."
            ),
            "resolution": (
                "Admit the one-atom reduction and finite source semantics "
                "only. Close occurrence-materialized split tables and "
                "represented quotient/grid/resultant routes at the frozen "
                "caps, while preserving an endpoint-compressed, "
                "output-sensitive nonlinear C5 index. Do not infer an "
                "arbitrary data-structure or arithmetic-circuit lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_output_sensitive_nonlinear_c5_source_index_scope",
            "observed": (
                "R119 proves that an iid random prime-cyclic C deck has "
                "B^(15/4+o(1)) canonical five-sum endpoint support with "
                "probability 1-o(1). All eight finite R82 hash decks have "
                "injective canonical C5 endpoint maps with exact source and "
                "empty-query replay. Explicit endpoint dictionaries, "
                "radical/source selectors, and output-linear image "
                "compilers exceed B^(9/4) setup."
            ),
            "resolution": (
                "Admit the iid support theorem and finite source semantics "
                "only. Do not transfer the theorem to every filtered deck "
                "or infer an arithmetic-circuit, cell-probe, or RAM lower "
                "bound. Preserve a sub-output target-specialized nonlinear "
                "C5 membership/source circuit with exact empty certification "
                "and reverse projective backpointers."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_suboutput_implicit_c5_character_pairing_scope",
            "observed": (
                "R120 proves base-field algebraic characters and self-Weil "
                "pairings are trivial for the required cyclic-group "
                "character. An independent torsion pairing gives an exact "
                "injective five-sum to five-product encoding, but target cost "
                "scales with k=ord_q(p). All eight R82 controls are "
                "supersingular k=2 exceptions, while four exact prime-order "
                "curves realize finite maximal degree k=q-1."
            ),
            "resolution": (
                "Admit the exact forward pairing semantics and finite source "
                "replay only. Charge extension, torsion, Miller, and final "
                "exponentiation work; give no generic credit to k=2 fixtures "
                "or unit-cost field DLP. Preserve a small-k sub-output "
                "multiplicative C5 circuit and keep pairing-unfriendly inputs "
                "as an explicit unsolved generic branch."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_small_k_multiplicative_c5_moment_torus_scope",
            "observed": (
                "R121 gives a complete norm-one Cayley chart and exact "
                "degree-five target form on all eight R82 pairing decks. "
                "Complete-homogeneous endpoint moments have BM order equal "
                "to the injective B^(15/4) C5 support. The cap-tight C3 "
                "table leaves B^(3/2) query work, while the direct kSUM and "
                "Theorem 4.1 routes are over cap or outside their stated "
                "output-universe hypothesis."
            ),
            "resolution": (
                "Admit the exact torus, moment, annihilator, and split source "
                "semantics only. Do not infer a general arithmetic-circuit "
                "or data-structure lower bound. Preserve a target-injected "
                "nonlinear torus C5 circuit outside full moments, explicit "
                "split scans, integer residue maps, and field DLP."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_explicit_split_global_rebalance_scope",
            "observed": (
                "R122 proves that every fixed-arity one-C branch using a "
                "stored C_s table and enumerated C_r complement has charged "
                "relation-collection exponent at least 11/4+beta, after "
                "density supply, the B^(9/4) setup cap, and B^beta "
                "meaningful rows. The R115/R121 C3|C2 vertex is B^(11/4) "
                "fresh work and B^(7/2) relation collection."
            ),
            "resolution": (
                "Close only explicit occurrence/output split tables and "
                "their iid-distinct support analogues across arities. Do not "
                "transfer the iid theorem to filtered decks or infer a "
                "general circuit/data-structure lower bound. Preserve a "
                "target-specialized nonoccurrence torus C5 circuit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_fourier_product_resultant_scope",
            "observed": (
                "R123 verifies exact multiplicative Fourier inversion "
                "without field DLP and the ordered P2|P3 product-resultant "
                "identity. Full inversion has B^5 modes, the ordered moment "
                "sequence has B^(15/4) BM order, a represented fresh-target "
                "P2|P3 resultant costs B^(9/4), and symbolic P5 has "
                "B^(15/4) output."
            ),
            "resolution": (
                "Admit the exact Fourier and product-resultant semantics "
                "only. Close full mode, Prony/BM, represented target "
                "resultant, symbolic P5, and full-grid grammars. Preserve a "
                "nonrepresented target-specialized circuit and do not infer "
                "a general arithmetic-circuit or data-structure lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_linear_sketch_circulant_scope",
            "observed": (
                "R124 writes C5 counts as translated C3 against C2 inner "
                "products and proves that a universal linear C3 sketch with "
                "linear exact-count decoding needs the full C2-circulant "
                "rank. For a proper nonempty binary deck at prime order, "
                "the characteristic-zero rank is q=B^5."
            ),
            "resolution": (
                "Close only universal target-independent linear "
                "measurements with linear exact-count decoding. Preserve "
                "nonlinear preprocessing specialized jointly to "
                "(u^(*2),u^(*3)), nonlinear membership-only decoding, and "
                "adaptive data structures; infer no general lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_prime_order_homomorphic_fingerprint_scope",
            "observed": (
                "R125 proves that a product-preserving fingerprint from "
                "the prime-order pairing image is trivial or injective. "
                "The trivial map fails empty targets; every nontrivial "
                "power map is a DLP-free permutation with q=B^5 image. "
                "Finite tuples retain the same kernel dichotomy."
            ),
            "resolution": (
                "Close only pure single or finite-tuples of group "
                "homomorphisms. Preserve nonhomomorphic and adaptive "
                "fingerprints with explicit product-law correction data; "
                "infer no general data-structure or circuit lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_explicit_hash_correction_support_scope",
            "observed": (
                "R126 proves that exact per-bucket product-correction lists "
                "cover C5 and therefore contain at least |C5| represented "
                "entries. Global deduplication still has |C5| keys, which "
                "is B^(15/4) under the inherited iid-support theorem. "
                "Twenty-four coordinate-hash controls replay all products "
                "and C2+C3 sources without field DLP."
            ),
            "resolution": (
                "Close only explicit per-bucket product lists and global "
                "exact product dictionaries. Preserve implicit correction "
                "circuits, adaptive probes, and nonlinear nonlisting "
                "certificates; infer no general circuit or data-structure "
                "lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_bucket_resultant_routing_tradeoff_scope",
            "observed": (
                "R127 gives optimistic one-pair resultant work "
                "B^(9/4-gamma) for H=B^gamma balanced buckets. All-pair "
                "queries cost B^(9/4+gamma), and quotient-style H-pair "
                "routing remains B^(9/4). Polylog query isolates "
                "gamma=9/4 and rho=0: singleton C3 buckets with an "
                "implicit constant-pair router."
            ),
            "resolution": (
                "Close only independent all-pair, quotient-style, symbolic, "
                "and dense represented routing grammars. Preserve shared "
                "transposed evaluation and the cap-tight singleton-C3 "
                "constant-pair router; infer no general lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_rational_selector_degree_scope",
            "observed": (
                "R128 proves that one rational C2 selector has degree at "
                "least |C5|/|C2|=B^(9/4) by counting output fibers. All "
                "eight actual canonical selector interpolants have full "
                "|C5|-1 degree. Dense represented evaluation fits setup "
                "but costs B^(9/4) per target."
            ),
            "resolution": (
                "Close only dense single rational selectors, explicit "
                "target tables, and densely represented branches. Preserve "
                "high-degree low-SLP and compact piecewise selectors because "
                "degree is not a circuit-size lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_piecewise_selector_decision_dag_scope",
            "observed": (
                "R129 reduces injective source-monomial C2 branches to an "
                "edge set hitting every square-free five-subset. Turan's "
                "theorem gives an exact B^(3/2) optimal branch count. All "
                "eight actual optimal branch covers return sources, while "
                "sequential scans cost B^(3/2) per target."
            ),
            "resolution": (
                "Close only sequential piecewise-constant branch scans and "
                "explicit B^(15/4) target routers. Preserve compact shared-"
                "predicate decision DAGs and high-degree low-SLP selectors; "
                "infer no general circuit or data-structure lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_sparse_fourier_predicate_transfer_scope",
            "observed": (
                "R130 closes complex sparse-Fourier predicates but shows "
                "that all four actual pairing fields have "
                "ord_q(characteristic)=2, outside the pinned primitive "
                "finite-field Chebotarev condition. An exact GF(2^10) "
                "five-mode polynomial has five order-11 zeros."
            ),
            "resolution": (
                "Reject unproved complex-to-finite uncertainty transfer. "
                "Preserve order-two finite-field Fourier analysis, direct "
                "non-Fourier shared predicates, and low-SLP selectors; infer "
                "no actual-field circuit or data-structure lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_consecutive_mode_predicate_scope",
            "observed": (
                "R131 applies the field-independent polynomial root bound "
                "inside all eight actual order-two controls. Thirty active "
                "color predicates have nonzero Vandermonde determinants "
                "and exact dense annihilator zero sets, while a balanced "
                "color contains B^(15/4) accepted targets."
            ),
            "resolution": (
                "Close only dense consecutive-mode predicates and explicit "
                "root product trees at B^(15/4) state and sequential query "
                "cost. Preserve lacunary high-degree low-SLP predicates and "
                "non-Fourier shared DAGs; infer no general circuit or data-"
                "structure lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_base_field_frobenius_predicate_dag_scope",
            "observed": (
                "R132 proves that univariate base-field polynomial and "
                "rational zero-test outcomes are invariant under z to "
                "z^(-1) because Frobenius is inversion. In all eight actual "
                "controls every positive C5 target has an empty inverse, "
                "and exact support annihilators need extension coefficients."
            ),
            "resolution": (
                "Close only univariate base-field zero/definedness DAGs on "
                "the exact inversion-disjoint controls. Preserve asymmetric "
                "F_(p^2) lacunary predicates, Frobenius-aware coordinate "
                "DAGs, and nonzero-value tests; assign no generic "
                "asymptotic or circuit lower-bound credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_sparse_monomial_root_bound_scope",
            "observed": (
                "R133 adapts the pinned Kelley sparse-root proof to the "
                "prime-order subgroup, excluding one through four represented "
                "extension modes for a q^(3/4) root set. A separate pure-"
                "fifth union bound excludes up to one quarter log2(q) modes "
                "only under the explicit uniform-random-deck model."
            ),
            "resolution": (
                "Admit the deterministic small-mode negative and retain the "
                "small-log result as model-bound only. Do not transfer it to "
                "the structured factor base. Preserve five-mode and larger-"
                "polylog predicates, high-expansion low-SLP predicates, and "
                "multi-predicate Frobenius-coordinate DAGs; infer no general "
                "circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_two_atom_geometric_progression_scope",
            "observed": (
                "R134 proves that two atoms from one color force six "
                "accepted C5 targets in geometric progression. Their ratio "
                "has prime order, so every at-most-six-mode evaluation "
                "matrix is Vandermonde. Twelve finite color witnesses replay "
                "the progression and source tuples exactly."
            ),
            "resolution": (
                "Close only a single represented at-most-six-mode zero or "
                "pole set on the asymptotic structured colors. Give no "
                "credit to singleton fixture gaps. Preserve seven-or-more-"
                "mode predicates, multiple-predicate DAGs, nonzero-value "
                "Frobenius branches, and high-expansion low-SLP predicates; "
                "infer no general circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_khatri_rao_kruskal_amplification_scope",
            "observed": (
                "R135 identifies the all-five-from-one-color C5 "
                "evaluation submatrix as a row-deduplicated fifth "
                "Khatri-Rao power. Its field-independent Kruskal-rank "
                "inequality recovers six deterministic modes, while a "
                "uniform-random-deck sparse-root bound amplifies to "
                "(5-o(1))*log2(q) modes only under that model."
            ),
            "resolution": (
                "Admit the ordinary finite-field rank lemma and the "
                "model-bound negative, but do not transfer random-deck "
                "Kruskal rank to the structured factor base. Preserve "
                "structured seven-plus-mode predicates, composed "
                "predicate DAGs, nonzero-value Frobenius branches, and "
                "high-expansion low-SLP predicates; infer no general "
                "circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_all_nonzero_path_product_scope",
            "observed": (
                "R136 collapses the all-nonzero path of a sparse "
                "polynomial zero-test tree to a product polynomial. A "
                "rejecting leaf forces that product to cover the positive "
                "support; an accepting leaf forces the node root union to "
                "cover the subgroup complement. Exact trees therefore "
                "need expanded product support above the R135 rank "
                "threshold or a near-linear root budget."
            ),
            "resolution": (
                "Close only zero-test DAGs whose expanded all-nonzero path "
                "product remains within the admitted rank threshold. "
                "Preserve growing-support low-SLP paths, nonzero-value "
                "Frobenius-coordinate circuits, and structured seven-plus-"
                "mode products; infer no straight-line-program, circuit, "
                "RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_binomial_node_union_depth_scope",
            "observed": (
                "R137 applies multiplicative Cauchy-Davenport to give "
                "every m-atom structured color at least 5m-4 accepted "
                "fivefold products. A prime-order binomial has at most one "
                "root, so an exact binomial zero-test tree needs "
                "B^(3/4+o(1)) all-nonzero path depth regardless of product "
                "expansion. Random support extends polynomial depth only "
                "through four-mode nodes."
            ),
            "resolution": (
                "Close all structured binomial zero-test trees and retain "
                "the three- and four-mode result as random-model-only. "
                "Preserve structured three-plus-mode nodes, five-plus-mode "
                "low-SLP circuits, and nonzero-value Frobenius-coordinate "
                "branches; infer no general SLP, circuit, RAM, or cell-"
                "probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_chebotarev_fiber_cover_scope",
            "observed": (
                "R138 proves that atom-restricted t-column full spark "
                "forces a rejecting all-nonzero path to have depth at "
                "least m/(t-1), conditionally giving B^(3/4+o(1)) for "
                "trinomials. The corrected finite-field Chebotarev source "
                "requires primitive characteristic order, while every "
                "actual norm-one family has order two; forcing primitive "
                "order costs extension degree q-1=B^(5+o(1))."
            ),
            "resolution": (
                "Admit the tuple-fiber lemma and reject only the generic "
                "primitive-order theorem transfer. Do not infer actual "
                "atom-restricted full spark from finite controls. Preserve "
                "a characteristic-specific restricted-minor proof, four-"
                "plus-mode low-SLP circuits, and nonzero-value Frobenius "
                "branches; infer no general SLP, circuit, RAM, or cell-"
                "probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_order_two_three_minor_rigidity_scope",
            "observed": (
                "R139 uses p=-1 modulo q Frobenius to prove every "
                "three-by-three prime-order Fourier minor nonsingular. "
                "Every structured atom color therefore has three-column "
                "full spark. R138's fiber cover becomes unconditional for "
                "trinomials, forcing B^(3/4+o(1)) rejecting-path depth or "
                "B^(5/2+o(1)) accepting-path depth."
            ),
            "resolution": (
                "Close structured represented-trinomial zero-test trees "
                "without a random-support assumption. Preserve four-plus-"
                "mode low-SLP circuits and nonzero-value Frobenius branches; "
                "infer no general sparse-polynomial, SLP, circuit, RAM, or "
                "cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_order_two_four_minor_claw_scope",
            "observed": (
                "R140 factors the normalized four-minor determinant as "
                "an explicit norm-one Mobius claw and finds rank-three "
                "witnesses on all twelve available actual six-point "
                "progressions. Each kernel hits exactly four checked "
                "progression points. Recovering the hidden image as an "
                "integer Fourier mode invokes a field DLP or the standard "
                "q^(1/2)=B^(5/2) known-mode collision baseline."
            ),
            "resolution": (
                "Reject universal four-column full spark, but give the "
                "finite witnesses no asymptotic selector credit. Preserve "
                "a sub-q^(9/20) structured claw, direct hidden-mode "
                "evaluation, dense four-mode atom roots, and nonzero-value "
                "Frobenius branches. Treat B^(5/2) as a generic baseline, "
                "not a universal circuit or data-structure lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_sextic_mobius_character_router_scope",
            "observed": (
                "R141 evaluates chi_z(x)=T_z(x)^q in polylogarithmic "
                "work and uses Weil cancellation to obtain logarithmic "
                "fixed-parameter inverse separation. All actual inverse "
                "controls pass. The C2-by-C3 character matrices have full "
                "row rank, multiplicative defects attain all six cosets, "
                "and Parseval forces Omega(q)=B^(5+o(1)) translated-linear "
                "character state."
            ),
            "resolution": (
                "Close the base-field inversion symmetry and translated "
                "linear convolution routes. Preserve nonlinear, "
                "nontranslation character composition and adaptive source "
                "routing; infer no general nonlinear circuit, RAM, or "
                "cell-probe lower bound. Inverse separation alone is not "
                "source selection."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_adaptive_character_decision_router_scope",
            "observed": (
                "R142 exhausts adaptive trees over the frozen nonidentity "
                "deck parameters. Five of eight controls retain an "
                "inadmissible full-signature cell; the other three need "
                "minimum leaf counts 33, 360, and 645. Every exact-tree "
                "positive and inverse-empty path replays."
            ),
            "resolution": (
                "Close direct deck-landmark character trees on the finite "
                "controls without asymptotic credit. Preserve arbitrary "
                "parameters and arithmetic or algebraic composition of "
                "labels across C2-by-C3; infer no circuit, RAM, cell-probe, "
                "or structured asymptotic lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "torus_c5_label_congruence_correction_scope",
            "observed": (
                "R143 derives the exact fixed-degree Mobius-character "
                "product defect, proves that a total label-only law on a "
                "prime-order group is constant or injective, and exhausts "
                "all deck-parameter subsets on eight controls. Two controls "
                "remain nondeterministic; the other six require the full "
                "C2-by-C3 operand table. An ideal explicit-row four-list "
                "merge has envelope max(beta,1-beta)."
            ),
            "resolution": (
                "Close fixed-size total label composition and the ideal "
                "explicit-row birthday shortcut. Preserve arithmetic "
                "circuits, nonuniform structure, compressed linear algebra, "
                "and implicit transposed summation-polynomial/FFE "
                "relation-span operators. Infer no general ECDLP lower "
                "bound from the finite controls or explicit-row model."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_weighted_fiber_marginal_log_operator_scope",
            "observed": (
                "R144 sums every ordered six-factor relation in a target "
                "fiber into one count and A/C marginal row. The exact "
                "known-RHS identity solves meaningful factor logs, and the "
                "same row recovers shifted target logs without opening a "
                "source. All eight finite aggregate matrices have full "
                "meaningful rank. The conditional precomputation envelope "
                "is B^(9/4)=N^(9/20)."
            ),
            "resolution": (
                "Admit the source-free algebraic reduction and conditional "
                "envelope only. Require a scalar-blind weighted S7/S13 count "
                "index, reusable offline/online transposed derivative state, "
                "structured rank and density, complete charts and integer "
                "lifting, and full bit costs. Baur-Strassen alone does not "
                "preserve the preprocessing/query split."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_weighted_c3_mobius_gcd_trace_scope",
            "observed": (
                "R145 realizes each weighted C6 target count exactly as a "
                "root-free quotient trace on the common divisor of the C3 "
                "support polynomial and its Mobius pullback. All eight "
                "finite controls replay positive and empty targets. The "
                "standard dense route costs B^(9/4) per target and "
                "B^(11/4)=N^(11/20) for the A6 batch."
            ),
            "resolution": (
                "Admit the exact divisor identity and close only the dense "
                "per-target coefficient route. Preserve implicit modular "
                "resultants, shared target-batched remainder trees, and "
                "other circuit or data-structure representations. Infer no "
                "lower bound, factor logs, descent, or ECDLP improvement."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_aggregate_marginal_singleton_source_equivalence_scope",
            "observed": (
                "R146 proves that exact count plus full A/C marginals "
                "recovers every atom multiplicity on a singleton canonical "
                "fiber. Between 98.18% and 100% of positive fibers are "
                "singleton on the eight controls; the independent-uniform "
                "dense M6 model predicts conditional singleton probability "
                "0.9999990355. Mean occupancy and uniform hit density are "
                "reciprocal at fixed source density."
            ),
            "resolution": (
                "Charge the count-and-marginal output as source-equivalent "
                "on singleton fibers. Retain the implicit batched index, but "
                "give no cost credit merely for omitting an explicit source "
                "row. Treat the near-one occupancy probability as random-"
                "model-only and charge any superconstant occupancy against "
                "its hit-density retry unless a targetable family is built."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_occurrence_pair_resultant_local_valuation_scope",
            "observed": (
                "R147 represents the exact ordered C6 count as a local "
                "valuation of an implicit occurrence-pair resultant while "
                "retaining repeated C3 endpoint multiplicities. The "
                "material occurrence divisor has degree B^(9/4), but "
                "independent bounded-order truncation costs B^(9/4) per "
                "target and B^(7/2)=N^(7/10) over the relation stream."
            ),
            "resolution": (
                "Admit the exact local-valuation identity and close only "
                "componentwise or direct-product use of the cited "
                "truncated-resultant algorithm. Preserve a genuinely shared "
                "transposed multi-target valuation-and-marker operator as "
                "open. Infer no computational lower bound, marginals, logs, "
                "descent, or ECDLP improvement."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_static_3sum_indexing_tradeoff_scope",
            "observed": (
                "R148 identifies occurrence-pair existence with static "
                "3SUM-indexing on a length-B^(9/4) list. Linear state plus "
                "scanning costs B^(7/2)=N^(7/10) over the target stream. "
                "The Fiat-Naor linear-state endpoint is worse, while the "
                "Dinur-Golovnev improvement requires at least B^(27/8) "
                "state and B^(9/2) preprocessing."
            ),
            "resolution": (
                "Reject an unsupported linear-state square-root query and "
                "close only the named published decision or witness data "
                "structures. Preserve an operator that exploits the compact "
                "C divisor and triple elliptic convolution before generic "
                "indexing. Require exact counts and marginals, and infer no "
                "data-structure lower bound or ECDLP improvement."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_actual_c6_shift_krylov_rank_scope",
            "observed": (
                "R149 proves that the actual fixed ordered-C6 count "
                "sequence u^(*6) and every exact C-atom marginal have all "
                "q characteristic-zero Fourier modes and full cyclic "
                "shift-Krylov dimension q=B^5. All eight actual torus and "
                "scalar multiplicity replays are exact."
            ),
            "resolution": (
                "Close constant-coefficient linear recurrences and linear "
                "shift-Krylov state at the B^(9/4) setup cap. Preserve "
                "nonlinear target-specialized compact-divisor arithmetic, "
                "adaptive data structures, and implicit summation-"
                "polynomial/FFE operators. Infer no general circuit or "
                "data-structure lower bound and no ECDLP improvement."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_rational_convolution_subalgebra_rigidity_scope",
            "observed": (
                "R150 proves that Q[C_q] has no small nontrivial unital "
                "rational algebra quotient and that the multiplication-"
                "closed subalgebra generated by the actual campaign deck "
                "has dimension at least (q-1)/|C|=B^(17/4-o(1)). All eight "
                "actual multiplier-orbit and augmentation controls are "
                "exact."
            ),
            "resolution": (
                "Close reusable characteristic-zero rational convolution "
                "quotients and multiplication-closed deck state at the "
                "B^(9/4) setup cap. Preserve bounded-depth nonhomomorphic "
                "circuits specialized only to U^6 and the fixed marker "
                "batch, adaptive data structures, and implicit summation-"
                "polynomial/FFE operators. Infer no general circuit or "
                "data-structure lower bound and no ECDLP improvement."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_matrix_free_marginal_jacobian_krylov_scope",
            "observed": (
                "R151 identifies the aggregate marginal matrix as a "
                "log-weight Jacobian. Forward directional evaluation gives "
                "Mx, reverse differentiation of a scalar contraction gives "
                "M^T lambda, and a conditional B^(5/4)-cost bidirectional "
                "operator yields B^(2+o(1)) Wiedemann linear algebra."
            ),
            "resolution": (
                "Do not require explicit marginal-row materialization, but "
                "do not infer reusable derivative state from Baur-Strassen. "
                "Require a weight-independent setup or compact reusable "
                "tangent and adjoint state, exact residual verification, "
                "structured generic-prime rank, factor logs, and identical "
                "descent before attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_geometry_only_weight_interpolation_adjoint_scope",
            "observed": (
                "R152 compiles arbitrary A/C atom weights, tangents, and "
                "transpose adjoints through public subproduct trees in "
                "B^(3/4+o(1)) state and application work. All inversions "
                "are fixed nonzero geometry constants and all eight actual "
                "A/C controls are scalar-blind and exact."
            ),
            "resolution": (
                "Admit reusable weight-independent derivative state at the "
                "leaves only. Require signed-point exactness and a "
                "weight-separable internal summation-polynomial/FFE DAG "
                "whose pivots and topology are geometry-only before "
                "granting marker, rank, log, descent, or attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_symmetric_shift_reverse_only_marginal_scope",
            "observed": (
                "R153 gives exact D_s^T=D_(-s), row-sum count recovery, "
                "and known-shift relation equations for a public known-A "
                "and inversion-closed C factor-base family. The eight "
                "finite stacked ranks are 4,0,0,0,0,0,0,0 and none is "
                "full rank."
            ),
            "resolution": (
                "Admit that one reverse-adjoint batch would supply both "
                "matrix directions and counts, eliminating a separate "
                "forward tangent. Do not credit the uniform density model. "
                "Require preregistered nonadaptive multiscale density and "
                "rank transfer together with the signed reverse circuit, "
                "exact residual, factor logs, and identical descent."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_signed_quotient_multiscale_rank_scope",
            "observed": (
                "R154 proves the public inversion-sign quotient and exact "
                "combined-rank formula. The eight actual signed ranks are "
                "1,0,0,0,0,0,0,0, while 11 of 48 preregistered synthetic "
                "prime-cyclic controls become full rank at higher finite "
                "occupancy."
            ),
            "resolution": (
                "Admit the quotient and finite rank transition only. The "
                "synthetic controls use verifier labels and do not prove "
                "random-rank concentration or hash-to-curve transfer. "
                "Require both of those results and the reverse signed FFE "
                "operator before rank, log, descent, or attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_singleton_relation_hypergraph_rank_scope",
            "observed": (
                "R155 normalizes every singleton signed relation to width "
                "at most seven and shows that singleton matrices explain "
                "all 11 finite full-rank controls. Seventeen fully covered "
                "controls remain deficient, and every row set contains "
                "exact opposite-row dependencies."
            ),
            "resolution": (
                "Admit the singleton formula, n log n independent-support "
                "coverage comparator, and logarithmic cost only. The "
                "pinned sparse-rank theorems require independent supports "
                "or a prescribed-degree random Tanner graph. Require a "
                "convolution-Tanner coupling or direct theorem, "
                "hash-to-curve transfer, and the reverse signed FFE "
                "operator before rank, log, descent, or attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_a_diversity_projective_rank_scope",
            "observed": (
                "R156 projectively removes exact opposite singleton rows "
                "and replays 108 preregistered controls. Full-rank counts "
                "are 18,22,22 by A-pair count and 6,21,35 by logarithmic "
                "factor; only one factor-eight control is deficient."
            ),
            "resolution": (
                "Admit the exact projective quotient and finite transition "
                "only. Require a direct asymptotic rank theorem controlling "
                "coverage, distinct-row supply, shared-deck dependencies, "
                "and residual nullity as A grows, plus hash-to-curve "
                "transfer and the reverse signed FFE operator before rank, "
                "log, descent, or attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_hash_to_curve_projective_rank_scope",
            "observed": (
                "R157 derives the exact cancellation-pair fiber formula "
                "and transfers singleton relation discovery to public "
                "curve-point equality. Twenty-three of 24 controls reach "
                "full rank and recover factor logs without BSGS or a DLP "
                "oracle; all recovered logs verify by scalar "
                "multiplication."
            ),
            "resolution": (
                "Admit the finite public-group transfer and factor-log "
                "recovery only. Charge explicit C6 endpoint enumeration at "
                "B^(9/2), above rho. Require asymptotic coefficient-map "
                "injectivity and rank, a reverse signed FFE replacement, "
                "and identical target descent before attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_short_relation_near_injectivity_supply_scope",
            "observed": (
                "R158 refutes global feasible-C6 coefficient-map "
                "injectivity: the expected number of collision pairs is "
                "Theta(B^4). It instead proves an O(B^(-1/2)) bad-source "
                "fraction, the exact l1-five-or-seven relation-row "
                "universe, pairwise-independent projective relation events, "
                "and concentrated Theta(B^(3/4) log B) distinct-row supply."
            ),
            "resolution": (
                "Admit near-injectivity and relation supply only. Pairwise "
                "independence yields a vanishing uncovered fraction, not "
                "zero uncovered columns or full rank, and exact transfer to "
                "the conditioned hash-to-curve sampler remains open. Charge "
                "explicit C6 endpoint enumeration at B^(9/2), above rho, "
                "and require full coverage/rank, reverse signed FFE, and "
                "identical descent before attack credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_random_diagonal_known_target_rank_scope",
            "observed": (
                "R159 replaces A6 relation events by direct known-scalar "
                "targets. It proves zero all-column coverage with "
                "B^(5/4)log(B) targets and full rank with failure at most "
                "d/q using independent diagonal shifts. All 12 public-"
                "curve controls recover verified factor logs and complete "
                "identical positive-C6 descent."
            ),
            "resolution": (
                "Admit the iid and conditioned ideal-sampler coverage/rank "
                "theorem and identical-descent reduction. The controls "
                "explicitly enumerate B^(9/2) positive-C6 endpoints and "
                "receive no attack credit. Require a public batched reverse "
                "FFE unique-source locator within B^(9/4) setup and "
                "B^(5/4) work, plus a stated hash-to-curve model, before "
                "algorithm, rho, or Shoup credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_positive_c6_generic_locator_reduction_scope",
            "observed": (
                "R160 embeds a generic DLP challenge as public factor points "
                "C_i=[a_i]G+[b_i]Q. Combined with R159 coverage and rank, an "
                "encoding-invariant locator at the requested caps would solve "
                "generic DLP in q^(9/20+o(1)); all six finite controls recover "
                "and publicly verify the embedded DLP and identical descent."
            ),
            "resolution": (
                "Admit the scoped generic-group reduction and exclude only an "
                "encoding-invariant locator under Shoup's classical generic "
                "lower bound. The finite controls use a B^(7/2) C3 scan and "
                "receive no attack credit. Preserve coordinate-aware S7, "
                "resultant, and FFE circuits, and require an explicit "
                "representation-specific operation plus full charged replay "
                "before algorithm, rho, or Shoup-improvement credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_signed_c3_divisor_translation_gcd_scope",
            "observed": (
                "R161 represents the injective signed C3 endpoint set by "
                "U,V, translates it by each target in F_p[X]/U, and "
                "recovers positive C3+C3 sources from an exact gcd. For a "
                "unique positive-C6 target the source factor has degree at "
                "most 20; all six positive, empty, and exceptional controls "
                "are exact."
            ),
            "resolution": (
                "Admit the coordinate quotient/gcd interface and constant "
                "source-factor degree. Charge independent composition over "
                "the complete target batch as B^(7/2), with no attack credit. "
                "Require a many-inner batched modular-composition and gcd "
                "source adjoint within B^(5/4) batch work before algorithm, "
                "rho, or Shoup-improvement credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_batch_inverse_transpose_modcomp_fit_scope",
            "observed": (
                "R162 batches U(u_j), detects exceptional roots, and "
                "transposes any fixed linear functional of the denominator "
                "quotients to multipoint evaluation. Published precomputation "
                "fixes the inner map and does not directly share across the "
                "target-varying phi_j maps."
            ),
            "resolution": (
                "Admit the linear layer and distinguish the strict B^(5/4) "
                "batch cap from the global B^(5/2) rho threshold. Permit a "
                "target-dependent B^(9/4) aggregate nonlinear pass, but "
                "require target-labeled source gcds, backpointers, and full "
                "charged replay before algorithm or improvement credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_aggregate_union_factor_label_recovery_scope",
            "observed": (
                "R163 couples signed x/y membership per target, aggregates "
                "the exact left-endpoint union semantics, bounds its degree "
                "by 20K=B^(3/4), and recovers target labels plus both C3 "
                "backpointers in B^2 once union roots are known."
            ),
            "resolution": (
                "Remove labels and source backpointers from the fast-constructor "
                "contract. Require only one unlabeled aggregate union factor "
                "below B^(5/2), with same-target coupling and exceptional-root "
                "handling, before invoking the admitted R163 postprocessor."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_randomized_target_divisor_norm_union_scope",
            "observed": (
                "R164 replaces the Fermat projector by a collision-safe "
                "target-label norm of a_j+r_j*b_j. It has no false negatives, "
                "bounds any regular false root by nN/p=B^(-3/2), verifies "
                "candidates exactly in expected B^2, and constructs the "
                "incidence branch in B^(9/4)."
            ),
            "resolution": (
                "Remove projector degree, target-coordinate collisions, "
                "false-positive correctness, and denominator incidents from "
                "the open contract. Require only an output-sensitive regular "
                "elliptic-translation norm below B^(5/2); charge the standard "
                "represented norm as B^(7/2) and give no unit-cost D5 oracle."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_global_randomizer_elliptic_translate_product_scope",
            "observed": (
                "R165 proves that one scalar r preserves the nN/p=B^(-3/2) "
                "bound and turns every randomized factor into a translate of "
                "f_r=U(x)+r(y-V(x)). The function has pole order 2n; tangents "
                "are global evaluations and only P=T maps to the pole."
            ),
            "resolution": (
                "Remove target-wise randomizer interpolation and unrelated "
                "per-target functions from the open contract. Require an "
                "output-sensitive product of arbitrary public translations of "
                "one fixed function modulo U. Charge explicit divisor or value "
                "expansion as B^(7/2), and do not infer a Miller scalar chain."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_kummer_x_translate_signed_verification_scope",
            "observed": (
                "R166 sets r=0 and reduces every constructor factor to "
                "U(x(T-P)). Its zero divisor is S+(-S): true signed matches "
                "are retained, opposite-sign roots are verified away, and "
                "the iid-model candidate output and verifier cost are "
                "B^(3/4) and B^2 respectively."
            ),
            "resolution": (
                "Remove the randomizer, V-dependent y residual, and full-pair "
                "signed filter from the open contract, while retaining V as "
                "the signed-divisor translation input. Require an output-sensitive "
                "arbitrary-target Kummer translate product modulo U, charge "
                "explicit divisor or value expansion as B^(7/2), retain signed "
                "verification on candidates, and require a separate proof for "
                "deterministic hash-to-curve density transfer."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_generalized_target_divisor_weil_reciprocity_swap_scope",
            "observed": (
                "R167 represents the N arbitrary targets by one degree-N "
                "principal-divisor quotient h, proves the corrected Weil-"
                "reciprocity swap on 202 selected endpoints, and preserves all "
                "140 finite candidate-zero rows by specialization. The compact "
                "h state is B^(5/4), but raw swapped evaluation is B^(9/2) and "
                "a standard represented elliptic resultant is B^(7/2)."
            ),
            "resolution": (
                "Remove target-divisor representation and reciprocity identity "
                "from the open contract. Require only a denominator-cleared "
                "SLP elliptic-resultant or tame-symbol remainder modulo U below "
                "B^(5/2), retain every auxiliary correction and the inherited "
                "signed verifier, and reject nN, n^2, or degree-Theta(nN) "
                "intermediates and unit-cost resultant or restriction oracles."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_log_derivative_elliptic_cauchy_trace_scope",
            "observed": (
                "R168 differentiates the R167 resultant into a corrected "
                "additive trace of Dh/h. Across six controls, 241 simple zero "
                "occurrences give 140 candidate poles with nonzero residues; "
                "six public P=T rational poles are detected by U,V and "
                "regularized away. Compact Dh/h state is B^(5/4), while direct, "
                "swapped, and tensor routes remain B^(7/2), B^(9/2), and B^(9/2)."
            ),
            "resolution": (
                "Remove multiplicative resultants, local multiplicity survival, "
                "and target-equality semantics from the open contract. Require "
                "only a denominator-aware transposed elliptic Cauchy trace "
                "modulo U below B^(5/2), preserving Fitting/subresultant state "
                "at candidate nonunits; reject generic-point value traces, "
                "candidate inversions, nN/n^2 tables, and tensor materialization."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_regularized_log_trace_displacement_rank_scope",
            "observed": (
                "R169 packages every R168 candidate multiplicity as the "
                "lambda-zero valuation of an exact scalar resolvent. Across six "
                "controls and a fourteen-row degree sweep, the regularized "
                "kernel and the tested ordinary diagonal x/y Sylvester and "
                "Stein displacements have full row rank. Generic lambda "
                "interpolation and full-rank generators each cost B^(9/2)."
            ),
            "resolution": (
                "Remove scalar-resolvent correctness and ordinary diagonal "
                "displacement testing from the open contract. Require only a "
                "fraction-free elliptic Fitting/subresultant denominator modulo "
                "U below B^(5/2), preserving the lambda-zero specialization; "
                "reject 2n+1 samples per endpoint, full-rank diagonal generators, "
                "candidate inversions, nN/n^2 tables, and tensor materialization."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_lambda_zero_fitting_target_norm_dedup_scope",
            "observed": (
                "R170 proves that the corrected lambda-zero Fitting support is "
                "the R167 aggregate Kummer target norm up to units. Six controls "
                "recover all 140 candidate roots; every represented target "
                "factor and aggregate is full degree and fully dense. Standard "
                "coefficient-ring work is B^(7/2), while the swapped matrix is "
                "B^(9/2)."
            ),
            "resolution": (
                "Deduplicate fraction-free lambda-zero Fitting from the open "
                "contract. Require only an SLP-streaming norm modulo U below "
                "B^(5/2); reject N dense quotient-ring elements, nN coefficient "
                "bodies, n^2 matrices, N modular compositions, candidate "
                "inversions, or uninstantiated analytic theta identities."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_balanced_miller_tree_norm_streaming_scope",
            "observed": (
                "R171 instantiates both generalized Miller functions as exact "
                "balanced line trees. Six controls verify 192 dense/tree ratios, "
                "1,222 admissible line-reciprocity rows, all 140 candidate roots, "
                "and signed telescoping to exactly the 40 original target leaves. "
                "At selected endpoints f0(-P)=0, so nodewise specialization is "
                "invalid until numerator/denominator origin factors cancel."
            ),
            "resolution": (
                "Remove balanced-tree construction, node-local norm streaming, "
                "and per-leaf correction cancellation from the open contract. "
                "Require only a nonlocal point-list-to-product elliptic batch "
                "operator below B^(5/2); reject nN pair visits, represented "
                "target factors, uncharged multipoint/resultant inputs, and "
                "individual line specialization at selected endpoints."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_target_sign_conjugate_s3_self_resultant_scope",
            "observed": (
                "R172 removes target-y dependence by multiplying the R171 "
                "locator with its target-sign conjugate. Six controls verify "
                "2,972 homogenized factors, 1,486 S3 conjugate identities, "
                "8,922 iterated-resultant rows, all 140 R171 plus roots, and one "
                "extra finite conjugate root. The represented reverse-resultant "
                "body is fully dense and full coefficient rank in these controls."
            ),
            "resolution": (
                "Remove target-sign elimination and denominator batching from "
                "the open contract. Require only the factored self-S3 resultant "
                "modulo arbitrary squarefree U below B^(5/2); reject the N^2 "
                "represented body, nN point/factor visits, local-x truncation "
                "substitution, candidate inversions, and uncharged resultant or "
                "multipoint inputs. Give finite density and root counts no "
                "asymptotic credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_s3_determinantal_transfer_noncommutativity_scope",
            "observed": (
                "R173 rewrites S3 as an exact 2 by 2 affine determinant pencil. "
                "Six controls verify 68,326 determinant rows, 1,486 separable "
                "discriminants, 1,675,890 noncommuting distinct-root pairs, all "
                "118 noncommuting target-factor pairs, and exact R172 determinant "
                "replay in both target orders. The represented matrix transfer "
                "uses 1,418 of 1,448 charged coefficient slots."
            ),
            "resolution": (
                "Close only the naive order-independent 2 by 2 transfer and its "
                "represented B^(5/2) product. Preserve a custom commutative or "
                "transposed pushforward using 4*V(X)*V_T(u) modulo arbitrary "
                "squarefree U; reject N^2 bodies, nN pair visits, target-dependent "
                "uncharged setup, and oracle-cost norms or resultants. Give finite "
                "noncommutativity and coefficient density no general lower-bound "
                "credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_confluent_signed_dual_chow_pushforward_scope",
            "observed": (
                "R174 constructs an exact signed target dual-Chow pushforward "
                "with separate off-diagonal divided-difference and geometric "
                "tangent charts. Six controls verify 68,326 signed factors, "
                "1,486 selected-Chow derivatives, 8,922 target-Chow evaluations, "
                "and all 140 R166 verified roots without the opposite-sign branch. "
                "The target and selected represented Chow forms fill 204 and "
                "4,770 coefficient slots."
            ),
            "resolution": (
                "Remove signed incidence, tangent correctness, and target-order "
                "interchange from the open contract. Require only a fused factored "
                "dual-Chow outer norm below B^(5/2). Reject N^2 or n^2 Chow "
                "coefficients, nN or n^2 query grids, N independent transforms, "
                "interpolant-derivative tangents, and uncharged multipoint or norm "
                "oracles. Give finite full density no lower-bound credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_scalar_subset_incidence_group_testing_scope",
            "observed": (
                "R175 reduces the exact R174 signed locator to scalar subset "
                "products and balanced zero-product group testing. Six controls "
                "make 358 exact queries, recover all 140 R174 roots, and query "
                "subset volume 1,238 under the proved finite bound 1,374. The "
                "conditional reusable-oracle envelope is B^(9/4), while direct "
                "factor expansion is B^(23/4)."
            ),
            "resolution": (
                "Remove scalar recovery, output-sensitive query scheduling, and "
                "compact subset descriptors from the open contract. Require only "
                "the reusable coordinate-specific scalar incidence oracle with "
                "softly O(n+N) preprocessing and O(|S|+N) query work. Reject leaf "
                "enumeration, represented Chow triangles, per-query target setup, "
                "candidate inversions, and generic locator oracles. Give finite "
                "tree counts no asymptotic credit."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_principal_target_pontryagin_resultant_scope",
            "observed": (
                "R176 rewrites the signed subset scalar as h(A*D) for the R167 "
                "principal target witness and the positive selected Pontryagin "
                "cycle. Six controls verify 8,922 pole-free pair evaluations, "
                "241 signed incidences, all 140 R175 roots, and 358 completed "
                "Weil-reciprocity identities. Standard represented pair cycles "
                "and Miller programs retain B^(9/2) state."
            ),
            "resolution": (
                "Remove principal-function signed incidence, tangent handling, "
                "tree scheduling, and ordinary represented reciprocity from the "
                "open contract. Require only a factored trilinear elliptic "
                "resultant on compact A,D,h inputs with reusable setup. Reject "
                "mn pair sums, degree-mn functions, n^2 kernels, per-node target "
                "setup, and candidate inversions. Treat the represented-degree "
                "barrier as a standard-route cost, not a circuit lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_global_marked_fitting_locator_scope",
            "observed": (
                "R177 replaces R175 subset queries by one global marked "
                "lambda-adic norm. Six controls verify kernel dimension 241, "
                "exact vanishing below lambda degree M, the restricted-X1 "
                "characteristic polynomial of degree 241, and all 140 distinct "
                "R176 roots after gcd with U. Standard pair-algebra, full-body, "
                "and explicit interpolation costs are B^(9/2), B^9, and B^6."
            ),
            "resolution": (
                "Remove subset-query scheduling and locator extraction from the "
                "open contract. Require only a fraction-free output-sensitive "
                "marked Fitting or subresultant constructor on compact U,V,h "
                "that emits M and det(AI-X_1|ker K). Reject n^2 pair or tensor "
                "state, generic n^2 matrix pencils, the full lambda/A body, M^2 "
                "marker interpolation, and candidate inversions. Treat the "
                "represented costs as standard-route negatives, not circuit "
                "lower bounds."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_marked_fitting_signed_norm_dedup_scope",
            "observed": (
                "R178 factors the R177 marker into signed-incidence threshold "
                "polynomials of global degrees 140, 71, 23, and 7. Six controls "
                "verify that the first threshold is byte-identical to both the "
                "R174 signed aggregate-norm factor and the R177 candidate gcd. "
                "The remaining layers add multiplicity but no candidate roots."
            ),
            "resolution": (
                "Close marked Fitting as a distinct ECDLP constructor lane. Return "
                "to the existing fused signed dual-Chow outer norm or equivalent "
                "nonlocal elliptic translate product modulo U. Reject nN target "
                "grids, N dense quotient elements, n^2 pair/Fitting state, the "
                "full marked determinant, and candidate inversions. Give higher "
                "multiplicity layers no ECDLP attack credit and treat represented "
                "costs as standard-route negatives, not circuit lower bounds."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_squarefree_truncated_resultant_applicability_scope",
            "observed": (
                "R179 replays all six R178 signed aggregates, verifies every "
                "selected U is squarefree, reconstructs all 202 aggregate slots "
                "from order-one CRT residues, and supplies exact local-power "
                "alias witnesses. The published x-adic algorithm specializes "
                "to B^(9/2) for either one order-n expansion or n local calls; "
                "represented target Chow reaches B^(5/2)."
            ),
            "resolution": (
                "Close only direct x-adic, per-component CRT, and represented-"
                "coefficient resultant applications. Preserve a factored "
                "arbitrary-squarefree dynamic-evaluation or transposed operator "
                "that shares work across CRT components and emits C_h mod U or "
                "G_1 in softly O(n+N). Reject n local resultants, N quotient "
                "elements, N^2 expansion, nN or n^2 grids, and candidate "
                "inversions. Infer no general resultant or circuit lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_d5_directed_evaluation_survivor_scope",
            "observed": (
                "R180 replays all six R174/R178 candidate factors from 40 "
                "individual target norms and exhaustively optimizes every finite "
                "early-split order. The 62 noncandidate components survive all "
                "target factors, forcing the literal D5 and directed-evaluation "
                "tree to B^(7/2); standard D5 half-GCD is B^(9/2), and the cited "
                "generic algebraic modular-composition exponent becomes B^3.02175."
            ),
            "resolution": (
                "Close literal target-factor streaming, successive D5 zero tests, "
                "directed panoramic evaluation of that same tree, and standard "
                "D5 half-GCD. Require a one-shot monogenic or bounded-bidegree "
                "compiler from compact U,V and the target Miller SLP, followed "
                "by charged finite-field modular composition. Reject N residue "
                "elements, nN coefficients, n^2 pair state, candidate inversions, "
                "and unit-cost composition. Infer no general circuit lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_monogenic_kernel_bidegree_scope",
            "observed": (
                "R181 constructs the canonical A_P(u)+vB_P(u) signed target "
                "kernel on nine controls, including seed 18104. Every body has "
                "3n dense slots per source point and pole order 3n; both its "
                "coefficient matrix and a fixed public n-target evaluation "
                "matrix have full source rank n. Flattened norms and complete "
                "composed resultants retain represented scale nN."
            ),
            "resolution": (
                "Close tautological post-construction H(a), explicit exact "
                "bounded-source-degree scalar kernels on the controls, the "
                "canonical 3n^2 body, flattened tensor norms, and full degree-nN "
                "composed output. Preserve compact high-degree SLPs and a "
                "gcd-equivalent output-sensitive elliptic composed resultant "
                "modulo U that may discard target-dependent units. Infer no "
                "asymptotic rank, circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_gcd_equivalent_target_subresultant_scope",
            "observed": (
                "R182 replays the exact target-divisor remainder, Newton norm, "
                "nonunit, and gcd identity on twelve controls, including held-out "
                "seed 18206. Every ordinary remainder body is dense and full "
                "target rank. A duplicate-plus-opposite-point control forces two "
                "x-injective elliptic target charts and preserves candidate roots."
            ),
            "resolution": (
                "Close standard represented nN remainder matrices, N degree-n "
                "Newton traces, per-component inverse certificates, triangular "
                "norms, and target half-GCD. Preserve an SLP-direct determinant-"
                "zero or candidate-only resultant-mod-U algorithm emitting G_1 "
                "in softly O(n+N) without represented nN state. Infer no general "
                "resultant, circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "m6_sparse_projector_prony_locator_scope",
            "observed": (
                "R183 replays all twelve R182 controls and one new strided "
                "divisor family. Target-chart Fermat-projector traces equal gcd "
                "multiplicities, and 2c weighted source moments recover each exact "
                "candidate locator by Berlekamp-Massey. A source-row swap preserves "
                "global target power sums but changes the candidate locator."
            ),
            "resolution": (
                "Admit the nested projector identity and weighted Prony output "
                "stage. Close source-blind traces as label-incomplete and charge "
                "represented projectors and A-linear target Krylov output at nN. "
                "Preserve only an SLP-direct nested-moment operator emitting 2c "
                "scalars in softly O(n+N+c). Infer no general nonlinear trace, "
                "circuit, RAM, or cell-probe lower bound."
            ),
            "blocks_promotion": False,
            "uncertainty_class": "non_blocking",
            "operator_interrupt_required": False,
        },
        {
            "id": "target_descent_absence",
            "observed": f"{missing_descents} full cells contain no target-descent trials.",
            "resolution": "Treat absent descent as untested, never as success.",
            "blocks_promotion": missing_descents > 0,
            "uncertainty_class": (
                "promotion_blocking" if missing_descents > 0 else "non_blocking"
            ),
            "operator_interrupt_required": missing_descents > 0,
        },
        {
            "id": "independent_audit_binding",
            "observed": (
                "The audit is absent or not bound to the source hash."
                if not audit_status["promotion_binding_valid"]
                else "The passing audit is bound to the exact source hash."
            ),
            "resolution": (
                "Withhold promotion."
                if not audit_status["promotion_binding_valid"]
                else "Permit this gate to pass; all arithmetic and cost gates still apply."
            ),
            "blocks_promotion": not audit_status["promotion_binding_valid"],
            "uncertainty_class": (
                "promotion_blocking"
                if not audit_status["promotion_binding_valid"]
                else "non_blocking"
            ),
            "operator_interrupt_required": not audit_status["promotion_binding_valid"],
        },
        {
            "id": "source_claim_status",
            "observed": f"Source large_prime_breakthrough={source_breakthrough}.",
            "resolution": "Never synthesize a breakthrough claim in post-processing.",
            "blocks_promotion": not source_breakthrough,
            "uncertainty_class": (
                "promotion_blocking" if not source_breakthrough else "non_blocking"
            ),
            "operator_interrupt_required": not source_breakthrough,
        },
    ]
    return resolutions


def build_report(
    payload: dict[str, Any],
    source_path: Path,
    fixed_config_name: str = DEFAULT_FIXED_CONFIG,
    focus_budget: int = 3,
    audit: dict[str, Any] | None = None,
    note_url: str = DEFAULT_NOTE_URL,
    frontier_preflights: dict[str, tuple[dict[str, Any], Path]] | None = None,
) -> dict[str, Any]:
    methodology = build_methodology(note_url)
    source_sha = sha256_file(source_path)
    cells = [
        cell_report(curve, policy, prefix, cell, fixed_config_name)
        for curve, policy, prefix, cell in iter_cells(payload)
    ]
    if not cells:
        raise ValueError("P1436 payload contains no curve/policy/prefix cells")
    full_cells = [cell for cell in cells if cell["prefix"] == "full"]
    natural = [cell["natural_route"] for cell in full_cells]
    bottlenecks = Counter(row["bottleneck"] for row in natural)
    uniform_controls = [
        cell["synthetic_uniform_occupancy_control"]
        for cell in full_cells
        if cell["synthetic_uniform_occupancy_control"].get("available")
    ]
    audit_status = audit_binding(audit, source_sha)
    frontier_status = frontier_lane_bindings(frontier_preflights)
    source_breakthrough = bool((payload.get("summary") or {}).get("large_prime_breakthrough"))
    summation_ffe_admission_complete = all(
        not row["has_summation_ffe_evidence"]
        or row["summation_ffe_new_factor_row_discovery_gate"]["lane_admitted"]
        for row in natural
    )
    natural_complete = bool(full_cells) and all(
        row["exact"]
        and row["full_rank"]
        and row["rhs_compatible"]
        and row["verified_factor_logs"]
        and row["target_descent_count"] > 0
        and row["target_descent_fraction"] == 1.0
        and row["below_rho_proxy"]
        for row in natural
    ) and summation_ffe_admission_complete
    promotion_allowed = source_breakthrough and natural_complete and audit_status["promotion_binding_valid"]
    steering = focus_plan(full_cells, focus_budget, frontier_status)
    routing_generalization = routing_generalization_summary(full_cells, fixed_config_name)
    routing_specificity = routing_specificity_summary(full_cells)
    major_result_replication_stats = major_result_replication_summary(full_cells)
    shoup_pressure = shoup_pressure_summary(full_cells)
    resolutions = ambiguity_resolutions(
        full_cells,
        audit_status,
        fixed_config_name,
        source_breakthrough,
    )
    compliance = guidance_compliance(steering, resolutions)
    lineage = experiment_lineage(source_sha, steering)

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note_url": methodology["source_post_url_with_query"],
        "claim_status": (
            "AUDIT_BOUND_PROMOTION_CANDIDATE"
            if promotion_allowed
            else "DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION"
        ),
        "methodology": methodology,
        "source_intake": {
            "tweet_source": build_tweet_source_payload(methodology),
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha,
            "schema": payload.get("schema"),
            "claim_status": payload.get("claim_status"),
        },
        "fixed_routing_intervention": fixed_config_name,
        "cell_reports": cells,
        "routing_generalization": routing_generalization,
        "routing_specificity_control": routing_specificity,
        "frontier_lane_preflights": frontier_status,
        "summary": {
            "cell_count": len(cells),
            "full_cell_count": len(full_cells),
            "natural_exact_full_cell_count": sum(row["exact"] for row in natural),
            "natural_stored_but_unusable_full_cell_count": sum(
                row["stored_but_unusable"] for row in natural
            ),
            "natural_full_rank_cell_count": sum(row["full_rank"] for row in natural),
            "natural_verified_log_cell_count": sum(
                row["verified_factor_logs"] for row in natural
            ),
            "natural_complete_descent_cell_count": sum(
                bool(row["target_descent_count"])
                and row["target_descent_fraction"] == 1.0
                for row in natural
            ),
            "natural_below_rho_cell_count": sum(row["below_rho_proxy"] for row in natural),
            "natural_summation_ffe_evidence_cell_count": sum(
                row["has_summation_ffe_evidence"] for row in natural
            ),
            "natural_summation_ffe_evidence_marker_count": sum(
                row["summation_ffe_evidence_count"] for row in natural
            ),
            "natural_summation_ffe_evidence_ready_cell_count": sum(
                row["summation_ffe_readiness"]["replay_readiness"]
                == "exact_replay_inputs_ready"
                for row in natural
            ),
            "natural_summation_ffe_evidence_missing_cell_count": sum(
                row["has_summation_ffe_evidence"]
                and row["summation_ffe_readiness"]["replay_readiness"]
                != "exact_replay_inputs_ready"
                for row in natural
            ),
            "natural_summation_ffe_evidence_marker_only_count": sum(
                row["summation_ffe_readiness"]["replay_readiness_class"] == "marker_only"
                for row in natural
            ),
            "natural_summation_ffe_evidence_partial_payload_count": sum(
                row["summation_ffe_readiness"]["replay_readiness_class"] == "partial_payload"
                for row in natural
            ),
            "natural_summation_ffe_lane_admitted_cell_count": sum(
                row["has_summation_ffe_evidence"]
                and row["summation_ffe_new_factor_row_discovery_gate"]["lane_admitted"]
                for row in natural
            ),
            "natural_summation_ffe_lane_blocked_cell_count": sum(
                row["has_summation_ffe_evidence"]
                and not row["summation_ffe_new_factor_row_discovery_gate"]["lane_admitted"]
                for row in natural
            ),
            "natural_summation_ffe_admission_complete": summation_ffe_admission_complete,
            "oracle_rank_headroom_cell_count": sum(
                float(cell["routing_gap"]["oracle_rank_headroom"] or 0.0) > 0.0
                for cell in full_cells
            ),
            "fixed_rank_headroom_cell_count": sum(
                float(cell["routing_gap"]["fixed_rank_headroom"] or 0.0) > 0.0
                for cell in full_cells
            ),
            "target_fixed_fraction_of_oracle_rank_headroom": routing_generalization[
                "all_target_cells"
            ]["fixed_fraction_of_oracle_rank_headroom"],
            "prospective_fixed_route_status": routing_generalization[
                "prospective_transfer"
            ]["status"],
            "routing_specificity_conclusion": routing_specificity["conclusion"],
            "shoup_pressure": shoup_pressure,
            "major_result_replication": major_result_replication_stats,
            "natural_bottleneck_counts": dict(sorted(bottlenecks.items())),
            "executed_uniform_control_cell_count": len(uniform_controls),
            "observed_collision_ratio_vs_uniform_range": (
                [
                    min(
                        row["observed_collision_ratio_vs_executed_uniform"]
                        for row in uniform_controls
                        if row["observed_collision_ratio_vs_executed_uniform"] is not None
                    ),
                    max(
                        row["observed_collision_ratio_vs_executed_uniform"]
                        for row in uniform_controls
                        if row["observed_collision_ratio_vs_executed_uniform"] is not None
                    ),
                ]
                if any(
                    row["observed_collision_ratio_vs_executed_uniform"] is not None
                    for row in uniform_controls
                )
                else None
            ),
            "source_breakthrough_claim": source_breakthrough,
            "natural_pipeline_complete": natural_complete,
            "promotion_allowed": promotion_allowed,
            "frontier_preflight_provided_count": frontier_status[
                "provided_count"
            ],
            "frontier_closed_lane_count": frontier_status[
                "closed_lane_count"
            ],
            "frontier_closed_lanes": frontier_status["closed_lanes"],
        },
        "focus_queue": steering["selected"],
        "next_action": next_action(steering),
        "autoresearch_steering": {
            "critical_experiment_budget": steering["budget"],
            "candidate_count": steering["candidate_count"],
            "deferred_experiments": steering["deferred"],
            "focus_accounting": steering["focus_accounting"],
            "selection_policy": steering["selection_policy"],
            "ambiguity_resolutions": resolutions,
            "guidance_compliance": compliance,
            "experiment_lineage": lineage,
            "self_resolution_policy": (
                "Resolve non-blocking representational ambiguity deterministically and record it. "
                "Stop and withhold promotion whenever exactness, descent evidence, source authority, "
                "or independent audit binding is missing."
            ),
            "operator_interrupt_policy": (
                "Blocking uncertainty records require operator interrupt. Non-blocking ambiguity is "
                "resolved deterministically inside the harness."
            ),
        },
        "independent_audit": audit_status,
        "promotion_boundary": (
            "Intervention-selected improvements are diagnostic only. Promotion requires the natural "
            "preregistered route to pass exact relation supply, full RHS-compatible rank, verified "
            "factor logs, separate target descent, the P1436 cost/exponent gate, any applicable "
            "R68 new-factor-row, R69 closure-collision, and applicable structured-source admission gates, and an independent audit cryptographically bound "
            "to this exact source JSON."
        ),
    }
    return report


def render_note(report: dict[str, Any]) -> str:
    summary = report["summary"]
    tweet_source = report["source_intake"]["tweet_source"]
    lines = [
        "# P1436 autoresearch focus report",
        "",
        "## Status",
        "",
        f"`{report['claim_status']}`",
        "",
        (
            "Natural full cells: exact "
            f"`{summary['natural_exact_full_cell_count']}/{summary['full_cell_count']}`, "
            f"full-rank `{summary['natural_full_rank_cell_count']}/{summary['full_cell_count']}`, "
            f"verified logs `{summary['natural_verified_log_cell_count']}/{summary['full_cell_count']}`, "
            f"below-rho `{summary['natural_below_rho_cell_count']}/{summary['full_cell_count']}`."
        ),
        "",
        (
            "Stored-but-unusable / oracle-rank-headroom / fixed-rank-headroom cells: "
            f"`{summary['natural_stored_but_unusable_full_cell_count']}/"
            f"{summary['oracle_rank_headroom_cell_count']}/"
            f"{summary['fixed_rank_headroom_cell_count']}`."
        ),
        (
            "Summation/FFE evidence markers: "
            f"`{summary['natural_summation_ffe_evidence_cell_count']}` cells with "
            f"`{summary['natural_summation_ffe_evidence_marker_count']}` markers, "
            f"exact-replay ready for "
            f"`{summary['natural_summation_ffe_evidence_ready_cell_count']}` cells and "
            f"missing exact inputs for "
            f"`{summary['natural_summation_ffe_evidence_missing_cell_count']}` cells; "
            f"R68 lane-admitted "
            f"`{summary['natural_summation_ffe_lane_admitted_cell_count']}` and blocked "
            f"`{summary['natural_summation_ffe_lane_blocked_cell_count']}`."
        ),
        (
            "Bound frontier preflights: "
            f"`{summary['frontier_preflight_provided_count']}`; currently closed lanes: "
            f"`{', '.join(summary['frontier_closed_lanes']) or 'none'}`."
        ),
        "",
        (
            "Target-policy fixed fraction of oracle rank headroom: "
            f"`{summary['target_fixed_fraction_of_oracle_rank_headroom']}`. "
            "Prospective transfer: "
            f"`{summary['prospective_fixed_route_status']}`."
        ),
        "",
        (
            "Matched hash-policy specificity control: "
            f"`{summary['routing_specificity_conclusion']}`."
        ),
        "",
        f"Audit-bound promotion allowed: `{summary['promotion_allowed']}`.",
        "",
        (
            "Shoup pressure status: "
            f"`{summary['shoup_pressure']['status']}` ("
            f"passes gate: {summary['shoup_pressure']['meets_shoup_pressure_gate']})"
        ),
        (
            "Shoup gate inputs: "
            f"{summary['shoup_pressure']['eligible_cell_count']} exact+verified cells, "
            f"{summary['shoup_pressure']['distinct_scale_count']} scales, "
            f"threshold exponent < {summary['shoup_pressure']['exponent_threshold']} "
            f"and residual <= {summary['shoup_pressure']['residual_tolerance']}."
        ),
        "",
        "## Shoup Pressure",
        "",
        (
            "Scale coverage: "
            f"`{summary['shoup_pressure']['distinct_scale_count']}` (required "
            f"`{summary['shoup_pressure']['required_scale_count']}`), status: "
            f"`{summary['shoup_pressure']['status']}`."
        ),
        (
            "Fitted exponent (log charged operations vs log group order): "
            f"`{summary['shoup_pressure']['fit']['exponent_in_group_order'] if summary['shoup_pressure']['fit'] else None}`; "
            f"max abs residual log: "
            f"`{summary['shoup_pressure']['fit']['residual_abs_max'] if summary['shoup_pressure']['fit'] else None}`."
        ),
        "",
        "Method source: " + report["methodology"]["source_post_url_with_query"],
        (
            "Tweet source captured: "
            f"{tweet_source['tweet_url']} "
            f"(text_included={tweet_source['tweet_text_included']})"
        ),
        (
            "Tweet source summary: "
            f"{tweet_source['tweet_summary']}"
        ),
        (
            f"Tweet source summary verbatim? "
            f"{tweet_source['tweet_summary_is_verbatim']}"
        ),
        (
            f"Tweet posted at: "
            f"{tweet_source['tweet_posted_at'] or 'unknown'}"
        ),
        (
            "Tweet intake mode: "
            f"{tweet_source['tweet_intake_mode']}"
        ),
        (
            "Tweet source title: "
            f"{tweet_source['tweet_source_title'] or 'unknown'}"
            f" (source: {tweet_source['tweet_source_url'] or 'unknown'})"
        ),
        (
            "Paper title as written in post: "
            f"{tweet_source['tweet_referenced_paper_title'] or 'unknown'}"
        ),
        (
            "Tweet media: "
            f"{tweet_source['tweet_media_count']} items, "
            f"has_media={tweet_source['tweet_has_media']} "
            f"({', '.join(tweet_source['tweet_media_types'])})"
            if tweet_source["tweet_media_count"] > 0
            else "Tweet media: none"
        ),
        (
            "Tweet hashtags: "
            f"{', '.join(tweet_source['tweet_hashtags'])}"
            if tweet_source["tweet_hashtags"]
            else "Tweet hashtags: none"
        ),
        "",
        "## Next Action",
        (
            "1. `" + report["next_action"]["focus_id"] + "`: "
            + report["next_action"]["action"]
        ),
        (
            "   Decisive test: "
            f"{report['next_action']['decisive_test']}"
        ),
        (
            "   Falsifier: "
            f"{report['next_action']['falsifier']}"
        ),
        (
            "   Required artifacts: "
            f"{', '.join(report['next_action']['required_artifacts'])}"
        ),
        "",
        "## Focus Queue",
        "",
    ]
    for index, item in enumerate(report["focus_queue"], start=1):
        experiment = item["experiment"]
        lines.extend(
            [
                f"{index}. `{item['id']}`: {item['action']}",
                f"   Hypothesis: {experiment['hypothesis']}",
                f"   Falsifier: {experiment['falsifier']}",
            ]
        )
    steering = report["autoresearch_steering"]
    specificity = report["routing_specificity_control"]
    lines.extend([
        "",
        "## Routing Controls",
        "",
        (
            "Matched curves / positive coordinate-specific excess: "
            f"`{specificity['matched_curve_count']}/"
            f"{specificity['positive_coordinate_specific_excess_count']}`."
        ),
        "",
        report["methodology"]["fidelity_boundary"],
    ])
    lines.extend(
        [
            "",
            "## Autoresearch Guidance",
            (
                "Source: "
                f"{report['methodology']['source_post_url_with_query']}"
                " (canonical "
                f"{report['methodology']['source_post_url']})"
            ),
            (
                "Bounded critical set: "
                f"{report['methodology']['tweet_guidance']['bounded_critical_set']} "
                f"- {report['methodology']['tweet_guidance']['bounded_critical_set_note']}"
            ),
            (
                "Deterministic non-blocking ambiguity handling: "
                f"{report['methodology']['tweet_guidance']['non_blocking_ambiguity_is_deterministic']} "
                f"- {report['methodology']['tweet_guidance']['non_blocking_ambiguity_note']}"
            ),
            (
                "Peripheral scope deference: "
                f"{report['methodology']['tweet_guidance']['peripheral_scope_defer']} "
                f"- {report['methodology']['tweet_guidance']['peripheral_scope_defer_note']}"
            ),
            (
                "Operator interrupt policy: "
                f"{report['methodology']['tweet_guidance']['operator_interrupt_only_for_blocking_uncertainty']} "
                "- blocking uncertainty requires operator action before any promotion."
            ),
            (
                "Experiment lineage source: "
                f"{report['methodology']['openresearch_cli_url']} "
                "(logical lineage only; no branch/run materialization is claimed)."
            ),
            "",
            "## Guidance Compliance",
            (
                "Bounded critical set enforced: "
                f"{steering['guidance_compliance']['bounded_critical_set_enforced']}"
            ),
            (
                "Peripheral scope deferral enforced: "
                f"{steering['guidance_compliance']['peripheral_scope_defer_enforced']}"
            ),
            (
                "Non-blocking ambiguity resolutions recorded: "
                f"{steering['guidance_compliance']['non_blocking_ambiguity_resolutions_recorded']}"
            ),
            (
                "Operator interrupt alignment: "
                f"{steering['guidance_compliance']['operator_interrupt_alignment']}"
            ),
            (
                "Selected focus candidates fully specified: "
                f"{steering['guidance_compliance']['selected_candidates_have_full_spec']}"
            ),
        ]
    )
    lineage = steering["experiment_lineage"]
    lines.extend(
        [
            "",
            "## Experiment Lineage",
            (
                "Baseline / queued / deferred nodes: "
                f"`1/{lineage['queued_count']}/{lineage['deferred_count']}`; "
                f"mode `{lineage['lineage_mode']}`."
            ),
            lineage["materialization_boundary"],
        ]
    )
    lines.extend(
        [
            "",
            "## Major Result Replication",
            (
                "Fully replicated natural cells: "
                f"{summary['major_result_replication']['fully_replicated_cells']}/"
                f"{summary['major_result_replication']['cell_count']}"
            ),
            f"Per-stage status counts: {summary['major_result_replication']['stage_status_counts']}",
        ]
    )
    lines.extend(["", "## Deferred Experiments", ""])
    if steering["deferred_experiments"]:
        for item in steering["deferred_experiments"]:
            lines.append(
                f"- `{item['id']}` (rank {item['rank']}): {item['reason']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Ambiguity Resolutions", ""])
    for item in steering["ambiguity_resolutions"]:
        lines.append(
            f"- `{item['id']}`: {item['resolution']} "
            f"Blocks promotion: `{item['blocks_promotion']}`."
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        report["promotion_boundary"],
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summation-ffe-evidence-inventory",
        type=Path,
        default=None,
        help=(
            "Output path for the summation/FFE evidence inventory. "
            "Defaults to <output-parent>/summation_ffe_evidence_inventory.json."
        ),
    )
    parser.add_argument(
        "--summation-ffe-replay-plan",
        type=Path,
        default=None,
        help=(
            "Output path for the summation/FFE replay plan. "
            "Defaults to <output-parent>/summation_ffe_evidence_replay_plan.json."
        ),
    )
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--note-url", type=str, default=DEFAULT_NOTE_URL)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument(
        "--idea340-preflight",
        type=Path,
        default=DEFAULT_IDEA340_PREFLIGHT,
    )
    parser.add_argument(
        "--slice-source-preflight",
        type=Path,
        default=DEFAULT_SLICE_SOURCE_PREFLIGHT,
    )
    parser.add_argument(
        "--presurface-full-charge-preflight",
        type=Path,
        default=DEFAULT_PRESURFACE_FULL_CHARGE_PREFLIGHT,
    )
    parser.add_argument(
        "--factor-line-equivalence-probe",
        type=Path,
        default=DEFAULT_FACTOR_LINE_EQUIVALENCE_PROBE,
    )
    parser.add_argument(
        "--constructive-closure-collision-gate",
        type=Path,
        default=DEFAULT_CONSTRUCTIVE_CLOSURE_COLLISION_GATE,
    )
    parser.add_argument(
        "--multiplicative-x-s3-closure-screen",
        type=Path,
        default=DEFAULT_MULTIPLICATIVE_X_S3_CLOSURE_SCREEN,
    )
    parser.add_argument(
        "--s4-centered-carry-rank-probe",
        type=Path,
        default=DEFAULT_S4_CENTERED_CARRY_RANK_PROBE,
    )
    parser.add_argument(
        "--s6-centered-carry-rank-minor-probe",
        type=Path,
        default=DEFAULT_S6_CENTERED_CARRY_RANK_MINOR_PROBE,
    )
    parser.add_argument(
        "--resultant-valuation-trace-grammar",
        type=Path,
        default=DEFAULT_RESULTANT_VALUATION_TRACE_GRAMMAR,
    )
    parser.add_argument(
        "--s6-residual-decision-diagram-probe",
        type=Path,
        default=DEFAULT_S6_RESIDUAL_DECISION_DIAGRAM_PROBE,
    )
    parser.add_argument(
        "--s6-iterated-norm-support-probe",
        type=Path,
        default=DEFAULT_S6_ITERATED_NORM_SUPPORT_PROBE,
    )
    parser.add_argument(
        "--s6-subset-incidence-mobius-probe",
        type=Path,
        default=DEFAULT_S6_SUBSET_INCIDENCE_MOBIUS_PROBE,
    )
    parser.add_argument(
        "--target-translated-frequency-orbit-probe",
        type=Path,
        default=DEFAULT_TARGET_TRANSLATED_FREQUENCY_ORBIT_PROBE,
    )
    parser.add_argument(
        "--actual-s6-fermat-tensor-train-probe",
        type=Path,
        default=DEFAULT_ACTUAL_S6_FERMAT_TT_PROBE,
    )
    parser.add_argument(
        "--scalar-only-nested-norm-slp-probe",
        type=Path,
        default=DEFAULT_SCALAR_ONLY_NESTED_NORM_SLP_PROBE,
    )
    parser.add_argument(
        "--batched-nested-norm-node-compiler-probe",
        type=Path,
        default=DEFAULT_BATCHED_NESTED_NORM_NODE_COMPILER_PROBE,
    )
    parser.add_argument(
        "--full-multiplicative-x-coset-endpoint-probe",
        type=Path,
        default=DEFAULT_FULL_MULTIPLICATIVE_X_COSET_ENDPOINT_PROBE,
    )
    parser.add_argument(
        "--cartesian-sum-compact-divisor-probe",
        type=Path,
        default=DEFAULT_CARTESIAN_SUM_COMPACT_DIVISOR_PROBE,
    )
    parser.add_argument(
        "--5a5c-coordinate-filtration-probe",
        dest="five_a_five_c_coordinate_filtration_probe",
        type=Path,
        default=DEFAULT_5A5C_COORDINATE_FILTRATION_PROBE,
    )
    parser.add_argument(
        "--5a5c-marked-resultant-source-section-probe",
        dest="five_a_five_c_marked_resultant_source_section_probe",
        type=Path,
        default=DEFAULT_5A5C_MARKED_RESULTANT_SOURCE_SECTION_PROBE,
    )
    parser.add_argument(
        "--5a5c-target-uniform-precoefficient-circuit-probe",
        dest="five_a_five_c_target_uniform_precoefficient_circuit_probe",
        type=Path,
        default=DEFAULT_5A5C_TARGET_UNIFORM_PRECOEFFICIENT_CIRCUIT_PROBE,
    )
    parser.add_argument(
        "--5a5c-sparse-multihomogeneous-moment-recurrence-probe",
        dest="five_a_five_c_sparse_multihomogeneous_moment_recurrence_probe",
        type=Path,
        default=DEFAULT_5A5C_SPARSE_MULTIHOMOGENEOUS_MOMENT_RECURRENCE_PROBE,
    )
    parser.add_argument(
        "--5a5c-jet-preserving-addition-pushforward-probe",
        dest="five_a_five_c_jet_preserving_addition_pushforward_probe",
        type=Path,
        default=DEFAULT_5A5C_JET_PRESERVING_ADDITION_PUSHFORWARD_PROBE,
    )
    parser.add_argument(
        "--5a5c-black-box-resultant-localizer-probe",
        dest="five_a_five_c_black_box_resultant_localizer_probe",
        type=Path,
        default=DEFAULT_5A5C_BLACK_BOX_RESULTANT_LOCALIZER_PROBE,
    )
    parser.add_argument(
        "--5a5c-fixed-marker-scalar-recurrence-probe",
        dest="five_a_five_c_fixed_marker_scalar_recurrence_probe",
        type=Path,
        default=DEFAULT_5A5C_FIXED_MARKER_SCALAR_RECURRENCE_PROBE,
    )
    parser.add_argument(
        "--5a5c-nonlocal-moment-hankel-translation-probe",
        dest="five_a_five_c_nonlocal_moment_hankel_translation_probe",
        type=Path,
        default=DEFAULT_5A5C_NONLOCAL_MOMENT_HANKEL_TRANSLATION_PROBE,
    )
    parser.add_argument(
        "--5a5c-unequal-list-subfunction-inversion-probe",
        dest="five_a_five_c_unequal_list_subfunction_inversion_probe",
        type=Path,
        default=DEFAULT_5A5C_UNEQUAL_LIST_SUBFUNCTION_INVERSION_PROBE,
    )
    parser.add_argument(
        "--5a5c-compact-elliptic-subfunction-map-probe",
        dest="five_a_five_c_compact_elliptic_subfunction_map_probe",
        type=Path,
        default=DEFAULT_5A5C_COMPACT_ELLIPTIC_SUBFUNCTION_MAP_PROBE,
    )
    parser.add_argument(
        "--5a5c-shared-semilinear-incidence-probe",
        dest="five_a_five_c_shared_semilinear_incidence_probe",
        type=Path,
        default=DEFAULT_5A5C_SHARED_SEMILINEAR_INCIDENCE_PROBE,
    )
    parser.add_argument(
        "--5a5c-implicit-veronese-hyperplane-source-index-probe",
        dest="five_a_five_c_implicit_veronese_hyperplane_source_index_probe",
        type=Path,
        default=DEFAULT_5A5C_IMPLICIT_VERONESE_HYPERPLANE_SOURCE_INDEX_PROBE,
    )
    parser.add_argument(
        "--5a5c-aggregate-veronese-projector-recurrence-probe",
        dest="five_a_five_c_aggregate_veronese_projector_recurrence_probe",
        type=Path,
        default=DEFAULT_5A5C_AGGREGATE_VERONESE_PROJECTOR_RECURRENCE_PROBE,
    )
    parser.add_argument(
        "--5a5c-modular-frobenius-trace-recurrence-probe",
        dest="five_a_five_c_modular_frobenius_trace_recurrence_probe",
        type=Path,
        default=DEFAULT_5A5C_MODULAR_FROBENIUS_TRACE_RECURRENCE_PROBE,
    )
    parser.add_argument(
        "--5a5c-factored-transposed-projector-trace-probe",
        dest="five_a_five_c_factored_transposed_projector_trace_probe",
        type=Path,
        default=DEFAULT_5A5C_FACTORED_TRANSPOSED_PROJECTOR_TRACE_PROBE,
    )
    parser.add_argument(
        "--5a5c-nonlinear-tensor-tower-trace-probe",
        dest="five_a_five_c_nonlinear_tensor_tower_trace_probe",
        type=Path,
        default=DEFAULT_5A5C_NONLINEAR_TENSOR_TOWER_TRACE_PROBE,
    )
    parser.add_argument(
        "--5a5c-multiedge-digitized-equality-projector-probe",
        dest="five_a_five_c_multiedge_digitized_equality_projector_probe",
        type=Path,
        default=DEFAULT_5A5C_MULTIEDGE_DIGITIZED_EQUALITY_PROJECTOR_PROBE,
    )
    parser.add_argument(
        "--5a5c-succinct-aggregate-digit-trie-probe",
        dest="five_a_five_c_succinct_aggregate_digit_trie_probe",
        type=Path,
        default=DEFAULT_5A5C_SUCCINCT_AGGREGATE_DIGIT_TRIE_PROBE,
    )
    parser.add_argument(
        "--5a5c-actual-divisor-image-entropy-merge-probe",
        dest="five_a_five_c_actual_divisor_image_entropy_merge_probe",
        type=Path,
        default=DEFAULT_5A5C_ACTUAL_DIVISOR_IMAGE_ENTROPY_MERGE_PROBE,
    )
    parser.add_argument(
        "--5a5c-two-sided-implicit-join-probe",
        dest="five_a_five_c_two_sided_implicit_join_probe",
        type=Path,
        default=DEFAULT_5A5C_TWO_SIDED_IMPLICIT_JOIN_PROBE,
    )
    parser.add_argument(
        "--5a5c-target-forced-algebraic-join-filter-probe",
        dest="five_a_five_c_target_forced_algebraic_join_filter_probe",
        type=Path,
        default=DEFAULT_5A5C_TARGET_FORCED_ALGEBRAIC_JOIN_FILTER_PROBE,
    )
    parser.add_argument(
        "--5a5c-compact-preendpoint-s3-ffe-pushdown-probe",
        dest="five_a_five_c_compact_preendpoint_s3_ffe_pushdown_probe",
        type=Path,
        default=DEFAULT_5A5C_COMPACT_PREENDPOINT_S3_FFE_PUSHDOWN_PROBE,
    )
    parser.add_argument(
        "--5a5c-actual-deck-nonmergeable-target-pullback-probe",
        dest="five_a_five_c_actual_deck_nonmergeable_target_pullback_probe",
        type=Path,
        default=DEFAULT_5A5C_ACTUAL_DECK_NONMERGEABLE_TARGET_PULLBACK_PROBE,
    )
    parser.add_argument(
        "--5a5c-scalar-target-norm-count-circuit-probe",
        dest="five_a_five_c_scalar_target_norm_count_circuit_probe",
        type=Path,
        default=DEFAULT_5A5C_SCALAR_TARGET_NORM_COUNT_CIRCUIT_PROBE,
    )
    parser.add_argument(
        "--5a5c-noncharacter-algebraic-target-norm-resultant-probe",
        dest=(
            "five_a_five_c_noncharacter_algebraic_target_norm_"
            "resultant_probe"
        ),
        type=Path,
        default=(
            DEFAULT_5A5C_NONCHARACTER_ALGEBRAIC_TARGET_NORM_RESULTANT_PROBE
        ),
    )
    parser.add_argument(
        "--5a5c-factored-elliptic-lambda-ring-chow-norm-probe",
        dest="five_a_five_c_factored_elliptic_lambda_ring_chow_norm_probe",
        type=Path,
        default=DEFAULT_5A5C_FACTORED_ELLIPTIC_LAMBDA_RING_CHOW_NORM_PROBE,
    )
    parser.add_argument(
        "--5a5c-poincare-theta-target-section-rank-probe",
        dest="five_a_five_c_poincare_theta_target_section_rank_probe",
        type=Path,
        default=DEFAULT_5A5C_POINCARE_THETA_TARGET_SECTION_RANK_PROBE,
    )
    parser.add_argument(
        "--5a5c-theta-addition-cancellation-network-probe",
        dest="five_a_five_c_theta_addition_cancellation_network_probe",
        type=Path,
        default=DEFAULT_5A5C_THETA_ADDITION_CANCELLATION_NETWORK_PROBE,
    )
    parser.add_argument(
        "--5a5c-finite-deck-alternant-annihilator-probe",
        dest="five_a_five_c_finite_deck_alternant_annihilator_probe",
        type=Path,
        default=DEFAULT_5A5C_FINITE_DECK_ALTERNANT_ANNIHILATOR_PROBE,
    )
    parser.add_argument(
        "--5a5c-gauge-normalized-endpoint-query2p1-probe",
        dest="five_a_five_c_gauge_normalized_endpoint_query2p1_probe",
        type=Path,
        default=DEFAULT_5A5C_GAUGE_NORMALIZED_ENDPOINT_QUERY2P1_PROBE,
    )
    parser.add_argument(
        "--5a5c-nonlinear-elliptic-orbit-product-probe",
        dest="five_a_five_c_nonlinear_elliptic_orbit_product_probe",
        type=Path,
        default=DEFAULT_5A5C_NONLINEAR_ELLIPTIC_ORBIT_PRODUCT_PROBE,
    )
    parser.add_argument(
        "--5a5c-transposed-nonuniform-c5-leaf-generator-probe",
        dest="five_a_five_c_transposed_nonuniform_c5_leaf_generator_probe",
        type=Path,
        default=DEFAULT_5A5C_TRANSPOSED_NONUNIFORM_C5_LEAF_GENERATOR_PROBE,
    )
    parser.add_argument(
        "--relation-arity-factor-base-transposed-interface-probe",
        dest="relation_arity_factor_base_transposed_interface_probe",
        type=Path,
        default=DEFAULT_RELATION_ARITY_FACTOR_BASE_TRANSPOSED_INTERFACE_PROBE,
    )
    parser.add_argument(
        "--m6-a6-batched-c3-pair-sum-source-locator-probe",
        dest="m6_a6_batched_c3_pair_sum_source_locator_probe",
        type=Path,
        default=DEFAULT_M6_A6_BATCHED_C3_PAIR_SUM_SOURCE_LOCATOR_PROBE,
    )
    parser.add_argument(
        "--m6-target-batched-c3-elliptic-transpose-probe",
        dest="m6_target_batched_c3_elliptic_transpose_probe",
        type=Path,
        default=DEFAULT_M6_TARGET_BATCHED_C3_ELLIPTIC_TRANSPOSE_PROBE,
    )
    parser.add_argument(
        "--m6-nonlinear-value-sensitive-c6-source-locator-probe",
        dest="m6_nonlinear_value_sensitive_c6_source_locator_probe",
        type=Path,
        default=DEFAULT_M6_NONLINEAR_VALUE_SENSITIVE_C6_SOURCE_LOCATOR_PROBE,
    )
    parser.add_argument(
        "--m6-output-sensitive-nonlinear-c5-source-index-probe",
        dest="m6_output_sensitive_nonlinear_c5_source_index_probe",
        type=Path,
        default=DEFAULT_M6_OUTPUT_SENSITIVE_NONLINEAR_C5_SOURCE_INDEX_PROBE,
    )
    parser.add_argument(
        "--m6-suboutput-implicit-c5-character-pairing-probe",
        dest="m6_suboutput_implicit_c5_character_pairing_probe",
        type=Path,
        default=DEFAULT_M6_SUBOUTPUT_IMPLICIT_C5_CHARACTER_PAIRING_PROBE,
    )
    parser.add_argument(
        "--m6-small-k-multiplicative-c5-moment-torus-probe",
        dest="m6_small_k_multiplicative_c5_moment_torus_probe",
        type=Path,
        default=DEFAULT_M6_SMALL_K_MULTIPLICATIVE_C5_MOMENT_TORUS_PROBE,
    )
    parser.add_argument(
        "--torus-c5-explicit-split-global-rebalance-probe",
        dest="torus_c5_explicit_split_global_rebalance_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_EXPLICIT_SPLIT_GLOBAL_REBALANCE_PROBE,
    )
    parser.add_argument(
        "--torus-c5-fourier-product-resultant-probe",
        dest="torus_c5_fourier_product_resultant_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_FOURIER_PRODUCT_RESULTANT_PROBE,
    )
    parser.add_argument(
        "--torus-c5-linear-sketch-circulant-probe",
        dest="torus_c5_linear_sketch_circulant_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_LINEAR_SKETCH_CIRCULANT_PROBE,
    )
    parser.add_argument(
        "--torus-c5-prime-order-homomorphic-fingerprint-probe",
        dest="torus_c5_prime_order_homomorphic_fingerprint_probe",
        type=Path,
        default=(
            DEFAULT_TORUS_C5_PRIME_ORDER_HOMOMORPHIC_FINGERPRINT_PROBE
        ),
    )
    parser.add_argument(
        "--torus-c5-explicit-hash-correction-support-probe",
        dest="torus_c5_explicit_hash_correction_support_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_EXPLICIT_HASH_CORRECTION_SUPPORT_PROBE,
    )
    parser.add_argument(
        "--torus-c5-bucket-resultant-routing-tradeoff-probe",
        dest="torus_c5_bucket_resultant_routing_tradeoff_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_BUCKET_RESULTANT_ROUTING_TRADEOFF_PROBE,
    )
    parser.add_argument(
        "--torus-c5-rational-selector-degree-probe",
        dest="torus_c5_rational_selector_degree_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_RATIONAL_SELECTOR_DEGREE_PROBE,
    )
    parser.add_argument(
        "--torus-c5-piecewise-selector-decision-dag-probe",
        dest="torus_c5_piecewise_selector_decision_dag_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_PIECEWISE_SELECTOR_DECISION_DAG_PROBE,
    )
    parser.add_argument(
        "--torus-c5-sparse-fourier-predicate-transfer-probe",
        dest="torus_c5_sparse_fourier_predicate_transfer_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_SPARSE_FOURIER_PREDICATE_TRANSFER_PROBE,
    )
    parser.add_argument(
        "--torus-c5-consecutive-mode-predicate-probe",
        dest="torus_c5_consecutive_mode_predicate_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_CONSECUTIVE_MODE_PREDICATE_PROBE,
    )
    parser.add_argument(
        "--torus-c5-base-field-frobenius-predicate-dag-probe",
        dest="torus_c5_base_field_frobenius_predicate_dag_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_BASE_FIELD_FROBENIUS_PREDICATE_DAG_PROBE,
    )
    parser.add_argument(
        "--torus-c5-sparse-monomial-root-bound-probe",
        dest="torus_c5_sparse_monomial_root_bound_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_SPARSE_MONOMIAL_ROOT_BOUND_PROBE,
    )
    parser.add_argument(
        "--torus-c5-two-atom-geometric-progression-probe",
        dest="torus_c5_two_atom_geometric_progression_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_TWO_ATOM_GEOMETRIC_PROGRESSION_PROBE,
    )
    parser.add_argument(
        "--torus-c5-khatri-rao-kruskal-amplification-probe",
        dest="torus_c5_khatri_rao_kruskal_amplification_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_KHATRI_RAO_KRUSKAL_AMPLIFICATION_PROBE,
    )
    parser.add_argument(
        "--torus-c5-all-nonzero-path-product-probe",
        dest="torus_c5_all_nonzero_path_product_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_ALL_NONZERO_PATH_PRODUCT_PROBE,
    )
    parser.add_argument(
        "--torus-c5-binomial-node-union-depth-probe",
        dest="torus_c5_binomial_node_union_depth_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_BINOMIAL_NODE_UNION_DEPTH_PROBE,
    )
    parser.add_argument(
        "--torus-c5-chebotarev-fiber-cover-probe",
        dest="torus_c5_chebotarev_fiber_cover_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_CHEBOTAREV_FIBER_COVER_PROBE,
    )
    parser.add_argument(
        "--torus-c5-order-two-three-minor-rigidity-probe",
        dest="torus_c5_order_two_three_minor_rigidity_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_ORDER_TWO_THREE_MINOR_RIGIDITY_PROBE,
    )
    parser.add_argument(
        "--torus-c5-order-two-four-minor-claw-probe",
        dest="torus_c5_order_two_four_minor_claw_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_ORDER_TWO_FOUR_MINOR_CLAW_PROBE,
    )
    parser.add_argument(
        "--torus-c5-sextic-mobius-character-router-probe",
        dest="torus_c5_sextic_mobius_character_router_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_SEXTIC_MOBIUS_CHARACTER_ROUTER_PROBE,
    )
    parser.add_argument(
        "--torus-c5-adaptive-character-decision-router-probe",
        dest="torus_c5_adaptive_character_decision_router_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_ADAPTIVE_CHARACTER_DECISION_ROUTER_PROBE,
    )
    parser.add_argument(
        "--torus-c5-label-congruence-correction-probe",
        dest="torus_c5_label_congruence_correction_probe",
        type=Path,
        default=DEFAULT_TORUS_C5_LABEL_CONGRUENCE_CORRECTION_PROBE,
    )
    parser.add_argument(
        "--m6-weighted-fiber-marginal-log-operator-probe",
        dest="m6_weighted_fiber_marginal_log_operator_probe",
        type=Path,
        default=DEFAULT_M6_WEIGHTED_FIBER_MARGINAL_LOG_OPERATOR_PROBE,
    )
    parser.add_argument(
        "--m6-weighted-c3-mobius-gcd-trace-probe",
        dest="m6_weighted_c3_mobius_gcd_trace_probe",
        type=Path,
        default=DEFAULT_M6_WEIGHTED_C3_MOBIUS_GCD_TRACE_PROBE,
    )
    parser.add_argument(
        "--m6-aggregate-marginal-singleton-source-equivalence-probe",
        dest="m6_aggregate_marginal_singleton_source_equivalence_probe",
        type=Path,
        default=(
            DEFAULT_M6_AGGREGATE_MARGINAL_SINGLETON_SOURCE_EQUIVALENCE_PROBE
        ),
    )
    parser.add_argument(
        "--m6-occurrence-pair-resultant-local-valuation-probe",
        dest="m6_occurrence_pair_resultant_local_valuation_probe",
        type=Path,
        default=DEFAULT_M6_OCCURRENCE_PAIR_RESULTANT_LOCAL_VALUATION_PROBE,
    )
    parser.add_argument(
        "--m6-static-3sum-indexing-tradeoff-probe",
        dest="m6_static_3sum_indexing_tradeoff_probe",
        type=Path,
        default=DEFAULT_M6_STATIC_3SUM_INDEXING_TRADEOFF_PROBE,
    )
    parser.add_argument(
        "--m6-actual-c6-shift-krylov-rank-probe",
        dest="m6_actual_c6_shift_krylov_rank_probe",
        type=Path,
        default=DEFAULT_M6_ACTUAL_C6_SHIFT_KRYLOV_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-rational-convolution-subalgebra-rigidity-probe",
        dest="m6_rational_convolution_subalgebra_rigidity_probe",
        type=Path,
        default=DEFAULT_M6_RATIONAL_CONVOLUTION_SUBALGEBRA_RIGIDITY_PROBE,
    )
    parser.add_argument(
        "--m6-matrix-free-marginal-jacobian-krylov-probe",
        dest="m6_matrix_free_marginal_jacobian_krylov_probe",
        type=Path,
        default=DEFAULT_M6_MATRIX_FREE_MARGINAL_JACOBIAN_KRYLOV_PROBE,
    )
    parser.add_argument(
        "--m6-geometry-only-weight-interpolation-adjoint-probe",
        dest="m6_geometry_only_weight_interpolation_adjoint_probe",
        type=Path,
        default=DEFAULT_M6_GEOMETRY_ONLY_WEIGHT_INTERPOLATION_ADJOINT_PROBE,
    )
    parser.add_argument(
        "--m6-symmetric-shift-reverse-only-marginal-probe",
        dest="m6_symmetric_shift_reverse_only_marginal_probe",
        type=Path,
        default=DEFAULT_M6_SYMMETRIC_SHIFT_REVERSE_ONLY_MARGINAL_PROBE,
    )
    parser.add_argument(
        "--m6-signed-quotient-multiscale-rank-probe",
        dest="m6_signed_quotient_multiscale_rank_probe",
        type=Path,
        default=DEFAULT_M6_SIGNED_QUOTIENT_MULTISCALE_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-singleton-relation-hypergraph-rank-probe",
        dest="m6_singleton_relation_hypergraph_rank_probe",
        type=Path,
        default=DEFAULT_M6_SINGLETON_RELATION_HYPERGRAPH_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-a-diversity-projective-rank-probe",
        dest="m6_a_diversity_projective_rank_probe",
        type=Path,
        default=DEFAULT_M6_A_DIVERSITY_PROJECTIVE_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-hash-to-curve-projective-rank-probe",
        dest="m6_hash_to_curve_projective_rank_probe",
        type=Path,
        default=DEFAULT_M6_HASH_TO_CURVE_PROJECTIVE_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-short-relation-near-injectivity-supply-probe",
        dest="m6_short_relation_near_injectivity_supply_probe",
        type=Path,
        default=DEFAULT_M6_SHORT_RELATION_NEAR_INJECTIVITY_SUPPLY_PROBE,
    )
    parser.add_argument(
        "--m6-random-diagonal-known-target-rank-probe",
        dest="m6_random_diagonal_known_target_rank_probe",
        type=Path,
        default=DEFAULT_M6_RANDOM_DIAGONAL_KNOWN_TARGET_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-positive-c6-generic-locator-reduction-probe",
        dest="m6_positive_c6_generic_locator_reduction_probe",
        type=Path,
        default=DEFAULT_M6_POSITIVE_C6_GENERIC_LOCATOR_REDUCTION_PROBE,
    )
    parser.add_argument(
        "--m6-signed-c3-divisor-translation-gcd-probe",
        dest="m6_signed_c3_divisor_translation_gcd_probe",
        type=Path,
        default=DEFAULT_M6_SIGNED_C3_DIVISOR_TRANSLATION_GCD_PROBE,
    )
    parser.add_argument(
        "--m6-batch-inverse-transpose-modcomp-fit-probe",
        dest="m6_batch_inverse_transpose_modcomp_fit_probe",
        type=Path,
        default=DEFAULT_M6_BATCH_INVERSE_TRANSPOSE_MODCOMP_FIT_PROBE,
    )
    parser.add_argument(
        "--m6-aggregate-union-factor-label-recovery-probe",
        dest="m6_aggregate_union_factor_label_recovery_probe",
        type=Path,
        default=DEFAULT_M6_AGGREGATE_UNION_FACTOR_LABEL_RECOVERY_PROBE,
    )
    parser.add_argument(
        "--m6-randomized-target-divisor-norm-union-probe",
        dest="m6_randomized_target_divisor_norm_union_probe",
        type=Path,
        default=DEFAULT_M6_RANDOMIZED_TARGET_DIVISOR_NORM_UNION_PROBE,
    )
    parser.add_argument(
        "--m6-global-randomizer-elliptic-translate-product-probe",
        dest="m6_global_randomizer_elliptic_translate_product_probe",
        type=Path,
        default=DEFAULT_M6_GLOBAL_RANDOMIZER_ELLIPTIC_TRANSLATE_PRODUCT_PROBE,
    )
    parser.add_argument(
        "--m6-kummer-x-translate-signed-verification-probe",
        dest="m6_kummer_x_translate_signed_verification_probe",
        type=Path,
        default=DEFAULT_M6_KUMMER_X_TRANSLATE_SIGNED_VERIFICATION_PROBE,
    )
    parser.add_argument(
        "--m6-generalized-target-divisor-weil-reciprocity-swap-probe",
        dest="m6_generalized_target_divisor_weil_reciprocity_swap_probe",
        type=Path,
        default=DEFAULT_M6_GENERALIZED_TARGET_DIVISOR_WEIL_RECIPROCITY_SWAP_PROBE,
    )
    parser.add_argument(
        "--m6-log-derivative-elliptic-cauchy-trace-probe",
        dest="m6_log_derivative_elliptic_cauchy_trace_probe",
        type=Path,
        default=DEFAULT_M6_LOG_DERIVATIVE_ELLIPTIC_CAUCHY_TRACE_PROBE,
    )
    parser.add_argument(
        "--m6-regularized-log-trace-displacement-rank-probe",
        dest="m6_regularized_log_trace_displacement_rank_probe",
        type=Path,
        default=DEFAULT_M6_REGULARIZED_LOG_TRACE_DISPLACEMENT_RANK_PROBE,
    )
    parser.add_argument(
        "--m6-lambda-zero-fitting-target-norm-dedup-probe",
        dest="m6_lambda_zero_fitting_target_norm_dedup_probe",
        type=Path,
        default=DEFAULT_M6_LAMBDA_ZERO_FITTING_TARGET_NORM_DEDUP_PROBE,
    )
    parser.add_argument(
        "--m6-balanced-miller-tree-norm-streaming-probe",
        dest="m6_balanced_miller_tree_norm_streaming_probe",
        type=Path,
        default=DEFAULT_M6_BALANCED_MILLER_TREE_NORM_STREAMING_PROBE,
    )
    parser.add_argument(
        "--m6-target-sign-conjugate-s3-self-resultant-probe",
        dest="m6_target_sign_conjugate_s3_self_resultant_probe",
        type=Path,
        default=DEFAULT_M6_TARGET_SIGN_CONJUGATE_S3_SELF_RESULTANT_PROBE,
    )
    parser.add_argument(
        "--m6-s3-determinantal-transfer-noncommutativity-probe",
        dest="m6_s3_determinantal_transfer_noncommutativity_probe",
        type=Path,
        default=DEFAULT_M6_S3_DETERMINANTAL_TRANSFER_NONCOMMUTATIVITY_PROBE,
    )
    parser.add_argument(
        "--m6-confluent-signed-dual-chow-pushforward-probe",
        dest="m6_confluent_signed_dual_chow_pushforward_probe",
        type=Path,
        default=DEFAULT_M6_CONFLUENT_SIGNED_DUAL_CHOW_PUSHFORWARD_PROBE,
    )
    parser.add_argument(
        "--m6-scalar-subset-incidence-group-testing-probe",
        dest="m6_scalar_subset_incidence_group_testing_probe",
        type=Path,
        default=DEFAULT_M6_SCALAR_SUBSET_INCIDENCE_GROUP_TESTING_PROBE,
    )
    parser.add_argument(
        "--m6-principal-target-pontryagin-resultant-probe",
        dest="m6_principal_target_pontryagin_resultant_probe",
        type=Path,
        default=DEFAULT_M6_PRINCIPAL_TARGET_PONTRYAGIN_RESULTANT_PROBE,
    )
    parser.add_argument(
        "--m6-global-marked-fitting-locator-probe",
        dest="m6_global_marked_fitting_locator_probe",
        type=Path,
        default=DEFAULT_M6_GLOBAL_MARKED_FITTING_LOCATOR_PROBE,
    )
    parser.add_argument(
        "--m6-marked-fitting-signed-norm-dedup-probe",
        dest="m6_marked_fitting_signed_norm_dedup_probe",
        type=Path,
        default=DEFAULT_M6_MARKED_FITTING_SIGNED_NORM_DEDUP_PROBE,
    )
    parser.add_argument(
        "--m6-squarefree-truncated-resultant-applicability-probe",
        dest="m6_squarefree_truncated_resultant_applicability_probe",
        type=Path,
        default=DEFAULT_M6_SQUAREFREE_TRUNCATED_RESULTANT_APPLICABILITY_PROBE,
    )
    parser.add_argument(
        "--m6-d5-directed-evaluation-survivor-probe",
        dest="m6_d5_directed_evaluation_survivor_probe",
        type=Path,
        default=DEFAULT_M6_D5_DIRECTED_EVALUATION_SURVIVOR_PROBE,
    )
    parser.add_argument(
        "--m6-monogenic-kernel-bidegree-probe",
        dest="m6_monogenic_kernel_bidegree_probe",
        type=Path,
        default=DEFAULT_M6_MONOGENIC_KERNEL_BIDEGREE_PROBE,
    )
    parser.add_argument(
        "--m6-gcd-equivalent-target-subresultant-probe",
        dest="m6_gcd_equivalent_target_subresultant_probe",
        type=Path,
        default=DEFAULT_M6_GCD_EQUIVALENT_TARGET_SUBRESULTANT_PROBE,
    )
    parser.add_argument(
        "--m6-sparse-projector-prony-locator-probe",
        dest="m6_sparse_projector_prony_locator_probe",
        type=Path,
        default=DEFAULT_M6_SPARSE_PROJECTOR_PRONY_LOCATOR_PROBE,
    )
    parser.add_argument("--fixed-config", default=DEFAULT_FIXED_CONFIG)
    parser.add_argument("--focus-budget", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.focus_budget < 1:
        raise SystemExit("--focus-budget must be positive")
    if args.summation_ffe_evidence_inventory is None:
        args.summation_ffe_evidence_inventory = args.output.with_name(
            "summation_ffe_evidence_inventory.json"
        )
    if args.summation_ffe_replay_plan is None:
        args.summation_ffe_replay_plan = args.output.with_name(
            "summation_ffe_evidence_replay_plan.json"
        )
    payload = read_json(args.input)
    audit = read_json(args.audit_json) if args.audit_json else None
    frontier_preflights = {}
    for name, path in (
        ("idea340_public_chart", args.idea340_preflight),
        ("slice_quadratic_public_source", args.slice_source_preflight),
        ("presurface_full_charge", args.presurface_full_charge_preflight),
        (
            "factor_line_direct_root_equivalence",
            args.factor_line_equivalence_probe,
        ),
        (
            "constructive_closure_collision",
            args.constructive_closure_collision_gate,
        ),
        (
            "multiplicative_x_s3_closure",
            args.multiplicative_x_s3_closure_screen,
        ),
        (
            "s4_centered_carry_rank",
            args.s4_centered_carry_rank_probe,
        ),
        (
            "s6_centered_carry_rank_minor",
            args.s6_centered_carry_rank_minor_probe,
        ),
        (
            "resultant_valuation_trace_grammar",
            args.resultant_valuation_trace_grammar,
        ),
        (
            "s6_residual_decision_diagram",
            args.s6_residual_decision_diagram_probe,
        ),
        (
            "s6_iterated_norm_support",
            args.s6_iterated_norm_support_probe,
        ),
        (
            "s6_subset_incidence_mobius",
            args.s6_subset_incidence_mobius_probe,
        ),
        (
            "target_translated_frequency_orbit",
            args.target_translated_frequency_orbit_probe,
        ),
        (
            "actual_s6_fermat_tensor_train",
            args.actual_s6_fermat_tensor_train_probe,
        ),
        (
            "scalar_only_nested_norm_slp",
            args.scalar_only_nested_norm_slp_probe,
        ),
        (
            "batched_nested_norm_node_compiler",
            args.batched_nested_norm_node_compiler_probe,
        ),
        (
            "full_multiplicative_x_coset_endpoint",
            args.full_multiplicative_x_coset_endpoint_probe,
        ),
        (
            "cartesian_sum_compact_divisor",
            args.cartesian_sum_compact_divisor_probe,
        ),
        (
            "5a5c_coordinate_filtration",
            args.five_a_five_c_coordinate_filtration_probe,
        ),
        (
            "5a5c_marked_resultant_source_section",
            args.five_a_five_c_marked_resultant_source_section_probe,
        ),
        (
            "5a5c_target_uniform_precoefficient_circuit",
            args.five_a_five_c_target_uniform_precoefficient_circuit_probe,
        ),
        (
            "5a5c_sparse_multihomogeneous_moment_recurrence",
            args.five_a_five_c_sparse_multihomogeneous_moment_recurrence_probe,
        ),
        (
            "5a5c_jet_preserving_addition_pushforward",
            args.five_a_five_c_jet_preserving_addition_pushforward_probe,
        ),
        (
            "5a5c_black_box_resultant_localizer",
            args.five_a_five_c_black_box_resultant_localizer_probe,
        ),
        (
            "5a5c_fixed_marker_scalar_recurrence",
            args.five_a_five_c_fixed_marker_scalar_recurrence_probe,
        ),
        (
            "5a5c_nonlocal_moment_hankel_translation",
            args.five_a_five_c_nonlocal_moment_hankel_translation_probe,
        ),
        (
            "5a5c_unequal_list_subfunction_inversion",
            args.five_a_five_c_unequal_list_subfunction_inversion_probe,
        ),
        (
            "5a5c_compact_elliptic_subfunction_map",
            args.five_a_five_c_compact_elliptic_subfunction_map_probe,
        ),
        (
            "5a5c_shared_semilinear_incidence_correspondence",
            args.five_a_five_c_shared_semilinear_incidence_probe,
        ),
        (
            "5a5c_implicit_veronese_hyperplane_source_index",
            args.five_a_five_c_implicit_veronese_hyperplane_source_index_probe,
        ),
        (
            "5a5c_aggregate_veronese_projector_recurrence",
            args.five_a_five_c_aggregate_veronese_projector_recurrence_probe,
        ),
        (
            "5a5c_modular_frobenius_trace_recurrence",
            args.five_a_five_c_modular_frobenius_trace_recurrence_probe,
        ),
        (
            "5a5c_factored_transposed_projector_trace",
            args.five_a_five_c_factored_transposed_projector_trace_probe,
        ),
        (
            "5a5c_nonlinear_tensor_tower_trace",
            args.five_a_five_c_nonlinear_tensor_tower_trace_probe,
        ),
        (
            "5a5c_multiedge_digitized_equality_projector",
            args.five_a_five_c_multiedge_digitized_equality_projector_probe,
        ),
        (
            "5a5c_succinct_aggregate_digit_trie",
            args.five_a_five_c_succinct_aggregate_digit_trie_probe,
        ),
        (
            "5a5c_actual_divisor_image_entropy_merge",
            args.five_a_five_c_actual_divisor_image_entropy_merge_probe,
        ),
        (
            "5a5c_two_sided_implicit_join",
            args.five_a_five_c_two_sided_implicit_join_probe,
        ),
        (
            "5a5c_target_forced_algebraic_join_filter",
            args.five_a_five_c_target_forced_algebraic_join_filter_probe,
        ),
        (
            "5a5c_compact_preendpoint_s3_ffe_pushdown",
            args.five_a_five_c_compact_preendpoint_s3_ffe_pushdown_probe,
        ),
        (
            "5a5c_actual_deck_nonmergeable_target_pullback",
            args.five_a_five_c_actual_deck_nonmergeable_target_pullback_probe,
        ),
        (
            "5a5c_scalar_target_norm_count_circuit",
            args.five_a_five_c_scalar_target_norm_count_circuit_probe,
        ),
        (
            "5a5c_noncharacter_algebraic_target_norm_resultant",
            args.five_a_five_c_noncharacter_algebraic_target_norm_resultant_probe,
        ),
        (
            "5a5c_factored_elliptic_lambda_ring_chow_norm",
            args.five_a_five_c_factored_elliptic_lambda_ring_chow_norm_probe,
        ),
        (
            "5a5c_poincare_theta_target_section_rank",
            args.five_a_five_c_poincare_theta_target_section_rank_probe,
        ),
        (
            "5a5c_theta_addition_cancellation_network",
            args.five_a_five_c_theta_addition_cancellation_network_probe,
        ),
        (
            "5a5c_finite_deck_alternant_annihilator",
            args.five_a_five_c_finite_deck_alternant_annihilator_probe,
        ),
        (
            "5a5c_gauge_normalized_endpoint_query2p1",
            args.five_a_five_c_gauge_normalized_endpoint_query2p1_probe,
        ),
        (
            "5a5c_nonlinear_elliptic_orbit_product",
            args.five_a_five_c_nonlinear_elliptic_orbit_product_probe,
        ),
        (
            "5a5c_transposed_nonuniform_c5_leaf_generator",
            args.five_a_five_c_transposed_nonuniform_c5_leaf_generator_probe,
        ),
        (
            "relation_arity_factor_base_transposed_interface_rebalance",
            args.relation_arity_factor_base_transposed_interface_probe,
        ),
        (
            "m6_a6_batched_c3_pair_sum_source_locator",
            args.m6_a6_batched_c3_pair_sum_source_locator_probe,
        ),
        (
            "m6_target_batched_c3_elliptic_transpose",
            args.m6_target_batched_c3_elliptic_transpose_probe,
        ),
        (
            "m6_nonlinear_value_sensitive_c6_source_locator",
            args.m6_nonlinear_value_sensitive_c6_source_locator_probe,
        ),
        (
            "m6_output_sensitive_nonlinear_c5_source_index",
            args.m6_output_sensitive_nonlinear_c5_source_index_probe,
        ),
        (
            "m6_suboutput_implicit_c5_character_pairing",
            args.m6_suboutput_implicit_c5_character_pairing_probe,
        ),
        (
            "m6_small_k_multiplicative_c5_moment_torus",
            args.m6_small_k_multiplicative_c5_moment_torus_probe,
        ),
        (
            "torus_c5_explicit_split_global_rebalance",
            args.torus_c5_explicit_split_global_rebalance_probe,
        ),
        (
            "torus_c5_fourier_product_resultant",
            args.torus_c5_fourier_product_resultant_probe,
        ),
        (
            "torus_c5_linear_sketch_circulant",
            args.torus_c5_linear_sketch_circulant_probe,
        ),
        (
            "torus_c5_prime_order_homomorphic_fingerprint",
            args.torus_c5_prime_order_homomorphic_fingerprint_probe,
        ),
        (
            "torus_c5_explicit_hash_correction_support",
            args.torus_c5_explicit_hash_correction_support_probe,
        ),
        (
            "torus_c5_bucket_resultant_routing_tradeoff",
            args.torus_c5_bucket_resultant_routing_tradeoff_probe,
        ),
        (
            "torus_c5_rational_selector_degree",
            args.torus_c5_rational_selector_degree_probe,
        ),
        (
            "torus_c5_piecewise_selector_decision_dag",
            args.torus_c5_piecewise_selector_decision_dag_probe,
        ),
        (
            "torus_c5_sparse_fourier_predicate_transfer",
            args.torus_c5_sparse_fourier_predicate_transfer_probe,
        ),
        (
            "torus_c5_consecutive_mode_predicate",
            args.torus_c5_consecutive_mode_predicate_probe,
        ),
        (
            "torus_c5_base_field_frobenius_predicate_dag",
            args.torus_c5_base_field_frobenius_predicate_dag_probe,
        ),
        (
            "torus_c5_sparse_monomial_root_bound",
            args.torus_c5_sparse_monomial_root_bound_probe,
        ),
        (
            "torus_c5_two_atom_geometric_progression",
            args.torus_c5_two_atom_geometric_progression_probe,
        ),
        (
            "torus_c5_khatri_rao_kruskal_amplification",
            args.torus_c5_khatri_rao_kruskal_amplification_probe,
        ),
        (
            "torus_c5_all_nonzero_path_product",
            args.torus_c5_all_nonzero_path_product_probe,
        ),
        (
            "torus_c5_binomial_node_union_depth",
            args.torus_c5_binomial_node_union_depth_probe,
        ),
        (
            "torus_c5_chebotarev_fiber_cover",
            args.torus_c5_chebotarev_fiber_cover_probe,
        ),
        (
            "torus_c5_order_two_three_minor_rigidity",
            args.torus_c5_order_two_three_minor_rigidity_probe,
        ),
        (
            "torus_c5_order_two_four_minor_claw",
            args.torus_c5_order_two_four_minor_claw_probe,
        ),
        (
            "torus_c5_sextic_mobius_character_router",
            args.torus_c5_sextic_mobius_character_router_probe,
        ),
        (
            "torus_c5_adaptive_character_decision_router",
            args.torus_c5_adaptive_character_decision_router_probe,
        ),
        (
            "torus_c5_label_congruence_correction",
            args.torus_c5_label_congruence_correction_probe,
        ),
        (
            "m6_weighted_fiber_marginal_log_operator",
            args.m6_weighted_fiber_marginal_log_operator_probe,
        ),
        (
            "m6_weighted_c3_mobius_gcd_trace",
            args.m6_weighted_c3_mobius_gcd_trace_probe,
        ),
        (
            "m6_aggregate_marginal_singleton_source_equivalence",
            args.m6_aggregate_marginal_singleton_source_equivalence_probe,
        ),
        (
            "m6_occurrence_pair_resultant_local_valuation",
            args.m6_occurrence_pair_resultant_local_valuation_probe,
        ),
        (
            "m6_static_3sum_indexing_tradeoff",
            args.m6_static_3sum_indexing_tradeoff_probe,
        ),
        (
            "m6_actual_c6_shift_krylov_rank",
            args.m6_actual_c6_shift_krylov_rank_probe,
        ),
        (
            "m6_rational_convolution_subalgebra_rigidity",
            args.m6_rational_convolution_subalgebra_rigidity_probe,
        ),
        (
            "m6_matrix_free_marginal_jacobian_krylov",
            args.m6_matrix_free_marginal_jacobian_krylov_probe,
        ),
        (
            "m6_geometry_only_weight_interpolation_adjoint",
            args.m6_geometry_only_weight_interpolation_adjoint_probe,
        ),
        (
            "m6_symmetric_shift_reverse_only_marginal",
            args.m6_symmetric_shift_reverse_only_marginal_probe,
        ),
        (
            "m6_signed_quotient_multiscale_rank",
            args.m6_signed_quotient_multiscale_rank_probe,
        ),
        (
            "m6_singleton_relation_hypergraph_rank",
            args.m6_singleton_relation_hypergraph_rank_probe,
        ),
        (
            "m6_a_diversity_projective_rank",
            args.m6_a_diversity_projective_rank_probe,
        ),
        (
            "m6_hash_to_curve_projective_rank",
            args.m6_hash_to_curve_projective_rank_probe,
        ),
        (
            "m6_short_relation_near_injectivity_supply",
            args.m6_short_relation_near_injectivity_supply_probe,
        ),
        (
            "m6_random_diagonal_known_target_rank",
            args.m6_random_diagonal_known_target_rank_probe,
        ),
        (
            "m6_positive_c6_generic_locator_reduction",
            args.m6_positive_c6_generic_locator_reduction_probe,
        ),
        (
            "m6_signed_c3_divisor_translation_gcd",
            args.m6_signed_c3_divisor_translation_gcd_probe,
        ),
        (
            "m6_batch_inverse_transpose_modcomp_fit",
            args.m6_batch_inverse_transpose_modcomp_fit_probe,
        ),
        (
            "m6_aggregate_union_factor_label_recovery",
            args.m6_aggregate_union_factor_label_recovery_probe,
        ),
        (
            "m6_randomized_target_divisor_norm_union",
            args.m6_randomized_target_divisor_norm_union_probe,
        ),
        (
            "m6_global_randomizer_elliptic_translate_product",
            args.m6_global_randomizer_elliptic_translate_product_probe,
        ),
        (
            "m6_kummer_x_translate_signed_verification",
            args.m6_kummer_x_translate_signed_verification_probe,
        ),
        (
            "m6_generalized_target_divisor_weil_reciprocity_swap",
            args.m6_generalized_target_divisor_weil_reciprocity_swap_probe,
        ),
        (
            "m6_log_derivative_elliptic_cauchy_trace",
            args.m6_log_derivative_elliptic_cauchy_trace_probe,
        ),
        (
            "m6_regularized_log_trace_displacement_rank",
            args.m6_regularized_log_trace_displacement_rank_probe,
        ),
        (
            "m6_lambda_zero_fitting_target_norm_dedup",
            args.m6_lambda_zero_fitting_target_norm_dedup_probe,
        ),
        (
            "m6_balanced_miller_tree_norm_streaming",
            args.m6_balanced_miller_tree_norm_streaming_probe,
        ),
        (
            "m6_target_sign_conjugate_s3_self_resultant",
            args.m6_target_sign_conjugate_s3_self_resultant_probe,
        ),
        (
            "m6_s3_determinantal_transfer_noncommutativity",
            args.m6_s3_determinantal_transfer_noncommutativity_probe,
        ),
        (
            "m6_confluent_signed_dual_chow_pushforward",
            args.m6_confluent_signed_dual_chow_pushforward_probe,
        ),
        (
            "m6_scalar_subset_incidence_group_testing",
            args.m6_scalar_subset_incidence_group_testing_probe,
        ),
        (
            "m6_principal_target_pontryagin_resultant",
            args.m6_principal_target_pontryagin_resultant_probe,
        ),
        (
            "m6_global_marked_fitting_locator",
            args.m6_global_marked_fitting_locator_probe,
        ),
        (
            "m6_marked_fitting_signed_norm_dedup",
            args.m6_marked_fitting_signed_norm_dedup_probe,
        ),
        (
            "m6_squarefree_truncated_resultant_applicability",
            args.m6_squarefree_truncated_resultant_applicability_probe,
        ),
        (
            "m6_d5_directed_evaluation_survivor",
            args.m6_d5_directed_evaluation_survivor_probe,
        ),
        (
            "m6_monogenic_kernel_bidegree",
            args.m6_monogenic_kernel_bidegree_probe,
        ),
        (
            "m6_gcd_equivalent_target_subresultant",
            args.m6_gcd_equivalent_target_subresultant_probe,
        ),
        (
            "m6_sparse_projector_prony_locator",
            args.m6_sparse_projector_prony_locator_probe,
        ),
    ):
        if path.exists():
            frontier_preflights[name] = (read_json(path), path)
    report = build_report(
        payload,
        args.input,
        fixed_config_name=args.fixed_config,
        focus_budget=args.focus_budget,
        audit=audit,
        note_url=args.note_url,
        frontier_preflights=frontier_preflights,
    )
    summation_ffe_inventory, summation_ffe_replay_plan = build_summation_ffe_artifacts(
        payload, note_url=args.note_url
    )
    write_json(args.output, report)
    write_json(args.summation_ffe_evidence_inventory, summation_ffe_inventory)
    write_json(args.summation_ffe_replay_plan, summation_ffe_replay_plan)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text(render_note(report), encoding="utf-8")
    summary = report["summary"]
    print(
        f"claim={report['claim_status']} full={summary['full_cell_count']} "
        f"rank={summary['natural_full_rank_cell_count']} "
        f"logs={summary['natural_verified_log_cell_count']} "
        f"below_rho={summary['natural_below_rho_cell_count']} "
        f"focus={','.join(row['id'] for row in report['focus_queue'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
