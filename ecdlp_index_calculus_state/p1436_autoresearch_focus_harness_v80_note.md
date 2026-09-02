# P1436 autoresearch focus report

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

Natural full cells: exact `1/1`, full-rank `1/1`, verified logs `1/1`, below-rho `0/1`.

Stored-but-unusable / oracle-rank-headroom / fixed-rank-headroom cells: `0/0/0`.
Summation/FFE evidence markers: `0` cells with `0` markers, exact-replay ready for `0` cells and missing exact inputs for `0` cells; R68 lane-admitted `0` and blocked `0`.
Bound frontier preflights: `67`; currently closed lanes: `5a5c_actual_deck_nonmergeable_target_pullback, 5a5c_actual_divisor_image_entropy_merge, 5a5c_aggregate_veronese_projector_recurrence, 5a5c_black_box_resultant_localizer, 5a5c_compact_elliptic_subfunction_map, 5a5c_compact_preendpoint_s3_ffe_pushdown, 5a5c_coordinate_filtration, 5a5c_factored_elliptic_lambda_ring_chow_norm, 5a5c_factored_transposed_projector_trace, 5a5c_finite_deck_alternant_annihilator, 5a5c_fixed_marker_scalar_recurrence, 5a5c_gauge_normalized_endpoint_query2p1, 5a5c_implicit_veronese_hyperplane_source_index, 5a5c_jet_preserving_addition_pushforward, 5a5c_marked_resultant_source_section, 5a5c_modular_frobenius_trace_recurrence, 5a5c_multiedge_digitized_equality_projector, 5a5c_noncharacter_algebraic_target_norm_resultant, 5a5c_nonlinear_elliptic_orbit_product, 5a5c_nonlinear_tensor_tower_trace, 5a5c_nonlocal_moment_hankel_translation, 5a5c_poincare_theta_target_section_rank, 5a5c_scalar_target_norm_count_circuit, 5a5c_shared_semilinear_incidence_correspondence, 5a5c_sparse_multihomogeneous_moment_recurrence, 5a5c_succinct_aggregate_digit_trie, 5a5c_target_forced_algebraic_join_filter, 5a5c_target_uniform_precoefficient_circuit, 5a5c_theta_addition_cancellation_network, 5a5c_transposed_nonuniform_c5_leaf_generator, 5a5c_two_sided_implicit_join, 5a5c_unequal_list_subfunction_inversion, actual_s6_fermat_tensor_train, batched_nested_norm_node_compiler, cartesian_sum_compact_divisor, constructive_closure_collision, factor_line_direct_root_equivalence, full_multiplicative_x_coset_endpoint, idea340_public_chart, m6_a6_batched_c3_pair_sum_source_locator, m6_nonlinear_value_sensitive_c6_source_locator, m6_output_sensitive_nonlinear_c5_source_index, m6_small_k_multiplicative_c5_moment_torus, m6_suboutput_implicit_c5_character_pairing, m6_target_batched_c3_elliptic_transpose, multiplicative_x_s3_closure, presurface_full_charge, relation_arity_factor_base_transposed_interface_rebalance, resultant_valuation_trace_grammar, s4_centered_carry_rank, s6_centered_carry_rank_minor, s6_iterated_norm_support, s6_residual_decision_diagram, s6_subset_incidence_mobius, scalar_only_nested_norm_slp, slice_quadratic_public_source, target_translated_frequency_orbit, torus_c5_bucket_resultant_routing_tradeoff, torus_c5_consecutive_mode_predicate, torus_c5_explicit_hash_correction_support, torus_c5_explicit_split_global_rebalance, torus_c5_fourier_product_resultant, torus_c5_linear_sketch_circulant, torus_c5_piecewise_selector_decision_dag, torus_c5_prime_order_homomorphic_fingerprint, torus_c5_rational_selector_degree, torus_c5_sparse_fourier_predicate_transfer`.

Target-policy fixed fraction of oracle rank headroom: `None`. Prospective transfer: `invalid_exactness_or_missing_fixed_route`.

Matched hash-policy specificity control: `matched_hash_controls_unavailable`.

Audit-bound promotion allowed: `False`.

Shoup pressure status: `insufficient_comparable_cells` (passes gate: False)
Shoup gate inputs: 1 exact+verified cells, 1 scales, threshold exponent < 0.5 and residual <= 1.0.

## Shoup Pressure

Scale coverage: `1` (required `2`), status: `insufficient_comparable_cells`.
Fitted exponent (log charged operations vs log group order): `None`; max abs residual log: `None`.

Method source: https://x.com/askalphaxiv/status/2076737985559822734?s=46
Tweet source captured: https://x.com/askalphaxiv/status/2076737985559822734 (text_included=True)
Tweet source summary: Introducing autoresearch with GPT 5.6

We had GPT 5.6 Sol reproduce the key findings from “Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning”

Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, critical experiments and spent less time on peripheral details. It also asked fewer “clarification” questions and independently resolved ambiguities instead of pushing them back to us

@OpenAI pushing the boundaries of the automated research loop with models that don’t have handcuffs
Tweet source summary verbatim? True
Tweet posted at: 2026-07-13T18:38:14.340000+00:00
Tweet intake mode: public_snapshot_summary
Tweet source title: Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning (source: https://arxiv.org/abs/2607.08393)
Paper title as written in post: Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning
Tweet media: 1 items, has_media=True (video)
Tweet hashtags: none

## Next Action
1. `s23_lacunary_order2_torus_c5_selector_predicate`: Construct or refute a lacunary predicate in the actual ord_q(characteristic)=2 fields, or a non-Fourier shared decision DAG. It must exploit more than a dense consecutive mode block, choose a valid C2 branch in polylogarithmic arbitrary-target work, return exact C2+C3 sources or an empty certificate, fit B^(9/4+o(1)) state, avoid field DLP, and include rank, logs, identical descent, memory, field-operation, and bit costs.
   Decisive test: Freeze every lacunary mode, SLP node, or non-Fourier DAG node, the exact empty path, and reverse C2+C3 source pointer. Replay the circuit in each actual field before composing with rank, logs, and identical descent.
   Falsifier: The representation expands to a dense B^(15/4) mode block, scans B^(3/2) branches, stores B^(15/4) targets, hides a DLP, permits false answers, or fails five sources, rank, logs, descent, memory, operations, or bit costs.
   Required artifacts: frozen_torus_c5_consecutive_mode_predicate.json, torus_c5_consecutive_mode_predicate_cost_ledger.json, torus_c5_consecutive_mode_predicate_replay.json, torus_c5_consecutive_mode_predicate_controls.json, factor_logs_and_identical_descent_r131.json

## Focus Queue

1. `s23_lacunary_order2_torus_c5_selector_predicate`: Construct or refute a lacunary predicate in the actual ord_q(characteristic)=2 fields, or a non-Fourier shared decision DAG. It must exploit more than a dense consecutive mode block, choose a valid C2 branch in polylogarithmic arbitrary-target work, return exact C2+C3 sources or an empty certificate, fit B^(9/4+o(1)) state, avoid field DLP, and include rank, logs, identical descent, memory, field-operation, and bit costs.
   Hypothesis: The actual ord_q(characteristic)=2 pairing fields admit a lacunary high-degree low-SLP predicate or a non-Fourier shared decision DAG with polylogarithmic arbitrary-target evaluation.
   Falsifier: The representation expands to a dense B^(15/4) mode block, scans B^(3/2) branches, stores B^(15/4) targets, hides a DLP, permits false answers, or fails five sources, rank, logs, descent, memory, operations, or bit costs.
2. `end_to_end_cost_reduction`: Reduce charged source generation, retained residual traffic, or matrix work without changing the frozen evidence set.
   Hypothesis: The verified natural pipeline can cross the charged rho gate without changing its evidence set.
   Falsifier: The fitted total exponent is not below the preregistered gate or any cell exceeds rho.
3. `shoup_pressure_scaling_probe`: Fit charged natural-route operations versus group order and check explicit Shoup-pressure residuals.
   Hypothesis: Single-scale experiments can hide generic-bound pressure. A stable multiscale cost fit is required before claiming any asymptotic Shoup-bound progress.
   Falsifier: Any fitted exponent is >= 0.5, or multiscale points are not exactly comparable, or scaling residuals invalidate the fit.

## Routing Controls

Matched curves / positive coordinate-specific excess: `0/0`.

P1436 configurations rerun the collector under different frozen routing rules; they do not patch an identical intermediate EC state. Their deltas are matched configuration ablations, not causal self-patching evidence.

## Autoresearch Guidance
Source: https://x.com/askalphaxiv/status/2076737985559822734?s=46 (canonical https://x.com/askalphaxiv/status/2076737985559822734)
Bounded critical set: True - Keep critical experiments bounded to the smallest unresolved bottlenecks.
Deterministic non-blocking ambiguity handling: True - Record non-blocking ambiguities as deterministic resolutions and keep them diagnostic-only.
Peripheral scope deference: True - Defer peripheral branches until bounded critical queue is exhausted or falsified.
Operator interrupt policy: True - blocking uncertainty requires operator action before any promotion.
Experiment lineage source: https://github.com/alphaXiv/openresearch-cli (logical lineage only; no branch/run materialization is claimed).

## Guidance Compliance
Bounded critical set enforced: True
Peripheral scope deferral enforced: True
Non-blocking ambiguity resolutions recorded: True
Operator interrupt alignment: True
Selected focus candidates fully specified: True

## Experiment Lineage
Baseline / queued / deferred nodes: `1/3/0`; mode `logical_plan_only`.
This report records parentage and evidence contracts only. A node becomes executed evidence only after a separate branch/run receipt binds its artifacts to the immutable source hash.

## Major Result Replication
Fully replicated natural cells: 0/1
Per-stage status counts: {'replay_and_exact_residual': {'passed': 1}, 'collision_supply': {'passed': 1}, 'cross_shift_routing': {'passed': 1}, 'source_row_elimination': {'passed': 1}, 'relation_rank': {'passed': 1}, 'rhs_compatibility': {'passed': 1}, 'factor_log_verification': {'passed': 1}, 'target_descent': {'passed': 1}, 'total_cost': {'blocked': 1}}

## Deferred Experiments

- None.

## Ambiguity Resolutions

- `self_patching_fidelity`: Label every delta a matched configuration ablation and prohibit causal self-patching language. Blocks promotion: `False`.
- `paper_headroom_scope`: Recompute headroom from P1436 rank fractions; never import the paper's numeric range as an ECDLP gate. Blocks promotion: `False`.
- `posthoc_oracle_scope`: Break score ties by configuration name and keep every oracle choice diagnostic-only. Blocks promotion: `False`.
- `synthetic_null_scope`: Use the synthetic stream for occupancy only; assign no rows, logs, or descent credit. Blocks promotion: `False`.
- `fixed_route_availability`: Fall back to the natural route and mark the fixed intervention unavailable. Blocks promotion: `False`.
- `experiment_lineage_materialization`: Label lineage logical-plan-only until a separate branch/run receipt binds artifacts to the source hash. Blocks promotion: `False`.
- `ffe_information_conservation_scope`: Assign zero information credit to product/summation quotient rows and require a scalar-blind new-factor-row source below direct pair-complement cost. Blocks promotion: `False`.
- `constructive_closure_collision_scope`: Assign zero rank-reduction credit to fresh residual rows and require a prospective coordinate-defined collision locator below both pair materialization and rho. Blocks promotion: `False`.
- `multiplicative_x_s3_scope`: Close that frozen prefix candidate and do not treat FFT-compatible coordinates as a source locator without a prospective density theorem and explicit sub-pair algorithm. Blocks promotion: `False`.
- `s4_centered_carry_scope`: Close only the named k=1 S4 canonical-carry CP precursor. Preserve S6-specific, noncanonical, and non-CP interfaces, and require an exact branch-complete source router with direct costs. Blocks promotion: `False`.
- `s6_centered_carry_scope`: Close those two explicit S6 carry lifts. Route work to R9/R10's non-CP exact trace contraction, preserving other lifts only when accompanied by a new cancellation and source theorem. Blocks promotion: `False`.
- `resultant_valuation_trace_scope`: Close only the frozen resultant-valuation grammar: its expanded state is B^4, specialized queries cost B^2 each, and its actual S6 extension emits B^3 triple occurrences. Preserve structured quotient-algebra transducers and unrestricted circuits. Blocks promotion: `False`.
- `s6_residual_decision_diagram_scope`: Close only squarefree residual-radical memoization. Assign zero constructive credit to the tiny oracle diagram because building its signatures presupposes relation incidences; route work to a support-adaptive constructor with blind zero certificates. Blocks promotion: `False`.
- `s6_iterated_norm_support_scope`: Close only expanded iterated-norm/remainder representations. Preserve a transposed scalar functional that never materializes the coefficient cube or B^3 values and still returns exact multiplicity, zero certificates, children, and one source. Blocks promotion: `False`.
- `s6_subset_incidence_mobius_scope`: Close only the explicit subset-histogram representation and do not promote its exact finite counts. Route the next bounded test to a target-translated frequency oracle that avoids both prefix-triple and target-pair enumeration under direct caps. Blocks promotion: `False`.
- `target_translated_frequency_orbit_scope`: Close only universal exact linear shift-equivariant frequency sketches. Keep nonlinear target-specialized nested resultants open, and require them to replay R76 multiplicity, zero, source, children, and direct costs without character coordinates. Blocks promotion: `False`.
- `actual_s6_fermat_tensor_train_scope`: Close only the value-first exact TT and binary-Hadamard Fermat grammar. Keep scalar-only straight-line nested norms open, but expose and charge every node and require R76 exact count, source, multiplicity, blind zero, and child replay. Blocks promotion: `False`.
- `scalar_only_nested_norm_slp_scope`: Close only the explicit scalar-leaf product/norm SLP. Route the remaining possibility to a batched norm-node compiler whose resultants, subproducts, remainders, transposition, source, and child operations all receive field-level receipts. Blocks promotion: `False`.
- `batched_nested_norm_node_compiler_scope`: Close only the standard materialized S4 subproduct/gcd compiler. Route the residual to one scalar-blind structured factor-base geometry with subcap triple-endpoint compression, prospective density, matched rank controls, factor logs, and identical target descent. Blocks promotion: `False`.
- `full_multiplicative_x_coset_endpoint_scope`: Close only complete one-dimensional multiplicative-x cosets with the canonical lift mask and cofactor map. Route the remaining test to a compact divisor or straight-line factor-base description whose S4 endpoint compiler, source query, density, rank, logs, and identical descent are all prospectively charged. Blocks promotion: `False`.
- `cartesian_sum_compact_divisor_scope`: Preserve the passing local S4 compiler, close only the Cartesian addition-pushforward full pipeline under explicit equality joins and quotient-free generic collisions, and route the residual to one addition-compatible field filtration with exact FFE or summation-polynomial sources. Blocks promotion: `False`.
- `5a5c_coordinate_filtration_scope`: Close only explicit coordinate buckets and target-coupled offset replay. Preserve target-forced algebraic selectors and route the next test to a marked resultant with exact containment and one jointly coupled source section. Blocks promotion: `False`.
- `5a5c_marked_resultant_source_section_scope`: Close only explicit dense endpoint-polynomial and packed-interpolant source sections. Preserve P1510's output-sensitive positive control and route the residual to one target-uniform circuit before coefficient or provenance-leaf emission. Blocks promotion: `False`.
- `5a5c_target_uniform_precoefficient_circuit_scope`: Close fixed exact target-equivariant quotients and the bound standard circuit grammars only. Route the surviving target-specialized mechanism to P1514's explicit sparse multihomogeneous moment-constructor exception. Blocks promotion: `False`.
- `5a5c_sparse_multihomogeneous_moment_scope`: Close supplied-jet decoding and the charged standard multigraded constructors only. Preserve the compositional exception and route it to one jet-preserving A+C addition-pushforward intertwiner with full source and exceptional-chart replay. Blocks promotion: `False`.
- `5a5c_jet_pushforward_scope`: Close target-local jet propagation and explicit translated remainders only. Preserve the fixed-marker and implicit scalar exceptions, and route one black-box translated resultant/gcd source localizer with every representation and exceptional chart charged. Blocks promotion: `False`.
- `5a5c_black_box_resultant_scope`: Close materialized half-gcd and the frozen scalar/block Krylov grammar only. Preserve non-Krylov arithmetic circuits and route the residual to one coefficient-free fixed-marker scalar resultant recurrence with exact multiplicity and projective source replay. Blocks promotion: `False`.
- `5a5c_fixed_marker_scalar_recurrence_scope`: Close target-local fixed-marker jets and explicit shift or slot-support recurrences only. Preserve target-independent nonlocal nonlinear states, and require one frozen deck-update and translation law below both caps with exact source and exceptional-branch replay. Blocks promotion: `False`.
- `5a5c_nonlocal_moment_hankel_translation_scope`: Close only exponential-moment, Newton, and Hankel/Padé translation states, whose exact order is C^5=B^3. Preserve non-moment source-reporting indices and route one unequal-list subfunction-inversion construction with setup, query, reporting, memory, and exceptional branches fully charged. Blocks promotion: `False`.
- `5a5c_unequal_list_subfunction_inversion_scope`: Close direct applications of the bound unequal-list and balanced kSUM theorems to explicit endpoint lists only. Preserve a compact elliptic subfunction decomposition and require public MAP1, MAP2, f_d, and TR circuits acting directly on D_A,D_C with all costs and projective branches charged. Blocks promotion: `False`.
- `5a5c_compact_elliptic_subfunction_map_scope`: Close direct endpoint partitions with D independently preprocessed generic subfunctions only. Preserve target-dependent overlapping incidence correspondences with a jointly compressed shared operator, and require exact projective source unranking with all overlap, elimination, state, and fresh-work costs charged. Blocks promotion: `False`.
- `5a5c_shared_semilinear_incidence_scope`: Close materialized per-source Veronese rows and low-rank value-kernel arguments only. Preserve an implicit Cartesian algebraic range index that acts before feature emission, and require exact coupled source return with all projective and nonreduced branches charged. Blocks promotion: `False`.
- `5a5c_implicit_veronese_hyperplane_index_scope`: Close those three standard routes only. Full finite projector rank does not refute a nonlinear aggregate trace, so route one exact cap-sized projector-count recurrence with dyadic coupled-source and complete exceptional-branch replay; withhold every rank, log, descent, and Shoup claim. Blocks promotion: `False`.
- `5a5c_aggregate_veronese_projector_scope`: Close only the explicit all-monomial moment contraction. Treat the finite full-rank sweep as diagnostic rather than an asymptotic lower bound, and route one modular Frobenius trace with explicit state transitions, integer lifting, dyadic source return, and all costs and branches charged. Blocks promotion: `False`.
- `5a5c_modular_frobenius_trace_scope`: Close only explicit quotient bases, matrices, and dyadic range quotients. Preserve a factored transposed trace that never emits those bodies, but require an explicit adjoint identity, integer lift, range restriction, multiplicity-complete source return, and full cost ledger. Blocks promotion: `False`.
- `5a5c_factored_transposed_projector_scope`: Close pointwise Fermat powering, source-valued product trees, reverse adjoints, and linearized dyadic masks only. The finite full-rank sweep is not an asymptotic lower bound; route one nonlinear tensor-tower trace whose node states arise directly from compact A/C divisor circuits. Blocks promotion: `False`.
- `5a5c_nonlinear_tensor_tower_scope`: Close one-bond contractions even with arbitrary nonlinear local encoders. Preserve multi-edge digitized encodings and actual-image restrictions, but require a frozen algebraic extractor, injectivity, total cut capacity, no p-size advice, and complete occurrence-source replay. Blocks promotion: `False`.
- `5a5c_multiedge_digitized_projector_scope`: Credit the small-edge equality representation but close full-field tables, fiber root/coefficient tables, and per-source extraction. Route only a leaf-free aggregate digit trie compiled directly from compact A/C divisor circuits, with all merge, query, and source costs charged. Blocks promotion: `False`.
- `5a5c_succinct_aggregate_digit_trie_scope`: Close only universal arbitrary-set indices and explicit occurrence tries. Do not transfer that bound to R84's structured 3A+2C image; route an actual-image theorem that either supplies a leaf-free mergeable summary or an injective reachable high-entropy subfamily. Blocks promotion: `False`.
- `5a5c_actual_divisor_image_entropy_merge_scope`: Credit both structured local oracles inside the direct caps, but do not credit a 5A+5C relation source. Route one two-sided implicit join requiring exact joint multiplicity and one coupled source without enumerating either side. Blocks promotion: `False`.
- `5a5c_two_sided_implicit_join_scope`: Close direct meet-in-the-middle splits and extra disjoint or independent filters only. Preserve target-forced algebraic identities that every true L+R=T pair satisfies, but require compact pre-endpoint evaluation, exact multiplicity/source, and full false-positive costs. Blocks promotion: `False`.
- `5a5c_target_forced_algebraic_join_filter_scope`: Close pointwise S3, explicit endpoint-polynomial resultants, and materialized FFE factor lists only. Route one sign-resolved target-specialized pushdown through the compact divisor circuit before endpoint or provenance-leaf emission, with field-level state and source receipts. Blocks promotion: `False`.
- `5a5c_compact_preendpoint_s3_ffe_pushdown_scope`: Close explicit residual sets and universal mergeable child summaries only. Preserve one actual-deck-specific, non-mergeable target circuit that injects T before any child summary, while requiring fully expanded gate, source-adjoint, matched-random, and exceptional receipts. Blocks promotion: `False`.
- `5a5c_actual_deck_nonmergeable_target_pullback_scope`: Credit the source adjoint only after a scalar circuit is supplied, and assign no cost credit to the verifier's B^5 source product. Route the remaining work solely to a compact scalar target norm/count constructor with generic multiplicity and integer-lift gates. Blocks promotion: `False`.
- `5a5c_scalar_target_norm_count_circuit_scope`: Close explicit complex-character diagonalizations and equivalent mode tables only. Preserve a non-character target-injected algebraic norm/resultant circuit, while forbidding DLP coordinates and requiring every degree, marker, integer-lift, and exceptional cost receipt. Blocks promotion: `False`.
- `5a5c_noncharacter_algebraic_target_norm_resultant_scope`: Close explicit coefficient, Sylvester/subresultant, quotient-free cofactor, and sparse Macaulay grammars only. Preserve a factored elliptic lambda-ring/Chow recurrence that computes the canonical norm directly, with no general arithmetic-circuit lower-bound claim. Blocks promotion: `False`.
- `5a5c_factored_elliptic_lambda_ring_chow_norm_scope`: Admit the canonical cycle-weight correction and reject termwise norm evaluation. Preserve only a Poincare/theta factorization of the pulled-back target section; the theorem-of-the-cube line-bundle identity alone receives no scalar-evaluator credit. Blocks promotion: `False`.
- `5a5c_poincare_theta_target_section_rank_scope`: Close regular pure pairwise products and uniform separated section state only. Preserve an exact rational finite-field theta-addition circuit that generates high rank implicitly; require explicit pole cancellation and contraction costs. Blocks promotion: `False`.
- `5a5c_theta_addition_cancellation_network_scope`: Admit the exact target predicate and close analytic-only, unshifted-repeat, and uncharged exceptional-chart concerns. Preserve only a finite-deck annihilator which returns an existence bit and coupled source without the B^5 source body, B^(12/5) incidence, or global Fermat row mode. Blocks promotion: `False`.
- `5a5c_finite_deck_alternant_annihilator_scope`: Close raw univariate and standard row- or column-conditioned annihilators only. Give no constructor credit to low rank measured after forming the mask, and preserve the gauge-normalized endpoint Query2P1 index as a representation-sensitive exception with full source-unranking charges. Blocks promotion: `False`.
- `5a5c_gauge_normalized_endpoint_query2p1_scope`: Admit typed endpoint normalization and the thin setup tables. Close direct hashing, pair materialization, and the standard IDEA-012 R3/R4 indexing/resultant grammars only. Preserve one gauge-invariant nonlinear orbit-product recurrence with explicit order, target update, and source adjoint. Blocks promotion: `False`.
- `5a5c_nonlinear_elliptic_orbit_product_scope`: Close fixed translation orbits, standard product trees, and bounded prefix state as a work claim. Preserve only a transposed nonuniform C5 leaf generator with a charged target specialization and exact source adjoint. Blocks promotion: `False`.
- `5a5c_transposed_nonuniform_c5_leaf_generator_scope`: Close standard product reverse AD, first/second derivative localization as a cheap-constructor claim, and canonical typed preleaf propagation only. Preserve a broader relation-arity and asymmetric factor-base exponent rebalance with all source-to-target costs charged. Blocks promotion: `False`.
- `relation_arity_factor_base_transposed_interface_rebalance_scope`: Admit the vertex as a necessary planning envelope only. Give no algorithm, relation-rank, log, descent, or Shoup credit until an implicit S7/S13 3F self-convolution returns six coupled sources without half-occurrence enumeration. Blocks promotion: `False`.
- `m6_a6_batched_c3_pair_sum_source_locator_scope`: Admit the exact reduction and source semantics only. Close the bound integer-indexing, explicit translated-divisor, materialized resultant, and group-algebra routes at the frozen caps. Preserve one jointly transposed elliptic coefficient functional with a source adjoint. Blocks promotion: `False`.
- `m6_target_batched_c3_elliptic_transpose_scope`: Close universal characteristic-zero linear shift sketches, single regular rational target sections, and the charged current k=7 routes only. Do not infer a base-field nonlinear circuit or data-structure lower bound; preserve a target-specialized value-sensitive six-C source locator with explicit branching and reverse backpointers. Blocks promotion: `False`.
- `m6_nonlinear_value_sensitive_c6_source_locator_scope`: Admit the one-atom reduction and finite source semantics only. Close occurrence-materialized split tables and represented quotient/grid/resultant routes at the frozen caps, while preserving an endpoint-compressed, output-sensitive nonlinear C5 index. Do not infer an arbitrary data-structure or arithmetic-circuit lower bound. Blocks promotion: `False`.
- `m6_output_sensitive_nonlinear_c5_source_index_scope`: Admit the iid support theorem and finite source semantics only. Do not transfer the theorem to every filtered deck or infer an arithmetic-circuit, cell-probe, or RAM lower bound. Preserve a sub-output target-specialized nonlinear C5 membership/source circuit with exact empty certification and reverse projective backpointers. Blocks promotion: `False`.
- `m6_suboutput_implicit_c5_character_pairing_scope`: Admit the exact forward pairing semantics and finite source replay only. Charge extension, torsion, Miller, and final exponentiation work; give no generic credit to k=2 fixtures or unit-cost field DLP. Preserve a small-k sub-output multiplicative C5 circuit and keep pairing-unfriendly inputs as an explicit unsolved generic branch. Blocks promotion: `False`.
- `m6_small_k_multiplicative_c5_moment_torus_scope`: Admit the exact torus, moment, annihilator, and split source semantics only. Do not infer a general arithmetic-circuit or data-structure lower bound. Preserve a target-injected nonlinear torus C5 circuit outside full moments, explicit split scans, integer residue maps, and field DLP. Blocks promotion: `False`.
- `torus_c5_explicit_split_global_rebalance_scope`: Close only explicit occurrence/output split tables and their iid-distinct support analogues across arities. Do not transfer the iid theorem to filtered decks or infer a general circuit/data-structure lower bound. Preserve a target-specialized nonoccurrence torus C5 circuit. Blocks promotion: `False`.
- `torus_c5_fourier_product_resultant_scope`: Admit the exact Fourier and product-resultant semantics only. Close full mode, Prony/BM, represented target resultant, symbolic P5, and full-grid grammars. Preserve a nonrepresented target-specialized circuit and do not infer a general arithmetic-circuit or data-structure lower bound. Blocks promotion: `False`.
- `torus_c5_linear_sketch_circulant_scope`: Close only universal target-independent linear measurements with linear exact-count decoding. Preserve nonlinear preprocessing specialized jointly to (u^(*2),u^(*3)), nonlinear membership-only decoding, and adaptive data structures; infer no general lower bound. Blocks promotion: `False`.
- `torus_c5_prime_order_homomorphic_fingerprint_scope`: Close only pure single or finite-tuples of group homomorphisms. Preserve nonhomomorphic and adaptive fingerprints with explicit product-law correction data; infer no general data-structure or circuit lower bound. Blocks promotion: `False`.
- `torus_c5_explicit_hash_correction_support_scope`: Close only explicit per-bucket product lists and global exact product dictionaries. Preserve implicit correction circuits, adaptive probes, and nonlinear nonlisting certificates; infer no general circuit or data-structure lower bound. Blocks promotion: `False`.
- `torus_c5_bucket_resultant_routing_tradeoff_scope`: Close only independent all-pair, quotient-style, symbolic, and dense represented routing grammars. Preserve shared transposed evaluation and the cap-tight singleton-C3 constant-pair router; infer no general lower bound. Blocks promotion: `False`.
- `torus_c5_rational_selector_degree_scope`: Close only dense single rational selectors, explicit target tables, and densely represented branches. Preserve high-degree low-SLP and compact piecewise selectors because degree is not a circuit-size lower bound. Blocks promotion: `False`.
- `torus_c5_piecewise_selector_decision_dag_scope`: Close only sequential piecewise-constant branch scans and explicit B^(15/4) target routers. Preserve compact shared-predicate decision DAGs and high-degree low-SLP selectors; infer no general circuit or data-structure lower bound. Blocks promotion: `False`.
- `torus_c5_sparse_fourier_predicate_transfer_scope`: Reject unproved complex-to-finite uncertainty transfer. Preserve order-two finite-field Fourier analysis, direct non-Fourier shared predicates, and low-SLP selectors; infer no actual-field circuit or data-structure lower bound. Blocks promotion: `False`.
- `torus_c5_consecutive_mode_predicate_scope`: Close only dense consecutive-mode predicates and explicit root product trees at B^(15/4) state and sequential query cost. Preserve lacunary high-degree low-SLP predicates and non-Fourier shared DAGs; infer no general circuit or data-structure lower bound. Blocks promotion: `False`.
- `target_descent_absence`: Treat absent descent as untested, never as success. Blocks promotion: `False`.
- `independent_audit_binding`: Withhold promotion. Blocks promotion: `True`.
- `source_claim_status`: Never synthesize a breakthrough claim in post-processing. Blocks promotion: `True`.

## Boundary

Intervention-selected improvements are diagnostic only. Promotion requires the natural preregistered route to pass exact relation supply, full RHS-compatible rank, verified factor logs, separate target descent, the P1436 cost/exponent gate, any applicable R68 new-factor-row, R69 closure-collision, and applicable structured-source admission gates, and an independent audit cryptographically bound to this exact source JSON.
