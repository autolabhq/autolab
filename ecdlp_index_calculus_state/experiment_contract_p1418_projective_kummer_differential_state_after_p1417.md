# Experiment Contract: P1418 projective Kummer differential-state compiler after P1417

## Candidate

Replace P1417's full-column-rank CRT quotient action with an exact projective Kummer differential-addition representation. For each normalized pair state `A` and factor point `F`, construct and charge the augmented state `(X_A:Z_A, X_F:Z_F, X_{A-F}:Z_{A-F})`; use xADD to recover `x(A+F)` and preserve source witnesses only through a compressed state directory.

## Status

`CONJECTURE / PROJECTIVE-KUMMER / DIFFERENTIAL-ADDITION / PRIME-FIELD-GENERIC / TOY-EXPERIMENT`

## Hypothesis

At least one frozen coordinate factor-base family has persistent collisions or composable structure in its augmented pair-factor difference states, so exact triple outputs and source witnesses can be represented and queried in fewer than `B^2.5` field operations and less than `B^2.5` advice, with a fitted advantage of at least `0.05` over every matched hash base.

## Null hypothesis

The required known differences form `Theta(B^3)` distinct or witness-bearing states, xADD only changes constants, orientation labels restore the omitted information, or projective exceptional handling and memory exceed P1416/P1417 controls.

## Frozen inputs

- All nine P1416 random ordinary curves and eleven public factor bases at `B=ceil(r^(1/5))`.
- P1416 exact pair/triple state hashes, relation rows, factor-log solutions, and target descents.
- P1417 exact CRT action/root/tuple hashes and its independently audited full-column-rank boundary.
- Primary coordinate policies: interval, residue, rational-map images/compositions, and map union.
- Negative controls: three matched hash bases and public label shuffles.
- Stored factor logs, target scalars, relation outcomes, ranks, and held-out timings are forbidden selectors.

## Exact objects

- Preserve every normalized source pair witness before any x or projective quotient.
- For each pair point `A` and factor point `F`, construct full points `A-F` and `A+F` independently.
- Encode affine points as projective Kummer `(X:Z)` states, including explicit infinity, inverse, doubling, and repeated-x tags.
- Verify xADD output against the independently computed full point `A+F` for every augmented state.
- Measure deduplication separately for difference x, `(xA,xF,xDifference)`, output x, full augmented state, and source-witness-bearing state.
- Recover every retained triple output to exact y/sign orientation and source factors, then reproduce P1416 triple hashes and executed five-term query tuples.

## Policies

1. `p1416_explicit_affine_triples`: exact cubic control.
2. `projective_xadd_full_difference`: exact augmented-state control.
3. `difference_x_directory`: quotient by difference x with explicit orientation/source labels.
4. `augmented_kummer_directory`: quotient by `(xA,xF,xDifference)`.
5. `output_x_directory`: optimistic output quotient with exact witness recovery charged.
6. `matched_hash_augmented_directory`: identical construction on hash bases.

## Cost model

- Count EC additions/subtractions used to construct `A-F`; a known difference is never free.
- Count projective multiplications, squarings, additions, normalizations, exceptional fallbacks, orientation recovery, and exact replay.
- Count state keys, source labels, witness lists, projective coordinates, buffers, advice bits, peak memory, reads, and writes.
- Separate fixed-curve offline construction, per-row query work, supported targets, success probability, relation collection, sparse linear algebra, and individual target descent.
- Report `S*T^2/(epsilon*r)` and compare with sign-aware rho.

## Positive controls

- Every projective xADD state must equal the independently computed affine `x(A+F)`.
- The expanded augmented directory must reproduce P1416's full triple-state and source-witness hashes.
- Synthetic ordinary, inverse, doubling, infinity, repeated-x, and zero-difference cases must pass.
- Public label shuffling must preserve state counts and exact outputs.

## Negative controls

- Three matched hash bases at identical `B` and state dimensions.
- P1416 explicit triples with all `B^3` source witnesses charged.
- A no-difference x-only policy that is expected to fail exact composability.

## Metrics

- full/difference/augmented/output support counts and ordered multiplicities;
- exact xADD output and triple/source hashes;
- orientation labels, false merges, and invalid witness count;
- raw/projective field operations, state bytes, traffic, and peak memory;
- symbolic and fitted exponents in `B` and `r`;
- executed relation/target tuple equivalence, rank, sparse-LA cost, and descent success;
- offline/online fixed-curve tradeoff and `S*T^2/(epsilon*r)`.

## Success criterion

Every held-out curve for one frozen coordinate policy must satisfy:

- exact xADD, P1416 triple, relation, and target equivalence with zero invalid witnesses;
- complete projective exceptional-branch coverage;
- measured and symbolic augmented-state construction, source-witness advice, and query work below `B^2.5`;
- at least `0.05` fitted exponent advantage over every matched hash and explicit affine control;
- no worse peak-memory or traffic exponent than the best matched hash control;
- total end-to-end exponent below `0.5`, or a fixed-curve online frontier improvement with all offline work and advice reported.

## Falsification criterion

Narrow P1418 to a differential-state boundary if the exact augmented/source-witness state remains `Theta(B^3)`, every coordinate policy matches hash controls, xADD savings are only constant factor, or exact orientation recovery reconstructs the omitted cubic inventory. This does not rule out symmetric-square divisors, adaptive zero-product trees, or other non-enumerative representations.

## Red-team checks

- Never grant `x(A-F)` as free side information.
- Do not count output-x deduplication while omitting source-witness lists or orientation labels.
- Separate projective operation savings from state-count or exponent improvements.
- Include every inverse, doubling, infinity, repeated-x, and zero-difference branch.
- Prevent factor logs, relation hits, or target outcomes from choosing state quotients.
- Compare against identical hash construction and sign-aware rho.

## Reproduction command

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_p1418_projective_kummer_differential_state_after_p1417.py
```

## Next branch on failure

Formalize the measured known-difference state boundary, then test the symmetric-square degree-two divisor representation rather than another x-only relabeling.

## Results
- Timestamp: `2026-07-14T00:58:17.609258+00:00`.
- Claim status: `NEGATIVE_RESULT_P1418_KNOWN_DIFFERENCE_STATE_BOUNDARY`.
- Curves / policy-curves / queries: `9/99/297`.
- Exact xADD cells: `99/99`.
- Exact P1416 triple hashes: `99/99`.
- Exact query tuple sets: `297/297`.
- Invalid witnesses: `0`.
- Promotion audit: `{'policies': {'quadratic_image': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'mobius_image': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'quadratic_after_mobius': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'mobius_after_quadratic': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'quadratic_after_quadratic': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'two_map_union': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'interval_x': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}, 'x_residue_mod4': {'exact_triple_query_and_branch_gate': True, 'measured_below_B2_5_gate': False, 'symbolic_source_witness_below_B2_5_gate': False, 'fitted_hash_advantage_gate': False, 'end_to_end_or_fixed_curve_frontier_gate': False, 'promoted': False}}, 'promoted_coordinate_policies': [], 'strict_generic_prime_field_speedup': False, 'fixed_curve_frontier_improvement': False}`.

## Interpretation
P1418 charges the known difference required by projective xADD and preserves every normalized source triple. The experiment separates projective field-operation savings from state and witness complexity. If exact augmented and lossless directories retain cubic scaling like matched hashes, this is a scoped negative for the known-difference representation, not for other divisor, adaptive, or non-enumerative index-calculus constructions.
