# P1436 autoresearch focus harness V99 result

Date: 2026-07-29

## Result

V99 binds R150 as the 86th closed frontier lane and routes the highest
priority action to
`s42_finite_depth_nonhomomorphic_u6_marker_circuit`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

For prime `q`, the rational cyclic convolution algebra satisfies

```text
Q[C_q] = Q[X]/(X^q-1)
       = Q direct-product Q(zeta_q).
```

Its nonzero unital quotient dimensions are therefore `1`, `q-1`, and `q`.
The one-dimensional quotient is augmentation and cannot recover varying
selected C6 counts.

For the actual binary deck element

```text
U(X) = sum_(a in C) X^a
```

define

```text
H_C = {k in F_q^*: kC=C}.
```

The conjugates `U(zeta_q^k)` agree exactly on cosets of `H_C`, so

```text
[Q(U(zeta_q)):Q] = (q-1)/|H_C|.
```

If the campaign deck contains a nonzero atom, every nonzero `H_C` orbit in
`C` has size `|H_C|`. Hence every reusable characteristic-zero
multiplication-closed representation containing `U` has dimension at least

```text
(q-1)/|C|.
```

At `q=B^5` and `|C|=B^(3/4)`, this is

```text
B^(17/4-o(1)),
```

above both the `B^(9/4)` setup cap and `B^(5/2)` rho scale. All eight
finite actual decks have trivial multiplier stabilizer, but that stronger
finite observation receives no generic-family asymptotic credit.

This closes only rational algebra quotients and reusable
multiplication-closed deck representations. It is not a time lower bound
for a bounded-depth nonhomomorphic circuit specialized to the single power
`U^6` and fixed marker batch, nor for adaptive data structures, bounded
error, or implicit summation-polynomial/FFE elimination. The elementary
derivation is novelty-unverified.

Verifier BSGS labels supply only finite orbit and same-augmentation
collision checks. No candidate DLP, root, Fourier, recurrence,
algebra-state, count, marginal, rank, or source oracle is supplied. No
factor-log recovery, identical target descent, generic transfer,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R150 tests: 11 passed.
- Full harness tests: 116 passed.
- Full ECDLP suite: 770 passed.
- R150 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R150: 75 receipts, 1,385 bindings, 0 mismatches.
- V99 frontier preflights: 86 provided, 86 closed.
- R150 obligations: 14 of 24 passed; lane admission false.
- R150 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V99 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R150 producer: `7f9217358c57a6cdb84f7f0d731e003e973096e4ff01a780ae0b3b8f482a755e`
- R150 report: `598fd0b95622354f22bc6ce531fc02aa2b5f61a1578d06dae36fbd39086ae6d3`
- R150 frozen interface: `04f989b1af422a1e42456bda7f95fbfdbb9941f709a75aed1de456d98c261e2b`
- R150 cost ledger: `57a022f3ffd5e149a95236f913cb612faff214f2ec2ef2d69a48f7af219017cd`
- R150 source replay: `4fb314326c46d9cf2988866435d4fdc7705f0dd28444c288002925d4cf833341`
- R150 controls: `cc838d5f5cd05b3c43ebfae096d3b9110fa06e02a2cb3ea10285155d119c2a28`
- R150 logs/descent: `cf0aa2174f836375165628813ee8f16ea29005a6345aec5af42f071f3367d1e5`
- R150 tests: `e5ff8f6cc11cf7e252e1aa49b456942c7d9bcfae660707c0bf75b9626cf7bf03`
- R150 gate: `1def188f9b0c5834889a937f7de9629f78b98c0e4f3b3249b1ba170a404483b6`
- R150 parent: `81761afebf32a86199a3aef3bad3520a9a4a25654ceef051a547d6829e9098b6`
- harness: `26a79348740739637d9e7b84e2fdcaeb14b3c5e41d420180a455b1e686a3c87c`
- harness tests: `d54bc8a185218354a197671a1fdfbfbaa4af51bd61b5d291dfefcc5adf22b09b`
- report: `5c2d452fbfb258aa0cc08ca0f2c9b2697174425be39b671f1efff14f1b775df7`
- note: `c04fd502dd7b472ddbe2617d054234c18bc635ebe4ac0d0da3ec4c2e319cdf0f`
- inventory: `13301443231a8c4982e9bd206add11fe2e15a10fc2cc0ef5635d9c265d00b4f0`
- replay plan: `ca88d0177457cd0d98f327a4ad6a68c8dc79198d06829998a6c2b44056bab3c6`
