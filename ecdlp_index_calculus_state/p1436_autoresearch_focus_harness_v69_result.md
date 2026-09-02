# P1436 autoresearch focus harness V69 result

Date: 2026-07-29

## Result

V69 binds R120 as the 56th closed frontier lane and routes the highest
priority action to
`s12_m6_small_k_multiplicative_c5_membership_source_circuit`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

Every algebraic character from the elliptic curve to the base-field
multiplicative group is trivial, and the Weil self-pairing is trivial on the
prime-order cyclic subgroup. Pairing against an independent q-torsion point
does give an injective character and converts an exact five-sum into an exact
five-product while preserving source backpointers.

That conversion lives over an extension containing the q-th roots of unity.
With embedding degree `k=ord_q(p)`, even one extension-field operation carries
at least `k` base-field coordinates. The charged R118 batch then has exponent
`5/4 + log_B(k)`, so only `k=B^(o(1))` can remain within the frozen cap.

All eight R82 fixtures are supersingular `j=0`, embedding-degree-two cases.
An explicit Fp2 Miller loop and final exponentiation verifies their nontrivial
pairing characters, exact canonical C5 products, projective source replay, and
empty answers. Four finite prime-order curves with embedding degree `q-1`
provide an exceptional-family control. Neither finite control receives
asymptotic credit.

On the surviving small-k branch, pairing only reencodes exact multiplicative
five-product membership. Explicit product support still costs
`B^(15/4+o(1))`, and the current split index costs `B^(33/8+o(1))`. A
sub-output multiplicative C5 membership/source circuit remains open, while
pairing-unfriendly inputs remain a separate unsolved generic branch. No
Shoup improvement or ECDLP breakthrough is claimed.

## Verification

- Focused R120 tests: 10 passed.
- Full harness tests: 86 passed.
- Full ECDLP suite: 430 passed.
- R120 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R120: 45 receipts, 761 bindings, 0 mismatches.
- V69 frontier preflights: 56 provided, 56 closed.
- Structured true breakthrough/Shoup flags across R120 and V69: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R120 producer: `e758a1fed3cb0a41d18d74fd0fd4ffd05f047bd4d37d2d9be15f821195a9a131`
- R120 report: `de3e063e1d11a61cea49896a9a1e5e252b0e167cd558e6b50affe733bdeb60e5`
- R120 frozen interface: `1a4f30674052f3b6ab598b79fd06d6fed300c23f9ca6001a09826f7a514715c8`
- R120 cost ledger: `0e12afd3f31209dd707c022280c3868eec5a7d5533afcd7692f5bb19c0f71a42`
- R120 source replay: `ab50bec7273b718bcde1492df984d4dac054973a0706674d04b8b595b19aa600`
- R120 controls: `3dc277dab4854e9a4da6de31d7eb953ca228ac232d6cb7cffa586788988f2244`
- R120 logs/descent: `3b79bac599448a55a16d43a85af68e1519a72dfcb8ac1d74f9e2716bf5f40cbf`
- R120 tests: `baa460274249fb402d78b7738815b70d96f32baa8f0d2fcbd2bca91b857b68fb`
- R120 gate: `ab8efbe571f310a510f943e45f9fb65bac34fd37b2ee4d04337cdf3a5c9c8da1`
- R120 parent: `4e51a5d10a1debf66f7b4305f173cd818914e48e61512e37d11a6ea45e55169d`
- harness: `51e5f27d9265c465bf03688340058a730c5077d692ef2871a73a293714780412`
- harness tests: `361f902ead41adadee86357b9d288ce4cd43343bae461681b66f439a8f2bad05`
- report: `4f795ab47f11d8e85cef5adbd0fd09ae1eba0d14157943f0ce9d1bcc85d503b5`
- note: `b90fdb15611c9a9600aa06316d98368ddd0a4381777af059d518761a44f84b10`
- inventory: `7d76b238c5efcbc64d87b425b9c856700d203181f1c1b0d112bf85f5fef7deaf`
- replay plan: `e7d36c8e6d99627af128e4c05d7ed2546ea70ce894cd6a549797595343143ad7`
- Enge pairing reference: `93b99fa2d13e09c1bc8282b58d472be6285ceea9b594e9b97d765212a3aaa8e4`
- Miller pairing reference: `39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166`
