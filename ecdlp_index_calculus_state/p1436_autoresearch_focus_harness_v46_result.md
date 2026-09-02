# P1436 Autoresearch Focus Harness V46 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R96: modular Frobenius trace

R96 instantiates the R95 Fermat projector in a finite quotient algebra:

```text
A = F_p[x]/F(x),
zero count = Tr_A(1-M_h^(p-1)).
```

On a reduced split `F_11` quotient with roots `1,2,4,7`, the trace exactly
counts the two zeros of `h=(x-1)(x-4)`. Frobenius is the identity matrix, so
it supplies no state contraction.

Adding a duplicate occurrence of root one produces a degree-five nonreduced
quotient. The projector trace rises from two to three, but Frobenius rank is
four rather than five: it kills the nilpotent occurrence direction.
Radicalization loses the duplicate source, while retaining multiplicity keeps
the full source body.

A blind query returns trace zero. Dyadic quotient traces return one exact zero
occurrence with fewer than three root-body dimensions queried in total.

At the actual R84 split, the standard source-complete quotient basis costs

```text
B^(12/5),
```

outside the `B^(9/4)` setup and `B^(5/4)` fresh caps. Explicit multiplication
or Frobenius matrices cost `B^(24/5)`. This closes explicit quotient bases,
matrices, and dyadic range quotients only; a genuinely factored transposed
trace remains open.

```text
R96 producer  92f646378156476cf897729b6bca6d67bc47398663137aee1ab17d910511f960
R96 report    e4e667703d22b96c5ef64b3bfbbd24119f128cd00f68d0d7d850ba41c0c0e163
R96 gate      55655280c02bbef1e5f0f29bcad1beb1b21d6809e79b7de400a58997b0caa64d
R96 parent    bd7045674bc1faffb4154b902c4274d0bf298281b202289cd8cc5c57cf1746a9
R96 tests     a4b76d8a31f6e3982c690defffcdcbe244866dfb5a932295ada81a2170b19206
```

R94 and R95 details and receipts are recorded in
`ecdlp_index_calculus_state/p1436_autoresearch_focus_harness_v45_result.md`
with SHA-256
`f03c6c888d4eb96268baa9840732f7f7cab388dbdc7e5b17354b7352ddd64e1e`.

## Harness routing

V46 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v46.json`

SHA-256:

`92cbfaef6a6fabb09cbd6c9929d4f267193c0d9ccd7b95675288bca8488143f9`

Schema: `ecdlp.p1436_autoresearch_focus_report.v38`.

Thirty-two provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_factored_transposed_projector_trace
```

The next experiment must exhibit an explicit factored adjoint/trace identity,
integer lift, range restriction, and multiplicity-complete coupled-source
reporter without emitting a `B^(12/5)` quotient, root, moment, endpoint, or
adjoint body.

V46 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              41ee849b8bd395229bb0ab9114cf713b9019ca350390d030904f5d9795c2defc
harness tests        4675b15bedfafb05799a0084ff8a37a80e5e8d2ea6d133bc577dc5efce42deb1
V46 focus note       141c79eb9c8da70ba3fdf2b706861257d4ededb0a7b2e8205a9b9a58aa3c86d6
V46 FFE inventory    ede6f5f167bd56d4e89090e1c18ec7206eb4add4d38509e93c14d200503b55fe
V46 FFE replay       730b08845c6b9307a3c2377df5e8adbdb2df64e90f8efad966c359812beb3616
```

## Verification

- R96 targeted tests: 7 passed.
- R96 plus harness focused tests: 69 passed.
- Full ECDLP task suite: 230 tests passed.
- Twenty-two R76-R96 and harness Python modules compiled.
- Twenty-one R76-R96 parent YAML receipts parsed.
- All 290 declared input/artifact hashes matched.
- No parent receipt or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- Clean R94, R95, and R96 reruns each reproduced all six generated JSON
  artifacts byte-for-byte.
- V46 has 32 closed lanes, `promotion_allowed=false`, and the expected
  factored transposed projector-trace action.
