# P1436 Autoresearch Focus Harness V39 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R89: fixed-marker scalar recurrence screen

R89 removes R87's arbitrary-marker caveat for target-local first-jet states.
It freezes two equal-size five-deck systems over F_17, with two choices in
every deck and marker equal to the source-choice index plus one.

At target zero, both systems have the same nonzero seven-scalar norm/marker
jet:

```text
[4, 2, 13, 13, 13, 5, 15]
```

At translated target one, the first system has jet
`[0, 12, 5, 5, 5, 10, 5]`, while the second has the zero jet. Thus the same
fixed-marker local state leads to simple versus multiple or nonreduced
branches after translation.

The deterministic F_1009 control has a degree-32 norm polynomial and five
degree-31 marker derivatives. Modulo the frozen degree-16 quotient, the norm
and all five marker translation channels have full rank 16.

The explicit five-slot constructor has support exponents

```text
C^1  B^0.6
C^2  B^1.2
C^3  B^1.8
C^4  B^2.4
C^5  B^3
```

so it first exceeds the B^(9/4) setup cap at the fourth slot. The finite
size-two and size-three controls have full support; the size-four control has
at least 99 percent occupancy.

This closes target-local fixed-marker jets and explicit shift/support
recurrences only. Target-independent nonlocal nonlinear translation states
remain open.

```text
report  50a075dd73d3c5298be00efdbf5186734ddf694ed14bbb38cb37f890815a3b63
gate    2e6ac3545bb9db94e5d8b54eed5b4a67a22cf870d544dc0af80b96b41b444858
parent  ec8fcedad29442b1d2ef92d2507dd3cc318bf5a6d5eb3c6dd3f8d3b4558558c0
```

## Harness routing

V39 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v39.json`

SHA-256:

`0bffe1c5a033683b0c9e20c5a005e88144933f48a2bd3c4dcd99e27f3430428d`

Schema: `ecdlp.p1436_autoresearch_focus_report.v31`.

Twenty-five hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_nonlocal_nonlinear_translation_sketch
```

The next experiment must freeze a target-independent nonlinear state and
public deck-update/translation law. Setup and state must be at most B^(9/4);
fresh translation and source return must be at most B^(5/4). It may not hide
a shift-value, coefficient, quotient, endpoint, or source table, and it must
replay every reduced, multiple, nonreduced, signed, infinity, tangent, and
exceptional branch.

V39 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R89 producer         a7dcd1f5f55484e5190cacc97671df02fdd95f109a9c4a504b3ab5059e8044d3
R89 tests            9deafd859521493ad46fe575f25ce456650744cc8e8fed7151587452a355248b
harness              f0bb735115d10cf129d095ed44b7e439ee25bde520d488c9db2cc82a8719610c
harness tests        a21fbca2c527ee8f6ea45ae44ad91b0e09fdfc065e79dd75de9b3faa877a952b
V39 focus note       1a7eb13130ddc8c9e97914650d53304e5df5c8726e80aa25615843c48582ab5a
V39 FFE inventory    ad262c0140551907223570614412a8192e845a1672129b14250ba5b4dad5c840
V39 FFE replay       a1aceb03a3b1cfa50952023c9d7e38121f1816331f785ad120d7c94530eb51a7
```

## Verification

- R89 targeted tests: 5 passed.
- Harness tests: 55 passed.
- Full ECDLP task suite: 175 tests passed.
- Fifteen R76-R89 and harness Python modules compiled.
- Fourteen R76-R89 parent YAML receipts parsed.
- All 171 declared input/artifact hashes matched.
- Every parent receipt and nested result has `breakthrough=false`.
- A clean R89 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V39 has 25 closed lanes, `promotion_allowed=false`, and the expected
  nonlocal nonlinear translation-sketch action.
