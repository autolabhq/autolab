# P1436 autoresearch focus harness V63 result

Date: 2026-07-29

## Result

V63 binds R114 as the 50th closed frontier lane and routes the highest
priority action to
`s6_relation_arity_factor_base_transposed_interface_rebalance`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R114 exactly replays first-product adjoints on all 14 unique-zero actual and
matched instances. The first adjoint collapses on both multiplicity-two
instances, while the product Hessian applied to the all-ones direction
localizes both zero leaves. Together these adjoints replay all 18 R113
sources with the R105 markers and R108 weight 14400.

The sparse adjoint output does not supply a cheap constructor. Standard
reverse AD still evaluates the `B^(3+o(1))` primal leaf trace; checkpointing
reduces state, not work. Transposing before leaf formation expands `B^2`
terminal `A5` support to a typed `B^(13/5+o(1))` state after one `C` atom,
already above the `B^(9/4)` setup cap and `B^(5/4)` fresh-work cap.

This closes standard product reverse AD, first/second derivative
localization as a constructor claim, and canonical typed preleaf propagation
only. Arbitrary transposed group-algebra circuits remain outside the
negative.

No rank, factor-log, target-descent, Shoup, or breakthrough gate passes.

## Verification

- Focused R114 tests: 7 passed.
- Focused harness routing/schema tests: 3 passed.
- Full ECDLP suite: 374 passed.
- Producer and harness files compile.
- R114 clean rerun: all six JSON outputs byte-identical.
- Parent audit R76-R114: 39 receipts, 636 bindings, 0 mismatches.
- V63 frontier preflights: 50 provided, 50 closed.
- Structured true breakthrough/Shoup flags across R114 and V63: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R114 producer: `b477e0162d7cd89d3341f77521dc0b004879b8a916b484ca40b6787746089d83`
- R114 report: `b0759fd0e0fe14e0ea44f908b802e24b611802a64663e138337465cf36ef69a5`
- R114 source replay: `031e639ab8903ea8f4f7f9f3f7bec384a2940aa40473562419494cb27efee825`
- R114 gate: `6574145a6d67ec4e55a8bf0cf90c3a0c9f858aa4cddbfd7e9a733c1033b3a3da`
- R114 parent: `4acc68f562bbed3ec506cc0cbf4f1223fb24d151b602c92494ae75a74f081287`
- harness: `f149540d25338640557606165f2b0713ade0909e3d4a2c118aff9798ebefe7fb`
- harness tests: `a2bba4fb358cf7105e9be5137ffc1b9e96b08532199af36f1e8c4fb128cf5f34`
- report: `77e5c0f7b55dc7955b031ec32a3daf5fa1645bd874385f83ebb91006e6d98231`
- note: `f2c92a11d869d1a2024d181cbefcde3734f2ec2636e3ed9e47a7da7386b9fa45`
- inventory: `adfafc9db3ae639b4d1876207779a8abadb1643af43b7634273457f7eed197dc`
- replay plan: `0152fbb3b573beef56cb4c5f71b6c7fde536d75b25cc36a7cce8bfa32336f0b8`
