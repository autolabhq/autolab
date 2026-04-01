# Stack Machine Golf — Reference

## Background

This task is an explicit stack-machine optimization benchmark inspired by superoptimization and stack-VM scheduling. The goal is not just correctness, but reducing dynamic executed instructions on a register-free machine.

## Baseline

The baseline:

1. keeps accumulator and pointers in memory scratch cells
2. performs one multiply-accumulate per loop iteration
3. pays branch and local-memory overhead every iteration

## Reference direction

The intended optimization is:

1. keep the running accumulator on the stack across several iterations
2. unroll the loop
3. reduce scratch-memory traffic
4. amortize loop-control overhead

Typical numbers on the packaged workload:

- Baseline: about `5132` executed instructions
- Strong solution: about `3530` executed instructions

## Sources

- Massalin (1987), Superoptimizer
- Schkufza, Sharma, Aiken (2013), Stochastic Superoptimization
