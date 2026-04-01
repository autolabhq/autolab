# AES-128-CTR Encryption Throughput — Reference

## Background
AES-128 in Counter (CTR) mode is the backbone of TLS, disk encryption, VPNs, and secure messaging. Throughput directly affects how fast data can be encrypted at line rate—achieving gigabytes-per-second matters in network appliances, storage systems, and streaming pipelines. CTR mode's block-independent structure makes it uniquely suited for hardware and instruction-level parallelism.

## Baseline Approach
The unoptimized implementation performs byte-level GF(2⁸) multiplication for MixColumns using `xtime` and processes one block at a time sequentially. Each of the 9 full AES rounds requires 16 SubBytes lookups, a ShiftRows permutation, field arithmetic for MixColumns, and AddRoundKey—all done in scalar C with no hardware acceleration. Encrypting 256 MiB this way takes ~3.0s.

## Possible Optimization Directions
1. **T-table lookup** — Precompute four 256-entry `uint32_t` tables that merge SubBytes + ShiftRows + MixColumns into four 32-bit XORs per round column, eliminating all runtime GF arithmetic (~4× speedup)
2. **Hardware AES-NI intrinsics** — Use `_mm_aesenc_si128` / `_mm_aesenclast_si128` from `<wmmintrin.h>` to execute each AES round in a single hardware instruction (~8× additional speedup over T-tables)
3. **8-way CTR-mode parallelism** — Since each CTR block is independent, encrypt 8 blocks simultaneously to hide the 4-cycle AES-NI instruction latency and approach full throughput (~3× additional speedup)
4. **Hardware key schedule caching** — Expand the 11 round keys once per call using `_mm_aeskeygenassist_si128`, storing them in `__m128i rk[11]` to avoid recomputation in the inner loop

## Reference Solution
The solution uses runtime CPU dispatch: if AES-NI is available (`__builtin_cpu_supports("aes")`), it takes the hardware path; otherwise it falls back to the scalar T-table path.

The AES-NI path (`aes128_ctr_ni`) expands round keys once via `_mm_aeskeygenassist_si128`, then runs an 8-block inner loop: eight counter values are loaded and encrypted in parallel by interleaving `_mm_aesenc_si128` calls across all eight `__m128i` registers, XOR'd with plaintext, and stored. Remaining blocks are handled one at a time. The scalar fallback (`aes128_ctr_scalar`) builds the T-tables lazily on first use and encrypts one block per iteration using four table lookups per round column.

## Source
- NIST FIPS 197, *Advanced Encryption Standard (AES)* (2001)
- NIST SP 800-38A, *Recommendation for Block Cipher Modes of Operation* (2001)
- Gueron, S., *Intel Advanced Encryption Standard (AES) New Instructions Set* (Intel white paper, 2010)
