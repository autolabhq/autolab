# SHA-256 Throughput — Reference

## Background
SHA-256 is the dominant cryptographic hash function in TLS certificates, Git object storage, Bitcoin proof-of-work, code signing, and filesystem integrity checks. Sustained throughput over large buffers (e.g., disk images, firmware blobs, large file checksums) is a common bottleneck in security tooling and data pipelines.

## Baseline Approach
The unoptimized implementation is a portable FIPS 180-4 scalar loop: it precomputes all 64 message schedule words `w[0..63]` into a stack array before starting the compression loop, uses ROR via two shifts, and processes one 64-byte block sequentially. On a 512 MiB input (~8 million blocks × 64 rounds each) this takes ~2.5s with no instruction-level parallelism exploited.

## Possible Optimization Directions
1. **Message schedule interleaving** — compute `w[i]` on-the-fly inside the compression loop instead of in a separate pass, keeping schedule words in registers and reducing memory traffic (~1.3x improvement)
2. **Full loop unrolling** — manually unroll all 64 rounds to eliminate loop-carried dependencies the compiler cannot remove, exposing more ILP (~1.5x improvement)
3. **Explicit-type ROR macros** — cast operands to `uint32_t` explicitly so GCC emits native `ror` instructions instead of two-shift sequences (~1.2x improvement)
4. **SHA-NI hardware acceleration** — use Intel SHA Extensions (`sha256rnds2`, `sha256msg1`, `sha256msg2`) with `__attribute__((target("sse4.1,sha")))`, processing 2 rounds per instruction (~8–12x improvement)
5. **2-way SHA-NI parallel interleaving** — maintain two independent compression streams simultaneously to hide the 4-cycle latency of `sha256rnds2`, doubling block throughput from ~130 cycles/block to ~70 cycles/block (~1.8x additional improvement)

## Reference Solution
Runtime CPUID dispatch: checks for SHA-NI + SSE4.1 support (CPUID leaf 7 EBX bit 29, leaf 1 ECX bit 19) and routes to one of two paths:

- **SHA-NI path**: processes one 64-byte block using Intel SHA Extensions. Loads state into ABEF/CDGH register layout, byte-swaps the message block with `pshufb`, then runs 16 groups of 4 rounds using `DO4RNDS` (two `sha256rnds2` calls per group). Message schedule words 16–63 are produced on-the-fly interleaved with the rounds via `sha256msg1`, `alignr`, add, and `sha256msg2`. No scalar memory is touched during compression.
- **Scalar fallback**: fully unrolled 64-round loop with explicit-type ROR macros, compiling to native `ror` instructions under GCC -O2.

## Source
- NIST FIPS 180-4, *Secure Hash Standard* (2015)
- Intel, *Intel SHA Extensions* white paper (2013) — https://www.intel.com/content/dam/develop/external/us/en/documents/256808-sha-extensions-white-paper.pdf
- Gulley et al., *New Instructions Supporting the Secure Hash Algorithm on Intel Architecture Processors* (2013)
- Go standard library, `src/crypto/sha256/sha256block_amd64.s`
- Adam Langley's portable SHA-256 implementation
