# Fast Fourier Transform in Rust — Reference

## Background
The Discrete Fourier Transform (DFT) decomposes a signal into its constituent frequencies and is the mathematical engine behind audio/image compression (MP3, JPEG), spectrum analysers, convolution filters, fast polynomial multiplication, and PDE solvers. Computing it naively costs O(n²) trig calls — for n = 32,768 that is ~10⁹ `sin_cos` calls and roughly 10 seconds on a modern CPU. The FFT family of algorithms reduces this to O(n log n), enabling the same transform in under 1 millisecond.

## Baseline Approach
The unoptimized implementation computes the DFT directly: for each output frequency bin k, it iterates over all n input samples and computes a fresh `sin_cos` call per (j, k) pair. This is O(n²) trig evaluations — ~10⁹ calls for n = 32,768 — taking ~10 seconds.

## Possible Optimization Directions
1. **Recursive Cooley-Tukey FFT** — split even/odd samples recursively into two n/2 DFTs combined with twiddle factors; reduces to O(n log n); ~300–1000x speedup
2. **Iterative FFT with precomputed twiddles** — bottom-up butterfly stages with twiddle factors precomputed once (n/2 `sin_cos` calls total instead of n²); eliminates trig from the hot loop; ~2000–5000x speedup
3. **Bit-reversal permutation** — reorder input in one O(n) pass before butterfly stages, enabling fully in-place computation with sequential memory access
4. **SIMD butterfly (AVX2 / `std::simd`)** — pack two butterfly pairs into 256-bit registers and process with `_mm256_fmadd_pd`/`_mm256_fmsub_pd`; ~2–4x over scalar iterative
5. **Split-radix algorithm** — Duhamel & Hollmann (1984) variant uses ~15% fewer multiplications than radix-2 Cooley-Tukey

## Reference Solution
Iterative Cooley-Tukey (Gentleman–Sande decimation-in-time) FFT. Precomputes all n/2 twiddle factors in one pass, performs a bit-reversal permutation on the input, then runs log₂(n) in-place butterfly stages bottom-up. Each butterfly computes a complex multiply of the twiddle factor with the lower element, then adds/subtracts to produce the two outputs. For n = 32,768 this requires ~60,000 trig calls total instead of ~10⁹.

## Source
- Cooley & Tukey (1965), *An Algorithm for the Machine Calculation of Complex Fourier Series*, Math. Comput. 19(90):297–301
- Gentleman & Sande (1966), *Fast Fourier Transforms — For Fun and Profit*, AFIPS Conf. Proc. 29:563–578
- Duhamel & Hollmann (1984), *Split radix FFT algorithm*, Electronics Letters 20(1):14–16
- Frigo & Johnson (2005), *The Design and Implementation of FFTW3*, Proc. IEEE 93(2):216–231
