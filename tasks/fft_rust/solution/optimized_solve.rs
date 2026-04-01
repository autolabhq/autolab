/// optimized_solve.rs — reference solution: iterative Cooley-Tukey FFT.
///
/// Implements the standard Gentleman–Sande ("decimation-in-time") iterative FFT:
///
///   1. Precompute all n/2 twiddle factors (one pass of sin_cos calls).
///   2. Reorder input with a bit-reversal permutation (O(n) swaps).
///   3. Run log₂(n) butterfly stages bottom-up, each touching n/2 pairs.
///
/// Compared to the naive DFT baseline:
///   • sin_cos calls:   n/2 (precomputed once)  vs  n² (recomputed per pair)
///   • Total work:      O(n log n)               vs  O(n²)
///   • For n=32 768:    ~60 000 trig calls        vs  ~10⁹
///
/// This implementation should reach ~1–3 ms on a 3 GHz CPU.
///
/// Further improvements beyond this reference:
///   • SIMD butterfly: AVX2 processes 2 butterfly pairs per instruction,
///     doubling throughput.  Use `std::arch::x86_64::_mm256_*` intrinsics
///     or the nightly `std::simd` portable SIMD API.
///   • Cache-oblivious FFT (Frigo 1999): recursive tiling hides cache misses
///     for n > L2-cache size.
///   • Split-radix (Duhamel & Hollmann 1984): ~15% fewer multiplications than
///     Cooley-Tukey radix-2.

use std::f64::consts::PI;

pub fn fft(signal: &[f64]) -> Vec<[f64; 2]> {
    let n = signal.len();
    assert!(n.is_power_of_two() && n >= 2, "n must be a power of 2 and ≥ 2");

    // ── Step 1: precompute twiddle factors ω_n^k = e^{-2πik/n} for k=0..n/2 ──
    // Stored as (cos, sin) pairs.  Index j in butterfly stage with stride s
    // uses twiddle[j * (n / (2*s))].
    let twiddle: Vec<[f64; 2]> = (0..n / 2)
        .map(|k| {
            let theta = -2.0 * PI * k as f64 / n as f64;
            [theta.cos(), theta.sin()]
        })
        .collect();

    // ── Step 2: bit-reversal permutation ─────────────────────────────────────
    // Place input x[j] at index bit_reverse(j, log2(n)) in array a[].
    let mut a: Vec<[f64; 2]> = signal.iter().map(|&x| [x, 0.0]).collect();
    let log2n = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if j > i {
            a.swap(i, j);
        }
    }

    // ── Step 3: iterative butterfly network (log₂ n stages) ──────────────────
    // Stage s processes sub-DFTs of size 2s, combining pairs with distance s.
    // The twiddle for butterfly (k, s) is twiddle[k * (n / (2*s))].
    let mut half = 1usize; // half-size of current sub-DFT
    while half < n {
        let stride = n / (half << 1); // twiddle stride for this stage
        let mut base = 0;
        while base < n {
            for k in 0..half {
                let tw = twiddle[k * stride];
                let u = a[base + k];
                let v = a[base + k + half];
                // Complex multiply: tw × v
                let tv_re = tw[0] * v[0] - tw[1] * v[1];
                let tv_im = tw[0] * v[1] + tw[1] * v[0];
                // Butterfly: u + tw×v,  u - tw×v
                a[base + k]        = [u[0] + tv_re, u[1] + tv_im];
                a[base + k + half] = [u[0] - tv_re, u[1] - tv_im];
            }
            base += half << 1;
        }
        half <<= 1;
    }

    a
}

#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}
