/// solve.rs — Discrete Fourier Transform
///
/// The Discrete Fourier Transform (DFT) of a length-n real signal x is:
///
///   X[k] = Σ_{j=0}^{n-1}  x[j] · e^{-2πi·j·k/n}      k = 0, …, n-1
///
/// DFT is the mathematical foundation of signal processing, convolution,
/// polynomial multiplication, spectral analysis, audio/image compression,
/// and fast solvers for PDEs.
///
/// This baseline evaluates the definition directly: for each of the n output
/// bins k, it sums n complex exponentials. Every (j, k) pair recomputes
/// sin and cos from scratch. Correct but O(n²), with expensive trig calls in
/// the inner loop.
///
/// PUBLIC INTERFACE — do NOT change this signature:
///
///   pub fn fft(signal: &[f64]) -> Vec<[f64; 2]>
///
/// Input:
///   signal — real-valued input of length n, where n must be a power of 2.
///
/// Output:
///   A Vec<[f64; 2]> of length n.
///   output[k] = [Re(X[k]), Im(X[k])]  (the k-th complex DFT coefficient).
///
/// Rewrite fft() to make it faster. Do not modify anything under /tests/ or
/// src/main.rs.
pub fn fft(signal: &[f64]) -> Vec<[f64; 2]> {
    let n = signal.len();
    let mut output = vec![[0.0f64; 2]; n];

    let tau = -2.0 * std::f64::consts::PI / n as f64;

    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for j in 0..n {
            let (s, c) = (tau * (k * j) as f64).sin_cos();
            re += signal[j] * c;
            im += signal[j] * s;
        }
        output[k] = [re, im];
    }

    output
}
