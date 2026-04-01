/// main.rs — benchmark driver for fft_rust.
///
/// DO NOT MODIFY THIS FILE.
///
/// Generates a deterministic real-valued signal, calls fft(), verifies
/// correctness via Parseval's theorem and a spectral checksum, and reports
/// median wall-clock time over N_RUNS timed trials.
///
/// Output (single line to stdout):
///   n=<N> runs=<R> time=<seconds> checksum=<f64> verify=ok
///   OR  …  verify=FAIL
mod solve;

use std::time::Instant;

// ── Benchmark parameters ───────────────────────────────────────────────────────
const N_DEFAULT: usize = 1 << 15; // 32 768
const N_RUNS: usize = 3;

// ── Signal generation ──────────────────────────────────────────────────────────
//
// Generates a deterministic real signal that is a superposition of K sinusoids
// at known frequencies.  The DFT of this signal has sharp peaks at those
// frequencies, enabling a checksum that validates spectral correctness.
fn gen_signal(n: usize) -> Vec<f64> {
    // Sum of K sinusoids at different normalised frequencies.
    // Amplitudes and phases chosen for a non-trivial spectrum.
    const FREQS: &[(usize, f64, f64)] = &[
        (1, 1.0, 0.0),
        (3, 0.5, 0.5),
        (7, 0.25, 1.0),
        (13, 0.125, 1.5),
        (31, 0.0625, 0.3),
        (63, 0.03125, 0.7),
        (127, 0.015625, 1.1),
    ];
    let tau = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|j| {
            FREQS
                .iter()
                .map(|&(f, a, phi)| a * (tau * (f * j) as f64 / n as f64 + phi).sin())
                .sum()
        })
        .collect()
}

// ── Verification ───────────────────────────────────────────────────────────────
//
// Two independent correctness checks:
//
//  1. Parseval: Σ |X[k]|² / n  ==  Σ |x[j]|²   (energy conservation).
//     A wrong FFT will almost certainly violate this to within 1e-6 tolerance.
//
//  2. Spectral checksum: Σ_k  |X[k]|² × k   (weighted sum of power spectrum).
//     Sensitive to permuted or shifted frequency bins.
//
// Returns (ok, checksum) where checksum is the spectral checksum rounded to
// 6 significant figures for stable cross-run comparison.
fn verify(signal: &[f64], spectrum: &[[f64; 2]]) -> (bool, f64) {
    let n = signal.len();
    if spectrum.len() != n {
        return (false, 0.0);
    }

    // Parseval check (relative tolerance 1e-6)
    let energy_time: f64 = signal.iter().map(|&x| x * x).sum();
    let energy_freq: f64 = spectrum.iter().map(|&[re, im]| re * re + im * im).sum::<f64>() / n as f64;
    let parseval_err = ((energy_time - energy_freq) / energy_time.max(1e-30)).abs();
    if parseval_err > 1e-6 {
        return (false, 0.0);
    }

    // Spectral checksum: weighted power sum
    let checksum: f64 = spectrum
        .iter()
        .enumerate()
        .map(|(k, &[re, im])| (re * re + im * im) * k as f64)
        .sum();

    let rounded_checksum = if checksum == 0.0 {
        0.0
    } else {
        let magnitude = checksum.abs().log10().floor();
        let factor = 10f64.powf(magnitude - 5.0);
        (checksum / factor).round() * factor
    };

    (true, rounded_checksum) // round to 6 significant figures
}

fn main() {
    let n: usize = std::env::var("VERIFY_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_DEFAULT);

    assert!(n.is_power_of_two(), "n must be a power of 2");

    let signal = gen_signal(n);

    // ── Warm-up ────────────────────────────────────────────────────────────────
    let _ = solve::fft(&signal);

    // ── Timed runs ─────────────────────────────────────────────────────────────
    let mut times = [0f64; N_RUNS];
    let mut last_spectrum = Vec::new();

    for t in &mut times {
        let start = Instant::now();
        let spectrum = solve::fft(&signal);
        *t = start.elapsed().as_secs_f64();
        last_spectrum = spectrum;
    }

    let mut sorted_times = times;
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_times[N_RUNS / 2];

    // ── Verify last result ─────────────────────────────────────────────────────
    let (ok, checksum) = verify(&signal, &last_spectrum);
    let verify_tag = if ok { "ok" } else { "FAIL" };

    println!("n={n} runs={N_RUNS} time={median:.6} checksum={checksum:.6e} verify={verify_tag}");
}
