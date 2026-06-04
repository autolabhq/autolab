mod solve;

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

const FALLBACK_SEED: &str = "b1c4e5d6f0a9876c";

#[derive(Clone, Copy)]
struct Query {
    xlo: u32,
    ylo: u32,
    xhi: u32,
    yhi: u32,
}

struct Case {
    input: String,
    queries: Vec<Query>,
}

// A naive scan that the verifier uses as the timing reference and as
// the correctness oracle. Kept private to this binary so /app/solve.rs
// cannot link against it.
struct SlowSet {
    points: Vec<(u32, u32)>,
}

impl SlowSet {
    fn new(input: &str) -> Self {
        let mut points = Vec::new();
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_ascii_whitespace();
            let x = match it.next().and_then(|s| s.parse::<u32>().ok()) {
                Some(v) => v,
                None => continue,
            };
            let y = match it.next().and_then(|s| s.parse::<u32>().ok()) {
                Some(v) => v,
                None => continue,
            };
            points.push((x, y));
        }
        Self { points }
    }

    fn count_in(&self, xlo: u32, ylo: u32, xhi: u32, yhi: u32) -> u32 {
        let mut c = 0u32;
        for &(x, y) in &self.points {
            if x >= xlo && x <= xhi && y >= ylo && y <= yhi {
                c += 1;
            }
        }
        c
    }
}

#[derive(Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn range(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            0
        } else {
            ((self.next_u64() >> 32) as u32) % upper
        }
    }

    fn rand16(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32 & 0xFFFF
    }
}

fn parse_hex_seed(text: &str) -> u64 {
    let mut value = 0u64;
    for byte in text.bytes().take(16) {
        let digit = match byte {
            b'0'..=b'9' => (byte - b'0') as u64,
            b'a'..=b'f' => (byte - b'a' + 10) as u64,
            b'A'..=b'F' => (byte - b'A' + 10) as u64,
            _ => continue,
        };
        value = (value << 4) | digit;
    }
    value
}

fn runtime_salt() -> u64 {
    let mut buf = [0u8; 8];
    if File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut buf))
        .is_ok()
    {
        u64::from_le_bytes(buf)
    } else {
        Instant::now().elapsed().as_nanos() as u64
    }
}

fn make_cases(seed: u64, small: bool) -> Vec<Case> {
    let mut rng = Rng::new(seed);
    let case_count = if small { 1 } else { 2 };
    let point_count = if small { 35_000usize } else { 800_000usize };
    let query_count = if small { 800usize } else { 8_000usize };
    let cluster_count = if small { 6usize } else { 12usize };

    (0..case_count)
        .map(|case_id| make_case(&mut rng, case_id, point_count, query_count, cluster_count))
        .collect()
}

fn make_case(
    rng: &mut Rng,
    case_id: usize,
    point_count: usize,
    query_count: usize,
    cluster_count: usize,
) -> Case {
    // Build a list of cluster centers and radii.
    let mut centers: Vec<(u32, u32, u32)> = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let cx = rng.rand16();
        let cy = rng.rand16();
        let r = 200 + rng.range(1500);
        centers.push((cx, cy, r));
    }

    // Render input lines: ~70% in clusters, 30% uniform.
    let mut input = String::with_capacity(point_count * 12);
    for _ in 0..point_count {
        let bucket = rng.range(100);
        let (x, y) = if bucket < 70 {
            let (cx, cy, r) = centers[rng.range(cluster_count as u32) as usize];
            let dx = rng.range(2 * r + 1) as i64 - r as i64;
            let dy = rng.range(2 * r + 1) as i64 - r as i64;
            let xi = (cx as i64 + dx).clamp(0, 65535);
            let yi = (cy as i64 + dy).clamp(0, 65535);
            (xi as u32, yi as u32)
        } else {
            (rng.rand16(), rng.rand16())
        };
        // Use itoa-ish fast path via format
        push_u32(&mut input, x);
        input.push(' ');
        push_u32(&mut input, y);
        input.push('\n');
    }

    // Queries: a mix of small / medium / large rectangles.
    let mut queries = Vec::with_capacity(query_count);
    for q in 0..query_count {
        let bucket = rng.range(100);
        let half = if bucket < 85 {
            // narrow window: ~0.005-0.05% of grid area
            16 + rng.range(96)
        } else if bucket < 97 {
            // medium window: ~0.4-1.5% of grid area
            220 + rng.range(420)
        } else {
            // wider window: ~3-10% of grid area
            900 + rng.range(1100)
        };
        // Center: ~70% near a cluster, 30% uniform.
        let (cx, cy) = if rng.range(100) < 70 + (q as u32 & 0) {
            let (cx, cy, _) = centers[rng.range(cluster_count as u32) as usize];
            let jitter = 256 + rng.range(512);
            let jx = rng.range(2 * jitter + 1) as i64 - jitter as i64;
            let jy = rng.range(2 * jitter + 1) as i64 - jitter as i64;
            (
                (cx as i64 + jx).clamp(0, 65535) as u32,
                (cy as i64 + jy).clamp(0, 65535) as u32,
            )
        } else {
            (rng.rand16(), rng.rand16())
        };
        let xlo = cx.saturating_sub(half);
        let ylo = cy.saturating_sub(half);
        let xhi = (cx + half).min(65535);
        let yhi = (cy + half).min(65535);
        queries.push(Query { xlo, ylo, xhi, yhi });
    }

    let _ = case_id;
    Case { input, queries }
}

fn push_u32(out: &mut String, mut v: u32) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut len = 0;
    while v > 0 {
        buf[len] = b'0' + (v % 10) as u8;
        v /= 10;
        len += 1;
    }
    for i in (0..len).rev() {
        out.push(buf[i] as char);
    }
}

fn mix(mut acc: u64, value: u64) -> u64 {
    acc ^= value
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(acc << 6)
        .wrapping_add(acc >> 2);
    acc.rotate_left(27).wrapping_mul(0x3c79_ac49_2ba7_b653)
}

fn run_slow(cases: &[Case]) -> (u64, f64) {
    let start = Instant::now();
    let mut digest = 0x6a09_e667_f3bc_c909u64;
    for case in cases {
        let set = SlowSet::new(black_box(&case.input));
        for q in &case.queries {
            let v = set.count_in(
                black_box(q.xlo),
                black_box(q.ylo),
                black_box(q.xhi),
                black_box(q.yhi),
            );
            digest = mix(digest, v as u64);
        }
    }
    (black_box(digest), start.elapsed().as_secs_f64())
}

fn run_candidate(cases: &[Case]) -> (u64, f64) {
    let start = Instant::now();
    let mut digest = 0x6a09_e667_f3bc_c909u64;
    for case in cases {
        let set = solve::PointSet::new(black_box(&case.input));
        for q in &case.queries {
            let v = set.count_in(
                black_box(q.xlo),
                black_box(q.ylo),
                black_box(q.xhi),
                black_box(q.yhi),
            );
            digest = mix(digest, v as u64);
        }
    }
    (black_box(digest), start.elapsed().as_secs_f64())
}

fn min_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(f64::INFINITY, |best, value| best.min(value))
}

fn main() {
    let verify_only = env::args().any(|arg| arg == "--verify");
    let build_seed = parse_hex_seed(option_env!("TASK_SEED").unwrap_or(FALLBACK_SEED));
    let seed = build_seed ^ runtime_salt().rotate_left(13);

    if verify_only {
        let cases = make_cases(seed ^ 0x91e6_2b4d_5fa1_88c3, true);
        let (expected, _) = run_slow(&cases);
        let (actual, _) = run_candidate(&cases);
        if expected == actual {
            println!("OK digest={expected:016x}");
            return;
        }
        eprintln!("wrong digest: expected={expected:016x} actual={actual:016x}");
        std::process::exit(1);
    }

    let cases = make_cases(seed, false);
    let (expected, _) = run_slow(&cases);
    let (actual, _) = run_candidate(&cases);
    if expected != actual {
        println!("__VERIFIER__ speedup=0 result=wrong");
        std::process::exit(1);
    }

    let mut slow_times = Vec::with_capacity(4);
    let mut candidate_times = Vec::with_capacity(4);
    for _ in 0..4 {
        let (slow_digest_1, slow_time_1) = run_slow(&cases);
        let (candidate_digest_1, candidate_time_1) = run_candidate(&cases);
        let (candidate_digest_2, candidate_time_2) = run_candidate(&cases);
        let (slow_digest_2, slow_time_2) = run_slow(&cases);

        if slow_digest_1 != expected
            || slow_digest_2 != expected
            || candidate_digest_1 != expected
            || candidate_digest_2 != expected
        {
            println!("__VERIFIER__ speedup=0 result=wrong");
            std::process::exit(1);
        }

        slow_times.push((slow_time_1 + slow_time_2) * 0.5);
        candidate_times.push((candidate_time_1 + candidate_time_2) * 0.5);
    }

    // Late-slice probe with a fresh seed: defeats stateful "fake-after-N"
    // attacks where the candidate returns canned answers after the timed
    // loop. Probe runs after timing on a different distribution.
    let probe_cases = make_cases(seed ^ 0xa53b_07e8_d2c4_f019, true);
    let (probe_expected, _) = run_slow(&probe_cases);
    let (probe_actual, _) = run_candidate(&probe_cases);
    if probe_expected != probe_actual {
        println!("__VERIFIER__ speedup=0 result=wrong");
        std::process::exit(1);
    }

    let slow_min = min_value(&slow_times);
    let candidate_min = min_value(&candidate_times).max(1e-9);
    let speedup = slow_min / candidate_min;

    println!(
        "__VERIFIER__ speedup={speedup:.6} result=ok candidate_time={candidate_min:.6} baseline_time={slow_min:.6} digest={expected:016x}"
    );
}
