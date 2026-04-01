mod solve;

use std::time::Instant;

const BEN_HAYSTACKS: usize = 100000;
const VERIFY_HAYSTACKS: usize = 400;
const N_RUNS: usize = 5;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn patterns() -> Vec<String> {
    [
        r"^GET /api/v[0-9]+/users/[0-9]+$",
        r"^POST /api/v[0-9]+/orders$",
        r"^DELETE /api/v[0-9]+/orders/[0-9]+$",
        r"(ERROR|WARN|INFO) [A-Z_]+ [0-9][0-9][0-9]",
        r"user=[a-z]+_[a-z]+",
        r"session=[0-9a-f]+",
        r"trace=[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
        r"^/static/[a-z0-9_/.-]+\.js$",
        r"^/assets/[a-z0-9_/.-]+\.(css|png|svg)$",
        r"status=(200|201|204|400|401|403|404|500|503)",
        r"region=(us|eu|ap)-(east|west|south|north)-[0-9]",
        r"([a-z0-9._-]+)@([a-z0-9._-]+)\.(com|net|org)",
        r"^item:[0-9]+:(open|closed|pending)$",
        r"([A-Z][a-z]+ )?[A-Z][a-z]+",
        r"^([a-z]+:)?//[a-z0-9.-]+(/[a-z0-9._/-]+)?$",
        r"sha1=[0-9a-f]+",
        r"build-[0-9]+-(debug|release)",
        r#"msg="[A-Za-z0-9 _./:-]+""#,
        r"(cache|db|queue)_(hit|miss|timeout)",
        r"^([A-Z0-9_]+)=([A-Za-z0-9_./:-]+)$",
        r"[a-z]+(/[a-z0-9_-]+)+",
        r"^([A-Z][A-Z0-9_]*)(\|[A-Z][A-Z0-9_]*)+$",
        r"(foo|bar|baz)+qux",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn rand_alpha(rng: &mut u32, len: usize, upper: bool) -> String {
    let base = if upper { b'A' } else { b'a' };
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        s.push((base + (xorshift32(rng) % 26) as u8) as char);
    }
    s
}

fn rand_hex(rng: &mut u32, len: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        s.push(HEX[(xorshift32(rng) % 16) as usize] as char);
    }
    s
}

fn rand_num(rng: &mut u32, digits: usize) -> String {
    let mut s = String::with_capacity(digits);
    for i in 0..digits {
        let d = if i == 0 { (xorshift32(rng) % 9 + 1) as u8 } else { (xorshift32(rng) % 10) as u8 };
        s.push((b'0' + d) as char);
    }
    s
}

fn haystacks(n: usize, seed: u32) -> Vec<Vec<u8>> {
    let mut rng = seed;
    let mut out = Vec::with_capacity(n);
    let methods = ["GET", "POST", "DELETE", "PATCH"];
    let levels = ["ERROR", "WARN", "INFO", "DEBUG"];
    let regions = ["us-east-1", "eu-west-2", "ap-south-1", "us-north-1"];
    let statuses = ["200", "201", "204", "400", "401", "403", "404", "500", "503"];
    let assets = ["main.js", "app.css", "logo.svg", "image.png"];

    for i in 0..n {
        let kind = (xorshift32(&mut rng) % 14) as usize;
        let s = match kind {
            0 => format!(
                "{} /api/v{}/users/{}",
                methods[(xorshift32(&mut rng) % methods.len() as u32) as usize],
                (xorshift32(&mut rng) % 4) + 1,
                rand_num(&mut rng, 4),
            ),
            1 => format!(
                "{} {} {}",
                levels[(xorshift32(&mut rng) % levels.len() as u32) as usize],
                rand_alpha(&mut rng, 6, true),
                statuses[(xorshift32(&mut rng) % statuses.len() as u32) as usize]
            ),
            2 => format!(
                "user={}_{} session={} trace={}",
                rand_alpha(&mut rng, 5, false),
                rand_alpha(&mut rng, 6, false),
                rand_hex(&mut rng, 12),
                rand_hex(&mut rng, 8)
            ),
            3 => format!(
                "/assets/{}/{}",
                rand_alpha(&mut rng, 4, false),
                assets[(xorshift32(&mut rng) % assets.len() as u32) as usize]
            ),
            4 => format!(
                "status={} region={}",
                statuses[(xorshift32(&mut rng) % statuses.len() as u32) as usize],
                regions[(xorshift32(&mut rng) % regions.len() as u32) as usize]
            ),
            5 => format!(
                "{}@{}.{}",
                rand_alpha(&mut rng, 6, false),
                rand_alpha(&mut rng, 5, false),
                ["com", "net", "org"][(xorshift32(&mut rng) % 3) as usize]
            ),
            6 => format!(
                "item:{}:{}",
                rand_num(&mut rng, 5),
                ["open", "closed", "pending", "stale"][(xorshift32(&mut rng) % 4) as usize]
            ),
            7 => format!(
                "{}/{}",
                rand_alpha(&mut rng, 5, false),
                rand_alpha(&mut rng, 7, false)
            ),
            8 => format!(
                "sha1={} msg=\"{}/{}/{}\"",
                rand_hex(&mut rng, 40),
                rand_alpha(&mut rng, 4, true),
                rand_alpha(&mut rng, 5, false),
                rand_num(&mut rng, 3)
            ),
            9 => format!(
                "build-{}-{}",
                rand_num(&mut rng, 6),
                ["debug", "release", "asan"][(xorshift32(&mut rng) % 3) as usize]
            ),
            10 => format!(
                "{}_{}",
                ["cache", "db", "queue", "auth"][(xorshift32(&mut rng) % 4) as usize],
                ["hit", "miss", "timeout", "evict"][(xorshift32(&mut rng) % 4) as usize]
            ),
            11 => format!(
                "{}={}",
                rand_alpha(&mut rng, 5, true),
                rand_alpha(&mut rng, 7, false)
            ),
            12 => format!(
                "{}://{}/{}",
                ["http", "https", "ftp"][(xorshift32(&mut rng) % 3) as usize],
                rand_alpha(&mut rng, 8, false),
                rand_alpha(&mut rng, 6, false)
            ),
            _ => {
                let mut s = String::new();
                for j in 0..(2 + (i % 4)) {
                    if j > 0 {
                        s.push('|');
                    }
                    s.push_str(&rand_alpha(&mut rng, 4, true));
                }
                s
            }
        };
        out.push(s.into_bytes());
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verify = args.len() >= 2 && args[1] == "--verify";
    let pats = patterns();
    let texts = haystacks(
        if verify { VERIFY_HAYSTACKS } else { BEN_HAYSTACKS },
        if verify { 0x1234_abcd } else { 0xdead_beef },
    );

    if verify {
        let summary = solve::run_workload(&pats, &texts);
        println!(
            "verify patterns={} haystacks={} matches={} checksum={}",
            pats.len(),
            texts.len(),
            summary.matches,
            summary.checksum
        );
        return;
    }

    let _ = solve::run_workload(&pats, &texts);
    let mut times = [0.0; N_RUNS];
    let mut summary = solve::Summary { matches: 0, checksum: 0 };
    for t in &mut times {
        let start = Instant::now();
        summary = solve::run_workload(&pats, &texts);
        *t = start.elapsed().as_secs_f64();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "patterns={} haystacks={} time={:.6} matches={} checksum={}",
        pats.len(),
        texts.len(),
        times[N_RUNS / 2],
        summary.matches,
        summary.checksum
    );
}
