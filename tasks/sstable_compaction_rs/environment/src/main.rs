mod block;
mod builder;
mod codec;
mod compaction;
mod iter;
mod merge;
mod types;

use codec::put_varint;
use compaction::compact_tables;
use std::time::Instant;
use types::{Kind, KEY_SIZE, TABLES, VALUE_SIZE, BLOCK_RECORDS};

const DOMAIN_KEYS: usize = 420_000;
const TABLE_KEYS: usize = 210_000;
const VERIFY_TABLE_KEYS: usize = 30_000;
const N_RUNS: usize = 5;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn make_value(key: u64, table_idx: usize) -> [u8; VALUE_SIZE] {
    let mut out = [0u8; VALUE_SIZE];
    let mut x = key ^ ((table_idx as u64) << 48) ^ 0xdead_beef_cafe_f00d;
    for chunk in out.chunks_mut(8) {
        x = splitmix64(x);
        chunk.copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn encode_records(keys: &[u64], table_idx: usize) -> Vec<u8> {
    let mut table = Vec::new();
    let mut block_buf = Vec::new();
    let mut prev_key = [0u8; KEY_SIZE];
    let mut block_count = 0usize;

    for &key in keys {
        let seqno = 10_000_000u32 - (table_idx as u32) * 1000;
        let tombstone = (splitmix64(key ^ (table_idx as u64 * 17)) % 11 == 0) && table_idx < 3;
        let kind = if tombstone { Kind::Delete } else { Kind::Put };
        let key_bytes = key.to_be_bytes();
        let mut shared = 0usize;
        while shared < KEY_SIZE && prev_key[shared] == key_bytes[shared] {
            shared += 1;
        }
        let suffix = &key_bytes[shared..];
        put_varint(shared, &mut block_buf);
        put_varint(suffix.len(), &mut block_buf);
        put_varint(if matches!(kind, Kind::Put) { VALUE_SIZE } else { 0 }, &mut block_buf);
        block_buf.push(match kind { Kind::Put => 0, Kind::Delete => 1 });
        block_buf.extend_from_slice(&seqno.to_le_bytes());
        block_buf.extend_from_slice(suffix);
        if matches!(kind, Kind::Put) {
            block_buf.extend_from_slice(&make_value(key, table_idx));
        }
        prev_key = key_bytes;
        block_count += 1;
        if block_count == BLOCK_RECORDS {
            let len = block_buf.len() as u32;
            table.extend_from_slice(&len.to_le_bytes());
            table.extend_from_slice(&block_buf);
            block_buf.clear();
            block_count = 0;
            prev_key = [0u8; KEY_SIZE];
        }
    }

    if block_count != 0 {
        let len = block_buf.len() as u32;
        table.extend_from_slice(&len.to_le_bytes());
        table.extend_from_slice(&block_buf);
    }

    table
}

fn generate_tables(per_table: usize) -> Vec<Vec<u8>> {
    let mut tables = Vec::new();
    for table_idx in 0..TABLES {
        let mut keys = Vec::with_capacity(per_table);
        let stride = 1 + table_idx * 3;
        let offset = table_idx * 97;
        for i in 0..per_table {
            let id = ((i * stride + offset) % DOMAIN_KEYS) as u64;
            let mixed = (id << 20) | ((splitmix64(id + table_idx as u64) as u32 & 0xfffff) as u64);
            keys.push(mixed);
        }
        keys.sort_unstable();
        keys.dedup();
        tables.push(encode_records(&keys, table_idx));
    }
    tables
}

fn run_case(per_table: usize) -> Result<(u64, u64, u64), String> {
    let tables = generate_tables(per_table);
    let stats = compact_tables(&tables)?;
    Ok((stats.live_entries, stats.checksum, stats.output_bytes))
}

fn bench_case() -> Result<(u64, u64, u64, f64), String> {
    let tables = generate_tables(TABLE_KEYS);
    let warm = compact_tables(&tables)?;
    let mut times = Vec::with_capacity(N_RUNS);
    let mut last = warm;
    for _ in 0..N_RUNS {
        let t0 = Instant::now();
        last = compact_tables(&tables)?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((last.live_entries, last.checksum, last.output_bytes, times[N_RUNS / 2]))
}

fn main() {
    let verify = run_case(VERIFY_TABLE_KEYS);
    let bench = bench_case();
    match (verify, bench) {
        (Ok((vlive, vhash, vbytes)), Ok((blive, bhash, bbytes, secs))) => {
            println!(
                "verify=ok verify_live={} verify_hash={} verify_bytes={} bench_live={} bench_hash={} bench_bytes={} time={:.6}",
                vlive, vhash, vbytes, blive, bhash, bbytes, secs
            );
        }
        (Err(e), _) | (_, Err(e)) => {
            println!("verify=FAIL error={}", e.replace(' ', "_"));
        }
    }
}
