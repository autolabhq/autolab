use crate::block::Entry;
use crate::codec::put_varint;
use crate::types::{CompactionStats, Kind, BLOCK_RECORDS, KEY_SIZE, VALUE_SIZE};

pub struct TableBuilder {
    output: Vec<u8>,
    block_buf: Vec<u8>,
    block_count: usize,
    prev_key: [u8; KEY_SIZE],
    has_prev: bool,
}

impl TableBuilder {
    pub fn new() -> Self {
        Self {
            output: Vec::with_capacity(8 << 20),
            block_buf: Vec::with_capacity(256 << 10),
            block_count: 0,
            prev_key: [0; KEY_SIZE],
            has_prev: false,
        }
    }

    pub fn add(&mut self, rec: &Entry) {
        let mut shared = 0usize;
        if self.has_prev {
            while shared < KEY_SIZE && self.prev_key[shared] == rec.key[shared] {
                shared += 1;
            }
        }
        let suffix = &rec.key[shared..];
        put_varint(shared, &mut self.block_buf);
        put_varint(suffix.len(), &mut self.block_buf);
        put_varint(if matches!(rec.kind, Kind::Put) { VALUE_SIZE } else { 0 }, &mut self.block_buf);
        self.block_buf.push(match rec.kind { Kind::Put => 0, Kind::Delete => 1 });
        self.block_buf.extend_from_slice(&rec.seqno.to_le_bytes());
        self.block_buf.extend_from_slice(suffix);
        if matches!(rec.kind, Kind::Put) {
            self.block_buf.extend_from_slice(&rec.value);
        }
        self.prev_key = rec.key;
        self.has_prev = true;
        self.block_count += 1;
        if self.block_count == BLOCK_RECORDS {
            self.finish_block();
        }
    }

    fn finish_block(&mut self) {
        if self.block_count == 0 {
            return;
        }
        let len = self.block_buf.len() as u32;
        self.output.extend_from_slice(&len.to_le_bytes());
        self.output.extend_from_slice(&self.block_buf);
        self.block_buf.clear();
        self.block_count = 0;
        self.has_prev = false;
    }

    pub fn finish(mut self) -> CompactionStats {
        self.finish_block();
        let mut hash = 1469598103934665603u64;
        for &b in &self.output {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        CompactionStats {
            live_entries: 0,
            checksum: hash,
            output_bytes: self.output.len() as u64,
        }
    }
}
