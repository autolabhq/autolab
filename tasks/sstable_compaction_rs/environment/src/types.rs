pub const KEY_SIZE: usize = 8;
pub const VALUE_SIZE: usize = 24;
pub const BLOCK_RECORDS: usize = 256;
pub const TABLES: usize = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Put = 0,
    Delete = 1,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub key: Vec<u8>,
    pub seqno: u32,
    pub kind: Kind,
    pub value: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompactionStats {
    pub live_entries: u64,
    pub checksum: u64,
    pub output_bytes: u64,
}
