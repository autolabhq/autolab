use crate::block::{decode_block, Entry};

pub struct TableIter<'a> {
    table: &'a [u8],
    pos: usize,
    block_records: Vec<Entry>,
    idx: usize,
}

impl<'a> TableIter<'a> {
    pub fn new(table: &'a [u8]) -> Result<Self, String> {
        let mut it = Self { table, pos: 0, block_records: Vec::new(), idx: 0 };
        it.load_next_block()?;
        Ok(it)
    }

    fn load_next_block(&mut self) -> Result<(), String> {
        if self.pos >= self.table.len() {
            self.block_records.clear();
            self.idx = 0;
            return Ok(());
        }
        if self.pos + 4 > self.table.len() {
            return Err("truncated block length".to_string());
        }
        let len = u32::from_le_bytes([
            self.table[self.pos],
            self.table[self.pos + 1],
            self.table[self.pos + 2],
            self.table[self.pos + 3],
        ]) as usize;
        self.pos += 4;
        if self.pos + len > self.table.len() {
            return Err("truncated block bytes".to_string());
        }
        self.block_records = decode_block(&self.table[self.pos..self.pos + len])?;
        self.pos += len;
        self.idx = 0;
        Ok(())
    }

    pub fn current(&self) -> Option<Entry> {
        self.block_records.get(self.idx).copied()
    }

    pub fn advance(&mut self) -> Result<(), String> {
        self.idx += 1;
        if self.idx >= self.block_records.len() {
            self.load_next_block()?;
        }
        Ok(())
    }
}
