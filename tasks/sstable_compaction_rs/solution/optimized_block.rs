use crate::codec::read_varint;
use crate::types::{Kind, KEY_SIZE, VALUE_SIZE};

#[derive(Clone, Copy)]
pub struct Entry {
    pub key: [u8; KEY_SIZE],
    pub seqno: u32,
    pub kind: Kind,
    pub value: [u8; VALUE_SIZE],
}

pub fn decode_block(block: &[u8]) -> Result<Vec<Entry>, String> {
    let mut pos = 0usize;
    let mut prev_key = [0u8; KEY_SIZE];
    let mut out = Vec::with_capacity(block.len() / 24);

    while pos < block.len() {
        let shared = read_varint(block, &mut pos)?;
        let suffix_len = read_varint(block, &mut pos)?;
        let value_len = read_varint(block, &mut pos)?;
        if shared > KEY_SIZE || shared + suffix_len != KEY_SIZE {
            return Err("bad key prefix encoding".to_string());
        }
        if pos + 1 + 4 + suffix_len + value_len > block.len() {
            return Err("truncated block record".to_string());
        }
        let kind = match block[pos] {
            0 => Kind::Put,
            1 => Kind::Delete,
            _ => return Err("bad kind".to_string()),
        };
        pos += 1;
        let seqno = u32::from_le_bytes([block[pos], block[pos + 1], block[pos + 2], block[pos + 3]]);
        pos += 4;

        let mut key = prev_key;
        key[shared..KEY_SIZE].copy_from_slice(&block[pos..pos + suffix_len]);
        pos += suffix_len;

        let mut value = [0u8; VALUE_SIZE];
        if value_len != 0 {
            value.copy_from_slice(&block[pos..pos + VALUE_SIZE]);
        }
        pos += value_len;
        prev_key = key;
        out.push(Entry { key, seqno, kind, value });
    }

    Ok(out)
}
