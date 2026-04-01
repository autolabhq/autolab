use crate::codec::read_varint;
use crate::types::{Kind, Record, KEY_SIZE};

pub fn decode_block(block: &[u8]) -> Result<Vec<Record>, String> {
    let mut pos = 0usize;
    let mut prev_key = vec![0u8; KEY_SIZE];
    let mut out = Vec::new();

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

        let mut key = prev_key[..shared].to_vec();
        key.extend_from_slice(&block[pos..pos + suffix_len]);
        pos += suffix_len;
        let value = block[pos..pos + value_len].to_vec();
        pos += value_len;
        prev_key = key.clone();
        out.push(Record { key, seqno, kind, value });
    }

    Ok(out)
}
