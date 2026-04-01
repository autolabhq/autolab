pub fn put_varint(mut v: usize, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub fn read_varint(data: &[u8], pos: &mut usize) -> Result<usize, String> {
    if *pos >= data.len() {
        return Err("truncated varint".to_string());
    }
    let b0 = data[*pos];
    *pos += 1;
    if b0 < 0x80 {
        return Ok(b0 as usize);
    }
    let mut value = (b0 & 0x7f) as usize;
    let mut shift = 7usize;
    loop {
        if *pos >= data.len() {
            return Err("truncated varint".to_string());
        }
        let byte = data[*pos];
        *pos += 1;
        value |= ((byte & 0x7f) as usize) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err("varint too long".to_string());
        }
    }
}
