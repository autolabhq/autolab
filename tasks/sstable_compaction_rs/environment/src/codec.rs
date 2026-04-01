pub fn put_varint(mut v: usize, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn read_varint(data: &[u8], pos: &mut usize) -> Result<usize, String> {
    let mut shift = 0usize;
    let mut value = 0usize;
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
