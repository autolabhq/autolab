// Optimized: bin every (x, y) into a coarse 2D cell grid via a counting
// sort. Each query reduces to a scan of overlapping cells; cells fully
// inside the query window contribute their full size in O(1).

const GRID_BITS: u32 = 9; // 512 x 512 cells over the 65536 x 65536 grid
const GRID_SIDE: u32 = 1 << GRID_BITS;
const GRID_CELLS: usize = (GRID_SIDE as usize) * (GRID_SIDE as usize);
const SHIFT: u32 = 16 - GRID_BITS;
const CELL_W: u32 = 1 << SHIFT;

pub struct PointSet {
    // Points laid out cell-by-cell in (cy * GRID_SIDE + cx) order.
    xs: Vec<u16>,
    ys: Vec<u16>,
    // cell_start[c] = first index in xs/ys for cell c.
    // cell_start[GRID_CELLS] = total point count. Length is GRID_CELLS+1.
    cell_start: Vec<u32>,
}

impl PointSet {
    pub fn new(input: &str) -> Self {
        // Pass 1: parse points byte-by-byte; tally per-cell counts.
        let cap = input.len() / 8 + 1;
        let mut tmp_x: Vec<u16> = Vec::with_capacity(cap);
        let mut tmp_y: Vec<u16> = Vec::with_capacity(cap);
        let mut tmp_cell: Vec<u32> = Vec::with_capacity(cap);
        let mut counts: Vec<u32> = vec![0u32; GRID_CELLS];

        let bytes = input.as_bytes();
        let mut i = 0usize;
        let len = bytes.len();
        while i < len {
            while i < len
                && (bytes[i] == b' '
                    || bytes[i] == b'\t'
                    || bytes[i] == b'\n'
                    || bytes[i] == b'\r')
            {
                i += 1;
            }
            if i >= len {
                break;
            }
            let mut x: u32 = 0;
            let mut any_x = false;
            while i < len {
                let b = bytes[i];
                if b >= b'0' && b <= b'9' {
                    x = x.wrapping_mul(10).wrapping_add((b - b'0') as u32);
                    any_x = true;
                    i += 1;
                } else {
                    break;
                }
            }
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\t') {
                i += 1;
            }
            let mut y: u32 = 0;
            let mut any_y = false;
            while i < len {
                let b = bytes[i];
                if b >= b'0' && b <= b'9' {
                    y = y.wrapping_mul(10).wrapping_add((b - b'0') as u32);
                    any_y = true;
                    i += 1;
                } else {
                    break;
                }
            }
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            if any_x && any_y && x <= 65535 && y <= 65535 {
                let cell = (y >> SHIFT) * GRID_SIDE + (x >> SHIFT);
                tmp_x.push(x as u16);
                tmp_y.push(y as u16);
                tmp_cell.push(cell);
                counts[cell as usize] += 1;
            }
        }

        let n = tmp_x.len();

        // Build cell_start as a prefix sum of counts.
        let mut cell_start: Vec<u32> = vec![0u32; GRID_CELLS + 1];
        let mut s: u32 = 0;
        for c in 0..GRID_CELLS {
            cell_start[c] = s;
            s += counts[c];
        }
        cell_start[GRID_CELLS] = s;

        // Pass 2: scatter every (x, y) into its cell slot.
        let mut xs: Vec<u16> = vec![0u16; n];
        let mut ys: Vec<u16> = vec![0u16; n];
        let mut cursor: Vec<u32> = cell_start[..GRID_CELLS].to_vec();
        for k in 0..n {
            let c = tmp_cell[k] as usize;
            let pos = cursor[c] as usize;
            xs[pos] = tmp_x[k];
            ys[pos] = tmp_y[k];
            cursor[c] += 1;
        }

        Self { xs, ys, cell_start }
    }

    pub fn count_in(&self, xlo: u32, ylo: u32, xhi: u32, yhi: u32) -> u32 {
        if xlo > xhi || ylo > yhi {
            return 0;
        }
        let xlo = xlo.min(65535);
        let ylo = ylo.min(65535);
        let xhi = xhi.min(65535);
        let yhi = yhi.min(65535);

        let cx_lo = xlo >> SHIFT;
        let cx_hi = xhi >> SHIFT;
        let cy_lo = ylo >> SHIFT;
        let cy_hi = yhi >> SHIFT;

        let xlo16 = xlo as u16;
        let ylo16 = ylo as u16;
        let xhi16 = xhi as u16;
        let yhi16 = yhi as u16;

        let mut count: u32 = 0;
        let xs = &self.xs;
        let ys = &self.ys;
        let cs = &self.cell_start;

        for cy in cy_lo..=cy_hi {
            // A cell row is fully within the y-bounds if cy is strictly
            // between cy_lo and cy_hi (open interval).
            let cy_full = cy > cy_lo && cy < cy_hi;
            let cy_base = (cy * GRID_SIDE) as usize;
            for cx in cx_lo..=cx_hi {
                let cell_idx = cy_base + cx as usize;
                let lo = cs[cell_idx] as usize;
                let hi = cs[cell_idx + 1] as usize;
                if hi == lo {
                    continue;
                }
                let cx_full = cx > cx_lo && cx < cx_hi;
                if cy_full && cx_full {
                    count += (hi - lo) as u32;
                    continue;
                }
                // Boundary cell: classify which axis must be tested.
                let need_x = !cx_full;
                let need_y = !cy_full;
                let mut k = lo;
                if need_x && need_y {
                    while k < hi {
                        let x = xs[k];
                        let y = ys[k];
                        if x >= xlo16 && x <= xhi16 && y >= ylo16 && y <= yhi16 {
                            count += 1;
                        }
                        k += 1;
                    }
                } else if need_x {
                    while k < hi {
                        let x = xs[k];
                        if x >= xlo16 && x <= xhi16 {
                            count += 1;
                        }
                        k += 1;
                    }
                } else {
                    // need_y only
                    while k < hi {
                        let y = ys[k];
                        if y >= ylo16 && y <= yhi16 {
                            count += 1;
                        }
                        k += 1;
                    }
                }
                // If cell is fully spanned by both axes (caught above) we
                // never reach here; the partial-only branches let the
                // compiler vectorize the inner test.
                let _ = (xlo16, ylo16, xhi16, yhi16, CELL_W);
            }
        }
        count
    }
}
