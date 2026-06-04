pub struct PointSet {
    points: Vec<(u32, u32)>,
}

impl PointSet {
    pub fn new(input: &str) -> Self {
        let mut points = Vec::new();
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_ascii_whitespace();
            let x = match it.next() {
                Some(s) => match s.parse::<u32>() {
                    Ok(v) => v,
                    Err(_) => continue,
                },
                None => continue,
            };
            let y = match it.next() {
                Some(s) => match s.parse::<u32>() {
                    Ok(v) => v,
                    Err(_) => continue,
                },
                None => continue,
            };
            points.push((x, y));
        }
        Self { points }
    }

    pub fn count_in(&self, xlo: u32, ylo: u32, xhi: u32, yhi: u32) -> u32 {
        let mut count = 0u32;
        for &(x, y) in &self.points {
            if x >= xlo && x <= xhi && y >= ylo && y <= yhi {
                count += 1;
            }
        }
        count
    }
}
