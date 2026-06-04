# 2D Range Count — Reference

## Background

Two-dimensional rectangular range counting is a classical primitive:
"how many points fall inside this axis-aligned box?". It shows up in
spatial databases (PostGIS, MonetDB), GIS tile servers, time-series
multi-dimensional indexes, and OLAP cube cell queries. With small
integer coordinates (16 bits per axis here) the problem fits a wide
family of well-known structures: kd-trees, R-trees, BKD-trees,
multi-bit grids, and space-filling-curve indexes.

## Baseline Approach

The starter implementation parses the coordinate stream once into a
flat `Vec<(u32, u32)>` and answers each query by scanning every
point. That is `O(N)` per query, regardless of how small the query
window is — and queries here range from windows that cover ~0.01%
of the grid to ~10%, so most of the scanned points are irrelevant.

## Possible Optimization Directions

1. **Sort by a key that preserves 2D locality.** If points that are
   close in (x, y) end up close in the sorted order, you can localize
   each query to a contiguous slice. Bit-interleaving an `(x, y)` pair
   into a single 32-bit integer (with `x` on even bits and `y` on odd
   bits) gives such a key — points inside any axis-aligned rectangle
   land in a small number of contiguous runs after sorting.
2. **Skip step.** When you binary-search to the first sorted entry
   inside the query's key range, then scan forward, most stretches of
   that range are still outside the query rectangle (the recursive
   "Z" shape weaves in and out of the box). Given a current key that
   lies between the query corners but outside the box, you can
   compute the next key that re-enters the box by walking the bits
   from MSB to LSB, tracking which axis is currently outside its
   range, and "loading" or "saving" that axis (Tropf and Herzog,
   1981). Then a single binary search jumps you there.
3. **Compact storage.** Storing the 32-bit key plus two `u16`
   coordinates per point (8 bytes) keeps the inner loop cache-warm.
4. **Coarser per-cell counts.** A grid of bucket counts gives an
   answer in one pass per overlapping cell when the query is much
   larger than the cell — but when the workload mixes very small and
   very large queries, a single grid stride is hard to tune.

The reference picks (1)+(2)+(3): a compact sort-by-locality key plus
the BIGMIN skip step.

## Reference Solution

`solve_optimized.rs`:

- In `new()`, parse the coordinate stream byte-by-byte (no
  `String::parse` overhead per line), interleave the bits of `x` and
  `y` into a 32-bit Morton key, and sort `(key, x, y)` triples by
  key.
- In `count_in(xlo, ylo, xhi, yhi)`, compute the keys of the rect
  corners `(xlo, ylo)` and `(xhi, yhi)`. Binary-search for the first
  index whose key is `>=` the lower-corner key. Walk forward; for
  each entry whose key is `<=` the upper-corner key, check if the
  point is inside the rect — count it if so, otherwise call BIGMIN
  (Tropf and Herzog 1981) to compute the smallest key `>=` the
  current key that is inside the rect. Binary-search forward to that
  key and continue.

The BIGMIN routine inspects bits from MSB to LSB. At each bit it
looks at three bit values — current key, lower-corner key, upper-
corner key — and picks one of six branches that either (a) descends
into the next bit, (b) sets a candidate `BIGMIN` and restricts the
upper corner, or (c) commits and returns. The runtime cost is one
loop of 32 bit-decisions per skip.

## Lessons Learned

- The skip step is what beats a `kd`-tree on this distribution: for
  small queries the BIGMIN skip is essentially free per query, while
  a kd-tree pays log-N for every per-point candidate.
- Storing `(x, y)` as `u16` next to the key avoids a round-trip
  through `morton_decode` per candidate, which is the inner-loop
  bottleneck on small windows that mostly fall inside the rect.
- A single Morton key fits in a `u32` because each axis is 16 bits.
  Packing into `u64` for 32-bit-per-axis would cost double the cache
  footprint and is unnecessary at this resolution.

## Sources

- Tropf, Herzog. **Multidimensional Range Search in Dynamically
  Balanced Trees.** Angewandte Informatik, 1981.
- Morton, G. M. **A computer Oriented Geodetic Data Base; and a New
  Technique in File Sequencing.** IBM Ltd, 1966.
