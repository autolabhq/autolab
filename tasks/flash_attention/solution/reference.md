# Flash Attention — Reference

## Background
Scaled dot-product attention (`softmax(QK^T / sqrt(d)) V`) is the core compute kernel in every transformer model (BERT, GPT, LLaMA, etc.). At n=4096 and d=64, the naive implementation allocates 256 MB of intermediate state that never fits in CPU cache, making it the dominant bottleneck in inference and training pipelines.

## Baseline Approach
The unoptimized implementation allocates a full n×n score matrix `S` in double precision (128 MB), computes all dot products, then runs a three-pass softmax (find max → exp+sum → divide) over each row, allocates another 128 MB probability matrix `P`, and finally computes `P @ V`. With n=4096, nearly every memory access pays DRAM latency. Runs in ~0.75s.

## Possible Optimization Directions
1. **Flash Attention tiling** — process Q in Br-row blocks, stream K/V in Bc-row blocks; working set (~28 KB) fits in L1 cache; eliminates the n×n matrix entirely (~3× speedup)
2. **Online softmax** — maintain running (max, sum) state per row across K-blocks; merges three softmax passes into one and never materializes the full P matrix
3. **float32 instead of double** — halves memory traffic; AVX2 processes 8 floats/register vs 4 doubles (~2× speedup)
4. **AVX2 SIMD dot products** — reduces d=64 inner product to 8 `vfmadd` instructions per (i,j) pair instead of 64 scalar fmadds (~4× speedup over scalar float32)
5. **Fast vectorized exp approximation** — 5th-order polynomial with range reduction via `exp(x) = 2^k * exp(x - k*ln2)`; ~4 cycles/value vs ~15–20 for `expf()`; applied in AVX2 batches of 8

## Reference Solution
Implements Flash Attention tiling (Br=32, Bc=32) with online softmax, float32 throughout, and AVX2 SIMD for all hot loops: dot products use `vfmadd` with horizontal reduction, exp uses a vectorized 5th-order Horner polynomial with float bit-manipulation for `2^k` scaling, and the P@V accumulation uses `vfmadd` per V row. Memory usage is O(n·d) instead of O(n²). Scalar fallbacks are provided for non-AVX2 machines.

## Source
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022, arXiv:2205.14135)
- Milakov & Gimelshein, *Online Normalizer Calculation for Softmax* (arXiv:1805.02867, 2018)
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (ICLR 2024, arXiv:2307.08691)
- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
