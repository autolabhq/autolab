# ICP Correspondence Step -- Reference

## Background

The Iterative Closest Point (ICP) algorithm and its many variants (point-to-plane,
GICP, VGICP) sit at the core of LiDAR odometry, real-time SLAM, and 3D scan
registration. Each ICP iteration is dominated by a *correspondence step*: for
every source point find the closest target point, then accumulate the 3x3
cross-covariance matrix `H = sum_i (p_i - p_bar)(q_i - q_bar)^T` and the sum of
squared residuals `err = sum_i ||p_i - q_i||^2` over the matched pairs. This
3x3 matrix is then SVD-decomposed off-GPU to recover the next rigid-body
update. With a 200K source / 500K target cloud the correspondence step runs
many times per second on a robot, so it pays to push it onto the GPU.

## Baseline Approach

The naive baseline is `O(N * M)`: one CUDA thread per source point loops over
the entire target cloud to find its nearest neighbor, then atomically adds its
contribution to a 9-float global cross-covariance buffer. With `N = 200,000`
and `M = 500,000` that is `10^11` distance evaluations, none of the
`atomicAdd`s are coalesced, and the implementation has to sync back to the
host between the centroid pass and the cross-covariance pass to divide by the
match count. On an H100 the brute-force version is GMEM-latency-bound on the
target reads and atomic-bound on the H accumulation.

## Possible Optimization Directions

1. **KD-tree NN search.** A precomputed balanced KD-tree over the target cloud
   (built once on the host before timing begins; passed as a flat node array
   via `kd_nodes`) reduces NN cost from `O(M)` to roughly `O(log M)` per
   source point. Iterative DFS with a small register stack (depth ~ `log2 M`
   plus a slack of 8) avoids any global-memory stack. Visit the nearer child
   first so the running `best_d2` tightens early and prunes more subtrees.
   Use a squared-distance-to-AABB test for branch elimination.

2. **Block- and warp-level reductions for H.** Each thread computes its 9
   covariance contributions and a single `err` scalar, then the warp reduces
   them with `__shfl_down_sync`, the lane-0 of each warp adds into a
   shared-memory tile, and one `atomicAdd` per block reaches global memory.
   This drops global atomic traffic by ~`BLOCK_SIZE` x.

3. **Skip the host round-trip for the centroid divide.** A tiny single-block
   kernel reads the partial sums and the count, divides on-GPU, and writes
   the centroids back. The second pass picks up the centroids without ever
   `cudaMemcpy`ing back to the host.

4. **fp64 accumulators, fp32 outputs.** With `~10^5` matched pairs, fp32
   summation of `H` accumulates noticeable round-off; using fp64 accumulators
   keeps the `1e-4` relative tolerance slack while only doubling register
   pressure on a small constant set of values.

## Reference Solution

The shipped reference (`solution/solve_optimized.cu`) combines all four
directions:

- `nn_pass_kernel`: KD-tree iterative DFS with a 64-entry register stack,
  near-first ordered traversal, AABB pruning. Each thread that finds a valid
  match contributes `(p, q, 1)` into a per-warp shuffle reduction, then one
  block-level atomic into 6 doubles + 1 int.
- `centroid_div_kernel`: 1-block, 32-thread divide-by-count.
- `H_pass_kernel`: each thread computes `(p-cP)(q-cQ)^T` and the squared
  distance contribution, warp-reduces all 10 doubles with `__shfl_down_sync`,
  block-reduces in shared memory, then one `atomicAdd` per H entry per block.
- `finalize_kernel`: 1-block fp64 -> fp32 cast.

## Sources

- Koide, K., Yokozuka, M., Oishi, S., Banno, A. (2021). *Voxelized GICP for
  Fast and Accurate 3D Point Cloud Registration.* ICRA 2021.
  https://staff.aist.go.jp/shuji.oishi/assets/papers/preprint/VoxelGICP_ICRA2021.pdf
- `fast_gicp` reference implementation (CPU + CUDA VGICP):
  https://github.com/koide3/fast_gicp
- Popov, S., Gunther, J., Seidel, H.-P., Slusallek, P. (2007). *Stackless
  KD-Tree Traversal for High Performance GPU Ray Tracing.* Computer Graphics
  Forum 26(3): 415-424.
- Wald, I. (2022). *cudaKDTree: A KD-tree library for CUDA with GPU build and
  GPU traversal for k-NN / closest-point queries.*
  https://github.com/ingowald/cudaKDTree
- Brown, S., Snoeyink, J. (2010). *GPU Nearest Neighbor Searches Using a
  Minimal kd-tree.* NVIDIA GTC 2010.
  https://www.nvidia.com/content/gtc-2010/pdfs/2140_gtc2010.pdf
- Heo, J., Kim, T. (2016). *Parallel Tree Traversal for Nearest Neighbor
  Query on the GPU.* International Conference on Parallel Processing (ICPP).
  http://dicl.skku.edu/publications/icpp2016.pdf
