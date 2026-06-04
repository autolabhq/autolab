#!/bin/bash
# Apply the optimized Huffman decode reference implementation.
set -e
cp /solution/solve_optimized.cu /app/solve.cu
make -C /app -B
