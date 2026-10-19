#!/bin/bash
# Opening-angle sweep over FMM orders 3 to 9 at bucket 64 with the direct
# reference over 1024 particles, one result directory per theta, for the
# matched-accuracy comparison. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_231325d5
for theta in 0.5 0.6 0.7 0.8 0.9 1.0; do
    python3 benchmarks/run_cuda_kick_arena.py \
        --binary arena=$BIN:arena --policy auto= \
        --output docs/cuda_kick_shared_memory_results/theta_sweep/theta_$theta \
        --fixtures lattice,clustered,adaptive --buckets 64 --orders 3,4,5,6,7,8,9 \
        --pairs 2 --samples 15 --reference 1024 --theta $theta --timeout 3600
done
