#!/bin/bash
# Does restoring the source tiles after the automatic sink-tile fallback
# recover the bucket-1024 loss? Baseline against the fixed arena build with the
# automatic policy and the explicit 128-particle sink tile. Run from the
# repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_fix
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena_fix=$BIN:arena \
    --policy auto= \
    --policy tile128=--compute.cudaKickSinkTile=128 \
    --output docs/cuda_kick_shared_memory_results/layout_fix_bucket1024 \
    --fixtures lattice,clustered,adaptive --buckets 512,1024 --orders 7 \
    --pairs 2 --samples 11 --reference 1024 --timeout 1800
