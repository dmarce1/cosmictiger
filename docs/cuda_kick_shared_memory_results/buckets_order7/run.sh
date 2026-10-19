#!/bin/bash
# Bucket sweep at FMM order 7: the baseline (585ee6c0, fixed order 7) against
# the 1024-cap arena build with whole-bucket, 128- and 64-particle sink tiles
# and global force backing, all three fixtures, direct reference over 1024
# particles. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_231325d5
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena=$BIN:arena \
    --policy auto= \
    --policy tile128=--compute.cudaKickSinkTile=128 \
    --policy tile64=--compute.cudaKickSinkTile=64 \
    --policy global=--compute.cudaKickSinkTile=-1 \
    --output docs/cuda_kick_shared_memory_results/buckets_order7 \
    --fixtures lattice,clustered,adaptive --buckets 32,64,128,256,512,1024 \
    --orders 7 --pairs 3 --samples 15 --reference 1024 --timeout 3600
