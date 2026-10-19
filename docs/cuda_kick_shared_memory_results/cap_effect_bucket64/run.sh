#!/bin/bash
# Does the compile-time bucket cap change the order-7, bucket-64 kick? Baseline
# (585ee6c0) against the 128-cap arena binary of the earlier measurements and
# the 1024-cap build used for the bucket sweep. Run from the repository root.
set -eu
B128=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger/765541c5-5b91-4b03-b04b-7071cc6a863c/scratchpad/cosmictiger_b128
B1024=build/arena_b1024/frozen/cosmictiger_b1024_231325d5
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena_cap128=$B128:arena \
    --binary arena_cap1024=$B1024:arena \
    --policy auto= \
    --output docs/cuda_kick_shared_memory_results/cap_effect_bucket64 \
    --fixtures lattice,clustered,adaptive --buckets 64 --orders 7 \
    --pairs 3 --samples 15 --reference 0 --timeout 1800
