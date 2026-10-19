#!/bin/bash
# Where is the handoff optimum? Divisors 164, 82, 41 and 20 blocks (subtrees of
# about 200, 400, 800 and 1,600 particles for 32,768 particles) against the
# automatic rule, two HPX threads. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_fix2
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena=$BIN:arena \
    --policy auto= \
    --policy handoff164=--compute.cudaKickHandoffBlocks=164 \
    --policy handoff82=--compute.cudaKickHandoffBlocks=82 \
    --policy handoff41=--compute.cudaKickHandoffBlocks=41 \
    --policy handoff20=--compute.cudaKickHandoffBlocks=20 \
    --output docs/cuda_kick_shared_memory_results/handoff_coarse \
    --fixtures lattice,clustered,adaptive --buckets 64,128 --orders 7 \
    --pairs 2 --samples 11 --reference 0 --timeout 1800
