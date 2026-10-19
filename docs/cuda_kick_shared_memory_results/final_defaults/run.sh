#!/bin/bash
# Final check of the arena with its selected defaults (automatic layout,
# residency at the register limit, 4096-entry list reservation, handoff
# divisor of two blocks per multiprocessor) against the baseline, plus the
# arena with the baseline's 492-block divisor to separate the handoff effect
# from the arena itself. Three pairs of 15 samples, direct reference over
# 1,024 particles. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_final
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena=$BIN:arena \
    --policy auto= \
    --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
    --output docs/cuda_kick_shared_memory_results/final_defaults \
    --fixtures lattice,clustered,adaptive --buckets 32,64,128,256 --orders 7 \
    --pairs 3 --samples 15 --reference 1024 --timeout 1800
