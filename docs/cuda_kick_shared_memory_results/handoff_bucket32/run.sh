#!/bin/bash
# Bucket-32 diagnostic at FMM order 7: does the CPU/GPU handoff divisor (656
# arena blocks against the baseline's 492) explain the arena's loss at small
# buckets? Arena auto against the divisor pinned to 492, residency 6 (which
# also gives 492 blocks), and no list reservation. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_231325d5
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena=$BIN:arena \
    --policy auto= \
    --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
    --policy res6=--compute.cudaKickResidency=6 \
    --policy reserve0=--compute.cudaKickListReserve=0 \
    --output docs/cuda_kick_shared_memory_results/handoff_bucket32 \
    --fixtures lattice,clustered,adaptive --buckets 32,64 --orders 7 \
    --pairs 2 --samples 11 --reference 0 --timeout 1800
