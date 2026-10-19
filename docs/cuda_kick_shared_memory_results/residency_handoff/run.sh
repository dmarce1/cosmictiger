#!/bin/bash
# Residency and handoff sweep at FMM order 7 with the build whose requested
# residency is honoured (fewer launched blocks and a coarser CPU/GPU handoff):
# baseline against the arena with the automatic policy (8 blocks/SM, 656
# blocks), residency 6 and 7, and the handoff divisor pinned to 492, 328 and
# 164 blocks at full residency. Run from the repository root.
set -eu
BIN=build/arena_b1024/frozen/cosmictiger_b1024_fix2
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary arena=$BIN:arena \
    --policy auto= \
    --policy res6=--compute.cudaKickResidency=6 \
    --policy res7=--compute.cudaKickResidency=7 \
    --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
    --policy handoff328=--compute.cudaKickHandoffBlocks=328 \
    --policy handoff164=--compute.cudaKickHandoffBlocks=164 \
    --output docs/cuda_kick_shared_memory_results/residency_handoff \
    --fixtures lattice,clustered,adaptive --buckets 32,64,128 --orders 7 \
    --pairs 2 --samples 11 --reference 0 --timeout 1800
