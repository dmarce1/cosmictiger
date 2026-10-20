#!/bin/bash
# PP loop schedule: the arena with the restructured PP loop against the
# unchanged baseline and the arena head it was built from (same walk, old
# loop). Three pairs of 15 samples, direct reference over 1,024 particles,
# buckets 32 to 512, order 7. Run from the repository root with a quiet
# machine; the frozen binaries are copies of build/arena_b1024/cosmictiger.
set -eu
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
HEAD=build/arena_b1024/frozen/cosmictiger_b1024_head_cf66eacc
ARENA=build/arena_b1024/frozen/cosmictiger_b1024_v2
python3 benchmarks/run_cuda_kick_arena.py \
    --binary baseline=$BASELINE \
    --binary head=$HEAD:arena \
    --binary arena=$ARENA:arena \
    --policy auto= \
    --output docs/cuda_kick_shared_memory_results/pp_schedule \
    --fixtures lattice,clustered,adaptive --buckets 32,64,128,256,512 --orders 7 \
    --pairs 3 --samples 15 --reference 1024 --timeout 1800
