#!/bin/bash
# Per-launch list slab against the pre-slab branch head (eebab5b1, the merge of
# the arena fixes into the current main) and the fixed-order baseline: automatic
# policy, buckets 64 and 128 at FMM order 7, three pairs of 15 samples, direct
# reference over 1,024 particles. The "matched" policy pins the arena binaries
# to the baseline's work division and opening rule (492-block handoff, open
# both cells) so their kernels do the baseline's work. Both arena binaries come
# from the same 1024-bucket configuration (COSMICTIGER_WITH_MAX_BUCKET=1024).
# Run from the repository root.
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired"
set -eu
HEAD=build/arena_b1024/frozen/cosmictiger_b1024_eebab5b1
SLAB=build/arena_b1024/frozen/cosmictiger_b1024_slab
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
python3 benchmarks/run_cuda_kick_arena.py \
    --binary head=$HEAD:arena \
    --binary slab=$SLAB:arena \
    --binary baseline=$BASELINE \
    --policy auto= \
    --policy matched=--compute.cudaKickHandoffBlocks=492,--gravity.openLargerCell=false \
    --output docs/cuda_kick_shared_memory_results/list_slab \
    --fixtures lattice,clustered,adaptive --buckets 64,128 --orders 7 \
    --pairs 3 --samples 15 --reference 1024 --timeout 1800
