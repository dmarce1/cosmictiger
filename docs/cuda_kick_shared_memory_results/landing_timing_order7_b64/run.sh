#!/bin/bash
# Whole-kick timing of the merged main (819d0013: per-launch list slab, PP
# loop schedule and racecheck explanation landed on the reorganized layout)
# against the pre-merge main (00909d26), both built from the same 1024-bucket
# Release configuration (COSMICTIGER_WITH_MAX_BUCKET=1024, CUDA on, order 7
# selected at run time): automatic policy, bucket 64, FMM order 7, all three
# fixtures with the potential off and on, five pairs of 15 samples, direct
# reference over 1,024 particles. The frozen binaries are copies of each
# worktree's build/release/cosmictiger. Run from the repository root with a
# quiet machine (no builds, no other GPU client; the driver also waits for
# compilers, profilers and GPU compute processes before each run).
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -eu
PREMERGE=build/release/frozen/cosmictiger_b1024_premerge_00909d26
MERGED=build/release/frozen/cosmictiger_b1024_merged_819d0013
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/landing_timing_order7_b64}
python3 benchmarks/run_cuda_kick_arena.py \
    --binary premerge=$PREMERGE:arena \
    --binary merged=$MERGED:arena \
    --policy auto= \
    --output "$OUTPUT" \
    --fixtures lattice,clustered,adaptive --buckets 64 --orders 7 \
    --pairs 5 --samples 15 --reference 1024 --timeout 1800 \
    --idle-wait 'ninja|ncu|nsys|cc1plus|cicc|ptxas' --wait-gpu-idle
python3 benchmarks/check_kick_arena_pairs.py "$OUTPUT" || true
python3 benchmarks/compare_kick_arena_binaries.py "$OUTPUT" premerge merged \
    --output "$OUTPUT/premerge_vs_merged.md"
