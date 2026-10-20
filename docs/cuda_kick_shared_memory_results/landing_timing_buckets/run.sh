#!/bin/bash
# Whole-kick timing of the merged main (819d0013: per-launch list slab, PP
# loop schedule and racecheck explanation landed on the reorganized layout)
# against the pre-merge main (00909d26) at buckets 128, 256 and 512, extending
# landing_timing_order7_b64/ with the same frozen 1024-bucket Release binaries
# (COSMICTIGER_WITH_MAX_BUCKET=1024, CUDA on, order 7 selected at run time):
# automatic policy, FMM order 7, all three fixtures with the potential off and
# on, five pairs of 15 samples, direct reference over 1,024 particles. One
# driver run per bucket (b128/, b256/, b512/); a stage whose worst pair spread
# exceeds 1.15 is moved to contaminated/ and repeated, at most three attempts.
# Each stage ends with the force-export comparison (premerge_vs_merged.md) and
# the paired speedup table (speedup.md). Run from the repository root with a
# quiet machine (no builds, no other GPU client; the driver also waits for
# compilers, profilers and GPU compute processes before each run).
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -u
PREMERGE=build/release/frozen/cosmictiger_b1024_premerge_00909d26
MERGED=build/release/frozen/cosmictiger_b1024_merged_819d0013
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/landing_timing_buckets}
for bucket in 128 256 512; do
    stage=$OUTPUT/b$bucket
    for attempt in 1 2 3; do
        rm -rf "$stage"
        python3 benchmarks/run_cuda_kick_arena.py \
            --binary premerge=$PREMERGE:arena \
            --binary merged=$MERGED:arena \
            --policy auto= \
            --output "$stage" \
            --fixtures lattice,clustered,adaptive --buckets $bucket --orders 7 \
            --pairs 5 --samples 15 --reference 1024 --timeout 1800 \
            --idle-wait 'ninja|ncu|nsys|cc1plus|cicc|ptxas' --wait-gpu-idle
        if python3 benchmarks/check_kick_arena_pairs.py "$stage" --max-spread 1.15; then
            break
        fi
        mkdir -p "$OUTPUT/contaminated"
        mv "$stage" "$OUTPUT/contaminated/b${bucket}_attempt$attempt"
        echo "$(date -Is) bucket $bucket attempt $attempt exceeded the 1.15 pair spread; repeating"
    done
    if [ ! -d "$stage" ]; then
        echo "$(date -Is) bucket $bucket: three contaminated attempts, no clean stage"
        continue
    fi
    python3 benchmarks/compare_kick_arena_binaries.py "$stage" premerge merged \
        --output "$stage/premerge_vs_merged.md"
    python3 benchmarks/compare_kick_arena_runs.py "$stage" --reference premerge:auto \
        --output "$stage/speedup.md"
done
echo "$(date -Is) bucket stages done"
