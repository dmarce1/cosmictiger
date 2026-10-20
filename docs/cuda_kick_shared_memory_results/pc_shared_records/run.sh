#!/bin/bash
# Whole-kick timing of the PC loop reading its staged records and arena force
# planes through pointers derived from the dynamic shared-memory symbol (LDS
# per record word, LDS/STS force updates) against the head it was cut from
# (da3c1f4e, origin/main), both built from the same 1024-bucket Release
# configuration (COSMICTIGER_WITH_MAX_BUCKET=1024, CUDA on, order 7 selected at
# run time): automatic policy, buckets 64 and 512, FMM order 7, all three
# fixtures with the potential off and on, five pairs of 15 samples, direct
# reference over 1,024 particles. Both binaries are this worktree's
# build/release/cosmictiger frozen after the respective build (frozen/); the
# register-copy form's run of the same protocol is in registers_variant/. Run
# from the repository root with a quiet machine (no builds, no other GPU
# client; the driver also waits for compilers, profilers and GPU compute
# processes before each run). One GPU job at a time across the concurrent
# sessions of this workstation: re-execute under the shared benchmark lock,
# taken only after the machine has been quiet (stage.sh waits first, then
# locks).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -eu
HEAD=build/release/frozen/cosmictiger_b1024_head_da3c1f4e
SHARED=build/release/frozen/cosmictiger_b1024_pc_shared
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/pc_shared_records}
python3 benchmarks/run_cuda_kick_arena.py \
    --binary head=$HEAD:arena \
    --binary shared=$SHARED:arena \
    --policy auto= \
    --output "$OUTPUT" \
    --fixtures lattice,clustered,adaptive --buckets 64,512 --orders 7 \
    --pairs 5 --samples 15 --reference 1024 --timeout 1800 \
    --idle-wait 'ninja|ncu|nsys|cc1plus|cicc|ptxas' --wait-gpu-idle
python3 benchmarks/check_kick_arena_pairs.py "$OUTPUT" || true
python3 benchmarks/compare_kick_arena_binaries.py "$OUTPUT" head shared \
    --output "$OUTPUT/head_vs_shared.md"
python3 benchmarks/compare_kick_arena_runs.py "$OUTPUT" --reference head:auto \
    --output "$OUTPUT/paired_speedup.md"
