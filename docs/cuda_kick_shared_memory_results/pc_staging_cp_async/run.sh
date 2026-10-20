#!/bin/bash
# Whole-kick timing of the aligned PC multipole staging (cp.async copies of
# whole 16-byte-stride records) against the head it was cut from (819d0013,
# the code of c4cc57d5), both built from the same 1024-bucket Release
# configuration (COSMICTIGER_WITH_MAX_BUCKET=1024, CUDA on, order 7 selected
# at run time): automatic policy, buckets 64 and 512, FMM order 7, all three
# fixtures with the potential off and on, five pairs of 15 samples, direct
# reference over 1,024 particles. The head binary is the landing-timing
# session's frozen copy of 819d0013 (sha256 67f8fca1...); the candidate is this
# worktree's build/release/cosmictiger frozen after the build. Run from the
# repository root with a quiet machine (no builds, no other GPU client; the
# driver also waits for compilers, profilers and GPU compute processes before
# each run). One GPU job at a time across the concurrent sessions of this
# workstation: re-execute under the shared benchmark lock, taken only after the
# machine has been quiet (wait first, then lock).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -eu
HEAD=build/release/frozen/cosmictiger_b1024_head_819d0013
ALIGNED=build/release/frozen/cosmictiger_b1024_pc_aligned
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/pc_staging_cp_async}
python3 benchmarks/run_cuda_kick_arena.py \
    --binary head=$HEAD:arena \
    --binary aligned=$ALIGNED:arena \
    --policy auto= \
    --output "$OUTPUT" \
    --fixtures lattice,clustered,adaptive --buckets 64,512 --orders 7 \
    --pairs 5 --samples 15 --reference 1024 --timeout 1800 \
    --idle-wait 'ninja|ncu|nsys|cc1plus|cicc|ptxas' --wait-gpu-idle
python3 benchmarks/check_kick_arena_pairs.py "$OUTPUT" || true
python3 benchmarks/compare_kick_arena_binaries.py "$OUTPUT" head aligned \
    --output "$OUTPUT/head_vs_aligned.md"
