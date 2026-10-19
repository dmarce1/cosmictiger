#!/bin/bash
# Re-check of the cube-root rule after merging main (34d4aa88 keeps empty
# source cells out of the CPU walk, 327555a8 replaces FastFuture): automatic
# divisor against pinned 164 and 328 at 262,144 particles, 8 worker threads,
# bucket 128, order 7, two pairs of 11 samples, under the shared GPU lock.
# Run from the repository root.
set -eu
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_rule_merged}
OUT=docs/cuda_kick_shared_memory_results/handoff_rule_check
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
stage=$OUT/t8_n64_merged
[ -e "$stage/runs.json" ] && { echo "skip $stage (exists)"; exit 0; }
echo "== $stage $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg) GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
flock -w 14400 "$GPU_LOCK" \
    python3 benchmarks/run_cuda_kick_arena.py \
    --binary merged=$BIN:arena \
    --policy auto= \
    --policy handoff164=--compute.cudaKickHandoffBlocks=164 \
    --policy handoff328=--compute.cudaKickHandoffBlocks=328 \
    --output $stage \
    --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
    --threads 8 --particles-per-dimension 64 \
    --pairs 2 --samples 11 --reference 0 --timeout 3600 \
    --idle-wait 'ninja|ncu|nsys' --wait-gpu-idle
python3 benchmarks/check_kick_arena_pairs.py $stage --max-spread 1.2 || echo "WARNING: pairs disagree, repeat this stage"
