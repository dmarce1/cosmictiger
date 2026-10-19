#!/bin/bash
# Check that the automatic handoff rule with kKickHandoffMaxScale = 3 (six
# blocks per multiprocessor at and above 884,736 particles) reproduces the
# pinned 492-block division it was derived from at 2,097,152 particles: auto
# against handoff328 (the previous rule's value) and handoff492 at side 128,
# 8 worker threads, bucket 128, order 7, two pairs of 11 samples, under the
# shared GPU lock and the quiet gate of ../handoff_large_sizes/run.sh. Run
# from the repository root.
set -eu
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_handoff_large_rule}
OUT=docs/cuda_kick_shared_memory_results/handoff_rule_check_large
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
for threads in ${THREADS:-8}; do
    stage=$OUT/t${threads}_n128
    [ -e "$stage/runs.json" ] && { echo "skip $stage (exists)"; continue; }
    exec 9>"$GPU_LOCK"
    flock -w 14400 9
    quiet=0
    while [ $quiet -lt 3 ]; do
        if pgrep -x 'ninja|ncu|nsys' > /dev/null ||
            pgrep -f '[r]un_cuda_kick_arena.py' > /dev/null ||
            [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
            echo "waiting for a running build or benchmark $(date '+%H:%M:%S')"
            quiet=0
        else
            quiet=$((quiet + 1))
        fi
        sleep 10
    done
    echo "== $stage $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg) GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
    python3 benchmarks/run_cuda_kick_arena.py \
        --binary rule=$BIN:arena \
        --policy auto= \
        --policy handoff328=--compute.cudaKickHandoffBlocks=328 \
        --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
        --output $stage \
        --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
        --threads $threads --particles-per-dimension 128 \
        --pairs 2 --samples 11 --reference 0 --timeout 7200 \
        --idle-wait 'ninja|ncu|nsys' --wait-gpu-idle
    flock -u 9
    python3 benchmarks/check_kick_arena_pairs.py $stage --max-spread 1.2 || echo "WARNING: pairs disagree, repeat this stage"
done
