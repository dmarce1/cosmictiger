#!/bin/bash
# Ewald cell-cell schedule: the arena with the Fourier sum evaluated ahead of
# the real-space sum ("ewald") against the unchanged baseline and the head it
# was built from (origin/main da3c1f4e, "head"), buckets 128, 256 and 512,
# order 7, all three fixtures with the potential off and on, three pairs of
# 15 samples, direct reference over 1,024 particles. One driver run per
# bucket (b128/, b256/, b512/); a stage whose worst pair spread exceeds 1.15
# is moved to contaminated/ and repeated, at most three attempts. Each stage
# ends with the force-export comparison (head_vs_ewald.md) and the paired
# speedup tables against the head and the baseline. Run from the repository
# root on a quiet machine; the driver also waits for compilers, profilers
# and GPU compute processes before each run. One GPU job at a time across
# the sessions of this workstation: re-executes itself under the shared
# benchmark lock.
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    # Wait for a quiet machine first (no builds, profilers or GPU clients for
    # 30 s), then take the lock; the lock is not held while waiting.
    quiet=0
    while [ $quiet -lt 2 ]; do
        if pgrep -x "ninja|ncu|nsys|cc1plus|cicc|ptxas" > /dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
            quiet=0
        else
            quiet=$((quiet + 1))
        fi
        sleep 15
    done
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -u
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
HEAD=build/arena_b1024/frozen/cosmictiger_b1024_head_da3c1f4e
EWALD=build/arena_b1024/frozen/cosmictiger_b1024_ewald
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/ewald_schedule}
PAIRS=${PAIRS:-3}
for bucket in ${BUCKETS:-128 256 512}; do
    stage=$OUTPUT/b$bucket
    for attempt in 1 2 3; do
        rm -rf "$stage"
        python3 benchmarks/run_cuda_kick_arena.py \
            --binary baseline=$BASELINE \
            --binary head=$HEAD:arena \
            --binary ewald=$EWALD:arena \
            --policy auto= \
            --output "$stage" \
            --fixtures lattice,clustered,adaptive --buckets $bucket --orders 7 \
            --pairs $PAIRS --samples 15 --reference 1024 --timeout 1800 \
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
    python3 benchmarks/compare_kick_arena_binaries.py "$stage" head ewald \
        --output "$stage/head_vs_ewald.md"
    python3 benchmarks/compare_kick_arena_runs.py "$stage" --reference head:auto \
        --output "$stage/speedup_vs_head.md"
    python3 benchmarks/compare_kick_arena_runs.py "$stage" --reference baseline:auto \
        --output "$stage/speedup_vs_baseline.md"
done
echo "$(date -Is) bucket stages done"
