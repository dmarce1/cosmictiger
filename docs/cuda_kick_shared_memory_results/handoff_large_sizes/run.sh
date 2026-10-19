#!/bin/bash
# Should the handoff divisor keep growing above four blocks per
# multiprocessor at millions of particles? Divisors 328 (the rule's value at
# and above 262,144 particles: four blocks per multiprocessor), 492, 656,
# 984 and 1312 blocks (6, 8, 12 and 16 per multiprocessor) at 8 and 12 HPX
# worker threads for 2,097,152 particles (initial.particlesPerDimension 128),
# all three fixtures, bucket 128, order 7, two pairs of 11 samples, forces
# and potential. One driver run per thread count: t<threads>_n128/. The
# thresholds N / divisor are 6,394, 4,262, 3,197, 2,131 and 1,598 particles,
# all below the 16,384-particle CUDA_KICK_PARTS_MAX cap. The workers are
# pinned to CPUs 1 through the thread count.
#
# Same hygiene as ../handoff_threads_sizes/run.sh: a stage takes the lock
# file that the GPU-timing sessions share (held for the stage), starts only
# after half a minute without ninja, ncu, nsys, any GPU compute process
# (nvidia-smi) or another driver, and a stage whose pairs disagree by more
# than 20% is moved to ../handoff_large_sizes_contaminated/ and repeated, up
# to three times. Run from the repository root.
set -eu
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_handoff_large}
OUT=docs/cuda_kick_shared_memory_results/handoff_large_sizes
BAD=docs/cuda_kick_shared_memory_results/handoff_large_sizes_contaminated
for side in ${SIDES:-128}; do
    for threads in ${THREADS:-8 12}; do
        stage=$OUT/t${threads}_n${side}
        for attempt in 1 2 3; do
            if [ -e "$stage/runs.json" ]; then
                echo "skip $stage (exists)"
                break
            fi
            # Take the lock first so that locked sessions alternate, then
            # require half a minute without unlocked GPU users. A running
            # cosmictiger is named main-thread#0 by HPX, so GPU users are
            # taken from nvidia-smi rather than by process name.
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
            echo "== $stage attempt $attempt $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg) GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
            python3 benchmarks/run_cuda_kick_arena.py \
                --binary arena=$BIN:arena \
                --policy handoff328=--compute.cudaKickHandoffBlocks=328 \
                --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
                --policy handoff656=--compute.cudaKickHandoffBlocks=656 \
                --policy handoff984=--compute.cudaKickHandoffBlocks=984 \
                --policy handoff1312=--compute.cudaKickHandoffBlocks=1312 \
                --output $stage \
                --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
                --threads $threads --particles-per-dimension $side \
                --pairs 2 --samples 11 --reference 0 --timeout 7200 \
                --idle-wait 'ninja|ncu|nsys' --wait-gpu-idle
            flock -u 9
            if ! python3 benchmarks/check_kick_arena_pairs.py $stage --max-spread 1.2; then
                mkdir -p $BAD
                mv $stage $BAD/t${threads}_n${side}_$(date '+%H%M%S')
                echo "contaminated: moved to $BAD"
            fi
        done
    done
done
