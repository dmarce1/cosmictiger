#!/bin/bash
# Does the handoff divisor depend on the thread count or the particle count?
# Divisors 82, 164 (the default: two blocks per multiprocessor), 328, 492 and
# 656 blocks at 2, 4, 8 and 12 HPX worker threads for 32,768 and 262,144
# particles (initial.particlesPerDimension 32 and 64), all three fixtures,
# bucket 128, order 7, two pairs of 11 samples, forces and potential. One
# driver run per thread count and size: t<threads>_n<side>/. The workers are
# pinned to CPUs 1 through the thread count (CPUs 1-11 are distinct physical
# cores of the 12-core workstation; CPU 12 is the sibling of core 0).
#
# Other sessions build and benchmark on this workstation while this runs, so
# a stage takes the lock file that the GPU-timing sessions share (GPU_LOCK,
# held for the duration of the stage; locked sessions alternate), then starts
# only after half a minute without ninja, ncu, nsys, any GPU compute process
# (nvidia-smi) or another run_cuda_kick_arena.py driver (whose processes have
# gaps between them); every process waits for those to finish, and a stage
# whose pairs disagree by more than 20% (clean stages stay within 10%) is
# moved to ../handoff_threads_sizes_contaminated/ and repeated, up to three
# times. Run from the repository root.
set -eu
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_handoff}
OUT=docs/cuda_kick_shared_memory_results/handoff_threads_sizes
BAD=docs/cuda_kick_shared_memory_results/handoff_threads_sizes_contaminated
for side in ${SIDES:-32 64}; do
    for threads in ${THREADS:-2 4 8 12}; do
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
                --policy handoff82=--compute.cudaKickHandoffBlocks=82 \
                --policy handoff164=--compute.cudaKickHandoffBlocks=164 \
                --policy handoff328=--compute.cudaKickHandoffBlocks=328 \
                --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
                --policy handoff656=--compute.cudaKickHandoffBlocks=656 \
                --output $stage \
                --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
                --threads $threads --particles-per-dimension $side \
                --pairs 2 --samples 11 --reference 0 --timeout 3600 \
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
