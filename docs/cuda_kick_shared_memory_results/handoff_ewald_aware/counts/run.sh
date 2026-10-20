#!/bin/bash
# Handoff counts per policy at 2,097,152 particles, bucket 128, order 7, 8
# threads: the stage's policies run once (one pair, one sample) with
# cosmictiger_b1024_ewald_handoff_counts, the frozen binary with
# instrumentation.patch applied (KickWorkspace::toGpu prints the workitems and
# the checks they carry, the STF walk the CPU walk's interaction counts).
# Counts only: the instrumented binary is not timed. Under the shared GPU lock
# after the quiet wait of ../run.sh; the exports are deleted afterwards. Run
# from the repository root.
set -eu
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_ewald_handoff_counts}
OUT=docs/cuda_kick_shared_memory_results/handoff_ewald_aware/counts
BUSY='ninja|ncu|nsys|cc1plus|cicc|ptxas|nvcc'
stage=$OUT/t8_n128
quiet=0
while [ $quiet -lt 3 ]; do
    if pgrep -x "$BUSY" > /dev/null || pgrep -f '[r]un_cuda_kick_arena.py' > /dev/null ||
        [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        quiet=0
    else
        quiet=$((quiet + 1))
    fi
    sleep 10
done
exec 9>"$GPU_LOCK"
flock -w 14400 9
python3 benchmarks/run_cuda_kick_arena.py \
    --binary counts=$BIN:arena \
    --policy auto= \
    --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
    --policy handoff656=--compute.cudaKickHandoffBlocks=656 \
    --policy fine656=--compute.cudaKickFineHandoffBlocks=656 \
    --policy fine984=--compute.cudaKickFineHandoffBlocks=984 \
    --policy fine1312=--compute.cudaKickFineHandoffBlocks=1312 \
    --output $stage \
    --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
    --threads 8 --particles-per-dimension 128 \
    --pairs 1 --samples 1 --reference 0 --timeout 7200 \
    --idle-wait "$BUSY" --wait-gpu-idle
flock -u 9
python3 benchmarks/kick_arena_fixture_digests.py $stage --output $stage/fixture_digests.md
rm -f $stage/runs/*/stf_walk_forces.csv
python3 $OUT/summarize_counts.py $stage --output $OUT/counts.md
