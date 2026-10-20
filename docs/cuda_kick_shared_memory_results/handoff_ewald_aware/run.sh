#!/bin/bash
# Ewald-aware CPU/GPU handoff at 2,097,152 particles (side 128): a node that
# still carries Ewald checks goes to the GPU at the automatic rule's threshold
# (492 blocks, 4,262 particles), one without them at the finer threshold of
# compute.cudaKickFineHandoffBlocks (656, 984 or 1,312 blocks: 3,197, 2,131 or
# 1,598 particles). Policies: auto (the rule, 492), handoff492 and handoff656
# (pinned, every node), fine656, fine984 and fine1312, all three fixtures,
# bucket 128, order 7, two pairs of 11 samples, forces and potential, at 8
# and 12 HPX worker threads pinned to CPUs 1 through the thread count. One
# driver run per thread count: t<threads>_n128/.
#
# Hygiene as in ../handoff_large_sizes/run.sh, with the quiet wait moved in
# front of the lock: wait for a minute without builds, profilers, GPU compute
# processes or another driver and with less than four cores busy, then take
# the lock file the GPU-timing sessions share and re-check; a throwaway
# side-64 process ramps the GPU clocks before pair 0. A stage whose pairs
# disagree by more than 20% is moved to ../handoff_ewald_aware_contaminated/
# and repeated, up to three times. After a clean stage the per-fixture digests
# of the force exports go to fixture_digests.{json,md}, then the 900 MB
# exports are deleted (runs.json keeps each export's sha256). Run from the
# repository root.
set -eu
GPU_LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
BIN=${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_ewald_handoff}
OUT=docs/cuda_kick_shared_memory_results/handoff_ewald_aware
BAD=docs/cuda_kick_shared_memory_results/handoff_ewald_aware_contaminated
WARMUP=${WARMUP:-$(mktemp -d)}
BUSY='ninja|ncu|nsys|cc1plus|cicc|ptxas|nvcc'

# CPU time outside idle and IO wait over two seconds, in cores, of at least
# four. Not the load average: a home-directory backup and a file-system scan
# kept eight processes in IO wait (load 12 with the CPUs 64% idle) during this
# stage, and IO wait does not compete with the benchmark's threads.
cpu_busy() {
    local cpu u1 n1 s1 i1 w1 q1 sq1 st1 u2 n2 s2 i2 w2 q2 sq2 st2 rest
    read -r cpu u1 n1 s1 i1 w1 q1 sq1 st1 rest < /proc/stat
    sleep 2
    read -r cpu u2 n2 s2 i2 w2 q2 sq2 st2 rest < /proc/stat
    local used=$(((u2 + n2 + s2 + q2 + sq2 + st2) - (u1 + n1 + s1 + q1 + sq1 + st1)))
    local idle=$(((i2 + w2) - (i1 + w1)))
    awk -v used=$used -v idle=$idle -v cpus=$(nproc) \
        'BEGIN { exit !(used * cpus >= 4 * (used + idle)) }'
}

busy() {
    pgrep -x "$BUSY" > /dev/null ||
        pgrep -f '[r]un_cuda_kick_arena.py' > /dev/null ||
        [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] ||
        cpu_busy
}

wait_quiet() {
    local quiet=0
    while [ $quiet -lt "$1" ]; do
        if busy; then
            quiet=0
        else
            quiet=$((quiet + 1))
        fi
        sleep 10
    done
}

for threads in ${THREADS:-8 12}; do
    stage=$OUT/t${threads}_n128
    for attempt in 1 2 3; do
        if [ -e "$stage/runs.json" ]; then
            echo "skip $stage (exists)"
            break
        fi
        while true; do
            echo "waiting for a quiet machine $(date '+%H:%M:%S')"
            wait_quiet 6
            exec 9>"$GPU_LOCK"
            flock -w 14400 9
            if ! busy; then
                break
            fi
            flock -u 9
        done
        echo "== $stage attempt $attempt $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg)"
        (cd "$WARMUP" && COSMICTIGER_STF_WALK_SAMPLES=2 COSMICTIGER_STF_WALK_BUCKET=128 \
            COSMICTIGER_STF_WALK_FIXTURE=lattice taskset -c 1-$threads "$OLDPWD/$BIN" \
            --run.test=stfBenchmark --initial.particlesPerDimension=64 --hydro.enabled=false \
            --gravity.fmmOrder=7 --compute.cuda=true --compute.bucketSize=128 \
            --hpx:threads=$threads --hpx:bind=none --analysis.views=false \
            --analysis.power=false --analysis.groups=false --lightcone.enabled=false \
            > warmup.log 2>&1; rm -f stf_walk_forces.csv)
        python3 benchmarks/run_cuda_kick_arena.py \
            --binary ewald=$BIN:arena \
            --policy auto= \
            --policy handoff492=--compute.cudaKickHandoffBlocks=492 \
            --policy handoff656=--compute.cudaKickHandoffBlocks=656 \
            --policy fine656=--compute.cudaKickFineHandoffBlocks=656 \
            --policy fine984=--compute.cudaKickFineHandoffBlocks=984 \
            --policy fine1312=--compute.cudaKickFineHandoffBlocks=1312 \
            --output $stage \
            --fixtures lattice,clustered,adaptive --buckets 128 --orders 7 \
            --threads $threads --particles-per-dimension 128 \
            --pairs 2 --samples 11 --reference 0 --timeout 7200 \
            --idle-wait "$BUSY" --wait-gpu-idle
        flock -u 9
        if python3 benchmarks/check_kick_arena_pairs.py $stage --max-spread 1.2; then
            python3 benchmarks/kick_arena_fixture_digests.py $stage \
                --output $stage/fixture_digests.md
            rm -f $stage/runs/*/stf_walk_forces.csv
        else
            mkdir -p $BAD
            moved=$BAD/t${threads}_n128_$(date '+%H%M%S')
            mv $stage $moved
            rm -f $moved/runs/*/stf_walk_forces.csv
            echo "contaminated: moved to $moved"
        fi
    done
done
