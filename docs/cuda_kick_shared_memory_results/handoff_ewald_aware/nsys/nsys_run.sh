#!/bin/bash
# Nsight Systems host/kernel split of the Ewald-aware handoff on the adaptive
# fixture at 2,097,152 particles, where the timing stages put fine656-1312
# behind the automatic rule: auto (492 blocks for every node) against fine656
# and fine1312 (492 for nodes with Ewald checks, the finer divisor for the
# rest) and the pinned 656. Bucket 128, FMM order 7, 8 HPX worker threads
# pinned to CPUs 1-8, 11 samples, forces and potential, the frozen binary of
# ../run.sh. Three rounds with the variant order alternating; CPU sampling
# and context-switch tracing are off, as in ../../nsys_handoff_large/.
#
# Hygiene as in ../run.sh: wait for a minute without builds, profilers, GPU
# compute processes or a kick arena driver and with less than four cores
# busy, then take the shared GPU lock for the whole stage and re-check before
# each process. Each process's 300 MB stf_walk_forces.csv is deleted once it
# has exited. Run from the repository root; summarize with
#   python3 benchmarks/summarize_kick_handoff_profile.py DIR > DIR/summary.md
set -u
LOCK=${GPU_LOCK:-/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock}
DIR=docs/cuda_kick_shared_memory_results/handoff_ewald_aware/nsys
BIN=$PWD/${BIN:-build/arena_b1024/frozen/cosmictiger_b1024_ewald_handoff}
NSYS=${NSYS:-nsys}
THREADS=8
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=128 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=$THREADS --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=128 --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=11 COSMICTIGER_STF_WALK_BUCKET=128 COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=$THREADS OMP_THREAD_LIMIT=$THREADS
BUSY='ninja|ncu|nsys|cc1plus|cicc|ptxas|nvcc'
source <(sed -n '/^cpu_busy() {/,/^}/p; /^busy() {/,/^}/p' docs/cuda_kick_shared_memory_results/handoff_ewald_aware/run.sh)
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
declare -A OPTIONS=([auto]="" [fine656]="--compute.cudaKickFineHandoffBlocks=656"
    [fine1312]="--compute.cudaKickFineHandoffBlocks=1312"
    [h656]="--compute.cudaKickHandoffBlocks=656")
echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
while true; do
    echo "waiting for a quiet machine $(date '+%H:%M:%S')"
    wait_quiet 6
    exec 9>"$LOCK"
    flock -w 14400 9
    if ! busy; then
        break
    fi
    flock -u 9
done
echo "$(date -Is) GPU lock acquired"
: > $DIR/status.txt
for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then order="auto fine656 fine1312 h656"; else order="h656 fine1312 fine656 auto"; fi
    for v in $order; do
        d=$DIR/adaptive128_${v}_$round
        mkdir -p $d
        wait_quiet 1
        echo "== $d $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg)"
        (cd $d && COSMICTIGER_STF_WALK_FIXTURE=adaptive \
            timeout 1800 $NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true -o kick \
            taskset -c 1-$THREADS $BIN $COMMON ${OPTIONS[$v]} > run.log 2>&1
            echo "$d exit $? $(date '+%H:%M:%S')" >> ../status.txt)
        rm -f $d/stf_walk_forces.csv
        (cd $d && $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum,cuda_api_trace -f csv -o stats kick.nsys-rep > stats.log 2>&1)
    done
done
flock -u 9
echo DONE >> $DIR/status.txt
