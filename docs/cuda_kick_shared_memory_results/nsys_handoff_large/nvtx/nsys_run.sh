#!/bin/bash
# The stage of ../nsys_run.sh (handoff divisors 328 and 656 on the lattice
# and adaptive fixtures, 2,097,152 particles, bucket 128, order 7, 8 worker
# threads on CPUs 1-8, 11 samples, three rounds; the lattice also with 4
# workers on CPUs 1-4, to see which phases scale with the thread count) with
# a binary built from faaf984a plus instrumentation.patch: NVTX start/end
# ranges around the kick in the walk benchmark, KickWorkspace::addWork (entry
# and mutex hold), the stages of KickWorkspace::toGpu and the packing and
# device round trip of cudaExecuteKicks, and a mark with the workitem and list
# counts. Same build configuration as the frozen binary (Release, sm_86,
# MAX_BUCKET 1024, order 7, STF benchmarks). Start/end ranges rather than
# push/pop because an HPX thread that waits on the workspace mutex or on
# futures may resume on another worker. Summarize with
#   python3 ../../../../benchmarks/summarize_kick_handoff_profile.py --nvtx . > summary.md
#   python3 ../../../../benchmarks/summarize_kick_handoff_profile.py . >> summary.md
# Lock, quiet gate and force-export deletion as in ../nsys_run.sh.
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired"
set -u
cd "$(dirname "$0")"
BIN=${BIN:?set BIN to the instrumented cosmictiger binary}
NSYS=${NSYS:-nsys}
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=128 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=128 --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=11 COSMICTIGER_STF_WALK_BUCKET=128 COSMICTIGER_STF_WALK_REFERENCE=0
echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
wait_quiet() {
    local quiet=0
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
}
: > status.txt
for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then order="h328 h656"; else order="h656 h328"; fi
    for config in lattice:8 adaptive:8 lattice:4; do
        fixture=${config%%:*}
        threads=${config##*:}
        for v in $order; do
            d=${fixture}128_t${threads}_${v}_$round
            mkdir -p $d
            wait_quiet
            echo "== $d $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg)"
            (cd $d && COSMICTIGER_STF_WALK_FIXTURE=$fixture OMP_NUM_THREADS=$threads OMP_THREAD_LIMIT=$threads \
                timeout 1800 $NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true -o kick \
                taskset -c 1-$threads $BIN $COMMON --hpx:threads=$threads --compute.cudaKickHandoffBlocks=${v#h} > run.log 2>&1
                echo "$d exit $? $(date '+%H:%M:%S')" >> ../status.txt)
            rm -f $d/stf_walk_forces.csv
            (cd $d && $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum,cuda_api_trace,nvtx_sum -f csv -o stats kick.nsys-rep > stats.log 2>&1)
        done
    done
done
echo DONE >> status.txt
