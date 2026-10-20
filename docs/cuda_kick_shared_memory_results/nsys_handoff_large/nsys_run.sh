#!/bin/bash
# Nsight Systems host/kernel split of the CPU/GPU handoff at 2,097,152
# particles (initial.particlesPerDimension 128): compute.cudaKickHandoffBlocks
# 328 (512 lattice workitems of 4,096 particles) against 656 (1,024 of
# 2,048), on the lattice, where 656 loses 10-22% in handoff_large_sizes/, and
# on the adaptive fixture, where it gains, as the contrast. Bucket 128, FMM
# order 7, 8 HPX worker threads pinned to CPUs 1-8, 11 samples, forces and
# potential, the frozen faaf984a binary of handoff_rule_check_large/ (its
# automatic divisor is overridden here). Three rounds with the variant order
# alternating; CPU sampling and context-switch tracing are off so the host
# side carries only the CUDA trace overhead. Besides the kernel summary and
# GPU trace the CUDA API trace is exported, so the host share can be split at
# the kick's first CUDA call (summarize_kick_handoff_profile.py).
#
# One GPU job at a time across the concurrent sessions of this workstation:
# the stage re-executes under the shared benchmark lock, and every process
# starts only after half a minute without ninja, ncu, nsys, another GPU
# compute process or a kick arena driver. Each process writes a 300 MB
# stf_walk_forces.csv; it is deleted once that process has exited. Run from
# this directory; summarize with
#   python3 ../../../benchmarks/summarize_kick_profiles.py nsys . > summary.md
#   python3 ../../../benchmarks/summarize_kick_handoff_profile.py . >> summary.md
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired"
set -u
cd "$(dirname "$0")"
BIN=${BIN:-/home/dmarce1/workspace/cosmictiger/.claude/worktrees/inspiring-booth-ae1206/build/arena_b1024/frozen/cosmictiger_b1024_handoff_large_rule}
NSYS=${NSYS:-nsys}
THREADS=8
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=128 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=$THREADS --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=128 --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=11 COSMICTIGER_STF_WALK_BUCKET=128 COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=$THREADS OMP_THREAD_LIMIT=$THREADS
echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
wait_quiet() {
    # A running cosmictiger is named main-thread#0 by HPX, so GPU users are
    # taken from nvidia-smi rather than by process name.
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
    for fixture in lattice adaptive; do
        for v in $order; do
            d=${fixture}128_${v}_$round
            mkdir -p $d
            wait_quiet
            echo "== $d $(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg)"
            (cd $d && COSMICTIGER_STF_WALK_FIXTURE=$fixture \
                timeout 1800 $NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true -o kick \
                taskset -c 1-$THREADS $BIN $COMMON --compute.cudaKickHandoffBlocks=${v#h} > run.log 2>&1
                echo "$d exit $? $(date '+%H:%M:%S')" >> ../status.txt)
            rm -f $d/stf_walk_forces.csv
            (cd $d && $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum,cuda_api_trace -f csv -o stats kick.nsys-rep > stats.log 2>&1)
        done
    done
done
echo DONE >> status.txt
