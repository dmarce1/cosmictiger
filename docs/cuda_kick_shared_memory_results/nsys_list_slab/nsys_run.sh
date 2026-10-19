#!/bin/bash
# Nsight Systems host/kernel split of the per-launch list slab against the
# pre-slab head and the baseline on the adaptive fixture at buckets 64 and 128
# (FMM order 7, automatic policy). Three rounds with the variant order
# alternating; CPU sampling and context-switch tracing are off. Run from this
# directory; summarize with
#   python3 ../../../benchmarks/summarize_kick_profiles.py nsys . > summary.md
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired"
set -u
cd "$(dirname "$0")"
ROOT=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/wizardly-leakey-12daaf
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
HEAD=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_eebab5b1
SLAB=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_slab
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true"
export COSMICTIGER_STF_WALK_SAMPLES=15 COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then order="baseline head slab"; else order="slab head baseline"; fi
    for config in adaptive:64 adaptive:128; do
        fixture=${config%%:*}
        bucket=${config##*:}
        for v in $order; do
            case $v in
                baseline) BIN=$BASELINE; EXTRA="";;
                head) BIN=$HEAD; EXTRA="--gravity.fmmOrder=7";;
                slab) BIN=$SLAB; EXTRA="--gravity.fmmOrder=7";;
            esac
            d=${fixture}${bucket}_${v}_$round
            mkdir -p $d
            (cd $d && COSMICTIGER_STF_WALK_FIXTURE=$fixture COSMICTIGER_STF_WALK_BUCKET=$bucket \
                timeout 1800 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true -o kick \
                taskset -c 1,2 $BIN $COMMON --compute.bucketSize=$bucket $EXTRA > run.log 2>&1; echo "$d exit $?" >> ../status.txt)
            (cd $d && nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum -f csv -o stats kick.nsys-rep > stats.log 2>&1)
        done
    done
done
echo DONE >> status.txt
