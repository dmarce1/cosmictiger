#!/bin/bash
# Nsight Systems host/kernel split at FMM order 7: baseline (585ee6c0) against
# the arena (auto policy and residency 6) on the adaptive fixture at bucket 64,
# plus the lattice at bucket 32 and the adaptive fixture at bucket 512 where
# the bucket sweep showed the largest arena losses. Three rounds with the
# variant order alternating. CPU sampling and context-switch tracing are off so
# the host-side timing carries only the CUDA/NVTX trace overhead.
set -u
cd "$(dirname "$0")"
SRC=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/gifted-euler-74225b
BASELINE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
ARENA=$SRC/build/arena_b1024/frozen/cosmictiger_b1024_231325d5
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true"
export COSMICTIGER_STF_WALK_SAMPLES=15 COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then order="baseline arena_auto arena_res6"; else order="arena_res6 arena_auto baseline"; fi
    for config in adaptive:64 lattice:32 adaptive:512; do
        fixture=${config%%:*}
        bucket=${config##*:}
        for v in $order; do
            case $v in
                baseline) BIN=$BASELINE; EXTRA="";;
                arena_auto) BIN=$ARENA; EXTRA="--gravity.fmmOrder=7";;
                arena_res6) BIN=$ARENA; EXTRA="--gravity.fmmOrder=7 --compute.cudaKickResidency=6";;
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
