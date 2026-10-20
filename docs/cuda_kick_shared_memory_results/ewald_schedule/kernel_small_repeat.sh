#!/bin/bash
# Three alternating kernel-only replays of the clustered fixture at buckets 32
# and 64 (head, ewald), to read the small-bucket kernel time against its
# run-to-run spread. Run from this directory under the shared GPU lock.
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
HEAD=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_head_da3c1f4e
EWALD=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_ewald
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --gravity.fmmOrder=7"
METRICS=gpu__time_duration.sum,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio,launch__grid_size
mkdir -p kernel_small_repeat
: > kernel_small_repeat/summary.txt
for rep in 1 2 3; do
    for cell in clustered:32 clustered:64; do
        FIX=${cell%:*}; BUCKET=${cell#*:}
        for v in head ewald; do
            if [ $v = head ]; then BIN=$HEAD; else BIN=$EWALD; fi
            LABEL=kernel_small_repeat/${v}_${FIX}_b${BUCKET}_r$rep
            mkdir -p $LABEL
            (cd $LABEL && COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=$BUCKET COSMICTIGER_STF_WALK_FIXTURE=$FIX COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 timeout 900 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --metrics $METRICS taskset -c 1,2 $BIN $COMMON --compute.bucketSize=$BUCKET > ncu.log 2>&1; echo "$LABEL $(grep -E 'gpu__time_duration|inst_executed|warp_latency|grid_size' ncu.log | awk '{printf "%s=%s ", $1, $NF}')" >> ../summary.txt)
        done
    done
done
echo DONE >> kernel_small_repeat/summary.txt
