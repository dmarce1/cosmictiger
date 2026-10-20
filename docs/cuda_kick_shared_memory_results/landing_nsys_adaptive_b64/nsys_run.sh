#!/bin/bash
# Nsight Systems host/kernel split of the merged main (819d0013) against the
# pre-merge main (00909d26) on the adaptive fixture at bucket 64 (FMM order 7),
# the same frozen 1024-bucket binaries as landing_timing_order7_b64/: the
# automatic policy and, as "matched", both binaries pinned to the baseline's
# work division and opening rule (492-block handoff, gravity.openLargerCell=
# false) so the kernels do the same work, as in nsys_list_slab_matched/.
# Three rounds with the variant order alternating; CPU sampling and
# context-switch tracing are off; before each process the script waits for
# compilers, profilers and GPU compute processes. Run from this directory;
# summarize with
#   python3 ../../../benchmarks/summarize_kick_profiles.py nsys . > summary.md
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -u
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
PREMERGE=$ROOT/build/release/frozen/cosmictiger_b1024_premerge_00909d26
MERGED=$ROOT/build/release/frozen/cosmictiger_b1024_merged_819d0013
MATCHED="--compute.cudaKickHandoffBlocks=492 --gravity.openLargerCell=false"
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=15 COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
wait_idle() {
    while pgrep -x 'ninja|ncu|nsys|cc1plus|cicc|ptxas' > /dev/null \
        || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
        sleep 5
    done
}
: > status.txt
for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then
        order="premerge merged premerge_matched merged_matched"
    else
        order="merged_matched premerge_matched merged premerge"
    fi
    for v in $order; do
        case $v in
            premerge) BIN=$PREMERGE; EXTRA="";;
            merged) BIN=$MERGED; EXTRA="";;
            premerge_matched) BIN=$PREMERGE; EXTRA=$MATCHED;;
            merged_matched) BIN=$MERGED; EXTRA=$MATCHED;;
        esac
        d=adaptive64_${v}_$round
        mkdir -p $d
        wait_idle
        (cd $d && COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_BUCKET=64 \
            timeout 1800 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true -o kick \
            taskset -c 1,2 $BIN $COMMON --compute.bucketSize=64 $EXTRA > run.log 2>&1; echo "$d exit $? $(date -Is) load $(cut -d' ' -f1 /proc/loadavg)" >> ../status.txt)
        (cd $d && nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum -f csv -o stats kick.nsys-rep > stats.log 2>&1)
    done
done
echo DONE >> status.txt
