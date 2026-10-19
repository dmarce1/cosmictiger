#!/bin/bash
# Nsight Compute replay of the second force-only kick kernel launch on the
# adaptive fixture at bucket 64: the pre-slab head against the per-launch list
# slab (FMM order 7, automatic policy), to attribute the warp instructions of
# the pool allocator. Run from this directory.
# One GPU job at a time across the concurrent sessions of this workstation:
# re-execute under the shared benchmark lock (the sessions' agreed convention).
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired"
cd "$(dirname "$0")"
ROOT=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/wizardly-leakey-12daaf
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=64 --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=64 COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for v in head slab; do
    BIN=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_$([ $v = head ] && echo eebab5b1 || echo slab)
    mkdir -p $v
    (cd $v && timeout 1800 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --set full -o kick -f $BIN $COMMON > ncu.log 2>&1; echo "$v exit $?" >> ../status.txt)
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page raw --csv > $v/raw.csv 2>/dev/null
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page details > $v/details.txt 2>/dev/null
done
echo DONE >> status.txt
