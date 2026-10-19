#!/bin/bash
# Nsight Compute replay of the second force-only kick kernel launch on the
# adaptive fixture at bucket 512 (64 leaves, one block each): baseline against
# the arena's automatic policy. Run from this directory.
cd "$(dirname "$0")"
SRC=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/gifted-euler-74225b
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=512"
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=512 COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for v in baseline arena; do
    if [ $v = baseline ]; then BIN=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger; EXTRA=""; else BIN=$SRC/build/arena_b1024/frozen/cosmictiger_b1024_fix2; EXTRA="--gravity.fmmOrder=7"; fi
    mkdir -p $v
    (cd $v && timeout 1800 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --set full -o kick -f $BIN $COMMON $EXTRA > ncu.log 2>&1; echo "$v exit $?" >> ../status.txt)
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page raw --csv > $v/raw.csv 2>/dev/null
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page details > $v/details.txt 2>/dev/null
done
echo DONE >> status.txt
