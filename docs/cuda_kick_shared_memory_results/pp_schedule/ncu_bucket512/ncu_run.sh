#!/bin/bash
# Nsight Compute replay of the second force-only kick kernel launch on the
# adaptive fixture at bucket 512 for the arena head (old PP loop) and the
# arena with the restructured loop; the baseline's replay is the one in
# ../../ncu_bucket512/baseline (same binary, sha256 136f06e4...). Run from
# this directory.
cd "$(dirname "$0")"
SRC=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/unruffled-robinson-73f6f1
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=512 --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=512 COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for v in "$@"; do
    if [ $v = head ]; then BIN=$SRC/build/arena_b1024/frozen/cosmictiger_b1024_head_cf66eacc; else BIN=$SRC/build/arena_b1024/frozen/cosmictiger_b1024_v2; fi
    mkdir -p $v
    (cd $v && timeout 1800 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --set full -o kick -f $BIN $COMMON > ncu.log 2>&1; echo "$v exit $?" >> ../status.txt)
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page raw --csv > $v/raw.csv 2>/dev/null
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page details > $v/details.txt 2>/dev/null
    /usr/local/cuda/bin/ncu --import $v/kick.ncu-rep --page source --csv --print-source sass > $v/source.csv 2>/dev/null
done
echo DONE >> status.txt
