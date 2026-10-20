#!/bin/bash
# Nsight Compute replay (--set full) of the second force-only kick kernel on
# the adaptive fixture at bucket 512 for the head (origin/main da3c1f4e) and
# the Ewald schedule binary. Writes <variant>/kick.ncu-rep plus the raw,
# details and SASS-with-source exports (source.csv is not committed; the
# per-region summary from benchmarks/summarize_ncu_regions.py is). Run from
# this directory under the shared GPU lock with nothing else on the machine:
#   flock LOCK bash ncu_run.sh head ewald
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=${BUCKET:-512} --gravity.fmmOrder=7"
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=${BUCKET:-512} COSMICTIGER_STF_WALK_FIXTURE=${FIXTURE:-adaptive} COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
: > status.txt
for v in "$@"; do
    case $v in
        head) BIN=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_head_da3c1f4e ;;
        *) BIN=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_$v ;;
    esac
    mkdir -p ncu_bucket${BUCKET:-512}/$v
    (cd ncu_bucket${BUCKET:-512}/$v && timeout 1800 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --set full -o kick -f taskset -c 1,2 $BIN $COMMON > ncu.log 2>&1; echo "$v exit $?" >> ../../status.txt)
    D=ncu_bucket${BUCKET:-512}/$v
    /usr/local/cuda/bin/ncu --import $D/kick.ncu-rep --page raw --csv > $D/raw.csv 2>/dev/null
    /usr/local/cuda/bin/ncu --import $D/kick.ncu-rep --page details > $D/details.txt 2>/dev/null
    /usr/local/cuda/bin/ncu --import $D/kick.ncu-rep --page source --csv --print-source sass,cuda > $D/source.csv 2>/dev/null
    /usr/local/cuda/bin/ncu --import $D/kick.ncu-rep --page source --csv --print-source sass > $D/source_sass.csv 2>/dev/null; python3 $ROOT/benchmarks/summarize_ncu_sections.py $D/source_sass.csv > $D/sections.txt 2>/dev/null; rm -f $D/source_sass.csv
done
echo DONE >> status.txt
