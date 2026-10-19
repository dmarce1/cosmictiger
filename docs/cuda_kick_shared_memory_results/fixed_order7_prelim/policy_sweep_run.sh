#!/bin/bash
cd "$(dirname "$0")"
ARENA=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/cosmictiger
BASE=/home/dmarce1/workspace/cosmictiger/build/baseline_main/cosmictiger
WALK="--run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=true --hpx:threads=2 --hpx:bind=none"
echo "=== sanitizers $(date)" >> status.txt
for tool in synccheck racecheck; do
  timeout 1800 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 1 $ARENA $WALK > $tool.log 2>&1
  echo "$tool exit $? : $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $tool.log | tail -1)" >> status.txt
done
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=64"
export COSMICTIGER_STF_WALK_SAMPLES=7 COSMICTIGER_STF_WALK_BUCKET=64 COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
run() { name=$1; shift; mkdir -p $name; (cd $name && timeout 900 taskset -c 1,2 "$@" > out.log 2>&1); echo "$name exit $? : $(grep -E 'STF_WALK_MEDIAN,adaptive,cuda,0' $name/out.log | cut -d, -f9) $(grep -E 'STF_WALK_MEDIAN,adaptive,cuda,1' $name/out.log | cut -d, -f9) | $(grep -E 'CUDA kick arena|Occupancy is' $name/out.log | head -1)" >> status.txt; }
for round in 1 2; do
  run baseline_$round $BASE $COMMON
  run auto_$round $ARENA $COMMON
  run r6_$round $ARENA $COMMON --compute.cudaKickResidency=6
  run r7_$round $ARENA $COMMON --compute.cudaKickResidency=7
  run globalexp_$round $ARENA $COMMON --compute.cudaKickSharedExpansion=false
  run globalforce_$round $ARENA $COMMON --compute.cudaKickSinkTile=-1
  run r6_globalexp_globalforce_$round $ARENA $COMMON --compute.cudaKickResidency=6 --compute.cudaKickSharedExpansion=false --compute.cudaKickSinkTile=-1
  run p256_$round $ARENA $COMMON --compute.cudaKickParticleTile=256
  run p768_r6_$round $ARENA $COMMON --compute.cudaKickParticleTile=768 --compute.cudaKickResidency=6
done
echo DONE >> status.txt
