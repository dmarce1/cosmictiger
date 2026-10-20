#!/bin/bash
# racecheck of the CUDA fmmWalk regression at order 7 with the aligned binary,
# after 60 s of quiet, under the shared GPU lock (one run, print limit 50).
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-festive-easley-59df3c/494d385c-f529-4643-8d8d-d3d837c7695a/scratchpad
REPO=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/festive-easley-59df3c
OUT=$REPO/docs/cuda_kick_shared_memory_results/pc_staging_cp_async
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
WALK="--run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=true --hpx:threads=2 --hpx:bind=none --gravity.fmmOrder=7"
SAN=/usr/local/cuda/bin/compute-sanitizer
mkdir -p $S/rc_wd && cd $S/rc_wd
for bin in aligned head; do
    if [ $bin = aligned ]; then B=$REPO/build/release/frozen/cosmictiger_b1024_pc_aligned; else B=$REPO/build/release/frozen/cosmictiger_b1024_head_819d0013; fi
    flock -w 14400 $LOCK bash -c "echo \"\$(date -Is) racecheck $bin start\"; timeout 1800 $SAN --tool racecheck --print-limit 50 $B $WALK > $OUT/racecheck_$bin.log 2>&1; echo \"\$(date -Is) racecheck $bin exit \$? \$(grep -E 'RACECHECK SUMMARY' $OUT/racecheck_$bin.log)\""
done
