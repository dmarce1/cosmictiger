#!/bin/bash
# run_one.sh NAME BINARY [sanitizer args...]: racecheck the CUDA fmmWalk regression
# at order 7 with BINARY after a quiet GPU, under the shared GPU lock.
set -u
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-nifty-lederberg-7f7b6e/d666ad06-9a49-415b-a5ba-0124ef8a78e3/scratchpad/rc
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
STATUS=$S/status.txt
WALK="--run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=true --hpx:threads=2 --hpx:bind=none --gravity.fmmOrder=7"
SAN=/usr/local/cuda/bin/compute-sanitizer
name=$1; bin=$2; shift 2
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -ne 0 ] || pgrep -x nsys >/dev/null || pgrep -x ncu >/dev/null || pgrep -x compute-sanitizer >/dev/null || pgrep -f run_cuda_kick_arena.py >/dev/null; }
quiet=0; waited=0
while [ $quiet -lt 30 ]; do if busy; then quiet=0; else quiet=$((quiet+5)); fi; sleep 5; waited=$((waited+5)); done
mkdir -p $S/wd_$name && cd $S/wd_$name
flock -w 14400 $LOCK bash -c "timeout 1800 $SAN --tool racecheck $* $bin $WALK > $S/logs/$name.log 2>&1; echo \"\$(date -Is) $name: exit \$? \$(grep -E 'RACECHECK SUMMARY' $S/logs/$name.log) \$(grep -c 'Race reported' $S/logs/$name.log) analysis reports printed (quiet wait ${waited}s)\" >> $STATUS"
