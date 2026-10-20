#!/bin/bash
# All remaining racecheck runs in one lock session (about eight one-minute
# runs), so the other sessions' benchmark stages are interrupted once.
# Quiet first (bounded, the lock itself serializes GPU use), then the lock,
# then a short quiet check inside before the runs start.
set -u
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-nifty-lederberg-7f7b6e/d666ad06-9a49-415b-a5ba-0124ef8a78e3/scratchpad/rc
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
STATUS=$S/status.txt
WALK="--run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=true --hpx:threads=2 --hpx:bind=none --gravity.fmmOrder=7"
SAN=/usr/local/cuda/bin/compute-sanitizer
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -ne 0 ] || pgrep -x nsys >/dev/null || pgrep -x ncu >/dev/null || pgrep -x compute-sanitizer >/dev/null; }
quiet=0; waited=0
while [ $quiet -lt 30 ] && [ $waited -lt 300 ]; do if busy; then quiet=0; else quiet=$((quiet+5)); fi; sleep 5; waited=$((waited+5)); done
echo "$(date -Is) queue_all: quiet $quiet s after ${waited}s; waiting for lock" >> $STATUS
run() { local name=$1 bin=$2; shift 2; mkdir -p $S/wd_$name && cd $S/wd_$name; timeout 1800 $SAN --tool racecheck "$@" $bin $WALK > $S/logs/$name.log 2>&1; echo "$(date -Is) $name: exit $? $(grep -E 'RACECHECK SUMMARY' $S/logs/$name.log) $(grep -c 'Race reported' $S/logs/$name.log) reports" >> $STATUS; }
export -f run busy
export S STATUS WALK SAN
flock -w 14400 $LOCK bash -c '
  echo "$(date -Is) queue_all: lock taken" >> $STATUS
  q=0; w=0; while [ $q -lt 10 ] && [ $w -lt 60 ]; do if busy; then q=0; else q=$((q+5)); fi; sleep 5; w=$((w+5)); done
  run mb_handoff_all /home/dmarce1/workspace/cosmictiger/.claude/worktrees/modest-borg-b7dbb2/build/arena_b1024/frozen/cosmictiger_b1024_handoff --racecheck-report all --print-level info --print-limit 20000
  run ur_head_all /home/dmarce1/workspace/cosmictiger/.claude/worktrees/unruffled-robinson-73f6f1/build/arena_b1024/frozen/cosmictiger_b1024_head_cf66eacc --racecheck-report all --print-level info --print-limit 20000
  run scratch_base_all /home/dmarce1/workspace/cosmictiger/.claude/worktrees/nifty-lederberg-7f7b6e/build/racecheck/frozen/cosmictiger_base --racecheck-report all --print-level info --print-limit 20000
  for v in launder_copy launder_wait launder_both block_sync sync_copy; do run scratch_$v /home/dmarce1/workspace/cosmictiger/.claude/worktrees/nifty-lederberg-7f7b6e/build/racecheck/frozen/cosmictiger_$v --print-limit 0; done
  echo "$(date -Is) queue_all: lock released" >> $STATUS
'
echo "$(date -Is) QUEUE_ALL DONE" >> $STATUS
