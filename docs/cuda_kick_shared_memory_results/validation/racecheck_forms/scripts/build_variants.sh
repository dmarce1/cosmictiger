#!/bin/bash
# Build each diagnostic staging variant in build/racecheck (6 jobs), freeze the
# binary, survey its operand forms, and restore the committed source.
set -u
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-nifty-lederberg-7f7b6e/d666ad06-9a49-415b-a5ba-0124ef8a78e3/scratchpad
W=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/nifty-lederberg-7f7b6e
B=$W/build/racecheck
LOG=$S/variants_status.txt
: # base already surveyed
echo "$(date -Is) base surveyed" >> $LOG
for v in "$@"; do
  while [ "$(pgrep -c -x ninja)" -ge 2 ]; do sleep 15; done
  python3 -I $S/apply_variant.py $v >> $LOG 2>&1
  ninja -C $B -j6 cosmictiger > $B/build_$v.log 2>&1; rc=$?
  echo "$(date -Is) $v: ninja exit $rc" >> $LOG
  if [ $rc -eq 0 ]; then
    cp $B/cosmictiger $B/frozen/cosmictiger_$v
    $S/survey_one.sh scratch_$v $B/frozen/cosmictiger_$v
    echo "$(date -Is) $v: frozen and surveyed" >> $LOG
  else
    grep -n "error" $B/build_$v.log | head -5 >> $LOG
  fi
  python3 -I $S/apply_variant.py base >> $LOG 2>&1
done
echo "$(date -Is) VARIANTS DONE" >> $LOG
