#!/bin/bash
# Runs run.test=force CPU-only for both walk rules on a cosmological and a Plummer
# initial condition, 32^3 particles, and leaves a run.log per case.
set -u
BIN=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/cosmictiger
RECFAST=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/recfast
POWER=/home/dmarce1/workspace/cosmictiger/power.init
ROOT=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger/fdd3544f-0407-4144-8d36-8667f5e57461/scratchpad/runs_cost
for ic in "$@"; do
  for rule in false true; do
    dir=$ROOT/${ic}_open_${rule}
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$RECFAST" "$dir/recfast"; cp "$POWER" "$dir/power.init"
    extra=""
    if [ "$ic" = plummer ]; then extra="--initial.plummer=true"; fi
    ( cd "$dir" && "$BIN" --run.test=force --compute.cuda=false --hydro.enabled=false \
        --analysis.views=false --analysis.power=false \
        --analysis.groups=false --lightcone.enabled=false --initial.particlesPerDimension=32 \
        --gravity.openLargerCell=$rule $extra --hpx:threads=8 --hpx:bind=none > run.log 2>&1
      echo "EXIT $?" >> run.log )
    echo "$ic openLargerCell=$rule: $(tail -1 "$dir/run.log")"
  done
done
