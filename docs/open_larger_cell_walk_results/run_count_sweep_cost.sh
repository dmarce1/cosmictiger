#!/bin/bash
# GPU-free count sweep: run.test=kick at each opening angle for both walk rules,
# CPU only, THREADS HPX threads. Writes one log per case under runs_kick/.
set -u
THREADS=${THREADS:-5}
BIN=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/cosmictiger
RECFAST=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/recfast
POWER=/home/dmarce1/workspace/cosmictiger/power.init
ROOT=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger/fdd3544f-0407-4144-8d36-8667f5e57461/scratchpad/runs_kick_cost
THETAS="0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"
for ic in "$@"; do
  for rule in false true; do
    dir=$ROOT/${ic}_open_${rule}
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$RECFAST" "$dir/recfast"; cp "$POWER" "$dir/power.init"
    extra=""
    if [ "$ic" = plummer ]; then extra="--initial.plummer=true"; fi
    : > "$dir/sweep.log"
    for theta in $THETAS; do
      ( cd "$dir" && "$BIN" --run.test=kick --compute.cuda=false --hydro.enabled=false \
          --analysis.views=false --analysis.power=false --analysis.groups=false \
          --lightcone.enabled=false --initial.particlesPerDimension=32 \
          --gravity.openLargerCell=$rule --gravity.openingAngle=$theta $extra \
          --hpx:threads=$THREADS --hpx:bind=none > run_$theta.log 2>&1
        echo "theta $theta EXIT $?" >> sweep.log
        grep -h "^walk theta\|^tree_kick" run_$theta.log >> sweep.log )
    done
    echo "$ic openLargerCell=$rule: $(grep -c EXIT\ 0 "$dir/sweep.log") of 16 runs ok"
  done
done
