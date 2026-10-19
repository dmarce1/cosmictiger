#!/bin/bash
# Equal-accuracy comparison of the two walk rules on one tree: run.test=force
# with run.treeOpeningAngle=TREE_THETA builds the tree once and sweeps only the
# kick's opening angle from TREE_THETA to 0.95 in steps of 0.025. The walk is
# CPU only (compute.cuda=false); the analytic comparison uses the GPU for its
# direct reference sums. Usage: run_fixed_tree_sweep.sh cosmo [plummer]; writes
# ROOT/<ic>_tree<theta>_open_<rule>/run.log. BIN selects the binary.
set -u
THREADS=${THREADS:-8}
TREE_THETA=${TREE_THETA:-0.4}
BIN=${BIN:-/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/cosmictiger}
RECFAST=/home/dmarce1/workspace/cosmictiger/build/order_shared_arena/recfast
POWER=/home/dmarce1/workspace/cosmictiger/power.init
ROOT=${ROOT:-/tmp/cosmictiger_open_larger/runs_fixed_tree}
for ic in "$@"; do
  for rule in false true; do
    dir=$ROOT/${ic}_tree${TREE_THETA}_open_${rule}
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$RECFAST" "$dir/recfast"; cp "$POWER" "$dir/power.init"
    extra=""
    if [ "$ic" = plummer ]; then extra="--initial.plummer=true"; fi
    ( cd "$dir" && "$BIN" --run.test=force --run.treeOpeningAngle=$TREE_THETA \
        --compute.cuda=false --hydro.enabled=false --analysis.views=false \
        --analysis.power=false --analysis.groups=false --lightcone.enabled=false \
        --initial.particlesPerDimension=32 --gravity.openLargerCell=$rule $extra \
        --hpx:threads=$THREADS --hpx:bind=none > run.log 2>&1
      echo "EXIT $?" >> run.log )
    echo "$ic tree theta $TREE_THETA openLargerCell=$rule: $(tail -1 "$dir/run.log")"
  done
done
