#!/bin/bash
# Kernel-only metrics of the second force-only kick launch for binary $1 at
# bucket $2, fixture $3, label $4; extra options follow.
BIN=$1; BUCKET=$2; FIX=$3; LABEL=$4; shift 4
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=$BUCKET"
mkdir -p $LABEL && cd $LABEL
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=$BUCKET COSMICTIGER_STF_WALK_FIXTURE=$FIX COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
timeout 900 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,launch__grid_size,sm__warps_active.avg.pct_of_peak_sustained_active taskset -c 1,2 $BIN $COMMON "$@" > ncu.log 2>&1
echo "== $LABEL ncu exit $?"
grep -E "gpu__time_duration|inst_executed|warp_latency|short_scoreboard|no_instruction|branch_resolving|mem_local|grid_size|warps_active" ncu.log
