#!/bin/bash
# Kernel-only duration of the second force-only kick launch (adaptive fixture,
# bucket $2) for binary $1, plus the harness's own kick times with 5 samples.
# Usage: kernel_time.sh BINARY BUCKET LABEL [extra options]
BIN=$1; BUCKET=$2; LABEL=$3; shift 3
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --compute.bucketSize=$BUCKET"
mkdir -p $LABEL && cd $LABEL
export COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=$BUCKET COSMICTIGER_STF_WALK_FIXTURE=adaptive COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
timeout 900 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,launch__registers_per_thread,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum taskset -c 1,2 $BIN $COMMON "$@" > ncu.log 2>&1
echo "ncu exit $?"
grep -E "gpu__time_duration|inst_executed|warp_latency|short_scoreboard|long_scoreboard|stalled_wait|registers_per_thread|mem_local|mem_shared" ncu.log
export COSMICTIGER_STF_WALK_SAMPLES=5
taskset -c 1,2 $BIN $COMMON "$@" > harness.log 2>&1
echo "harness exit $?"
grep -E "^STF_WALK_MEDIAN|^CUDA kick arena" harness.log
