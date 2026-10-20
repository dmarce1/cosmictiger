#!/bin/bash
# Kernel-only metrics of the second force-only kick launch at the small
# buckets (head against the Ewald schedule binary), to check that the
# fully occupied regime does not regress. Run from this directory under the
# shared GPU lock on a quiet machine. Output: kernel_small_buckets/<label>/ncu.log
# and head_vs_ewald.txt.
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
HEAD=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_head_da3c1f4e
EWALD=$ROOT/build/arena_b1024/frozen/cosmictiger_b1024_ewald
COMMON="--run.test=stfBenchmark --initial.particlesPerDimension=32 --hydro.enabled=false --gravity.openingAngle=0.8 --gravity.effectiveRadius=0.01876988456953534 --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 --hpx:threads=2 --hpx:bind=none --analysis.views=false --analysis.power=false --analysis.groups=false --lightcone.enabled=false --compute.cuda=true --gravity.fmmOrder=7"
METRICS=gpu__time_duration.sum,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,launch__registers_per_thread,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,launch__grid_size
mkdir -p kernel_small_buckets
: > kernel_small_buckets/head_vs_ewald.txt
for cell in clustered:32 clustered:64 adaptive:64 adaptive:128 adaptive:256; do
    FIX=${cell%:*}; BUCKET=${cell#*:}
    for v in head ewald; do
        if [ $v = head ]; then BIN=$HEAD; else BIN=$EWALD; fi
        LABEL=kernel_small_buckets/${v}_${FIX}_b$BUCKET
        mkdir -p $LABEL
        (cd $LABEL && COSMICTIGER_STF_WALK_SAMPLES=1 COSMICTIGER_STF_WALK_BUCKET=$BUCKET COSMICTIGER_STF_WALK_FIXTURE=$FIX COSMICTIGER_STF_WALK_REFERENCE=0 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 timeout 900 /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 1 --kernel-name regex:cudaKickKernel --metrics $METRICS taskset -c 1,2 $BIN $COMMON --compute.bucketSize=$BUCKET > ncu.log 2>&1; echo "== $LABEL ncu exit $?" >> ../head_vs_ewald.txt; grep -E "gpu__time_duration|inst_executed|warp_latency|short_scoreboard|no_instruction|stalled_wait|registers_per_thread|mem_local|grid_size" ncu.log >> ../head_vs_ewald.txt)
    done
done
echo DONE >> kernel_small_buckets/head_vs_ewald.txt
