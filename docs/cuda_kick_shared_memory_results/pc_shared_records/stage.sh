#!/bin/bash
# Stage driver: wait until the workstation has been quiet for 60 s (no
# ninja/compiler/profiler/sanitizer process, no GPU compute process, one-minute
# load below 6), then take the shared GPU lock and run, in one holding, the
# CUDA regressions (ctest fmm_walk_cuda, sph_gravity_cuda) and run.sh. The lock
# is never held while waiting for quiet. Run from the repository root; the
# transcript goes to driver.log in the stage directory.
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/pc_shared_records}
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
quiet_reason() {
    if pgrep -x 'ninja|cc1plus|cicc|ptxas|ncu|nsys|compute-sanitizer|nvcc|fatbinary|nvlink' > /dev/null; then
        echo build; return
    fi
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
        echo gpu; return
    fi
    if [ "$(awk '{print ($1 < 6.0)}' /proc/loadavg)" != 1 ]; then
        echo "load $(cut -d' ' -f1 /proc/loadavg)"; return
    fi
    echo ok
}
quiet_since=""
while true; do
    reason=$(quiet_reason)
    if [ "$reason" = ok ]; then
        if [ -z "$quiet_since" ]; then quiet_since=$(date +%s); fi
        if [ $(( $(date +%s) - quiet_since )) -ge 60 ]; then
            echo "$(date +%T) quiet for 60 s (load $(cut -d' ' -f1 /proc/loadavg)); taking the GPU lock"
            break
        fi
    else
        quiet_since=""
        echo "$(date +%T) not quiet: $reason"
    fi
    sleep 20
done
flock -w 14400 "$LOCK" bash -c "
    echo \"\$(date -Is) GPU lock acquired for ctest, load \$(cut -d' ' -f1 /proc/loadavg)\"
    ctest --test-dir build/release -R 'fmm_walk_cuda|sph_gravity_cuda' > '$OUTPUT/ctest_cuda.log' 2>&1
    echo \"\$(date -Is) ctest exit \$? (\$(grep -E 'tests passed' '$OUTPUT/ctest_cuda.log'))\"
    GPU_LOCK_HELD=1 bash '$OUTPUT/run.sh' '$OUTPUT'
"
echo "$(date -Is) stage done"
