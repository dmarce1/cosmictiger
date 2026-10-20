#!/bin/bash
# wait_quiet_then.sh COMMAND...: wait until the workstation has been quiet for
# 60 s (no ninja/compiler/profiler/sanitizer process, no GPU compute process,
# one-minute load below 6), then run COMMAND, which takes the shared GPU lock
# itself. The lock is never held while waiting.
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
            echo "$(date +%T) quiet for 60 s (load $(cut -d' ' -f1 /proc/loadavg)); running: $*"
            break
        fi
    else
        quiet_since=""
        echo "$(date +%T) not quiet: $reason"
    fi
    sleep 20
done
exec "$@"
