#!/usr/bin/env bash
# Sample ordinary functions under the HPX task names forwarded by APEX.
set -euo pipefail

if (($# == 0)); then
    echo "Usage: $0 EXECUTABLE [application arguments...]" >&2
    echo "Set COSMICTIGER_PROFILE_DIR to choose the report directory." >&2
    exit 2
fi

tau_exec_path="${TAU_EXEC:-}"
if [[ -z "$tau_exec_path" ]]; then
    tau_exec_path="$(command -v tau_exec || true)"
fi
if [[ -z "$tau_exec_path" ]]; then
    for candidate in "${TAU_ROOT:-$HOME/local/tau}/bin/tau_exec" \
        "${TAU_ROOT:-$HOME/local/tau}/x86_64/bin/tau_exec" \
        "${TAU_ROOT:-$HOME/local/tau}/arm64_linux/bin/tau_exec"; do
        if [[ -x "$candidate" ]]; then
            tau_exec_path="$candidate"
            break
        fi
    done
fi
if [[ -z "$tau_exec_path" || ! -x "$tau_exec_path" ]]; then
    echo "TAU is unavailable. Run scripts/build_tau.sh or set TAU_EXEC to tau_exec." >&2
    exit 1
fi

report_directory="${COSMICTIGER_PROFILE_DIR:-$PWD/profiles/tau-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$report_directory"
report_directory="$(cd "$report_directory" && pwd -P)"
export PROFILEDIR="${PROFILEDIR:-$report_directory/tau}"
export APEX_OUTPUT_FILE_PATH="${APEX_OUTPUT_FILE_PATH:-$report_directory/apex}"
mkdir -p "$PROFILEDIR" "$APEX_OUTPUT_FILE_PATH"

# These are intrinsic to this launcher, not optional reporting defaults.
export APEX_TAU=1
export TAU_SAMPLING=1
export TAU_PROFILE=1
export TAU_TRACE=0
# Keep the profiled application at the launcher's PID so MPI supervision and
# Linux parent-death cleanup apply to the application, not a waiting shell.
export COSMICTIGER_TAU_EXEC_REPLACE=1
# Preserve explicit report and sampling choices. No stack unwinding is needed
# to associate a sampled instruction with its active APEX task context.
export APEX_SCREEN_OUTPUT="${APEX_SCREEN_OUTPUT-0}"
export APEX_CSV_OUTPUT="${APEX_CSV_OUTPUT-1}"
export APEX_TASKTREE_OUTPUT="${APEX_TASKTREE_OUTPUT-1}"
export APEX_THROTTLE_TIMERS="${APEX_THROTTLE_TIMERS-0}"
export TAU_THROTTLE="${TAU_THROTTLE-0}"
export TAU_CALLPATH="${TAU_CALLPATH-1}"
export TAU_EBS_SOURCE="${TAU_EBS_SOURCE-itimer}"
export TAU_EBS_PERIOD="${TAU_EBS_PERIOD-10000}"
export TAU_EBS_RESOLUTION="${TAU_EBS_RESOLUTION-function}"
export TAU_EBS_UNWIND="${TAU_EBS_UNWIND-0}"
export TAU_EBS_KEEP_UNRESOLVED_ADDR="${TAU_EBS_KEEP_UNRESOLVED_ADDR-1}"

printf 'TAU task/function profiles: %s\nAPEX task reports: %s\n' \
    "$PROFILEDIR" "$APEX_OUTPUT_FILE_PATH"
exec "$tau_exec_path" -T mpi,pthread -ebs "$@"
