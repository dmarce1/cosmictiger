#!/bin/bash
# Run pass B under the GPU benchmark lock the sibling sessions share (flock
# on gpu_benchmark.lock in the shared notes folder), which serializes their
# GPU benchmarks and profiles. Builds and the desktop run outside the lock,
# so disturbed runs are detected afterwards instead: a run whose lattice tree
# build takes more than 3.5 ms (2 ms when the machine is quiet) or whose
# process ran longer than 4 s is contaminated (benchmarks/check_kick_arena_pairs.py
# in the handoff session's worktree flags pair spreads the same way). Earlier
# versions of this script also waited for a quiet machine, but their process
# check counted the sibling sessions' waiting flock wrappers, whose command
# lines name cosmictiger, and never started. Run from anywhere.
cd "$(dirname "$0")/../../.."
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
echo "waiting for the GPU lock at $(date +%T)"
exec flock -w 7200 "$LOCK" bash -c 'echo "lock taken at $(date +%T), load $(cut -d" " -f1 /proc/loadavg)"; bash docs/cuda_kick_shared_memory_results/pp_schedule/run_pass_b.sh'
