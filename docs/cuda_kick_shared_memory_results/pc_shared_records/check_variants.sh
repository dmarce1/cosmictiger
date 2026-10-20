#!/bin/bash
# Force-export comparison of the record-read variants against the head, one
# process per binary and bucket (one pair, one sample, no direct reference):
# correctness only, not timing. Under the shared GPU lock, taken after the
# machine has been quiet (stage.sh's rule); the driver still waits for
# compilers, profilers and GPU compute processes before each run. Run from the
# repository root. Variants: registers = the record copied to registers with
# ten LDS.128 (rejected: the compiler re-pairs the M2P operator's fused
# multiply-adds), scalar = the record viewed in place through the shared
# pointer (LDS per word), aligned = the same view at the slot's 16-byte
# alignment.
LOCK=/home/dmarce1/.config/Claude/side-session-notes/local_1aab8699-560a-48fb-895b-1e8a0aa36de6/gpu_benchmark.lock
if [ "${GPU_LOCK_HELD:-}" != 1 ]; then
    GPU_LOCK_HELD=1 exec flock -w 14400 "$LOCK" bash "$0" "$@"
fi
echo "$(date -Is) GPU lock acquired, load $(cut -d' ' -f1 /proc/loadavg)"
set -u
OUTPUT=${1:-docs/cuda_kick_shared_memory_results/pc_shared_records/variants}
python3 benchmarks/run_cuda_kick_arena.py \
    --binary head=build/release/frozen/cosmictiger_b1024_head_da3c1f4e:arena \
    --binary registers=build/release/frozen/cosmictiger_b1024_pc_registers:arena \
    --binary scalar=build/release/frozen/cosmictiger_b1024_pc_shared_a:arena \
    --binary aligned=build/release/frozen/cosmictiger_b1024_pc_shared_c:arena \
    --policy auto= \
    --output "$OUTPUT" \
    --fixtures lattice,clustered,adaptive --buckets 64,512 --orders 7 \
    --pairs 1 --samples 1 --reference 0 --timeout 1800 \
    --idle-wait 'ninja|ncu|nsys|cc1plus|cicc|ptxas' --wait-gpu-idle
for v in registers scalar aligned; do
    python3 benchmarks/compare_kick_arena_binaries.py "$OUTPUT" head $v \
        --output "$OUTPUT/head_vs_$v.md"
done
echo "$(date -Is) variant check done"
