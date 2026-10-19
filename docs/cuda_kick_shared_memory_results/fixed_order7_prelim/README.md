# Preliminary fixed-order (order 7) arena measurements, 2026-10-19

These runs compare the first shared-memory arena build (commit dc70579e, still a
compile-time order-7 build) with the unchanged baseline (585ee6c0) on the
adaptive 32768-particle fixture, bucket 64, theta 0.8, CUDA backend, GPU shared
with the desktop. They predate the runtime-order conversion and the caching fix
to `kickBlockCount()`; they are kept because they identified two costs.

- `policy_sweep_status.txt`: 7-sample whole-kick medians (seconds, force-only
  then with potential) for the baseline and eight arena variants (residency 6,
  7, 8; shared or global active expansion; shared or global forces; particle
  tiles 256/512/768). Every arena variant measured about 21 ms against about 19
  ms for the baseline, independent of the policy, which pointed at a cost common
  to all variants. The first arena build's `kickBlockCount()` re-queried the
  device attribute on every CPU-walk node; the baseline cached it. The
  runtime-order dispatcher caches the value per order.
- `ncu_*`: Nsight Compute replay of the force-only adaptive kick kernel
  (`--set full`, second launch). Kernel duration 5.25 ms (baseline) versus 5.55
  ms (arena); executed warp instructions 238.7 M versus 255.6 M. The arena has 0
  static shared bytes (baseline 16), 7,888 dynamic shared bytes (baseline
  15,184), 8 resident blocks per SM (baseline 6), identical 255 registers and
  essentially identical register-spill instruction counts (7.32 M versus 7.27 M).
- `ncu_function_attribution.md`: executed warp instructions per device function.
  PP, PC, CP, direct and Ewald CC and the walk itself are within 1-6% of the
  baseline; the difference is the pool allocator (`cudaAllocate` 42.0 M to 64.3 M,
  `cudaRelease` 9.4 M to 14.9 M). The pool's lock-free queues are scanned with
  atomics, so per-block construction/teardown cost scales with the number of
  concurrently allocating blocks: 656 blocks (8 per SM) against 492 (6 per SM).
  This motivated the `compute.cudaKickListReserve` option and keeping the
  residency a measured choice.
- `sanitizer_*`: compute-sanitizer on the CUDA fmmWalk regression. memcheck
  clean; racecheck 0 errors / 0 warnings after the control-block construction
  barrier fix (the first arena build reported 408 warnings, the baseline 132,
  all from lane-redundant constructor writes); synccheck reports the same 32
  `cuda::barrier` "missing init/wait" diagnostics for the baseline and the arena.
