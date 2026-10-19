# STF benchmark results — 2026-10-18

CPU complete walks improved by **1.08–1.36x** in these fixtures. CUDA complete
walks show **no consistent overall speedup**: aggregate tree-plus-kick ratios
range from 0.964x to 1.011x. The clustered force-only CUDA case is about 3.7%
slower overall (6.9% slower in the kick). The generated cooperative CUDA L2L
itself is **3.74–4.38x faster** than the legacy warp implementation.

These are measured workload results, not estimates from arithmetic counts.
Ratios are old time / STF time; values below one mean STF is slower.

## Configuration and provenance

- AMD Ryzen 9 3900X; NVIDIA GeForce RTX 3090 (`sm_86`); GCC 13.3; CUDA 12.8.
- Baseline commit: `91fccebc1a36eb9ae001aa617c53329c7ae74b3c`, including the existing
  standard-library SIMD implementation. Imported generator source:
  `dmarce1/experimental` commit `7ba8f1829fdede7348a1418b7cb36c0e7a955302`.
- Both application builds use matching Release flags: host `-O3 -ffast-math
  -march=native`, CUDA `-O3 --use_fast_math`. Identical benchmark harness and test
  dispatch were injected into the otherwise unchanged baseline snapshot.
- Source degree 5, local degree 6, 32,768 particles, opening angle 0.8, bucket
  size 64, support radius 0.0008798383391969691, one locality, two HPX workers,
  `--hpx:bind=none`. CPU and CUDA runs select the backend in a CUDA-enabled binary.
- All timing jobs ran serially with our builds/tests idle. GPU clocks were not
  locked; CPU microbenchmarks were unpinned. Small differences require caution.

Exact commands and source/executable hashes are in
[metadata.json](stf_benchmark_results/metadata.json),
[build_manifest.json](stf_benchmark_results/build_manifest.json), and
[walk_runs.json](stf_benchmark_results/walk_runs.json).

## Complete periodic walks

The lattice is deterministically jittered; the clustered fixture contains eight
compact groups. Positions, velocities, forces, and rungs are reset each time.
Timing includes the root background calculation, Ewald and direct interactions,
and the production CUDA L2L path. Input fingerprints match across variants.

Each process performs one warmup and 15 measured iterations per fixture/potential
mode. CPU entries are medians of those 15 iterations. CUDA was repeated in three
paired trials, alternating old/STF order; table entries are medians of the three
process medians. Raw per-iteration data are retained under
[stf_benchmark_results](stf_benchmark_results/); reset, output, and cleanup are
outside tree-plus-kick timing. Medians of totals need not equal sums of medians.

| Backend | Fixture | Potential | Old kick ms | STF kick ms | Old tree+kick ms | STF tree+kick ms | Total speedup |
|---|---|---|---:|---:|---:|---:|---:|
| CPU | lattice | off | 118.355 | 108.754 | 120.406 | 111.098 | 1.084x |
| CPU | lattice | on | 122.938 | 109.931 | 124.898 | 111.821 | 1.117x |
| CPU | clustered | off | 25.312 | 18.997 | 28.133 | 21.749 | 1.294x |
| CPU | clustered | on | 28.782 | 20.429 | 31.455 | 23.105 | 1.361x |
| CUDA | lattice | off | 33.641 | 33.206 | 35.945 | 35.538 | 1.011x |
| CUDA | lattice | on | 32.299 | 33.052 | 34.631 | 35.283 | 0.982x |
| CUDA | clustered | off | 7.101 | 7.590 | 10.293 | 10.674 | 0.964x |
| CUDA | clustered | on | 7.439 | 7.328 | 10.445 | 10.423 | 1.002x |

CUDA timings varied by several percent between paired trials. The force-only
clustered kick was slower in all three pairs; the other small changes do not
support a general CUDA speedup claim. The CPU gain is much larger than these
small CUDA differences.

## Cooperative CUDA L2L

This benchmark uses one expansion per 32-thread block and the production
source-read, lane-zero write, warp synchronization, and coalesced result-store
pattern. It uses 32 blocks, 32,768 expansions, 20 sweeps, and seven repetitions;
CUDA events measure execution, excluding host transfers and fixture setup.

| Implementation | Potential off, ns | Potential on, ns |
|---|---:|---:|
| Legacy cooperative warp | 345.99 | 364.01 |
| Interim STF lane zero | 67.82 | 63.68 |
| Final generated cooperative warp | 92.49 | 83.04 |

The final code retains the requested full-warp execution. It takes 1.30–1.36x
as long as the interim lane-zero implementation in this isolated measurement.
A first cooperative schedule took 210.63/208.53 ns; replacing dynamically indexed
basis arrays with lane-owned registers and shuffle lookups removed that overhead.
The final focused kernel uses 94 registers and zero stack bytes, with 86 shuffles
per lane versus the legacy kernel's 250. These register counts are for the
standalone benchmark, not the entire production kick kernel.

The legacy CUDA kernel fails the new repeated-translation reference check:
its table has 616 nonpotential terms but its loop processes only 615. The omitted
term is `Lzzzzzz * dz^4 / 24` in `Lzz`, causing trace drift. The baseline remains
unchanged for timing; its failed reference result is explicitly logged.

The final warp passes every-coefficient checks in every lane, both potential
modes, nonzero trace, linear/root-quadratic fields, 1 and 17 in-place translations,
and three independent warps per block. Maximum normalized Cartesian-reference
error was `5.08e-7` (normalization `max(1, abs(reference))`). Compute Sanitizer
`synccheck` reports zero errors. Exact commands, checks, and timings:
[l2l_warp_report.json](stf_benchmark_results/l2l_warp_report.json),
[l2l_commands.sh](stf_benchmark_results/l2l_commands.sh).

## Individual operators

The same harness compares native old/STF representations, preserving nonzero
dipoles and including the old Green-tensor generation in direct CC/PC timings.
CPU cases use 128 inputs, 1,000 sweeps, and seven repetitions; CUDA cases use
32,768 inputs, 100 sweeps, seven repetitions, and 32 one-warp blocks. Results
include loads/stores and consume every output. These limited-concurrency CUDA
results are not full-device throughput estimates. CUDA L2L in this table is
per-thread scalar L2L, separate from the cooperative measurement above.

| Operator | Potential | CPU scalar | CPU SIMD8 | CUDA per-thread |
|---|---|---:|---:|---:|
| p2m | off | 0.99x | 0.73x | 1.19x |
| m2m | off | 1.50x | 0.89x | 0.86x |
| cc_direct | off | 2.75x | 1.99x | 0.89x |
| cc_direct | on | 3.03x | 2.21x | 0.92x |
| pc_direct | off | 4.56x | 3.98x | 0.89x |
| pc_direct | on | 5.72x | 4.91x | 0.84x |
| l2l | off | 1.59x | 1.39x | 1.34x |
| l2l | on | 1.89x | 1.65x | 1.25x |
| l2p | off | 2.41x | 2.50x | 0.83x |
| l2p | on | 3.49x | 3.55x | 0.93x |

The SIMD direct interactions and evaluations improve substantially, while SIMD
P2M/M2M and several CUDA scalar operators regress. Whole-walk timing determines
the net result; operation-count savings alone do not predict it.

## Numerical comparison and validation

All eight final application exports contain 65,536 unique fixture/position rows,
with finite force and potential values. Rows are matched by fixed-point position,
not tree ordering. Comparing STF with the CPU baseline gives relative force L2
differences of `1.65e-5` (lattice) and `1.96e-7` (clustered). STF CUDA versus STF
CPU gives `9.13e-6` and `2.13e-7`, respectively; relative potential L2 differences
are `3.45e-5` and `2.98e-7`. These are agreement measures, not absolute error bounds.

The old/new CUDA force difference is larger (`1.18e-2` on the lattice), so this
is not a bitwise-compatible refactor. The legacy CUDA L2L arithmetic defect above
is independently demonstrated; both new CPU/CUDA paths pass the independent
Cartesian and periodic Ewald reference tests. Absolute potential differences and
full comparison statistics are in
[force_comparison.json](stf_benchmark_results/force_comparison.json).

All **15 final CTests passed**, covering scalar/SIMD kernels, generated-file
consistency, the CUDA-disabled Ewald initializer, device kernels and cooperative
L2L, CPU/CUDA gravity and periodic walks, three-rank MPI trees/walks, and DM/SPH
smoke runs. See [validation.log](stf_benchmark_results/validation.log).
The full application with `COSMICTIGER_WITH_CUDA=OFF` still has unrelated,
pre-existing unconditional CUDA calls; CPU runtime tests used the CPU backend
of the CUDA-enabled application.

## Reproduction

See [benchmark instructions](../benchmarks/stf_benchmarks.md). To prepare the
baseline, extract the recorded commit into a separate directory, copy the current
`benchmarks/stf_walk.cpp` into its `benchmarks/` directory, and apply
[baseline_benchmark.patch](stf_benchmark_results/baseline_benchmark.patch).
Configure both builds with `COSMICTIGER_BUILD_STF_BENCHMARKS=ON` and the matching
settings in `metadata.json`. Set `COSMICTIGER_STF_WALK_SAMPLES=15` for the recorded
walk sample count, and use each exact argument array in `walk_runs.json`.
Recover the microbenchmark baseline header with:

```sh
git show 91fccebc1a36eb9ae001aa617c53329c7ae74b3c:cosmictiger/fmm/fmm_kernels.hpp > /tmp/fmm_kernels_before.hpp
```

Raw per-particle exports remain in the recorded `/tmp`
run directories; compact timing, hash, comparison, and validation evidence is
saved alongside this report.
