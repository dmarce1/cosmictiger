# STF performance comparisons

Run comparisons after the correctness checks pass, with other compilation and
benchmark jobs idle. Microbenchmarks measure particular operators. The complete
walk includes tree creation, acceptance decisions, direct and periodic
interactions, and the production CUDA work distribution.

## Microbenchmarks

Preserve the legacy `cosmictiger/fmm_kernels.hpp` before replacing it. The harness
includes that external file through `COSMICTIGER_STF_BASELINE_HEADER`; it uses the
current common math, SIMD, tensor, and container headers for both implementations.
This compares kernel implementations with the same supporting infrastructure,
rather than two independently changed SIMD implementations.

The runner prints compiler commands when neither action flag is supplied:

```sh
python3 benchmarks/run_stf.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/stf_microbench
```

Compile all six binaries serially:

```sh
python3 benchmarks/run_stf.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/stf_microbench --build
```

The defaults are GCC 13, C++20, `-O3 -ffast-math -march=native`, CUDA `sm_86`, and
`-O3 --use_fast_math`. Both CUDA variants use `-rdc=true`, required by the legacy
header's inline constant-memory arrays. CUDA headers are also needed for the
host compiler because the legacy header declares CUDA storage. Override
`--cxx`, `--cuda-root`, and `--arch` together for a different toolchain.
`--skip-cuda` omits the CUDA binaries; it does not remove the CUDA-header
requirement for this legacy source. The STF variants include the generated
kernels from a configured build tree (`ninja -C build
cosmictiger_generated_kernels`), so the compile commands need
`-I<build>/generated` in addition to the source root; they time the build's
default FMM order (`ORDER`, 7), the only order the legacy header implements.

Use a fresh output directory. The build manifest records compiler versions,
exact argument arrays, source/header hashes, and executable hashes. Builds do
not start timing runs. To execute the prepared binaries, select an allowed CPU
that is not occupied by another workload:

```sh
python3 benchmarks/run_stf.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/stf_microbench \
  --run --cpu 2
```

The runner validates the recorded sources and binaries, runs one process at a
time, pins CPU runs to one logical CPU, and alternates legacy/STF ordering across
three pairs. Each process reports the median of seven measurements after two
warmup sweeps. Results, commands, stderr, and paired ratios are saved beneath
`measurements/`. `median_stf_over_legacy < 1` indicates a shorter STF execution
time for that case. Existing measurements are never overwritten.

CUDA runs are limited to 32 blocks of 32 threads, with a grid-stride loop. This
is below half of the 82 SMs on the RTX 3090 workstation. It measures deliberately
limited concurrency; it is not an estimate of unrestricted full-GPU throughput.

The timed operations are:

| Case | Legacy | STF |
|---|---|---|
| P2M | Point-to-moment projection | Generated point-to-STF projection |
| M2M | Legacy moment translation | Generated STF moment translation |
| Direct CC | Green derivatives plus full M2L | Direct partial-contraction M2L |
| Direct PC | Green derivatives plus potential/gradient M2L | Direct M2P |
| L2L | Legacy local translation | Generated local translation |
| L2P | Legacy local evaluation | Generated local evaluation |

CC, PC, L2L, and L2P each run with potential enabled and disabled. Setup creates
the same physical point source and geometry independently in each representation.
Legacy odd moment ranks have the opposite sign; legacy P2M/M2M also modify their
position argument. The harness accounts for both differences. The direct kernels
retain nonzero dipoles in these workloads.

All output coefficients are stored and consumed. CPU compiler barriers prevent
removing repeated sweeps. CUDA timing uses events around kernel launches and
includes device loads/stores; allocation, input upload, output download, fixture
construction, checksums, and CSV output are outside timing. Native source
storage is 37 legacy coefficients versus 36 STF coefficients, so the associated
storage traffic is part of the measured implementation difference.

Raw moment and local coefficient checksums must not be compared across layouts.
PC/L2P checksums use the common potential/x-gradient/y-gradient/z-gradient order,
but checksums are still only a consumption/sanity check, not a numerical error
norm. Correctness comes from the independent kernel tests and position-matched
complete-walk force output. Compare legacy versus STF within the same type and
backend: scalar and SIMD cases have different lane counts and working-set sizes.
The scalar and SIMD source-offset sequences also differ by their lane width.

CUDA L2L microbenchmarks execute one independent expansion per thread. They do
not measure the legacy cooperative warp L2L routine. The complete-walk comparison
is required to assess the replacement's effect on that production path.

`benchmarks/stf_l2l_cuda.cu` separately measures the production arrangement:
one expansion per 32-thread block, including the caller's in-place update and
synchronization. Compile it with the same CUDA flags above for the generated
warp implementation, with `COSMICTIGER_STF_L2L_LANE0` for the interim scalar-lane
implementation, or with `COSMICTIGER_STF_BASELINE_HEADER` for the legacy warp.
Run `--check` for validation alone, which covers every FMM order 3 through 9
(local degrees 2 through 8), or pass sample, round, and repetition counts for
timing. Timing runs one order: the build's default (`ORDER`, 7) unless
`--order N` selects another; `--order` also narrows `--check` to that order. The
generated warp build also validates and times `translateInto`, which writes into
a shared-memory expansion instead of broadcasting to every lane; its rows carry
the `_into` suffix, and the CSV includes the local degree. The legacy header is
degree-six only, so its build requires `ORDER=7` and validates that order alone.
The legacy implementation has a known omitted translation term;
its diagnostic error is reported separately from timing and must not be treated
as a passing correctness result.

## Complete periodic walks

Configure both source snapshots with
`-DCOSMICTIGER_BUILD_STF_BENCHMARKS=ON`, and inject the identical
`benchmarks/stf_walk.cpp` and test dispatch into the baseline. Use matching
compiler, build type, kernel choices, opening angle, softening radius, and HPX
settings. Use the same CUDA-enabled binary for its CPU and CUDA runs, selecting
the backend with `--compute.cuda=false` or `--compute.cuda=true`.

The test entry point is:

```sh
/absolute/path/to/cosmictiger --run.test=stfBenchmark \
  --initial.particlesPerDimension=32 --hydro.enabled=false \
  --compute.cuda=false --hpx:threads=2 --hpx:bind=none
```

Run each build/backend in its own empty working directory. The harness requires
one locality and two HPX workers. It creates 32^3 particles directly, without
power-spectrum initial conditions, for a jittered lattice and eight dense
clusters. It resets positions, velocities, forces, potentials, and rungs before
each iteration and provides the complete root direct and Ewald checklists.
Each case has one warmup and five measured samples; override the latter with
`COSMICTIGER_STF_WALK_SAMPLES`.

`STF_WALK_CONFIG` records the actual workload settings, including the runtime
FMM order (`order=`, from `--gravity.fmmOrder`). `STF_WALK_SAMPLE` and
`STF_WALK_MEDIAN` report reset, tree, and kick times separately, plus an
order-independent input-position fingerprint and force/potential checksums.
`run_cuda_kick_arena.py --orders 5,7,9` sweeps orders; the order appears in run
names, result keys, and every summary table.
Checksums, file output, tree cleanup, and kick workspace cleanup are outside
the timed tree/kick intervals.

`stf_walk_forces.csv` contains the final potential-enabled force results for both
fixtures. Match rows by `(fixture, x_raw, y_raw, z_raw)` before computing maximum
and RMS force/potential differences; tree ordering may differ. Check equal row
counts and position keys first. For potential, report absolute differences as
well as relative errors so a background-gauge shift cannot be hidden.
