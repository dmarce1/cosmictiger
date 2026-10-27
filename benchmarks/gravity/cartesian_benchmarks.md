# Cartesian performance comparisons

The Cartesian structures are `cosmictiger::gravity::cartesian::Multipole<Real, order>`
and `cartesian::Expansion<Real, order>`, indexed by the application FMM order.
Generated operators retain degree-based arguments in the same namespace and the
standard abbreviations `P2M`, `M2M`, `M2L`, `M2P`, `P2L`, `L2L`, and `L2P`.

Run comparisons after the correctness checks pass, with other compilation and
benchmark jobs idle. Microbenchmarks measure particular operators. The complete
walk includes tree creation, acceptance decisions, direct and periodic
interactions, and the production CUDA work distribution.

## Microbenchmarks

Preserve the legacy `cosmictiger/fmm_kernels.hpp` before replacing it. The harness
includes that external file through `COSMICTIGER_CARTESIAN_BASELINE_HEADER`; it uses the
current common math, SIMD, tensor, and container headers for both implementations.
This compares kernel implementations with the same supporting infrastructure,
rather than two independently changed SIMD implementations.

The runner prints compiler commands when neither action flag is supplied:

```sh
python3 benchmarks/gravity/run_cartesian.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/cartesian_microbench
```

Compile all six binaries serially:

```sh
python3 benchmarks/gravity/run_cartesian.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/cartesian_microbench --build
```

The defaults are GCC 13, C++20, `-O3 -ffast-math -march=native`, CUDA `sm_86`, and
`-O3 --use_fast_math`. Both CUDA variants use `-rdc=true`, required by the legacy
header's inline constant-memory arrays. CUDA headers are also needed for the
host compiler because the legacy header declares CUDA storage. Override
`--cxx`, `--cuda-root`, and `--arch` together for a different toolchain.
`--skip-cuda` omits the CUDA binaries; it does not remove the CUDA-header
requirement for this legacy source. The Cartesian variants include the generated
kernels checked in under `cosmictiger/gravity/`, using the source-root include
path already supplied by the runner. After changing a generator, explicitly
regenerate with `python3 tools/generate_fmm.py`; ordinary builds leave the
headers unchanged. The runner fixes the FMM order at `ORDER=7`, the application
default and the only order the legacy header implements.

Use a fresh output directory. The build manifest records compiler versions,
exact argument arrays, source/header hashes, and executable hashes. Builds do
not start timing runs. To execute the prepared binaries, select an allowed CPU
that is not occupied by another workload:

```sh
python3 benchmarks/gravity/run_cartesian.py \
  --baseline-header /tmp/fmm_kernels_before.hpp --output /tmp/cartesian_microbench \
  --run --cpu 2
```

The runner validates the recorded sources and binaries, runs one process at a
time, pins CPU runs to one logical CPU, and alternates legacy/Cartesian ordering across
three pairs. Each process reports the median of seven measurements after two
warmup sweeps. Results, commands, stderr, and paired ratios are saved beneath
`measurements/`. `median_cartesian_over_legacy < 1` indicates a shorter Cartesian execution
time for that case. Existing measurements are never overwritten.

CUDA runs are limited to 32 blocks of 32 threads, with a grid-stride loop. This
is below half of the 82 SMs on the RTX 3090 workstation. It measures deliberately
limited concurrency; it is not an estimate of unrestricted full-GPU throughput.

The timed operations are:

| Case | Legacy | Cartesian |
|---|---|---|
| P2M | Point-to-moment projection | Generated point-to-Cartesian projection |
| M2M | Legacy moment translation | Generated Cartesian moment translation |
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
storage is 37 legacy coefficients versus 36 Cartesian coefficients, so the associated
storage traffic is part of the measured implementation difference.

Raw moment and local coefficient checksums must not be compared across layouts.
PC/L2P checksums use the common potential/x-gradient/y-gradient/z-gradient order,
but checksums are still only a consumption/sanity check, not a numerical error
norm. Correctness comes from the independent kernel tests and position-matched
complete-walk force output. Compare legacy versus Cartesian within the same type and
backend: scalar and SIMD cases have different lane counts and working-set sizes.
The scalar and SIMD source-offset sequences also differ by their lane width.

CUDA L2L microbenchmarks execute one independent expansion per thread. They do
not measure the legacy cooperative warp L2L routine. The complete-walk comparison
is required to assess the replacement's effect on that production path.

`benchmarks/gravity/cartesian_l2l_cuda.cu` separately measures the production arrangement:
one expansion per 32-thread block, including the caller's in-place update and
synchronization. Compile it with the same CUDA flags above for the generated
warp implementation, with `COSMICTIGER_CARTESIAN_L2L_LANE0` for the interim scalar-lane
implementation, or with `COSMICTIGER_CARTESIAN_BASELINE_HEADER` for the legacy warp.
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
`-DCOSMICTIGER_BUILD_CARTESIAN_BENCHMARKS=ON -DBUILD_TESTING=ON`, and inject the identical
`benchmarks/gravity/dehnen/cartesian_walk.cpp` and test dispatch into the baseline. Use matching
compiler, build type, kernel choices, opening angle, softening radius, and HPX
settings. Use the same CUDA-enabled binary for its CPU and CUDA runs, selecting
the backend with `--compute.cuda=false` or `--compute.cuda=true`.

The test entry point is:

```sh
/absolute/path/to/cosmictiger --run.test=cartesianBenchmark \
  --initial.particlesPerDimension=32 --hydro.enabled=false \
  --compute.cuda=false --hpx:threads=2 --hpx:bind=none
```

Run each build/backend in its own empty working directory. The harness requires
one locality and runs on the configured number of HPX worker threads
(`--hpx:threads`, recorded as `threads=` in `CARTESIAN_WALK_CONFIG`). It creates
`initial.particlesPerDimension` cubed particles directly (32^3 above; the
adaptive fixture's dense-cube selection is exact for power-of-two counts),
without power-spectrum initial conditions, for a jittered lattice and eight
dense clusters. It resets positions, velocities, forces, potentials, and rungs before
each iteration and provides the complete root direct and Ewald checklists.
Each case has one warmup and five measured samples; override the latter with
`COSMICTIGER_CARTESIAN_WALK_SAMPLES`.

`CARTESIAN_WALK_CONFIG` records the actual workload settings, including the compiled
FMM order (`order=`, selected by `COSMICTIGER_WITH_ORDER`). `CARTESIAN_WALK_SAMPLE` and
`CARTESIAN_WALK_MEDIAN` report reset, tree, and kick times separately, plus an
order-independent input-position fingerprint and force/potential checksums.
Build a separate FMM executable for each order. For example,
`run_cuda_kick_arena.py --binary o5=/path/order5/cosmictiger:arena
--binary o7=/path/order7/cosmictiger:arena --binary o9=/path/order9/cosmictiger:arena
--orders 5,7,9` selects those compiled orders; the order appears in run names,
result keys, and every summary table. The driver reads each binary's build
configuration with `--help`, rejects PMMM builds, and never passes an order option
to a current executable. Historical binaries with an advertised runtime order
option retain their original invocation. `run_production_accuracy.py` requires
current compiled-order binaries; supply additional timing candidates with
`--order-binary ORDER=PATH`.
Checksums, file output, tree cleanup, and kick workspace cleanup are outside
the timed tree/kick intervals.

`cartesian_walk_forces.csv` contains the final potential-enabled force results for both
fixtures. Match rows by `(fixture, x_raw, y_raw, z_raw)` before computing maximum
and RMS force/potential differences; tree ordering may differ. Check equal row
counts and position keys first. For potential, report absolute differences as
well as relative errors so a background-gauge shift cannot be hidden.
