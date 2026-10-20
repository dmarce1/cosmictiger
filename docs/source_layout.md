# Source layout

`src/` and `cosmictiger/` are organized into the same twelve subdirectories,
one level deep, so `cosmictiger/<dir>/x.hpp` pairs with `src/<dir>/x.cpp`.
Every project include names its directory: `#include <cosmictiger/gravity/kick.hpp>`.
`src/main.cpp` stays at the root as the entry point, and `src/tools/` holds the
standalone executables. Developer regressions live in `tests/`, including the
colocated in-binary dispatcher source and header, `tests/test.cpp` and
`tests/test.hpp`.

| directory | contents |
|---|---|
| `core` | application skeleton: assert, defs, containers, HPX plumbing, options, the shared force stepping and scenario evolution drivers, timers, profiler, flop counters, memory and stack-trace diagnostics |
| `math` | numerics with no simulation state: math helpers, complex, constants, units, fixed-point positions, double-float, half floats, SIMD, tensors, ranges, radial kernels and the kernel tables |
| `cuda` | device runtime wrappers: streams, memory, reductions, device vectors, device hash map |
| `particles` | the particle store, domain decomposition and drift |
| `fmm` | multipole algebra: order traits, expansion types, kernels, acceptance tests, Ewald indices and tables, the Ewald codegen; the generated kernels land in `${build}/generated/cosmictiger/fmm/` |
| `gravity` | the tree, the CPU walk, the CUDA kick (kernel, launch, per-order instances), the kick workspace, the Barnes-Hut potential solver, and the analytic gravity reference evaluator |
| `sph` | hydrodynamics: CPU and CUDA runs, SPH particles and tree, chemistry bridge, Monaghan (1992) formulas, exact Sod and Sedov reference solutions, and CPU/CUDA spherical-shell relaxation helpers |
| `chemistry` | the primordial chemistry library (IEEE-strict, its own static library) |
| `cosmology` | background cosmology, recombination, the Boltzmann solver, cosmological initial-condition machinery and the distributed FFT |
| `analysis` | group finding, halo catalogues (ROCKSTAR), the lightcone and its compression, the power spectrum |
| `io` | SILO views, tracer and slice output, checkpoints |
| `scenarios` | first-class scenario selection and initializers: Plummer, Sod, blast, Rayleigh–Taylor, Kelvin–Helmholtz, disc, polytropic star, hydro wave, cosmology and glass |

Rules of thumb when adding a file:

- Pair production headers in `cosmictiger/<family>/` with sources in
  `src/<family>/`, and list both in `CMakeLists.txt`
  (`cosmictiger_source_files`, `cosmictiger_header_files`). Test-only headers
  stay beside their implementations in `tests/`.
- `fmm` must not depend on `particles` or `gravity`: the STF unit tests build
  it standalone.
- Physical initializers and scenario-specific completion diagnostics go in
  `scenarios/`; they use the common evolution drivers in `core/`. Solver
  reference solutions and initialization helpers belong to their solver family:
  `gravity/analytic.*`, and `sph/exact_sod.*`, `sph/sedov.*`, `sph/sphere.*`.
- Developer regressions and benchmarks stay in `tests/` or `benchmarks/`. The
  in-binary regression dispatcher is `tests/test.cpp`, with its declaration in
  `tests/test.hpp`; other tests compiled into the main binary include
  `tests/rockstar.cpp` and `tests/groups_catalogue.cpp`.
- Code outside the production path that needs a finder's internal tables gets
  a narrow read accessor (see `detail::groupsForEachCandidate` in
  `cosmictiger/analysis/groups.hpp`), not a shared static.
- The clang-tidy header filter in `.clang-tidy` admits files directly under
  `cosmictiger/`, `src/`, or `tests/`, plus one directory level; do not nest deeper.
