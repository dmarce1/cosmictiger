# Source layout

`src/` and `cosmictiger/` are organized into the same twelve subdirectories,
one level deep, so `cosmictiger/<dir>/x.hpp` pairs with `src/<dir>/x.cpp`.
Every project include names its directory: `#include <cosmictiger/gravity/kick.hpp>`.
`src/main.cpp` stays at the root as the entry point, and `src/tools/` holds the
standalone executables.

| directory | contents |
|---|---|
| `core` | application skeleton: assert, defs, containers, HPX plumbing, options, the driver loop, timers, profiler, flop counters, memory and stack-trace diagnostics |
| `math` | numerics with no simulation state: math helpers, complex, constants, units, fixed-point positions, double-float, half floats, SIMD, tensors, ranges, radial kernels and the kernel tables |
| `cuda` | device runtime wrappers: streams, memory, reductions, device vectors, device hash map |
| `particles` | the particle store, domain decomposition and drift |
| `fmm` | multipole algebra: order traits, expansion types, kernels, acceptance tests, Ewald indices and tables, the Ewald codegen; the generated kernels land in `${build}/generated/cosmictiger/fmm/` |
| `gravity` | the tree, the CPU walk, the CUDA kick (kernel, launch, per-order instances), the kick workspace and the Barnes-Hut potential solver |
| `sph` | hydrodynamics: SPH driver, CPU and CUDA runs, SPH particles and tree, chemistry bridge, Monaghan (1992) formulas |
| `chemistry` | the primordial chemistry library (IEEE-strict, its own static library) |
| `cosmology` | background cosmology, recombination, the Boltzmann solver, initial conditions and the distributed FFT |
| `analysis` | group finding, halo catalogues (ROCKSTAR), the lightcone and its compression, the power spectrum |
| `io` | SILO views, tracer and slice output, checkpoints |
| `problems` | test problems and analytic references (hydro problems, Sedov, Sod, Plummer, sphere surface, analytic gravity) and the in-binary test dispatcher |

Rules of thumb when adding a file:

- Put the header and its source in the same directory, and list both in
  `CMakeLists.txt` (`cosmictiger_source_files`, `cosmictiger_header_files`).
- `fmm` must not depend on `particles` or `gravity`: the STF unit tests build
  it standalone.
- Test problems and in-binary tests go in `problems/` or `tests/`, not next to
  the production code they exercise; `tests/*.cpp` files compiled into the
  main binary (for example `tests/rockstar.cpp`, `tests/groups_catalogue.cpp`)
  are the precedent.
- Code outside the production path that needs a finder's internal tables gets
  a narrow read accessor (see `detail::groupsForEachCandidate` in
  `cosmictiger/analysis/groups.hpp`), not a shared static.
- The clang-tidy header filter in `.clang-tidy` admits exactly one directory
  level; do not nest deeper.
