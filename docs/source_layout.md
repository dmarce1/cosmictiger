# Source organization

Public headers live under `cosmictiger/` and implementations under `src/`.
Both trees use the same algorithm directories. Common machinery sits directly
in its subsystem directory; there is no `shared/` layer.

## Gravity

```text
gravity/
├── fmm_types.hpp, fmm_order.*, fmm_kernels.hpp
├── fmm_acceptance.hpp, fmm_geometry.hpp
├── sources.*, source_access.*
├── tree.hpp, tree_build.*, tree_arena.cpp, tree_multipole.hpp
├── gravity.*, gravity_cpu.hpp, gravity_cuda.hpp
├── kick.*, kick_context.hpp, kick_walk.hpp, kick_workspace.*
├── kick_executor.cpp, kick_cuda*.hpp, kick_interaction_counts.*
├── dehnen/
│   ├── distributed_tree.*
│   ├── remote_cache.*, particle_exchange.*
│   ├── execution.*, gpu_staging.*
│   ├── periodic.*, periodic_cpu.hpp, periodic_cuda.hpp
│   └── ewald_indices.*, ewald_tables.hpp, ewald_kernels.hpp
└── nitadori/
    ├── solver.*
    ├── fft.cpp, mesh_exchange.cpp, evaluation.cpp
    ├── ghost_storage.*, ghost_exchange.cpp
    ├── cell_tree.*, cell_schedule.cpp, fmm_batch.*
    ├── kernels.*, near_pp.*
    ├── kick.*, collective_dispatch.*
    └── state.hpp, actions.hpp                 # Private implementation headers
```

An asterisk denotes the applicable header, C++ implementation, and/or CUDA
implementation, rather than a requirement that every file have all three.
The `state.hpp` and `actions.hpp` headers belong only to `src/gravity/nitadori/`.

`dehnen/` contains the global adaptive Cartesian FMM solver, including distributed
tree construction, remote node and particle access, and Cartesian Ewald
corrections. `nitadori/` contains the particle mesh multipole solver: spherical
moments and FFT convolution, whole-cell ghost exchange, readiness scheduling,
and the direct-PP or cell-FMM near-field adapters. The directory names identify
method families; the numerical documentation records the particular conventions
and implementation choices.

Both solvers use the parent directory's Cartesian operators, local tree
construction, acceptance tests, traversal, and CPU/CUDA workspaces. In
particular, the PMMM cell FMM uses these common routines without invoking the
global solver. The common tree builder accepts bounded-center and child-routing
callbacks; the common walk receives solver execution callbacks. Remote source
and tree access are registered by the active solver. These adapters run at task,
node, or source-copy boundaries; pair operators remain inline CPU/CUDA code.

Owned particle storage remains in `particles/`. It is also used by drift,
domain migration, SPH, checkpoints, and output. Common gravity source views
borrow that storage. Nitadori owns a separate ghost arena and publishes its
borrowed source view before readers start. Ghost packets contain source
positions, particle types, and frozen timestep rungs, and never become owned
particles. Generic particle-release hooks let solver-owned auxiliary storage
follow particle teardown without a dependency from `particles/` to a solver.

`core/tree_id.hpp` contains the tree identity and hashing types used by both
gravity and SPH. The analysis Barnes–Hut tree lives in `analysis/`; analytic
force references used by tests live in `tests/gravity/`.

## SPH

```text
sph/
├── sph.*, sph_cpu.hpp, sph_cuda.hpp, sph_kernel.hpp
├── sph_particles.*, sph_tree.*
├── sph_prehydro_cpu.cpp, sph_prehydro.cu
├── sph_hydro_cpu.cpp, sph_hydro.cu
├── sph_rungs_cpu.cpp, sph_rungs.cu
├── sph_chemistry.*, sph_flops.hpp, kernels.hpp, viscosity_state.hpp
├── monaghan/
│   ├── equations.hpp
│   └── viscosity.hpp
├── morris_and_monaghan/
│   └── viscosity.*
├── cullen_and_dehnen/
│   ├── viscosity.*
│   ├── state.hpp
│   └── particle_exchange.*
└── balsara/
    └── shear_limiter.hpp
```

The parent directory owns particle storage, neighbor traversal, smoothing and
density loops, execution, timesteps, and common viscosity state. The Monaghan
module supplies the pressure/thermal-work equations and pair viscosity;
Morris–Monaghan and Cullen–Dehnen supply the coefficient-evolution laws. The
Balsara module supplies the optional shear limiter. The common CPU and CUDA
loops call those modules, so selecting a viscosity model does not create a
second neighbor engine.

The runtime selector remains
`hydro.viscosityEvolution=none|morrisAndMonaghan|cullenAndDehnen`. Model-specific
state allocation, migration, checkpoint identity, and output are handled at the
existing storage and integration boundaries. Cullen–Dehnen's additional
particle exchange is owned by its module. The author-directory reorganization
does not introduce Hopkins/MFV hydrodynamics.

Exact Sod, Sedov, and hydrostatic sphere reference/initial-condition helpers live
in `scenarios/`, alongside the scenarios that use them.

## Generated code and build integration

The old `cosmictiger/fmm/` and `src/fmm/` directories are folded into gravity.
Generated headers are written beneath the build tree's
`generated/cosmictiger/gravity/`:

- Common Cartesian free-space operators and configuration stay at the parent level.
- Cartesian Ewald operators go into `dehnen/`.
- Spherical PMMM and spherical Ewald operators go into `nitadori/`.

Generators remain organized by representation under `tools/cartesian/` and
`tools/fmm/`, with `tools/generate_fmm.py` as the entry point. Per-order CPU/CUDA
instantiation templates remain in `src/gravity/`; they assemble common and
solver-specific instantiations exactly once per order. These are build
composition points, not an algorithm dependency from common numerical code to
a solver.

A file move must update includes, CMake sources, generated output paths,
per-source compiler options, and active test or benchmark inputs together.
Tests mirror solver ownership under `tests/gravity/` and `tests/sph/`, with
common numerical tests at the parent level and solver tests in author
subdirectories. Gravity benchmarks use the same convention beneath
`benchmarks/gravity/`. Historical benchmark manifests and patches retain the
paths of the source versions they describe. Public option keys, serialized fields, and test names
are independent of the source-directory names.
