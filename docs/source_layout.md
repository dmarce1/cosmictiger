# Source organization

Public headers live under `cosmictiger/` and implementations under `src/`.
Both trees are organized by subsystem and algorithm. A backend with one source
file can live directly in its subsystem directory, as `src/chemistry/recombination.cpp`
does. Do not create single-file directories: flatten the file into its parent with
a prefix naming the former directory. Within a subsystem directory, omit a
redundant subsystem prefix (for example, `sph/cuda.hpp`, not `sph/sph_cuda.hpp`).
This applies independently to the header and source trees; the single-file
PMMM PP header is `gravity/pmmm/pp_near_pp.hpp`. Common machinery sits directly
in its subsystem directory; there is no `shared/` layer.

## Build libraries

The application combines capability-dependent objects with smaller static
libraries selected in `cmake/variants.cmake`:

| Library | Contents | Build requirement |
| --- | --- | --- |
| `cosmictiger_core` | Assertions, counters, memory utilities, kernels, and CUDA utilities | All builds |
| `cosmictiger_cosmology` | Background evolution, Einstein–Boltzmann solver, cosmological initial conditions, and background plotting | Gravity |
| `cosmictiger_fft` | Distributed transforms and FFTW threading support | Gravity |
| `cosmictiger_power` | CPU/CUDA power-spectrum analysis | Gravity |
| `cosmictiger_recombination` | Homogeneous recombination history | Gravity |
| `cosmictiger_sph_chemistry` | GADGET-4 particle cooling | SPH and chemistry |

The cosmology library depends on FFT and recombination; power analysis depends
on FFT. Particle cooling does not depend on homogeneous recombination. The
generic core has no cosmology or analysis sources and does not inherit the
application's capability definitions.

Disabling gravity (`COSMICTIGER_ENABLE_GRAVITY=OFF`) excludes the
gravity-only libraries, cosmological scenario, and their tests and reference
executables. Such builds also omit GSL, FFTW, and both HEALPix dependencies.
HPX, MPI, threads, and Silo remain available for hydrodynamics, distribution,
and output. Capability definitions propagate through the selected physics
libraries and application objects; standalone tests receive the same
definitions without linking unrelated physics libraries.

## Gravity

```text
gravity/
├── kernels/                               # Generated STF Cartesian and spherical kernels
├── fmm_types.hpp, fmm_order.*
├── fmm_acceptance.hpp, fmm_geometry.hpp, softening.hpp
├── sources.*, source_access.*
├── tree*, gravity*, kick*                  # Local tree/walk/CPU/CUDA execution of the near field
├── error.*, direct_reference.hpp, reference_table.*   # Sampled direct-force diagnostics
├── pm/                                    # Mesh infrastructure shared by the PMMM stages
│   ├── collective_dispatch.*, fft_flops.hpp
│   ├── kick.*                             # Streamed field consumer with long half-kicks
│   └── outer_interval.*                   # Outer interval selection
└── pmmm/
    ├── solver.*, kernels.*, mesh_size.hpp, fft.cpp, mesh_exchange.cpp, interval.cpp
    ├── evaluation.cpp, kick.*
    ├── state.hpp, actions.hpp              # Private source implementation headers
    ├── pp_near_pp.hpp, pp/near_pp.*        # Direct particle-pair near field (fallback)
    └── fmm/{cell_tree.*,fmm_batch.*,cell_schedule.cpp,ghost_storage.*,ghost_exchange.cpp}
```

An asterisk denotes the applicable header, C++ implementation, and/or CUDA
implementation, rather than a requirement that every file have all three.
The `state.hpp` and `actions.hpp` headers belong only to `src/gravity/pmmm/`.

Directories identify solver families, not paper authors. `pmmm/` owns the
multipole mesh convolution and its FMM and PP near fields; the local tree and
walk machinery of the FMM near field sits at the gravity root. Generic FFT
accounting, collective dispatch, the per-particle long acceleration consumer and
the outer interval selection live in `pm/`. PMMM's tensor convolution is not a
scalar PM solve merely because it uses FFTs.

`COSMICTIGER_ENABLE_GRAVITY=OFF` excludes the whole solver. Public runtime
selectors and serialized layouts are preserved: `gravity.method.near` keeps its
name, and checkpoints written by the retired global FMM, PM and P3M solvers are
rejected rather than misread.

The PMMM near FMM uses the parent directory's Cartesian operators, local tree
construction, acceptance tests, traversal, and CPU/CUDA workspaces.
The common tree builder accepts bounded-center and child-routing
callbacks; the common walk receives solver execution callbacks. Remote source
and tree access are registered by the active solver. These adapters run at task,
node, or source-copy boundaries; pair operators remain inline CPU/CUDA code.

Owned particle storage remains in `particles/`. It is also used by drift,
domain migration, SPH, checkpoints, and output. Common gravity source views
borrow that storage. PMMM owns a separate ghost arena and publishes its
borrowed source view before readers start. Ghost packets contain source
positions, particle types, and frozen timestep rungs, and never become owned
particles. Generic particle-release hooks let solver-owned auxiliary storage
follow particle teardown without a dependency from `particles/` to a solver.

`core/tree_id.hpp` contains the tree identity and hashing types used by both
gravity and SPH. The analysis Barnes–Hut tree lives in `analysis/`; analytic
force references used by tests live in `tests/gravity/`.

## SPH

`COSMICTIGER_ENABLE_SPH` defaults to `ON`. Setting it to `OFF` excludes SPH,
hydrodynamic scenarios, particle chemistry, and their tests. Homogeneous
recombination remains available for collisionless cosmology.
`COSMICTIGER_ENABLE_CHEMISTRY` independently controls particle chemistry and
defaults to `ON`. Particle chemistry sources and tests are included only when
both SPH and chemistry are enabled. With SPH off, the chemistry setting is
inactive; homogeneous recombination remains available with either setting.
Homogeneous recombination is built only when a gravity algorithm is enabled.

```text
sph/                                      # Under both cosmictiger/ and src/
├── sph.*, cpu.*, cuda.*
├── particles.*, tree.*
├── prehydro_cpu.cpp, prehydro.cu
├── hydro_cpu.cpp, hydro.cu
├── rungs_cpu.cpp, rungs.cu
├── run_data.hpp, flops.hpp, volume.hpp, variable_smoothing_lengths.hpp
└── viscosity/
    └── monaghan.hpp, morris_and_monaghan.hpp, balsara.hpp

chemistry/sph.*                            # Particle cooling integration
stars/
├── star_formation.*, star_formation_cuda.*, star_formation_kernel.hpp
└── springel_and_hernquist/                 # Header-only equations and cooling
    ├── equations.hpp
    └── cooling.hpp
```

The `sph/` directory owns particle storage, neighbor traversal, smoothing and
density loops, execution, and timesteps. `sph/viscosity/` owns the viscosity
models. The Monaghan module supplies pair viscosity; Morris–Monaghan supplies the
coefficient-evolution law. The Balsara module supplies the shear limiter. `volume.hpp` holds the weighted-volume sums, support derivatives,
constraint, and grad-H factors. `variable_smoothing_lengths.hpp` holds the
compatible pressure force and thermal work used for every SPH run
([variational SPH](variational_sph.md)). The common CPU and CUDA loops call
those modules, so selecting a viscosity model does not create a second
neighbor engine.

`tree.*` builds eight-way spatial subdivisions for each rank's SPH particles.
Octant bit `d` selects the upper half of dimension `d`; empty octants have invalid
child IDs and are skipped by both construction and neighbor traversal. The
existing binary domain-key hierarchy remains the routing layer between ranks.
Depths 0 through 15 are allowed. A node that still needs subdivision at depth 15
throws `std::out_of_range` because its children would require depth 16; this also
rejects unresolved coincident-particle buckets. Child tasks finish before a
construction exception propagates, so the arena can be safely reset or released.

`run_data.hpp` contains production CPU/CUDA run data and shared source-completion
and support helpers; it is not a smoothing-kernel catalogue. Ideal-gas entropy
conversions live with the entropy-backed particle accessors in `particles.hpp`.
The independent cubic-spline reference formulas live in `tests/sph/kernels.hpp`;
production smoothing kernels remain in `cosmictiger/math/`.
Cooling stores a plain `float electronFraction` in batch and migration records,
with the particle array named `sphParticlesElectronFractions`. The default guess
remains `1.0f`, and checkpoints retain one float per gas particle.

There is no viscosity selector: every SPH run stores, migrates, checkpoints and
outputs the per-particle α and Balsara factor.
Morris–Monaghan alpha updates only at completed own interval endpoints; startup
does not advance it. Midpoint forces use the last endpoint's alpha. Balsara is
computed with separate SPH divergence/curl sums during the current density pass,
is always on for Morris–Monaghan, and is optional for constant α. Morris–Monaghan
retains no divergence history.
Alpha is two-byte binary16 and Balsara storage is one `uint8_t`, for three bytes
per gas particle. Balsara factors remain transient, are never checkpointed or
migrated, and default on for constant α.
`SphParticleFlags` replaces separate convergence, semiactivity and proposed-rung
arrays with one byte: bits 0/1 are the flags and bits 2–7 are the proposed rung,
with no unset sentinel. Current rungs are 0–15. Explicit setters are
required because packed bits cannot expose mutable references. Memory,
checkpoint and migration storage each shrink by two bytes per gas particle.
SPH checkpoint base 130000 introduced this packed layout; base 140000 removes the
gradient-model state. See
[viscosity timing and storage](monaghan_sph.md#morrismonaghan-variable-viscosity).

Exact Sod, Sedov, and hydrostatic sphere reference/initial-condition helpers live
in `scenarios/`, alongside the scenarios that use them.

## Generated code and build integration

The old `cosmictiger/fmm/` and `src/fmm/` directories are folded into gravity.
Generated headers are checked in beneath `cosmictiger/gravity/kernels/`,
with flat `cartesian_tracefree_*` names for the free-space near-field operators
and `spherical_*` names for the free-space and Ewald far-field operators.

Generators remain organized by representation under `tools/cartesian/` and
`tools/fmm/`, with `tools/generate_fmm.py` as the entry point. Running it without
arguments regenerates both families for FMM orders 3 through 9 in the source
header directory, using the repository's clang-format configuration.
`generate_fmm_kernels` is the explicit CMake regeneration target;
`generate_cartesian_kernels` and `cosmictiger_generated_kernels` are aliases.
These targets always regenerate all internal orders 3 through 9, corresponding
to potential degrees 2 through 8. `COSMICTIGER_EXPANSION_DEGREE` selects a scalar
degree or inclusive range, defaulting to `4-6`. CMake converts that selection
to internal orders by adding one, then uses `FMM_MIN_ORDER`/`FMM_MAX_ORDER`
preprocessor guards and per-order instantiations. Runtime
`gravity.expansionDegree` chooses one compiled degree, defaulting to the largest.
Ordinary builds use the checked-in headers without rewriting them, and
`python3 tools/generate_fmm.py --check` verifies them without writing.

Per-order CPU/CUDA
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
