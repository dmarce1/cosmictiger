# Source organization

Public headers live under `cosmictiger/` and implementations under `src/`.
Both trees are organized by subsystem and algorithm. A backend with one source
file can live directly in its subsystem directory, as `src/chemistry/recombination.cpp`
does. Do not create single-file directories: flatten the file into its parent with
a prefix naming the former directory. Within a subsystem directory, omit a
redundant subsystem prefix (for example, `sph/cuda.hpp`, not `sph/sph_cuda.hpp`).
This applies independently to the header and source trees. Common machinery sits directly
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

`gravity/global_fmm/` contains distributed tree construction, remote node caches,
particle exchange, task execution, and CUDA staging. Shared tree construction,
Cartesian force operators, periodic Ewald corrections, and CPU/CUDA walks live
directly in `gravity/`. Generated Cartesian kernels live in
`gravity/kernels/`; direct-force diagnostics use `error.*`, `direct_reference.hpp`,
and `reference_table.*`.

`COSMICTIGER_ENABLE_GRAVITY=OFF` excludes the solver. Global FMM uses owned
particle storage and distributed caches in `particles/`, shared with drift,
domain migration, SPH, checkpointing, and output. Mesh gravity and its retained
long-force and frozen-cell storage have been removed. Checkpoint headers from
retired mesh solvers are rejected before their payload is read.

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
├── parent_tree.*, ghosts.*                # Collectives over locality domains; ghost leaves
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
child IDs and are skipped by both construction and neighbor traversal. Each
locality builds its own octree at depth 0; there are no rank-routing nodes.
`parent_tree.*` holds the parent tree over the locality domains, whose topology
drives every SPH collective, and `ghosts.*` mirrors the remote leaves a walk can
reach, so walks and passes read only local and ghost data
([parent tree plan](sph_parent_tree_plan.md)).
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

Generated Cartesian trace-free declarations are checked in beneath
`cosmictiger/gravity/kernels/`, with implementations beneath `src/gravity/kernels/`
and counting-test definitions beneath `tests/gravity/kernels/instrumentation/`.
Free-space and periodic Ewald operators share the `cartesian_tracefree/` directory.

The generator entry point is `tools/generate_fmm.py`, with supporting emitters in
`tools/cartesian/`. Running it without arguments regenerates the degree-6 source
references using the repository's clang-format configuration; `--check` verifies
those references without writing. CMake generates one kernel set per build for
`COSMICTIGER_EXPANSION_DEGREE` (default 6, supported values 2 through 8). Build
outputs live in `generated_fmm/` and take precedence over the source references.
The internal FMM order is the potential degree plus one. Runtime degree selection
is removed; a different degree requires another build directory.

`generate_fmm_kernels` is the explicit CMake generation target;
`generate_cartesian_kernels` and `cosmictiger_generated_kernels` are aliases.
The static `cosmictiger_fmm` library contains the selected scalar, SIMD, and,
when enabled, CUDA definitions. CPU/CUDA instantiation templates in `src/gravity/`
assemble the solver at that build's single order.

A file move must update includes, CMake sources, generated output paths,
per-source compiler options, and active test or benchmark inputs together.
Tests mirror solver ownership under `tests/gravity/` and `tests/sph/`; global
FMM tree and periodic-force regressions live in `tests/gravity/global_fmm/`. Historical benchmark manifests and patches retain the
paths of the source versions they describe. Public option keys, serialized fields, and test names
are independent of the source-directory names.
