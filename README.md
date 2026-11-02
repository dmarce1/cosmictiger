# CosmicTiger - A Cosmological N-Body Code

## License

CosmicTiger's original code is distributed under the
[Boost Software License 1.0](LICENSE), the license used by HPX.
Bundled and adapted third-party code retains its own terms and attribution;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The current application
includes and links GPL-licensed components, so this change does not make the
complete executable a Boost-only distribution.

## Command-line options

Application options use `groupName.optionName` with camelCase names, except for
the top-level `--help` and `--configuration` options. Run `./cosmictiger --help`
for all options and defaults. HPX's own options retain their `--hpx:*` syntax.

| Group | Controls |
| --- | --- |
| `run` | Test selection, iteration limit, time-step hierarchy, verbose diagnostics |
| `compute` | Position representation and tree bucket sizes |
| `drift` | Particle drift execution backend |
| `sort` | Particle sorting execution backend |
| `checkpoint` | Checkpoint interval and restart index |
| `initial` | Particle lattice, glass/Plummer/Gadget input, initial power spectrum |
| `output` | Particle Silo output, cadence, directory, unit system, frame, and selected box |
| `analysis` | Power spectrum, halos, tracers, slices, and views |
| `lightcone` | Lightcone output and group finding |
| `gravity` | Mesh selection, softening, global FMM opening angle, and timestep |
| `cosmology` | Redshifts, density parameters, and primordial spectrum |
| `units` | Particle mass unit in solar masses |
| `hydro` | SPH, equation of state, smoothing radii, and artificial viscosity |

In C++, each category is an ordinary public struct in `Options`: `Gravity gravity`,
`Output output`, and likewise `Run`, `Compute`, `Drift`, `Sort`, `Checkpoint`, `Initial`, `Analysis`,
`Lightcone`, `Cosmology`, `Units`, and `Hydro`. Code accesses members directly, for
example `getOptions().gravity.enabled` or `options.output.directory`.
Radius aliases resolve to canonical support radii in box units; unit conversions
are stored as `units.massG`, `units.lengthCm`, and `units.timeSeconds`.
Curvature `cosmology.omegaK` is derived from the density budget and is no longer an
accepted input, since a supplied value was always overwritten.

The old ungrouped and snake_case option names have been replaced. For example,
`--parts_dim=16 --max_iter=2 --cuda=true` is now
`--initial.particlesPerDimension=16 --run.maxIterations=2 --gravity.cuda=true`,
and `--hydro.sph=true` is now `--hydro.enabled=true`.

CUDA selection is independent for each module. All backend selectors default to
`false`: `gravity.cuda`, `hydro.cuda`, `drift.cuda`, `initial.cuda`,
`chemistry.cuda`, `starFormation.cuda`, `gravity.error.cuda`,
`analysis.power.cuda`, `analysis.groups.cuda`, and `lightcone.cuda`.
Gravity kernel tuning belongs under `gravity.cuda.*`, such as
`gravity.cuda.kickResidency`. Mixed runs can enable GPU gravity while keeping
hydrodynamics on the CPU. `hydro.cuda=true` supports SPH; MFM/MFV are CPU only.
`sort.cuda`, `analysis.tracers.cuda`, `analysis.slice.cuda`, and
`analysis.views.cuda` are accepted only as `false` because those CUDA paths are
not implemented. There is no global CUDA selector.

Use `--configuration=simulation.ini` to read an INI file. Both dotted keys and
INI sections work; explicit command-line values override configuration values
for the same key. Radius representations use the precedence described below:

```ini
[initial]
particlesPerDimension = 16
[compute]
cuda = true
[hydro]
enabled = true
viscosityAlpha = 1.0
viscosityBeta = 2.0
viscosityEpsilon = 0.01
gamma = 1.666666666666667
neighborNumber = 60
cfl = 0.22821773229381923
[run]
maxIterations = 2
```

[default.ini](default.ini) documents every application option except `help` and
`configuration`, with active source defaults and commented position-width and
radius settings.
The complete, independently editable problem files in [configurations/](configurations/README.md) contain the same
option set, with problem-specific values and inline help comments. Files are
loaded explicitly; there is no implicit loading of `default.ini`.

Positions use 32-bit fixed point by default. Select 64-bit fixed point at CMake
configuration time with `-DCOSMICTIGER_POSITION_TYPE=fixed64`. Both choices store
normalized positions in `[0, 1)`; their coordinate
increments are `2^-32` and `2^-64`, respectively. The 64-bit choice doubles position
storage and preserves smaller separations and accumulated displacements. Force
and velocity arithmetic retain their existing precision.

Each build directory uses one position representation for `cosmictiger`; its
name stays the same for either width.
The runtime `compute.positionType` defaults to that compiled representation.
An explicit CLI or INI value must match it; selecting the other width requires
reconfiguring and rebuilding. Use separate build directories to keep both widths.
Checkpoints record the representation and must be restarted with the same width.
The supplied INIs leave `positionType` commented so they use the compiled width.
The legacy `glass.bin` exchange format remains Q0.32 in both modes; checkpoints
retain the selected position precision.

One CMake configuration builds one `cosmictiger` executable with the enabled
physics and integrators:

```sh
cmake -S . -B build/release_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_WITH_CUDA=OFF -DBUILD_TESTING=OFF \
  -DCOSMICTIGER_ENABLE_FMM=ON -DCOSMICTIGER_ENABLE_PMMM_PP=ON \
  -DCOSMICTIGER_ENABLE_INTEGRATOR_GLOBAL=ON \
  -DCOSMICTIGER_EXPANSION_DEGREE=4-6
cmake --build build/release_cpu --target cosmictiger --parallel 3
build/release_cpu/cosmictiger --gravity.method.far=pmmm \
  --gravity.method.near=pp --gravity.meshSize=16 \
  --run.timeIntegrator=global --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
```

Fresh defaults enable FMM, SPH, equilibrium cooling, traditional nested integration,
and potential expansion degrees 4-6. PMMM with PP or FMM near fields, hierarchical
integration, and global integration can be enabled independently. Both PMMM options default
to `OFF` and may be enabled together. A single eligible algorithm is
automatically selected; multiple eligible algorithms require a CLI/INI selection.

Use `gravity.method.far=fmm|pmmm`, `gravity.method.near=pp|fmm` for PMMM,
`gravity.expansionDegree`, and `run.timeIntegrator=traditional|hierarchical|global`.
The highest compiled degree is the runtime default. Potential degree d is old
internal order d+1, so degrees 4-6 correspond to orders 5-7. PMMM and its near FMM
share the degree. Old checkpoint order integers retain their original meaning.

Traditional integration uses individual nested block timesteps. Hierarchical
integration uses a Hamiltonian split and currently supports collisionless runs.
Global integration advances every particle with the minimum raw allowed timestep
across all particles and ranks, without power-of-two rounding. It supports SPH,
gravity, and combined runs, including explicit chemistry source intervals.

Set `COSMICTIGER_ENABLE_SPH=OFF` for gravity-only builds, or disable
`COSMICTIGER_ENABLE_FMM`, `COSMICTIGER_ENABLE_PMMM_PP`, and
`COSMICTIGER_ENABLE_PMMM_FMM` for SPH-only builds.
`COSMICTIGER_ENABLE_HOPKINS` defaults to `OFF`. Set it to `ON` to compile
Hopkins MFM/MFV in addition to SPH; it requires `COSMICTIGER_ENABLE_SPH=ON`
and `COSMICTIGER_POSITION_TYPE=fixed64`.
`COSMICTIGER_ENABLE_CHEMISTRY` defaults to `ON`; set it to `OFF` to retain SPH
without equilibrium cooling. Equilibrium cooling is compiled only when SPH and
chemistry are both enabled. The chemistry setting is inactive when SPH is off;
homogeneous cosmological recombination remains available in every configuration.
Disabled physics and its test problems are excluded from compilation and test
registration. Coupled tests require both. Shared sources compile once, while
selected degrees keep separate CPU/CUDA instantiation units.

`./build.sh` configures these features interactively. `cosmictiger_help` reports
the configured capabilities. See [CMake options](CMAKE_OPTIONS.md) for the full
reference, defaults, and migration from the retired named physics targets.

PMMM needs `gravity.meshSize`; `gravity.pmmmNearWidth` accepts 1-4 and defaults
to 2. It fixes near-FMM theta to `sqrt(3)/(pmmmNearWidth+1)`. Global FMM retains
`gravity.openingAngle` and cosmological `gravity.openingSchedule` controls.
See [PMMM](docs/pmmm.md) for particle/FFT ownership and solver details.

Use `--gravity.errorInterval=10 --gravity.errorSampleSize=100` to compare the
gravity solver with direct accelerations on evenly spaced particles at coarse
steps 0, 10, 20, and so on. Results are written to `gravity_error.csv` in the output
directory; see [gravity accuracy diagnostics](docs/gravity_error.md).

FMM gravity and SPH can select radial kernels independently. Both use an effective radius
`effectiveRadius = 2 sigma`, where sigma is the standard deviation along one
coordinate axis. This preserves the existing numerical length scale: adopting
radius terminology does not introduce a factor-of-two conversion.

```ini
[gravity]
kernelFamily = ferrers
kernelDegree = 8
effectiveRadius = 0.01876988456953534
[hydro]
kernelFamily = wendland
kernelDegree = 8
neighborNumber = 120
```

The supported density polynomial degrees are spline 2–9, Wendland 2/5/8,
and Ferrers 2/4/6/8. Specify `gravity.effectiveRadius` or `gravity.supportRadius`,
in mean particle spacings. The imposed central force similarly accepts
`hydro.centralEffectiveRadius` or `hydro.centralSupportRadius`, in box units,
using the gravity kernel. `gravity.softening` and `hydro.centralSoftening`
remain aliases for their effective radii.

SPH adapts each particle's support radius to its density. Its bounds accept
`hydro.minEffectiveRadius` or `hydro.minSupportRadius`, and
`hydro.maxEffectiveRadius` or `hydro.maxSupportRadius`, in initial interparticle
spacings, using the hydro kernel. Neighbor and force evaluation use support
radii; timestep limits use effective radii.

Both representations may be supplied. An explicit effective-radius key wins
over its legacy alias, and either wins over a support-radius key, including
across command-line and INI inputs. A warning names each ignored key. To select
a support radius in an example INI, remove or comment out its active effective
radius or softening alias. See [kernel definitions and migration](docs/kernels.md)
for conversions and timestep conventions.

For the Kelvin–Helmholtz scenario, run from a build directory:

```sh
./cosmictiger --configuration=../../configurations/kelvin_helmholtz.ini --hpx:threads=12 --hpx:bind=none
```

This file selects `run.scenario=kelvinHelmholtz`, CUDA, nominal resolution 32, and full-box
particle Silo output. The initializer creates 32,600 gas particles and runs to
time 10. Output is configured directly in the INI:

```ini
[output]
silo = true
siloInterval = 1
directory = ./output/kelvinHelmholtz
xMin = 0
xMax = 1
yMin = 0
yMax = 1
zMin = 0
zMax = 1
```

Output is controlled only by CLI/INI options; CLI values override the INI.
`output.siloInterval` is a positive number of synchronized coarse steps between
snapshots, defaulting to 1. For KH this means every `10/1024 = 0.009765625`
simulation-time units; setting it to 10 writes every 0.09765625 time units.
Finite-duration scenarios retain their initial and final snapshots. Snapshot names
retain their coarse cycle number, so skipped cycles leave gaps in the names.

`output.directory` defaults to `./output/<scenarioName>`, relative to the working
directory, and is created automatically for all scenario products. Snapshots are
named `output/<scenarioName>/view.0.<cycle>.silo` with the default directory.
Absolute directory paths are also supported. `output.silo=false` disables
particle snapshots while retaining other selected outputs. See
[scenario configurations](configurations/README.md) for all ten initializations,
their solver choices, and durations.
`analysis.views` independently controls HEALPix maps.

Each snapshot is a master Silo file referencing one particle file per locality
under its sibling `view.0.<cycle>.silo.data` directory. Localities write their
owned particles independently; the master is published after all particle files
are closed. Open the master in VisIt, and keep its `.silo.data` directory beside
it when copying or moving a snapshot.

Gas and dark-matter particles are written as Silo point meshes named `gas` and
`dark_matter`. Their fields are node-centered point variables, with one value per
particle. The gas `kernel_support_radius` scalar retains the SPH support radius;
positions and support radii follow the selected output units and frame.

Particle positions retain the fixed-point representation used by the solver and
checkpoints. Set `output.units=cgs` for CGS quantities or
`output.units=cosmological` for Mpc/h, solar masses/h, and km/s. Set
`output.frame=comoving` or `output.frame=lab` independently; lab positions and
densities are proper quantities. Converted velocities are peculiar velocities
in either frame and never include Hubble flow. `output.units=both` and
`output.frame=both` write all four converted views in one Silo snapshot. Physical
coordinate conversion uses double precision directly from fixed point.

The defaults, `output.units=code` and `output.frame=comoving`, preserve the native
mesh names and evolution normalization. Code units require the comoving
selection. Bounds remain normalized comoving coordinates in every output mode.
See [units and frames](docs/units.md) for conversion formulas, view paths, and
the distinction between stored and physical velocity and internal energy.

The conversion utilities use the same grouping style:
`healpix2silo --input.file=view.dat --output.file=view.silo` and
`groups2halofunc --input.groupNumber=0 --analysis.boxSize=684 --analysis.numberOfBins=1000`.

## SPH

The gas solver evolves density and internal energy, using the
[variational pressure operator](docs/variational_sph.md) with grad-H corrections
for all SPH runs. It uses Monaghan artificial viscosity, FMM or PMMM gravity, and
shared particle storage. See
[the kernel, viscosity, and paper references](docs/monaghan_sph.md) for the equations,
cosmological conversions, and deliberate implementation choices. SPH support is
built by default; `-DCOSMICTIGER_ENABLE_SPH=OFF` excludes it. Enable gas
explicitly with `--hydro.enabled=true`. SPH supports traditional block timesteps
and global adaptive timesteps with FMM, either PMMM near method, or gravity
disabled. Hierarchical hydro integration is not implemented.
Select SIMD CPU SPH with `--hydro.cuda=false` or CUDA SPH with `--hydro.cuda=true`.
Particle drift selects its backend with `drift.cuda`: the CPU path uses SIMD position updates
and HPX workers, including rung filtering, glass damping, and lightcone crossings.
CPU runs can also use a build configured with `COSMICTIGER_WITH_CUDA=OFF`.

Cosmological gas runs initialize a displaced gas lattice alongside the dark
matter lattice. `initial.particlesPerDimension` specifies the lattice size for each species. With
SPH enabled, the total particle count doubles, while the gas and dark-matter
masses are weighted by `omega_b / omega_m` and `omega_c / omega_m` respectively.
The power-spectrum deposition uses these masses too.

For a small cosmological run using the native Einstein–Boltzmann power spectrum:

```sh
./cosmictiger --hydro.enabled=true --gravity.cuda=true --hydro.cuda=true --drift.cuda=true --initial.particlesPerDimension=16 --run.maxIterations=2 --hpx:threads=4
```

Select `hydro.method=sph|mfm|mfv` to choose SPH or the restored
[Hopkins MFM/MFV solver](docs/hopkins_hydro.md). The Hopkins implementation uses
double precision hydrodynamic state on the CPU and supports traditional individual
steps or global continuous steps, with FMM or PMMM gravity. It requires a Fixed64
position build. MFM accepts cooling and star formation; MFV requires both disabled.
`configurations/sound_wave.ini` and
`configurations/gresho.ini` provide meshless test problems.
Compile the meshless methods with `-DCOSMICTIGER_ENABLE_HOPKINS=ON`;
the default SPH build omits them.

The gas controls appear under `hydro.*` in `--help` and can also be
specified in the INI configuration file. The compatibility group `chemistry.*`
selects [GADGET-4 equilibrium atomic cooling](docs/primordial_chemistry.md), with
fixed H/He composition and no external UV background. The cosmological
configurations enable cooling; `configurations/star_formation/lcdm.ini` and
`configurations/lcdm_present.ini` additionally enable star formation and require
`COSMICTIGER_ENABLE_STARS=ON` as well as SPH and cooling.
The variational pressure operator supports both cooling and star formation.
The `starFormation.*` controls select GADGET-4's algebraic multiphase target,
bulk-energy relaxation, and stochastic whole-particle conversion. Converted
stars retain gravitational interactions. See [star formation](docs/star_formation.md)
for source ordering, thresholds and configuration. The former reaction networks,
manual species fractions, Anninos controls and wind options are not accepted.
Diffusion and conduction remain unsupported. Their former options, the obsolete `hydro.chemistry` switch,
`hydro.yReflect`, and `gravity.sphSmoothing` are no longer accepted, including
false entries in older INI files. Gas smoothing lengths remain adaptive;
FMM gravity uses one fixed comoving radius for gas and dark matter. With SPH on,
the FOF/ROCKSTAR halo finder and the lightcone halo catalogue link dark matter
only, each particle counted at the total-matter mass of its lattice site (the
ROCKSTAR `RESCALE_PARTICLE_MASSES` convention); gas and stars enter the power
spectrum and the lightcone HEALPix maps weighted by their mass. Option metadata
records `analysis.haloParticles=darkMatter`; the catalogue `.format` sidecars
record `halo_particles dark_matter` and `particle_mass total_matter_rescaled`.
Fixed-time SPH scenarios (Sod, blast, discs, waves) have no dark matter and
reject halo groups and lightcones; the dark-matter-only Gadget/Plummer/glass
loaders still reject SPH runs.

Gas state, including fields used by staggered timesteps, follows particles during
domain redistribution and is retained in versioned checkpoints. Restart using
the same scenario, physics, and particle-layout options. Checkpoints predating
first-class scenarios lack scenario metadata and require their original executable
to resume. New SPH
checkpoints store and validate the layout, kernel selection, viscosity, smoothing,
timestep, and external-force parameters.

### Checkpoint and restart

Set `checkpoint.intervalSeconds` to a wall-clock interval in seconds. The default
is `1000000000`, so periodic checkpointing is effectively disabled unless the
interval is changed. CosmicTiger writes a checkpoint at a completed iteration
boundary when the interval expires, and writes one more at normal exit. A
`checkpoint.txt` file containing one redshift per line requests checkpoints
when those redshifts are crossed.

Each checkpoint resides under `output.directory` in a numbered `checkpoint.N`
directory, where `N` is the iteration. It contains one `checkpoint.R.dat` file per HPX locality and a
versioned `checkpoint.meta` manifest with the locality count, file sizes, and
checksums. A checksummed `scenario.meta` records the scenario, requested duration,
and gravity selection. Files are written in `checkpoint.N.incomplete` and the directory is
renamed to `checkpoint.N` only after all localities finish and the manifest is
durable. An interrupted write leaves the incomplete directory for inspection;
it is not a restart candidate. Existing numbered checkpoints are never
overwritten. Redshift requests may also create a `checkpoint.z.Z` symlink for
convenience; the numbered directory is the restart source.

Restart with `--checkpoint.readIndex=N` and the same `output.directory`, using
the same scenario, duration, number of HPX localities, and compatible physics, particle-layout,
units, and timestep settings. A new checkpoint is validated against its
manifest before any locality loads state. Older payload decoding remains
available to developer regressions; physical scenarios require the new metadata.
Rank payloads still use native
binary structures, so moving checkpoints between different architectures or
compiler ABIs is not guaranteed.

## Terminal output

Normal runs show the CosmicTiger banner, a short configuration summary, and an
aligned progress table. The table repeats its column labels every 25 rows and
starts with a fresh header after restart. Fixed-duration scenarios report code
time, timestep, completion percentage, minimum/maximum rung, total wall time,
and time spent on the current step. Cosmological runs show redshift and scale
factor alongside timestep and timing information. Warnings and errors remain
visible. Plain-text output also stays readable when redirected to a log.

Both progress tables include `TFLOP/s` without enabling verbose output. This is
an estimate of instrumented gravity, SPH, Hopkins and tree arithmetic divided by the wall time
of force advancement, summed across CPU workers, CUDA kernels and localities.
SPH includes neighbor geometry, each density/support-radius iteration, pressure
and viscosity pairs, timestep/rung calculations, and predictor/corrector kicks.
PMMM includes its spherical operators, spectral contraction, and cached FFTW
execution counts. Chemistry, other FFT paths and other uninstrumented work are not counted;
when they run inside force advancement their elapsed time is included. The
tables also separate `FP32 TF/s`, `FP64 TF/s`, and `Mixed TF/s`. Hopkins uses
FP64; `Mixed` retains legacy counts whose arithmetic precision has not yet been
split, rather than attributing them to the wrong precision. Cosmological
`progress.txt` preserves its existing fields and appends `Tflops32`, `Tflops64`,
and `TflopsMixed`.

Counts use 64-bit integers until the final rate conversion. Arithmetic and
math-function results count as one operation, FMA as two. A conversion between
floating precisions counts once at the higher precision (float-to-double and
double-to-float both count as FP64). Integer operations, comparisons, selections
and sign changes (including absolute values) are excluded.
Force and potential kernels count; diagnostic energy/momentum bookkeeping,
diagnostic reductions and diagnostic-only signal speeds do not. SIMD padding
and numerical CUDA reductions count when executed. These are source-operation estimates, not
hardware instruction counts: compiler folding/fusion and math-library
implementations can produce different instruction totals. Fixed kernel costs
are batched, CPU totals are sharded by worker, and each CUDA tree walk submits
its accumulated count once per warp. Reset and reporting happen after work
completion; reporting uses a monotonic clock and excludes counter collection
latency from the measured interval.

See [the function-by-function FLOP audit](docs/flop_accounting.md) for the counted
paths, diagnostic exclusions and regression coverage.

Startup always prints the complete resolved options. Set `--run.verbose=true`
to print detailed runtime diagnostics, including tree, gravity, SPH, and CUDA setup information.
The default is `false`. The equivalent INI setting is:

```ini
[run]
verbose = true
```

The tiger and sky artwork is copied exactly from
[`dmarce1/experimental` at `7ba8f1829fdede7348a1418b7cb36c0e7a955302`](https://github.com/dmarce1/experimental/blob/7ba8f1829fdede7348a1418b7cb36c0e7a955302/cosmictiger/startup.hpp).
Its progress display also supplies the aligned-column and repeating-header
pattern used here. See the [output-file inventory](docs/output_files.txt) for
runtime diagnostics, analysis products, and tool outputs other than Silo and
checkpoints, including their names and the conditions that create them.

## Phase profiling

`COSMICTIGER_PROFILE=ON` is the CMake default for every build type and compiles
the manual phase profiler; configure with `-DCOSMICTIGER_PROFILE=OFF` to turn
every call into an empty inline. Coordinator code brackets phases with
`profile::profileBegin("name")` and `profile::profileEnd()` or the exception-safe
`profile::ScopedPhase` from `cosmictiger/core/profiler.hpp`: initialization,
transfer functions, domains, rebound, rung sort, tree creation, kicks, mesh
solves, hydro, chemistry, star formation, drift, checkpoints, output, lightcone,
groups, power spectra, and gravity-error passes. Phases nest on the coordinator's
logical control flow; entering a child pauses its parent, so each name collects
exclusive wall time and the rows sum to the instrumented total. These
coordinator scopes exclude individual kernels, per-particle loops, and work
on concurrent HPX tasks: the single phase stack belongs to the coordinator.

The report uses dotted names for finer stages. SPH separates tree creation,
neighbor boxes/lists, density/support solves, viscosity targets, closing/opening
forces, predictor/corrector updates, rung limiting, sources, and cleanup.
Hopkins MFM/MFV also separates gather/scatter, geometry, gradients, pair-flux
solves/integration, and timestep selection. Gravity separates tree allocation
and construction, the FMM root walk/background/result wait, PMMM
preparation/deposition/transforms/evaluation, and cleanup. Drift separates
CPU/CUDA execution, memory advice, and remaining remote waits.
Domain migration, cosmological initialization, checkpoint I/O, particle output,
group finding, lightcone processing, and power spectra have their own subphases.
Only stages actually executed appear. A parent row (for example `hydro`) now
contains its remaining overhead, excluding all its named children; its share
is not the total share of that subsystem. In PMMM, the near-field pipeline can
overlap the transform: `transform` includes that overlapping wall interval,
and `finishNearPipeline` measures only the remaining wait.

Normal driver exit and a clean first Ctrl-C print the table (phase, calls,
total, mean, share) to the terminal and save it as `profile.txt` in the
simulation output directory. Only rank 0 writes it; worker localities keep no
phase profile. The report
measures the coordinator's wall time, including remote waits.

Gravity also appends a detailed breakdown to the terminal and the same
`profile.txt`. The host table separates CPU setup (including L2L), periodic
work, tree acceptance and interaction-list processing (`kick.cpu.traversal`),
M2L, P2L, M2P, P2P, task scheduling, and leaf expansion evaluation/velocity/rung updates
(`kick.cpu.leafKick`). It also measures GPU work-packet queueing, source
grouping/collection, remapping, sorting/fetch completion, memory advice,
launch preparation, submission/completion, cleanup, and child/result waits.
Calls count timed sections, including interaction routines given empty lists.
Mean durations in these detail tables are in microseconds so short operations
remain visible.

These are accumulated elapsed intervals from rank 0's local tasks, including
any suspension or waiting, not CPU execution time or an all-rank reduction.
Timers stop before recursive descent and GPU execution handoff. Concurrent
tasks and nested waits can overlap; detail rows have no percentage share or
grand total and must not be added to the coordinator total. The independent
`profile::ScopedWork` counters do not use the coordinator's phase stack or a
system-thread-local stack. They also cover force-only uses of the shared FMM
walk, such as gravity error checks and PMMM near interactions when enabled.

For CUDA, a separate device table uses stream events to measure the explicit
launch-packet host-to-device copy, fused gravity kernel, and result
device-to-host copy. Events are read after the existing stream completion
wait; profiling adds no extra synchronization. Kernel time includes managed
memory stalls, and the copy rows do not account for all managed-memory
migration. `COSMICTIGER_PROFILE=OFF` removes the host detail timers and CUDA
timing events. Use APEX and TAU below for broader HPX task attribution.

Configure a CUDA build with `-DCOSMICTIGER_PROFILE=ON` and
`-DCOSMICTIGER_DEVICE_PROFILE=ON` to also time sections inside the fused gravity
kernel. Device profiling is off by default and requires CUDA and the host
profiler. Its `kick.device.*` rows match the CPU sections: `prepare` (including
L2L and force-tile setup), `periodic`, `traversal` (acceptance tests and
interaction lists), `M2L`, `P2L`, `M2P`, `P2P`, and `leafKick` (L2P and particle
updates). Additional rows separate `schedule` (work-queue and walk-stack
operations), `copyWait` (remaining checklist-copy wait), and `cleanup`.
The asynchronous checklist copy still overlaps M2L. Calls count phase visits
across blocks, including empty interaction lists, rather than particle pairs;
CPU and CUDA visit counts can differ because the CUDA walk processes tiles.

The device report uses `clock64()` elapsed cycles, accumulated independently
by each block, with calls, mean cycles, and shares of summed instrumented block
cycles. It also reports min/mean/max total cycles across recorded blocks and
launches. These include stalls, scheduling, and instrumentation overhead;
overlapping block cycles are not kernel wall time and are never converted
using a nominal GPU frequency. Keep the event-based kernel duration as the
elapsed-time measurement. Only rank 0 imports device records, after the
existing stream wait. The counter transfer is separate from the ordinary
result-copy event interval.

`profile::DeviceProfiler` is collective on exactly one complete 32-thread
warp per block, as used by the kick kernel. All lanes must reach its
constructor, `next`, and `finish` calls with identical phase choices and a
block-owned output record. Counters are distributed across warp lanes and
written once to disjoint global records, without profiling atomics or shared
counter arrays. `COSMICTIGER_DEVICE_PROFILE=OFF` compiles out its state, clock
reads, synchronization, and record transfers.

The first Ctrl-C requests a clean stop at the next safe phase or completed
substep. Active work finishes, the simulation's ordinary final checkpoint is
written when a completed state is available, and HPX/MPI shutdown finalizes the
APEX and TAU reports. Long initialization batches or force calculations must
finish before shutdown; the signal handler never cancels a worker or calls MPI.
A second Ctrl-C forces immediate termination and can leave reports or
checkpoints incomplete. A clean interrupted run exits successfully.

### HPX task profiles with APEX

The phase profile above only knows the coordinator's named phases; it does not
retain the identity, parent, or suspend/resume history of the HPX tasks that
run inside them.

Use an HPX installation built with `HPX_WITH_APEX=ON` for task instrumentation.
HPX then creates APEX tasks from the descriptions it already knows and starts,
yields, resumes, and stops their timers at scheduler boundaries. HPX owns APEX
initialization and finalization; CosmicTiger must not initialize it again.
Explicit `hpx::annotated_function` labels distinguish anonymous task entry
points. Use a stable string literal for each phase so repeated task instances
aggregate under the same name. This applies to `async`, `post`, continuations,
and `run_as_os_thread` callables. Parallel sort, copy, and fill use
`annotatedParallelPolicy("phase.name")` to name their generated worker tasks.
Registered actions already supply their action names. HPX's FULL description
fallback uses the object address for unnamed lambdas and functors; those
addresses can appear as `UNRESOLVED ADDR` even with debug symbols. When HPX was
built without `HPX_WITH_THREAD_DESCRIPTION_FULL`, unannotated local callables
can inherit a parent name instead of getting distinct function names.
Full descriptions allow ordinary function pointers to be resolved. Anonymous
callables still need explicit labels when HPX supplies a functor object's address.

`COSMICTIGER_REQUIRE_APEX=ON` rejects an HPX installation without that support.
It defaults to ON for RelWithDebInfo, Debug, and Profile, and OFF otherwise.
An existing CMake cache retains its explicitly selected value. Set it to OFF
to deliberately allow an HPX installation without APEX; this does not disable
APEX in an installation that already has it. Configuration always reports the
selected HPX's actual instrumentation status. `build.sh` requires APEX for its
non-Release modes and selects the RelWithDebInfo HPX prefix for Profile.
Setting `HPX_WITH_APEX=ON` in the application cannot retrofit an installed HPX.

For example, select a compatible APEX-enabled HPX installation (built against
the same MPI and CUDA toolkit as CosmicTiger):

```sh
cmake -S . -B build/apex -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=/path/to/apex-enabled-hpx \
  -DCOSMICTIGER_REQUIRE_APEX=ON
cmake --build build/apex --target cosmictiger --parallel 4
./build/apex/cosmictiger --configuration=configurations/lcdm_present.ini --hpx:threads=4
```

With an APEX-enabled HPX, CosmicTiger enables `apex_profiles.csv` and
`apex_tasktree.csv` by default, independently of `COSMICTIGER_PROFILE`.
The detailed screen report is off by default, including through
`scripts/profile_tau.sh`. Set `APEX_SCREEN_OUTPUT=1` to request it explicitly, or
`APEX_CSV_OUTPUT=0` and `APEX_TASKTREE_OUTPUT=0` to disable individual file reports.
Existing environment settings are preserved. `APEX_OUTPUT_FILE_PATH` selects their directory;
otherwise APEX writes to the working directory. APEX 1.11 gives an explicitly
requested `APEX_TASKGRAPH_OUTPUT=1` precedence over task-tree output.

APEX's task summary, CSV, and task tree complement `profile.txt`. Task timers
measure execution intervals between scheduler events; they are not the phase
profile's exclusive coordinator wall times.
`APEX_TRACE_EVENT=1` additionally requests a task timeline viewable in Perfetto.
APEX task instrumentation alone does not establish CUDA kernel profiling;
device activity requires a compatible GPU instrumentation backend.

### Combining task context and ordinary function samples

Use the TAU launcher to collect ordinary function samples under the active HPX
task name. APEX forwards task start/resume and yield/stop events to TAU. HPX
1.11 requires the TAU shutdown fix described below; the local Debug, Release,
and RelWithDebInfo HPX installations already include it. Build the pinned TAU
runtime once, then launch the application:

```sh
./scripts/build_tau.sh
./scripts/profile_tau.sh ./build/apex/cosmictiger \
  --configuration=configurations/lcdm_present.ini --hpx:threads=4
```

The dependency script installs TAU 2.35.2 with MPI, pthread, BFD symbol
resolution, and libunwind into `~/local/tau`. It verifies archive checksums and
limits build parallelism. `TAU_PREFIX`, `TAU_BUILD_ROOT`, and `TAU_BUILD_JOBS`
override those choices. The launcher finds `tau_exec` through `TAU_EXEC`, PATH,
or `TAU_ROOT` (default `~/local/tau`). It keeps the application's working
directory and arguments unchanged. Prefer RelWithDebInfo for debug symbols;
optimized/inlined code may not retain a separate sample entry for every source
function.

Each launch creates `profiles/tau-<timestamp>-<pid>/`, with TAU's combined
`profile.<rank>.<context>.<thread>` files in `tau/` and APEX reports in `apex/`.
Set `COSMICTIGER_PROFILE_DIR` to choose a stable output root. `PROFILEDIR` and
`APEX_OUTPUT_FILE_PATH` can override the individual directories. For MPI, pass
the same output root to every rank and use the Ctrl-C-aware MPI wrapper shown
below.

Open the TAU directory in ParaProf, or run `~/local/tau/x86_64/bin/pprof -a`
from that directory. Contextual entries connect the task timer to a
`[CONTEXT]` node (`[INTERMEDIATE]` in older releases) and then a `[SAMPLE]`
function. These are the combined task/function profiles; `profile.txt` remains
the separate coordinator phase report.

The launcher enables `APEX_TAU=1`, TAU profiling and sampling, and disables
TAU tracing. Defaults are function resolution, a 10 ms sampling period,
callpaths enabled, and stack unwinding and timer throttling disabled. The
`TAU_EBS_*`, `TAU_CALLPATH`, `TAU_THROTTLE`, and `APEX_THROTTLE_TIMERS`
environment settings override these defaults. No stack unwinding is needed
for the function-to-task association. `APEX_PROFILE_OUTPUT=1` alone only
writes APEX timers in TAU format; it does not collect ordinary function samples.
The local dependency scripts include compatibility fixes for TAU's Linux
PIE/shared-library address resolution, APEX 1.11's HPX shutdown ordering, and
thread-safe initialization of APEX's task-tree root.
The dependency fixes are preserved in `cmake/patches/`. For HPX 1.11, apply
`hpx_full_thread_descriptions.patch` from the HPX source root. Apply both
`hpx_apex_compatibility.patch` and `hpx_apex_tau.patch` from its APEX source root
(normally `<hpx-build>/_deps/apex-src` after configuration). Each patch records
its source revision. Check with `git apply --check /absolute/path/to/patch`
before `git apply /absolute/path/to/patch`. Rebuild and reinstall HPX with
`HPX_WITH_APEX=ON` and `HPX_WITH_THREAD_DESCRIPTION_FULL=ON`; rebuilding only
APEX does not install the HPX header fixes. The local `~/scripts/build-hpx.sh`
applies these fixes idempotently.
APEX's synthetic root is omitted from TAU, which already supplies an application
root. TAU's wrappers finalize after HPX workers exit, avoiding cross-thread
timer closure during HPX teardown; still-active shutdown timers can therefore
include teardown time. APEX's own root report and actual HPX task events are
retained. MPI task-tree merging uses the reporting thread without creating
an additional worker pool. The launcher
requires active APEX instrumentation and leaves optional OS-thread lifetime
timers disabled (`APEX_TIME_TOP_LEVEL_OS_THREADS=0`).
CosmicTiger finalizes APEX reports on HPX's native I/O pool in an early
pre-shutdown callback. This keeps HPX workers available to deliver every
rank's shutdown actions, including runs with one worker per rank.
See the [APEX TAU listener](https://github.com/STEllAR-GROUP/apex/blob/hpx_v1.11.0/src/apex/tau_listener.cpp)
and [TAU sampling options](https://www.cs.uoregon.edu/research/tau/docs/html-docs/latest/newguide/environment.html).

The listener forwards active task names, not the complete asynchronous parent
graph; keep APEX's task tree for that information. TAU timer call counts include
resumptions, whereas APEX reports completed tasks and yields separately. Linux
TAU timer sampling uses a wall-clock timer on every worker; the phase profile
measures the coordinator alone. Their totals are not directly interchangeable.

Ctrl-C uses the same clean HPX/MPI shutdown as a completed run. For MPI, place
the supplied wrapper outside the launcher so terminal Ctrl-C requests shutdown
instead of invoking Open MPI's job-abort behavior:

```sh
COSMICTIGER_PROFILE_DIR="$PWD/profiles/tau_mpi" \
  ./scripts/run_mpi.py -n 2 ./scripts/profile_tau.sh ./build/apex/cosmictiger \
    --configuration=configurations/lcdm_present.ini --hpx:threads=4
```

The wrapper queues an early interrupt until handlers are ready and forwards a
cooperative SIGUSR1 request. A second Ctrl-C aborts the job. It can also launch
the application directly without TAU. This wrapper requires Linux, Open MPI,
and a working directory visible at the same path on every host. Direct SIGINT or SIGUSR1 to an application
rank requests a clean stop across the localities. See
[Open MPI signal handling](https://docs.open-mpi.org/en/v5.0.10/man-openmpi/man1/mpirun.1.html#signal-handling).

The `tau_profiler_default` CTest is registered when `tau_exec` is found (or set
`COSMICTIGER_TAU_EXEC` at CMake configuration). It requires sampled ordinary
functions to be connected to their distinct task contexts, rejects crossed
parent/child attribution, and observes an explicit worker migration. The APEX
tests also cover default reporting and explicit report opt-outs.
`tau_profiler_config` checks configuration-file precedence, and
`tau_profiler_mpi_startup` checks a two-rank application startup and shutdown
with separate rank profiles and a bounded task-tree merge. The interrupt
regressions check first-interrupt finalization, second-interrupt emergency
termination, and the completed APEX, TAU, and `profile.txt` reports.

## Source layout

Common gravity and SPH machinery lives directly in each subsystem directory.
The same subsystem organization is used beneath `cosmictiger/` for headers and `src/` for
implementations. Gravity solver code is in `gravity/dehnen/` for global adaptive
FMM and `gravity/nitadori/` for PMMM. SPH numerical modules are in
`hydro/sph/variational/`, `hydro/sph/monaghan/`, `hydro/sph/morris_and_monaghan/`,
`hydro/sph/cullen_and_dehnen/`, `hydro/sph/price_etal/`, and `hydro/sph/balsara/`.
Hopkins MFM/MFV headers live in `cosmictiger/hydro/hopkins/`;
`src/hydro/hopkins.cpp` implements their hydrodynamic stage within the shared
simulation driver.
Owned particle storage stays in `particles/`; PMMM owns its
separate ghost arena in `gravity/nitadori/`.

See [the source organization guide](docs/source_layout.md) for file responsibilities,
dependency boundaries, and generated-code placement.

## Build

Use CMake 3.24 or newer, C++20, HPX with CUDA and networking enabled, CUDA,
MPI, FFTW3 with single-precision serial and MPI libraries (`fftw3f` and
`fftw3f_mpi`), GSL, Silo, HEALPix C and C++, Boost program options and Boost.Units
headers, plus libbacktrace for stack-trace symbolization. CMake searches installed
libbacktrace packages and the selected compiler's implicit include/library paths
(including GCC's bundled copy). For another prefix, set `CMAKE_PREFIX_PATH` or
`Libbacktrace_INCLUDE_DIR` and `Libbacktrace_LIBRARY`.
Install the development packages for these dependencies; on
Debian/Ubuntu, FFTW3, GSL and HEALPix are provided by `libfftw3-dev`,
`libfftw3-mpi-dev`, `libgsl-dev`, `libchealpix-dev`, and `libhealpix-cxx-dev`,
alongside the development package for the selected MPI implementation. CMake
finds system installations automatically and verifies that the single-precision
FFTW MPI API links with the selected MPI library. The build links the installed
HEALPix C++ library.

Fatal signals (`SIGABRT`, `SIGSEGV`, `SIGBUS`, `SIGFPE`, and `SIGILL`) report the
host, process ID, HPX locality, and faulting thread's stack to stderr and append
to `stack_trace.txt` in the configured output directory. Before options are
installed, the log defaults to the working directory. The handler is installed
on every HPX locality at runtime startup and preserves termination by the
original signal, including core dumps when enabled. Use `RelWithDebInfo` or
`Debug` for source-file and line information; symbolization after a fatal fault
is best effort. This handler is adapted from OctoTiger's `stacktrace_fix` commit
`a0720ee02feebf776b797b2fdc74edf6bcc5bd83`.

Run `./build.sh` in a terminal for the interactive builder. Choose Release,
RelWithDebInfo, Debug, or Profile (host gprof instrumentation), CUDA, SPH, Hopkins,
equilibrium cooling, job counts, HPX prefix, CUDA architectures, additional
dependency prefixes, enabled gravity and near methods, potential expansion degrees
(a scalar or inclusive range),
enabled time integrators, and fixed32 or fixed64 position storage.
Builds use `build/<mode>_<cuda|cpu>` (for example `build/release_cuda`), with live
output and a `builder.*.log`. The default builds `cosmictiger` with FMM, SPH,
equilibrium cooling, traditional integration, potential degrees 4–6, and fixed32
positions. Building and running the test suite is opt-in. Build concurrency
defaults to the detected system thread count, with no upper cap. Use `./build.sh -j 4` to select a count,
or `./build.sh -j` (also `-j max`) to select the detected thread count.
Builds show full compiler and linker commands using CMake's `--verbose` option
(Make's `VERBOSE=1`). Test concurrency accounts for two MPI localities with two
HPX threads each.

The builder uses `~/local/hpx/<build type>` when installed (RelWithDebInfo for Profile),
or normal CMake package discovery. Additional dependency prefixes default to the
system search paths; use the menu or `CMAKE_PREFIX_PATH` only for installations
outside those paths. The builder refreshes cached GSL and HEALPix locations when
configuring, so old temporary installations do not override newly installed
system packages. It reuses CUDA architectures from this checkout's CMake build
caches, with a menu override or `CUDAARCHS` for explicit selection. Without a
cached or explicit CUDA architecture, it uses `native`. For a manual build:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/hpx -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/release --parallel 4
```

Select the CUDA toolkit explicitly when several versions are installed. HPX and
CosmicTiger should use the same toolkit major version and host compiler. HPX,
FFTW3-MPI and CosmicTiger must use the same MPI installation; rebuild HPX and
FFTW3-MPI after changing it. The MPI implementation must provide
`MPI_THREAD_MULTIPLE` so application MPI calls can overlap HPX parcelport
communication. CosmicTiger requests this level before starting HPX, including
single-process launches, and exits with an error if any rank supplies less.
A CUDA-enabled HPX also checks its CUDA compiler when configuring
CosmicTiger with CUDA disabled; a separate CPU-only HPX installation is optional.

Before MPI startup, CosmicTiger defaults `HWLOC_COMPONENTS` to `-gl` to avoid
hwloc's OpenGL discovery blocking on an X server, even for `--help`. CPU topology
discovery and CUDA execution remain available. An explicitly set
`HWLOC_COMPONENTS` is preserved. If an MPI launcher itself hangs before starting
CosmicTiger, apply the setting to the launcher as well, for example
`HWLOC_COMPONENTS=-gl mpirun -n 2 ./cosmictiger --help --hpx:threads=2`.
See hwloc's [component selection documentation](https://www.open-mpi.org/projects/hwloc/doc/v2.13.0/envvar.html).

On Ubuntu 26.04 with GCC 15 and glibc 2.43, CUDA 12.8 fails its compiler check or
conflicts with the C library's math declarations. Ubuntu's CUDA 13.1 packages,
including `cuda-crt-13-1` version `13.1.115-0ubuntu1.1` or newer, contain the
matching declarations. Rebuild HPX with that toolkit, then use a fresh build:

```sh
cmake -S . -B build/cuda13 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-13.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/path/to/cuda-13.1 \
  -DHPX_DIR=/path/to/matching/hpx/lib/cmake/HPX \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda13 --parallel 12
ctest --test-dir build/cuda13 --output-on-failure --parallel 2
```

Initial power and recombination histories are computed internally. No external
CAMB power table or RECFAST executable is required. Particle Silo output writes
separate gas and dark-matter point meshes, with gas thermodynamic fields.

## Regression checks

```sh
ctest --test-dir build/release --output-on-failure --parallel 4
```

C++ and CUDA correctness checks use GoogleTest, with each named case registered
with CTest. CMake uses an installed GoogleTest package or fetches a pinned source
revision when `BUILD_TESTING=ON`. For an offline build, pass
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest` at configure time.
Application checks selected by `--run.test` also report through GoogleTest in a
testing build. Python integration runners stay registered directly with CTest;
reference generators, plotting scripts, and benchmarks remain support tools.

Application executables remain in the build root. Static libraries go in `lib/`;
standalone test programs and support libraries go in `tests/bin/`, CTest discovery
files in `tests/cmake/`, and default test working files in `tests/work/`.
`BUILD_TESTING=OFF` omits the test targets. Halo tools are built only when at least
one gravity solver is enabled.

List cases with `ctest --test-dir build/release -N`, or select a family using its
label, for example `ctest --test-dir build/release -L '^monaghan_sph$'`.
Standalone executables also accept GoogleTest options such as
`--gtest_filter=MonaghanSph.*` and `--gtest_list_tests`. GPU checks report a skip
when no CUDA device or driver is available.

The scalar SPH formulas, grouped options, checkpoint metadata, tracer selection,
numerical utility, and compression regressions run on the CPU without initializing
a GPU:

```sh
ctest --test-dir build/release --output-on-failure -R '^(monaghan_sph\.|grouped_options$|checkpoint_state_dm$|particle_selection\.|numerical_utilities\.|compression\.)'
```

The numerical utility and compression targets still require CUDA headers. The
build also compiles a device-side range regression without launching a kernel.

The direct `monaghan_sph_cuda` test exercises production density and force kernels,
smoothing bounds, periodic support geometry, pair conservation, and cosmological
thermal work. The integration checks cover mixed-mass CPU/CUDA force evaluation, the gas gravity handoff,
synchronized diagnostics, gas reordering, checkpoint payloads and configuration
compatibility, a coarse Sod shock tube, dark-matter startup, and a short
cosmological SPH run with restart. Simulation runs in the CUDA-enabled executable
initialize the GPU even when selecting CPU gravity. The full suite therefore
requires an available CUDA device; help and option validation do not. These are correctness and integration checks, not a
resolution-convergence study or a performance benchmark.
