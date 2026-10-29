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
| `compute` | CUDA and tree bucket size |
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
`Output output`, and likewise `Run`, `Compute`, `Checkpoint`, `Initial`, `Analysis`,
`Lightcone`, `Cosmology`, `Units`, and `Hydro`. Code accesses members directly, for
example `getOptions().gravity.enabled` or `options.output.directory`.
Radius aliases resolve to canonical support radii in box units; unit conversions
are stored as `units.massG`, `units.lengthCm`, and `units.timeSeconds`.
Curvature `cosmology.omegaK` is derived from the density budget and is no longer an
accepted input, since a supplied value was always overwritten.

The old ungrouped and snake_case option names have been replaced. For example,
`--parts_dim=16 --max_iter=2 --cuda=true` is now
`--initial.particlesPerDimension=16 --run.maxIterations=2 --compute.cuda=true`,
and `--hydro.sph=true` is now `--hydro.enabled=true`.

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

Each build directory uses one position representation for `cosmictiger` and all
named variants; target and executable names stay the same for either width.
The runtime `compute.positionType` defaults to that compiled representation.
An explicit CLI or INI value must match it; selecting the other width requires
reconfiguring and rebuilding. Use separate build directories to keep both widths.
Checkpoints record the representation and must be restarted with the same width.
The supplied INIs leave `positionType` commented so they use the compiled width.
The legacy `glass.bin` exchange format remains Q0.32 in both modes; checkpoints
retain the selected position precision.

One CMake configuration provides named targets for every supported combination
of time integrator, gravity solver, SPH support, and expansion order. Configure
once with Unix Makefiles, then build any desired target:

```sh
cmake -S . -B build/release_cuda -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make -C build/release_cuda -j4 cosmictiger_leapfrog_pmmm_pp_7
make -C build/release_cuda -j4 cosmictiger_leapfrog_fmm_sph_7
make -C build/release_cuda -j4 cosmictiger_leapfrog_pmmm_pp_sph_7
make -C build/release_cuda -j4 cosmictiger_hierarchical_pmmm_fmm_8
```

Targets have the form `cosmictiger_<integrator>_<gravity>[_sph]_<order>`:

| Component | Supported values |
| --- | --- |
| Integrator | `leapfrog` (ordinary block-timestep KDK), `hierarchical` |
| Gravity | `fmm`, `pmmm_pp`, `pmmm_fmm` |
| SPH | Omit `_sph` for collisionless targets; include `_sph` with leapfrog and any gravity method |
| Order | `3` through `9` (multipole degree plus two) |

There are 42 collisionless variants (2 integrators × 3 gravity methods × 7
orders) and 21 SPH variants (leapfrog × 3 gravity methods × 7 orders), giving
63 physics variants and 63 named executables per build directory.
Run `make cosmictiger_help` from a configured build
directory, or `make -C build/release_cuda cosmictiger_help`, to see those families,
selection examples, the current default, CPU/CUDA mode, and position storage
without compiling. Standard `make help` lists available targets, including
`cosmictiger_help`.

Each requested target reuses the shared core library.
Configuration-independent sources compile once per build directory;
variant sources have their own target-specific definitions and source lists.
Global FMM variants exclude PMMM sources, PMMM variants include only their
selected near-field implementation, and collisionless variants exclude SPH sources.
Other named variants are excluded from the default `all` build, so plain `make`
builds only the configured default executable and enabled tests.

`./build.sh` offers the same choices interactively and builds the selected named
target in `build/<mode>_<cuda|cpu>`, such as `build/release_cuda`. Changing physics
choices reuses that directory and keeps previously built variants available.
The initial choices are leapfrog, FMM, SPH, order 7, and fixed32 positions.
Position storage is a menu choice applied to the whole build directory; changing
it rebuilds the requested target under the same name. Selecting PMMM changes
the menu's default order to 8 unless an order was explicitly selected; disable
SPH when selecting hierarchical integration. For CPU-only builds, use
`-DCOSMICTIGER_WITH_CUDA=OFF` in a separate directory such as `build/release_cpu`.

The existing `COSMICTIGER_GRAVITY_METHOD`, `COSMICTIGER_PMMM_NEAR_METHOD`,
`COSMICTIGER_WITH_ORDER`, `COSMICTIGER_TIME_INTEGRATOR`, and
`COSMICTIGER_WITH_SPH` CMake options select the default `cosmictiger` executable and
its tests; they do not limit which named variants are available. The builder
sets these options to match the selected target. See
[CMake options](CMAKE_OPTIONS.md) for details and all build settings.

The former `gravity.method`, `gravity.pmmmNearMethod`, `gravity.fmmOrder`, and
`gravity.pmmmOrder` CLI/INI settings are no longer accepted. `--help` and the
`build.*` startup fields identify the compiled variant. SPH is available with
leapfrog and all three gravity methods. The hierarchical driver lacks the SPH
integration stages, so its supported targets remain collisionless.

Run a PMMM executable with `--gravity.meshSize=32 --gravity.pmmmNearWidth=2`.
Select `--hydro.enabled=true` in an SPH build or `--hydro.enabled=false` for a
collisionless run. PMMM uses the spherical FMM's total-degree M2L truncation,
one FFTW3-MPI group containing all localities, cached FFT plans and Green spectra,
and the compiled direct-pair or adaptive Cartesian FMM near field. The compiled
order applies to both the PMMM expansion and near FMM. Ghost width remains a
runtime option: it defaults to two, accepts one through four, and fixes near FMM
theta to `sqrt(3)/(pmmmNearWidth+1)`.
Global FMM cosmology runs can select redshift/theta pairs with
`--gravity.openingSchedule='49:0.41,20:0.51,2:0.61'`. Expansion applies the
listed changes; contraction reverses them at the same boundaries, restoring
the previous theta. See the [opening schedule rules](docs/pmmm.md#global-fmm-opening-schedules).
PMMM uses the compiled time integrator;
`compute.cuda` selects CPU or CUDA near-field execution and drift. See [PMMM](docs/pmmm.md)
for order conventions, independent particle and FFT ownership, memory costs, CPU/CUDA
near fields, and restarts.

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

The gas solver uses Monaghan's 1992 density/internal-energy SPH: the cubic spline,
symmetric pressure forces and thermal work, and constant alpha/beta artificial
viscosity. It is integrated with FMM or PMMM gravity and particle storage. See
[the formulation and paper references](docs/monaghan_sph.md) for the equations,
cosmological conversions, and deliberate implementation choices. SPH support is
built by default; `-DCOSMICTIGER_WITH_SPH=OFF` excludes it. Enable gas
explicitly with `--hydro.enabled=true` in a `leapfrog_<gravity>_sph` build. SPH
uses the ordinary block-timestep gravity/SPH path with global FMM or either PMMM
near method. Hierarchical hydro integration is not implemented. Hydrodynamics without
gravity remains supported.
Select SIMD CPU SPH with `--compute.cuda=false` or CUDA SPH with `--compute.cuda=true`.
Particle drift also follows `compute.cuda`: the CPU path uses SIMD position updates
and HPX workers, including rung filtering, glass damping, and lightcone crossings.
CPU runs can also use a build configured with `COSMICTIGER_WITH_CUDA=OFF`.

Cosmological gas runs initialize a displaced gas lattice alongside the dark
matter lattice. `initial.particlesPerDimension` specifies the lattice size for each species. With
SPH enabled, the total particle count doubles, while the gas and dark-matter
masses are weighted by `omega_b / omega_m` and `omega_c / omega_m` respectively.
The power-spectrum deposition uses these masses too.

For a small cosmological run using the native Einstein–Boltzmann power spectrum:

```sh
./cosmictiger --hydro.enabled=true --compute.cuda=true --initial.particlesPerDimension=16 --run.maxIterations=2 --hpx:threads=4
```

The gas controls appear under `hydro.*` in `--help` and can also be
specified in the INI configuration file. Primordial chemistry is configured under
`chemistry.*` and enabled by the cosmology scenario INI. Stellar particles, star
formation, diffusion, and conduction are unsupported. Their former options,
the obsolete `hydro.chemistry` switch,
`hydro.yReflect`, and `gravity.sphSmoothing` are no longer accepted, including
false entries in older INI files. Gas smoothing lengths remain adaptive;
FMM gravity uses one fixed comoving radius for gas and dark matter. The legacy
equal-mass lightcone and halo finders, and dark-matter-only Gadget/Plummer/glass
loaders, reject SPH runs.

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
an estimate of instrumented gravity, SPH and tree arithmetic divided by the wall time
of force advancement, summed across CPU workers, CUDA kernels and localities.
SPH includes neighbor geometry, each density/support-radius iteration, pressure
and viscosity pairs, timestep/rung calculations, and predictor/corrector kicks.
PMMM includes its spherical operators, spectral contraction, and cached FFTW
execution counts. Chemistry, other FFT paths and other uninstrumented work are not counted;
when they run inside force advancement their elapsed time is included. The
cosmological `progress.txt` retains its existing `Tflops` field and layout.

Counts use 64-bit integers until the final rate conversion. Arithmetic and
math-function results count as one operation, FMA as two; integer operations,
comparisons, selections and sign changes (including absolute values) are excluded.
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

Set `--run.verbose=true` to print the complete resolved options and detailed
runtime diagnostics, including tree, gravity, SPH, and CUDA setup information.
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

## CPU profiling

Normal simulation exit and a clean first Ctrl-C print a compact **Major section
timings** table to the terminal and save the same table as `section_timings.txt`
in the simulation output directory. It lists call count, total wall seconds, and
mean wall seconds for the major sections that ran (initial conditions, domains,
tree build, gravity, hydrodynamics, drift, checkpoints, snapshots, and analysis).
These are inclusive coordinator timings; nested sections overlap and must not be
added together as a breakdown of total runtime. Detailed profiler reports stay
in files by default.

`COSMICTIGER_ENABLE_PROFILING=ON` is the CMake default. The internal profiler is
the statistical CPU sampler adapted from
[`dmarce1/experimental` at `7ba8f1829fdede7348a1418b7cb36c0e7a955302`](https://github.com/dmarce1/experimental/blob/7ba8f1829fdede7348a1418b7cb36c0e7a955302/src/profiler.cpp).
It requires Linux on x86-64 or AArch64. Configure with
`-DCOSMICTIGER_ENABLE_PROFILING=OFF` to disable it.

Sampling starts before HPX initialization. Each attached OS worker has a timer
that samples its interrupted instruction address every 10 ms of that worker's
CPU time. HPX startup and shutdown hooks attach and detach workers automatically,
so HPX tasks can suspend, wait for futures, and resume on other workers without
manual profiling markers. There is no separate sampling thread. Optional
`PROFILE()` calls register metadata once per function specialization; they do
not set the current function or delimit a timed scope. They remain valid in
C++20 `constexpr` functions and become no-ops with `PROFILE_DISABLE` or in CUDA
device compilation.

Reports resolve sampled instruction addresses to ELF function symbols, including
functions without `PROFILE()` calls. Rows contain samples, estimated exclusive
CPU seconds, and their percentage of delivered samples. Inlined work belongs to
the enclosing compiled function. Parallel CPU totals can exceed elapsed wall
time. This does not measure call counts, inclusive time, blocked waiting time,
or GPU execution; simulation progress tables still report elapsed wall time.
Keep executable and shared-library symbol tables for useful function names.

Normal process exit writes the detailed report to
`cosmictiger.profile.<pid>.txt` in the working directory. The driver does not
print reports after individual steps. The first Ctrl-C requests a clean stop at
the next safe phase or completed substep. Active work finishes, the simulation's
ordinary final checkpoint is written when a completed state is available, and
HPX/MPI shutdown finalizes the CPU, APEX, and TAU reports. The handler remains
available when CPU sampling is disabled. Long initialization batches or force
calculations must finish before shutdown; the signal handler never cancels a
worker or calls MPI. A second Ctrl-C forces immediate termination and can leave
reports or checkpoints incomplete. A clean interrupted run exits successfully.
Each process has its own file; reports are not reduced across MPI localities.
The report includes dropped samples, missed timer ticks, and setup failures so
sampling losses remain visible. The bounded signal-handler buffers are allocated
before sampling; symbol resolution and demangling happen while reporting.

### HPX task profiles with APEX

The CPU sampler sees ordinary functions executing inside HPX tasks, including
`async`, `post`, actions, and continuations. A context switch does not hide the
currently executing instruction. Its flat report does not retain the active
HPX task's identity, parent, or suspend/resume history.

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
./build/apex/cosmictiger --configuration=lcdm.ini --hpx:threads=4
```

With an APEX-enabled HPX, CosmicTiger enables `apex_profiles.csv` and
`apex_tasktree.csv` by default, even when the separate CPU sampler is disabled.
The detailed screen report is off by default, including through
`scripts/profile_tau.sh`. Set `APEX_SCREEN_OUTPUT=1` to request it explicitly, or
`APEX_CSV_OUTPUT=0` and `APEX_TASKTREE_OUTPUT=0` to disable individual file reports.
Existing environment settings are preserved. `APEX_OUTPUT_FILE_PATH` selects their directory;
otherwise APEX writes to the working directory. APEX 1.11 gives an explicitly
requested `APEX_TASKGRAPH_OUTPUT=1` precedence over task-tree output.

APEX's task summary, CSV, and task tree complement the existing
`cosmictiger.profile.<pid>.txt`. Task timers measure execution intervals between
scheduler events; they are not the sampler's exclusive CPU-time estimates.
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
  --configuration=lcdm.ini --hpx:threads=4
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
function. These are the combined task/function profiles; the old
`cosmictiger.profile.<pid>.txt` remains a separate flat CPU report.

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
TAU timer sampling uses a wall-clock timer; CosmicTiger's independent sampler
uses a thread CPU clock. Their totals are not directly interchangeable.

Ctrl-C uses the same clean HPX/MPI shutdown as a completed run. For MPI, place
the supplied wrapper outside the launcher so terminal Ctrl-C requests shutdown
instead of invoking Open MPI's job-abort behavior:

```sh
COSMICTIGER_PROFILE_DIR="$PWD/profiles/tau_mpi" \
  ./scripts/run_mpi.py -n 2 ./scripts/profile_tau.sh ./build/apex/cosmictiger \
    --configuration=lcdm.ini --hpx:threads=4
```

The wrapper queues an early interrupt until handlers are ready and forwards a
cooperative SIGUSR1 request. A second Ctrl-C aborts the job. It can also launch
the application directly without TAU. This wrapper requires Linux, Open MPI,
and a working directory visible at the same path on every host. Direct SIGINT or SIGUSR1 to an application
rank requests a clean stop across the localities. See
[Open MPI signal handling](https://docs.open-mpi.org/en/v5.0.10/man-openmpi/man1/mpirun.1.html#signal-handling).

The `tau_profiler_enabled` and
`tau_profiler_disabled` CTests are registered when `tau_exec` is found (or set
`COSMICTIGER_TAU_EXEC` at CMake configuration). They require sampled ordinary
functions to be connected to their distinct task contexts, reject crossed
parent/child attribution, observe an explicit worker migration, and check
coexistence with the independent CPU sampler. The APEX tests also cover default reporting and
explicit report opt-outs. `tau_profiler_config` checks configuration-file
precedence, and `tau_profiler_mpi_startup` checks a two-rank application startup
and shutdown with separate rank profiles and a bounded task-tree merge.
The interrupt regressions check first-interrupt finalization, second-interrupt
emergency termination, and APEX/TAU output with CPU sampling compiled out.

## Source layout

Common gravity and SPH machinery lives directly in each subsystem directory.
The same layout is used beneath `cosmictiger/` for headers and `src/` for
implementations. Gravity solver code is in `gravity/dehnen/` for global adaptive
FMM and `gravity/nitadori/` for PMMM. SPH numerical modules are in
`sph/monaghan/`, `sph/morris_and_monaghan/`, `sph/cullen_and_dehnen/`,
`sph/price_etal/`, and `sph/balsara/`.
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

Run `./build.sh` in a terminal for the interactive builder. It uses a full-screen
menu when `whiptail` is available, with a text menu as a fallback. Choose Release,
RelWithDebInfo, Debug, or Profile (host gprof instrumentation), CUDA, SPH, job counts,
HPX prefix, CUDA architectures, additional dependency prefixes, gravity variant
(FMM, PMMM-FMM, or PMMM-PP), one kernel order from 3 through 9, time integrator,
and fixed32 or fixed64 position storage.
Builds share one generated Makefile in `build/<mode>_<cuda|cpu>` (for example
`build/release_cuda`), with live output and a `builder.*.log`. The default builds
`cosmictiger_leapfrog_fmm_sph_7` with fixed32 positions; building and running
the test suite is opt-in. Build jobs accept
any positive count. Use `./build.sh -j 24` to select a count, or `./build.sh -j`
(also `-j max`) to use all detected CPUs. Default test concurrency accounts for
two MPI localities with two HPX threads each.

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
