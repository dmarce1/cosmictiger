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
| `gravity` | Gravity, softening, opening angle, and time-step parameter |
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
`configuration`, with active source defaults and commented radius alternatives.
The complete, independently editable problem files in [configurations/](configurations/README.md) contain the same
option set, with problem-specific values and inline help comments. Files are
loaded explicitly; there is no implicit loading of `default.ini`.

Gravity and SPH can select kernels independently. Both use an effective radius
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

Gas particles are written as Silo CSG spheres in the `gas` mesh. Each sphere's
radius is the stored SPH kernel support radius, also available as the
zone-centered `kernel_support_radius` field. Sphere centers and radii follow the
selected output units and frame. Dark matter uses a point mesh.
VisIt discretizes overlapping CSG zones for display, which can show seams where
spheres overlap.

Particle positions retain the fixed-point representation used by the solver and
checkpoints. Set `output.units=cgs` for CGS quantities or
`output.units=cosmological` for Mpc/h, solar masses/h, and km/s. Set
`output.frame=comoving` or `output.frame=lab` independently; lab positions and
densities are proper quantities. Converted velocities are peculiar velocities
in either frame and never include Hubble flow. `output.units=both` and
`output.frame=both` write all four converted views in one Silo snapshot. Physical
coordinate conversion uses double precision directly from fixed point.

The defaults, `output.units=code` and `output.frame=comoving`, preserve the native
output layout and evolution normalization. Code units require the comoving
selection. Bounds remain normalized comoving coordinates in every output mode.
See [units and frames](docs/units.md) for conversion formulas, view paths, and
the distinction between stored and physical velocity and internal energy.

The conversion utilities use the same grouping style:
`healpix2silo --input.file=view.dat --output.file=view.silo` and
`groups2halofunc --input.groupNumber=0 --analysis.boxSize=684 --analysis.numberOfBins=1000`.

## SPH

The gas solver uses Monaghan's 1992 density/internal-energy SPH: the cubic spline,
symmetric pressure forces and thermal work, and constant alpha/beta artificial
viscosity. It is integrated with the Cartesian FMM and particle storage. See
[the formulation and paper references](docs/monaghan_sph.md) for the equations,
cosmological conversions, and deliberate implementation choices. The normal
dark-matter-only mode keeps its hierarchical gravity integrator. Enable gas explicitly
with `--hydro.enabled=true`; gas runs use the ordinary block-timestep gravity/SPH path.
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
gravity uses one fixed comoving radius for gas and dark matter. The legacy
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

## Build

Use CMake 3.24 or newer, C++20, HPX with CUDA and networking enabled, CUDA,
serial FFTW3 (single precision), GSL, Silo, HEALPix C and C++, Boost program options
and Boost.Units headers. Install the development packages for these dependencies;
on Debian/Ubuntu, GSL and HEALPix are provided by `libgsl-dev`, `libchealpix-dev`,
and `libhealpix-cxx-dev`. CMake finds system installations automatically. The
build links the installed HEALPix C++ library instead of compiling the bundled
`healpix/` sources.

Run `./build.sh` in a terminal for the interactive builder. It uses a full-screen
menu when `whiptail` is available, with a text menu as a fallback. Choose Release,
RelWithDebInfo, Debug, or Profile (host gprof instrumentation), CUDA, job counts,
HPX prefix, CUDA architectures, and additional dependency prefixes. Builds go in
`build/<mode>` with live output and a `builder.*.log`. The default builds
`cosmictiger`; building and running the test suite is opt-in. Build jobs accept
any positive count. Use `./build.sh -j 24` to select a count, or `./build.sh -j`
(also `-j max`) to use all detected CPUs. Default test concurrency accounts for
two MPI localities with two HPX threads each.

The builder uses `~/local/hpx/<build type>` when installed (Release for Profile),
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

Initial power and recombination histories are computed internally. No external
CAMB power table or RECFAST executable is required. Particle Silo output writes
separate gas and dark-matter point meshes, with gas thermodynamic fields.

## Regression checks

```sh
ctest --test-dir build/release --output-on-failure
```

The scalar SPH formulas, grouped options, checkpoint metadata, tracer selection,
numerical utility, and compression regressions run on the CPU without initializing
a GPU:

```sh
ctest --test-dir build/release --output-on-failure -R '^(monaghan_sph$|grouped_options$|checkpoint_state_dm$|particle_selection|numerical_utilities|compression_)'
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
