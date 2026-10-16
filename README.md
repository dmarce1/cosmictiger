# CosmicTiger - A Cosmological N-Body Code

## Command-line options

Application options use `groupName.optionName` with camelCase names, except for
the top-level `--help` and `--configuration` options. Run `./cosmictiger --help`
for all options and defaults. HPX's own options retain their `--hpx:*` syntax.

| Group | Controls |
| --- | --- |
| `run` | Test selection, iteration limit, time-step hierarchy |
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

The old ungrouped and snake_case option names have been replaced. For example,
`--parts_dim=16 --max_iter=2 --cuda=true` is now
`--initial.particlesPerDimension=16 --run.maxIterations=2 --compute.cuda=true`,
and `--hydro.sph=true` is now `--hydro.enabled=true`.

Use `--configuration=simulation.ini` to read an INI file. Both dotted keys and
INI sections work; explicit command-line values override configuration values:

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
cfl = 0.25
[run]
maxIterations = 2
```

[default.ini](default.ini) contains every application option at its source
default, except `help` and `configuration`. The complete, independently editable
problem files in [configurations/](configurations/README.md) contain the same
option set, with problem-specific values and inline help comments. Files are
loaded explicitly; there is no implicit loading of `default.ini`.

For the existing Kelvin–Helmholtz test, run from a build directory:

```sh
./cosmictiger --configuration=../../configurations/kelvinHelmholz.ini --hpx:threads=12 --hpx:bind=none
```

This file selects `run.test=helmholtz`, CUDA, nominal resolution 32, and full-box
particle Silo output. The initializer creates 32,600 gas particles and runs to
time 10. Output is configured directly in the INI:

```ini
[output]
silo = true
siloInterval = 1
directory = ./output
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
Standalone hydro tests retain their initial and final snapshots. Snapshot names
retain their coarse cycle number, so skipped cycles leave gaps in the names.

`output.directory` defaults to `./output`, relative to the working directory,
and is created automatically when Silo output is enabled. Snapshots are named
`output/view.0.<cycle>.silo` with the default directory. Absolute directory paths
are also supported. `output.silo=false` disables particle output.
`analysis.views` independently controls HEALPix maps.

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
dark-matter-only mode keeps its hierarchical gravity integrator. Enable gas explicitly with `--hydro.enabled=true`; gas runs use
the ordinary block-timestep gravity/SPH path and require CUDA.

Cosmological gas runs initialize a displaced gas lattice alongside the dark
matter lattice. `initial.particlesPerDimension` specifies the lattice size for each species. With
SPH enabled, the total particle count doubles, while the gas and dark-matter
masses are weighted by `omega_b / omega_m` and `omega_c / omega_m` respectively.
The power-spectrum deposition uses these masses too.

For a small cosmological run from a build directory containing `power.init`:

```sh
./cosmictiger --hydro.enabled=true --compute.cuda=true --initial.particlesPerDimension=16 --run.maxIterations=2 --hpx:threads=4
```

The gas controls appear under `hydro.*` in `--help` and can also be
specified in the INI configuration file. Chemistry, diffusion, and conduction
are detached; enabling their legacy flags is rejected. Star formation and adaptive gravitational
softening are not exposed by this integration; gas smoothing lengths remain
adaptive. The legacy equal-mass lightcone and halo finders, and dark-matter-only
Gadget/Plummer/glass loaders, reject SPH runs.

Gas state, including fields used by staggered timesteps, follows particles during
domain redistribution and is retained in versioned checkpoints. Restart using
the same physics and particle-layout options. Existing main-branch dark-matter
checkpoints remain readable in the original mode. Earlier SPH checkpoints are
rejected because their formulation differs. New SPH checkpoints store and validate
the viscosity, smoothing, timestep, and external-force parameters.

## Build

Use CMake 3.24 or newer, C++20, HPX with CUDA and networking enabled, CUDA,
serial FFTW3 (single precision), GSL, Silo, HEALPix C, Boost program options and
Boost.Units headers,
and a Fortran compiler. Select the CUDA architecture for the target machine:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/hpx -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/release --parallel 4
```

The build copies the initial power spectrum and builds `recfast`. Run from the
build directory so these inputs are available. Particle Silo output writes
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
