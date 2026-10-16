# CosmicTiger - A Cosmological N-Body Code

## SPH

The SPH implementation from `new_age` is integrated with the current Cartesian
FMM and particle storage. The normal dark-matter-only mode keeps its hierarchical
gravity integrator. Enable gas explicitly with `--hydro.sph=true`; gas runs use
the ordinary block-timestep gravity/SPH path and require CUDA.

Cosmological gas runs initialize a displaced gas lattice alongside the dark
matter lattice. `parts_dim` specifies the lattice size for each species. With
SPH enabled, the total particle count doubles, while the gas and dark-matter
masses are weighted by `omega_b / omega_m` and `omega_c / omega_m` respectively.
The power-spectrum deposition uses these masses too.

For a small cosmological run from a build directory containing `power.init`:

```sh
./cosmictiger --hydro.sph=true --cuda=true --parts_dim=16 --max_iter=2 --hpx:threads=4
```

The restored gas controls appear under `hydro.*` in `--help` and can also be
specified in the existing INI configuration file. Chemistry, diffusion, and
conduction default to disabled. Star formation and adaptive gravitational
softening are not exposed by this integration; gas smoothing lengths remain
adaptive. The legacy equal-mass lightcone and halo finders, and dark-matter-only
Gadget/Plummer/glass loaders, reject SPH runs.

Gas state, including fields used by staggered timesteps, follows particles during
domain redistribution and is retained in versioned checkpoints. Restart using
the same physics and particle-layout options. Existing main-branch dark-matter
checkpoints remain readable in the original mode. Unversioned checkpoints from
old SPH branches are not supported.

## Build

Use CMake 3.24 or newer, C++20, HPX with CUDA and networking enabled, CUDA,
serial FFTW3 (single precision), GSL, Silo, HEALPix C, Boost program options,
and a Fortran compiler. Select the CUDA architecture for the target machine:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/hpx -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/release --parallel 4
```

The build copies the initial power spectrum and builds `recfast`. Run from the
build directory so these inputs are available. The existing `view.txt` mechanism
writes separate gas and dark-matter Silo point meshes, with gas thermodynamic
fields.

## Regression checks

```sh
ctest --test-dir build/release --output-on-failure
```

The standalone tracer selection, numerical utility, and compression regressions
run on the CPU without initializing a GPU:

```sh
ctest --test-dir build/release --output-on-failure -R '^(particle_selection|numerical_utilities|compression_)'
```

The numerical utility and compression targets still require CUDA headers. The
build also compiles a device-side range regression without launching a kernel.

The checks cover mixed-mass CPU/CUDA force evaluation, the gas gravity handoff,
synchronized diagnostics, gas reordering, checkpoint payloads and configuration
compatibility, a coarse Sod shock tube, dark-matter startup, and a short
cosmological SPH run with restart. The CUDA-enabled executable initializes the
GPU even when selecting CPU gravity, so this test suite requires an available
CUDA device. These are correctness and integration checks, not a
resolution-convergence study or a performance benchmark.
