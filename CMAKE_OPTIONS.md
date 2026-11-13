# CosmicTiger CMake options

Configure one `cosmictiger` executable with the capabilities needed for a run.
Disabled physics and its test problems are excluded from compilation. Shared
sources compile once; each selected expansion degree has separate CPU/CUDA
instantiation units.

```sh
cmake -S . -B build/release_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_WITH_CUDA=OFF -DBUILD_TESTING=OFF \
  -DCOSMICTIGER_ENABLE_GRAVITY=ON \
  -DCOSMICTIGER_EXPANSION_DEGREE=6
cmake --build build/release_cpu --target cosmictiger --parallel 3
build/release_cpu/cosmictiger \
  --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
```

## Small matrix algorithms

The six `COSMICTIGER_MATRIX_{GENERAL,SYMMETRIC}_{INVERSE,DETERMINANT,SOLVE}_THRESHOLD`
cache settings select the largest dimension using explicit cofactors. Defaults
are **4 for inverse/determinant** and **3 for solve**, independently configurable
for general and symmetric storage. Values are integers 0 through 6; 0 selects
pivoting for every non-diagonal automatic operation. Diagonal matrices always use
direct formulas. See [matrix algorithms and measurements](docs/matrix.md).

## Project options

| Option | Fresh default | Purpose |
| --- | --- | --- |
| `COSMICTIGER_ENABLE_GRAVITY` | `ON` | Compile the global periodic FMM, including direct leaf particle pairs. |
| `COSMICTIGER_DIMENSION_COUNT` | `3` | Spatial dimensions: 1, 2, or 3. Below 3 requires SPH, gravity OFF, chemistry OFF, and stars OFF. See [dimensional SPH](docs/sph_dimensions.md). |
| `COSMICTIGER_ENABLE_SPH` | `ON` | Compile SPH and hydro problems. |
| `COSMICTIGER_ENABLE_CHEMISTRY` | `ON` | Compile equilibrium cooling when SPH is also enabled. Inactive with SPH disabled. |
| `COSMICTIGER_ENABLE_STARS` | `OFF` | Compile GADGET-4 multiphase star formation. Requires equilibrium cooling; inactive with SPH disabled. |
| `COSMICTIGER_EXPANSION_DEGREE` | `6` | One potential degree per executable, from 2 through 8. |
| `COSMICTIGER_POSITION_TYPE` | `fixed32` | `fixed32` or `fixed64` particle positions for the build. |
| `COSMICTIGER_WITH_CUDA` | `ON` | Compile CUDA; requires a CUDA-enabled HPX installation. |
| `COSMICTIGER_WITH_LONG_LONG_PART_INT` | `OFF` | Use long long local particle indices. |
| `COSMICTIGER_PROFILE` | `ON` | Manual phase profiler (`profileBegin`/`profileEnd`, `profile.txt`). |
| `COSMICTIGER_DEVICE_PROFILE` | `OFF` | Per-block CUDA gravity phase cycles in `profile.txt`; requires `COSMICTIGER_PROFILE` and `COSMICTIGER_WITH_CUDA`. |
| `COSMICTIGER_REQUIRE_APEX` | Build-dependent | ON for Debug, RelWithDebInfo, Profile; otherwise OFF. Requires an APEX-enabled HPX installation. |
| `BUILD_TESTING` | `ON` | Compile and register tests supported by the enabled capabilities. |

At least one physics component must be enabled. Hierarchical global FMM gravity
and midpoint SPH form the sole integration scheme.
Gas and dark matter can evolve together under global FMM self-gravity.
To build SPH only, disable `COSMICTIGER_ENABLE_GRAVITY`.
To build gravity only, disable `COSMICTIGER_ENABLE_SPH`.

Equilibrium cooling requires both `COSMICTIGER_ENABLE_SPH=ON` and
`COSMICTIGER_ENABLE_CHEMISTRY=ON`. Disable chemistry to retain SPH without the
GADGET-4 atomic cooling module. With SPH disabled, the chemistry setting is inactive
and remains unchanged in the cache. Homogeneous cosmological recombination is
available regardless of these two options.

Star formation requires `COSMICTIGER_ENABLE_STARS=ON`,
`COSMICTIGER_ENABLE_CHEMISTRY=ON`, and `COSMICTIGER_ENABLE_SPH=ON`.
Configuration rejects stars enabled with chemistry disabled. With SPH disabled,
the stars and chemistry settings remain cached but their sources and tests
are excluded. Runtime `starFormation.enabled=true` requires
`hydro.enabled=true` and `chemistry.enabled=true` with `hydro.heatCapacityRatio=5/3`; any
compiled integrator may advance it. See [the model documentation](docs/star_formation.md) for the
equilibrium cooling assumptions, configuration example, and validation targets.

The former `COSMICTIGER_ENABLE_STAR_FORMATION` name initializes `STARS` when
the canonical option is absent, including an existing cache created before
the rename. Subsequent configuration uses `COSMICTIGER_ENABLE_STARS` and
synchronizes the legacy cache entry and compiler macro with it. Thus
`-DCOSMICTIGER_ENABLE_STARS=OFF` disables stars even if the old cache entry
was `ON`; after migration, change `STARS` to select either value.

Global FMM is the sole gravity solver. Hierarchical gravity kicks and midpoint
SPH are automatic. `gravity.method.near` and `run.timeIntegrator`
are retired. Restart requires the saved expansion degree to match the executable.

`run.numberOfOutputs` counts equally spaced output phases including `t=0` and
`t=end` (at least two). The spacing is physical code time for fixed scenarios
and conformal code time for cosmology. It replaces `run.numberOfSteps` and
`output.siloInterval`; enabled Silo snapshots are written at every phase.
To retain a former interval count `M`, request `numberOfOutputs=M+1`. The
checkpoint header continues storing `M`. The rung clock subdivides each output interval dyadically.

`COSMICTIGER_EXPANSION_DEGREE` selects one degree at build time (default 6).
Potential degree d means source multipole degree d-1 and internal order d+1.
The global tree uses this degree for every interaction. Each build generates only its
Cartesian operators under its own build directory; separate
build directories can use different degrees without changing source headers.
`gravity.expansionDegree` is no longer a CLI/INI option. Checkpoints retain the
historical internal order and reject a binary compiled with a different degree.
SPH-only builds instantiate no gravity expansions.

Gravity-only builds omit hydro problems/tests; SPH-only builds omit gravity
problems/tests. Coupled tests require both capabilities, and backend-specific
tests require that backend. Hydro problems with imposed forces remain available
without self-gravity. Common infrastructure and homogeneous recombination remain
available. The shared cosmology implementation still requires FFTW/MPI.

There are no named physics-combination executables. `cosmictiger_help` prints
the configured capabilities. Old exclusive CMake selections are retired; a fresh
build directory avoids ambiguous cached selections. See configure diagnostics for
migration guidance. CPU/CUDA and position width remain build-directory settings.
An explicit runtime `compute.positionType` must match the compiled width.

Gravity leaf capacity is the runtime option `compute.bucketSize` (default 128),
with no build-time ceiling or power-of-two requirement. CUDA shared-memory
capacity limits the sink tile; larger leaves span multiple tiles. Runtime
force-buffer index and allocation limits still apply.

## Common CMake and toolchain settings

These are standard CMake controls relevant to this project, rather than
additional `COSMICTIGER_*` options.

| Setting | Default or use |
| --- | --- |
| `CMAKE_BUILD_TYPE` | For single-configuration generators such as Unix Makefiles and Ninja: typically `Release`, `RelWithDebInfo`, `Debug`, or `MinSizeRel`. The project does not choose a default when invoked directly. |
| `CMAKE_CXX_STANDARD` | Project default `20`; `26` selects standard SIMD when available. A host-only FP16 target independently requires C++23 support. CUDA sources use C++20. |
| `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` | Select C and C++ compiler executables when first configuring. Use a fresh build directory when switching toolchains. |
| `CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_HOST_COMPILER` | Select the CUDA compiler and its host compiler when first configuring a CUDA toolchain. |
| `CMAKE_CUDA_ARCHITECTURES` | Project default `native` for CUDA builds if not already defined. Set explicit GPU architecture numbers when needed, such as `-DCMAKE_CUDA_ARCHITECTURES=86` for a matching GPU, or a quoted semicolon-separated list. |
| `CMAKE_TOOLCHAIN_FILE` | Path to a CMake toolchain file, normally supplied on the first configuration. |
| `CMAKE_PREFIX_PATH` | Dependency installation prefixes, for example `-DCMAKE_PREFIX_PATH="/opt/hpx;/opt/dependencies"`. The command-line/cache form uses semicolons; the Unix environment form uses colons. |
| `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, `CMAKE_CUDA_FLAGS` | Additional language flags. Build-type-specific forms include `CMAKE_CXX_FLAGS_RELEASE` and `CMAKE_CXX_FLAGS_DEBUG`. Target-specific flags in `CMakeLists.txt` still apply. |
| `CMAKE_EXE_LINKER_FLAGS` | Additional executable linker flags; build-type-specific forms are also available. |

The project enables `compile_commands.json` generation directly. Its
`CMAKE_CUDA_STANDARD=20` and `CMAKE_EXPORT_COMPILE_COMMANDS=ON` assignments are
not exposed as configurable project options.

`Profile` is a custom build type supported by `build.sh`. Selecting only
`-DCMAKE_BUILD_TYPE=Profile` in a direct CMake command does not reproduce the
script's flags. The script also supplies:

```sh
-DCMAKE_C_FLAGS_PROFILE="-O2 -g -pg -DNDEBUG"
-DCMAKE_CXX_FLAGS_PROFILE="-O2 -g -pg -DNDEBUG"
-DCMAKE_CUDA_FLAGS_PROFILE="-O2 -g -DNDEBUG -Xcompiler=-pg"
-DCMAKE_EXE_LINKER_FLAGS_PROFILE="-pg"
```

## Dependency selection

Dependencies are discovered automatically. Use these overrides when an
installation is not found or when selecting a particular installation.
`*_INCLUDE_DIR` values are directories containing the named header;
`*_LIBRARY` values are full library file paths.

| Variable(s) | Selects |
| --- | --- |
| `HPX_DIR` | Directory containing the installed `HPXConfig.cmake`. Alternatively supply the HPX installation prefix through `CMAKE_PREFIX_PATH`. |
| `Boost_DIR` | Directory containing `BoostConfig.cmake`; Boost.Program_options is required. |
| `CUDAToolkit_ROOT` | CUDA toolkit installation prefix for CUDA builds. |
| `MPI_CXX_COMPILER` | MPI C++ compiler wrapper used by CMake's MPI finder. MPI C++ support is required even for CPU-only builds. |
| `MPI_CXX_HEADER_DIR` | Explicit MPI header directory when automatic discovery needs assistance. |
| `FFTW3F_INCLUDE_DIR`, `FFTW3F_LIBRARY` | Single-precision FFTW: `fftw3.h` and `fftw3f`. |
| `FFTW3F_MPI_INCLUDE_DIR`, `FFTW3F_MPI_LIBRARY` | Single-precision MPI FFTW: `fftw3-mpi.h` and `fftw3f_mpi`. |
| `SILO_INCLUDE_DIR`, `SILO_LIBRARY` | Silo: `silo.h` and `siloh5` or `silo`. |
| `CHEALPIX_INCLUDE_DIR`, `CHEALPIX_LIBRARY` | HEALPix C: `chealpix.h` and `chealpix`. |
| `HEALPIX_CXX_INCLUDE_DIR`, `HEALPIX_CXX_LIBRARY` | HEALPix C++: `healpix_base.h` and `healpix_cxx`. |
| `Libbacktrace_INCLUDE_DIR`, `Libbacktrace_LIBRARY` | `backtrace.h` and `backtrace`; required on non-MSVC toolchains. These are advanced cache entries. |
| `Python3_EXECUTABLE` | Python interpreter. Required even with tests disabled because the explicit kernel-generation target uses it. |
| `GTest_DIR` | Directory containing GoogleTest's CMake package, used when `BUILD_TESTING=ON`. |
| `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` | Existing GoogleTest source directory for the FetchContent fallback when no installed GoogleTest package is found. |

GSL and Threads are also required and discovered with CMake's standard finders;
use `CMAKE_PREFIX_PATH` for a nonstandard GSL installation. FFTW's
single-precision MPI API must pass a compile/link check during configuration.
A gravity-only build still discovers the project's full required dependency set.

When tests are enabled and no installed GoogleTest is found, configuration
fetches a pinned GoogleTest source archive unless its source directory is
overridden. The fallback forces `BUILD_GMOCK=OFF` and `INSTALL_GTEST=OFF`.
The available tests also depend on the selected solver, SPH, equilibrium cooling, star formation,
CUDA, HPX features, and optional Python packages such as NumPy, SciPy, and Matplotlib.

`HPX_WITH_CUDA` and `HPX_WITH_APEX` describe the selected HPX installation.
Setting these in CosmicTiger's CMake command cannot add missing features to
that installation. Select an appropriately built HPX instead. An HPX package
itself built with CUDA may require CUDA toolchain discovery even when
`COSMICTIGER_WITH_CUDA=OFF`.

## Using build.sh

`./build.sh` offers an interactive menu for build mode, enabled physics (including stars), near
methods, integrators, potential degree, CUDA, position storage, dependency
paths, and tests. It builds `cosmictiger` in `build/<mode>_<cpu|cuda>`. Changing
capabilities reconfigures that directory. Use separate directories for builds
that need to remain available simultaneously. Local jobs are capped at half the
available cores; concurrent work shares that budget.

Gravity uses hierarchical global FMM kicks; SPH uses midpoint kicks. There is no integrator build selector.
