# CosmicTiger CMake options

Configure one `cosmictiger` executable with the capabilities needed for a run.
Disabled physics and its test problems are excluded from compilation. Shared
sources compile once; each selected expansion degree has separate CPU/CUDA
instantiation units.

```sh
cmake -S . -B build/release_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_WITH_CUDA=OFF -DBUILD_TESTING=OFF \
  -DCOSMICTIGER_ENABLE_FMM=ON -DCOSMICTIGER_ENABLE_PMMM=ON \
  -DCOSMICTIGER_ENABLE_PMMM_NEAR_FMM=ON \
  -DCOSMICTIGER_ENABLE_INTEGRATOR_GLOBAL=ON \
  -DCOSMICTIGER_EXPANSION_DEGREE=4-6
cmake --build build/release_cpu --target cosmictiger --parallel 3
build/release_cpu/cosmictiger --gravity.method.far=pmmm \
  --gravity.method.near=pp --gravity.meshSize=16 \
  --gravity.expansionDegree=6 --run.timeIntegrator=global --compute.cuda=false
```

## Project options

| Option | Fresh default | Purpose |
| --- | --- | --- |
| `COSMICTIGER_ENABLE_FMM` | `ON` | Compile global FMM gravity. |
| `COSMICTIGER_ENABLE_PMMM` | `OFF` | Compile PMMM gravity. |
| `COSMICTIGER_ENABLE_SPH` | `ON` | Compile SPH, particle chemistry, and hydro problems. |
| `COSMICTIGER_ENABLE_PMMM_NEAR_PP` | `ON` | Include direct near interactions when PMMM is enabled. |
| `COSMICTIGER_ENABLE_PMMM_NEAR_FMM` | `OFF` | Include cell FMM near interactions when PMMM is enabled. Includes shared FMM machinery without requiring global FMM. |
| `COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL` | `ON` | Traditional nested individual block timesteps. |
| `COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL` | `OFF` | Hierarchical Hamiltonian integration for collisionless gravity. |
| `COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL` | `OFF` | All particles share the smallest raw allowable timestep, without power-of-two quantization. |
| `COSMICTIGER_EXPANSION_DEGREE` | `4-6` | Inclusive potential-degree range, or one degree such as `6`; supported degrees are 2 through 8. |
| `COSMICTIGER_POSITION_TYPE` | `fixed32` | `fixed32` or `fixed64` particle positions for the build. |
| `COSMICTIGER_WITH_CUDA` | `ON` | Compile CUDA; requires a CUDA-enabled HPX installation. |
| `COSMICTIGER_WITH_MAX_BUCKET` | `128` | Maximum tree leaf bucket: 32, 64, 128, 256, 512, or 1024. |
| `COSMICTIGER_WITH_LONG_LONG_PART_INT` | `OFF` | Use long long local particle indices. |
| `COSMICTIGER_ENABLE_PROFILING` | `ON` | CPU sampling and PROFILE metadata. |
| `COSMICTIGER_REQUIRE_APEX` | Build-dependent | ON for Debug, RelWithDebInfo, Profile; otherwise OFF. Requires an APEX-enabled HPX installation. |
| `BUILD_TESTING` | `ON` | Compile and register tests supported by the enabled capabilities. |

At least one physics component and one integrator must be enabled. PMMM needs
at least one near method. SPH needs traditional or global integration available;
a collisionless run may select hierarchical integration even when SPH is compiled.
To build SPH only, disable both `COSMICTIGER_ENABLE_FMM` and
`COSMICTIGER_ENABLE_PMMM`. To build gravity only, disable `COSMICTIGER_ENABLE_SPH`.

Runtime selection uses `gravity.method.far=fmm|pmmm`,
`gravity.method.near=pp|fmm` (PMMM), and
`run.timeIntegrator=traditional|hierarchical|global`. A sole eligible method is
automatic; multiple eligible methods require an explicit CLI/INI choice. A saved
checkpoint selection also resolves these choices on restart. Help and startup
metadata distinguish compiled capabilities from the selected run.

`gravity.expansionDegree` defaults to the highest compiled degree. Potential
degree d means source multipole degree d-1 and historical internal order d+1.
The default degrees 4-6 therefore instantiate old orders 5-7. PMMM far and near
FMM share the selected degree. Internal tensor and old checkpoint integers keep
their historical order convention. PMMM formerly defaulted to old order 8
(potential degree 7); request a range including 7 to retain that accuracy setting.
SPH-only builds instantiate no gravity expansions.

The global integrator reduces raw gravity/hydro limits over all particles and
all localities before opening the next half-step. Cosmological and configured
maximum-step limits also apply. Final-time and scenario-stage boundaries may
shorten a step. Kicks, drifts, SPH and chemistry use explicit old/new intervals;
checkpoints retain the actual pending interval independently of the integer rung
clock. Traditional and hierarchical integration retain their individual timesteps.

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
An FMM-only build still discovers the project's full required dependency set.

When tests are enabled and no installed GoogleTest is found, configuration
fetches a pinned GoogleTest source archive unless its source directory is
overridden. The fallback forces `BUILD_GMOCK=OFF` and `INSTALL_GTEST=OFF`.
The available tests also depend on the selected solver, SPH, CUDA, HPX
features, and optional Python packages such as NumPy, SciPy, and Matplotlib.

`HPX_WITH_CUDA` and `HPX_WITH_APEX` describe the selected HPX installation.
Setting these in CosmicTiger's CMake command cannot add missing features to
that installation. Select an appropriately built HPX instead. An HPX package
itself built with CUDA may require CUDA toolchain discovery even when
`COSMICTIGER_WITH_CUDA=OFF`.

## Using build.sh

`./build.sh` offers an interactive menu for build mode, enabled physics, near
methods, integrators, potential-degree range, CUDA, position storage, dependency
paths, and tests. It builds `cosmictiger` in `build/<mode>_<cpu|cuda>`. Changing
capabilities reconfigures that directory. Use separate directories for builds
that need to remain available simultaneously. Local jobs are capped at half the
available cores; concurrent work shares that budget.
