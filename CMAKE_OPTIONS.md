# CosmicTiger CMake options

This is a source-level reference for the configuration inputs in
`CMakeLists.txt` and `cmake/FindLibbacktrace.cmake`. No build directory is needed
to read it. Pass options as `-DNAME=value` when configuring with CMake 3.24 or
newer:

```sh
cmake -S . -B build/fmm_order7 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_GRAVITY_METHOD=fmm \
  -DCOSMICTIGER_WITH_ORDER=7 \
  -DCOSMICTIGER_WITH_CUDA=OFF \
  -DBUILD_TESTING=OFF
cmake --build build/fmm_order7 --parallel 4
```

Run examples from the repository root. CMake creates the directory specified by
`-B`; configuration discovers dependencies and generates build rules, while
`cmake --build` compiles them. Add dependency paths below as needed.

## Project options

Defaults apply to a fresh CMake configuration. Boolean options accept `ON` or
`OFF`; solver and integrator names are lowercase.

| Option | Default | Values and purpose |
| --- | --- | --- |
| `COSMICTIGER_GRAVITY_METHOD` | `fmm` | `fmm` for global FMM, or `pmmm` for PMMM. Selects the gravity solver at compile time. |
| `COSMICTIGER_PMMM_NEAR_METHOD` | `pp` | `pp` for direct particle-particle near interactions, or `fmm` for cell FMM near interactions. Affects PMMM builds. |
| `COSMICTIGER_WITH_ORDER` | `7` | One integer from `3` through `9`. Compiles one FMM kernel order, also used by PMMM expansions. The convention is multipole degree plus two. |
| `COSMICTIGER_TIME_INTEGRATOR` | `leapfrog` | `leapfrog` for ordinary block-timestep kick-drift-kick (KDK), or `hierarchical` for hierarchical gravity. Selects the time integrator at compile time. Hierarchical builds reject SPH; rebuild with `leapfrog` to enable hydro. |
| `COSMICTIGER_WITH_MAX_BUCKET` | `128` | One of `32`, `64`, `128`, `256`, `512`, `1024`. Compile-time ceiling for the runtime `compute.bucketSize`. |
| `COSMICTIGER_WITH_CUDA` | `ON` | Builds CUDA code. Requires the CUDA toolkit and HPX built with CUDA support. |
| `COSMICTIGER_WITH_LONG_LONG_PART_INT` | `OFF` | Uses `long long` to index local particles. |
| `COSMICTIGER_ENABLE_PROFILING` | `ON` | Enables statistical CPU sampling and `PROFILE()` metadata. Independent of the installed HPX's APEX support. |
| `COSMICTIGER_REQUIRE_APEX` | Depends on build type | Defaults to `ON` for `Debug`, `RelWithDebInfo`, and `Profile`; otherwise `OFF`. Rejects an HPX installation without APEX when enabled. |
| `BUILD_TESTING` | `ON` | Standard CTest option used here to include standalone tests and embedded `run.test` application tests. `OFF` excludes both. |

Both `cosmictiger` (32-bit fixed-point positions) and `cosmictiger_fixed64`
(64-bit fixed-point positions) are built. Building the `cosmictiger` target also
builds its fixed64 companion, selected at runtime with
`--compute.positionType=fixed64`. Position storage precision is independent of
floating-point arithmetic precision. Both executables use the same selected
gravity solver and kernel order.

The former `COSMICTIGER_FMM_MIN_ORDER` and `COSMICTIGER_FMM_MAX_ORDER` options
are retired. CMake removes their cached values and derives both from
`COSMICTIGER_WITH_ORDER`. Generated source headers still cover orders 3–9, but
each executable compiles only its selected order.

Preprocessor definitions such as `ORDER`, `FMM_MIN_ORDER`, `FMM_MAX_ORDER`,
`MAX_BUCKET_SIZE`, `COSMICTIGER_GRAVITY_PMMM`, and `COSMICTIGER_PMMM_NEAR_FMM`
are derived from these settings, not independent CMake options.

The solver, PMMM near method, kernel order, and time integrator require rebuilding
to change. Leapfrog is the default for both dark matter and SPH. The hierarchical
integrator supports dark matter only; `hydro.enabled=true` requires a build with
`-DCOSMICTIGER_TIME_INTEGRATOR=leapfrog`.
PMMM ghost width remains a runtime option, `--gravity.pmmmNearWidth` (1–2,
default 2). Global FMM theta remains `--gravity.openingAngle`. PMMM also requires
the runtime `--gravity.meshSize` setting.

## Solver combinations

| Executable variant | CMake settings |
| --- | --- |
| Global FMM | `-DCOSMICTIGER_GRAVITY_METHOD=fmm` |
| PMMM-FMM | `-DCOSMICTIGER_GRAVITY_METHOD=pmmm -DCOSMICTIGER_PMMM_NEAR_METHOD=fmm` |
| PMMM-PP | `-DCOSMICTIGER_GRAVITY_METHOD=pmmm -DCOSMICTIGER_PMMM_NEAR_METHOD=pp` |

Use separate build directories to keep variants available together. For example:

```sh
cmake -S . -B build/pmmm_fmm_order7 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_GRAVITY_METHOD=pmmm \
  -DCOSMICTIGER_PMMM_NEAR_METHOD=fmm \
  -DCOSMICTIGER_WITH_ORDER=7 \
  -DCOSMICTIGER_WITH_CUDA=OFF \
  -DBUILD_TESTING=OFF

cmake -S . -B build/pmmm_pp_order5 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_GRAVITY_METHOD=pmmm \
  -DCOSMICTIGER_PMMM_NEAR_METHOD=pp \
  -DCOSMICTIGER_WITH_ORDER=5 \
  -DCOSMICTIGER_WITH_CUDA=OFF \
  -DBUILD_TESTING=OFF
```

## Common CMake and toolchain settings

These are standard CMake controls relevant to this project, rather than
additional `COSMICTIGER_*` options.

| Setting | Default or use |
| --- | --- |
| `CMAKE_BUILD_TYPE` | For single-configuration generators such as Ninja: typically `Release`, `RelWithDebInfo`, `Debug`, or `MinSizeRel`. The project does not choose a default when invoked directly. |
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
| `Boost_DIR` | Directory containing `BoostConfig.cmake` when using package-config discovery; Boost.Program_options is required. `CMAKE_PREFIX_PATH` can select its installation prefix. |
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
The available tests also depend on the selected solver, CUDA, HPX features,
and optional Python packages such as NumPy, SciPy, and Matplotlib.

`HPX_WITH_CUDA` and `HPX_WITH_APEX` describe the selected HPX installation.
Setting these in CosmicTiger's CMake command cannot add missing features to
that installation. Select an appropriately built HPX instead. An HPX package
itself built with CUDA may require CUDA toolchain discovery even when
`COSMICTIGER_WITH_CUDA=OFF`.

## Using build.sh

`./build.sh` provides an interactive configure/build menu. Its initial choices
are `Release`, global FMM, order 7, leapfrog, CUDA enabled, and tests disabled. In particular, its `BUILD_TESTING=OFF` differs from direct CMake's
default `ON`. It uses Ninja and build directories named
`build/<mode>_<variant>_order<order>_<integrator>`, such as
`build/release_pmmm_fmm_order7_leapfrog`. Selecting the other integrator uses a
separate build directory.

The script builds both position-storage executables, chooses the APEX
requirement from the build mode, supplies the custom `Profile` flags above,
and rediscovers cached HPX, MPI, GSL, and HEALPix paths. `Profile` uses a
`RelWithDebInfo` HPX installation. For settings not offered by the menu,
configure directly with CMake.

## Inspecting and changing a configured build

```sh
# List cached options, including advanced entries and help descriptions.
# This reads the existing cache without configuring or compiling.
cmake -LAH -N build/fmm_order7

# Change an option, reconfigure, then rebuild.
cmake -S . -B build/fmm_order7 -DCOSMICTIGER_WITH_MAX_BUCKET=256
cmake --build build/fmm_order7 --parallel 4

# Help for CMake's built-in variables; project variables are documented above.
cmake --help-variable-list
cmake --help-variable CMAKE_BUILD_TYPE
```

Options persist in `CMakeCache.txt`; omitting a previously supplied option does
not reset it. This includes `COSMICTIGER_REQUIRE_APEX`: its build-type-dependent
default is chosen when the cache entry is first created. Supply an explicit
value when changing an existing configuration. A new build directory gives
fresh defaults.

The cache listing contains dependency settings and detected results as well
as user options. Internal probe results such as `COSMICTIGER_FFTW3F_MPI_LINKS`
are not feature switches and should not be set to bypass dependency checks.
CMake has no general source-only command that lists every possible project
option: options can be created conditionally while `CMakeLists.txt` executes.
