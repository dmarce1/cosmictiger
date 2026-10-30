# CosmicTiger CMake options

This is a source-level reference for the configuration inputs in
`CMakeLists.txt` and `cmake/FindLibbacktrace.cmake`. No build directory is needed
to read it. Pass options as `-DNAME=value` when configuring with CMake 3.24 or
newer:

```sh
cmake -S . -B build/release_cpu -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMICTIGER_WITH_CUDA=OFF \
  -DBUILD_TESTING=OFF
make -C build/release_cpu -j4 cosmictiger_leapfrog_fmm_sph_7
make -C build/release_cpu -j4 cosmictiger_hierarchical_pmmm_pp_8
```

Run examples from the repository root. CMake creates the directory specified by
`-B`; configuration discovers dependencies and generates build rules, while
`make` or `cmake --build` compiles them. Both examples use the same generated
Makefile and shared core library. Add dependency paths below as needed.

## Project options

Defaults apply to a fresh CMake configuration. Boolean options accept `ON` or
`OFF`; solver and integrator names are lowercase.

| Option | Default | Values and purpose |
| --- | --- | --- |
| `COSMICTIGER_GRAVITY_METHOD` | `fmm` | `none` for SPH without gravity, `fmm` for global FMM, or `pmmm` for PMMM. `none` requires SPH and leapfrog. Selects the compiled gravity solver for the default executable and tests. |
| `COSMICTIGER_PMMM_NEAR_METHOD` | `pp` | `pp` for direct particle-particle near interactions, or `fmm` for cell FMM near interactions. Selects the default PMMM near solver. |
| `COSMICTIGER_WITH_ORDER` | `7` for FMM; `8` for PMMM | One integer from `3` through `9`, selecting the default executable's shared FMM/PMMM kernel order. The convention is multipole degree plus two. Explicit or cached selections override the default. Ignored by SPH-only targets; their compiled order is zero. |
| `COSMICTIGER_TIME_INTEGRATOR` | `leapfrog` | `leapfrog` for ordinary block-timestep kick-drift-kick (KDK), or `hierarchical` for hierarchical gravity. Selects the compiled integrator for the default executable and tests. Hierarchical hydro integration is not implemented, so hierarchical builds currently require SPH disabled. |
| `COSMICTIGER_POSITION_TYPE` | `fixed32` | `fixed32` or `fixed64` position storage for the default executable and every named variant in this build directory. Target and executable names are unchanged. |
| `COSMICTIGER_WITH_MAX_BUCKET` | `128` | One of `32`, `64`, `128`, `256`, `512`, `1024`. Compile-time ceiling for the runtime `compute.bucketSize`. |
| `COSMICTIGER_WITH_CUDA` | `ON` | Builds CUDA code. Requires the CUDA toolkit and HPX built with CUDA support. |
| `COSMICTIGER_WITH_SPH` | `ON` | Selects SPH support for the default executable and, when testing is enabled, SPH tests. `OFF` excludes SPH implementation sources from that executable and omits SPH tests. SPH supports leapfrog without gravity, with global FMM, or with either PMMM near method. Named targets include `_sph` for SPH support and omit it for collisionless builds. |
| `COSMICTIGER_WITH_LONG_LONG_PART_INT` | `OFF` | Uses `long long` to index local particles. |
| `COSMICTIGER_ENABLE_PROFILING` | `ON` | Enables statistical CPU sampling and `PROFILE()` metadata. Independent of the installed HPX's APEX support. |
| `COSMICTIGER_REQUIRE_APEX` | Depends on build type | Defaults to `ON` for `Debug`, `RelWithDebInfo`, and `Profile`; otherwise `OFF`. Rejects an HPX installation without APEX when enabled. |
| `BUILD_TESTING` | `ON` | Standard CTest option used here to include standalone tests and embedded `run.test` application tests. `OFF` excludes both. |

`cosmictiger` and all named variants use the position storage selected by
`COSMICTIGER_POSITION_TYPE`. Position storage precision is independent of
floating-point arithmetic precision. Runtime `compute.positionType` defaults to
the compiled width; an explicit CLI or INI value must match it. There are no
companion executables or automatic runtime switches. The five physics cache
options select the default executable and its tests; all supported named
variants are available regardless of the selected default.

To build with 64-bit positions while preserving a separate fixed32 build:

```sh
cmake -S . -B build/release_cpu_fixed64 -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DCOSMICTIGER_WITH_CUDA=OFF \
  -DCOSMICTIGER_POSITION_TYPE=fixed64 -DBUILD_TESTING=OFF
make -C build/release_cpu_fixed64 -j4 cosmictiger_leapfrog_fmm_sph_7
```

Alternatively, change the option in an existing build directory and rebuild the
desired targets. Their names remain unchanged. Checkpoints require the same
position width as the executable used to write them.
The supplied INIs leave `compute.positionType` commented to use the compiled
width automatically.

The former `COSMICTIGER_FMM_MIN_ORDER` and `COSMICTIGER_FMM_MAX_ORDER` options
are retired. CMake removes their cached values and derives both from
`COSMICTIGER_WITH_ORDER`. Generated source headers still cover orders 3–9, but
each gravity-enabled executable compiles only its selected order. SPH-only targets compile no FMM instances.

Preprocessor definitions such as `ORDER`, `FMM_MIN_ORDER`, `FMM_MAX_ORDER`,
`MAX_BUCKET_SIZE`, `COSMICTIGER_WITH_GRAVITY`, `COSMICTIGER_GRAVITY_PMMM`, and `COSMICTIGER_PMMM_NEAR_FMM`
are derived from these settings, not independent CMake options.

The solver, PMMM near method, kernel order, time integrator, and SPH support are
compiled into each executable. Switch them by building and running the matching
named target. Leapfrog is the default for both dark matter and SPH. The
hierarchical integrator supports dark matter only; `hydro.enabled=true` requires
a `leapfrog_<gravity>_sph` variant. The hierarchical driver lacks SPH integration
stages. Leapfrog SPH supports no gravity, global FMM, PMMM-PP, and PMMM-FMM.
PMMM ghost width remains a runtime option, `--gravity.pmmmNearWidth` (1–4,
default 2). PMMM derives its theta from `sqrt(3)/(pmmmNearWidth+1)` and requires
the runtime `--gravity.meshSize` setting. Global FMM keeps `--gravity.openingAngle`
and the optional cosmological `--gravity.openingSchedule` as runtime controls.

For the SPH-only default, set `COSMICTIGER_GRAVITY_METHOD=none`,
`COSMICTIGER_WITH_SPH=ON`, and `COSMICTIGER_TIME_INTEGRATOR=leapfrog`.
It excludes gravity sources, generated FMM instances, gravity tests, and halo
potential analysis. It retains cosmology, imposed hydro fields, and shared
FFTW/MPI dependencies. The cached FMM order is preserved for later gravity
builds; the SPH-only executable has no order and reports zero in build metadata.

## Named physics targets

After configuring, print the physics target guide without compiling:

```sh
make -C build/release_cpu cosmictiger_help
```

From inside the build directory, use `make cosmictiger_help`. It shows the nine gravity
families, the SPH-only target, orders, selection examples, current default, CPU/CUDA mode,
and position storage. Standard `make help` lists available targets,
including `cosmictiger_help`.

Every configuration provides gravity-enabled targets named
`cosmictiger_<integrator>_<gravity>[_sph]_<order>` and the order-independent
`cosmictiger_leapfrog_sph` target. The optional `_sph` selects SPH;
omit it for collisionless builds. Replace `<order>` in each supported family
below with any integer from 3 through 9:

| Target family | Physics |
| --- | --- |
| `cosmictiger_leapfrog_sph` | Leapfrog, SPH only; no gravity or expansion order |
| `cosmictiger_leapfrog_fmm_sph_<order>` | Leapfrog, global FMM, SPH available |
| `cosmictiger_leapfrog_pmmm_pp_sph_<order>` | Leapfrog, PMMM with direct near pairs, SPH available |
| `cosmictiger_leapfrog_pmmm_fmm_sph_<order>` | Leapfrog, PMMM with FMM near interactions, SPH available |
| `cosmictiger_leapfrog_fmm_<order>` | Leapfrog, global FMM, no SPH |
| `cosmictiger_leapfrog_pmmm_pp_<order>` | Leapfrog, PMMM with direct near pairs, no SPH |
| `cosmictiger_leapfrog_pmmm_fmm_<order>` | Leapfrog, PMMM with FMM near interactions, no SPH |
| `cosmictiger_hierarchical_fmm_<order>` | Hierarchical, global FMM, no SPH |
| `cosmictiger_hierarchical_pmmm_pp_<order>` | Hierarchical, PMMM with direct near pairs, no SPH |
| `cosmictiger_hierarchical_pmmm_fmm_<order>` | Hierarchical, PMMM with FMM near interactions, no SPH |

The 64 physics variants comprise 42 collisionless combinations (2 integrators
× 3 gravity methods × 7 orders), 21 SPH combinations with gravity (leapfrog × 3
gravity methods × 7 orders), and one SPH-only target. These produce 64 named
executables, each using the position width selected for the build directory. Target names never include a position-width suffix.

The `cosmictiger_core` static library contains configuration-independent
implementation sources and compiles once per build directory. Remaining sources
use per-variant objects and compile definitions, including position storage
precision. Building another variant reuses the shared core and any previously
built objects for that exact variant. A default executable and the corresponding
named executable reuse the same variant objects.

Source selection follows the compiled solver: global FMM builds exclude PMMM
implementation sources, and PMMM builds include only their selected near-field
implementation. Shared gravity utilities remain available to the selected solver.

Named variants are excluded from the default `all` build. Plain `make` builds
the configured `cosmictiger` executable and enabled tests, without compiling the whole
matrix. Request additional variants from the same Makefile, without
reconfiguring:

```sh
make -C build/release_cpu -j4 cosmictiger_leapfrog_pmmm_fmm_7
make -C build/release_cpu -j4 cosmictiger_leapfrog_pmmm_pp_5
make -C build/release_cpu -j4 cosmictiger_leapfrog_pmmm_fmm_sph_7

# Equivalent generator-independent command:
cmake --build build/release_cpu --parallel 4 \
  --target cosmictiger_leapfrog_pmmm_pp_5
```

For a different default executable and matching test selection, reconfigure
the existing directory with the five cache options above. For example, select
hierarchical PMMM-PP at order 5 with `COSMICTIGER_TIME_INTEGRATOR=hierarchical`,
`COSMICTIGER_GRAVITY_METHOD=pmmm`, `COSMICTIGER_PMMM_NEAR_METHOD=pp`,
`COSMICTIGER_WITH_ORDER=5`, and `COSMICTIGER_WITH_SPH=OFF`. Changing defaults
does not remove other named targets or their existing objects. SPH is supported
only with leapfrog; hierarchical SPH combinations are
rejected during configuration.

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

`./build.sh` provides an interactive configure/build menu. Its initial choices
are `Release`, global FMM, order 7, leapfrog, fixed32 positions, CUDA and SPH enabled,
and tests disabled. Choosing a PMMM solver changes the default order to 8; an order
explicitly chosen in the menu is retained when changing solvers. In particular,
its `BUILD_TESTING=OFF` differs from direct CMake's
default `ON`. It generates Unix Makefiles in `build/<mode>_<cuda|cpu>`, such as
`build/release_cuda` or `build/debug_cpu`. Physics selections share one build
directory and preserve previously compiled named variants. Build mode and CUDA
support remain separate because they affect the common library and toolchain.
Position storage is selected for the whole build directory; changing it rebuilds
the requested target under the same name.

Disable "Build SPH support" in the menu when selecting hierarchical
integration. The menu reports unsupported SPH combinations and returns to its
settings without starting a build. A valid selection determines both the named
target to build and the default physics cache options, so enabling tests tests
the selected physics combination.

The script builds the named executable at the selected position width, chooses the APEX
requirement from the build mode, supplies the custom `Profile` flags above,
and rediscovers cached HPX, MPI, GSL, and HEALPix paths. `Profile` uses a
`RelWithDebInfo` HPX installation. For settings not offered by the menu,
configure directly with CMake.

## Inspecting and changing a configured build

```sh
# List cached options, including advanced entries and help descriptions.
# This reads the existing cache without configuring or compiling.
cmake -LAH -N build/release_cpu

# Change an option, reconfigure, then rebuild.
cmake -S . -B build/release_cpu -DCOSMICTIGER_WITH_MAX_BUCKET=256
cmake --build build/release_cpu --parallel 4

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
