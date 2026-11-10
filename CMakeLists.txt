# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

cmake_minimum_required(VERSION 3.24)
project(cosmictiger LANGUAGES CXX C)

find_program(cosmictiger_ccache_program NAMES ccache NO_CACHE)
set(cosmictiger_use_ccache_default ON)
if(NOT cosmictiger_ccache_program)
    set(cosmictiger_use_ccache_default OFF)
    if(NOT DEFINED COSMICTIGER_USE_CCACHE)
        message(WARNING
            "ccache is recommended but optional and is not installed. "
            "Changing the default for COSMICTIGER_USE_CCACHE from ON to OFF.")
    endif()
endif()
option(COSMICTIGER_USE_CCACHE "Use ccache for compilation when available"
    ${cosmictiger_use_ccache_default})
if(COSMICTIGER_USE_CCACHE)
    if(cosmictiger_ccache_program)
        foreach(cosmictiger_language IN ITEMS C CXX CUDA)
            # Preserve user launchers, including CUDA's not-yet-initialized environment value.
            if(NOT DEFINED CMAKE_${cosmictiger_language}_COMPILER_LAUNCHER AND
                NOT DEFINED ENV{CMAKE_${cosmictiger_language}_COMPILER_LAUNCHER})
                set(CMAKE_${cosmictiger_language}_COMPILER_LAUNCHER "${cosmictiger_ccache_program}")
            endif()
        endforeach()
    else()
        message(WARNING
            "COSMICTIGER_USE_CCACHE is ON, but ccache is not installed. "
            "ccache is recommended but optional; continuing without automatic compiler caching.")
    endif()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
include(CTest)

# C++20 uses experimental SIMD; C++26 selects standard SIMD when available.
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(COSMICTIGER_WITH_CUDA "Enable CUDA" ON)
set(COSMICTIGER_POSITION_TYPE "fixed32" CACHE STRING
    "Particle position storage for every target: fixed32 or fixed64")
set_property(CACHE COSMICTIGER_POSITION_TYPE PROPERTY STRINGS fixed32 fixed64)
if(NOT COSMICTIGER_POSITION_TYPE MATCHES "^(fixed32|fixed64)$")
    message(FATAL_ERROR "COSMICTIGER_POSITION_TYPE must be fixed32 or fixed64")
endif()
string(REPLACE "fixed" "" COSMICTIGER_POSITION_BITS "${COSMICTIGER_POSITION_TYPE}")
option(COSMICTIGER_PROFILE "Compile the manual phase profiler (profileBegin/profileEnd)" ON)
option(COSMICTIGER_DEVICE_PROFILE "Compile per-block CUDA gravity cycle profiling" OFF)
if(COSMICTIGER_DEVICE_PROFILE AND (NOT COSMICTIGER_PROFILE OR NOT COSMICTIGER_WITH_CUDA))
    message(FATAL_ERROR "COSMICTIGER_DEVICE_PROFILE requires COSMICTIGER_PROFILE and COSMICTIGER_WITH_CUDA")
endif()
set(cosmictiger_require_apex_default OFF)
if(CMAKE_BUILD_TYPE MATCHES "^(RelWithDebInfo|Debug|Profile)$")
    set(cosmictiger_require_apex_default ON)
endif()
option(COSMICTIGER_REQUIRE_APEX "Require HPX task profiling through an APEX-enabled HPX installation"
    ${cosmictiger_require_apex_default})
option(COSMICTIGER_WITH_LONG_LONG_PART_INT "Use long long to index local particles" OFF)
include(cmake/features.cmake)
include(cmake/matrix.cmake)
# Compile-time cap on the tree leaf bucket. compute.bucketSize may be any
# runtime value up to this power of two; leaves then hold between half the
# runtime bucket plus one and the runtime bucket particles. The CUDA kick
# shared-memory policy is tuned for a runtime bucket equal to this cap.
set(COSMICTIGER_WITH_MAX_BUCKET "128" CACHE STRING "Maximum gravity tree leaf bucket size (power of two)")
if(NOT COSMICTIGER_WITH_MAX_BUCKET MATCHES "^(32|64|128|256|512|1024)$")
    message(FATAL_ERROR
        "COSMICTIGER_WITH_MAX_BUCKET must be a power of two from 32 to 1024; got '${COSMICTIGER_WITH_MAX_BUCKET}'")
endif()
if(COSMICTIGER_WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
if(COSMICTIGER_REQUIRE_APEX AND NOT HPX_WITH_APEX)
    message(FATAL_ERROR
        "APEX task profiling requires HPX built with HPX_WITH_APEX=ON. "
        "Select that installation with HPX_DIR or CMAKE_PREFIX_PATH; setting "
        "HPX_WITH_APEX on CosmicTiger cannot enable it in an installed HPX. "
        "To explicitly build without task profiling, set COSMICTIGER_REQUIRE_APEX=OFF.")
endif()
if(HPX_WITH_APEX)
    message(STATUS "CosmicTiger HPX task profiling: APEX enabled (HPX owns task lifecycle instrumentation)")
else()
    message(STATUS "CosmicTiger HPX task profiling: APEX disabled in the selected HPX installation")
endif()
add_library(cosmictiger_profiler STATIC src/core/profiler.cpp src/core/interrupt_profile.cpp
    src/core/profiling_runtime.cpp)
target_include_directories(cosmictiger_profiler PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(cosmictiger_profiler PUBLIC cxx_std_20)
target_link_libraries(cosmictiger_profiler PUBLIC HPX::hpx ${CMAKE_DL_LIBS}
    "$<$<PLATFORM_ID:Linux>:rt>")
if(COSMICTIGER_PROFILE)
    target_compile_definitions(cosmictiger_profiler PUBLIC COSMICTIGER_PROFILE=1)
endif()
if(COSMICTIGER_DEVICE_PROFILE)
    target_compile_definitions(cosmictiger_profiler PUBLIC COSMICTIGER_DEVICE_PROFILE=1)
endif()
if(COSMICTIGER_WITH_CUDA AND NOT HPX_WITH_CUDA)
    message(FATAL_ERROR
        "CUDA builds require HPX built with HPX_WITH_CUDA=ON; otherwise configure COSMICTIGER_WITH_CUDA=OFF")
endif()
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
add_library(cosmictiger_stacktrace STATIC src/core/stack_trace.cpp)
target_include_directories(cosmictiger_stacktrace PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(cosmictiger_stacktrace PRIVATE Boost::boost ${CMAKE_DL_LIBS})
if(NOT MSVC)
    # Unlike older addr2line backends, libbacktrace relocates PIE addresses.
    find_package(Libbacktrace REQUIRED)
    target_link_libraries(cosmictiger_stacktrace PRIVATE Libbacktrace::libbacktrace)
    target_compile_definitions(cosmictiger_stacktrace PRIVATE BOOST_STACKTRACE_USE_BACKTRACE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cosmictiger_stacktrace PUBLIC
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>")
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
if(COSMICTIGER_WITH_GRAVITY)
    find_package(GSL REQUIRED)
    find_path(FFTW3F_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED)
    find_library(FFTW3F_THREADS_LIBRARY NAMES fftw3f_threads REQUIRED)
    find_path(FFTW3F_MPI_INCLUDE_DIR fftw3-mpi.h HINTS ${FFTW3F_INCLUDE_DIR} REQUIRED)
    find_library(FFTW3F_MPI_LIBRARY NAMES fftw3f_mpi REQUIRED)

    # Check the threaded single-precision MPI API without running it during
    # configuration. Keep MPI and threads before their serial FFTW dependency.
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${FFTW3F_INCLUDE_DIR} ${FFTW3F_MPI_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${FFTW3F_MPI_LIBRARY} ${FFTW3F_THREADS_LIBRARY}
        ${FFTW3F_LIBRARY} MPI::MPI_CXX Threads::Threads)
    check_cxx_source_compiles([=[
#include <fftw3-mpi.h>

int main() {
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (!fftwf_init_threads()) {
        return 1;
    }
    fftwf_mpi_init();
    fftwf_threads_set_callback(nullptr, nullptr);
    fftwf_plan_with_nthreads(2);
    fftwf_mpi_cleanup();
    fftwf_cleanup_threads();
    return provided;
}
]=] COSMICTIGER_FFTW3F_MPI_CALLBACK_LINKS)
    cmake_pop_check_state()
    if(NOT COSMICTIGER_FFTW3F_MPI_CALLBACK_LINKS)
        message(FATAL_ERROR
            "FFTW3 single-precision MPI and thread callback support must compile and link with the selected MPI installation")
    endif()
endif()

find_path(SILO_INCLUDE_DIR silo.h REQUIRED)
find_library(SILO_LIBRARY NAMES siloh5 silo REQUIRED)
if(COSMICTIGER_WITH_GRAVITY)
    find_path(CHEALPIX_INCLUDE_DIR chealpix.h REQUIRED)
    find_library(CHEALPIX_LIBRARY NAMES chealpix REQUIRED)
endif()
if(COSMICTIGER_WITH_ROCKSTAR)
    find_path(HEALPIX_CXX_INCLUDE_DIR healpix_base.h PATH_SUFFIXES healpix_cxx REQUIRED)
    find_library(HEALPIX_CXX_LIBRARY NAMES healpix_cxx REQUIRED)
endif()

message(STATUS "cosmictiger SPH = ${COSMICTIGER_WITH_SPH}")
message(STATUS "cosmictiger ROCKSTAR = ${COSMICTIGER_WITH_ROCKSTAR} (requires gravity)")
message(STATUS "cosmictiger equilibrium cooling = ${COSMICTIGER_WITH_CHEMISTRY} (requires SPH)")
message(STATUS "cosmictiger stars = ${COSMICTIGER_WITH_STARS} (COSMICTIGER_ENABLE_STARS; requires chemistry and SPH)")
message(STATUS "cosmictiger position storage = ${COSMICTIGER_POSITION_TYPE}")
message(STATUS "cosmictiger integration: hierarchical short gravity, outer long kicks, midpoint SPH")
message(STATUS "cosmictiger gravity = ${COSMICTIGER_WITH_GRAVITY} (PMMM: FMM near field with PP fallback)")
if(COSMICTIGER_WITH_GRAVITY)
    message(STATUS "cosmictiger potential expansion degrees = ${COSMICTIGER_EXPANSION_DEGREE}")
endif()
message(STATUS "cosmictiger maximum bucket size = ${COSMICTIGER_WITH_MAX_BUCKET}")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

include(cmake/core_sources.cmake)
include(cmake/variant_sources.cmake)

set(cosmictiger_header_files
    cosmictiger/analysis/compress.hpp
    cosmictiger/analysis/group_entry.hpp
    cosmictiger/analysis/group_header.hpp
    cosmictiger/analysis/group_tree.hpp
    cosmictiger/analysis/groups.hpp
    cosmictiger/analysis/groups_find.hpp
    cosmictiger/analysis/lightcone.hpp
    cosmictiger/analysis/lightcone_entry.hpp
    cosmictiger/analysis/power.hpp
    cosmictiger/analysis/power_cpu.hpp
    cosmictiger/analysis/power_cuda.hpp
    cosmictiger/analysis/rockstar.hpp
    cosmictiger/analysis/rockstar_core.hpp
    cosmictiger/analysis/rockstar_properties.hpp
    cosmictiger/analysis/rockstar_record.hpp
    cosmictiger/chemistry/gadget4_cooling.hpp
    cosmictiger/chemistry/gadget4_cooling_impl.hpp
    cosmictiger/chemistry/recombination.hpp
    cosmictiger/core/assert.hpp
    cosmictiger/core/containers.hpp
    cosmictiger/core/defs.hpp
    cosmictiger/core/driver.hpp
    cosmictiger/core/force_step.hpp
    cosmictiger/core/diagnostic_output.hpp
    cosmictiger/core/fixed_capacity_vector.hpp
    cosmictiger/core/flops.hpp
    cosmictiger/core/hpx.hpp
    cosmictiger/core/memused.hpp
    cosmictiger/core/options.hpp
    cosmictiger/core/options_metadata.hpp
    cosmictiger/core/profiler.hpp
    cosmictiger/core/safe_io.hpp
    cosmictiger/core/semaphore.hpp
    cosmictiger/core/stack_trace.hpp
    cosmictiger/core/stack_vector.hpp
    cosmictiger/core/time.hpp
    cosmictiger/core/timer.hpp
    cosmictiger/core/unordered_set_ts.hpp
    cosmictiger/cosmology/boltzmann.hpp
    cosmictiger/cosmology/background.hpp
    cosmictiger/cosmology/background_plot.hpp
    cosmictiger/cosmology/evolution.hpp
    cosmictiger/cosmology/friedmann.hpp
    cosmictiger/cosmology/cosmology.hpp
    cosmictiger/cosmology/fft.hpp
    cosmictiger/cosmology/initialize.hpp
    cosmictiger/cosmology/lpt.hpp
    cosmictiger/cosmology/recombination_thermal.hpp
    cosmictiger/cosmology/native_recombination.hpp
    cosmictiger/cosmology/zero_order.hpp
    cosmictiger/cuda/cuda.hpp
    cosmictiger/cuda/cuda_export.hpp
    cosmictiger/cuda/cuda_memory.hpp
    cosmictiger/cuda/cuda_memory_advice.hpp
    cosmictiger/cuda/cuda_reduce.hpp
    cosmictiger/cuda/cuda_stream.hpp
    cosmictiger/cuda/cuda_unordered_map.hpp
    cosmictiger/cuda/device_vector.hpp
    cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp
    cosmictiger/gravity/fmm_acceptance.hpp
    cosmictiger/gravity/softening.hpp
    cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp
    cosmictiger/gravity/fmm_order.hpp
    cosmictiger/gravity/fmm_types.hpp
    cosmictiger/gravity/kernels/spherical_types.hpp
    cosmictiger/analysis/bh.hpp
    cosmictiger/gravity/fmm_cpu_instances.hpp
    cosmictiger/gravity/gravity.hpp
    cosmictiger/gravity/gravity_cpu.hpp
    cosmictiger/gravity/gravity_cuda.hpp
    cosmictiger/gravity/kick.hpp
    cosmictiger/gravity/kick_cuda.hpp
    cosmictiger/gravity/kick_cuda_instances.hpp
    cosmictiger/gravity/kick_cuda_kernel.hpp
    cosmictiger/gravity/kick_cuda_launch.hpp
    cosmictiger/gravity/kick_interaction_counts.hpp
    cosmictiger/gravity/kick_walk.hpp
    cosmictiger/gravity/kick_workspace.hpp
    cosmictiger/gravity/tree.hpp
    cosmictiger/gravity/tree_multipole.hpp
    cosmictiger/io/checkpoint.hpp
    cosmictiger/io/output.hpp
    cosmictiger/io/view.hpp
    cosmictiger/math/complex.hpp
    cosmictiger/math/constants.hpp
    cosmictiger/math/fixed.hpp
    cosmictiger/math/fp16.hpp
    cosmictiger/math/interpolation.hpp
    cosmictiger/math/kernel.hpp
    cosmictiger/math/math.hpp
    cosmictiger/math/matrix.hpp
    cosmictiger/math/matrix_eigen.hpp
    cosmictiger/math/position.hpp
    cosmictiger/math/radial_density_kernel.hpp
    cosmictiger/math/radial_kernel.hpp
    cosmictiger/math/range.hpp
    cosmictiger/math/simd_backend.hpp
    cosmictiger/math/simd_math.hpp
    cosmictiger/math/simulation_units.hpp
    cosmictiger/math/units.hpp
    cosmictiger/math/vector3.hpp
    cosmictiger/particles/domain.hpp
    cosmictiger/particles/drift.hpp
    cosmictiger/particles/drift_cpu.hpp
    cosmictiger/particles/particle_selection.hpp
    cosmictiger/particles/particles.hpp
    cosmictiger/scenarios/scenario.hpp
    cosmictiger/scenarios/plummer.hpp
    cosmictiger/scenarios/sod.hpp
    cosmictiger/scenarios/blast.hpp
    cosmictiger/scenarios/rayleigh_taylor.hpp
    cosmictiger/math/cell_lattice.hpp
    cosmictiger/scenarios/kelvin_helmholtz_lattice.hpp
    cosmictiger/scenarios/kelvin_helmholtz.hpp
    cosmictiger/scenarios/disc.hpp
    cosmictiger/scenarios/polytropic_star.hpp
    cosmictiger/scenarios/hydro_wave.hpp
    cosmictiger/scenarios/jeans_wave.hpp
    cosmictiger/scenarios/cosmology.hpp
    cosmictiger/scenarios/glass.hpp
    cosmictiger/scenarios/exact_sod.hpp
    cosmictiger/scenarios/sedov.hpp
    cosmictiger/sph/sph.hpp
    cosmictiger/chemistry/sph.hpp
    cosmictiger/sph/cpu.hpp
    cosmictiger/sph/cuda.hpp
    cosmictiger/sph/neighbor_buffer.hpp
    cosmictiger/sph/run_data.hpp
    cosmictiger/sph/particles.hpp
    cosmictiger/sph/particle_flags.hpp
    cosmictiger/sph/tree.hpp
    cosmictiger/sph/parent_tree.hpp
    cosmictiger/sph/ghosts.hpp
    cosmictiger/scenarios/sphere.hpp
    cosmictiger/core/tree_id.hpp
    cosmictiger/gravity/fmm_geometry.hpp
    cosmictiger/gravity/kick_context.hpp
    cosmictiger/gravity/fmm/cell_tree.hpp
    cosmictiger/gravity/collective_dispatch.hpp
    cosmictiger/gravity/fmm/fmm_batch.hpp
    cosmictiger/gravity/fmm/ghost_storage.hpp
    cosmictiger/gravity/pmmm_kernels.hpp
    cosmictiger/gravity/pmmm_kick.hpp
    cosmictiger/gravity/pp_near_pp.hpp
    cosmictiger/gravity/solver.hpp
    cosmictiger/gravity/source_access.hpp
    cosmictiger/gravity/sources.hpp
    cosmictiger/gravity/tree_build.hpp
    cosmictiger/sph/viscosity/balsara.hpp
    cosmictiger/sph/viscosity/monaghan.hpp
    cosmictiger/sph/viscosity/morris_and_monaghan.hpp
    cosmictiger/sph/flops.hpp
    cosmictiger/sph/variable_smoothing_lengths.hpp
    cosmictiger/sph/volume.hpp
    cosmictiger/sph/thermodynamics.hpp
    cosmictiger/math/float16_storage.hpp)

# Completed midpoint intervals apply cooling and star formation to entropy.
set(cosmictiger_entropy_sources_ready ON)
if(BUILD_TESTING AND COSMICTIGER_DIMENSION_COUNT EQUAL 3)
    set(cosmictiger_rockstar_test_args)
    if(NOT COSMICTIGER_WITH_ROCKSTAR)
        list(APPEND cosmictiger_rockstar_test_args --without-rockstar)
    endif()
    set(cosmictiger_chemistry_test_args)
    if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_ENABLE_CHEMISTRY OR NOT cosmictiger_entropy_sources_ready)
        list(APPEND cosmictiger_chemistry_test_args --without-chemistry)
    endif()
    # Help and build metadata describe compiled scalar capabilities.
    set(cosmictiger_options_capability_args)
    if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_ENABLE_CHEMISTRY)
        list(APPEND cosmictiger_options_capability_args --without-chemistry)
    endif()
    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_STAR_FORMATION)
        list(APPEND cosmictiger_options_capability_args --with-star-formation)
    endif()
    if(NOT cosmictiger_entropy_sources_ready)
        list(APPEND cosmictiger_options_capability_args --without-entropy-sources)
    endif()
    # Only option validation and dispatch depend on test availability.
    set_property(SOURCE src/main.cpp src/core/options.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS COSMICTIGER_BUILD_TESTING)
endif()

if(COSMICTIGER_WITH_ROCKSTAR)
    # Keep the C++23 standard half type in one host-only translation unit. The
    # public two-byte storage type remains usable by the C++20/CUDA application.
    add_library(cosmictiger_fp16 STATIC src/math/fp16.cpp)
    set_target_properties(cosmictiger_fp16 PROPERTIES
        CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_include_directories(cosmictiger_fp16 PUBLIC ${PROJECT_SOURCE_DIR})
    target_compile_options(cosmictiger_fp16 PRIVATE -fno-fast-math -ffp-contract=off)

    add_executable(halo_catalog_export src/tools/halo_catalog_export.cpp)
    target_link_libraries(halo_catalog_export PRIVATE cosmictiger_fp16)
    target_compile_options(halo_catalog_export PRIVATE -fno-fast-math)
endif()

# Equilibrium cooling must not inherit the application's fast-math/FTZ settings: the
# scalar cooling arithmetic uses matching IEEE semantics on CPU and CUDA.
target_link_libraries(cosmictiger_profiler PRIVATE Threads::Threads)
target_link_libraries(cosmictiger_stacktrace PRIVATE Threads::Threads)
# Homogeneous recombination supplies the gravity-enabled cosmological background.
# Particle cooling is independent and requires both SPH and chemistry.
if(COSMICTIGER_WITH_GRAVITY)
    add_library(cosmictiger_recombination STATIC src/chemistry/recombination.cpp)
    target_include_directories(cosmictiger_recombination PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries(cosmictiger_recombination PUBLIC Threads::Threads Boost::boost)
    target_compile_options(cosmictiger_recombination PRIVATE -fno-fast-math -ffp-contract=off)
endif()
if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_CHEMISTRY)
    add_library(cosmictiger_sph_chemistry STATIC src/chemistry/gadget4_cooling.cpp)
    target_include_directories(cosmictiger_sph_chemistry PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries(cosmictiger_sph_chemistry PUBLIC Threads::Threads Boost::boost)
    target_compile_options(cosmictiger_sph_chemistry PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
    if(COSMICTIGER_WITH_CUDA)
        target_sources(cosmictiger_sph_chemistry PRIVATE src/chemistry/gadget4_cooling.cu)
        target_compile_definitions(cosmictiger_sph_chemistry PRIVATE USE_CUDA)
        target_link_libraries(cosmictiger_sph_chemistry PUBLIC CUDA::cudart)
        target_compile_options(cosmictiger_sph_chemistry PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off>)
    endif()
endif()
set_source_files_properties(src/chemistry/sph.cpp src/io/view.cpp tests/sph/chemistry.cpp
    src/stars/star_formation.cpp tests/sph/star_formation.cpp tests/sph/source_timing.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Star-formation device physics shares the chemistry rounding contract.
set_source_files_properties(src/stars/star_formation_cuda.cu PROPERTIES COMPILE_OPTIONS
    "--expt-relaxed-constexpr;--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off")

# The adaptive Boltzmann solve needs finite-value checks and accurate
# cancellation in the synchronous Einstein constraints.
set_source_files_properties(src/cosmology/boltzmann.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Background and clock guards must remain effective in release builds.
set_source_files_properties(src/core/driver.cpp src/gravity/error.cpp src/gravity/reference_table.cpp
    src/gravity/solver.cpp src/gravity/pmmm_kernels.cpp src/gravity/pp/near_pp.cpp tests/gravity/pmmm.cpp
    src/gravity/pmmm_kick.cpp src/cosmology/friedmann.cpp src/cosmology/cosmology.cpp
    src/cosmology/background.cpp src/cosmology/evolution.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Keep the diagnostic's compensated double sums intact on both host and device.
set_source_files_properties(src/gravity/error_cuda.cu PROPERTIES COMPILE_OPTIONS
    "--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off")
set_source_files_properties(
    src/gravity/fft.cpp
    src/gravity/mesh_exchange.cpp
    src/gravity/fmm/ghost_exchange.cpp
    src/gravity/fmm/cell_schedule.cpp
    src/gravity/evaluation.cpp
    src/gravity/pm_kick.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Domain boundary coordinates must round identically when stored and reloaded.
set_source_files_properties(src/particles/domain.cpp tests/mesh_domains.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")

# Checked-in kernels for potential degrees 2 through 8, regenerated only on
# explicit request. Cartesian FMM and spherical PMMM share this degree range.
# Ordinary builds use the source headers.
set(COSMICTIGER_GENERATED_FMM_DIR ${PROJECT_SOURCE_DIR}/cosmictiger/gravity/kernels)
set(COSMICTIGER_GENERATED_FMM_FILES
    ${COSMICTIGER_GENERATED_FMM_DIR}/basis_conversion.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree_config.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree_types.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/geometry.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/p2m.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/p2l.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/m2m.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/m2l.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/m2p.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/l2l.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/l2p.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_tracefree/l2l_warp.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical_ewald_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/p2m.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/l2p.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/green.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/ewald_common.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/ewald_green.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical/ewald_l2p.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/operation_counts.json)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(COSMICTIGER_WITH_GRAVITY)
    # Embed the renderer so copied executables do not depend on the source tree.
    # Matplotlib is needed on the execution host only; missing runtime support
    # produces a warning while preserving the background samples and simulation.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/scripts/plot_cosmology.py)
    file(READ ${PROJECT_SOURCE_DIR}/scripts/plot_cosmology.py cosmictiger_background_plot_script)
    configure_file(cmake/background_plot_script.hpp.in
        ${PROJECT_BINARY_DIR}/generated/cosmictiger/background_plot_script.hpp @ONLY)
    set_source_files_properties(src/cosmology/background_plot.cpp PROPERTIES
        INCLUDE_DIRECTORIES "${PROJECT_BINARY_DIR}/generated")
endif()
# Do not declare source headers as OUTPUT/BYPRODUCTS or mark them GENERATED:
# build-system clean targets must never delete these version-controlled files.
add_custom_target(generate_fmm_kernels
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_fmm.py
        --output-dir ${COSMICTIGER_GENERATED_FMM_DIR} --min-order 2 --max-order 8
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Regenerating Cartesian and Spherical kernels for potential degrees 2 through 8"
    VERBATIM)
add_custom_target(generate_cartesian_kernels DEPENDS generate_fmm_kernels)
add_custom_target(cosmictiger_generated_kernels DEPENDS generate_fmm_kernels)
# Compile a target against the generated kernels and selected order/bucket cap.
function(cosmictiger_use_generated_kernels target)
    target_include_directories(${target} BEFORE PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(${target} PRIVATE ORDER=${cosmictiger_default_fmm_order}
        MAX_BUCKET_SIZE=${COSMICTIGER_WITH_MAX_BUCKET})
endfunction()
include(cmake/variants.cmake)

if(COSMICTIGER_WITH_GRAVITY)
    add_executable(healpix2silo src/tools/healpix2silo.cpp)
    target_link_libraries(healpix2silo PRIVATE ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} Boost::program_options dl)
    target_include_directories(healpix2silo PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(healpix2silo SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})

    set(cosmictiger_analysis_tools healpix2silo)
    if(COSMICTIGER_WITH_ROCKSTAR)
        add_hpx_executable(groups2halofunc
            SOURCES src/core/assert.cpp src/core/hpx.cpp src/tools/groups2halofunc.cpp src/analysis/group_entry.cpp)
        target_link_libraries(groups2halofunc PUBLIC Boost::program_options PRIVATE cosmictiger_stacktrace)
        set_target_properties(groups2halofunc PROPERTIES ENABLE_EXPORTS ON)
        target_include_directories(groups2halofunc PRIVATE ${PROJECT_SOURCE_DIR})
        list(APPEND cosmictiger_analysis_tools groups2halofunc)
        target_link_libraries(halo_catalog_export PRIVATE cosmictiger_default_config)
    endif()

    foreach(tool_target IN LISTS cosmictiger_analysis_tools)
        target_compile_options(${tool_target} PRIVATE -Wno-attributes -ffast-math)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(${tool_target} PRIVATE -march=native)
        endif()
    endforeach()

    foreach(tool_target IN LISTS cosmictiger_analysis_tools)
        target_link_libraries(${tool_target} PRIVATE cosmictiger_default_config)
    endforeach()
endif()

# Capture standalone test targets so they inherit the same capability definitions.
get_property(cosmictiger_targets_before_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)

# Register distributed regressions only when the configured launcher can run them.
set(cosmictiger_mpi_tests FALSE)
set(cosmictiger_mpi_three_ranks FALSE)
set(cosmictiger_mpi_four_ranks FALSE)
if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
    if(NOT DEFINED MPIEXEC_MAX_NUMPROCS OR MPIEXEC_MAX_NUMPROCS GREATER_EQUAL 2)
        set(cosmictiger_mpi_tests TRUE)
    endif()
    if(NOT DEFINED MPIEXEC_MAX_NUMPROCS OR MPIEXEC_MAX_NUMPROCS GREATER_EQUAL 3)
        set(cosmictiger_mpi_three_ranks TRUE)
    endif()
    if(NOT DEFINED MPIEXEC_MAX_NUMPROCS OR MPIEXEC_MAX_NUMPROCS GREATER_EQUAL 4)
        set(cosmictiger_mpi_four_ranks TRUE)
    endif()
endif()

if(BUILD_TESTING AND COSMICTIGER_DIMENSION_COUNT EQUAL 3)
    foreach(profile_mode IN ITEMS enabled disabled host_only)
        add_executable(phase_profiler_${profile_mode}_test
            tests/phase_profiler.cpp src/core/profiler.cpp)
        target_include_directories(phase_profiler_${profile_mode}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_features(phase_profiler_${profile_mode}_test PRIVATE cxx_std_20)
        target_link_libraries(phase_profiler_${profile_mode}_test PRIVATE Threads::Threads)
        target_compile_definitions(phase_profiler_${profile_mode}_test PRIVATE
            COSMICTIGER_PROFILE=$<IF:$<STREQUAL:${profile_mode},disabled>,0,1>
            COSMICTIGER_DEVICE_PROFILE=$<IF:$<STREQUAL:${profile_mode},host_only>,0,1>)
        add_test(NAME phase_profiler_${profile_mode}
            COMMAND phase_profiler_${profile_mode}_test
                ${PROJECT_BINARY_DIR}/phase_profiler_${profile_mode}.txt)
        set_tests_properties(phase_profiler_${profile_mode} PROPERTIES
            TIMEOUT 10 PROCESSORS 1 LABELS "profiler;cpu")
    endforeach()
    if(COSMICTIGER_WITH_CUDA)
        foreach(profile_mode IN ITEMS enabled disabled)
            add_executable(device_profiler_${profile_mode}_test tests/device_profiler.cu)
            target_include_directories(device_profiler_${profile_mode}_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(device_profiler_${profile_mode}_test PRIVATE
                COSMICTIGER_PROFILE=1
                COSMICTIGER_DEVICE_PROFILE=$<IF:$<STREQUAL:${profile_mode},enabled>,1,0>)
            target_link_libraries(device_profiler_${profile_mode}_test PRIVATE CUDA::cudart)
            set_target_properties(device_profiler_${profile_mode}_test PROPERTIES CUDA_STANDARD 20)
            add_test(NAME device_profiler_${profile_mode} COMMAND device_profiler_${profile_mode}_test)
            set_tests_properties(device_profiler_${profile_mode} PROPERTIES
                TIMEOUT 30 PROCESSORS 1 RUN_SERIAL TRUE SKIP_RETURN_CODE 77 LABELS "profiler;cuda")
        endforeach()
        if(COSMICTIGER_WITH_GRAVITY)
            foreach(profile_mode IN ITEMS enabled disabled)
                add_executable(kick_device_profile_${profile_mode}_test
                    tests/gravity/kick_device_profile.cu tests/cuda_memory_backend.cpp
                    src/core/flops.cu src/math/kernel.cu src/cuda/cuda_memory.cu src/core/assert.cu)
                target_include_directories(kick_device_profile_${profile_mode}_test PRIVATE
                    ${PROJECT_SOURCE_DIR})
                target_compile_definitions(kick_device_profile_${profile_mode}_test PRIVATE
                    USE_CUDA CUDA_MEM_STACK_SIZE=128 COSMICTIGER_PROFILE=1
                    COSMICTIGER_DEVICE_PROFILE=$<IF:$<STREQUAL:${profile_mode},enabled>,1,0>)
                target_compile_options(kick_device_profile_${profile_mode}_test PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)
                target_link_libraries(kick_device_profile_${profile_mode}_test PRIVATE
                    CUDA::cudart HPX::hpx)
                set_target_properties(kick_device_profile_${profile_mode}_test PROPERTIES
                    CUDA_STANDARD 20 CUDA_SEPARABLE_COMPILATION ON)
                add_test(NAME kick_device_profile_${profile_mode}
                    COMMAND kick_device_profile_${profile_mode}_test)
                set_tests_properties(kick_device_profile_${profile_mode} PROPERTIES
                    TIMEOUT 60 PROCESSORS 1 RUN_SERIAL TRUE SKIP_RETURN_CODE 77
                    LABELS "profiler;cuda;gravity")
            endforeach()
        endif()
    endif()
    if(COSMICTIGER_PROFILE AND COSMICTIGER_WITH_SPH)
        set(cosmictiger_phase_profile_args)
        if(COSMICTIGER_WITH_GRAVITY)
            list(APPEND cosmictiger_phase_profile_args --gravity)
        endif()
        add_test(NAME phase_profile_application COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_phase_profile.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/phase_profile ${cosmictiger_phase_profile_args})
        set_tests_properties(phase_profile_application PROPERTIES
            TIMEOUT 150 PROCESSORS 2 LABELS "profiler;cpu;application")
    endif()
    add_test(NAME build_features COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_build_features.py ${PROJECT_BINARY_DIR})
    set_tests_properties(build_features PROPERTIES TIMEOUT 30 LABELS "build")
    add_test(NAME build_variants_checker COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_build_variants.py)
    set_tests_properties(build_variants_checker PROPERTIES TIMEOUT 30 LABELS "build")
    add_test(NAME python_regressions COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/test_python_regressions.py)
    add_test(NAME python_regressions_optimized COMMAND ${Python3_EXECUTABLE} -O
        ${PROJECT_SOURCE_DIR}/tests/test_python_regressions.py)
    set_tests_properties(python_regressions python_regressions_optimized PROPERTIES
        TIMEOUT 60 PROCESSORS 1 LABELS "python")
    set(cosmictiger_startup_test_args)
    if(NOT COSMICTIGER_WITH_SPH)
        list(APPEND cosmictiger_startup_test_args --without-sph)
    endif()
    add_test(NAME startup_options COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_options.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/startup_options --startup-options-only --concurrent-cases=1

--order ${cosmictiger_default_fmm_order}
        --time-integrator ${cosmictiger_default_integrator}
        ${cosmictiger_options_capability_args}
        ${cosmictiger_startup_test_args})
    set_tests_properties(startup_options PROPERTIES TIMEOUT 120 PROCESSORS 2
        LABELS "build;cpu")
    set(COSMICTIGER_TEST_CUDA_EXECUTABLE "" CACHE FILEPATH
        "Optional second CUDA-enabled build for cross-build SPH comparisons")
    if(COSMICTIGER_TEST_CUDA_EXECUTABLE AND NOT EXISTS "${COSMICTIGER_TEST_CUDA_EXECUTABLE}")
        message(FATAL_ERROR "COSMICTIGER_TEST_CUDA_EXECUTABLE does not exist")
    endif()
    set(cosmictiger_mpi_transport_args --hpx:ini=hpx.parcel.mpi.enable=1)
    if(HPX_WITH_PARCELPORT_TCP)
        list(APPEND cosmictiger_mpi_transport_args --hpx:ini=hpx.parcel.tcp.enable=0)
    endif()
    set(cosmictiger_python_mpi_args --mpi-exec=${MPIEXEC_EXECUTABLE}
        --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
        list(APPEND cosmictiger_python_mpi_args "--mpi-preflag=${flag}")
    endforeach()
    foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
        list(APPEND cosmictiger_python_mpi_args "--mpi-postflag=${flag}")
    endforeach()
    foreach(flag IN LISTS cosmictiger_mpi_transport_args)
        list(APPEND cosmictiger_python_mpi_args "--hpx-argument=${flag}")
    endforeach()
    # Test fixtures select algorithms explicitly so mixed-capability builds
    # exercise their intended path without changing normal option validation.
    set(cosmictiger_test_integrator_args --run.timeIntegrator=${cosmictiger_default_integrator})
    # SPH with PMMM self-gravity runs only on the hierarchical ladder.
    set(cosmictiger_sph_gravity_integrator_args ${cosmictiger_test_integrator_args})
    if(COSMICTIGER_WITH_GRAVITY)
        set(cosmictiger_sph_gravity_integrator_args --run.timeIntegrator=hierarchical)
    endif()
    # PMMM is the only gravity solver; its near field defaults to fmm.
    set(cosmictiger_test_gravity_args)
    if(COSMICTIGER_WITH_GRAVITY)
        set(cosmictiger_test_gravity_args --gravity.meshSize=8)
    endif()
    # These SPH fixtures exercise hydrodynamics without self-gravity.
    set(cosmictiger_test_hydro_args --hydro.enabled=true --gravity.enabled=false)
    add_library(cosmictiger_application_tests STATIC tests/application_test.cpp)
    target_include_directories(cosmictiger_application_tests PRIVATE ${PROJECT_SOURCE_DIR})
    foreach(application_target IN LISTS cosmictiger_application_targets)
        target_link_libraries(${application_target} PRIVATE cosmictiger_application_tests)
    endforeach()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(googletest)
    endif()
    include(GoogleTest)

    if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_executable(stack_trace_test tests/stack_trace.cpp)
        target_link_libraries(stack_trace_test PRIVATE cosmictiger_stacktrace Threads::Threads)
        target_compile_options(stack_trace_test PRIVATE -g -O0 -fno-optimize-sibling-calls)
        set_target_properties(stack_trace_test PROPERTIES ENABLE_EXPORTS ON)
        add_test(NAME stack_trace COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_stack_trace.py $<TARGET_FILE:stack_trace_test>
            ${PROJECT_BINARY_DIR}/stack_trace_test_output)
        set_tests_properties(stack_trace PROPERTIES TIMEOUT 90 LABELS "diagnostics;cpu")

        add_hpx_executable(stack_trace_hpx_test SOURCES tests/stack_trace_hpx.cpp)
        target_link_libraries(stack_trace_hpx_test PRIVATE cosmictiger_stacktrace)
        target_compile_options(stack_trace_hpx_test PRIVATE -g -O0 -fno-optimize-sibling-calls)
        set_target_properties(stack_trace_hpx_test PROPERTIES ENABLE_EXPORTS ON)
        add_test(NAME stack_trace_hpx COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_stack_trace_hpx.py $<TARGET_FILE:stack_trace_hpx_test>
            ${PROJECT_BINARY_DIR}/stack_trace_hpx_test_output)
        set_tests_properties(stack_trace_hpx PROPERTIES TIMEOUT 120 PROCESSORS 2
            LABELS "diagnostics;cpu;hpx")
        if(cosmictiger_mpi_tests)
            set(stack_trace_mpi_args --mpi-exec=${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG} --mpi-only)
            foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                list(APPEND stack_trace_mpi_args --mpi-preflag=${flag})
            endforeach()
            foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                list(APPEND stack_trace_mpi_args --mpi-postflag=${flag})
            endforeach()
            add_test(NAME stack_trace_hpx_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_stack_trace_hpx.py $<TARGET_FILE:stack_trace_hpx_test>
                ${PROJECT_BINARY_DIR}/stack_trace_hpx_mpi_test_output ${stack_trace_mpi_args})
            set_tests_properties(stack_trace_hpx_mpi PROPERTIES TIMEOUT 120 PROCESSORS 4
                LABELS "diagnostics;cpu;hpx;mpi")
        endif()
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME mpi_interrupt_launcher COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_mpi_launcher.py ${PROJECT_SOURCE_DIR}/scripts/run_mpi.py)
        set_tests_properties(mpi_interrupt_launcher PROPERTIES TIMEOUT 60 PROCESSORS 2
            LABELS "profiler;interrupt;cpu")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(HPX_WITH_APEX)
            add_hpx_executable(apex_profiler_test SOURCES tests/apex_profiler.cpp)
            # Match the application's exported symbols for APEX address lookup.
            set_target_properties(apex_profiler_test PROPERTIES ENABLE_EXPORTS ON)
            target_link_libraries(apex_profiler_test PRIVATE cosmictiger_profiler)
            add_test(NAME apex_profiler COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_apex_profiler.py $<TARGET_FILE:apex_profiler_test>)
            set_tests_properties(apex_profiler
                PROPERTIES TIMEOUT 90 PROCESSORS 2
                LABELS "profiler;apex;cpu;hpx"
                ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
            add_test(NAME apex_reports_disabled COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_apex_profiler.py $<TARGET_FILE:apex_profiler_test>
                --reports-disabled)
            set_tests_properties(apex_reports_disabled PROPERTIES TIMEOUT 90 PROCESSORS 2
                LABELS "profiler;apex;cpu;hpx")
            add_test(NAME apex_interrupt COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_interrupt_profiler.py
                $<TARGET_FILE:apex_profiler_test>)
            set_tests_properties(apex_interrupt PROPERTIES
                TIMEOUT 120 PROCESSORS 2 LABELS "profiler;apex;cpu;hpx;interrupt")
            set(cosmictiger_profile_test_args)
            if(NOT COSMICTIGER_PROFILE)
                set(cosmictiger_profile_test_args --profile-disabled)
            endif()
            if(COSMICTIGER_WITH_SPH)
                add_test(NAME apex_interrupt_simulation_hydro COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_interrupt_simulation.py
                    $<TARGET_FILE:cosmictiger> --scenario=hydroWave
                    ${cosmictiger_profile_test_args})
                set_tests_properties(apex_interrupt_simulation_hydro PROPERTIES TIMEOUT 240
                    PROCESSORS 2 LABELS "profiler;apex;cpu;hpx;interrupt")
            endif()
            if(COSMICTIGER_WITH_GRAVITY AND cosmictiger_mpi_tests)
                set(apex_mpi_args --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
                foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                    list(APPEND apex_mpi_args --mpi-preflag=${flag})
                endforeach()
                foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                    list(APPEND apex_mpi_args --mpi-postflag=${flag})
                endforeach()
                add_test(NAME apex_interrupt_simulation_mpi_worker COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_interrupt_simulation.py
                    $<TARGET_FILE:cosmictiger> --scenario=plummer
                    --mpi-exec=${MPIEXEC_EXECUTABLE} ${apex_mpi_args}
                    ${cosmictiger_profile_test_args})
                set_tests_properties(apex_interrupt_simulation_mpi_worker PROPERTIES
                    TIMEOUT 240 PROCESSORS 4 LABELS "profiler;apex;cpu;hpx;mpi;interrupt")
            endif()
        endif()
    endif()

    include(cmake/test_layout.cmake)

    # Exercise the actual allocator on CPU with controlled scheduling and CUDA
    # allocation failures, including in configurations without a CUDA toolkit.
    add_executable(cuda_memory_host_test tests/cuda_memory.cpp)
    target_include_directories(cuda_memory_host_test BEFORE PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/cuda_memory_stubs ${PROJECT_SOURCE_DIR})
    target_compile_definitions(cuda_memory_host_test PRIVATE CUDA_MEM_STACK_SIZE=16)
    target_link_libraries(cuda_memory_host_test PRIVATE Threads::Threads)
    cosmictiger_add_gtest(NAME cuda_memory_host TARGET cuda_memory_host_test
        PROPERTIES TIMEOUT 60 PROCESSORS 4)

    if(COSMICTIGER_WITH_CUDA)
        add_executable(cuda_memory_cuda_test tests/cuda_memory_cuda.cu
            tests/cuda_memory_backend.cpp src/cuda/cuda_memory.cu src/core/assert.cu)
        target_include_directories(cuda_memory_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(cuda_memory_cuda_test PRIVATE USE_CUDA CUDA_MEM_STACK_SIZE=128)
        target_link_libraries(cuda_memory_cuda_test PRIVATE CUDA::cudart)
        set_target_properties(cuda_memory_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        cosmictiger_add_gtest(NAME cuda_memory_cuda TARGET cuda_memory_cuda_test
            PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
    endif()

    # The selected run.test application check executes inside the HPX runtime.
    target_link_libraries(cosmictiger_application_tests PRIVATE GTest::gtest)
    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(cosmology_flops_test tests/cosmology_flops.cpp src/cosmology/cosmology.cpp
            src/cosmology/friedmann.cpp)
        target_include_directories(cosmology_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(cosmology_flops_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME cosmology_flops TARGET cosmology_flops_test
            PROPERTIES TIMEOUT 30)
        target_compile_options(cosmology_flops_test PRIVATE -fno-fast-math -ffp-contract=off)
        add_executable(friedmann_test tests/friedmann.cpp src/cosmology/friedmann.cpp)
        target_include_directories(friedmann_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(friedmann_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME friedmann TARGET friedmann_test PROPERTIES TIMEOUT 60)

        add_executable(cosmology_evolution_test tests/cosmology_evolution.cpp)
        target_include_directories(cosmology_evolution_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(cosmology_evolution_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME cosmology_evolution TARGET cosmology_evolution_test
            PROPERTIES TIMEOUT 30)
    endif()
    add_executable(safe_io_test tests/safe_io.cpp)
    target_include_directories(safe_io_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME safe_io TARGET safe_io_test PROPERTIES TIMEOUT 10)
    if(COSMICTIGER_WITH_GRAVITY AND COSMICTIGER_WITH_SPH)
        add_test(NAME particle_count_application COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_particle_counts.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/particle_count_application)
        set_tests_properties(particle_count_application PROPERTIES RUN_SERIAL TRUE TIMEOUT 900 PROCESSORS 2)
    endif()
    add_executable(particle_count_test tests/particle_count.cpp)
    target_include_directories(particle_count_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME particle_count TARGET particle_count_test PROPERTIES TIMEOUT 10)
    add_executable(terminal_test tests/terminal.cpp)
    target_include_directories(terminal_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(terminal_test PRIVATE -ffast-math)
    cosmictiger_add_gtest(NAME terminal TARGET terminal_test PROPERTIES TIMEOUT 10)
    add_executable(rung_merge_test tests/rung_merge.cpp)
    target_include_directories(rung_merge_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(rung_merge_test PRIVATE -ffast-math)
    cosmictiger_add_gtest(NAME rung_merge TARGET rung_merge_test PROPERTIES TIMEOUT 10)
    # The collisionless checks also exercise terminal hierarchy synchronization.
    # Gas evolution uses midpoint SPH with the shared gravity ladder.
    if(COSMICTIGER_WITH_GRAVITY)
        if(COSMICTIGER_WITH_GRAVITY)
            add_test(NAME rung_merge_application COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_rung_merge.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/rung_merge_application)
            set_tests_properties(rung_merge_application
                PROPERTIES TIMEOUT 300 PROCESSORS 2 LABELS "gravity;cpu;hierarchical")
        endif()
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_hpx_executable(fftw_threads_test
            SOURCES tests/fftw_threads.cpp src/cosmology/fftw_threads.cpp)
        target_include_directories(fftw_threads_test PRIVATE ${PROJECT_SOURCE_DIR} ${FFTW3F_INCLUDE_DIR})
        target_link_libraries(fftw_threads_test PRIVATE
            ${FFTW3F_THREADS_LIBRARY} ${FFTW3F_LIBRARY} Threads::Threads)
        cosmictiger_add_gtest(NAME fftw_hpx_threads TARGET fftw_threads_test HPX_MAIN
            ARGS --hpx:threads=2 --hpx:bind=none
            PROPERTIES TIMEOUT 30 PROCESSORS 2 LABELS "fft;cpu;threads")
        add_test(NAME fftw_hpx_single_worker COMMAND $<TARGET_FILE:fftw_threads_test>
            --hpx:threads=1 --hpx:bind=none)
        set_tests_properties(fftw_hpx_single_worker PROPERTIES
            TIMEOUT 30 PROCESSORS 1 LABELS "fft;cpu;threads")
    endif()

    add_hpx_executable(flops_test SOURCES tests/flops.cpp src/core/flops.cpp)
    target_include_directories(flops_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME flops TARGET flops_test HPX_MAIN
        ARGS --hpx:threads=2 --hpx:bind=none
        PROPERTIES TIMEOUT 30 PROCESSORS 6)
    if(cosmictiger_mpi_three_ranks)
        add_test(NAME flops_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:flops_test> ${MPIEXEC_POSTFLAGS}
            --hpx:threads=2 --hpx:bind=none --hpx:localities=3)
        set_tests_properties(flops_mpi PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE PROCESSORS 10)
    endif()
    if(COSMICTIGER_WITH_CUDA)
        add_executable(flops_cuda_test tests/flops_cuda.cu src/core/flops.cu)
        target_include_directories(flops_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(flops_cuda_test PRIVATE USE_CUDA)
        target_link_libraries(flops_cuda_test PRIVATE CUDA::cudart)
        set_target_properties(flops_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        cosmictiger_add_gtest(NAME flops_cuda TARGET flops_cuda_test
            PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
    endif()

    if(COSMICTIGER_WITH_ROCKSTAR)
        add_executable(halo_catalog_fixture tests/halo_catalog_fixture.cpp)
        target_link_libraries(halo_catalog_fixture PRIVATE cosmictiger_fp16)
        target_compile_options(halo_catalog_fixture PRIVATE -fno-fast-math)
        add_test(NAME halo_viewer COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_halo_viewer.py
            --exporter $<TARGET_FILE:halo_catalog_export>
            --fixture $<TARGET_FILE:halo_catalog_fixture>
            --viewer ${PROJECT_SOURCE_DIR}/tools/halo_viewer/halo_viewer.py)
        set_tests_properties(halo_viewer PROPERTIES TIMEOUT 180 PROCESSORS 2)
    endif()

    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_STAR_FORMATION)
        add_executable(springel_and_hernquist_test tests/springel_and_hernquist.cpp)
        target_include_directories(springel_and_hernquist_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(springel_and_hernquist_test PRIVATE cosmictiger_sph_chemistry)
        target_compile_options(springel_and_hernquist_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME springel_and_hernquist TARGET springel_and_hernquist_test
            PROPERTIES TIMEOUT 120 PROCESSORS 1)
        # Coupled particle evolution shares the endpoint source contract.
        if(cosmictiger_entropy_sources_ready)
            set(source_timing_integrator hierarchical)
            if(COSMICTIGER_WITH_GRAVITY)
                set(star_formation_test_integrator hierarchical)
            else()
                set(star_formation_test_integrator ${source_timing_integrator})
            endif()
            set(star_formation_workdir "${PROJECT_BINARY_DIR}/tests/star_formation_cpu")
            file(MAKE_DIRECTORY "${star_formation_workdir}")
            add_test(NAME star_formation_cpu COMMAND cosmictiger --run.test=star_formation
                --hydro.enabled=true --starFormation.enabled=true --chemistry.enabled=true --chemistry.initializationTemperature=10000
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --run.timeIntegrator=${star_formation_test_integrator}
                ${cosmictiger_test_gravity_args}
                --hpx:threads=1 --hpx:bind=none)
            set_tests_properties(star_formation_cpu PROPERTIES
                WORKING_DIRECTORY "${star_formation_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 1)
            set(star_formation_particles_workdir "${PROJECT_BINARY_DIR}/tests/star_formation_particles_cpu")
            file(MAKE_DIRECTORY "${star_formation_particles_workdir}")
            add_test(NAME star_formation_particles_cpu COMMAND cosmictiger --run.test=starFormationParticles
                --hydro.enabled=true --starFormation.enabled=true --chemistry.enabled=true --chemistry.initializationTemperature=10000
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --run.timeIntegrator=${star_formation_test_integrator}
                ${cosmictiger_test_gravity_args}
                --hpx:threads=1 --hpx:bind=none)
            set_tests_properties(star_formation_particles_cpu PROPERTIES
                WORKING_DIRECTORY "${star_formation_particles_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 1)
            foreach(source_backend IN ITEMS cpu mpi cuda)
                if((source_backend STREQUAL "mpi" AND NOT cosmictiger_mpi_tests) OR
                        (source_backend STREQUAL "cuda" AND NOT COSMICTIGER_WITH_CUDA))
                    continue()
                endif()
                set(source_timing_name "sph_source_timing_${source_backend}")
                set(source_timing_workdir "${PROJECT_BINARY_DIR}/tests/${source_timing_name}")
                file(MAKE_DIRECTORY "${source_timing_workdir}")
                set(source_timing_launcher)
                set(source_timing_postflags)
                set(source_timing_args)
                set(source_timing_cuda false)
                if(source_backend STREQUAL "mpi")
                    set(source_timing_launcher ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS})
                    set(source_timing_postflags ${MPIEXEC_POSTFLAGS})
                    set(source_timing_args --hpx:localities=2 ${cosmictiger_mpi_transport_args})
                elseif(source_backend STREQUAL "cuda")
                    set(source_timing_cuda true)
                endif()
                add_test(NAME ${source_timing_name} COMMAND ${source_timing_launcher}
                    $<TARGET_FILE:cosmictiger> ${source_timing_postflags} --run.test=sphSourceTiming
                    ${cosmictiger_test_hydro_args} --chemistry.enabled=true --starFormation.enabled=true
                    --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --chemistry.initializationTemperature=10000
                    --gravity.cuda=${source_timing_cuda} --hydro.cuda=${source_timing_cuda}
                    --drift.cuda=${source_timing_cuda} --chemistry.cuda=${source_timing_cuda}
                    --starFormation.cuda=${source_timing_cuda}
                    --run.timeIntegrator=${source_timing_integrator}
                    ${cosmictiger_test_gravity_args}
                    --hpx:threads=1 --hpx:bind=none ${source_timing_args})
                set_tests_properties(${source_timing_name} PROPERTIES
                    WORKING_DIRECTORY "${source_timing_workdir}" RUN_SERIAL TRUE TIMEOUT 180
                    PROCESSORS 2 LABELS "sph;star_formation;${source_backend}")
            endforeach()
            if(cosmictiger_mpi_tests)
                foreach(star_mpi_fixture IN ITEMS star_formation star_formation_particles)
                    set(star_mpi_test "${star_mpi_fixture}_mpi")
                    set(star_mpi_workdir "${PROJECT_BINARY_DIR}/tests/${star_mpi_test}")
                    file(MAKE_DIRECTORY "${star_mpi_workdir}")
                    set(star_mpi_run_test star_formation)
                    set(star_mpi_gravity_args)
                    if(NOT star_formation_test_integrator STREQUAL "hierarchical")
                        set(star_mpi_gravity_args --gravity.enabled=false)
                    endif()
                    if(star_mpi_fixture STREQUAL "star_formation_particles")
                        set(star_mpi_run_test starFormationParticles)
                        set(star_mpi_gravity_args)
                    endif()
                    add_test(NAME ${star_mpi_test} COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                        --run.test=${star_mpi_run_test}
                        --hydro.enabled=true --starFormation.enabled=true --chemistry.enabled=true --chemistry.initializationTemperature=10000
                        --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                        --run.timeIntegrator=${star_formation_test_integrator}
                        ${cosmictiger_test_gravity_args}
                        ${star_mpi_gravity_args}
                        --hpx:threads=1 --hpx:localities=2 --hpx:bind=none
                        ${cosmictiger_mpi_transport_args})
                    set_tests_properties(${star_mpi_test} PROPERTIES
                        WORKING_DIRECTORY "${star_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 180
                        PROCESSORS 4 LABELS "star_formation;cpu;mpi"
                        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
                endforeach()
                add_test(NAME star_formation_ids_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_star_formation_ids.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/star_formation_ids_mpi ${cosmictiger_python_mpi_args})
                set_tests_properties(star_formation_ids_mpi PROPERTIES
                    RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2 LABELS "integration;star_formation;cpu;mpi"
                    ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
            endif()
            add_test(NAME star_formation_application COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_star_formation.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/star_formation_application)
            set_tests_properties(star_formation_application PROPERTIES
                RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2 LABELS "integration;star_formation;cpu")
            if(cosmictiger_mpi_tests)
                add_test(NAME star_formation_application_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_star_formation.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/star_formation_application_mpi
                    --skip-fixtures ${cosmictiger_python_mpi_args})
                set_tests_properties(star_formation_application_mpi PROPERTIES
                    RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2 LABELS "integration;star_formation;mpi")
            endif()
            if(COSMICTIGER_WITH_CUDA)
                add_test(NAME star_formation_application_cuda COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_star_formation.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/star_formation_application_cuda --skip-fixtures --cuda)
                set_tests_properties(star_formation_application_cuda PROPERTIES
                    RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2 LABELS "integration;star_formation;cuda")
            endif()
        endif()
        if(COSMICTIGER_WITH_CUDA)
            add_executable(star_formation_cuda_test tests/sph/star_formation_cuda.cu
                src/stars/star_formation_cuda.cu)
            target_include_directories(star_formation_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(star_formation_cuda_test PRIVATE USE_CUDA)
            target_link_libraries(star_formation_cuda_test PRIVATE cosmictiger_sph_chemistry CUDA::cudart Boost::boost)
            target_compile_options(star_formation_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off>)
            cosmictiger_add_gtest(NAME star_formation_cuda TARGET star_formation_cuda_test
                PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE PROCESSORS 1)
            if(cosmictiger_entropy_sources_ready)
                set(star_cuda_test star_formation_production_cuda)
                set(star_cuda_workdir "${PROJECT_BINARY_DIR}/tests/${star_cuda_test}")
                file(MAKE_DIRECTORY "${star_cuda_workdir}")
                add_test(NAME ${star_cuda_test} COMMAND cosmictiger --run.test=star_formation
                    --hydro.enabled=true --starFormation.enabled=true --chemistry.enabled=true
                    --chemistry.initializationTemperature=10000
                    --gravity.cuda=true --hydro.cuda=true --drift.cuda=true
                    --chemistry.cuda=true --starFormation.cuda=true --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                    --run.timeIntegrator=${star_formation_test_integrator}
                    ${cosmictiger_test_gravity_args}
                    --hpx:threads=1 --hpx:bind=none)
                set_tests_properties(${star_cuda_test} PROPERTIES
                    WORKING_DIRECTORY "${star_cuda_workdir}" RUN_SERIAL TRUE TIMEOUT 180
                    PROCESSORS 1 LABELS "star_formation;cuda")
            endif()
            add_executable(springel_and_hernquist_cuda_test tests/springel_and_hernquist_cuda.cu)
            target_include_directories(springel_and_hernquist_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(springel_and_hernquist_cuda_test PRIVATE cosmictiger_sph_chemistry CUDA::cudart)
            target_compile_options(springel_and_hernquist_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off>)
            cosmictiger_add_gtest(NAME springel_and_hernquist_cuda TARGET springel_and_hernquist_cuda_test
                PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE PROCESSORS 1)
        endif()
    endif()

    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_CHEMISTRY)
        add_executable(gadget4_cooling_test tests/gadget4_cooling.cpp)
        target_link_libraries(gadget4_cooling_test PRIVATE cosmictiger_sph_chemistry)
        target_compile_options(gadget4_cooling_test PRIVATE -fno-fast-math -ffp-contract=off)
        if(COSMICTIGER_WITH_CUDA)
            target_compile_definitions(gadget4_cooling_test PRIVATE USE_CUDA)
        endif()
        cosmictiger_add_gtest(NAME gadget4_cooling TARGET gadget4_cooling_test
            PROPERTIES TIMEOUT 180 PROCESSORS 2 RUN_SERIAL TRUE)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        # These analysis utilities also contain independent numerical regressions.
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, scipy, matplotlib"
            RESULT_VARIABLE ANALYSIS_PYTHON_STATUS OUTPUT_QUIET ERROR_QUIET)
        if(ANALYSIS_PYTHON_STATUS EQUAL 0)
            foreach(analysis_check IN ITEMS linear_growth glass_fixed_evolution glass_ic_corrections)
                if(analysis_check STREQUAL "linear_growth")
                    set(analysis_check_argument --self-check)
                else()
                    set(analysis_check_argument self-check)
                endif()
                set(analysis_check_name "${analysis_check}_analysis_self_check")
                set(analysis_check_workdir "${PROJECT_BINARY_DIR}/tests/${analysis_check_name}")
                file(MAKE_DIRECTORY "${analysis_check_workdir}")
                add_test(NAME ${analysis_check_name} COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/analyze_${analysis_check}.py ${analysis_check_argument})
                set_tests_properties(${analysis_check_name} PROPERTIES TIMEOUT 120 PROCESSORS 1
                    WORKING_DIRECTORY "${analysis_check_workdir}"
                    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;MPLCONFIGDIR=${analysis_check_workdir}/matplotlib"
                    LABELS "analysis;python")
            endforeach()
        endif()
    endif()
    if(COSMICTIGER_WITH_GRAVITY AND COSMICTIGER_WITH_CHEMISTRY AND cosmictiger_entropy_sources_ready)
        set(chemistry_timesteps_workdir "${PROJECT_BINARY_DIR}/tests/sph_chemistry_timesteps_cpu")
        file(MAKE_DIRECTORY "${chemistry_timesteps_workdir}")
        add_test(NAME sph_chemistry_timesteps_cpu COMMAND cosmictiger
            --run.test=sphChemistryTimesteps --run.scenario=cosmology
            --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --chemistry.enabled=true
            --chemistry.initializationTemperature=10000
            --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(sph_chemistry_timesteps_cpu PROPERTIES
            WORKING_DIRECTORY "${chemistry_timesteps_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)
        if(cosmictiger_mpi_tests)
            set(chemistry_mpi_workdir "${PROJECT_BINARY_DIR}/tests/sph_chemistry_timesteps_mpi")
            file(MAKE_DIRECTORY "${chemistry_mpi_workdir}")
            add_test(NAME sph_chemistry_timesteps_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                --run.test=sphChemistryTimesteps --run.scenario=cosmology
                --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --chemistry.enabled=true
                --starFormation.enabled=false --chemistry.initializationTemperature=10000
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=1 --hpx:localities=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args}
                ${cosmictiger_mpi_transport_args})
            set_tests_properties(sph_chemistry_timesteps_mpi PROPERTIES
                WORKING_DIRECTORY "${chemistry_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 180
                PROCESSORS 2 LABELS "chemistry;cpu;mpi"
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        endif()
    endif()
    if(COSMICTIGER_WITH_CHEMISTRY AND cosmictiger_entropy_sources_ready)
        foreach(integrator IN ITEMS hierarchical)
            foreach(backend IN ITEMS cpu mpi cuda)
                if((backend STREQUAL "mpi" AND NOT cosmictiger_mpi_tests) OR
                        (backend STREQUAL "cuda" AND NOT COSMICTIGER_WITH_CUDA))
                    continue()
                endif()
                set(cooling_test "gadget4_cooling_integration_${backend}")
                if(NOT integrator STREQUAL "traditional")
                    set(cooling_test "gadget4_cooling_integration_${integrator}_${backend}")
                endif()
                set(cooling_backend_args)
                if(backend STREQUAL "mpi")
                    set(cooling_backend_args ${cosmictiger_python_mpi_args})
                elseif(backend STREQUAL "cuda")
                    set(cooling_backend_args --cuda)
                endif()
                add_test(NAME ${cooling_test} COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_gadget4_cooling.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/${cooling_test}
                    --time-integrator=${integrator} ${cooling_backend_args})
                set_tests_properties(${cooling_test} PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE
                    PROCESSORS 2 SKIP_RETURN_CODE 77 LABELS "integration;chemistry;${backend}")
            endforeach()
        endforeach()
    endif()

    if(COSMICTIGER_WITH_GRAVITY AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_library(mpi_thread_override SHARED tests/mpi_thread_override.cpp)
        target_link_libraries(mpi_thread_override PRIVATE MPI::MPI_CXX)
        add_test(NAME mpi_thread_support COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_mpi_startup.py $<TARGET_FILE:cosmictiger>
            $<TARGET_FILE:mpi_thread_override> ${PROJECT_BINARY_DIR}/tests/mpi_thread_support
            --position-type ${COSMICTIGER_POSITION_TYPE})
        set_tests_properties(mpi_thread_support PROPERTIES TIMEOUT 240 PROCESSORS 2
            LABELS "mpi;startup")
        if(cosmictiger_mpi_tests)
            set(mpi_startup_launcher_args --mpi-exec=${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
            foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                list(APPEND mpi_startup_launcher_args "--mpi-preflag=${flag}")
            endforeach()
            foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                list(APPEND mpi_startup_launcher_args "--mpi-postflag=${flag}")
            endforeach()
            add_test(NAME mpi_thread_support_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_mpi_startup.py $<TARGET_FILE:cosmictiger>
                $<TARGET_FILE:mpi_thread_override> ${PROJECT_BINARY_DIR}/tests/mpi_thread_support_mpi
                ${mpi_startup_launcher_args} --position-type ${COSMICTIGER_POSITION_TYPE})
            set_tests_properties(mpi_thread_support_mpi PROPERTIES TIMEOUT 120 PROCESSORS 4
                LABELS "mpi;startup")
        endif()
        add_library(no_cuda_runtime SHARED tests/no_cuda_runtime.cpp)
        if(COSMICTIGER_WITH_GRAVITY)
            if(COSMICTIGER_WITH_ROCKSTAR)
                add_test(NAME analysis_cpu_runtime COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_analysis_cpu.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/analysis_cpu_runtime
                    --guard $<TARGET_FILE:no_cuda_runtime>)
                set_tests_properties(analysis_cpu_runtime PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE)
                if(COSMICTIGER_WITH_SPH)
                    # Gas joins the maps and spectra; the lightcone finder keeps dark matter.
                    add_test(NAME analysis_cpu_hydro_runtime COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_analysis_cpu.py $<TARGET_FILE:cosmictiger>
                        ${PROJECT_BINARY_DIR}/tests/analysis_cpu_hydro_runtime
                        --guard $<TARGET_FILE:no_cuda_runtime> --hydro)
                    set_tests_properties(analysis_cpu_hydro_runtime PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE)
                endif()
                if(cosmictiger_mpi_tests)
                    set(analysis_mpi_launcher_args --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
                    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                        list(APPEND analysis_mpi_launcher_args --mpi-preflag=${flag})
                    endforeach()
                    foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                        list(APPEND analysis_mpi_launcher_args --mpi-postflag=${flag})
                    endforeach()
                    foreach(flag IN LISTS cosmictiger_mpi_transport_args)
                        list(APPEND analysis_mpi_launcher_args --hpx-option=${flag})
                    endforeach()
                    add_test(NAME analysis_cpu_mpi COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_analysis_cpu.py $<TARGET_FILE:cosmictiger>
                        ${PROJECT_BINARY_DIR}/tests/analysis_cpu_mpi
                        --guard $<TARGET_FILE:no_cuda_runtime> --mpi-exec ${MPIEXEC_EXECUTABLE}
                        ${analysis_mpi_launcher_args} --ranks 2)
                    set_tests_properties(analysis_cpu_mpi PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE PROCESSORS 4)
                endif()
            endif()
        endif()
    endif()
    if(COSMICTIGER_WITH_GRAVITY)
        foreach(recombination_variant IN ITEMS standard fast_math)
            set(thermal_target recombination_thermal_${recombination_variant}_test)
            add_executable(${thermal_target} tests/recombination_thermal.cpp)
            target_include_directories(${thermal_target} PRIVATE ${PROJECT_SOURCE_DIR})
            set(background_target recombination_background_${recombination_variant}_test)
            add_executable(${background_target} tests/recombination_background.cpp
                src/cosmology/zero_order.cpp)
            target_include_directories(${background_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${background_target} PRIVATE cosmictiger_recombination)
            if(recombination_variant STREQUAL "fast_math")
                target_compile_options(${thermal_target} PRIVATE -ffast-math)
                target_compile_options(${background_target} PRIVATE -ffast-math)
            endif()
            cosmictiger_add_gtest(NAME recombination_thermal_${recombination_variant}
                TARGET ${thermal_target} PROPERTIES TIMEOUT 30)
            cosmictiger_add_gtest(NAME recombination_background_${recombination_variant}
                TARGET ${background_target} PROPERTIES TIMEOUT 30)
        endforeach()

        add_executable(recombination_reference tests/recombination_reference.cpp)
        target_link_libraries(recombination_reference PRIVATE cosmictiger_recombination)
        target_compile_options(recombination_reference PRIVATE -fno-fast-math -ffp-contract=off)
        add_test(NAME native_recombination_reference COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/compare_native_recombination.py check-reference
            --native-executable $<TARGET_FILE:recombination_reference>
            --reference-root ${PROJECT_SOURCE_DIR}/docs/native_recombination_results_20261020/recfast
            --workdir ${PROJECT_BINARY_DIR}/tests/native_recfast_regression)
        set_tests_properties(native_recombination_reference PROPERTIES TIMEOUT 60)
        foreach(native_variant IN ITEMS standard fast_math)
            set(native_target native_recombination_${native_variant}_test)
            add_executable(${native_target} tests/native_recombination.cpp src/cosmology/zero_order.cpp)
            target_link_libraries(${native_target} PRIVATE cosmictiger_recombination)
            if(native_variant STREQUAL "fast_math")
                target_compile_options(${native_target} PRIVATE -ffast-math)
            endif()
            cosmictiger_add_gtest(NAME native_recombination_${native_variant} TARGET ${native_target}
                PROPERTIES TIMEOUT 180)
        endforeach()

        add_executable(linear_power_test tests/linear_power.cpp)
        target_include_directories(linear_power_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(linear_power_test PRIVATE -ffast-math)
        cosmictiger_add_gtest(NAME linear_power TARGET linear_power_test
            PROPERTIES TIMEOUT 30)
        add_executable(lpt_test tests/lpt.cpp)
        target_include_directories(lpt_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(lpt_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME lpt TARGET lpt_test PROPERTIES TIMEOUT 30)
    endif()
    if(COSMICTIGER_WITH_SPH)
        add_executable(sedov_test tests/scenarios/sedov.cpp src/scenarios/sedov.cpp)
        target_include_directories(sedov_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(sedov_test PRIVATE HPX::hpx quadmath)
        target_compile_options(sedov_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME sedov TARGET sedov_test LABELS "scenarios;cpu" PROPERTIES TIMEOUT 30)
    endif()
    if(COSMICTIGER_WITH_ROCKSTAR)
        add_executable(analysis_tools_test tests/analysis_tools.cpp src/analysis/group_entry.cpp)
        target_include_directories(analysis_tools_test PRIVATE ${PROJECT_SOURCE_DIR}
            ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})
        target_link_libraries(analysis_tools_test PRIVATE Boost::program_options
            ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} dl)
        cosmictiger_add_gtest(NAME analysis_tools TARGET analysis_tools_test
            LABELS "analysis;cpu" PROPERTIES TIMEOUT 60)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        foreach(boltzmann_target IN ITEMS boltzmann_test boltzmann_reference)
            if(boltzmann_target STREQUAL "boltzmann_test")
                set(boltzmann_test_source tests/boltzmann.cpp)
            else()
                set(boltzmann_test_source tests/boltzmann_reference.cpp)
            endif()
            add_executable(${boltzmann_target} ${boltzmann_test_source}
                src/cosmology/boltzmann.cpp src/cosmology/zero_order.cpp)
            target_include_directories(${boltzmann_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${boltzmann_target} PRIVATE cosmictiger_recombination)
        endforeach()
        cosmictiger_add_gtest(NAME boltzmann TARGET boltzmann_test
            PROPERTIES TIMEOUT 120)
        if(COSMICTIGER_WITH_GRAVITY)
            foreach(boltzmann_two_lpt IN ITEMS false true)
                set(boltzmann_smoke boltzmann_initialization_${boltzmann_two_lpt}_cpu)
                add_test(NAME ${boltzmann_smoke} COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/${boltzmann_smoke}
                    --extra-option=--scenario.cosmology.twoLpt=${boltzmann_two_lpt}
                    --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
                set_tests_properties(${boltzmann_smoke}
                    PROPERTIES RUN_SERIAL TRUE TIMEOUT 240 PROCESSORS 2)
            endforeach()
        endif()
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        foreach(fft_mode IN ITEMS fft1 fft2)
            set(fft_workdir "${PROJECT_BINARY_DIR}/tests/${fft_mode}_cpu")
            file(MAKE_DIRECTORY "${fft_workdir}")
            add_test(NAME ${fft_mode}_cpu COMMAND $<TARGET_FILE:cosmictiger>
                --run.test=${fft_mode} ${cosmictiger_test_gravity_args} --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --hydro.enabled=false --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(${fft_mode}_cpu PROPERTIES TIMEOUT 120 PROCESSORS 2
                WORKING_DIRECTORY "${fft_workdir}" LABELS "fft;cpu"
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1")
            if(cosmictiger_mpi_tests)
                set(fft_mpi_workdir "${PROJECT_BINARY_DIR}/tests/${fft_mode}_mpi")
                file(MAKE_DIRECTORY "${fft_mpi_workdir}")
                add_test(NAME ${fft_mode}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=${fft_mode} ${cosmictiger_test_gravity_args} --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                    --hydro.enabled=false --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                    --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                    ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
                set_tests_properties(${fft_mode}_mpi PROPERTIES TIMEOUT 180 PROCESSORS 4
                    WORKING_DIRECTORY "${fft_mpi_workdir}" LABELS "fft;cpu;mpi"
                    ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1")
            endif()
        endforeach()
    endif()

    if(COSMICTIGER_WITH_ROCKSTAR)
        foreach(fp16_variant IN ITEMS standard fast_math)
            add_executable(fp16_${fp16_variant}_test tests/fp16.cpp)
            target_include_directories(fp16_${fp16_variant}_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(fp16_${fp16_variant}_test PRIVATE cosmictiger_fp16)
            if(fp16_variant STREQUAL "fast_math")
                target_compile_options(fp16_${fp16_variant}_test PRIVATE -ffast-math)
            endif()
            cosmictiger_add_gtest(NAME fp16_${fp16_variant} TARGET fp16_${fp16_variant}_test
                PROPERTIES TIMEOUT 30)
        endforeach()
    endif()

    foreach(fixed_variant IN ITEMS normal fast_math)
        if(fixed_variant STREQUAL "normal")
            set(fixed_target fixed_test)
            set(fixed_test_name fixed)
        else()
            set(fixed_target fixed_fast_math_test)
            set(fixed_test_name fixed_fast_math)
        endif()
        add_executable(${fixed_target} tests/fixed.cpp)
        target_include_directories(${fixed_target} PRIVATE ${PROJECT_SOURCE_DIR})
        if(fixed_variant STREQUAL "fast_math")
            target_compile_options(${fixed_target} PRIVATE -ffast-math)
        endif()
        cosmictiger_add_gtest(NAME ${fixed_test_name} TARGET ${fixed_target}
            PROPERTIES TIMEOUT 30)
    endforeach()

    if(COSMICTIGER_WITH_ROCKSTAR)
        foreach(halo_unit IN ITEMS rockstar_core rockstar_properties lightcone_epoch)
            add_executable(${halo_unit}_test tests/${halo_unit}.cpp)
            target_include_directories(${halo_unit}_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(${halo_unit}_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
            cosmictiger_use_generated_kernels(${halo_unit}_test)
            cosmictiger_add_gtest(NAME ${halo_unit} TARGET ${halo_unit}_test
                PROPERTIES TIMEOUT 30)
        endforeach()
        target_sources(lightcone_epoch_test PRIVATE src/cosmology/background.cpp
            src/cosmology/friedmann.cpp)
        target_link_libraries(lightcone_epoch_test PRIVATE HPX::hpx Boost::boost)
        target_compile_options(lightcone_epoch_test PRIVATE -fno-fast-math -ffp-contract=off)
        add_executable(rockstar_properties_fast_math_test tests/rockstar_properties.cpp)
        target_include_directories(rockstar_properties_fast_math_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(rockstar_properties_fast_math_test PRIVATE -ffast-math)
        cosmictiger_add_gtest(NAME rockstar_properties_fast_math TARGET rockstar_properties_fast_math_test
            PROPERTIES TIMEOUT 30)
        foreach(halo_mode IN ITEMS rockstar bhPotential groupsCatalogue)
            foreach(halo_backend IN ITEMS cpu cuda)
                if(halo_backend STREQUAL "cuda" AND (NOT COSMICTIGER_WITH_CUDA OR halo_mode STREQUAL "groupsCatalogue"))
                    continue()
                endif()
                if(halo_backend STREQUAL "cuda")
                    set(halo_cuda true)
                else()
                    set(halo_cuda false)
                endif()
                set(halo_workdir "${PROJECT_BINARY_DIR}/tests/${halo_mode}_${halo_backend}")
                file(MAKE_DIRECTORY "${halo_workdir}")
                add_test(NAME ${halo_mode}_${halo_backend} COMMAND cosmictiger
                    --run.test=${halo_mode} ${cosmictiger_test_gravity_args} --analysis.groups.cuda=${halo_cuda} --gravity.cuda=${halo_cuda} --hydro.cuda=false --drift.cuda=${halo_cuda} --hydro.enabled=false
                    --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --analysis.groups=true
                    --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
                set_tests_properties(${halo_mode}_${halo_backend} PROPERTIES
                    WORKING_DIRECTORY "${halo_workdir}" TIMEOUT 180 PROCESSORS 2)
            endforeach()
        endforeach()
        if(COSMICTIGER_WITH_SPH)
            # The same catalogue fixture with gas and stars present: the finder
            # stays dark-matter-only and the records gain rvir aperture masses.
            set(halo_workdir "${PROJECT_BINARY_DIR}/tests/groupsCatalogue_hydro_cpu")
            file(MAKE_DIRECTORY "${halo_workdir}")
            add_test(NAME groupsCatalogue_hydro_cpu COMMAND cosmictiger
                --run.test=groupsCatalogue ${cosmictiger_test_gravity_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hydro.enabled=true
                --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --analysis.groups=true
                --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_sph_gravity_integrator_args})
            set_tests_properties(groupsCatalogue_hydro_cpu PROPERTIES
                WORKING_DIRECTORY "${halo_workdir}" TIMEOUT 180 PROCESSORS 2)
        endif()
        if(cosmictiger_mpi_tests)
            set(halo_mpi_variants dm)
            if(COSMICTIGER_WITH_SPH)
                list(APPEND halo_mpi_variants hydro)
            endif()
            foreach(halo_mpi_variant IN LISTS halo_mpi_variants)
                if(halo_mpi_variant STREQUAL "hydro")
                    set(halo_mpi_test groupsCatalogue_hydro_mpi)
                    set(halo_mpi_hydro true)
                    set(halo_mpi_integrator_args ${cosmictiger_sph_gravity_integrator_args})
                else()
                    set(halo_mpi_test groupsCatalogue_mpi)
                    set(halo_mpi_hydro false)
                    set(halo_mpi_integrator_args ${cosmictiger_test_integrator_args})
                endif()
                set(halo_mpi_workdir "${PROJECT_BINARY_DIR}/tests/${halo_mpi_test}")
                file(MAKE_DIRECTORY "${halo_mpi_workdir}")
                add_test(NAME ${halo_mpi_test} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger>
                    ${cosmictiger_mpi_transport_args}
                    --run.test=groupsCatalogueDistributed ${cosmictiger_test_gravity_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hydro.enabled=${halo_mpi_hydro}
                    --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --analysis.groups=true
                    --hpx:threads=2 --hpx:bind=none --hpx:localities=2 ${MPIEXEC_POSTFLAGS}
                    ${halo_mpi_integrator_args})
                set_tests_properties(${halo_mpi_test} PROPERTIES
                    WORKING_DIRECTORY "${halo_mpi_workdir}" TIMEOUT 240 PROCESSORS 4 RUN_SERIAL TRUE)
            endforeach()
        endif()
    endif()

    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_WITH_GRAVITY)
        add_test(NAME scenarios COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_scenarios.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/scenarios ${cosmictiger_chemistry_test_args})
        set_tests_properties(scenarios PROPERTIES RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2)
    endif()

    set(cosmictiger_sph_test_args)
    if(NOT COSMICTIGER_WITH_SPH)
        list(APPEND cosmictiger_sph_test_args --without-sph)
    endif()
    if(COSMICTIGER_WITH_GRAVITY)
        set(rockstar_capability_args)
        if(COSMICTIGER_WITH_ROCKSTAR)
            list(APPEND rockstar_capability_args --enabled)
        endif()
        add_test(NAME rockstar_capability COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_rockstar_capability.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/rockstar_capability ${rockstar_capability_args})
        set_tests_properties(rockstar_capability PROPERTIES TIMEOUT 120 PROCESSORS 2 LABELS "build;cpu")
        add_test(NAME grouped_options COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_options.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/grouped_options ${cosmictiger_sph_test_args}
            ${cosmictiger_options_capability_args}
            ${cosmictiger_rockstar_test_args}

--order ${cosmictiger_default_fmm_order}
            --time-integrator ${cosmictiger_default_integrator})
        # run_options.py overlaps four two-thread cosmictiger launches.
        set_tests_properties(grouped_options PROPERTIES RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 8)
        add_test(NAME hydro_methods COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_options.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/hydro_methods ${cosmictiger_sph_test_args}
            ${cosmictiger_options_capability_args}
            ${cosmictiger_rockstar_test_args}

--order ${cosmictiger_default_fmm_order}
            --time-integrator ${cosmictiger_default_integrator} --hydro-methods-only)
        set_tests_properties(hydro_methods PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2
            LABELS "build;cpu")
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        set(gravity_error_name gravity_error_cpu)
        add_test(NAME ${gravity_error_name} COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_gravity_error.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/${gravity_error_name}

            --position-type ${COSMICTIGER_POSITION_TYPE})
        set_tests_properties(${gravity_error_name} PROPERTIES TIMEOUT 300 PROCESSORS 2
            LABELS "gravity;accuracy")
        if(COSMICTIGER_WITH_CUDA)
            set(gravity_error_name gravity_error_cuda)
            add_test(NAME ${gravity_error_name} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_gravity_error.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/${gravity_error_name} --cuda

                --position-type ${COSMICTIGER_POSITION_TYPE})
            set_tests_properties(${gravity_error_name} PROPERTIES TIMEOUT 300 PROCESSORS 2
                RUN_SERIAL TRUE LABELS "gravity;accuracy;cuda")
        endif()
        if(cosmictiger_mpi_tests)
            set(gravity_error_mpi_args --mpi-exec=${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
            foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                list(APPEND gravity_error_mpi_args "--mpi-preflag=${flag}")
            endforeach()
            foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                list(APPEND gravity_error_mpi_args "--mpi-postflag=${flag}")
            endforeach()
            if(HPX_WITH_PARCELPORT_TCP)
                list(APPEND gravity_error_mpi_args --disable-tcp)
            endif()
            add_test(NAME gravity_error_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_gravity_error.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/gravity_error_mpi ${gravity_error_mpi_args} --ranks 2
                --position-type ${COSMICTIGER_POSITION_TYPE})
            set_tests_properties(gravity_error_mpi PROPERTIES TIMEOUT 300 PROCESSORS 4
                RUN_SERIAL TRUE LABELS "gravity;accuracy;mpi")
        endif()
    endif()

    add_executable(units_test tests/units.cpp)
    target_include_directories(units_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(units_test PRIVATE Boost::boost)
    cosmictiger_add_gtest(NAME units TARGET units_test
        PROPERTIES TIMEOUT 20)

    add_executable(units_fast_math_test tests/units.cpp)
    target_include_directories(units_fast_math_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(units_fast_math_test PRIVATE Boost::boost)
    target_compile_options(units_fast_math_test PRIVATE -ffast-math)
    cosmictiger_add_gtest(NAME units_fast_math TARGET units_fast_math_test
        PROPERTIES TIMEOUT 20)

    include(CheckIncludeFileCXX)
    check_include_file_cxx("experimental/simd" COSMICTIGER_HAS_EXPERIMENTAL_SIMD)
    foreach(simd_variant IN ITEMS auto baseline native)
        set(simd_target "simd_${simd_variant}_test")
        add_executable(${simd_target} tests/simd.cpp)
        target_include_directories(${simd_target} PRIVATE ${PROJECT_SOURCE_DIR})
        if(NOT simd_variant STREQUAL "auto" AND COSMICTIGER_HAS_EXPERIMENTAL_SIMD)
            target_compile_definitions(${simd_target} PRIVATE
                COSMICTIGER_SIMD_USE_EXPERIMENTAL=1 COSMICTIGER_SIMD_EXPECT_STANDARD=0)
        endif()
        if(simd_variant STREQUAL "native" AND COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(${simd_target} PRIVATE -march=native)
        elseif(simd_variant STREQUAL "baseline" AND
                CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$" AND
                CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${simd_target} PRIVATE
                -msse2 -mno-avx -mno-avx2 -mno-avx512f -mno-fma)
        endif()
        cosmictiger_add_gtest(NAME simd_${simd_variant} TARGET ${simd_target}
            PROPERTIES TIMEOUT 20)
    endforeach()

    if(COSMICTIGER_WITH_SPH)
        set(units_output_workdir "${PROJECT_BINARY_DIR}/tests/units_output")
        file(MAKE_DIRECTORY "${units_output_workdir}")
        add_test(NAME units_output COMMAND cosmictiger --run.test=unitsOutput
            --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
            --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(units_output PROPERTIES
            WORKING_DIRECTORY "${units_output_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
        # Keep the existing cases on the default bounds and cover constant alpha (equal bounds).
        set(units_workdir "${PROJECT_BINARY_DIR}/tests/units_output_none")
        file(MAKE_DIRECTORY "${units_workdir}")
        add_test(NAME units_output_none COMMAND cosmictiger --run.test=unitsOutput
            --compute.positionType=${COSMICTIGER_POSITION_TYPE}
            --hydro.viscosityCoefficientMin=1 --hydro.viscosityCoefficientMax=1
            --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
            --hpx:threads=2 --hpx:bind=none
            ${cosmictiger_test_integrator_args})
        set_tests_properties(units_output_none PROPERTIES
            WORKING_DIRECTORY "${units_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)

        set(distributed_output_workdir "${PROJECT_BINARY_DIR}/tests/distributed_output")
        file(MAKE_DIRECTORY "${distributed_output_workdir}")
        add_test(NAME distributed_output COMMAND cosmictiger --run.test=distributedOutput
            --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
            --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(distributed_output PROPERTIES WORKING_DIRECTORY "${distributed_output_workdir}"
            RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)
    endif()

    add_executable(radial_kernel_test tests/radial_kernel.cpp)
    target_include_directories(radial_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(radial_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(radial_kernel_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME radial_kernel TARGET radial_kernel_test
        PROPERTIES TIMEOUT 30)

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(cartesian_kernels_test tests/gravity/cartesian_kernels.cpp)
        target_include_directories(cartesian_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(cartesian_kernels_test)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(cartesian_kernels_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME cartesian_kernels TARGET cartesian_kernels_test
            PROPERTIES TIMEOUT 60)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(fmm_api_test tests/gravity/fmm_api.cpp)
        target_include_directories(fmm_api_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(fmm_api_test)
        cosmictiger_add_gtest(NAME fmm_api TARGET fmm_api_test
            PROPERTIES TIMEOUT 30)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        get_target_property(source_capabilities cosmictiger_default_config INTERFACE_COMPILE_DEFINITIONS)
        list(FILTER source_capabilities EXCLUDE REGEX "^(COSMICTIGER_POSITION_BITS|FMM_MIN_ORDER|FMM_MAX_ORDER|ORDER)=")
        foreach(position_bits IN ITEMS 32 64)
            set(source_target "tree_source_moments_fixed${position_bits}_test")
            add_executable(${source_target} tests/gravity/tree_source_moments.cpp)
            target_include_directories(${source_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${source_target} PRIVATE HPX::hpx Boost::boost)
            target_compile_definitions(${source_target} PRIVATE ${source_capabilities}
                COSMICTIGER_POSITION_BITS=${position_bits} ORDER=7 FMM_MIN_ORDER=3 FMM_MAX_ORDER=9)
            target_compile_options(${source_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "tree_source_moments_fixed${position_bits}" COMMAND ${source_target})
            set_tests_properties("tree_source_moments_fixed${position_bits}" PROPERTIES
                TIMEOUT 60 PROCESSORS 1 LABELS "fmm;pmmm;cpu")
        endforeach()
    endif()

    # Standalone generator families: these tests do not select a new runtime walk.
    if(COSMICTIGER_WITH_GRAVITY)
        foreach(representation IN ITEMS spherical_ewald)
            set(generated_kernel_target "generated_${representation}_test")
            add_executable(${generated_kernel_target} "tests/gravity/generated_${representation}.cpp")
            target_include_directories(${generated_kernel_target} PRIVATE ${PROJECT_SOURCE_DIR})
            cosmictiger_use_generated_kernels(${generated_kernel_target})
            target_compile_options(${generated_kernel_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "generated_${representation}" COMMAND ${generated_kernel_target})
            set_tests_properties("generated_${representation}" PROPERTIES
                TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator;cpu")
        endforeach()
        # Splitting the spherical templates by degree bounds compiler memory use.
        foreach(spherical_order IN LISTS COSMICTIGER_FMM_ORDERS)
            set(conversion_target "basis_conversion_order_${spherical_order}_test")
            add_executable(${conversion_target} tests/gravity/basis_conversion.cpp)
            target_include_directories(${conversion_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(${conversion_target} PRIVATE
                BASIS_CONVERSION_TEST_ORDER=${spherical_order})
            cosmictiger_use_generated_kernels(${conversion_target})
            target_compile_options(${conversion_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "basis_conversion_order_${spherical_order}" COMMAND ${conversion_target})
            set_tests_properties("basis_conversion_order_${spherical_order}" PROPERTIES
                TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator;cpu")
            math(EXPR spherical_degree "${spherical_order} - 1")
            set(spherical_target "generated_spherical_degree_${spherical_degree}_test")
            add_executable(${spherical_target} tests/gravity/generated_spherical.cpp)
            target_include_directories(${spherical_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(${spherical_target} PRIVATE
                GENERATED_SPHERICAL_TEST_DEGREE=${spherical_degree})
            cosmictiger_use_generated_kernels(${spherical_target})
            target_compile_options(${spherical_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "generated_spherical_degree_${spherical_degree}" COMMAND ${spherical_target})
            set_tests_properties("generated_spherical_degree_${spherical_degree}" PROPERTIES
                TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator;cpu")
        endforeach()
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_test(NAME generated_fmm_current COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/generate_fmm.py --check
            --output-dir ${COSMICTIGER_GENERATED_FMM_DIR} --min-order 2 --max-order 8)
        set_tests_properties(generated_fmm_current PROPERTIES
            TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator")
        add_test(NAME generated_fmm_flops COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/gravity/check_generated_fmm_flops.py
            --generated-dir ${COSMICTIGER_GENERATED_FMM_DIR}
            --min-fmm-order ${COSMICTIGER_FMM_MIN_ORDER}
            --max-fmm-order ${COSMICTIGER_FMM_MAX_ORDER}
            --compiler ${CMAKE_CXX_COMPILER})
        set_tests_properties(generated_fmm_flops PROPERTIES
            TIMEOUT 300 PROCESSORS 1 LABELS "fmm;generator")
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(softening_flops_test tests/softening_flops.cpp)
        target_include_directories(softening_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_include_directories(softening_flops_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
        target_compile_definitions(softening_flops_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
        cosmictiger_use_generated_kernels(softening_flops_test)
        target_link_libraries(softening_flops_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME softening_flops TARGET softening_flops_test
            PROPERTIES TIMEOUT 30)
    endif()

    if(COSMICTIGER_WITH_SPH)
        add_executable(sph_flops_test tests/sph/flops.cpp)
        target_include_directories(sph_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(sph_flops_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
        cosmictiger_use_generated_kernels(sph_flops_test)
        target_link_libraries(sph_flops_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME sph_flops TARGET sph_flops_test
            PROPERTIES TIMEOUT 30)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(fmm_acceptance_test tests/gravity/fmm_acceptance.cpp)
        target_include_directories(fmm_acceptance_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(fmm_acceptance_test PRIVATE -ffast-math)
        target_compile_definitions(fmm_acceptance_test PRIVATE FMM_ACCEPTANCE_TEST_SIMD)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(fmm_acceptance_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME fmm_acceptance TARGET fmm_acceptance_test
            PROPERTIES TIMEOUT 30)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(mesh_domain_test tests/mesh_domain.cpp)
        target_include_directories(mesh_domain_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(mesh_domain_test PRIVATE HPX::hpx Boost::boost)
        target_compile_options(mesh_domain_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME mesh_domain TARGET mesh_domain_test
            PROPERTIES TIMEOUT 30 LABELS "particle_mesh;cpu")

        add_executable(pmmm_kernels_test tests/gravity/pmmm_kernels.cpp
            tests/gravity/pmmm_kernel_flops.cpp src/gravity/pmmm_kernels.cpp)
        target_include_directories(pmmm_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(pmmm_kernels_test)
        target_link_libraries(pmmm_kernels_test PRIVATE HPX::hpx Boost::boost)
        target_compile_options(pmmm_kernels_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME pmmm_kernels TARGET pmmm_kernels_test
            PROPERTIES TIMEOUT 180 LABELS "pmmm;cpu")
        add_executable(pmmm_near_flops_test tests/gravity/pp/near_pp_flops.cpp)
        target_include_directories(pmmm_near_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(pmmm_near_flops_test PRIVATE HPX::hpx Boost::boost)
        target_compile_options(pmmm_near_flops_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME pmmm_near_flops TARGET pmmm_near_flops_test
            PROPERTIES TIMEOUT 60 LABELS "pmmm;cpu;flops")
        add_executable(pmmm_fft_flops_test tests/gravity/fft_flops.cpp)
        target_include_directories(pmmm_fft_flops_test PRIVATE ${PROJECT_SOURCE_DIR} ${FFTW3F_INCLUDE_DIR})
        target_link_libraries(pmmm_fft_flops_test PRIVATE HPX::hpx Boost::boost ${FFTW3F_LIBRARY})
        target_compile_options(pmmm_fft_flops_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME pmmm_fft_flops TARGET pmmm_fft_flops_test
            PROPERTIES TIMEOUT 60 LABELS "pmmm;cpu;flops")
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        set(pmmm_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_cpu")
        file(MAKE_DIRECTORY "${pmmm_workdir}")
        add_test(NAME pmmm_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=pmmm --run.darkMatterParticleCount=512 --run.gasParticleCount=512
            --gravity.meshSize=8 --hydro.enabled=false --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
            --compute.positionType=${COSMICTIGER_POSITION_TYPE} --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(pmmm_cpu PROPERTIES TIMEOUT 300 PROCESSORS 2
            WORKING_DIRECTORY "${pmmm_workdir}" LABELS "pmmm;cpu")
        set(pmmm_source_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_sources_cpu")
        file(MAKE_DIRECTORY "${pmmm_source_workdir}")
        add_test(NAME pmmm_sources_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=pmmmSources --gravity.meshSize=8 --hydro.enabled=false
            --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
            --hpx:threads=2 --hpx:bind=none ${cosmictiger_test_integrator_args})
        set_tests_properties(pmmm_sources_cpu PROPERTIES TIMEOUT 300 PROCESSORS 2
            WORKING_DIRECTORY "${pmmm_source_workdir}" LABELS "pmmm;source;cpu")
        foreach(near_method IN ITEMS pp fmm)
            set(near_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_near_${near_method}")
            file(MAKE_DIRECTORY "${near_workdir}")
            add_test(NAME pmmm_near_${near_method} COMMAND cosmictiger --run.test=pmmm
            --gravity.method.near=${near_method}
                --gravity.meshSize=8 --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --hydro.enabled=false
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(pmmm_near_${near_method} PROPERTIES TIMEOUT 300 PROCESSORS 2
                WORKING_DIRECTORY "${near_workdir}" LABELS "pmmm;near;cpu")
        endforeach()
        set(mesh_domain_workdir "${PROJECT_BINARY_DIR}/tests/mesh_domains_cpu")
        file(MAKE_DIRECTORY "${mesh_domain_workdir}")
        add_test(NAME mesh_domains_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=meshDomains --run.darkMatterParticleCount=512 --run.gasParticleCount=512
            --gravity.meshSize=8 --hydro.enabled=false
            --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
            --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(mesh_domains_cpu PROPERTIES TIMEOUT 180 PROCESSORS 2
            WORKING_DIRECTORY "${mesh_domain_workdir}" LABELS "particle_mesh;cpu")
        set(pmmm_sph_args)
        if(COSMICTIGER_WITH_SPH)
            list(APPEND pmmm_sph_args --with-sph)
        endif()
        add_test(NAME pmmm_evolution_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_cpu
            --order ${cosmictiger_default_fmm_order}
            --time-integrator ${cosmictiger_default_integrator}
            --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
        set_tests_properties(pmmm_evolution_cpu PROPERTIES TIMEOUT 600 PROCESSORS 2
            RUN_SERIAL TRUE LABELS "pmmm;cpu")
        # Gas and dark matter under PMMM gravity on the hierarchical ladder, the only
        # integrator that couples SPH to PMMM: four steps, gas gravity, restart.
        if(COSMICTIGER_WITH_SPH)
            add_test(NAME pmmm_evolution_sph_hierarchical_cpu COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_sph_hierarchical_cpu
                --order ${cosmictiger_default_fmm_order} --time-integrator hierarchical
                --position-type ${COSMICTIGER_POSITION_TYPE} --with-sph)
            set_tests_properties(pmmm_evolution_sph_hierarchical_cpu PROPERTIES TIMEOUT 900
                PROCESSORS 2 LABELS "pmmm;sph;hierarchical;cpu")
        endif()
        # Outer interval with frozen near membership (docs/pmmm.md).
        if(COSMICTIGER_WITH_GRAVITY)
            add_test(NAME pmmm_interval_cpu COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_interval.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_interval_cpu
                --order ${cosmictiger_default_fmm_order} --time-integrator ${cosmictiger_default_integrator}
                --position-type ${COSMICTIGER_POSITION_TYPE})
            set_tests_properties(pmmm_interval_cpu PROPERTIES TIMEOUT 900 PROCESSORS 2
                RUN_SERIAL TRUE LABELS "pmmm;particle_mesh;cpu")
        endif()
        if(COSMICTIGER_WITH_CUDA)
            add_test(NAME pmmm_evolution_cuda COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_cuda --cuda
                --order ${cosmictiger_default_fmm_order}
                --time-integrator ${cosmictiger_default_integrator}
                --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
            set_tests_properties(pmmm_evolution_cuda PROPERTIES TIMEOUT 600 PROCESSORS 2
                RUN_SERIAL TRUE LABELS "pmmm;cuda")
            file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/tests/pmmm_cuda")
            add_test(NAME pmmm_cuda COMMAND $<TARGET_FILE:cosmictiger>
                --run.test=pmmm --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --gravity.meshSize=8 --hydro.enabled=false --gravity.cuda=true --hydro.cuda=false --drift.cuda=true
                --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(pmmm_cuda PROPERTIES TIMEOUT 300 PROCESSORS 2
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/tests/pmmm_cuda" LABELS "pmmm;cuda")
        endif()
        foreach(pm_check IN ITEMS particleMeshSort particleMeshTimestep)
            set(pm_test_name "${pm_check}")
            set(pm_test_workdir "${PROJECT_BINARY_DIR}/tests/${pm_test_name}")
            file(MAKE_DIRECTORY "${pm_test_workdir}")
            add_test(NAME ${pm_test_name} COMMAND $<TARGET_FILE:cosmictiger>
                --run.test=${pm_check} --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --gravity.meshSize=8 --hydro.enabled=false
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
                --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(${pm_test_name} PROPERTIES TIMEOUT 180 PROCESSORS 2
                WORKING_DIRECTORY "${pm_test_workdir}" LABELS "particle_mesh;cpu")
        endforeach()
        if(cosmictiger_mpi_tests)
            set(pm_launcher_args --mpi-exec=${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
            if(HPX_WITH_PARCELPORT_TCP)
                list(APPEND pm_launcher_args --disable-tcp)
            endif()
            foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                list(APPEND pm_launcher_args "--mpi-preflag=${flag}")
            endforeach()
            foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                list(APPEND pm_launcher_args "--mpi-postflag=${flag}")
            endforeach()
            set(pmmm_threaded_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_mpi_threads")
            file(MAKE_DIRECTORY "${pmmm_threaded_workdir}")
            add_test(NAME pmmm_mpi_threads COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                --run.test=pmmm --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                --gravity.meshSize=8 --hydro.enabled=false --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
            # FFTW and the near-field tasks share each rank's HPX workers.
            set_tests_properties(pmmm_mpi_threads PROPERTIES TIMEOUT 300 PROCESSORS 4
                WORKING_DIRECTORY "${pmmm_threaded_workdir}" LABELS "pmmm;cpu;mpi;threads")
            # PMMM requires power-of-two locality counts.
            foreach(source_ranks IN ITEMS 2 4)
                set(source_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_sources_mpi_${source_ranks}")
                file(MAKE_DIRECTORY "${source_workdir}")
                add_test(NAME pmmm_sources_mpi_${source_ranks} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${source_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=pmmmSources --gravity.meshSize=8 --hydro.enabled=false
                    --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                    --hpx:threads=1 --hpx:bind=none --hpx:localities=${source_ranks}
                    ${cosmictiger_mpi_transport_args} ${cosmictiger_test_integrator_args})
                set_tests_properties(pmmm_sources_mpi_${source_ranks} PROPERTIES TIMEOUT 300
                    PROCESSORS ${source_ranks} WORKING_DIRECTORY "${source_workdir}"
                    LABELS "pmmm;source;mpi")
            endforeach()
            foreach(pmmm_ranks IN ITEMS 4 8)
                set(pmmm_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_mpi_${pmmm_ranks}")
                file(MAKE_DIRECTORY "${pmmm_workdir}")
                add_test(NAME pmmm_mpi_${pmmm_ranks} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${pmmm_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=pmmm --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                    --gravity.meshSize=8 --hydro.enabled=false --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                    --hpx:threads=1 --hpx:bind=none --hpx:localities=${pmmm_ranks}
                    ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
                set_tests_properties(pmmm_mpi_${pmmm_ranks} PROPERTIES TIMEOUT 300
                    PROCESSORS ${pmmm_ranks} WORKING_DIRECTORY "${pmmm_workdir}" LABELS "pmmm;mpi")
            endforeach()
            foreach(mesh_domain_ranks IN ITEMS 2 4 8)
                set(mesh_domain_name "mesh_domains_mpi_${mesh_domain_ranks}")
                set(mesh_domain_workdir "${PROJECT_BINARY_DIR}/tests/${mesh_domain_name}")
                file(MAKE_DIRECTORY "${mesh_domain_workdir}")
                add_test(NAME ${mesh_domain_name} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${mesh_domain_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=meshDomains --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                    --gravity.meshSize=8 --hydro.enabled=false
                    --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
                    --hpx:threads=1 --hpx:bind=none --hpx:localities=${mesh_domain_ranks}
                    ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
                set_tests_properties(${mesh_domain_name} PROPERTIES TIMEOUT 180
                    PROCESSORS ${mesh_domain_ranks} WORKING_DIRECTORY "${mesh_domain_workdir}"
                    LABELS "particle_mesh;mpi")
            endforeach()
            add_test(NAME pmmm_evolution_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_mpi ${pm_launcher_args} --ranks 2
                --order ${cosmictiger_default_fmm_order}
                --time-integrator ${cosmictiger_default_integrator}
                --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
            set_tests_properties(pmmm_evolution_mpi PROPERTIES TIMEOUT 600 PROCESSORS 4
                RUN_SERIAL TRUE LABELS "pmmm;mpi")
            if(COSMICTIGER_WITH_SPH)
                add_test(NAME pmmm_evolution_sph_hierarchical_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/gravity/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_sph_hierarchical_mpi
                    ${pm_launcher_args} --ranks 2 --order ${cosmictiger_default_fmm_order}
                    --time-integrator hierarchical --position-type ${COSMICTIGER_POSITION_TYPE}
                    --with-sph)
                set_tests_properties(pmmm_evolution_sph_hierarchical_mpi PROPERTIES TIMEOUT 900
                    PROCESSORS 4 LABELS "pmmm;sph;hierarchical;mpi")
            endif()
            add_test(NAME particle_mesh_dispatch_failures_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/run_particle_mesh_failures.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/particle_mesh_dispatch_failures_mpi ${pm_launcher_args})
            set_tests_properties(particle_mesh_dispatch_failures_mpi PROPERTIES TIMEOUT 100 PROCESSORS 4
                RUN_SERIAL TRUE LABELS "particle_mesh;mpi")
            # The fixture migrates to spatial ownership, then checks the global
            # timestep reduction; the FMM near solver is covered by streamed tests.
            if(cosmictiger_mpi_four_ranks)
                set(pm_timestep_workdir "${PROJECT_BINARY_DIR}/tests/particle_mesh_timestep_mpi")
                file(MAKE_DIRECTORY "${pm_timestep_workdir}")
                add_test(NAME particle_mesh_timestep_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger>
                    ${MPIEXEC_POSTFLAGS} --run.test=particleMeshTimestep --gravity.method.near=pp --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                    --gravity.meshSize=8 --hydro.enabled=false
                    --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=4
                    ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
                set_tests_properties(particle_mesh_timestep_mpi PROPERTIES TIMEOUT 180 PROCESSORS 8
                    WORKING_DIRECTORY "${pm_timestep_workdir}" LABELS "particle_mesh;mpi")
            endif()
        endif()
    endif()
    if(COSMICTIGER_WITH_SPH)
        add_executable(gresho_test tests/gresho.cpp)
        target_include_directories(gresho_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME gresho TARGET gresho_test PROPERTIES TIMEOUT 30 LABELS "sph;cpu")
        add_executable(radial_density_kernel_test tests/radial_density_kernel.cpp)
        target_include_directories(radial_density_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(radial_density_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(radial_density_kernel_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME radial_density_kernel TARGET radial_density_kernel_test
            PROPERTIES TIMEOUT 30)

        add_executable(sph_kernels_test tests/sph/kernels.cpp)
        target_include_directories(sph_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME sph_kernels TARGET sph_kernels_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(morris_and_monaghan_test tests/sph/morris_and_monaghan/viscosity.cpp)
        target_include_directories(morris_and_monaghan_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME morris_and_monaghan TARGET morris_and_monaghan_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(balsara_test tests/sph/balsara/shear_limiter.cpp)
        target_include_directories(balsara_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME balsara TARGET balsara_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(monaghan_sph_test tests/sph/monaghan/equations.cpp)
        target_include_directories(monaghan_sph_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME monaghan_sph TARGET monaghan_sph_test
            PROPERTIES TIMEOUT 20)

        # Weighted-volume and grad-H algebra of the variational pressure operator.
        add_executable(sph_volume_test tests/sph/volume.cpp)
        target_include_directories(sph_volume_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME sph_volume TARGET sph_volume_test
            PROPERTIES TIMEOUT 120 LABELS "sph;cpu")
    endif()

    foreach(analysis_kernel IN ITEMS power lightcone)
        if(NOT COSMICTIGER_WITH_GRAVITY OR
                (analysis_kernel STREQUAL "lightcone" AND NOT COSMICTIGER_WITH_ROCKSTAR))
            continue()
        endif()
        add_executable(${analysis_kernel}_cpu_test
            tests/${analysis_kernel}_cpu.cpp src/analysis/${analysis_kernel}_cpu.cpp)
        target_include_directories(${analysis_kernel}_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${analysis_kernel}_cpu_test PRIVATE HPX::hpx)
        target_compile_options(${analysis_kernel}_cpu_test PRIVATE -ffast-math)
        cosmictiger_add_gtest(NAME ${analysis_kernel}_cpu TARGET ${analysis_kernel}_cpu_test
            PROPERTIES TIMEOUT 60)
    endforeach()

    if(COSMICTIGER_WITH_CUDA)
        if(COSMICTIGER_WITH_GRAVITY)
            add_executable(power_cuda_test tests/power_cuda.cu src/analysis/power_cpu.cpp src/analysis/power_cuda.cu
                src/core/assert.cu)
            target_include_directories(power_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(power_cuda_test PRIVATE USE_CUDA)
            target_link_libraries(power_cuda_test PRIVATE HPX::hpx CUDA::cudart)
            target_compile_options(power_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
            set_target_properties(power_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
            cosmictiger_add_gtest(NAME power_cuda TARGET power_cuda_test
                PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

            if(COSMICTIGER_WITH_ROCKSTAR)
                add_executable(lightcone_cuda_test tests/lightcone_cuda.cu src/analysis/lightcone.cu
                    src/analysis/lightcone_cpu.cpp src/core/assert.cu src/cuda/cuda_stream.cpp)
                target_include_directories(lightcone_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
                target_compile_definitions(lightcone_cuda_test PRIVATE USE_CUDA ORDER=${cosmictiger_default_fmm_order})
                cosmictiger_use_generated_kernels(lightcone_cuda_test)
                target_link_libraries(lightcone_cuda_test PRIVATE HPX::hpx CUDA::cudart)
                target_compile_options(lightcone_cuda_test PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
                set_target_properties(lightcone_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                cosmictiger_add_gtest(NAME lightcone_cuda TARGET lightcone_cuda_test
                    PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
            endif()
        endif()

        add_hpx_executable(cuda_async_test
            SOURCES tests/cuda_async.cpp tests/cuda_async.cu src/cuda/cuda_stream.cpp)
        target_include_directories(cuda_async_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(cuda_async_test PRIVATE USE_CUDA)
        target_link_libraries(cuda_async_test PRIVATE CUDA::cudart)
        cosmictiger_add_gtest(NAME cuda_async TARGET cuda_async_test HPX_MAIN
            ARGS --hpx:threads=1 --hpx:bind=none
            ENVIRONMENT "OMP_NUM_THREADS=1;OMP_THREAD_LIMIT=1"
            PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE SKIP_RETURN_CODE 77 PROCESSORS 1)
    endif()

    # Compile CPU drift with ordinary host arrays and no CUDA device kernels.
    add_executable(drift_cpu_test tests/drift_cpu.cpp src/particles/drift_cpu.cpp)
    target_include_directories(drift_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(drift_cpu_test PRIVATE HPX::hpx)
    if(COSMICTIGER_WITH_GRAVITY)
        target_include_directories(drift_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
        target_link_libraries(drift_cpu_test PRIVATE ${CHEALPIX_LIBRARY})
    endif()
    target_compile_options(drift_cpu_test PRIVATE -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(drift_cpu_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME drift_cpu TARGET drift_cpu_test
        PROPERTIES TIMEOUT 30)

    add_hpx_executable(drift_dispatch_test
        SOURCES tests/drift_dispatch.cpp src/particles/drift.cpp src/particles/drift_cpu.cpp)
    target_include_directories(drift_dispatch_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(drift_dispatch_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
    cosmictiger_use_generated_kernels(drift_dispatch_test)
    if(COSMICTIGER_WITH_GRAVITY)
        target_include_directories(drift_dispatch_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
        target_link_libraries(drift_dispatch_test PRIVATE ${CHEALPIX_LIBRARY})
    endif()
    target_compile_options(drift_dispatch_test PRIVATE -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(drift_dispatch_test PRIVATE -march=native)
    endif()
    if(COSMICTIGER_WITH_CUDA)
        target_compile_definitions(drift_dispatch_test PRIVATE USE_CUDA)
        target_link_libraries(drift_dispatch_test PRIVATE CUDA::cudart)
    endif()
    cosmictiger_add_gtest(NAME drift_dispatch TARGET drift_dispatch_test HPX_MAIN
        ARGS --hpx:threads=2 --hpx:bind=none
        ENVIRONMENT "CUDA_VISIBLE_DEVICES="
        PROPERTIES TIMEOUT 30 PROCESSORS 2)
    if(COSMICTIGER_WITH_SPH)
        # Relax sphere-surface shells on the CPU path that run.scenario=polytropicStar uses
        # without CUDA.
        add_hpx_executable(sphere_surface_test
            SOURCES tests/scenarios/sphere_surface.cpp src/scenarios/sphere.cpp src/core/assert.cpp
                src/core/hpx.cpp)
        target_link_libraries(sphere_surface_test PRIVATE cosmictiger_stacktrace)
        target_include_directories(sphere_surface_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(sphere_surface_test PRIVATE -ffast-math)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(sphere_surface_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME sphere_surface TARGET sphere_surface_test HPX_MAIN
            ARGS --hpx:threads=2 --hpx:bind=none
            ENVIRONMENT "CUDA_VISIBLE_DEVICES="
            PROPERTIES TIMEOUT 120 PROCESSORS 2)

        # Compile the production SIMD path without USE_CUDA or device kernels.
        add_executable(sph_cpu_test tests/sph/cpu.cpp src/sph/cpu.cpp
            src/sph/prehydro_cpu.cpp src/sph/hydro_cpu.cpp src/sph/rungs_cpu.cpp src/math/kernel.cpp)
        target_include_directories(sph_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(sph_cpu_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
        cosmictiger_use_generated_kernels(sph_cpu_test)
        target_link_libraries(sph_cpu_test PRIVATE HPX::hpx)
        target_compile_options(sph_cpu_test PRIVATE -ffast-math)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(sph_cpu_test PRIVATE -march=native)
        endif()
        # The compensated accumulation regression exercises four concurrent CPU writers.
        cosmictiger_add_gtest(NAME sph_cpu TARGET sph_cpu_test
            PROPERTIES TIMEOUT 120 PROCESSORS 4)

    endif()

    foreach(matrix_variant IN ITEMS standard fast_math)
        set(matrix_target matrix_${matrix_variant}_test)
        add_executable(${matrix_target} tests/matrix.cpp tests/matrix_eigen.cpp)
        target_include_directories(${matrix_target} PRIVATE ${PROJECT_SOURCE_DIR})
        if(matrix_variant STREQUAL "fast_math")
            target_compile_options(${matrix_target} PRIVATE -ffast-math)
        else()
            target_compile_options(${matrix_target} PRIVATE -fno-fast-math -ffp-contract=off)
        endif()
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(${matrix_target} PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME matrix_${matrix_variant} TARGET ${matrix_target}
            PROPERTIES TIMEOUT 30 LABELS "math;cpu")
    endforeach()
    if(COSMICTIGER_WITH_CUDA)
        add_executable(matrix_cuda_test tests/matrix_cuda.cu tests/matrix_eigen_cuda.cu)
        target_include_directories(matrix_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(matrix_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(matrix_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        cosmictiger_add_gtest(NAME matrix_cuda TARGET matrix_cuda_test
            PROPERTIES TIMEOUT 30 RESOURCE_LOCK cuda LABELS "math;cuda")
    endif()
    add_executable(matrix_benchmark EXCLUDE_FROM_ALL benchmarks/math/matrix.cpp)
    target_include_directories(matrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(matrix_benchmark PRIVATE -O3 -fno-fast-math -ffp-contract=off)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(matrix_benchmark PRIVATE -march=native)
    endif()

    add_executable(vector_test tests/vector.cpp)
    target_include_directories(vector_test PRIVATE ${PROJECT_SOURCE_DIR})
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(vector_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME vector TARGET vector_test PROPERTIES TIMEOUT 10)

    # Range and position arithmetic use the selected width and no GPU runtime.
    add_executable(numerical_utilities_test tests/numerical_utilities.cpp)
    target_include_directories(numerical_utilities_test PRIVATE ${PROJECT_SOURCE_DIR})
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(numerical_utilities_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME numerical_utilities TARGET numerical_utilities_test
        PROPERTIES TIMEOUT 10)

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(fmm_geometry_test tests/gravity/fmm_geometry.cpp)
        target_include_directories(fmm_geometry_test PRIVATE ${PROJECT_SOURCE_DIR})
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(fmm_geometry_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME fmm_geometry TARGET fmm_geometry_test
            PROPERTIES TIMEOUT 10)
    endif()

    if(COSMICTIGER_WITH_SPH)
        add_executable(rayleigh_taylor_test tests/rayleigh_taylor.cpp src/scenarios/rayleigh_taylor.cpp)
        target_include_directories(rayleigh_taylor_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(rayleigh_taylor_test PRIVATE ORDER=${cosmictiger_default_fmm_order})
        cosmictiger_use_generated_kernels(rayleigh_taylor_test)
        target_link_libraries(rayleigh_taylor_test PRIVATE HPX::hpx)
        cosmictiger_add_gtest(NAME rayleigh_taylor TARGET rayleigh_taylor_test
            PROPERTIES TIMEOUT 120)
    endif()

    add_executable(particle_selection_test tests/particle_selection.cpp)
    target_include_directories(particle_selection_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME particle_selection TARGET particle_selection_test
        PROPERTIES TIMEOUT 10)

    if(COSMICTIGER_WITH_GRAVITY)
        add_executable(gravity_direct_reference_test tests/gravity/direct_reference.cpp)
        target_include_directories(gravity_direct_reference_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(gravity_direct_reference_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME gravity_direct_reference TARGET gravity_direct_reference_test
            PROPERTIES TIMEOUT 30)

        add_executable(gravity_reference_table_test tests/gravity/reference_table.cpp
            src/gravity/reference_table.cpp)
        target_include_directories(gravity_reference_table_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(gravity_reference_table_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME gravity_reference_table TARGET gravity_reference_table_test
            PROPERTIES TIMEOUT 60)

        if(COSMICTIGER_WITH_CUDA)
            foreach(position_bits IN ITEMS 32 64)
                set(reference_target gravity_error_cuda_reference_${position_bits}_test)
                add_executable(${reference_target} tests/gravity/error_cuda.cpp
                    src/gravity/error_cuda.cu src/gravity/reference_table.cpp src/cuda/cuda_stream.cpp)
                target_include_directories(${reference_target} PRIVATE ${PROJECT_SOURCE_DIR})
                target_compile_definitions(${reference_target} PRIVATE USE_CUDA
                    COSMICTIGER_POSITION_BITS=${position_bits})
                target_link_libraries(${reference_target} PRIVATE HPX::hpx CUDA::cudart)
                target_compile_options(${reference_target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
                set_target_properties(${reference_target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                cosmictiger_add_gtest(NAME gravity_error_cuda_reference_${position_bits}
                    TARGET ${reference_target}
                    PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
            endforeach()
        endif()
    endif()

    if(COSMICTIGER_WITH_CUDA)
        if(COSMICTIGER_WITH_GRAVITY)
            add_executable(cartesian_kernels_cuda_test tests/gravity/cartesian_kernels_cuda.cu)
            target_include_directories(cartesian_kernels_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            cosmictiger_use_generated_kernels(cartesian_kernels_cuda_test)
            target_link_libraries(cartesian_kernels_cuda_test PRIVATE CUDA::cudart)
            cosmictiger_add_gtest(NAME cartesian_kernels_cuda TARGET cartesian_kernels_cuda_test
                PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

            add_executable(cartesian_l2l_cuda_test tests/gravity/cartesian_l2l_cuda.cu)
            target_include_directories(cartesian_l2l_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(cartesian_l2l_cuda_test PRIVATE USE_CUDA ORDER=${cosmictiger_default_fmm_order})
            cosmictiger_use_generated_kernels(cartesian_l2l_cuda_test)
            target_link_libraries(cartesian_l2l_cuda_test PRIVATE CUDA::cudart)
            target_compile_options(cartesian_l2l_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)
            cosmictiger_add_gtest(NAME cartesian_l2l_cuda TARGET cartesian_l2l_cuda_test
                PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

            add_executable(fmm_acceptance_cuda_test tests/gravity/fmm_acceptance_cuda.cu)
            target_include_directories(fmm_acceptance_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(fmm_acceptance_cuda_test PRIVATE USE_CUDA)
            target_link_libraries(fmm_acceptance_cuda_test PRIVATE CUDA::cudart)
            target_compile_options(fmm_acceptance_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math;-Xcompiler=-ffast-math>)
            cosmictiger_add_gtest(NAME fmm_acceptance_cuda TARGET fmm_acceptance_cuda_test
                PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)
        endif()

        add_executable(radial_kernel_cuda_test tests/radial_kernel_cuda.cu)
        target_include_directories(radial_kernel_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(radial_kernel_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(radial_kernel_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        cosmictiger_add_gtest(NAME radial_kernel_cuda TARGET radial_kernel_cuda_test
            PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)

        if(COSMICTIGER_WITH_SPH)
            add_executable(monaghan_sph_cuda_test tests/sph/cuda.cu
                tests/cuda_memory_backend.cpp
                src/core/flops.cu
                src/sph/prehydro.cu src/sph/hydro.cu src/sph/rungs.cu
                src/sph/cpu.cpp src/sph/prehydro_cpu.cpp src/sph/hydro_cpu.cpp src/sph/rungs_cpu.cpp
                src/math/kernel.cpp src/math/kernel.cu src/cuda/cuda_memory.cu src/core/assert.cu)
            target_include_directories(monaghan_sph_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(monaghan_sph_cuda_test PRIVATE
                USE_CUDA ORDER=${cosmictiger_default_fmm_order})
            cosmictiger_use_generated_kernels(monaghan_sph_cuda_test)
            target_link_libraries(monaghan_sph_cuda_test PRIVATE CUDA::cudart HPX::hpx)
            target_compile_options(monaghan_sph_cuda_test PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>
                $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
            if(COMPILER_SUPPORTS_MARCH_NATIVE)
                target_compile_options(monaghan_sph_cuda_test PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
            endif()
            set_target_properties(monaghan_sph_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
            cosmictiger_add_gtest(NAME monaghan_sph_cuda TARGET monaghan_sph_cuda_test
                PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
        endif()

        if(COSMICTIGER_WITH_ROCKSTAR)
            # These host regressions need CUDA headers, but do not initialize or run a GPU.
            add_executable(compression_test tests/compression.cpp src/analysis/compress.cpp)
            foreach(test_target IN ITEMS compression_test)
                target_include_directories(${test_target} PRIVATE ${PROJECT_SOURCE_DIR})
                target_compile_definitions(${test_target} PRIVATE USE_CUDA)
                target_link_libraries(${test_target} PRIVATE CUDA::cudart)
                if(COMPILER_SUPPORTS_MARCH_NATIVE)
                    target_compile_options(${test_target} PRIVATE -march=native)
                endif()
            endforeach()
            cosmictiger_add_gtest(NAME compression TARGET compression_test PROPERTIES TIMEOUT 10)
        endif()

        # Instantiate device-only template branches without requiring GPU execution.
        add_library(range_cuda_compile OBJECT tests/range_cuda_compile.cu)
        target_include_directories(range_cuda_compile PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(range_cuda_compile PRIVATE USE_CUDA)
        target_link_libraries(range_cuda_compile PRIVATE CUDA::cudart)
        set_target_properties(range_cuda_compile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        add_library(units_cuda_compile OBJECT tests/units_cuda_compile.cu)
        target_include_directories(units_cuda_compile PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(units_cuda_compile PRIVATE USE_CUDA)
        target_link_libraries(units_cuda_compile PRIVATE CUDA::cudart Boost::boost)
        set_target_properties(units_cuda_compile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        foreach(test_name IN ITEMS checkpoint_state_dm checkpoint_state_sph)
            if((NOT COSMICTIGER_WITH_SPH) AND
                    NOT test_name STREQUAL "checkpoint_state_dm")
                continue()
            endif()
            set(test_workdir "${PROJECT_BINARY_DIR}/tests/${test_name}")
            file(MAKE_DIRECTORY "${test_workdir}")
            if(test_name STREQUAL "checkpoint_state_sph")
                set(test_physics_args ${cosmictiger_test_hydro_args})
            else()
                set(test_physics_args --hydro.enabled=false)
            endif()
            if(COSMICTIGER_WITH_CUDA AND test_name STREQUAL "checkpoint_state_sph")
                set(test_cuda true)
            else()
                set(test_cuda false)
            endif()
            set(test_position_type ${COSMICTIGER_POSITION_TYPE})
            add_test(NAME ${test_name} COMMAND cosmictiger
                --run.test=checkpointState ${cosmictiger_test_gravity_args} --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${test_physics_args}
                --gravity.cuda=${test_cuda} --hydro.cuda=${test_cuda} --drift.cuda=${test_cuda} --compute.positionType=${test_position_type}
                --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY "${test_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)
        endforeach()
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        if(COSMICTIGER_WITH_SPH)
            add_test(NAME sph_rebuild_fraction_application COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_sph_rebuild_fraction.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/sph_rebuild_fraction_application)
            set_tests_properties(sph_rebuild_fraction_application PROPERTIES
                RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2 LABELS "integration;sph;gravity;octree;options")
        endif()
        if(COSMICTIGER_WITH_SPH)
            add_test(NAME sph_octree_application COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_sph_octree.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/sph_octree_application)
            set_tests_properties(sph_octree_application PROPERTIES
                RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2 LABELS "integration;sph;gravity;octree")
            if(COSMICTIGER_WITH_CUDA)
                add_test(NAME sph_octree_application_cuda COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_sph_octree.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/sph_octree_application_cuda --cuda)
                set_tests_properties(sph_octree_application_cuda PROPERTIES
                    RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2 LABELS "integration;sph;gravity;octree;cuda")
            endif()
            if(cosmictiger_mpi_tests)
                add_test(NAME sph_octree_application_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_sph_octree.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/sph_octree_application_mpi
                    ${cosmictiger_python_mpi_args})
                set_tests_properties(sph_octree_application_mpi PROPERTIES
                    RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 4 LABELS "integration;sph;gravity;octree;mpi")
            endif()
        endif()
        if(COSMICTIGER_WITH_SPH)
            set(sph_tree_workdir "${PROJECT_BINARY_DIR}/tests/sph_tree")
            file(MAKE_DIRECTORY "${sph_tree_workdir}")
            add_test(NAME sph_tree COMMAND cosmictiger --run.test=sphTree
                --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
            set_tests_properties(sph_tree PROPERTIES
                WORKING_DIRECTORY "${sph_tree_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 6)
        endif()
        if(COSMICTIGER_WITH_SPH)
            # SPH parent tree of HPX components (docs/sph_parent_tree_plan.md, M1): topology
            # for every rank count, then the distributed tree on 1, 2, 3 and 4 localities.
            set(sph_parent_tree_workdir "${PROJECT_BINARY_DIR}/tests/sph_parent_tree")
            file(MAKE_DIRECTORY "${sph_parent_tree_workdir}")
            add_test(NAME sph_parent_tree COMMAND cosmictiger --run.test=sphParentTree
                --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args}
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                --hpx:threads=2 --hpx:bind=none ${cosmictiger_test_integrator_args})
            set_tests_properties(sph_parent_tree PROPERTIES
                WORKING_DIRECTORY "${sph_parent_tree_workdir}" TIMEOUT 120 PROCESSORS 2
                LABELS "sph;cpu;hpx")
            if(cosmictiger_mpi_tests)
                foreach(parent_tree_ranks IN ITEMS 2 3 4)
                    set(parent_tree_name sph_parent_tree_mpi${parent_tree_ranks})
                    set(parent_tree_workdir "${PROJECT_BINARY_DIR}/tests/${parent_tree_name}")
                    file(MAKE_DIRECTORY "${parent_tree_workdir}")
                    add_test(NAME ${parent_tree_name} COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} ${parent_tree_ranks} ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS} --run.test=sphParentTree
                        --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                        ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false
                        --drift.cuda=false --hpx:threads=1 --hpx:bind=none
                        --hpx:localities=${parent_tree_ranks} ${cosmictiger_mpi_transport_args}
                        ${cosmictiger_test_integrator_args})
                    set_tests_properties(${parent_tree_name} PROPERTIES
                        WORKING_DIRECTORY "${parent_tree_workdir}" TIMEOUT 180
                        PROCESSORS ${parent_tree_ranks} LABELS "sph;cpu;hpx;mpi")
                endforeach()
            endif()
            # SPH coupled to PMMM self-gravity on the hierarchical ladder (M7): the
            # linear Jeans wave on 1 locality, and on 2 and 4 against it.
            if(COSMICTIGER_WITH_GRAVITY AND
                    COSMICTIGER_DIMENSION_COUNT EQUAL 3)
                set(jeans_wave_mpi_args)
                set(jeans_wave_processors 8)
                if(cosmictiger_mpi_tests)
                    set(jeans_wave_mpi_args ${cosmictiger_python_mpi_args})
                endif()
                add_test(NAME sph_jeans_wave COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_jeans_wave.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/sph_jeans_wave ${jeans_wave_mpi_args})
                set_tests_properties(sph_jeans_wave PROPERTIES
                    TIMEOUT 1500 PROCESSORS ${jeans_wave_processors}
                    LABELS "integration;sph;gravity;hierarchical;mpi")
                # Restart inside a PMMM outer interval with gas (M8), on 1 and 2 localities.
                add_test(NAME sph_pmmm_interval_restart COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_sph_pmmm_interval.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/sph_pmmm_interval_restart ${jeans_wave_mpi_args})
                set_tests_properties(sph_pmmm_interval_restart PROPERTIES
                    TIMEOUT 2400 PROCESSORS 4
                    LABELS "integration;sph;gravity;hierarchical;checkpoint;mpi")
            endif()
            # Distributed SPH evolution on ghost leaves against one locality.
            if(cosmictiger_mpi_tests)
                add_test(NAME sph_ghosts_application_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_sph_ghosts.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/sph_ghosts_application_mpi
                    ${cosmictiger_python_mpi_args})
                set_tests_properties(sph_ghosts_application_mpi PROPERTIES
                    TIMEOUT 1500 PROCESSORS 4 LABELS "integration;sph;octree;mpi")
            endif()
            # Ghost leaves (M2): the local ghost walk equals the global fetch walk.
            set(sph_ghosts_workdir "${PROJECT_BINARY_DIR}/tests/sph_ghosts")
            file(MAKE_DIRECTORY "${sph_ghosts_workdir}")
            add_test(NAME sph_ghosts COMMAND cosmictiger --run.test=sphGhosts
                --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args}
                --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                --hpx:threads=2 --hpx:bind=none ${cosmictiger_test_integrator_args})
            set_tests_properties(sph_ghosts PROPERTIES
                WORKING_DIRECTORY "${sph_ghosts_workdir}" TIMEOUT 120 PROCESSORS 2
                LABELS "sph;cpu;hpx")
            if(cosmictiger_mpi_tests)
                foreach(ghost_ranks IN ITEMS 2 3 4)
                    set(ghost_name sph_ghosts_mpi${ghost_ranks})
                    set(ghost_workdir "${PROJECT_BINARY_DIR}/tests/${ghost_name}")
                    file(MAKE_DIRECTORY "${ghost_workdir}")
                    add_test(NAME ${ghost_name} COMMAND ${MPIEXEC_EXECUTABLE}
                        ${MPIEXEC_NUMPROC_FLAG} ${ghost_ranks} ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS} --run.test=sphGhosts
                        --run.darkMatterParticleCount=512 --run.gasParticleCount=512
                        ${cosmictiger_test_hydro_args} --gravity.cuda=false --hydro.cuda=false
                        --drift.cuda=false --hpx:threads=1 --hpx:bind=none
                        --hpx:localities=${ghost_ranks} ${cosmictiger_mpi_transport_args}
                        ${cosmictiger_test_integrator_args})
                    set_tests_properties(${ghost_name} PROPERTIES
                        WORKING_DIRECTORY "${ghost_workdir}" TIMEOUT 180
                        PROCESSORS ${ghost_ranks} LABELS "sph;cpu;hpx;mpi")
                endforeach()
            endif()
        endif()
        foreach(kernel_case IN ITEMS wendland_spline ferrers_wendland)
            if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_WITH_CUDA)
                continue()
            endif()
            if(kernel_case STREQUAL "wendland_spline")
                set(kernel_options --extra-option=--hydro.kernelFamily=wendland
                    --extra-option=--hydro.kernelDegree=8 --extra-option=--gravity.kernelFamily=spline
                    --extra-option=--gravity.kernelDegree=9)
            else()
                set(kernel_options --extra-option=--hydro.kernelFamily=ferrers
                    --extra-option=--hydro.kernelDegree=6 --extra-option=--gravity.kernelFamily=wendland
                    --extra-option=--gravity.kernelDegree=5)
            endif()
            add_test(NAME smoke_kernels_${kernel_case} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/smoke_kernels_${kernel_case}
                --extra-option=--hydro.neighborNumber=120 ${kernel_options}
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
            set_tests_properties(smoke_kernels_${kernel_case} PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
        endforeach()
    endif()

    if(NOT COSMICTIGER_WITH_GRAVITY)
        add_test(NAME sph_only_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_sph_only.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/sph_only_cpu --position-type ${COSMICTIGER_POSITION_TYPE})
        set_tests_properties(sph_only_cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 600
            PROCESSORS 2 LABELS "sph;cpu;gravity_free")

        set(sph_only_tree_workdir "${PROJECT_BINARY_DIR}/tests/sph_tree_cpu")
        file(MAKE_DIRECTORY "${sph_only_tree_workdir}")
        add_test(NAME sph_tree_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=sphTree --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --hydro.enabled=true
            --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none
                ${cosmictiger_test_integrator_args})
        set_tests_properties(sph_tree_cpu PROPERTIES WORKING_DIRECTORY "${sph_only_tree_workdir}"
            RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2 LABELS "sph;cpu;gravity_free"
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1")

        if(cosmictiger_mpi_tests)
            set(sph_only_mpi_workdir "${PROJECT_BINARY_DIR}/tests/sph_only_mpi")
            set(sph_only_mpi_setup "${PROJECT_BINARY_DIR}/tests/prepare_sph_only_mpi.cmake")
            file(GENERATE OUTPUT "${sph_only_mpi_setup}" CONTENT
                "file(REMOVE_RECURSE [==[${sph_only_mpi_workdir}]==])\nfile(MAKE_DIRECTORY [==[${sph_only_mpi_workdir}]==])\n")
            add_test(NAME sph_only_mpi_prepare COMMAND ${CMAKE_COMMAND} -P "${sph_only_mpi_setup}")
            set_tests_properties(sph_only_mpi_prepare PROPERTIES
                FIXTURES_SETUP sph_only_mpi_directory LABELS "sph;cpu;gravity_free;mpi")
            add_test(NAME sph_only_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                --run.scenario=hydroWave --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --hydro.enabled=true
                --run.maxIterations=1 --run.endTime=1e-5 --gravity.cuda=false --hydro.cuda=false --drift.cuda=false
                --chemistry.enabled=false --output.directory=. --output.silo=false
                --analysis.groups=false --analysis.views=false --analysis.power=false
                --lightcone.enabled=false --checkpoint.intervalSeconds=0
                --hpx:threads=1 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_mpi_transport_args}
                ${cosmictiger_test_integrator_args})
            set_tests_properties(sph_only_mpi PROPERTIES WORKING_DIRECTORY "${sph_only_mpi_workdir}"
                FIXTURES_REQUIRED sph_only_mpi_directory RUN_SERIAL TRUE TIMEOUT 240 PROCESSORS 2
                LABELS "sph;cpu;gravity_free;mpi"
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        endif()
    endif()

    set(rejected_options_workdir "${PROJECT_BINARY_DIR}/tests/rejected_options")
    file(MAKE_DIRECTORY "${rejected_options_workdir}")
    add_test(NAME rejected_options COMMAND cosmictiger --invalid_option_for_test=true
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(rejected_options PROPERTIES WORKING_DIRECTORY "${rejected_options_workdir}"
        RUN_SERIAL TRUE TIMEOUT 20 WILL_FAIL TRUE)
    if(COSMICTIGER_WITH_GRAVITY)
        if(cosmictiger_mpi_tests)
            if(COSMICTIGER_WITH_SPH)
                set(distributed_output_mpi_workdir "${PROJECT_BINARY_DIR}/tests/distributed_output_mpi")
                file(MAKE_DIRECTORY "${distributed_output_mpi_workdir}")
                add_test(NAME distributed_output_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=distributedOutput --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args}
                    --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_test_integrator_args})
                set_tests_properties(distributed_output_mpi PROPERTIES
                    WORKING_DIRECTORY "${distributed_output_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 180 PROCESSORS 4)
            endif()
            if(COSMICTIGER_WITH_SPH)
                set(units_mpi_workdir "${PROJECT_BINARY_DIR}/tests/units_output_mpi")
                file(MAKE_DIRECTORY "${units_mpi_workdir}")
                add_test(NAME units_output_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=unitsOutput --run.darkMatterParticleCount=512 --run.gasParticleCount=512 ${cosmictiger_test_hydro_args}
                    --gravity.cuda=false --hydro.cuda=false --drift.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_test_integrator_args})
                set_tests_properties(units_output_mpi PROPERTIES
                    WORKING_DIRECTORY "${units_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
                if(COSMICTIGER_WITH_CUDA)
                add_test(NAME smoke_sph_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph_mpi
                    $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/smoke_sph_mpi --mpi-exec=${MPIEXEC_EXECUTABLE}
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
                set_tests_properties(smoke_sph_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
                endif()
                add_test(NAME smoke_sph_cpu_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph_mpi
                    $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/smoke_sph_cpu_mpi --mpi-exec=${MPIEXEC_EXECUTABLE} --cpu
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
                set_tests_properties(smoke_sph_cpu_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
            endif()
        endif()
        foreach(smoke_mode IN ITEMS sod sph sph_analysis)
            if(NOT COSMICTIGER_WITH_SPH)
                continue()
            endif()
            if(smoke_mode STREQUAL "sph_analysis" AND NOT COSMICTIGER_WITH_ROCKSTAR)
                continue()
            endif()
            add_test(NAME smoke_${smoke_mode}_cpu COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode}_cpu --cpu
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
            set_tests_properties(smoke_${smoke_mode}_cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        endforeach()
        # Variational SPH in Sod, blast and hydroWave, plus CFL refinement. These
        # ideal-gas cases must remain registered when equilibrium cooling is disabled.
        if(COSMICTIGER_WITH_SPH)
            foreach(backend IN ITEMS cpu mpi cuda cross_build)
                if((backend STREQUAL "mpi" AND NOT cosmictiger_mpi_tests) OR
                        (backend STREQUAL "cuda" AND NOT COSMICTIGER_WITH_CUDA) OR
                        (backend STREQUAL "cross_build" AND NOT COSMICTIGER_TEST_CUDA_EXECUTABLE))
                    continue()
                endif()
                set(variational_test variational_sph_application)
                set(variational_backend_args)
                if(backend STREQUAL "mpi")
                    set(variational_backend_args ${cosmictiger_python_mpi_args})
                elseif(backend STREQUAL "cuda")
                    set(variational_backend_args --cuda)
                elseif(backend STREQUAL "cross_build")
                    set(variational_backend_args --cuda
                        --cuda-executable=${COSMICTIGER_TEST_CUDA_EXECUTABLE})
                endif()
                if(NOT backend STREQUAL "cpu")
                    string(APPEND variational_test "_${backend}")
                endif()
                add_test(NAME ${variational_test} COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_variational_sph.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/${variational_test} --quick --jobs=1
                    ${variational_backend_args})
                set_tests_properties(${variational_test} PROPERTIES RUN_SERIAL TRUE TIMEOUT 900
                    PROCESSORS 4 SKIP_RETURN_CODE 77 LABELS "integration;sph;${backend}")
            endforeach()
        endif()
        foreach(smoke_mode IN ITEMS sod dm_cpu dm_cuda sph)
            if(NOT COSMICTIGER_WITH_CUDA AND NOT smoke_mode STREQUAL "dm_cpu")
                continue()
            endif()
            if((NOT COSMICTIGER_WITH_SPH) AND
                    NOT smoke_mode MATCHES "^dm_")
                continue()
            endif()
            add_test(NAME smoke_${smoke_mode} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode}
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
            set_tests_properties(smoke_${smoke_mode} PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        endforeach()
        add_test(NAME smoke_dm_min_rung COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu
            $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/smoke_dm_min_rung --minimum-rung=10
                --extra-option=--run.timeIntegrator=${cosmictiger_default_integrator})
        set_tests_properties(smoke_dm_min_rung PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        if(COSMICTIGER_WITH_GRAVITY)
            add_test(NAME checkpoint_integrity COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_checkpoint.py
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/checkpoint_integrity ${cosmictiger_sph_test_args})
            set_tests_properties(checkpoint_integrity PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
        endif()
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        set(position_selection_args)
        if(NOT COSMICTIGER_WITH_SPH)
            list(APPEND position_selection_args --without-sph)
        endif()
        add_test(NAME position_selection COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_position_selection.py
            $<TARGET_FILE:cosmictiger> ${PROJECT_BINARY_DIR}/tests/position_selection
            ${position_selection_args} --position-type ${COSMICTIGER_POSITION_TYPE}
            --time-integrator ${cosmictiger_default_integrator}
            --order ${cosmictiger_default_fmm_order})
        set_tests_properties(position_selection PROPERTIES RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2)
    endif()

endif()

if(BUILD_TESTING AND COSMICTIGER_DIMENSION_COUNT EQUAL 3)
    get_property(cosmictiger_targets_after_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    list(REMOVE_ITEM cosmictiger_targets_after_tests ${cosmictiger_targets_before_tests})
    cosmictiger_finalize_test_layout(${cosmictiger_targets_after_tests})
    # These standalone tests deliberately compile both position widths, rather
    # than inheriting the configured application's position definition.
    list(REMOVE_ITEM cosmictiger_targets_after_tests
        gravity_error_cuda_reference_32_test gravity_error_cuda_reference_64_test
        tree_source_moments_fixed32_test tree_source_moments_fixed64_test)
    foreach(test_target IN LISTS cosmictiger_targets_after_tests)
        get_target_property(test_type ${test_target} TYPE)
        if(NOT test_type STREQUAL "UTILITY")
            target_link_libraries(${test_target} PRIVATE cosmictiger_default_config)
        endif()
    endforeach()
endif()

# Dimensional mathematical checks are independent of HPX and the 3D fixture suite.
if(BUILD_TESTING AND COSMICTIGER_WITH_SPH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME sph_dimension_configuration COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_sph_dimensions.py ${PROJECT_SOURCE_DIR})
    if(COSMICTIGER_DIMENSION_COUNT LESS 3)
        add_test(NAME sph_dimension_application COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_sph_dimensions.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/dimensions ${COSMICTIGER_DIMENSION_COUNT}
            --integrator=${cosmictiger_default_integrator})
        set_tests_properties(sph_dimension_application PROPERTIES PROCESSORS 2 TIMEOUT 300 RUN_SERIAL TRUE)
    endif()
    foreach(dimension RANGE 1 3)
        add_executable(sph_dimension_${dimension}_test tests/sph/dimensions.cpp)
        target_include_directories(sph_dimension_${dimension}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(sph_dimension_${dimension}_test PRIVATE COSMICTIGER_DIMENSION_COUNT=${dimension})
        add_test(NAME sph_dimension_${dimension} COMMAND sph_dimension_${dimension}_test)
    endforeach()
endif()

# Cell-lattice geometry and selector validation do not depend on HPX or gravity.
if(BUILD_TESTING)
    add_executable(cell_lattice_test tests/cell_lattice.cpp)
    target_include_directories(cell_lattice_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(cell_lattice_test PRIVATE COSMICTIGER_DIMENSION_COUNT=${COSMICTIGER_DIMENSION_COUNT})
    add_test(NAME cell_lattice COMMAND cell_lattice_test)
    set_tests_properties(cell_lattice PROPERTIES TIMEOUT 30 LABELS "scenarios;cpu")
endif()

if(BUILD_TESTING AND COSMICTIGER_WITH_SPH AND COSMICTIGER_DIMENSION_COUNT GREATER_EQUAL 2)
    add_test(NAME kelvin_helmholtz_packing_application COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_kelvin_helmholtz_packings.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/kelvin_helmholtz_packings ${COSMICTIGER_DIMENSION_COUNT})
    set_tests_properties(kelvin_helmholtz_packing_application PROPERTIES
        PROCESSORS 2 TIMEOUT 300 RUN_SERIAL TRUE LABELS "scenarios;sph;cpu")
endif()
