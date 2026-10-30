# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

cmake_minimum_required(VERSION 3.24)
project(cosmictiger LANGUAGES CXX C)
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
option(COSMICTIGER_WITH_SPH "Enable SPH in the default executable and tests" ON)
set(COSMICTIGER_POSITION_TYPE "fixed32" CACHE STRING
    "Particle position storage for every target: fixed32 or fixed64")
set_property(CACHE COSMICTIGER_POSITION_TYPE PROPERTY STRINGS fixed32 fixed64)
if(NOT COSMICTIGER_POSITION_TYPE MATCHES "^(fixed32|fixed64)$")
    message(FATAL_ERROR "COSMICTIGER_POSITION_TYPE must be fixed32 or fixed64")
endif()
string(REPLACE "fixed" "" COSMICTIGER_POSITION_BITS "${COSMICTIGER_POSITION_TYPE}")
option(COSMICTIGER_ENABLE_PROFILING "Enable statistical CPU sampling and PROFILE() metadata" ON)
set(cosmictiger_require_apex_default OFF)
if(CMAKE_BUILD_TYPE MATCHES "^(RelWithDebInfo|Debug|Profile)$")
    set(cosmictiger_require_apex_default ON)
endif()
option(COSMICTIGER_REQUIRE_APEX "Require HPX task profiling through an APEX-enabled HPX installation"
    ${cosmictiger_require_apex_default})
option(COSMICTIGER_WITH_LONG_LONG_PART_INT "Use long long to index local particles" OFF)
# Each executable has one time integrator, one gravity solver and one FMM order.
set(COSMICTIGER_TIME_INTEGRATOR "leapfrog" CACHE STRING
    "Default particle time integrator: leapfrog or hierarchical")
set_property(CACHE COSMICTIGER_TIME_INTEGRATOR PROPERTY STRINGS leapfrog hierarchical)
if(NOT COSMICTIGER_TIME_INTEGRATOR MATCHES "^(leapfrog|hierarchical)$")
    message(FATAL_ERROR "COSMICTIGER_TIME_INTEGRATOR must be leapfrog or hierarchical")
endif()
set(COSMICTIGER_GRAVITY_METHOD "fmm" CACHE STRING "Default gravity solver: fmm or pmmm")
set_property(CACHE COSMICTIGER_GRAVITY_METHOD PROPERTY STRINGS fmm pmmm)
if(NOT COSMICTIGER_GRAVITY_METHOD MATCHES "^(fmm|pmmm)$")
    message(FATAL_ERROR "COSMICTIGER_GRAVITY_METHOD must be fmm or pmmm")
endif()
set(COSMICTIGER_PMMM_NEAR_METHOD "pp" CACHE STRING "Default PMMM near solver: pp or fmm")
set_property(CACHE COSMICTIGER_PMMM_NEAR_METHOD PROPERTY STRINGS pp fmm)
if(NOT COSMICTIGER_PMMM_NEAR_METHOD MATCHES "^(pp|fmm)$")
    message(FATAL_ERROR "COSMICTIGER_PMMM_NEAR_METHOD must be pp or fmm")
endif()
if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
    # The unused near method must not create a second global-FMM configuration.
    set(COSMICTIGER_PMMM_NEAR_METHOD pp)
endif()
# FMM order: multipole degree p plus two. Source multipoles carry degree p,
# local expansions degree p + 1, and Ewald derivatives ranks through p + 1.
set(cosmictiger_default_order 7)
if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
    set(cosmictiger_default_order 8)
endif()
set(COSMICTIGER_WITH_ORDER "${cosmictiger_default_order}" CACHE STRING
    "Default FMM/PMMM order (3 to 9; default 7 for FMM, 8 for PMMM)")
set_property(CACHE COSMICTIGER_WITH_ORDER PROPERTY STRINGS 3 4 5 6 7 8 9)
if(NOT COSMICTIGER_WITH_ORDER MATCHES "^[3-9]$")
    message(FATAL_ERROR
        "COSMICTIGER_WITH_ORDER must be an integer from 3 to 9; got '${COSMICTIGER_WITH_ORDER}'")
endif()
# Retire the runtime-order range, including values from an older build cache.
unset(COSMICTIGER_FMM_MIN_ORDER CACHE)
unset(COSMICTIGER_FMM_MAX_ORDER CACHE)
set(COSMICTIGER_FMM_MIN_ORDER ${COSMICTIGER_WITH_ORDER})
set(COSMICTIGER_FMM_MAX_ORDER ${COSMICTIGER_WITH_ORDER})
set(COSMICTIGER_FMM_ORDERS ${COSMICTIGER_WITH_ORDER})
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
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
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
if(NOT COSMICTIGER_ENABLE_PROFILING)
    target_compile_definitions(cosmictiger_profiler PUBLIC PROFILE_DISABLE=1)
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
find_package(GSL REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
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

find_path(SILO_INCLUDE_DIR silo.h REQUIRED)
find_library(SILO_LIBRARY NAMES siloh5 silo REQUIRED)
find_path(CHEALPIX_INCLUDE_DIR chealpix.h REQUIRED)
find_library(CHEALPIX_LIBRARY NAMES chealpix REQUIRED)
find_path(HEALPIX_CXX_INCLUDE_DIR healpix_base.h PATH_SUFFIXES healpix_cxx REQUIRED)
find_library(HEALPIX_CXX_LIBRARY NAMES healpix_cxx REQUIRED)

message(STATUS "cosmictiger SPH = ${COSMICTIGER_WITH_SPH}")
message(STATUS "cosmictiger position storage = ${COSMICTIGER_POSITION_TYPE}")
message(STATUS "cosmictiger time integrator = ${COSMICTIGER_TIME_INTEGRATOR}")
message(STATUS "cosmictiger gravity solver = ${COSMICTIGER_GRAVITY_METHOD}")
if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
    message(STATUS "cosmictiger PMMM near solver = ${COSMICTIGER_PMMM_NEAR_METHOD}")
endif()
message(STATUS "cosmictiger compiled FMM order = ${COSMICTIGER_WITH_ORDER}")
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
    cosmictiger/chemistry/primordial_chemistry.hpp
    cosmictiger/chemistry/primordial_eos.hpp
    cosmictiger/chemistry/primordial_float_environment.hpp
    cosmictiger/chemistry/primordial_rates.hpp
    cosmictiger/chemistry/primordial_solver.hpp
    cosmictiger/chemistry/primordial_types.hpp
    cosmictiger/chemistry/primordial_units.hpp
    cosmictiger/chemistry/recombination.hpp
    cosmictiger/core/assert.hpp
    cosmictiger/core/containers.hpp
    cosmictiger/core/defs.hpp
    cosmictiger/core/driver.hpp
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
    cosmictiger/cosmology/evolution.hpp
    cosmictiger/cosmology/friedmann.hpp
    cosmictiger/cosmology/cosmology.hpp
    cosmictiger/cosmology/fft.hpp
    cosmictiger/cosmology/initialize.hpp
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
    cosmictiger/gravity/dehnen/ewald_indices.hpp
    cosmictiger/gravity/dehnen/ewald_tables.hpp
    cosmictiger/gravity/fmm_kernels/cartesian_types.hpp
    cosmictiger/gravity/fmm_acceptance.hpp
    cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp
    cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp
    cosmictiger/gravity/fmm_order.hpp
    cosmictiger/gravity/fmm_types.hpp
    cosmictiger/gravity/fmm_kernels/spherical_operators.hpp
    cosmictiger/gravity/fmm_kernels/spherical_types.hpp
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
    cosmictiger/math/float_float.hpp
    cosmictiger/math/fp16.hpp
    cosmictiger/math/interpolation.hpp
    cosmictiger/math/kernel.hpp
    cosmictiger/math/math.hpp
    cosmictiger/math/position.hpp
    cosmictiger/math/radial_density_kernel.hpp
    cosmictiger/math/radial_kernel.hpp
    cosmictiger/math/range.hpp
    cosmictiger/math/simd.hpp
    cosmictiger/math/simd_backend.hpp
    cosmictiger/math/simd_math.hpp
    cosmictiger/math/simulation_units.hpp
    cosmictiger/math/tensor.hpp
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
    cosmictiger/scenarios/kelvin_helmholtz.hpp
    cosmictiger/scenarios/disc.hpp
    cosmictiger/scenarios/polytropic_star.hpp
    cosmictiger/scenarios/hydro_wave.hpp
    cosmictiger/scenarios/cosmology.hpp
    cosmictiger/scenarios/glass.hpp
    cosmictiger/scenarios/exact_sod.hpp
    cosmictiger/scenarios/sedov.hpp
    cosmictiger/sph/sph.hpp
    cosmictiger/sph/sph_chemistry.hpp
    cosmictiger/sph/sph_cpu.hpp
    cosmictiger/sph/sph_cuda.hpp
    cosmictiger/sph/sph_kernel.hpp
    cosmictiger/sph/sph_particles.hpp
    cosmictiger/sph/sph_tree.hpp
    cosmictiger/scenarios/sphere.hpp
    cosmictiger/core/tree_id.hpp
    cosmictiger/gravity/dehnen/cpu_instances.hpp
    cosmictiger/gravity/dehnen/distributed_tree.hpp
    cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp
    cosmictiger/gravity/dehnen/execution.hpp
    cosmictiger/gravity/dehnen/gpu_staging.hpp
    cosmictiger/gravity/dehnen/particle_exchange.hpp
    cosmictiger/gravity/dehnen/periodic.hpp
    cosmictiger/gravity/dehnen/periodic_cpu.hpp
    cosmictiger/gravity/dehnen/periodic_cuda.hpp
    cosmictiger/gravity/dehnen/remote_cache.hpp
    cosmictiger/gravity/fmm_geometry.hpp
    cosmictiger/gravity/kick_context.hpp
    cosmictiger/gravity/nitadori/cell_tree.hpp
    cosmictiger/gravity/nitadori/collective_dispatch.hpp
    cosmictiger/gravity/nitadori/fmm_batch.hpp
    cosmictiger/gravity/nitadori/ghost_storage.hpp
    cosmictiger/gravity/nitadori/kernels.hpp
    cosmictiger/gravity/nitadori/kick.hpp
    cosmictiger/gravity/nitadori/near_pp.hpp
    cosmictiger/gravity/nitadori/solver.hpp
    cosmictiger/gravity/source_access.hpp
    cosmictiger/gravity/sources.hpp
    cosmictiger/gravity/tree_build.hpp
    cosmictiger/sph/balsara/shear_limiter.hpp
    cosmictiger/sph/cullen_and_dehnen/particle_exchange.hpp
    cosmictiger/sph/cullen_and_dehnen/state.hpp
    cosmictiger/sph/cullen_and_dehnen/viscosity.hpp
    cosmictiger/sph/kernels.hpp
    cosmictiger/sph/monaghan/equations.hpp
    cosmictiger/sph/monaghan/viscosity.hpp
    cosmictiger/sph/morris_and_monaghan/viscosity.hpp
    cosmictiger/sph/price_etal/viscosity.hpp
    cosmictiger/sph/sph_flops.hpp
    cosmictiger/sph/viscosity_state.hpp
)

if(BUILD_TESTING)
    # Only option validation and dispatch depend on test availability.
    set_property(SOURCE src/main.cpp src/core/options.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS COSMICTIGER_BUILD_TESTING)
endif()

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

# Chemistry must not inherit the application's fast-math/FTZ settings: the
# signed-complement and FloatFloat contracts require IEEE rounding and -0.
target_link_libraries(cosmictiger_profiler PRIVATE Threads::Threads)
target_link_libraries(cosmictiger_stacktrace PRIVATE Threads::Threads)
# Homogeneous recombination is shared by every variant. Particle chemistry is
# compiled only when an SPH executable (or its tests) is requested.
add_library(cosmictiger_recombination STATIC src/chemistry/recombination.cpp)
add_library(cosmictiger_sph_chemistry STATIC EXCLUDE_FROM_ALL src/chemistry/primordial_chemistry.cpp)
foreach(chemistry_target IN ITEMS cosmictiger_recombination cosmictiger_sph_chemistry)
    target_include_directories(${chemistry_target} PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries(${chemistry_target} PUBLIC Threads::Threads Boost::boost)
    target_compile_options(${chemistry_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
endforeach()
target_link_libraries(cosmictiger_sph_chemistry PUBLIC cosmictiger_recombination)
if(COSMICTIGER_WITH_CUDA)
    target_sources(cosmictiger_sph_chemistry PRIVATE src/chemistry/primordial_chemistry.cu)
    target_compile_definitions(cosmictiger_sph_chemistry PRIVATE USE_CUDA)
    target_link_libraries(cosmictiger_sph_chemistry PUBLIC CUDA::cudart)
    target_compile_options(cosmictiger_sph_chemistry PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false>)
endif()
# Preserve the existing test-facing dependency for the configured default.
add_library(primordial_chemistry INTERFACE)
target_link_libraries(primordial_chemistry INTERFACE cosmictiger_recombination)
if(COSMICTIGER_WITH_SPH)
    target_link_libraries(primordial_chemistry INTERFACE cosmictiger_sph_chemistry)
endif()
set_source_files_properties(src/sph/sph_chemistry.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# The adaptive Boltzmann solve needs finite-value checks and accurate
# cancellation in the synchronous Einstein constraints.
set_source_files_properties(src/cosmology/boltzmann.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Background and clock guards must remain effective in release builds.
set_source_files_properties(src/core/driver.cpp src/gravity/error.cpp src/gravity/reference_table.cpp
    src/gravity/nitadori/solver.cpp src/gravity/nitadori/kernels.cpp src/gravity/nitadori/near_pp.cpp tests/gravity/nitadori/pmmm.cpp
    src/gravity/nitadori/kick.cpp src/cosmology/friedmann.cpp src/cosmology/cosmology.cpp
    src/cosmology/background.cpp src/cosmology/evolution.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Keep the diagnostic's compensated double sums intact on both host and device.
set_source_files_properties(src/gravity/error_cuda.cu PROPERTIES COMPILE_OPTIONS
    "--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off")
set_source_files_properties(
    src/gravity/nitadori/fft.cpp
    src/gravity/nitadori/mesh_exchange.cpp
    src/gravity/nitadori/ghost_exchange.cpp
    src/gravity/nitadori/cell_schedule.cpp
    src/gravity/nitadori/evaluation.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# Domain boundary coordinates must round identically when stored and reloaded.
set_source_files_properties(src/particles/domain.cpp tests/mesh_domains.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")

# Checked-in kernels for FMM orders 3 to 9, regenerated only on explicit request.
# Both representations share the same order range. Only Cartesian is currently
# connected to the application walk. Ordinary builds use the source headers.
set(COSMICTIGER_GENERATED_FMM_DIR ${PROJECT_SOURCE_DIR}/cosmictiger/gravity/fmm_kernels)
set(COSMICTIGER_GENERATED_FMM_FILES
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_config.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_m2l.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/cartesian_ewald_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/spherical_ewald_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/operation_counts.json)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Do not declare source headers as OUTPUT/BYPRODUCTS or mark them GENERATED:
# build-system clean targets must never delete these version-controlled files.
add_custom_target(generate_fmm_kernels
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_fmm.py
        --output-dir ${COSMICTIGER_GENERATED_FMM_DIR} --min-order 2 --max-order 8
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Regenerating source headers for Cartesian and Spherical FMM orders 3 to 9"
    VERBATIM)
add_custom_target(generate_cartesian_kernels DEPENDS generate_fmm_kernels)
add_custom_target(cosmictiger_generated_kernels DEPENDS generate_fmm_kernels)
# Compile a target against the generated kernels and selected order/bucket cap.
function(cosmictiger_use_generated_kernels target)
    target_include_directories(${target} BEFORE PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(${target} PRIVATE ORDER=${COSMICTIGER_WITH_ORDER}
        MAX_BUCKET_SIZE=${COSMICTIGER_WITH_MAX_BUCKET})
endfunction()
include(cmake/variants.cmake)

add_executable(healpix2silo src/tools/healpix2silo.cpp)
target_link_libraries(healpix2silo PRIVATE ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} Boost::program_options dl)
target_include_directories(healpix2silo PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(healpix2silo SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})

add_hpx_executable(groups2halofunc
    SOURCES src/core/assert.cpp src/core/hpx.cpp src/tools/groups2halofunc.cpp src/analysis/group_entry.cpp)
target_link_libraries(groups2halofunc PUBLIC Boost::program_options PRIVATE cosmictiger_stacktrace)
set_target_properties(groups2halofunc PROPERTIES ENABLE_EXPORTS ON)
target_include_directories(groups2halofunc PRIVATE ${PROJECT_SOURCE_DIR})

foreach(tool_target IN ITEMS healpix2silo groups2halofunc)
    target_compile_options(${tool_target} PRIVATE -Wno-attributes -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(${tool_target} PRIVATE -march=native)
    endif()
endforeach()

foreach(tool_target IN ITEMS halo_catalog_export healpix2silo groups2halofunc)
    target_link_libraries(${tool_target} PRIVATE cosmictiger_default_config)
endforeach()

# Tests exercise the selected default. Capture their targets rather than adding
# directory-wide definitions, which would also contaminate the named variants.
get_property(cosmictiger_targets_before_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)

if(BUILD_TESTING)
    add_test(NAME build_features COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_build_features.py ${PROJECT_BINARY_DIR})
    set_tests_properties(build_features PROPERTIES TIMEOUT 30 LABELS "build")
    add_test(NAME build_variants_checker COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_build_variants.py)
    set_tests_properties(build_variants_checker PROPERTIES TIMEOUT 30 LABELS "build")
    set(cosmictiger_mpi_transport_args --hpx:ini=hpx.parcel.mpi.enable=1)
    if(HPX_WITH_PARCELPORT_TCP)
        list(APPEND cosmictiger_mpi_transport_args --hpx:ini=hpx.parcel.tcp.enable=0)
    endif()
    set(cosmictiger_test_gravity_args)
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
        set(cosmictiger_test_gravity_args --gravity.meshSize=8)
    endif()
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
        if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
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

    add_executable(profiler_disabled_test tests/profiler_disabled.cpp)
    target_include_directories(profiler_disabled_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(profiler_disabled_test PRIVATE PROFILE_DISABLE=1)
    add_test(NAME profiler_disabled COMMAND profiler_disabled_test)
    set_tests_properties(profiler_disabled PROPERTIES TIMEOUT 10 LABELS "profiler;cpu")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME mpi_interrupt_launcher COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_mpi_launcher.py ${PROJECT_SOURCE_DIR}/scripts/run_mpi.py)
        set_tests_properties(mpi_interrupt_launcher PROPERTIES TIMEOUT 60 PROCESSORS 2
            LABELS "profiler;interrupt;cpu")
    endif()
    if(COSMICTIGER_ENABLE_PROFILING AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64)$")
        add_hpx_executable(profiler_test SOURCES tests/profiler.cpp)
        target_link_libraries(profiler_test PRIVATE cosmictiger_profiler)
        add_test(NAME profiler COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_profiler.py $<TARGET_FILE:profiler_test>)
        set_tests_properties(profiler PROPERTIES TIMEOUT 90 PROCESSORS 2
            LABELS "profiler;cpu" ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        if(HPX_WITH_APEX)
            add_hpx_executable(apex_profiler_test SOURCES tests/apex_profiler.cpp)
            # Match the application's exported symbols for APEX address lookup.
            set_target_properties(apex_profiler_test PROPERTIES ENABLE_EXPORTS ON)
            target_link_libraries(apex_profiler_test PRIVATE cosmictiger_profiler)
            add_test(NAME apex_profiler COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_apex_profiler.py $<TARGET_FILE:apex_profiler_test>)
            add_test(NAME apex_profiler_without_sampler COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_apex_profiler.py $<TARGET_FILE:apex_profiler_test>
                --sampler-disabled)
            set_tests_properties(apex_profiler apex_profiler_without_sampler
                PROPERTIES TIMEOUT 90 PROCESSORS 2
                LABELS "profiler;apex;cpu;hpx"
                ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
            add_test(NAME apex_reports_disabled COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_apex_profiler.py $<TARGET_FILE:apex_profiler_test>
                --reports-disabled)
            set_tests_properties(apex_reports_disabled PROPERTIES TIMEOUT 90 PROCESSORS 2
                LABELS "profiler;apex;cpu;hpx")
            add_hpx_executable(apex_interrupt_disabled_test SOURCES tests/apex_profiler.cpp
                src/core/profiler.cpp src/core/interrupt_profile.cpp src/core/profiling_runtime.cpp)
            target_include_directories(apex_interrupt_disabled_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(apex_interrupt_disabled_test PRIVATE PROFILE_DISABLE=1)
            target_link_libraries(apex_interrupt_disabled_test PRIVATE ${CMAKE_DL_LIBS} rt)
            set_target_properties(apex_interrupt_disabled_test PROPERTIES ENABLE_EXPORTS ON)
            foreach(interrupt_mode IN ITEMS enabled disabled)
                set(interrupt_fixture apex_profiler_test)
                set(interrupt_args)
                if(interrupt_mode STREQUAL "disabled")
                    set(interrupt_fixture apex_interrupt_disabled_test)
                    list(APPEND interrupt_args --require-compiled-disabled)
                endif()
                add_test(NAME apex_interrupt_${interrupt_mode} COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_interrupt_profiler.py
                    $<TARGET_FILE:${interrupt_fixture}> ${interrupt_args})
                set_tests_properties(apex_interrupt_${interrupt_mode} PROPERTIES
                    TIMEOUT 120 PROCESSORS 2 LABELS "profiler;apex;cpu;hpx;interrupt")
            endforeach()
            find_program(COSMICTIGER_TAU_EXEC tau_exec
                HINTS "$ENV{TAU_ROOT}/bin" "$ENV{TAU_ROOT}/x86_64/bin"
                    "$ENV{TAU_ROOT}/arm64_linux/bin" "$ENV{HOME}/local/tau/bin"
                    "$ENV{HOME}/local/tau/x86_64/bin" "$ENV{HOME}/local/tau/arm64_linux/bin")
            if(COSMICTIGER_TAU_EXEC)
                foreach(sampler_mode IN ITEMS enabled disabled config interrupt interrupt_disabled interrupt_usr1)
                    set(tau_test_args)
                    set(tau_fixture apex_profiler_test)
                    if(sampler_mode STREQUAL "disabled")
                        list(APPEND tau_test_args --sampler-disabled)
                    elseif(sampler_mode STREQUAL "config")
                        list(APPEND tau_test_args --apex-config)
                    elseif(sampler_mode MATCHES "^interrupt")
                        list(APPEND tau_test_args --interrupt-exit)
                        if(sampler_mode STREQUAL "interrupt_disabled")
                            set(tau_fixture apex_interrupt_disabled_test)
                            list(APPEND tau_test_args --require-compiled-disabled)
                        elseif(sampler_mode STREQUAL "interrupt_usr1")
                            list(APPEND tau_test_args --interrupt-signal=SIGUSR1)
                        endif()
                    endif()
                    add_test(NAME tau_profiler_${sampler_mode} COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_tau_profiler.py
                        $<TARGET_FILE:${tau_fixture}>
                        ${PROJECT_SOURCE_DIR}/scripts/profile_tau.sh ${COSMICTIGER_TAU_EXEC}
                        ${tau_test_args})
                    set_tests_properties(tau_profiler_${sampler_mode} PROPERTIES
                        TIMEOUT 120 PROCESSORS 2 LABELS "profiler;apex;tau;cpu;hpx")
                endforeach()
                if(COSMICTIGER_WITH_SPH AND COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                    add_test(NAME tau_interrupt_simulation_hydro COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_interrupt_simulation.py
                        $<TARGET_FILE:cosmictiger> --scenario=hydroWave
                        --launcher=${PROJECT_SOURCE_DIR}/scripts/profile_tau.sh
                        --tau-exec=${COSMICTIGER_TAU_EXEC})
                    set_tests_properties(tau_interrupt_simulation_hydro PROPERTIES TIMEOUT 240
                        PROCESSORS 2 LABELS "profiler;apex;tau;cpu;hpx;interrupt")
                endif()
                if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
                    set(tau_mpi_args --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG})
                    foreach(flag IN LISTS MPIEXEC_PREFLAGS)
                        list(APPEND tau_mpi_args --mpi-preflag=${flag})
                    endforeach()
                    foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
                        list(APPEND tau_mpi_args --mpi-postflag=${flag})
                    endforeach()
                    add_test(NAME tau_profiler_mpi_startup COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_tau_mpi_startup.py
                        $<TARGET_FILE:cosmictiger>
                        ${PROJECT_SOURCE_DIR}/scripts/profile_tau.sh ${COSMICTIGER_TAU_EXEC}
                        ${MPIEXEC_EXECUTABLE} ${tau_mpi_args})
                    set_tests_properties(tau_profiler_mpi_startup PROPERTIES
                        TIMEOUT 120 PROCESSORS 4 LABELS "profiler;apex;tau;cpu;hpx;mpi")
                    add_test(NAME tau_interrupt_simulation_mpi_worker COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/run_interrupt_simulation.py
                        $<TARGET_FILE:cosmictiger> --scenario=plummer
                        --mpi-exec=${MPIEXEC_EXECUTABLE} ${tau_mpi_args}
                        --launcher=${PROJECT_SOURCE_DIR}/scripts/profile_tau.sh
                        --tau-exec=${COSMICTIGER_TAU_EXEC})
                    set_tests_properties(tau_interrupt_simulation_mpi_worker PROPERTIES
                        TIMEOUT 240 PROCESSORS 4 LABELS "profiler;apex;tau;cpu;hpx;mpi;interrupt")
                endif()
            endif()
        endif()
    endif()

    # Keep each assertion group visible to CTest, with the same resource limits
    # and compiler variants as the standalone regression executables.
    function(cosmictiger_add_gtest)
        cmake_parse_arguments(PARSE_ARGV 0 test "HPX_MAIN" "NAME;TARGET;WORKING_DIRECTORY;ENVIRONMENT"
            "ARGS;PROPERTIES")
        if(test_HPX_MAIN)
            target_link_libraries(${test_TARGET} PRIVATE GTest::gtest)
        else()
            target_link_libraries(${test_TARGET} PRIVATE GTest::gtest_main)
        endif()
        if(NOT test_WORKING_DIRECTORY)
            set(test_WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
        endif()
        gtest_discover_tests(${test_TARGET}
            NO_PRETTY_VALUES
            DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 30
            TEST_PREFIX "${test_NAME}." EXTRA_ARGS ${test_ARGS}
            WORKING_DIRECTORY "${test_WORKING_DIRECTORY}"
            PROPERTIES LABELS "${test_NAME}" ${test_PROPERTIES})
        # GoogleTest's discovery helper expands list-valued properties. Apply
        # environment lists after discovery so semicolons remain part of the value.
        if(DEFINED test_ENVIRONMENT)
            set(environment_file "${PROJECT_BINARY_DIR}/${test_TARGET}_environment.cmake")
            file(WRITE "${environment_file}"
                "if(DEFINED ${test_TARGET}_TESTS)\n"
                "    set_tests_properties(\${${test_TARGET}_TESTS} PROPERTIES\n"
                "        ENVIRONMENT [==[${test_ENVIRONMENT}]==])\n"
                "endif()\n")
            set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${environment_file}")
        endif()
    endfunction()

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

    add_executable(opening_schedule_test tests/gravity/opening_schedule.cpp)
    target_include_directories(opening_schedule_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(opening_schedule_test PRIVATE -ffast-math)
    cosmictiger_add_gtest(NAME opening_schedule TARGET opening_schedule_test
        PROPERTIES TIMEOUT 30)

    add_executable(cosmology_evolution_test tests/cosmology_evolution.cpp)
    target_include_directories(cosmology_evolution_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(cosmology_evolution_test PRIVATE -fno-fast-math -ffp-contract=off)
    cosmictiger_add_gtest(NAME cosmology_evolution TARGET cosmology_evolution_test
        PROPERTIES TIMEOUT 30)
    add_executable(terminal_test tests/terminal.cpp)
    target_include_directories(terminal_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(terminal_test PRIVATE -ffast-math)
    cosmictiger_add_gtest(NAME terminal TARGET terminal_test PROPERTIES TIMEOUT 10)
    # The collisionless checks also exercise terminal hierarchy synchronization.
    # Only leapfrog builds with SPH include the additional gas evolution case.
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
        foreach(cosmology_backend IN ITEMS cpu cuda)
            if(cosmology_backend STREQUAL "cuda" AND NOT COSMICTIGER_WITH_CUDA)
                continue()
            endif()
            set(cosmology_integration_name "cosmology_integration_${cosmology_backend}")
            if(cosmology_backend STREQUAL "cpu")
                set(cosmology_integration_name cosmology_integration)
            endif()
            set(cosmology_backend_args)
            if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                list(APPEND cosmology_backend_args --collisionless-only)
            endif()
            if(cosmology_backend STREQUAL "cuda")
                list(APPEND cosmology_backend_args --cuda)
            endif()
            add_test(NAME ${cosmology_integration_name} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_cosmology.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/${cosmology_integration_name}
                --position-type ${COSMICTIGER_POSITION_TYPE} ${cosmology_backend_args})
            set_tests_properties(${cosmology_integration_name}
                PROPERTIES RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2 LABELS "cosmology;${cosmology_backend}")
        endforeach()
    endif()

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

    add_hpx_executable(flops_test SOURCES tests/flops.cpp src/core/flops.cpp)
    target_include_directories(flops_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME flops TARGET flops_test HPX_MAIN
        ARGS --hpx:threads=2 --hpx:bind=none
        PROPERTIES TIMEOUT 30 PROCESSORS 6)
    if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
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

    add_executable(halo_catalog_fixture tests/halo_catalog_fixture.cpp)
    target_link_libraries(halo_catalog_fixture PRIVATE cosmictiger_fp16)
    target_compile_options(halo_catalog_fixture PRIVATE -fno-fast-math)
    add_test(NAME halo_viewer COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_halo_viewer.py
        --exporter $<TARGET_FILE:halo_catalog_export>
        --fixture $<TARGET_FILE:halo_catalog_fixture>
        --viewer ${PROJECT_SOURCE_DIR}/tools/halo_viewer/halo_viewer.py)
    set_tests_properties(halo_viewer PROPERTIES TIMEOUT 180 PROCESSORS 2)

    if(COSMICTIGER_WITH_SPH)
        foreach(chemistry_test IN ITEMS primordial_rates primordial_solver primordial_units)
            add_executable(${chemistry_test}_test tests/${chemistry_test}.cpp)
            target_link_libraries(${chemistry_test}_test PRIVATE primordial_chemistry)
            target_compile_options(${chemistry_test}_test PRIVATE -fno-fast-math -ffp-contract=off)
            if(chemistry_test STREQUAL "primordial_solver")
                set(chemistry_processors 2)
            else()
                set(chemistry_processors 1)
            endif()
            cosmictiger_add_gtest(NAME ${chemistry_test} TARGET ${chemistry_test}_test
                PROPERTIES TIMEOUT 180 PROCESSORS ${chemistry_processors})
        endforeach()
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, scipy"
            RESULT_VARIABLE PRIMORDIAL_SCIPY_STATUS OUTPUT_QUIET ERROR_QUIET)
        if(PRIMORDIAL_SCIPY_STATUS EQUAL 0)
            add_library(primordial_reference SHARED tests/primordial_reference.cpp)
            target_include_directories(primordial_reference PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(primordial_reference PRIVATE Boost::boost)
            target_compile_options(primordial_reference PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME primordial_radau_reference COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/primordial_reference.py $<TARGET_FILE:primordial_reference>)
            set_tests_properties(primordial_radau_reference PROPERTIES TIMEOUT 240
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
        endif()
    endif()

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
    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm" AND
            COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
        add_test(NAME primordial_integration_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_primordial_integration.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/primordial_integration_cpu
            --recombination-reference $<TARGET_FILE:recombination_reference>)
        set_tests_properties(primordial_integration_cpu PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_library(mpi_thread_override SHARED tests/mpi_thread_override.cpp)
        target_link_libraries(mpi_thread_override PRIVATE MPI::MPI_CXX)
        add_test(NAME mpi_thread_support COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_mpi_startup.py $<TARGET_FILE:cosmictiger>
            $<TARGET_FILE:mpi_thread_override> ${PROJECT_BINARY_DIR}/tests/mpi_thread_support
            --position-type ${COSMICTIGER_POSITION_TYPE})
        set_tests_properties(mpi_thread_support PROPERTIES TIMEOUT 240 PROCESSORS 2
            LABELS "mpi;startup")
        if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
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
        if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
            if(COSMICTIGER_WITH_SPH AND COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                add_test(NAME cpu_only_runtime COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_cpu_only.py $<TARGET_FILE:cosmictiger>
                    $<TARGET_FILE:no_cuda_runtime> ${PROJECT_BINARY_DIR}/tests/cpu_only)
                set_tests_properties(cpu_only_runtime PROPERTIES TIMEOUT 250 RUN_SERIAL TRUE)
            endif()
            add_test(NAME analysis_cpu_runtime COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_analysis_cpu.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/analysis_cpu_runtime
                --guard $<TARGET_FILE:no_cuda_runtime>)
            set_tests_properties(analysis_cpu_runtime PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE)
            if(MPIEXEC_EXECUTABLE)
                add_test(NAME analysis_cpu_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_analysis_cpu.py $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/analysis_cpu_mpi
                    --guard $<TARGET_FILE:no_cuda_runtime> --mpi-exec ${MPIEXEC_EXECUTABLE} --ranks 2)
                set_tests_properties(analysis_cpu_mpi PROPERTIES TIMEOUT 240 RUN_SERIAL TRUE PROCESSORS 4)
            endif()
        endif()
    endif()
    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_WITH_CUDA)
        target_compile_definitions(primordial_solver_test PRIVATE USE_CUDA)
        add_executable(primordial_cuda_test tests/primordial_cuda.cu)
        target_link_libraries(primordial_cuda_test PRIVATE primordial_chemistry CUDA::cudart)
        target_compile_options(primordial_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false>)
        cosmictiger_add_gtest(NAME primordial_cuda TARGET primordial_cuda_test
            PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    foreach(recombination_variant IN ITEMS standard fast_math)
        set(thermal_target recombination_thermal_${recombination_variant}_test)
        add_executable(${thermal_target} tests/recombination_thermal.cpp)
        target_include_directories(${thermal_target} PRIVATE ${PROJECT_SOURCE_DIR})
        set(background_target recombination_background_${recombination_variant}_test)
        add_executable(${background_target} tests/recombination_background.cpp
            src/cosmology/zero_order.cpp)
        target_include_directories(${background_target} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${background_target} PRIVATE primordial_chemistry)
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
    target_link_libraries(recombination_reference PRIVATE primordial_chemistry)
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
        target_link_libraries(${native_target} PRIVATE primordial_chemistry)
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
    foreach(boltzmann_target IN ITEMS boltzmann_test boltzmann_reference)
        if(boltzmann_target STREQUAL "boltzmann_test")
            set(boltzmann_test_source tests/boltzmann.cpp)
        else()
            set(boltzmann_test_source tests/boltzmann_reference.cpp)
        endif()
        add_executable(${boltzmann_target} ${boltzmann_test_source}
            src/cosmology/boltzmann.cpp src/cosmology/zero_order.cpp)
        target_include_directories(${boltzmann_target} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${boltzmann_target} PRIVATE primordial_chemistry)
    endforeach()
    cosmictiger_add_gtest(NAME boltzmann TARGET boltzmann_test
        PROPERTIES TIMEOUT 120)
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
        foreach(boltzmann_two_lpt IN ITEMS false true)
            set(boltzmann_smoke boltzmann_initialization_${boltzmann_two_lpt}_cpu)
            add_test(NAME ${boltzmann_smoke} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/${boltzmann_smoke}
                --extra-option=--initial.twoLpt=${boltzmann_two_lpt})
            set_tests_properties(${boltzmann_smoke}
                PROPERTIES RUN_SERIAL TRUE TIMEOUT 240 PROCESSORS 2)
        endforeach()
    endif()

    foreach(fft_mode IN ITEMS fft1 fft2)
        set(fft_workdir "${PROJECT_BINARY_DIR}/tests/${fft_mode}_cpu")
        file(MAKE_DIRECTORY "${fft_workdir}")
        add_test(NAME ${fft_mode}_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=${fft_mode} ${cosmictiger_test_gravity_args} --initial.particlesPerDimension=8
            --hydro.enabled=false --compute.cuda=false --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(${fft_mode}_cpu PROPERTIES TIMEOUT 120 PROCESSORS 2
            WORKING_DIRECTORY "${fft_workdir}" LABELS "fft;cpu"
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1")
        if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
            set(fft_mpi_workdir "${PROJECT_BINARY_DIR}/tests/${fft_mode}_mpi")
            file(MAKE_DIRECTORY "${fft_mpi_workdir}")
            add_test(NAME ${fft_mode}_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                --run.test=${fft_mode} ${cosmictiger_test_gravity_args} --initial.particlesPerDimension=8
                --hydro.enabled=false --compute.cuda=false
                --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_mpi_transport_args})
            set_tests_properties(${fft_mode}_mpi PROPERTIES TIMEOUT 180 PROCESSORS 4
                WORKING_DIRECTORY "${fft_mpi_workdir}" LABELS "fft;cpu;mpi"
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1")
        endif()
    endforeach()

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

    add_executable(rockstar_eigenpair_test tests/rockstar_eigenpair.cpp src/math/math.cpp)
    target_include_directories(rockstar_eigenpair_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(rockstar_eigenpair_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(rockstar_eigenpair_test)
    cosmictiger_add_gtest(NAME rockstar_eigenpair TARGET rockstar_eigenpair_test
        PROPERTIES TIMEOUT 30)
    foreach(halo_unit IN ITEMS rockstar_core rockstar_properties lightcone_epoch)
        add_executable(${halo_unit}_test tests/${halo_unit}.cpp)
        target_include_directories(${halo_unit}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(${halo_unit}_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
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
                --run.test=${halo_mode} ${cosmictiger_test_gravity_args} --compute.cuda=${halo_cuda} --hydro.enabled=false
                --initial.particlesPerDimension=8 --analysis.groups=true
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(${halo_mode}_${halo_backend} PROPERTIES
                WORKING_DIRECTORY "${halo_workdir}" TIMEOUT 180 PROCESSORS 2)
        endforeach()
    endforeach()
    if(MPIEXEC_EXECUTABLE)
        set(halo_mpi_workdir "${PROJECT_BINARY_DIR}/tests/groupsCatalogue_mpi")
        file(MAKE_DIRECTORY "${halo_mpi_workdir}")
        add_test(NAME groupsCatalogue_mpi COMMAND ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger>
            --run.test=groupsCatalogueDistributed ${cosmictiger_test_gravity_args} --compute.cuda=false --hydro.enabled=false
            --initial.particlesPerDimension=8 --analysis.groups=true
            --hpx:threads=2 --hpx:bind=none --hpx:localities=2 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(groupsCatalogue_mpi PROPERTIES
            WORKING_DIRECTORY "${halo_mpi_workdir}" TIMEOUT 240 PROCESSORS 4 RUN_SERIAL TRUE)
    endif()
    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm" AND
            COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
        add_test(NAME scenarios COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_scenarios.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/scenarios)
        set_tests_properties(scenarios PROPERTIES RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2)
    endif()

    set(cosmictiger_sph_test_args)
    if(NOT COSMICTIGER_WITH_SPH)
        list(APPEND cosmictiger_sph_test_args --without-sph)
    endif()
    add_test(NAME grouped_options COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_options.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/grouped_options ${cosmictiger_sph_test_args}
        --gravity-method ${COSMICTIGER_GRAVITY_METHOD}
        --near-method ${COSMICTIGER_PMMM_NEAR_METHOD} --order ${COSMICTIGER_WITH_ORDER}
        --time-integrator ${COSMICTIGER_TIME_INTEGRATOR})
    # Hundreds of expected rejections each produce a symbolized stack trace.
    set_tests_properties(grouped_options PROPERTIES RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2)

    set(gravity_error_name gravity_error_cpu)
    add_test(NAME ${gravity_error_name} COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_gravity_error.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/${gravity_error_name}
        --gravity-method ${COSMICTIGER_GRAVITY_METHOD}
        --position-type ${COSMICTIGER_POSITION_TYPE})
    set_tests_properties(${gravity_error_name} PROPERTIES TIMEOUT 300 PROCESSORS 2
        LABELS "gravity;accuracy")
    if(COSMICTIGER_WITH_CUDA)
        set(gravity_error_name gravity_error_cuda)
        add_test(NAME ${gravity_error_name} COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_gravity_error.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/${gravity_error_name} --cuda
            --gravity-method ${COSMICTIGER_GRAVITY_METHOD}
            --position-type ${COSMICTIGER_POSITION_TYPE})
        set_tests_properties(${gravity_error_name} PROPERTIES TIMEOUT 300 PROCESSORS 2
            RUN_SERIAL TRUE LABELS "gravity;accuracy;cuda")
    endif()
    if(MPIEXEC_EXECUTABLE)
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
            --position-type ${COSMICTIGER_POSITION_TYPE}
            --gravity-method ${COSMICTIGER_GRAVITY_METHOD})
        set_tests_properties(gravity_error_mpi PROPERTIES TIMEOUT 300 PROCESSORS 4
            RUN_SERIAL TRUE LABELS "gravity;accuracy;mpi")
    endif()

    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm" AND
            COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
        add_test(NAME linear_growth_response COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_linear_growth.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/linear_growth_response)
        set_tests_properties(linear_growth_response PROPERTIES RUN_SERIAL TRUE TIMEOUT 200)
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
    foreach(float_float_variant IN ITEMS standard fast_math)
        set(float_float_target float_float_${float_float_variant}_test)
        add_executable(${float_float_target} tests/float_float.cpp)
        target_include_directories(${float_float_target} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${float_float_target} PRIVATE Boost::boost)
        if(float_float_variant STREQUAL "fast_math")
            target_compile_options(${float_float_target} PRIVATE -ffast-math)
        endif()
        cosmictiger_add_gtest(NAME float_float_${float_float_variant} TARGET ${float_float_target}
            PROPERTIES TIMEOUT 30)
    endforeach()

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

    # Keep the long-double oracle in a separate translation unit so the
    # application's fast-math flags cannot change the reference calculations.
    add_library(special_functions_reference STATIC tests/special_functions_reference.cpp)
    target_compile_options(special_functions_reference PRIVATE -fno-fast-math -ffp-contract=off)
    foreach(special_variant IN ITEMS baseline native fast_math)
        set(special_target special_functions_${special_variant}_test)
        add_executable(${special_target} tests/special_functions.cpp)
        target_include_directories(${special_target} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${special_target} PRIVATE special_functions_reference)
        if(special_variant STREQUAL "baseline")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
                target_compile_options(${special_target} PRIVATE
                    -msse2 -mno-avx -mno-avx2 -mno-avx512f -mno-fma)
            endif()
        elseif(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(${special_target} PRIVATE -march=native)
        endif()
        if(special_variant STREQUAL "fast_math")
            target_compile_options(${special_target} PRIVATE -ffast-math)
        endif()
        cosmictiger_add_gtest(NAME special_functions_${special_variant} TARGET ${special_target}
            PROPERTIES TIMEOUT 60)
    endforeach()
    foreach(special_variant IN ITEMS standard fast_math)
        set(special_benchmark special_functions_${special_variant}_benchmark)
        add_executable(${special_benchmark} EXCLUDE_FROM_ALL tests/special_functions_benchmark.cpp)
        target_include_directories(${special_benchmark} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${special_benchmark} PRIVATE special_functions_reference)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(${special_benchmark} PRIVATE -march=native)
        endif()
        if(special_variant STREQUAL "fast_math")
            target_compile_options(${special_benchmark} PRIVATE -ffast-math)
        endif()
    endforeach()

    if(COSMICTIGER_WITH_SPH AND COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm" AND
            COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
        set(units_output_workdir "${PROJECT_BINARY_DIR}/tests/units_output")
        file(MAKE_DIRECTORY "${units_output_workdir}")
        add_test(NAME units_output COMMAND cosmictiger --run.test=unitsOutput
            --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
            --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(units_output PROPERTIES
            WORKING_DIRECTORY "${units_output_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
        # Keep the existing cases on the default model and cover the other state layouts.
        foreach(units_viscosity IN ITEMS none cullen_and_dehnen price_etal)
            if(units_viscosity STREQUAL "none")
                set(units_viscosity_option none)
            elseif(units_viscosity STREQUAL "price_etal")
                set(units_viscosity_option priceEtAl)
            else()
                set(units_viscosity_option cullenAndDehnen)
            endif()
            set(units_test "units_output_${units_viscosity}")
            set(units_workdir "${PROJECT_BINARY_DIR}/tests/${units_test}")
            file(MAKE_DIRECTORY "${units_workdir}")
            add_test(NAME ${units_test} COMMAND cosmictiger --run.test=unitsOutput
                --compute.positionType=${COSMICTIGER_POSITION_TYPE}
                --hydro.viscosityEvolution=${units_viscosity_option}
                --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(${units_test} PROPERTIES
                WORKING_DIRECTORY "${units_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)
        endforeach()

        set(distributed_output_workdir "${PROJECT_BINARY_DIR}/tests/distributed_output")
        file(MAKE_DIRECTORY "${distributed_output_workdir}")
        add_test(NAME distributed_output COMMAND cosmictiger --run.test=distributedOutput
            --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
            --hpx:threads=2 --hpx:bind=none)
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

    add_executable(cartesian_kernels_test tests/gravity/cartesian_kernels.cpp)
    target_include_directories(cartesian_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_use_generated_kernels(cartesian_kernels_test)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(cartesian_kernels_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME cartesian_kernels TARGET cartesian_kernels_test
        PROPERTIES TIMEOUT 60)

    add_executable(fmm_api_test tests/gravity/fmm_api.cpp)
    target_include_directories(fmm_api_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_use_generated_kernels(fmm_api_test)
    cosmictiger_add_gtest(NAME fmm_api TARGET fmm_api_test
        PROPERTIES TIMEOUT 30)

    # Standalone generator families: these tests do not select a new runtime walk.
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
        foreach(representation IN ITEMS spherical_ewald)
            set(generated_kernel_target "generated_${representation}_test")
            add_executable(${generated_kernel_target} "tests/gravity/nitadori/generated_${representation}.cpp")
            target_include_directories(${generated_kernel_target} PRIVATE ${PROJECT_SOURCE_DIR})
            cosmictiger_use_generated_kernels(${generated_kernel_target})
            target_compile_options(${generated_kernel_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "generated_${representation}" COMMAND ${generated_kernel_target})
            set_tests_properties("generated_${representation}" PROPERTIES
                TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator;cpu")
        endforeach()
        # Splitting the spherical templates by order bounds compiler memory use.
        foreach(spherical_order IN LISTS COSMICTIGER_FMM_ORDERS)
            math(EXPR spherical_degree "${spherical_order} - 1")
            set(spherical_target "generated_spherical_order_${spherical_order}_test")
            add_executable(${spherical_target} tests/gravity/nitadori/generated_spherical.cpp)
            target_include_directories(${spherical_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(${spherical_target} PRIVATE
                GENERATED_SPHERICAL_TEST_DEGREE=${spherical_degree})
            cosmictiger_use_generated_kernels(${spherical_target})
            target_compile_options(${spherical_target} PRIVATE -fno-fast-math -ffp-contract=off)
            add_test(NAME "generated_spherical_order_${spherical_order}" COMMAND ${spherical_target})
            set_tests_properties("generated_spherical_order_${spherical_order}" PROPERTIES
                TIMEOUT 180 PROCESSORS 1 LABELS "fmm;generator;cpu")
        endforeach()
    endif()

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
    # Exercise the CUDA-disabled Ewald initializer even in a CUDA-enabled build.
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
        add_executable(cartesian_ewald_cpu_test tests/gravity/dehnen/cartesian_ewald_cpu.cpp src/gravity/dehnen/ewald_indices.cpp src/gravity/fmm_order.cpp)
        target_include_directories(cartesian_ewald_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_include_directories(cartesian_ewald_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
        target_compile_definitions(cartesian_ewald_cpu_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(cartesian_ewald_cpu_test)
        target_link_libraries(cartesian_ewald_cpu_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME cartesian_ewald_cpu TARGET cartesian_ewald_cpu_test
            PROPERTIES TIMEOUT 30)

        add_executable(fmm_flops_test tests/gravity/dehnen/fmm_flops.cpp src/gravity/dehnen/ewald_indices.cpp src/gravity/fmm_order.cpp)
        target_include_directories(fmm_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_include_directories(fmm_flops_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
        target_compile_definitions(fmm_flops_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(fmm_flops_test)
        target_link_libraries(fmm_flops_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME fmm_flops TARGET fmm_flops_test
            PROPERTIES TIMEOUT 30)
    endif()

    add_executable(softening_flops_test tests/softening_flops.cpp)
    target_include_directories(softening_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(softening_flops_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_compile_definitions(softening_flops_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(softening_flops_test)
    target_link_libraries(softening_flops_test PRIVATE HPX::hpx Boost::boost)
    cosmictiger_add_gtest(NAME softening_flops TARGET softening_flops_test
        PROPERTIES TIMEOUT 30)

    if(COSMICTIGER_WITH_SPH)
        add_executable(sph_flops_test tests/sph/flops.cpp)
        target_include_directories(sph_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(sph_flops_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(sph_flops_test)
        target_link_libraries(sph_flops_test PRIVATE HPX::hpx Boost::boost)
        cosmictiger_add_gtest(NAME sph_flops TARGET sph_flops_test
            PROPERTIES TIMEOUT 30)
    endif()

    add_executable(fmm_acceptance_test tests/gravity/fmm_acceptance.cpp)
    target_include_directories(fmm_acceptance_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(fmm_acceptance_test PRIVATE -ffast-math)
    target_compile_definitions(fmm_acceptance_test PRIVATE FMM_ACCEPTANCE_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(fmm_acceptance_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME fmm_acceptance TARGET fmm_acceptance_test
        PROPERTIES TIMEOUT 30)

    add_executable(mesh_domain_test tests/mesh_domain.cpp)
    target_include_directories(mesh_domain_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(mesh_domain_test PRIVATE HPX::hpx Boost::boost)
    target_compile_options(mesh_domain_test PRIVATE -fno-fast-math -ffp-contract=off)
    cosmictiger_add_gtest(NAME mesh_domain TARGET mesh_domain_test
        PROPERTIES TIMEOUT 30 LABELS "particle_mesh;cpu")
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
        add_executable(pmmm_kernels_test tests/gravity/nitadori/pmmm_kernels.cpp
            tests/gravity/nitadori/pmmm_kernel_flops.cpp src/gravity/nitadori/kernels.cpp)
        target_include_directories(pmmm_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(pmmm_kernels_test)
        target_link_libraries(pmmm_kernels_test PRIVATE HPX::hpx Boost::boost)
        target_compile_options(pmmm_kernels_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME pmmm_kernels TARGET pmmm_kernels_test
            PROPERTIES TIMEOUT 180 LABELS "pmmm;cpu")
        if(COSMICTIGER_PMMM_NEAR_METHOD STREQUAL "pp")
            add_executable(pmmm_near_flops_test tests/gravity/nitadori/near_pp_flops.cpp)
            target_include_directories(pmmm_near_flops_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(pmmm_near_flops_test PRIVATE HPX::hpx Boost::boost)
            target_compile_options(pmmm_near_flops_test PRIVATE -fno-fast-math -ffp-contract=off)
            cosmictiger_add_gtest(NAME pmmm_near_flops TARGET pmmm_near_flops_test
                PROPERTIES TIMEOUT 60 LABELS "pmmm;cpu;flops")
        endif()
        add_executable(pmmm_fft_flops_test tests/gravity/nitadori/fft_flops.cpp)
        target_include_directories(pmmm_fft_flops_test PRIVATE ${PROJECT_SOURCE_DIR} ${FFTW3F_INCLUDE_DIR})
        target_link_libraries(pmmm_fft_flops_test PRIVATE HPX::hpx Boost::boost ${FFTW3F_LIBRARY})
        target_compile_options(pmmm_fft_flops_test PRIVATE -fno-fast-math -ffp-contract=off)
        cosmictiger_add_gtest(NAME pmmm_fft_flops TARGET pmmm_fft_flops_test
            PROPERTIES TIMEOUT 60 LABELS "pmmm;cpu;flops")
    endif()

    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
        set(pmmm_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_cpu")
        file(MAKE_DIRECTORY "${pmmm_workdir}")
        add_test(NAME pmmm_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=pmmm --initial.particlesPerDimension=8
            --gravity.meshSize=8 --hydro.enabled=false --compute.cuda=false
            --compute.positionType=${COSMICTIGER_POSITION_TYPE} --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(pmmm_cpu PROPERTIES TIMEOUT 300 PROCESSORS 2
            WORKING_DIRECTORY "${pmmm_workdir}" LABELS "pmmm;cpu")
        set(mesh_domain_workdir "${PROJECT_BINARY_DIR}/tests/mesh_domains_cpu")
        file(MAKE_DIRECTORY "${mesh_domain_workdir}")
        add_test(NAME mesh_domains_cpu COMMAND $<TARGET_FILE:cosmictiger>
            --run.test=meshDomains --initial.particlesPerDimension=8
            --gravity.meshSize=7 --hydro.enabled=false
            --compute.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
            --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(mesh_domains_cpu PROPERTIES TIMEOUT 180 PROCESSORS 2
            WORKING_DIRECTORY "${mesh_domain_workdir}" LABELS "particle_mesh;cpu")
        set(pmmm_sph_args)
        if(COSMICTIGER_WITH_SPH)
            list(APPEND pmmm_sph_args --with-sph)
        endif()
        add_test(NAME pmmm_evolution_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/gravity/nitadori/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_cpu
            --near-method ${COSMICTIGER_PMMM_NEAR_METHOD} --order ${COSMICTIGER_WITH_ORDER}
            --time-integrator ${COSMICTIGER_TIME_INTEGRATOR}
            --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
        set_tests_properties(pmmm_evolution_cpu PROPERTIES TIMEOUT 600 PROCESSORS 2
            RUN_SERIAL TRUE LABELS "pmmm;cpu")
        if(COSMICTIGER_WITH_CUDA)
            add_test(NAME pmmm_evolution_cuda COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/nitadori/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_cuda --cuda
                --near-method ${COSMICTIGER_PMMM_NEAR_METHOD} --order ${COSMICTIGER_WITH_ORDER}
                --time-integrator ${COSMICTIGER_TIME_INTEGRATOR}
                --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
            set_tests_properties(pmmm_evolution_cuda PROPERTIES TIMEOUT 600 PROCESSORS 2
                RUN_SERIAL TRUE LABELS "pmmm;cuda")
            file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/tests/pmmm_cuda")
            add_test(NAME pmmm_cuda COMMAND $<TARGET_FILE:cosmictiger>
                --run.test=pmmm --initial.particlesPerDimension=8
                --gravity.meshSize=8 --hydro.enabled=false --compute.cuda=true
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(pmmm_cuda PROPERTIES TIMEOUT 300 PROCESSORS 2
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/tests/pmmm_cuda" LABELS "pmmm;cuda")
        endif()
        foreach(pm_check IN ITEMS particleMeshSort particleMeshTimestep)
            set(pm_test_name "${pm_check}")
            set(pm_test_workdir "${PROJECT_BINARY_DIR}/tests/${pm_test_name}")
            file(MAKE_DIRECTORY "${pm_test_workdir}")
            add_test(NAME ${pm_test_name} COMMAND $<TARGET_FILE:cosmictiger>
                --run.test=${pm_check} --initial.particlesPerDimension=8
                --gravity.meshSize=8 --hydro.enabled=false
                --compute.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(${pm_test_name} PROPERTIES TIMEOUT 180 PROCESSORS 2
                WORKING_DIRECTORY "${pm_test_workdir}" LABELS "particle_mesh;cpu")
        endforeach()
        if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
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
                --run.test=pmmm --initial.particlesPerDimension=8
                --gravity.meshSize=8 --hydro.enabled=false --compute.cuda=false
                --hpx:threads=2 --hpx:bind=none --hpx:localities=2
                ${cosmictiger_mpi_transport_args})
            # FFTW and the near-field tasks share each rank's HPX workers.
            set_tests_properties(pmmm_mpi_threads PROPERTIES TIMEOUT 300 PROCESSORS 4
                WORKING_DIRECTORY "${pmmm_threaded_workdir}" LABELS "pmmm;cpu;mpi;threads")
            foreach(pmmm_ranks IN ITEMS 3 6)
                set(pmmm_workdir "${PROJECT_BINARY_DIR}/tests/pmmm_mpi_${pmmm_ranks}")
                file(MAKE_DIRECTORY "${pmmm_workdir}")
                add_test(NAME pmmm_mpi_${pmmm_ranks} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${pmmm_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=pmmm --initial.particlesPerDimension=8
                    --gravity.meshSize=8 --hydro.enabled=false --compute.cuda=false
                    --hpx:threads=1 --hpx:bind=none --hpx:localities=${pmmm_ranks}
                    ${cosmictiger_mpi_transport_args})
                set_tests_properties(pmmm_mpi_${pmmm_ranks} PROPERTIES TIMEOUT 300
                    PROCESSORS ${pmmm_ranks} WORKING_DIRECTORY "${pmmm_workdir}" LABELS "pmmm;mpi")
            endforeach()
            foreach(mesh_domain_ranks IN ITEMS 2 3 6 9)
                set(mesh_domain_name "mesh_domains_mpi_${mesh_domain_ranks}")
                set(mesh_domain_workdir "${PROJECT_BINARY_DIR}/tests/${mesh_domain_name}")
                file(MAKE_DIRECTORY "${mesh_domain_workdir}")
                add_test(NAME ${mesh_domain_name} COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${mesh_domain_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=meshDomains --initial.particlesPerDimension=8
                    --gravity.meshSize=7 --hydro.enabled=false
                    --compute.cuda=false --compute.positionType=${COSMICTIGER_POSITION_TYPE}
                    --hpx:threads=1 --hpx:bind=none --hpx:localities=${mesh_domain_ranks}
                    ${cosmictiger_mpi_transport_args})
                set_tests_properties(${mesh_domain_name} PROPERTIES TIMEOUT 180
                    PROCESSORS ${mesh_domain_ranks} WORKING_DIRECTORY "${mesh_domain_workdir}"
                    LABELS "particle_mesh;mpi")
            endforeach()
            add_test(NAME pmmm_evolution_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/nitadori/run_pmmm_evolution.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/pmmm_evolution_mpi ${pm_launcher_args} --ranks 2
                --near-method ${COSMICTIGER_PMMM_NEAR_METHOD} --order ${COSMICTIGER_WITH_ORDER}
                --time-integrator ${COSMICTIGER_TIME_INTEGRATOR}
                --position-type ${COSMICTIGER_POSITION_TYPE} ${pmmm_sph_args})
            set_tests_properties(pmmm_evolution_mpi PROPERTIES TIMEOUT 600 PROCESSORS 4
                RUN_SERIAL TRUE LABELS "pmmm;mpi")
            add_test(NAME particle_mesh_dispatch_failures_mpi COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/gravity/nitadori/run_particle_mesh_failures.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/particle_mesh_dispatch_failures_mpi ${pm_launcher_args})
            set_tests_properties(particle_mesh_dispatch_failures_mpi PROPERTIES TIMEOUT 100 PROCESSORS 4
                RUN_SERIAL TRUE LABELS "particle_mesh;mpi")
            # This fixture uses arbitrary per-rank ownership and a global reduction.
            # The FMM near solver requires spatial ownership, covered by streamed tests.
            if(COSMICTIGER_PMMM_NEAR_METHOD STREQUAL "pp")
                set(pm_timestep_workdir "${PROJECT_BINARY_DIR}/tests/particle_mesh_timestep_mpi")
                file(MAKE_DIRECTORY "${pm_timestep_workdir}")
                add_test(NAME particle_mesh_timestep_mpi COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger>
                    ${MPIEXEC_POSTFLAGS} --run.test=particleMeshTimestep --initial.particlesPerDimension=8
                    --gravity.meshSize=8 --hydro.enabled=false
                    --compute.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=3
                    ${cosmictiger_mpi_transport_args})
                set_tests_properties(particle_mesh_timestep_mpi PROPERTIES TIMEOUT 180 PROCESSORS 6
                    WORKING_DIRECTORY "${pm_timestep_workdir}" LABELS "particle_mesh;mpi")
            endif()
        endif()
    endif()
    if(COSMICTIGER_WITH_SPH)
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

        add_executable(cullen_and_dehnen_test tests/sph/cullen_and_dehnen/viscosity.cpp)
        target_include_directories(cullen_and_dehnen_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME cullen_and_dehnen TARGET cullen_and_dehnen_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(price_etal_test tests/sph/price_etal/viscosity.cpp src/sph/price_etal/viscosity.cpp)
        target_include_directories(price_etal_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME price_etal TARGET price_etal_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(balsara_test tests/sph/balsara/shear_limiter.cpp)
        target_include_directories(balsara_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME balsara TARGET balsara_test
            PROPERTIES TIMEOUT 30 LABELS "sph;cpu")

        add_executable(monaghan_sph_test tests/sph/monaghan/equations.cpp)
        target_include_directories(monaghan_sph_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_add_gtest(NAME monaghan_sph TARGET monaghan_sph_test
            PROPERTIES TIMEOUT 20)
    endif()

    foreach(analysis_kernel IN ITEMS power lightcone)
        add_executable(${analysis_kernel}_cpu_test
            tests/${analysis_kernel}_cpu.cpp src/analysis/${analysis_kernel}_cpu.cpp)
        target_include_directories(${analysis_kernel}_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${analysis_kernel}_cpu_test PRIVATE HPX::hpx)
        target_compile_options(${analysis_kernel}_cpu_test PRIVATE -ffast-math)
        cosmictiger_add_gtest(NAME ${analysis_kernel}_cpu TARGET ${analysis_kernel}_cpu_test
            PROPERTIES TIMEOUT 60)
    endforeach()

    if(COSMICTIGER_WITH_CUDA)
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

        add_executable(lightcone_cuda_test tests/lightcone_cuda.cu src/analysis/lightcone.cu
            src/analysis/lightcone_cpu.cpp src/core/assert.cu src/cuda/cuda_stream.cpp)
        target_include_directories(lightcone_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(lightcone_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(lightcone_cuda_test)
        target_link_libraries(lightcone_cuda_test PRIVATE HPX::hpx CUDA::cudart)
        target_compile_options(lightcone_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        set_target_properties(lightcone_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        cosmictiger_add_gtest(NAME lightcone_cuda TARGET lightcone_cuda_test
            PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

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
    target_include_directories(drift_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_link_libraries(drift_cpu_test PRIVATE HPX::hpx ${CHEALPIX_LIBRARY})
    target_compile_options(drift_cpu_test PRIVATE -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(drift_cpu_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME drift_cpu TARGET drift_cpu_test
        PROPERTIES TIMEOUT 30)

    add_hpx_executable(drift_dispatch_test
        SOURCES tests/drift_dispatch.cpp src/particles/drift.cpp src/particles/drift_cpu.cpp)
    target_include_directories(drift_dispatch_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(drift_dispatch_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_compile_definitions(drift_dispatch_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(drift_dispatch_test)
    target_link_libraries(drift_dispatch_test PRIVATE ${CHEALPIX_LIBRARY})
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
            SOURCES tests/scenarios/sphere_surface.cpp src/scenarios/sphere.cpp)
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
        add_executable(sph_cpu_test tests/sph/cpu.cpp src/sph/sph_cpu.cpp
            src/sph/sph_prehydro_cpu.cpp src/sph/sph_hydro_cpu.cpp src/sph/sph_rungs_cpu.cpp src/math/kernel.cpp)
        target_include_directories(sph_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(sph_cpu_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(sph_cpu_test)
        target_link_libraries(sph_cpu_test PRIVATE HPX::hpx)
        target_compile_options(sph_cpu_test PRIVATE -ffast-math)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(sph_cpu_test PRIVATE -march=native)
        endif()
        cosmictiger_add_gtest(NAME sph_cpu TARGET sph_cpu_test
            PROPERTIES TIMEOUT 120)

    endif()

    # Range and position arithmetic use the selected width and no GPU runtime.
    add_executable(numerical_utilities_test tests/numerical_utilities.cpp)
    target_include_directories(numerical_utilities_test PRIVATE ${PROJECT_SOURCE_DIR})
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(numerical_utilities_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME numerical_utilities TARGET numerical_utilities_test
        PROPERTIES TIMEOUT 10)

    add_executable(fmm_geometry_test tests/gravity/fmm_geometry.cpp)
    target_include_directories(fmm_geometry_test PRIVATE ${PROJECT_SOURCE_DIR})
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(fmm_geometry_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME fmm_geometry TARGET fmm_geometry_test
        PROPERTIES TIMEOUT 10)

    if(COSMICTIGER_WITH_SPH)
        add_executable(rayleigh_taylor_test tests/rayleigh_taylor.cpp src/scenarios/rayleigh_taylor.cpp)
        target_include_directories(rayleigh_taylor_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(rayleigh_taylor_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(rayleigh_taylor_test)
        target_link_libraries(rayleigh_taylor_test PRIVATE HPX::hpx)
        cosmictiger_add_gtest(NAME rayleigh_taylor TARGET rayleigh_taylor_test
            PROPERTIES TIMEOUT 120)
    endif()

    add_executable(particle_selection_test tests/particle_selection.cpp)
    target_include_directories(particle_selection_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME particle_selection TARGET particle_selection_test
        PROPERTIES TIMEOUT 10)

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

    if(COSMICTIGER_WITH_CUDA)
        foreach(special_variant IN ITEMS standard fast_math)
            set(special_target special_functions_cuda_${special_variant}_test)
            add_executable(${special_target} tests/special_functions_cuda.cu)
            target_include_directories(${special_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${special_target} PRIVATE CUDA::cudart special_functions_reference)
            set(special_benchmark special_functions_cuda_${special_variant}_benchmark)
            add_executable(${special_benchmark} EXCLUDE_FROM_ALL
                tests/special_functions_benchmark_cuda.cu)
            target_include_directories(${special_benchmark} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${special_benchmark} PRIVATE CUDA::cudart special_functions_reference)
            if(special_variant STREQUAL "fast_math")
                foreach(special_program IN ITEMS ${special_target} ${special_benchmark})
                    target_compile_options(${special_program} PRIVATE
                        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-ffast-math>)
                endforeach()
            endif()
            cosmictiger_add_gtest(NAME special_functions_cuda_${special_variant} TARGET ${special_target}
                PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
        endforeach()

        foreach(float_float_variant IN ITEMS standard fast_math)
            set(float_float_target float_float_cuda_${float_float_variant}_test)
            add_executable(${float_float_target} tests/float_float_cuda.cu)
            target_include_directories(${float_float_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${float_float_target} PRIVATE CUDA::cudart Boost::boost)
            if(float_float_variant STREQUAL "fast_math")
                target_compile_options(${float_float_target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-ffast-math>)
            endif()
            cosmictiger_add_gtest(NAME float_float_cuda_${float_float_variant} TARGET ${float_float_target}
                PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
        endforeach()

        add_executable(cartesian_kernels_cuda_test tests/gravity/cartesian_kernels_cuda.cu)
        target_include_directories(cartesian_kernels_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(cartesian_kernels_cuda_test)
        target_link_libraries(cartesian_kernels_cuda_test PRIVATE CUDA::cudart)
        cosmictiger_add_gtest(NAME cartesian_kernels_cuda TARGET cartesian_kernels_cuda_test
            PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(cartesian_l2l_cuda_test tests/gravity/cartesian_l2l_cuda.cu)
        target_include_directories(cartesian_l2l_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(cartesian_l2l_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
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
                src/sph/sph_prehydro.cu src/sph/sph_hydro.cu src/sph/sph_rungs.cu
                src/sph/sph_cpu.cpp src/sph/sph_prehydro_cpu.cpp src/sph/sph_hydro_cpu.cpp src/sph/sph_rungs_cpu.cpp
                src/math/kernel.cpp src/math/kernel.cu src/cuda/cuda_memory.cu src/core/assert.cu)
            target_include_directories(monaghan_sph_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(monaghan_sph_cuda_test PRIVATE
                USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
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

    foreach(test_name IN ITEMS sph_gravity_cpu sph_gravity_cuda
        checkpoint_state_dm checkpoint_state_sph)
        if((NOT COSMICTIGER_WITH_SPH OR COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm" OR
                COSMICTIGER_TIME_INTEGRATOR STREQUAL "hierarchical") AND
                NOT test_name STREQUAL "checkpoint_state_dm")
            continue()
        endif()
        set(test_workdir "${PROJECT_BINARY_DIR}/tests/${test_name}")
        file(MAKE_DIRECTORY "${test_workdir}")
        if(test_name MATCHES "^sph_gravity")
            set(test_mode sphGravity)
            set(test_sph true)
        else()
            set(test_mode checkpointState)
            if(test_name STREQUAL "checkpoint_state_sph")
                set(test_sph true)
            else()
                set(test_sph false)
            endif()
        endif()
        if(test_name STREQUAL "sph_gravity_cuda" OR test_name STREQUAL "checkpoint_state_sph")
            set(test_cuda true)
        else()
            set(test_cuda false)
        endif()
        set(test_position_type ${COSMICTIGER_POSITION_TYPE})
        add_test(NAME ${test_name} COMMAND cosmictiger
            --run.test=${test_mode} ${cosmictiger_test_gravity_args} --initial.particlesPerDimension=8 --hydro.enabled=${test_sph}
            --compute.cuda=${test_cuda} --compute.positionType=${test_position_type}
            --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${test_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)
    endforeach()
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
        if(COSMICTIGER_WITH_SPH AND COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
            set(sph_tree_workdir "${PROJECT_BINARY_DIR}/tests/sph_tree")
            file(MAKE_DIRECTORY "${sph_tree_workdir}")
            add_test(NAME sph_tree COMMAND cosmictiger --run.test=sphTree
                --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(sph_tree PROPERTIES
                WORKING_DIRECTORY "${sph_tree_workdir}" RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 6)
        endif()

        set(fmm_tree_workdir "${PROJECT_BINARY_DIR}/tests/fmm_tree")
        file(MAKE_DIRECTORY "${fmm_tree_workdir}")
        add_test(NAME fmm_tree COMMAND cosmictiger --run.test=fmmTree
            --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=false
            --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(fmm_tree PROPERTIES
            WORKING_DIRECTORY "${fmm_tree_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
        foreach(fmm_backend IN ITEMS cpu cuda)
            if(fmm_backend STREQUAL "cuda")
                if(NOT COSMICTIGER_WITH_CUDA)
                    continue()
                endif()
                set(fmm_cuda true)
            else()
                set(fmm_cuda false)
            endif()
            set(fmm_walk_workdir "${PROJECT_BINARY_DIR}/tests/fmm_walk_${fmm_backend}")
            file(MAKE_DIRECTORY "${fmm_walk_workdir}")
            add_test(NAME fmm_walk_${fmm_backend} COMMAND cosmictiger --run.test=fmmWalk
                --initial.particlesPerDimension=8 --hydro.enabled=false --compute.cuda=${fmm_cuda}
                --hpx:threads=2 --hpx:bind=none)
            set_tests_properties(fmm_walk_${fmm_backend} PROPERTIES
                WORKING_DIRECTORY "${fmm_walk_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
        endforeach()
        foreach(kernel_backend IN ITEMS cpu cuda)
            if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                continue()
            endif()
            if(kernel_backend STREQUAL "cuda" AND NOT COSMICTIGER_WITH_CUDA)
                continue()
            endif()
            set(kernel_backend_flag)
            if(kernel_backend STREQUAL "cuda")
                set(kernel_backend_flag --cuda)
            endif()
            add_test(NAME kernel_selection_${kernel_backend} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_kernel_selection.py $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/kernel_selection_${kernel_backend} ${kernel_backend_flag})
            set_tests_properties(kernel_selection_${kernel_backend} PROPERTIES RUN_SERIAL TRUE TIMEOUT 900)
        endforeach()
        foreach(kernel_case IN ITEMS wendland_spline ferrers_wendland)
            if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_WITH_CUDA OR
                    NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
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
                --extra-option=--hydro.neighborNumber=120 ${kernel_options})
            set_tests_properties(smoke_kernels_${kernel_case} PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
        endforeach()
    endif()

    set(rejected_options_workdir "${PROJECT_BINARY_DIR}/tests/rejected_options")
    file(MAKE_DIRECTORY "${rejected_options_workdir}")
    add_test(NAME rejected_options COMMAND cosmictiger --invalid_option_for_test=true
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(rejected_options PROPERTIES WORKING_DIRECTORY "${rejected_options_workdir}"
        RUN_SERIAL TRUE TIMEOUT 20 WILL_FAIL TRUE)
    if(COSMICTIGER_GRAVITY_METHOD STREQUAL "fmm")
        if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
            set(fmm_tree_mpi_workdir "${PROJECT_BINARY_DIR}/tests/fmm_tree_mpi")
            if(COSMICTIGER_WITH_SPH AND COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                set(distributed_output_mpi_workdir "${PROJECT_BINARY_DIR}/tests/distributed_output_mpi")
                file(MAKE_DIRECTORY "${distributed_output_mpi_workdir}")
                add_test(NAME distributed_output_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=distributedOutput --initial.particlesPerDimension=8 --hydro.enabled=true
                    --compute.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=2)
                set_tests_properties(distributed_output_mpi PROPERTIES
                    WORKING_DIRECTORY "${distributed_output_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 180 PROCESSORS 4)
            endif()
            file(MAKE_DIRECTORY "${fmm_tree_mpi_workdir}")
            add_test(NAME fmm_tree_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                --run.test=fmmTree --initial.particlesPerDimension=8 --hydro.enabled=false
                --compute.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=3)
            set_tests_properties(fmm_tree_mpi PROPERTIES
                WORKING_DIRECTORY "${fmm_tree_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
            if(COSMICTIGER_WITH_CUDA)
                set(fmm_walk_mpi_workdir "${PROJECT_BINARY_DIR}/tests/fmm_walk_mpi")
                file(MAKE_DIRECTORY "${fmm_walk_mpi_workdir}")
                add_test(NAME fmm_walk_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false
                    --compute.cuda=true --hpx:threads=2 --hpx:bind=none --hpx:localities=3)
                set_tests_properties(fmm_walk_mpi PROPERTIES
                    WORKING_DIRECTORY "${fmm_walk_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
            endif()
            if(COSMICTIGER_WITH_SPH AND COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                set(units_mpi_workdir "${PROJECT_BINARY_DIR}/tests/units_output_mpi")
                file(MAKE_DIRECTORY "${units_mpi_workdir}")
                add_test(NAME units_output_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=unitsOutput --initial.particlesPerDimension=8 --hydro.enabled=true
                    --compute.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=2)
                set_tests_properties(units_output_mpi PROPERTIES
                    WORKING_DIRECTORY "${units_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
                set(migration_workdir "${PROJECT_BINARY_DIR}/tests/sph_migration_mpi")
                file(MAKE_DIRECTORY "${migration_workdir}")
                add_test(NAME sph_migration_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
                    --run.test=sphGravity --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
                    --hpx:threads=2 --hpx:bind=none --hpx:localities=2)
                set_tests_properties(sph_migration_mpi PROPERTIES WORKING_DIRECTORY "${migration_workdir}"
                    RUN_SERIAL TRUE TIMEOUT 120)
                add_test(NAME smoke_sph_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph_mpi
                    $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/smoke_sph_mpi --mpi-exec=${MPIEXEC_EXECUTABLE})
                set_tests_properties(smoke_sph_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
                add_test(NAME smoke_sph_cpu_mpi COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph_mpi
                    $<TARGET_FILE:cosmictiger>
                    ${PROJECT_BINARY_DIR}/tests/smoke_sph_cpu_mpi --mpi-exec=${MPIEXEC_EXECUTABLE} --cpu)
                set_tests_properties(smoke_sph_cpu_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
            endif()
        endif()
        foreach(smoke_mode IN ITEMS sod sph)
            if(NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
                continue()
            endif()
            add_test(NAME smoke_${smoke_mode}_cpu COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode}_cpu --cpu)
            set_tests_properties(smoke_${smoke_mode}_cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        endforeach()
        foreach(smoke_mode IN ITEMS sod dm_cpu dm_cuda sph)
            if((NOT COSMICTIGER_WITH_SPH OR NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog") AND
                    NOT smoke_mode MATCHES "^dm_")
                continue()
            endif()
            add_test(NAME smoke_${smoke_mode} COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode})
            set_tests_properties(smoke_${smoke_mode} PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        endforeach()
        add_test(NAME smoke_dm_min_rung COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu
            $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/smoke_dm_min_rung --minimum-rung=10)
        set_tests_properties(smoke_dm_min_rung PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
        if(COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
            add_test(NAME checkpoint_integrity COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/run_checkpoint.py
                $<TARGET_FILE:cosmictiger>
                ${PROJECT_BINARY_DIR}/tests/checkpoint_integrity ${cosmictiger_sph_test_args})
            set_tests_properties(checkpoint_integrity PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
        endif()
    endif()

    set(position_selection_args)
    if(NOT COSMICTIGER_WITH_SPH)
        list(APPEND position_selection_args --without-sph)
    endif()
    add_test(NAME position_selection COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_position_selection.py
        $<TARGET_FILE:cosmictiger> ${PROJECT_BINARY_DIR}/tests/position_selection
        ${position_selection_args} --position-type ${COSMICTIGER_POSITION_TYPE}
        --gravity-method ${COSMICTIGER_GRAVITY_METHOD}
        --time-integrator ${COSMICTIGER_TIME_INTEGRATOR}
        --near-method ${COSMICTIGER_PMMM_NEAR_METHOD} --order ${COSMICTIGER_WITH_ORDER})
    set_tests_properties(position_selection PROPERTIES RUN_SERIAL TRUE TIMEOUT 600 PROCESSORS 2)
endif()

if(BUILD_TESTING)
    get_property(cosmictiger_targets_after_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    list(REMOVE_ITEM cosmictiger_targets_after_tests ${cosmictiger_targets_before_tests})
    # These standalone tests deliberately compile both position widths, rather
    # than inheriting the configured application's position definition.
    list(REMOVE_ITEM cosmictiger_targets_after_tests
        gravity_error_cuda_reference_32_test gravity_error_cuda_reference_64_test)
    foreach(test_target IN LISTS cosmictiger_targets_after_tests)
        get_target_property(test_type ${test_target} TYPE)
        if(NOT test_type STREQUAL "UTILITY")
            target_link_libraries(${test_target} PRIVATE cosmictiger_default_config)
        endif()
    endforeach()
endif()
