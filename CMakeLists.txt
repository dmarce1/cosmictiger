# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

cmake_minimum_required(VERSION 3.24)
project(cosmictiger LANGUAGES CXX C)

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
option(COSMICTIGER_WITH_LONG_LONG_PART_INT "Use long long to index local particles" OFF)
# FMM order: multipole degree p plus two. Source multipoles carry degree p,
# local expansions degree p + 1, and Ewald derivatives ranks through p + 1.
# The supported FMM orders span 3 to 9. Builds compile a configurable subset
# and the run selects one of those orders with gravity.fmmOrder.
set(COSMICTIGER_FMM_MIN_ORDER "4" CACHE STRING "Smallest FMM order to compile (3 to 9)")
set(COSMICTIGER_FMM_MAX_ORDER "8" CACHE STRING "Largest FMM order to compile (3 to 9)")
if(NOT COSMICTIGER_FMM_MIN_ORDER MATCHES "^[3-9]$" OR
   NOT COSMICTIGER_FMM_MAX_ORDER MATCHES "^[3-9]$")
    message(FATAL_ERROR "COSMICTIGER_FMM_MIN_ORDER and COSMICTIGER_FMM_MAX_ORDER must be from 3 to 9")
endif()
if(COSMICTIGER_FMM_MIN_ORDER GREATER COSMICTIGER_FMM_MAX_ORDER)
    message(FATAL_ERROR "COSMICTIGER_FMM_MIN_ORDER must not exceed COSMICTIGER_FMM_MAX_ORDER")
endif()
set(COSMICTIGER_WITH_ORDER "7" CACHE STRING "Default FMM order (gravity.fmmOrder), 3 to 9")
if(NOT COSMICTIGER_WITH_ORDER MATCHES "^[3-9]$")
    message(FATAL_ERROR
        "COSMICTIGER_WITH_ORDER must be an integer from 3 to 9; got '${COSMICTIGER_WITH_ORDER}'")
endif()
if(COSMICTIGER_WITH_ORDER LESS COSMICTIGER_FMM_MIN_ORDER OR
   COSMICTIGER_WITH_ORDER GREATER COSMICTIGER_FMM_MAX_ORDER)
    message(FATAL_ERROR "COSMICTIGER_WITH_ORDER must be inside the configured FMM order range")
endif()
foreach(fmm_order RANGE ${COSMICTIGER_FMM_MIN_ORDER} ${COSMICTIGER_FMM_MAX_ORDER})
    list(APPEND COSMICTIGER_FMM_ORDERS ${fmm_order})
endforeach()
add_compile_definitions(FMM_MIN_ORDER=${COSMICTIGER_FMM_MIN_ORDER}
    FMM_MAX_ORDER=${COSMICTIGER_FMM_MAX_ORDER})
# Compile-time cap on the tree leaf bucket. compute.bucketSize may be any
# runtime value up to this power of two; leaves then hold between half the
# runtime bucket plus one and the runtime bucket particles. The CUDA kick
# shared-memory policy is tuned for a runtime bucket equal to this cap.
set(COSMICTIGER_WITH_MAX_BUCKET "128" CACHE STRING "Maximum tree leaf bucket size (power of two)")
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
if(COSMICTIGER_WITH_CUDA AND NOT HPX_WITH_CUDA)
    message(FATAL_ERROR
        "CUDA builds require HPX built with HPX_WITH_CUDA=ON; otherwise configure COSMICTIGER_WITH_CUDA=OFF")
endif()
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(GSL REQUIRED)
find_path(FFTW3F_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED)
find_path(SILO_INCLUDE_DIR silo.h REQUIRED)
find_library(SILO_LIBRARY NAMES siloh5 silo REQUIRED)
find_path(CHEALPIX_INCLUDE_DIR chealpix.h REQUIRED)
find_library(CHEALPIX_LIBRARY NAMES chealpix REQUIRED)
find_path(HEALPIX_CXX_INCLUDE_DIR healpix_base.h PATH_SUFFIXES healpix_cxx REQUIRED)
find_library(HEALPIX_CXX_LIBRARY NAMES healpix_cxx REQUIRED)

include(cmake/voronoi.cmake)
target_sources(cosmictiger_voronoi PRIVATE src/geometry/distributed_voronoi.cpp)
add_library(cosmictiger_silo_voronoi STATIC src/io/silo_voronoi.cpp)
target_include_directories(cosmictiger_silo_voronoi SYSTEM PUBLIC ${SILO_INCLUDE_DIR})
target_link_libraries(cosmictiger_silo_voronoi PUBLIC cosmictiger_voronoi ${SILO_LIBRARY})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cosmictiger_silo_voronoi PRIVATE -fno-fast-math -ffp-contract=off)
elseif(MSVC)
    target_compile_options(cosmictiger_silo_voronoi PRIVATE /fp:strict)
endif()

message(STATUS "cosmictiger FMM orders = ${COSMICTIGER_FMM_MIN_ORDER} to ${COSMICTIGER_FMM_MAX_ORDER} (default ${COSMICTIGER_WITH_ORDER})")
message(STATUS "cosmictiger maximum bucket size = ${COSMICTIGER_WITH_MAX_BUCKET}")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

set(cosmictiger_source_files
    src/analysis/compress.cpp
    src/analysis/group_entry.cpp
    src/analysis/group_tree.cpp
    src/analysis/groups.cpp
    src/analysis/groups_find.cpp
    src/analysis/healpix.cu
    src/analysis/lightcone.cpp
    src/analysis/lightcone.cu
    src/analysis/lightcone_cpu.cpp
    src/analysis/power.cpp
    src/analysis/power.cu
    src/analysis/power_cpu.cpp
    src/analysis/power_cuda.cu
    src/analysis/rockstar.cpp
    src/analysis/rockstar.cu
    src/analysis/rockstar_catalogue.cpp
    src/core/assert.cpp
    src/core/assert.cu
    src/core/driver.cpp
    src/core/scenario_driver.cpp
    src/core/flops.cpp
    src/core/flops.cu
    src/core/hpx.cpp
    src/core/memused.cpp
    src/core/options.cpp
    src/core/profiler.cpp
    src/core/stack_trace.cpp
    src/cosmology/boltzmann.cpp
    src/cosmology/cosmology.cpp
    src/cosmology/fft.cpp
    src/cosmology/initialize.cpp
    src/cosmology/zero_order.cpp
    src/cuda/cuda.cpp
    src/cuda/cuda_memory.cu
    src/cuda/cuda_stream.cpp
    src/fmm/ewald_indices.cpp
    src/fmm/ewald_indices.cu
    src/fmm/fmm_order.cpp
    src/gravity/analytic.cpp
    src/gravity/analytic.cu
    src/gravity/bh.cpp
    src/gravity/bh.cu
    src/gravity/gravity.cpp
    src/gravity/kick.cpp
    src/gravity/kick.cu
    src/gravity/kick_workspace.cpp
    src/gravity/tree.cpp
    src/io/checkpoint.cpp
    src/io/output.cpp
    src/io/view.cpp
    src/main.cpp
    src/math/kernel.cpp
    src/math/kernel.cu
    src/math/math.cpp
    src/particles/domain.cpp
    src/particles/drift.cpp
    src/particles/drift.cu
    src/particles/drift_cpu.cpp
    src/particles/particles.cpp
    src/scenarios/scenario.cpp
    src/scenarios/plummer.cpp
    src/scenarios/sod.cpp
    src/scenarios/blast.cpp
    src/scenarios/rayleigh_taylor.cpp
    src/scenarios/kelvin_helmholtz.cpp
    src/scenarios/disc.cpp
    src/scenarios/polytropic_star.cpp
    src/scenarios/hydro_wave.cpp
    src/scenarios/cosmology.cpp
    src/scenarios/glass.cpp
    src/sph/exact_sod.cpp
    src/sph/sedov.cpp
    src/sph/sph.cpp
    src/sph/sph.cu
    src/sph/sph_chemistry.cpp
    src/sph/sph_cpu.cpp
    src/sph/sph_hydro.cu
    src/sph/sph_hydro_cpu.cpp
    src/sph/sph_particles.cpp
    src/sph/sph_prehydro.cu
    src/sph/sph_prehydro_cpu.cpp
    src/sph/sph_rungs.cu
    src/sph/sph_rungs_cpu.cpp
    src/sph/sph_tree.cpp
    src/sph/sphere.cpp
    src/sph/sphere.cu
    tests/analysis_backends.cpp
    tests/bh.cpp
    tests/fmm_tree.cpp
    tests/fmm_walk.cpp
    tests/groups_catalogue.cpp
    tests/kernel_selection_cuda.cu
    tests/linear_growth.cpp
    tests/rockstar.cpp
    tests/sph_gravity.cpp
    tests/test.cpp
    tests/units_output.cpp
    tests/voronoi_output.cpp
)

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
    cosmictiger/cosmology/cosmology.hpp
    cosmictiger/cosmology/fft.hpp
    cosmictiger/cosmology/initialize.hpp
    cosmictiger/cosmology/recombination_thermal.hpp
    cosmictiger/cosmology/native_recombination.hpp
    cosmictiger/cosmology/zero_order.hpp
    cosmictiger/cuda/cuda.hpp
    cosmictiger/cuda/cuda_memory.hpp
    cosmictiger/cuda/cuda_reduce.hpp
    cosmictiger/cuda/cuda_stream.hpp
    cosmictiger/cuda/cuda_unordered_map.hpp
    cosmictiger/cuda/device_vector.hpp
    cosmictiger/fmm/ewald_indices.hpp
    cosmictiger/fmm/ewald_tables.hpp
    cosmictiger/fmm/fmm_acceptance.hpp
    cosmictiger/fmm/fmm_kernels.hpp
    cosmictiger/fmm/fmm_order.hpp
    cosmictiger/fmm/fmm_types.hpp
    cosmictiger/gravity/analytic.hpp
    cosmictiger/gravity/bh.hpp
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
    cosmictiger/io/silo_spheres.hpp
    cosmictiger/io/view.hpp
    cosmictiger/math/complex.hpp
    cosmictiger/math/constants.hpp
    cosmictiger/math/fixed.hpp
    cosmictiger/math/float_float.hpp
    cosmictiger/math/fp16.hpp
    cosmictiger/math/interpolation.hpp
    cosmictiger/math/kernel.hpp
    cosmictiger/math/math.hpp
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
    cosmictiger/sph/exact_sod.hpp
    cosmictiger/sph/monaghan_sph.hpp
    cosmictiger/sph/sedov.hpp
    cosmictiger/sph/sph.hpp
    cosmictiger/sph/sph_chemistry.hpp
    cosmictiger/sph/sph_cpu.hpp
    cosmictiger/sph/sph_cuda.hpp
    cosmictiger/sph/sph_kernel.hpp
    cosmictiger/sph/sph_particles.hpp
    cosmictiger/sph/sph_tree.hpp
    cosmictiger/sph/sphere.hpp
    tests/test.hpp
)

if(NOT COSMICTIGER_WITH_CUDA)
    list(FILTER cosmictiger_source_files EXCLUDE REGEX "\\.cu$")
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
find_package(Threads REQUIRED)
add_library(primordial_chemistry STATIC src/chemistry/primordial_chemistry.cpp
    src/chemistry/recombination.cpp)
target_include_directories(primordial_chemistry PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(primordial_chemistry PUBLIC Threads::Threads Boost::boost)
target_compile_options(primordial_chemistry PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
if(COSMICTIGER_WITH_CUDA)
    target_sources(primordial_chemistry PRIVATE src/chemistry/primordial_chemistry.cu)
    target_compile_definitions(primordial_chemistry PRIVATE USE_CUDA)
    target_link_libraries(primordial_chemistry PUBLIC CUDA::cudart)
    target_compile_options(primordial_chemistry PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false>)
endif()
set_source_files_properties(src/sph/sph_chemistry.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
# The adaptive Boltzmann solve needs finite-value checks and accurate
# cancellation in the synchronous Einstein constraints.
set_source_files_properties(src/cosmology/boltzmann.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")

add_hpx_executable(cosmictiger
    DEPENDENCIES Boost::boost Boost::program_options GSL::gsl GSL::gslcblas
        ${FFTW3F_LIBRARY} ${SILO_LIBRARY} ${CHEALPIX_LIBRARY} ${HEALPIX_CXX_LIBRARY} dl quadmath
    SOURCES ${cosmictiger_source_files}
    HEADERS ${cosmictiger_header_files})
option(COSMICTIGER_BUILD_STF_BENCHMARKS "Build the opt-in STF comparison harness" OFF)
if(COSMICTIGER_BUILD_STF_BENCHMARKS)
    target_sources(cosmictiger PRIVATE benchmarks/stf_walk.cpp)
    set_source_files_properties(tests/test.cpp PROPERTIES COMPILE_DEFINITIONS COSMICTIGER_STF_BENCHMARKS)
endif()
target_link_libraries(cosmictiger PRIVATE primordial_chemistry cosmictiger_fp16 cosmictiger_silo_voronoi)
target_include_directories(cosmictiger PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(cosmictiger SYSTEM PRIVATE
    ${FFTW3F_INCLUDE_DIR} ${SILO_INCLUDE_DIR} ${CHEALPIX_INCLUDE_DIR} ${HEALPIX_CXX_INCLUDE_DIR})
target_compile_definitions(cosmictiger PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
target_compile_options(cosmictiger PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes;-ffast-math>)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
if(COSMICTIGER_WITH_LONG_LONG_PART_INT)
    target_compile_definitions(cosmictiger PRIVATE LONG_LONG_PART_INT)
endif()
if(COSMICTIGER_WITH_CUDA)
    target_link_libraries(cosmictiger PUBLIC CUDA::cudart)
    target_compile_definitions(cosmictiger PRIVATE USE_CUDA BOOST_STACKTRACE_USE_ADDR2LINE)
    target_compile_options(cosmictiger PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)
    set_target_properties(cosmictiger PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(src/particles/drift.cu PROPERTIES COMPILE_OPTIONS -maxrregcount=64)
    set_source_files_properties(src/sph/sph.cu src/sph/sph_prehydro.cu src/sph/sph_hydro.cu src/sph/sph_rungs.cu
        PROPERTIES COMPILE_OPTIONS -maxrregcount=128)
endif()

# Generated kernels for every FMM order, produced together in the build tree.
# The Python generator emits specialized STF kernel classes and one Ewald
# derivative overload per order, so a single build serves all orders and selects
# one at run time.
set(COSMICTIGER_GENERATED_INCLUDE_DIR ${PROJECT_BINARY_DIR}/generated)
set(COSMICTIGER_GENERATED_FMM_DIR
    ${COSMICTIGER_GENERATED_INCLUDE_DIR}/cosmictiger/fmm)
set(COSMICTIGER_GENERATED_FMM_FILES
    ${COSMICTIGER_GENERATED_FMM_DIR}/generated_config.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/generated_m2l.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/generated_operators.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/ewald_kernels.hpp
    ${COSMICTIGER_GENERATED_FMM_DIR}/operation_counts.json)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(COSMICTIGER_FMM_GENERATOR_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/generate_stf_m2l.py
    ${PROJECT_SOURCE_DIR}/tools/stf/stf_cuda.py
    ${PROJECT_SOURCE_DIR}/tools/stf/stf_ewald.py
    ${PROJECT_SOURCE_DIR}/tools/stf/stf_particles.py
    ${PROJECT_SOURCE_DIR}/tools/stf/stf_translations.py
    ${PROJECT_SOURCE_DIR}/tools/stf/generated_notice.txt
    ${PROJECT_SOURCE_DIR}/cosmictiger/core/defs.hpp)
add_custom_command(
    OUTPUT ${COSMICTIGER_GENERATED_FMM_FILES}
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_stf_m2l.py
        --output-dir ${COSMICTIGER_GENERATED_FMM_DIR} --no-format
    DEPENDS ${COSMICTIGER_FMM_GENERATOR_SOURCES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generating STF kernels and Ewald derivatives for FMM orders 3 to 9")
add_custom_target(generate_fmm_kernels DEPENDS ${COSMICTIGER_GENERATED_FMM_FILES})
add_custom_target(generate_stf_kernels DEPENDS ${COSMICTIGER_GENERATED_FMM_FILES})
add_custom_target(cosmictiger_generated_kernels
    DEPENDS ${COSMICTIGER_GENERATED_FMM_FILES})
# Compile a target against the generated kernels and the order/bucket defaults.
function(cosmictiger_use_generated_kernels target)
    target_include_directories(${target} BEFORE PRIVATE ${COSMICTIGER_GENERATED_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE ORDER=${COSMICTIGER_WITH_ORDER}
        MAX_BUCKET_SIZE=${COSMICTIGER_WITH_MAX_BUCKET})
    add_dependencies(${target} cosmictiger_generated_kernels)
endfunction()
cosmictiger_use_generated_kernels(cosmictiger)
# Per-order instantiation units: the heavy order-specific code (CPU gravity and
# walk, tree multipoles, Ewald tables, CUDA kick) compiles once per order in its
# own translation unit, so the orders build in parallel.
set(COSMICTIGER_FMM_INSTANCE_DIR ${PROJECT_BINARY_DIR}/instances)
foreach(fmm_order IN LISTS COSMICTIGER_FMM_ORDERS)
    set(FMM_INSTANCE_ORDER ${fmm_order})
    configure_file(src/gravity/fmm_instance_cpu.cpp.in
        ${COSMICTIGER_FMM_INSTANCE_DIR}/fmm_instance_cpu_${fmm_order}.cpp @ONLY)
    target_sources(cosmictiger PRIVATE ${COSMICTIGER_FMM_INSTANCE_DIR}/fmm_instance_cpu_${fmm_order}.cpp)
    if(COSMICTIGER_WITH_CUDA)
        configure_file(src/gravity/fmm_instance_cuda.cu.in
            ${COSMICTIGER_FMM_INSTANCE_DIR}/fmm_instance_cuda_${fmm_order}.cu @ONLY)
        target_sources(cosmictiger PRIVATE ${COSMICTIGER_FMM_INSTANCE_DIR}/fmm_instance_cuda_${fmm_order}.cu)
    endif()
endforeach()

add_executable(healpix2silo src/tools/healpix2silo.cpp)
target_link_libraries(healpix2silo PRIVATE ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} Boost::program_options dl)
target_include_directories(healpix2silo PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(healpix2silo SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})

add_hpx_executable(groups2halofunc
    SOURCES src/core/assert.cpp src/core/hpx.cpp src/tools/groups2halofunc.cpp src/analysis/group_entry.cpp)
target_link_libraries(groups2halofunc PUBLIC Boost::program_options)
target_include_directories(groups2halofunc PRIVATE ${PROJECT_SOURCE_DIR})

foreach(tool_target IN ITEMS healpix2silo groups2halofunc)
    target_compile_options(${tool_target} PRIVATE -Wno-attributes -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(${tool_target} PRIVATE -march=native)
    endif()
endforeach()

include(CTest)
# Keep the application compilation independent of the selected test reporter.
add_library(cosmictiger_application_tests STATIC)
target_include_directories(cosmictiger_application_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cosmictiger PRIVATE cosmictiger_application_tests)
if(BUILD_TESTING)
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

    # The selected run.test application check executes inside the HPX runtime.
    target_sources(cosmictiger_application_tests PRIVATE tests/application_test.cpp)
    target_link_libraries(cosmictiger_application_tests PRIVATE GTest::gtest)
    foreach(voronoi_test IN ITEMS voronoi distributed_voronoi silo_voronoi)
        add_executable(${voronoi_test}_test tests/${voronoi_test}.cpp)
        target_link_libraries(${voronoi_test}_test PRIVATE cosmictiger_silo_voronoi)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${voronoi_test}_test PRIVATE -fno-fast-math -ffp-contract=off)
        endif()
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/tests/${voronoi_test}")
        cosmictiger_add_gtest(NAME ${voronoi_test} TARGET ${voronoi_test}_test
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/tests/${voronoi_test}"
            PROPERTIES TIMEOUT 120)
    endforeach()
    add_executable(halo_catalog_fixture tests/halo_catalog_fixture.cpp)
    target_link_libraries(halo_catalog_fixture PRIVATE cosmictiger_fp16)
    target_compile_options(halo_catalog_fixture PRIVATE -fno-fast-math)
    add_test(NAME halo_viewer COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_halo_viewer.py
        --exporter $<TARGET_FILE:halo_catalog_export>
        --fixture $<TARGET_FILE:halo_catalog_fixture>
        --viewer ${PROJECT_SOURCE_DIR}/tools/halo_viewer/halo_viewer.py)
    set_tests_properties(halo_viewer PROPERTIES TIMEOUT 180 PROCESSORS 2)

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
    add_test(NAME primordial_integration_cpu COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_primordial_integration.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/primordial_integration_cpu
        --recombination-reference $<TARGET_FILE:recombination_reference>)
    set_tests_properties(primordial_integration_cpu PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_library(no_cuda_runtime SHARED tests/no_cuda_runtime.cpp)
        add_test(NAME cpu_only_runtime COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_cpu_only.py $<TARGET_FILE:cosmictiger>
            $<TARGET_FILE:no_cuda_runtime> ${PROJECT_BINARY_DIR}/tests/cpu_only)
        set_tests_properties(cpu_only_runtime PROPERTIES TIMEOUT 250 RUN_SERIAL TRUE)
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
    if(COSMICTIGER_WITH_CUDA)
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
    foreach(boltzmann_two_lpt IN ITEMS false true)
        set(boltzmann_smoke boltzmann_initialization_${boltzmann_two_lpt}_cpu)
        add_test(NAME ${boltzmann_smoke} COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/${boltzmann_smoke}
            --extra-option=--initial.twoLpt=${boltzmann_two_lpt})
        set_tests_properties(${boltzmann_smoke}
            PROPERTIES RUN_SERIAL TRUE TIMEOUT 240 PROCESSORS 2)
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
                --run.test=${halo_mode} --compute.cuda=${halo_cuda} --hydro.enabled=false
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
            --run.test=groupsCatalogueDistributed --compute.cuda=false --hydro.enabled=false
            --initial.particlesPerDimension=8 --analysis.groups=true
            --hpx:threads=2 --hpx:bind=none --hpx:localities=2 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(groupsCatalogue_mpi PROPERTIES
            WORKING_DIRECTORY "${halo_mpi_workdir}" TIMEOUT 240 PROCESSORS 4 RUN_SERIAL TRUE)
    endif()
    add_test(NAME scenarios COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_scenarios.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/scenarios)
    set_tests_properties(scenarios PROPERTIES RUN_SERIAL TRUE TIMEOUT 1200 PROCESSORS 2)

    add_test(NAME grouped_options COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_options.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/grouped_options)
    set_tests_properties(grouped_options PROPERTIES RUN_SERIAL TRUE TIMEOUT 240)

    add_test(NAME linear_growth_response COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_linear_growth.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/linear_growth_response)
    set_tests_properties(linear_growth_response PROPERTIES RUN_SERIAL TRUE TIMEOUT 200)

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

    set(units_output_workdir "${PROJECT_BINARY_DIR}/tests/units_output")
    file(MAKE_DIRECTORY "${units_output_workdir}")
    add_test(NAME units_output COMMAND cosmictiger --run.test=unitsOutput
        --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(units_output PROPERTIES
        WORKING_DIRECTORY "${units_output_workdir}" RUN_SERIAL TRUE TIMEOUT 120)

    set(voronoi_output_workdir "${PROJECT_BINARY_DIR}/tests/voronoi_output")
    file(MAKE_DIRECTORY "${voronoi_output_workdir}")
    add_test(NAME voronoi_output COMMAND cosmictiger --run.test=voronoiOutput
        --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=false
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(voronoi_output PROPERTIES WORKING_DIRECTORY "${voronoi_output_workdir}"
        RUN_SERIAL TRUE TIMEOUT 120 PROCESSORS 2)

    add_executable(radial_kernel_test tests/radial_kernel.cpp)
    target_include_directories(radial_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(radial_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(radial_kernel_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME radial_kernel TARGET radial_kernel_test
        PROPERTIES TIMEOUT 30)

    add_executable(stf_kernels_test tests/stf_kernels.cpp)
    target_include_directories(stf_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_use_generated_kernels(stf_kernels_test)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(stf_kernels_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME stf_kernels TARGET stf_kernels_test
        PROPERTIES TIMEOUT 60)
    # Exercise the CUDA-disabled Ewald initializer even in a CUDA-enabled build.
    add_executable(stf_ewald_cpu_test tests/stf_ewald_cpu.cpp src/fmm/ewald_indices.cpp src/fmm/fmm_order.cpp)
    target_include_directories(stf_ewald_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(stf_ewald_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_compile_definitions(stf_ewald_cpu_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(stf_ewald_cpu_test)
    target_link_libraries(stf_ewald_cpu_test PRIVATE HPX::hpx Boost::boost)
    cosmictiger_add_gtest(NAME stf_ewald_cpu TARGET stf_ewald_cpu_test
        PROPERTIES TIMEOUT 30)

    add_executable(fmm_acceptance_test tests/fmm_acceptance.cpp)
    target_include_directories(fmm_acceptance_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(fmm_acceptance_test PRIVATE -ffast-math)
    target_compile_definitions(fmm_acceptance_test PRIVATE FMM_ACCEPTANCE_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(fmm_acceptance_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME fmm_acceptance TARGET fmm_acceptance_test
        PROPERTIES TIMEOUT 30)

    add_executable(radial_density_kernel_test tests/radial_density_kernel.cpp)
    target_include_directories(radial_density_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(radial_density_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(radial_density_kernel_test PRIVATE -march=native)
    endif()
    cosmictiger_add_gtest(NAME radial_density_kernel TARGET radial_density_kernel_test
        PROPERTIES TIMEOUT 30)

    add_executable(monaghan_sph_test tests/monaghan_sph.cpp)
    target_include_directories(monaghan_sph_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME monaghan_sph TARGET monaghan_sph_test
        PROPERTIES TIMEOUT 20)

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
    # Relax sphere-surface shells on the CPU path that run.scenario=polytropicStar uses
    # without CUDA.
    add_hpx_executable(sphere_surface_test
        SOURCES tests/sphere_surface.cpp src/sph/sphere.cpp)
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
    add_executable(sph_cpu_test tests/sph_cpu.cpp src/sph/sph_cpu.cpp
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

    add_executable(rayleigh_taylor_test tests/rayleigh_taylor.cpp src/scenarios/rayleigh_taylor.cpp)
    target_include_directories(rayleigh_taylor_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(rayleigh_taylor_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(rayleigh_taylor_test)
    target_link_libraries(rayleigh_taylor_test PRIVATE HPX::hpx)
    cosmictiger_add_gtest(NAME rayleigh_taylor TARGET rayleigh_taylor_test
        PROPERTIES TIMEOUT 120)

    add_executable(silo_spheres_test tests/silo_spheres.cpp)
    target_include_directories(silo_spheres_test PRIVATE ${PROJECT_SOURCE_DIR} ${SILO_INCLUDE_DIR})
    target_link_libraries(silo_spheres_test PRIVATE ${SILO_LIBRARY})
    set(silo_spheres_workdir "${PROJECT_BINARY_DIR}/tests/silo_spheres")
    file(MAKE_DIRECTORY "${silo_spheres_workdir}")
    cosmictiger_add_gtest(NAME silo_spheres TARGET silo_spheres_test
        WORKING_DIRECTORY "${silo_spheres_workdir}"
        PROPERTIES TIMEOUT 20)

    add_executable(particle_selection_test tests/particle_selection.cpp)
    target_include_directories(particle_selection_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_add_gtest(NAME particle_selection TARGET particle_selection_test
        PROPERTIES TIMEOUT 10)

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

        add_executable(stf_kernels_cuda_test tests/stf_kernels_cuda.cu)
        target_include_directories(stf_kernels_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(stf_kernels_cuda_test)
        target_link_libraries(stf_kernels_cuda_test PRIVATE CUDA::cudart)
        cosmictiger_add_gtest(NAME stf_kernels_cuda TARGET stf_kernels_cuda_test
            PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(stf_l2l_cuda_test tests/stf_l2l_cuda.cu)
        target_include_directories(stf_l2l_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(stf_l2l_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(stf_l2l_cuda_test)
        target_link_libraries(stf_l2l_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(stf_l2l_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)
        cosmictiger_add_gtest(NAME stf_l2l_cuda TARGET stf_l2l_cuda_test
            PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(fmm_acceptance_cuda_test tests/fmm_acceptance_cuda.cu)
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

        add_executable(monaghan_sph_cuda_test tests/monaghan_sph_cuda.cu
            tests/cuda_memory_backend.cpp
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

        # These host regressions need CUDA headers, but do not initialize or run a GPU.
        add_executable(numerical_utilities_test tests/numerical_utilities.cpp)
        add_executable(compression_test tests/compression.cpp src/analysis/compress.cpp)
        foreach(test_target IN ITEMS numerical_utilities_test compression_test)
            target_include_directories(${test_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_compile_definitions(${test_target} PRIVATE USE_CUDA)
            target_link_libraries(${test_target} PRIVATE CUDA::cudart)
            if(COMPILER_SUPPORTS_MARCH_NATIVE)
                target_compile_options(${test_target} PRIVATE -march=native)
            endif()
        endforeach()
        cosmictiger_add_gtest(NAME numerical_utilities TARGET numerical_utilities_test
            PROPERTIES TIMEOUT 10)
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

    foreach(test_name IN ITEMS sph_gravity_cpu sph_gravity_cuda checkpoint_state_dm checkpoint_state_sph)
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
        add_test(NAME ${test_name} COMMAND cosmictiger
            --run.test=${test_mode} --initial.particlesPerDimension=8 --hydro.enabled=${test_sph}
            --compute.cuda=${test_cuda} --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${test_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
    endforeach()
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
        if(NOT COSMICTIGER_WITH_CUDA)
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
    set(rejected_options_workdir "${PROJECT_BINARY_DIR}/tests/rejected_options")
    file(MAKE_DIRECTORY "${rejected_options_workdir}")
    add_test(NAME rejected_options COMMAND cosmictiger --invalid_option_for_test=true
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(rejected_options PROPERTIES WORKING_DIRECTORY "${rejected_options_workdir}"
        RUN_SERIAL TRUE TIMEOUT 20 WILL_FAIL TRUE)
    if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
        set(fmm_tree_mpi_workdir "${PROJECT_BINARY_DIR}/tests/fmm_tree_mpi")
        set(voronoi_output_mpi_workdir "${PROJECT_BINARY_DIR}/tests/voronoi_output_mpi")
        file(MAKE_DIRECTORY "${voronoi_output_mpi_workdir}")
        add_test(NAME voronoi_output_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
            --run.test=voronoiOutput --initial.particlesPerDimension=8 --hydro.enabled=true
            --compute.cuda=false --hpx:threads=2 --hpx:bind=none --hpx:localities=2)
        set_tests_properties(voronoi_output_mpi PROPERTIES
            WORKING_DIRECTORY "${voronoi_output_mpi_workdir}" RUN_SERIAL TRUE TIMEOUT 180 PROCESSORS 4)
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
    foreach(smoke_mode IN ITEMS sod sph)
        add_test(NAME smoke_${smoke_mode}_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
            $<TARGET_FILE:cosmictiger>
            ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode}_cpu --cpu)
        set_tests_properties(smoke_${smoke_mode}_cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
    endforeach()
    foreach(smoke_mode IN ITEMS sod dm_cpu dm_cuda sph)
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
    add_test(NAME checkpoint_integrity COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_checkpoint.py
        $<TARGET_FILE:cosmictiger>
        ${PROJECT_BINARY_DIR}/tests/checkpoint_integrity)
    set_tests_properties(checkpoint_integrity PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
else()
    target_sources(cosmictiger_application_tests PRIVATE tests/application_test_direct.cpp)
endif()
