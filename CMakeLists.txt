# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

cmake_minimum_required(VERSION 3.24)
project(cosmictiger LANGUAGES CXX C Fortran)

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
# Every order from 3 to 9 is compiled in (per-order instantiations) and the
# run selects one with gravity.fmmOrder; this value is only that option's
# default. Order 3 (dipoles, quadratic expansions) is the smallest order whose
# local expansion holds the periodic background quadratic; order 9 (degree-7
# multipoles, degree-8 expansions) is where single precision stops gaining.
set(COSMICTIGER_WITH_ORDER "7" CACHE STRING "Default FMM order (gravity.fmmOrder), 3 to 9")
if(NOT COSMICTIGER_WITH_ORDER MATCHES "^(3|4|5|6|7|8|9)$")
    message(FATAL_ERROR
        "COSMICTIGER_WITH_ORDER must be an integer from 3 to 9; got '${COSMICTIGER_WITH_ORDER}'")
endif()
set(COSMICTIGER_FMM_ORDERS 3 4 5 6 7 8 9)
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

message(STATUS "cosmictiger default FMM order = ${COSMICTIGER_WITH_ORDER} (orders 3 to 9 compiled)")
message(STATUS "cosmictiger maximum bucket size = ${COSMICTIGER_WITH_MAX_BUCKET}")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

set(cosmictiger_source_files
    healpix/error_handling.cc
    healpix/geom_utils.cc
    healpix/healpix_base.cc
    healpix/healpix_tables.cc
    healpix/pointing.cc
    healpix/string_utils.cc
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
    src/cosmology/recfast.cpp
    src/cosmology/zero_order.cpp
    src/cuda/cuda.cpp
    src/cuda/cuda_memory.cu
    src/cuda/cuda_stream.cpp
    src/fmm/ewald_indices.cpp
    src/fmm/ewald_indices.cu
    src/fmm/fmm_order.cpp
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
    src/problems/analytic.cpp
    src/problems/analytic.cu
    src/problems/exact_sod.cpp
    src/problems/hydro.cpp
    src/problems/plummer.cpp
    src/problems/sedov.cpp
    src/problems/sphere.cpp
    src/problems/sphere.cu
    src/problems/test.cpp
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
    tests/analysis_backends.cpp
    tests/bh.cpp
    tests/fmm_tree.cpp
    tests/fmm_walk.cpp
    tests/groups_catalogue.cpp
    tests/kernel_selection_cuda.cu
    tests/linear_growth.cpp
    tests/rockstar.cpp
    tests/sph_gravity.cpp
    tests/units_output.cpp
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
    cosmictiger/core/assert.hpp
    cosmictiger/core/containers.hpp
    cosmictiger/core/defs.hpp
    cosmictiger/core/driver.hpp
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
    cosmictiger/cosmology/recfast.hpp
    cosmictiger/cosmology/recombination_thermal.hpp
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
    cosmictiger/problems/analytic.hpp
    cosmictiger/problems/exact_sod.hpp
    cosmictiger/problems/hydro.hpp
    cosmictiger/problems/plummer.hpp
    cosmictiger/problems/sedov.hpp
    cosmictiger/problems/sphere.hpp
    cosmictiger/problems/test.hpp
    cosmictiger/sph/monaghan_sph.hpp
    cosmictiger/sph/sph.hpp
    cosmictiger/sph/sph_chemistry.hpp
    cosmictiger/sph/sph_cpu.hpp
    cosmictiger/sph/sph_cuda.hpp
    cosmictiger/sph/sph_kernel.hpp
    cosmictiger/sph/sph_particles.hpp
    cosmictiger/sph/sph_tree.hpp
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
add_library(primordial_chemistry STATIC src/chemistry/primordial_chemistry.cpp)
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

add_hpx_executable(cosmictiger
    DEPENDENCIES Boost::boost Boost::program_options GSL::gsl GSL::gslcblas
        ${FFTW3F_LIBRARY} ${SILO_LIBRARY} ${CHEALPIX_LIBRARY} dl quadmath
    SOURCES ${cosmictiger_source_files}
    HEADERS ${cosmictiger_header_files})
option(COSMICTIGER_BUILD_STF_BENCHMARKS "Build the opt-in STF comparison harness" OFF)
if(COSMICTIGER_BUILD_STF_BENCHMARKS)
    target_sources(cosmictiger PRIVATE benchmarks/stf_walk.cpp)
    set_source_files_properties(src/problems/test.cpp PROPERTIES COMPILE_DEFINITIONS COSMICTIGER_STF_BENCHMARKS)
endif()
target_link_libraries(cosmictiger PRIVATE primordial_chemistry cosmictiger_fp16)
target_include_directories(cosmictiger PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(cosmictiger SYSTEM PRIVATE
    ${FFTW3F_INCLUDE_DIR} ${SILO_INCLUDE_DIR} ${CHEALPIX_INCLUDE_DIR})
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

add_executable(recfast recfast/recfast.f)
# Do not silently change fixed-form equations by truncating long statements.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(recfast PRIVATE -Werror=line-truncation)
endif()
configure_file(power.init power.init COPYONLY)
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
if(BUILD_TESTING)
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
        add_test(NAME ${chemistry_test} COMMAND ${chemistry_test}_test)
        set_tests_properties(${chemistry_test} PROPERTIES TIMEOUT 180)
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
    add_test(NAME primordial_integration_cpu COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_primordial_integration.py $<TARGET_FILE:cosmictiger>
        ${PROJECT_SOURCE_DIR}/power.init ${PROJECT_BINARY_DIR}/tests/primordial_integration_cpu)
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
        add_test(NAME primordial_cuda COMMAND primordial_cuda_test)
        set_tests_properties(primordial_cuda PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
    foreach(recfast_variant IN ITEMS standard fast_math)
        set(recfast_target recfast_${recfast_variant}_test)
        add_executable(${recfast_target} tests/recfast.cpp src/cosmology/recfast.cpp)
        target_include_directories(${recfast_target} PRIVATE ${PROJECT_SOURCE_DIR})
        set(thermal_target recombination_thermal_${recfast_variant}_test)
        add_executable(${thermal_target} tests/recombination_thermal.cpp)
        target_include_directories(${thermal_target} PRIVATE ${PROJECT_SOURCE_DIR})
        set(background_target recombination_background_${recfast_variant}_test)
        add_executable(${background_target} tests/recombination_background.cpp
            src/cosmology/zero_order.cpp src/cosmology/recfast.cpp)
        target_include_directories(${background_target} PRIVATE ${PROJECT_SOURCE_DIR})
        if(recfast_variant STREQUAL "fast_math")
            target_compile_options(${recfast_target} PRIVATE -ffast-math)
            target_compile_options(${thermal_target} PRIVATE -ffast-math)
            target_compile_options(${background_target} PRIVATE -ffast-math)
        endif()
        add_test(NAME recfast_${recfast_variant} COMMAND ${recfast_target})
        add_test(NAME recombination_thermal_${recfast_variant} COMMAND ${thermal_target})
        add_test(NAME recombination_background_${recfast_variant} COMMAND ${background_target}
            ${PROJECT_BINARY_DIR}/tests/recfast_history/fiducial/history.dat)
        set_tests_properties(recombination_background_${recfast_variant}
            PROPERTIES FIXTURES_REQUIRED recfast_history TIMEOUT 30)
        set_tests_properties(recfast_${recfast_variant} recombination_thermal_${recfast_variant}
            PROPERTIES TIMEOUT 30)
    endforeach()
    add_test(NAME recfast_history COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_recfast.py --executable $<TARGET_FILE:recfast>
        --reader $<TARGET_FILE:recfast_standard_test> $<TARGET_FILE:recfast_fast_math_test>
        --workdir ${PROJECT_BINARY_DIR}/tests/recfast_history)
    set_tests_properties(recfast_history PROPERTIES FIXTURES_SETUP recfast_history TIMEOUT 90)

    foreach(fp16_variant IN ITEMS standard fast_math)
        add_executable(fp16_${fp16_variant}_test tests/fp16.cpp)
        target_include_directories(fp16_${fp16_variant}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(fp16_${fp16_variant}_test PRIVATE cosmictiger_fp16)
        if(fp16_variant STREQUAL "fast_math")
            target_compile_options(fp16_${fp16_variant}_test PRIVATE -ffast-math)
        endif()
        add_test(NAME fp16_${fp16_variant} COMMAND fp16_${fp16_variant}_test)
        set_tests_properties(fp16_${fp16_variant} PROPERTIES TIMEOUT 30)
    endforeach()

    add_executable(rockstar_eigenpair_test tests/rockstar_eigenpair.cpp src/math/math.cpp)
    target_include_directories(rockstar_eigenpair_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(rockstar_eigenpair_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(rockstar_eigenpair_test)
    add_test(NAME rockstar_eigenpair COMMAND rockstar_eigenpair_test)
    set_tests_properties(rockstar_eigenpair PROPERTIES TIMEOUT 30)
    foreach(halo_unit IN ITEMS rockstar_core rockstar_properties lightcone_epoch)
        add_executable(${halo_unit}_test tests/${halo_unit}.cpp)
        target_include_directories(${halo_unit}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(${halo_unit}_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(${halo_unit}_test)
        add_test(NAME ${halo_unit} COMMAND ${halo_unit}_test)
        set_tests_properties(${halo_unit} PROPERTIES TIMEOUT 30)
    endforeach()
    add_executable(rockstar_properties_fast_math_test tests/rockstar_properties.cpp)
    target_include_directories(rockstar_properties_fast_math_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(rockstar_properties_fast_math_test PRIVATE -ffast-math)
    add_test(NAME rockstar_properties_fast_math COMMAND rockstar_properties_fast_math_test)
    set_tests_properties(rockstar_properties_fast_math PROPERTIES TIMEOUT 30)
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
    add_test(NAME units COMMAND units_test)
    set_tests_properties(units PROPERTIES TIMEOUT 20)

    add_executable(units_fast_math_test tests/units.cpp)
    target_include_directories(units_fast_math_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(units_fast_math_test PRIVATE Boost::boost)
    target_compile_options(units_fast_math_test PRIVATE -ffast-math)
    add_test(NAME units_fast_math COMMAND units_fast_math_test)
    set_tests_properties(units_fast_math PROPERTIES TIMEOUT 20)

    include(CheckIncludeFileCXX)
    foreach(float_float_variant IN ITEMS standard fast_math)
        set(float_float_target float_float_${float_float_variant}_test)
        add_executable(${float_float_target} tests/float_float.cpp)
        target_include_directories(${float_float_target} PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${float_float_target} PRIVATE Boost::boost)
        if(float_float_variant STREQUAL "fast_math")
            target_compile_options(${float_float_target} PRIVATE -ffast-math)
        endif()
        add_test(NAME float_float_${float_float_variant} COMMAND ${float_float_target})
        set_tests_properties(float_float_${float_float_variant} PROPERTIES TIMEOUT 30)
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
        add_test(NAME simd_${simd_variant} COMMAND ${simd_target})
        set_tests_properties(simd_${simd_variant} PROPERTIES TIMEOUT 20)
    endforeach()

    set(units_output_workdir "${PROJECT_BINARY_DIR}/tests/units_output")
    file(MAKE_DIRECTORY "${units_output_workdir}")
    add_test(NAME units_output COMMAND cosmictiger --run.test=unitsOutput
        --initial.particlesPerDimension=8 --hydro.enabled=true --compute.cuda=true
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(units_output PROPERTIES
        WORKING_DIRECTORY "${units_output_workdir}" RUN_SERIAL TRUE TIMEOUT 120)

    add_executable(radial_kernel_test tests/radial_kernel.cpp)
    target_include_directories(radial_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(radial_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(radial_kernel_test PRIVATE -march=native)
    endif()
    add_test(NAME radial_kernel COMMAND radial_kernel_test)
    set_tests_properties(radial_kernel PROPERTIES TIMEOUT 30)

    add_executable(stf_kernels_test tests/stf_kernels.cpp)
    target_include_directories(stf_kernels_test PRIVATE ${PROJECT_SOURCE_DIR})
    cosmictiger_use_generated_kernels(stf_kernels_test)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(stf_kernels_test PRIVATE -march=native)
    endif()
    add_test(NAME stf_kernels COMMAND stf_kernels_test)
    set_tests_properties(stf_kernels PROPERTIES TIMEOUT 60)
    # Exercise the CUDA-disabled Ewald initializer even in a CUDA-enabled build.
    add_executable(stf_ewald_cpu_test tests/stf_ewald_cpu.cpp src/fmm/ewald_indices.cpp src/fmm/fmm_order.cpp)
    target_include_directories(stf_ewald_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(stf_ewald_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_compile_definitions(stf_ewald_cpu_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    cosmictiger_use_generated_kernels(stf_ewald_cpu_test)
    target_link_libraries(stf_ewald_cpu_test PRIVATE HPX::hpx Boost::boost)
    add_test(NAME stf_ewald_cpu COMMAND stf_ewald_cpu_test)
    set_tests_properties(stf_ewald_cpu PROPERTIES TIMEOUT 30)

    add_executable(fmm_acceptance_test tests/fmm_acceptance.cpp)
    target_include_directories(fmm_acceptance_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(fmm_acceptance_test PRIVATE -ffast-math)
    target_compile_definitions(fmm_acceptance_test PRIVATE FMM_ACCEPTANCE_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(fmm_acceptance_test PRIVATE -march=native)
    endif()
    add_test(NAME fmm_acceptance COMMAND fmm_acceptance_test)
    set_tests_properties(fmm_acceptance PROPERTIES TIMEOUT 30)

    add_executable(radial_density_kernel_test tests/radial_density_kernel.cpp)
    target_include_directories(radial_density_kernel_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_definitions(radial_density_kernel_test PRIVATE RADIAL_KERNEL_TEST_SIMD)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(radial_density_kernel_test PRIVATE -march=native)
    endif()
    add_test(NAME radial_density_kernel COMMAND radial_density_kernel_test)
    set_tests_properties(radial_density_kernel PROPERTIES TIMEOUT 30)

    add_executable(monaghan_sph_test tests/monaghan_sph.cpp)
    target_include_directories(monaghan_sph_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME monaghan_sph COMMAND monaghan_sph_test)
    set_tests_properties(monaghan_sph PROPERTIES TIMEOUT 20)

    foreach(analysis_kernel IN ITEMS power lightcone)
        add_executable(${analysis_kernel}_cpu_test
            tests/${analysis_kernel}_cpu.cpp src/analysis/${analysis_kernel}_cpu.cpp)
        target_include_directories(${analysis_kernel}_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(${analysis_kernel}_cpu_test PRIVATE HPX::hpx)
        target_compile_options(${analysis_kernel}_cpu_test PRIVATE -ffast-math)
        add_test(NAME ${analysis_kernel}_cpu COMMAND ${analysis_kernel}_cpu_test)
        set_tests_properties(${analysis_kernel}_cpu PROPERTIES TIMEOUT 60)
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
        add_test(NAME power_cuda COMMAND power_cuda_test)
        set_tests_properties(power_cuda PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(lightcone_cuda_test tests/lightcone_cuda.cu src/analysis/lightcone.cu
            src/analysis/lightcone_cpu.cpp src/core/assert.cu src/cuda/cuda_stream.cpp)
        target_include_directories(lightcone_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(lightcone_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(lightcone_cuda_test)
        target_link_libraries(lightcone_cuda_test PRIVATE HPX::hpx CUDA::cudart)
        target_compile_options(lightcone_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        set_target_properties(lightcone_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME lightcone_cuda COMMAND lightcone_cuda_test)
        set_tests_properties(lightcone_cuda PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_hpx_executable(cuda_async_test
            SOURCES tests/cuda_async.cpp tests/cuda_async.cu src/cuda/cuda_stream.cpp)
        target_include_directories(cuda_async_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(cuda_async_test PRIVATE USE_CUDA)
        target_link_libraries(cuda_async_test PRIVATE CUDA::cudart)
        add_test(NAME cuda_async COMMAND cuda_async_test --hpx:threads=1 --hpx:bind=none)
        set_tests_properties(cuda_async PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77 PROCESSORS 1 ENVIRONMENT "OMP_NUM_THREADS=1;OMP_THREAD_LIMIT=1")
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
    add_test(NAME drift_cpu COMMAND drift_cpu_test)
    set_tests_properties(drift_cpu PROPERTIES TIMEOUT 30)

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
    add_test(NAME drift_dispatch COMMAND drift_dispatch_test --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(drift_dispatch PROPERTIES TIMEOUT 30 PROCESSORS 2
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
    # Relax sphere-surface shells on the CPU path that run.test=star uses
    # without CUDA.
    add_hpx_executable(sphere_surface_test
        SOURCES tests/sphere_surface.cpp src/problems/sphere.cpp)
    target_include_directories(sphere_surface_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(sphere_surface_test PRIVATE -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(sphere_surface_test PRIVATE -march=native)
    endif()
    add_test(NAME sphere_surface COMMAND sphere_surface_test --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(sphere_surface PROPERTIES TIMEOUT 120 PROCESSORS 2
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

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
    add_test(NAME sph_cpu COMMAND sph_cpu_test)
    set_tests_properties(sph_cpu PROPERTIES TIMEOUT 120)

    add_executable(silo_spheres_test tests/silo_spheres.cpp)
    target_include_directories(silo_spheres_test PRIVATE ${PROJECT_SOURCE_DIR} ${SILO_INCLUDE_DIR})
    target_link_libraries(silo_spheres_test PRIVATE ${SILO_LIBRARY})
    set(silo_spheres_workdir "${PROJECT_BINARY_DIR}/tests/silo_spheres")
    file(MAKE_DIRECTORY "${silo_spheres_workdir}")
    add_test(NAME silo_spheres COMMAND silo_spheres_test)
    set_tests_properties(silo_spheres PROPERTIES
        WORKING_DIRECTORY "${silo_spheres_workdir}" TIMEOUT 20)

    add_executable(particle_selection_test tests/particle_selection.cpp)
    target_include_directories(particle_selection_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME particle_selection COMMAND particle_selection_test)
    set_tests_properties(particle_selection PROPERTIES TIMEOUT 10)

    if(COSMICTIGER_WITH_CUDA)
        foreach(float_float_variant IN ITEMS standard fast_math)
            set(float_float_target float_float_cuda_${float_float_variant}_test)
            add_executable(${float_float_target} tests/float_float_cuda.cu)
            target_include_directories(${float_float_target} PRIVATE ${PROJECT_SOURCE_DIR})
            target_link_libraries(${float_float_target} PRIVATE CUDA::cudart Boost::boost)
            if(float_float_variant STREQUAL "fast_math")
                target_compile_options(${float_float_target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-ffast-math>)
            endif()
            add_test(NAME float_float_cuda_${float_float_variant} COMMAND ${float_float_target})
            set_tests_properties(float_float_cuda_${float_float_variant}
                PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
        endforeach()

        add_executable(stf_kernels_cuda_test tests/stf_kernels_cuda.cu)
        target_include_directories(stf_kernels_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        cosmictiger_use_generated_kernels(stf_kernels_cuda_test)
        target_link_libraries(stf_kernels_cuda_test PRIVATE CUDA::cudart)
        add_test(NAME stf_kernels_cuda COMMAND stf_kernels_cuda_test)
        set_tests_properties(stf_kernels_cuda PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(stf_l2l_cuda_test tests/stf_l2l_cuda.cu)
        target_include_directories(stf_l2l_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(stf_l2l_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
        cosmictiger_use_generated_kernels(stf_l2l_cuda_test)
        target_link_libraries(stf_l2l_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(stf_l2l_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)
        add_test(NAME stf_l2l_cuda COMMAND stf_l2l_cuda_test)
        set_tests_properties(stf_l2l_cuda PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(fmm_acceptance_cuda_test tests/fmm_acceptance_cuda.cu)
        target_include_directories(fmm_acceptance_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(fmm_acceptance_cuda_test PRIVATE USE_CUDA)
        target_link_libraries(fmm_acceptance_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(fmm_acceptance_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math;-Xcompiler=-ffast-math>)
        add_test(NAME fmm_acceptance_cuda COMMAND fmm_acceptance_cuda_test)
        set_tests_properties(fmm_acceptance_cuda PROPERTIES
            RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)

        add_executable(radial_kernel_cuda_test tests/radial_kernel_cuda.cu)
        target_include_directories(radial_kernel_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(radial_kernel_cuda_test PRIVATE CUDA::cudart)
        target_compile_options(radial_kernel_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        add_test(NAME radial_kernel_cuda COMMAND radial_kernel_cuda_test)
        set_tests_properties(radial_kernel_cuda PROPERTIES RUN_SERIAL TRUE TIMEOUT 120 SKIP_RETURN_CODE 77)

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
        add_test(NAME monaghan_sph_cuda COMMAND monaghan_sph_cuda_test)
        set_tests_properties(monaghan_sph_cuda PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)

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
        add_test(NAME numerical_utilities COMMAND numerical_utilities_test)
        set_tests_properties(numerical_utilities PROPERTIES TIMEOUT 10)
        foreach(compression_case IN ITEMS empty single repeated repeated_clusters zero_position_spread
                zero_velocity_spread normal)
            add_test(NAME compression_${compression_case}
                COMMAND compression_test ${compression_case})
            set_tests_properties(compression_${compression_case} PROPERTIES TIMEOUT 10)
        endforeach()

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
            ${PROJECT_SOURCE_DIR}/power.init ${PROJECT_BINARY_DIR}/tests/smoke_kernels_${kernel_case}
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
            --compute.cuda=true --hpx:threads=2 --hpx:bind=none --hpx:localities=2)
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
            $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
            ${PROJECT_BINARY_DIR}/tests/smoke_sph_mpi --mpi-exec=${MPIEXEC_EXECUTABLE})
        set_tests_properties(smoke_sph_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
        add_test(NAME smoke_sph_cpu_mpi COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py sph_mpi
            $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
            ${PROJECT_BINARY_DIR}/tests/smoke_sph_cpu_mpi --mpi-exec=${MPIEXEC_EXECUTABLE} --cpu)
        set_tests_properties(smoke_sph_cpu_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 260)
    endif()
    foreach(smoke_mode IN ITEMS sod sph)
        add_test(NAME smoke_${smoke_mode}_cpu COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
            $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
            ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode}_cpu --cpu)
        set_tests_properties(smoke_${smoke_mode}_cpu PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
    endforeach()
    foreach(smoke_mode IN ITEMS sod dm_cpu dm_cuda sph)
        add_test(NAME smoke_${smoke_mode} COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/run_smoke.py ${smoke_mode}
            $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
            ${PROJECT_BINARY_DIR}/tests/smoke_${smoke_mode})
        set_tests_properties(smoke_${smoke_mode} PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
    endforeach()
    add_test(NAME smoke_dm_min_rung COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_smoke.py dm_cpu
        $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
        ${PROJECT_BINARY_DIR}/tests/smoke_dm_min_rung --minimum-rung=10)
    set_tests_properties(smoke_dm_min_rung PROPERTIES RUN_SERIAL TRUE TIMEOUT 500)
    add_test(NAME checkpoint_integrity COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/run_checkpoint.py
        $<TARGET_FILE:cosmictiger> ${PROJECT_SOURCE_DIR}/power.init
        ${PROJECT_BINARY_DIR}/tests/checkpoint_integrity)
    set_tests_properties(checkpoint_integrity PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
endif()
