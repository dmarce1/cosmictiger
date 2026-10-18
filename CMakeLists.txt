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
set(COSMICTIGER_WITH_ORDER "7" CACHE STRING "Expansion order P+1")
if(NOT COSMICTIGER_WITH_ORDER STREQUAL "7")
    message(FATAL_ERROR "The checked-in STF kernels use source degree 5 and local degree 6")
endif()
if(COSMICTIGER_WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(GSL REQUIRED)
find_path(FFTW3F_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED)
find_path(SILO_INCLUDE_DIR silo.h REQUIRED)
find_library(SILO_LIBRARY NAMES siloh5 silo REQUIRED)
find_path(CHEALPIX_INCLUDE_DIR chealpix.h REQUIRED)
find_library(CHEALPIX_LIBRARY NAMES chealpix REQUIRED)

message(STATUS "cosmictiger expansion order P+1 = ${COSMICTIGER_WITH_ORDER}")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

set(cosmictiger_source_files
    healpix/error_handling.cc
    healpix/geom_utils.cc
    healpix/healpix_base.cc
    healpix/healpix_tables.cc
    healpix/pointing.cc
    healpix/string_utils.cc
    src/analytic.cu
    src/analytic.cpp
    src/assert.cpp
    src/assert.cu
    src/bh.cpp
    src/bh.cu
    src/boltzmann.cpp
    src/compress.cpp
    src/cosmology.cpp
    src/cuda.cpp
    src/cuda_memory.cu
    src/drift.cpp
    src/drift.cu
    src/driver.cpp
    src/domain.cpp
    src/ewald_indices.cpp
    src/ewald_indices.cu
    src/flops.cpp
    src/flops.cu
    src/fft.cpp
    src/gravity.cpp
    src/gravity.cu
    src/group_tree.cpp
    src/groups.cpp
    src/group_entry.cpp
    src/groups_find.cpp
    src/initialize.cpp
    src/hpx.cpp
    src/healpix.cu
    src/kick.cpp
    src/kick.cu
    src/kick_workspace.cpp
    src/lightcone.cpp
    src/lightcone.cu
    src/main.cpp
    src/math.cpp
    src/memused.cpp
    src/options.cpp
    src/output.cpp
    src/particles.cpp
    src/power.cpp
    src/power.cu
    src/profiler.cpp
    src/rockstar.cpp
    src/rockstar.cu
    src/stack_trace.cpp
    src/test.cpp
    src/tree.cpp
    src/view.cpp
    src/zero_order.cpp
    src/hydro.cpp
    src/kernel.cpp
    src/kernel.cu
    src/exact_sod.cpp
    src/sedov.cpp
    src/sph_particles.cpp
    src/sph.cpp
    src/sph.cu
    src/sph_prehydro.cu
    src/sph_hydro.cu
    src/sph_rungs.cu
    src/sphere.cu
    src/sph_tree.cpp
    tests/kernel_selection_cuda.cu
    tests/fmm_tree.cpp
    tests/fmm_walk.cpp
    tests/sph_gravity.cpp
    tests/linear_growth.cpp
    tests/units_output.cpp
)

set(cosmictiger_header_files
    cosmictiger/assert.hpp
    cosmictiger/bh.hpp
    cosmictiger/boltzmann.hpp
    cosmictiger/complex.hpp
    cosmictiger/containers.hpp
    cosmictiger/compress.hpp
    cosmictiger/cosmology.hpp
    cosmictiger/cuda.hpp
    cosmictiger/cuda_reduce.hpp
    cosmictiger/defs.hpp
    cosmictiger/device_vector.hpp
    cosmictiger/domain.hpp
    cosmictiger/drift.hpp
    cosmictiger/driver.hpp
    cosmictiger/ewald_indices.hpp
    cosmictiger/fixed.hpp
    cosmictiger/fft.hpp
    cosmictiger/fixed_capacity_vector.hpp
    cosmictiger/gravity.hpp
    cosmictiger/initialize.hpp
    cosmictiger/interpolation.hpp
    cosmictiger/flops.hpp
    cosmictiger/fp16.hpp
    cosmictiger/group_tree.hpp
    cosmictiger/groups.hpp
    cosmictiger/group_entry.hpp
    cosmictiger/groups_find.hpp
    cosmictiger/hpx.hpp
    cosmictiger/kick.hpp
    cosmictiger/kick_workspace.hpp
    cosmictiger/lightcone.hpp
    cosmictiger/math.hpp
    cosmictiger/memused.hpp
    cosmictiger/options.hpp
    cosmictiger/options_metadata.hpp
    cosmictiger/output.hpp
    cosmictiger/particles.hpp
    cosmictiger/power.hpp
    cosmictiger/profiler.hpp
    cosmictiger/random_number_generator.hpp
    cosmictiger/range.hpp
    cosmictiger/rockstar.hpp
    cosmictiger/safe_io.hpp
    cosmictiger/sedov.hpp
    cosmictiger/silo_spheres.hpp
    cosmictiger/simd.hpp
    cosmictiger/simd_backend.hpp
    cosmictiger/simd_math.hpp
    cosmictiger/stack_vector.hpp
    cosmictiger/stack_trace.hpp
    cosmictiger/test.hpp
    cosmictiger/tree.hpp
    cosmictiger/timer.hpp
    cosmictiger/units.hpp
    cosmictiger/simulation_units.hpp
    cosmictiger/unordered_set_ts.hpp
    cosmictiger/view.hpp
    cosmictiger/zero_order.hpp
    cosmictiger/hydro.hpp
    cosmictiger/kernel.hpp
    cosmictiger/radial_kernel.hpp
    cosmictiger/radial_density_kernel.hpp
    cosmictiger/monaghan_sph.hpp
    cosmictiger/exact_sod.hpp
    cosmictiger/sph.hpp
    cosmictiger/sph_cuda.hpp
    cosmictiger/sph_particles.hpp
    cosmictiger/sph_tree.hpp
    cosmictiger/sphere.hpp
)

if(NOT COSMICTIGER_WITH_CUDA)
    list(FILTER cosmictiger_source_files EXCLUDE REGEX "\\.cu$")
endif()

add_hpx_executable(cosmictiger
    DEPENDENCIES Boost::boost Boost::program_options GSL::gsl GSL::gslcblas
        ${FFTW3F_LIBRARY} ${SILO_LIBRARY} ${CHEALPIX_LIBRARY} dl quadmath
    SOURCES ${cosmictiger_source_files}
    HEADERS ${cosmictiger_header_files})
option(COSMICTIGER_BUILD_STF_BENCHMARKS "Build the opt-in STF comparison harness" OFF)
if(COSMICTIGER_BUILD_STF_BENCHMARKS)
    target_sources(cosmictiger PRIVATE benchmarks/stf_walk.cpp)
    set_source_files_properties(src/test.cpp PROPERTIES COMPILE_DEFINITIONS COSMICTIGER_STF_BENCHMARKS)
endif()
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
    set_source_files_properties(src/drift.cu PROPERTIES COMPILE_OPTIONS -maxrregcount=64)
    set_source_files_properties(src/sph.cu src/sph_prehydro.cu src/sph_hydro.cu src/sph_rungs.cu
        PROPERTIES COMPILE_OPTIONS -maxrregcount=128)
endif()

add_executable(recfast recfast/recfast.f)
add_executable(fmm_kernel_codegen src/fmm_kernel_codegen.cpp cosmictiger/tensor.hpp)
target_link_libraries(fmm_kernel_codegen PRIVATE dl)
target_include_directories(fmm_kernel_codegen PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(fmm_kernel_codegen PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
if(COSMICTIGER_WITH_CUDA)
    target_compile_definitions(fmm_kernel_codegen PRIVATE USE_CUDA)
    target_link_libraries(fmm_kernel_codegen PRIVATE CUDA::cudart)
endif()
configure_file(power.init power.init COPYONLY)
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/generated/cosmictiger/ewald_kernels.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/generated/cosmictiger
    COMMAND fmm_kernel_codegen > ${PROJECT_BINARY_DIR}/generated/cosmictiger/ewald_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "Generating Ewald derivatives for comparison with the checked-in header")
add_custom_target(generate_fmm_kernels DEPENDS ${PROJECT_BINARY_DIR}/generated/cosmictiger/ewald_kernels.hpp)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(generate_stf_kernels
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_stf_m2l.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generating source degree 5 and local degree 6 STF kernels")
add_custom_target(check_stf_kernels
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_stf_m2l.py --check
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(healpix2silo src/healpix2silo.cpp)
target_link_libraries(healpix2silo PRIVATE ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} Boost::program_options dl)
target_include_directories(healpix2silo PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(healpix2silo SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})

add_hpx_executable(groups2halofunc
    SOURCES src/assert.cpp src/hpx.cpp src/groups2halofunc.cpp src/group_entry.cpp)
target_link_libraries(groups2halofunc PUBLIC Boost::program_options)
target_include_directories(groups2halofunc PRIVATE ${PROJECT_SOURCE_DIR})

foreach(tool_target IN ITEMS fmm_kernel_codegen healpix2silo groups2halofunc)
    target_compile_options(${tool_target} PRIVATE -Wno-attributes -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(${tool_target} PRIVATE -march=native)
    endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
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
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(stf_kernels_test PRIVATE -march=native)
    endif()
    add_test(NAME stf_kernels COMMAND stf_kernels_test)
    set_tests_properties(stf_kernels PROPERTIES TIMEOUT 60)
    # Exercise the CUDA-disabled Ewald initializer even in a CUDA-enabled build.
    add_executable(stf_ewald_cpu_test tests/stf_ewald_cpu.cpp src/ewald_indices.cpp)
    target_include_directories(stf_ewald_cpu_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(stf_ewald_cpu_test SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR})
    target_compile_definitions(stf_ewald_cpu_test PRIVATE ORDER=${COSMICTIGER_WITH_ORDER})
    target_link_libraries(stf_ewald_cpu_test PRIVATE HPX::hpx Boost::boost)
    add_test(NAME stf_ewald_cpu COMMAND stf_ewald_cpu_test)
    set_tests_properties(stf_ewald_cpu PROPERTIES TIMEOUT 30)
    add_test(NAME stf_generated_current
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_stf_m2l.py --check)
    set_tests_properties(stf_generated_current PROPERTIES TIMEOUT 120)

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
        add_executable(stf_kernels_cuda_test tests/stf_kernels_cuda.cu)
        target_include_directories(stf_kernels_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_link_libraries(stf_kernels_cuda_test PRIVATE CUDA::cudart)
        add_test(NAME stf_kernels_cuda COMMAND stf_kernels_cuda_test)
        set_tests_properties(stf_kernels_cuda PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)

        add_executable(stf_l2l_cuda_test tests/stf_l2l_cuda.cu)
        target_include_directories(stf_l2l_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(stf_l2l_cuda_test PRIVATE USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
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
            src/sph_prehydro.cu src/sph_hydro.cu src/kernel.cpp src/kernel.cu src/cuda_memory.cu src/assert.cu)
        target_include_directories(monaghan_sph_cuda_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_definitions(monaghan_sph_cuda_test PRIVATE
            USE_CUDA ORDER=${COSMICTIGER_WITH_ORDER})
        target_link_libraries(monaghan_sph_cuda_test PRIVATE CUDA::cudart HPX::hpx)
        target_compile_options(monaghan_sph_cuda_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math>)
        set_target_properties(monaghan_sph_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME monaghan_sph_cuda COMMAND monaghan_sph_cuda_test)
        set_tests_properties(monaghan_sph_cuda PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)

        # These host regressions need CUDA headers, but do not initialize or run a GPU.
        add_executable(numerical_utilities_test tests/numerical_utilities.cpp)
        add_executable(compression_test tests/compression.cpp src/compress.cpp)
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
    endif()
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
