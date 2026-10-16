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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(COSMICTIGER_WITH_CUDA "Enable CUDA" ON)
option(COSMICTIGER_WITH_LONG_LONG_PART_INT "Use long long to index local particles" OFF)
set(COSMICTIGER_WITH_ORDER "7" CACHE STRING "Expansion order P+1")
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
    src/cuda_mem.cu
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
    src/chemistry.cpp
    src/chemistry.cu
    src/hydro.cpp
    src/kernel.cu
    src/exact_sod.cpp
    src/sedov.cpp
    src/sph_particles.cpp
    src/sph.cpp
    src/sph.cu
    src/sph_aux.cu
    src/sph_cond_init.cu
    src/sph_prehydro.cu
    src/sph_conduction.cu
    src/sph_hydro.cu
    src/sph_rungs.cu
    src/sphere.cu
    src/sph_tree.cpp
    src/stars.cpp
    tests/sph_gravity.cpp
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
    cosmictiger/fixedcapvec.hpp
    cosmictiger/gravity.hpp
    cosmictiger/initialize.hpp
    cosmictiger/interp.hpp
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
    cosmictiger/output.hpp
    cosmictiger/particles.hpp
    cosmictiger/power.hpp
    cosmictiger/profiler.hpp
    cosmictiger/rand.hpp
    cosmictiger/range.hpp
    cosmictiger/rockstar.hpp
    cosmictiger/safe_io.hpp
    cosmictiger/sedov.hpp
    cosmictiger/simd.hpp
    cosmictiger/stack_vector.hpp
    cosmictiger/stack_trace.hpp
    cosmictiger/test.hpp
    cosmictiger/tree.hpp
    cosmictiger/timer.hpp
    cosmictiger/unordered_set_ts.hpp
    cosmictiger/view.hpp
    cosmictiger/zero_order.hpp
    cosmictiger/chemistry.hpp
    cosmictiger/hydro.hpp
    cosmictiger/kernel.hpp
    cosmictiger/exact_sod.hpp
    cosmictiger/sph.hpp
    cosmictiger/sph_cuda.hpp
    cosmictiger/sph_particles.hpp
    cosmictiger/sph_tree.hpp
    cosmictiger/sphere.hpp
    cosmictiger/stars.hpp
)

add_hpx_executable(cosmictiger
    DEPENDENCIES Boost::boost Boost::program_options GSL::gsl GSL::gslcblas
        ${FFTW3F_LIBRARY} ${SILO_LIBRARY} ${CHEALPIX_LIBRARY} dl quadmath
    SOURCES ${cosmictiger_source_files}
    HEADERS ${cosmictiger_header_files})
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
    set_source_files_properties(src/chemistry.cu PROPERTIES COMPILE_OPTIONS -maxrregcount=40)
    set_source_files_properties(src/sph.cu src/sph_prehydro.cu src/sph_aux.cu
        src/sph_cond_init.cu src/sph_hydro.cu src/sph_conduction.cu src/sph_rungs.cu
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
    OUTPUT ${PROJECT_BINARY_DIR}/cosmictiger/fmm_kernels.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/cosmictiger
    COMMAND fmm_kernel_codegen > ${PROJECT_BINARY_DIR}/cosmictiger/fmm_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "Generating FMM kernels for comparison with the checked-in header")
add_custom_target(generate_fmm_kernels DEPENDS ${PROJECT_BINARY_DIR}/cosmictiger/fmm_kernels.hpp)

add_executable(lc2rockstar src/lc2rockstar.cpp src/compress.cpp)
target_compile_definitions(lc2rockstar PRIVATE LC2GADGET2)
target_include_directories(lc2rockstar PRIVATE ${PROJECT_SOURCE_DIR})
if(COSMICTIGER_WITH_CUDA)
    target_link_libraries(lc2rockstar PRIVATE CUDA::cudart)
endif()

add_executable(healpix2silo src/healpix2silo.cpp)
target_link_libraries(healpix2silo PRIVATE ${CHEALPIX_LIBRARY} ${SILO_LIBRARY} Boost::program_options dl)
target_include_directories(healpix2silo PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(healpix2silo SYSTEM PRIVATE ${CHEALPIX_INCLUDE_DIR} ${SILO_INCLUDE_DIR})

add_hpx_executable(groups2halofunc
    SOURCES src/assert.cpp src/hpx.cpp src/groups2halofunc.cpp src/group_entry.cpp)
target_link_libraries(groups2halofunc PUBLIC Boost::program_options)
target_include_directories(groups2halofunc PRIVATE ${PROJECT_SOURCE_DIR})

foreach(tool_target IN ITEMS fmm_kernel_codegen lc2rockstar healpix2silo groups2halofunc)
    target_compile_options(${tool_target} PRIVATE -Wno-attributes -ffast-math)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(${tool_target} PRIVATE -march=native)
    endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
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
            --test=${test_mode} --parts_dim=8 --hydro.sph=${test_sph}
            --cuda=${test_cuda} --hpx:threads=2 --hpx:bind=none)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${test_workdir}" RUN_SERIAL TRUE TIMEOUT 120)
    endforeach()
    set(rejected_options_workdir "${PROJECT_BINARY_DIR}/tests/rejected_options")
    file(MAKE_DIRECTORY "${rejected_options_workdir}")
    add_test(NAME rejected_options COMMAND cosmictiger --invalid_option_for_test=true
        --hpx:threads=2 --hpx:bind=none)
    set_tests_properties(rejected_options PROPERTIES WORKING_DIRECTORY "${rejected_options_workdir}"
        RUN_SERIAL TRUE TIMEOUT 20 WILL_FAIL TRUE)
    if(HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE)
        set(migration_workdir "${PROJECT_BINARY_DIR}/tests/sph_migration_mpi")
        file(MAKE_DIRECTORY "${migration_workdir}")
        add_test(NAME sph_migration_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cosmictiger> ${MPIEXEC_POSTFLAGS}
            --test=sphGravity --parts_dim=8 --hydro.sph=true --cuda=false
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
endif()
