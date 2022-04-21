cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(cosmictiger CXX C Fortran)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
option(COSMICTIGER_WITH_CUDA "Enable CUDA" ON)
option(COSMICTIGER_WITH_LONG_LONG_PART_INT "Use long long to index local particles" OFF)
set(COSMICTIGER_WITH_ORDER "7" )
if( COSMICTIGER_WITH_CUDA ) 
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
endif()
message(STATUS "cosmictiger expansion order P+1 = ${COSMICTIGER_WITH_ORDER}")
set(CMAKE_CXX_FLAGS "-Wno-attributes --fast-math")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
CHECK_CXX_COMPILER_FLAG("-maltivec" COMPILER_SUPPORTS_MALTIVEC)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-pie -no-pie")
endif()
if(COMPILER_SUPPORTS_MALTIVEC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec")
endif()

execute_process(COMMAND g++ ${PROJECT_SOURCE_DIR}/src/simd_type.cpp -march=native -o simd_detect)
execute_process(COMMAND ./simd_detect OUTPUT_VARIABLE SIMD_ARCH)
message(STATUS "${SIMD_ARCH}")


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
    src/chemistry.cpp
    src/chemistry.cu
 	src/cosmology.cpp
    src/cuda.cpp
    src/cuda_mem.cu
    src/drift.cpp
    src/driver.cpp
    src/domain.cpp
    src/ewald_indices.cpp
    src/ewald_indices.cu
    src/fft.cpp
    src/gravity.cpp
    src/gravity.cu
    src/group_tree.cpp
    src/groups.cpp
    src/group_entry.cpp
    src/groups_find.cpp
    src/hydro.cpp
    src/initialize.cpp
    src/hpx.cpp
    src/kernel.cu
    src/kick.cpp
    src/kick.cu
    src/exact_sod.cpp
    src/kick_workspace.cpp
    src/lightcone.cpp
    src/main.cpp
    src/memused.cpp
    src/options.cpp
    src/output.cpp
    src/particles.cpp
    src/power.cpp
    src/profiler.cpp
    src/rockstar.cpp
    src/rockstar.cu
    src/sedov.cpp
    src/sph_particles.cpp
    src/sph.cpp
    src/sph.cu
    src/sph_aux.cu
    src/sph_cond_init.cu
    src/sph_smoothlen.cu
    src/sph_prehydro.cu
    src/sph_conduction.cu
    src/sph_hydro.cu
    src/sph_rungs.cu
    src/sphere.cu
    src/sph_tree.cpp
    src/stack_trace.cpp
    src/stars.cpp
    src/test.cpp
    src/tree.cpp
    src/view.cpp
    src/zero_order.cpp
)      
                       
                        
                        

set(cosmictiger_header_files
    cosmictiger/assert.hpp
    cosmictiger/bh.hpp
    cosmictiger/boltzmann.hpp
    cosmictiger/chemistry.hpp
    cosmictiger/complex.hpp
    cosmictiger/containers.hpp
    cosmictiger/cosmology.hpp
    cosmictiger/cuda.hpp
    cosmictiger/cuda_reduce.hpp
    cosmictiger/cuda_vector.hpp
    cosmictiger/defs.hpp
    cosmictiger/domain.hpp
    cosmictiger/drift.hpp
    cosmictiger/driver.hpp
    cosmictiger/ewald_incides.hpp
    cosmictiger/fft.hpp
    cosmictiger/fixedcapvec.hpp
    cosmictiger/fixes.hpp
    cosmictiger/gravity.hpp
    cosmictiger/initialize.hpp
    cosmictiger/interp.hpp
    cosmictiger/group_tree.hpp
    cosmictiger/groups.hpp
    cosmictiger/group_entry.hpp
    cosmictiger/groups_find.hpp
    cosmictiger/hpx.hpp
    cosmictiger/hydro.hpp
    cosmictiger/kernel.hpp
    cosmictiger/kick.hpp
    cosmictiger/kick_workspace.hpp
    cosmictiger/lc_group_archive.hpp
    cosmictiger/exact_sod.hpp
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
    cosmictiger/sph.hpp
    cosmictiger/sph_cuda.hpp
    cosmictiger/sph_particles.hpp
    cosmictiger/sph_tree.hpp
    cosmictiger/stack_vector.hpp
    cosmictiger/stack_trace.hpp
    cosmictiger/stars.hpp
    cosmictiger/test.hpp
    cosmictiger/tree.hpp
    cosmictiger/timer.hpp
    cosmictiger/unordered_map_ts.hpp
    cosmictiger/unordered_set_ts.hpp
    cosmictiger/view.hpp
    cosmictiger/zero_order.hpp
)

add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost fftw3f gsl gslcblas dl silo chealpix quadmath
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 


 
add_executable( recfast recfast/recfast.f )
add_executable( fmm_kernel_codegen src/fmm_kernel_codegen.cpp  cosmictiger/tensor.hpp )
target_link_libraries( fmm_kernel_codegen PUBLIC dl )
execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)
 
set_source_files_properties(./src/gravity.cu PROPERTIES COMPILE_FLAGS -maxrregcount=160)
set_source_files_properties(./src/kick.cu PROPERTIES COMPILE_FLAGS -maxrregcount=160)

set_source_files_properties(./src/cuda_mem.cu PROPERTIES COMPILE_FLAGS -maxrregcount=40)
set_source_files_properties(./src/chemistry.cu PROPERTIES COMPILE_FLAGS -maxrregcount=40)


set_source_files_properties(./src/sph.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_smoothlen.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_prehydro.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_aux.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_cond_init.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_hydro.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_conduction.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)
set_source_files_properties(./src/sph_rungs.cu PROPERTIES COMPILE_FLAGS -maxrregcount=128)

if( COSMICTIGER_WITH_CUDA )
	target_link_libraries(cosmictiger PUBLIC cudart)
	target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -DUSE_CUDA --use_fast_math ;
                        >)
    target_compile_options(cosmictiger PRIVATE -DUSE_CUDA -DBOOST_STACKTRACE_USE_ADDR2LINE )
	target_compile_options(fmm_kernel_codegen PRIVATE -DUSE_CUDA )
	set_property(TARGET cosmictiger PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
	set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
 
 
 
add_custom_command(
    OUTPUT cosmictiger/fmm_kernels.hpp
    COMMAND ./fmm_kernel_codegen > ${PROJECT_SOURCE_DIR}/cosmictiger/fmm_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "generating cosmictiger/fmm_kernels.hpp"
)
add_custom_target( generate_fmm_kernels DEPENDS cosmictiger/fmm_kernels.hpp COMMENT "" )
#add_dependencies( cosmictiger generate_fmm_kernels )
 
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(fmm_kernel_codegen PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(cosmictiger PUBLIC ORDER=${COSMICTIGER_WITH_ORDER})
if( COSMICTIGER_WITH_LONG_LONG_PART_INT ) 
	target_compile_definitions(cosmictiger PUBLIC LONG_LONG_PART_INT)
endif()
target_compile_definitions(fmm_kernel_codegen PUBLIC ORDER=${COSMICTIGER_WITH_ORDER})

add_executable(
 	healpix2silo
 	src/healpix2silo.cpp
 )
 
 
 
 
target_link_libraries(healpix2silo PUBLIC chealpix silo boost_program_options dl)
target_include_directories(healpix2silo PUBLIC ${PROJECT_SOURCE_DIR})



add_hpx_executable(
 	groups2halofunc
   SOURCES
 	src/assert.cpp
 	src/hpx.cpp
 	src/groups2halofunc.cpp
 	src/group_entry.cpp
 )

target_link_libraries(groups2halofunc PUBLIC boost_program_options)
target_include_directories(groups2halofunc PUBLIC ${PROJECT_SOURCE_DIR})
