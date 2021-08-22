cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(cosmictiger CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
option(COSMICTIGER_WITH_CUDA "Enable CUDA" ON)
set(COSMICTIGER_WITH_ORDER "7" )
if( COSMICTIGER_WITH_CUDA ) 
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
endif()
message(STATUS "cosmictiger expansion order P+1 = ${COSMICTIGER_WITH_ORDER}")
set(CMAKE_CXX_FLAGS "-ffast-math -march=native  -no-pie -fno-pie -Wno-attributes")


set(cosmictiger_source_files
    src/analytic.cu
    src/analytic.cpp
    src/assert.cpp
    src/assert.cu
    src/bh.cpp
 	src/cosmology.cpp
    src/cuda.cpp
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
    src/initialize.cpp
    src/hpx.cpp
    src/kick.cpp
    src/kick.cu
    src/kick_workspace.cpp
    src/main.cpp
    src/map.cpp
    src/options.cpp
    src/output.cpp
    src/particles.cpp
    src/power.cpp
    src/range_set.cpp
    src/rockstar.cpp
    src/test.cpp
    src/tree.cpp
    src/view.cpp
)      
                       
                        
                        

set(cosmictiger_header_files
    cosmictiger/assert.hpp
    cosmictiger/bh.hpp
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
    cosmictiger/group_tree.hpp
    cosmictiger/groups.hpp
    cosmictiger/group_entry.hpp
    cosmictiger/groups_find.hpp
    cosmictiger/hpx.hpp
    cosmictiger/kick.hpp
    cosmictiger/kick_workspace.hpp
    cosmictiger/map.hpp
    cosmictiger/math.hpp
    cosmictiger/options.hpp
    cosmictiger/output.hpp
    cosmictiger/particles.hpp
    cosmictiger/power.hpp
    cosmictiger/range.hpp
    cosmictiger/range_set.hpp
    cosmictiger/rockstar.hpp
    cosmictiger/safe_io.hpp
    cosmictiger/simd.hpp
    cosmictiger/stack_vector.hpp
    cosmictiger/test.hpp
    cosmictiger/tree.hpp
    cosmictiger/timer.hpp
    cosmictiger/unordered_map_ts.hpp
    cosmictiger/unordered_set_ts.hpp
    cosmictiger/view.hpp
)

add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost fftw3f gsl gslcblas chealpix dl silo
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 
 add_executable( fmm_kernel_codegen src/fmm_kernel_codegen.cpp  cosmictiger/tensor.hpp )
 target_link_libraries( fmm_kernel_codegen PUBLIC dl )
 execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)
 
if( COSMICTIGER_WITH_CUDA )
	target_link_libraries(cosmictiger PUBLIC cudart)
	target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math --maxrregcount=160;
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
add_dependencies( cosmictiger generate_fmm_kernels )
 
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(fmm_kernel_codegen PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(cosmictiger PUBLIC ORDER=${COSMICTIGER_WITH_ORDER})
target_compile_definitions(fmm_kernel_codegen PUBLIC ORDER=${COSMICTIGER_WITH_ORDER})

add_executable(
 	healpix2silo
 	src/healpix2silo.cpp
 )
 
 
target_link_libraries(healpix2silo PUBLIC chealpix silo boost_program_options dl)
target_include_directories(healpix2silo PUBLIC ${PROJECT_SOURCE_DIR})



add_hpx_executable(
 	groups2silo
   SOURCES
 	src/assert.cpp
 	src/hpx.cpp
 	src/groups2silo.cpp
 	src/group_entry.cpp
 )

target_link_libraries(groups2silo PUBLIC silo boost_program_options)
target_include_directories(groups2silo PUBLIC ${PROJECT_SOURCE_DIR})
