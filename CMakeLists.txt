cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(cosmictiger CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
enable_language(CUDA)
set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
set(CMAKE_CXX_FLAGS "-ffast-math -Wno-deprecated-declarations -Wno-format-security")
execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
set(CMAKE_CXX_FLAGS "-ffast-math -march=native")


set(cosmictiger_source_files
    src/analytic.cu
    src/analytic.cpp
    src/cuda.cpp
    src/drift.cpp
    src/driver.cpp
    src/domain.cpp
    src/ewald_indices.cpp
    src/ewald_indices.cu
    src/fft.cpp
    src/gravity.cpp
    src/gravity.cu
    src/initialize.cpp
    src/hpx.cpp
    src/kick.cpp
    src/kick.cu
    src/kick_workspace.cpp
    src/main.cpp
    src/map.cpp
    src/options.cpp
    src/particles.cpp
    src/range_set.cpp
    src/test.cpp
    src/tree.cpp
)      
                       
                        
                        

set(cosmictiger_header_files
    cosmictiger/complex.hpp
    cosmictiger/containers.hpp
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
    cosmictiger/hpx.hpp
    cosmictiger/kick.hpp
    cosmictiger/kick_workspace.hpp
    cosmictiger/map.hpp
    cosmictiger/math.hpp
    cosmictiger/options.hpp
    cosmictiger/particles.hpp
    cosmictiger/range.hpp
    cosmictiger/range_set.hpp
    cosmictiger/safe_io.hpp
    cosmictiger/simd.hpp
    cosmictiger/stack_vector.hpp
    cosmictiger/test.hpp
    cosmictiger/tree.hpp
    cosmictiger/timer.hpp
    cosmictiger/unordered_map_ts.hpp
    cosmictiger/unordered_set_ts.hpp
)

add_hpx_executable(
  cosmictiger
  DEPENDENCIES
    Boost::boost cudart fftw3f gsl gslcblas chealpix
  SOURCES
    ${cosmictiger_source_files}
  HEADERS
    ${cosmictiger_header_files}
 )
 
 add_executable( fmm_kernel_codegen src/fmm_kernel_codegen.cpp cosmictiger/tensor.hpp )
 execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)
 
 
 add_custom_command(
    OUTPUT cosmictiger/fmm_kernels.hpp
    COMMAND ./fmm_kernel_codegen > ${PROJECT_SOURCE_DIR}/cosmictiger/fmm_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "Generating FFM kernels"
)
add_custom_target( generate_fmm_kernels DEPENDS cosmictiger/fmm_kernels.hpp COMMENT "" )
add_dependencies( cosmictiger generate_fmm_kernels )
 
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(fmm_kernel_codegen PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(cosmictiger PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math ;
                        >)
                        
set_property(TARGET cosmictiger PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


add_executable(
 	healpix2silo
 	src/healpix2silo.cpp
 )
 
target_link_libraries(healpix2silo PUBLIC chealpix siloh5 boost_program_options)
target_include_directories(healpix2silo PUBLIC ${PROJECT_SOURCE_DIR})

