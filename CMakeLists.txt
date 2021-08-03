cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(tigerfmm CXX C)
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


set(tigerfmm_source_files
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
                       
                        
                        

set(tigerfmm_header_files
    tigerfmm/complex.hpp
    tigerfmm/containers.hpp
    tigerfmm/cuda.hpp
    tigerfmm/cuda_reduce.hpp
    tigerfmm/cuda_vector.hpp
    tigerfmm/defs.hpp
    tigerfmm/domain.hpp
    tigerfmm/drift.hpp
    tigerfmm/driver.hpp
    tigerfmm/ewald_incides.hpp
    tigerfmm/fft.hpp
    tigerfmm/fixedcapvec.hpp
    tigerfmm/fixes.hpp
    tigerfmm/gravity.hpp
    tigerfmm/initialize.hpp
    tigerfmm/hpx.hpp
    tigerfmm/kick.hpp
    tigerfmm/kick_workspace.hpp
    tigerfmm/map.hpp
    tigerfmm/math.hpp
    tigerfmm/options.hpp
    tigerfmm/particles.hpp
    tigerfmm/range.hpp
    tigerfmm/range_set.hpp
    tigerfmm/safe_io.hpp
    tigerfmm/simd.hpp
    tigerfmm/stack_vector.hpp
    tigerfmm/test.hpp
    tigerfmm/tree.hpp
    tigerfmm/timer.hpp
    tigerfmm/unordered_map_ts.hpp
    tigerfmm/unordered_set_ts.hpp
)

add_hpx_executable(
  tigerfmm
  DEPENDENCIES
    Boost::boost cudart fftw3 gsl gslcblas chealpix
  SOURCES
    ${tigerfmm_source_files}
  HEADERS
    ${tigerfmm_header_files}
 )
 
 add_executable( fmm_kernel_codegen src/fmm_kernel_codegen.cpp tigerfmm/tensor.hpp )
 execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)
 
 
 add_custom_command(
    OUTPUT tigerfmm/fmm_kernels.hpp
    COMMAND ./fmm_kernel_codegen > ${PROJECT_SOURCE_DIR}/tigerfmm/fmm_kernels.hpp
    DEPENDS fmm_kernel_codegen
    COMMENT "Generating FFM kernels"
)
add_custom_target( generate_fmm_kernels DEPENDS tigerfmm/fmm_kernels.hpp COMMENT "" )
add_dependencies( tigerfmm generate_fmm_kernels )
 
set_property(TARGET tigerfmm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tigerfmm PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(fmm_kernel_codegen PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigerfmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math ;
                        >)
                        
set_property(TARGET tigerfmm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
                        

add_executable(
 	healpix2silo
 	src/healpix2silo.cpp
 )
 
target_link_libraries(healpix2silo PUBLIC chealpix siloh5 boost_program_options)
target_include_directories(healpix2silo PUBLIC ${PROJECT_SOURCE_DIR})


add_library(hpx_lite 
	hpx_lite/src/hpx.cpp
	hpx_lite/src/id_type.cpp
	hpx_lite/src/thread.cpp
	hpx_lite/src/future.cpp
	hpx_lite/src/mutex.cpp
	hpx_lite/src/serialize.cpp
)
#target_compile_options( hpx_lite PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
target_include_directories(hpx_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite  ${PROJECT_SOURCE_DIR}/hpxfft)

add_executable( tigerfmm_lite ${tigerfmm_source_files})
#target_compile_options( tigerfmm_lite PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0 -DHPX_LITE")
target_compile_options( tigerfmm_lite PUBLIC "-DHPX_LITE")
target_link_libraries(tigerfmm_lite PUBLIC boost_program_options cudart fftw3 gsl gslcblas chealpix hpx_lite tbbmalloc  hwloc)
set_property(TARGET tigerfmm_lite PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tigerfmm_lite PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigerfmm_lite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math ;
                        >)

set_property(TARGET tigerfmm_lite PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
                        