cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(tigerfmm CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-ffast-math -march=native -Wno-deprecated-declarations -Wno-format-security")
execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")


set(tigerfmm_source_files
    src/domain.cpp
    src/hpx.cpp
    src/main.cpp
    src/options.cpp
    src/particles.cpp
    src/test.cpp
)      
                       
                        
                        

set(tigerfmm_header_files
    tigerfmm/containers.hpp
    tigerfmm/cuda.hpp
    tigerfmm/defs.hpp
    tigerfmm/domain.hpp
    tigerfmm/fixes.hpp
    tigerfmm/hpx.hpp
    tigerfmm/options.hpp
    tigerfmm/particles.hpp
    tigerfmm/range.hpp
    tigerfmm/safe_io.hpp
    tigerfmm/test.hpp
    tigerfmm/timer.hpp
)



add_hpx_executable(
  tigerfmm
  DEPENDENCIES
    Boost::boost cudart fftw3 gsl gslcblas
  SOURCES
    ${tigerfmm_source_files}
  HEADERS
    ${tigerfmm_header_files}
 )

set_property(TARGET tigerfmm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tigerfmm PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigerfmm PUBLIC "-DUSE_HPX")
target_compile_options(tigerfmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math ;
                        >)
                        
set_property(TARGET tigerfmm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

