cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(cosmictiger CXX C Fortran CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
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
option(BUILD_SHARED_LIBS "" OFF)
execute_process(COMMAND g++ ${PROJECT_SOURCE_DIR}/src/simd_type.cpp -march=native -o simd_detect)
execute_process(COMMAND ./simd_detect OUTPUT_VARIABLE SIMD_ARCH)
message(STATUS "${SIMD_ARCH}")

set(host_files
    healpix/error_handling.cc
    healpix/geom_utils.cc
    healpix/healpix_base.cc
    healpix/healpix_tables.cc
    healpix/pointing.cc
    healpix/string_utils.cc
    src/analytic.cpp
    src/assert.cpp
    src/bh.cpp
    src/boltzmann.cpp
    src/chemistry.cpp
    src/cosmology.cpp
    src/cuda.cpp
    src/drift.cpp
    src/driver.cpp
    src/domain.cpp
    src/ewald_indices.cpp
    src/fft.cpp
    src/gravity.cpp
    src/group_tree.cpp
    src/groups.cpp
    src/group_entry.cpp
    src/groups_find.cpp
    src/hydro.cpp
    src/initialize.cpp
    src/hpx.cpp
    src/kick.cpp
    src/exact_sod.cpp
    src/kick_workspace.cpp
    src/lightcone.cpp
    src/memused.cpp
    src/options.cpp
    src/output.cpp
    src/particles.cpp
    src/power.cpp
    src/profiler.cpp
    src/rockstar.cpp
    src/sedov.cpp
    src/sph_particles.cpp
    src/sph.cpp
    src/sph_tree.cpp
    src/stack_trace.cpp
    src/stars.cpp
    src/test.cpp
    src/tree.cpp
    src/view.cpp
    src/zero_order.cpp
)      
set(device_files
    src/analytic.cu
    src/assert.cu
    src/bh.cu
    src/chemistry.cu
    src/cuda_mem.cu
    src/ewald_indices.cu
    src/gravity.cu
    src/kernel.cu
    src/kick.cu
    src/rockstar.cu
    src/sph.cu
    src/sph_aux.cu
    src/sph_cond_init.cu
    src/sph_prehydro.cu
    src/sph_conduction.cu
    src/sph_hydro.cu
    src/sph_rungs.cu
    src/sphere.cu
)      
                  
set(header_files
    cosmictiger/assert.hpp
    cosmictiger/bh.hpp
    cosmictiger/boltzmann.hpp
    cosmictiger/chemistry.hpp
    cosmictiger/complex.hpp
    cosmictiger/containers.hpp
    cosmictiger/cosmology.hpp
    cosmictiger/cuda.hpp
    cosmictiger/cuda_reduce.hpp
    cosmictiger/defs.hpp
    cosmictiger/domain.hpp
    cosmictiger/drift.hpp
    cosmictiger/driver.hpp
    cosmictiger/fft.hpp
    cosmictiger/fixedcapvec.hpp
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
    cosmictiger/unordered_set_ts.hpp
    cosmictiger/view.hpp
    cosmictiger/zero_order.hpp
)

cuda_add_library(
	device 
	STATIC
	${device_files} ${header_files}
	OPTIONS
		-dc --diag-suppress=177 --diag-suppress=550 --diag-suppress=20054 -I${PROJECT_SOURCE_DIR} -DORDER=${COSMICTIGER_WITH_ORDER} -DUSE_CUDA
)
target_link_libraries(device cudart)
set_property(TARGET device PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET device PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

add_hpx_library(
	host
	STATIC
    DEPENDENCIES
    	siloh5 simd gsl fftw3f quadmath chealpix
    SOURCES
		${host_files}
    HEADERS
    	${header_files}
)
target_compile_options(host PUBLIC -I${PROJECT_SOURCE_DIR} -DORDER=${COSMICTIGER_WITH_ORDER} -DUSE_CUDA -march=native -Wno-attributes --fast-math )
set_property(TARGET host PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET host PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

                                                    
add_hpx_executable(
  cosmictiger
  STATIC
  DEPENDENCIES
    host device siloh5 simd gsl fftw3f quadmath chealpix
  SOURCES
    src/main.cpp
  HEADERS
    ${header_files}
)
target_compile_options(cosmictiger PUBLIC -I${PROJECT_SOURCE_DIR} -DORDER=${COSMICTIGER_WITH_ORDER} -DUSE_CUDA -march=native -Wno-attributes --fast-math )
set_property(TARGET cosmictiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cosmictiger PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_include_directories(cosmictiger PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(cosmictiger PUBLIC ORDER=${COSMICTIGER_WITH_ORDER})
set_property(TARGET cosmictiger PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


add_executable(
 	healpix2silo
 	src/healpix2silo.cpp
)
target_link_libraries(healpix2silo chealpix siloh5 boost_program_options dl)
target_include_directories(healpix2silo PUBLIC ${PROJECT_SOURCE_DIR})


add_hpx_executable(
 	groups2halofunc
 	DEPENDENCIES
 		boost_program_options
    SOURCES
 		src/assert.cpp
 		src/hpx.cpp
 		src/groups2halofunc.cpp
 		src/group_entry.cpp
)
target_include_directories(groups2halofunc PUBLIC ${PROJECT_SOURCE_DIR})

