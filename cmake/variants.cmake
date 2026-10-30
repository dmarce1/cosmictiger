# Build one application with the configured capabilities.
add_library(cosmictiger_platform INTERFACE)
target_include_directories(cosmictiger_platform INTERFACE ${PROJECT_SOURCE_DIR})
target_include_directories(cosmictiger_platform SYSTEM INTERFACE
    ${FFTW3F_INCLUDE_DIR} ${FFTW3F_MPI_INCLUDE_DIR}
    ${SILO_INCLUDE_DIR} ${CHEALPIX_INCLUDE_DIR} ${HEALPIX_CXX_INCLUDE_DIR})
target_link_libraries(cosmictiger_platform INTERFACE
    HPX::hpx Boost::boost Boost::program_options GSL::gsl GSL::gslcblas
    ${FFTW3F_MPI_LIBRARY} ${FFTW3F_THREADS_LIBRARY} ${FFTW3F_LIBRARY}
    MPI::MPI_CXX Threads::Threads
    ${SILO_LIBRARY} ${CHEALPIX_LIBRARY} ${HEALPIX_CXX_LIBRARY} dl quadmath)
target_compile_options(cosmictiger_platform INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes;-ffast-math>)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(cosmictiger_platform INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
if(COSMICTIGER_WITH_LONG_LONG_PART_INT)
    target_compile_definitions(cosmictiger_platform INTERFACE LONG_LONG_PART_INT)
endif()
if(COSMICTIGER_WITH_CUDA)
    target_link_libraries(cosmictiger_platform INTERFACE CUDA::cudart)
    target_compile_definitions(cosmictiger_platform INTERFACE USE_CUDA)
    target_compile_options(cosmictiger_platform INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)
    set_source_files_properties(src/particles/drift.cu PROPERTIES COMPILE_OPTIONS -maxrregcount=64)
    set_source_files_properties(src/sph/sph.cu src/sph/sph_prehydro.cu
        src/sph/sph_hydro.cu src/sph/sph_rungs.cu PROPERTIES COMPILE_OPTIONS -maxrregcount=128)
endif()

add_library(cosmictiger_core STATIC ${cosmictiger_core_source_files})
target_link_libraries(cosmictiger_core PUBLIC cosmictiger_platform cosmictiger_recombination
    cosmictiger_fp16 cosmictiger_profiler cosmictiger_stacktrace)
if(COSMICTIGER_WITH_CUDA)
    # Resolve device symbols when linking the final executable, alongside its
    # capability-dependent CUDA objects.
    set_target_properties(cosmictiger_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

# Capability definitions are shared by the executable and standalone tests.
add_library(cosmictiger_default_config INTERFACE)
foreach(feature IN ITEMS FMM PMMM_PP PMMM_FMM SPH CHEMISTRY
        INTEGRATOR_TRADITIONAL INTEGRATOR_HIERARCHICAL INTEGRATOR_GLOBAL)
    target_compile_definitions(cosmictiger_default_config INTERFACE
        COSMICTIGER_ENABLE_${feature}=$<BOOL:${COSMICTIGER_ENABLE_${feature}}>)
endforeach()
target_compile_definitions(cosmictiger_default_config INTERFACE
    COSMICTIGER_WITH_GRAVITY=$<BOOL:${COSMICTIGER_WITH_GRAVITY}>
    COSMICTIGER_WITH_SPH=$<BOOL:${COSMICTIGER_WITH_SPH}>
    COSMICTIGER_POSITION_BITS=${COSMICTIGER_POSITION_BITS}
    ORDER=${cosmictiger_default_fmm_order}
    FMM_MIN_ORDER=${COSMICTIGER_FMM_MIN_ORDER}
    FMM_MAX_ORDER=${COSMICTIGER_FMM_MAX_ORDER}
    MAX_BUCKET_SIZE=${COSMICTIGER_WITH_MAX_BUCKET})

cosmictiger_feature_sources(cosmictiger_enabled_sources)
if(COSMICTIGER_ENABLE_FMM OR
        COSMICTIGER_ENABLE_PMMM_FMM)
    foreach(order IN LISTS COSMICTIGER_FMM_ORDERS)
        set(FMM_INSTANCE_ORDER ${order})
        configure_file(src/gravity/fmm_instance_cpu.cpp.in
            ${PROJECT_BINARY_DIR}/instances/fmm_instance_cpu_${order}.cpp @ONLY)
        list(APPEND cosmictiger_enabled_sources
            ${PROJECT_BINARY_DIR}/instances/fmm_instance_cpu_${order}.cpp)
        if(COSMICTIGER_WITH_CUDA)
            configure_file(src/gravity/fmm_instance_cuda.cu.in
                ${PROJECT_BINARY_DIR}/instances/fmm_instance_cuda_${order}.cu @ONLY)
            list(APPEND cosmictiger_enabled_sources
                ${PROJECT_BINARY_DIR}/instances/fmm_instance_cuda_${order}.cu)
        endif()
    endforeach()
endif()
add_library(cosmictiger_objects OBJECT ${cosmictiger_enabled_sources}
    ${cosmictiger_header_files} ${COSMICTIGER_GENERATED_FMM_FILES})
target_link_libraries(cosmictiger_objects PUBLIC cosmictiger_core cosmictiger_default_config)
if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_CHEMISTRY)
    target_link_libraries(cosmictiger_objects PUBLIC cosmictiger_sph_chemistry)
endif()
if(COSMICTIGER_WITH_CUDA)
    set_target_properties(cosmictiger_objects PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
add_hpx_executable(cosmictiger SOURCES src/main.cpp)
target_link_libraries(cosmictiger PRIVATE cosmictiger_objects cosmictiger_core)
set_target_properties(cosmictiger PROPERTIES ENABLE_EXPORTS ON)
if(COSMICTIGER_WITH_CUDA)
    set_target_properties(cosmictiger PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
set(cosmictiger_application_targets cosmictiger)
configure_file(cmake/variant_help.txt.in ${PROJECT_BINARY_DIR}/cosmictiger_build_help.txt @ONLY)
add_custom_target(cosmictiger_help
    COMMAND ${CMAKE_COMMAND} -E cat ${PROJECT_BINARY_DIR}/cosmictiger_build_help.txt
    VERBATIM)
