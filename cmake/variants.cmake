# One build graph contains every supported physics configuration. Platform,
# compiler, CUDA, and build type remain settings of the complete build.
if(COSMICTIGER_WITH_SPH AND NOT COSMICTIGER_TIME_INTEGRATOR STREQUAL "leapfrog")
    message(FATAL_ERROR "SPH requires leapfrog. Select that default or "
        "set COSMICTIGER_WITH_SPH=OFF; all supported named targets remain available.")
endif()

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
    # variant-specific CUDA objects.
    set_target_properties(cosmictiger_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

function(cosmictiger_variant_config target integrator gravity near sph order)
    add_library(${target} INTERFACE)
    target_compile_definitions(${target} INTERFACE
        COSMICTIGER_INTEGRATOR_HIERARCHICAL=$<STREQUAL:${integrator},hierarchical>
        COSMICTIGER_GRAVITY_PMMM=$<STREQUAL:${gravity},pmmm>
        COSMICTIGER_PMMM_NEAR_FMM=$<STREQUAL:${near},fmm>
        COSMICTIGER_WITH_SPH=$<BOOL:${sph}>
        COSMICTIGER_POSITION_BITS=${COSMICTIGER_POSITION_BITS}
        ORDER=${order} FMM_MIN_ORDER=${order} FMM_MAX_ORDER=${order}
        MAX_BUCKET_SIZE=${COSMICTIGER_WITH_MAX_BUCKET})
endfunction()

function(cosmictiger_variant_executable target objects)
    cmake_parse_arguments(application "DEFAULT" "" "" ${ARGN})
    if(application_DEFAULT)
        add_hpx_executable(${target} SOURCES src/main.cpp)
    else()
        add_hpx_executable(${target} EXCLUDE_FROM_ALL SOURCES src/main.cpp)
    endif()
    target_link_libraries(${target} PRIVATE ${objects} cosmictiger_core)
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON)
    if(COSMICTIGER_WITH_CUDA)
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(BUILD_TESTING)
        target_link_libraries(${target} PRIVATE cosmictiger_application_tests)
    endif()
endfunction()

function(cosmictiger_add_variant integrator gravity near sph order)
    set(solver ${gravity})
    if(gravity STREQUAL "pmmm")
        string(APPEND solver "_${near}")
    endif()
    if(sph)
        set(hydro _sph)
    else()
        set(hydro "")
    endif()
    set(base cosmictiger_${integrator}_${solver}${hydro}_${order})
    set_property(GLOBAL APPEND PROPERTY COSMICTIGER_VARIANT_TARGETS ${base})
    cosmictiger_variant_config(${base}_config ${integrator} ${gravity} ${near} ${sph} ${order})
    cosmictiger_variant_sources(sources ${gravity} ${near} ${sph})
    if(gravity STREQUAL "fmm" OR near STREQUAL "fmm")
        list(APPEND sources ${PROJECT_BINARY_DIR}/instances/fmm_instance_cpu_${order}.cpp)
        if(COSMICTIGER_WITH_CUDA)
            list(APPEND sources ${PROJECT_BINARY_DIR}/instances/fmm_instance_cuda_${order}.cu)
        endif()
    endif()
    set(objects ${base}_objects)
    add_library(${objects} OBJECT EXCLUDE_FROM_ALL ${sources} ${cosmictiger_header_files}
        ${COSMICTIGER_GENERATED_FMM_FILES})
    target_link_libraries(${objects} PUBLIC cosmictiger_core ${base}_config)
    if(sph)
        target_link_libraries(${objects} PUBLIC cosmictiger_sph_chemistry)
    endif()
    if(COSMICTIGER_WITH_CUDA)
        set_target_properties(${objects} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    cosmictiger_variant_executable(${base} ${objects})
endfunction()

# The generated source is order-specific, but its objects also depend on the
# solver and SPH. Each object library compiles its own instance with the build's
# selected position storage.
foreach(order RANGE 3 9)
    set(FMM_INSTANCE_ORDER ${order})
    configure_file(src/gravity/fmm_instance_cpu.cpp.in
        ${PROJECT_BINARY_DIR}/instances/fmm_instance_cpu_${order}.cpp @ONLY)
    if(COSMICTIGER_WITH_CUDA)
        configure_file(src/gravity/fmm_instance_cuda.cu.in
            ${PROJECT_BINARY_DIR}/instances/fmm_instance_cuda_${order}.cu @ONLY)
    endif()
    foreach(integrator IN ITEMS leapfrog hierarchical)
        cosmictiger_add_variant(${integrator} fmm pp OFF ${order})
        cosmictiger_add_variant(${integrator} pmmm pp OFF ${order})
        cosmictiger_add_variant(${integrator} pmmm fmm OFF ${order})
    endforeach()
    cosmictiger_add_variant(leapfrog fmm pp ON ${order})
    cosmictiger_add_variant(leapfrog pmmm pp ON ${order})
    cosmictiger_add_variant(leapfrog pmmm fmm ON ${order})
endforeach()

# The legacy entry point and tests select one default from the same object graph.
set(cosmictiger_default_solver ${COSMICTIGER_GRAVITY_METHOD})
if(COSMICTIGER_GRAVITY_METHOD STREQUAL "pmmm")
    string(APPEND cosmictiger_default_solver "_${COSMICTIGER_PMMM_NEAR_METHOD}")
endif()
set(cosmictiger_default_hydro "")
if(COSMICTIGER_WITH_SPH)
    set(cosmictiger_default_hydro _sph)
endif()
set(cosmictiger_default_variant
    cosmictiger_${COSMICTIGER_TIME_INTEGRATOR}_${cosmictiger_default_solver}${cosmictiger_default_hydro}_${COSMICTIGER_WITH_ORDER})
add_library(cosmictiger_default_config INTERFACE)
target_link_libraries(cosmictiger_default_config INTERFACE ${cosmictiger_default_variant}_config)
set(cosmictiger_application_targets cosmictiger)
cosmictiger_variant_executable(cosmictiger ${cosmictiger_default_variant}_objects DEFAULT)

# Derive the guide from the targets actually registered above, so the supported
# families and counts stay in sync with the build graph.
get_property(cosmictiger_variant_targets GLOBAL PROPERTY COSMICTIGER_VARIANT_TARGETS)
list(LENGTH cosmictiger_variant_targets cosmictiger_variant_count)
set(cosmictiger_variant_families)
set(cosmictiger_variant_orders)
set(cosmictiger_sph_variant_count 0)
foreach(variant IN LISTS cosmictiger_variant_targets)
    string(REGEX REPLACE "_[0-9]+$" "_<order>" family "${variant}")
    string(REGEX REPLACE "^.*_([0-9]+)$" "\\1" order "${variant}")
    list(APPEND cosmictiger_variant_families "${family}")
    list(APPEND cosmictiger_variant_orders "${order}")
    if(variant MATCHES "_sph_")
        math(EXPR cosmictiger_sph_variant_count "${cosmictiger_sph_variant_count} + 1")
    endif()
endforeach()
math(EXPR cosmictiger_nosph_variant_count
    "${cosmictiger_variant_count} - ${cosmictiger_sph_variant_count}")
list(REMOVE_DUPLICATES cosmictiger_variant_families)
list(REMOVE_DUPLICATES cosmictiger_variant_orders)
list(LENGTH cosmictiger_variant_orders cosmictiger_variant_order_count)
list(JOIN cosmictiger_variant_orders ", " cosmictiger_variant_order_help)
set(cosmictiger_variant_family_help "")
foreach(family IN LISTS cosmictiger_variant_families)
    string(APPEND cosmictiger_variant_family_help "  ${family}\n")
endforeach()
configure_file(cmake/variant_help.txt.in ${PROJECT_BINARY_DIR}/cosmictiger_build_help.txt @ONLY)
# CMake reserves the name 'help' for its generic target listing.
add_custom_target(cosmictiger_help
    COMMAND ${CMAKE_COMMAND} -E cat ${PROJECT_BINARY_DIR}/cosmictiger_build_help.txt
    VERBATIM)
message(STATUS "cosmictiger default target = ${cosmictiger_default_variant}")
message(STATUS "cosmictiger named variants = ${cosmictiger_variant_count} (make cosmictiger_help for selection help)")
