# Geogram's BSD-licensed standalone Delaunay module. Pin the source and verify
# every downloaded file; an offline checkout may be supplied with this option.
set(COSMICTIGER_GEOGRAM_SOURCE_DIR "" CACHE PATH "Directory containing pinned Geogram Delaunay PSM files")
set(_cosmictiger_geogram_revision 9d4b56b9c1b0ae419baf4ec1425cfa6da769cd31)
if(NOT COSMICTIGER_GEOGRAM_SOURCE_DIR)
    set(_cosmictiger_geogram_dir "${CMAKE_BINARY_DIR}/_deps/geogram_psm")
else()
    set(_cosmictiger_geogram_dir "${COSMICTIGER_GEOGRAM_SOURCE_DIR}")
endif()
file(MAKE_DIRECTORY "${_cosmictiger_geogram_dir}")
set(_cosmictiger_geogram_files Delaunay_psm.h Delaunay_psm.cpp LICENSE)
set(_cosmictiger_geogram_hashes
    a9d292ef24a8fd9e2353cf9fa3e232473eabc25c06241ab7b96586cfefcfb328
    8b93e9ad904f7f035112e453187bf24ca5d192deb902bb8bb8e2e266fa8c0591
    5071ceca467d28e6fefb5f0dba4066abfd03d8b8b5b3af968d0aa7754c5e3f9c)
foreach(_cosmictiger_geogram_index RANGE 0 2)
    list(GET _cosmictiger_geogram_files ${_cosmictiger_geogram_index} _cosmictiger_geogram_file)
    list(GET _cosmictiger_geogram_hashes ${_cosmictiger_geogram_index} _cosmictiger_geogram_hash)
    set(_cosmictiger_geogram_path "${_cosmictiger_geogram_dir}/${_cosmictiger_geogram_file}")
    if(NOT EXISTS "${_cosmictiger_geogram_path}")
        if(COSMICTIGER_GEOGRAM_SOURCE_DIR)
            message(FATAL_ERROR "Missing pinned Geogram file: ${_cosmictiger_geogram_path}")
        endif()
        file(DOWNLOAD
            "https://raw.githubusercontent.com/BrunoLevy/geogram.psm.Delaunay/${_cosmictiger_geogram_revision}/${_cosmictiger_geogram_file}"
            "${_cosmictiger_geogram_path}"
            EXPECTED_HASH "SHA256=${_cosmictiger_geogram_hash}" TLS_VERIFY ON
            STATUS _cosmictiger_geogram_status TIMEOUT 90)
        list(GET _cosmictiger_geogram_status 0 _cosmictiger_geogram_error)
        if(NOT _cosmictiger_geogram_error EQUAL 0)
            message(FATAL_ERROR "Cannot download pinned Geogram: ${_cosmictiger_geogram_status}")
        endif()
    endif()
    file(SHA256 "${_cosmictiger_geogram_path}" _cosmictiger_geogram_actual_hash)
    if(NOT _cosmictiger_geogram_actual_hash STREQUAL _cosmictiger_geogram_hash)
        message(FATAL_ERROR "Geogram source hash mismatch: ${_cosmictiger_geogram_path}")
    endif()
endforeach()

find_package(Threads REQUIRED)
if(NOT TARGET Boost::headers AND NOT TARGET Boost::boost)
    find_package(Boost REQUIRED)
endif()
get_filename_component(_cosmictiger_voronoi_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(cosmictiger_geogram STATIC "${_cosmictiger_geogram_dir}/Delaunay_psm.cpp")
target_include_directories(cosmictiger_geogram SYSTEM PUBLIC "${_cosmictiger_geogram_dir}")
target_compile_features(cosmictiger_geogram PUBLIC cxx_std_17)
target_compile_definitions(cosmictiger_geogram PUBLIC GEO_STATIC_LIBS)
target_link_libraries(cosmictiger_geogram PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_library(cosmictiger_voronoi STATIC "${_cosmictiger_voronoi_root}/src/geometry/voronoi.cpp")
target_include_directories(cosmictiger_voronoi PUBLIC "${_cosmictiger_voronoi_root}")
target_compile_features(cosmictiger_voronoi PUBLIC cxx_std_20)
target_link_libraries(cosmictiger_voronoi PRIVATE cosmictiger_geogram)
if(TARGET Boost::headers)
    target_link_libraries(cosmictiger_voronoi PRIVATE Boost::headers)
else()
    target_link_libraries(cosmictiger_voronoi PRIVATE Boost::boost)
endif()
# Filtered/exact predicates require IEEE semantics. These options intentionally
# override any inherited fast-math flags and must never be replaced by CUDA's
# --use_fast_math or host contraction/reassociation settings.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cosmictiger_geogram PRIVATE -fno-fast-math -frounding-math -ffp-contract=off)
    target_compile_options(cosmictiger_voronoi PRIVATE -fno-fast-math -frounding-math -ffp-contract=off)
elseif(MSVC)
    target_compile_options(cosmictiger_geogram PRIVATE /fp:strict)
    target_compile_options(cosmictiger_voronoi PRIVATE /fp:strict)
endif()
