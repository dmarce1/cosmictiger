# Keep standalone test binaries and discovery metadata out of the build root.
# Registration stays alongside the target definitions; discovery runs later in
# its own CMake directory so GoogleTest uses the intended metadata directory.
function(cosmictiger_add_gtest)
    cmake_parse_arguments(PARSE_ARGV 0 test "HPX_MAIN" "NAME;TARGET;WORKING_DIRECTORY;ENVIRONMENT"
        "ARGS;PROPERTIES;LABELS")
    if(test_HPX_MAIN)
        target_link_libraries(${test_TARGET} PRIVATE GTest::gtest)
    else()
        target_link_libraries(${test_TARGET} PRIVATE GTest::gtest_main)
    endif()
    if(NOT test_WORKING_DIRECTORY)
        set(test_WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/tests/work/${test_TARGET}")
        file(MAKE_DIRECTORY "${test_WORKING_DIRECTORY}")
    endif()
    foreach(field IN ITEMS NAME ARGS PROPERTIES LABELS WORKING_DIRECTORY)
        set_property(TARGET ${test_TARGET} PROPERTY COSMICTIGER_TEST_${field} "${test_${field}}")
    endforeach()
    if(DEFINED test_ENVIRONMENT)
        set_property(TARGET ${test_TARGET} PROPERTY COSMICTIGER_TEST_ENVIRONMENT "${test_ENVIRONMENT}")
    endif()
    set_property(GLOBAL APPEND PROPERTY COSMICTIGER_GTEST_TARGETS "${test_TARGET}")
endfunction()

function(cosmictiger_finalize_test_layout)
    foreach(test_target IN LISTS ARGN)
        get_target_property(test_type ${test_target} TYPE)
        if(test_type STREQUAL "EXECUTABLE")
            set_target_properties(${test_target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/bin")
        elseif(test_type STREQUAL "SHARED_LIBRARY" OR test_type STREQUAL "MODULE_LIBRARY")
            set_target_properties(${test_target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/bin"
                LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/bin")
        endif()
    endforeach()
    add_subdirectory("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/test_discovery"
        "${PROJECT_BINARY_DIR}/tests/cmake")
endfunction()
