# PRE_TEST discovery supports targets defined in the parent directory. Calling
# GoogleTest here places its generated includes and discovered tests alongside
# our labels and environment metadata without overriding CMake directory state.
get_property(cosmictiger_gtest_targets GLOBAL PROPERTY COSMICTIGER_GTEST_TARGETS)
foreach(test_target IN LISTS cosmictiger_gtest_targets)
    foreach(field IN ITEMS NAME ARGS PROPERTIES LABELS WORKING_DIRECTORY)
        get_target_property(test_${field} ${test_target} COSMICTIGER_TEST_${field})
    endforeach()
    gtest_discover_tests(${test_target}
        NO_PRETTY_VALUES
        DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 30
        TEST_PREFIX "${test_NAME}." EXTRA_ARGS ${test_ARGS}
        WORKING_DIRECTORY "${test_WORKING_DIRECTORY}"
        PROPERTIES LABELS "${test_NAME}" ${test_PROPERTIES})

    # Apply list-valued properties after discovery so GoogleTest's argument
    # expansion cannot split the label and environment values.
    string(REPLACE "\\;" ";" test_LABELS "${test_LABELS}")
    set(labels_file "${CMAKE_CURRENT_BINARY_DIR}/${test_target}_labels.cmake")
    file(WRITE "${labels_file}"
        "if(DEFINED ${test_target}_TESTS)\n"
        "    set_tests_properties(\${${test_target}_TESTS} PROPERTIES\n"
        "        LABELS [==[${test_NAME};${test_LABELS}]==])\n"
        "endif()\n")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${labels_file}")

    get_property(has_environment TARGET ${test_target} PROPERTY COSMICTIGER_TEST_ENVIRONMENT SET)
    if(has_environment)
        get_target_property(test_environment ${test_target} COSMICTIGER_TEST_ENVIRONMENT)
        set(environment_file "${CMAKE_CURRENT_BINARY_DIR}/${test_target}_environment.cmake")
        file(WRITE "${environment_file}"
            "if(DEFINED ${test_target}_TESTS)\n"
            "    set_tests_properties(\${${test_target}_TESTS} PROPERTIES\n"
            "        ENVIRONMENT [==[${test_environment}]==])\n"
            "endif()\n")
        set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${environment_file}")
    endif()
endforeach()
