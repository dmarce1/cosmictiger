# Compile-time dispatch for small scalar matrices. Diagonal operations stay explicit.
# Defaults are measured by benchmarks/math/matrix.cpp; see docs/matrix.md.
foreach(matrix_structure IN ITEMS GENERAL SYMMETRIC)
    foreach(matrix_operation IN ITEMS INVERSE DETERMINANT SOLVE)
        set(matrix_option COSMICTIGER_MATRIX_${matrix_structure}_${matrix_operation}_THRESHOLD)
        if(matrix_operation STREQUAL "SOLVE")
            set(matrix_default 3)
        else()
            set(matrix_default 4)
        endif()
        set(${matrix_option} ${matrix_default} CACHE STRING
            "Largest ${matrix_structure} matrix using explicit ${matrix_operation} (0 disables; 1..6)")
        set_property(CACHE ${matrix_option} PROPERTY STRINGS 0 1 2 3 4 5 6)
        if(NOT "${${matrix_option}}" MATCHES "^[0-6]$")
            message(FATAL_ERROR "${matrix_option} must be an integer from 0 through 6")
        endif()
        add_compile_definitions(${matrix_option}=${${matrix_option}})
    endforeach()
endforeach()
