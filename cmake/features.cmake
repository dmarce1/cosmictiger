# A build contains exactly the enabled algorithms. Selection among available
# algorithms happens at startup; these switches describe capabilities only.
set(cosmictiger_retired_cache_options)
foreach(legacy IN ITEMS COSMICTIGER_TIME_INTEGRATOR COSMICTIGER_GRAVITY_METHOD
        COSMICTIGER_PMMM_NEAR_METHOD COSMICTIGER_WITH_ORDER
        COSMICTIGER_FMM_MIN_ORDER COSMICTIGER_FMM_MAX_ORDER COSMICTIGER_WITH_SPH
        COSMICTIGER_ENABLE_PMMM COSMICTIGER_ENABLE_PMMM_NEAR_PP
        COSMICTIGER_ENABLE_PMMM_NEAR_FMM COSMICTIGER_ENABLE_FMM
        COSMICTIGER_ENABLE_PMMM_PP COSMICTIGER_ENABLE_PMMM_FMM COSMICTIGER_ENABLE_PM
        COSMICTIGER_ENABLE_P3M_PP COSMICTIGER_ENABLE_P3M_FMM)
    if(DEFINED CACHE{${legacy}})
        list(APPEND cosmictiger_retired_cache_options ${legacy})
    endif()
endforeach()
if(cosmictiger_retired_cache_options)
    list(JOIN cosmictiger_retired_cache_options ", " retired_names)
    message(FATAL_ERROR "Retired CosmicTiger cache options: ${retired_names}. "
        "Select algorithms with COSMICTIGER_ENABLE_* and potential degrees with "
        "COSMICTIGER_EXPANSION_DEGREE (old order minus one). Remove the old entries "
        "using cmake -U COSMICTIGER_TIME_INTEGRATOR -U COSMICTIGER_GRAVITY_METHOD "
        "-U COSMICTIGER_PMMM_NEAR_METHOD -U COSMICTIGER_WITH_ORDER "
        "-U COSMICTIGER_FMM_MIN_ORDER -U COSMICTIGER_FMM_MAX_ORDER "
        "-U COSMICTIGER_WITH_SPH -U COSMICTIGER_ENABLE_PMMM "
        "-U COSMICTIGER_ENABLE_PMMM_NEAR_PP -U COSMICTIGER_ENABLE_PMMM_NEAR_FMM "
        "-U COSMICTIGER_ENABLE_FMM -U COSMICTIGER_ENABLE_PMMM_PP "
        "-U COSMICTIGER_ENABLE_PMMM_FMM -U COSMICTIGER_ENABLE_PM "
        "-U COSMICTIGER_ENABLE_P3M_PP -U COSMICTIGER_ENABLE_P3M_FMM, "
        "and supply the desired new feature options. The only gravity solver is "
        "PMMM with FMM near interactions, selected by COSMICTIGER_ENABLE_GRAVITY.")
endif()
# PMMM gravity: periodic FFT far field and FMM near interactions. Close leaf
# interactions use the particle-pair kernels inside the FMM tree walk.
option(COSMICTIGER_ENABLE_GRAVITY "Compile PMMM gravity with FMM near interactions" ON)
option(COSMICTIGER_ENABLE_SPH "Compile SPH hydrodynamics" ON)
option(COSMICTIGER_ENABLE_ROCKSTAR "Compile ROCKSTAR halo finding and lightcone analysis when gravity is enabled" ON)
option(COSMICTIGER_ENABLE_CHEMISTRY "Compile GADGET-4 equilibrium cooling when SPH is enabled" ON)
# The legacy name initializes a fresh/migrated cache once. Thereafter STARS is
# authoritative, so explicitly disabling it cannot be undone by a stale alias.
set(cosmictiger_stars_default OFF)
if(NOT DEFINED COSMICTIGER_ENABLE_STARS AND DEFINED COSMICTIGER_ENABLE_STAR_FORMATION)
    set(cosmictiger_stars_default "${COSMICTIGER_ENABLE_STAR_FORMATION}")
endif()
option(COSMICTIGER_ENABLE_STARS "Compile Springel-Hernquist star formation when SPH is enabled"
    "${cosmictiger_stars_default}")
set(COSMICTIGER_ENABLE_STAR_FORMATION "${COSMICTIGER_ENABLE_STARS}" CACHE BOOL
    "Legacy initializer for COSMICTIGER_ENABLE_STARS; thereafter follows STARS" FORCE)
set(COSMICTIGER_ENABLE_STAR_FORMATION "${COSMICTIGER_ENABLE_STARS}")
mark_as_advanced(COSMICTIGER_ENABLE_STAR_FORMATION)
if(COSMICTIGER_ENABLE_STARS AND NOT COSMICTIGER_ENABLE_CHEMISTRY)
    message(FATAL_ERROR "COSMICTIGER_ENABLE_STARS=ON requires COSMICTIGER_ENABLE_CHEMISTRY=ON")
endif()
# One integration scheme: hierarchical short gravity, outer long kicks, midpoint SPH.
# Clear obsolete cache selectors so an existing build migrates to the sole scheme.
foreach(mode IN ITEMS TRADITIONAL HIERARCHICAL GLOBAL)
    unset(COSMICTIGER_ENABLE_INTEGRATOR_${mode} CACHE)
endforeach()
set(COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL OFF)
set(COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL ON)
set(COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL OFF)
set(COSMICTIGER_DIMENSION_COUNT "3" CACHE STRING "Spatial dimensions for SPH (1, 2, or 3; gravity requires 3)")
set_property(CACHE COSMICTIGER_DIMENSION_COUNT PROPERTY STRINGS 1 2 3)
if(NOT COSMICTIGER_DIMENSION_COUNT MATCHES "^[123]$")
    message(FATAL_ERROR "COSMICTIGER_DIMENSION_COUNT must be 1, 2, or 3")
endif()
set(COSMICTIGER_WITH_SPH ${COSMICTIGER_ENABLE_SPH})
# Preserve requested cache settings while exposing only compiled capabilities.
set(COSMICTIGER_WITH_CHEMISTRY OFF)
set(COSMICTIGER_WITH_STARS OFF)
if(COSMICTIGER_WITH_SPH AND COSMICTIGER_ENABLE_CHEMISTRY)
    set(COSMICTIGER_WITH_CHEMISTRY ON)
    if(COSMICTIGER_ENABLE_STARS)
        set(COSMICTIGER_WITH_STARS ON)
    endif()
endif()
set(COSMICTIGER_WITH_GRAVITY ${COSMICTIGER_ENABLE_GRAVITY})
if(COSMICTIGER_DIMENSION_COUNT LESS 3)
    if(COSMICTIGER_WITH_GRAVITY OR NOT COSMICTIGER_WITH_SPH)
        message(FATAL_ERROR "COSMICTIGER_DIMENSION_COUNT=1 or 2 requires SPH and COSMICTIGER_ENABLE_GRAVITY=OFF")
    endif()
    if(COSMICTIGER_WITH_CHEMISTRY OR COSMICTIGER_WITH_STARS)
        message(FATAL_ERROR "Lower-dimensional SPH requires COSMICTIGER_ENABLE_CHEMISTRY=OFF and COSMICTIGER_ENABLE_STARS=OFF; volumetric source physics is three-dimensional")
    endif()
endif()
message(STATUS "CosmicTiger spatial dimensions: ${COSMICTIGER_DIMENSION_COUNT}")
set(COSMICTIGER_WITH_ROCKSTAR OFF)
if(COSMICTIGER_WITH_GRAVITY AND COSMICTIGER_ENABLE_ROCKSTAR)
    set(COSMICTIGER_WITH_ROCKSTAR ON)
endif()
if(NOT COSMICTIGER_WITH_GRAVITY AND NOT COSMICTIGER_ENABLE_SPH)
    message(FATAL_ERROR "Enable at least one of COSMICTIGER_ENABLE_GRAVITY or COSMICTIGER_ENABLE_SPH")
endif()
set(COSMICTIGER_EXPANSION_DEGREE "4-6" CACHE STRING
    "Compiled maximum potential degrees: one degree or inclusive range, from 2 through 8")
if(COSMICTIGER_EXPANSION_DEGREE MATCHES "^([2-8])-([2-8])$")
    set(COSMICTIGER_MIN_EXPANSION_DEGREE ${CMAKE_MATCH_1})
    set(COSMICTIGER_MAX_EXPANSION_DEGREE ${CMAKE_MATCH_2})
elseif(COSMICTIGER_EXPANSION_DEGREE MATCHES "^[2-8]$")
    set(COSMICTIGER_MIN_EXPANSION_DEGREE ${COSMICTIGER_EXPANSION_DEGREE})
    set(COSMICTIGER_MAX_EXPANSION_DEGREE ${COSMICTIGER_EXPANSION_DEGREE})
else()
    message(FATAL_ERROR "COSMICTIGER_EXPANSION_DEGREE must be a degree from 2 through 8 or an inclusive range such as 4-6")
endif()
if(COSMICTIGER_MIN_EXPANSION_DEGREE GREATER COSMICTIGER_MAX_EXPANSION_DEGREE)
    message(FATAL_ERROR "COSMICTIGER_EXPANSION_DEGREE range must be increasing")
endif()
# Kernel templates and checkpoints retain their historical order convention.
# A potential of degree d uses order d+1 and source multipoles of degree d-1.
math(EXPR COSMICTIGER_FMM_MIN_ORDER "${COSMICTIGER_MIN_EXPANSION_DEGREE} + 1")
math(EXPR COSMICTIGER_FMM_MAX_ORDER "${COSMICTIGER_MAX_EXPANSION_DEGREE} + 1")
set(cosmictiger_default_fmm_order ${COSMICTIGER_FMM_MAX_ORDER})
set(COSMICTIGER_FMM_ORDERS)
foreach(order RANGE ${COSMICTIGER_FMM_MIN_ORDER} ${COSMICTIGER_FMM_MAX_ORDER})
    list(APPEND COSMICTIGER_FMM_ORDERS ${order})
endforeach()
if(NOT COSMICTIGER_WITH_GRAVITY)
    set(cosmictiger_default_fmm_order 0)
    set(COSMICTIGER_FMM_MIN_ORDER 0)
    set(COSMICTIGER_FMM_MAX_ORDER 0)
    set(COSMICTIGER_FMM_ORDERS)
endif()
set(cosmictiger_default_integrator hierarchical)
