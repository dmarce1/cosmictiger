# Select the union of enabled algorithms, keeping the shared core separate.
function(cosmictiger_feature_sources output)
    set(cosmictiger_source_files
        src/core/assert.cpp
        src/core/assert.cu
        src/core/driver.cpp
        src/core/scenario_driver.cpp
        src/core/flops.cpp
        src/core/flops.cu
        # HPX initialization selects gravity-dependent CUDA analysis setup.
        src/core/hpx.cpp
        src/core/memused.cpp
        src/core/options.cpp
        src/cuda/cuda.cpp
        src/cuda/cuda_memory.cu
        src/io/checkpoint.cpp
        src/io/output.cpp
        src/io/view.cpp
        src/main.cpp
        src/math/kernel.cpp
        src/math/kernel.cu
        src/particles/domain.cpp
        src/particles/drift.cpp
        src/particles/drift.cu
        src/particles/drift_cpu.cpp
        src/particles/particles.cpp
        src/scenarios/scenario.cpp
        src/particles/particle_partition.cpp
    )

    if(COSMICTIGER_WITH_GRAVITY)
        list(APPEND cosmictiger_source_files
            src/gravity/fmm_order.cpp
            src/gravity/error.cpp
            src/gravity/error_cuda.cu
            src/gravity/reference_table.cpp
            src/scenarios/cosmology.cpp
            src/scenarios/plummer.cpp
            src/scenarios/glass.cpp
        )
    endif()
    if(COSMICTIGER_WITH_ROCKSTAR)
        list(APPEND cosmictiger_source_files
            src/analysis/compress.cpp
            src/analysis/group_entry.cpp
            src/analysis/healpix.cu
            src/analysis/group_tree.cpp
            src/analysis/groups.cpp
            src/analysis/groups_find.cpp
            src/analysis/lightcone.cpp
            src/analysis/lightcone.cu
            src/analysis/lightcone_cpu.cpp
            src/analysis/rockstar.cpp
            src/analysis/rockstar.cu
            src/analysis/rockstar_catalogue.cpp
            src/analysis/bh.cpp
            src/analysis/bh.cu
        )
    endif()

    if(COSMICTIGER_WITH_GRAVITY)
        # The tree walk serves the PMMM near-cell FMM.
        list(APPEND cosmictiger_source_files
            src/gravity/gravity.cpp
            src/gravity/kick.cpp
            src/gravity/kick.cu
            src/gravity/kick_workspace.cpp
            src/gravity/tree_build.cpp
            src/gravity/kick_executor.cpp
            src/gravity/source_access.cpp
            src/gravity/sources.cpp
            src/gravity/tree_arena.cpp
        )
    endif()

    # PMMM: collective dispatch, the owned-particle mesh sort, the streamed
    # field consumer that applies rung kicks, the FMM near field and its PP
    # fallback.
    if(COSMICTIGER_WITH_GRAVITY)
        list(APPEND cosmictiger_source_files
            src/gravity/collective_dispatch.cpp
            src/gravity/pm_kick.cpp
            src/gravity/outer_interval.cpp
            src/particles/particle_mesh_sort.cpp
            src/gravity/pmmm_kick.cpp
            src/gravity/interval.cpp
            src/gravity/solver.cpp
            src/gravity/pmmm_kernels.cpp
            src/gravity/evaluation.cpp
            src/gravity/fft.cpp
            src/gravity/mesh_exchange.cpp
            src/gravity/fmm/cell_schedule.cpp
            src/gravity/fmm/cell_tree.cpp
            src/gravity/fmm/fmm_batch.cpp
            src/gravity/fmm/ghost_exchange.cpp
            src/gravity/fmm/ghost_storage.cpp
            src/gravity/pp/near_pp.cpp
            src/gravity/pp/near_pp.cu
        )
    endif()

    if(COSMICTIGER_WITH_SPH)
        list(APPEND cosmictiger_source_files
            src/scenarios/sod.cpp
            src/scenarios/blast.cpp
            src/scenarios/rayleigh_taylor.cpp
            src/scenarios/kelvin_helmholtz.cpp
            src/scenarios/disc.cpp
            src/scenarios/polytropic_star.cpp
            src/scenarios/hydro_wave.cpp
            src/scenarios/jeans_wave.cpp
            src/scenarios/gresho.cpp
            src/scenarios/exact_sod.cpp
            src/scenarios/sedov.cpp
            src/sph/sph.cpp
            src/sph/cuda.cu
            src/sph/cpu.cpp
            src/sph/hydro.cu
            src/sph/hydro_cpu.cpp
            src/sph/particles.cpp
            src/sph/prehydro.cu
            src/sph/prehydro_cpu.cpp
            src/sph/rungs.cu
            src/sph/rungs_cpu.cpp
            src/sph/tree.cpp
            src/sph/parent_tree.cpp
            src/sph/ghosts.cpp
            src/sph/time_integration.cpp
            src/scenarios/sphere.cpp
            src/scenarios/sphere.cu
        )
        if(COSMICTIGER_DIMENSION_COUNT LESS 3)
            list(REMOVE_ITEM cosmictiger_source_files src/scenarios/rayleigh_taylor.cpp
                src/scenarios/disc.cpp src/scenarios/polytropic_star.cpp
                src/scenarios/sphere.cpp src/scenarios/sphere.cu)
        endif()
        if(COSMICTIGER_DIMENSION_COUNT LESS 2)
            list(REMOVE_ITEM cosmictiger_source_files src/scenarios/kelvin_helmholtz.cpp
                src/scenarios/gresho.cpp)
        endif()
        if(COSMICTIGER_ENABLE_CHEMISTRY)
            list(APPEND cosmictiger_source_files src/chemistry/sph.cpp)
        endif()
        if(COSMICTIGER_ENABLE_STAR_FORMATION)
            list(APPEND cosmictiger_source_files src/stars/star_formation.cpp
                src/stars/star_formation_cuda.cu)
        endif()
    endif()

    if(BUILD_TESTING AND COSMICTIGER_DIMENSION_COUNT EQUAL 3)
        list(APPEND cosmictiger_source_files
            tests/test.cpp)
        if(COSMICTIGER_WITH_ROCKSTAR)
            list(APPEND cosmictiger_source_files
                tests/analysis_backends.cpp tests/bh.cpp tests/groups_catalogue.cpp tests/rockstar.cpp)
        endif()
        if(COSMICTIGER_WITH_GRAVITY)
            list(APPEND cosmictiger_source_files
                tests/mesh_domains.cpp
                tests/gravity/particle_mesh_dispatch.cpp
                tests/particle_mesh_sort.cpp
                tests/gravity/particle_mesh_timestep.cpp
                tests/gravity/pmmm.cpp)
        else()
            list(APPEND cosmictiger_source_files tests/sph/no_gravity.cpp)
        endif()
        if(COSMICTIGER_WITH_SPH)
            list(APPEND cosmictiger_source_files
                tests/sph/tree.cpp tests/sph/parent_tree.cpp tests/sph/ghosts.cpp tests/units_output.cpp
                tests/distributed_output.cpp)
            if(COSMICTIGER_ENABLE_CHEMISTRY)
                list(APPEND cosmictiger_source_files tests/sph/chemistry.cpp)
            endif()
            if(COSMICTIGER_ENABLE_STAR_FORMATION)
                list(APPEND cosmictiger_source_files tests/sph/star_formation.cpp
                    tests/star_formation_particles.cpp tests/sph/source_timing.cpp)
            endif()
        endif()
    endif()

    if(NOT COSMICTIGER_WITH_CUDA)
        list(FILTER cosmictiger_source_files EXCLUDE REGEX "\\.cu$")
    endif()

    list(REMOVE_ITEM cosmictiger_source_files src/main.cpp ${cosmictiger_core_source_files})
    set(${output} ${cosmictiger_source_files} PARENT_SCOPE)
endfunction()
