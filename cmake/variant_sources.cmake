# Select the union of enabled algorithms, keeping the shared core separate.
function(cosmictiger_feature_sources output)
    set(cosmictiger_source_files
        src/analysis/compress.cpp
        src/analysis/group_entry.cpp
        src/analysis/healpix.cu
        src/analysis/power.cpp
        src/analysis/power.cu
        src/analysis/power_cpu.cpp
        src/analysis/power_cuda.cu
        src/core/assert.cpp
        src/core/assert.cu
        src/core/driver.cpp
        src/core/scenario_driver.cpp
        src/core/flops.cpp
        src/core/flops.cu
        src/core/hpx.cpp
        src/core/memused.cpp
        src/core/options.cpp
        src/cosmology/boltzmann.cpp
        src/cosmology/background.cpp
        src/cosmology/background_plot.cpp
        src/cosmology/evolution.cpp
        src/cosmology/friedmann.cpp
        src/cosmology/cosmology.cpp
        src/cosmology/fft.cpp
        src/cosmology/fftw_threads.cpp
        src/cosmology/initialize.cpp
        src/cosmology/zero_order.cpp
        src/cuda/cuda.cpp
        src/cuda/cuda_memory.cu
        src/cuda/cuda_stream.cpp
        src/io/checkpoint.cpp
        src/io/output.cpp
        src/io/view.cpp
        src/main.cpp
        src/math/kernel.cpp
        src/math/kernel.cu
        src/math/math.cpp
        src/particles/domain.cpp
        src/particles/drift.cpp
        src/particles/drift.cu
        src/particles/drift_cpu.cpp
        src/particles/particles.cpp
        src/scenarios/scenario.cpp
        src/scenarios/cosmology.cpp
        src/particles/particle_partition.cpp
    )

    if(COSMICTIGER_WITH_GRAVITY)
        list(APPEND cosmictiger_source_files
            src/analysis/group_tree.cpp
            src/analysis/groups.cpp
            src/analysis/groups_find.cpp
            src/analysis/lightcone.cpp
            src/analysis/lightcone.cu
            src/analysis/lightcone_cpu.cpp
            src/analysis/rockstar.cpp
            src/analysis/rockstar.cu
            src/analysis/rockstar_catalogue.cpp
            src/gravity/fmm_order.cpp
            src/gravity/error.cpp
            src/gravity/error_cuda.cu
            src/gravity/reference_table.cpp
            src/analysis/bh.cpp
            src/analysis/bh.cu
            src/scenarios/plummer.cpp
            src/scenarios/glass.cpp
        )
    endif()

    # The tree walk is shared by global FMM and PMMM's optional near-cell FMM.
    set(cosmictiger_uses_fmm OFF)
    if(COSMICTIGER_ENABLE_FMM OR COSMICTIGER_ENABLE_PMMM_FMM)
        set(cosmictiger_uses_fmm ON)
    endif()
    if(cosmictiger_uses_fmm)
        list(APPEND cosmictiger_source_files
            src/gravity/gravity.cpp
            src/gravity/kick.cpp
            src/gravity/kick.cu
            src/gravity/kick_workspace.cpp
            src/gravity/tree_build.cpp
            src/gravity/kick_executor.cpp
            src/gravity/kick_interaction_counts.cpp
            src/gravity/source_access.cpp
            src/gravity/sources.cpp
            src/gravity/tree_arena.cpp
        )
    endif()

    if(COSMICTIGER_ENABLE_FMM)
        list(APPEND cosmictiger_source_files
            src/gravity/dehnen/ewald_indices.cpp
            src/gravity/dehnen/ewald_indices.cu
            src/gravity/dehnen/distributed_tree.cpp
            src/gravity/dehnen/execution.cpp
            src/gravity/dehnen/gpu_staging.cpp
            src/gravity/dehnen/particle_exchange.cpp
            src/gravity/dehnen/periodic.cpp
            src/gravity/dehnen/remote_cache.cpp
        )
    endif()

    if(COSMICTIGER_ENABLE_PMMM_PP OR COSMICTIGER_ENABLE_PMMM_FMM)
        list(APPEND cosmictiger_source_files
            src/gravity/nitadori/collective_dispatch.cpp
            src/gravity/nitadori/kick.cpp
            src/gravity/nitadori/solver.cpp
            src/gravity/nitadori/kernels.cpp
            src/particles/particle_mesh_sort.cpp
            src/gravity/nitadori/evaluation.cpp
            src/gravity/nitadori/fft.cpp
            src/gravity/nitadori/mesh_exchange.cpp
        )
        if(COSMICTIGER_ENABLE_PMMM_FMM)
            list(APPEND cosmictiger_source_files
                src/gravity/nitadori/cell_schedule.cpp
                src/gravity/nitadori/cell_tree.cpp
                src/gravity/nitadori/fmm_batch.cpp
                src/gravity/nitadori/ghost_exchange.cpp
                src/gravity/nitadori/ghost_storage.cpp
            )
        endif()
        if(COSMICTIGER_ENABLE_PMMM_PP)
            list(APPEND cosmictiger_source_files
                src/gravity/nitadori/near_pp.cpp
                src/gravity/nitadori/near_pp.cu
            )
        endif()
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
            src/scenarios/exact_sod.cpp
            src/scenarios/sedov.cpp
            src/sph/sph.cpp
            src/sph/sph.cu
            src/sph/sph_cpu.cpp
            src/sph/sph_hydro.cu
            src/sph/sph_hydro_cpu.cpp
            src/sph/sph_particles.cpp
            src/sph/sph_prehydro.cu
            src/sph/sph_prehydro_cpu.cpp
            src/sph/sph_rungs.cu
            src/sph/sph_rungs_cpu.cpp
            src/sph/sph_tree.cpp
            src/scenarios/sphere.cpp
            src/scenarios/sphere.cu
            src/sph/cullen_and_dehnen/particle_exchange.cpp
            src/sph/cullen_and_dehnen/viscosity.cpp
            src/sph/morris_and_monaghan/viscosity.cpp
            src/sph/price_etal/viscosity.cpp
        )
        if(COSMICTIGER_ENABLE_CHEMISTRY)
            list(APPEND cosmictiger_source_files src/sph/sph_chemistry.cpp)
        endif()
        if(COSMICTIGER_ENABLE_STAR_FORMATION)
            list(APPEND cosmictiger_source_files src/sph/star_formation.cpp
                src/sph/star_formation_cuda.cu src/sph/springel_and_hernquist/cooling.cpp)
        endif()
    endif()

    if(BUILD_TESTING)
        list(APPEND cosmictiger_source_files
            tests/test.cpp)
        if(COSMICTIGER_WITH_GRAVITY)
            list(APPEND cosmictiger_source_files
                tests/analysis_backends.cpp tests/bh.cpp tests/groups_catalogue.cpp tests/rockstar.cpp)
        endif()
        if(COSMICTIGER_ENABLE_PMMM_PP OR COSMICTIGER_ENABLE_PMMM_FMM)
            list(APPEND cosmictiger_source_files
                tests/mesh_domains.cpp
                tests/gravity/nitadori/particle_mesh_dispatch.cpp
                tests/particle_mesh_sort.cpp
                tests/gravity/nitadori/particle_mesh_timestep.cpp
                tests/gravity/nitadori/pmmm.cpp)
        endif()
        if(COSMICTIGER_ENABLE_FMM)
            list(APPEND cosmictiger_source_files
                tests/gravity/dehnen/fmm_tree.cpp tests/gravity/dehnen/fmm_walk.cpp
                tests/gravity/analytic.cpp tests/gravity/analytic.cu tests/linear_growth.cpp)
        endif()
        if(NOT COSMICTIGER_WITH_GRAVITY)
            list(APPEND cosmictiger_source_files tests/sph/no_gravity.cpp)
        endif()
        if(COSMICTIGER_WITH_SPH)
            list(APPEND cosmictiger_source_files
                tests/sph/tree.cpp tests/units_output.cpp tests/distributed_output.cpp)
            if(COSMICTIGER_ENABLE_CHEMISTRY)
                list(APPEND cosmictiger_source_files tests/sph/chemistry.cpp)
            endif()
            if(COSMICTIGER_ENABLE_STAR_FORMATION)
                list(APPEND cosmictiger_source_files tests/sph/star_formation.cpp
                    tests/star_formation_particles.cpp)
            endif()
            if(COSMICTIGER_ENABLE_FMM)
                list(APPEND cosmictiger_source_files tests/sph/gravity.cpp
                    tests/kernel_selection_cuda.cu)
            endif()
        endif()
    endif()

    if(NOT COSMICTIGER_WITH_CUDA)
        list(FILTER cosmictiger_source_files EXCLUDE REGEX "\\.cu$")
    endif()

    list(REMOVE_ITEM cosmictiger_source_files src/main.cpp ${cosmictiger_core_source_files})
    set(${output} ${cosmictiger_source_files} PARENT_SCOPE)
endfunction()
