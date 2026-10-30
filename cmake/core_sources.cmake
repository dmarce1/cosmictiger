# Sources shared by every integrator, gravity solver, SPH setting, expansion
# order, and position precision. Their project-header dependencies do not
# include the capability build configuration or the Position type.
#
# Keep source-specific floating-point options when compiling these in the
# shared library; in particular, Boltzmann and Friedmann require IEEE checks.
# CUDA support is selected for the whole build, so its utility objects may
# also be shared by the configured application and its tests.
set(cosmictiger_core_source_files
    src/analysis/group_entry.cpp
    src/core/assert.cpp
    src/core/flops.cpp
    src/core/hpx.cpp
    src/core/memused.cpp
    src/cosmology/boltzmann.cpp
    src/cosmology/friedmann.cpp
    src/cosmology/fftw_threads.cpp
    src/cosmology/zero_order.cpp
    src/cuda/cuda_stream.cpp
    src/math/kernel.cpp
    src/math/math.cpp
)

if(COSMICTIGER_WITH_CUDA)
    list(APPEND cosmictiger_core_source_files
        src/analysis/healpix.cu
        src/core/assert.cu
        src/core/flops.cu
        src/cuda/cuda_memory.cu
        src/math/kernel.cu
    )
endif()
