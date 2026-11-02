# Sources independent of the configured integrator, expansion order, and
# position precision. Their project-header dependencies do not include the
# capability build configuration or the Position type.
#
# CUDA support is selected for the whole build, so its utility objects may
# also be shared by the configured application and its tests.
set(cosmictiger_core_source_files
    src/core/assert.cpp
    src/core/flops.cpp
    src/core/memused.cpp
    src/math/kernel.cpp
)

if(COSMICTIGER_WITH_CUDA)
    list(APPEND cosmictiger_core_source_files
        src/core/assert.cu
        src/core/flops.cu
        src/cuda/cuda_memory.cu
        src/cuda/cuda_stream.cpp
        src/math/kernel.cu
    )
endif()
