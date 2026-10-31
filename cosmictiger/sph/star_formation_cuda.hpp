/* CosmicTiger - SH03 CUDA batch boundary; BSL-1.0. */
#pragma once

#include <cstddef>

#include <cosmictiger/sph/star_formation_kernel.hpp>

namespace star_formation_backend {
// All outputs are staged. A failed particle or CUDA operation throws before the
// caller can commit any particle state. A CUDA request never falls back to CPU.
void runCuda(Operation operation, Input const* inputs, Output* outputs, size_t count,
    Parameters const& parameters);
}    // namespace star_formation_backend
