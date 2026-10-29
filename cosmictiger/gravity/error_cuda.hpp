/* CosmicTiger - CUDA sampled gravity reference; BSL-1.0. */
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace detail {

using GravityErrorPosition = std::array<Position, 3>;
using GravityErrorForce = std::array<double, 3>;

struct GravityErrorSource {
    GravityErrorPosition position;
    double mass;
};

using GravityErrorSourceLoader = std::function<void(size_t, std::span<GravityErrorSource>)>;

#ifdef USE_CUDA
// The loader fills consecutive owned sources starting at its first argument.
// Results include source masses but not G; positions retain their fixed-point
// representation until doubleDistance forms each wrapped pair separation.
std::vector<GravityErrorForce> gravityErrorDirectCuda(std::span<GravityErrorPosition const> sinks,
    size_t sourceCount, GravityErrorSourceLoader const& loadSources,
    radial_kernel::KernelTable const& kernel, double supportRadius,
    std::span<GravityErrorForce const> correctionValues);
#endif

}    // namespace detail
