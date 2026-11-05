#pragma once

#include <cmath>
#include <numbers>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/vector.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace sph_geometry {

inline constexpr double kUnitBallVolume = kSpatialDimensions == 1 ?
    2.0 :
    (kSpatialDimensions == 2 ? std::numbers::pi : 4.0 * std::numbers::pi / 3.0);

template <class Real>
using SpatialVector = math::Vector<Real, kSpatialDimensions>;

// Padded XYZ records are an I/O/storage convention, not extra spatial directions.
template <class Real>
CUDA_EXPORT inline SpatialVector<Real> spatialVector(Vector3<Real> const& padded) {
    SpatialVector<Real> result;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        result[dim] = padded[dim];
    }
    return result;
}

template <class Real>
CUDA_EXPORT inline Real spatialDot(Vector3<Real> const& left, Vector3<Real> const& right) {
    return spatialVector(left).dot(spatialVector(right));
}

// Derived from W_D = H⁻ᴰ w_D(r/H): density scales as H⁻ᴰ and its
// gradient as H⁻⁽ᴰ⁺¹⁾. D → kSpatialDimensions; H⁻¹ → inverseSupport.
// Hopkins (2013), https://arxiv.org/abs/1206.5006, equations (8)–(13),
// generalized here from 3D. Retain the established 3D multiplication order.
template <class Real>
CUDA_EXPORT inline Real inverseVolume(Real inverseSupport) {
    if constexpr (kSpatialDimensions == 1) {
        return inverseSupport;
    } else if constexpr (kSpatialDimensions == 2) {
        return inverseSupport * inverseSupport;
    } else {
        return inverseSupport * inverseSupport * inverseSupport;
    }
}

template <class Real>
CUDA_EXPORT inline Real inverseGradientScale(Real inverseSupport) {
    if constexpr (kSpatialDimensions == 3) {
        Real const squared = inverseSupport * inverseSupport;
        return squared * squared;
    } else {
        return inverseVolume(inverseSupport) * inverseSupport;
    }
}

inline double supportRadius(double mass, double density, double neighbors) {
    return std::pow(mass * neighbors / (kUnitBallVolume * density), 1.0 / kSpatialDimensions);
}

}    // namespace sph_geometry
