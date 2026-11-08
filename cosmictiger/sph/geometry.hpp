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

// One particle's vector, gathered from separate component arrays over the active axes.
template <class Real, class Index>
CUDA_EXPORT inline SpatialVector<Real> loadSpatialVector(
    Real const* x, Real const* y, Real const* z, Index index) {
    Real const* const components[kDimensions] = {x, y, z};
    SpatialVector<Real> result;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        result[dim] = components[dim][index];
    }
    return result;
}

// Padded XYZ records are an I/O/storage convention, not extra spatial directions. Interfaces
// outside the SPH kernels and the three-component curl still take them; pad only at those
// boundaries. Inactive axes hold zero.
template <class Real>
CUDA_EXPORT inline Vector3<Real> paddedVector(SpatialVector<Real> const& spatial) {
    Vector3<Real> padded(Real(0));
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        padded[dim] = spatial[dim];
    }
    return padded;
}

// Derived from W_D = h⁻ᴰ w_D(r/h): density scales as h⁻ᴰ and its
// gradient as h⁻⁽ᴰ⁺¹⁾. D → spatial dimensions; h⁻¹, h⁻² → inverse support and its square.
// Code names: ih = h⁻¹; ih2 = h⁻².
// Hopkins (2013), https://arxiv.org/abs/1206.5006, equations (8)–(13),
// generalized here from 3D. Retain the established 3D multiplication order.
template <class Real>
CUDA_EXPORT inline Real inverseVolume(Real ih) {
    if constexpr (kSpatialDimensions == 1) {
        return ih;
    } else if constexpr (kSpatialDimensions == 2) {
        return ih * ih;
    } else {
        return ih * ih * ih;
    }
}

template <class Real>
CUDA_EXPORT inline Real inverseGradientScale(Real ih) {
    if constexpr (kSpatialDimensions == 3) {
        Real const ih2 = ih * ih;
        return ih2 * ih2;
    } else {
        return inverseVolume(ih) * ih;
    }
}

// Derived from V_D hᴰ ρ/m = N, the neighbor count of a uniform density:
//   h = [m N/(V_D ρ)]^(1/D).
// m → particle mass; ρ → density; N → neighbor count; V_D → unit ball volume;
// D → spatial dimensions; h → support radius (return value).
inline double supportRadius(double m, double ρ, double N) {
    return std::pow(m * N / (kUnitBallVolume * ρ), 1.0 / kSpatialDimensions);
}

}    // namespace sph_geometry

// Component separations r𞁑𞁓 = x𞁑 − x𞁓 of two position vectors over the active axes, using
// the scalar distance() of the coordinate type. x𞁑, x𞁓 → sink, neighbor position;
// r𞁑𞁓 → separation (return value). Global so that distance(x𞁑, x𞁓) resolves in kernels.
template <class Coordinate>
CUDA_EXPORT inline sph_geometry::SpatialVector<float> distance(
    sph_geometry::SpatialVector<Coordinate> const& x𞁑,
    sph_geometry::SpatialVector<Coordinate> const& x𞁓) {
    sph_geometry::SpatialVector<float> r𞁑𞁓;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        r𞁑𞁓[dim] = distance(x𞁑[dim], x𞁓[dim]);
    }
    return r𞁑𞁓;
}
