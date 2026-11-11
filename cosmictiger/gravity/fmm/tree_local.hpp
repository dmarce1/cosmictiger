/* CosmicTiger - completed mesh locals in Cartesian tree coordinates; BSL-1.0. */
#pragma once

#include <type_traits>

#include <cosmictiger/gravity/kernels/basis_conversion.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/fixed.hpp>

namespace cosmictiger::gravity::tree_local {
namespace cartesian = gravity::cartesian;

// Derived boundary formulas, docs/fmm_basis_conversion.md:
// H′ = H + 2βₘ U₂₀; Cₙ = κ⁻¹(κh)⁻ⁿ Bₙ Lₙ; Φ = H′ + Kz²/2.
// K = -3βₘ/(κ³h²), where tree y=κr and mesh u=r/h.
// Symbol directory: H is the spherical harmonic potential; H′ is its harmonic
// proxy; βₘ is the mesh background coefficient; C is a Cartesian derivative;
// B is the local basis map; L is a spherical local; κ is kScaleFactor;
// h is mesh width; K is the tree residual curvature; z is tree displacement.
// Code names: meshBackground = βₘ; curvature = K; source = L; target = C.
CUDA_EXPORT inline int fromMesh(cartesian::Expansion<double>& target, double const* source,
    double meshBackground, int meshSize) {
    spherical::Expansion<double> harmonic;
    for (int i = 0; i < harmonic.kCount_; ++i) {
        harmonic[i] = source[i];
    }
    harmonic.real(2, 0) += 2 * meshBackground;
    target = static_cast<cartesian::Expansion<double>>(harmonic);
    double const κ = kScaleFactor;
    double const ratio = meshSize / κ;
    double factor = 1 / κ;
    for (int n = 0; n < kFmmOrder; ++n) {
        for (int i = n * n; i < (n + 1) * (n + 1); ++i) {
            target[i] *= factor;
        }
        factor *= ratio;
    }
    return basis_conversion::kExpansionCastConventionalFlops<kFmmOrder> + kFmmOrder * kFmmOrder +
        kFmmOrder + 4;
}

/// Convert the mesh isotropic-background coefficient to tree-coordinate curvature using
/// mesh and coordinate scales.
CUDA_EXPORT inline double backgroundCurvature(double meshBackground, int meshSize) {
    double const κ = kScaleFactor;
    return -3 * meshBackground * double(meshSize) * meshSize / (κ * κ * κ);
}

/// Subtract native fixed-point coordinates before conversion and scale the unwrapped
/// displacement into tree units.
template <class Real = double, class PositionType>
CUDA_EXPORT Vector3<Real> displacement(
    Vector3<PositionType> const& target, Vector3<PositionType> const& source) {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
    Vector3<Real> result;
    for (int axis = 0; axis < kDimensions; ++axis) {
        // Subtract native fixed-point values before converting, including Fixed64.
        if constexpr (std::is_same_v<Real, float>) {
            result[axis] = kScaleFactor * unwrappedDistance(target[axis], source[axis]);
        } else {
            result[axis] =
                double(kScaleFactor) * unwrappedDoubleDistance(target[axis], source[axis]);
        }
    }
    return result;
}

// Derived recentering: H′new(x)=H′old(x+d)+K d_z x_z+K d_z²/2.
// Symbol directory: H′ is the harmonic proxy; d is the tree center shift;
// K is the known residual curvature, carried once per solve, not in each local.
// Code names: curvature = K.
template <class Real>
CUDA_EXPORT int translate(cartesian::Expansion<Real>& Lo, cartesian::Expansion<Real> const& Li,
    Vector3<Real> const& d, Real curvature, bool doPotential) {
    int const flops = cartesian::L2L(Li, d, doPotential, Lo);
    Lo(0, 0, 1) += curvature * d[2];
    if (doPotential) {
        Lo[0] += Real(.5) * curvature * d[2] * d[2];
    }
    return flops + (doPotential ? 6 : 2);
}

// Derived field conversion: Φphys=κΦtree, aphys=κ²atree;
// Φtree=H′+Kz²/2, a_z=-∂zH′-Kz.
// Symbol directory: Φ is potential; a is acceleration; K is residual curvature;
// κ is kScaleFactor; y is tree displacement with z=y_z; H′ is the harmonic proxy.
// Code names: curvature = K.
template <class Real>
CUDA_EXPORT int evaluate(Field<Real>& target, cartesian::Expansion<Real> const& source,
    Vector3<Real> const& y, Real curvature, bool doPotential) {
    Field<Real> field{};
    int const flops = cartesian::L2P(source, y, doPotential, field);
    if (doPotential) {
        target[0] += Real(kScaleFactor) * (field[0] + Real(.5) * curvature * y[2] * y[2]);
    }
    field[3] -= curvature * y[2];
    for (int axis = 1; axis <= kDimensions; ++axis) {
        target[axis] += Real(kScaleFactor2) * field[axis];
    }
    return flops + (doPotential ? 14 : 8);
}
}    // namespace cosmictiger::gravity::tree_local
