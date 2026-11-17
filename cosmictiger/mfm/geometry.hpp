/*
 CosmicTiger - A cosmological N-Body code
 Real-precision MFM distances, support geometry, and checked inverse moments.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/matrix.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace mfm {

inline constexpr int kMaximumSmoothingIterations = 128;

template <typename Real>
CUDA_EXPORT constexpr bool finite(Real value) {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
    if constexpr (std::is_same_v<Real, float>) {
        return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
    } else {
        return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }
}

template <typename Real, int dimensions>
CUDA_EXPORT constexpr Real unitBallVolume() {
    static_assert(dimensions >= 1 && dimensions <= kDimensions);
    if constexpr (dimensions == 1) {
        return Real(2);
    } else if constexpr (dimensions == 2) {
        return std::numbers::pi_v<Real>;
    } else {
        return Real(4) * std::numbers::pi_v<Real> / Real(3);
    }
}

// Derived support target: V_D hᴰ n=N, n=h⁻ᴰ Σw, hence Σw=N/V_D.
// Hopkins (2015), equation (26), uses C_1=1. Preserve CosmicTiger's existing SPH
// convention V_1=2, so its 1D neighbor target is twice that paper's target at fixed h.
// Symbol directory: N → weighted neighbor target; V_D → unit D-ball volume.
template <typename Real, int dimensions>
CUDA_EXPORT constexpr Real targetKernelSum(Real N) {
    return N / unitBallVolume<Real, dimensions>();
}

// Derived h⁻ᴰ; symbol directory: h⁻¹ → inverse full support; Code names: ih = h⁻¹.
template <typename Real, int dimensions>
CUDA_EXPORT constexpr Real inverseVolume(Real ih) {
    static_assert(dimensions >= 1 && dimensions <= kDimensions);
    if constexpr (dimensions == 1) {
        return ih;
    } else if constexpr (dimensions == 2) {
        return ih * ih;
    } else {
        return ih * ih * ih;
    }
}

// Minimum-image r𞁑𞁓=x𞁑−x𞁓. Subtract integer coordinates before converting to Real.
// Symbol directory: x𞁑, x𞁓 → particle positions; r𞁑𞁓 → periodic separation.
template <typename Real, int dimensions>
CUDA_EXPORT inline math::Vector<Real, dimensions> separation(
    math::Vector<Position, dimensions> const& x𞁑, math::Vector<Position, dimensions> const& x𞁓) {
    math::Vector<Real, dimensions> r𞁑𞁓;
    for (int dim = 0; dim < dimensions; ++dim) {
        if constexpr (std::is_same_v<Real, float>) {
            r𞁑𞁓[dim] = distance(x𞁑[dim], x𞁓[dim]);
        } else {
            static_assert(std::is_same_v<Real, double>);
            r𞁑𞁓[dim] = doubleDistance(x𞁑[dim], x𞁓[dim]);
        }
    }
    return r𞁑𞁓;
}

template <typename Real>
CUDA_EXPORT inline bool supportIntersects(
    SphBox const& sinkBox, Vector3<Position> const& position, Real smoothingLength) {
    if (!finite(smoothingLength) || !(smoothingLength > Real(0))) {
        return false;
    }
    SphBox expanded = sinkBox;
    for (int dim = 0; dim < kDimensions; ++dim) {
        expanded.begin[dim] -= RangeFixed(smoothingLength);
        expanded.end[dim] += RangeFixed(smoothingLength);
    }
    return expanded.contains(position);
}

/*
 * Hopkins (2015), MNRAS 450, 53, equations (6)-(7), (13)-(14), arXiv:1409.7395:
 * Eᵢ=Σⱼ ψⱼ(xᵢ)(xⱼ−xᵢ)⊗(xⱼ−xᵢ) (14), Bᵢ=Eᵢ⁻¹ (13).
 * With ψⱼ(xᵢ)=w(qᵢⱼ)/Σw, derived dimensionless construction:
 * Qᵢ=Σⱼ w(qᵢⱼ) qᵢⱼ⊗qᵢⱼ, Mᵢ=Qᵢ/Σw=Eᵢ/hᵢ², Bᵢ=Mᵢ⁻¹/hᵢ².
 * Only Mᵢ is inverted/conditioned, so validity is independent of length units.
 * Symbol directory: Qᵢ → unnormalized dimensionless moment; Σw → kernel weight sum;
 * hᵢ → full support; Mᵢ → normalized dimensionless moment; Bᵢ → physical inverse moment.
 * Code names: ΣW = Σw; iMᵢ = Mᵢ⁻¹; ihᵢ = hᵢ⁻¹.
 */
template <typename Real, int dimensions>
CUDA_EXPORT inline bool invertMoment(math::SymmetricMatrix<Real, dimensions> const& Qᵢ, Real ΣW,
    Real hᵢ, Real maximumCondition, math::SymmetricMatrix<Real, dimensions>& Bᵢ) {
    Bᵢ = math::SymmetricMatrix<Real, dimensions>{};
    if (!finite(ΣW) || !(ΣW > Real(0)) || !finite(hᵢ) || !(hᵢ > Real(0)) ||
        !finite(maximumCondition) || maximumCondition < Real(1)) {
        return false;
    }
    math::SymmetricMatrix<Real, dimensions> Mᵢ;
    for (int row = 0; row < dimensions; ++row) {
        for (int col = 0; col <= row; ++col) {
            if (!finite(Qᵢ(row, col))) {
                return false;
            }
            Mᵢ(row, col) = Qᵢ(row, col) / ΣW;
        }
        if (!(Mᵢ(row, row) > Real(0))) {
            return false;
        }
    }
    math::SymmetricMatrix<Real, dimensions> iMᵢ;
    if (math::tryInverse(Mᵢ, iMᵢ) != math::MatrixStatus::Success) {
        return false;
    }
    Real norm = Real(0);
    Real inverseNorm = Real(0);
    for (int row = 0; row < dimensions; ++row) {
        Real rowSum = Real(0);
        Real inverseRowSum = Real(0);
        for (int col = 0; col < dimensions; ++col) {
            rowSum += math::max(Mᵢ(row, col), -Mᵢ(row, col));
            inverseRowSum += math::max(iMᵢ(row, col), -iMᵢ(row, col));
        }
        norm = math::max(norm, rowSum);
        inverseNorm = math::max(inverseNorm, inverseRowSum);
        if (!(iMᵢ(row, row) > Real(0))) {
            return false;
        }
    }
    Real const condition = norm * inverseNorm;
    if (!finite(condition) || condition > maximumCondition) {
        return false;
    }
    Real const ihᵢ = Real(1) / hᵢ;
    math::SymmetricMatrix<Real, dimensions> candidate;
    for (int row = 0; row < dimensions; ++row) {
        for (int col = 0; col <= row; ++col) {
            candidate(row, col) = (iMᵢ(row, col) * ihᵢ) * ihᵢ;
            if (!finite(candidate(row, col))) {
                return false;
            }
        }
        if (!(candidate(row, row) > Real(0))) {
            return false;
        }
    }
    Bᵢ = candidate;
    return true;
}

}    // namespace mfm
