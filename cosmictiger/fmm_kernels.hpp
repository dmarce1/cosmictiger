/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */
#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/ewald_kernels.hpp>
#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/fmm_scale.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>
#include <cosmictiger/simd_math.hpp>

// The generated STF operators are emitted for one (source degree, local degree)
// pair. The build directory shadows the checked-in defaults when
// COSMICTIGER_WITH_ORDER selects another order; see fmm_order.hpp.
static_assert(cosmictiger::gravity::stf::kMinOrder == kFmmMultipoleDegree &&
        cosmictiger::gravity::stf::kMaxOrder == kFmmLocalDegree,
    "Generated STF kernels do not match ORDER; regenerate with tools/generate_stf_m2l.py");

// Compact rank-major STF storage. Only local expansions carry a Laplacian;
// source moments contain exactly (degree + 1)^2 coefficients, including dipoles.
template <class Real, int degree, bool withTrace = false>
class StfTensor : public Array<Real, (degree + 1) * (degree + 1) + int(withTrace)> {
public:
    static constexpr int kCount_ = (degree + 1) * (degree + 1);
    CUDA_EXPORT static constexpr int index(int nx, int ny, int nz) {
        int const rank = nx + ny + nz;
        return rank * rank + (nz == 0 ? nx : rank + 1 + nx);
    }
    CUDA_EXPORT StfTensor& operator=(Real const& value) {
        for (int i = 0; i < kCount_ + int(withTrace); i++) {
            (*this)[i] = value;
        }
        return *this;
    }
    CUDA_EXPORT Real& operator()(int nx, int ny, int nz) {
        ASSERT(nz <= 1);
        return (*this)[index(nx, ny, nz)];
    }
    CUDA_EXPORT Real operator()(int nx, int ny, int nz) const {
        if (nz < 2) {
            return (*this)[index(nx, ny, nz)];
        }
        Real result = -(*this)(nx + 2, ny, nz - 2) - (*this)(nx, ny + 2, nz - 2);
        if constexpr (withTrace) {
            if (nx == 0 && ny == 0 && nz == 2) {
                result += (*this)[kCount_];
            }
        }
        return result;
    }
};

template <class Real>
using Multipole = StfTensor<Real, kFmmMultipoleDegree>;
template <class Real>
using Expansion = StfTensor<Real, kFmmLocalDegree, true>;
template <class Real>
using Expansion2 = StfTensor<Real, 1>;
#define MULTIPOLE_SIZE kFmmMultipoleCoefficients
#define EXPANSION_SIZE kFmmLocalCoefficients
static_assert(sizeof(Multipole<float>) == 4 * kFmmMultipoleCoefficients, "Multipole storage");
static_assert(sizeof(Expansion<float>) == kFmmLocalExpansionBytes, "Local expansion storage");

namespace detail {
namespace stf = cosmictiger::gravity::stf::detail;

template <class Real>
CUDA_EXPORT Real fmmInverseRadius(Array<Real, NDIM> const& position) {
    Real const radius = simdSqrt(simdFma(
        position[0], position[0], simdFma(position[1], position[1], position[2] * position[2])));
    return simdSelect(
               radius > simdBroadcast<Real>(0), simdBroadcast<Real>(1), simdBroadcast<Real>(0)) /
        simdMax(radius, simdBroadcast<Real>(1.0e-20));
}
}    // namespace detail

template <class Real>
CUDA_EXPORT Multipole<Real> p2M(Array<Real, NDIM> const& position) {
    Multipole<Real> result{};
    detail::stf::P2mKernel<kFmmMultipoleDegree>::add(result, simdBroadcast<Real>(1),
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR));
    return result;
}

template <class Real>
CUDA_EXPORT Multipole<Real> m2M(Multipole<Real> const& source, Array<Real, NDIM> position) {
    Multipole<Real> result{};
    detail::stf::M2mKernel<kFmmMultipoleDegree>::add(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    return result;
}

template <class Real>
CUDA_EXPORT int greensFunction(
    Expansion<Real>& local, Array<Real, NDIM> position, bool scale = true) {
    if (scale) {
        for (int dim = 0; dim < NDIM; dim++) {
            position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
        }
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    local = simdBroadcast<Real>(0);
    detail::stf::P2lKernel<kFmmLocalDegree>::add(local, simdBroadcast<Real>(1),
        position[0] * inverseRadius, position[1] * inverseRadius, position[2] * inverseRadius,
        inverseRadius, inverseRadius);
    return detail::stf::P2lKernel<kFmmLocalDegree>::kFlops_ + 14;
}

// Direct M2L uses the imported partial-contraction kernel, avoiding the
// intermediate Green tensor. centered=true is valid only for zero source dipole.
template <class Real, bool centered = false>
CUDA_EXPORT int m2LDirect(
    Expansion<Real>& local, Multipole<Real> const& source, Array<Real, NDIM> position, bool doPhi) {
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Real const oldPotential = local[0];
    detail::stf::M2lKernel<kFmmLocalDegree, centered>::template add<Real, Expansion<Real>,
        Multipole<Real>, kFmmMultipoleDegree>(local, source, position[0] * inverseRadius,
        position[1] * inverseRadius, position[2] * inverseRadius, inverseRadius, inverseRadius,
        inverseRadius);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::stf::M2lKernel<kFmmLocalDegree, centered>::kFlops_ + 14;
}

template <class Real, bool centered = false>
CUDA_EXPORT int m2P(Expansion2<Real>& local, Multipole<Real> const& source,
    Array<Real, NDIM> position, bool doPhi) {
    for (int dim = 0; dim < NDIM; dim++) {
        position[dim] *= simdBroadcast<Real>(SCALE_FACTOR);
    }
    Real const inverseRadius = detail::fmmInverseRadius(position);
    Array<Real, NDIM + 1> field{};
    detail::stf::M2pKernel<kFmmMultipoleDegree, centered>::add(field, source,
        position[0] * inverseRadius, position[1] * inverseRadius, position[2] * inverseRadius,
        inverseRadius, inverseRadius);
    if (doPhi) {
        local[0] += field[0];
    }
    local(1, 0, 0) -= field[1];
    local(0, 1, 0) -= field[2];
    local(0, 0, 1) -= field[3];
    return detail::stf::M2pKernel<kFmmMultipoleDegree, centered>::kFlops_ + 18;
}

// Ewald is evaluated on the nearest-image branch. Harmonicize it using the
// unwrapped separation, so the removed quadratic sums to one root polynomial.
template <class Real>
CUDA_EXPORT int ewaldGreensFunction(Expansion<Real>& local,
    Array<Real, NDIM> const& wrappedPosition, Array<Real, NDIM> const& unwrappedPosition) {
    SymmetricTracelessTensor<Real, kFmmEwaldTensorOrder> derivatives;
    int const flops = detail::ewaldCorrectionDerivatives(derivatives, wrappedPosition);
    for (int rank = 0; rank <= kFmmLocalDegree; rank++) {
        for (int nz = 0; nz <= 1 && nz <= rank; nz++) {
            for (int nx = 0; nx <= rank - nz; nx++) {
                int const ny = rank - nx - nz;
                local(nx, ny, nz) = derivatives(nx, ny, nz);
            }
        }
    }
    Real const coefficient = simdBroadcast<Real>(4.0 * M_PI / 3.0);
    Real radiusSquared = simdBroadcast<Real>(0);
    for (int dim = 0; dim < NDIM; dim++) {
        radiusSquared = simdFma(unwrappedPosition[dim], unwrappedPosition[dim], radiusSquared);
        int const nx = dim == XDIM;
        int const ny = dim == YDIM;
        int const nz = dim == ZDIM;
        local(nx, ny, nz) +=
            coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV2) * unwrappedPosition[dim];
    }
    local[0] += simdBroadcast<Real>(0.5 * SCALE_FACTOR_INV1) * coefficient * radiusSquared;
    local(2, 0, 0) += coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV3);
    local(0, 2, 0) += coefficient * simdBroadcast<Real>(SCALE_FACTOR_INV3);
    local[kFmmLocalTraceIndex] = simdBroadcast<Real>(0);
    return flops + 25;
}

template <class Real>
CUDA_EXPORT int ewaldGreensFunction(Expansion<Real>& local, Array<Real, NDIM> const& position) {
    return ewaldGreensFunction(local, position, position);
}

template <class Real, bool centered = false>
CUDA_EXPORT int m2L(Expansion<Real>& local, Multipole<Real> const& source,
    Expansion<Real> const& derivatives, bool doPhi) {
    Real const oldPotential = local[0];
    detail::stf::EwaldM2lKernel<kFmmLocalDegree, centered>::template add<Real>(
        local, source, derivatives);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::stf::EwaldM2lKernel<kFmmLocalDegree, centered>::kFlops_;
}

template <class Real>
CUDA_EXPORT Expansion<Real> l2L(
    Expansion<Real> const& source, Array<Real, NDIM> position, bool doPhi) {
    Expansion<Real> result{};
    detail::stf::L2lKernel<kFmmLocalDegree, true>::add(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    if (!doPhi) {
        result[0] = source[0];
    }
    return result;
}

template <class Real>
CUDA_EXPORT Expansion2<Real> l2P(
    Expansion<Real> const& source, Array<Real, NDIM> position, bool doPhi) {
    Array<Real, NDIM + 1> field{};
    detail::stf::L2pKernel<kFmmLocalDegree, true>::add(field, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), simdBroadcast<Real>(1));
    Expansion2<Real> result{};
    result[0] = doPhi ? field[0] : simdBroadcast<Real>(0);
    result(1, 0, 0) = -field[1];
    result(0, 1, 0) = -field[2];
    result(0, 0, 1) = -field[3];
    return result;
}

inline void initializeBackgroundExpansion(
    Expansion<float>& local, Multipole<float> const& rootMultipole, double secondMomentTrace) {
    double const coefficient = 4.0 * M_PI / 3.0;
    local = 0.0f;
    local[0] = -0.5 * coefficient * secondMomentTrace * SCALE_FACTOR_INV1;
    local(1, 0, 0) = coefficient * rootMultipole(1, 0, 0) * SCALE_FACTOR_INV3;
    local(0, 1, 0) = coefficient * rootMultipole(0, 1, 0) * SCALE_FACTOR_INV3;
    local(0, 0, 1) = coefficient * rootMultipole(0, 0, 1) * SCALE_FACTOR_INV3;
    local(2, 0, 0) = local(0, 2, 0) = -coefficient * rootMultipole[0] * SCALE_FACTOR_INV3;
    local[kFmmLocalTraceIndex] = -4.0 * M_PI * rootMultipole[0] * SCALE_FACTOR_INV3;
}

#ifdef __CUDACC__
template <class Real>
__device__ Expansion<Real> l2LCuda(
    Expansion<Real> const& source, Array<Real, NDIM> position, bool doPhi) {
    Expansion<Real> result{};
    detail::stf::L2lWarpKernel<kFmmLocalDegree, true>::translate(result, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
    return result;
}

// Cooperative translation: group leaders write their coefficients straight into
// `target`, which must not alias `source`. Complete on return for every lane.
template <class Real, int localDegree = kFmmLocalDegree>
__device__ void l2LCudaInto(
    Real* target, Expansion<Real> const& source, Array<Real, NDIM> position, bool doPhi) {
    detail::stf::L2lWarpKernel<localDegree, true>::translateInto(target, source,
        position[0] * simdBroadcast<Real>(SCALE_FACTOR),
        position[1] * simdBroadcast<Real>(SCALE_FACTOR),
        position[2] * simdBroadcast<Real>(SCALE_FACTOR), doPhi);
}
#endif
