/* CosmicTiger - Dehnen periodic FMM operators; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/dehnen/generated_cartesian_ewald.hpp>
#include <cosmictiger/gravity/fmm_kernels.hpp>

// Ewald is evaluated on the nearest-image branch. Harmonicize it using the
// unwrapped separation, so the removed quadratic sums to one root polynomial.
template <class Real, int degree>
CUDA_EXPORT int ewaldGreensFunction(Cartesian<Real, degree, true>& local,
    Array<Real, NDIM> const& wrappedPosition, Array<Real, NDIM> const& unwrappedPosition) {
    constexpr int kTensorOrder = degree + 1;
    SymmetricTracelessTensor<Real, kTensorOrder> derivatives;
    int const flops = detail::ewaldCorrectionDerivatives(derivatives, wrappedPosition);
    for (int rank = 0; rank <= degree; rank++) {
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
    local[Cartesian<Real, degree, true>::kCount_] = simdBroadcast<Real>(0);
    return flops + 22;
}

template <class Real, int degree>
CUDA_EXPORT int ewaldGreensFunction(
    Cartesian<Real, degree, true>& local, Array<Real, NDIM> const& position) {
    return ewaldGreensFunction(local, position, position);
}

template <class Real, int localDegree, int sourceDegree, bool centered = false>
CUDA_EXPORT int m2L(Cartesian<Real, localDegree, true>& local,
    Cartesian<Real, sourceDegree> const& source,
    Cartesian<Real, localDegree, true> const& derivatives, bool doPhi) {
    static_assert(localDegree == sourceDegree + 1, "Local degree is the source degree plus one");
    Real const oldPotential = local[0];
    detail::cartesian::EwaldM2lKernel<localDegree, centered>::template add<Real>(
        local, source, derivatives);
    if (!doPhi) {
        local[0] = oldPotential;
    }
    return detail::cartesian::EwaldM2lKernel<localDegree, centered>::kFlops_;
}

template <int order>
inline void initializeBackgroundExpansion(FmmExpansion<float, order>& local,
    FmmMultipole<float, order> const& rootMultipole, double secondMomentTrace) {
    double const coefficient = 4.0 * M_PI / 3.0;
    local = 0.0f;
    local[0] = -0.5 * coefficient * secondMomentTrace * SCALE_FACTOR_INV1;
    local(1, 0, 0) = coefficient * rootMultipole(1, 0, 0) * SCALE_FACTOR_INV3;
    local(0, 1, 0) = coefficient * rootMultipole(0, 1, 0) * SCALE_FACTOR_INV3;
    local(0, 0, 1) = coefficient * rootMultipole(0, 0, 1) * SCALE_FACTOR_INV3;
    local(2, 0, 0) = local(0, 2, 0) = -coefficient * rootMultipole[0] * SCALE_FACTOR_INV3;
    local[FmmOrder<order>::kLocalTraceIndex_] = -4.0 * M_PI * rootMultipole[0] * SCALE_FACTOR_INV3;
}
