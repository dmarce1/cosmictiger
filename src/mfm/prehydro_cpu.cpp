/*
 CosmicTiger - A cosmological N-Body code
 SIMD CPU MFM support, density, and persistent checked inverse-moment preparation.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <vector>

#include <cosmictiger/mfm/cpu.hpp>

namespace mfm {
namespace detail {

    template <typename Real, int dimensions>
    void invalidateGeometry(RunData<Real, dimensions> const& data, int sinkIndex) {
        data.sinkGeometryValid[sinkIndex] = 0;
        data.sinkInverseMoment[sinkIndex] = math::SymmetricMatrix<Real, dimensions>{};
    }

    /*
     * Hopkins (2015), arXiv:1409.7395, equation (26): adapt h to a fixed weighted neighbor
     * number. Derived with full-support kernel W=h⁻ᴰw(r/h) and the existing SPH unit-ball
     * convention (V_1=2, twice the paper's C_1=1):
     * ΣW=Σ𞁓w(q𞁑𞁓)=Nₛₑₜ/V_D, dΣW/dh𞁑=−Σ𞁓q𞁑𞁓w′(q𞁑𞁓)/h𞁑,
     * ρ𞁑=m h𞁑⁻ᴰ ΣW. Newton steps are bounded by ±h𞁑/5 and the retained bracket.
     * Symbol directory: h𞁑 → sink full support; hₘᵢₙ,hₘₐₓ → support bounds;
     * q𞁑𞁓 → normalized pair distance; ΣW → kernel weight sum; Nₛₑₜ → weighted target;
     * V_D → unit-ball volume; m → equal gas mass; ρ𞁑 → mass density.
     * Code names: abs_r𞁑𞁓=|r𞁑𞁓|; ih𞁑=h𞁑⁻¹; dΣW_dh𞁑=dΣW/dh𞁑.
     */
    template <typename Real, int dimensions>
    bool solveSmoothingLength(PrehydroParameters<Real> const& parameters,
        RunData<Real, dimensions> const& data, SphTreeNode const& targetNode,
        std::vector<int> const& sources, int particleIndex, Real& h𞁑) {
        Real const hₘᵢₙ = parameters.minimumSmoothingLength;
        Real const hₘₐₓ = parameters.maximumSmoothingLength;
        ALWAYS_ASSERT(finite(hₘᵢₙ) && finite(hₘₐₓ) && finite(h𞁑) && hₘᵢₙ > Real(0) &&
            hₘₐₓ >= hₘᵢₙ && hₘₐₓ <= Real(0.5) && finite(data.targetNeighborCount) &&
            data.targetNeighborCount > Real(0) && finite(parameters.smoothingLengthTolerance) &&
            parameters.smoothingLengthTolerance > Real(0));
        h𞁑 = std::clamp(h𞁑, hₘᵢₙ, hₘₐₓ);
        auto const paddedPosition = data.sourcePaddedPosition(particleIndex);
        std::vector<Pack<Real>> abs_r𞁑𞁓;
        abs_r𞁑𞁓.reserve((sources.size() + kCpuLaneCount - 1) / kCpuLaneCount);
        for (size_t first = 0; first < sources.size(); first += kCpuLaneCount) {
            abs_r𞁑𞁓.push_back(separations(data, particleIndex, sources, first).norm());
        }
        Real lower = hₘᵢₙ;
        Real upper = hₘₐₓ;
        Real const ΣWₛₑₜ = targetKernelSum<Real, dimensions>(data.targetNeighborCount);
        for (int iteration = 0; iteration < kMaximumSmoothingIterations; ++iteration) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                if (targetNode.outerBox.end[dim] <
                        RangeFixed(paddedPosition[dim]) + RangeFixed(h𞁑) + RangeFixed::min() ||
                    RangeFixed(paddedPosition[dim]) <
                        targetNode.outerBox.begin[dim] + RangeFixed(h𞁑) + RangeFixed::min()) {
                    return false;
                }
            }
            Pack<Real> const ih𞁑Pack(Real(1) / h𞁑);
            Pack<Real> ΣWPack(Real(0));
            Pack<Real> dΣW_dh𞁑Pack(Real(0));
            for (size_t first = 0; first < sources.size(); first += kCpuLaneCount) {
                auto const q𞁑𞁓 = abs_r𞁑𞁓[first / kCpuLaneCount] * ih𞁑Pack;
                auto const contributes =
                    validLanes<Real>(sources, first) && q𞁑𞁓 < Pack<Real>(Real(1));
                ΣWPack += math::select(
                    contributes, kernelWeight<Pack<Real>, dimensions>(q𞁑𞁓), Pack<Real>(Real(0)));
                dΣW_dh𞁑Pack -= math::select(contributes,
                    q𞁑𞁓 * kernelDerivative<Pack<Real>, dimensions>(q𞁑𞁓) * ih𞁑Pack,
                    Pack<Real>(Real(0)));
            }
            Real const ΣW = simdSum(ΣWPack);
            auto const publishInitialDensity = [&]() {
                if (parameters.initializeDensity) {
                    int const sinkIndex = targetNode.sinkParticleRange.first -
                        targetNode.particleRange.first + particleIndex;
                    data.sinkDensity[sinkIndex] =
                        data.gasMass * inverseVolume<Real, dimensions>(Real(1) / h𞁑) * ΣW;
                }
            };
            Real const residual = ΣW - ΣWₛₑₜ;
            Real const tolerance = parameters.smoothingLengthTolerance * ΣWₛₑₜ;
            if (hₘₐₓ >= Real(0.5) && h𞁑 >= hₘₐₓ && residual < -tolerance) {
                throw std::runtime_error(std::format("MFM smoothing length exceeds half the "
                                                     "periodic box: h={:e} sum={:e} target={:e}",
                    h𞁑, ΣW, ΣWₛₑₜ));
            }
            if (std::abs(residual) <= tolerance || (h𞁑 <= hₘᵢₙ && residual >= Real(0)) ||
                (h𞁑 >= hₘₐₓ && residual <= Real(0))) {
                publishInitialDensity();
                return true;
            }
            if (residual > Real(0)) {
                upper = h𞁑;
            } else {
                lower = h𞁑;
            }
            Real const dΣW_dh𞁑 = simdSum(dΣW_dh𞁑Pack);
            Real const maximumChange = h𞁑 / Real(5);
            Real change = dΣW_dh𞁑 > Real(0) ? -residual / dΣW_dh𞁑 : maximumChange;
            change = std::clamp(change, -maximumChange, maximumChange);
            Real next = std::clamp(h𞁑 + change, hₘᵢₙ, hₘₐₓ);
            if (next < lower || next > upper) {
                next = (lower + upper) / Real(2);
            }
            if (next == h𞁑) {
                publishInitialDensity();
                return true;
            }
            h𞁑 = next;
        }
        throw std::runtime_error(std::format(
            "MFM smoothing-length solve did not converge: h={:e} bounds=[{:e},{:e}] target={:e}",
            h𞁑, hₘᵢₙ, hₘₐₓ, data.targetNeighborCount));
    }

    // A pair is touched when r𞁑𞁓²<max(h𞁑²,h𞁓²) and the neighbor is active.
    // Symbol directory: r𞁑𞁓 → periodic pair separation; h𞁑,h𞁓 → sink/source support.
    // Code names: h2𞁑=h𞁑²; r2𞁑𞁓=r𞁑𞁓².
    template <typename Real, int dimensions>
    bool hasActiveNeighbor(PrehydroParameters<Real> const& parameters,
        RunData<Real, dimensions> const& data, std::vector<int> const& sources, int particleIndex,
        Real h𞁑) {
        Pack<Real> const h2𞁑(h𞁑 * h𞁑);
        for (size_t first = 0; first < sources.size(); first += kCpuLaneCount) {
            auto const r𞁑𞁓 = separations(data, particleIndex, sources, first);
            auto const r2𞁑𞁓 = r𞁑𞁓.dot(r𞁑𞁓);
            auto const h𞁓 = load(data.sourceSmoothingLength, sources, first, Real(0));
            Pack<Real> const active([&](auto lane) {
                size_t const index = first + size_t(lane);
                return index < sources.size() &&
                        parameters.isActiveRung(data.sourceRungs[sources[index]]) ?
                    Real(1) :
                    Real(0);
            });
            auto const intersects = r2𞁑𞁓 < math::max(h2𞁑, h𞁓 * h𞁓);
            if (math::anyOf(validLanes<Real>(sources, first) && active > Pack<Real>(Real(0)) &&
                    intersects)) {
                return true;
            }
        }
        return false;
    }

}    // namespace detail

template <typename Real, int dimensions>
PrehydroResult<Real> smoothingLengthsCpu(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, int targetNodeIndex) {
    PrehydroResult<Real> result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const sources = detail::neighbors(data, targetNode, false);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        if (active && !data.sinkFlags[sinkIndex].converged()) {
            detail::invalidateGeometry(data, sinkIndex);
            Real& h𞁑 = data.sinkSmoothingLength[sinkIndex];
            bool const converged = detail::solveSmoothingLength(
                parameters, data, targetNode, sources, particleIndex, h𞁑);
            data.sinkFlags[sinkIndex].setConverged(converged);
            if (converged) {
                result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, h𞁑);
                result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, h𞁑);
            } else {
                result.needsAnotherPass = true;
            }
        }
    }
    return result;
}

/*
 * Hopkins (2015), arXiv:1409.7395, equations (6)-(7), (13)-(14), with equal masses:
 * ψⱼ(xᵢ)=Wᵢⱼ/ΣₖWᵢₖ, Eᵢ=Σⱼψⱼ(xᵢ)(xⱼ−xᵢ)⊗(xⱼ−xᵢ), Bᵢ=Eᵢ⁻¹.
 * Derived full-support, dimensionless accumulation:
 * ρᵢ=m hᵢ⁻ᴰΣw, Qᵢ=Σⱼw(qᵢⱼ)qᵢⱼ⊗qᵢⱼ, Eᵢ=hᵢ² Qᵢ/Σw.
 * Symbol directory: hᵢ → full support; m → equal gas mass; ρᵢ → mass density;
 * rᵢⱼ → periodic separation; qᵢⱼ=rᵢⱼ/hᵢ → dimensionless separation;
 * Qᵢ → unnormalized dimensionless moment; Σw → kernel weight sum.
 * Code names: ihᵢ=hᵢ⁻¹; ΣW=Σw; ΣWPack, QᵢPack → SIMD partial sums.
 */
template <typename Real, int dimensions>
PrehydroResult<Real> densityAndGeometryCpu(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, int targetNodeIndex) {
    using Pack = detail::Pack<Real>;
    PrehydroResult<Real> result;
    auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
    auto const sources = detail::neighbors(data, targetNode, true);
    for (int particleIndex = targetNode.particleRange.first;
        particleIndex < targetNode.particleRange.second; ++particleIndex) {
        int const sinkIndex =
            targetNode.sinkParticleRange.first - targetNode.particleRange.first + particleIndex;
        bool const active =
            parameters.isActiveRung(data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
        Real& hᵢ = data.sinkSmoothingLength[sinkIndex];
        bool const semiactive =
            active || detail::hasActiveNeighbor(parameters, data, sources, particleIndex, hᵢ);
        data.sinkFlags[sinkIndex].setSemiactive(semiactive);
        if (!semiactive) {
            continue;
        }
        detail::invalidateGeometry(data, sinkIndex);
        if (!active && !data.sinkFlags[sinkIndex].converged() &&
            !detail::solveSmoothingLength(
                parameters, data, targetNode, sources, particleIndex, hᵢ)) {
            data.sinkFlags[sinkIndex].setConverged(false);
            result.needsAnotherPass = true;
            continue;
        }
        if (!finite(hᵢ) || !(hᵢ > Real(0))) {
            ++result.geometryFailures;
            continue;
        }
        Real const ihᵢ = Real(1) / hᵢ;
        Pack ΣWPack(Real(0));
        math::SymmetricMatrix<Pack, dimensions> QᵢPack;
        for (size_t first = 0; first < sources.size(); first += detail::kCpuLaneCount) {
            auto const rᵢⱼ = detail::separations(data, particleIndex, sources, first);
            auto const qᵢⱼ = rᵢⱼ * Pack(ihᵢ);
            Pack const w = math::select(detail::validLanes<Real>(sources, first),
                kernelWeight<Pack, dimensions>(qᵢⱼ.norm()), Pack(Real(0)));
            ΣWPack += w;
            for (int row = 0; row < dimensions; ++row) {
                for (int col = 0; col <= row; ++col) {
                    QᵢPack(row, col) += (w * qᵢⱼ[row]) * qᵢⱼ[col];
                }
            }
        }
        Real const ΣW = simdSum(ΣWPack);
        Real const ρᵢ = data.gasMass * inverseVolume<Real, dimensions>(ihᵢ) * ΣW;
        data.sinkDensity[sinkIndex] = ρᵢ;
        math::SymmetricMatrix<Real, dimensions> Qᵢ;
        for (int row = 0; row < dimensions; ++row) {
            for (int col = 0; col <= row; ++col) {
                Qᵢ(row, col) = simdSum(QᵢPack(row, col));
            }
        }
        bool const usable = finite(ρᵢ) && ρᵢ > Real(0) &&
            invertMoment(
                Qᵢ, ΣW, hᵢ, parameters.maximumMomentCondition, data.sinkInverseMoment[sinkIndex]);
        data.sinkGeometryValid[sinkIndex] = uint8_t(usable);
        if (!usable) {
            ++result.geometryFailures;
        }
        // A singular geometry is not a request to repeat the same support/neighbor search.
        data.sinkFlags[sinkIndex].setConverged(true);
        result.minimumSmoothingLength = std::min(result.minimumSmoothingLength, hᵢ);
        result.maximumSmoothingLength = std::max(result.maximumSmoothingLength, hᵢ);
    }
    return result;
}

#define COSMICTIGER_MFM_CPU_INSTANTIATE(Real, dimensions)                        \
    template PrehydroResult<Real> smoothingLengthsCpu<Real, dimensions>(         \
        PrehydroParameters<Real> const&, RunData<Real, dimensions> const&, int); \
    template PrehydroResult<Real> densityAndGeometryCpu<Real, dimensions>(       \
        PrehydroParameters<Real> const&, RunData<Real, dimensions> const&, int)

COSMICTIGER_MFM_CPU_INSTANTIATE(float, 1);
COSMICTIGER_MFM_CPU_INSTANTIATE(float, 2);
COSMICTIGER_MFM_CPU_INSTANTIATE(float, 3);
COSMICTIGER_MFM_CPU_INSTANTIATE(double, 1);
COSMICTIGER_MFM_CPU_INSTANTIATE(double, 2);
COSMICTIGER_MFM_CPU_INSTANTIATE(double, 3);

#undef COSMICTIGER_MFM_CPU_INSTANTIATE

}    // namespace mfm
