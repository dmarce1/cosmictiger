/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cmath>

#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/gravity/stf/generated/generated_config.hpp>
#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_STF_TEST_HD __host__ __device__
#else
#define COSMICTIGER_STF_TEST_HD
#endif

namespace detail {

namespace kernels = cosmictiger::gravity::stf::detail;

// The build selects one multipole degree p (ORDER - 2); the local expansion
// carries degree p + 1. The generated kernels must have been produced for the
// same pair, otherwise the explicit specializations below do not exist.
constexpr int kMultipoleDegree = kFmmMultipoleDegree;
constexpr int kLocalDegree = kFmmLocalDegree;
static_assert(cosmictiger::gravity::stf::kMinOrder == kMultipoleDegree &&
        cosmictiger::gravity::stf::kMaxOrder == kLocalDegree,
    "The generated STF kernels were produced for a different ORDER than this build");

// Compact rank-major STF storage: (degree + 1)^2 harmonic coefficients. Local
// expansions append one constant Laplacian (the periodic background trace);
// only the trace-carrying L2L output writes that extra slot.
constexpr int kMultipoleCount = kFmmMultipoleCoefficients;
constexpr int kLocalHarmonicCount = kFmmLocalHarmonicCoefficients;
constexpr int kLocalTraceIndex = kFmmLocalTraceIndex;
constexpr int kLocalCount = kFmmLocalCoefficients;
constexpr int kFieldCount = 4;
constexpr int kGeometryCount = 12;

template <typename Real, int count>
struct Coefficients {
    Real values[count]{};

    COSMICTIGER_STF_TEST_HD Real& operator[](int index) {
        return values[index];
    }

    COSMICTIGER_STF_TEST_HD Real const& operator[](int index) const {
        return values[index];
    }
};

constexpr int kP2mOffset = 0;
constexpr int kM2mOffset = kP2mOffset + kMultipoleCount;
constexpr int kP2lOffset = kM2mOffset + kMultipoleCount;
constexpr int kM2lOffset = kP2lOffset + kLocalHarmonicCount;
constexpr int kM2lCenteredOffset = kM2lOffset + kLocalHarmonicCount;
constexpr int kL2lOffset = kM2lCenteredOffset + kLocalHarmonicCount;
constexpr int kL2lHarmonicOffset = kL2lOffset + kLocalCount;
constexpr int kM2pOffset = kL2lHarmonicOffset + kLocalHarmonicCount;
constexpr int kM2pCenteredOffset = kM2pOffset + kFieldCount;
constexpr int kL2pOffset = kM2pCenteredOffset + kFieldCount;
constexpr int kL2pHarmonicOffset = kL2pOffset + kFieldCount;
constexpr int kEwaldM2lOffset = kL2pHarmonicOffset + kFieldCount;
constexpr int kEwaldM2lCenteredOffset = kEwaldM2lOffset + kLocalHarmonicCount;
constexpr int kResultCount = kEwaldM2lCenteredOffset + kLocalHarmonicCount;

template <typename Real>
struct KernelInput {
    Coefficients<Real, kMultipoleCount> multipole;
    Coefficients<Real, kLocalCount> local;
    // Unit direction, inverse distance, source/distance, target/distance,
    // normalized translation, scale ratio, particle mass, inverse local scale.
    Coefficients<Real, kGeometryCount> geometry;
};

template <typename Real>
using KernelResult = Coefficients<Real, kResultCount>;

template <typename Real>
COSMICTIGER_STF_TEST_HD void evaluateKernels(
    KernelInput<Real> const& input, KernelResult<Real>& result) {
    using Multipole = Coefficients<Real, kMultipoleCount>;
    auto const& g = input.geometry;
    auto* p2m = result.values + kP2mOffset;
    kernels::P2mKernel<kMultipoleDegree>::add(p2m, g[10], g[6], g[7], g[8]);
    auto* m2m = result.values + kM2mOffset;
    kernels::M2mKernel<kMultipoleDegree>::add(m2m, input.multipole, g[6], g[7], g[8], g[9]);
    auto* p2l = result.values + kP2lOffset;
    kernels::P2lKernel<kLocalDegree>::add(p2l, g[10], g[0], g[1], g[2], g[3], g[5]);
    auto* m2l = result.values + kM2lOffset;
    kernels::M2lKernel<kLocalDegree, false>::template add<Real, Real*, Multipole, kMultipoleDegree>(
        m2l, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* m2lCentered = result.values + kM2lCenteredOffset;
    kernels::M2lKernel<kLocalDegree, true>::template add<Real, Real*, Multipole, kMultipoleDegree>(
        m2lCentered, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* l2l = result.values + kL2lOffset;
    kernels::L2lKernel<kLocalDegree, true>::add(l2l, input.local, g[6], g[7], g[8], g[9]);
    auto* l2lHarmonic = result.values + kL2lHarmonicOffset;
    kernels::L2lKernel<kLocalDegree, false>::add(l2lHarmonic, input.local, g[6], g[7], g[8], g[9]);
    auto* m2p = result.values + kM2pOffset;
    kernels::M2pKernel<kMultipoleDegree, false>::add(
        m2p, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* m2pCentered = result.values + kM2pCenteredOffset;
    kernels::M2pKernel<kMultipoleDegree, true>::add(
        m2pCentered, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* l2p = result.values + kL2pOffset;
    kernels::L2pKernel<kLocalDegree, true>::add(l2p, input.local, g[6], g[7], g[8], g[11]);
    auto* l2pHarmonic = result.values + kL2pHarmonicOffset;
    kernels::L2pKernel<kLocalDegree, false>::add(l2pHarmonic, input.local, g[6], g[7], g[8], g[11]);
    auto* ewaldM2l = result.values + kEwaldM2lOffset;
    kernels::EwaldM2lKernel<kLocalDegree, false>::template add<Real>(
        ewaldM2l, input.multipole, input.local);
    auto* ewaldM2lCentered = result.values + kEwaldM2lCenteredOffset;
    kernels::EwaldM2lKernel<kLocalDegree, true>::template add<Real>(
        ewaldM2lCentered, input.multipole, input.local);
}

template <typename Real>
KernelInput<Real> makeKernelInput(int sample) {
    KernelInput<Real> input{};
    for (int i = 0; i < kMultipoleCount; ++i) {
        input.multipole[i] = Real(.012 * std::sin((sample + 1) * (i + 2) * .71));
    }
    input.multipole[0] = Real(1.0 + .013 * sample);
    for (int i = 0; i < kLocalCount; ++i) {
        input.local[i] = Real(.2 * std::cos((sample + 2) * (i + 1) * .43));
    }
    double const x = std::sin(.73 * sample + .11);
    double const y = std::cos(.39 * sample + .17);
    double const z = std::sin(.53 * sample + .29);
    double const inverseNorm = 1 / std::sqrt(x * x + y * y + z * z);
    input.geometry[0] = Real(x * inverseNorm);
    input.geometry[1] = Real(y * inverseNorm);
    input.geometry[2] = Real(z * inverseNorm);
    input.geometry[3] = Real(.41 + .002 * sample);
    input.geometry[4] = Real(.17 + .001 * sample);
    input.geometry[5] = Real(.23 + .001 * sample);
    input.geometry[6] = Real(.13 * x);
    input.geometry[7] = Real(.09 * y);
    input.geometry[8] = Real(.11 * z);
    input.geometry[9] = Real(.67 + .003 * sample);
    input.geometry[10] = Real(.7 + .011 * sample);
    input.geometry[11] = Real(1.1 + .002 * sample);
    return input;
}

}    // namespace detail

#undef COSMICTIGER_STF_TEST_HD
