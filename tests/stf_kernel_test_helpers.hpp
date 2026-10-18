/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cmath>

#include <cosmictiger/gravity/stf/generated/generated_m2l.hpp>
#include <cosmictiger/gravity/stf/generated/generated_operators.hpp>

#ifdef __CUDACC__
#define COSMICTIGER_STF_TEST_HD __host__ __device__
#else
#define COSMICTIGER_STF_TEST_HD
#endif

namespace detail {

namespace kernels = cosmictiger::gravity::stf::detail;

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
constexpr int kM2mOffset = kP2mOffset + 36;
constexpr int kP2lOffset = kM2mOffset + 36;
constexpr int kM2lOffset = kP2lOffset + 49;
constexpr int kM2lCenteredOffset = kM2lOffset + 49;
constexpr int kL2lOffset = kM2lCenteredOffset + 49;
constexpr int kL2lHarmonicOffset = kL2lOffset + 50;
constexpr int kM2pOffset = kL2lHarmonicOffset + 49;
constexpr int kM2pCenteredOffset = kM2pOffset + 4;
constexpr int kL2pOffset = kM2pCenteredOffset + 4;
constexpr int kL2pHarmonicOffset = kL2pOffset + 4;
constexpr int kEwaldM2lOffset = kL2pHarmonicOffset + 4;
constexpr int kEwaldM2lCenteredOffset = kEwaldM2lOffset + 49;
constexpr int kResultCount = kEwaldM2lCenteredOffset + 49;

template <typename Real>
struct KernelInput {
    Coefficients<Real, 36> multipole;
    Coefficients<Real, 50> local;
    // Unit direction, inverse distance, source/distance, target/distance,
    // normalized translation, scale ratio, particle mass, inverse local scale.
    Coefficients<Real, 12> geometry;
};

template <typename Real>
using KernelResult = Coefficients<Real, kResultCount>;

template <typename Real>
COSMICTIGER_STF_TEST_HD void evaluateKernels(
    KernelInput<Real> const& input, KernelResult<Real>& result) {
    auto const& g = input.geometry;
    auto* p2m = result.values + kP2mOffset;
    kernels::P2mKernel<5>::add(p2m, g[10], g[6], g[7], g[8]);
    auto* m2m = result.values + kM2mOffset;
    kernels::M2mKernel<5>::add(m2m, input.multipole, g[6], g[7], g[8], g[9]);
    auto* p2l = result.values + kP2lOffset;
    kernels::P2lKernel<6>::add(p2l, g[10], g[0], g[1], g[2], g[3], g[5]);
    auto* m2l = result.values + kM2lOffset;
    kernels::M2lKernel<6, false>::template add<Real, Real*, Coefficients<Real, 36>, 5>(
        m2l, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* m2lCentered = result.values + kM2lCenteredOffset;
    kernels::M2lKernel<6, true>::template add<Real, Real*, Coefficients<Real, 36>, 5>(
        m2lCentered, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* l2l = result.values + kL2lOffset;
    kernels::L2lKernel<6, true>::add(l2l, input.local, g[6], g[7], g[8], g[9]);
    auto* l2lHarmonic = result.values + kL2lHarmonicOffset;
    kernels::L2lKernel<6, false>::add(l2lHarmonic, input.local, g[6], g[7], g[8], g[9]);
    auto* m2p = result.values + kM2pOffset;
    kernels::M2pKernel<5, false>::add(m2p, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* m2pCentered = result.values + kM2pCenteredOffset;
    kernels::M2pKernel<5, true>::add(m2pCentered, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* l2p = result.values + kL2pOffset;
    kernels::L2pKernel<6, true>::add(l2p, input.local, g[6], g[7], g[8], g[11]);
    auto* l2pHarmonic = result.values + kL2pHarmonicOffset;
    kernels::L2pKernel<6, false>::add(l2pHarmonic, input.local, g[6], g[7], g[8], g[11]);
    auto* ewaldM2l = result.values + kEwaldM2lOffset;
    kernels::EwaldM2lKernel<6, false>::template add<Real>(ewaldM2l, input.multipole, input.local);
    auto* ewaldM2lCentered = result.values + kEwaldM2lCenteredOffset;
    kernels::EwaldM2lKernel<6, true>::template add<Real>(
        ewaldM2lCentered, input.multipole, input.local);
}

template <typename Real>
KernelInput<Real> makeKernelInput(int sample) {
    KernelInput<Real> input{};
    for (int i = 0; i < 36; ++i) {
        input.multipole[i] = Real(.012 * std::sin((sample + 1) * (i + 2) * .71));
    }
    input.multipole[0] = Real(1.0 + .013 * sample);
    for (int i = 0; i < 50; ++i) {
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
