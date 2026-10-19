/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <cmath>
#include <utility>

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

// Every FMM order from kFmmMinOrder to kFmmMaxOrder is compiled into the
// binary. An order selects the multipole degree p = order - 2 and the local
// degree p + 1; the generated kernels must cover that whole range.
static_assert(cosmictiger::gravity::stf::kMinLocalDegree <= fmmLocalDegree(kFmmMinOrder) &&
        cosmictiger::gravity::stf::kMaxLocalDegree >= fmmLocalDegree(kFmmMaxOrder),
    "The generated STF kernels do not cover FMM orders 3 to 9");

constexpr int kFieldCount = 4;
constexpr int kGeometryCount = 12;
constexpr int kOrderCount = kFmmMaxOrder - kFmmMinOrder + 1;

// Compact rank-major STF storage: (degree + 1)^2 harmonic coefficients. Local
// expansions append one constant Laplacian (the periodic background trace);
// only the trace-carrying L2L output writes that extra slot. The result buffer
// concatenates every kernel output of one order at these offsets.
template <int order>
struct KernelLayout {
    using Order = FmmOrder<order>;
    static constexpr int kMultipoleDegree_ = Order::kMultipoleDegree_;
    static constexpr int kLocalDegree_ = Order::kLocalDegree_;
    static constexpr int kMultipoleCount_ = Order::kMultipoleCoefficients_;
    static constexpr int kLocalHarmonicCount_ = Order::kLocalHarmonicCoefficients_;
    static constexpr int kLocalTraceIndex_ = Order::kLocalTraceIndex_;
    static constexpr int kLocalCount_ = Order::kLocalCoefficients_;
    static constexpr int kP2mOffset_ = 0;
    static constexpr int kM2mOffset_ = kP2mOffset_ + kMultipoleCount_;
    static constexpr int kP2lOffset_ = kM2mOffset_ + kMultipoleCount_;
    static constexpr int kM2lOffset_ = kP2lOffset_ + kLocalHarmonicCount_;
    static constexpr int kM2lCenteredOffset_ = kM2lOffset_ + kLocalHarmonicCount_;
    static constexpr int kL2lOffset_ = kM2lCenteredOffset_ + kLocalHarmonicCount_;
    static constexpr int kL2lHarmonicOffset_ = kL2lOffset_ + kLocalCount_;
    static constexpr int kM2pOffset_ = kL2lHarmonicOffset_ + kLocalHarmonicCount_;
    static constexpr int kM2pCenteredOffset_ = kM2pOffset_ + kFieldCount;
    static constexpr int kL2pOffset_ = kM2pCenteredOffset_ + kFieldCount;
    static constexpr int kL2pHarmonicOffset_ = kL2pOffset_ + kFieldCount;
    static constexpr int kEwaldM2lOffset_ = kL2pHarmonicOffset_ + kFieldCount;
    static constexpr int kEwaldM2lCenteredOffset_ = kEwaldM2lOffset_ + kLocalHarmonicCount_;
    static constexpr int kResultCount_ = kEwaldM2lCenteredOffset_ + kLocalHarmonicCount_;
};

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

template <typename Real, int order>
struct KernelInput {
    Coefficients<Real, KernelLayout<order>::kMultipoleCount_> multipole;
    Coefficients<Real, KernelLayout<order>::kLocalCount_> local;
    // Unit direction, inverse distance, source/distance, target/distance,
    // normalized translation, scale ratio, particle mass, inverse local scale.
    Coefficients<Real, kGeometryCount> geometry;
};

template <typename Real, int order>
using KernelResult = Coefficients<Real, KernelLayout<order>::kResultCount_>;

template <typename Real, int order>
COSMICTIGER_STF_TEST_HD void evaluateKernels(
    KernelInput<Real, order> const& input, KernelResult<Real, order>& result) {
    using Layout = KernelLayout<order>;
    using Multipole = Coefficients<Real, Layout::kMultipoleCount_>;
    constexpr int kMultipoleDegree = Layout::kMultipoleDegree_;
    constexpr int kLocalDegree = Layout::kLocalDegree_;
    auto const& g = input.geometry;
    auto* p2m = result.values + Layout::kP2mOffset_;
    kernels::P2mKernel<kMultipoleDegree>::add(p2m, g[10], g[6], g[7], g[8]);
    auto* m2m = result.values + Layout::kM2mOffset_;
    kernels::M2mKernel<kMultipoleDegree>::add(m2m, input.multipole, g[6], g[7], g[8], g[9]);
    auto* p2l = result.values + Layout::kP2lOffset_;
    kernels::P2lKernel<kLocalDegree>::add(p2l, g[10], g[0], g[1], g[2], g[3], g[5]);
    auto* m2l = result.values + Layout::kM2lOffset_;
    kernels::M2lKernel<kLocalDegree, false>::template add<Real, Real*, Multipole, kMultipoleDegree>(
        m2l, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* m2lCentered = result.values + Layout::kM2lCenteredOffset_;
    kernels::M2lKernel<kLocalDegree, true>::template add<Real, Real*, Multipole, kMultipoleDegree>(
        m2lCentered, input.multipole, g[0], g[1], g[2], g[3], g[4], g[5]);
    auto* l2l = result.values + Layout::kL2lOffset_;
    kernels::L2lKernel<kLocalDegree, true>::add(l2l, input.local, g[6], g[7], g[8], g[9]);
    auto* l2lHarmonic = result.values + Layout::kL2lHarmonicOffset_;
    kernels::L2lKernel<kLocalDegree, false>::add(l2lHarmonic, input.local, g[6], g[7], g[8], g[9]);
    auto* m2p = result.values + Layout::kM2pOffset_;
    kernels::M2pKernel<kMultipoleDegree, false>::add(
        m2p, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* m2pCentered = result.values + Layout::kM2pCenteredOffset_;
    kernels::M2pKernel<kMultipoleDegree, true>::add(
        m2pCentered, input.multipole, g[0], g[1], g[2], g[3], g[4]);
    auto* l2p = result.values + Layout::kL2pOffset_;
    kernels::L2pKernel<kLocalDegree, true>::add(l2p, input.local, g[6], g[7], g[8], g[11]);
    auto* l2pHarmonic = result.values + Layout::kL2pHarmonicOffset_;
    kernels::L2pKernel<kLocalDegree, false>::add(l2pHarmonic, input.local, g[6], g[7], g[8], g[11]);
    auto* ewaldM2l = result.values + Layout::kEwaldM2lOffset_;
    kernels::EwaldM2lKernel<kLocalDegree, false>::template add<Real>(
        ewaldM2l, input.multipole, input.local);
    auto* ewaldM2lCentered = result.values + Layout::kEwaldM2lCenteredOffset_;
    kernels::EwaldM2lKernel<kLocalDegree, true>::template add<Real>(
        ewaldM2lCentered, input.multipole, input.local);
}

template <typename Real, int order>
KernelInput<Real, order> makeKernelInput(int sample) {
    using Layout = KernelLayout<order>;
    KernelInput<Real, order> input{};
    for (int i = 0; i < Layout::kMultipoleCount_; ++i) {
        input.multipole[i] = Real(.012 * std::sin((sample + 1) * (i + 2) * .71));
    }
    input.multipole[0] = Real(1.0 + .013 * sample);
    for (int i = 0; i < Layout::kLocalCount_; ++i) {
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

// Call f(std::integral_constant<int, order>{}) for every supported FMM order,
// from kFmmMinOrder upward. The standalone tests do not link the application's
// assertion handler, so they do not use fmmDispatch.
template <class F, int... offsets>
void forEachFmmOrderImpl(F&& f, std::integer_sequence<int, offsets...>) {
    (f(std::integral_constant<int, kFmmMinOrder + offsets>{}), ...);
}

template <class F>
void forEachFmmOrder(F&& f) {
    forEachFmmOrderImpl(std::forward<F>(f), std::make_integer_sequence<int, kOrderCount>{});
}

}    // namespace detail

#undef COSMICTIGER_STF_TEST_HD
