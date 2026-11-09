/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cmath>
#include <utility>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>

namespace detail {

namespace kernels = cosmictiger::gravity::cartesian::tracefree::detail;

// Every FMM order from kFmmMinOrder to kFmmMaxOrder is compiled into the
// binary. An order selects the multipole degree p = order - 2 and the local
// degree p + 1; the generated kernels must cover that whole range.
static_assert(
    cosmictiger::gravity::cartesian::tracefree::kMinLocalDegree <= fmmLocalDegree(kFmmMinOrder) &&
        cosmictiger::gravity::cartesian::tracefree::kMaxLocalDegree >= fmmLocalDegree(kFmmMaxOrder),
    "The generated Cartesian kernels do not cover FMM orders 3 to 9");

constexpr int kFieldCount = 4;
constexpr int kGeometryCount = 9;
constexpr int kOrderCount = kFmmMaxOrder - kFmmMinOrder + 1;

// Compact rank-major Cartesian storage: (degree + 1)^2 harmonic coefficients.
// Local expansions append one constant Laplacian (the periodic background trace).
// The result buffer concatenates every kernel
// output of one order at these offsets.
template <int order>
struct KernelLayout {
    using Order = FmmOrder<order>;
    static constexpr int kMultipoleDegree_ = Order::kMultipoleDegree_;
    static constexpr int kLocalDegree_ = Order::kLocalDegree_;
    static constexpr int kMultipoleCount_ = Order::kMultipoleCoefficients_;
    static constexpr int kLocalHarmonicCount_ = Order::kLocalHarmonicCoefficients_;
    static constexpr int kLocalCount_ = Order::kLocalCoefficients_;
    static constexpr int kP2mOffset_ = 0;
    static constexpr int kM2mOffset_ = kP2mOffset_ + kMultipoleCount_;
    static constexpr int kP2lOffset_ = kM2mOffset_ + kMultipoleCount_;
    static constexpr int kM2lOffset_ = kP2lOffset_ + kLocalHarmonicCount_;
    static constexpr int kL2lOffset_ = kM2lOffset_ + kLocalHarmonicCount_;
    static constexpr int kM2pOffset_ = kL2lOffset_ + kLocalCount_;
    static constexpr int kL2pOffset_ = kM2pOffset_ + kFieldCount;
    static constexpr int kEwaldM2lOffset_ = kL2pOffset_ + kFieldCount;
    static constexpr int kResultCount_ = kEwaldM2lOffset_ + kLocalHarmonicCount_;
};

template <typename Real, int count>
struct Coefficients {
    Real values[count]{};

    CUDA_EXPORT Real& operator[](int index) {
        return values[index];
    }

    CUDA_EXPORT Real const& operator[](int index) const {
        return values[index];
    }
};

template <typename Real, int order>
struct KernelInput {
    cosmictiger::gravity::cartesian::tracefree::Multipole<Real, order> multipole{};
    cosmictiger::gravity::cartesian::tracefree::Expansion<Real, order> local{};
    // Separation, center translation, translation scale ratio, particle mass,
    // and inverse local scale.
    Coefficients<Real, kGeometryCount> geometry;
};

template <typename Real, int order>
using KernelResult = Coefficients<Real, KernelLayout<order>::kResultCount_>;

template <typename Real, int order>
CUDA_EXPORT void evaluateKernels(
    KernelInput<Real, order> const& input, KernelResult<Real, order>& result) {
    using Layout = KernelLayout<order>;
    using Multipole = cosmictiger::gravity::cartesian::tracefree::Multipole<Real, order>;
    using Expansion = cosmictiger::gravity::cartesian::tracefree::Expansion<Real, order>;
    using Green = cosmictiger::gravity::cartesian::tracefree::Green<Real, order>;
    using Field = cosmictiger::gravity::Field<Real>;
    auto const& g = input.geometry;
    Vector3<Real> const separation{g[0], g[1], g[2]};
    Vector3<Real> const translation{g[3], g[4], g[5]};
    auto store = [&](auto const& value, int offset, int count) {
        for (int i = 0; i < count; i++) {
            result[offset + i] = value[i];
        }
    };
    Multipole p2m{};
    kernels::P2M(p2m, g[7], translation);
    store(p2m, Layout::kP2mOffset_, Layout::kMultipoleCount_);
    Multipole m2m{};
    kernels::M2M(m2m, input.multipole, translation, g[6]);
    store(m2m, Layout::kM2mOffset_, Layout::kMultipoleCount_);
    Expansion p2l{};
    kernels::P2L(p2l, g[7], separation);
    store(p2l, Layout::kP2lOffset_, Layout::kLocalHarmonicCount_);
    Expansion m2l{};
    kernels::M2L(m2l, input.multipole, separation);
    store(m2l, Layout::kM2lOffset_, Layout::kLocalHarmonicCount_);
    Expansion l2l{};
    kernels::L2L(l2l, input.local, translation, g[6]);
    store(l2l, Layout::kL2lOffset_, Layout::kLocalCount_);
    Field m2p{};
    kernels::M2P(m2p, input.multipole, separation);
    store(m2p, Layout::kM2pOffset_, kFieldCount);
    Field l2p{};
    kernels::L2P(l2p, input.local, translation, g[8]);
    store(l2p, Layout::kL2pOffset_, kFieldCount);
    Green derivatives{};
    for (int i = 0; i < Layout::kLocalCount_; i++) {
        derivatives[i] = input.local[i];
    }
    Expansion ewald{};
    kernels::M2L(ewald, input.multipole, derivatives);
    store(ewald, Layout::kEwaldM2lOffset_, Layout::kLocalHarmonicCount_);
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
    input.geometry[0] = Real(x);
    input.geometry[1] = Real(y);
    input.geometry[2] = Real(z);
    input.geometry[3] = Real(.13 * x);
    input.geometry[4] = Real(.09 * y);
    input.geometry[5] = Real(.11 * z);
    input.geometry[6] = Real(.67 + .003 * sample);
    input.geometry[7] = Real(.7 + .011 * sample);
    input.geometry[8] = Real(1.1 + .002 * sample);
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
