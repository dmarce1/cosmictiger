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
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>

namespace detail {

namespace kernels = cosmictiger::gravity::cartesian;

// Kernels and fixtures must agree on the single build degree.
static_assert(cosmictiger::gravity::cartesian::kMinLocalDegree <= fmmLocalDegree(kFmmOrder) &&
        cosmictiger::gravity::cartesian::kMaxLocalDegree >= fmmLocalDegree(kFmmOrder),
    "The generated Cartesian kernels do not cover FMM orders 3 to 9");

constexpr int kFieldCount = 4;
constexpr int kGeometryCount = 8;
constexpr int kOrderCount = 1;

// Compact rank-major Cartesian storage, with the actual zz component appended
// when independent quadratic trace storage is enabled.
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
    static constexpr int kM2lOffset_ = kP2lOffset_ + kLocalCount_;
    static constexpr int kEwaldM2lOffset_ = kM2lOffset_ + kLocalCount_;
    static constexpr int kL2lOffset_ = kEwaldM2lOffset_ + kLocalCount_;
    static constexpr int kM2pOffset_ = kL2lOffset_ + kLocalCount_;
    static constexpr int kL2pOffset_ = kM2pOffset_ + kFieldCount;
    static constexpr int kResultCount_ = kL2pOffset_ + kFieldCount;
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
    cosmictiger::gravity::cartesian::Multipole<Real> multipole{};
    cosmictiger::gravity::cartesian::Expansion<Real> local{};
    // Separation, center translation, unit translation scale ratio, particle mass.
    Coefficients<Real, kGeometryCount> geometry;
};

template <typename Real, int order>
using KernelResult = Coefficients<Real, KernelLayout<order>::kResultCount_>;

template <typename Real, int order>
CUDA_EXPORT void evaluateKernels(
    KernelInput<Real, order> const& input, KernelResult<Real, order>& result) {
    using Layout = KernelLayout<order>;
    using Multipole = cosmictiger::gravity::cartesian::Multipole<Real>;
    using Expansion = cosmictiger::gravity::cartesian::Expansion<Real>;
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
    kernels::P2M(g[7], translation, p2m);
    store(p2m, Layout::kP2mOffset_, Layout::kMultipoleCount_);
    auto const& Mi = input.multipole;
    Multipole Mo{};
    kernels::M2M(Mi, translation, g[6], Mo);
    store(Mo, Layout::kM2mOffset_, Layout::kMultipoleCount_);
    Expansion p2l{};
    kernels::P2L(g[7], separation, p2l);
    store(p2l, Layout::kP2lOffset_, Layout::kLocalCount_);
    Expansion m2l{};
    kernels::M2L(input.multipole, separation, m2l);
    store(m2l, Layout::kM2lOffset_, Layout::kLocalCount_);
    cosmictiger::gravity::cartesian::Green<Real> green{};
    for (int i = 0; i < Layout::kLocalCount_; ++i) {
        green[i] = input.local[i];
    }
    Expansion ewaldM2l{};
    kernels::M2Lewald(input.multipole, green, ewaldM2l);
    store(ewaldM2l, Layout::kEwaldM2lOffset_, Layout::kLocalCount_);
    auto const& Li = input.local;
    Expansion Lo{};
    kernels::L2L(Li, translation, g[6], Lo);
    store(Lo, Layout::kL2lOffset_, Layout::kLocalCount_);
    Field m2p{};
    kernels::M2P(input.multipole, separation, m2p);
    store(m2p, Layout::kM2pOffset_, kFieldCount);
    Field l2p{};
    kernels::L2P(input.local, translation, l2p);
    store(l2p, Layout::kL2pOffset_, kFieldCount);
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
    input.geometry[6] = Real(1);
    input.geometry[7] = Real(.7 + .011 * sample);
    return input;
}

// Call f(std::integral_constant<int, order>{}) for every supported FMM order,
// at the build order.
template <class F>
void forEachFmmOrder(F&& f) {
    std::forward<F>(f)(std::integral_constant<int, kFmmOrder>{});
}

}    // namespace detail
