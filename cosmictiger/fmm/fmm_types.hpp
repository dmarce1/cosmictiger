/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/math/fixed.hpp>

// Order-dependent FMM value types. This header is light: it carries no
// generated kernels, so order-agnostic code (tree nodes, work items, options)
// can include it everywhere. The operators live in fmm_kernels.hpp.

// Compact rank-major STF storage. Only local expansions carry a Laplacian;
// source moments contain exactly (degree + 1)^2 coefficients, including dipoles.
template <class Real, int degree, bool withTrace = false>
class StfTensor : public Array<Real, (degree + 1) * (degree + 1) + int(withTrace)> {
public:
    static constexpr int kDegree_ = degree;
    static constexpr int kCount_ = (degree + 1) * (degree + 1);
    static constexpr int kSize_ = kCount_ + int(withTrace);
    CUDA_EXPORT static constexpr int index(int nx, int ny, int nz) {
        int const rank = nx + ny + nz;
        return rank * rank + (nz == 0 ? nx : rank + 1 + nx);
    }
    CUDA_EXPORT StfTensor& operator=(Real const& value) {
        for (int i = 0; i < kSize_; i++) {
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

// Degree-p source multipole and degree-(p + 1) local expansion of one FMM order.
template <class Real, int order>
using FmmMultipole = StfTensor<Real, fmmMultipoleDegree(order)>;
template <class Real, int order>
using FmmExpansion = StfTensor<Real, fmmLocalDegree(order), true>;
// Potential plus gradient at a point.
template <class Real>
using Expansion2 = StfTensor<Real, 1>;

static_assert(sizeof(FmmMultipole<float, 7>) == 4 * fmmMultipoleCoefficientCount(7));
static_assert(sizeof(FmmExpansion<float, 7>) == fmmLocalExpansionBytes(7));

// Positioned multipole record. The center comes first so order-agnostic tree
// code reads it through MultipolePosition; the per-order view adds the moments.
struct MultipolePosition {
    Array<Fixed32, NDIM> pos;
};

template <int order>
struct FmmMultiPos {
    Array<Fixed32, NDIM> pos;
    FmmMultipole<float, order> multi;
};

static_assert(std::is_standard_layout_v<FmmMultiPos<kFmmDefaultOrder>>);
static_assert(sizeof(FmmMultiPos<7>) == fmmPositionedMultipoleBytes(7));
static_assert(sizeof(FmmMultiPos<9>) == fmmPositionedMultipoleBytes(9));

template <int order>
CUDA_EXPORT inline FmmMultiPos<order>& fmmMultiPos(MultipolePosition* record) {
    return *reinterpret_cast<FmmMultiPos<order>*>(record);
}

template <int order>
CUDA_EXPORT inline FmmMultiPos<order> const& fmmMultiPos(MultipolePosition const* record) {
    return *reinterpret_cast<FmmMultiPos<order> const*>(record);
}

// Transport buffers sized for the largest order. Order-agnostic interfaces
// (kick work items, HPX actions, tree returns) pass these; the per-order code
// views the leading coefficients through expansion<order>() / multipole<order>().
struct ExpansionBuffer {
    Array<float, kFmmMaxLocalCoefficients> coefficients;

    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxLocalCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    template <int order>
    CUDA_EXPORT FmmExpansion<float, order>& expansion() {
        return *reinterpret_cast<FmmExpansion<float, order>*>(coefficients.data());
    }
    template <int order>
    CUDA_EXPORT FmmExpansion<float, order> const& expansion() const {
        return *reinterpret_cast<FmmExpansion<float, order> const*>(coefficients.data());
    }
    template <int order>
    CUDA_EXPORT void assign(FmmExpansion<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int const count = fmmLocalCoefficientCount(fmmOrder());
        for (int i = 0; i < count; i++) {
            arc & coefficients[i];
        }
    }
};

struct MultipoleBuffer {
    Array<float, kFmmMaxMultipoleCoefficients> coefficients;

    CUDA_EXPORT float& operator[](int i) {
        return coefficients[i];
    }
    CUDA_EXPORT float const& operator[](int i) const {
        return coefficients[i];
    }
    CUDA_EXPORT void zero() {
        for (int i = 0; i < kFmmMaxMultipoleCoefficients; i++) {
            coefficients[i] = 0.0f;
        }
    }
    template <int order>
    CUDA_EXPORT FmmMultipole<float, order>& multipole() {
        return *reinterpret_cast<FmmMultipole<float, order>*>(coefficients.data());
    }
    template <int order>
    CUDA_EXPORT FmmMultipole<float, order> const& multipole() const {
        return *reinterpret_cast<FmmMultipole<float, order> const*>(coefficients.data());
    }
    template <int order>
    CUDA_EXPORT void assign(FmmMultipole<float, order> const& value) {
        for (int i = 0; i < FmmOrder<order>::kMultipoleCoefficients_; i++) {
            coefficients[i] = value[i];
        }
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        int const count = fmmMultipoleCoefficientCount(fmmOrder());
        for (int i = 0; i < count; i++) {
            arc & coefficients[i];
        }
    }
};
