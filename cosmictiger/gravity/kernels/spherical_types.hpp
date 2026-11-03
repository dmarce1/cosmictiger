/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>

namespace cosmictiger::gravity::spherical {
namespace detail {

    // Slot n(n+1)+m: m>=0 is real; m<0 is the imaginary part of +|m|.
    template <class Real, int degree>
    class Coefficients {
        Real coefficients_[(degree + 1) * (degree + 1)]{};

    public:
        static constexpr int kDegree_ = degree;
        static constexpr int kCount_ = (degree + 1) * (degree + 1);
        CUDA_EXPORT static constexpr int index(int n, int m) {
            return n * (n + 1) + m;
        }
        CUDA_EXPORT Real& operator[](int i) {
            return coefficients_[i];
        }
        CUDA_EXPORT Real const& operator[](int i) const {
            return coefficients_[i];
        }
        CUDA_EXPORT Real* data() {
            return coefficients_;
        }
        CUDA_EXPORT Real const* data() const {
            return coefficients_;
        }
        CUDA_EXPORT Coefficients& operator=(Real const& value) {
            for (int i = 0; i < kCount_; i++) {
                coefficients_[i] = value;
            }
            return *this;
        }
    };

}    // namespace detail

template <class Real, int order>
class Multipole : public detail::Coefficients<Real, (order - 2)> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 2)>;

public:
    using Base::operator=;
};

template <class Real, int order>
class Expansion : public detail::Coefficients<Real, (order - 1)> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 1)>;

public:
    using Base::operator=;
};

template <class Real, int order>
class Green : public detail::Coefficients<Real, (order - 1)> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 1)>;

public:
    using Base::operator=;
};

}    // namespace cosmictiger::gravity::spherical
