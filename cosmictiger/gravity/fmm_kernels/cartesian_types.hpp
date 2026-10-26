/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/core/containers.hpp>

namespace cosmictiger::gravity::cartesian {
namespace detail {

    template <class Real, int degree, bool withTrace = false>
    class Coefficients : public Array<Real, (degree + 1) * (degree + 1) + int(withTrace)> {
    public:
        static constexpr int kDegree_ = degree;
        static constexpr int kCount_ = (degree + 1) * (degree + 1);
        static constexpr int kSize_ = kCount_ + int(withTrace);

        CUDA_EXPORT static constexpr int index(int nx, int ny, int nz) {
            int const rank = nx + ny + nz;
            return rank * rank + (nz == 0 ? nx : rank + 1 + nx);
        }
        CUDA_EXPORT Coefficients& operator=(Real const& value) {
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

}    // namespace detail

template <class Real, int order>
class Multipole : public detail::Coefficients<Real, (order - 2)> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 2)>;

public:
    using Base::operator=;
};

template <class Real, int order>
class Expansion : public detail::Coefficients<Real, (order - 1), true> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 1), true>;

public:
    using Base::operator=;
};

template <class Real, int order>
class Green : public detail::Coefficients<Real, (order - 1), true> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, (order - 1), true>;

public:
    using Base::operator=;
};

}    // namespace cosmictiger::gravity::cartesian
