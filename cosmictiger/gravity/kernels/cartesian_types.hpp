/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/core/containers.hpp>

namespace cosmictiger::gravity::cartesian {

CUDA_EXPORT constexpr int coefficientCount(int degree) {
    return (degree + 1) * (degree + 2) * (degree + 3) / 6;
}

CUDA_EXPORT constexpr int index(int nx, int ny, int nz) {
    int const rank = nx + ny + nz;
    return rank * (rank + 1) * (rank + 2) / 6 + nx * (rank + 1) - nx * (nx - 1) / 2 + ny;
}

namespace detail {

    // One entry for every symmetric Cartesian multi-index, including all traces.
    // No harmonic reconstruction or implicit conversion to tracefree storage.
    template <class Real, int degree>
    class Coefficients : public Array<Real, coefficientCount(degree)> {
    public:
        static_assert(degree >= 0);
        static constexpr int kDegree_ = degree;
        static constexpr int kCount_ = coefficientCount(degree);
        static constexpr int kSize_ = kCount_;

        CUDA_EXPORT static constexpr int index(int nx, int ny, int nz) {
            return cartesian::index(nx, ny, nz);
        }
        CUDA_EXPORT Coefficients& operator=(Real const& value) {
            for (int component = 0; component < kSize_; ++component) {
                (*this)[component] = value;
            }
            return *this;
        }
        CUDA_EXPORT Real& operator()(int nx, int ny, int nz) {
            ASSERT(nx >= 0 && ny >= 0 && nz >= 0 && nx + ny + nz <= degree);
            return (*this)[index(nx, ny, nz)];
        }
        CUDA_EXPORT Real const& operator()(int nx, int ny, int nz) const {
            ASSERT(nx >= 0 && ny >= 0 && nz >= 0 && nx + ny + nz <= degree);
            return (*this)[index(nx, ny, nz)];
        }
    };

}    // namespace detail

// Raw moments sum(mass * offset^alpha), through total degree order - 2.
template <class Real, int order>
class Multipole : public detail::Coefficients<Real, order - 2> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, order - 2>;

public:
    CUDA_EXPORT Multipole& operator=(Real const& value) {
        Base::operator=(value);
        return *this;
    }
};

// Unscaled derivatives of the negative potential, through degree order - 1.
template <class Real, int order>
class Expansion : public detail::Coefficients<Real, order - 1> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, order - 1>;

public:
    CUDA_EXPORT Expansion& operator=(Real const& value) {
        Base::operator=(value);
        return *this;
    }
};

template <class Real, int order>
class Green : public detail::Coefficients<Real, order - 1> {
    static_assert(order >= 3 && order <= 9);
    using Base = detail::Coefficients<Real, order - 1>;

public:
    CUDA_EXPORT Green& operator=(Real const& value) {
        Base::operator=(value);
        return *this;
    }
};

}    // namespace cosmictiger::gravity::cartesian
