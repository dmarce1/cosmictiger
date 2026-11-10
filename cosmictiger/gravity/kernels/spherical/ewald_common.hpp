/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cmath>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace cosmictiger::gravity::spherical {
namespace detail {
    template <typename Real>
    class EwaldWorkingType {
    public:
        using Type = Real;
    };
    template <>
    class EwaldWorkingType<float> {
    public:
        using Type = double;
    };
#ifndef __CUDACC__
    template <int lanes>
    class EwaldWorkingType<SimdVector<float, lanes>> {
    public:
        using Type = SimdVector<double, lanes>;
    };
#endif
    template <typename Destination, typename Source>
    CUDA_EXPORT inline Destination ewaldConvert(Source const& value) {
        using namespace math;
        if constexpr (std::is_same_v<Destination, Source>) {
            return value;
        } else if constexpr (std::is_arithmetic_v<Destination>) {
            return static_cast<Destination>(value);
        } else {
            return Destination([&](auto lane) {
                return static_cast<typename Destination::value_type>(value[lane]);
            });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldExp(Real const& value) {
        using namespace math;
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::exp(value);
        } else {
            return Real([&](auto lane) { return std::exp(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldErfc(Real const& value) {
        using namespace math;
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::erfc(value);
        } else {
            return Real([&](auto lane) { return std::erfc(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldCos(Real const& value) {
        using namespace math;
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::cos(value);
        } else {
            return Real([&](auto lane) { return std::cos(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldSin(Real const& value) {
        using namespace math;
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::sin(value);
        } else {
            return Real([&](auto lane) { return std::sin(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline void ewaldAccumulate(Real& sum, Real& correction, Real value) {
        using namespace math;
        Real const adjusted = value - correction;
        Real const next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
}    // namespace detail
template <int degree>
class EwaldGreen;
}    // namespace cosmictiger::gravity::spherical
