/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once
#include <cmath>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/gravity/fmm_kernels/spherical_kernels.hpp>
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
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::exp(value);
        } else {
            return Real([&](auto lane) { return std::exp(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldErfc(Real const& value) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::erfc(value);
        } else {
            return Real([&](auto lane) { return std::erfc(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldCos(Real const& value) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::cos(value);
        } else {
            return Real([&](auto lane) { return std::cos(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline Real ewaldSin(Real const& value) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return std::sin(value);
        } else {
            return Real([&](auto lane) { return std::sin(value[lane]); });
        }
    }
    template <typename Real>
    CUDA_EXPORT inline void ewaldAccumulate(Real& sum, Real& correction, Real value) {
        Real const adjusted = value - correction;
        Real const next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
}    // namespace detail
template <int degree>
class EwaldGreen;
template <int degree>
class EwaldM2L;
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 3) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 3)
template <>
class EwaldGreen<2> {
public:
    static constexpr int kFlops_ = 114444;
    static constexpr int kConventionalFlops_ = 119214;
    static constexpr int kDegree_ = 2;
    static constexpr int kCount_ = 9;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[9];
        Real compensation[9];
        Real imagePair[9];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[9];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 9; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<2> {
public:
    static constexpr int kFlops_ = 35;
    static constexpr int kConventionalFlops_ = 58;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<2>::template add<Real>(local, moments, greens);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 4) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 4)
template <>
class EwaldGreen<3> {
public:
    static constexpr int kFlops_ = 194963;
    static constexpr int kConventionalFlops_ = 206716;
    static constexpr int kDegree_ = 3;
    static constexpr int kCount_ = 16;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[16];
        Real compensation[16];
        Real imagePair[16];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[16];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 16; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<3> {
public:
    static constexpr int kFlops_ = 100;
    static constexpr int kConventionalFlops_ = 173;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<3>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 5) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 5)
template <>
class EwaldGreen<4> {
public:
    static constexpr int kFlops_ = 299055;
    static constexpr int kConventionalFlops_ = 320565;
    static constexpr int kDegree_ = 4;
    static constexpr int kCount_ = 25;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        Real const zFactor4_0 = simdBroadcast<Real>(7) * z;
        output[20] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor4_0, output[12], -radiusSquared * output[6]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        Real const zFactor4_1 = simdBroadcast<Real>(7) * z;
        output[21] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[13], -radiusSquared * output[7]);
        output[19] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[11], -radiusSquared * output[5]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        Real const zFactor4_2 = simdBroadcast<Real>(7) * z;
        output[22] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[14], -radiusSquared * output[8]);
        output[18] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[10], -radiusSquared * output[4]);
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
        output[23] = z * output[15];
        output[17] = z * output[9];
        output[24] = simdBroadcast<Real>(0.125) * simdFma(x, output[15], -y * output[9]);
        output[16] = simdBroadcast<Real>(0.125) * simdFma(y, output[15], x * output[9]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[25];
        Real compensation[25];
        Real imagePair[25];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        Real series4 = simdBroadcast<Real>(1);
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        Real const radial4 = simdBroadcast<Real>(9.7816742188851205) * exponential * series4;
        output[16] = simdBroadcast<Real>(40320) * radial4 * harmonics[16];
        compensation[16] = simdBroadcast<Real>(0);
        output[17] = simdBroadcast<Real>(5040) * radial4 * harmonics[17];
        compensation[17] = simdBroadcast<Real>(0);
        output[18] = simdBroadcast<Real>(1440) * radial4 * harmonics[18];
        compensation[18] = simdBroadcast<Real>(0);
        output[19] = simdBroadcast<Real>(720) * radial4 * harmonics[19];
        compensation[19] = simdBroadcast<Real>(0);
        output[20] = simdBroadcast<Real>(576) * radial4 * harmonics[20];
        compensation[20] = simdBroadcast<Real>(0);
        output[21] = simdBroadcast<Real>(720) * radial4 * harmonics[21];
        compensation[21] = simdBroadcast<Real>(0);
        output[22] = simdBroadcast<Real>(1440) * radial4 * harmonics[22];
        compensation[22] = simdBroadcast<Real>(0);
        output[23] = simdBroadcast<Real>(5040) * radial4 * harmonics[23];
        compensation[23] = simdBroadcast<Real>(0);
        output[24] = simdBroadcast<Real>(40320) * radial4 * harmonics[24];
        compensation[24] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.22222222222222221);
                        Real const imageRadial4 = -screening * inversePower;
                        Real const imageValue16 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[16];
                        if (sign < 0) {
                            imagePair[16] = imageValue16;
                        } else {
                            imagePair[16] += imageValue16;
                        }
                        Real const imageValue17 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[17];
                        if (sign < 0) {
                            imagePair[17] = imageValue17;
                        } else {
                            imagePair[17] += imageValue17;
                        }
                        Real const imageValue18 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[18];
                        if (sign < 0) {
                            imagePair[18] = imageValue18;
                        } else {
                            imagePair[18] += imageValue18;
                        }
                        Real const imageValue19 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[19];
                        if (sign < 0) {
                            imagePair[19] = imageValue19;
                        } else {
                            imagePair[19] += imageValue19;
                        }
                        Real const imageValue20 =
                            simdBroadcast<Real>(576) * imageRadial4 * harmonics[20];
                        if (sign < 0) {
                            imagePair[20] = imageValue20;
                        } else {
                            imagePair[20] += imageValue20;
                        }
                        Real const imageValue21 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[21];
                        if (sign < 0) {
                            imagePair[21] = imageValue21;
                        } else {
                            imagePair[21] += imageValue21;
                        }
                        Real const imageValue22 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[22];
                        if (sign < 0) {
                            imagePair[22] = imageValue22;
                        } else {
                            imagePair[22] += imageValue22;
                        }
                        Real const imageValue23 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[23];
                        if (sign < 0) {
                            imagePair[23] = imageValue23;
                        } else {
                            imagePair[23] += imageValue23;
                        }
                        Real const imageValue24 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[24];
                        if (sign < 0) {
                            imagePair[24] = imageValue24;
                        } else {
                            imagePair[24] += imageValue24;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                    detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                    detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                    detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                    detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                    detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                    detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                    detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                    detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                    Real const waveRadial4 =
                        simdBroadcast<Real>(14.843290062324177) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[16], compensation[16],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[16]);
                    detail::ewaldAccumulate(output[17], compensation[17],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[17]);
                    detail::ewaldAccumulate(output[18], compensation[18],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[18]);
                    detail::ewaldAccumulate(output[19], compensation[19],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[19]);
                    detail::ewaldAccumulate(output[20], compensation[20],
                        simdBroadcast<Real>(576) * waveRadial4 * harmonics[20]);
                    detail::ewaldAccumulate(output[21], compensation[21],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[21]);
                    detail::ewaldAccumulate(output[22], compensation[22],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[22]);
                    detail::ewaldAccumulate(output[23], compensation[23],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[23]);
                    detail::ewaldAccumulate(output[24], compensation[24],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[24]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[25];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 25; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<4> {
public:
    static constexpr int kFlops_ = 228;
    static constexpr int kConventionalFlops_ = 409;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<4>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 6) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 6)
template <>
class EwaldGreen<5> {
public:
    static constexpr int kFlops_ = 426720;
    static constexpr int kConventionalFlops_ = 460761;
    static constexpr int kDegree_ = 5;
    static constexpr int kCount_ = 36;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        Real const zFactor4_0 = simdBroadcast<Real>(7) * z;
        output[20] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor4_0, output[12], -radiusSquared * output[6]);
        Real const zFactor5_0 = simdBroadcast<Real>(9) * z;
        output[30] = simdBroadcast<Real>(0.040000000000000001) *
            simdFma(zFactor5_0, output[20], -radiusSquared * output[12]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        Real const zFactor4_1 = simdBroadcast<Real>(7) * z;
        output[21] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[13], -radiusSquared * output[7]);
        output[19] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[11], -radiusSquared * output[5]);
        Real const zFactor5_1 = simdBroadcast<Real>(9) * z;
        output[31] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[21], -radiusSquared * output[13]);
        output[29] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[19], -radiusSquared * output[11]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        Real const zFactor4_2 = simdBroadcast<Real>(7) * z;
        output[22] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[14], -radiusSquared * output[8]);
        output[18] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[10], -radiusSquared * output[4]);
        Real const zFactor5_2 = simdBroadcast<Real>(9) * z;
        output[32] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[22], -radiusSquared * output[14]);
        output[28] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[18], -radiusSquared * output[10]);
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
        output[23] = z * output[15];
        output[17] = z * output[9];
        Real const zFactor5_3 = simdBroadcast<Real>(9) * z;
        output[33] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[23], -radiusSquared * output[15]);
        output[27] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[17], -radiusSquared * output[9]);
        output[24] = simdBroadcast<Real>(0.125) * simdFma(x, output[15], -y * output[9]);
        output[16] = simdBroadcast<Real>(0.125) * simdFma(y, output[15], x * output[9]);
        output[34] = z * output[24];
        output[26] = z * output[16];
        output[35] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(x, output[24], -y * output[16]);
        output[25] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(y, output[24], x * output[16]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[36];
        Real compensation[36];
        Real imagePair[36];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        Real series4 = simdBroadcast<Real>(1);
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        Real const radial4 = simdBroadcast<Real>(9.7816742188851205) * exponential * series4;
        output[16] = simdBroadcast<Real>(40320) * radial4 * harmonics[16];
        compensation[16] = simdBroadcast<Real>(0);
        output[17] = simdBroadcast<Real>(5040) * radial4 * harmonics[17];
        compensation[17] = simdBroadcast<Real>(0);
        output[18] = simdBroadcast<Real>(1440) * radial4 * harmonics[18];
        compensation[18] = simdBroadcast<Real>(0);
        output[19] = simdBroadcast<Real>(720) * radial4 * harmonics[19];
        compensation[19] = simdBroadcast<Real>(0);
        output[20] = simdBroadcast<Real>(576) * radial4 * harmonics[20];
        compensation[20] = simdBroadcast<Real>(0);
        output[21] = simdBroadcast<Real>(720) * radial4 * harmonics[21];
        compensation[21] = simdBroadcast<Real>(0);
        output[22] = simdBroadcast<Real>(1440) * radial4 * harmonics[22];
        compensation[22] = simdBroadcast<Real>(0);
        output[23] = simdBroadcast<Real>(5040) * radial4 * harmonics[23];
        compensation[23] = simdBroadcast<Real>(0);
        output[24] = simdBroadcast<Real>(40320) * radial4 * harmonics[24];
        compensation[24] = simdBroadcast<Real>(0);
        Real series5 = simdBroadcast<Real>(1);
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        Real const radial5 = simdBroadcast<Real>(-7.1139448864619048) * exponential * series5;
        output[25] = simdBroadcast<Real>(3628800) * radial5 * harmonics[25];
        compensation[25] = simdBroadcast<Real>(0);
        output[26] = simdBroadcast<Real>(362880) * radial5 * harmonics[26];
        compensation[26] = simdBroadcast<Real>(0);
        output[27] = simdBroadcast<Real>(80640) * radial5 * harmonics[27];
        compensation[27] = simdBroadcast<Real>(0);
        output[28] = simdBroadcast<Real>(30240) * radial5 * harmonics[28];
        compensation[28] = simdBroadcast<Real>(0);
        output[29] = simdBroadcast<Real>(17280) * radial5 * harmonics[29];
        compensation[29] = simdBroadcast<Real>(0);
        output[30] = simdBroadcast<Real>(14400) * radial5 * harmonics[30];
        compensation[30] = simdBroadcast<Real>(0);
        output[31] = simdBroadcast<Real>(17280) * radial5 * harmonics[31];
        compensation[31] = simdBroadcast<Real>(0);
        output[32] = simdBroadcast<Real>(30240) * radial5 * harmonics[32];
        compensation[32] = simdBroadcast<Real>(0);
        output[33] = simdBroadcast<Real>(80640) * radial5 * harmonics[33];
        compensation[33] = simdBroadcast<Real>(0);
        output[34] = simdBroadcast<Real>(362880) * radial5 * harmonics[34];
        compensation[34] = simdBroadcast<Real>(0);
        output[35] = simdBroadcast<Real>(3628800) * radial5 * harmonics[35];
        compensation[35] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.22222222222222221);
                        Real const imageRadial4 = -screening * inversePower;
                        Real const imageValue16 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[16];
                        if (sign < 0) {
                            imagePair[16] = imageValue16;
                        } else {
                            imagePair[16] += imageValue16;
                        }
                        Real const imageValue17 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[17];
                        if (sign < 0) {
                            imagePair[17] = imageValue17;
                        } else {
                            imagePair[17] += imageValue17;
                        }
                        Real const imageValue18 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[18];
                        if (sign < 0) {
                            imagePair[18] = imageValue18;
                        } else {
                            imagePair[18] += imageValue18;
                        }
                        Real const imageValue19 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[19];
                        if (sign < 0) {
                            imagePair[19] = imageValue19;
                        } else {
                            imagePair[19] += imageValue19;
                        }
                        Real const imageValue20 =
                            simdBroadcast<Real>(576) * imageRadial4 * harmonics[20];
                        if (sign < 0) {
                            imagePair[20] = imageValue20;
                        } else {
                            imagePair[20] += imageValue20;
                        }
                        Real const imageValue21 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[21];
                        if (sign < 0) {
                            imagePair[21] = imageValue21;
                        } else {
                            imagePair[21] += imageValue21;
                        }
                        Real const imageValue22 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[22];
                        if (sign < 0) {
                            imagePair[22] = imageValue22;
                        } else {
                            imagePair[22] += imageValue22;
                        }
                        Real const imageValue23 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[23];
                        if (sign < 0) {
                            imagePair[23] = imageValue23;
                        } else {
                            imagePair[23] += imageValue23;
                        }
                        Real const imageValue24 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[24];
                        if (sign < 0) {
                            imagePair[24] = imageValue24;
                        } else {
                            imagePair[24] += imageValue24;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.18181818181818182);
                        Real const imageRadial5 = screening * inversePower;
                        Real const imageValue25 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[25];
                        if (sign < 0) {
                            imagePair[25] = imageValue25;
                        } else {
                            imagePair[25] += imageValue25;
                        }
                        Real const imageValue26 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[26];
                        if (sign < 0) {
                            imagePair[26] = imageValue26;
                        } else {
                            imagePair[26] += imageValue26;
                        }
                        Real const imageValue27 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[27];
                        if (sign < 0) {
                            imagePair[27] = imageValue27;
                        } else {
                            imagePair[27] += imageValue27;
                        }
                        Real const imageValue28 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[28];
                        if (sign < 0) {
                            imagePair[28] = imageValue28;
                        } else {
                            imagePair[28] += imageValue28;
                        }
                        Real const imageValue29 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[29];
                        if (sign < 0) {
                            imagePair[29] = imageValue29;
                        } else {
                            imagePair[29] += imageValue29;
                        }
                        Real const imageValue30 =
                            simdBroadcast<Real>(14400) * imageRadial5 * harmonics[30];
                        if (sign < 0) {
                            imagePair[30] = imageValue30;
                        } else {
                            imagePair[30] += imageValue30;
                        }
                        Real const imageValue31 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[31];
                        if (sign < 0) {
                            imagePair[31] = imageValue31;
                        } else {
                            imagePair[31] += imageValue31;
                        }
                        Real const imageValue32 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[32];
                        if (sign < 0) {
                            imagePair[32] = imageValue32;
                        } else {
                            imagePair[32] += imageValue32;
                        }
                        Real const imageValue33 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[33];
                        if (sign < 0) {
                            imagePair[33] = imageValue33;
                        } else {
                            imagePair[33] += imageValue33;
                        }
                        Real const imageValue34 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[34];
                        if (sign < 0) {
                            imagePair[34] = imageValue34;
                        } else {
                            imagePair[34] += imageValue34;
                        }
                        Real const imageValue35 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[35];
                        if (sign < 0) {
                            imagePair[35] = imageValue35;
                        } else {
                            imagePair[35] += imageValue35;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                    detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                    detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                    detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                    detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                    detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                    detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                    detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                    detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                    detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                    detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                    detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                    detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                    detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                    detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                    detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                    detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                    detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                    detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                    detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                    Real const waveRadial4 =
                        simdBroadcast<Real>(14.843290062324177) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[16], compensation[16],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[16]);
                    detail::ewaldAccumulate(output[17], compensation[17],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[17]);
                    detail::ewaldAccumulate(output[18], compensation[18],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[18]);
                    detail::ewaldAccumulate(output[19], compensation[19],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[19]);
                    detail::ewaldAccumulate(output[20], compensation[20],
                        simdBroadcast<Real>(576) * waveRadial4 * harmonics[20]);
                    detail::ewaldAccumulate(output[21], compensation[21],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[21]);
                    detail::ewaldAccumulate(output[22], compensation[22],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[22]);
                    detail::ewaldAccumulate(output[23], compensation[23],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[23]);
                    detail::ewaldAccumulate(output[24], compensation[24],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[24]);
                    Real const waveRadial5 =
                        simdBroadcast<Real>(10.36257133664445) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[25], compensation[25],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[25]);
                    detail::ewaldAccumulate(output[26], compensation[26],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[26]);
                    detail::ewaldAccumulate(output[27], compensation[27],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[27]);
                    detail::ewaldAccumulate(output[28], compensation[28],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[28]);
                    detail::ewaldAccumulate(output[29], compensation[29],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[29]);
                    detail::ewaldAccumulate(output[30], compensation[30],
                        simdBroadcast<Real>(14400) * waveRadial5 * harmonics[30]);
                    detail::ewaldAccumulate(output[31], compensation[31],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[31]);
                    detail::ewaldAccumulate(output[32], compensation[32],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[32]);
                    detail::ewaldAccumulate(output[33], compensation[33],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[33]);
                    detail::ewaldAccumulate(output[34], compensation[34],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[34]);
                    detail::ewaldAccumulate(output[35], compensation[35],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[35]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[36];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 36; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<5> {
public:
    static constexpr int kFlops_ = 457;
    static constexpr int kConventionalFlops_ = 841;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<5>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 7) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 7)
template <>
class EwaldGreen<6> {
public:
    static constexpr int kFlops_ = 577958;
    static constexpr int kConventionalFlops_ = 627304;
    static constexpr int kDegree_ = 6;
    static constexpr int kCount_ = 49;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        Real const zFactor4_0 = simdBroadcast<Real>(7) * z;
        output[20] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor4_0, output[12], -radiusSquared * output[6]);
        Real const zFactor5_0 = simdBroadcast<Real>(9) * z;
        output[30] = simdBroadcast<Real>(0.040000000000000001) *
            simdFma(zFactor5_0, output[20], -radiusSquared * output[12]);
        Real const zFactor6_0 = simdBroadcast<Real>(11) * z;
        output[42] = simdBroadcast<Real>(0.027777777777777776) *
            simdFma(zFactor6_0, output[30], -radiusSquared * output[20]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        Real const zFactor4_1 = simdBroadcast<Real>(7) * z;
        output[21] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[13], -radiusSquared * output[7]);
        output[19] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[11], -radiusSquared * output[5]);
        Real const zFactor5_1 = simdBroadcast<Real>(9) * z;
        output[31] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[21], -radiusSquared * output[13]);
        output[29] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[19], -radiusSquared * output[11]);
        Real const zFactor6_1 = simdBroadcast<Real>(11) * z;
        output[43] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[31], -radiusSquared * output[21]);
        output[41] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[29], -radiusSquared * output[19]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        Real const zFactor4_2 = simdBroadcast<Real>(7) * z;
        output[22] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[14], -radiusSquared * output[8]);
        output[18] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[10], -radiusSquared * output[4]);
        Real const zFactor5_2 = simdBroadcast<Real>(9) * z;
        output[32] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[22], -radiusSquared * output[14]);
        output[28] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[18], -radiusSquared * output[10]);
        Real const zFactor6_2 = simdBroadcast<Real>(11) * z;
        output[44] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[32], -radiusSquared * output[22]);
        output[40] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[28], -radiusSquared * output[18]);
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
        output[23] = z * output[15];
        output[17] = z * output[9];
        Real const zFactor5_3 = simdBroadcast<Real>(9) * z;
        output[33] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[23], -radiusSquared * output[15]);
        output[27] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[17], -radiusSquared * output[9]);
        Real const zFactor6_3 = simdBroadcast<Real>(11) * z;
        output[45] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[33], -radiusSquared * output[23]);
        output[39] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[27], -radiusSquared * output[17]);
        output[24] = simdBroadcast<Real>(0.125) * simdFma(x, output[15], -y * output[9]);
        output[16] = simdBroadcast<Real>(0.125) * simdFma(y, output[15], x * output[9]);
        output[34] = z * output[24];
        output[26] = z * output[16];
        Real const zFactor6_4 = simdBroadcast<Real>(11) * z;
        output[46] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[34], -radiusSquared * output[24]);
        output[38] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[26], -radiusSquared * output[16]);
        output[35] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(x, output[24], -y * output[16]);
        output[25] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(y, output[24], x * output[16]);
        output[47] = z * output[35];
        output[37] = z * output[25];
        output[48] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(x, output[35], -y * output[25]);
        output[36] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(y, output[35], x * output[25]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[49];
        Real compensation[49];
        Real imagePair[49];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        Real series4 = simdBroadcast<Real>(1);
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        Real const radial4 = simdBroadcast<Real>(9.7816742188851205) * exponential * series4;
        output[16] = simdBroadcast<Real>(40320) * radial4 * harmonics[16];
        compensation[16] = simdBroadcast<Real>(0);
        output[17] = simdBroadcast<Real>(5040) * radial4 * harmonics[17];
        compensation[17] = simdBroadcast<Real>(0);
        output[18] = simdBroadcast<Real>(1440) * radial4 * harmonics[18];
        compensation[18] = simdBroadcast<Real>(0);
        output[19] = simdBroadcast<Real>(720) * radial4 * harmonics[19];
        compensation[19] = simdBroadcast<Real>(0);
        output[20] = simdBroadcast<Real>(576) * radial4 * harmonics[20];
        compensation[20] = simdBroadcast<Real>(0);
        output[21] = simdBroadcast<Real>(720) * radial4 * harmonics[21];
        compensation[21] = simdBroadcast<Real>(0);
        output[22] = simdBroadcast<Real>(1440) * radial4 * harmonics[22];
        compensation[22] = simdBroadcast<Real>(0);
        output[23] = simdBroadcast<Real>(5040) * radial4 * harmonics[23];
        compensation[23] = simdBroadcast<Real>(0);
        output[24] = simdBroadcast<Real>(40320) * radial4 * harmonics[24];
        compensation[24] = simdBroadcast<Real>(0);
        Real series5 = simdBroadcast<Real>(1);
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        Real const radial5 = simdBroadcast<Real>(-7.1139448864619048) * exponential * series5;
        output[25] = simdBroadcast<Real>(3628800) * radial5 * harmonics[25];
        compensation[25] = simdBroadcast<Real>(0);
        output[26] = simdBroadcast<Real>(362880) * radial5 * harmonics[26];
        compensation[26] = simdBroadcast<Real>(0);
        output[27] = simdBroadcast<Real>(80640) * radial5 * harmonics[27];
        compensation[27] = simdBroadcast<Real>(0);
        output[28] = simdBroadcast<Real>(30240) * radial5 * harmonics[28];
        compensation[28] = simdBroadcast<Real>(0);
        output[29] = simdBroadcast<Real>(17280) * radial5 * harmonics[29];
        compensation[29] = simdBroadcast<Real>(0);
        output[30] = simdBroadcast<Real>(14400) * radial5 * harmonics[30];
        compensation[30] = simdBroadcast<Real>(0);
        output[31] = simdBroadcast<Real>(17280) * radial5 * harmonics[31];
        compensation[31] = simdBroadcast<Real>(0);
        output[32] = simdBroadcast<Real>(30240) * radial5 * harmonics[32];
        compensation[32] = simdBroadcast<Real>(0);
        output[33] = simdBroadcast<Real>(80640) * radial5 * harmonics[33];
        compensation[33] = simdBroadcast<Real>(0);
        output[34] = simdBroadcast<Real>(362880) * radial5 * harmonics[34];
        compensation[34] = simdBroadcast<Real>(0);
        output[35] = simdBroadcast<Real>(3628800) * radial5 * harmonics[35];
        compensation[35] = simdBroadcast<Real>(0);
        Real series6 = simdBroadcast<Real>(1);
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        Real const radial6 = simdBroadcast<Real>(4.3778122378227113) * exponential * series6;
        output[36] = simdBroadcast<Real>(479001600) * radial6 * harmonics[36];
        compensation[36] = simdBroadcast<Real>(0);
        output[37] = simdBroadcast<Real>(39916800) * radial6 * harmonics[37];
        compensation[37] = simdBroadcast<Real>(0);
        output[38] = simdBroadcast<Real>(7257600) * radial6 * harmonics[38];
        compensation[38] = simdBroadcast<Real>(0);
        output[39] = simdBroadcast<Real>(2177280) * radial6 * harmonics[39];
        compensation[39] = simdBroadcast<Real>(0);
        output[40] = simdBroadcast<Real>(967680) * radial6 * harmonics[40];
        compensation[40] = simdBroadcast<Real>(0);
        output[41] = simdBroadcast<Real>(604800) * radial6 * harmonics[41];
        compensation[41] = simdBroadcast<Real>(0);
        output[42] = simdBroadcast<Real>(518400) * radial6 * harmonics[42];
        compensation[42] = simdBroadcast<Real>(0);
        output[43] = simdBroadcast<Real>(604800) * radial6 * harmonics[43];
        compensation[43] = simdBroadcast<Real>(0);
        output[44] = simdBroadcast<Real>(967680) * radial6 * harmonics[44];
        compensation[44] = simdBroadcast<Real>(0);
        output[45] = simdBroadcast<Real>(2177280) * radial6 * harmonics[45];
        compensation[45] = simdBroadcast<Real>(0);
        output[46] = simdBroadcast<Real>(7257600) * radial6 * harmonics[46];
        compensation[46] = simdBroadcast<Real>(0);
        output[47] = simdBroadcast<Real>(39916800) * radial6 * harmonics[47];
        compensation[47] = simdBroadcast<Real>(0);
        output[48] = simdBroadcast<Real>(479001600) * radial6 * harmonics[48];
        compensation[48] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.22222222222222221);
                        Real const imageRadial4 = -screening * inversePower;
                        Real const imageValue16 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[16];
                        if (sign < 0) {
                            imagePair[16] = imageValue16;
                        } else {
                            imagePair[16] += imageValue16;
                        }
                        Real const imageValue17 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[17];
                        if (sign < 0) {
                            imagePair[17] = imageValue17;
                        } else {
                            imagePair[17] += imageValue17;
                        }
                        Real const imageValue18 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[18];
                        if (sign < 0) {
                            imagePair[18] = imageValue18;
                        } else {
                            imagePair[18] += imageValue18;
                        }
                        Real const imageValue19 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[19];
                        if (sign < 0) {
                            imagePair[19] = imageValue19;
                        } else {
                            imagePair[19] += imageValue19;
                        }
                        Real const imageValue20 =
                            simdBroadcast<Real>(576) * imageRadial4 * harmonics[20];
                        if (sign < 0) {
                            imagePair[20] = imageValue20;
                        } else {
                            imagePair[20] += imageValue20;
                        }
                        Real const imageValue21 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[21];
                        if (sign < 0) {
                            imagePair[21] = imageValue21;
                        } else {
                            imagePair[21] += imageValue21;
                        }
                        Real const imageValue22 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[22];
                        if (sign < 0) {
                            imagePair[22] = imageValue22;
                        } else {
                            imagePair[22] += imageValue22;
                        }
                        Real const imageValue23 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[23];
                        if (sign < 0) {
                            imagePair[23] = imageValue23;
                        } else {
                            imagePair[23] += imageValue23;
                        }
                        Real const imageValue24 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[24];
                        if (sign < 0) {
                            imagePair[24] = imageValue24;
                        } else {
                            imagePair[24] += imageValue24;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.18181818181818182);
                        Real const imageRadial5 = screening * inversePower;
                        Real const imageValue25 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[25];
                        if (sign < 0) {
                            imagePair[25] = imageValue25;
                        } else {
                            imagePair[25] += imageValue25;
                        }
                        Real const imageValue26 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[26];
                        if (sign < 0) {
                            imagePair[26] = imageValue26;
                        } else {
                            imagePair[26] += imageValue26;
                        }
                        Real const imageValue27 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[27];
                        if (sign < 0) {
                            imagePair[27] = imageValue27;
                        } else {
                            imagePair[27] += imageValue27;
                        }
                        Real const imageValue28 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[28];
                        if (sign < 0) {
                            imagePair[28] = imageValue28;
                        } else {
                            imagePair[28] += imageValue28;
                        }
                        Real const imageValue29 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[29];
                        if (sign < 0) {
                            imagePair[29] = imageValue29;
                        } else {
                            imagePair[29] += imageValue29;
                        }
                        Real const imageValue30 =
                            simdBroadcast<Real>(14400) * imageRadial5 * harmonics[30];
                        if (sign < 0) {
                            imagePair[30] = imageValue30;
                        } else {
                            imagePair[30] += imageValue30;
                        }
                        Real const imageValue31 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[31];
                        if (sign < 0) {
                            imagePair[31] = imageValue31;
                        } else {
                            imagePair[31] += imageValue31;
                        }
                        Real const imageValue32 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[32];
                        if (sign < 0) {
                            imagePair[32] = imageValue32;
                        } else {
                            imagePair[32] += imageValue32;
                        }
                        Real const imageValue33 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[33];
                        if (sign < 0) {
                            imagePair[33] = imageValue33;
                        } else {
                            imagePair[33] += imageValue33;
                        }
                        Real const imageValue34 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[34];
                        if (sign < 0) {
                            imagePair[34] = imageValue34;
                        } else {
                            imagePair[34] += imageValue34;
                        }
                        Real const imageValue35 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[35];
                        if (sign < 0) {
                            imagePair[35] = imageValue35;
                        } else {
                            imagePair[35] += imageValue35;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.15384615384615385);
                        Real const imageRadial6 = -screening * inversePower;
                        Real const imageValue36 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[36];
                        if (sign < 0) {
                            imagePair[36] = imageValue36;
                        } else {
                            imagePair[36] += imageValue36;
                        }
                        Real const imageValue37 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[37];
                        if (sign < 0) {
                            imagePair[37] = imageValue37;
                        } else {
                            imagePair[37] += imageValue37;
                        }
                        Real const imageValue38 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[38];
                        if (sign < 0) {
                            imagePair[38] = imageValue38;
                        } else {
                            imagePair[38] += imageValue38;
                        }
                        Real const imageValue39 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[39];
                        if (sign < 0) {
                            imagePair[39] = imageValue39;
                        } else {
                            imagePair[39] += imageValue39;
                        }
                        Real const imageValue40 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[40];
                        if (sign < 0) {
                            imagePair[40] = imageValue40;
                        } else {
                            imagePair[40] += imageValue40;
                        }
                        Real const imageValue41 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[41];
                        if (sign < 0) {
                            imagePair[41] = imageValue41;
                        } else {
                            imagePair[41] += imageValue41;
                        }
                        Real const imageValue42 =
                            simdBroadcast<Real>(518400) * imageRadial6 * harmonics[42];
                        if (sign < 0) {
                            imagePair[42] = imageValue42;
                        } else {
                            imagePair[42] += imageValue42;
                        }
                        Real const imageValue43 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[43];
                        if (sign < 0) {
                            imagePair[43] = imageValue43;
                        } else {
                            imagePair[43] += imageValue43;
                        }
                        Real const imageValue44 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[44];
                        if (sign < 0) {
                            imagePair[44] = imageValue44;
                        } else {
                            imagePair[44] += imageValue44;
                        }
                        Real const imageValue45 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[45];
                        if (sign < 0) {
                            imagePair[45] = imageValue45;
                        } else {
                            imagePair[45] += imageValue45;
                        }
                        Real const imageValue46 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[46];
                        if (sign < 0) {
                            imagePair[46] = imageValue46;
                        } else {
                            imagePair[46] += imageValue46;
                        }
                        Real const imageValue47 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[47];
                        if (sign < 0) {
                            imagePair[47] = imageValue47;
                        } else {
                            imagePair[47] += imageValue47;
                        }
                        Real const imageValue48 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[48];
                        if (sign < 0) {
                            imagePair[48] = imageValue48;
                        } else {
                            imagePair[48] += imageValue48;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                    detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                    detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                    detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                    detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                    detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                    detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                    detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                    detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                    detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                    detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                    detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                    detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                    detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                    detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                    detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                    detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                    detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                    detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                    detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                    detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                    detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                    detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                    detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                    detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                    detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                    detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                    detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                    detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                    detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                    detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                    detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                    detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                    Real const waveRadial4 =
                        simdBroadcast<Real>(14.843290062324177) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[16], compensation[16],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[16]);
                    detail::ewaldAccumulate(output[17], compensation[17],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[17]);
                    detail::ewaldAccumulate(output[18], compensation[18],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[18]);
                    detail::ewaldAccumulate(output[19], compensation[19],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[19]);
                    detail::ewaldAccumulate(output[20], compensation[20],
                        simdBroadcast<Real>(576) * waveRadial4 * harmonics[20]);
                    detail::ewaldAccumulate(output[21], compensation[21],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[21]);
                    detail::ewaldAccumulate(output[22], compensation[22],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[22]);
                    detail::ewaldAccumulate(output[23], compensation[23],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[23]);
                    detail::ewaldAccumulate(output[24], compensation[24],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[24]);
                    Real const waveRadial5 =
                        simdBroadcast<Real>(10.36257133664445) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[25], compensation[25],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[25]);
                    detail::ewaldAccumulate(output[26], compensation[26],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[26]);
                    detail::ewaldAccumulate(output[27], compensation[27],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[27]);
                    detail::ewaldAccumulate(output[28], compensation[28],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[28]);
                    detail::ewaldAccumulate(output[29], compensation[29],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[29]);
                    detail::ewaldAccumulate(output[30], compensation[30],
                        simdBroadcast<Real>(14400) * waveRadial5 * harmonics[30]);
                    detail::ewaldAccumulate(output[31], compensation[31],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[31]);
                    detail::ewaldAccumulate(output[32], compensation[32],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[32]);
                    detail::ewaldAccumulate(output[33], compensation[33],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[33]);
                    detail::ewaldAccumulate(output[34], compensation[34],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[34]);
                    detail::ewaldAccumulate(output[35], compensation[35],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[35]);
                    Real const waveRadial6 =
                        simdBroadcast<Real>(5.9190869060913398) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[36], compensation[36],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[36]);
                    detail::ewaldAccumulate(output[37], compensation[37],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[37]);
                    detail::ewaldAccumulate(output[38], compensation[38],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[38]);
                    detail::ewaldAccumulate(output[39], compensation[39],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[39]);
                    detail::ewaldAccumulate(output[40], compensation[40],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[40]);
                    detail::ewaldAccumulate(output[41], compensation[41],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[41]);
                    detail::ewaldAccumulate(output[42], compensation[42],
                        simdBroadcast<Real>(518400) * waveRadial6 * harmonics[42]);
                    detail::ewaldAccumulate(output[43], compensation[43],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[43]);
                    detail::ewaldAccumulate(output[44], compensation[44],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[44]);
                    detail::ewaldAccumulate(output[45], compensation[45],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[45]);
                    detail::ewaldAccumulate(output[46], compensation[46],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[46]);
                    detail::ewaldAccumulate(output[47], compensation[47],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[47]);
                    detail::ewaldAccumulate(output[48], compensation[48],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[48]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[49];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 49; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<6> {
public:
    static constexpr int kFlops_ = 829;
    static constexpr int kConventionalFlops_ = 1553;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<6>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 8) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 8)
template <>
class EwaldGreen<7> {
public:
    static constexpr int kFlops_ = 752769;
    static constexpr int kConventionalFlops_ = 820194;
    static constexpr int kDegree_ = 7;
    static constexpr int kCount_ = 64;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        Real const zFactor4_0 = simdBroadcast<Real>(7) * z;
        output[20] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor4_0, output[12], -radiusSquared * output[6]);
        Real const zFactor5_0 = simdBroadcast<Real>(9) * z;
        output[30] = simdBroadcast<Real>(0.040000000000000001) *
            simdFma(zFactor5_0, output[20], -radiusSquared * output[12]);
        Real const zFactor6_0 = simdBroadcast<Real>(11) * z;
        output[42] = simdBroadcast<Real>(0.027777777777777776) *
            simdFma(zFactor6_0, output[30], -radiusSquared * output[20]);
        Real const zFactor7_0 = simdBroadcast<Real>(13) * z;
        output[56] = simdBroadcast<Real>(0.020408163265306121) *
            simdFma(zFactor7_0, output[42], -radiusSquared * output[30]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        Real const zFactor4_1 = simdBroadcast<Real>(7) * z;
        output[21] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[13], -radiusSquared * output[7]);
        output[19] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[11], -radiusSquared * output[5]);
        Real const zFactor5_1 = simdBroadcast<Real>(9) * z;
        output[31] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[21], -radiusSquared * output[13]);
        output[29] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[19], -radiusSquared * output[11]);
        Real const zFactor6_1 = simdBroadcast<Real>(11) * z;
        output[43] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[31], -radiusSquared * output[21]);
        output[41] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[29], -radiusSquared * output[19]);
        Real const zFactor7_1 = simdBroadcast<Real>(13) * z;
        output[57] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor7_1, output[43], -radiusSquared * output[31]);
        output[55] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor7_1, output[41], -radiusSquared * output[29]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        Real const zFactor4_2 = simdBroadcast<Real>(7) * z;
        output[22] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[14], -radiusSquared * output[8]);
        output[18] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[10], -radiusSquared * output[4]);
        Real const zFactor5_2 = simdBroadcast<Real>(9) * z;
        output[32] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[22], -radiusSquared * output[14]);
        output[28] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[18], -radiusSquared * output[10]);
        Real const zFactor6_2 = simdBroadcast<Real>(11) * z;
        output[44] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[32], -radiusSquared * output[22]);
        output[40] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[28], -radiusSquared * output[18]);
        Real const zFactor7_2 = simdBroadcast<Real>(13) * z;
        output[58] = simdBroadcast<Real>(0.022222222222222223) *
            simdFma(zFactor7_2, output[44], -radiusSquared * output[32]);
        output[54] = simdBroadcast<Real>(0.022222222222222223) *
            simdFma(zFactor7_2, output[40], -radiusSquared * output[28]);
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
        output[23] = z * output[15];
        output[17] = z * output[9];
        Real const zFactor5_3 = simdBroadcast<Real>(9) * z;
        output[33] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[23], -radiusSquared * output[15]);
        output[27] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[17], -radiusSquared * output[9]);
        Real const zFactor6_3 = simdBroadcast<Real>(11) * z;
        output[45] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[33], -radiusSquared * output[23]);
        output[39] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[27], -radiusSquared * output[17]);
        Real const zFactor7_3 = simdBroadcast<Real>(13) * z;
        output[59] = simdBroadcast<Real>(0.025000000000000001) *
            simdFma(zFactor7_3, output[45], -radiusSquared * output[33]);
        output[53] = simdBroadcast<Real>(0.025000000000000001) *
            simdFma(zFactor7_3, output[39], -radiusSquared * output[27]);
        output[24] = simdBroadcast<Real>(0.125) * simdFma(x, output[15], -y * output[9]);
        output[16] = simdBroadcast<Real>(0.125) * simdFma(y, output[15], x * output[9]);
        output[34] = z * output[24];
        output[26] = z * output[16];
        Real const zFactor6_4 = simdBroadcast<Real>(11) * z;
        output[46] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[34], -radiusSquared * output[24]);
        output[38] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[26], -radiusSquared * output[16]);
        Real const zFactor7_4 = simdBroadcast<Real>(13) * z;
        output[60] = simdBroadcast<Real>(0.030303030303030304) *
            simdFma(zFactor7_4, output[46], -radiusSquared * output[34]);
        output[52] = simdBroadcast<Real>(0.030303030303030304) *
            simdFma(zFactor7_4, output[38], -radiusSquared * output[26]);
        output[35] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(x, output[24], -y * output[16]);
        output[25] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(y, output[24], x * output[16]);
        output[47] = z * output[35];
        output[37] = z * output[25];
        Real const zFactor7_5 = simdBroadcast<Real>(13) * z;
        output[61] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor7_5, output[47], -radiusSquared * output[35]);
        output[51] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor7_5, output[37], -radiusSquared * output[25]);
        output[48] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(x, output[35], -y * output[25]);
        output[36] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(y, output[35], x * output[25]);
        output[62] = z * output[48];
        output[50] = z * output[36];
        output[63] =
            simdBroadcast<Real>(0.071428571428571425) * simdFma(x, output[48], -y * output[36]);
        output[49] =
            simdBroadcast<Real>(0.071428571428571425) * simdFma(y, output[48], x * output[36]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[64];
        Real compensation[64];
        Real imagePair[64];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        Real series4 = simdBroadcast<Real>(1);
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        Real const radial4 = simdBroadcast<Real>(9.7816742188851205) * exponential * series4;
        output[16] = simdBroadcast<Real>(40320) * radial4 * harmonics[16];
        compensation[16] = simdBroadcast<Real>(0);
        output[17] = simdBroadcast<Real>(5040) * radial4 * harmonics[17];
        compensation[17] = simdBroadcast<Real>(0);
        output[18] = simdBroadcast<Real>(1440) * radial4 * harmonics[18];
        compensation[18] = simdBroadcast<Real>(0);
        output[19] = simdBroadcast<Real>(720) * radial4 * harmonics[19];
        compensation[19] = simdBroadcast<Real>(0);
        output[20] = simdBroadcast<Real>(576) * radial4 * harmonics[20];
        compensation[20] = simdBroadcast<Real>(0);
        output[21] = simdBroadcast<Real>(720) * radial4 * harmonics[21];
        compensation[21] = simdBroadcast<Real>(0);
        output[22] = simdBroadcast<Real>(1440) * radial4 * harmonics[22];
        compensation[22] = simdBroadcast<Real>(0);
        output[23] = simdBroadcast<Real>(5040) * radial4 * harmonics[23];
        compensation[23] = simdBroadcast<Real>(0);
        output[24] = simdBroadcast<Real>(40320) * radial4 * harmonics[24];
        compensation[24] = simdBroadcast<Real>(0);
        Real series5 = simdBroadcast<Real>(1);
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        Real const radial5 = simdBroadcast<Real>(-7.1139448864619048) * exponential * series5;
        output[25] = simdBroadcast<Real>(3628800) * radial5 * harmonics[25];
        compensation[25] = simdBroadcast<Real>(0);
        output[26] = simdBroadcast<Real>(362880) * radial5 * harmonics[26];
        compensation[26] = simdBroadcast<Real>(0);
        output[27] = simdBroadcast<Real>(80640) * radial5 * harmonics[27];
        compensation[27] = simdBroadcast<Real>(0);
        output[28] = simdBroadcast<Real>(30240) * radial5 * harmonics[28];
        compensation[28] = simdBroadcast<Real>(0);
        output[29] = simdBroadcast<Real>(17280) * radial5 * harmonics[29];
        compensation[29] = simdBroadcast<Real>(0);
        output[30] = simdBroadcast<Real>(14400) * radial5 * harmonics[30];
        compensation[30] = simdBroadcast<Real>(0);
        output[31] = simdBroadcast<Real>(17280) * radial5 * harmonics[31];
        compensation[31] = simdBroadcast<Real>(0);
        output[32] = simdBroadcast<Real>(30240) * radial5 * harmonics[32];
        compensation[32] = simdBroadcast<Real>(0);
        output[33] = simdBroadcast<Real>(80640) * radial5 * harmonics[33];
        compensation[33] = simdBroadcast<Real>(0);
        output[34] = simdBroadcast<Real>(362880) * radial5 * harmonics[34];
        compensation[34] = simdBroadcast<Real>(0);
        output[35] = simdBroadcast<Real>(3628800) * radial5 * harmonics[35];
        compensation[35] = simdBroadcast<Real>(0);
        Real series6 = simdBroadcast<Real>(1);
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        Real const radial6 = simdBroadcast<Real>(4.3778122378227113) * exponential * series6;
        output[36] = simdBroadcast<Real>(479001600) * radial6 * harmonics[36];
        compensation[36] = simdBroadcast<Real>(0);
        output[37] = simdBroadcast<Real>(39916800) * radial6 * harmonics[37];
        compensation[37] = simdBroadcast<Real>(0);
        output[38] = simdBroadcast<Real>(7257600) * radial6 * harmonics[38];
        compensation[38] = simdBroadcast<Real>(0);
        output[39] = simdBroadcast<Real>(2177280) * radial6 * harmonics[39];
        compensation[39] = simdBroadcast<Real>(0);
        output[40] = simdBroadcast<Real>(967680) * radial6 * harmonics[40];
        compensation[40] = simdBroadcast<Real>(0);
        output[41] = simdBroadcast<Real>(604800) * radial6 * harmonics[41];
        compensation[41] = simdBroadcast<Real>(0);
        output[42] = simdBroadcast<Real>(518400) * radial6 * harmonics[42];
        compensation[42] = simdBroadcast<Real>(0);
        output[43] = simdBroadcast<Real>(604800) * radial6 * harmonics[43];
        compensation[43] = simdBroadcast<Real>(0);
        output[44] = simdBroadcast<Real>(967680) * radial6 * harmonics[44];
        compensation[44] = simdBroadcast<Real>(0);
        output[45] = simdBroadcast<Real>(2177280) * radial6 * harmonics[45];
        compensation[45] = simdBroadcast<Real>(0);
        output[46] = simdBroadcast<Real>(7257600) * radial6 * harmonics[46];
        compensation[46] = simdBroadcast<Real>(0);
        output[47] = simdBroadcast<Real>(39916800) * radial6 * harmonics[47];
        compensation[47] = simdBroadcast<Real>(0);
        output[48] = simdBroadcast<Real>(479001600) * radial6 * harmonics[48];
        compensation[48] = simdBroadcast<Real>(0);
        Real series7 = simdBroadcast<Real>(1);
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.018018018018018018), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        Real const radial7 = simdBroadcast<Real>(-2.334833193505446) * exponential * series7;
        output[49] = simdBroadcast<Real>(87178291200) * radial7 * harmonics[49];
        compensation[49] = simdBroadcast<Real>(0);
        output[50] = simdBroadcast<Real>(6227020800) * radial7 * harmonics[50];
        compensation[50] = simdBroadcast<Real>(0);
        output[51] = simdBroadcast<Real>(958003200) * radial7 * harmonics[51];
        compensation[51] = simdBroadcast<Real>(0);
        output[52] = simdBroadcast<Real>(239500800) * radial7 * harmonics[52];
        compensation[52] = simdBroadcast<Real>(0);
        output[53] = simdBroadcast<Real>(87091200) * radial7 * harmonics[53];
        compensation[53] = simdBroadcast<Real>(0);
        output[54] = simdBroadcast<Real>(43545600) * radial7 * harmonics[54];
        compensation[54] = simdBroadcast<Real>(0);
        output[55] = simdBroadcast<Real>(29030400) * radial7 * harmonics[55];
        compensation[55] = simdBroadcast<Real>(0);
        output[56] = simdBroadcast<Real>(25401600) * radial7 * harmonics[56];
        compensation[56] = simdBroadcast<Real>(0);
        output[57] = simdBroadcast<Real>(29030400) * radial7 * harmonics[57];
        compensation[57] = simdBroadcast<Real>(0);
        output[58] = simdBroadcast<Real>(43545600) * radial7 * harmonics[58];
        compensation[58] = simdBroadcast<Real>(0);
        output[59] = simdBroadcast<Real>(87091200) * radial7 * harmonics[59];
        compensation[59] = simdBroadcast<Real>(0);
        output[60] = simdBroadcast<Real>(239500800) * radial7 * harmonics[60];
        compensation[60] = simdBroadcast<Real>(0);
        output[61] = simdBroadcast<Real>(958003200) * radial7 * harmonics[61];
        compensation[61] = simdBroadcast<Real>(0);
        output[62] = simdBroadcast<Real>(6227020800) * radial7 * harmonics[62];
        compensation[62] = simdBroadcast<Real>(0);
        output[63] = simdBroadcast<Real>(87178291200) * radial7 * harmonics[63];
        compensation[63] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.22222222222222221);
                        Real const imageRadial4 = -screening * inversePower;
                        Real const imageValue16 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[16];
                        if (sign < 0) {
                            imagePair[16] = imageValue16;
                        } else {
                            imagePair[16] += imageValue16;
                        }
                        Real const imageValue17 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[17];
                        if (sign < 0) {
                            imagePair[17] = imageValue17;
                        } else {
                            imagePair[17] += imageValue17;
                        }
                        Real const imageValue18 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[18];
                        if (sign < 0) {
                            imagePair[18] = imageValue18;
                        } else {
                            imagePair[18] += imageValue18;
                        }
                        Real const imageValue19 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[19];
                        if (sign < 0) {
                            imagePair[19] = imageValue19;
                        } else {
                            imagePair[19] += imageValue19;
                        }
                        Real const imageValue20 =
                            simdBroadcast<Real>(576) * imageRadial4 * harmonics[20];
                        if (sign < 0) {
                            imagePair[20] = imageValue20;
                        } else {
                            imagePair[20] += imageValue20;
                        }
                        Real const imageValue21 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[21];
                        if (sign < 0) {
                            imagePair[21] = imageValue21;
                        } else {
                            imagePair[21] += imageValue21;
                        }
                        Real const imageValue22 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[22];
                        if (sign < 0) {
                            imagePair[22] = imageValue22;
                        } else {
                            imagePair[22] += imageValue22;
                        }
                        Real const imageValue23 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[23];
                        if (sign < 0) {
                            imagePair[23] = imageValue23;
                        } else {
                            imagePair[23] += imageValue23;
                        }
                        Real const imageValue24 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[24];
                        if (sign < 0) {
                            imagePair[24] = imageValue24;
                        } else {
                            imagePair[24] += imageValue24;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.18181818181818182);
                        Real const imageRadial5 = screening * inversePower;
                        Real const imageValue25 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[25];
                        if (sign < 0) {
                            imagePair[25] = imageValue25;
                        } else {
                            imagePair[25] += imageValue25;
                        }
                        Real const imageValue26 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[26];
                        if (sign < 0) {
                            imagePair[26] = imageValue26;
                        } else {
                            imagePair[26] += imageValue26;
                        }
                        Real const imageValue27 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[27];
                        if (sign < 0) {
                            imagePair[27] = imageValue27;
                        } else {
                            imagePair[27] += imageValue27;
                        }
                        Real const imageValue28 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[28];
                        if (sign < 0) {
                            imagePair[28] = imageValue28;
                        } else {
                            imagePair[28] += imageValue28;
                        }
                        Real const imageValue29 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[29];
                        if (sign < 0) {
                            imagePair[29] = imageValue29;
                        } else {
                            imagePair[29] += imageValue29;
                        }
                        Real const imageValue30 =
                            simdBroadcast<Real>(14400) * imageRadial5 * harmonics[30];
                        if (sign < 0) {
                            imagePair[30] = imageValue30;
                        } else {
                            imagePair[30] += imageValue30;
                        }
                        Real const imageValue31 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[31];
                        if (sign < 0) {
                            imagePair[31] = imageValue31;
                        } else {
                            imagePair[31] += imageValue31;
                        }
                        Real const imageValue32 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[32];
                        if (sign < 0) {
                            imagePair[32] = imageValue32;
                        } else {
                            imagePair[32] += imageValue32;
                        }
                        Real const imageValue33 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[33];
                        if (sign < 0) {
                            imagePair[33] = imageValue33;
                        } else {
                            imagePair[33] += imageValue33;
                        }
                        Real const imageValue34 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[34];
                        if (sign < 0) {
                            imagePair[34] = imageValue34;
                        } else {
                            imagePair[34] += imageValue34;
                        }
                        Real const imageValue35 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[35];
                        if (sign < 0) {
                            imagePair[35] = imageValue35;
                        } else {
                            imagePair[35] += imageValue35;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.15384615384615385);
                        Real const imageRadial6 = -screening * inversePower;
                        Real const imageValue36 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[36];
                        if (sign < 0) {
                            imagePair[36] = imageValue36;
                        } else {
                            imagePair[36] += imageValue36;
                        }
                        Real const imageValue37 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[37];
                        if (sign < 0) {
                            imagePair[37] = imageValue37;
                        } else {
                            imagePair[37] += imageValue37;
                        }
                        Real const imageValue38 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[38];
                        if (sign < 0) {
                            imagePair[38] = imageValue38;
                        } else {
                            imagePair[38] += imageValue38;
                        }
                        Real const imageValue39 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[39];
                        if (sign < 0) {
                            imagePair[39] = imageValue39;
                        } else {
                            imagePair[39] += imageValue39;
                        }
                        Real const imageValue40 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[40];
                        if (sign < 0) {
                            imagePair[40] = imageValue40;
                        } else {
                            imagePair[40] += imageValue40;
                        }
                        Real const imageValue41 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[41];
                        if (sign < 0) {
                            imagePair[41] = imageValue41;
                        } else {
                            imagePair[41] += imageValue41;
                        }
                        Real const imageValue42 =
                            simdBroadcast<Real>(518400) * imageRadial6 * harmonics[42];
                        if (sign < 0) {
                            imagePair[42] = imageValue42;
                        } else {
                            imagePair[42] += imageValue42;
                        }
                        Real const imageValue43 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[43];
                        if (sign < 0) {
                            imagePair[43] = imageValue43;
                        } else {
                            imagePair[43] += imageValue43;
                        }
                        Real const imageValue44 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[44];
                        if (sign < 0) {
                            imagePair[44] = imageValue44;
                        } else {
                            imagePair[44] += imageValue44;
                        }
                        Real const imageValue45 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[45];
                        if (sign < 0) {
                            imagePair[45] = imageValue45;
                        } else {
                            imagePair[45] += imageValue45;
                        }
                        Real const imageValue46 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[46];
                        if (sign < 0) {
                            imagePair[46] = imageValue46;
                        } else {
                            imagePair[46] += imageValue46;
                        }
                        Real const imageValue47 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[47];
                        if (sign < 0) {
                            imagePair[47] = imageValue47;
                        } else {
                            imagePair[47] += imageValue47;
                        }
                        Real const imageValue48 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[48];
                        if (sign < 0) {
                            imagePair[48] = imageValue48;
                        } else {
                            imagePair[48] += imageValue48;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.13333333333333333);
                        Real const imageRadial7 = screening * inversePower;
                        Real const imageValue49 =
                            simdBroadcast<Real>(87178291200) * imageRadial7 * harmonics[49];
                        if (sign < 0) {
                            imagePair[49] = imageValue49;
                        } else {
                            imagePair[49] += imageValue49;
                        }
                        Real const imageValue50 =
                            simdBroadcast<Real>(6227020800) * imageRadial7 * harmonics[50];
                        if (sign < 0) {
                            imagePair[50] = imageValue50;
                        } else {
                            imagePair[50] += imageValue50;
                        }
                        Real const imageValue51 =
                            simdBroadcast<Real>(958003200) * imageRadial7 * harmonics[51];
                        if (sign < 0) {
                            imagePair[51] = imageValue51;
                        } else {
                            imagePair[51] += imageValue51;
                        }
                        Real const imageValue52 =
                            simdBroadcast<Real>(239500800) * imageRadial7 * harmonics[52];
                        if (sign < 0) {
                            imagePair[52] = imageValue52;
                        } else {
                            imagePair[52] += imageValue52;
                        }
                        Real const imageValue53 =
                            simdBroadcast<Real>(87091200) * imageRadial7 * harmonics[53];
                        if (sign < 0) {
                            imagePair[53] = imageValue53;
                        } else {
                            imagePair[53] += imageValue53;
                        }
                        Real const imageValue54 =
                            simdBroadcast<Real>(43545600) * imageRadial7 * harmonics[54];
                        if (sign < 0) {
                            imagePair[54] = imageValue54;
                        } else {
                            imagePair[54] += imageValue54;
                        }
                        Real const imageValue55 =
                            simdBroadcast<Real>(29030400) * imageRadial7 * harmonics[55];
                        if (sign < 0) {
                            imagePair[55] = imageValue55;
                        } else {
                            imagePair[55] += imageValue55;
                        }
                        Real const imageValue56 =
                            simdBroadcast<Real>(25401600) * imageRadial7 * harmonics[56];
                        if (sign < 0) {
                            imagePair[56] = imageValue56;
                        } else {
                            imagePair[56] += imageValue56;
                        }
                        Real const imageValue57 =
                            simdBroadcast<Real>(29030400) * imageRadial7 * harmonics[57];
                        if (sign < 0) {
                            imagePair[57] = imageValue57;
                        } else {
                            imagePair[57] += imageValue57;
                        }
                        Real const imageValue58 =
                            simdBroadcast<Real>(43545600) * imageRadial7 * harmonics[58];
                        if (sign < 0) {
                            imagePair[58] = imageValue58;
                        } else {
                            imagePair[58] += imageValue58;
                        }
                        Real const imageValue59 =
                            simdBroadcast<Real>(87091200) * imageRadial7 * harmonics[59];
                        if (sign < 0) {
                            imagePair[59] = imageValue59;
                        } else {
                            imagePair[59] += imageValue59;
                        }
                        Real const imageValue60 =
                            simdBroadcast<Real>(239500800) * imageRadial7 * harmonics[60];
                        if (sign < 0) {
                            imagePair[60] = imageValue60;
                        } else {
                            imagePair[60] += imageValue60;
                        }
                        Real const imageValue61 =
                            simdBroadcast<Real>(958003200) * imageRadial7 * harmonics[61];
                        if (sign < 0) {
                            imagePair[61] = imageValue61;
                        } else {
                            imagePair[61] += imageValue61;
                        }
                        Real const imageValue62 =
                            simdBroadcast<Real>(6227020800) * imageRadial7 * harmonics[62];
                        if (sign < 0) {
                            imagePair[62] = imageValue62;
                        } else {
                            imagePair[62] += imageValue62;
                        }
                        Real const imageValue63 =
                            simdBroadcast<Real>(87178291200) * imageRadial7 * harmonics[63];
                        if (sign < 0) {
                            imagePair[63] = imageValue63;
                        } else {
                            imagePair[63] += imageValue63;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                    detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                    detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                    detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                    detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                    detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                    detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                    detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                    detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                    detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                    detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                    detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                    detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                    detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                    detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                    detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                    detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                    detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                    detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                    detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                    detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                    detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                    detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                    detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                    detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                    detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                    detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                    detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                    detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                    detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                    detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                    detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                    detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                    detail::ewaldAccumulate(output[49], compensation[49], imagePair[49]);
                    detail::ewaldAccumulate(output[50], compensation[50], imagePair[50]);
                    detail::ewaldAccumulate(output[51], compensation[51], imagePair[51]);
                    detail::ewaldAccumulate(output[52], compensation[52], imagePair[52]);
                    detail::ewaldAccumulate(output[53], compensation[53], imagePair[53]);
                    detail::ewaldAccumulate(output[54], compensation[54], imagePair[54]);
                    detail::ewaldAccumulate(output[55], compensation[55], imagePair[55]);
                    detail::ewaldAccumulate(output[56], compensation[56], imagePair[56]);
                    detail::ewaldAccumulate(output[57], compensation[57], imagePair[57]);
                    detail::ewaldAccumulate(output[58], compensation[58], imagePair[58]);
                    detail::ewaldAccumulate(output[59], compensation[59], imagePair[59]);
                    detail::ewaldAccumulate(output[60], compensation[60], imagePair[60]);
                    detail::ewaldAccumulate(output[61], compensation[61], imagePair[61]);
                    detail::ewaldAccumulate(output[62], compensation[62], imagePair[62]);
                    detail::ewaldAccumulate(output[63], compensation[63], imagePair[63]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                    Real const waveRadial4 =
                        simdBroadcast<Real>(14.843290062324177) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[16], compensation[16],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[16]);
                    detail::ewaldAccumulate(output[17], compensation[17],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[17]);
                    detail::ewaldAccumulate(output[18], compensation[18],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[18]);
                    detail::ewaldAccumulate(output[19], compensation[19],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[19]);
                    detail::ewaldAccumulate(output[20], compensation[20],
                        simdBroadcast<Real>(576) * waveRadial4 * harmonics[20]);
                    detail::ewaldAccumulate(output[21], compensation[21],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[21]);
                    detail::ewaldAccumulate(output[22], compensation[22],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[22]);
                    detail::ewaldAccumulate(output[23], compensation[23],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[23]);
                    detail::ewaldAccumulate(output[24], compensation[24],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[24]);
                    Real const waveRadial5 =
                        simdBroadcast<Real>(10.36257133664445) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[25], compensation[25],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[25]);
                    detail::ewaldAccumulate(output[26], compensation[26],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[26]);
                    detail::ewaldAccumulate(output[27], compensation[27],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[27]);
                    detail::ewaldAccumulate(output[28], compensation[28],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[28]);
                    detail::ewaldAccumulate(output[29], compensation[29],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[29]);
                    detail::ewaldAccumulate(output[30], compensation[30],
                        simdBroadcast<Real>(14400) * waveRadial5 * harmonics[30]);
                    detail::ewaldAccumulate(output[31], compensation[31],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[31]);
                    detail::ewaldAccumulate(output[32], compensation[32],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[32]);
                    detail::ewaldAccumulate(output[33], compensation[33],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[33]);
                    detail::ewaldAccumulate(output[34], compensation[34],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[34]);
                    detail::ewaldAccumulate(output[35], compensation[35],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[35]);
                    Real const waveRadial6 =
                        simdBroadcast<Real>(5.9190869060913398) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[36], compensation[36],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[36]);
                    detail::ewaldAccumulate(output[37], compensation[37],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[37]);
                    detail::ewaldAccumulate(output[38], compensation[38],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[38]);
                    detail::ewaldAccumulate(output[39], compensation[39],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[39]);
                    detail::ewaldAccumulate(output[40], compensation[40],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[40]);
                    detail::ewaldAccumulate(output[41], compensation[41],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[41]);
                    detail::ewaldAccumulate(output[42], compensation[42],
                        simdBroadcast<Real>(518400) * waveRadial6 * harmonics[42]);
                    detail::ewaldAccumulate(output[43], compensation[43],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[43]);
                    detail::ewaldAccumulate(output[44], compensation[44],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[44]);
                    detail::ewaldAccumulate(output[45], compensation[45],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[45]);
                    detail::ewaldAccumulate(output[46], compensation[46],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[46]);
                    detail::ewaldAccumulate(output[47], compensation[47],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[47]);
                    detail::ewaldAccumulate(output[48], compensation[48],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[48]);
                    Real const waveRadial7 =
                        simdBroadcast<Real>(2.8608246061747828) * wavePotential * sine;
                    detail::ewaldAccumulate(output[49], compensation[49],
                        simdBroadcast<Real>(87178291200) * waveRadial7 * harmonics[49]);
                    detail::ewaldAccumulate(output[50], compensation[50],
                        simdBroadcast<Real>(6227020800) * waveRadial7 * harmonics[50]);
                    detail::ewaldAccumulate(output[51], compensation[51],
                        simdBroadcast<Real>(958003200) * waveRadial7 * harmonics[51]);
                    detail::ewaldAccumulate(output[52], compensation[52],
                        simdBroadcast<Real>(239500800) * waveRadial7 * harmonics[52]);
                    detail::ewaldAccumulate(output[53], compensation[53],
                        simdBroadcast<Real>(87091200) * waveRadial7 * harmonics[53]);
                    detail::ewaldAccumulate(output[54], compensation[54],
                        simdBroadcast<Real>(43545600) * waveRadial7 * harmonics[54]);
                    detail::ewaldAccumulate(output[55], compensation[55],
                        simdBroadcast<Real>(29030400) * waveRadial7 * harmonics[55]);
                    detail::ewaldAccumulate(output[56], compensation[56],
                        simdBroadcast<Real>(25401600) * waveRadial7 * harmonics[56]);
                    detail::ewaldAccumulate(output[57], compensation[57],
                        simdBroadcast<Real>(29030400) * waveRadial7 * harmonics[57]);
                    detail::ewaldAccumulate(output[58], compensation[58],
                        simdBroadcast<Real>(43545600) * waveRadial7 * harmonics[58]);
                    detail::ewaldAccumulate(output[59], compensation[59],
                        simdBroadcast<Real>(87091200) * waveRadial7 * harmonics[59]);
                    detail::ewaldAccumulate(output[60], compensation[60],
                        simdBroadcast<Real>(239500800) * waveRadial7 * harmonics[60]);
                    detail::ewaldAccumulate(output[61], compensation[61],
                        simdBroadcast<Real>(958003200) * waveRadial7 * harmonics[61]);
                    detail::ewaldAccumulate(output[62], compensation[62],
                        simdBroadcast<Real>(6227020800) * waveRadial7 * harmonics[62]);
                    detail::ewaldAccumulate(output[63], compensation[63],
                        simdBroadcast<Real>(87178291200) * waveRadial7 * harmonics[63]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[64];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 64; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<7> {
public:
    static constexpr int kFlops_ = 1396;
    static constexpr int kConventionalFlops_ = 2649;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<7>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 9) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 9)
template <>
class EwaldGreen<8> {
public:
    static constexpr int kFlops_ = 951153;
    static constexpr int kConventionalFlops_ = 1039431;
    static constexpr int kDegree_ = 8;
    static constexpr int kCount_ = 81;
    static constexpr int kLatticeRadius_ = 6;
    template <typename Real>
    CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
        output[0] = simdBroadcast<Real>(1);
        output[2] = z * output[0];
        Real const zFactor2_0 = simdBroadcast<Real>(3) * z;
        output[6] =
            simdBroadcast<Real>(0.25) * simdFma(zFactor2_0, output[2], -radiusSquared * output[0]);
        Real const zFactor3_0 = simdBroadcast<Real>(5) * z;
        output[12] = simdBroadcast<Real>(0.1111111111111111) *
            simdFma(zFactor3_0, output[6], -radiusSquared * output[2]);
        Real const zFactor4_0 = simdBroadcast<Real>(7) * z;
        output[20] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor4_0, output[12], -radiusSquared * output[6]);
        Real const zFactor5_0 = simdBroadcast<Real>(9) * z;
        output[30] = simdBroadcast<Real>(0.040000000000000001) *
            simdFma(zFactor5_0, output[20], -radiusSquared * output[12]);
        Real const zFactor6_0 = simdBroadcast<Real>(11) * z;
        output[42] = simdBroadcast<Real>(0.027777777777777776) *
            simdFma(zFactor6_0, output[30], -radiusSquared * output[20]);
        Real const zFactor7_0 = simdBroadcast<Real>(13) * z;
        output[56] = simdBroadcast<Real>(0.020408163265306121) *
            simdFma(zFactor7_0, output[42], -radiusSquared * output[30]);
        Real const zFactor8_0 = simdBroadcast<Real>(15) * z;
        output[72] = simdBroadcast<Real>(0.015625) *
            simdFma(zFactor8_0, output[56], -radiusSquared * output[42]);
        output[3] = x * simdBroadcast<Real>(0.5);
        output[1] = y * simdBroadcast<Real>(0.5);
        output[7] = z * output[3];
        output[5] = z * output[1];
        Real const zFactor3_1 = simdBroadcast<Real>(5) * z;
        output[13] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[7], -radiusSquared * output[3]);
        output[11] =
            simdBroadcast<Real>(0.125) * simdFma(zFactor3_1, output[5], -radiusSquared * output[1]);
        Real const zFactor4_1 = simdBroadcast<Real>(7) * z;
        output[21] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[13], -radiusSquared * output[7]);
        output[19] = simdBroadcast<Real>(0.066666666666666666) *
            simdFma(zFactor4_1, output[11], -radiusSquared * output[5]);
        Real const zFactor5_1 = simdBroadcast<Real>(9) * z;
        output[31] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[21], -radiusSquared * output[13]);
        output[29] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor5_1, output[19], -radiusSquared * output[11]);
        Real const zFactor6_1 = simdBroadcast<Real>(11) * z;
        output[43] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[31], -radiusSquared * output[21]);
        output[41] = simdBroadcast<Real>(0.028571428571428571) *
            simdFma(zFactor6_1, output[29], -radiusSquared * output[19]);
        Real const zFactor7_1 = simdBroadcast<Real>(13) * z;
        output[57] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor7_1, output[43], -radiusSquared * output[31]);
        output[55] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor7_1, output[41], -radiusSquared * output[29]);
        Real const zFactor8_1 = simdBroadcast<Real>(15) * z;
        output[73] = simdBroadcast<Real>(0.015873015873015872) *
            simdFma(zFactor8_1, output[57], -radiusSquared * output[43]);
        output[71] = simdBroadcast<Real>(0.015873015873015872) *
            simdFma(zFactor8_1, output[55], -radiusSquared * output[41]);
        output[8] = simdBroadcast<Real>(0.25) * simdFma(x, output[3], -y * output[1]);
        output[4] = simdBroadcast<Real>(0.25) * simdFma(y, output[3], x * output[1]);
        output[14] = z * output[8];
        output[10] = z * output[4];
        Real const zFactor4_2 = simdBroadcast<Real>(7) * z;
        output[22] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[14], -radiusSquared * output[8]);
        output[18] = simdBroadcast<Real>(0.083333333333333329) *
            simdFma(zFactor4_2, output[10], -radiusSquared * output[4]);
        Real const zFactor5_2 = simdBroadcast<Real>(9) * z;
        output[32] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[22], -radiusSquared * output[14]);
        output[28] = simdBroadcast<Real>(0.047619047619047616) *
            simdFma(zFactor5_2, output[18], -radiusSquared * output[10]);
        Real const zFactor6_2 = simdBroadcast<Real>(11) * z;
        output[44] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[32], -radiusSquared * output[22]);
        output[40] = simdBroadcast<Real>(0.03125) *
            simdFma(zFactor6_2, output[28], -radiusSquared * output[18]);
        Real const zFactor7_2 = simdBroadcast<Real>(13) * z;
        output[58] = simdBroadcast<Real>(0.022222222222222223) *
            simdFma(zFactor7_2, output[44], -radiusSquared * output[32]);
        output[54] = simdBroadcast<Real>(0.022222222222222223) *
            simdFma(zFactor7_2, output[40], -radiusSquared * output[28]);
        Real const zFactor8_2 = simdBroadcast<Real>(15) * z;
        output[74] = simdBroadcast<Real>(0.016666666666666666) *
            simdFma(zFactor8_2, output[58], -radiusSquared * output[44]);
        output[70] = simdBroadcast<Real>(0.016666666666666666) *
            simdFma(zFactor8_2, output[54], -radiusSquared * output[40]);
        output[15] =
            simdBroadcast<Real>(0.16666666666666666) * simdFma(x, output[8], -y * output[4]);
        output[9] = simdBroadcast<Real>(0.16666666666666666) * simdFma(y, output[8], x * output[4]);
        output[23] = z * output[15];
        output[17] = z * output[9];
        Real const zFactor5_3 = simdBroadcast<Real>(9) * z;
        output[33] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[23], -radiusSquared * output[15]);
        output[27] = simdBroadcast<Real>(0.0625) *
            simdFma(zFactor5_3, output[17], -radiusSquared * output[9]);
        Real const zFactor6_3 = simdBroadcast<Real>(11) * z;
        output[45] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[33], -radiusSquared * output[23]);
        output[39] = simdBroadcast<Real>(0.037037037037037035) *
            simdFma(zFactor6_3, output[27], -radiusSquared * output[17]);
        Real const zFactor7_3 = simdBroadcast<Real>(13) * z;
        output[59] = simdBroadcast<Real>(0.025000000000000001) *
            simdFma(zFactor7_3, output[45], -radiusSquared * output[33]);
        output[53] = simdBroadcast<Real>(0.025000000000000001) *
            simdFma(zFactor7_3, output[39], -radiusSquared * output[27]);
        Real const zFactor8_3 = simdBroadcast<Real>(15) * z;
        output[75] = simdBroadcast<Real>(0.018181818181818181) *
            simdFma(zFactor8_3, output[59], -radiusSquared * output[45]);
        output[69] = simdBroadcast<Real>(0.018181818181818181) *
            simdFma(zFactor8_3, output[53], -radiusSquared * output[39]);
        output[24] = simdBroadcast<Real>(0.125) * simdFma(x, output[15], -y * output[9]);
        output[16] = simdBroadcast<Real>(0.125) * simdFma(y, output[15], x * output[9]);
        output[34] = z * output[24];
        output[26] = z * output[16];
        Real const zFactor6_4 = simdBroadcast<Real>(11) * z;
        output[46] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[34], -radiusSquared * output[24]);
        output[38] = simdBroadcast<Real>(0.050000000000000003) *
            simdFma(zFactor6_4, output[26], -radiusSquared * output[16]);
        Real const zFactor7_4 = simdBroadcast<Real>(13) * z;
        output[60] = simdBroadcast<Real>(0.030303030303030304) *
            simdFma(zFactor7_4, output[46], -radiusSquared * output[34]);
        output[52] = simdBroadcast<Real>(0.030303030303030304) *
            simdFma(zFactor7_4, output[38], -radiusSquared * output[26]);
        Real const zFactor8_4 = simdBroadcast<Real>(15) * z;
        output[76] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor8_4, output[60], -radiusSquared * output[46]);
        output[68] = simdBroadcast<Real>(0.020833333333333332) *
            simdFma(zFactor8_4, output[52], -radiusSquared * output[38]);
        output[35] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(x, output[24], -y * output[16]);
        output[25] =
            simdBroadcast<Real>(0.10000000000000001) * simdFma(y, output[24], x * output[16]);
        output[47] = z * output[35];
        output[37] = z * output[25];
        Real const zFactor7_5 = simdBroadcast<Real>(13) * z;
        output[61] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor7_5, output[47], -radiusSquared * output[35]);
        output[51] = simdBroadcast<Real>(0.041666666666666664) *
            simdFma(zFactor7_5, output[37], -radiusSquared * output[25]);
        Real const zFactor8_5 = simdBroadcast<Real>(15) * z;
        output[77] = simdBroadcast<Real>(0.02564102564102564) *
            simdFma(zFactor8_5, output[61], -radiusSquared * output[47]);
        output[67] = simdBroadcast<Real>(0.02564102564102564) *
            simdFma(zFactor8_5, output[51], -radiusSquared * output[37]);
        output[48] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(x, output[35], -y * output[25]);
        output[36] =
            simdBroadcast<Real>(0.083333333333333329) * simdFma(y, output[35], x * output[25]);
        output[62] = z * output[48];
        output[50] = z * output[36];
        Real const zFactor8_6 = simdBroadcast<Real>(15) * z;
        output[78] = simdBroadcast<Real>(0.035714285714285712) *
            simdFma(zFactor8_6, output[62], -radiusSquared * output[48]);
        output[66] = simdBroadcast<Real>(0.035714285714285712) *
            simdFma(zFactor8_6, output[50], -radiusSquared * output[36]);
        output[63] =
            simdBroadcast<Real>(0.071428571428571425) * simdFma(x, output[48], -y * output[36]);
        output[49] =
            simdBroadcast<Real>(0.071428571428571425) * simdFma(y, output[48], x * output[36]);
        output[79] = z * output[63];
        output[65] = z * output[49];
        output[80] = simdBroadcast<Real>(0.0625) * simdFma(x, output[63], -y * output[49]);
        output[64] = simdBroadcast<Real>(0.0625) * simdFma(y, output[63], x * output[49]);
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluateWorking(
        Output& output, Real& greenTrace, Real x, Real y, Real z) {
        Real harmonics[81];
        Real compensation[81];
        Real imagePair[81];
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        regular(harmonics, x, y, z, radiusSquared);
        Real const argument = simdBroadcast<Real>(4) * radiusSquared;
        Real const exponential = detail::ewaldExp(-argument);
        Real series0 = simdBroadcast<Real>(1);
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        series0 = simdFma(
            series0, argument * simdBroadcast<Real>(0.66666666666666663), simdBroadcast<Real>(1));
        Real const radial0 = simdBroadcast<Real>(2.2567583341910251) * exponential * series0;
        output[0] = radial0 * harmonics[0];
        compensation[0] = simdBroadcast<Real>(0);
        Real series1 = simdBroadcast<Real>(1);
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        series1 = simdFma(
            series1, argument * simdBroadcast<Real>(0.40000000000000002), simdBroadcast<Real>(1));
        Real const radial1 = simdBroadcast<Real>(-6.0180222245093997) * exponential * series1;
        output[1] = simdBroadcast<Real>(2) * radial1 * harmonics[1];
        compensation[1] = simdBroadcast<Real>(0);
        output[2] = radial1 * harmonics[2];
        compensation[2] = simdBroadcast<Real>(0);
        output[3] = simdBroadcast<Real>(2) * radial1 * harmonics[3];
        compensation[3] = simdBroadcast<Real>(0);
        Real series2 = simdBroadcast<Real>(1);
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        series2 = simdFma(
            series2, argument * simdBroadcast<Real>(0.2857142857142857), simdBroadcast<Real>(1));
        Real const radial2 = simdBroadcast<Real>(9.6288355592150427) * exponential * series2;
        output[4] = simdBroadcast<Real>(24) * radial2 * harmonics[4];
        compensation[4] = simdBroadcast<Real>(0);
        output[5] = simdBroadcast<Real>(6) * radial2 * harmonics[5];
        compensation[5] = simdBroadcast<Real>(0);
        output[6] = simdBroadcast<Real>(4) * radial2 * harmonics[6];
        compensation[6] = simdBroadcast<Real>(0);
        output[7] = simdBroadcast<Real>(6) * radial2 * harmonics[7];
        compensation[7] = simdBroadcast<Real>(0);
        output[8] = simdBroadcast<Real>(24) * radial2 * harmonics[8];
        compensation[8] = simdBroadcast<Real>(0);
        Real series3 = simdBroadcast<Real>(1);
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        series3 = simdFma(
            series3, argument * simdBroadcast<Real>(0.22222222222222221), simdBroadcast<Real>(1));
        Real const radial3 = simdBroadcast<Real>(-11.004383496245763) * exponential * series3;
        output[9] = simdBroadcast<Real>(720) * radial3 * harmonics[9];
        compensation[9] = simdBroadcast<Real>(0);
        output[10] = simdBroadcast<Real>(120) * radial3 * harmonics[10];
        compensation[10] = simdBroadcast<Real>(0);
        output[11] = simdBroadcast<Real>(48) * radial3 * harmonics[11];
        compensation[11] = simdBroadcast<Real>(0);
        output[12] = simdBroadcast<Real>(36) * radial3 * harmonics[12];
        compensation[12] = simdBroadcast<Real>(0);
        output[13] = simdBroadcast<Real>(48) * radial3 * harmonics[13];
        compensation[13] = simdBroadcast<Real>(0);
        output[14] = simdBroadcast<Real>(120) * radial3 * harmonics[14];
        compensation[14] = simdBroadcast<Real>(0);
        output[15] = simdBroadcast<Real>(720) * radial3 * harmonics[15];
        compensation[15] = simdBroadcast<Real>(0);
        Real series4 = simdBroadcast<Real>(1);
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        series4 = simdFma(
            series4, argument * simdBroadcast<Real>(0.18181818181818182), simdBroadcast<Real>(1));
        Real const radial4 = simdBroadcast<Real>(9.7816742188851205) * exponential * series4;
        output[16] = simdBroadcast<Real>(40320) * radial4 * harmonics[16];
        compensation[16] = simdBroadcast<Real>(0);
        output[17] = simdBroadcast<Real>(5040) * radial4 * harmonics[17];
        compensation[17] = simdBroadcast<Real>(0);
        output[18] = simdBroadcast<Real>(1440) * radial4 * harmonics[18];
        compensation[18] = simdBroadcast<Real>(0);
        output[19] = simdBroadcast<Real>(720) * radial4 * harmonics[19];
        compensation[19] = simdBroadcast<Real>(0);
        output[20] = simdBroadcast<Real>(576) * radial4 * harmonics[20];
        compensation[20] = simdBroadcast<Real>(0);
        output[21] = simdBroadcast<Real>(720) * radial4 * harmonics[21];
        compensation[21] = simdBroadcast<Real>(0);
        output[22] = simdBroadcast<Real>(1440) * radial4 * harmonics[22];
        compensation[22] = simdBroadcast<Real>(0);
        output[23] = simdBroadcast<Real>(5040) * radial4 * harmonics[23];
        compensation[23] = simdBroadcast<Real>(0);
        output[24] = simdBroadcast<Real>(40320) * radial4 * harmonics[24];
        compensation[24] = simdBroadcast<Real>(0);
        Real series5 = simdBroadcast<Real>(1);
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        series5 = simdFma(
            series5, argument * simdBroadcast<Real>(0.15384615384615385), simdBroadcast<Real>(1));
        Real const radial5 = simdBroadcast<Real>(-7.1139448864619048) * exponential * series5;
        output[25] = simdBroadcast<Real>(3628800) * radial5 * harmonics[25];
        compensation[25] = simdBroadcast<Real>(0);
        output[26] = simdBroadcast<Real>(362880) * radial5 * harmonics[26];
        compensation[26] = simdBroadcast<Real>(0);
        output[27] = simdBroadcast<Real>(80640) * radial5 * harmonics[27];
        compensation[27] = simdBroadcast<Real>(0);
        output[28] = simdBroadcast<Real>(30240) * radial5 * harmonics[28];
        compensation[28] = simdBroadcast<Real>(0);
        output[29] = simdBroadcast<Real>(17280) * radial5 * harmonics[29];
        compensation[29] = simdBroadcast<Real>(0);
        output[30] = simdBroadcast<Real>(14400) * radial5 * harmonics[30];
        compensation[30] = simdBroadcast<Real>(0);
        output[31] = simdBroadcast<Real>(17280) * radial5 * harmonics[31];
        compensation[31] = simdBroadcast<Real>(0);
        output[32] = simdBroadcast<Real>(30240) * radial5 * harmonics[32];
        compensation[32] = simdBroadcast<Real>(0);
        output[33] = simdBroadcast<Real>(80640) * radial5 * harmonics[33];
        compensation[33] = simdBroadcast<Real>(0);
        output[34] = simdBroadcast<Real>(362880) * radial5 * harmonics[34];
        compensation[34] = simdBroadcast<Real>(0);
        output[35] = simdBroadcast<Real>(3628800) * radial5 * harmonics[35];
        compensation[35] = simdBroadcast<Real>(0);
        Real series6 = simdBroadcast<Real>(1);
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        series6 = simdFma(
            series6, argument * simdBroadcast<Real>(0.13333333333333333), simdBroadcast<Real>(1));
        Real const radial6 = simdBroadcast<Real>(4.3778122378227113) * exponential * series6;
        output[36] = simdBroadcast<Real>(479001600) * radial6 * harmonics[36];
        compensation[36] = simdBroadcast<Real>(0);
        output[37] = simdBroadcast<Real>(39916800) * radial6 * harmonics[37];
        compensation[37] = simdBroadcast<Real>(0);
        output[38] = simdBroadcast<Real>(7257600) * radial6 * harmonics[38];
        compensation[38] = simdBroadcast<Real>(0);
        output[39] = simdBroadcast<Real>(2177280) * radial6 * harmonics[39];
        compensation[39] = simdBroadcast<Real>(0);
        output[40] = simdBroadcast<Real>(967680) * radial6 * harmonics[40];
        compensation[40] = simdBroadcast<Real>(0);
        output[41] = simdBroadcast<Real>(604800) * radial6 * harmonics[41];
        compensation[41] = simdBroadcast<Real>(0);
        output[42] = simdBroadcast<Real>(518400) * radial6 * harmonics[42];
        compensation[42] = simdBroadcast<Real>(0);
        output[43] = simdBroadcast<Real>(604800) * radial6 * harmonics[43];
        compensation[43] = simdBroadcast<Real>(0);
        output[44] = simdBroadcast<Real>(967680) * radial6 * harmonics[44];
        compensation[44] = simdBroadcast<Real>(0);
        output[45] = simdBroadcast<Real>(2177280) * radial6 * harmonics[45];
        compensation[45] = simdBroadcast<Real>(0);
        output[46] = simdBroadcast<Real>(7257600) * radial6 * harmonics[46];
        compensation[46] = simdBroadcast<Real>(0);
        output[47] = simdBroadcast<Real>(39916800) * radial6 * harmonics[47];
        compensation[47] = simdBroadcast<Real>(0);
        output[48] = simdBroadcast<Real>(479001600) * radial6 * harmonics[48];
        compensation[48] = simdBroadcast<Real>(0);
        Real series7 = simdBroadcast<Real>(1);
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.018018018018018018), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        series7 = simdFma(
            series7, argument * simdBroadcast<Real>(0.11764705882352941), simdBroadcast<Real>(1));
        Real const radial7 = simdBroadcast<Real>(-2.334833193505446) * exponential * series7;
        output[49] = simdBroadcast<Real>(87178291200) * radial7 * harmonics[49];
        compensation[49] = simdBroadcast<Real>(0);
        output[50] = simdBroadcast<Real>(6227020800) * radial7 * harmonics[50];
        compensation[50] = simdBroadcast<Real>(0);
        output[51] = simdBroadcast<Real>(958003200) * radial7 * harmonics[51];
        compensation[51] = simdBroadcast<Real>(0);
        output[52] = simdBroadcast<Real>(239500800) * radial7 * harmonics[52];
        compensation[52] = simdBroadcast<Real>(0);
        output[53] = simdBroadcast<Real>(87091200) * radial7 * harmonics[53];
        compensation[53] = simdBroadcast<Real>(0);
        output[54] = simdBroadcast<Real>(43545600) * radial7 * harmonics[54];
        compensation[54] = simdBroadcast<Real>(0);
        output[55] = simdBroadcast<Real>(29030400) * radial7 * harmonics[55];
        compensation[55] = simdBroadcast<Real>(0);
        output[56] = simdBroadcast<Real>(25401600) * radial7 * harmonics[56];
        compensation[56] = simdBroadcast<Real>(0);
        output[57] = simdBroadcast<Real>(29030400) * radial7 * harmonics[57];
        compensation[57] = simdBroadcast<Real>(0);
        output[58] = simdBroadcast<Real>(43545600) * radial7 * harmonics[58];
        compensation[58] = simdBroadcast<Real>(0);
        output[59] = simdBroadcast<Real>(87091200) * radial7 * harmonics[59];
        compensation[59] = simdBroadcast<Real>(0);
        output[60] = simdBroadcast<Real>(239500800) * radial7 * harmonics[60];
        compensation[60] = simdBroadcast<Real>(0);
        output[61] = simdBroadcast<Real>(958003200) * radial7 * harmonics[61];
        compensation[61] = simdBroadcast<Real>(0);
        output[62] = simdBroadcast<Real>(6227020800) * radial7 * harmonics[62];
        compensation[62] = simdBroadcast<Real>(0);
        output[63] = simdBroadcast<Real>(87178291200) * radial7 * harmonics[63];
        compensation[63] = simdBroadcast<Real>(0);
        Real series8 = simdBroadcast<Real>(1);
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.017699115044247787), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.018018018018018018), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.01834862385321101), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.018691588785046728), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.019047619047619049), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.019417475728155338), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.019801980198019802), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.020202020202020204), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.020618556701030927), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.021052631578947368), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.021505376344086023), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.02197802197802198), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.02247191011235955), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.022988505747126436), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.023529411764705882), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.024096385542168676), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.024691358024691357), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.025316455696202531), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.025974025974025976), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.026666666666666668), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.027397260273972601), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.028169014084507043), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.028985507246376812), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.029850746268656716), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.030769230769230771), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.031746031746031744), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.032786885245901641), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.033898305084745763), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.035087719298245612), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.036363636363636362), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.037735849056603772), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.039215686274509803), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.040816326530612242), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.042553191489361701), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.044444444444444446), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.046511627906976744), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.04878048780487805), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.05128205128205128), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.054054054054054057), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.057142857142857141), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.060606060606060608), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.064516129032258063), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.068965517241379309), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.07407407407407407), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.080000000000000002), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.086956521739130432), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.095238095238095233), simdBroadcast<Real>(1));
        series8 = simdFma(
            series8, argument * simdBroadcast<Real>(0.10526315789473684), simdBroadcast<Real>(1));
        Real const radial8 = simdBroadcast<Real>(1.0987450322378569) * exponential * series8;
        output[64] = simdBroadcast<Real>(20922789888000) * radial8 * harmonics[64];
        compensation[64] = simdBroadcast<Real>(0);
        output[65] = simdBroadcast<Real>(1307674368000) * radial8 * harmonics[65];
        compensation[65] = simdBroadcast<Real>(0);
        output[66] = simdBroadcast<Real>(174356582400) * radial8 * harmonics[66];
        compensation[66] = simdBroadcast<Real>(0);
        output[67] = simdBroadcast<Real>(37362124800) * radial8 * harmonics[67];
        compensation[67] = simdBroadcast<Real>(0);
        output[68] = simdBroadcast<Real>(11496038400) * radial8 * harmonics[68];
        compensation[68] = simdBroadcast<Real>(0);
        output[69] = simdBroadcast<Real>(4790016000) * radial8 * harmonics[69];
        compensation[69] = simdBroadcast<Real>(0);
        output[70] = simdBroadcast<Real>(2612736000) * radial8 * harmonics[70];
        compensation[70] = simdBroadcast<Real>(0);
        output[71] = simdBroadcast<Real>(1828915200) * radial8 * harmonics[71];
        compensation[71] = simdBroadcast<Real>(0);
        output[72] = simdBroadcast<Real>(1625702400) * radial8 * harmonics[72];
        compensation[72] = simdBroadcast<Real>(0);
        output[73] = simdBroadcast<Real>(1828915200) * radial8 * harmonics[73];
        compensation[73] = simdBroadcast<Real>(0);
        output[74] = simdBroadcast<Real>(2612736000) * radial8 * harmonics[74];
        compensation[74] = simdBroadcast<Real>(0);
        output[75] = simdBroadcast<Real>(4790016000) * radial8 * harmonics[75];
        compensation[75] = simdBroadcast<Real>(0);
        output[76] = simdBroadcast<Real>(11496038400) * radial8 * harmonics[76];
        compensation[76] = simdBroadcast<Real>(0);
        output[77] = simdBroadcast<Real>(37362124800) * radial8 * harmonics[77];
        compensation[77] = simdBroadcast<Real>(0);
        output[78] = simdBroadcast<Real>(174356582400) * radial8 * harmonics[78];
        compensation[78] = simdBroadcast<Real>(0);
        output[79] = simdBroadcast<Real>(1307674368000) * radial8 * harmonics[79];
        compensation[79] = simdBroadcast<Real>(0);
        output[80] = simdBroadcast<Real>(20922789888000) * radial8 * harmonics[80];
        compensation[80] = simdBroadcast<Real>(0);
        output[0] += simdBroadcast<Real>(0.78539816339744828);
        greenTrace = simdBroadcast<Real>(4.1887902047863905);
        for (int ix = -6; ix <= 6; ++ix) {
            for (int iy = -6; iy <= 6; ++iy) {
                for (int iz = -6; iz <= 6; ++iz) {
                    int const latticeSquared = ix * ix + iy * iy + iz * iz;
                    if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 ||
                        (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {
                        continue;
                    }
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Real const imageX = x + simdBroadcast<Real>(sign * ix);
                        Real const imageY = y + simdBroadcast<Real>(sign * iy);
                        Real const imageZ = z + simdBroadcast<Real>(sign * iz);
                        Real const imageSquared =
                            simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));
                        Real const imageRadius = simdSqrt(imageSquared);
                        Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;
                        Real const inverseSquared = inverseRadius * inverseRadius;
                        regular(harmonics, imageX, imageY, imageZ, imageSquared);
                        Real const imageArgument = simdBroadcast<Real>(4) * imageSquared;
                        Real const imageExp = detail::ewaldExp(-imageArgument);
                        Real screening = detail::ewaldErfc(simdBroadcast<Real>(2) * imageRadius);
                        Real gammaTerm =
                            simdBroadcast<Real>(2.2567583341910251) * imageRadius * imageExp;
                        Real inversePower = inverseRadius;
                        Real const imageRadial0 = -screening * inversePower;
                        Real const imageValue0 = imageRadial0 * harmonics[0];
                        if (sign < 0) {
                            imagePair[0] = imageValue0;
                        } else {
                            imagePair[0] += imageValue0;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.66666666666666663);
                        Real const imageRadial1 = screening * inversePower;
                        Real const imageValue1 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[1];
                        if (sign < 0) {
                            imagePair[1] = imageValue1;
                        } else {
                            imagePair[1] += imageValue1;
                        }
                        Real const imageValue2 = imageRadial1 * harmonics[2];
                        if (sign < 0) {
                            imagePair[2] = imageValue2;
                        } else {
                            imagePair[2] += imageValue2;
                        }
                        Real const imageValue3 =
                            simdBroadcast<Real>(2) * imageRadial1 * harmonics[3];
                        if (sign < 0) {
                            imagePair[3] = imageValue3;
                        } else {
                            imagePair[3] += imageValue3;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.40000000000000002);
                        Real const imageRadial2 = -screening * inversePower;
                        Real const imageValue4 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[4];
                        if (sign < 0) {
                            imagePair[4] = imageValue4;
                        } else {
                            imagePair[4] += imageValue4;
                        }
                        Real const imageValue5 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[5];
                        if (sign < 0) {
                            imagePair[5] = imageValue5;
                        } else {
                            imagePair[5] += imageValue5;
                        }
                        Real const imageValue6 =
                            simdBroadcast<Real>(4) * imageRadial2 * harmonics[6];
                        if (sign < 0) {
                            imagePair[6] = imageValue6;
                        } else {
                            imagePair[6] += imageValue6;
                        }
                        Real const imageValue7 =
                            simdBroadcast<Real>(6) * imageRadial2 * harmonics[7];
                        if (sign < 0) {
                            imagePair[7] = imageValue7;
                        } else {
                            imagePair[7] += imageValue7;
                        }
                        Real const imageValue8 =
                            simdBroadcast<Real>(24) * imageRadial2 * harmonics[8];
                        if (sign < 0) {
                            imagePair[8] = imageValue8;
                        } else {
                            imagePair[8] += imageValue8;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.2857142857142857);
                        Real const imageRadial3 = screening * inversePower;
                        Real const imageValue9 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[9];
                        if (sign < 0) {
                            imagePair[9] = imageValue9;
                        } else {
                            imagePair[9] += imageValue9;
                        }
                        Real const imageValue10 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[10];
                        if (sign < 0) {
                            imagePair[10] = imageValue10;
                        } else {
                            imagePair[10] += imageValue10;
                        }
                        Real const imageValue11 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[11];
                        if (sign < 0) {
                            imagePair[11] = imageValue11;
                        } else {
                            imagePair[11] += imageValue11;
                        }
                        Real const imageValue12 =
                            simdBroadcast<Real>(36) * imageRadial3 * harmonics[12];
                        if (sign < 0) {
                            imagePair[12] = imageValue12;
                        } else {
                            imagePair[12] += imageValue12;
                        }
                        Real const imageValue13 =
                            simdBroadcast<Real>(48) * imageRadial3 * harmonics[13];
                        if (sign < 0) {
                            imagePair[13] = imageValue13;
                        } else {
                            imagePair[13] += imageValue13;
                        }
                        Real const imageValue14 =
                            simdBroadcast<Real>(120) * imageRadial3 * harmonics[14];
                        if (sign < 0) {
                            imagePair[14] = imageValue14;
                        } else {
                            imagePair[14] += imageValue14;
                        }
                        Real const imageValue15 =
                            simdBroadcast<Real>(720) * imageRadial3 * harmonics[15];
                        if (sign < 0) {
                            imagePair[15] = imageValue15;
                        } else {
                            imagePair[15] += imageValue15;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.22222222222222221);
                        Real const imageRadial4 = -screening * inversePower;
                        Real const imageValue16 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[16];
                        if (sign < 0) {
                            imagePair[16] = imageValue16;
                        } else {
                            imagePair[16] += imageValue16;
                        }
                        Real const imageValue17 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[17];
                        if (sign < 0) {
                            imagePair[17] = imageValue17;
                        } else {
                            imagePair[17] += imageValue17;
                        }
                        Real const imageValue18 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[18];
                        if (sign < 0) {
                            imagePair[18] = imageValue18;
                        } else {
                            imagePair[18] += imageValue18;
                        }
                        Real const imageValue19 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[19];
                        if (sign < 0) {
                            imagePair[19] = imageValue19;
                        } else {
                            imagePair[19] += imageValue19;
                        }
                        Real const imageValue20 =
                            simdBroadcast<Real>(576) * imageRadial4 * harmonics[20];
                        if (sign < 0) {
                            imagePair[20] = imageValue20;
                        } else {
                            imagePair[20] += imageValue20;
                        }
                        Real const imageValue21 =
                            simdBroadcast<Real>(720) * imageRadial4 * harmonics[21];
                        if (sign < 0) {
                            imagePair[21] = imageValue21;
                        } else {
                            imagePair[21] += imageValue21;
                        }
                        Real const imageValue22 =
                            simdBroadcast<Real>(1440) * imageRadial4 * harmonics[22];
                        if (sign < 0) {
                            imagePair[22] = imageValue22;
                        } else {
                            imagePair[22] += imageValue22;
                        }
                        Real const imageValue23 =
                            simdBroadcast<Real>(5040) * imageRadial4 * harmonics[23];
                        if (sign < 0) {
                            imagePair[23] = imageValue23;
                        } else {
                            imagePair[23] += imageValue23;
                        }
                        Real const imageValue24 =
                            simdBroadcast<Real>(40320) * imageRadial4 * harmonics[24];
                        if (sign < 0) {
                            imagePair[24] = imageValue24;
                        } else {
                            imagePair[24] += imageValue24;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.18181818181818182);
                        Real const imageRadial5 = screening * inversePower;
                        Real const imageValue25 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[25];
                        if (sign < 0) {
                            imagePair[25] = imageValue25;
                        } else {
                            imagePair[25] += imageValue25;
                        }
                        Real const imageValue26 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[26];
                        if (sign < 0) {
                            imagePair[26] = imageValue26;
                        } else {
                            imagePair[26] += imageValue26;
                        }
                        Real const imageValue27 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[27];
                        if (sign < 0) {
                            imagePair[27] = imageValue27;
                        } else {
                            imagePair[27] += imageValue27;
                        }
                        Real const imageValue28 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[28];
                        if (sign < 0) {
                            imagePair[28] = imageValue28;
                        } else {
                            imagePair[28] += imageValue28;
                        }
                        Real const imageValue29 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[29];
                        if (sign < 0) {
                            imagePair[29] = imageValue29;
                        } else {
                            imagePair[29] += imageValue29;
                        }
                        Real const imageValue30 =
                            simdBroadcast<Real>(14400) * imageRadial5 * harmonics[30];
                        if (sign < 0) {
                            imagePair[30] = imageValue30;
                        } else {
                            imagePair[30] += imageValue30;
                        }
                        Real const imageValue31 =
                            simdBroadcast<Real>(17280) * imageRadial5 * harmonics[31];
                        if (sign < 0) {
                            imagePair[31] = imageValue31;
                        } else {
                            imagePair[31] += imageValue31;
                        }
                        Real const imageValue32 =
                            simdBroadcast<Real>(30240) * imageRadial5 * harmonics[32];
                        if (sign < 0) {
                            imagePair[32] = imageValue32;
                        } else {
                            imagePair[32] += imageValue32;
                        }
                        Real const imageValue33 =
                            simdBroadcast<Real>(80640) * imageRadial5 * harmonics[33];
                        if (sign < 0) {
                            imagePair[33] = imageValue33;
                        } else {
                            imagePair[33] += imageValue33;
                        }
                        Real const imageValue34 =
                            simdBroadcast<Real>(362880) * imageRadial5 * harmonics[34];
                        if (sign < 0) {
                            imagePair[34] = imageValue34;
                        } else {
                            imagePair[34] += imageValue34;
                        }
                        Real const imageValue35 =
                            simdBroadcast<Real>(3628800) * imageRadial5 * harmonics[35];
                        if (sign < 0) {
                            imagePair[35] = imageValue35;
                        } else {
                            imagePair[35] += imageValue35;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.15384615384615385);
                        Real const imageRadial6 = -screening * inversePower;
                        Real const imageValue36 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[36];
                        if (sign < 0) {
                            imagePair[36] = imageValue36;
                        } else {
                            imagePair[36] += imageValue36;
                        }
                        Real const imageValue37 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[37];
                        if (sign < 0) {
                            imagePair[37] = imageValue37;
                        } else {
                            imagePair[37] += imageValue37;
                        }
                        Real const imageValue38 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[38];
                        if (sign < 0) {
                            imagePair[38] = imageValue38;
                        } else {
                            imagePair[38] += imageValue38;
                        }
                        Real const imageValue39 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[39];
                        if (sign < 0) {
                            imagePair[39] = imageValue39;
                        } else {
                            imagePair[39] += imageValue39;
                        }
                        Real const imageValue40 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[40];
                        if (sign < 0) {
                            imagePair[40] = imageValue40;
                        } else {
                            imagePair[40] += imageValue40;
                        }
                        Real const imageValue41 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[41];
                        if (sign < 0) {
                            imagePair[41] = imageValue41;
                        } else {
                            imagePair[41] += imageValue41;
                        }
                        Real const imageValue42 =
                            simdBroadcast<Real>(518400) * imageRadial6 * harmonics[42];
                        if (sign < 0) {
                            imagePair[42] = imageValue42;
                        } else {
                            imagePair[42] += imageValue42;
                        }
                        Real const imageValue43 =
                            simdBroadcast<Real>(604800) * imageRadial6 * harmonics[43];
                        if (sign < 0) {
                            imagePair[43] = imageValue43;
                        } else {
                            imagePair[43] += imageValue43;
                        }
                        Real const imageValue44 =
                            simdBroadcast<Real>(967680) * imageRadial6 * harmonics[44];
                        if (sign < 0) {
                            imagePair[44] = imageValue44;
                        } else {
                            imagePair[44] += imageValue44;
                        }
                        Real const imageValue45 =
                            simdBroadcast<Real>(2177280) * imageRadial6 * harmonics[45];
                        if (sign < 0) {
                            imagePair[45] = imageValue45;
                        } else {
                            imagePair[45] += imageValue45;
                        }
                        Real const imageValue46 =
                            simdBroadcast<Real>(7257600) * imageRadial6 * harmonics[46];
                        if (sign < 0) {
                            imagePair[46] = imageValue46;
                        } else {
                            imagePair[46] += imageValue46;
                        }
                        Real const imageValue47 =
                            simdBroadcast<Real>(39916800) * imageRadial6 * harmonics[47];
                        if (sign < 0) {
                            imagePair[47] = imageValue47;
                        } else {
                            imagePair[47] += imageValue47;
                        }
                        Real const imageValue48 =
                            simdBroadcast<Real>(479001600) * imageRadial6 * harmonics[48];
                        if (sign < 0) {
                            imagePair[48] = imageValue48;
                        } else {
                            imagePair[48] += imageValue48;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.13333333333333333);
                        Real const imageRadial7 = screening * inversePower;
                        Real const imageValue49 =
                            simdBroadcast<Real>(87178291200) * imageRadial7 * harmonics[49];
                        if (sign < 0) {
                            imagePair[49] = imageValue49;
                        } else {
                            imagePair[49] += imageValue49;
                        }
                        Real const imageValue50 =
                            simdBroadcast<Real>(6227020800) * imageRadial7 * harmonics[50];
                        if (sign < 0) {
                            imagePair[50] = imageValue50;
                        } else {
                            imagePair[50] += imageValue50;
                        }
                        Real const imageValue51 =
                            simdBroadcast<Real>(958003200) * imageRadial7 * harmonics[51];
                        if (sign < 0) {
                            imagePair[51] = imageValue51;
                        } else {
                            imagePair[51] += imageValue51;
                        }
                        Real const imageValue52 =
                            simdBroadcast<Real>(239500800) * imageRadial7 * harmonics[52];
                        if (sign < 0) {
                            imagePair[52] = imageValue52;
                        } else {
                            imagePair[52] += imageValue52;
                        }
                        Real const imageValue53 =
                            simdBroadcast<Real>(87091200) * imageRadial7 * harmonics[53];
                        if (sign < 0) {
                            imagePair[53] = imageValue53;
                        } else {
                            imagePair[53] += imageValue53;
                        }
                        Real const imageValue54 =
                            simdBroadcast<Real>(43545600) * imageRadial7 * harmonics[54];
                        if (sign < 0) {
                            imagePair[54] = imageValue54;
                        } else {
                            imagePair[54] += imageValue54;
                        }
                        Real const imageValue55 =
                            simdBroadcast<Real>(29030400) * imageRadial7 * harmonics[55];
                        if (sign < 0) {
                            imagePair[55] = imageValue55;
                        } else {
                            imagePair[55] += imageValue55;
                        }
                        Real const imageValue56 =
                            simdBroadcast<Real>(25401600) * imageRadial7 * harmonics[56];
                        if (sign < 0) {
                            imagePair[56] = imageValue56;
                        } else {
                            imagePair[56] += imageValue56;
                        }
                        Real const imageValue57 =
                            simdBroadcast<Real>(29030400) * imageRadial7 * harmonics[57];
                        if (sign < 0) {
                            imagePair[57] = imageValue57;
                        } else {
                            imagePair[57] += imageValue57;
                        }
                        Real const imageValue58 =
                            simdBroadcast<Real>(43545600) * imageRadial7 * harmonics[58];
                        if (sign < 0) {
                            imagePair[58] = imageValue58;
                        } else {
                            imagePair[58] += imageValue58;
                        }
                        Real const imageValue59 =
                            simdBroadcast<Real>(87091200) * imageRadial7 * harmonics[59];
                        if (sign < 0) {
                            imagePair[59] = imageValue59;
                        } else {
                            imagePair[59] += imageValue59;
                        }
                        Real const imageValue60 =
                            simdBroadcast<Real>(239500800) * imageRadial7 * harmonics[60];
                        if (sign < 0) {
                            imagePair[60] = imageValue60;
                        } else {
                            imagePair[60] += imageValue60;
                        }
                        Real const imageValue61 =
                            simdBroadcast<Real>(958003200) * imageRadial7 * harmonics[61];
                        if (sign < 0) {
                            imagePair[61] = imageValue61;
                        } else {
                            imagePair[61] += imageValue61;
                        }
                        Real const imageValue62 =
                            simdBroadcast<Real>(6227020800) * imageRadial7 * harmonics[62];
                        if (sign < 0) {
                            imagePair[62] = imageValue62;
                        } else {
                            imagePair[62] += imageValue62;
                        }
                        Real const imageValue63 =
                            simdBroadcast<Real>(87178291200) * imageRadial7 * harmonics[63];
                        if (sign < 0) {
                            imagePair[63] = imageValue63;
                        } else {
                            imagePair[63] += imageValue63;
                        }
                        inversePower *= inverseSquared;
                        screening += gammaTerm;
                        gammaTerm *= imageArgument * simdBroadcast<Real>(0.11764705882352941);
                        Real const imageRadial8 = -screening * inversePower;
                        Real const imageValue64 =
                            simdBroadcast<Real>(20922789888000) * imageRadial8 * harmonics[64];
                        if (sign < 0) {
                            imagePair[64] = imageValue64;
                        } else {
                            imagePair[64] += imageValue64;
                        }
                        Real const imageValue65 =
                            simdBroadcast<Real>(1307674368000) * imageRadial8 * harmonics[65];
                        if (sign < 0) {
                            imagePair[65] = imageValue65;
                        } else {
                            imagePair[65] += imageValue65;
                        }
                        Real const imageValue66 =
                            simdBroadcast<Real>(174356582400) * imageRadial8 * harmonics[66];
                        if (sign < 0) {
                            imagePair[66] = imageValue66;
                        } else {
                            imagePair[66] += imageValue66;
                        }
                        Real const imageValue67 =
                            simdBroadcast<Real>(37362124800) * imageRadial8 * harmonics[67];
                        if (sign < 0) {
                            imagePair[67] = imageValue67;
                        } else {
                            imagePair[67] += imageValue67;
                        }
                        Real const imageValue68 =
                            simdBroadcast<Real>(11496038400) * imageRadial8 * harmonics[68];
                        if (sign < 0) {
                            imagePair[68] = imageValue68;
                        } else {
                            imagePair[68] += imageValue68;
                        }
                        Real const imageValue69 =
                            simdBroadcast<Real>(4790016000) * imageRadial8 * harmonics[69];
                        if (sign < 0) {
                            imagePair[69] = imageValue69;
                        } else {
                            imagePair[69] += imageValue69;
                        }
                        Real const imageValue70 =
                            simdBroadcast<Real>(2612736000) * imageRadial8 * harmonics[70];
                        if (sign < 0) {
                            imagePair[70] = imageValue70;
                        } else {
                            imagePair[70] += imageValue70;
                        }
                        Real const imageValue71 =
                            simdBroadcast<Real>(1828915200) * imageRadial8 * harmonics[71];
                        if (sign < 0) {
                            imagePair[71] = imageValue71;
                        } else {
                            imagePair[71] += imageValue71;
                        }
                        Real const imageValue72 =
                            simdBroadcast<Real>(1625702400) * imageRadial8 * harmonics[72];
                        if (sign < 0) {
                            imagePair[72] = imageValue72;
                        } else {
                            imagePair[72] += imageValue72;
                        }
                        Real const imageValue73 =
                            simdBroadcast<Real>(1828915200) * imageRadial8 * harmonics[73];
                        if (sign < 0) {
                            imagePair[73] = imageValue73;
                        } else {
                            imagePair[73] += imageValue73;
                        }
                        Real const imageValue74 =
                            simdBroadcast<Real>(2612736000) * imageRadial8 * harmonics[74];
                        if (sign < 0) {
                            imagePair[74] = imageValue74;
                        } else {
                            imagePair[74] += imageValue74;
                        }
                        Real const imageValue75 =
                            simdBroadcast<Real>(4790016000) * imageRadial8 * harmonics[75];
                        if (sign < 0) {
                            imagePair[75] = imageValue75;
                        } else {
                            imagePair[75] += imageValue75;
                        }
                        Real const imageValue76 =
                            simdBroadcast<Real>(11496038400) * imageRadial8 * harmonics[76];
                        if (sign < 0) {
                            imagePair[76] = imageValue76;
                        } else {
                            imagePair[76] += imageValue76;
                        }
                        Real const imageValue77 =
                            simdBroadcast<Real>(37362124800) * imageRadial8 * harmonics[77];
                        if (sign < 0) {
                            imagePair[77] = imageValue77;
                        } else {
                            imagePair[77] += imageValue77;
                        }
                        Real const imageValue78 =
                            simdBroadcast<Real>(174356582400) * imageRadial8 * harmonics[78];
                        if (sign < 0) {
                            imagePair[78] = imageValue78;
                        } else {
                            imagePair[78] += imageValue78;
                        }
                        Real const imageValue79 =
                            simdBroadcast<Real>(1307674368000) * imageRadial8 * harmonics[79];
                        if (sign < 0) {
                            imagePair[79] = imageValue79;
                        } else {
                            imagePair[79] += imageValue79;
                        }
                        Real const imageValue80 =
                            simdBroadcast<Real>(20922789888000) * imageRadial8 * harmonics[80];
                        if (sign < 0) {
                            imagePair[80] = imageValue80;
                        } else {
                            imagePair[80] += imageValue80;
                        }
                    }
                    detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                    detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                    detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                    detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                    detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                    detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                    detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                    detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                    detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                    detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                    detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                    detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                    detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                    detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                    detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                    detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                    detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                    detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                    detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                    detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                    detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                    detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                    detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                    detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                    detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                    detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                    detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                    detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                    detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                    detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                    detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                    detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                    detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                    detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                    detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                    detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                    detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                    detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                    detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                    detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                    detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                    detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                    detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                    detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                    detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                    detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                    detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                    detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                    detail::ewaldAccumulate(output[49], compensation[49], imagePair[49]);
                    detail::ewaldAccumulate(output[50], compensation[50], imagePair[50]);
                    detail::ewaldAccumulate(output[51], compensation[51], imagePair[51]);
                    detail::ewaldAccumulate(output[52], compensation[52], imagePair[52]);
                    detail::ewaldAccumulate(output[53], compensation[53], imagePair[53]);
                    detail::ewaldAccumulate(output[54], compensation[54], imagePair[54]);
                    detail::ewaldAccumulate(output[55], compensation[55], imagePair[55]);
                    detail::ewaldAccumulate(output[56], compensation[56], imagePair[56]);
                    detail::ewaldAccumulate(output[57], compensation[57], imagePair[57]);
                    detail::ewaldAccumulate(output[58], compensation[58], imagePair[58]);
                    detail::ewaldAccumulate(output[59], compensation[59], imagePair[59]);
                    detail::ewaldAccumulate(output[60], compensation[60], imagePair[60]);
                    detail::ewaldAccumulate(output[61], compensation[61], imagePair[61]);
                    detail::ewaldAccumulate(output[62], compensation[62], imagePair[62]);
                    detail::ewaldAccumulate(output[63], compensation[63], imagePair[63]);
                    detail::ewaldAccumulate(output[64], compensation[64], imagePair[64]);
                    detail::ewaldAccumulate(output[65], compensation[65], imagePair[65]);
                    detail::ewaldAccumulate(output[66], compensation[66], imagePair[66]);
                    detail::ewaldAccumulate(output[67], compensation[67], imagePair[67]);
                    detail::ewaldAccumulate(output[68], compensation[68], imagePair[68]);
                    detail::ewaldAccumulate(output[69], compensation[69], imagePair[69]);
                    detail::ewaldAccumulate(output[70], compensation[70], imagePair[70]);
                    detail::ewaldAccumulate(output[71], compensation[71], imagePair[71]);
                    detail::ewaldAccumulate(output[72], compensation[72], imagePair[72]);
                    detail::ewaldAccumulate(output[73], compensation[73], imagePair[73]);
                    detail::ewaldAccumulate(output[74], compensation[74], imagePair[74]);
                    detail::ewaldAccumulate(output[75], compensation[75], imagePair[75]);
                    detail::ewaldAccumulate(output[76], compensation[76], imagePair[76]);
                    detail::ewaldAccumulate(output[77], compensation[77], imagePair[77]);
                    detail::ewaldAccumulate(output[78], compensation[78], imagePair[78]);
                    detail::ewaldAccumulate(output[79], compensation[79], imagePair[79]);
                    detail::ewaldAccumulate(output[80], compensation[80], imagePair[80]);
                    Real const waveX = simdBroadcast<Real>(ix);
                    Real const waveY = simdBroadcast<Real>(iy);
                    Real const waveZ = simdBroadcast<Real>(iz);
                    Real const waveSquared = simdBroadcast<Real>(latticeSquared);
                    regular(harmonics, waveX, waveY, waveZ, waveSquared);
                    Real const phase = simdBroadcast<Real>(6.2831853071795862) *
                        simdFma(waveX, x, simdFma(waveY, y, waveZ * z));
                    Real const cosine = detail::ewaldCos(phase);
                    Real const sine = detail::ewaldSin(phase);
                    Real const wavePotential = simdBroadcast<Real>(-0.63661977236758138) *
                        detail::ewaldExp(simdBroadcast<Real>(-2.4674011002723395) * waveSquared) /
                        waveSquared;
                    Real const waveRadial0 = simdBroadcast<Real>(1) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                    Real const waveRadial1 =
                        simdBroadcast<Real>(6.2831853071795862) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[1], compensation[1],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[1]);
                    detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                    detail::ewaldAccumulate(output[3], compensation[3],
                        simdBroadcast<Real>(2) * waveRadial1 * harmonics[3]);
                    Real const waveRadial2 =
                        simdBroadcast<Real>(13.159472534785811) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[4], compensation[4],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[4]);
                    detail::ewaldAccumulate(output[5], compensation[5],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[5]);
                    detail::ewaldAccumulate(output[6], compensation[6],
                        simdBroadcast<Real>(4) * waveRadial2 * harmonics[6]);
                    detail::ewaldAccumulate(output[7], compensation[7],
                        simdBroadcast<Real>(6) * waveRadial2 * harmonics[7]);
                    detail::ewaldAccumulate(output[8], compensation[8],
                        simdBroadcast<Real>(24) * waveRadial2 * harmonics[8]);
                    Real const waveRadial3 =
                        simdBroadcast<Real>(16.536680896159901) * wavePotential * sine;
                    detail::ewaldAccumulate(output[9], compensation[9],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[9]);
                    detail::ewaldAccumulate(output[10], compensation[10],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[10]);
                    detail::ewaldAccumulate(output[11], compensation[11],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[11]);
                    detail::ewaldAccumulate(output[12], compensation[12],
                        simdBroadcast<Real>(36) * waveRadial3 * harmonics[12]);
                    detail::ewaldAccumulate(output[13], compensation[13],
                        simdBroadcast<Real>(48) * waveRadial3 * harmonics[13]);
                    detail::ewaldAccumulate(output[14], compensation[14],
                        simdBroadcast<Real>(120) * waveRadial3 * harmonics[14]);
                    detail::ewaldAccumulate(output[15], compensation[15],
                        simdBroadcast<Real>(720) * waveRadial3 * harmonics[15]);
                    Real const waveRadial4 =
                        simdBroadcast<Real>(14.843290062324177) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[16], compensation[16],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[16]);
                    detail::ewaldAccumulate(output[17], compensation[17],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[17]);
                    detail::ewaldAccumulate(output[18], compensation[18],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[18]);
                    detail::ewaldAccumulate(output[19], compensation[19],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[19]);
                    detail::ewaldAccumulate(output[20], compensation[20],
                        simdBroadcast<Real>(576) * waveRadial4 * harmonics[20]);
                    detail::ewaldAccumulate(output[21], compensation[21],
                        simdBroadcast<Real>(720) * waveRadial4 * harmonics[21]);
                    detail::ewaldAccumulate(output[22], compensation[22],
                        simdBroadcast<Real>(1440) * waveRadial4 * harmonics[22]);
                    detail::ewaldAccumulate(output[23], compensation[23],
                        simdBroadcast<Real>(5040) * waveRadial4 * harmonics[23]);
                    detail::ewaldAccumulate(output[24], compensation[24],
                        simdBroadcast<Real>(40320) * waveRadial4 * harmonics[24]);
                    Real const waveRadial5 =
                        simdBroadcast<Real>(10.36257133664445) * wavePotential * -sine;
                    detail::ewaldAccumulate(output[25], compensation[25],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[25]);
                    detail::ewaldAccumulate(output[26], compensation[26],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[26]);
                    detail::ewaldAccumulate(output[27], compensation[27],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[27]);
                    detail::ewaldAccumulate(output[28], compensation[28],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[28]);
                    detail::ewaldAccumulate(output[29], compensation[29],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[29]);
                    detail::ewaldAccumulate(output[30], compensation[30],
                        simdBroadcast<Real>(14400) * waveRadial5 * harmonics[30]);
                    detail::ewaldAccumulate(output[31], compensation[31],
                        simdBroadcast<Real>(17280) * waveRadial5 * harmonics[31]);
                    detail::ewaldAccumulate(output[32], compensation[32],
                        simdBroadcast<Real>(30240) * waveRadial5 * harmonics[32]);
                    detail::ewaldAccumulate(output[33], compensation[33],
                        simdBroadcast<Real>(80640) * waveRadial5 * harmonics[33]);
                    detail::ewaldAccumulate(output[34], compensation[34],
                        simdBroadcast<Real>(362880) * waveRadial5 * harmonics[34]);
                    detail::ewaldAccumulate(output[35], compensation[35],
                        simdBroadcast<Real>(3628800) * waveRadial5 * harmonics[35]);
                    Real const waveRadial6 =
                        simdBroadcast<Real>(5.9190869060913398) * wavePotential * -cosine;
                    detail::ewaldAccumulate(output[36], compensation[36],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[36]);
                    detail::ewaldAccumulate(output[37], compensation[37],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[37]);
                    detail::ewaldAccumulate(output[38], compensation[38],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[38]);
                    detail::ewaldAccumulate(output[39], compensation[39],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[39]);
                    detail::ewaldAccumulate(output[40], compensation[40],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[40]);
                    detail::ewaldAccumulate(output[41], compensation[41],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[41]);
                    detail::ewaldAccumulate(output[42], compensation[42],
                        simdBroadcast<Real>(518400) * waveRadial6 * harmonics[42]);
                    detail::ewaldAccumulate(output[43], compensation[43],
                        simdBroadcast<Real>(604800) * waveRadial6 * harmonics[43]);
                    detail::ewaldAccumulate(output[44], compensation[44],
                        simdBroadcast<Real>(967680) * waveRadial6 * harmonics[44]);
                    detail::ewaldAccumulate(output[45], compensation[45],
                        simdBroadcast<Real>(2177280) * waveRadial6 * harmonics[45]);
                    detail::ewaldAccumulate(output[46], compensation[46],
                        simdBroadcast<Real>(7257600) * waveRadial6 * harmonics[46]);
                    detail::ewaldAccumulate(output[47], compensation[47],
                        simdBroadcast<Real>(39916800) * waveRadial6 * harmonics[47]);
                    detail::ewaldAccumulate(output[48], compensation[48],
                        simdBroadcast<Real>(479001600) * waveRadial6 * harmonics[48]);
                    Real const waveRadial7 =
                        simdBroadcast<Real>(2.8608246061747828) * wavePotential * sine;
                    detail::ewaldAccumulate(output[49], compensation[49],
                        simdBroadcast<Real>(87178291200) * waveRadial7 * harmonics[49]);
                    detail::ewaldAccumulate(output[50], compensation[50],
                        simdBroadcast<Real>(6227020800) * waveRadial7 * harmonics[50]);
                    detail::ewaldAccumulate(output[51], compensation[51],
                        simdBroadcast<Real>(958003200) * waveRadial7 * harmonics[51]);
                    detail::ewaldAccumulate(output[52], compensation[52],
                        simdBroadcast<Real>(239500800) * waveRadial7 * harmonics[52]);
                    detail::ewaldAccumulate(output[53], compensation[53],
                        simdBroadcast<Real>(87091200) * waveRadial7 * harmonics[53]);
                    detail::ewaldAccumulate(output[54], compensation[54],
                        simdBroadcast<Real>(43545600) * waveRadial7 * harmonics[54]);
                    detail::ewaldAccumulate(output[55], compensation[55],
                        simdBroadcast<Real>(29030400) * waveRadial7 * harmonics[55]);
                    detail::ewaldAccumulate(output[56], compensation[56],
                        simdBroadcast<Real>(25401600) * waveRadial7 * harmonics[56]);
                    detail::ewaldAccumulate(output[57], compensation[57],
                        simdBroadcast<Real>(29030400) * waveRadial7 * harmonics[57]);
                    detail::ewaldAccumulate(output[58], compensation[58],
                        simdBroadcast<Real>(43545600) * waveRadial7 * harmonics[58]);
                    detail::ewaldAccumulate(output[59], compensation[59],
                        simdBroadcast<Real>(87091200) * waveRadial7 * harmonics[59]);
                    detail::ewaldAccumulate(output[60], compensation[60],
                        simdBroadcast<Real>(239500800) * waveRadial7 * harmonics[60]);
                    detail::ewaldAccumulate(output[61], compensation[61],
                        simdBroadcast<Real>(958003200) * waveRadial7 * harmonics[61]);
                    detail::ewaldAccumulate(output[62], compensation[62],
                        simdBroadcast<Real>(6227020800) * waveRadial7 * harmonics[62]);
                    detail::ewaldAccumulate(output[63], compensation[63],
                        simdBroadcast<Real>(87178291200) * waveRadial7 * harmonics[63]);
                    Real const waveRadial8 =
                        simdBroadcast<Real>(1.1983394087956813) * wavePotential * cosine;
                    detail::ewaldAccumulate(output[64], compensation[64],
                        simdBroadcast<Real>(20922789888000) * waveRadial8 * harmonics[64]);
                    detail::ewaldAccumulate(output[65], compensation[65],
                        simdBroadcast<Real>(1307674368000) * waveRadial8 * harmonics[65]);
                    detail::ewaldAccumulate(output[66], compensation[66],
                        simdBroadcast<Real>(174356582400) * waveRadial8 * harmonics[66]);
                    detail::ewaldAccumulate(output[67], compensation[67],
                        simdBroadcast<Real>(37362124800) * waveRadial8 * harmonics[67]);
                    detail::ewaldAccumulate(output[68], compensation[68],
                        simdBroadcast<Real>(11496038400) * waveRadial8 * harmonics[68]);
                    detail::ewaldAccumulate(output[69], compensation[69],
                        simdBroadcast<Real>(4790016000) * waveRadial8 * harmonics[69]);
                    detail::ewaldAccumulate(output[70], compensation[70],
                        simdBroadcast<Real>(2612736000) * waveRadial8 * harmonics[70]);
                    detail::ewaldAccumulate(output[71], compensation[71],
                        simdBroadcast<Real>(1828915200) * waveRadial8 * harmonics[71]);
                    detail::ewaldAccumulate(output[72], compensation[72],
                        simdBroadcast<Real>(1625702400) * waveRadial8 * harmonics[72]);
                    detail::ewaldAccumulate(output[73], compensation[73],
                        simdBroadcast<Real>(1828915200) * waveRadial8 * harmonics[73]);
                    detail::ewaldAccumulate(output[74], compensation[74],
                        simdBroadcast<Real>(2612736000) * waveRadial8 * harmonics[74]);
                    detail::ewaldAccumulate(output[75], compensation[75],
                        simdBroadcast<Real>(4790016000) * waveRadial8 * harmonics[75]);
                    detail::ewaldAccumulate(output[76], compensation[76],
                        simdBroadcast<Real>(11496038400) * waveRadial8 * harmonics[76]);
                    detail::ewaldAccumulate(output[77], compensation[77],
                        simdBroadcast<Real>(37362124800) * waveRadial8 * harmonics[77]);
                    detail::ewaldAccumulate(output[78], compensation[78],
                        simdBroadcast<Real>(174356582400) * waveRadial8 * harmonics[78]);
                    detail::ewaldAccumulate(output[79], compensation[79],
                        simdBroadcast<Real>(1307674368000) * waveRadial8 * harmonics[79]);
                    detail::ewaldAccumulate(output[80], compensation[80],
                        simdBroadcast<Real>(20922789888000) * waveRadial8 * harmonics[80]);
                }
            }
        }
    }
    template <typename Real, typename Output>
    CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {
        using Working = typename detail::EwaldWorkingType<Real>::Type;
        Working temporary[81];
        Working trace;
        evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x),
            detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
        for (int i = 0; i < 81; ++i) {
            output[i] = detail::ewaldConvert<Real>(temporary[i]);
        }
        greenTrace = detail::ewaldConvert<Real>(trace);
    }
};
template <>
class EwaldM2L<8> {
public:
    static constexpr int kFlops_ = 2216;
    static constexpr int kConventionalFlops_ = 4245;
    template <typename Real, typename Local, typename Moments, typename Greens>
    CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments,
        Real sourceTrace, Greens const& greens, Real greenTrace) {
        detail::M2LDerivatives<8>::template add<Real>(local, moments, greens);
        local[0] = simdFma(-simdBroadcast<Real>(0.5) * greenTrace, sourceTrace, local[0]);
        local[3] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[3], local[3]);
        local[1] = simdFma(-simdBroadcast<Real>(2) * greenTrace, moments[1], local[1]);
        local[2] = simdFma(-greenTrace, moments[2], local[2]);
        localTrace = simdFma(greenTrace, moments[0], localTrace);
    }
};
#endif
template <int degree>
class EwaldL2L {
public:
    static constexpr int kFlops_ = detail::L2L<degree>::kFlops_ + 9;
    static constexpr int kConventionalFlops_ = detail::L2L<degree>::kConventionalFlops_ + 15;
    template <typename Real, typename Local, typename Source>
    CUDA_EXPORT static void add(Local& output, Real& outputTrace, Source const& source,
        Real sourceTrace, Real x, Real y, Real z) {
        detail::L2L<degree>::template add<Real>(output, source, x, y, z);
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        output[0] = simdFma(-simdBroadcast<Real>(0.5) * sourceTrace, radiusSquared, output[0]);
        output[3] = simdFma(-sourceTrace, x, output[3]);
        output[1] = simdFma(-sourceTrace, y, output[1]);
        output[2] = simdFma(-sourceTrace, z, output[2]);
        outputTrace += sourceTrace;
    }
};
template <int degree>
class EwaldL2P {
public:
    static constexpr int kFlops_ = detail::L2P<degree>::kFlops_ + 8;
    static constexpr int kConventionalFlops_ = detail::L2P<degree>::kConventionalFlops_ + 14;
    template <typename Real, typename Field, typename Source>
    CUDA_EXPORT static void add(
        Field& field, Source const& source, Real sourceTrace, Real x, Real y, Real z) {
        detail::L2P<degree>::template add<Real>(field, source, x, y, z);
        Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));
        field[0] = simdFma(-simdBroadcast<Real>(0.5) * sourceTrace, radiusSquared, field[0]);
        field[1] = simdFma(sourceTrace, x, field[1]);
        field[2] = simdFma(sourceTrace, y, field[2]);
        field[3] = simdFma(sourceTrace, z, field[3]);
    }
};
}    // namespace cosmictiger::gravity::spherical
