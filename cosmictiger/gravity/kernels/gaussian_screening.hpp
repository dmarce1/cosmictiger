/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cmath>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/special_error.hpp>

namespace cosmictiger::gravity {

enum class GaussianComponent { Short, Long };

// alpha is an inverse length in the same coordinates as the displacement.
// Its precondition is finite alpha > 0; the split radius is rs = 1/(2 alpha).
template <class Scalar>
class GaussianScreening {
    Scalar alpha_;
    Scalar alphaSquared_;

public:
    CUDA_EXPORT explicit constexpr GaussianScreening(Scalar alpha)
      : alpha_(alpha)
      , alphaSquared_(alpha * alpha) {}

    CUDA_EXPORT static constexpr GaussianScreening fromSplitRadius(Scalar radius) {
        return GaussianScreening(Scalar(0.5) / radius);
    }
    CUDA_EXPORT constexpr Scalar alpha() const {
        return alpha_;
    }
    CUDA_EXPORT constexpr Scalar alphaSquared() const {
        return alphaSquared_;
    }
};

namespace detail {

    template <int exponent, class Real>
    CUDA_EXPORT inline Real gaussianIntegerPower(Real value) {
        static_assert(exponent >= 0);
        if constexpr (exponent == 0) {
            return simdBroadcast<Real>(1);
        } else if constexpr (exponent == 1) {
            return value;
        } else {
            Real const half = gaussianIntegerPower<exponent / 2>(value);
            if constexpr (exponent % 2 == 0) {
                return half * half;
            } else {
                return half * half * value;
            }
        }
    }

    CUDA_EXPORT constexpr int gaussianPowerFlops(int exponent) {
        return exponent <= 1 ? 0 : gaussianPowerFlops(exponent / 2) + 1 + exponent % 2;
    }

    // Keep the small screened factor in every partial product. Forming the
    // inverse-length power alone can overflow while the final derivative fits.
    template <int exponent, class Real>
    CUDA_EXPORT inline Real gaussianScaleProduct(Real value, Real scale) {
        static_assert(exponent >= 0);
        if constexpr (exponent == 0) {
            return value;
        } else {
            return gaussianScaleProduct<exponent - 1>(value * scale, scale);
        }
    }

    // ADL retains the independent counting-scalar hooks used by Ewald tests.
    // The existing accurate float implementation already handles SIMD floats;
    // double SIMD currently uses scalar library evaluations lane by lane.
    template <class Real>
    CUDA_EXPORT inline void gaussianErfcexp(Real argument, Real* complement, Real* exponential) {
        if constexpr (!std::is_arithmetic_v<Real> && std::is_same_v<SimdScalar<Real>, double> &&
            requires(Real value) { value[0]; }) {
#ifndef __CUDACC__
            for (size_t lane = 0; lane < Real::size(); ++lane) {
                double erfcValue;
                double expValue;
                special_functions::erfcexp(double(argument[lane]), &erfcValue, &expValue);
                simdSetLane(*complement, lane, erfcValue);
                simdSetLane(*exponential, lane, expValue);
            }
#endif
        } else {
            using special_functions::erfcexp;
            erfcexp(argument, complement, exponential);
        }
    }

    template <class Real>
    CUDA_EXPORT inline Real gaussianExp(Real argument) {
        if constexpr (!std::is_arithmetic_v<Real> && std::is_same_v<SimdScalar<Real>, double> &&
            requires(Real value) { value[0]; }) {
#ifndef __CUDACC__
            return Real([&](auto lane) { return std::exp(double(argument[lane])); });
#endif
        } else {
            using special_functions::exp;
            return exp(argument);
        }
    }

    CUDA_EXPORT constexpr double gaussianSeriesCoefficient(int degree, int term) {
        double factorial = 1;
        for (int i = 2; i <= term; ++i) {
            factorial *= i;
        }
        return (term % 2 ? -1.0 : 1.0) / (factorial * (2 * degree + 2 * term + 1));
    }

    template <int degree, int terms, int term = 0, class Real>
    CUDA_EXPORT inline Real gaussianSmallSeries(Real argument) {
        constexpr double kCoefficient = gaussianSeriesCoefficient(degree, term);
        if constexpr (term + 1 == terms) {
            return simdBroadcast<Real>(kCoefficient);
        } else {
            return simdFma(gaussianSmallSeries<degree, terms, term + 1>(argument), argument,
                simdBroadcast<Real>(kCoefficient));
        }
    }

}    // namespace detail

// Store scaled radial factors, never the potentially underflowing bare A_j.
// coefficient<j,t>() returns r^(j-t) A_j, where A_j=(1/r d/dr)^j g and
// g is -erfc(alpha*r)/r (Short) or -erf(alpha*r)/r (Long). Its Cartesian
// derivative rank is j+t. Multiplication by unit-direction monomials and
// symmetric Kronecker deltas reconstructs the full derivative tensor.
// Relative accuracy is limited once the dimensionless Gaussian tail becomes
// subnormal. Subsequent dimensional rescaling cannot recover a tail that has
// already rounded to zero; this is not an arbitrary-dynamic-range evaluator.
template <GaussianComponent component, class Real, int degree>
class GaussianRadial {
    static_assert(degree >= 0 && degree <= 8);
    Real factors_[degree + 1]{};
    Real inverseRadius_;
    Real inverseScale_;
    Real scaledRadius_;
    int flops_ = 0;

public:
    CUDA_EXPORT Real inverseRadius() const {
        return inverseRadius_;
    }
    CUDA_EXPORT int flops() const {
        return flops_;
    }
    template <int radialOrder, int traces>
    CUDA_EXPORT static constexpr int coefficientFlops() {
        static_assert(traces >= 0 && radialOrder >= traces && radialOrder + traces <= degree);
        if constexpr (component == GaussianComponent::Short) {
            return traces;
        } else {
            return radialOrder + traces + 1 +
                (radialOrder != traces ? detail::gaussianPowerFlops(radialOrder - traces) + 1 : 0);
        }
    }
    template <int radialOrder, int traces>
    CUDA_EXPORT Real coefficient() const {
        static_assert(traces >= 0 && radialOrder >= traces && radialOrder + traces <= degree);
        Real value = factors_[radialOrder];
        if constexpr (component == GaussianComponent::Short) {
            return detail::gaussianScaleProduct<traces>(value, inverseRadius_);
        } else {
            if constexpr (radialOrder != traces) {
                value *= detail::gaussianIntegerPower<radialOrder - traces>(scaledRadius_);
            }
            return detail::gaussianScaleProduct<radialOrder + traces + 1>(value, inverseScale_);
        }
    }

    // knownSmallDomain is an explicit contract alpha^2*r^2 <= 3. It removes
    // general-range dispatch for the existing Ewald primary-image correction.
    template <bool knownSmallDomain = false>
    CUDA_EXPORT static GaussianRadial evaluate(
        Real radiusSquared, GaussianScreening<SimdScalar<Real>> const& screening) {
        static_assert(!knownSmallDomain || component == GaussianComponent::Long);
        GaussianRadial result;
        Real const zero = simdBroadcast<Real>(0);
        Real const one = simdBroadcast<Real>(1);
        Real const alpha = simdBroadcast<Real>(screening.alpha());
        Real const radius = simdSqrt(radiusSquared);
        auto const nonzero = radius > zero;
        result.inverseRadius_ = simdSelect(nonzero, one, zero) / simdSelect(nonzero, radius, one);
        // Beyond z=32, degree-eight dimensionless Gaussian tails underflow in double.
        // Clamping also prevents 0*infinity in the tail recurrence.
        Real const z = simdMin(alpha * radius, simdBroadcast<Real>(32));
        Real const y = z * z;
        result.flops_ = 4;    // sqrt, reciprocal, alpha*r, z*z
        Real complement = zero;
        Real exponential = zero;
        if constexpr (knownSmallDomain) {
            exponential = detail::gaussianExp(-y);
            result.flops_ += 1;
        } else {
            detail::gaussianErfcexp(z, &complement, &exponential);
            result.flops_ += 2;
        }

        if constexpr (component == GaussianComponent::Short) {
            // Short has a genuine point singularity: radiusSquared > 0 is a
            // precondition. Store d_n = r^n A_n; the same-sign recurrence
            // shares the dimensional scaling between all tensor components.
            result.inverseScale_ = result.inverseRadius_;
            result.scaledRadius_ = one;
            result.factors_[0] = -complement * result.inverseRadius_;
            result.flops_ += 1;
            if constexpr (degree > 0) {
                Real term = simdBroadcast<Real>(1.1283791670955125739) * z * exponential;
                term = detail::gaussianScaleProduct<2>(term, result.inverseRadius_);
                Real const negativeTwoY = simdBroadcast<Real>(-2) * y;
                result.flops_ += 5;
                for (int n = 0; n < degree; ++n) {
                    result.factors_[n + 1] =
                        simdFma(simdBroadcast<Real>(-2 * n - 1) * result.factors_[n],
                            result.inverseRadius_, term);
                    result.flops_ += 3;
                    if (n + 1 != degree) {
                        term = (term * negativeTwoY) * result.inverseRadius_;
                        result.flops_ += 2;
                    }
                }
            }
        } else {
            auto const small = knownSmallDomain ? y >= zero : y < simdBroadcast<Real>(degree + 1);
            result.inverseScale_ = simdSelect(small, alpha, result.inverseRadius_);
            result.scaledRadius_ = simdSelect(small, z, one);
            if (knownSmallDomain || simdAnyOf(small)) {
                Real const smallY = simdSelect(small, y, zero);
                Real const smallExp = simdSelect(small, exponential, one);
                Real integral;
                if constexpr (knownSmallDomain) {
                    constexpr int kTerms = std::is_same_v<SimdScalar<Real>, float> ? 20 : 32;
                    integral = detail::gaussianSmallSeries<degree, kTerms>(smallY);
                    result.flops_ += 2 * (kTerms - 1);
                } else {
                    // S_n(y)=exp(-y)/(2n+1) times a positive series. Starting
                    // at n=degree with y<degree+1 makes its terms decrease.
                    constexpr int kTerms = std::is_same_v<SimdScalar<Real>, float> ? 32 : 64;
                    Real term = one;
                    Real sum = one;
                    for (int k = 1; k < kTerms; ++k) {
                        term *= smallY * simdBroadcast<Real>(2.0 / (2 * degree + 2 * k + 1));
                        sum += term;
                    }
                    integral = (sum * smallExp) * simdBroadcast<Real>(1.0 / (2 * degree + 1));
                    result.flops_ += 3 * (kTerms - 1) + 2;
                }
                Real integrals[degree + 1];
                integrals[degree] = integral;
                Real const twoY = smallY + smallY;
                result.flops_ += 1;
                for (int n = degree; n > 0; --n) {
                    integrals[n - 1] = simdFma(twoY, integrals[n], smallExp) *
                        simdBroadcast<Real>(1.0 / (2 * n - 1));
                    result.flops_ += 3;
                }
                double factor = -1.1283791670955125739;
                for (int n = 0; n <= degree; ++n) {
                    result.factors_[n] = simdBroadcast<Real>(factor) * integrals[n];
                    factor *= -2;
                }
                result.flops_ += degree + 1;
            }
            if constexpr (!knownSmallDomain) {
                if (!simdAllOf(small)) {
                    Real const largeZ = simdSelect(small, one, z);
                    Real const largeY = simdSelect(small, one, y);
                    Real const largeExp = simdSelect(small, zero, exponential);
                    Real radial = complement - one;
                    result.factors_[0] = simdSelect(small, result.factors_[0], radial);
                    result.flops_ += 1;
                    if constexpr (degree > 0) {
                        Real term = simdBroadcast<Real>(1.1283791670955125739) * largeZ * largeExp;
                        Real const negativeTwoY = simdBroadcast<Real>(-2) * largeY;
                        result.flops_ += 3;
                        for (int n = 0; n < degree; ++n) {
                            radial = simdFma(simdBroadcast<Real>(-2 * n - 1), radial, -term);
                            result.factors_[n + 1] =
                                simdSelect(small, result.factors_[n + 1], radial);
                            result.flops_ += 2;
                            if (n + 1 != degree) {
                                term *= negativeTwoY;
                                result.flops_ += 1;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};

template <GaussianComponent component, int degree, bool knownSmallDomain = false, class Real>
CUDA_EXPORT inline GaussianRadial<component, Real, degree> evaluateGaussianRadial(
    Real radiusSquared, GaussianScreening<SimdScalar<Real>> const& screening) {
    return GaussianRadial<component, Real, degree>::template evaluate<knownSmallDomain>(
        radiusSquared, screening);
}

}    // namespace cosmictiger::gravity
