/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace hopkins {

template <class Real>
using ArithmeticVector = std::array<Real, 3>;

template <class Real>
using ArithmeticMask = decltype(Real{} > Real{});

template <class Real>
struct PrimitiveValues {
    Real density = Real(0.0);
    ArithmeticVector<Real> velocity{};
    Real pressure = Real(0.0);
};

template <class Real>
struct GradientValues {
    ArithmeticVector<Real> density{};
    std::array<ArithmeticVector<Real>, 3> velocity{};
    ArithmeticVector<Real> pressure{};
};

template <class Real>
struct FluxValues {
    Real mass = Real(0.0);
    ArithmeticVector<Real> momentum{};
    Real energy = Real(0.0);
    Real signalSpeed = Real(0.0);
    Real contactSpeed = Real(0.0);
};

#ifndef __CUDACC__
using SimdReal = NativeSimdVector<double>;
using SimdVector3 = ArithmeticVector<SimdReal>;
using SimdPrimitive = PrimitiveValues<SimdReal>;
using SimdGradient = GradientValues<SimdReal>;
#endif

namespace detail {

    // Count executed lanes, including padding. Mask operations, signs, comparisons,
    // and arithmetic used only for validation are excluded, as in hopkins.hpp.
    template <class Real>
    inline void countArithmeticFlops(FlopCount* operations, FlopCount count) {
        if (operations != nullptr) {
            if constexpr (std::is_arithmetic_v<Real>) {
                *operations += count;
            } else {
                *operations += count * Real::size();
            }
        }
    }

    template <class Real>
    inline ArithmeticMask<Real> arithmeticFinite(Real const& value) {
        Real const limit(std::numeric_limits<SimdScalar<Real>>::max());
        return (value <= limit) && (value >= -limit);
    }

    template <class Exception, class Mask>
    inline void requireArithmetic(Mask const& failure, char const* message) {
        if (simdAnyOf(failure)) {
            if constexpr (std::is_same_v<Mask, bool>) {
                throw Exception(message);
            } else {
                // Scalar indexing is restricted to the exceptional diagnostic path.
                for (size_t lane = 0; lane < Mask::size(); lane++) {
                    if (failure[lane]) {
                        throw Exception(std::string(message) + " in lane " + std::to_string(lane));
                    }
                }
            }
        }
    }

    template <class Real>
    inline Real arithmeticDot(ArithmeticVector<Real> const& left,
        ArithmeticVector<Real> const& right, FlopCount* operations) {
        countArithmeticFlops<Real>(operations, 5);
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    template <class Real>
    inline ArithmeticMask<Real> arithmeticPhysical(PrimitiveValues<Real> const& state) {
        return (state.density > Real(0.0)) && (state.pressure > Real(0.0)) &&
            arithmeticFinite(state.density) && arithmeticFinite(state.pressure) &&
            arithmeticFinite(state.velocity[0]) && arithmeticFinite(state.velocity[1]) &&
            arithmeticFinite(state.velocity[2]);
    }

    template <class Real>
    inline PrimitiveValues<Real> selectPrimitive(ArithmeticMask<Real> const& mask,
        PrimitiveValues<Real> const& yes, PrimitiveValues<Real> const& no) {
        PrimitiveValues<Real> result;
        result.density = simdSelect(mask, yes.density, no.density);
        result.pressure = simdSelect(mask, yes.pressure, no.pressure);
        for (int dim = 0; dim < 3; dim++) {
            result.velocity[dim] = simdSelect(mask, yes.velocity[dim], no.velocity[dim]);
        }
        return result;
    }

    template <class Real>
    struct ConservedValues {
        Real mass = Real(0.0);
        ArithmeticVector<Real> momentum{};
        Real energy = Real(0.0);
    };

    template <class Real>
    inline ConservedValues<Real> arithmeticConserved(
        PrimitiveValues<Real> const& state, Real const& gammaMinusOne, FlopCount* operations) {
        ConservedValues<Real> result;
        result.mass = state.density;
        countArithmeticFlops<Real>(operations, 7);
        for (int dim = 0; dim < 3; dim++) {
            result.momentum[dim] = state.density * state.velocity[dim];
        }
        result.energy = state.pressure / gammaMinusOne +
            Real(0.5) * state.density * arithmeticDot(state.velocity, state.velocity, operations);
        return result;
    }

    template <class Real>
    inline ConservedValues<Real> arithmeticStar(PrimitiveValues<Real> const& state,
        ConservedValues<Real> const& conservedState, ArithmeticVector<Real> const& normal,
        Real const& normalVelocity, Real const& waveRelativeVelocity, Real const& waveSpeed,
        Real const& contact, Real const& starPressure, FlopCount* operations) {
        countArithmeticFlops<Real>(operations, 19);
        Real const width = waveSpeed - contact;
        ConservedValues<Real> result;
        result.mass = state.density * waveRelativeVelocity / width;
        Real const velocityChange = contact - normalVelocity;
        for (int dim = 0; dim < 3; dim++) {
            Real const velocity = state.velocity[dim] + velocityChange * normal[dim];
            result.momentum[dim] = result.mass * velocity;
        }
        result.energy = (waveRelativeVelocity * conservedState.energy -
                            state.pressure * normalVelocity + starPressure * contact) /
            width;
        return result;
    }

    template <class Real>
    inline ArithmeticMask<Real> arithmeticPhysical(ConservedValues<Real> const& state) {
        // This energy check is validation-only; it does not contribute to FLOPs.
        Real const internalEnergy = state.energy -
            Real(0.5) * arithmeticDot(state.momentum, state.momentum, nullptr) / state.mass;
        return (state.mass > Real(0.0)) && arithmeticFinite(state.mass) &&
            arithmeticFinite(state.energy) && arithmeticFinite(state.momentum[0]) &&
            arithmeticFinite(state.momentum[1]) && arithmeticFinite(state.momentum[2]) &&
            (internalEnergy > Real(0.0)) && arithmeticFinite(internalEnergy);
    }

    template <class Real>
    inline ConservedValues<Real> selectConserved(ArithmeticMask<Real> const& mask,
        ConservedValues<Real> const& yes, ConservedValues<Real> const& no) {
        ConservedValues<Real> result;
        result.mass = simdSelect(mask, yes.mass, no.mass);
        result.energy = simdSelect(mask, yes.energy, no.energy);
        for (int dim = 0; dim < 3; dim++) {
            result.momentum[dim] = simdSelect(mask, yes.momentum[dim], no.momentum[dim]);
        }
        return result;
    }

}    // namespace detail

// Appendix B, equation (76), evaluated by masks. Safe divisors prevent an
// unselected sign-preserving correction from dividing by zero in another lane.
template <class Real>
inline Real pairLimiterValues(Real const& center, Real const& neighbor, Real const& reconstructed,
    Real const& faceFraction, FlopCount* operations = nullptr) {
    detail::countArithmeticFlops<Real>(operations, 15);
    Real const zero(0.0);
    Real const one(1.0);
    Real const change = neighbor - center;
    Real const difference = simdMax(change, -change);
    Real const delta1 = Real(0.5) * difference;
    Real const delta2 = Real(0.25) * difference;
    Real const minimum = simdMin(center, neighbor);
    Real const maximum = simdMax(center, neighbor);
    Real const lower = minimum - delta1;
    Real const upper = maximum + delta1;
    Real const safeMinimum = simdSelect(minimum > zero, minimum, one);
    Real const safeMaximum = simdSelect(maximum < zero, -maximum, one);
    Real const positiveLower = minimum / (one + delta1 / safeMinimum);
    Real const negativeUpper = maximum / (one + delta1 / safeMaximum);
    Real const limitedLower = simdSelect(minimum == zero, zero,
        simdSelect((minimum > zero) && (lower <= zero), positiveLower, lower));
    Real const limitedUpper = simdSelect(maximum == zero, zero,
        simdSelect((maximum < zero) && (upper >= zero), negativeUpper, upper));
    Real const mean = center + faceFraction * change;
    Real const result =
        simdSelect(center < neighbor, simdMax(limitedLower, simdMin(mean + delta2, reconstructed)),
            simdMin(limitedUpper, simdMax(mean - delta2, reconstructed)));
    return simdSelect(center == neighbor, center, result);
}

// Spatial reconstruction, pair limiter, and MUSCL-Hancock prediction operate on
// independent faces in each lane. The reconstruction's existing cell-value
// limiter is retained; the Riemann solver itself has no fallback.
template <class Real>
inline PrimitiveValues<Real> reconstructValues(PrimitiveValues<Real> const& center,
    GradientValues<Real> const& gradient, ArithmeticVector<Real> const& offset,
    ArithmeticVector<Real> const& frameVelocity, Real const& halfTimestep, double gamma,
    PrimitiveValues<Real> const& neighbor, Real const& faceFraction,
    FlopCount* operations = nullptr) {
    detail::countArithmeticFlops<Real>(operations, 32);
    PrimitiveValues<Real> result = center;
    result.density += detail::arithmeticDot(gradient.density, offset, operations);
    result.pressure += detail::arithmeticDot(gradient.pressure, offset, operations);
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] += detail::arithmeticDot(gradient.velocity[dim], offset, operations);
    }
    result.density = pairLimiterValues(
        center.density, neighbor.density, result.density, faceFraction, operations);
    result.pressure = pairLimiterValues(
        center.pressure, neighbor.pressure, result.pressure, faceFraction, operations);
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] = pairLimiterValues(center.velocity[dim], neighbor.velocity[dim],
            result.velocity[dim], faceFraction, operations);
    }
    ArithmeticVector<Real> relativeVelocity{};
    Real divergence(0.0);
    for (int dim = 0; dim < 3; dim++) {
        relativeVelocity[dim] = center.velocity[dim] - frameVelocity[dim];
        divergence += gradient.velocity[dim][dim];
    }
    result.density -= halfTimestep *
        (detail::arithmeticDot(relativeVelocity, gradient.density, operations) +
            center.density * divergence);
    result.pressure -= halfTimestep *
        (detail::arithmeticDot(relativeVelocity, gradient.pressure, operations) +
            Real(gamma) * center.pressure * divergence);
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] -= halfTimestep *
            (detail::arithmeticDot(relativeVelocity, gradient.velocity[dim], operations) +
                gradient.pressure[dim] / center.density);
    }
    return detail::selectPrimitive(detail::arithmeticPhysical(result), result, center);
}

// Single-pass HLLC, using the two-sided Davis wave estimates (Toro, HLLC
// lecture, equation 48): SL=min(uL-cL,uR-cR), SR=max(uL+cL,uR+cR).
// https://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf
// Unclipped waves construct the contact/star states. Clipping only the selected
// flux correction to min(0,SL) or max(0,SR) reduces four regions to two sides.
// Invalid active inputs or nonphysical HLLC states are fatal to the caller.
template <class Real>
inline FluxValues<Real> riemannFluxValues(PrimitiveValues<Real> left, PrimitiveValues<Real> right,
    ArithmeticVector<Real> const& inputArea, ArithmeticVector<Real> const& inputFrameVelocity,
    double gamma, bool finiteMass, ArithmeticMask<Real> const& valid,
    FlopCount* operations = nullptr) {
    if (!(gamma > 1.0) || !std::isfinite(gamma)) {
        throw std::invalid_argument("Hopkins HLLC requires finite gamma greater than one");
    }
    Real const zero(0.0);
    Real const one(1.0);
    PrimitiveValues<Real> benign;
    benign.density = one;
    benign.pressure = one;
    // Sanitize padding before even validation comparisons: an inactive NaN
    // must not raise FE_INVALID in an otherwise physical packed solve.
    left = detail::selectPrimitive(valid, left, benign);
    right = detail::selectPrimitive(valid, right, benign);
    ArithmeticVector<Real> area{};
    ArithmeticVector<Real> frameVelocity{};
    for (int dim = 0; dim < 3; dim++) {
        area[dim] = simdSelect(valid, inputArea[dim], zero);
        frameVelocity[dim] = simdSelect(valid, inputFrameVelocity[dim], zero);
    }
    auto inputPhysical = detail::arithmeticPhysical(left) && detail::arithmeticPhysical(right);
    for (int dim = 0; dim < 3; dim++) {
        inputPhysical = inputPhysical && detail::arithmeticFinite(area[dim]) &&
            detail::arithmeticFinite(frameVelocity[dim]);
    }
    detail::requireArithmetic<std::invalid_argument>(
        valid && !inputPhysical, "Hopkins HLLC requires physical finite inputs");
    detail::countArithmeticFlops<Real>(operations, 1);
    Real const areaMagnitude = simdSqrt(detail::arithmeticDot(area, area, operations));
    detail::requireArithmetic<std::runtime_error>(
        valid && !detail::arithmeticFinite(areaMagnitude), "Hopkins HLLC face area overflow");
    auto const solve = valid && (areaMagnitude > zero);
    FluxValues<Real> result;
    if (!simdAnyOf(solve)) {
        return result;
    }

    // A canonical positive state keeps inactive/zero-area lanes harmless in
    // packed divisions, roots, and star-state validation.
    left = detail::selectPrimitive(solve, left, benign);
    right = detail::selectPrimitive(solve, right, benign);
    ArithmeticVector<Real> normal{};
    Real const safeAreaMagnitude = simdSelect(solve, areaMagnitude, one);
    detail::countArithmeticFlops<Real>(operations, 9);
    for (int dim = 0; dim < 3; dim++) {
        frameVelocity[dim] = simdSelect(solve, frameVelocity[dim], zero);
        normal[dim] = simdSelect(solve, area[dim], Real(dim == 0 ? 1.0 : 0.0)) / safeAreaMagnitude;
        left.velocity[dim] -= frameVelocity[dim];
        right.velocity[dim] -= frameVelocity[dim];
    }
    Real const leftVelocity = detail::arithmeticDot(left.velocity, normal, operations);
    Real const rightVelocity = detail::arithmeticDot(right.velocity, normal, operations);
    detail::countArithmeticFlops<Real>(operations, 10);
    Real const leftSound = simdSqrt(Real(gamma) * left.pressure / left.density);
    Real const rightSound = simdSqrt(Real(gamma) * right.pressure / right.density);
    Real const leftWave = simdMin(leftVelocity - leftSound, rightVelocity - rightSound);
    Real const rightWave = simdMax(leftVelocity + leftSound, rightVelocity + rightSound);
    detail::requireArithmetic<std::runtime_error>(solve &&
            (!detail::arithmeticFinite(leftWave) || !detail::arithmeticFinite(rightWave) ||
                !(leftWave < rightWave)),
        "Hopkins HLLC has invalid signal speeds");
    detail::countArithmeticFlops<Real>(operations, 5);
    Real const leftRelativeWave = leftWave - leftVelocity;
    Real const rightRelativeWave = rightWave - rightVelocity;
    Real const leftImpedance = left.density * leftRelativeWave;
    Real const rightImpedance = right.density * rightRelativeWave;
    Real const denominator = leftImpedance - rightImpedance;
    detail::requireArithmetic<std::runtime_error>(
        solve && (!detail::arithmeticFinite(denominator) || !(denominator < zero)),
        "Hopkins HLLC has a degenerate contact denominator");
    detail::countArithmeticFlops<Real>(operations, 14);
    Real const contact = (right.pressure - left.pressure + leftImpedance * leftVelocity -
                             rightImpedance * rightVelocity) /
        denominator;
    Real const leftPressure = left.pressure + leftImpedance * (contact - leftVelocity);
    Real const rightPressure = right.pressure + rightImpedance * (contact - rightVelocity);
    Real const starPressure = Real(0.5) * (leftPressure + rightPressure);
    detail::requireArithmetic<std::runtime_error>(solve &&
            (!detail::arithmeticFinite(contact) || !(contact > leftWave) ||
                !(contact < rightWave) || !(starPressure > zero) ||
                !detail::arithmeticFinite(starPressure)),
        "Hopkins HLLC has a nonphysical contact state");
    detail::countArithmeticFlops<Real>(operations, 1);
    Real const gammaMinusOne = Real(gamma) - one;
    auto const leftState = detail::arithmeticConserved(left, gammaMinusOne, operations);
    auto const rightState = detail::arithmeticConserved(right, gammaMinusOne, operations);
    auto const leftStar = detail::arithmeticStar(left, leftState, normal, leftVelocity,
        leftRelativeWave, leftWave, contact, starPressure, operations);
    auto const rightStar = detail::arithmeticStar(right, rightState, normal, rightVelocity,
        rightRelativeWave, rightWave, contact, starPressure, operations);
    detail::requireArithmetic<std::runtime_error>(
        solve && (!detail::arithmeticPhysical(leftStar) || !detail::arithmeticPhysical(rightStar)),
        "Hopkins HLLC has a nonphysical star state");

    detail::ConservedValues<Real> flux;
    if (finiteMass) {
        detail::countArithmeticFlops<Real>(operations, 4);
        for (int dim = 0; dim < 3; dim++) {
            flux.momentum[dim] = starPressure * normal[dim];
        }
        flux.energy = starPressure * contact;
    } else {
        auto const fromLeft = contact >= zero;
        PrimitiveValues<Real> side;
        detail::ConservedValues<Real> state;
        detail::ConservedValues<Real> star;
        Real velocity;
        Real correctionWave;
        if (simdAllOf(!solve || fromLeft)) {
            side = left;
            state = leftState;
            star = leftStar;
            velocity = leftVelocity;
            correctionWave = simdMin(zero, leftWave);
        } else if (!simdAnyOf(solve && fromLeft)) {
            side = right;
            state = rightState;
            star = rightStar;
            velocity = rightVelocity;
            correctionWave = simdMax(zero, rightWave);
        } else {
            side = detail::selectPrimitive(fromLeft, left, right);
            state = detail::selectConserved(fromLeft, leftState, rightState);
            star = detail::selectConserved(fromLeft, leftStar, rightStar);
            velocity = simdSelect(fromLeft, leftVelocity, rightVelocity);
            correctionWave =
                simdSelect(fromLeft, simdMin(zero, leftWave), simdMax(zero, rightWave));
        }
        detail::countArithmeticFlops<Real>(operations, 27);
        flux.mass = side.density * velocity + correctionWave * (star.mass - state.mass);
        flux.energy = (state.energy + side.pressure) * velocity +
            correctionWave * (star.energy - state.energy);
        for (int dim = 0; dim < 3; dim++) {
            flux.momentum[dim] = state.momentum[dim] * velocity + side.pressure * normal[dim] +
                correctionWave * (star.momentum[dim] - state.momentum[dim]);
        }
    }
    detail::countArithmeticFlops<Real>(operations, 15);
    result.mass = flux.mass * areaMagnitude;
    result.energy =
        (flux.energy + detail::arithmeticDot(frameVelocity, flux.momentum, operations) +
            Real(0.5) * detail::arithmeticDot(frameVelocity, frameVelocity, operations) *
                flux.mass) *
        areaMagnitude;
    for (int dim = 0; dim < 3; dim++) {
        result.momentum[dim] =
            (flux.momentum[dim] + frameVelocity[dim] * flux.mass) * areaMagnitude;
    }
    result.signalSpeed = simdSelect(
        solve, simdMax(simdMax(leftWave, -leftWave), simdMax(rightWave, -rightWave)), zero);
    result.contactSpeed = simdSelect(solve, contact, zero);
    auto finite = detail::arithmeticFinite(result.mass) && detail::arithmeticFinite(result.energy);
    for (int dim = 0; dim < 3; dim++) {
        finite = finite && detail::arithmeticFinite(result.momentum[dim]);
    }
    detail::requireArithmetic<std::runtime_error>(
        solve && !finite, "Hopkins HLLC returned a non-finite flux");
    return result;
}

}    // namespace hopkins
