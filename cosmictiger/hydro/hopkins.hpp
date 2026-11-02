/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/flops.hpp>

// Hopkins (2015), MNRAS 450, 53, doi:10.1093/mnras/stv195.
// https://arxiv.org/html/1409.7395v2
// All primitives are in one consistent physical frame and system of units.
// The application bridge performs the comoving-variable transformations.
namespace hopkins {

namespace detail {

    // Optional local accounting; no shared counter is touched inside a math helper.
    // All counted expressions use double. Comparisons, absolute values, sign changes,
    // and validation-only arithmetic are excluded, as in the solver-wide convention.
    inline void countFlops(FlopCount* operations, FlopCount count) {
        if (operations != nullptr) {
            *operations += count;
        }
    }

}    // namespace detail

using Vector = std::array<double, 3>;
using Matrix = std::array<Vector, 3>;

struct Primitive {
    double density = 0.0;
    Vector velocity{};
    double pressure = 0.0;
};

struct Gradient {
    Vector density{};
    Matrix velocity{};    // [velocity component][spatial direction]
    Vector pressure{};
};

struct Geometry {
    double volume = 0.0;
    Matrix inverseMoment{};
    double conditionNumber = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct GeometryRate {
    double volume = 0.0;
    Matrix inverseMoment{};
};

struct Flux {
    double mass = 0.0;
    Vector momentum{};
    double energy = 0.0;
    double signalSpeed = 0.0;     // Maximum wave speed relative to the input frame.
    double contactSpeed = 0.0;    // Contact speed relative to the input frame.
    bool usedFallback = false;
};

struct Conserved {
    double mass = 0.0;
    Vector momentum{};
    double energy = 0.0;
};

// Only the low clock phase is stored with an inactive predictor. The scheduler
// bounds its actual age by one base interval (32768 finest ticks), so unsigned
// subtraction recovers that age uniquely across any number of clock wraps.
// Retain larger results for the caller's rung/age validation to reject.
inline uint16_t elapsedTicks(uint64_t tick, uint16_t lastKickTick) {
    return uint16_t(uint16_t(tick) - lastKickTick);
}

// Integrate a linear-in-time cached flux over only the elapsed part of its
// interval. The complete interval is the MUSCL-Hancock midpoint update; an
// interrupted interval remains time-centered without booking future exchanges.
// Callers apply this single increment with opposite signs at the two endpoints
// and supply fluxes and times in the same units (including cosmological factors).
inline Conserved integrateFlux(Conserved const& startFlux, Conserved const& midpointFlux,
    double interval, double elapsed, double duration, FlopCount* operations = nullptr) {
    if (!(interval > 0.0) || !std::isfinite(interval) || !(elapsed >= 0.0) ||
        !std::isfinite(elapsed) || !(duration >= 0.0) || !std::isfinite(duration)) {
        throw std::invalid_argument("Hopkins cached flux requires finite nonnegative times");
    }
    double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() * interval;
    if (elapsed > interval + tolerance || duration > interval - elapsed + tolerance) {
        throw std::invalid_argument("Hopkins cached flux integration exceeds its interval");
    }
    detail::countFlops(operations, 4);
    double const fraction = elapsed / interval + (elapsed + duration) / interval;
    auto integrate = [&](double start, double midpoint) {
        if (!std::isfinite(start) || !std::isfinite(midpoint)) {
            throw std::invalid_argument("Hopkins cached flux must be finite");
        }
        if (duration == 0.0) {
            return 0.0;
        }
        detail::countFlops(operations, 4);
        double const result = duration * std::fma(fraction, midpoint - start, start);
        if (!std::isfinite(result)) {
            throw std::overflow_error("Hopkins cached flux integral is not finite");
        }
        return result;
    };
    Conserved result;
    result.mass = integrate(startFlux.mass, midpointFlux.mass);
    result.energy = integrate(startFlux.energy, midpointFlux.energy);
    for (int dim = 0; dim < 3; dim++) {
        result.momentum[dim] = integrate(startFlux.momentum[dim], midpointFlux.momentum[dim]);
    }
    return result;
}

// A shared face is evaluated on the shorter endpoint timestep (Appendix G).
inline double pairTimestep(
    double baseTimestep, int leftRung, int rightRung, FlopCount* operations = nullptr) {
    if (!(baseTimestep > 0.0) || !std::isfinite(baseTimestep) || leftRung < 0 || rightRung < 0) {
        throw std::invalid_argument(
            "Hopkins pair timestep requires a positive base and valid rungs");
    }
    detail::countFlops(operations, 1);
    double const result = std::ldexp(baseTimestep, -std::max(leftRung, rightRung));
    if (!(result > 0.0) || !std::isfinite(result)) {
        throw std::invalid_argument("Hopkins pair timestep is not representable");
    }
    return result;
}

// Replace a constant old-acceleration predictor by a linear interpolation to
// the newly measured acceleration in dv/dt = g(t) - damping*v. Multiplying this
// weight by g_new-g_old gives the endpoint velocity correction. The uncorrected
// position already has O(duration^3) local error, as in velocity Verlet.
inline double sourceCorrectionWeight(
    double duration, double damping, FlopCount* operations = nullptr) {
    if (!(duration >= 0.0) || !std::isfinite(duration) || !(damping >= 0.0) ||
        !std::isfinite(damping)) {
        throw std::invalid_argument("Hopkins source correction requires finite nonnegative inputs");
    }
    detail::countFlops(operations, 1);
    double const argument = damping * duration;
    if (argument < 1.0e-3) {
        detail::countFlops(operations, 9);
        return duration *
            (0.5 +
                argument *
                    (-1.0 / 6.0 +
                        argument * (1.0 / 24.0 + argument * (-1.0 / 120.0 + argument / 720.0))));
    }
    // The division order stays finite even when damping*duration overflows.
    detail::countFlops(operations, 4);
    return (1.0 + std::expm1(-argument) / argument) / damping;
}

struct PairGravityWork {
    double leftEnergyRate = 0.0;
    double rightEnergyRate = 0.0;
};

inline double dot(Vector const& a, Vector const& b, FlopCount* operations = nullptr) {
    detail::countFlops(operations, 5);
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(Vector const& a, FlopCount* operations = nullptr) {
    detail::countFlops(operations, 1);
    return std::sqrt(dot(a, a, operations));
}

// Appendix H.1: mass crossing a finite-volume interface does
// gravitational work in addition to the usual particle-motion source kick.
// Here a symmetric midpoint spatial quadrature shares that work between the
// endpoints: their sum is massFlux * (a_i+a_j)/2 dot (x_j-x_i). It cancels the
// change of cell-centered potential energy exactly for a linear acceleration
// field. massFlux is positive from i to j and includes the effective face area;
// displacement is x_j-x_i. These are two SOURCE terms with the same sign, not
// an antisymmetric hydrodynamic energy flux. MFM vanishes identically.
//
// All arguments must use consistent coordinates/time/energy units. With the
// application's comoving displacement, dw/dtau acceleration, and w=a*v flux,
// multiply both returned rates by a before an update of duration dtau/a.
inline Vector pairGravityWorkVector(
    double massFlux, Vector const& displacement, FlopCount* operations = nullptr) {
    detail::countFlops(operations, 6);
    return {0.5 * massFlux * displacement[0], 0.5 * massFlux * displacement[1],
        0.5 * massFlux * displacement[2]};
}

inline PairGravityWork pairGravityWork(double massFlux, Vector const& displacement,
    Vector const& leftAcceleration, Vector const& rightAcceleration,
    FlopCount* operations = nullptr) {
    auto const work = pairGravityWorkVector(massFlux, displacement, operations);
    return {dot(leftAcceleration, work, operations), dot(rightAcceleration, work, operations)};
}

struct PairGravityWorkIntegral {
    Vector integral{};
    // First temporal moment measured from the start of the actual subinterval.
    Vector firstMoment{};
};

// Integrate q(t)=massFlux(t)*displacement(t)/2, represented by its beginning
// and midpoint values on a cached pair interval. A source anchor of age A at
// the subinterval start accumulates A*integral+firstMoment. At its next fresh
// force, multiplying that sum by (g_new-g_old)/sourceDuration supplies exactly
// the missing work for a linearly varying acceleration, including subdivisions.
inline PairGravityWorkIntegral integratePairGravityWork(Vector const& initial,
    Vector const& midpoint, double interval, double elapsed, double duration,
    FlopCount* operations = nullptr) {
    if (!(interval > 0.0) || !std::isfinite(interval) || !(elapsed >= 0.0) ||
        !std::isfinite(elapsed) || !(duration >= 0.0) || !std::isfinite(duration)) {
        throw std::invalid_argument("Hopkins pair gravity work requires finite nonnegative times");
    }
    double const tolerance = 32.0 * std::numeric_limits<double>::epsilon() * interval;
    if (elapsed > interval + tolerance || duration > interval - elapsed + tolerance) {
        throw std::invalid_argument("Hopkins pair gravity work integration exceeds its interval");
    }
    PairGravityWorkIntegral result;
    for (int dim = 0; dim < 3; dim++) {
        detail::countFlops(operations, 17);
        double const slope = 2.0 * (midpoint[dim] - initial[dim]) / interval;
        double const atStart = initial[dim] + slope * elapsed;
        result.integral[dim] = duration * (atStart + 0.5 * duration * slope);
        result.firstMoment[dim] =
            0.5 * duration * duration * atStart + duration * duration * duration * slope / 3.0;
    }
    return result;
}

inline Vector multiply(
    Matrix const& matrix, Vector const& vector, FlopCount* operations = nullptr) {
    return {dot(matrix[0], vector, operations), dot(matrix[1], vector, operations),
        dot(matrix[2], vector, operations)};
}

// Safeguarded Newton step for the monotonically increasing kernel-count sum.
// Keep the bracket across neighbor re-queries. Float storage cannot resolve a
// root strictly between adjacent floats; retain the current bracket endpoint
// once no further representable refinement is possible.
inline double supportRadiusStep(double radius, double residual, double derivative, double minimum,
    double maximum, double& lower, double& upper, bool floatStorage = false,
    FlopCount* operations = nullptr, FlopCount* singleOperations = nullptr) {
    if (residual == 0.0) {
        return radius;
    }
    if (residual > 0.0) {
        upper = std::min(upper, radius);
    } else {
        lower = std::max(lower, radius);
    }
    // The optional adjacency probe performs one float operation; its narrowing
    // input and promotion for the comparison both count at double precision.
    // The production driver uses double storage, so neither path executes there.
    if (floatStorage) {
        detail::countFlops(operations, 2);
        detail::countFlops(singleOperations, 1);
    }
    if (floatStorage &&
        std::nextafter(float(lower), std::numeric_limits<float>::infinity()) >= upper) {
        return radius;
    }
    detail::countFlops(operations, 4);
    double const change =
        derivative > 0.0 ? -residual / derivative : std::copysign(0.2 * radius, -residual);
    double next =
        std::clamp(radius + std::clamp(change, -0.2 * radius, 0.2 * radius), minimum, maximum);
    if (next < lower || next > upper || next == radius) {
        detail::countFlops(operations, 3);
        next = 0.5 * lower + 0.5 * upper;
    }
    if (floatStorage) {
        detail::countFlops(operations, 2);    // Narrow the stored radius, then promote its result.
    }
    return floatStorage ? double(float(next)) : next;
}

// Equations (24)-(26), (27), and (77). kernelSum includes the self contribution;
// kernelWeightedMoment is sum W(r,h) r tensor r, without particle-mass weighting.
// A rejected matrix must trigger a larger neighbor search (Appendix C), never a
// replacement with an SPH gradient. No diagonal regularization hides singularity.
inline Geometry makeGeometry(double kernelSum, Matrix const& kernelWeightedMoment,
    double maximumCondition = 100.0, int dimensions = 3, FlopCount* operations = nullptr) {
    Geometry result;
    if (!(kernelSum > 0.0) || !std::isfinite(kernelSum) || dimensions < 1 || dimensions > 3) {
        return result;
    }
    detail::countFlops(operations, 1);
    result.volume = 1.0 / kernelSum;
    Matrix moment{};
    double scale = 0.0;
    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j < dimensions; j++) {
            detail::countFlops(operations, 1);
            moment[i][j] = kernelWeightedMoment[i][j] / kernelSum;
            scale = std::max(scale, std::abs(moment[i][j]));
        }
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return result;
    }
    Matrix lower{};
    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j <= i; j++) {
            detail::countFlops(operations, 1);
            double value = moment[i][j] / scale;
            for (int k = 0; k < j; k++) {
                detail::countFlops(operations, 2);
                value -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!(value > 64.0 * std::numeric_limits<double>::epsilon())) {
                    return result;
                }
                detail::countFlops(operations, 1);
                lower[i][j] = std::sqrt(value);
            } else {
                detail::countFlops(operations, 1);
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    for (int column = 0; column < dimensions; column++) {
        Vector solution{};
        for (int i = 0; i < dimensions; i++) {
            double value = i == column ? 1.0 : 0.0;
            for (int j = 0; j < i; j++) {
                detail::countFlops(operations, 2);
                value -= lower[i][j] * solution[j];
            }
            detail::countFlops(operations, 1);
            solution[i] = value / lower[i][i];
        }
        for (int i = dimensions - 1; i >= 0; i--) {
            double value = solution[i];
            for (int j = i + 1; j < dimensions; j++) {
                detail::countFlops(operations, 2);
                value -= lower[j][i] * solution[j];
            }
            detail::countFlops(operations, 1);
            solution[i] = value / lower[i][i];
            detail::countFlops(operations, 1);
            result.inverseMoment[i][column] = solution[i] / scale;
        }
    }
    double momentNorm = 0.0;
    double inverseNorm = 0.0;
    for (int i = 0; i < dimensions; i++) {
        detail::countFlops(operations, 2);
        momentNorm += dot(moment[i], moment[i], operations);
        inverseNorm += dot(result.inverseMoment[i], result.inverseMoment[i], operations);
    }
    detail::countFlops(operations, 3);
    result.conditionNumber = std::sqrt(momentNorm * inverseNorm) / double(dimensions);
    result.valid =
        std::isfinite(result.conditionNumber) && result.conditionNumber <= maximumCondition;
    return result;
}

// Differentiate B = E^-1, where E is the normalized moment used by makeGeometry.
// momentRate is dE/dt, not the derivative of its unnormalized kernel sum.
inline GeometryRate geometryRate(Geometry const& geometry, double volumeRate,
    Matrix const& momentRate, FlopCount* operations = nullptr) {
    GeometryRate result;
    result.volume = volumeRate;
    Matrix intermediate{};
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            for (int k = 0; k < 3; k++) {
                detail::countFlops(operations, 2);
                intermediate[row][column] += geometry.inverseMoment[row][k] * momentRate[k][column];
            }
        }
    }
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            for (int k = 0; k < 3; k++) {
                detail::countFlops(operations, 2);
                result.inverseMoment[row][column] -=
                    intermediate[row][k] * geometry.inverseMoment[k][column];
            }
        }
    }
    return result;
}

inline Vector gradientWeight(Geometry const& geometry, Vector const& displacement,
    double kernelWeight, FlopCount* operations = nullptr) {
    Vector result = multiply(geometry.inverseMoment, displacement, operations);
    for (int dim = 0; dim < 3; dim++) {
        detail::countFlops(operations, 2);
        result[dim] *= kernelWeight * geometry.volume;
    }
    return result;
}

// Equation (31). displacement is x_j-x_i; A_ij points from i towards j.
// Pairwise antisymmetry holds for unequal volumes and unequal kernel radii.
inline Vector faceArea(Geometry const& left, Geometry const& right, Vector const& displacement,
    double kernelLeft, double kernelRight, FlopCount* operations = nullptr) {
    Vector const leftWeight = gradientWeight(left, displacement, kernelLeft, operations);
    Vector const rightWeight = gradientWeight(right, displacement, kernelRight, operations);
    Vector result{};
    for (int dim = 0; dim < 3; dim++) {
        detail::countFlops(operations, 3);
        result[dim] = left.volume * leftWeight[dim] + right.volume * rightWeight[dim];
    }
    return result;
}

// The endpoint-state midpoint update must center A as well as the Riemann flux.
// Differentiate each endpoint contribution V^2 W B (x_j-x_i) to equation (31).
inline Vector faceAreaRate(Geometry const& left, Geometry const& right,
    GeometryRate const& leftRate, GeometryRate const& rightRate, Vector const& displacement,
    Vector const& displacementRate, double kernelLeft, double kernelRight, double kernelRateLeft,
    double kernelRateRight, FlopCount* operations = nullptr) {
    auto const contribution = [&](Geometry const& geometry, GeometryRate const& rate, double kernel,
                                  double kernelRate) {
        Vector const projected = multiply(geometry.inverseMoment, displacement, operations);
        Vector const projectedRate = multiply(rate.inverseMoment, displacement, operations);
        Vector const motion = multiply(geometry.inverseMoment, displacementRate, operations);
        detail::countFlops(operations, 6);
        double const volumeSquared = geometry.volume * geometry.volume;
        double const coefficientRate =
            2.0 * geometry.volume * rate.volume * kernel + volumeSquared * kernelRate;
        Vector result{};
        for (int dim = 0; dim < 3; dim++) {
            detail::countFlops(operations, 5);
            result[dim] = coefficientRate * projected[dim] +
                volumeSquared * kernel * (projectedRate[dim] + motion[dim]);
        }
        return result;
    };
    Vector result = contribution(left, leftRate, kernelLeft, kernelRateLeft);
    Vector const other = contribution(right, rightRate, kernelRight, kernelRateRight);
    for (int dim = 0; dim < 3; dim++) {
        detail::countFlops(operations, 1);
        result[dim] += other[dim];
    }
    return result;
}

// Equations (74)-(75). Extrema include the central value. The two reconstructed
// extrema can be accumulated at the actual quadrature points in the neighbor loop.
inline double gradientLimiter(double center, double minimum, double maximum,
    double reconstructedMinimum, double reconstructedMaximum, double conditionNumber,
    FlopCount* operations = nullptr) {
    detail::countFlops(operations, 2);
    double const beta = std::max(1.0, 2.0 * std::min(1.0, 100.0 / conditionNumber));
    double limiter = 1.0;
    if (reconstructedMaximum > center) {
        detail::countFlops(operations, 4);
        limiter = std::min(limiter, beta * (maximum - center) / (reconstructedMaximum - center));
    }
    if (reconstructedMinimum < center) {
        detail::countFlops(operations, 4);
        limiter = std::min(limiter, beta * (center - minimum) / (center - reconstructedMinimum));
    }
    return std::clamp(limiter, 0.0, 1.0);
}

// Equation (76), psi_1=1/2 and psi_2=1/4. This pair limiter is applied to
// spatial reconstruction before the half-step time prediction.
inline double pairLimiter(double center, double neighbor, double reconstructed, double faceFraction,
    FlopCount* operations = nullptr) {
    if (center == neighbor) {
        return center;
    }
    detail::countFlops(operations, 9);
    double const difference = std::abs(neighbor - center);
    double const delta1 = 0.5 * difference;
    double const delta2 = 0.25 * difference;
    double const minimum = std::min(center, neighbor);
    double const maximum = std::max(center, neighbor);
    double lower = minimum - delta1;
    double upper = maximum + delta1;
    if (minimum > 0.0 && lower <= 0.0) {
        detail::countFlops(operations, 3);
        lower = minimum / (1.0 + delta1 / minimum);
    } else if (minimum == 0.0) {
        lower = 0.0;
    }
    if (maximum < 0.0 && upper >= 0.0) {
        detail::countFlops(operations, 3);
        upper = maximum / (1.0 + delta1 / std::abs(maximum));
    } else if (maximum == 0.0) {
        upper = 0.0;
    }
    double const mean = center + faceFraction * (neighbor - center);
    return center < neighbor ? std::max(lower, std::min(mean + delta2, reconstructed)) :
                               std::min(upper, std::max(mean - delta2, reconstructed));
}

inline bool physical(Primitive const& state, FlopCount* = nullptr) {
    return state.density > 0.0 && state.pressure > 0.0 && std::isfinite(state.density) &&
        std::isfinite(state.pressure) && std::isfinite(state.velocity[0]) &&
        std::isfinite(state.velocity[1]) && std::isfinite(state.velocity[2]);
}

// Appendix G: inactive cells retain their last active primitive/gradient
// snapshot. Predict that snapshot along the fluid trajectory to the current
// particle center before doing the separate face reconstruction. Sources and
// the conversion of elapsed time to hydrodynamic time belong to the caller.
// Return nonphysical predictions unchanged so the driver can wake the cell.
inline Primitive predictPrimitive(Primitive const& anchor, Gradient const& gradient, double elapsed,
    double gamma, FlopCount* operations = nullptr) {
    if (!(elapsed >= 0.0) || !std::isfinite(elapsed) || !(gamma > 1.0) || !std::isfinite(gamma)) {
        throw std::invalid_argument("Hopkins primitive prediction requires valid time and gamma");
    }
    detail::countFlops(operations, 19);
    Primitive result = anchor;
    double divergence = 0.0;
    for (int dim = 0; dim < 3; dim++) {
        divergence += gradient.velocity[dim][dim];
    }
    result.density -= elapsed * anchor.density * divergence;
    result.pressure -= elapsed * gamma * anchor.pressure * divergence;
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] -= elapsed * gradient.pressure[dim] / anchor.density;
    }
    return result;
}

// Appendix A, equations (58)-(60). Returned velocity remains in the laboratory
// frame; frameVelocity only sets the derivative's advective velocity. Passing a
// neighbor applies the Appendix B pair limiter before the temporal predictor.
// If reconstruction leaves the physical domain, fall back to the cell value.
inline Primitive reconstruct(Primitive const& center, Gradient const& gradient,
    Vector const& offset, Vector const& frameVelocity, double halfTimestep, double gamma,
    Primitive const* neighbor = nullptr, double faceFraction = 0.5,
    FlopCount* operations = nullptr) {
    detail::countFlops(operations, 32);
    Primitive result = center;
    result.density += dot(gradient.density, offset, operations);
    result.pressure += dot(gradient.pressure, offset, operations);
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] += dot(gradient.velocity[dim], offset, operations);
    }
    if (neighbor != nullptr) {
        result.density = pairLimiter(
            center.density, neighbor->density, result.density, faceFraction, operations);
        result.pressure = pairLimiter(
            center.pressure, neighbor->pressure, result.pressure, faceFraction, operations);
        for (int dim = 0; dim < 3; dim++) {
            result.velocity[dim] = pairLimiter(center.velocity[dim], neighbor->velocity[dim],
                result.velocity[dim], faceFraction, operations);
        }
    }
    Vector relativeVelocity{};
    double divergence = 0.0;
    for (int dim = 0; dim < 3; dim++) {
        relativeVelocity[dim] = center.velocity[dim] - frameVelocity[dim];
        divergence += gradient.velocity[dim][dim];
    }
    result.density -= halfTimestep *
        (dot(relativeVelocity, gradient.density, operations) + center.density * divergence);
    result.pressure -= halfTimestep *
        (dot(relativeVelocity, gradient.pressure, operations) +
            gamma * center.pressure * divergence);
    for (int dim = 0; dim < 3; dim++) {
        result.velocity[dim] -= halfTimestep *
            (dot(relativeVelocity, gradient.velocity[dim], operations) +
                gradient.pressure[dim] / center.density);
    }
    return physical(result, operations) ? result : center;
}

namespace detail {

    using Conserved = hopkins::Conserved;

    inline Conserved conserved(
        Primitive const& state, double gamma, FlopCount* operations = nullptr) {
        Conserved result;
        countFlops(operations, 8);
        result.mass = state.density;
        for (int dim = 0; dim < 3; dim++) {
            result.momentum[dim] = state.density * state.velocity[dim];
        }
        result.energy = state.pressure / (gamma - 1.0) +
            0.5 * state.density * dot(state.velocity, state.velocity, operations);
        return result;
    }

    inline Conserved physicalFlux(Primitive const& state, Vector const& normal, double gamma,
        FlopCount* operations = nullptr) {
        countFlops(operations, 12);
        double const normalVelocity = dot(state.velocity, normal, operations);
        Conserved result = conserved(state, gamma, operations);
        result.mass *= normalVelocity;
        for (int dim = 0; dim < 3; dim++) {
            result.momentum[dim] =
                result.momentum[dim] * normalVelocity + state.pressure * normal[dim];
        }
        result.energy = (result.energy + state.pressure) * normalVelocity;
        return result;
    }

    inline Conserved starState(Primitive const& state, Vector const& normal, double gamma,
        double waveSpeed, double contactSpeed, double starPressure,
        FlopCount* operations = nullptr) {
        double const normalVelocity = dot(state.velocity, normal, operations);
        Conserved result;
        countFlops(operations, 24);
        result.mass = state.density * (waveSpeed - normalVelocity) / (waveSpeed - contactSpeed);
        Vector velocity = state.velocity;
        for (int dim = 0; dim < 3; dim++) {
            velocity[dim] += (contactSpeed - normalVelocity) * normal[dim];
            result.momentum[dim] = result.mass * velocity[dim];
        }
        result.energy = ((waveSpeed - normalVelocity) * conserved(state, gamma, operations).energy -
                            state.pressure * normalVelocity + starPressure * contactSpeed) /
            (waveSpeed - contactSpeed);
        return result;
    }

    inline bool physical(Conserved const& state, FlopCount* operations = nullptr) {
        if (!(state.mass > 0.0) || !std::isfinite(state.mass) || !std::isfinite(state.energy)) {
            return false;
        }
        countFlops(operations, 3);
        return state.energy - 0.5 * dot(state.momentum, state.momentum, operations) / state.mass >
            0.0;
    }

    inline double pressureFunction(
        double pressure, Primitive const& state, double gamma, FlopCount* operations = nullptr) {
        countFlops(operations, 3);
        double const sound = std::sqrt(gamma * state.pressure / state.density);
        if (pressure > state.pressure) {
            countFlops(operations, 12);
            double const a = 2.0 / ((gamma + 1.0) * state.density);
            double const b = (gamma - 1.0) / (gamma + 1.0) * state.pressure;
            return (pressure - state.pressure) * std::sqrt(a / (pressure + b));
        }
        countFlops(operations, 10);
        return 2.0 * sound / (gamma - 1.0) *
            (std::pow(pressure / state.pressure, (gamma - 1.0) / (2.0 * gamma)) - 1.0);
    }

    // Bracketed ideal-gas exact pressure solve used only if all HLLC estimates fail
    // in MFM. The zero-pressure vacuum solution has identically zero contact flux.
    inline std::array<double, 2> exactContact(Primitive const& left, Primitive const& right,
        Vector const& normal, double gamma, FlopCount* operations = nullptr) {
        double const leftVelocity = dot(left.velocity, normal, operations);
        double const rightVelocity = dot(right.velocity, normal, operations);
        double const leftSound = std::sqrt(gamma * left.pressure / left.density);
        double const rightSound = std::sqrt(gamma * right.pressure / right.density);
        countFlops(operations, 11);
        if (rightVelocity - leftVelocity >= 2.0 * (leftSound + rightSound) / (gamma - 1.0)) {
            countFlops(operations, 2);
            return {0.0, 0.5 * (leftVelocity + rightVelocity)};
        }
        auto residual = [&](double pressure, FlopCount* residualOperations) {
            countFlops(residualOperations, 3);
            return pressureFunction(pressure, left, gamma, residualOperations) +
                pressureFunction(pressure, right, gamma, residualOperations) + rightVelocity -
                leftVelocity;
        };
        double lower = 0.0;
        double upper = std::max(left.pressure, right.pressure);
        for (int iteration = 0; iteration < 128 && residual(upper, operations) < 0.0; iteration++) {
            countFlops(operations, 1);
            upper *= 2.0;
        }
        if (!std::isfinite(upper) || residual(upper, nullptr) < 0.0) {
            throw std::runtime_error("Hopkins Riemann pressure could not be bracketed");
        }
        for (int iteration = 0; iteration < 160; iteration++) {
            countFlops(operations, 4);
            double const midpoint = 0.5 * (lower + upper);
            if (residual(midpoint, operations) > 0.0) {
                upper = midpoint;
            } else {
                lower = midpoint;
            }
            if (upper - lower <= 1.0e-12 * std::max(upper, 1.0e-100)) {
                break;
            }
        }
        countFlops(operations, 6);
        double const pressure = 0.5 * (lower + upper);
        double const velocity = 0.5 *
            (leftVelocity + rightVelocity + pressureFunction(pressure, right, gamma, operations) -
                pressureFunction(pressure, left, gamma, operations));
        return {pressure, velocity};
    }

}    // namespace detail

// An outward (i -> j) flux, including |area|. Apply -dt*flux to i and +dt*flux
// to j, once per pair. Appendix A supplies the Galilean boost/deboost. In MFM
// the Riemann face additionally moves at S_* (section 2.6): mass flux is exactly
// zero, momentum flux is P_* n, and energy flux is P_* (S_*+v_frame dot n).
inline Flux riemannFlux(Primitive left, Primitive right, Vector const& area,
    Vector const& frameVelocity, double gamma, bool finiteMass, FlopCount* operations = nullptr) {
    if (!physical(left, operations) || !physical(right, operations) || !(gamma > 1.0) ||
        !std::isfinite(gamma)) {
        throw std::invalid_argument("Hopkins Riemann solver requires physical ideal-gas states");
    }
    Flux result;
    double const areaMagnitude = norm(area, operations);
    if (areaMagnitude == 0.0) {
        return result;
    }
    detail::countFlops(operations, 48);
    Vector normal{};
    for (int dim = 0; dim < 3; dim++) {
        normal[dim] = area[dim] / areaMagnitude;
        left.velocity[dim] -= frameVelocity[dim];
        right.velocity[dim] -= frameVelocity[dim];
    }
    double const leftVelocity = dot(left.velocity, normal, operations);
    double const rightVelocity = dot(right.velocity, normal, operations);
    double const leftSound = std::sqrt(gamma * left.pressure / left.density);
    double const rightSound = std::sqrt(gamma * right.pressure / right.density);
    auto const leftState = detail::conserved(left, gamma, operations);
    auto const rightState = detail::conserved(right, gamma, operations);
    auto const leftFlux = detail::physicalFlux(left, normal, gamma, operations);
    auto const rightFlux = detail::physicalFlux(right, normal, gamma, operations);
    double const rootLeft = std::sqrt(left.density);
    double const rootRight = std::sqrt(right.density);
    Vector roeVelocity{};
    for (int dim = 0; dim < 3; dim++) {
        roeVelocity[dim] = (rootLeft * left.velocity[dim] + rootRight * right.velocity[dim]) /
            (rootLeft + rootRight);
    }
    double const roeEnthalpy = ((leftState.energy + left.pressure) / rootLeft +
                                   (rightState.energy + right.pressure) / rootRight) /
        (rootLeft + rootRight);
    double const roeSound = std::sqrt(std::max(
        0.0, (gamma - 1.0) * (roeEnthalpy - 0.5 * dot(roeVelocity, roeVelocity, operations))));
    double leftWave =
        std::min(leftVelocity - leftSound, dot(roeVelocity, normal, operations) - roeSound);
    double rightWave =
        std::max(rightVelocity + rightSound, dot(roeVelocity, normal, operations) + roeSound);
    double starPressure = 0.0;
    double contact = 0.0;
    detail::Conserved leftStar;
    detail::Conserved rightStar;
    bool good = false;
    for (int attempt = 0; attempt < 3; attempt++) {
        if (attempt == 1) {
            detail::countFlops(operations, 4);
            leftWave = std::min(leftVelocity - leftSound, rightVelocity - rightSound);
            rightWave = std::max(leftVelocity + leftSound, rightVelocity + rightSound);
        } else if (attempt == 2) {
            detail::countFlops(operations, 2);
            double const speed =
                std::max(std::abs(leftVelocity) + leftSound, std::abs(rightVelocity) + rightSound);
            leftWave = -speed;
            rightWave = speed;
        }
        detail::countFlops(operations, 19);
        double const leftImpedance = left.density * (leftWave - leftVelocity);
        double const rightImpedance = right.density * (rightWave - rightVelocity);
        contact = (right.pressure - left.pressure + leftImpedance * leftVelocity -
                      rightImpedance * rightVelocity) /
            (leftImpedance - rightImpedance);
        double const leftPressure = left.pressure + leftImpedance * (contact - leftVelocity);
        double const rightPressure = right.pressure + rightImpedance * (contact - rightVelocity);
        starPressure = 0.5 * (leftPressure + rightPressure);
        if (std::isfinite(contact) && contact > leftWave && contact < rightWave &&
            starPressure > 0.0 && std::isfinite(starPressure)) {
            leftStar =
                detail::starState(left, normal, gamma, leftWave, contact, starPressure, operations);
            rightStar = detail::starState(
                right, normal, gamma, rightWave, contact, starPressure, operations);
            good =
                detail::physical(leftStar, operations) && detail::physical(rightStar, operations);
        }
        if (good) {
            result.usedFallback = attempt != 0;
            break;
        }
    }
    result.signalSpeed = std::max(std::abs(leftWave), std::abs(rightWave));
    detail::Conserved flux;
    if (finiteMass) {
        if (!good) {
            auto const exact = detail::exactContact(left, right, normal, gamma, operations);
            starPressure = exact[0];
            contact = exact[1];
            result.usedFallback = true;
        }
        detail::countFlops(operations, 4);
        for (int dim = 0; dim < 3; dim++) {
            flux.momentum[dim] = starPressure * normal[dim];
        }
        flux.energy = starPressure * contact;
    } else if (leftWave >= 0.0) {
        flux = leftFlux;
    } else if (rightWave <= 0.0) {
        flux = rightFlux;
    } else if (good) {
        auto const& state = contact >= 0.0 ? leftState : rightState;
        auto const& star = contact >= 0.0 ? leftStar : rightStar;
        flux = contact >= 0.0 ? leftFlux : rightFlux;
        double const wave = contact >= 0.0 ? leftWave : rightWave;
        detail::countFlops(operations, 15);
        flux.mass += wave * (star.mass - state.mass);
        flux.energy += wave * (star.energy - state.energy);
        for (int dim = 0; dim < 3; dim++) {
            flux.momentum[dim] += wave * (star.momentum[dim] - state.momentum[dim]);
        }
    } else {
        // A positivity-preserving two-wave HLLE flux. Symmetric final waves
        // reduce this to local Lax-Friedrichs for the pathological fallback.
        detail::countFlops(operations, 2);
        double const inverseWidth = 1.0 / (rightWave - leftWave);
        auto combine = [&](double fLeft, double fRight, double uLeft, double uRight) {
            detail::countFlops(operations, 8);
            return (rightWave * fLeft - leftWave * fRight +
                       leftWave * rightWave * (uRight - uLeft)) *
                inverseWidth;
        };
        flux.mass = combine(leftFlux.mass, rightFlux.mass, leftState.mass, rightState.mass);
        flux.energy =
            combine(leftFlux.energy, rightFlux.energy, leftState.energy, rightState.energy);
        for (int dim = 0; dim < 3; dim++) {
            flux.momentum[dim] = combine(leftFlux.momentum[dim], rightFlux.momentum[dim],
                leftState.momentum[dim], rightState.momentum[dim]);
        }
        result.usedFallback = true;
    }
    detail::countFlops(operations, 15);
    result.contactSpeed = contact;
    result.mass = flux.mass * areaMagnitude;
    result.energy = (flux.energy + dot(frameVelocity, flux.momentum, operations) +
                        0.5 * dot(frameVelocity, frameVelocity, operations) * flux.mass) *
        areaMagnitude;
    for (int dim = 0; dim < 3; dim++) {
        result.momentum[dim] =
            (flux.momentum[dim] + frameVelocity[dim] * flux.mass) * areaMagnitude;
    }
    if (!std::isfinite(result.mass) || !std::isfinite(result.energy) ||
        !std::isfinite(norm(result.momentum))) {
        throw std::runtime_error("Hopkins Riemann solver returned a non-finite flux");
    }
    return result;
}

}    // namespace hopkins
