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
#include <cosmictiger/hydro/hopkins/hopkins_simd.hpp>
#include <cosmictiger/math/matrix.hpp>
#include <cosmictiger/math/vector3.hpp>

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

using Vector = Vector3<double>;
using Matrix = math::Matrix<double, 3>;

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
    double contactSpeed = 0.0;    // Contact, or HLL massless-face speed, in the input frame.
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

template <class LeftDerived, class RightDerived>
inline double dot(math::Vector<double, 3, LeftDerived> const& a,
    math::Vector<double, 3, RightDerived> const& b, FlopCount* operations = nullptr) {
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

// Integrated conserved variables require the volume at the same anchor time.
inline Primitive primitiveFromConserved(Conserved const& state, double volume, double gamma) {
    if (!(state.mass > 0.0) || !(volume > 0.0) || !(gamma > 1.0)) {
        throw std::invalid_argument("Invalid Hopkins conserved primitive conversion");
    }
    Primitive result;
    result.density = state.mass / volume;
    for (int dim = 0; dim < 3; ++dim) {
        result.velocity[dim] = state.momentum[dim] / state.mass;
    }
    result.pressure = (gamma - 1.0) *
        (state.energy - 0.5 * dot(state.momentum, state.momentum) / state.mass) / volume;
    return result;
}

// A single own-step source closure. The old-mass opening source and new-mass
// closing source surround the accumulated conservative exchange (Appendix H1).
inline Conserved closeConservedInterval(Conserved const& anchor, Conserved const& increment,
    Vector const& oldAcceleration, Vector const& newAcceleration, double duration, double damping,
    double oldExpansion, double newExpansion, double gravityWork = 0.0) {
    if (!(duration >= 0.0) || !std::isfinite(duration) || !(anchor.mass > 0.0) ||
        !(damping >= 0.0)) {
        throw std::invalid_argument("Invalid Hopkins own-step source interval");
    }
    if (duration == 0.0 && increment.mass == 0.0 && increment.energy == 0.0 &&
        dot(increment.momentum, increment.momentum) == 0.0 && gravityWork == 0.0) {
        return anchor;
    }
    double const half = 0.5 * duration;
    double const decay = std::exp(-damping * half);
    double const impulse = damping == 0.0 ? half : -std::expm1(-damping * half) / damping;
    Vector oldVelocity{}, velocity{};
    Conserved result;
    result.mass = anchor.mass + increment.mass;
    if (!(result.mass > 0.0) || !std::isfinite(result.mass)) {
        throw std::runtime_error("Nonpositive Hopkins mass at own-step closure");
    }
    double const initialThermal =
        anchor.energy - 0.5 * dot(anchor.momentum, anchor.momentum) / anchor.mass;
    double openingEnergy = anchor.energy + initialThermal * std::expm1(half * oldExpansion);
    for (int dim = 0; dim < 3; ++dim) {
        oldVelocity[dim] = anchor.momentum[dim] / anchor.mass;
        double const openingMomentum =
            decay * anchor.momentum[dim] + anchor.mass * impulse * oldAcceleration[dim];
        velocity[dim] = openingMomentum / anchor.mass;
        openingEnergy += 0.5 * anchor.mass * (velocity[dim] - oldVelocity[dim]) *
            (velocity[dim] + oldVelocity[dim]);
        result.momentum[dim] = openingMomentum + increment.momentum[dim];
    }
    result.energy = openingEnergy + increment.energy + gravityWork;
    double const thermal =
        result.energy - 0.5 * dot(result.momentum, result.momentum) / result.mass;
    result.energy += thermal * std::expm1(half * newExpansion);
    for (int dim = 0; dim < 3; ++dim) {
        oldVelocity[dim] = result.momentum[dim] / result.mass;
        result.momentum[dim] =
            decay * result.momentum[dim] + result.mass * impulse * newAcceleration[dim];
        velocity[dim] = result.momentum[dim] / result.mass;
        result.energy += 0.5 * result.mass * (velocity[dim] - oldVelocity[dim]) *
            (velocity[dim] + oldVelocity[dim]);
    }
    if (!std::isfinite(result.energy)) {
        throw std::runtime_error("Nonfinite Hopkins energy at own-step closure");
    }
    // The thermal part may be nonpositive here: the caller owns the dual-energy decision.
    return result;
}

// Thermal energy recovered from the conserved total by subtracting the kinetic energy.
inline double thermalEnergy(Conserved const& state, FlopCount* operations = nullptr) {
    if (!(state.mass > 0.0)) {
        throw std::invalid_argument("Hopkins thermal energy requires a positive mass");
    }
    detail::countFlops(operations, 3);
    return state.energy - 0.5 * dot(state.momentum, state.momentum, operations) / state.mass;
}

// Entropic function K = P rho^-gamma = (gamma-1) u rho^(1-gamma) of a cell with specific
// thermal energy u. It is the dual-energy auxiliary variable: an MFM cell exchanges no
// mass, so K is constant along the cell in smooth adiabatic flow and only shocks,
// the cosmological expansion source and the thermal sources change it. The same comoving
// variables as the conserved state are used throughout, so a homogeneous expansion
// scales K exactly as the closure scales the thermal energy.
inline double entropyFunction(
    double specificThermalEnergy, double density, double gamma, FlopCount* operations = nullptr) {
    if (!(density > 0.0) || !std::isfinite(density) || !(gamma > 1.0) || !std::isfinite(gamma) ||
        !std::isfinite(specificThermalEnergy)) {
        throw std::invalid_argument("Invalid Hopkins entropic function inputs");
    }
    detail::countFlops(operations, 3);
    return (gamma - 1.0) * specificThermalEnergy * std::pow(density, 1.0 - gamma);
}

// Specific thermal energy u = K rho^(gamma-1) / (gamma-1) implied by the entropic function.
inline double thermalEnergyFromEntropy(
    double entropy, double density, double gamma, FlopCount* operations = nullptr) {
    if (!(density > 0.0) || !std::isfinite(density) || !(gamma > 1.0) || !std::isfinite(gamma) ||
        !std::isfinite(entropy)) {
        throw std::invalid_argument("Invalid Hopkins entropy thermal energy inputs");
    }
    detail::countFlops(operations, 3);
    return entropy * std::pow(density, gamma - 1.0) / (gamma - 1.0);
}

struct DualEnergyClosure {
    Conserved state;
    bool switched = false;
    // Energy added to the cell by replacing the total-energy thermal estimate.
    double correction = 0.0;
};

// Dual-energy pressure switch of Bryan et al. (1995), Comput. Phys. Commun. 89, 149,
// equation 10, with the entropic function as the auxiliary variable. While the thermal
// energy recovered from the total energy exceeds pressureSwitch times the total energy,
// the conservative closure stands. Otherwise the thermal energy is swamped by kinetic
// cancellation (cold, high-Mach gas) and is replaced by the entropy estimate at the
// closing volume; the returned correction is the resulting change of total energy,
// which the caller books as a nonconservative source. A pressure switch of zero keeps
// every positive total-energy estimate and still rescues a nonpositive one.
inline DualEnergyClosure dualEnergyClosure(Conserved const& state, double entropy, double volume,
    double gamma, double pressureSwitch, FlopCount* operations = nullptr) {
    if (!(state.mass > 0.0) || !std::isfinite(state.mass) || !(volume > 0.0) ||
        !std::isfinite(volume) || !(gamma > 1.0) || !std::isfinite(gamma) || !(entropy > 0.0) ||
        !std::isfinite(entropy) || !(pressureSwitch >= 0.0) || !(pressureSwitch < 1.0)) {
        throw std::invalid_argument("Invalid Hopkins dual-energy closure");
    }
    DualEnergyClosure result;
    result.state = state;
    detail::countFlops(operations, 3);
    double const kinetic = 0.5 * dot(state.momentum, state.momentum, operations) / state.mass;
    double const thermal = state.energy - kinetic;
    if (std::isfinite(thermal) && thermal > pressureSwitch * state.energy) {
        return result;
    }
    detail::countFlops(operations, 4);
    double const replacement =
        state.mass * thermalEnergyFromEntropy(entropy, state.mass / volume, gamma, operations);
    if (!(replacement > 0.0) || !std::isfinite(replacement)) {
        throw std::runtime_error(
            "Nonpositive Hopkins thermal energy at own-step closure from both the total energy "
            "and the entropy; reduce hydro.cfl (published pair transfers cannot be retried "
            "one-sided)");
    }
    result.state.energy = kinetic + replacement;
    result.switched = true;
    result.correction = result.state.energy - state.energy;
    return result;
}

// Dual-energy update switch of Bryan et al. (1995), equation 11: the auxiliary follows
// the total-energy closure whenever the cell's thermal energy density reaches
// updateSwitch times the largest total energy density among the cell and its neighbors,
// so shock heating enters the entropy; a cold cell inside a hot neighborhood keeps its
// own adiabatic entropy. A nonpositive thermal estimate is never adopted.
inline bool entropyUpdateDue(
    double thermalEnergyDensity, double maximumEnergyDensity, double updateSwitch) {
    if (!(updateSwitch >= 0.0) || !std::isfinite(updateSwitch)) {
        throw std::invalid_argument("Invalid Hopkins dual-energy update switch");
    }
    return thermalEnergyDensity > 0.0 && std::isfinite(thermalEnergyDensity) &&
        thermalEnergyDensity >= updateSwitch * maximumEnergyDensity;
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
    detail::countFlops(operations, 15);
    auto const result = matrix * vector;
    return {result[0], result[1], result[2]};
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

namespace detail {

    template <int dimensions>
    inline Geometry invertMoment(Geometry result, Matrix const& moment, double scale,
        double maximumCondition, FlopCount* operations) {
        math::SymmetricMatrix<double, dimensions> activeMoment;
        math::SymmetricMatrix<double, dimensions> normalized;
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column <= row; ++column) {
                activeMoment(row, column) = moment(row, column);
                countFlops(operations, 1);
                normalized(row, column) = moment(row, column) / scale;
            }
        }
        // These leading-minor ratios are the squared Cholesky pivots. Preserve
        // the normalized SPD rejection policy without factoring the matrix.
        // This arithmetic is validation only, so it does not enter FLOP totals.
        constexpr double kPivotTolerance = 64.0 * std::numeric_limits<double>::epsilon();
        if (!(normalized(0, 0) > kPivotTolerance)) {
            return result;
        }
        if constexpr (dimensions >= 2) {
            double const minor =
                normalized(0, 0) * normalized(1, 1) - normalized(0, 1) * normalized(0, 1);
            if (!(minor / normalized(0, 0) > kPivotTolerance)) {
                return result;
            }
        }
        // The application SPD and condition checks own admissibility. The shared
        // inverse still checks exact singularity and finite inputs/results.
        math::SymmetricMatrix<double, dimensions> inverse;
        size_t inverseOperations = 0;
        auto const status =
            math::tryInverse(activeMoment, inverse, math::MatrixOptions<double>{0.0, 0.0},
                operations != nullptr ? &inverseOperations : nullptr);
        countFlops(operations, inverseOperations);
        if (status != math::MatrixStatus::Success) {
            return result;
        }
        // Cancel the physical scale before squaring either matrix. Computing
        // ||E||^2 ||E^-1||^2 first can overflow or underflow for a well-conditioned E.
        countFlops(operations, dimensions * (dimensions + 1) / 2);
        auto const normalizedInverse = inverse * scale;
        if constexpr (dimensions == 3) {
            // B_22 = minor_01/det(E) is the reciprocal of the final squared
            // Cholesky pivot. Reuse it instead of evaluating the determinant twice.
            double const inversePivot = normalizedInverse(2, 2);
            if (!(inversePivot > 0.0 && inversePivot < 1.0 / kPivotTolerance)) {
                return result;
            }
        }
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column < dimensions; ++column) {
                result.inverseMoment(row, column) = inverse(row, column);
            }
        }
        countFlops(operations, 4 * dimensions * dimensions + 3);
        result.conditionNumber =
            std::sqrt(math::squaredNorm(normalized) * math::squaredNorm(normalizedInverse)) /
            double(dimensions);
        result.valid =
            std::isfinite(result.conditionNumber) && result.conditionNumber <= maximumCondition;
        return result;
    }

}    // namespace detail

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
            moment(i, j) = kernelWeightedMoment(i, j) / kernelSum;
            if (!std::isfinite(moment(i, j))) {
                return result;
            }
            scale = std::max(scale, std::abs(moment(i, j)));
        }
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return result;
    }
    if (dimensions == 1) {
        return detail::invertMoment<1>(result, moment, scale, maximumCondition, operations);
    }
    if (dimensions == 2) {
        return detail::invertMoment<2>(result, moment, scale, maximumCondition, operations);
    }
    return detail::invertMoment<3>(result, moment, scale, maximumCondition, operations);
}

// Differentiate B = E^-1, where E is the normalized moment used by makeGeometry.
// momentRate is dE/dt, not the derivative of its unnormalized kernel sum.
inline GeometryRate geometryRate(Geometry const& geometry, double volumeRate,
    Matrix const& momentRate, FlopCount* operations = nullptr) {
    GeometryRate result;
    result.volume = volumeRate;
    detail::countFlops(operations, 90);
    result.inverseMoment = -(geometry.inverseMoment * momentRate * geometry.inverseMoment);
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

// An outward (i -> j) flux, including |area|. Apply -dt*flux to i and +dt*flux
// to j, once per pair. Appendix A supplies the Galilean boost/deboost. In MFM
// accepted HLLC uses a face moving at S_* (section 2.6): mass flux is exactly
// zero, momentum flux is P_* n, and energy flux is P_* (S_*+v_frame dot n).
// HLL fallback uses its massless-plane speed and the full moving-face flux.
// Scalar reference and packed production share the HLLC/HLL kernel.
inline Flux riemannFlux(Primitive const& left, Primitive const& right, Vector const& area,
    Vector const& frameVelocity, double gamma, bool finiteMass, FlopCount* operations = nullptr) {
    PrimitiveValues<double> const leftValues{left.density, left.velocity, left.pressure};
    PrimitiveValues<double> const rightValues{right.density, right.velocity, right.pressure};
    auto const result = riemannFluxValues(
        leftValues, rightValues, area, frameVelocity, gamma, finiteMass, true, operations);
    return {result.mass, result.momentum, result.energy, result.signalSpeed, result.contactSpeed};
}

}    // namespace hopkins
