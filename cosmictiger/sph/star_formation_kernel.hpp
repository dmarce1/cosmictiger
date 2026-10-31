/* CosmicTiger - SH03 per-particle source operations; BSL-1.0. */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/particles/star_formation_state.hpp>
#include <cosmictiger/sph/springel_and_hernquist/cooling.hpp>

namespace star_formation_backend {

// Host unit conversion supplies these scales once per batch. Keep the order of
// each conversion identical to the typed host boundary (including powers of a).
struct Parameters {
    springel_and_hernquist::Model model;
    springel_and_hernquist::AtomicCooling cooling;
    primordial::Environment environment;
    double densityScale{};
    double energyScale{};
    double hydrogenMass{};
    double scaleFactor{};
    double timeSeconds{};
    double lengthCm{};
    double gasMass{};
    double scaleDerivative{};
    double baseTimestep{};
    double explicitTimestep{-1.0};
    double metalYield{};
    double windMassLoading{};
    double windEnergyFraction{};
    double windDecouplingSeconds{};
    double windRecoupleDensityFraction{};
    // Newborn collisionless timestep and kick scales, already rounded as in
    // the host gravity boundary. baseTimestep retains double kick precision.
    float effectiveRadius{};
    float timestepBase{};
    float timestepCap{};
    float timestepFactor{};
    float kickSign{1.0f};
    int rungLimit{};
    uint64_t seed{};
    int minimumRung{};
    int uniformRung{-1};
    bool opening{};
    bool winds{};
};

struct Input {
    StarFormationState state;
    primordial::State chemistry;
    double chemistryHalfInterval{};
    double chemistryPendingHeating{};
    float velocity[3]{};
    float gravity[3]{};
    float energy{};
    float density{};
    float pressureRatio{};
    int rung{};
};
struct Output {
    StarFormationState state;
    primordial::State chemistry;
    double chemistryInterval{};
    double value{};    // Minimum timestep, pressure integral contribution, or heating.
    float velocity[3]{};
    float energy{};
    float pressureRatio{};
    float signalIndex{};
    int rung{};
    int status{};    // Zero means the entire particle operation succeeded.
    bool project{};
    bool convert{};
};
enum class Operation { eos, timestep, prepare, finish, pressure, openNewStar };

namespace detail {
    CUDA_EXPORT inline uint64_t mix(uint64_t value) {
        value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }
    CUDA_EXPORT inline double uniform(
        uint64_t identity, uint64_t event, uint64_t stream, uint64_t seed) {
        uint64_t const bits = mix(mix(identity ^ seed) ^ mix(event + UINT64_C(0x9e3779b97f4a7c15)) ^
            mix(stream + UINT64_C(0x632be59bd9b4e019)));
        return double(bits >> 11) * 0x1.0p-53;
    }
    CUDA_EXPORT inline double minimum(double a, double b) {
        return a < b ? a : b;
    }
    CUDA_EXPORT inline double maximum(double a, double b) {
        return a > b ? a : b;
    }
    CUDA_EXPORT inline bool finite(double value) {
        return springel_and_hernquist::detail::finite(value);
    }
    CUDA_EXPORT inline double density(Input const& input, Parameters const& p) {
        return double(input.density) * p.densityScale / p.scaleFactor / p.scaleFactor /
            p.scaleFactor;
    }
    CUDA_EXPORT inline double energy(float value, Parameters const& p) {
        return double(value) * p.energyScale / p.scaleFactor / p.scaleFactor;
    }
    CUDA_EXPORT inline bool checkedFloat(double value, float& result) {
        if (!finite(value) || ::fabs(value) > FLT_MAX ||
            (value != 0.0 && ::fabs(value) < FLT_MIN)) {
            return false;
        }
        result = float(value);
        return true;
    }
    CUDA_EXPORT inline bool synchronizeEnergy(
        primordial::State& state, float value, Parameters const& p) {
        return checkedFloat(
            energy(value, p) * p.hydrogenMass / (p.cooling.hydrogenMassFraction * 1.602176634e-12),
            state.thermalEnergy);
    }
    CUDA_EXPORT inline bool storedEnergy(double value, Parameters const& p, float& result) {
        double const stored = value / p.energyScale * p.scaleFactor * p.scaleFactor;
        if (!(stored > 0.0) || !finite(stored) || stored > FLT_MAX) {
            return false;
        }
        result = float(stored);
        return true;
    }
    CUDA_EXPORT inline bool equilibrium(
        double rho, Parameters const& p, springel_and_hernquist::Equilibrium& state) {
        return springel_and_hernquist::evaluateEquilibrium(p.model, p.cooling, rho, state);
    }
    CUDA_EXPORT inline bool starForming(Input const& input, Parameters const& p, double rho) {
        return rho >= p.model.thresholdDensity && input.state.windTimeRemaining <= 0.0;
    }
    CUDA_EXPORT inline Output initial(Input const& input) {
        Output result;
        result.state = input.state;
        result.chemistry = input.chemistry;
        result.energy = input.energy;
        result.rung = input.rung;
        for (int dim = 0; dim < 3; ++dim) {
            result.velocity[dim] = input.velocity[dim];
        }
        return result;
    }
}    // namespace detail

CUDA_EXPORT inline Output evaluateEos(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    double pressureRatio;
    double signalIndex;
    double const rho = detail::density(input, p);
    if (detail::starForming(input, p, rho)) {
        springel_and_hernquist::Equilibrium state, lower, upper;
        double const lowerDensity = detail::maximum(p.model.thresholdDensity, rho * 0.999);
        double const upperDensity = rho * 1.001;
        if (!detail::equilibrium(rho, p, state) || !detail::equilibrium(lowerDensity, p, lower) ||
            !detail::equilibrium(upperDensity, p, upper)) {
            output.status = 1;
            return output;
        }
        pressureRatio = (p.model.parameters.gamma - 1.0) * state.effectiveEnergy /
            detail::energy(input.energy, p);
        signalIndex = ::log(upper.pressure / lower.pressure) / ::log(upperDensity / lowerDensity);
    } else {
        primordial::Eos eos;
        if (!detail::synchronizeEnergy(output.chemistry, input.energy, p) ||
            primordial::evaluateEos(output.chemistry, p.environment, eos) !=
                primordial::Status::Success) {
            output.status = 2;
            return output;
        }
        pressureRatio = eos.pressureEnergyRatio;
        signalIndex = eos.adiabaticIndex;
    }
    if (!(pressureRatio > 0.0) || !(signalIndex > 0.0) || !detail::finite(pressureRatio) ||
        !detail::finite(signalIndex)) {
        output.status = 3;
        return output;
    }
    output.pressureRatio = float(pressureRatio);
    output.signalIndex = float(signalIndex);
    return output;
}
CUDA_EXPORT inline Output maximumTimestep(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    output.value = DBL_MAX;
    if (input.rung < p.minimumRung) {
        return output;
    }
    double physicalLimit = DBL_MAX;
    double const rho = detail::density(input, p);
    if (detail::starForming(input, p, rho)) {
        springel_and_hernquist::Equilibrium state;
        if (!detail::equilibrium(rho, p, state)) {
            output.status = 1;
            return output;
        }
        physicalLimit =
            0.05 / (state.specificStarFormationRate * (1.0 + (p.winds ? p.windMassLoading : 0.0)));
    } else if (input.state.windTimeRemaining > 0.0) {
        physicalLimit = input.state.windTimeRemaining;
    }
    output.value = physicalLimit / p.scaleFactor / p.timeSeconds;
    return output;
}
CUDA_EXPORT inline Output prepare(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    int const rung = p.opening && p.uniformRung >= 0 ? p.uniformRung : input.rung;
    double const halfInterval =
        p.explicitTimestep >= 0.0 ? 0.5 * p.explicitTimestep : ::ldexp(0.5 * p.baseTimestep, -rung);
    auto& state = output.state;
    if (!detail::finite(p.explicitTimestep) || state.id == 0 || !(halfInterval > 0.0) ||
        !detail::finite(halfInterval) ||
        (p.opening && (state.sourceHalfInterval != 0.0 || state.pendingHeating != 0.0)) ||
        (!p.opening && state.sourceHalfInterval != halfInterval) ||
        input.chemistryHalfInterval != 0.0 || input.chemistryPendingHeating != 0.0) {
        output.status = 4;
        return output;
    }
    double const midpointScale =
        p.scaleFactor + (p.opening ? 0.5 : -0.5) * p.scaleDerivative * halfInterval;
    double const physicalInterval = midpointScale * halfInterval * p.timeSeconds;
    if (!(physicalInterval > 0.0) || !detail::finite(physicalInterval)) {
        output.status = 5;
        return output;
    }
    double const rho = detail::density(input, p);
    if (rho < p.windRecoupleDensityFraction * p.model.thresholdDensity) {
        state.windTimeRemaining = 0.0;
    }
    double const windInterval = detail::minimum(state.windTimeRemaining, physicalInterval);
    state.windTimeRemaining -= windInterval;
    double const eligibleInterval = physicalInterval - windInterval;
    output.project = eligibleInterval > 0.0 && rho >= p.model.thresholdDensity;
    output.chemistryInterval = output.project ? windInterval : physicalInterval;
    double opticalDepth = 0.0;
    if (output.project) {
        springel_and_hernquist::Equilibrium phase;
        springel_and_hernquist::AtomicEquilibrium hot;
        if (!detail::equilibrium(rho, p, phase) ||
            !p.cooling.evaluateAtEnergy(phase.hotEnergy, hot) ||
            !detail::storedEnergy(phase.effectiveEnergy, p, output.energy)) {
            output.status = 1;
            return output;
        }
        output.chemistry = primordial::State{};
        double const heliumRatio =
            (1.0 - p.cooling.hydrogenMassFraction) / (4.0 * p.cooling.hydrogenMassFraction);
        output.chemistry.abundances[0] =
            float(phase.coldFraction + phase.hotFraction * hot.hydrogenNeutral);
        output.chemistry.abundances[1] = float(phase.hotFraction * hot.hydrogenIonized);
        output.chemistry.abundances[2] =
            float(heliumRatio * (phase.coldFraction + phase.hotFraction * hot.heliumNeutral));
        output.chemistry.abundances[3] = float(heliumRatio * phase.hotFraction * hot.heliumIonized);
        output.chemistry.abundances[4] =
            float(heliumRatio * phase.hotFraction * hot.heliumDoubleIonized);
        output.chemistry.abundances[5] = float(phase.hotFraction * hot.electronsPerHydrogen);
        if (!detail::synchronizeEnergy(output.chemistry, output.energy, p)) {
            output.status = 2;
            return output;
        }
        opticalDepth = phase.specificStarFormationRate * eligibleInterval;
        state.metallicity += float(
            springel_and_hernquist::enrichmentIncrement(phase, eligibleInterval, p.metalYield));
    }
    if (!(state.metallicity >= 0.0f && state.metallicity <= 1.0f)) {
        output.status = 6;
        return output;
    }
    state.formationOpticalDepth += opticalDepth;
    state.elapsedPhysicalTime += physicalInterval;
    state.sourceHalfInterval = halfInterval;
    return output;
}
// Chemistry has completed transactionally before this operation. Its output is
// applied only to ordinary gas; a wind recoupling midway ends on the SH03 branch.
CUDA_EXPORT inline Output finish(Input const& input, Output output, Parameters const& p) {
    auto& state = output.state;
    double kineticChange = 0.0;
    if (!p.opening) {
        double const probability = -::expm1(-state.formationOpticalDepth);
        output.convert = detail::uniform(state.id, state.eventCounter, 0, p.seed) < probability;
        if (output.convert) {
            output.energy = 0.0f;
        } else if (p.winds && state.windTimeRemaining == 0.0 &&
            detail::uniform(state.id, state.eventCounter, 1, p.seed) <
                -::expm1(-p.windMassLoading * state.formationOpticalDepth)) {
            double const speed = springel_and_hernquist::windSpeed(
                p.model.parameters, p.windMassLoading, p.windEnergyFraction);
            double const cosine =
                2.0 * detail::uniform(state.id, state.eventCounter, 2, p.seed) - 1.0;
            double const azimuth =
                6.2831853071795864769 * detail::uniform(state.id, state.eventCounter, 3, p.seed);
            double const sine = ::sqrt(detail::maximum(0.0, 1.0 - cosine * cosine));
            double const direction[] = {sine * ::cos(azimuth), sine * ::sin(azimuth), cosine};
            double const storedSpeed = speed * p.scaleFactor * p.timeSeconds / p.lengthCm;
            for (int dim = 0; dim < 3; ++dim) {
                double const before = output.velocity[dim];
                output.velocity[dim] = float(before + storedSpeed * direction[dim]);
                kineticChange +=
                    0.5 * (double(output.velocity[dim]) * output.velocity[dim] - before * before);
            }
            state.windTimeRemaining = p.windDecouplingSeconds;
        }
        ++state.eventCounter;
        state.formationOpticalDepth = 0.0;
    }
    double const change =
        -p.gasMass * (double(output.energy) - input.energy + kineticChange) / p.scaleFactor;
    if (p.opening) {
        state.pendingHeating = change;
    } else {
        output.value = input.state.pendingHeating + change;
        state.sourceHalfInterval = 0.0;
        state.pendingHeating = 0.0;
    }
    return output;
}
// A converted particle missed the collisionless gravity timestep proposal.
// Apply that criterion and its first opening kick before the next drift.
CUDA_EXPORT inline Output openNewStar(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    float accelerationSquared = 1.0e-35f;
    for (int dim = 0; dim < 3; ++dim) {
        accelerationSquared += input.gravity[dim] * input.gravity[dim];
    }
    float const gravityTimestep =
        p.timestepFactor * ::sqrtf(p.effectiveRadius / ::sqrtf(accelerationSquared));
    float const boundedTimestep =
        p.timestepBase < gravityTimestep ? p.timestepBase : gravityTimestep;
    float const timestep = boundedTimestep < p.timestepCap ? boundedTimestep : p.timestepCap;
    float const requestedRung = ::ceilf(::log2f(p.timestepBase / timestep));
    if (!(timestep > 0.0f) || !detail::finite(timestep) || !detail::finite(requestedRung) ||
        requestedRung >= p.rungLimit) {
        output.status = 8;
        return output;
    }
    int rung = int(requestedRung);
    if (rung < input.rung - 1) {
        rung = input.rung - 1;
    }
    if (rung < p.minimumRung) {
        rung = p.minimumRung;
    }
    if (rung < 1) {
        rung = 1;
    }
    output.rung = rung;
    double const half = p.kickSign * ::ldexp(0.5 * p.baseTimestep, -rung);
    for (int dim = 0; dim < 3; ++dim) {
        // Explicit FMA matches the host kick even with contraction disabled.
        output.velocity[dim] =
            float(::fma(double(input.gravity[dim]), half, double(input.velocity[dim])));
    }
    return output;
}
CUDA_EXPORT inline Output evaluate(
    Operation operation, Input const& input, Output output, Parameters const& p) {
    switch (operation) {
    case Operation::eos:
        return evaluateEos(input, p);
    case Operation::timestep:
        return maximumTimestep(input, p);
    case Operation::prepare:
        return prepare(input, p);
    case Operation::finish:
        return finish(input, output, p);
    case Operation::openNewStar:
        return openNewStar(input, p);
    case Operation::pressure:
        output.value = p.gasMass * double(input.pressureRatio) * input.energy /
            (p.scaleFactor * p.scaleFactor);
        return output;
    }
    output.status = 7;
    return output;
}
}    // namespace star_formation_backend
