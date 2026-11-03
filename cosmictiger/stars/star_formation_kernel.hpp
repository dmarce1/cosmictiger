/* CosmicTiger - GADGET-4 per-particle cooling and SF source; BSL-1.0. */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/stars/star_formation_state.hpp>
#include <cosmictiger/stars/springel_and_hernquist/cooling.hpp>

namespace star_formation_backend {

struct Parameters {
    springel_and_hernquist::Model model;
    gadget4::Parameters cooling;
    double densityScale{};
    double energyScale{};
    double scaleFactor{};
    double timeSeconds{};
    double gasMass{};
    double scaleDerivative{};
    double baseTimestep{};
    double explicitTimestep{-1.0};
    float effectiveRadius{};
    float timestepBase{};
    float timestepCap{};
    float timestepFactor{};
    float kickSign{1.0f};
    int rungLimit{};
    uint64_t seed{};
    int minimumRung{};
};
struct Input {
    StarFormationState state;
    float electronFraction{1.0f};
    Vector3<float> velocity{};
    Vector3<float> gravity{};
    float energy{};
    float density{};
    float pressureRatio{};
    int rung{};
};
struct Output {
    StarFormationState state;
    float electronFraction{1.0f};
    double value{};
    double physicalInterval{};
    double specificStarFormationRate{};
    double coldFraction{};
    double hotEnergy{};
    double effectiveEnergy{};
    Vector3<float> velocity{};
    float energy{};
    float pressureRatio{};
    float signalIndex{};
    int rung{};
    int status{};
    bool starForming{};
    bool convert{};
};
enum class Operation { eos, diagnostics, advance, pressure, openNewStar };
inline char const* statusName(int status) {
    switch (status) {
    case 0:
        return "success";
    case 1:
        return "equilibrium_failed";
    case 2:
        return "cooling_or_energy_storage_failed";
    case 3:
        return "invalid_bulk_energy";
    case 4:
        return "invalid_source_state_or_interval";
    case 7:
        return "unknown_operation";
    case 8:
        return "new_star_timestep_out_of_range";
    default:
        return "unknown_status";
    }
}

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
    CUDA_EXPORT inline bool finite(double value) {
        return springel_and_hernquist::detail::finite(value);
    }
    CUDA_EXPORT inline double density(Input const& input, Parameters const& parameters) {
        return double(input.density) * parameters.densityScale / parameters.scaleFactor /
            parameters.scaleFactor / parameters.scaleFactor;
    }
    CUDA_EXPORT inline double energy(float value, Parameters const& parameters) {
        return double(value) * parameters.energyScale / parameters.scaleFactor /
            parameters.scaleFactor;
    }
    CUDA_EXPORT inline bool storedEnergy(
        double value, Parameters const& parameters, float& result) {
        double const stored =
            value / parameters.energyScale * parameters.scaleFactor * parameters.scaleFactor;
        if (!(stored > 0.0) || !finite(stored) || stored > FLT_MAX) {
            return false;
        }
        result = float(stored);
        return result > 0.0f;
    }
    CUDA_EXPORT inline bool eligible(Input const& input, Parameters const& parameters) {
        return density(input, parameters) >= parameters.model.thresholdDensity &&
            (!parameters.cooling.comoving ||
                double(input.density) * parameters.densityScale >=
                    parameters.model.overdensityThreshold);
    }
    CUDA_EXPORT inline Output initial(Input const& input) {
        Output output;
        output.state = input.state;
        output.electronFraction = input.electronFraction;
        output.energy = input.energy;
        output.rung = input.rung;
        output.velocity = input.velocity;
        return output;
    }
    CUDA_EXPORT inline void recordPhase(
        Output& output, springel_and_hernquist::Equilibrium const& phase, double energy) {
        output.starForming = true;
        output.coldFraction = phase.coldFraction;
        output.hotEnergy = phase.hotEnergy;
        output.effectiveEnergy = phase.effectiveEnergy;
        output.specificStarFormationRate =
            energy > 1.01 * phase.effectiveEnergy ? 0.0 : phase.specificStarFormationRate;
    }
}    // namespace detail
CUDA_EXPORT inline Output evaluateEos(Input const& input, Parameters const&) {
    Output output = detail::initial(input);
    if (!(input.energy > 0.0f) || !detail::finite(input.energy)) {
        output.status = 3;
        return output;
    }
    // GADGET-4 hydrodynamics uses the carried bulk thermal energy and gamma,
    // not dP_eff/drho along an equilibrium curve.
    output.pressureRatio = float(2.0 / 3.0);
    output.signalIndex = float(5.0 / 3.0);
    return output;
}
CUDA_EXPORT inline Output diagnostics(Input const& input, Parameters const& parameters) {
    Output output = detail::initial(input);
    if (!detail::eligible(input, parameters)) {
        return output;
    }
    springel_and_hernquist::Equilibrium phase;
    double electrons = input.electronFraction;
    if (!springel_and_hernquist::evaluateEquilibrium(parameters.model, parameters.cooling,
            detail::density(input, parameters), electrons, phase)) {
        output.status = 1;
        return output;
    }
    detail::recordPhase(output, phase, detail::energy(input.energy, parameters));
    return output;
}
CUDA_EXPORT inline Output advance(Input const& input, Parameters const& parameters) {
    Output output = detail::initial(input);
    double const step = parameters.explicitTimestep >= 0.0 ?
        parameters.explicitTimestep :
        ::ldexp(parameters.baseTimestep, -input.rung);
    double const interval = parameters.scaleFactor * step * parameters.timeSeconds;
    auto& state = output.state;
    if (state.id == 0 || !detail::finite(parameters.explicitTimestep) || !(interval > 0.0) ||
        !detail::finite(interval) || !detail::finite(state.specificStarFormationRate) ||
        state.specificStarFormationRate < 0.0 || !detail::finite(state.elapsedPhysicalTime)) {
        output.status = 4;
        return output;
    }
    // Official run.cc calls sfr_create_star_particles BEFORE cooling_and_starformation.
    // Sample the saved previous SFR over this full selected SPH interval.
    double const probability =
        springel_and_hernquist::starFormationProbability(state.specificStarFormationRate, interval);
    output.convert =
        detail::uniform(state.id, state.eventCounter, 0, parameters.seed) < probability;
    ++state.eventCounter;
    state.elapsedPhysicalTime += interval;
    if (!detail::finite(state.elapsedPhysicalTime)) {
        output.status = 4;
        return output;
    }
    output.physicalInterval = interval;
    if (output.convert) {
        output.energy = 0.0f;
        state.specificStarFormationRate = 0.0;
        state.formationTime =
            parameters.cooling.comoving ? parameters.scaleFactor : state.elapsedPhysicalTime;
        output.value = parameters.gasMass * double(input.energy) / parameters.scaleFactor;
        return output;
    }
    double const physicalDensity = detail::density(input, parameters);
    double const oldEnergy = detail::energy(input.energy, parameters);
    double newEnergy = oldEnergy;
    double electrons = input.electronFraction;
    if (detail::eligible(input, parameters)) {
        springel_and_hernquist::Equilibrium phase;
        if (!springel_and_hernquist::evaluateEquilibrium(
                parameters.model, parameters.cooling, physicalDensity, electrons, phase) ||
            !springel_and_hernquist::clampSourceTime(
                parameters.model.parameters, interval, phase)) {
            output.status = 1;
            return output;
        }
        // The official dense branch persists Ne from the hot cooling-time
        // lookup. Its subsequent DoCooling uses a local Ne, not that field.
        double coolingElectrons = electrons;
        if (oldEnergy > phase.effectiveEnergy) {
            if (gadget4::doCooling(oldEnergy, physicalDensity, interval, parameters.cooling,
                    coolingElectrons, newEnergy) != gadget4::Status::Success) {
                output.status = 2;
                return output;
            }
            if (newEnergy < phase.effectiveEnergy) {
                newEnergy = phase.effectiveEnergy;
            }
        } else {
            // clang-format off
            /*
             * Springel & Hernquist (2003), MNRAS 339, 289, equation (12),
             * https://arxiv.org/abs/astro-ph/0206393: τ_h = t⋆ ρ_h/[β(A+1)ρ_c].
             * Derived frozen-equilibrium relaxation used by the GADGET-4 source:
             *   u(Δt) = u_eff + [u(0) − u_eff] exp(−Δt/τ_h).
             * Symbol directory:
             *   u(0) → oldEnergy; u_eff → phase.effectiveEnergy; Δt → interval
             *   τ_h → phase.relaxationTime; u(Δt) → newEnergy.
             */
            // clang-format on
            newEnergy = phase.effectiveEnergy +
                (oldEnergy - phase.effectiveEnergy) * ::exp(-interval / phase.relaxationTime);
        }
        if (newEnergy < parameters.cooling.minimumSpecificEnergy) {
            newEnergy = parameters.cooling.minimumSpecificEnergy;
        }
        detail::recordPhase(output, phase, newEnergy);
    } else {
        double const initialEnergy = oldEnergy < parameters.cooling.minimumSpecificEnergy ?
            parameters.cooling.minimumSpecificEnergy :
            oldEnergy;
        if (gadget4::doCooling(initialEnergy, physicalDensity, interval, parameters.cooling,
                electrons, newEnergy) != gadget4::Status::Success) {
            output.status = 2;
            return output;
        }
        if (newEnergy < parameters.cooling.minimumSpecificEnergy) {
            newEnergy = parameters.cooling.minimumSpecificEnergy;
        }
    }
    if (!detail::storedEnergy(newEnergy, parameters, output.energy) || !detail::finite(electrons) ||
        electrons < 0.0 || electrons > FLT_MAX) {
        output.status = 2;
        return output;
    }
    output.electronFraction = float(electrons);
    state.specificStarFormationRate = output.specificStarFormationRate;
    output.value =
        -parameters.gasMass * (double(output.energy) - input.energy) / parameters.scaleFactor;
    return output;
}
// A converted particle missed the collisionless gravity timestep proposal.
// Apply that criterion and its first opening kick before the next drift.
CUDA_EXPORT inline Output openNewStar(Input const& input, Parameters const& parameters) {
    Output output = detail::initial(input);
    float accelerationSquared = 1.0e-35f;
    for (int dim = 0; dim < 3; ++dim) {
        accelerationSquared += input.gravity[dim] * input.gravity[dim];
    }
    float const gravityTimestep = parameters.timestepFactor *
        ::sqrtf(parameters.effectiveRadius / ::sqrtf(accelerationSquared));
    float const boundedTimestep =
        parameters.timestepBase < gravityTimestep ? parameters.timestepBase : gravityTimestep;
    float const timestep =
        boundedTimestep < parameters.timestepCap ? boundedTimestep : parameters.timestepCap;
    float const requestedRung = ::ceilf(::log2f(parameters.timestepBase / timestep));
    if (!(timestep > 0.0f) || !detail::finite(timestep) || !detail::finite(requestedRung) ||
        requestedRung >= parameters.rungLimit) {
        output.status = 8;
        return output;
    }
    int rung = int(requestedRung);
    if (rung < input.rung - 1) {
        rung = input.rung - 1;
    }
    if (rung < parameters.minimumRung) {
        rung = parameters.minimumRung;
    }
    if (rung < 1) {
        rung = 1;
    }
    output.rung = rung;
    double const half = parameters.kickSign * ::ldexp(0.5 * parameters.baseTimestep, -rung);
    for (int dim = 0; dim < 3; ++dim) {
        // Explicit FMA matches the host kick even with contraction disabled.
        output.velocity[dim] =
            float(::fma(double(input.gravity[dim]), half, double(input.velocity[dim])));
    }
    return output;
}
CUDA_EXPORT inline Output evaluate(
    Operation operation, Input const& input, Output output, Parameters const& parameters) {
    switch (operation) {
    case Operation::eos:
        return evaluateEos(input, parameters);
    case Operation::diagnostics:
        return diagnostics(input, parameters);
    case Operation::advance:
        return advance(input, parameters);
    case Operation::openNewStar:
        return openNewStar(input, parameters);
    case Operation::pressure:
        output.value = parameters.gasMass * double(input.pressureRatio) * input.energy /
            (parameters.scaleFactor * parameters.scaleFactor);
        return output;
    }
    output.status = 7;
    return output;
}
}    // namespace star_formation_backend
