/* CosmicTiger - GADGET-4 per-particle cooling and SF source; BSL-1.0. */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cosmictiger/hydro/sph/springel_and_hernquist/cooling.hpp>
#include <cosmictiger/particles/star_formation_state.hpp>

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
    gadget4::State chemistry;
    float velocity[3]{};
    float gravity[3]{};
    float energy{};
    float density{};
    float pressureRatio{};
    int rung{};
};
struct Output {
    StarFormationState state;
    gadget4::State chemistry;
    double value{};
    double physicalInterval{};
    double specificStarFormationRate{};
    double coldFraction{};
    double hotEnergy{};
    double effectiveEnergy{};
    float velocity[3]{};
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
    CUDA_EXPORT inline double density(Input const& input, Parameters const& p) {
        return double(input.density) * p.densityScale / p.scaleFactor / p.scaleFactor /
            p.scaleFactor;
    }
    CUDA_EXPORT inline double energy(float value, Parameters const& p) {
        return double(value) * p.energyScale / p.scaleFactor / p.scaleFactor;
    }
    CUDA_EXPORT inline bool storedEnergy(double value, Parameters const& p, float& result) {
        double const stored = value / p.energyScale * p.scaleFactor * p.scaleFactor;
        if (!(stored > 0.0) || !finite(stored) || stored > FLT_MAX) {
            return false;
        }
        result = float(stored);
        return result > 0.0f;
    }
    CUDA_EXPORT inline bool eligible(Input const& input, Parameters const& p) {
        return density(input, p) >= p.model.thresholdDensity &&
            (!p.cooling.comoving ||
                double(input.density) * p.densityScale >= p.model.overdensityThreshold);
    }
    CUDA_EXPORT inline Output initial(Input const& input) {
        Output output;
        output.state = input.state;
        output.chemistry = input.chemistry;
        output.energy = input.energy;
        output.rung = input.rung;
        for (int dim = 0; dim < 3; ++dim) {
            output.velocity[dim] = input.velocity[dim];
        }
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
CUDA_EXPORT inline Output diagnostics(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    if (!detail::eligible(input, p)) {
        return output;
    }
    springel_and_hernquist::Equilibrium phase;
    double electrons = input.chemistry.electronFraction;
    if (!springel_and_hernquist::evaluateEquilibrium(
            p.model, p.cooling, detail::density(input, p), electrons, phase)) {
        output.status = 1;
        return output;
    }
    detail::recordPhase(output, phase, detail::energy(input.energy, p));
    return output;
}
CUDA_EXPORT inline Output advance(Input const& input, Parameters const& p) {
    Output output = detail::initial(input);
    double const step =
        p.explicitTimestep >= 0.0 ? p.explicitTimestep : ::ldexp(p.baseTimestep, -input.rung);
    double const interval = p.scaleFactor * step * p.timeSeconds;
    auto& state = output.state;
    if (state.id == 0 || !detail::finite(p.explicitTimestep) || !(interval > 0.0) ||
        !detail::finite(interval) || !detail::finite(state.specificStarFormationRate) ||
        state.specificStarFormationRate < 0.0 || !detail::finite(state.elapsedPhysicalTime)) {
        output.status = 4;
        return output;
    }
    // Official run.cc calls sfr_create_star_particles BEFORE cooling_and_starformation.
    // Sample the saved previous SFR over this full selected SPH interval.
    double const probability =
        springel_and_hernquist::starFormationProbability(state.specificStarFormationRate, interval);
    output.convert = detail::uniform(state.id, state.eventCounter, 0, p.seed) < probability;
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
        state.formationTime = p.cooling.comoving ? p.scaleFactor : state.elapsedPhysicalTime;
        output.value = p.gasMass * double(input.energy) / p.scaleFactor;
        return output;
    }
    double const rho = detail::density(input, p);
    double const oldEnergy = detail::energy(input.energy, p);
    double newEnergy = oldEnergy;
    double electrons = input.chemistry.electronFraction;
    if (detail::eligible(input, p)) {
        springel_and_hernquist::Equilibrium phase;
        if (!springel_and_hernquist::evaluateEquilibrium(
                p.model, p.cooling, rho, electrons, phase) ||
            !springel_and_hernquist::clampSourceTime(p.model.parameters, interval, phase)) {
            output.status = 1;
            return output;
        }
        // The official dense branch persists Ne from the hot cooling-time
        // lookup. Its subsequent DoCooling uses a local Ne, not that field.
        double coolingElectrons = electrons;
        if (oldEnergy > phase.effectiveEnergy) {
            if (gadget4::doCooling(oldEnergy, rho, interval, p.cooling, coolingElectrons,
                    newEnergy) != gadget4::Status::Success) {
                output.status = 2;
                return output;
            }
            if (newEnergy < phase.effectiveEnergy) {
                newEnergy = phase.effectiveEnergy;
            }
        } else {
            newEnergy = phase.effectiveEnergy +
                (oldEnergy - phase.effectiveEnergy) * ::exp(-interval / phase.relaxationTime);
        }
        if (newEnergy < p.cooling.minimumSpecificEnergy) {
            newEnergy = p.cooling.minimumSpecificEnergy;
        }
        detail::recordPhase(output, phase, newEnergy);
    } else {
        double const initialEnergy = oldEnergy < p.cooling.minimumSpecificEnergy ?
            p.cooling.minimumSpecificEnergy :
            oldEnergy;
        if (gadget4::doCooling(initialEnergy, rho, interval, p.cooling, electrons, newEnergy) !=
            gadget4::Status::Success) {
            output.status = 2;
            return output;
        }
        if (newEnergy < p.cooling.minimumSpecificEnergy) {
            newEnergy = p.cooling.minimumSpecificEnergy;
        }
    }
    if (!detail::storedEnergy(newEnergy, p, output.energy) || !detail::finite(electrons) ||
        electrons < 0.0 || electrons > FLT_MAX) {
        output.status = 2;
        return output;
    }
    output.chemistry.electronFraction = float(electrons);
    state.specificStarFormationRate = output.specificStarFormationRate;
    output.value = -p.gasMass * (double(output.energy) - input.energy) / p.scaleFactor;
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
    case Operation::diagnostics:
        return diagnostics(input, p);
    case Operation::advance:
        return advance(input, p);
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
