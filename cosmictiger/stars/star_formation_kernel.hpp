/* CosmicTiger - GADGET-4 per-particle cooling and SF source; BSL-1.0. */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/stars/springel_and_hernquist/cooling.hpp>
#include <cosmictiger/stars/star_formation_state.hpp>

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
    uint64_t seed{};
};
struct Input {
    StarFormationState state;
    float electronFraction{1.0f};
    Vector3<float> velocity{};
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
enum class Operation { eos, diagnostics, advance, pressure };
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
    double const Δt = parameters.scaleFactor * step * parameters.timeSeconds;
    auto& state = output.state;
    if (state.id == 0 || !detail::finite(parameters.explicitTimestep) || !(Δt > 0.0) ||
        !detail::finite(Δt) || !detail::finite(state.specificStarFormationRate) ||
        state.specificStarFormationRate < 0.0 || !detail::finite(state.elapsedPhysicalTime)) {
        output.status = 4;
        return output;
    }
    // Official run.cc calls sfr_create_star_particles BEFORE cooling_and_starformation.
    // Sample the saved previous SFR over this full selected SPH interval.
    double const probability =
        springel_and_hernquist::starFormationProbability(state.specificStarFormationRate, Δt);
    output.convert =
        detail::uniform(state.id, state.eventCounter, 0, parameters.seed) < probability;
    ++state.eventCounter;
    state.elapsedPhysicalTime += Δt;
    if (!detail::finite(state.elapsedPhysicalTime)) {
        output.status = 4;
        return output;
    }
    output.physicalInterval = Δt;
    if (output.convert) {
        output.energy = 0.0f;
        state.specificStarFormationRate = 0.0;
        state.formationTime =
            parameters.cooling.comoving ? parameters.scaleFactor : state.elapsedPhysicalTime;
        output.value = parameters.gasMass * double(input.energy) / parameters.scaleFactor;
        return output;
    }
    double const physicalDensity = detail::density(input, parameters);
    double const u_0 = detail::energy(input.energy, parameters);
    double u_Δt = u_0;
    double electrons = input.electronFraction;
    if (detail::eligible(input, parameters)) {
        springel_and_hernquist::Equilibrium phase;
        if (!springel_and_hernquist::evaluateEquilibrium(
                parameters.model, parameters.cooling, physicalDensity, electrons, phase) ||
            !springel_and_hernquist::clampSourceTime(parameters.model.parameters, Δt, phase)) {
            output.status = 1;
            return output;
        }
        // The official dense branch persists Ne from the hot cooling-time
        // lookup. Its subsequent DoCooling uses a local Ne, not that field.
        double coolingElectrons = electrons;
        if (u_0 > phase.effectiveEnergy) {
            if (gadget4::doCooling(u_0, physicalDensity, Δt, parameters.cooling, coolingElectrons,
                    u_Δt) != gadget4::Status::Success) {
                output.status = 2;
                return output;
            }
            if (u_Δt < phase.effectiveEnergy) {
                u_Δt = phase.effectiveEnergy;
            }
        } else {
            // clang-format off
            /*
             * Springel & Hernquist (2003), MNRAS 339, 289, equation (12),
             * https://arxiv.org/abs/astro-ph/0206393: τₕ = t⋆ ρₕ/[β(A+1)ρ𞁞].
             * Derived frozen-equilibrium relaxation used by the GADGET-4 source:
             *   u(Δt) = u_eff + [u(0) − u_eff] exp(−Δt/τₕ).
             * Symbol directory:
             *   u(0) → old energy; u_eff → effective energy; Δt → interval
             *   τₕ → relaxation time; u(Δt) → new energy.
             *   Code names: u_0 = u(0); u_Δt = u(Δt).
             */
            // clang-format on
            u_Δt = phase.effectiveEnergy +
                (u_0 - phase.effectiveEnergy) * ::exp(-Δt / phase.relaxationTime);
        }
        if (u_Δt < parameters.cooling.minimumSpecificEnergy) {
            u_Δt = parameters.cooling.minimumSpecificEnergy;
        }
        detail::recordPhase(output, phase, u_Δt);
    } else {
        double const initialEnergy = u_0 < parameters.cooling.minimumSpecificEnergy ?
            parameters.cooling.minimumSpecificEnergy :
            u_0;
        if (gadget4::doCooling(initialEnergy, physicalDensity, Δt, parameters.cooling, electrons,
                u_Δt) != gadget4::Status::Success) {
            output.status = 2;
            return output;
        }
        if (u_Δt < parameters.cooling.minimumSpecificEnergy) {
            u_Δt = parameters.cooling.minimumSpecificEnergy;
        }
    }
    if (!detail::storedEnergy(u_Δt, parameters, output.energy) || !detail::finite(electrons) ||
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
CUDA_EXPORT inline Output evaluate(
    Operation operation, Input const& input, Output output, Parameters const& parameters) {
    switch (operation) {
    case Operation::eos:
        return evaluateEos(input, parameters);
    case Operation::diagnostics:
        return diagnostics(input, parameters);
    case Operation::advance:
        return advance(input, parameters);
    case Operation::pressure:
        output.value = parameters.gasMass * double(input.pressureRatio) * input.energy /
            (parameters.scaleFactor * parameters.scaleFactor);
        return output;
    }
    output.status = 7;
    return output;
}
}    // namespace star_formation_backend
