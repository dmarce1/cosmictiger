/* CosmicTiger - GADGET-4 cooling adapter for its effective EOS; BSL-1.0. */
#pragma once

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/sph/springel_and_hernquist/equations.hpp>

namespace springel_and_hernquist {

// These are the constants used in the pinned GADGET-4 source, not updated CODATA.
inline constexpr double kBoltzmann = gadget4::kBoltzmann;
inline constexpr double kProtonMass = gadget4::kProtonMass;
inline constexpr double kHydrogenMassFraction = 0.76;
inline constexpr double kSecondsPerYear = 3.15576e7;
CUDA_EXPORT inline Parameters defaultParameters() {
    Parameters result;
    double const neutralWeight = 4.0 / (1.0 + 3.0 * kHydrogenMassFraction);
    double const ionizedWeight = 4.0 / (8.0 - 5.0 * (1.0 - kHydrogenMassFraction));
    result.coldEnergy =
        1.0 / neutralWeight / (result.gamma - 1.0) * (kBoltzmann / kProtonMass) * 1000.0;
    result.supernovaEnergy =
        1.0 / ionizedWeight / (result.gamma - 1.0) * (kBoltzmann / kProtonMass) * 1.0e8;
    result.warmEnergy = result.supernovaEnergy * 1.0e-4;
    return result;
}
CUDA_EXPORT inline bool makeModel(Parameters const& parameters, gadget4::Parameters cooling,
    double calibrationDensity, Model& result) {
    if (!(calibrationDensity > 0.0) || !detail::finite(calibrationDensity) ||
        !(parameters.evaporationFactor > 0.0)) {
        return false;
    }
    cooling.redshift = 0.0;
    double const energy = parameters.supernovaEnergy / parameters.evaporationFactor;
    double electrons = 1.0;
    double time;
    if (gadget4::coolingTime(energy, calibrationDensity, cooling, electrons, time) !=
            gadget4::Status::Success ||
        !(time > 0.0)) {
        return false;
    }
    return calibrate(parameters, energy / time / calibrationDensity, result);
}
CUDA_EXPORT inline bool evaluateEquilibrium(Model const& model, gadget4::Parameters const& cooling,
    double density, double& electrons, Equilibrium& result) {
    double candidateElectrons = electrons;
    double time;
    if (gadget4::coolingTime(hotEnergy(model, density), density, cooling, candidateElectrons,
            time) != gadget4::Status::Success ||
        !equilibrium(model, density, time, result)) {
        return false;
    }
    electrons = candidateElectrons;
    return true;
}
}    // namespace springel_and_hernquist
