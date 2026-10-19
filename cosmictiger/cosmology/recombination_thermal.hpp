#pragma once

#include <cmath>

#include <cosmictiger/math/constants.hpp>

namespace recombination {

// Dimensionless Compton temperature coupling, Gamma_C / H. Number densities
// and H are physical CGS quantities; omegaPhoton and omegaBaryon are present-day
// density parameters. Only baryons contribute to the gas heat capacity.
// Scott & Moss (2009), Eqs. (1)-(2), https://arxiv.org/abs/0902.3438:
// Gamma_C = 8 sigma_T a_R T_rad^4 / (3 m_e c) * xe / (1 + fHe + xe).
inline double comptonTemperatureCoupling(double scaleFactor, double omegaPhoton, double omegaBaryon,
    double hydrogenNumberDensity, double heliumNumberDensity, double electronNumberDensity,
    double hubbleRate) {
    double const meanParticleMass = (hydrogenNumberDensity + 4.0 * heliumNumberDensity) *
        constants::kMh / (hydrogenNumberDensity + heliumNumberDensity + electronNumberDensity);
    return (8.0 / 3.0) * meanParticleMass / constants::kMe * constants::kC * omegaPhoton /
        (scaleFactor * omegaBaryon) * constants::kSigmaT * electronNumberDensity / hubbleRate;
}

// The existing two-stage implicit background step for
// dT_gas/dln(a) = -2 T_gas + (Gamma_C/H) (T_rad - T_gas).
// Radiation temperature and coupling are held fixed during this nonnegative step.
inline double advanceGasTemperature(double gasTemperature, double radiationTemperature,
    double logScaleStep, double comptonCoupling) {
    if (logScaleStep == 0.0) {
        return gasTemperature;
    }
    double const gamma = 1.0 - 1.0 / std::sqrt(2.0);
    double const firstDerivative =
        ((gasTemperature + gamma * logScaleStep * comptonCoupling * radiationTemperature) /
                (1.0 + gamma * logScaleStep * (2.0 + comptonCoupling)) -
            gasTemperature) /
        (gamma * logScaleStep);
    double const stageTemperature =
        gasTemperature + (1.0 - 2.0 * gamma) * firstDerivative * logScaleStep;
    double const secondDerivative =
        ((stageTemperature + gamma * logScaleStep * comptonCoupling * radiationTemperature) /
                (1.0 + gamma * logScaleStep * (2.0 + comptonCoupling)) -
            stageTemperature) /
        (gamma * logScaleStep);
    return gasTemperature + 0.5 * (firstDerivative + secondDerivative) * logScaleStep;
}

}    // namespace recombination
