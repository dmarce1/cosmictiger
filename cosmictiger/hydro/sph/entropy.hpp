/* CosmicTiger - ideal-gas entropic function; BSL-1.0. */
#pragma once

#include <cmath>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace sph_entropy {

// A = P_physical / rho_physical^gamma, rho_c = a^3 rho_physical,
// E = a^2 u_physical. Therefore E = A rho_c^(gamma-1) a^(5-3 gamma)/(gamma-1).
// Treat the representable monatomic gamma exactly, including on the GPU.
CUDA_EXPORT inline double energyPerEntropy(double density, double gamma, double scaleFactor) {
    double const exponent =
        (gamma == 5.0 / 3.0 || gamma == double(float(5.0 / 3.0))) ? 0.0 : 5.0 - 3.0 * gamma;
    return std::pow(density, gamma - 1.0) * std::pow(scaleFactor, exponent) / (gamma - 1.0);
}

CUDA_EXPORT inline double storedEnergy(
    double entropy, double density, double gamma, double scaleFactor) {
    return entropy * energyPerEntropy(density, gamma, scaleFactor);
}

CUDA_EXPORT inline double fromStoredEnergy(
    double energy, double density, double gamma, double scaleFactor) {
    return energy / energyPerEntropy(density, gamma, scaleFactor);
}

}    // namespace sph_entropy
