/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace sph_entropy {

// Cosmological scale-factor conversions below deliberately retain their 3D
// convention. Lower-dimensional SPH is noncosmological (a = 1).

// A = P_physical / rho_physical^adiabaticIndex, rho_c = a^3 rho_physical,
// E = a^2 u_physical. Therefore E = A rho_c^(adiabaticIndex-1) a^(5-3
// adiabaticIndex)/(adiabaticIndex-1). Treat the representable monatomic adiabaticIndex exactly,
// including on the GPU.
// clang-format off
/*
 * Derived ideal-gas conversion from A = P_phys/ρ_phys^γ,
 * ρ_c = a³ ρ_phys, E = a² u_phys and P_phys = (γ−1)ρ_phys u_phys.
 *   E/A = ρ_c^(γ−1) a^(5−3γ)/(γ−1).
 * This is a local derivation, not a numbered source equation.
 * Symbol directory:
 *   ρ_c → density; γ → adiabaticIndex; a → scaleFactor; 5−3γ → exponent; E/A → return value.
 */
// clang-format on
CUDA_EXPORT inline float energyPerEntropy(float density, float adiabaticIndex, float scaleFactor) {
    float const exponent = (adiabaticIndex == 5.0f / 3.0f) ? 0.0f : 5.0f - 3.0f * adiabaticIndex;
    return std::pow(density, adiabaticIndex - 1.0f) * std::pow(scaleFactor, exponent) /
        (adiabaticIndex - 1.0f);
}

// clang-format off
/*
 * Derived ideal-gas conversion from A = P_phys/ρ_phys^γ,
 * ρ_c = a³ ρ_phys, E = a² u_phys and P_phys = (γ−1)ρ_phys u_phys.
 *   E = A ρ_c^(γ−1) a^(5−3γ)/(γ−1).
 * This is a local derivation, not a numbered source equation.
 * Symbol directory:
 *   A → entropy; ρ_c → density; γ → adiabaticIndex; a → scaleFactor; E → return value.
 */
// clang-format on
CUDA_EXPORT inline float storedEnergy(
    float entropy, float density, float adiabaticIndex, float scaleFactor) {
    return entropy * energyPerEntropy(density, adiabaticIndex, scaleFactor);
}

// clang-format off
/*
 * Derived ideal-gas conversion from A = P_phys/ρ_phys^γ,
 * ρ_c = a³ ρ_phys, E = a² u_phys and P_phys = (γ−1)ρ_phys u_phys.
 *   A = (γ−1) E ρ_c^(1−γ) a^(3γ−5).
 * This is a local derivation, not a numbered source equation.
 * Symbol directory:
 *   E → energy; ρ_c → density; γ → adiabaticIndex; a → scaleFactor; A → return value.
 */
// clang-format on
CUDA_EXPORT inline float fromStoredEnergy(
    float energy, float density, float adiabaticIndex, float scaleFactor) {
    return energy / energyPerEntropy(density, adiabaticIndex, scaleFactor);
}

struct PreparedThermalState {
    float entropy;
    float storedEnergy;
};

namespace detail {
    inline bool finiteThermalValue(float value) {
        return (std::bit_cast<uint32_t>(value) & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
    }
    inline float thermalStorageValue(float value) {
        // Sources must remain usable under the application's fast-math build;
        // subnormal storage is not reliable when the processor flushes it to zero.
        if (!finiteThermalValue(value) || value < std::numeric_limits<float>::min() ||
            value > std::numeric_limits<float>::max()) {
            throw std::range_error(
                "SPH source returned unrepresentable positive thermodynamic state");
        }
        return static_cast<float>(value);
    }
}    // namespace detail

// Stage the canonical float A before changing any particle. Reconstruct E from
// that exact representation so the source ledger matches the committed state.
inline PreparedThermalState prepareStoredEnergy(
    float energy, float density, float adiabaticIndex, float scaleFactor) {
    if (!detail::finiteThermalValue(energy) || !(energy > 0.0f) ||
        !detail::finiteThermalValue(density) || !(density > 0.0f) ||
        !detail::finiteThermalValue(adiabaticIndex) || !(adiabaticIndex > 1.0f) ||
        !detail::finiteThermalValue(scaleFactor) || !(scaleFactor > 0.0f)) {
        throw std::invalid_argument(
            "SPH source requires finite positive energy, density and scale");
    }
    float const factor = energyPerEntropy(density, adiabaticIndex, scaleFactor);
    if (!detail::finiteThermalValue(factor) || !(factor > 0.0f)) {
        throw std::range_error("SPH source has an invalid entropy conversion factor");
    }
    PreparedThermalState result;
    result.entropy = detail::thermalStorageValue(energy / factor);
    result.storedEnergy = detail::thermalStorageValue(result.entropy * factor);
    return result;
}

}    // namespace sph_entropy
