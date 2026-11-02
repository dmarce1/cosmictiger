/* CosmicTiger - immutable hierarchical SPH particle anchor; BSL-1.0. */

#pragma once

#include <array>
#include <cstdint>

namespace sph_hierarchical {

struct SourceState {
    uint64_t id = 0;
    std::array<double, 3> position{};
    std::array<double, 3> driftVelocity{};
    std::array<double, 3> synchronizedVelocity{};
    std::array<double, 3> hydroAcceleration{};
    double internalEnergy = 1.0;
    double internalEnergyRate = 0.0;
    double density = 1.0;
    double logDensityRate = 0.0;
    double physicalDivergence = 0.0;
    double supportRadius = 0.1;
    double minimumSupportRadius = 0.001;
    double maximumSupportRadius = 0.5;
    double gradHCorrection = 0.0;
    double gradHCorrectionRate = 0.0;
    double balsaraFactor = 1.0;
    double viscosityAlpha = 1.0;
    double pressureEnergyRatio = 2.0 / 3.0;
    double adiabaticIndex = 5.0 / 3.0;
    int rung = 0;
    double epoch = 0.0;
    double validUntil = 0.0;
    double driftScale = 1.0;

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & id;
        archive & position;
        archive & driftVelocity;
        archive & synchronizedVelocity;
        archive & hydroAcceleration;
        archive & internalEnergy;
        archive & internalEnergyRate;
        archive & density;
        archive & logDensityRate;
        archive & physicalDivergence;
        archive & supportRadius;
        archive & minimumSupportRadius;
        archive & maximumSupportRadius;
        archive & gradHCorrection;
        archive & gradHCorrectionRate;
        archive & balsaraFactor;
        archive & viscosityAlpha;
        archive & pressureEnergyRatio;
        archive & adiabaticIndex;
        archive & rung;
        archive & epoch;
        archive & validUntil;
        archive & driftScale;
    }
};

}    // namespace sph_hierarchical
