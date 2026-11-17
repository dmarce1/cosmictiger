/* CosmicTiger - shared CPU/CUDA conservative pair regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

inline void pairKickRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void pairKickClose(double actual, double expected, char const* message) {
    pairKickRequire(
        std::abs(actual - expected) <= 2.0e-6 * std::max(std::abs(expected), 1.0e-20), message);
}

// A compensated pair sum represents s − c (cosmictiger/sph/pair_kick.hpp).
inline double compensatedKickSum(SphCompensatedSum const& sum) {
    return double(float(sum)) - double(sum.correction());
}

template <class Fixture>
void testConservativePairKick(Fixture& fixture, bool morrisAndMonaghan) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    parameters.midpointHydro = true;
    parameters.minimumRung = 0;
    parameters.minimumProposedRung = 0;
    parameters.baseTimestep = 1.0e-5f;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.03f;
    parameters.hydroKickTimestep = 0.0005;
    data.selfGravityEnabled = false;
    if (morrisAndMonaghan) {
        fixture.enableVariableViscosity();
    }
    auto* identities = fixture.enablePairKickOutputs();
    identities[fixture.sourceIndex(0)] = {1, 37};
    identities[fixture.sourceIndex(1)] = {0, 92};
    std::array<std::array<float, 3>, 2> const velocities{
        {{0.2f, 0.07f, 0.04f}, {-0.1f, -0.03f, -0.06f}}};
    for (int index = 0; index < 2; ++index) {
        int const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = 0.47 + 0.06 * index;
        data.sourcePositionY[source] = 0.49 + 0.03 * index;
        data.sourcePositionZ[source] = 0.48 + 0.02 * index;
        data.sourceVelocityX[source] = velocities[index][0];
        data.sourceVelocityY[source] = velocities[index][1];
        data.sourceVelocityZ[source] = velocities[index][2];
        data.sourceDensity[source] = 1.3f + 0.8f * index;
        data.sourceInternalEnergy[source] = 0.8f + 0.6f * index;
        data.sourceSmoothingLength[source] = 0.2f + 0.05f * index;
        data.sourceGradHCorrection[source] = 0.0f;    // Fixed-support pair.
        if (morrisAndMonaghan) {
            data.sourceBulkViscosityCoefficient[source] = 0.4f + 0.5f * index;
        }
    }
    // The complete equal-rung evaluation, in which the lower identity owns the one pair, is
    // the reference for both sides' impulse and heating, including unequal particle alphas.
    // The fixture clears the pair updates before every pass.
    fixture.hydro(0);
    std::array<std::array<double, kDimensions>, 2> impulse;
    std::array<float, 2> entropyRate;
    for (int index = 0; index < 2; ++index) {
        auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            impulse[index][dimension] = compensatedKickSum(update.velocity[dimension]);
        }
        entropyRate[index] = data.sinkThermalRate[fixture.sinkIndex(index)];
        pairKickRequire(entropyRate[index] > 0.0f, "pair fixture missed viscous heating");
        pairKickRequire(impulse[index][0] != 0.0, "pair fixture missed the pair force");
    }
    data.sourceRungs[fixture.sourceIndex(0)] = data.sinkParticleRungs[fixture.particleIndex(0)] = 3;
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 1;
    auto requireZero = [&] {
        for (int index = 0; index < 2; ++index) {
            auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
            pairKickRequire(update.thermal == 0.0, "inactive pair generated entropy");
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                pairKickRequire(update.velocity[dimension] == 0.0,
                    "pair was integrated on its coarser endpoint's clock");
            }
        }
    };
    parameters.minimumRung = parameters.maximumRung = 1;
    fixture.hydro(1);
    requireZero();
    parameters.minimumRung = parameters.maximumRung = 3;
    fixture.hydro(1);
    for (int index = 0; index < 2; ++index) {
        auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
        pairKickClose(update.thermal, parameters.hydroKickTimestep * entropyRate[index],
            "pair kick lost an endpoint's viscous heating");
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            pairKickClose(compensatedKickSum(update.velocity[dimension]), impulse[index][dimension],
                "fine-owned pair impulse differs from the complete pair force");
        }
    }
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        double const firstIncrement =
            data.sourcePairUpdates[fixture.sourceIndex(0)].velocity[dimension];
        double const secondIncrement =
            data.sourcePairUpdates[fixture.sourceIndex(1)].velocity[dimension];
        pairKickRequire(firstIncrement == -secondIncrement,
            "pair endpoints did not receive the identical opposite impulse");
        float const firstVelocity = float(double(velocities[0][dimension]) + firstIncrement);
        float const secondVelocity = float(double(velocities[1][dimension]) + secondIncrement);
        double const momentumChange = double(data.gasMass) *
            (double(firstVelocity) + double(secondVelocity) - double(velocities[0][dimension]) -
                double(velocities[1][dimension]));
        double const roundingBound = 2.0 * std::numeric_limits<float>::epsilon() * data.gasMass *
            (std::abs(double(velocities[0][dimension])) +
                std::abs(double(velocities[1][dimension])) + std::abs(firstIncrement) +
                std::abs(secondIncrement));
        pairKickRequire(std::abs(momentumChange) <= roundingBound,
            "float velocity commits exceeded their momentum roundoff bound");
    }
    // Equal-rung ownership must integrate exactly once across ranks, and must
    // also use the local index to break ties within one rank.
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 3;
    for (int ownershipCase = 0; ownershipCase < 2; ++ownershipCase) {
        if (ownershipCase == 1) {
            identities[fixture.sourceIndex(0)] = {0, 37};
            identities[fixture.sourceIndex(1)] = {0, 92};
        }
        fixture.hydro(1);
        for (int index = 0; index < 2; ++index) {
            auto const& update = data.sourcePairUpdates[fixture.sourceIndex(index)];
            pairKickClose(update.thermal, parameters.hydroKickTimestep * entropyRate[index],
                "equal-rung pair heating was counted twice or lost");
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                pairKickClose(compensatedKickSum(update.velocity[dimension]),
                    impulse[index][dimension], "equal-rung pair force was counted twice or lost");
            }
        }
    }
    // Production does not need the persistent entropy-rate output.
    data.sinkThermalRate = nullptr;
    parameters.hydroKickTimestep = 0.0;
    fixture.hydro(1);
    requireZero();
    parameters.hydroKickTimestep = 0.0005;
    parameters.minimumRung = parameters.maximumRung = 1;
    data.sourceRungs[fixture.sourceIndex(1)] = data.sinkParticleRungs[fixture.particleIndex(1)] = 1;
    parameters.uniformAccelerationY = 0.4f;
    parameters.velocityDampingRate = 0.125f;
    fixture.hydro(1);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        pairKickRequire(data.sourcePairUpdates[fixture.sourceIndex(0)].velocity[dimension] == 0.0,
            "external force touched an inactive target");
        float const externalAcceleration = (dimension == 1 ? 0.4f : 0.0f) -
            parameters.velocityDampingRate * velocities[1][dimension];
        pairKickClose(data.sourcePairUpdates[fixture.sourceIndex(1)].velocity[dimension],
            parameters.hydroKickTimestep * externalAcceleration,
            "external force was omitted or counted more than once");
    }
}

// Both formulations must give the same force and dissipative exchange. Direct energy
// evolution additionally publishes each endpoint's own pressure work, including the
// inactive endpoint of a fine-owned pair.
template <class Fixture>
void testDensityEnergyPair(Fixture& fixture) {
    constexpr size_t kParticles = 2;
    constexpr int kDissipationCases = 4;
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    fixture.enableKickObservation();
    parameters.minimumRung = 0;
    parameters.maximumRung = kMaxRung - 1;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.0f;
    data.adiabaticIndex = 1.4f;
    data.selfGravityEnabled = false;
    for (size_t index = 0; index < kParticles; ++index) {
        auto const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = 0.47 + 0.06 * index;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = index == 0 ? 0.2f : -0.1f;
        data.sourceVelocityY[source] = data.sourceVelocityZ[source] = 0.0f;
        data.sourceDensity[source] = 1.3f + 0.8f * index;
        data.sourceInternalEnergy[source] = 0.8f + 0.6f * index;
        data.sourceSmoothingLength[source] = 0.2f + 0.05f * index;
        data.sourceGradHCorrection[source] = (-0.1f + 0.3f * index) * data.gasMass;
        data.sourceBalsaraFactor[source] = 1.0f;
    }
    std::array<double, kParticles> pressureWork;
    // Independent scalar form of Price (2012), equation (35), with the variational
    // grad-h correction and the stored comoving units derived in variable_smoothing_lengths.hpp:
    //   (dEᵢ/dτ)_P = a⁻¹ (γ−1) Eᵢ (m−ζᵢ)/ρᵢ w′(r/hᵢ) hᵢ⁻⁽ᴰ⁺¹⁾ Δv r/|r|.
    // Symbol directory: Eᵢ=a²uᵢ; r=x₀−x₁; Δv=v₀−v₁; hᵢ=support; ρᵢ=comoving
    // density; ζᵢ=grad-h correction; γ=adiabatic index; a=scale factor; D=dimensions.
    double const r = distance(
        data.sourcePositionX[fixture.sourceIndex(0)], data.sourcePositionX[fixture.sourceIndex(1)]);
    double const Δv = double(data.sourceVelocityX[fixture.sourceIndex(0)]) -
        data.sourceVelocityX[fixture.sourceIndex(1)];
    for (size_t index = 0; index < kParticles; ++index) {
        auto const source = fixture.sourceIndex(index);
        double const hᵢ = data.sourceSmoothingLength[source];
        double const Eᵢ = data.sourceInternalEnergy[source];
        double const ρᵢ = data.sourceDensity[source];
        double const ζᵢ = data.sourceGradHCorrection[source];
        pressureWork[index] = (double(data.adiabaticIndex) - 1.0) * Eᵢ * (data.gasMass - ζᵢ) /
            (parameters.scaleFactor * ρᵢ) * dkernelWDq(std::abs(r) / hᵢ) /
            std::pow(hᵢ, spatialDimensions() + 1) * (r < 0.0 ? -Δv : Δv);
        pairKickRequire(pressureWork[index] > 0.0, "compressive pair missed reversible heating");
    }
    std::array<float, kParticles> energyRates;
    for (int dissipation = 0; dissipation < kDissipationCases; ++dissipation) {
        fixture.setBulkViscosityCoefficient(dissipation & 1 ? 0.7f : 0.0f);
        parameters.conductionCoefficient = dissipation & 2 ? 0.3f : 0.0f;
        data.energyEvolution = false;
        fixture.hydro();
        std::array<double, kParticles> heating;
        std::array<sph_geometry::SpatialVector<float>, kParticles> acceleration;
        for (size_t index = 0; index < kParticles; ++index) {
            heating[index] = fixture.dissipativeHeating(index);
            acceleration[index] = fixture.kickAcceleration(index);
            if (dissipation == 0) {
                pairKickRequire(heating[index] == 0.0, "default entropy evolution gained PdV");
            }
        }
        data.energyEvolution = true;
        fixture.hydro();
        double totalEnergyRate = 0.0;
        double rateScale = 0.0;
        for (size_t index = 0; index < kParticles; ++index) {
            auto const source = fixture.sourceIndex(index);
            auto const sink = fixture.sinkIndex(index);
            energyRates[index] = data.sinkThermalRate[sink];
            double const expected = pressureWork[index] + heating[index];
            pairKickRequire(std::abs(energyRates[index] - expected) <=
                    5.0e-6 * (std::abs(pressureWork[index]) + std::abs(heating[index])),
                "density-energy rate lost pressure work, viscosity, or conduction");
            pairKickClose(compensatedKickSum(data.sourcePairUpdates[source].thermal),
                parameters.hydroKickTimestep * energyRates[index],
                "density-energy pair kick differs from its complete thermal rate");
            auto const actualAcceleration = fixture.kickAcceleration(index);
            double kineticRate = 0.0;
            for (int axis = 0; axis < spatialDimensions(); ++axis) {
                pairKickRequire(actualAcceleration[axis] == acceleration[index][axis],
                    "thermal formulation changed the pressure or viscosity force");
                kineticRate += double(data.sourceVelocity(source)[axis]) * actualAcceleration[axis];
            }
            totalEnergyRate += energyRates[index] + kineticRate;
            rateScale += std::abs(energyRates[index]) + std::abs(kineticRate);
        }
        pairKickRequire(std::abs(totalEnergyRate) <= 5.0e-6 * rateScale,
            "density-energy pair failed instantaneous kinetic plus thermal conservation");
    }
    // Either finer endpoint must supply both directed pressure-work terms, even without
    // the diagnostic sink-rate arrays used by the preceding reference evaluations.
    data.sinkThermalRate = nullptr;
    for (size_t owner = 0; owner < kParticles; ++owner) {
        for (size_t index = 0; index < kParticles; ++index) {
            int const rung = index == owner ? 3 : 1;
            data.sourceRungs[fixture.sourceIndex(index)] = rung;
            data.sinkParticleRungs[fixture.particleIndex(index)] = rung;
        }
        parameters.minimumRung = parameters.maximumRung = 3;
        fixture.hydro();
        for (size_t index = 0; index < kParticles; ++index) {
            pairKickClose(
                compensatedKickSum(data.sourcePairUpdates[fixture.sourceIndex(index)].thermal),
                parameters.hydroKickTimestep * energyRates[index],
                "fine-owned pair omitted or duplicated an endpoint's full thermal rate");
        }
    }
}

// Homogeneous expansion is a one-particle source, not a pair exchange. Check its
// gamma dependence, its vanishing entropy counterpart, and exact-rung ownership.
template <class Fixture>
void testDensityEnergyExpansion(Fixture& fixture) {
    auto& data = fixture.data_;
    auto& parameters = fixture.params_;
    fixture.enableMidpointOutputs();
    fixture.enableKickObservation();
    fixture.setBulkViscosityCoefficient(0.0f);
    parameters.minimumRung = 0;
    parameters.maximumRung = kMaxRung - 1;
    parameters.scaleFactor = 0.7f;
    parameters.expansionRate = 0.03f;
    data.selfGravityEnabled = false;
    for (int index = 0; index < fixture.count_; ++index) {
        auto const source = fixture.sourceIndex(index);
        data.sourcePositionX[source] = 0.47 + 0.06 * index;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = data.sourceVelocityY[source] = data.sourceVelocityZ[source] =
            0.0f;
        data.sourceDensity[source] = 1.3f;
        data.sourceInternalEnergy[source] = 0.8f + 0.6f * index;
        data.sourceSmoothingLength[source] = 0.2f;
        data.sourceGradHCorrection[source] = 0.0f;
    }
    for (float const γ : {1.4f, 5.0f / 3.0f, 1.8f}) {
        data.adiabaticIndex = γ;
        for (bool const energyEvolution : {false, true}) {
            data.energyEvolution = energyEvolution;
            fixture.hydro();
            for (int index = 0; index < fixture.count_; ++index) {
                auto const source = fixture.sourceIndex(index);
                // Derived E=a²u and du/dt=-3H(γ-1)u: dE/dτ=(5-3γ)(da/dτ)E/a.
                // Symbol directory: E=stored specific energy; γ=adiabatic index.
                double const E = data.sourceInternalEnergy[source];
                double const scale = parameters.expansionRate / parameters.scaleFactor * E;
                double const expected = energyEvolution ? (5.0 - 3.0 * double(γ)) * scale : 0.0;
                double const tolerance = 2.0e-6 * std::abs(scale);
                pairKickRequire(std::abs(data.sinkThermalRate[fixture.sinkIndex(index)] -
                                    expected) <= tolerance,
                    "homogeneous expansion rate has the wrong formulation or gamma dependence");
                pairKickRequire(
                    std::abs(compensatedKickSum(data.sourcePairUpdates[source].thermal) /
                            parameters.hydroKickTimestep -
                        expected) <= tolerance,
                    "homogeneous expansion kick was omitted or counted per neighbor");
            }
        }
    }
    data.adiabaticIndex = 1.4f;
    data.energyEvolution = true;
    data.sinkThermalRate = nullptr;
    for (int index = 0; index < fixture.count_; ++index) {
        data.sourceRungs[fixture.sourceIndex(index)] = 1 + 2 * index;
        data.sinkParticleRungs[fixture.particleIndex(index)] = 1 + 2 * index;
    }
    for (int active = 0; active < fixture.count_; ++active) {
        parameters.minimumRung = parameters.maximumRung = 1 + 2 * active;
        fixture.hydro();
        for (int index = 0; index < fixture.count_; ++index) {
            double const actual =
                compensatedKickSum(data.sourcePairUpdates[fixture.sourceIndex(index)].thermal);
            pairKickRequire(index == active ? actual > 0.0 : actual == 0.0,
                "a pair event applied homogeneous expansion on an inactive endpoint's clock");
        }
    }
}

}    // namespace detail
