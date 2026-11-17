/* CosmicTiger - SPH storage, fixed-dimension and saved-force regressions; BSL-1.0. */

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <tests/mfm/persistence_cases.hpp>

void sphOnlyDiagnosticsTest() {
    static_assert(COSMICTIGER_WITH_SPH);
    if (particlesSize() != 0 || sphParticlesSize() != 0 || hpxSize() != 1) {
        throw std::runtime_error("SPH-only diagnostics fixture requires empty single-rank state");
    }
    auto options = getOptions();
    options.hydro.enabled = true;
    // Exercise the runtime-off path even when gravity support is compiled into the binary.
    options.gravity.enabled = false;
    options.gravity.cuda = false;
    options.chemistry.enabled = false;
    options.analysis.saveForce = true;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.hydro.cuda = false;
    options.hydro.minSmoothingLength = 0.25;
    options.hydro.maxSmoothingLength = 0.25;
    options.hydro.damping = 0.0;
    options.scenario.disc.centralAcceleration = 0.0;
    options.hydro.yAcceleration = 0.0;
    options.scenario.rayleighTaylor.zPlaneAcceleration = 0.0;
    options.initial.gasMass = 0.125;
    options.run.minRung = 0;
    setOptions(options);
    constexpr ParticleInteger kParticleCount = 8;
    auto const checkDiagnostics = [](ParticleInteger begin, ParticleInteger end,
                                      char const* stage) {
        for (ParticleInteger index = begin; index < end; ++index) {
            // Integer checks also reject NaNs under the application's fast-math flags.
            if (std::bit_cast<uint32_t>(particlesPot(index)) != 0U) {
                throw std::runtime_error(std::string(stage) + ": nonzero saved gravity potential");
            }
            for (int dim = 0; dim < spatialDimensions(); ++dim) {
                if (std::bit_cast<uint32_t>(particlesGravityAcceleration(dim, index)) != 0U) {
                    throw std::runtime_error(
                        std::string(stage) + ": nonzero saved gravity acceleration");
                }
            }
        }
    };
    auto const poisonDiagnostics = []() {
        for (ParticleInteger index = 0; index < particlesSize(); ++index) {
            particlesPot(index) = -100.0f - index;
            for (int dim = 0; dim < spatialDimensions(); ++dim) {
                particlesGravityAcceleration(dim, index) = 100.0f + 3 * index + dim;
            }
        }
    };
    sphParticlesResize(kParticleCount);
    // Live selectors cannot change the meaning or allocation of resident thermal fields.
    for (bool const changeFormulation : {true, false}) {
        auto altered = options;
        if (changeFormulation) {
            altered.hydro.sphFormulation =
                usesSphEnergyEvolution(options.hydro) ? "densityEntropy" : "densityEnergy";
        } else {
            altered.hydro.hopkins = !options.hydro.hopkins;
        }
        auto const* const entropyStorage = sphParticlesEntropies;
        auto const* const energyStorage = sphParticlesInternalEnergies;
        bool rejected = false;
        try {
            setOptions(altered);
        } catch (std::invalid_argument const& error) {
            rejected =
                std::string(error.what()).find("after gas storage allocation") != std::string::npos;
        }
        if (!rejected || getOptions().hydro.sphFormulation != options.hydro.sphFormulation ||
            getOptions().hydro.hopkins != options.hydro.hopkins ||
            sphParticlesEntropies != entropyStorage ||
            sphParticlesInternalEnergies != energyStorage) {
            throw std::runtime_error("A live thermal selector change escaped its storage guard");
        }
    }
    checkDiagnostics(0, kParticleCount, "new allocation");
    auto const rejectDimensionChange = [&options]() {
        int const originalDimensions = spatialDimensions();
        auto* const originalPositions = particlesX[kXi];
        auto changed = options;
        changed.hydro.dimensions = originalDimensions == 1 ? 2 : 1;
        bool rejected = false;
        try {
            setOptions(changed);
        } catch (std::invalid_argument const& error) {
            rejected = std::string(error.what()) ==
                "hydro.dimensions cannot be changed after particle storage allocation";
            if (!rejected) {
                throw;
            }
        }
        if (!rejected || spatialDimensions() != originalDimensions ||
            getOptions().hydro.dimensions != originalDimensions ||
            particlesX[kXi] != originalPositions) {
            throw std::runtime_error("Rejected dimension change altered allocated particle state");
        }
    };
    rejectDimensionChange();
    // A zero logical count retains allocated capacity and must retain its dimensionality.
    sphParticlesResize(0);
    if (particlesSize() != 0 || particlesX[kXi] == nullptr) {
        throw std::runtime_error("Dimension-change fixture did not retain empty storage capacity");
    }
    rejectDimensionChange();
    sphParticlesResize(kParticleCount);
    checkDiagnostics(0, kParticleCount, "empty retained allocation");
    poisonDiagnostics();
    // Reusing capacity must initialize newly added logical particles as well.
    sphParticlesResize(kParticleCount / 2);
    sphParticlesResize(kParticleCount);
    checkDiagnostics(kParticleCount / 2, kParticleCount, "reused allocation");
    for (ParticleInteger index = 0; index < kParticleCount; ++index) {
        for (int dim = 0; dim < spatialDimensions(); ++dim) {
            particlesPos(dim, index) = 0.125 * index;
            particlesVel(dim, index) = 0.25f * (dim + 1);
        }
        particlesRung(index) = 1;
        sphParticlesSetInternalEnergy(index, 2.0f + index);
        sphParticlesSmoothingLength(index) = 0.25f;
        sphParticlesDensity(index) = 1.0f;
        sphParticlesSetProposedRung(index, 3);
        detail::setMfmFixtureGeometry(index, index);
    }
    UniqueFile checkpoint(std::tmpfile());
    if (checkpoint == nullptr) {
        throw std::runtime_error("Cannot create SPH-only particle checkpoint fixture");
    }
    particlesSave(checkpoint.get());
    poisonDiagnostics();
    sphParticlesSetInternalEnergy(0, 99.0f);
    particlesVel(kXi, 0) = 99.0f;
    sphParticlesSetProposedRung(0, 9);
    detail::setMfmFixtureGeometry(0, 99);
    std::rewind(checkpoint.get());
    // This deliberately retains the same particle count, bypassing allocation.
    particlesLoad(checkpoint.get());
    checkDiagnostics(0, kParticleCount, "same-size checkpoint load");
    for (ParticleInteger index = 0; index < kParticleCount; ++index) {
        detail::requireMfmFixtureGeometry(index, index);
    }
    if (particlesSize() != kParticleCount || sphParticlesSize() != kParticleCount ||
        sphParticlesInternalEnergy(0) != 2.0f || particlesVel(kXi, 0) != 0.25f ||
        sphParticlesProposedRung(0) != 3) {
        throw std::runtime_error("SPH-only checkpoint failed to preserve the gas payload");
    }
    sphCheckpointTest();
    // Exercise the selected canonical thermal field and live source staging.
    sphParticlesPrepareThermodynamics(0.5, true);
    float const initialEntropy = sphParticlesEntropyValue(0);
    float const initialEnergy = sphParticlesInternalEnergy(0);
    float const initialThermal = sphParticlesThermalVariable(0);
    sphParticlesDensity(0) *= 8.0f;
    sphParticlesPrepareThermodynamics(0.5, false);
    double const energyRatio = usesSphEnergyEvolution(options.hydro) ?
        1.0 :
        std::pow(8.0, options.hydro.heatCapacityRatio - 1.0);
    if (sphParticlesThermalVariable(0) != initialThermal ||
        std::abs(sphParticlesInternalEnergy(0) / initialEnergy - energyRatio) > 1.e-5) {
        throw std::runtime_error("Density refresh changed the canonical thermal variable");
    }
    if (options.hydro.hopkins &&
        (sphParticlesEntropies[0] != sphParticlesEntropyValue(0) ||
            sphParticlesInternalEnergies[0] != sphParticlesInternalEnergy(0))) {
        throw std::runtime_error("Density refresh left the auxiliary Hopkins field stale");
    }
    float predictedEnergy, support;
    ParticleGlobalRange range;
    range.proc = hpxRank();
    range.range.first = 0;
    range.range.second = 1;
    sphParticlesGlobalReadInternalEnergyAndSmoothingLength(range, &predictedEnergy, &support, 0);
    double const expectedEnergy = usesSphEnergyEvolution(options.hydro) ?
        initialEnergy :
        sph_entropy::storedEnergy(
            initialEntropy, sphParticlesDensity(0), options.hydro.heatCapacityRatio, 0.5);
    if (std::abs(predictedEnergy - expectedEnergy) > 1.e-5 * expectedEnergy) {
        throw std::runtime_error("Staging did not read the selected canonical thermal state");
    }
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        for (int dimension = 0; dimension < spatialDimensions(); ++dimension) {
            // The neighboring diagonal particles lie near the edge of h=0.25.
            // A weak compression heats by less than one float entropy unit at
            // dt=0.0005; use a resolved shock without changing that short step.
            sphParticlesVelocity(dimension, gas) =
                -10.0f * (sphParticlesPosition(dimension, gas).toDouble() - 0.5);
        }
    }
    auto const totals = []() {
        Array<double, 4> values{};
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            for (int dimension = 0; dimension < spatialDimensions(); ++dimension) {
                values[dimension] += sphParticlesVelocity(dimension, gas);
            }
            values[3] += sphParticlesThermalVariable(gas);
        }
        return values;
    };
    auto const initial = totals();
    EnergiesT energies{};
    auto const estimate = sphMidpointStep(1, 0.0, 0.0, 0.001, 0.5, 0.0, energies);
    auto const estimated = totals();
    for (int component = 0; component < 4; ++component) {
        if (std::abs(estimated[component] - initial[component]) > 1.0e-7) {
            throw std::runtime_error("Zero-duration SPH force estimate changed physical state");
        }
    }
    if (!(estimate.minimumTimestep > 0.0 && std::isfinite(estimate.minimumTimestep))) {
        throw std::runtime_error("SPH-only estimate did not return a positive timestep");
    }
    sphMidpointStep(1, 0.00025, 0.0005, 0.001, 0.5, 0.0, energies);
    auto const kicked = totals();
    for (int dimension = 0; dimension < spatialDimensions(); ++dimension) {
        if (std::abs(kicked[dimension] - initial[dimension]) > 3.0e-6) {
            throw std::runtime_error("SPH midpoint pair impulses failed momentum conservation");
        }
    }
    if (!(kicked[3] > initial[3])) {
        throw std::runtime_error("Converging midpoint SPH flow produced no irreversible heating");
    }
    std::printf("SPH-only midpoint thermal increment: %.9e\n", kicked[3] - initial[3]);
    checkDiagnostics(0, kParticleCount, "midpoint kick");
    sphCheckpointTest();
    std::cout << "SPH-only diagnostics and thermal lifecycle passed (live sources, midpoint pairs, "
                 "checkpoint)\n"
              << std::flush;
}
