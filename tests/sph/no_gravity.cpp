/* CosmicTiger - SPH-only saved-force regression; BSL-1.0. */

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>

void sphOnlyDiagnosticsTest() {
    static_assert(!COSMICTIGER_WITH_GRAVITY && COSMICTIGER_WITH_SPH);
    if (particlesSize() != 0 || sphParticlesSize() != 0 || hpxSize() != 1) {
        throw std::runtime_error("SPH-only diagnostics fixture requires empty single-rank state");
    }
    auto options = getOptions();
    options.hydro.enabled = true;
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
            for (int dim = 0; dim < kDimensions; ++dim) {
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
            for (int dim = 0; dim < kDimensions; ++dim) {
                particlesGravityAcceleration(dim, index) = 100.0f + 3 * index + dim;
            }
        }
    };
    sphParticlesResize(kParticleCount);
    checkDiagnostics(0, kParticleCount, "new allocation");
    poisonDiagnostics();
    // Reusing capacity must initialize newly added logical particles as well.
    sphParticlesResize(kParticleCount / 2);
    sphParticlesResize(kParticleCount);
    checkDiagnostics(kParticleCount / 2, kParticleCount, "reused allocation");
    for (ParticleInteger index = 0; index < kParticleCount; ++index) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, index) = 0.125 * index;
            particlesVel(dim, index) = 0.25f * (dim + 1);
        }
        particlesRung(index) = 1;
        sphParticlesSetInternalEnergy(index, 2.0f + index);
        sphParticlesSmoothingLength(index) = 0.25f;
        sphParticlesDensity(index) = 1.0f;
        sphParticlesSetProposedRung(index, 3);
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
    std::rewind(checkpoint.get());
    // This deliberately retains the same particle count, bypassing allocation.
    particlesLoad(checkpoint.get());
    checkDiagnostics(0, kParticleCount, "same-size checkpoint load");
    if (particlesSize() != kParticleCount || sphParticlesSize() != kParticleCount ||
        sphParticlesInternalEnergy(0) != 2.0f || particlesVel(kXi, 0) != 0.25f ||
        sphParticlesProposedRung(0) != 3) {
        throw std::runtime_error("SPH-only checkpoint failed to preserve the gas payload");
    }
    sphCheckpointTest();
    // Exercise canonical entropy, live source staging, and the actual frozen midpoint pass.
    sphParticlesPrepareThermodynamics(0.5, true);
    float const initialEntropy = sphParticlesEntropy(0);
    float const initialEnergy = sphParticlesInternalEnergy(0);
    sphParticlesDensity(0) *= 8.0f;
    if (sphParticlesEntropy(0) != initialEntropy ||
        std::abs(sphParticlesInternalEnergy(0) / initialEnergy -
            std::pow(8.0, options.hydro.heatCapacityRatio - 1.0)) > 1.e-5) {
        throw std::runtime_error("Adiabatic compression did not preserve entropy");
    }
    float predictedEnergy, support;
    ParticleGlobalRange range;
    range.proc = hpxRank();
    range.range.first = 0;
    range.range.second = 1;
    sphParticlesGlobalReadInternalEnergyAndSmoothingLength(range, &predictedEnergy, &support, 0);
    double const expectedEnergy = sph_entropy::storedEnergy(
        initialEntropy, sphParticlesDensity(0), options.hydro.heatCapacityRatio, 0.5);
    if (std::abs(predictedEnergy - expectedEnergy) > 1.e-5 * expectedEnergy) {
        throw std::runtime_error("Staging did not derive energy from live canonical entropy");
    }
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
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
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                values[dimension] += sphParticlesVelocity(dimension, gas);
            }
            values[3] += sphParticlesEntropy(gas);
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
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        if (std::abs(kicked[dimension] - initial[dimension]) > 3.0e-6) {
            throw std::runtime_error("SPH midpoint pair impulses failed momentum conservation");
        }
    }
    if (!(kicked[3] > initial[3])) {
        throw std::runtime_error("Converging midpoint SPH flow produced no irreversible heating");
    }
    std::printf("SPH-only midpoint entropy increment: %.9e\n", kicked[3] - initial[3]);
    checkDiagnostics(0, kParticleCount, "midpoint kick");
    sphCheckpointTest();
    std::cout << "SPH-only diagnostics and entropy lifecycle passed (live sources, midpoint pairs, "
                 "checkpoint)\n"
              << std::flush;
}
