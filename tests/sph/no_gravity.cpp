/* CosmicTiger - SPH-only saved-force regression; BSL-1.0. */

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/particles/particles.hpp>

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
            sphParticlesAcceleration(dim, index) = 0.5f * (dim + 1);
        }
        particlesRung(index) = 1;
        sphParticlesSetInternalEnergy(index, 2.0f + index);
        sphParticlesSupportRadius(index) = 0.25f;
        sphParticlesDensity(index) = 1.0f;
        sphParticlesEntropyRate(index) = 0.125f;
    }
    UniqueFile checkpoint(std::tmpfile());
    if (checkpoint == nullptr) {
        throw std::runtime_error("Cannot create SPH-only particle checkpoint fixture");
    }
    particlesSave(checkpoint.get());
    poisonDiagnostics();
    sphParticlesSetInternalEnergy(0, 99.0f);
    sphParticlesAcceleration(kXDimension, 0) = 99.0f;
    std::rewind(checkpoint.get());
    // This deliberately retains the same particle count, bypassing allocation.
    particlesLoad(checkpoint.get());
    checkDiagnostics(0, kParticleCount, "same-size checkpoint load");
    if (particlesSize() != kParticleCount || sphParticlesSize() != kParticleCount ||
        sphParticlesInternalEnergy(0) != 2.0f || sphParticlesAcceleration(kXDimension, 0) != 0.5f) {
        throw std::runtime_error("SPH-only checkpoint failed to preserve the gas payload");
    }
    sphCheckpointTest();
    // Exercise the actual entropy initialization, staged prediction, and both kick phases.
    sphParticlesPrepareThermodynamics(0.5, true);
    float const initialEntropy = sphParticlesEntropy(0);
    float const initialEnergy = sphParticlesInternalEnergy(0);
    sphParticlesDensity(0) *= 8.0f;
    if (sphParticlesEntropy(0) != initialEntropy ||
        std::abs(sphParticlesInternalEnergy(0) / initialEnergy -
            std::pow(8.0, options.hydro.gamma - 1.0)) > 1.e-5) {
        throw std::runtime_error("Adiabatic compression did not preserve entropy");
    }
    sphParticlesEntropyRate(0) = 0.25f;
    float const velocity = sphParticlesVelocity(kXDimension, 0);
    sphParticlesApplyUpdates(1, 0, 1.0f, 1.0f, 0.2);
    if (sphParticlesEntropy(0) != initialEntropy ||
        sphParticlesVelocity(kXDimension, 0) != velocity) {
        throw std::runtime_error("Entropy predictor mutated the staggered particle state");
    }
    float predictedEnergy, support;
    ParticleGlobalRange range;
    range.proc = hpxRank();
    range.range.first = 0;
    range.range.second = 1;
    sphParticlesGlobalReadInternalEnergyAndSupportRadius(range, &predictedEnergy, &support, 0);
    double const expectedEnergy = sph_entropy::storedEnergy(
        initialEntropy + 0.025, sphParticlesDensity(0), options.hydro.gamma, 0.5);
    if (std::abs(predictedEnergy - expectedEnergy) > 1.e-5 * expectedEnergy) {
        throw std::runtime_error("Staging did not derive energy from predicted entropy");
    }
    sphParticlesEntropyRate(0) = 0.5f;
    sphParticlesApplyUpdates(1, 1, 1.0f, 1.0f, 0.2);
    // Opening may use a different interval; reuse the fresh closing rate.
    sphParticlesApplyUpdates(1, 2, 1.0f, 1.0f, 0.4);
    if (std::abs(sphParticlesEntropy(0) - (initialEntropy + 0.15)) > 1.e-6) {
        throw std::runtime_error("Entropy closing/opening kicks used the wrong rates or intervals");
    }
    sphCheckpointTest();
    PRINT("SPH-only diagnostics and entropy lifecycle passed (prediction, kicks, checkpoint)\n");
}
