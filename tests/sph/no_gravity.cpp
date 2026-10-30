/* CosmicTiger - SPH-only saved-force regression; BSL-1.0. */

#include <bit>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

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
            sphParticlesPredictedAcceleration(dim, index) = 0.75f * (dim + 1);
        }
        particlesRung(index) = 1;
        sphParticlesThermodynamicState(index) = {2.0f + index, 0.25f};
        sphParticlesDensity(index) = 1.0f;
        sphParticlesInternalEnergyRate(index) = 0.125f;
        sphParticlesPredictedInternalEnergyRate(index) = 0.375f;
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> checkpoint(std::tmpfile(), std::fclose);
    if (checkpoint == nullptr) {
        throw std::runtime_error("Cannot create SPH-only particle checkpoint fixture");
    }
    particlesSave(checkpoint.get());
    poisonDiagnostics();
    sphParticlesThermodynamicState(0).internalEnergy = 99.0f;
    sphParticlesPredictedAcceleration(kXDimension, 0) = 99.0f;
    std::rewind(checkpoint.get());
    // This deliberately retains the same particle count, bypassing allocation.
    particlesLoad(checkpoint.get());
    checkDiagnostics(0, kParticleCount, "same-size checkpoint load");
    if (particlesSize() != kParticleCount || sphParticlesSize() != kParticleCount ||
        sphParticlesInternalEnergy(0) != 2.0f ||
        sphParticlesPredictedAcceleration(kXDimension, 0) != 0.75f) {
        throw std::runtime_error("SPH-only checkpoint failed to preserve the gas payload");
    }
    sphCheckpointTest();
    PRINT("SPH-only saved gravity diagnostics passed (allocation, reuse, and checkpoint load)\n");
}
