#include <algorithm>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/hydro/hopkins/hopkins_hydro.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/scenarios/sound_wave.hpp>

namespace scenarios {

ScenarioPlan initializeSoundWave() {
    auto options = getOptions();
    ParticleInteger const count = options.initial.particlesPerDimension;
    double const mass = 1.0 / count;
    options.initial.gasMass = mass;
    setOptions(options);
    sphParticlesResize(count);
    for (ParticleInteger i = 0; i < count; i++) {
        double const position = soundWavePosition((i + 0.5) * mass);
        sphParticlesPosition(kXDimension, i) = position;
        sphParticlesPosition(kYDimension, i) = 0.5;
        sphParticlesPosition(kZDimension, i) = 0.5;
        auto const primitive = soundWaveState(position);
        double const energy =
            primitive.pressure / ((options.hydro.gamma - 1.0) * primitive.density);
        double const support =
            std::clamp(0.5 * options.hydro.neighborNumber * mass / primitive.density,
                options.hydro.minSupportRadius, options.hydro.maxSupportRadius);
        sphParticlesSupportRadius(i) = support;
        sphParticlesVelocity(kXDimension, i) = primitive.velocity;
        sphParticlesVelocity(kYDimension, i) = 0.0;
        sphParticlesVelocity(kZDimension, i) = 0.0;
        sphParticlesRung(i) = 0;
        sphParticlesInternalEnergy(i) = energy;
        if (options.hydro.method == "mfv") {
            sphParticlesMfvMass(i) = mass;
        }
        // The 1e-6 mode must not be rounded through the ordinary float gas fields.
        auto& state = sphParticlesHopkinsState(i);
        state = HopkinsParticleState{};
        state.identifier = i;
        state.mass = mass;
        state.momentum[kXDimension] = mass * primitive.velocity;
        state.meshVelocity[kXDimension] = primitive.velocity;
        state.totalEnergy = mass * (energy + 0.5 * primitive.velocity * primitive.velocity);
        state.initialized = true;
        state.supportRadius = support;
    }
    return ScenarioPlan{{{1.0, 1}}, hopkinsSoundWaveDiagnostics};
}

}    // namespace scenarios
