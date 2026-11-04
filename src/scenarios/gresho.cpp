#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/scenarios/gresho.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

ScenarioPlan initializeGresho() {
    auto opts = getOptions();
    ParticleInteger const ndim = opts.initial.particlesPerDimension;
    if (opts.initial.numberOfParticles > size_t(std::numeric_limits<ParticleInteger>::max())) {
        throw std::overflow_error("Gresho particle count exceeds ParticleInteger range");
    }
    ParticleInteger const count = ParticleInteger(opts.initial.numberOfParticles);
    opts.initial.gasMass = 1.0 / count;
    setOptions(opts);
    double const supportRadius =
        std::cbrt(opts.initial.gasMass * opts.hydro.neighborNumber / (4.0 * std::acos(-1.0) / 3.0));
    sphParticlesResize(count);
    ParticleInteger i = 0;
    for (ParticleInteger ix = 0; ix < ndim; ix++) {
        for (ParticleInteger iy = 0; iy < ndim; iy++) {
            for (ParticleInteger iz = 0; iz < ndim; iz++) {
                for (int phase = 0; phase < (opts.initial.closePack ? 4 : 1); ++phase) {
                    auto const position =
                        greshoLatticePosition(ix, iy, iz, phase, ndim, opts.initial.closePack);
                    double const offsetX = position[0] - 0.5;
                    double const offsetY = position[1] - 0.5;
                    double const radius = std::hypot(offsetX, offsetY);
                    auto const state = greshoState(radius);
                    double const angularVelocity =
                        radius > 0.0 ? state.azimuthalVelocity / radius : 0.0;
                    sphParticlesSmoothingLength(i) = supportRadius;
                    sphParticlesPosition(kXDimension, i) = position[0];
                    sphParticlesPosition(kYDimension, i) = position[1];
                    sphParticlesPosition(kZDimension, i) = position[2];
                    sphParticlesVelocity(kXDimension, i) = -angularVelocity * offsetY;
                    sphParticlesVelocity(kYDimension, i) = angularVelocity * offsetX;
                    sphParticlesVelocity(kZDimension, i) = 0.0;
                    sphParticlesRung(i) = 0;
                    sphParticlesSetInternalEnergy(i, state.pressure / (opts.hydro.gamma - 1.0));
                    i++;
                }
            }
        }
    }
    return ScenarioPlan{{{3.0, 64}}};
}

}    // namespace scenarios
