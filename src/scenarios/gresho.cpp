#include <cmath>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/scenarios/gresho.hpp>

namespace scenarios {

ScenarioPlan initializeGresho() {
    auto opts = getOptions();
    ParticleInteger const ndim = opts.initial.particlesPerDimension;
    ParticleInteger const count = ndim * ndim * ndim;
    opts.initial.gasMass = 1.0 / count;
    setOptions(opts);
    double const spacing = 1.0 / ndim;
    double const supportRadius =
        std::cbrt(opts.initial.gasMass * opts.hydro.neighborNumber / (4.0 * std::acos(-1.0) / 3.0));
    sphParticlesResize(count);
    ParticleInteger i = 0;
    for (ParticleInteger ix = 0; ix < ndim; ix++) {
        for (ParticleInteger iy = 0; iy < ndim; iy++) {
            double const x = (ix + 0.5) * spacing;
            double const y = (iy + 0.5) * spacing;
            double const offsetX = x - 0.5;
            double const offsetY = y - 0.5;
            double const radius = std::hypot(offsetX, offsetY);
            auto const state = greshoState(radius);
            double const angularVelocity = radius > 0.0 ? state.azimuthalVelocity / radius : 0.0;
            for (ParticleInteger iz = 0; iz < ndim; iz++) {
                sphParticlesSupportRadius(i) = supportRadius;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = y;
                sphParticlesPosition(kZDimension, i) = (iz + 0.5) * spacing;
                sphParticlesVelocity(kXDimension, i) = -angularVelocity * offsetY;
                sphParticlesVelocity(kYDimension, i) = angularVelocity * offsetX;
                sphParticlesVelocity(kZDimension, i) = 0.0;
                sphParticlesRung(i) = 0;
                sphParticlesThermodynamicState(i).internalEnergy =
                    state.pressure / (opts.hydro.gamma - 1.0);
                i++;
            }
        }
    }
    return ScenarioPlan{{{3.0, 64}}};
}

}    // namespace scenarios
