#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/scenarios/gresho.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

ScenarioPlan initializeGresho() {
    auto opts = getOptions();
    auto const layout =
        particle_count::roundLattice(opts.run.gasParticleCount, opts.initial.closePack ? 4 : 1);
    ParticleInteger const ndim = layout.side;
    opts.run.gasParticleCount = layout.count;
    if (opts.run.gasParticleCount > size_t(std::numeric_limits<ParticleInteger>::max())) {
        throw std::overflow_error("Gresho particle count exceeds ParticleInteger range");
    }
    ParticleInteger const count = ParticleInteger(opts.run.gasParticleCount);
    opts.initial.gasMass = 1.0 / count;
    setOptions(opts);
    double const smoothingLength =
        sph_geometry::supportRadius(opts.initial.gasMass, 1.0, opts.hydro.neighborNumber);
    sphParticlesResize(count);
    ParticleInteger i = 0;
    for (ParticleInteger ix = 0; ix < ndim; ix++) {
        for (ParticleInteger iy = 0; iy < (kSpatialDimensions >= 2 ? ndim : 1); iy++) {
            for (ParticleInteger iz = 0; iz < (kSpatialDimensions >= 3 ? ndim : 1); iz++) {
                for (int phase = 0; phase < (opts.initial.closePack ? 4 : 1); ++phase) {
                    auto const position =
                        greshoLatticePosition(ix, iy, iz, phase, ndim, opts.initial.closePack);
                    double const offsetX = position[0] - 0.5;
                    double const offsetY = position[1] - 0.5;
                    double const radius = std::hypot(offsetX, offsetY);
                    auto const state = greshoState(radius);
                    double const angularVelocity =
                        radius > 0.0 ? state.azimuthalVelocity / radius : 0.0;
                    sphParticlesSmoothingLength(i) = smoothingLength;
                    sphParticlesPosition(kXDimension, i) = position[0];
                    sphParticlesPosition(kYDimension, i) =
                        kSpatialDimensions >= 2 ? position[1] : 0.5;
                    sphParticlesPosition(kZDimension, i) =
                        kSpatialDimensions >= 3 ? position[2] : 0.5;
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
