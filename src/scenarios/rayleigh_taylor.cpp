// Initialize mirrored Rayleigh-Taylor layers in two or three spatial dimensions.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace scenarios {

/// Initialize mirrored light/heavy gas layers with hydrostatic pressure and interface
/// perturbations.
ScenarioPlan initializeRayleighTaylor() {
    static_assert(kSpatialDimensions >= 2);
    constexpr int kVerticalAxis = kSpatialDimensions == 2 ? kYi : kZi;
    constexpr int kMirroredSideCount = 2;
    constexpr int kQuarterCount = 4;
    constexpr int kHeavySiteCount = 2;
    constexpr int kLightSiteCount = 1;
    constexpr int kLayerSiteCount = kHeavySiteCount + kLightSiteCount;
    constexpr double kHalfBox = 0.5;
    constexpr double kQuarterBox = 0.25;
    auto opts = getOptions();
    auto const lattice =
        particle_count::roundLattice(opts.run.gasParticleCount, kRayleighTaylorLatticeMultiplicity);
    int const quarterDim = lattice.side;
    if (quarterDim == 0) {
        throw std::runtime_error("run.gasParticleCount must be positive");
    }
    if (lattice.count > size_t(std::numeric_limits<ParticleInteger>::max()) ||
        quarterDim > std::numeric_limits<int>::max() / kQuarterCount) {
        throw std::overflow_error("Rayleigh-Taylor particle count exceeds the supported range");
    }
    int const halfDim = kMirroredSideCount * quarterDim;
    int const fullDim = kQuarterCount * quarterDim;
    int const transverseDim = kSpatialDimensions == 3 ? fullDim : 1;
    ParticleInteger const particleCount = lattice.count;
    ParticleInteger const lightCount = particleCount / kLayerSiteCount;
    constexpr double kLightDensity = 1.0;
    constexpr double kHeavyDensity = 2.0;
    constexpr double kBoundaryPressure = 1.5;
    constexpr double kVelocityAmplitude = 0.1;
    double const acceleration = opts.scenario.rayleighTaylor.zPlaneAcceleration;
    if (!(acceleration > 0.0)) {
        throw std::invalid_argument(
            "The Rayleigh-Taylor scenario requires scenario.rayleighTaylor.zPlaneAcceleration > 0");
    }
    opts.run.gasParticleCount = particleCount;
    opts.initial.gasMass = kHalfBox * kLightDensity / lightCount;
    setOptions(opts);
    double const mass = opts.initial.gasMass;
    double const dx = kHalfBox / halfDim;
    sphParticlesResize(particleCount);
    ParticleInteger index = 0;
    for (int ix = 0; ix < fullDim; ix++) {
        for (int transverseIndex = 0; transverseIndex < transverseDim; transverseIndex++) {
            for (int verticalIndex = 0; verticalIndex < halfDim; verticalIndex++) {
                bool const heavy = verticalIndex < quarterDim;
                double const density = heavy ? kHeavyDensity : kLightDensity;
                int const samples = heavy ? kHeavySiteCount : kLightSiteCount;
                double const smoothingLength =
                    sph_geometry::supportRadius(mass, density, opts.hydro.neighborNumber);
                for (int sample = 0; sample < samples; sample++) {
                    double const offset = heavy ? kQuarterBox + kHalfBox * sample : kHalfBox;
                    Position const x((ix + offset) * dx);
                    Position const vertical((verticalIndex + offset) * dx);
                    double const height = vertical.toDouble();
                    double const distanceFromMiddle = kHalfBox - height;
                    // Integrate the vertical pressure gradient from the outer boundary:
                    // dP/dy = rho * a_y in 2D; dP/dz = rho * a_z in 3D.
                    // This balances the imposed field; self-gravity is an independent option.
                    double const pressure = kBoundaryPressure +
                        acceleration *
                            (kHeavyDensity *
                                    (kHalfBox - std::fmax(distanceFromMiddle, kQuarterBox)) +
                                kLightDensity * std::fmax(kQuarterBox - distanceFromMiddle, 0.0));
                    double const internalEnergy =
                        pressure / ((opts.hydro.heatCapacityRatio - 1.0) * density);
                    double const transverse = x.toDouble() - kHalfBox;
                    // Retain the legacy multiharmonic seed (d8e95728), made
                    // antisymmetric in the vertical coordinate by 5e926504. The transverse Fourier
                    // weights are 1 + cos(4 phi) + (cos(phi)+cos(7 phi))/2,
                    // phi=2*pi*(x-0.5); kVelocityAmplitude is a scale, not its peak.
                    double const velocity = kVelocityAmplitude / 4.0 *
                        (1.0 +
                            std::cos(8.0 * std::numbers::pi * transverse) *
                                (1.0 + std::cos(6.0 * std::numbers::pi * transverse))) *
                        std::sin(2.0 * std::numbers::pi * height);
                    // Mirror in fixed-point coordinates so paired positions sum to one
                    // exactly. Scalars are even and the normal velocity is odd.
                    for (int side = 0; side < kMirroredSideCount; side++) {
                        sphParticlesPosition(kXi, index) = x;
                        if constexpr (kSpatialDimensions == 3) {
                            sphParticlesPosition(kYi, index) =
                                Position((transverseIndex + offset) * dx);
                        }
                        sphParticlesPosition(kVerticalAxis, index) =
                            side == 0 ? vertical : Position(0.0) - vertical;
                        for (int dim = 0; dim < kSpatialDimensions; dim++) {
                            sphParticlesVelocity(dim, index) = 0.0f;
                        }
                        sphParticlesVelocity(kVerticalAxis, index) =
                            side == 0 ? velocity : -velocity;
                        sphParticlesSmoothingLength(index) = smoothingLength;
                        sphParticlesSetInternalEnergy(index, internalEnergy);
                        sphParticlesRung(index) = 0;
                        index++;
                    }
                }
            }
        }
    }
    return ScenarioPlan{{{10.0, 256}}};
}

}    // namespace scenarios
