/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/kelvin_helmholtz.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

namespace scenarios {

namespace detail {

    void initializeKelvinHelmholtzThermodynamics(float pressure) {
        constexpr float kSupportSearchMultiplier = 1.2f;
        // All scenario particles are still on rank zero. The tree may sort gas
        // records, but does not migrate them; assign energies only after this pass.
        SphTreeCreateParameters treeParameters;
        treeParameters.minimumRung = 0;
        treeParameters.smoothingLengthMultiplier = kSupportSearchMultiplier;
        sphTreeCreate(treeParameters);
        SphRunParameters parameters;
        parameters.runType = kSphRunSmoothingLengths;
        parameters.phase = 0;
        parameters.minimumRung = 0;
        parameters.scaleFactor = 1.0f;
        parameters.expansionRate = 0.0f;
        parameters.baseTimestep = 0.0f;
        parameters.initializeDensity = true;
        SphTreeNeighborParameters neighbors;
        neighbors.minimumRung = 0;
        neighbors.smoothingLengthMultiplier = kSupportSearchMultiplier;
        neighbors.interactionDirection = SphInteractionDirection::TargetSupport;
        TreeId const root{0, 0};
        sphParticlesResetConverged();
        int passes = 0;
        while (true) {
            if (++passes > 256) {
                throw std::runtime_error("K-H initial smoothing lengths did not converge");
            }
            neighbors.operation = SphNeighborOperation::FindNeighbors;
            sphTreeNeighbor(neighbors, root, Vector<TreeId>{root}).get();
            // CPU dispatch calls sphCpuSmoothingLengths for each target leaf and
            // retains its actual kernel-sum density, including at support bounds.
            if (!sphRun(parameters, false).needsAnotherPass) {
                break;
            }
            neighbors.operation = SphNeighborOperation::RefreshBounds;
            sphTreeNeighbor(neighbors, root, Vector<TreeId>{}).get();
        }
        float const gammaMinusOne = getOptions().hydro.gamma - 1.0f;
        for (ParticleInteger particle = 0; particle < sphParticlesSize(); ++particle) {
            // Derived ideal-gas conversion, using the EOS in sphParticlesPressure:
            // P = (γ − 1) ρ u  =>  u = P / [(γ − 1) ρ].
            // P -> pressure; γ − 1 -> gammaMinusOne; ρ -> density;
            // u -> sphParticlesSetInternalEnergy. All quantities are in code units.
            float const density = sphParticlesDensity(particle);
            ALWAYS_ASSERT(density > 0.0f);
            sphParticlesSetInternalEnergy(particle, pressure / (gammaMinusOne * density));
        }
        sphTreeFreeNeighborList();
        sphTreeDestroy();
        sphParticlesClearCaches();
    }

}    // namespace detail

ScenarioPlan initializeKelvinHelmholtz() {
    ParticleInteger totalNumberOfParticles = getOptions().run.gasParticleCount;
    double rho1 = 0.5;
    double rho0 = 1.0;
    double vy1 = 1.0;
    double vy0 = -1.0;
    constexpr double kPerturbationAmplitude = 0.01;
    constexpr unsigned kPerturbationSeed = 42;
    auto const layout = particle_count::roundKelvinHelmholtz(totalNumberOfParticles);
    ParticleInteger const leftDim = layout.lightSide;
    ParticleInteger const rightDim = layout.heavySide;
    // Derived sampling limit for this y-only field: the two offset lattices
    // provide 4n distinct y planes, Δy = 1/(4n), and mode n_Nyquist = 2n.
    // Use the coarser fluid's limit and exclude the Nyquist mode itself and DC.
    // n -> leftDim or rightDim; retained mode count -> maximumPerturbationMode.
    int const maximumPerturbationMode = 2 * std::min(layout.lightSide, layout.heavySide) - 1;
    struct ModeAmplitudes {
        double sine;
        double cosine;
    };
    std::mt19937 generator(kPerturbationSeed);
    std::uniform_real_distribution<double> amplitude(
        -kPerturbationAmplitude, kPerturbationAmplitude);
    std::vector<ModeAmplitudes> modeAmplitudes(maximumPerturbationMode);
    // Derived RMS² = (1/2) Σ_k (a_k²+b_k²) -> velocityVariance.
    double velocityVariance = 0.0;
    for (auto& mode : modeAmplitudes) {
        mode.sine = amplitude(generator);
        mode.cosine = amplitude(generator);
        velocityVariance += 0.5 * (sqr(mode.sine) + sqr(mode.cosine));
    }
    // Prescribed random K-H spectrum (code units):
    // δv_x(y) = Σ_{k=1}^{K} [a_k sin(2πky) + b_k cos(2πky)],
    // a_k, b_k independently uniform on [−η,+η].
    // K -> maximumPerturbationMode; η -> kPerturbationAmplitude;
    // a_k, b_k -> modeAmplitudes[k−1].sine, .cosine; y -> streamwisePosition.
    // One reproducible coefficient set is shared by both fluids. Precompute
    // each y plane once; particle placement and tree ordering cannot reseed it.
    auto const transverseVelocities = [&](ParticleInteger halfBoxSide) {
        std::vector<double> velocities(4 * halfBoxSide, 0.0);
        for (size_t plane = 0; plane < velocities.size(); ++plane) {
            double const streamwisePosition = double(plane) / velocities.size();
            for (int mode = 1; mode <= maximumPerturbationMode; ++mode) {
                double const angle = 2.0 * std::numbers::pi * mode * streamwisePosition;
                velocities[plane] += modeAmplitudes[mode - 1].sine * std::sin(angle) +
                    modeAmplitudes[mode - 1].cosine * std::cos(angle);
            }
        }
        return velocities;
    };
    auto const leftTransverseVelocities = transverseVelocities(leftDim);
    auto const rightTransverseVelocities = transverseVelocities(rightDim);
    terminal::detailPrint(
        "K-H perturbation: seed=%u modes=1..%i coefficient_range=[%g,%g] rms=%e\n",
        kPerturbationSeed, maximumPerturbationMode, -kPerturbationAmplitude, kPerturbationAmplitude,
        std::sqrt(velocityVariance));
    ParticleInteger numberOfLeftParticles = particle_count::spatialPower(2 * leftDim);
    ParticleInteger numberOfRightParticles = particle_count::spatialPower(2 * rightDim);
    totalNumberOfParticles = numberOfLeftParticles + numberOfRightParticles;
    double dx = 0.5 / leftDim;
    ParticleInteger i = 0;
    auto opts = getOptions();
    opts.run.gasParticleCount = totalNumberOfParticles;
    opts.initial.gasMass = rho1 * 0.5 / numberOfLeftParticles;
    double const m = opts.initial.gasMass;
    setOptions(opts);
    rho1 = numberOfLeftParticles * m / 0.5f;
    rho0 = numberOfRightParticles * m / 0.5f;
    terminal::detailPrint("Kelvin-Helmholtz realized densities: left=%e right=%e ratio=%e\n", rho1,
        rho0, rho0 / rho1);
    for (int ix = 0; ix < leftDim; ix++) {
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? 2 * leftDim : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? 2 * leftDim : 1); iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double h = sph_geometry::supportRadius(m, rho1, getOptions().hydro.neighborNumber);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                sphParticlesVelocity(kXDimension, i) = leftTransverseVelocities[2 * iy + 1];
                sphParticlesVelocity(kYDimension, i) = vy1;
                sphParticlesVelocity(kZDimension, i) = 0.0;
                sphParticlesRung(i) = 0;

                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                sphParticlesVelocity(kXDimension, i) = leftTransverseVelocities[2 * iy];
                sphParticlesVelocity(kYDimension, i) = vy1;
                sphParticlesVelocity(kZDimension, i) = 0.0;
                sphParticlesRung(i) = 0;

                i++;
            }
        }
    }
    dx = 0.5 / rightDim;
    for (int ix0 = rightDim; ix0 < 2 * rightDim; ix0++) {
        int ix = ix0;
        for (int iy = 0; iy < (kSpatialDimensions >= 2 ? 2 * rightDim : 1); iy++) {
            for (int iz = 0; iz < (kSpatialDimensions >= 3 ? 2 * rightDim : 1); iz++) {
                double x = (ix + 0.5) * dx;
                double y = (iy + 0.5) * dx;
                double z = (iz + 0.5) * dx;
                double h = sph_geometry::supportRadius(m, rho0, getOptions().hydro.neighborNumber);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                sphParticlesVelocity(kXDimension, i) = rightTransverseVelocities[2 * iy + 1];
                sphParticlesVelocity(kYDimension, i) = vy0;
                sphParticlesVelocity(kZDimension, i) = 0.0;
                sphParticlesRung(i) = 0;

                i++;
                x = (ix) *dx;
                y = (iy) *dx;
                z = (iz) *dx;
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(i) = h;
                sphParticlesPosition(kXDimension, i) = x;
                sphParticlesPosition(kYDimension, i) = kSpatialDimensions >= 2 ? y : 0.5;
                sphParticlesPosition(kZDimension, i) = kSpatialDimensions >= 3 ? z : 0.5;
                sphParticlesVelocity(kXDimension, i) = rightTransverseVelocities[2 * iy];
                sphParticlesVelocity(kYDimension, i) = vy0;
                sphParticlesVelocity(kZDimension, i) = 0.0;
                sphParticlesRung(i) = 0;

                i++;
            }
        }
    }
    if (sphParticlesSize() == 0) {
        throw std::runtime_error("The Kelvin-Helmholtz scenario produced no gas particles; "
                                 "increase run.gasParticleCount");
    }
    detail::initializeKelvinHelmholtzThermodynamics(19.2f);
    return ScenarioPlan{{{10.0, 1024}}};
}

}    // namespace scenarios
