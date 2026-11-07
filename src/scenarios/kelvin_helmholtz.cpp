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
#include <cosmictiger/scenarios/kelvin_helmholtz_lattice.hpp>
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
    auto opts = getOptions();
    auto const packing =
        parseKelvinHelmholtzPacking(opts.initial.kelvinHelmholz.packing, kSpatialDimensions);
    auto const layout = particle_count::roundKelvinHelmholtz(
        opts.run.gasParticleCount, packing.low.sitesPerCell(kSpatialDimensions));
    int const halfBoxCells = layout.side;
    double const cellSpacing = 0.5 / halfBoxCells;
    constexpr double kLowDensity = 0.5;
    constexpr double kHighDensity = 2.0 * kLowDensity;
    constexpr double kPerturbationAmplitude = 0.01;
    constexpr unsigned kPerturbationSeed = 42;

    // Derived sampling limit: a lattice with p distinct y offsets per cell has
    // 2np equally spaced y planes, so its Nyquist mode is np. Retain positive
    // modes below both fluids' limits, excluding DC and the Nyquist endpoint.
    // n -> halfBoxCells; p -> planesPerCell; K -> maximumPerturbationMode.
    int const maximumPerturbationMode = halfBoxCells *
            std::min(packing.low.planesPerCell(kSpatialDimensions, kYi),
                packing.high.planesPerCell(kSpatialDimensions, kYi)) -
        1;
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
    // Precompute all possible half-cell y planes once for both fluids.
    std::vector<double> transverseVelocities(size_t(4) * halfBoxCells, 0.0);
    for (size_t plane = 0; plane < transverseVelocities.size(); ++plane) {
        double const streamwisePosition = double(plane) / transverseVelocities.size();
        for (int mode = 1; mode <= maximumPerturbationMode; ++mode) {
            double const angle = 2.0 * std::numbers::pi * mode * streamwisePosition;
            transverseVelocities[plane] += modeAmplitudes[mode - 1].sine * std::sin(angle) +
                modeAmplitudes[mode - 1].cosine * std::cos(angle);
        }
    }
    terminal::detailPrint(
        "K-H perturbation: seed=%u positive_modes=%i coefficient_range=[%g,%g] rms=%e\n",
        kPerturbationSeed, maximumPerturbationMode, -kPerturbationAmplitude, kPerturbationAmplitude,
        std::sqrt(velocityVariance));

    // Derived equal-mass normalization for half-box volume V=1/2:
    // m = ρ₀ V/N_low; N_high = 2 N_low ⇒ ρ_high = 2ρ₀ exactly in cell averages.
    // ρ₀ -> kLowDensity; N_low -> layout.lowCount; m -> opts.initial.gasMass.
    opts.run.gasParticleCount = layout.count;
    opts.initial.gasMass = kLowDensity * 0.5 / layout.lowCount;
    setOptions(opts);
    terminal::detailPrint(
        "Kelvin-Helmholtz packing: %s cell_spacing=%e low_count=%zu high_count=%zu\n",
        packing.name().c_str(), cellSpacing, layout.lowCount, layout.highCount);
    terminal::detailPrint("Kelvin-Helmholtz realized densities: left=%e right=%e ratio=%e\n",
        kLowDensity, kHighDensity, kHighDensity / kLowDensity);

    cell_lattice::CellLattice::CellCounts cells{
        halfBoxCells, 2 * halfBoxCells, kSpatialDimensions == 3 ? 2 * halfBoxCells : 1};
    sphParticlesResize(layout.count);
    ParticleInteger particle = 0;
    auto const addLayer = [&](cell_lattice::CellLattice const& lattice, double originX,
                              double density, double streamwiseVelocity) {
        double const supportRadius =
            sph_geometry::supportRadius(opts.initial.gasMass, density, opts.hydro.neighborNumber);
        lattice.generate(kSpatialDimensions, cells, cellSpacing, {originX, 0.0, 0.0},
            [&](cell_lattice::CellLattice::Position const& position) {
                for (int axis = 0; axis < kSpatialDimensions; ++axis) {
                    sphParticlesPosition(axis, particle) = position[axis];
                    sphParticlesVelocity(axis, particle) = 0.0;
                }
                size_t const plane =
                    size_t(std::llround(position[kYi] * transverseVelocities.size()));
                sphParticlesVelocity(kXi, particle) = transverseVelocities.at(plane);
                sphParticlesVelocity(kYi, particle) = streamwiseVelocity;
                sphParticlesSmoothingLength(particle) = supportRadius;
                sphParticlesRung(particle) = 0;
                ++particle;
            });
    };
    addLayer(packing.low, 0.0, kLowDensity, 1.0);
    addLayer(packing.high, 0.5, kHighDensity, -1.0);
    ALWAYS_ASSERT(size_t(particle) == layout.count);
    detail::initializeKelvinHelmholtzThermodynamics(19.2f);
    return ScenarioPlan{{{10.0, 1024}}};
}

}    // namespace scenarios
