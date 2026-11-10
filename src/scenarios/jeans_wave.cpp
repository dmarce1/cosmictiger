/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/scenarios/jeans_wave.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/particles.hpp>

// clang-format off
/*
 * Linear Jeans wave (Jeans 1902; derived for this setup). Uniform gas of density ρ₀ and
 * sound speed c in a periodic unit box, displaced along x by the Lagrangian standing wave
 *   ξ(x₀, t) = (A/k) sin(k x₀) cos(ωt),   ω² = c²k² − 4πGρ₀,   k = 2π,
 * so that ρ = ρ₀/(1 + A cos(k x₀) cos(ωt)) and u_x = −(Aω/k) sin(k x₀) sin(ωt). Periodic
 * gravity removes the mean density (the Jeans swindle), so the dispersion relation is
 * exact for the mode. The scenario sets c² = 8πGρ₀/k², so ω = ck/√2 with gravity and
 * ω = ck without, and evolves to t = 3π/(4 ω_J).
 *
 * Measured modes over the N gas particles (derived):
 *   V = (2/N) Σ u_x sin(k x) = −(Aω/k) sin(ωt),   D = (2/N) Σ cos(k x) = −A cos(ωt),
 * to first order in A, with x = x₀ + ξ; D is the particle number density mode and so carries
 * no kernel smoothing, unlike a mode of the SPH density estimate,
 * Artificial viscosity damps both modes by one common factor, which cancels in the phase:
 *   ωt = atan2(−kV/ω, −D),
 * solved by fixed-point iteration from the expected ω_e on the branch nearest ω_e t. The
 * damping is reported as the ratio of the measured to the undamped amplitude,
 *   [(kV/ω)² + D²]^(1/2) / A.
 * Symbol directory: A → relative density amplitude; k → wavenumber; c → sound speed;
 *   G → gravity.gm; ρ₀ → mean density; ω → angular frequency; ξ → displacement;
 *   u_x → velocity; V, D → velocity and density modes; ω_e → expected ω.
 *   Code names: rho_0 = ρ₀; omega_J = ω_J; omega_s = ck; omega_fit = measured ω; c2 = c².
 */
// clang-format on

namespace scenarios {
namespace detail {

    inline constexpr double kJeansAmplitude = 0.01;
    inline constexpr double kJeansWavenumber = 2.0 * std::numbers::pi;
    inline constexpr double kJeansDensity = 1.0;
    inline constexpr double kJeansPeriodFraction = 0.375;    // ωt = 3π/4

    struct JeansModes {
        double velocity = 0.0;    // Σ u_x sin(k x)
        double density = 0.0;     // Σ cos(k x), the particle number density mode
        double count = 0.0;

        /// Archive this record in the field order required by distributed
        /// transport.
        template <class Archive>
        void serialize(Archive& archive, unsigned) {
            archive & velocity & density & count;
        }
    };

    /// Accumulate local Fourier projections of gas density and velocity for the
    /// Jeans-wave diagnostic.
    JeansModes localJeansModes() {
        JeansModes modes;
        double const k = kJeansWavenumber;
        for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
            double const x = sphParticlesPositionValue(kXi, i).toDouble();
            modes.velocity += sphParticlesVelocityValue(kXi, i) * std::sin(k * x);
            modes.density += std::cos(k * x);
            modes.count += 1.0;
        }
        return modes;
    }

}    // namespace detail
}    // namespace scenarios

HPX_PLAIN_ACTION(scenarios::detail::localJeansModes, LocalJeansModesAction);

namespace scenarios {
namespace detail {

    // Sound speed squared from G: c² = 8πGρ₀/k², so 4πGρ₀ = c²k²/2.
    double jeansSoundSpeedSquared() {
        double const gm = getOptions().gravity.gm;
        if (!(gm > 0.0) || !std::isfinite(gm)) {
            throw std::invalid_argument("The jeansWave scenario needs a positive gravity.gm");
        }
        return 8.0 * std::numbers::pi * gm * kJeansDensity / sqr(kJeansWavenumber);
    }

    /// Reduce Fourier modes and compare the measured Jeans-wave frequency and amplitude
    /// with the reference.
    void finishJeansWave(double time) {
        JeansModes modes;
        std::vector<hpx::future<JeansModes>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<LocalJeansModesAction>(locality));
        }
        for (auto& future : futures) {
            auto const part = future.get();
            modes.velocity += part.velocity;
            modes.density += part.density;
            modes.count += part.count;
        }
        double const k = kJeansWavenumber;
        double const A = kJeansAmplitude;
        double const c2 = jeansSoundSpeedSquared();
        double const omega_s = std::sqrt(c2) * k;
        double const omega_J = std::sqrt(
            c2 * sqr(k) - 4.0 * std::numbers::pi * getOptions().gravity.gm * kJeansDensity);
        double const V = 2.0 * modes.velocity / modes.count;
        double const D = 2.0 * modes.density / modes.count;
        double const expected = getOptions().gravity.enabled ? omega_J : omega_s;
        double omega_fit = expected;
        for (int iteration = 0; iteration < 64; ++iteration) {
            double phase = std::atan2(-k * V / omega_fit, -D);
            // Choose the branch of the phase nearest the expected phase.
            double const target = expected * time;
            phase +=
                2.0 * std::numbers::pi * std::round((target - phase) / (2.0 * std::numbers::pi));
            omega_fit = phase / time;
        }
        double const amplitude = std::sqrt(sqr(k * V / omega_fit) + sqr(D)) / A;
        double const error = std::abs(omega_fit - expected) / expected;
        std::cout << std::format("Jeans wave: gravity={} time={:e} velocity_mode={:e} "
                                 "density_mode={:e} omega_fit={:e} omega_jeans={:e} "
                                 "omega_sound={:e} amplitude={:e} relative_error={:e}\n",
                         getOptions().gravity.enabled ? "T" : "F", time, V, D, omega_fit, omega_J,
                         omega_s, amplitude, error)
                  << std::flush;
    }

}    // namespace detail

/// Initialize the sinusoidal Jeans mode with consistent density, velocity, and thermal
/// support.
ScenarioPlan initializeJeansWave() {
    auto opts = getOptions();
    ParticleInteger const side = particle_count::roundLattice(opts.run.gasParticleCount).side;
    ParticleInteger const count = particle_count::spatialPower(side);
    opts.run.gasParticleCount = count;
    opts.initial.gasMass = detail::kJeansDensity / count;
    setOptions(opts);
    double const m = opts.initial.gasMass;
    double const gamma = opts.hydro.heatCapacityRatio;
    double const rho_0 = detail::kJeansDensity;
    double const c2 = detail::jeansSoundSpeedSquared();
    // u = c²/(γ(γ−1)) at ρ₀, adiabatic: u ∝ ρ^(γ−1).
    double const u_0 = c2 / (gamma * (gamma - 1.0));
    double const k = detail::kJeansWavenumber;
    double const A = detail::kJeansAmplitude;
    double const h = sph_geometry::supportRadius(m, rho_0, opts.hydro.neighborNumber);
    double const dx = 1.0 / side;
    ParticleInteger index = 0;
    for (ParticleInteger ix = 0; ix < side; ix++) {
        for (ParticleInteger iy = 0; iy < (kSpatialDimensions >= 2 ? side : 1); iy++) {
            for (ParticleInteger iz = 0; iz < (kSpatialDimensions >= 3 ? side : 1); iz++) {
                double const x0 = (ix + 0.5) * dx;
                double const x = x0 + A / k * std::sin(k * x0);
                double const compression = 1.0 + A * std::cos(k * x0);
                sphParticlesResize(sphParticlesSize() + 1);
                sphParticlesSmoothingLength(index) = h;
                sphParticlesPosition(kXi, index) = x - std::floor(x);
                if constexpr (kSpatialDimensions > 1) {
                    sphParticlesPosition(kYi, index) = (iy + 0.5) * dx;
                }
                if constexpr (kSpatialDimensions > 2) {
                    sphParticlesPosition(kZi, index) = (iz + 0.5) * dx;
                }
                for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                    sphParticlesVelocity(dim, index) = 0.0f;
                }
                sphParticlesRung(index) = 0;
                sphParticlesSetInternalEnergy(index, u_0 * std::pow(compression, 1.0 - gamma));
                index++;
            }
        }
    }
    double const omega_J =
        std::sqrt(c2 * sqr(k) - 4.0 * std::numbers::pi * opts.gravity.gm * rho_0);
    double const duration = 2.0 * std::numbers::pi * detail::kJeansPeriodFraction / omega_J;
    return ScenarioPlan{{{duration, 32}}, detail::finishJeansWave};
}

}    // namespace scenarios
