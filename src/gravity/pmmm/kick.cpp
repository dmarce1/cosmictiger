/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/gravity/pmmm/kick.hpp>
#include <cosmictiger/gravity/pmmm/solver.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#endif

HPX_PLAIN_ACTION(particleMeshPrepareKick, ParticleMeshPrepareKickAction);
HPX_PLAIN_ACTION(particleMeshFinishKick, ParticleMeshFinishKickAction);

KickReturn particleMeshPrepareKick(double scale, double baseTimestep, double maximumTimestep,
    bool firstCall, int minimumRung, double previousTimestep) {
    std::vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ParticleMeshPrepareKickAction>(
            child, scale, baseTimestep, maximumTimestep, firstCall, minimumRung, previousTimestep));
    }
    auto const& options = getOptions();
    double const effectiveRadius = pmmmEffectiveRadius();
    addCpuFlops(1, FlopPrecision::Double);    // Support-radius conversion, once per locality.
    double const sign = options.run.scenario == "glass" ? -1.0 : 1.0;
    size_t const count = particlesSize();
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), std::max(size_t(1), (count + 4095) / 4096));
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async(hpx::annotated_function(
                [=]() {
                    KickReturn result;
                    FlopCounter<> flops;
                    for (size_t i = worker * count / workers; i < (worker + 1) * count / workers;
                        ++i) {
                        auto const& acceleration = pmmmAcceleration(i);
                        particlesCaptureGravitySample(i, acceleration[kXDimension],
                            acceleration[kYDimension], acceleration[kZDimension]);
                        double accelerationSquared = 0.0;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            accelerationSquared += double(acceleration[dim]) * acceleration[dim];
                        }
                        double const timestep = std::min(maximumTimestep,
                            options.gravity.timestepEta *
                                std::sqrt(scale * effectiveRadius /
                                    std::max(std::sqrt(accelerationSquared), 1.0e-30)));
                        if (!(timestep > 0.0) || !std::isfinite(timestep)) {
                            THROW_ERROR("Particle mesh timestep is not finite and positive\n");
                        }
                        result.minimumTimestep = std::min(result.minimumTimestep, timestep);
                        int const previousRung = particlesRung(i);
                        if (previousTimestep < 0.0) {
                            double const requestedRung =
                                std::ceil(std::log2(baseTimestep / timestep));
                            if (!(requestedRung < kMaxRung)) {
                                THROW_ERROR(
                                    "Particle mesh timestep exceeds the available rung range\n");
                            }
                            result.maxRung = std::max(result.maxRung,
                                std::max({1, int(requestedRung), previousRung - 1, minimumRung}));
                        } else {
                            particlesRung(i) = 0;
                        }
#if COSMICTIGER_WITH_SPH
                        bool const gas = options.hydro.enabled && particlesType(i) == kSphType;
#else
                        bool const gas = false;
#endif
                        double const oldHalfStep = previousTimestep >= 0.0 ?
                            0.5 * previousTimestep :
                            (firstCall ? 0.0 : std::ldexp(0.5 * baseTimestep, -previousRung));
                        // Acceleration norm and timestep; the rung proposal and
                        // ldexp half-step occur only for the rung-based path.
                        // Each norm product promotes both stored floats to double.
                        flops += 4 * kDimensions + 5 + (previousTimestep < 0.0 ? 3 : 0) +
                            (previousTimestep >= 0.0 ? 1 : (firstCall ? 0 : 2));
                        double velocitySquared = 0.0;
                        for (int dim = 0; dim < kDimensions; ++dim) {
#if COSMICTIGER_WITH_SPH
                            if (gas) {
                                sphParticlesGravityAcceleration(dim, particlesCatIndex(i)) =
                                    sign * acceleration[dim];
                                // Promote acceleration, multiply, then narrow the result.
                                flops += 3;
                            } else
#endif
                            {
                                particlesVel(dim, i) = std::fma(
                                    sign * acceleration[dim], oldHalfStep, particlesVel(dim, i));
                                // Three arithmetic operations plus two promotions and narrowing.
                                flops += 6;
                            }
                            if (saveParticleAcceleration(options.analysis)) {
                                particlesGravityAcceleration(dim, i) = acceleration[dim];
                            }
                            double const velocity = particlesVel(dim, i);
                            velocitySquared += velocity * velocity;
                        }
                        double const mass = particlesMass(i);
                        double const potential = pmmmPotential(i);
                        if (saveParticlePotential(options.analysis)) {
                            particlesPot(i) = potential;
                        }
                        result.pot += 0.5 * mass * potential;
                        if (!gas) {
                            // Hydro sums the gas energies and momenta in its kick.
                            result.kin += 0.5 * mass * velocitySquared;
                            result.xmom += mass * particlesVel(kXDimension, i);
                            result.ymom += mass * particlesVel(kYDimension, i);
                            result.zmom += mass * particlesVel(kZDimension, i);
                            result.nmom += mass * std::sqrt(velocitySquared);
                        }
                    }
                    addCpuFlops(flops, FlopPrecision::Double);
                    return result;
                },
                "gravity.pmmm.prepareKick")));
        }
    } catch (...) {
        hpx::wait_all(futures);
        throw;
    }
    hpx::wait_all(futures);
    KickReturn result;
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

double particleMeshFinishKick(double baseTimestep, int rung, double explicitTimestep) {
    ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
    std::vector<hpx::future<double>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(
            hpx::async<ParticleMeshFinishKickAction>(child, baseTimestep, rung, explicitTimestep));
    }
    double const sign = getOptions().run.scenario == "glass" ? -1.0 : 1.0;
    double const halfStep = sign *
        (explicitTimestep >= 0.0 ? 0.5 * explicitTimestep : std::ldexp(0.5 * baseTimestep, -rung));
    // The explicit timestep skips ldexp; both paths multiply by one half and sign.
    addCpuFlops(explicitTimestep >= 0.0 ? 2 : 3, FlopPrecision::Double);
    bool const hydro = getOptions().hydro.enabled;
    size_t const count = particlesSize();
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), std::max(size_t(1), (count + 4095) / 4096));
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async(hpx::annotated_function(
                [=]() {
                    FlopCounter<> flops;
                    double kinetic = 0.0;
                    for (size_t i = worker * count / workers; i < (worker + 1) * count / workers;
                        ++i) {
                        particlesRung(i) = rung;
                        if (!hydro || particlesType(i) != kSphType) {
                            auto const& acceleration = pmmmAcceleration(i);
                            for (int dim = 0; dim < kDimensions; ++dim) {
                                particlesVel(dim, i) =
                                    std::fma(acceleration[dim], halfStep, particlesVel(dim, i));
                            }
                            // Each FMA promotes two floats and narrows its double result.
                            flops += 5 * kDimensions;
                        }
                        double velocitySquared = 0.0;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            double const velocity = particlesVel(dim, i);
                            velocitySquared += velocity * velocity;
                        }
                        kinetic += 0.5 * particlesMass(i) * velocitySquared;
                    }
                    addCpuFlops(flops, FlopPrecision::Double);
                    return kinetic;
                },
                "gravity.pmmm.finishKick")));
        }
    } catch (...) {
        hpx::wait_all(futures);
        throw;
    }
    hpx::wait_all(futures);
    double kinetic = 0.0;
    for (auto& future : futures) {
        kinetic += future.get();
    }
    return kinetic;
}

double particleMeshFinishGlobalKick(double timestep) {
    ALWAYS_ASSERT(std::isfinite(timestep) && timestep > 0.0);
    return particleMeshFinishKick(0.0, 0, timestep);
}
