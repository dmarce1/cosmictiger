/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/gravity/particle_mesh.hpp>
#include <cosmictiger/gravity/particle_mesh_kick.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

HPX_PLAIN_ACTION(particleMeshPrepareKick, ParticleMeshPrepareKickAction);
HPX_PLAIN_ACTION(particleMeshFinishKick, ParticleMeshFinishKickAction);

KickReturn particleMeshPrepareKick(
    double scale, double baseTimestep, double maximumTimestep, bool firstCall, int minimumRung) {
    std::vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ParticleMeshPrepareKickAction>(
            child, scale, baseTimestep, maximumTimestep, firstCall, minimumRung));
    }
    double const effectiveRadius = particleMeshEffectiveRadius();
    auto const& options = getOptions();
    double const sign = options.initial.createGlass ? -1.0 : 1.0;
    size_t const count = particlesSize();
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), std::max(size_t(1), (count + 4095) / 4096));
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async([=]() {
                KickReturn result;
                FlopCounter<> flops;
                for (size_t i = worker * count / workers; i < (worker + 1) * count / workers; ++i) {
                    auto const& acceleration = particleMeshAcceleration(i);
                    double accelerationSquared = 0.0;
                    for (int dim = 0; dim < NDIM; ++dim) {
                        accelerationSquared += double(acceleration[dim]) * acceleration[dim];
                    }
                    double const timestep = std::min(maximumTimestep,
                        options.gravity.timestepEta *
                            std::sqrt(scale * effectiveRadius /
                                std::max(std::sqrt(accelerationSquared), 1.0e-30)));
                    if (!(timestep > 0.0) || !std::isfinite(timestep)) {
                        THROW_ERROR("Particle mesh timestep is not finite and positive\n");
                    }
                    double const requestedRung = std::ceil(std::log2(baseTimestep / timestep));
                    if (!(requestedRung < MAX_RUNG)) {
                        THROW_ERROR("Particle mesh timestep exceeds the available rung range\n");
                    }
                    int const previousRung = particlesRung(i);
                    result.maxRung = std::max(result.maxRung,
                        std::max({1, int(requestedRung), previousRung - 1, minimumRung}));
                    bool const gas = options.hydro.enabled && particlesType(i) == SPH_TYPE;
                    double const oldHalfStep =
                        firstCall ? 0.0 : std::ldexp(0.5 * baseTimestep, -previousRung);
                    double velocitySquared = 0.0;
                    for (int dim = 0; dim < NDIM; ++dim) {
                        if (gas) {
                            sphParticlesGravityAcceleration(dim, particlesCatIndex(i)) =
                                sign * acceleration[dim];
                        } else {
                            particlesVel(dim, i) = std::fma(
                                sign * acceleration[dim], oldHalfStep, particlesVel(dim, i));
                        }
                        if (options.analysis.saveForce) {
                            particlesGravityAcceleration(dim, i) = acceleration[dim];
                        }
                        double const velocity = particlesVel(dim, i);
                        velocitySquared += velocity * velocity;
                    }
                    double const mass = particlesMass(i);
                    double const potential = particleMeshPotential(i);
                    if (options.analysis.saveForce) {
                        particlesPot(i) = potential;
                    }
                    result.pot += 0.5 * mass * potential;
                    result.kin += 0.5 * mass * velocitySquared;
                    result.xmom += mass * particlesVel(XDIM, i);
                    result.ymom += mass * particlesVel(YDIM, i);
                    result.zmom += mass * particlesVel(ZDIM, i);
                    result.nmom += mass * std::sqrt(velocitySquared);
                    flops += 32;
                }
                addCpuFlops(flops);
                return result;
            }));
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

void particleMeshFinishKick(double baseTimestep, int rung) {
    ALWAYS_ASSERT(rung >= 0 && rung < MAX_RUNG);
    std::vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ParticleMeshFinishKickAction>(child, baseTimestep, rung));
    }
    double const sign = getOptions().initial.createGlass ? -1.0 : 1.0;
    double const halfStep = sign * std::ldexp(0.5 * baseTimestep, -rung);
    bool const hydro = getOptions().hydro.enabled;
    size_t const count = particlesSize();
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), std::max(size_t(1), (count + 4095) / 4096));
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async([=]() {
                FlopCounter<> flops;
                for (size_t i = worker * count / workers; i < (worker + 1) * count / workers; ++i) {
                    particlesRung(i) = rung;
                    if (!hydro || particlesType(i) != SPH_TYPE) {
                        auto const& acceleration = particleMeshAcceleration(i);
                        for (int dim = 0; dim < NDIM; ++dim) {
                            particlesVel(dim, i) =
                                std::fma(acceleration[dim], halfStep, particlesVel(dim, i));
                        }
                        flops += 2 * NDIM;
                    }
                }
                addCpuFlops(flops);
            }));
        }
    } catch (...) {
        hpx::wait_all(futures);
        throw;
    }
    hpx::wait_all(futures);
    for (auto& future : futures) {
        future.get();
    }
}
