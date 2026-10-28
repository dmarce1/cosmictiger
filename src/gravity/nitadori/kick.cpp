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
#include <cosmictiger/gravity/nitadori/kick.hpp>
#include <cosmictiger/gravity/nitadori/solver.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph_particles.hpp>
#endif

HPX_PLAIN_ACTION(particleMeshPrepareKick, ParticleMeshPrepareKickAction);
HPX_PLAIN_ACTION(particleMeshFinishKick, ParticleMeshFinishKickAction);

KickReturn particleMeshApplyField(ParticleInteger index,
    Array<float, kDimensions> const& acceleration, float potential,
    KickParameters const& parameters, FlopCount* flopCount) {
    KickReturn result;
    FlopCounter<> flops;
    auto const& options = getOptions();
    auto& rung = particlesRung(index);
    if (rung < parameters.minRung) {
        return result;
    }
    bool const saveAcceleration =
        parameters.saveForce || saveParticleAcceleration(options.analysis);
    bool const savePotential = parameters.saveForce || saveParticlePotential(options.analysis);
    if (saveAcceleration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesGravityAcceleration(dim, index) = acceleration[dim];
        }
    }
    if (savePotential && parameters.doPhi) {
        particlesPot(index) = potential;
    }
    auto& vx = particlesVel(kXDimension, index);
    auto& vy = particlesVel(kYDimension, index);
    auto& vz = particlesVel(kZDimension, index);
    float const gx = acceleration[kXDimension];
    float const gy = acceleration[kYDimension];
    float const gz = acceleration[kZDimension];
    float const effectiveRadius =
        parameters.gravitySupportRadius / selectedGravityKernel().supportRadiusPerEffectiveRadius;
    flops += 1;
    float const mass = particlesMass(index);
    float const oldKinetic = 0.5f * mass * sqr(vx, vy, vz);
    int proposedRung = 0;
    if (!parameters.htime || parameters.descending) {
        float const accelerationSquared = sqr(gx, gy, gz) + 1.0e-35f;
        float const requiredTimestep = std::min(parameters.maxDt,
            std::min(parameters.eta * std::sqrt(parameters.a) *
                    std::sqrt(effectiveRadius / std::sqrt(accelerationSquared)),
                parameters.t0));
        if (!(requiredTimestep > 0.0f) || !std::isfinite(requiredTimestep)) {
            THROW_ERROR("PMMM kick timestep is not finite and positive\n");
        }
        float const requestedRung = std::ceil(std::log2(parameters.t0 / requiredTimestep));
        if (!std::isfinite(requestedRung) || requestedRung >= kMaxRung) {
            THROW_ERROR("PMMM kick timestep exceeds the available rung range\n");
        }
        proposedRung = int(requestedRung);
        // Squared acceleration, timestep bound, and division/log2/ceil rung proposal.
        flops += 15;
    }
    if (parameters.htime) {
        float const sign = parameters.sign * (parameters.top ? 1.0f : -1.0f);
        float const halfStep = 0.5f * rungDt[parameters.minRung] * parameters.t0;
        flops += 3;
        if (parameters.ascending && !parameters.firstCall) {
            vx = std::fma(sign * gx, halfStep, vx);
            vy = std::fma(sign * gy, halfStep, vy);
            vz = std::fma(sign * gz, halfStep, vz);
            flops += 3 * kDimensions;
        }
        result.kin = 0.5f * mass * sqr(vx, vy, vz);
        result.xmom = mass * vx;
        result.ymom = mass * vy;
        result.zmom = mass * vz;
        result.nmom = mass * std::sqrt(sqr(vx, vy, vz));
        if (parameters.descending) {
            rung = std::max(
                parameters.minRung + int(proposedRung > parameters.minRung), int(rung) - 1);
            result.maxRung = rung;
            ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
            vx = std::fma(sign * gx, halfStep, vx);
            vy = std::fma(sign * gy, halfStep, vy);
            vz = std::fma(sign * gz, halfStep, vz);
            flops += 3 * kDimensions;
        }
    } else {
        float const sign = parameters.sign;
        if (!parameters.firstCall) {
            float const halfStep = 0.5f * rungDt[rung] * parameters.t0;
            vx = std::fma(sign * gx, halfStep, vx);
            vy = std::fma(sign * gy, halfStep, vy);
            vz = std::fma(sign * gz, halfStep, vz);
            flops += 2 + 3 * kDimensions;
        }
        result.kin = 0.5f * mass * sqr(vx, vy, vz);
        result.xmom = mass * vx;
        result.ymom = mass * vy;
        result.zmom = mass * vz;
        result.nmom = mass * std::sqrt(sqr(vx, vy, vz));
        float const synchronizedVx = vx;
        float const synchronizedVy = vy;
        float const synchronizedVz = vz;
        rung = std::max({proposedRung, int(rung) - 1, parameters.minRung, 1});
        result.maxRung = rung;
        if (parameters.fixedRung != 0) {
            rung = parameters.fixedRung;
        }
        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        float const halfStep = 0.5f * rungDt[rung] * parameters.t0;
        vx = std::fma(sign * gx, halfStep, vx);
        vy = std::fma(sign * gy, halfStep, vy);
        vz = std::fma(sign * gz, halfStep, vz);
        flops += 2 + 3 * kDimensions;
        result.darkMatterKineticCorrection =
            0.5f * mass * (sqr(synchronizedVx, synchronizedVy, synchronizedVz) - sqr(vx, vy, vz));
        result.darkMatterXMomentumCorrection = mass * (synchronizedVx - vx);
        result.darkMatterYMomentumCorrection = mass * (synchronizedVy - vy);
        result.darkMatterZMomentumCorrection = mass * (synchronizedVz - vz);
        result.darkMatterMomentumNormCorrection = mass *
            (std::sqrt(sqr(synchronizedVx, synchronizedVy, synchronizedVz)) -
                std::sqrt(sqr(vx, vy, vz)));
    }
    result.pot = parameters.doPhi ? 0.5f * mass * potential : 0.0;
    result.dkin = 0.5f * mass * sqr(vx, vy, vz) - oldKinetic;
    // Energy/momentum diagnostics and their reductions are deliberately excluded.
    if (flopCount != nullptr) {
        *flopCount += flops;
    } else {
        addCpuFlops(flops);
    }
    return result;
}

KickReturn particleMeshPrepareKick(
    double scale, double baseTimestep, double maximumTimestep, bool firstCall, int minimumRung) {
    std::vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ParticleMeshPrepareKickAction>(
            child, scale, baseTimestep, maximumTimestep, firstCall, minimumRung));
    }
    auto const& options = getOptions();
    double const effectiveRadius = pmmmEffectiveRadius();
    addCpuFlops(1);    // Support-radius conversion, once per locality.
    double const sign = options.initial.createGlass ? -1.0 : 1.0;
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
                        double const requestedRung = std::ceil(std::log2(baseTimestep / timestep));
                        if (!(requestedRung < kMaxRung)) {
                            THROW_ERROR(
                                "Particle mesh timestep exceeds the available rung range\n");
                        }
                        int const previousRung = particlesRung(i);
                        result.maxRung = std::max(result.maxRung,
                            std::max({1, int(requestedRung), previousRung - 1, minimumRung}));
#if COSMICTIGER_WITH_SPH
                        bool const gas = options.hydro.enabled && particlesType(i) == kSphType;
#endif
                        double const oldHalfStep =
                            firstCall ? 0.0 : std::ldexp(0.5 * baseTimestep, -previousRung);
                        // Acceleration norm, timestep/rung proposal, and optional old half-step.
                        flops += 2 * kDimensions + 8 + (firstCall ? 0 : 2);
                        double velocitySquared = 0.0;
                        for (int dim = 0; dim < kDimensions; ++dim) {
#if COSMICTIGER_WITH_SPH
                            if (gas) {
                                sphParticlesGravityAcceleration(dim, particlesCatIndex(i)) =
                                    sign * acceleration[dim];
                                flops += 1;
                            } else
#endif
                            {
                                particlesVel(dim, i) = std::fma(
                                    sign * acceleration[dim], oldHalfStep, particlesVel(dim, i));
                                flops += 3;
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
                        result.kin += 0.5 * mass * velocitySquared;
                        result.xmom += mass * particlesVel(kXDimension, i);
                        result.ymom += mass * particlesVel(kYDimension, i);
                        result.zmom += mass * particlesVel(kZDimension, i);
                        result.nmom += mass * std::sqrt(velocitySquared);
                    }
                    addCpuFlops(flops);
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

void particleMeshFinishKick(double baseTimestep, int rung) {
    ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
    std::vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ParticleMeshFinishKickAction>(child, baseTimestep, rung));
    }
    double const sign = getOptions().initial.createGlass ? -1.0 : 1.0;
    double const halfStep = sign * std::ldexp(0.5 * baseTimestep, -rung);
    addCpuFlops(3);    // Half-step multiplication, ldexp, and sign multiplication.
    bool const hydro = getOptions().hydro.enabled;
    size_t const count = particlesSize();
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), std::max(size_t(1), (count + 4095) / 4096));
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async(hpx::annotated_function(
                [=]() {
                    FlopCounter<> flops;
                    for (size_t i = worker * count / workers; i < (worker + 1) * count / workers;
                        ++i) {
                        particlesRung(i) = rung;
                        if (!hydro || particlesType(i) != kSphType) {
                            auto const& acceleration = pmmmAcceleration(i);
                            for (int dim = 0; dim < kDimensions; ++dim) {
                                particlesVel(dim, i) =
                                    std::fma(acceleration[dim], halfStep, particlesVel(dim, i));
                            }
                            flops += 2 * kDimensions;
                        }
                    }
                    addCpuFlops(flops);
                },
                "gravity.pmmm.finishKick")));
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
