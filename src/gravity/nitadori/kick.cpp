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
    KickParameters const& parameters, FlopCount* flopCount, FlopCounts* precisionCounts) {
    KickReturn result;
    FlopCounter<> flops;
    auto const& options = getOptions();
    auto& rung = particlesRung(index);
    if (rung < parameters.minRung) {
        return result;
    }
    particlesCaptureGravitySample(
        index, acceleration[kXDimension], acceleration[kYDimension], acceleration[kZDimension]);
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
    auto submitFlops = [&]() {
        FlopCounts counts;
        if constexpr (kCountFlops) {
            // The radius expression promotes its float numerator, divides in
            // double precision, and narrows to float. All remaining solver
            // arithmetic in this kick uses float, including its FMA arguments.
            counts.singlePrecision = FlopCount(flops) - 1;
            counts.doublePrecision = 3;
        }
        if (flopCount != nullptr) {
            *flopCount += counts.total();
        }
        if (precisionCounts != nullptr) {
            *precisionCounts += counts;
        }
        if (flopCount == nullptr && precisionCounts == nullptr) {
            addCpuFlops(counts.singlePrecision, FlopPrecision::Single);
            addCpuFlops(counts.doublePrecision, FlopPrecision::Double);
        }
    };
    float const mass = particlesMass(index);
    float const oldKinetic = 0.5f * mass * sqr(vx, vy, vz);
#if COSMICTIGER_WITH_SPH
    if (!parameters.htime && options.hydro.enabled && particlesType(index) == kSphType) {
        // The hydro integrator owns the gas momentum and timestep. As in the
        // ordinary FMM kick, publish the active gas force without a second
        // gravity kick or a gravity-only rung update.
        ParticleInteger const gas = particlesCatIndex(index);
        for (int dim = 0; dim < kDimensions; ++dim) {
            sphParticlesGravityAcceleration(dim, gas) = acceleration[dim];
        }
        result.kin = oldKinetic;
        result.xmom = mass * vx;
        result.ymom = mass * vy;
        result.zmom = mass * vz;
        result.nmom = mass * std::sqrt(sqr(vx, vy, vz));
        result.pot = parameters.doPhi ? 0.5f * mass * potential : 0.0f;
        result.maxRung = rung;
        submitFlops();
        return result;
    }
#endif
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
    submitFlops();
    return result;
}

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
                        result.kin += 0.5 * mass * velocitySquared;
                        result.xmom += mass * particlesVel(kXDimension, i);
                        result.ymom += mass * particlesVel(kYDimension, i);
                        result.zmom += mass * particlesVel(kZDimension, i);
                        result.nmom += mass * std::sqrt(velocitySquared);
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
