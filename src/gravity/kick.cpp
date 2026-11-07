/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <stack>
#include <utility>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#if COSMICTIGER_ENABLE_FMM
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#endif
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#endif

namespace detail {

static thread_local std::stack<KickLists> localLists;
static Array<float*, kDimensions> globalAccelerations{};
static size_t globalAccelerationCount = 0;

enum class GasOpeningState { Disabled, Collecting, Sealed };
struct GasOpeningImpulse {
    ParticleInteger particleIndex;
    Vector3<float> velocityIncrement;
};
static GasOpeningState gasOpeningState = GasOpeningState::Disabled;
static Vector3<float>* gasOpeningIncrements = nullptr;
static size_t gasOpeningCount = 0;
static size_t gasOpeningParticleCount = 0;
static Vector<GasOpeningImpulse> gasOpeningImpulses;

KickLists kickAcquireLists() {
    if (localLists.empty()) {
        localLists.push(KickLists());
    }
    KickLists lists = std::move(localLists.top());
    localLists.pop();
    return std::move(lists);
}

void kickReleaseLists(KickLists&& lists) {
    lists.cclist.resize(0);
    lists.cplist.resize(0);
    lists.pclist.resize(0);
    lists.nextlist.resize(0);
    lists.leaflist.resize(0);
    localLists.push(std::move(lists));
}

bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

}    // namespace detail

KickReturn kickWalkDispatch(KickParameters params, ExpansionBuffer l, Vector3<Position> pos,
    TreeId self, Vector<TreeId> direct, Vector<TreeId> periodic,
    std::shared_ptr<KickWorkspace> workspace, KickContext const& context) {
    return fmmDispatch([&](auto orderTag) {
        return kickWalk<decltype(orderTag)::value>(params, std::move(l), pos, self,
            std::move(direct), std::move(periodic), std::move(workspace), context);
    });
}

HPX_PLAIN_ACTION(gravityBeginGlobalKick, GravityBeginGlobalKickAction);
HPX_PLAIN_ACTION(gravityFinishGlobalKick, GravityFinishGlobalKickAction);
HPX_PLAIN_ACTION(gravityFinishBalancedGasKick, GravityFinishBalancedGasKickAction);

void gravityBeginGlobalKick() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityBeginGlobalKickAction>(child));
    }
    size_t const count = particlesSize();
    if (count != detail::globalAccelerationCount) {
        std::array<std::unique_ptr<float, decltype(&freeManagedMemory)>, kDimensions> replacement{
            {{nullptr, &freeManagedMemory}, {nullptr, &freeManagedMemory},
                {nullptr, &freeManagedMemory}}};
        try {
            for (int dim = 0; dim < kDimensions; ++dim) {
                replacement[dim].reset(count == 0 ?
                        nullptr :
                        static_cast<float*>(allocateManagedMemory(count * sizeof(float))));
            }
        } catch (...) {
            hpx::wait_all(futures.begin(), futures.end());
            throw;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            freeManagedMemory(detail::globalAccelerations[dim]);
            detail::globalAccelerations[dim] = replacement[dim].release();
        }
        detail::globalAccelerationCount = count;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

float* gravityGlobalAcceleration(int dimension) {
    return detail::globalAccelerations[dimension];
}

KickReturn gravityFinishBalancedGasKick(double baseTimestep, float sign) {
    ALWAYS_ASSERT(std::isfinite(baseTimestep) && baseTimestep > 0.0);
    Vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(
            hpx::async<GravityFinishBalancedGasKickAction>(child, baseTimestep, sign));
    }
    ALWAYS_ASSERT(detail::globalAccelerationCount == particlesSize());
    KickReturn result;
#if COSMICTIGER_WITH_SPH
    ALWAYS_ASSERT(sphMidpointEnabled());
    FlopCount flops = 0;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        if (particlesType(index) != kSphType) {
            continue;
        }
        int const rung = particlesRung(index);
        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        result.maxRung = std::max(result.maxRung, rung);
        float const halfTimestep = sign * 0.5f * rungDt[rung] * float(baseTimestep);
        Vector3<float> before;
        Vector3<float> after;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            before[dimension] = particlesVel(dimension, index);
            after[dimension] = std::fma(
                detail::globalAccelerations[dimension][index], halfTimestep, before[dimension]);
            particlesVel(dimension, index) = after[dimension];
        }
        flops += 9;
        double const mass = particlesMass(index);
        double const kineticChange = 0.5 * mass * (after.dot(after) - before.dot(before));
        result.dkin += kineticChange;
        result.darkMatterKineticCorrection -= kineticChange;
        result.darkMatterXMomentumCorrection += mass * (before[kXi] - after[kXi]);
        result.darkMatterYMomentumCorrection += mass * (before[kYi] - after[kYi]);
        result.darkMatterZMomentumCorrection += mass * (before[kZi] - after[kZi]);
        result.darkMatterMomentumNormCorrection += mass * (before.norm() - after.norm());
    }
    addCpuFlops(flops, FlopPrecision::Single);
#endif
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        freeManagedMemory(detail::globalAccelerations[dimension]);
        detail::globalAccelerations[dimension] = nullptr;
    }
    detail::globalAccelerationCount = 0;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

double gravityFinishGlobalKick(double timestep) {
    ALWAYS_ASSERT(std::isfinite(timestep) && timestep > 0.0);
    Vector<hpx::future<double>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityFinishGlobalKickAction>(child, timestep));
    }
    ALWAYS_ASSERT(detail::globalAccelerationCount == particlesSize());
    auto const& options = getOptions();
#if COSMICTIGER_WITH_SPH
    bool const midpointHydro = sphMidpointEnabled();
#else
    bool const midpointHydro = false;
#endif
    double const halfStep = (options.run.scenario == "glass" ? -0.5 : 0.5) * timestep;
    double kinetic = 0.0;
    FlopCount flops = 1;
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        particlesRung(i) = 0;
        double velocitySquared = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (!options.hydro.enabled || particlesType(i) != kSphType || midpointHydro) {
                particlesVel(dim, i) =
                    std::fma(detail::globalAccelerations[dim][i], halfStep, particlesVel(dim, i));
                flops += 5;    // Double FMA, two float inputs, and float output conversion.
            }
            double const velocity = particlesVel(dim, i);
            velocitySquared += velocity * velocity;
        }
        kinetic += 0.5 * particlesMass(i) * velocitySquared;
    }
    addCpuFlops(flops, FlopPrecision::Double);
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        freeManagedMemory(detail::globalAccelerations[dimension]);
        detail::globalAccelerations[dimension] = nullptr;
    }
    detail::globalAccelerationCount = 0;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        kinetic += future.get();
    }
    return kinetic;
}

HPX_PLAIN_ACTION(gravityBeginDeferredGasOpening, GravityBeginDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravitySealDeferredGasOpening, GravitySealDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravityFinishDeferredGasOpening, GravityFinishDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravityDiscardDeferredGasOpening, GravityDiscardDeferredGasOpeningAction);

bool gravityGasOpeningDeferred() {
    return detail::gasOpeningState == detail::GasOpeningState::Collecting;
}

Vector3<float>* gravityDeferredGasOpeningData() {
    ALWAYS_ASSERT(gravityGasOpeningDeferred());
    return detail::gasOpeningIncrements;
}

void gravityBeginDeferredGasOpening() {
    ALWAYS_ASSERT(detail::gasOpeningState == detail::GasOpeningState::Disabled);
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityBeginDeferredGasOpeningAction>(child));
    }
    try {
#if COSMICTIGER_WITH_SPH
        ALWAYS_ASSERT(sphMidpointEnabled() && !usesGlobalIntegration(getOptions().run));
        size_t const count = sphParticlesSize();
        auto* increments = count == 0 ?
            nullptr :
            static_cast<Vector3<float>*>(allocateManagedMemory(count * sizeof(Vector3<float>)));
        if (count != 0) {
            std::fill_n(increments, count, Vector3<float>(0.0f));
        }
        detail::gasOpeningIncrements = increments;
        detail::gasOpeningCount = count;
        detail::gasOpeningState = detail::GasOpeningState::Collecting;
#else
        THROW_ERROR("Deferred gas opening requires SPH support\n");
#endif
    } catch (...) {
        auto const failure = std::current_exception();
        hpx::wait_all(futures.begin(), futures.end());
        std::rethrow_exception(failure);
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

void gravitySealDeferredGasOpening() {
    ALWAYS_ASSERT(gravityGasOpeningDeferred());
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravitySealDeferredGasOpeningAction>(child));
    }
    try {
#if COSMICTIGER_WITH_SPH
        ALWAYS_ASSERT(detail::gasOpeningCount == size_t(sphParticlesSize()));
        detail::gasOpeningImpulses.reserve(detail::gasOpeningCount);
        for (ParticleInteger gasIndex = 0; gasIndex < sphParticlesSize(); ++gasIndex) {
            auto const increment = detail::gasOpeningIncrements[gasIndex];
            if (increment[kXi] != 0.0f || increment[kYi] != 0.0f || increment[kZi] != 0.0f) {
                detail::gasOpeningImpulses.push_back(
                    {sphParticlesParticleIndex(gasIndex), increment});
            }
        }
#endif
        freeManagedMemory(detail::gasOpeningIncrements);
        detail::gasOpeningIncrements = nullptr;
        detail::gasOpeningCount = 0;
        detail::gasOpeningParticleCount = particlesOwnedSize();
        detail::gasOpeningState = detail::GasOpeningState::Sealed;
    } catch (...) {
        auto const failure = std::current_exception();
        hpx::wait_all(futures.begin(), futures.end());
        std::rethrow_exception(failure);
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

KickReturn gravityFinishDeferredGasOpening() {
    ALWAYS_ASSERT(detail::gasOpeningState == detail::GasOpeningState::Sealed);
    ALWAYS_ASSERT(detail::gasOpeningParticleCount == size_t(particlesOwnedSize()));
    Vector<hpx::future<KickReturn>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityFinishDeferredGasOpeningAction>(child));
    }
    KickReturn result;
    // Derived split ordering at a coincident hydro event:
    // Δv_G = Σ_l s_l (Δτ_l/2) a_l;  vₒᵤₜ = vᵢₙ + Δv_G.
    // The signed hierarchical ladder supplies s_l; traditional gravity has one
    // opening term. Accumulation changes no momentum until the hydro operator
    // has completed. Diagnostic corrections retain the pre-opening state.
    // Δv_G -> velocity increment; vᵢₙ -> velocity before the gravity increment; vₒᵤₜ ->
    // velocity after the gravity increment.
    for (auto const& impulse : detail::gasOpeningImpulses) {
        ParticleInteger const index = impulse.particleIndex;
        ALWAYS_ASSERT(particlesType(index) == kSphType);
        Vector3<float> vᵢₙ;
        Vector3<float> vₒᵤₜ;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            vᵢₙ[dimension] = particlesVel(dimension, index);
            vₒᵤₜ[dimension] = vᵢₙ[dimension] + impulse.velocityIncrement[dimension];
            particlesVel(dimension, index) = vₒᵤₜ[dimension];
        }
        double const mass = particlesMass(index);
        double const kineticChange = 0.5 * mass * (vₒᵤₜ.dot(vₒᵤₜ) - vᵢₙ.dot(vᵢₙ));
        result.dkin += kineticChange;
        result.darkMatterKineticCorrection -= kineticChange;
        result.darkMatterXMomentumCorrection += mass * (vᵢₙ[kXi] - vₒᵤₜ[kXi]);
        result.darkMatterYMomentumCorrection += mass * (vᵢₙ[kYi] - vₒᵤₜ[kYi]);
        result.darkMatterZMomentumCorrection += mass * (vᵢₙ[kZi] - vₒᵤₜ[kZi]);
        result.darkMatterMomentumNormCorrection += mass * (vᵢₙ.norm() - vₒᵤₜ.norm());
        result.maxRung = std::max(result.maxRung, int(particlesRung(index)));
    }
    addCpuFlops(3 * detail::gasOpeningImpulses.size(), FlopPrecision::Single);
    Vector<detail::GasOpeningImpulse>().swap(detail::gasOpeningImpulses);
    detail::gasOpeningParticleCount = 0;
    detail::gasOpeningState = detail::GasOpeningState::Disabled;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

void gravityDiscardDeferredGasOpening() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<GravityDiscardDeferredGasOpeningAction>(child));
    }
    freeManagedMemory(detail::gasOpeningIncrements);
    detail::gasOpeningIncrements = nullptr;
    detail::gasOpeningCount = 0;
    detail::gasOpeningParticleCount = 0;
    Vector<detail::GasOpeningImpulse>().swap(detail::gasOpeningImpulses);
    detail::gasOpeningState = detail::GasOpeningState::Disabled;
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}
