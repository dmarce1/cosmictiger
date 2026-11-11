/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <exception>
#include <memory>
#include <stack>
#include <stdexcept>
#include <utility>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
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

/// Take reusable interaction lists from the thread-local pool, creating a set when
/// necessary.
KickLists kickAcquireLists() {
    if (localLists.empty()) {
        localLists.push(KickLists());
    }
    KickLists lists = std::move(localLists.top());
    localLists.pop();
    return std::move(lists);
}

/// Clear interaction-list lengths and return their allocated storage to the thread-local
/// pool.
void kickReleaseLists(KickLists&& lists) {
    lists.cclist.resize(0);
    lists.cplist.resize(0);
    lists.pclist.resize(0);
    lists.nextlist.resize(0);
    lists.leaflist.resize(0);
    localLists.push(std::move(lists));
}

/// Test whether every source in the checklist belongs to a single-locality tree node.
bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

}    // namespace detail

/// Invoke the tree walk at the build order.
KickReturn kickTreeWalk(KickParameters params, ExpansionBuffer l, Vector3<Position> pos,
    TreeId self, Vector<TreeId> direct, std::shared_ptr<KickWorkspace> workspace,
    KickContext const& context) {
    return kickWalk<kFmmOrder>(
        params, std::move(l), pos, self, std::move(direct), std::move(workspace), context);
}

HPX_PLAIN_ACTION(gravityBeginDeferredGasOpening, GravityBeginDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravitySealDeferredGasOpening, GravitySealDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravityFinishDeferredGasOpening, GravityFinishDeferredGasOpeningAction);
HPX_PLAIN_ACTION(gravityDiscardDeferredGasOpening, GravityDiscardDeferredGasOpeningAction);

/// Report whether gas opening-kick increments are currently being collected.
bool gravityGasOpeningDeferred() {
    return detail::gasOpeningState == detail::GasOpeningState::Collecting;
}

/// Expose the gas-indexed increment buffer while deferred collection is active.
Vector3<float>* gravityDeferredGasOpeningData() {
    ALWAYS_ASSERT(gravityGasOpeningDeferred());
    return detail::gasOpeningIncrements;
}

/// Allocate zeroed gas opening-kick buffers and begin deferred collection across
/// localities.
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
        throw std::invalid_argument("Deferred gas opening requires SPH support");
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

/// Convert collected gas increments to stable base-particle impulses before gas reordering.
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

/// Apply sealed opening impulses after the coincident hydro event and return their kick
/// diagnostics.
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

/// Release deferred opening state throughout the locality tree without applying its
/// impulses.
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
