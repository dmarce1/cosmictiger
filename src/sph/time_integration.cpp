/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/ghosts.hpp>
#include <cosmictiger/sph/parent_tree.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#include <cosmictiger/sph/tree.hpp>

namespace detail {

static bool midpointEventActive = false;
static bool rungBalanceActive = false;
static char* rungCandidates = nullptr;
static Vector<char> rungSnapshot;

}    // namespace detail

void sphMidpointSetEventActive(bool active);
HPX_PLAIN_ACTION(sphMidpointSetEventActive, SphMidpointSetEventActiveAction);
HPX_PLAIN_ACTION(sphMidpointTargetCount, SphMidpointTargetCountAction);
HPX_PLAIN_ACTION(sphParticlesSelectMidpointRungs, SphParticlesSelectMidpointRungsAction);
HPX_PLAIN_ACTION(sphSeedRungProposals, SphSeedRungProposalsAction);
HPX_PLAIN_ACTION(sphCommitBalancedRungs, SphCommitBalancedRungsAction);

/// Seed each gas particle's proposed rung from its current rung before neighbor balancing.
void sphSeedRungProposals() {
    ALWAYS_ASSERT(!detail::rungBalanceActive);
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphSeedRungProposalsAction>(child));
    }
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        sphParticlesSetProposedRung(gas, sphParticlesRung(gas));
    }
    for (auto& child : children) {
        child.get();
    }
}

/// Install balanced gas rung proposals and return the global maximum rung.
int sphCommitBalancedRungs() {
    ALWAYS_ASSERT(!detail::rungBalanceActive);
    Vector<hpx::future<int>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphCommitBalancedRungsAction>(child));
    }
    int maximumRung = 0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto const candidate = sphParticlesProposedRung(gas);
        ALWAYS_ASSERT(candidate >= 0 && candidate < kMaxRung);
        sphParticlesRung(gas) = candidate;
        maximumRung = std::max(maximumRung, int(candidate));
    }
    for (auto& child : children) {
        maximumRung = std::max(maximumRung, child.get());
    }
    return maximumRung;
}

void sphPrepareRungCandidates(bool initialize);
SphRungBalanceReturn sphFinishRungCandidates(bool publish);
HPX_PLAIN_ACTION(sphPrepareRungCandidates, SphPrepareRungCandidatesAction);
HPX_PLAIN_ACTION(sphFinishRungCandidates, SphFinishRungCandidatesAction);

/// Report whether rung-candidate balancing storage is active.
bool sphRungBalanceActive() {
    return detail::rungBalanceActive;
}

/// Expose the base-particle-indexed candidate rung buffer for the current balance pass.
char* sphRungCandidatesData() {
    return detail::rungCandidates;
}

/// Allocate candidate storage when requested and snapshot gas candidates for a synchronous
/// balance pass.
void sphPrepareRungCandidates(bool initialize) {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphPrepareRungCandidatesAction>(child, initialize));
    }
    if (initialize) {
        ALWAYS_ASSERT(!detail::rungBalanceActive);
        detail::rungBalanceActive = true;
        detail::rungSnapshot.resize(sphParticlesSize());
        if (particlesSize() != 0) {
            detail::rungCandidates = static_cast<char*>(allocateManagedMemory(particlesSize()));
        }
    }
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto const particle = sphParticlesParticleIndex(gas);
        if (initialize) {
            auto const seed = sphParticlesProposedRung(gas);
            ALWAYS_ASSERT(seed >= 0 && seed < kMaxRung);
            detail::rungCandidates[particle] = seed;
        }
        detail::rungSnapshot[gas] = detail::rungCandidates[particle];
    }
    for (auto& child : children) {
        child.get();
    }
}

/// Report whether candidate snapshots are available for neighbor reads.
bool sphRungCandidatesActive() {
    return detail::rungBalanceActive;
}

/// Copy a validated range from the current local gas rung snapshot.
Vector<char> sphFetchRungCandidates(Pair<ParticleInteger> range) {
    ALWAYS_ASSERT(detail::rungBalanceActive && range.first >= 0 &&
        range.second <= detail::rungSnapshot.size());
    return Vector<char>(
        detail::rungSnapshot.begin() + range.first, detail::rungSnapshot.begin() + range.second);
}

/// Read candidate rungs from the local snapshot or staged remote ghost fields.
void sphReadRungCandidates(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    if (range.proc == hpxRank()) {
        ALWAYS_ASSERT(detail::rungBalanceActive);
        std::copy(detail::rungSnapshot.begin() + range.range.first,
            detail::rungSnapshot.begin() + range.range.second, rungs + offset);
    } else {
        // Remote candidates arrive with the ghost fields (docs/sph_parent_tree_plan.md, M5).
        sphGhostsRequireStaging(range);
        sphGhostsReadRungCandidates(range, rungs, offset);
    }
}

/// Publish candidate proposals and reduce changes, or release candidate storage when ending
/// the balance.
SphRungBalanceReturn sphFinishRungCandidates(bool publish) {
    Vector<hpx::future<SphRungBalanceReturn>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphFinishRungCandidatesAction>(child, publish));
    }
    SphRungBalanceReturn result;
    if (publish) {
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            auto const candidate = detail::rungCandidates[sphParticlesParticleIndex(gas)];
            result.changedParticles += candidate != sphParticlesProposedRung(gas);
            result.maximumRung = std::max(result.maximumRung, int(candidate));
            sphParticlesSetProposedRung(gas, candidate);
        }
    }
    if (detail::rungCandidates != nullptr) {
        freeManagedMemory(detail::rungCandidates);
        detail::rungCandidates = nullptr;
    }
    detail::rungSnapshot.clear();
    detail::rungBalanceActive = false;
    for (auto& child : children) {
        auto const remote = child.get();
        result.maximumRung = std::max(result.maximumRung, remote.maximumRung);
        result.changedParticles += remote.changedParticles;
    }
    return result;
}

/// Iterate neighbor rung closure to a bounded fixed point and clean up candidate state on
/// failure.
SphRungBalanceReturn sphBalanceRungs(TreeId root, float scaleFactor, bool cuda) {
    auto const& opts = getOptions();
    ALWAYS_ASSERT(hpxRank() == 0 && sphTreeAllValid());
    SphTreeNeighborParameters neighbors;
    neighbors.operation = SphNeighborOperation::RefreshBounds;
    neighbors.innerParticleSelection = neighbors.outerParticleSelection = SphParticleSelection::All;
    sphTreeNeighbor(neighbors, root, {}).get();
    neighbors.operation = SphNeighborOperation::FindNeighbors;
    neighbors.interactionDirection = SphInteractionDirection::EitherSupport;
    sphTreeNeighbor(neighbors, root, Vector<TreeId>{root}).get();

    try {
        sphPrepareRungCandidates(true);
        SphRunParameters parameters;
        parameters.runType = kSphRunRungs;
        parameters.phase = 0;
        parameters.minimumRung = 0;
        parameters.baseTimestep = 1.0f;
        parameters.scaleFactor = scaleFactor;
        parameters.expansionRate = 0.0f;
        parameters.courantFactor = opts.hydro.cfl;
        // Locally derived graph closure:
        //   r_i* = max_j (r_j^0 - dist(i,j), 0).
        // r_j^0 -> caller proposal; dist -> touching-kernel graph distance.
        // Each Jacobi pass crosses one edge. Since 0 <= r < kMaxRung, at most
        // kMaxRung-1 propagation passes and one unchanged pass suffice.
        for (int pass = 1; pass <= kMaxRung; ++pass) {
            auto const result = sphRun(parameters, cuda);
            if (!result.needsAnotherPass) {
                auto balanced = sphFinishRungCandidates(true);
                balanced.passes = pass;
                return balanced;
            }
            sphPrepareRungCandidates(false);
        }
        throw std::runtime_error("SPH rung closure did not reach its bounded fixed point");
    } catch (...) {
        sphFinishRungCandidates(false);
        throw;
    }
}

/// Report whether the resolved hydro options select the SPH midpoint event scheme.
bool sphMidpointEnabled() {
    auto const& options = getOptions();
    return options.hydro.enabled && options.hydro.method == "sph";
}

/// Report whether a midpoint source snapshot event is currently active.
bool sphMidpointEventActive() {
    return detail::midpointEventActive;
}

/// Propagate midpoint-event activity through the SPH collective topology.
void sphMidpointSetEventActive(bool active) {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphMidpointSetEventActiveAction>(child, active));
    }
    detail::midpointEventActive = active;
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        child.get();
    }
}

/// Require an idle midpoint integrator and mark the event active on all localities.
void sphMidpointBeginEvent() {
    ALWAYS_ASSERT(sphMidpointEnabled() && !detail::midpointEventActive);
    sphMidpointSetEventActive(true);
}

/// Clear midpoint-event activity on all localities.
void sphMidpointEndEvent() {
    sphMidpointSetEventActive(false);
}

/// Count gas particles on the exact requested rung, or all gas for a negative rung.
size_t sphMidpointTargetCount(int exactRung) {
    Vector<hpx::future<size_t>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphMidpointTargetCountAction>(child, exactRung));
    }
    size_t count = 0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        count += exactRung < 0 || sphParticlesRung(gas) == exactRung;
    }
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        count += child.get();
    }
    return count;
}

/// Choose next gas rungs from hydro bounds and allowed coarsening, returning the global
/// maximum.
int sphParticlesSelectMidpointRungs(int minimumRung) {
    auto const& opts = getOptions();
    ALWAYS_ASSERT(sphMidpointEnabled() && minimumRung >= 0);
    Vector<hpx::future<int>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphParticlesSelectMidpointRungsAction>(child, minimumRung));
    }
    int maximumRung = 0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto& rung = sphParticlesRung(gas);
        if (rung >= minimumRung) {
            int const nextRung = std::max(
                {minimumRung, int(rung) - 1, sphParticlesHydroRung(gas), opts.run.minRung});
            ALWAYS_ASSERT(nextRung >= 0 && nextRung < kMaxRung);
            rung = nextRung;
        }
        maximumRung = std::max(maximumRung, int(rung));
    }
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        maximumRung = std::max(maximumRung, child.get());
    }
    return maximumRung;
}
