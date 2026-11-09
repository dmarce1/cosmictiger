/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/ghosts.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/tree.hpp>

// Gate of docs/sph_parent_tree_plan.md milestone 2: with ghost leaves, every locality walks
// only its own tree and its ghosts, and the neighbor leaf lists equal those of the global
// walk with remote fetches, for every search direction, with no remote node read.

namespace detail {

void requireGhosts(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("SPH ghost regression: " + message);
    }
}

// Deterministic uniform deviate in [0, 1) from a counter (splitmix64).
double ghostDeviate(std::uint64_t counter) {
    std::uint64_t z = counter + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return double(z >> 11) * 0x1.0p-53;
}

constexpr int kGhostParticlesPerRank = 500;

// Random gas on every rank: positions over the whole box, so most particles migrate; a
// spread of supports, a few large; three rungs.
void initializeGhostParticles() {
    ParticleInteger const count = kGhostParticlesPerRank;
    particlesResize(count);
    sphParticlesResize(count, false);
    for (ParticleInteger i = 0; i < count; ++i) {
        std::uint64_t const base = (std::uint64_t(hpxRank()) << 32) + 16 * std::uint64_t(i);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = ghostDeviate(base + dim);
            particlesVel(dim, i) = 0.0f;
        }
        particlesRung(i) = char(int(3.0 * ghostDeviate(base + 4)));
        particlesType(i) = kSphType;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        double const draw = ghostDeviate(base + 5);
        sphParticlesSmoothingLength(i) = float(draw < 0.02 ? 0.2 : 0.03 + 0.06 * draw);
        sphParticlesSetInternalEnergy(i, 1.0f);
    }
}

// This locality's leaves with their neighbor leaves, sorted for comparison.
using NeighborTable = std::vector<std::pair<TreeId, std::vector<TreeId>>>;

NeighborTable localNeighborTable() {
    NeighborTable table;
    for (int index = 0; index < sphTreeLeaflistSize(); ++index) {
        auto const id = sphTreeGetLeaf(index);
        auto const& leaf = *sphTreeGetNode(id);
        std::vector<TreeId> neighbors;
        for (int k = leaf.neighborRange.first; k < leaf.neighborRange.second; ++k) {
            neighbors.push_back(sphTreeGetNeighbor(k));
        }
        std::sort(neighbors.begin(), neighbors.end());
        requireGhosts(std::adjacent_find(neighbors.begin(), neighbors.end()) == neighbors.end(),
            "a neighbor list repeats a leaf");
        table.emplace_back(id, std::move(neighbors));
    }
    std::sort(
        table.begin(), table.end(), [](auto const& a, auto const& b) { return a.first < b.first; });
    return table;
}

std::int64_t localRemoteNodeReads() {
    return sphTreeRemoteNodeReads();
}

void resetLocalRemoteNodeReads() {
    sphTreeResetRemoteNodeReads();
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeGhostParticles, InitializeGhostParticlesAction);
HPX_PLAIN_ACTION(detail::localNeighborTable, GhostLocalNeighborTableAction);
HPX_PLAIN_ACTION(detail::localRemoteNodeReads, GhostLocalRemoteNodeReadsAction);
HPX_PLAIN_ACTION(detail::resetLocalRemoteNodeReads, GhostResetRemoteNodeReadsAction);

namespace detail {

std::vector<NeighborTable> gatherNeighborTables() {
    std::vector<hpx::future<NeighborTable>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<GhostLocalNeighborTableAction>(locality));
    }
    std::vector<NeighborTable> tables;
    for (auto& future : futures) {
        tables.push_back(future.get());
    }
    return tables;
}

std::int64_t remoteNodeReads() {
    std::int64_t reads = 0;
    std::vector<hpx::future<std::int64_t>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<GhostLocalRemoteNodeReadsAction>(locality));
    }
    for (auto& future : futures) {
        reads += future.get();
    }
    return reads;
}

void resetRemoteNodeReads() {
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<GhostResetRemoteNodeReadsAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void findNeighbors(TreeId root, SphInteractionDirection direction) {
    SphTreeNeighborParameters parameters{};
    parameters.operation = SphNeighborOperation::FindNeighbors;
    parameters.interactionDirection = direction;
    parameters.minimumRung = 1;
    sphTreeNeighbor(parameters, root, Vector<TreeId>{root}).get();
}

void refreshBounds(TreeId root, SphParticleSelection outer) {
    SphTreeNeighborParameters parameters{};
    parameters.operation = SphNeighborOperation::RefreshBounds;
    parameters.smoothingLengthMultiplier = 1.21f;
    parameters.minimumRung = 1;
    parameters.innerParticleSelection = SphParticleSelection::All;
    parameters.outerParticleSelection = outer;
    sphTreeNeighbor(parameters, root, Vector<TreeId>{}).get();
}

}    // namespace detail

void sphGhostsTest() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.hydro.enabled = true;
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.sphBucketSize = 8;
    options.initial.numberOfParticles = detail::kGhostParticlesPerRank * hpxSize();
    setOptions(options);
    std::vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeGhostParticlesAction>(locality));
    }
    for (auto& future : initialization) {
        future.get();
    }
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    SphTreeCreateParameters parameters{};
    parameters.minimumRung = 1;
    parameters.smoothingLengthMultiplier = 1.21;
    auto const built = sphTreeCreate(parameters);
    int comparisons = 0;
    std::int64_t remoteLeafPairs = 0;
    struct Case {
        SphParticleSelection outer;
        SphInteractionDirection direction;
        char const* name;
    };
    Case const cases[] = {
        {SphParticleSelection::Active, SphInteractionDirection::TargetSupport, "target, active"},
        {SphParticleSelection::Active, SphInteractionDirection::SourceSupport, "source, active"},
        {SphParticleSelection::All, SphInteractionDirection::EitherSupport, "either, all"},
        {SphParticleSelection::All, SphInteractionDirection::TargetSupport, "target, all"},
    };
    for (auto const& testCase : cases) {
        detail::refreshBounds(built.id, testCase.outer);
        sphGhostsSetEnabled(false);
        detail::findNeighbors(built.id, testCase.direction);
        auto const fetched = detail::gatherNeighborTables();
        sphGhostsSetEnabled(true);
        detail::resetRemoteNodeReads();
        detail::findNeighbors(built.id, testCase.direction);
        auto const ghosted = detail::gatherNeighborTables();
        auto const reads = detail::remoteNodeReads();
        if (hpxSize() > 1) {
            detail::requireGhosts(reads == 0,
                std::format("{}: the ghost walk read {} remote nodes", testCase.name, reads));
        }
        detail::requireGhosts(fetched.size() == ghosted.size(), "locality count changed");
        for (size_t rank = 0; rank < fetched.size(); ++rank) {
            detail::requireGhosts(fetched[rank].size() == ghosted[rank].size(),
                std::format("{}: rank {} has a different leaf count", testCase.name, rank));
            for (size_t leaf = 0; leaf < fetched[rank].size(); ++leaf) {
                auto const& [id, expected] = fetched[rank][leaf];
                auto const& [ghostId, actual] = ghosted[rank][leaf];
                detail::requireGhosts(id == ghostId, "leaf identities differ");
                detail::requireGhosts(expected == actual,
                    std::format("{}: leaf ({}, {}) has {} neighbors with fetches and {} with "
                                "ghosts",
                        testCase.name, id.proc, id.index, expected.size(), actual.size()));
                for (auto const& neighbor : expected) {
                    remoteLeafPairs += neighbor.proc != id.proc;
                }
                comparisons++;
            }
        }
    }
    if (hpxSize() > 1) {
        detail::requireGhosts(remoteLeafPairs > 0, "the fixture has no cross-locality leaf pair");
    }
    auto const statistics = sphGhostsStatistics();
    std::cout << std::format("SPH ghost walk equals the fetch walk: ranks={} leaf lists={} "
                             "remote pairs={} ghost leaves={} ghost particles={}\n",
                     hpxSize(), comparisons, remoteLeafPairs, statistics.leaves,
                     statistics.particles)
              << std::flush;
    sphGhostsClear();
    sphTreeDestroy(true);
    setOptions(savedOptions);
}
