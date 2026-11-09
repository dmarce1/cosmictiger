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

// One leaf's identity and stored boxes, for the brute-force reference.
struct GhostLeafBoxes {
    TreeId id;
    Fixed32Range innerBox;
    Fixed32Range outerBox;

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & id;
        archive & innerBox;
        archive & outerBox;
    }
};

// This locality's leaves, read on their owner so the reference never touches a ghost.
std::vector<GhostLeafBoxes> localLeafBoxes() {
    std::vector<GhostLeafBoxes> leaves;
    for (int index = 0; index < sphTreeLeaflistSize(); ++index) {
        GhostLeafBoxes record;
        record.id = sphTreeGetLeaf(index);
        auto const& leaf = *sphTreeGetNode(record.id);
        record.innerBox = leaf.innerBox;
        record.outerBox = leaf.outerBox;
        leaves.push_back(record);
    }
    return leaves;
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
HPX_PLAIN_ACTION(detail::localLeafBoxes, GhostLocalLeafBoxesAction);
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

// Brute-force neighbor leaf lists: every leaf of every locality against every leaf of every
// locality with the walk's leaf test, target support outer(s) ∩ inner(l) and source support
// inner(s) ∩ outer(l). Node boxes contain their children's, so the walk opens every ancestor
// of such a leaf and its lists must equal these exactly.
std::vector<NeighborTable> bruteForceNeighborTables(SphInteractionDirection direction) {
    std::vector<hpx::future<std::vector<GhostLeafBoxes>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<GhostLocalLeafBoxesAction>(locality));
    }
    std::vector<std::vector<GhostLeafBoxes>> leavesByRank;
    for (auto& future : futures) {
        leavesByRank.push_back(future.get());
    }
    bool const target = direction == SphInteractionDirection::TargetSupport ||
        direction == SphInteractionDirection::EitherSupport;
    bool const source = direction == SphInteractionDirection::SourceSupport ||
        direction == SphInteractionDirection::EitherSupport;
    std::vector<NeighborTable> tables(leavesByRank.size());
    for (size_t rank = 0; rank < leavesByRank.size(); ++rank) {
        for (auto const& self : leavesByRank[rank]) {
            std::vector<TreeId> neighbors;
            for (auto const& others : leavesByRank) {
                for (auto const& other : others) {
                    if ((target && self.outerBox.periodicIntersects(other.innerBox)) ||
                        (source && self.innerBox.periodicIntersects(other.outerBox))) {
                        neighbors.push_back(other.id);
                    }
                }
            }
            std::sort(neighbors.begin(), neighbors.end());
            tables[rank].emplace_back(self.id, std::move(neighbors));
        }
        std::sort(tables[rank].begin(), tables[rank].end(),
            [](auto const& a, auto const& b) { return a.first < b.first; });
    }
    return tables;
}

// Every locality's walked lists equal the brute-force lists.
std::int64_t compareNeighborTables(std::vector<NeighborTable> const& expected,
    std::vector<NeighborTable> const& actual, char const* name) {
    std::int64_t remoteLeafPairs = 0;
    requireGhosts(expected.size() == actual.size(), "locality count changed");
    for (size_t rank = 0; rank < expected.size(); ++rank) {
        requireGhosts(expected[rank].size() == actual[rank].size(),
            std::format("{}: rank {} has a different leaf count", name, rank));
        for (size_t leaf = 0; leaf < expected[rank].size(); ++leaf) {
            auto const& [id, wanted] = expected[rank][leaf];
            auto const& [walkedId, walked] = actual[rank][leaf];
            requireGhosts(id == walkedId, "leaf identities differ");
            requireGhosts(wanted == walked,
                std::format("{}: leaf ({}, {}) has {} neighbors by brute force and {} walked", name,
                    id.proc, id.index, wanted.size(), walked.size()));
            for (auto const& neighbor : wanted) {
                remoteLeafPairs += neighbor.proc != id.proc;
            }
        }
    }
    return remoteLeafPairs;
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
        detail::compareNeighborTables(
            detail::bruteForceNeighborTables(testCase.direction), ghosted, testCase.name);
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
