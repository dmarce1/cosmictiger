/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// Gate of docs/sph_parent_tree_plan.md milestones 2 and 5: every locality walks only its own
// tree and its ghost leaves, and the neighbor leaf lists equal a brute-force test of every
// leaf against every leaf of every locality, for every search direction. A remote node that
// is not a ghost cannot be read at all.

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

// Every pushed field of one owner leaf, as the gas readers stage it.
struct GhostLeafFields {
    int owner = -1;
    Pair<ParticleInteger> range;
    Vector<Position> x;
    Vector<Position> y;
    Vector<Position> z;
    Vector<float> internalEnergy;
    Vector<float> smoothingLength;
    Vector<char> rung;
    Vector<float> vx;
    Vector<float> vy;
    Vector<float> vz;
    Vector<float> density;
    Vector<float> balsaraFactor;
    Vector<float> bulkViscosityCoefficient;
    Vector<float> gradHCorrection;

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & owner & range & x & y & z & internalEnergy & smoothingLength & rung & vx & vy &
            vz & density & balsaraFactor & bulkViscosityCoefficient & gradHCorrection;
    }

    // The first field whose bytes differ, or nullptr. Bytes, not values: an unbuilt grad-h
    // correction is NaN, and a ghost copy must carry it unchanged.
    char const* firstDifference(GhostLeafFields const& other) const {
        auto const same = [](auto const& a, auto const& b) {
            return a.size() == b.size() &&
                (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0);
        };
        if (owner != other.owner || range != other.range) {
            return "leaf";
        }
        if (!same(x, other.x) || !same(y, other.y) || !same(z, other.z)) {
            return "positions";
        }
        if (!same(internalEnergy, other.internalEnergy)) {
            return "internal energies";
        }
        if (!same(smoothingLength, other.smoothingLength)) {
            return "smoothing lengths";
        }
        if (!same(rung, other.rung)) {
            return "rungs";
        }
        if (!same(vx, other.vx) || !same(vy, other.vy) || !same(vz, other.vz)) {
            return "velocities";
        }
        if (!same(density, other.density) || !same(balsaraFactor, other.balsaraFactor) ||
            !same(bulkViscosityCoefficient, other.bulkViscosityCoefficient) ||
            !same(gradHCorrection, other.gradHCorrection)) {
            return "densities";
        }
        return nullptr;
    }
};

// Read one owner leaf through the gas readers: its owner reads local gas, an importer reads
// its ghost copy inside a staging bracket.
GhostLeafFields readLeafFields(int owner, Pair<ParticleInteger> range) {
    GhostLeafFields fields;
    fields.owner = owner;
    fields.range = range;
    size_t const count = range.second - range.first;
    for (auto* field : {&fields.x, &fields.y, &fields.z}) {
        field->resize(count);
    }
    for (auto* field : {&fields.internalEnergy, &fields.smoothingLength, &fields.vx, &fields.vy,
             &fields.vz, &fields.density, &fields.balsaraFactor, &fields.bulkViscosityCoefficient,
             &fields.gradHCorrection}) {
        field->resize(count);
    }
    fields.rung.resize(count);
    ParticleGlobalRange global;
    global.proc = owner;
    global.range = range;
    bool const remote = owner != hpxRank();
    if (remote) {
        sphGhostsBeginStaging();
    }
    try {
        sphParticlesGlobalReadPositions(
            global, fields.x.data(), fields.y.data(), fields.z.data(), 0);
        sphParticlesGlobalReadInternalEnergyAndSmoothingLength(
            global, fields.internalEnergy.data(), fields.smoothingLength.data(), 0);
        sphParticlesGlobalReadRungs(global, fields.rung.data(), 0);
        sphParticlesGlobalReadVelocities(
            global, fields.vx.data(), fields.vy.data(), fields.vz.data(), 0);
        sphParticlesGlobalReadDensity(global, fields.density.data(), 0, fields.balsaraFactor.data(),
            fields.bulkViscosityCoefficient.data(), fields.gradHCorrection.data());
    } catch (...) {
        if (remote) {
            sphGhostsEndStaging();
        }
        throw;
    }
    if (remote) {
        sphGhostsEndStaging();
    }
    return fields;
}

// This locality's ghost copies of every ghost leaf.
std::vector<GhostLeafFields> localGhostCopies() {
    std::vector<GhostLeafFields> copies;
    for (auto const id : sphGhostsLeafIds()) {
        auto const& leaf = *sphGhostsFindNode(id);
        copies.push_back(readLeafFields(id.proc, leaf.particleRange));
    }
    return copies;
}

// The owner's current values of the given local leaves.
std::vector<GhostLeafFields> ownerLeafFields(std::vector<Pair<ParticleInteger>> ranges) {
    std::vector<GhostLeafFields> fields;
    for (auto const& range : ranges) {
        fields.push_back(readLeafFields(hpxRank(), range));
    }
    return fields;
}

// Whether reading every ghost leaf's velocities throws, as it must when the last refresh did
// not carry them.
bool ghostVelocitiesThrow() {
    for (auto const id : sphGhostsLeafIds()) {
        auto const& leaf = *sphGhostsFindNode(id);
        ParticleGlobalRange global;
        global.proc = id.proc;
        global.range = leaf.particleRange;
        std::vector<float> v(leaf.numberOfParticles());
        sphGhostsBeginStaging();
        bool threw = false;
        try {
            sphParticlesGlobalReadVelocities(global, v.data(), v.data(), v.data(), 0);
        } catch (std::exception const&) {
            threw = true;
        }
        sphGhostsEndStaging();
        if (!threw) {
            return false;
        }
    }
    return true;
}

// Change every third particle's support, velocity and internal energy, and nudge every fifth
// position within its leaf, so a refresh has some changed and many unchanged slots.
void perturbGhostSources(int round) {
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        if ((i + round) % 3 == 0) {
            sphParticlesSmoothingLength(i) *= 1.01f;
            sphParticlesVelocity(kXi, i) += 0.001f * float(round + 1);
            sphParticlesSetInternalEnergy(i, 1.0f + 0.01f * float(round + 1));
        }
        if ((i + round) % 5 == 0) {
            sphParticlesPosition(kYi, i) += Position(1.0e-7);
        }
    }
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeGhostParticles, InitializeGhostParticlesAction);
HPX_PLAIN_ACTION(detail::localGhostCopies, GhostLocalCopiesAction);
HPX_PLAIN_ACTION(detail::ownerLeafFields, GhostOwnerLeafFieldsAction);
HPX_PLAIN_ACTION(detail::ghostVelocitiesThrow, GhostVelocitiesThrowAction);
HPX_PLAIN_ACTION(detail::perturbGhostSources, PerturbGhostSourcesAction);
HPX_PLAIN_ACTION(detail::localNeighborTable, GhostLocalNeighborTableAction);
HPX_PLAIN_ACTION(detail::localLeafBoxes, GhostLocalLeafBoxesAction);

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

// Every importer's ghost copy of every leaf equals its owner's current values bit for bit.
std::int64_t verifyGhostCopies(char const* stage) {
    std::vector<hpx::future<std::vector<GhostLeafFields>>> copies;
    for (auto const& locality : hpxLocalities()) {
        copies.push_back(hpx::async<GhostLocalCopiesAction>(locality));
    }
    std::int64_t particles = 0;
    for (auto& future : copies) {
        auto const held = future.get();
        std::vector<std::vector<Pair<ParticleInteger>>> ranges(hpxSize());
        for (auto const& copy : held) {
            ranges[copy.owner].push_back(copy.range);
        }
        std::vector<std::vector<GhostLeafFields>> truth(hpxSize());
        for (int owner = 0; owner < hpxSize(); ++owner) {
            if (!ranges[owner].empty()) {
                truth[owner] =
                    hpx::async<GhostOwnerLeafFieldsAction>(hpxLocalities()[owner], ranges[owner])
                        .get();
            }
        }
        std::vector<size_t> next(hpxSize(), 0);
        for (auto const& copy : held) {
            auto const& expected = truth[copy.owner][next[copy.owner]++];
            char const* const difference = copy.firstDifference(expected);
            requireGhosts(difference == nullptr,
                std::format("{}: the {} of a ghost copy of rank {} gas [{}, {}) differ from its "
                            "owner's",
                    stage, difference ? difference : "", copy.owner, copy.range.first,
                    copy.range.second));
            particles += copy.range.second - copy.range.first;
        }
    }
    return particles;
}

void perturbSources(int round) {
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<PerturbGhostSourcesAction>(locality, round));
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
        detail::findNeighbors(built.id, testCase.direction);
        auto const walked = detail::gatherNeighborTables();
        remoteLeafPairs += detail::compareNeighborTables(
            detail::bruteForceNeighborTables(testCase.direction), walked, testCase.name);
        for (auto const& table : walked) {
            comparisons += table.size();
        }
    }
    if (hpxSize() > 1) {
        // Changed-set pushes (M6): copies equal their owners after full, sparse and kept-block
        // refreshes, a group outside the last refresh cannot be read, and an unchanged source
        // set costs no field bytes.
        constexpr unsigned kTestFields = kSphGhostAllFields & ~kSphGhostRungCandidates;
        sphGhostsResetStatistics();
        sphGhostsRefresh(kTestFields);
        auto const copied = detail::verifyGhostCopies("first refresh");
        detail::requireGhosts(copied > 0, "the fixture holds no ghost particles");
        auto const first = sphGhostsStatistics();
        detail::requireGhosts(first.fieldBytes == first.neededFieldBytes,
            "a first refresh did not send every requested slot");
        sphGhostsRefresh(kTestFields);
        detail::verifyGhostCopies("unchanged refresh");
        auto const unchanged = sphGhostsStatistics();
        detail::requireGhosts(
            unchanged.fieldBytes == first.fieldBytes, "an unchanged refresh sent field bytes");
        detail::perturbSources(0);
        sphGhostsRefresh(kTestFields);
        detail::verifyGhostCopies("sparse refresh");
        auto const sparse = sphGhostsStatistics();
        std::int64_t const sparseBytes = sparse.fieldBytes - unchanged.fieldBytes;
        detail::requireGhosts(sparseBytes > 0 && sparseBytes < first.fieldBytes,
            "a partial change was not sent as a partial update");
        sphGhostsRefresh(kSphGhostPositions);
        bool allThrew = true;
        for (auto const& locality : hpxLocalities()) {
            allThrew = allThrew && hpx::async<GhostVelocitiesThrowAction>(locality).get();
        }
        detail::requireGhosts(allThrew, "velocities outside the last refresh were readable");
        // A walk resyncs every export; unchanged exports keep their blocks and fields.
        detail::perturbSources(1);
        detail::findNeighbors(built.id, SphInteractionDirection::TargetSupport);
        auto const beforeResyncRefresh = sphGhostsStatistics();
        sphGhostsRefresh(kTestFields);
        detail::verifyGhostCopies("refresh after a resync");
        auto const afterResyncRefresh = sphGhostsStatistics();
        // The same boxes give the same exports, so every block is kept and only the perturbed
        // particles are sent, not a dense resend of every requested slot.
        detail::requireGhosts(afterResyncRefresh.fieldBytes - beforeResyncRefresh.fieldBytes <
                afterResyncRefresh.neededFieldBytes - beforeResyncRefresh.neededFieldBytes,
            "a resync with unchanged exports did not keep the ghost blocks");
        std::cout << std::format("SPH ghost changed-set pushes: first={} sparse={} bytes\n",
                         first.fieldBytes, sparseBytes)
                  << std::flush;
        detail::requireGhosts(remoteLeafPairs > 0, "the fixture has no cross-locality leaf pair");
        // There is no fetch path: a remote node that is not a ghost leaf, and remote gas read
        // outside the staging of a pass, must both throw.
        TreeId missing;
        missing.proc = 1;
        missing.index = 0;
        while (sphGhostsFindNode(missing) != nullptr) {
            missing.index++;
        }
        bool nodeThrew = false;
        try {
            sphTreeGetNode(missing);
        } catch (std::exception const&) {
            nodeThrew = true;
        }
        detail::requireGhosts(nodeThrew, "a remote node that is not a ghost was readable");
        ParticleGlobalRange remote;
        remote.proc = 1;
        remote.range = Pair<ParticleInteger>(0, 1);
        Position x;
        Position y;
        Position z;
        bool gasThrew = false;
        try {
            sphParticlesGlobalReadPositions(remote, &x, &y, &z, 0);
        } catch (std::exception const&) {
            gasThrew = true;
        }
        detail::requireGhosts(gasThrew, "remote gas was readable outside staging");
    }
    auto const statistics = sphGhostsStatistics();
    std::cout << std::format("SPH ghost walk equals brute force: ranks={} leaf lists={} "
                             "remote pairs={} ghost leaves={} ghost particles={}\n",
                     hpxSize(), comparisons, remoteLeafPairs, statistics.leaves,
                     statistics.particles)
              << std::flush;
    sphGhostsClear();
    sphTreeDestroy(true);
    setOptions(savedOptions);
}
