/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

// Full PMMM mesh of the fixtures. Seam particles straddle its cell planes; the
// coarse meshes below cut the particle domains on a subset of those planes.
constexpr int kMeshDomainMeshSize = 16;
constexpr int kMeshDomainNearWidth = 1;
constexpr int kMeshDomainCoarseMeshSizes[] = {4, 8};
constexpr int kMeshDomainFixtures = 4;
constexpr int kMeshDomainDenseFixture = 2;

void requireMeshDomain(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Mesh domain regression: ") + message);
    }
}

void migrateMeshDomainFixture() {
    // These ownership checks move every rung. Hierarchical builds migrate one
    // rung per call; ordinary leapfrog migrates the whole population together.
    int const lastRung = getOptions().run.hierarchicalGravity ? 7 : 0;
    for (int rung = 0; rung <= lastRung; ++rung) {
        domainsBegin(rung);
        domainsEnd();
    }
}

int meshDomainFixtureCount(int fixture) {
    return fixture == 0 ? 126 : fixture == 1 ? 112 : fixture == kMeshDomainDenseFixture ? 37 : 0;
}

PositionInteger meshDomainPosition(int identity, int dim, int fixture, int phase) {
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << kPositionBits;
    unsigned __int128 const cellWidth = period / kMeshDomainMeshSize;
    unsigned __int128 raw;
    if (fixture == 0) {
        // Seam particles sit one unit below, exactly on, and one unit above the
        // full-mesh planes, so the half-open cell classification is exercised
        // on both sides of every coarse plane. Every 21st particle straddles
        // the periodic seam at 0 / 1.
        int const plane = 1 + ((identity / 3 + 2 * dim) % (kMeshDomainMeshSize - 1));
        raw = cellWidth * plane;
        int const offset = (identity + dim) % 3 - 1;
        raw = offset < 0 ? raw - 1 : raw + offset;
        if (identity % 21 == 0) {
            raw = dim % 2 == 0 ? 0 : period - 1;
        }
    } else if (fixture == kMeshDomainDenseFixture) {
        // Distinct coordinates inside one full-mesh cell, hence inside one cell
        // of every coarse mesh. Aligned cuts can never separate them.
        int const count = meshDomainFixtureCount(fixture);
        int const multiplier = dim == kXi ? 1 : dim == kYi ? 13 : 23;
        int const offset = 1 + identity * multiplier % count;
        raw = period / 2 + cellWidth * offset / (2 * (count + 1));
    } else {
        // Cell-centered particles. Most of the population crowds one full-mesh
        // cell in x, so the count-balanced plane differs from the midpoint and
        // from the plane nearest the continuous median.
        int const cell = dim == kXi ? (identity < 80 ? 3 : 4 + identity % 11) :
                                      (identity * 5 + 2 * dim) % kMeshDomainMeshSize;
        raw = cellWidth * (2 * cell + 1) / 2 + identity % 3;
    }
    // Half a period is an exact multiple of every power-of-two cell width, so
    // the translation moves particles through existing rank boundaries while
    // keeping each seam particle on its side of a mesh plane.
    raw += period / 2 * phase;
    return static_cast<PositionInteger>(raw);
}

struct MeshDomainRecord {
    Array<PositionInteger, kDimensions> raw;
    Vector3<float> velocity;
    GroupInteger lastGroup = kNoParticleGroup;
    int rung;
    int tracer;
    ParticleInteger gravitySample;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & raw & velocity & lastGroup & rung & tracer & gravitySample;
    }
};

struct MeshDomainSnapshot {
    Range<double> box;
    Vector<MeshDomainRecord> records;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & box & records;
    }
};

void initializeMeshDomains(int fixture) {
    int const count = hpxRank() == 0 ? meshDomainFixtureCount(fixture) : 0;
    particlesResize(count);
    for (int identity = 0; identity < count; ++identity) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, identity).setInteger(meshDomainPosition(identity, dim, fixture, 0));
            particlesVel(dim, identity) = float(identity + 1) + 0.25f * dim;
        }
        particlesRung(identity) = 1 + identity % 7;
        if constexpr (COSMICTIGER_ENABLE_ROCKSTAR) {
            particlesLastgroup(identity) = 1000 + identity;
        }
        particlesTracer(identity) = identity % 2;
        particlesGravitySample(identity) = identity % 3 == 0 ? identity : -1;
    }
}

void moveMeshDomains(int fixture, int phase) {
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        int const identity = int(particlesVel(kXi, index)) - 1;
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, index).setInteger(meshDomainPosition(identity, dim, fixture, phase));
        }
    }
}

MeshDomainSnapshot readMeshDomains() {
    MeshDomainSnapshot result;
    result.box = domainsFindMyBox();
    result.records.resize(particlesSize());
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        auto& record = result.records[index];
        for (int dim = 0; dim < kDimensions; ++dim) {
            record.raw[dim] = particlesPos(dim, index).raw();
            record.velocity[dim] = particlesVel(dim, index);
        }
        record.rung = particlesRung(index);
        if constexpr (COSMICTIGER_ENABLE_ROCKSTAR) {
            record.lastGroup = particlesLastgroup(index);
        }
        record.tracer = particlesTracer(index);
        record.gravitySample = particlesGravitySample(index);
    }
    return result;
}

void reloadMeshDomainLayout() {
    FILE* const file = std::tmpfile();
    requireMeshDomain(file != nullptr, "cannot create domain-layout round-trip file");
    domainsSave(file);
    requireMeshDomain(std::fflush(file) == 0, "cannot flush domain-layout round-trip file");
    std::rewind(file);
    domainsLoad(file);
    std::fclose(file);
}

void checkGravitySampleCheckpointReset() {
    FILE* const file = std::tmpfile();
    requireMeshDomain(file != nullptr, "cannot create particle-checkpoint round-trip file");
    particlesGravitySamplesBegin(1);
    if (particlesSize() > 0) {
        particlesGravitySample(0) = 0;
        particlesCaptureGravitySample(0, 1.0f, 2.0f, 3.0f);
    }
    particlesSave(file);
    requireMeshDomain(std::fflush(file) == 0, "cannot flush particle-checkpoint round-trip file");
    std::rewind(file);
    particlesLoad(file);
    std::fclose(file);
    requireMeshDomain(particlesGravitySampleAccelerations == nullptr,
        "checkpoint reload retained a transient gravity capture buffer");
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        requireMeshDomain(particlesGravitySample(index) == -1,
            "checkpoint reload retained a transient gravity sample slot");
    }
}

long double meshDomainCoordinate(PositionInteger raw) {
    return std::ldexp(static_cast<long double>(raw), -kPositionBits);
}

bool meshDomainContains(Range<double> const& box, MeshDomainRecord const& record) {
    for (int dim = 0; dim < kDimensions; ++dim) {
        auto const coordinate = meshDomainCoordinate(record.raw[dim]);
        if (coordinate < box.begin[dim] || coordinate >= box.end[dim]) {
            return false;
        }
    }
    return true;
}

// Integer plane index of a box boundary on the given mesh, or -1 when the
// boundary is not exactly an integer multiple of 1 / meshSize.
int meshDomainPlane(double coordinate, int meshSize) {
    auto const plane = std::llround(coordinate * meshSize);
    if (plane < 0 || plane > meshSize || double(plane) / meshSize != coordinate) {
        return -1;
    }
    return int(plane);
}

int meshDomainCellIndex(MeshDomainRecord const& record, int dim, int meshSize) {
    return int(particleMeshCoordinate(record.raw[dim], uint32_t(meshSize)));
}

size_t meshDomainCellKey(MeshDomainRecord const& record, int meshSize) {
    size_t key = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        key = key * size_t(meshSize) + size_t(meshDomainCellIndex(record, dim, meshSize));
    }
    return key;
}

int meshDomainLevels(int ranks) {
    int levels = 0;
    while ((1 << levels) < ranks) {
        ++levels;
    }
    requireMeshDomain((1 << levels) == ranks, "PMMM domains require a power-of-two rank count");
    return levels;
}

// Independent transcription of the aligned-cut rule. Admissible planes leave
// both children enough coarse cells for the cuts they still owe on this axis.
// Among them the plane whose below-count is closest to the balanced target
// wins; ties go to the plane nearest the parent's middle, then the lower one.
int meshDomainExpectedPlane(
    std::vector<int> const& coarseIndices, int lo, int hi, int depth, int ranks) {
    int const axis = depth % kDimensions;
    int const levels = meshDomainLevels(ranks);
    int remainingCuts = 0;
    for (int level = 1; level < levels; ++level) {
        if ((depth + level) % kDimensions == axis) {
            ++remainingCuts;
        }
    }
    int const need = 1 << remainingCuts;
    requireMeshDomain(
        hi - lo >= 2 * need, "parent domain has too few coarse cells for its remaining cuts");
    size_t const target = size_t(ranks / 2) * coarseIndices.size() / size_t(ranks);
    int best = -1;
    size_t bestError = std::numeric_limits<size_t>::max();
    int bestCenterDistance = std::numeric_limits<int>::max();
    for (int plane = lo + need; plane <= hi - need; ++plane) {
        size_t const below = size_t(std::count_if(coarseIndices.begin(), coarseIndices.end(),
            [plane](int const index) { return index < plane; }));
        size_t const error = below > target ? below - target : target - below;
        int const centerDistance = std::abs(2 * plane - (lo + hi));
        if (error < bestError || (error == bestError && centerDistance < bestCenterDistance)) {
            best = plane;
            bestError = error;
            bestCenterDistance = centerDistance;
        }
    }
    return best;
}

void verifyMeshDomainSplits(Vector<MeshDomainRecord> const& records, size_t key, int ranks,
    int depth, bool rebound, int coarseMeshSize) {
    auto const parent = domainsRange(key);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireMeshDomain(parent.begin[dim] >= 0.0 && parent.end[dim] <= 1.0 &&
                parent.begin[dim] <= parent.end[dim],
            "domain bounds do not form a valid spatial interval");
        // Every boundary, trivial or not, is an exact coarse-mesh plane.
        requireMeshDomain(meshDomainPlane(parent.begin[dim], coarseMeshSize) >= 0 &&
                meshDomainPlane(parent.end[dim], coarseMeshSize) >= 0,
            "domain boundary is not an exact coarse-mesh plane");
    }
    if (ranks == 1) {
        return;
    }
    auto const left = domainsRange(key * 2);
    auto const right = domainsRange(key * 2 + 1);
    int const axis = depth % kDimensions;
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireMeshDomain(left.begin[dim] == parent.begin[dim] &&
                right.end[dim] == parent.end[dim] &&
                (dim == axis ?
                        left.end[dim] == right.begin[dim] :
                        left.end[dim] == parent.end[dim] && right.begin[dim] == parent.begin[dim]),
            "child domains do not partition their parent along the selected axis");
    }
    int const leftRanks = ranks / 2;
    if (rebound) {
        int const lo = meshDomainPlane(parent.begin[axis], coarseMeshSize);
        int const hi = meshDomainPlane(parent.end[axis], coarseMeshSize);
        std::vector<int> coarseIndices;
        for (auto const& record : records) {
            if (meshDomainContains(parent, record)) {
                int const index = meshDomainCellIndex(record, axis, coarseMeshSize);
                // The integer classifier must agree with the exact coordinate test.
                requireMeshDomain(index >= lo && index < hi,
                    "mesh cell classification disagrees with domain containment");
                coarseIndices.push_back(index);
            }
        }
        int const expected = meshDomainExpectedPlane(coarseIndices, lo, hi, depth, ranks);
        requireMeshDomain(expected > lo && expected < hi, "no admissible coarse plane was found");
        requireMeshDomain(left.end[axis] == double(expected) / coarseMeshSize,
            "domain cut is not the count-balanced coarse-mesh plane");
    }
    verifyMeshDomainSplits(records, key * 2, leftRanks, depth + 1, rebound, coarseMeshSize);
    verifyMeshDomainSplits(
        records, key * 2 + 1, ranks - leftRanks, depth + 1, rebound, coarseMeshSize);
}

void readMeshDomainLayout(Vector<Range<double>>& boxes, size_t key, int ranks) {
    boxes.push_back(domainsRange(key));
    if (ranks > 1) {
        readMeshDomainLayout(boxes, key * 2, ranks / 2);
        readMeshDomainLayout(boxes, key * 2 + 1, ranks - ranks / 2);
    }
}

void verifyMeshDomainLayout(Vector<Range<double>> const& expected) {
    Vector<Range<double>> actual;
    readMeshDomainLayout(actual, 1, hpxSize());
    requireMeshDomain(actual.size() == expected.size(), "domain topology changed");
    for (size_t index = 0; index < actual.size(); ++index) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireMeshDomain(actual[index].begin[dim] == expected[index].begin[dim] &&
                    actual[index].end[dim] == expected[index].end[dim],
                "checkpoint reload or mesh size changed particle domain boundaries");
        }
    }
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeMeshDomains, InitializeMeshDomainsAction);
HPX_PLAIN_ACTION(detail::moveMeshDomains, MoveMeshDomainsAction);
HPX_PLAIN_ACTION(detail::readMeshDomains, ReadMeshDomainsAction);
HPX_PLAIN_ACTION(detail::reloadMeshDomainLayout, ReloadMeshDomainLayoutAction);
HPX_PLAIN_ACTION(
    detail::checkGravitySampleCheckpointReset, CheckGravitySampleCheckpointResetAction);

namespace detail {

// No cell of the coarse mesh, nor of any finer power-of-two mesh, may hold
// particles owned by two localities.
void verifyMeshDomainCellOwnership(
    Vector<MeshDomainRecord> const& records, Vector<int> const& owners, int meshSize) {
    std::vector<int> cellOwner(size_t(meshSize) * size_t(meshSize) * size_t(meshSize), -1);
    for (size_t index = 0; index < records.size(); ++index) {
        auto& owner = cellOwner[meshDomainCellKey(records[index], meshSize)];
        if (owner < 0) {
            owner = owners[index];
        } else {
            requireMeshDomain(
                owner == owners[index], "a mesh cell holds particles owned by two localities");
        }
    }
}

void verifyMeshDomains(int fixture, int phase, bool rebound, int coarseMeshSize) {
    Vector<hpx::future<MeshDomainSnapshot>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ReadMeshDomainsAction>(locality));
    }
    int const count = meshDomainFixtureCount(fixture);
    std::vector<bool> seen(count, false);
    Vector<MeshDomainRecord> allRecords;
    Vector<int> owners;
    int populatedRanks = 0;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const snapshot = pending[rank].get();
        populatedRanks += snapshot.records.empty() ? 0 : 1;
        for (auto const& record : snapshot.records) {
            int const identity = int(record.velocity[kXi]) - 1;
            requireMeshDomain(identity >= 0 && identity < count && !seen[identity],
                "migration lost or duplicated particle identities");
            seen[identity] = true;
            requireMeshDomain(meshDomainContains(snapshot.box, record),
                "particle lies outside its owning locality's spatial domain");
            for (int dim = 0; dim < kDimensions; ++dim) {
                requireMeshDomain(
                    record.raw[dim] == meshDomainPosition(identity, dim, fixture, phase) &&
                        record.velocity[dim] == float(identity + 1) + 0.25f * dim,
                    "partitioning or migration detached a particle position or velocity");
            }
            requireMeshDomain(record.rung == 1 + identity % 7 && record.tracer == identity % 2 &&
                    record.gravitySample == (identity % 3 == 0 ? identity : -1),
                "partitioning or migration detached a rung, tracer or gravity sample");
            if constexpr (COSMICTIGER_ENABLE_ROCKSTAR) {
                requireMeshDomain(record.lastGroup == 1000 + identity,
                    "partitioning or migration detached a particle's group history");
            }
            allRecords.push_back(record);
            owners.push_back(rank);
        }
    }
    requireMeshDomain(allRecords.size() == size_t(count), "migration changed particle count");
    auto const root = domainsRange(1);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireMeshDomain(root.begin[dim] == 0.0 && root.end[dim] == 1.0,
            "particle domains do not cover the periodic volume");
    }
    verifyMeshDomainSplits(allRecords, 1, hpxSize(), 0, rebound, coarseMeshSize);
    int const meshSizes[] = {coarseMeshSize, 2 * coarseMeshSize, getOptions().gravity.meshSize};
    for (int const meshSize : meshSizes) {
        verifyMeshDomainCellOwnership(allRecords, owners, meshSize);
    }
    if (fixture == kMeshDomainDenseFixture) {
        // One coarse cell cannot be split, so the dense population shares one
        // locality whatever the rank count; the other localities stay empty.
        requireMeshDomain(
            populatedRanks == 1, "particles of one coarse cell were spread over localities");
    }
}

void initializeMeshDomainFixture(int fixture) {
    Vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<InitializeMeshDomainsAction>(locality, fixture));
    }
    for (auto& future : pending) {
        future.get();
    }
}

}    // namespace detail

void meshDomainsTest() {
    detail::requireMeshDomain(
        (hpxSize() & (hpxSize() - 1)) == 0, "meshDomains needs a power-of-two locality count");
    auto options = getOptions();
    options.hydro.enabled = false;
    options.gravity.enabled = true;
    // setOptions skips command-line validation, so every PMMM mesh parameter the
    // domain cuts read is set explicitly here.
    options.gravity.meshSize = detail::kMeshDomainMeshSize;
    options.gravity.pmmmNearWidth = detail::kMeshDomainNearWidth;
    options.gravity.errorInterval = 1;
    options.analysis.groups = COSMICTIGER_ENABLE_ROCKSTAR != 0;
    options.analysis.tracers = true;
    options.analysis.saveForce = false;
    for (int const coarseMeshSize : detail::kMeshDomainCoarseMeshSizes) {
        options.gravity.pmmmCoarseMeshSize = coarseMeshSize;
        for (int fixture = 0; fixture < detail::kMeshDomainFixtures; ++fixture) {
            // The balancing target must use the current distributed population,
            // rather than an initial-condition count that can become stale.
            options.initial.numberOfParticles = detail::meshDomainFixtureCount(fixture) + 11;
            setOptions(options);
            detail::initializeMeshDomainFixture(fixture);
            domainsRebound();
            detail::migrateMeshDomainFixture();
            detail::verifyMeshDomains(fixture, 0, true, coarseMeshSize);
            Vector<hpx::future<void>> pending;
            for (auto const& locality : hpxLocalities()) {
                pending.push_back(hpx::async<MoveMeshDomainsAction>(locality, fixture, 1));
            }
            for (auto& future : pending) {
                future.get();
            }
            // Translated particles migrate across the stale layout without
            // rebalancing, then a fresh rebound realigns the cuts to them.
            detail::migrateMeshDomainFixture();
            detail::verifyMeshDomains(fixture, 1, false, coarseMeshSize);
            domainsRebound();
            detail::migrateMeshDomainFixture();
            detail::verifyMeshDomains(fixture, 1, true, coarseMeshSize);
        }
    }

    // Aligned cuts survive checkpoint round trips and full-mesh changes that keep
    // the coarse mesh. The crowded fixture places its x cut away from the middle,
    // so a layout silently rebuilt from defaults would be detected.
    constexpr int kFixture = 1;
    int const coarseMeshSize = detail::kMeshDomainCoarseMeshSizes[0];
    options.gravity.pmmmCoarseMeshSize = coarseMeshSize;
    options.gravity.meshSize = detail::kMeshDomainMeshSize;
    options.initial.numberOfParticles = detail::meshDomainFixtureCount(kFixture) + 11;
    setOptions(options);
    detail::initializeMeshDomainFixture(kFixture);
    domainsRebound();
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true, coarseMeshSize);
    Vector<Range<double>> originalLayout;
    detail::readMeshDomainLayout(originalLayout, 1, hpxSize());
    Vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ReloadMeshDomainLayoutAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true, coarseMeshSize);
    detail::verifyMeshDomainLayout(originalLayout);
    // Halving the full mesh keeps the coarse mesh, hence the layout and ownership.
    options.gravity.meshSize = detail::kMeshDomainMeshSize / 2;
    setOptions(options);
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true, coarseMeshSize);
    detail::verifyMeshDomainLayout(originalLayout);
    // Rebounding the same population reproduces the same aligned layout.
    domainsRebound();
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true, coarseMeshSize);
    detail::verifyMeshDomainLayout(originalLayout);
    pending.clear();
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<CheckGravitySampleCheckpointResetAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    std::cout << std::format("Coarse-aligned particle domains, unsplit cells, count-balanced cuts "
                             "and migration passed ({:d} bits, {:d} ranks)\n",
                     kPositionBits, hpxSize())
              << std::flush;
}
