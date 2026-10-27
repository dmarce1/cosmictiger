/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

constexpr int kMeshDomainSide = 7;

void requireMeshDomain(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Mesh domain regression: ") + message);
    }
}

void migrateMeshDomainFixture() {
    // These ownership checks move every rung. Hierarchical builds migrate one
    // rung per call; ordinary leapfrog migrates the whole population together.
    constexpr int kLastRung = integration::kBuildUsesHierarchical ? 7 : 0;
    for (int rung = 0; rung <= kLastRung; ++rung) {
        domainsBegin(rung);
        domainsEnd();
    }
}

int meshDomainFixtureCount(int fixture) {
    return fixture == 0 ? 126 : fixture == 1 ? 112 : fixture == 2 ? 37 : 0;
}

PositionInteger meshDomainPosition(int identity, int dim, int fixture, int phase) {
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << kPositionBits;
    unsigned __int128 raw;
    if (fixture == 0) {
        int const plane = 1 + ((identity / 3 + 2 * dim) % (kMeshDomainSide - 1));
        raw = (period * plane + kMeshDomainSide - 1) / kMeshDomainSide;
        int const offset = (identity + dim) % 3 - 1;
        raw = offset < 0 ? raw - 1 : raw + offset;
        if (identity % 21 == 0) {
            raw = dim % 2 == 0 ? 0 : period - 1;
        }
    } else if (fixture == 2) {
        // Distinct coordinates inside one cell must be balanced across ranks.
        // Keep their separation well above the domain search resolution.
        int const count = meshDomainFixtureCount(fixture);
        int const multiplier = dim == kXDimension ? 1 : dim == kYDimension ? 13 : 23;
        int const offset = 1 + identity * multiplier % count;
        raw = period / 2 + period * offset / (32 * (count + 1));
    } else {
        int const cell = dim == kXDimension ? (identity < 80 ? 1 : 2 + identity % 5) :
                                              (identity * 5 + 2 * dim) % kMeshDomainSide;
        raw = period * (2 * cell + 1) / (2 * kMeshDomainSide) + identity % 3;
    }
    // An exactly representable periodic translation moves particles through
    // existing rank boundaries without rounding the odd-mesh seam fixtures.
    raw += period / 2 * phase;
    return static_cast<PositionInteger>(raw);
}

int meshDomainReferenceCell(PositionInteger raw) {
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << kPositionBits;
    // Independent half-open cell oracle, using exact boundary division instead
    // of the multiplication used by the production mesh classifier.
    for (int plane = 1; plane < kMeshDomainSide; ++plane) {
        auto const boundary = (period * plane + kMeshDomainSide - 1) / kMeshDomainSide;
        if (static_cast<unsigned __int128>(raw) < boundary) {
            return plane - 1;
        }
    }
    return kMeshDomainSide - 1;
}

struct MeshDomainRecord {
    Array<PositionInteger, kDimensions> raw;
    Array<float, kDimensions> velocity;
    GroupInteger lastGroup;
    int rung;
    int tracer;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & raw & velocity & lastGroup & rung & tracer;
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
        particlesLastgroup(identity) = 1000 + identity;
        particlesTracer(identity) = identity % 2;
    }
}

void moveMeshDomains(int fixture, int phase) {
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        int const identity = int(particlesVel(kXDimension, index)) - 1;
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
        record.lastGroup = particlesLastgroup(index);
        record.tracer = particlesTracer(index);
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

Array<int, kDimensions> meshDomainRecordCell(MeshDomainRecord const& record) {
    Array<int, kDimensions> cell;
    for (int dim = 0; dim < kDimensions; ++dim) {
        cell[dim] = meshDomainReferenceCell(record.raw[dim]);
    }
    return cell;
}

void verifyMeshDomainSplits(
    Vector<MeshDomainRecord> const& records, size_t key, int ranks, int depth, bool balanced) {
    auto const parent = domainsRange(key);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireMeshDomain(parent.begin[dim] >= 0.0 && parent.end[dim] <= 1.0 &&
                parent.begin[dim] <= parent.end[dim],
            "domain bounds do not form a valid spatial interval");
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
    if (balanced) {
        std::vector<long double> coordinates;
        for (auto const& record : records) {
            if (meshDomainContains(parent, record)) {
                coordinates.push_back(meshDomainCoordinate(record.raw[axis]));
            }
        }
        std::sort(coordinates.begin(), coordinates.end());
        size_t const target = coordinates.size() * leftRanks / ranks;
        long double const lower = target == 0 ? parent.begin[axis] : coordinates[target - 1];
        long double const upper =
            target == coordinates.size() ? parent.end[axis] : coordinates[target];
        // A coordinate quantile is independent of the bisection implementation.
        // Coincident or sub-resolution points need not be split exactly evenly.
        long double const tolerance =
            std::ldexp(parent.end[axis] - parent.begin[axis], -kDomainReboundIterations) +
            std::ldexp(1.0L, -kPositionBits);
        requireMeshDomain(
            left.end[axis] >= lower - tolerance && left.end[axis] <= upper + tolerance,
            "domain cut does not balance the current particle population");
    }
    verifyMeshDomainSplits(records, key * 2, leftRanks, depth + 1, balanced);
    verifyMeshDomainSplits(records, key * 2 + 1, ranks - leftRanks, depth + 1, balanced);
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

namespace detail {

void verifyMeshDomains(int fixture, int phase, bool balanced) {
    Vector<hpx::future<MeshDomainSnapshot>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ReadMeshDomainsAction>(locality));
    }
    int const count = meshDomainFixtureCount(fixture);
    std::vector<bool> seen(count, false);
    std::vector<int> firstCellOwner(kMeshDomainSide * kMeshDomainSide * kMeshDomainSide, -1);
    bool splitCell = false;
    Vector<MeshDomainRecord> allRecords;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const snapshot = pending[rank].get();
        if (balanced && fixture == 2) {
            // A finite bisection may miss its target by one particle per cut.
            int tolerance = 0;
            for (int ranks = hpxSize(); ranks > 1; ranks = (ranks + 1) / 2) {
                ++tolerance;
            }
            requireMeshDomain(
                snapshot.records.size() >= size_t(std::max(0, count / hpxSize() - tolerance)) &&
                    snapshot.records.size() <=
                        size_t((count + hpxSize() - 1) / hpxSize() + tolerance),
                "particles in one mesh cell were not balanced across localities");
        }
        for (auto const& record : snapshot.records) {
            int const identity = int(record.velocity[kXDimension]) - 1;
            requireMeshDomain(identity >= 0 && identity < count && !seen[identity],
                "migration lost or duplicated particle identities");
            seen[identity] = true;
            requireMeshDomain(meshDomainContains(snapshot.box, record),
                "particle lies outside its owning locality's spatial domain");
            auto const cell = meshDomainRecordCell(record);
            int const key = (cell[0] * kMeshDomainSide + cell[1]) * kMeshDomainSide + cell[2];
            auto& owner = firstCellOwner[key];
            if (owner < 0) {
                owner = rank;
            } else {
                splitCell |= owner != rank;
            }
            for (int dim = 0; dim < kDimensions; ++dim) {
                requireMeshDomain(
                    record.raw[dim] == meshDomainPosition(identity, dim, fixture, phase) &&
                        record.velocity[dim] == float(identity + 1) + 0.25f * dim,
                    "partitioning or migration detached a particle position or velocity");
            }
            requireMeshDomain(record.rung == 1 + identity % 7 &&
                    record.lastGroup == 1000 + identity && record.tracer == identity % 2,
                "partitioning or migration detached a rung, group or tracer field");
            allRecords.push_back(record);
        }
    }
    requireMeshDomain(allRecords.size() == size_t(count), "migration changed particle count");
    auto const root = domainsRange(1);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireMeshDomain(root.begin[dim] == 0.0 && root.end[dim] == 1.0,
            "particle domains do not cover the periodic volume");
    }
    verifyMeshDomainSplits(allRecords, 1, hpxSize(), 0, balanced);
    if (balanced && fixture == 2 && hpxSize() > 1) {
        requireMeshDomain(splitCell, "dense mesh cell does not span particle localities");
    }
}

}    // namespace detail

void meshDomainsTest() {
    auto options = getOptions();
    options.hydro.enabled = false;
    options.gravity.enabled = true;
    options.gravity.meshSize = detail::kMeshDomainSide;
    options.run.hierarchicalGravity = integration::kBuildUsesHierarchical;
    options.analysis.groups = true;
    options.analysis.tracers = true;
    options.analysis.saveForce = false;
    for (int fixture = 0; fixture < 4; ++fixture) {
        // The balancing target must use the current distributed population,
        // rather than an initial-condition count that can become stale.
        options.initial.numberOfParticles = detail::meshDomainFixtureCount(fixture) + 11;
        setOptions(options);
        Vector<hpx::future<void>> pending;
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<InitializeMeshDomainsAction>(locality, fixture));
        }
        for (auto& future : pending) {
            future.get();
        }
        domainsRebound();
        detail::migrateMeshDomainFixture();
        detail::verifyMeshDomains(fixture, 0, true);
        pending.clear();
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<MoveMeshDomainsAction>(locality, fixture, 1));
        }
        for (auto& future : pending) {
            future.get();
        }
        detail::migrateMeshDomainFixture();
        detail::verifyMeshDomains(fixture, 1, false);
        domainsRebound();
        detail::migrateMeshDomainFixture();
        detail::verifyMeshDomains(fixture, 1, true);
    }

    // Continuous cuts survive checkpoint round trips and mesh-size changes.
    // Use the dense fixture so each nontrivial cut lies inside a mesh cell.
    constexpr int kFixture = 2;
    options.initial.numberOfParticles = detail::meshDomainFixtureCount(kFixture) + 11;
    setOptions(options);
    Vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<InitializeMeshDomainsAction>(locality, kFixture));
    }
    for (auto& future : pending) {
        future.get();
    }
    domainsRebound();
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true);
    if (hpxSize() > 1) {
        double const split = domainsRange(2).end[kXDimension];
        double const meshSplit =
            double(std::llround(split * detail::kMeshDomainSide)) / detail::kMeshDomainSide;
        detail::requireMeshDomain(split != meshSplit,
            "dense fixture did not produce a continuous non-mesh domain boundary");
    }
    Vector<Range<double>> originalLayout;
    detail::readMeshDomainLayout(originalLayout, 1, hpxSize());
    pending.clear();
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ReloadMeshDomainLayoutAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true);
    detail::verifyMeshDomainLayout(originalLayout);
    options.gravity.meshSize = 11;
    setOptions(options);
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true);
    detail::verifyMeshDomainLayout(originalLayout);
    domainsRebound();
    detail::migrateMeshDomainFixture();
    detail::verifyMeshDomains(kFixture, 0, true);
    detail::verifyMeshDomainLayout(originalLayout);
    PRINT("Independent particle domains, split cells, balancing and migration passed "
          "(%i bits, %i ranks)\n",
        kPositionBits, hpxSize());
}
