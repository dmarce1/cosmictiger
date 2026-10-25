/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    } else {
        int const cell = fixture == 2 ? 3 :
            dim == XDIM               ? (identity < 80 ? 1 : 2 + identity % 5) :
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
    Array<PositionInteger, NDIM> raw;
    Array<float, NDIM> velocity;
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
        for (int dim = 0; dim < NDIM; ++dim) {
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
        int const identity = int(particlesVel(XDIM, index)) - 1;
        for (int dim = 0; dim < NDIM; ++dim) {
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
        for (int dim = 0; dim < NDIM; ++dim) {
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

Range<int> meshDomainIntegerBox(Range<double> const& box) {
    Range<int> result;
    for (int dim = 0; dim < NDIM; ++dim) {
        result.begin[dim] = int(std::llround(box.begin[dim] * kMeshDomainSide));
        result.end[dim] = int(std::llround(box.end[dim] * kMeshDomainSide));
        requireMeshDomain(result.begin[dim] >= 0 && result.end[dim] <= kMeshDomainSide &&
                result.begin[dim] <= result.end[dim],
            "domain bounds do not form a valid mesh interval");
        requireMeshDomain(box.begin[dim] == double(result.begin[dim]) / kMeshDomainSide &&
                box.end[dim] == double(result.end[dim]) / kMeshDomainSide,
            "domain boundary is not an exact mesh plane");
    }
    return result;
}

Array<int, NDIM> meshDomainRecordCell(MeshDomainRecord const& record) {
    Array<int, NDIM> cell;
    for (int dim = 0; dim < NDIM; ++dim) {
        cell[dim] = meshDomainReferenceCell(record.raw[dim]);
    }
    return cell;
}

void verifyMeshDomainSplits(
    Vector<MeshDomainRecord> const& records, size_t key, int ranks, int depth) {
    if (ranks == 1) {
        return;
    }
    auto const parent = meshDomainIntegerBox(domainsRange(key));
    auto const left = meshDomainIntegerBox(domainsRange(key * 2));
    auto const right = meshDomainIntegerBox(domainsRange(key * 2 + 1));
    int const axis = depth % NDIM;
    int const actualPlane = left.end[axis];
    for (int dim = 0; dim < NDIM; ++dim) {
        requireMeshDomain(left.begin[dim] == parent.begin[dim] &&
                right.end[dim] == parent.end[dim] &&
                (dim == axis ?
                        left.end[dim] == right.begin[dim] :
                        left.end[dim] == parent.end[dim] && right.begin[dim] == parent.begin[dim]),
            "child domains do not partition their parent along the selected axis");
    }
    size_t total = 0;
    for (auto const& record : records) {
        total += parent.contains(meshDomainRecordCell(record));
    }
    int bestPlane = parent.begin[axis];
    uint64_t bestError = std::numeric_limits<uint64_t>::max();
    int bestDistance = std::numeric_limits<int>::max();
    int const leftRanks = ranks / 2;
    for (int plane = parent.begin[axis]; plane <= parent.end[axis]; ++plane) {
        size_t below = 0;
        for (auto const& record : records) {
            auto const cell = meshDomainRecordCell(record);
            below += parent.contains(cell) && cell[axis] < plane;
        }
        int64_t const residual = int64_t(below) * ranks - int64_t(total) * leftRanks;
        uint64_t const error = residual < 0 ? uint64_t(-residual) : uint64_t(residual);
        int const distance = std::abs((plane - parent.begin[axis]) * ranks -
            (parent.end[axis] - parent.begin[axis]) * leftRanks);
        if (error < bestError || (error == bestError && distance < bestDistance)) {
            bestError = error;
            bestDistance = distance;
            bestPlane = plane;
        }
    }
    requireMeshDomain(actualPlane == bestPlane,
        "selected plane is not the most balanced legal split with deterministic tie breaking");
    verifyMeshDomainSplits(records, key * 2, leftRanks, depth + 1);
    verifyMeshDomainSplits(records, key * 2 + 1, ranks - leftRanks, depth + 1);
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
    std::vector<int> cellOwners(kMeshDomainSide * kMeshDomainSide * kMeshDomainSide, -1);
    Vector<MeshDomainRecord> allRecords;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const snapshot = pending[rank].get();
        auto const box = meshDomainIntegerBox(snapshot.box);
        for (int x = box.begin[0]; x < box.end[0]; ++x) {
            for (int y = box.begin[1]; y < box.end[1]; ++y) {
                for (int z = box.begin[2]; z < box.end[2]; ++z) {
                    int const key = (x * kMeshDomainSide + y) * kMeshDomainSide + z;
                    requireMeshDomain(cellOwners[key] == -1, "one cell has multiple owners");
                    cellOwners[key] = rank;
                }
            }
        }
        for (auto const& record : snapshot.records) {
            int const identity = int(record.velocity[XDIM]) - 1;
            requireMeshDomain(identity >= 0 && identity < count && !seen[identity],
                "migration lost or duplicated particle identities");
            seen[identity] = true;
            requireMeshDomain(box.contains(meshDomainRecordCell(record)),
                "particle belongs to a mesh cell outside its owning locality");
            for (int dim = 0; dim < NDIM; ++dim) {
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
    requireMeshDomain(std::find(cellOwners.begin(), cellOwners.end(), -1) == cellOwners.end(),
        "domain partition left a mesh cell without an owner");
    if (balanced) {
        verifyMeshDomainSplits(allRecords, 1, hpxSize(), 0);
    }
}

}    // namespace detail

void meshDomainsTest() {
    auto options = getOptions();
    options.hydro.enabled = false;
    options.gravity.enabled = true;
    options.gravity.meshSize = detail::kMeshDomainSide;
    options.run.hierarchicalGravity = false;
    options.analysis.groups = true;
    options.analysis.tracers = true;
    options.analysis.saveForce = false;
    options.gravity.method = "pmmm";
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
        domainsBegin(0);
        domainsEnd();
        detail::verifyMeshDomains(fixture, 0, true);
        pending.clear();
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<MoveMeshDomainsAction>(locality, fixture, 1));
        }
        for (auto& future : pending) {
            future.get();
        }
        domainsBegin(0);
        domainsEnd();
        detail::verifyMeshDomains(fixture, 1, false);
        domainsRebound();
        domainsBegin(0);
        domainsEnd();
        detail::verifyMeshDomains(fixture, 1, true);
    }

    // Older mesh checkpoints contain the same continuous cuts that FMM
    // produces. Loading those domains must trigger an automatic mesh
    // rebalance before migration, without a caller-side rebound request.
    options.gravity.method = "fmm";
    options.initial.numberOfParticles = detail::meshDomainFixtureCount(0);
    setOptions(options);
    Vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<InitializeMeshDomainsAction>(locality, 0));
    }
    for (auto& future : pending) {
        future.get();
    }
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    if (hpxSize() > 1) {
        double const split = domainsRange(2).end[XDIM];
        double const meshSplit =
            double(std::llround(split * detail::kMeshDomainSide)) / detail::kMeshDomainSide;
        detail::requireMeshDomain(split != meshSplit,
            "legacy fixture did not produce a continuous non-mesh domain boundary");
    }
    options.gravity.method = "pmmm";
    setOptions(options);
    pending.clear();
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ReloadMeshDomainLayoutAction>(locality));
    }
    for (auto& future : pending) {
        future.get();
    }
    domainsBegin(0);
    domainsEnd();
    detail::verifyMeshDomains(0, 0, true);
    PRINT("Mesh-aligned domain ownership, balancing and migration passed (%i bits, %i ranks)\n",
        kPositionBits, hpxSize());
}
