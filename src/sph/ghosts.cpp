/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/sph/ghosts.hpp>
#include <cosmictiger/sph/parent_tree.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>

namespace detail {

// One locality's published interest: the stored boxes of a cut through its local tree, at
// most kInterestBoxes nodes, and their unions for a quick rejection. Every local leaf lies
// under one node of the cut, whose boxes contain the leaf's, so the export test against the
// cut is as complete as against the leaves themselves; a cut confines a stray leaf's long
// box to its own few nodes instead of one locality-wide union.
inline constexpr int kInterestBoxes = 64;

struct SphGhostInterest {
    int rank = -1;
    Fixed32Range innerUnion;
    Fixed32Range outerUnion;
    Vector<Fixed32Range> innerBoxes;
    Vector<Fixed32Range> outerBoxes;

    bool empty() const {
        return innerBoxes.empty();
    }

    // The walk's opening test of a leaf against this interest.
    bool reaches(SphTreeNode const& leaf) const {
        if (!leaf.innerBox.periodicIntersects(outerUnion) &&
            !leaf.outerBox.periodicIntersects(innerUnion)) {
            return false;
        }
        for (size_t node = 0; node < innerBoxes.size(); node++) {
            if (leaf.innerBox.periodicIntersects(outerBoxes[node]) ||
                leaf.outerBox.periodicIntersects(innerBoxes[node])) {
                return true;
            }
        }
        return false;
    }

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & rank;
        archive & innerUnion;
        archive & outerUnion;
        archive & innerBoxes;
        archive & outerBoxes;
    }
};

// The fields of an owner's exported particles, in the order of its exported leaves.
struct SphGhostFields {
    int owner = -1;
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
    Vector<char> rungCandidate;    // Empty unless the rung balance is active.

    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & owner;
        archive & x;
        archive & y;
        archive & z;
        archive & internalEnergy;
        archive & smoothingLength;
        archive & rung;
        archive & vx;
        archive & vy;
        archive & vz;
        archive & density;
        archive & balsaraFactor;
        archive & bulkViscosityCoefficient;
        archive & gradHCorrection;
        archive & rungCandidate;
    }

    std::int64_t bytes() const {
        return std::int64_t(x.size() + y.size() + z.size()) * sizeof(Position) +
            std::int64_t(internalEnergy.size() + smoothingLength.size() + vx.size() + vy.size() +
                vz.size() + density.size() + balsaraFactor.size() +
                bulkViscosityCoefficient.size() + gradHCorrection.size()) *
            sizeof(float) +
            std::int64_t(rung.size() + rungCandidate.size());
    }
};

// Ghost leaves held by this locality, and the fields of their particles.
struct SphGhostStore {
    std::vector<SphTreeNode> nodes;
    std::vector<ParticleInteger> nodeOffsets;
    std::unordered_map<std::uint64_t, int> nodeById;
    std::unordered_map<std::uint64_t, int> nodeByRangeStart;
    std::unordered_map<int, ParticleInteger> ownerOffsets;
    Vector<TreeId> leafIds;
    ParticleInteger size = 0;
    SphGhostFields fields;
};

static MutexType storeMutex;
static SphGhostStore store;
// Exports of this locality: importer rank to the indices of exported local leaves.
static std::unordered_map<int, std::vector<int>> exports;
static SphGhostStatistics counters;
static std::atomic<int> stagingDepth{0};

std::uint64_t ghostKey(int rank, std::int64_t index) {
    return (std::uint64_t(std::uint32_t(rank)) << 32) | std::uint64_t(std::uint32_t(index));
}

// The ghost leaf holding a whole remote range, and that range's first slot in the store.
ParticleInteger ghostOffset(ParticleGlobalRange const& range) {
    auto const found = store.nodeByRangeStart.find(ghostKey(range.proc, range.range.first));
    if (found == store.nodeByRangeStart.end()) {
        THROW_ERROR("SPH ghosts: rank %i holds no ghost leaf for rank %i gas range [%lli, %lli)\n",
            hpxRank(), range.proc, (long long) range.range.first, (long long) range.range.second);
    }
    auto const& node = store.nodes[found->second];
    if (range.range.second > node.particleRange.second) {
        THROW_ERROR("SPH ghosts: remote range outruns its ghost leaf\n");
    }
    return store.nodeOffsets[found->second];
}

template <class T>
void resizeField(Vector<T>& field, ParticleInteger size) {
    field.resize(size);
}

void resizeStore(ParticleInteger size) {
    auto& f = store.fields;
    resizeField(f.x, size);
    resizeField(f.y, size);
    resizeField(f.z, size);
    resizeField(f.internalEnergy, size);
    resizeField(f.smoothingLength, size);
    resizeField(f.rung, size);
    resizeField(f.vx, size);
    resizeField(f.vy, size);
    resizeField(f.vz, size);
    resizeField(f.density, size);
    resizeField(f.balsaraFactor, size);
    resizeField(f.bulkViscosityCoefficient, size);
    resizeField(f.gradHCorrection, size);
    resizeField(f.rungCandidate, size);
}

void clearLocalGhosts() {
    std::lock_guard<MutexType> lock(storeMutex);
    store = SphGhostStore();
    exports.clear();
}

void growUnion(Fixed32Range& total, Fixed32Range const& box) {
    for (int dim = 0; dim < kDimensions; dim++) {
        total.begin[dim] = std::min(total.begin[dim], box.begin[dim]);
        total.end[dim] = std::max(total.end[dim], box.end[dim]);
    }
}

// This locality's interest: the deepest level-by-level cut of the local tree with at most
// kInterestBoxes nonempty nodes.
SphGhostInterest localInterest() {
    SphGhostInterest interest;
    interest.rank = hpxRank();
    interest.innerUnion = emptySphBounds();
    interest.outerUnion = emptySphBounds();
    TreeId const root = sphTreeLocalRoot();
    if (root.index < 0 || sphTreeGetNode(root)->numberOfParticles() == 0) {
        return interest;
    }
    std::vector<TreeId> cut{root};
    while (true) {
        std::vector<TreeId> next;
        bool opened = false;
        for (auto const id : cut) {
            auto const& node = *sphTreeGetNode(id);
            if (node.leaf) {
                next.push_back(id);
                continue;
            }
            opened = true;
            for (auto const child : node.children) {
                if (child.index >= 0 && sphTreeGetNode(child)->numberOfParticles() > 0) {
                    next.push_back(child);
                }
            }
        }
        if (!opened || int(next.size()) > kInterestBoxes) {
            break;
        }
        cut = std::move(next);
    }
    for (auto const id : cut) {
        auto const& node = *sphTreeGetNode(id);
        interest.innerBoxes.push_back(node.innerBox);
        interest.outerBoxes.push_back(node.outerBox);
        growUnion(interest.innerUnion, node.innerBox);
        growUnion(interest.outerUnion, node.outerBox);
    }
    return interest;
}

// Pack the fields of local gas range [begin, end), converted as the cache-line fetches do.
void packRange(
    SphGhostFields& fields, ParticleInteger begin, ParticleInteger end, bool includeCandidates) {
    for (ParticleInteger i = begin; i < end; i++) {
        fields.x.push_back(sphParticlesPositionValue(kXi, i));
        fields.y.push_back(
            kSpatialDimensions > 1 ? sphParticlesPositionValue(kYi, i) : Position(0.5));
        fields.z.push_back(
            kSpatialDimensions > 2 ? sphParticlesPositionValue(kZi, i) : Position(0.5));
        fields.internalEnergy.push_back(sphParticlesInternalEnergy(i));
        fields.smoothingLength.push_back(sphParticlesSmoothingLength(i));
        fields.rung.push_back(sphParticlesRung(i));
        fields.vx.push_back(sphParticlesVelocityValue(kXi, i));
        fields.vy.push_back(kSpatialDimensions > 1 ? sphParticlesVelocityValue(kYi, i) : 0.0f);
        fields.vz.push_back(kSpatialDimensions > 2 ? sphParticlesVelocityValue(kZi, i) : 0.0f);
        fields.density.push_back(sphParticlesDensity(i));
        fields.balsaraFactor.push_back(sphParticlesBalsaraFactors ?
                balsara::decodeFactor(sphParticlesBalsaraFactors[i]) :
                1.0f);
        fields.bulkViscosityCoefficient.push_back(float(sphParticlesBulkViscosityCoefficients[i]));
        fields.gradHCorrection.push_back(sphParticlesGradHCorrections[i]);
    }
    if (includeCandidates) {
        auto const candidates = sphFetchRungCandidates(Pair<ParticleInteger>(begin, end));
        fields.rungCandidate.insert(
            fields.rungCandidate.end(), candidates.begin(), candidates.end());
    }
}

template <class T>
void copyField(Vector<T>& destination, Vector<T> const& source, ParticleInteger offset) {
    std::copy(source.begin(), source.end(), destination.begin() + offset);
}

void receiveGhostLeaves(int owner, Vector<SphTreeNode> records, Vector<int> indices) {
    std::lock_guard<MutexType> lock(storeMutex);
    ALWAYS_ASSERT(records.size() == indices.size());
    if (store.ownerOffsets.contains(owner)) {
        THROW_ERROR("SPH ghosts: rank %i received two syncs from rank %i\n", hpxRank(), owner);
    }
    store.ownerOffsets[owner] = store.size;
    for (size_t leaf = 0; leaf < records.size(); leaf++) {
        auto const& record = records[leaf];
        int const slot = store.nodes.size();
        store.nodes.push_back(record);
        store.nodeOffsets.push_back(store.size);
        store.nodeById[ghostKey(owner, indices[leaf])] = slot;
        store.nodeByRangeStart[ghostKey(owner, record.particleRange.first)] = slot;
        TreeId id;
        id.proc = owner;
        id.index = indices[leaf];
        store.leafIds.push_back(id);
        store.size += record.numberOfParticles();
    }
    // Neighbor lists follow this order; sort it so it does not depend on arrival order.
    std::sort(store.leafIds.begin(), store.leafIds.end());
    resizeStore(store.size);
}

void receiveGhostFields(SphGhostFields fields) {
    // The store was sized by the sync; owners write disjoint blocks, so no lock is needed
    // for the copies, only for the lookup of the owner's block.
    ParticleInteger offset;
    {
        std::lock_guard<MutexType> lock(storeMutex);
        auto const found = store.ownerOffsets.find(fields.owner);
        if (found == store.ownerOffsets.end()) {
            THROW_ERROR("SPH ghosts: rank %i received fields from rank %i without a sync\n",
                hpxRank(), fields.owner);
        }
        offset = found->second;
    }
    ALWAYS_ASSERT(offset + ParticleInteger(fields.x.size()) <= store.size);
    auto& f = store.fields;
    copyField(f.x, fields.x, offset);
    copyField(f.y, fields.y, offset);
    copyField(f.z, fields.z, offset);
    copyField(f.internalEnergy, fields.internalEnergy, offset);
    copyField(f.smoothingLength, fields.smoothingLength, offset);
    copyField(f.rung, fields.rung, offset);
    copyField(f.vx, fields.vx, offset);
    copyField(f.vy, fields.vy, offset);
    copyField(f.vz, fields.vz, offset);
    copyField(f.density, fields.density, offset);
    copyField(f.balsaraFactor, fields.balsaraFactor, offset);
    copyField(f.bulkViscosityCoefficient, fields.bulkViscosityCoefficient, offset);
    copyField(f.gradHCorrection, fields.gradHCorrection, offset);
    if (!fields.rungCandidate.empty()) {
        copyField(f.rungCandidate, fields.rungCandidate, offset);
    }
}

}    // namespace detail

void sphGhostsClear();
Vector<detail::SphGhostInterest> sphGhostsPublish();
void sphGhostsExport(Vector<detail::SphGhostInterest> interests);
void sphGhostsRefreshLocal();
SphGhostStatistics sphGhostsStatistics();
void sphGhostsResetStatistics();
HPX_PLAIN_ACTION(sphGhostsClear, SphGhostsClearAction);
HPX_PLAIN_ACTION(sphGhostsPublish, SphGhostsPublishAction);
HPX_PLAIN_ACTION(sphGhostsExport, SphGhostsExportAction);
HPX_PLAIN_ACTION(sphGhostsRefreshLocal, SphGhostsRefreshLocalAction);
HPX_PLAIN_ACTION(sphGhostsStatistics, SphGhostsStatisticsAction);
HPX_PLAIN_ACTION(sphGhostsResetStatistics, SphGhostsResetStatisticsAction);
HPX_PLAIN_ACTION(detail::receiveGhostLeaves, SphGhostsReceiveLeavesAction);
HPX_PLAIN_ACTION(detail::receiveGhostFields, SphGhostsReceiveFieldsAction);

namespace detail {

// Wait for every future and keep the first failure, so no collective returns while a child
// or a push it started is still running.
template <class T>
std::exception_ptr drainGhostFutures(Vector<hpx::future<T>>& futures, std::exception_ptr failure) {
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    return failure;
}

template <class T>
void joinGhostFutures(Vector<hpx::future<T>>& futures) {
    if (auto const failure = drainGhostFutures(futures, nullptr)) {
        std::rethrow_exception(failure);
    }
}

}    // namespace detail

bool sphGhostsEnabled() {
    return getOptions().compute.sphGhosts;
}

bool sphGhostsActive() {
    return getOptions().compute.sphGhosts && hpxSize() > 1;
}

void sphGhostsSetEnabled(bool enabled) {
    ALWAYS_ASSERT(hpxRank() == 0);
    auto options = getOptions();
    options.compute.sphGhosts = enabled;
    setOptions(options);
    sphGhostsClear();
}

void sphGhostsClear() {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsClearAction>(child));
    }
    detail::clearLocalGhosts();
    detail::joinGhostFutures(children);
}

// Collective: drop this locality's ghosts and gather every locality's interest.
Vector<detail::SphGhostInterest> sphGhostsPublish() {
    Vector<hpx::future<Vector<detail::SphGhostInterest>>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsPublishAction>(child));
    }
    detail::clearLocalGhosts();
    Vector<detail::SphGhostInterest> interests{detail::localInterest()};
    std::exception_ptr failure;
    for (auto& child : children) {
        try {
            auto const childInterests = child.get();
            interests.insert(interests.end(), childInterests.begin(), childInterests.end());
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return interests;
}

// Collective: select this locality's exports against every interest and push the records.
void sphGhostsExport(Vector<detail::SphGhostInterest> interests) {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsExportAction>(child, interests));
    }
    Vector<hpx::future<void>> pushes;
    std::int64_t exportedLeaves = 0;
    std::int64_t recordBytes = 0;
    for (auto const& interest : interests) {
        if (interest.rank == hpxRank() || interest.empty()) {
            continue;
        }
        Vector<SphTreeNode> records;
        Vector<int> indices;
        std::vector<int> exported;
        for (int index = 0; index < sphTreeLeaflistSize(); index++) {
            auto const id = sphTreeGetLeaf(index);
            auto const& leaf = *sphTreeGetNode(id);
            if (leaf.numberOfParticles() == 0) {
                continue;
            }
            // The walk's opening test against the importer's interest.
            if (interest.reaches(leaf)) {
                records.push_back(leaf);
                indices.push_back(id.index);
                exported.push_back(id.index);
            }
        }
        if (records.empty()) {
            continue;
        }
        exportedLeaves += records.size();
        recordBytes += std::int64_t(records.size()) * (sizeof(SphTreeNode) + sizeof(int));
        {
            std::lock_guard<MutexType> lock(detail::storeMutex);
            detail::exports[interest.rank] = std::move(exported);
        }
        pushes.push_back(hpx::async<SphGhostsReceiveLeavesAction>(
            hpxLocalities()[interest.rank], hpxRank(), std::move(records), std::move(indices)));
    }
    auto failure = detail::drainGhostFutures(pushes, nullptr);
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        detail::counters.syncs++;
        detail::counters.exportedLeaves += exportedLeaves;
        detail::counters.recordBytes += recordBytes;
    }
    failure = detail::drainGhostFutures(children, failure);
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void sphGhostsSync() {
    ALWAYS_ASSERT(hpxRank() == 0);
    auto interests = sphGhostsPublish();
    sphGhostsExport(std::move(interests));
}

// Collective: push the current fields of every exported particle to its importer.
void sphGhostsRefreshLocal() {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsRefreshLocalAction>(child));
    }
    bool const includeCandidates = sphRungCandidatesActive();
    std::unordered_map<int, std::vector<int>> localExports;
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        localExports = detail::exports;
    }
    Vector<hpx::future<void>> pushes;
    std::int64_t fieldBytes = 0;
    for (auto const& [importer, leaves] : localExports) {
        detail::SphGhostFields fields;
        fields.owner = hpxRank();
        for (int const index : leaves) {
            TreeId id;
            id.proc = hpxRank();
            id.index = index;
            auto const& leaf = *sphTreeGetNode(id);
            detail::packRange(
                fields, leaf.particleRange.first, leaf.particleRange.second, includeCandidates);
        }
        fieldBytes += fields.bytes();
        pushes.push_back(
            hpx::async<SphGhostsReceiveFieldsAction>(hpxLocalities()[importer], std::move(fields)));
    }
    auto failure = detail::drainGhostFutures(pushes, nullptr);
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        detail::counters.refreshes++;
        detail::counters.fieldBytes += fieldBytes;
    }
    failure = detail::drainGhostFutures(children, failure);
    if (failure) {
        std::rethrow_exception(failure);
    }
}

void sphGhostsRefresh() {
    ALWAYS_ASSERT(hpxRank() == 0);
    sphGhostsRefreshLocal();
}

void sphGhostsBeginStaging() {
    detail::stagingDepth++;
}

void sphGhostsEndStaging() {
    detail::stagingDepth--;
}

bool sphGhostsServe(ParticleGlobalRange const&) {
    // Ghost fields are pushed for, and current during, the staging of a pass; the read throws
    // if the range is not a ghost leaf. Other remote reads see the owner's live data.
    return sphGhostsActive() && detail::stagingDepth > 0;
}

SphTreeNode const* sphGhostsFindNode(TreeId id) {
    auto const found = detail::store.nodeById.find(detail::ghostKey(id.proc, id.index));
    return found == detail::store.nodeById.end() ? nullptr : &detail::store.nodes[found->second];
}

Vector<TreeId> const& sphGhostsLeafIds() {
    return detail::store.leafIds;
}

void sphGhostsReadPositions(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    for (ParticleInteger i = 0; i < count; i++) {
        x[offset + i] = f.x[source + i];
        y[offset + i] = f.y[source + i];
        z[offset + i] = f.z[source + i];
    }
}

void sphGhostsReadInternalEnergyAndSmoothingLength(ParticleGlobalRange range, float* internalEnergy,
    float* smoothingLength, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    for (ParticleInteger i = 0; i < count; i++) {
        if (internalEnergy) {
            internalEnergy[offset + i] = f.internalEnergy[source + i];
        }
        if (smoothingLength) {
            smoothingLength[offset + i] = f.smoothingLength[source + i];
        }
    }
}

void sphGhostsReadRungs(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    std::copy(f.rung.begin() + source, f.rung.begin() + source + count, rungs + offset);
}

void sphGhostsReadVelocities(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    for (ParticleInteger i = 0; i < count; i++) {
        vx[offset + i] = f.vx[source + i];
        vy[offset + i] = f.vy[source + i];
        vz[offset + i] = f.vz[source + i];
    }
}

void sphGhostsReadDensity(ParticleGlobalRange range, float* density, ParticleInteger offset,
    float* balsaraFactor, float* bulkViscosityCoefficient, float* gradHCorrection) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    for (ParticleInteger i = 0; i < count; i++) {
        if (density) {
            density[offset + i] = f.density[source + i];
        }
        if (balsaraFactor) {
            balsaraFactor[offset + i] = f.balsaraFactor[source + i];
        }
        if (bulkViscosityCoefficient) {
            bulkViscosityCoefficient[offset + i] = f.bulkViscosityCoefficient[source + i];
        }
        if (gradHCorrection) {
            gradHCorrection[offset + i] = f.gradHCorrection[source + i];
        }
    }
}

void sphGhostsReadRungCandidates(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    ParticleInteger const source = detail::ghostOffset(range);
    ParticleInteger const count = range.range.second - range.range.first;
    auto const& f = detail::store.fields;
    std::copy(
        f.rungCandidate.begin() + source, f.rungCandidate.begin() + source + count, rungs + offset);
}

SphGhostStatistics& SphGhostStatistics::operator+=(SphGhostStatistics const& other) {
    leaves += other.leaves;
    particles += other.particles;
    exportedLeaves += other.exportedLeaves;
    syncs = std::max(syncs, other.syncs);
    refreshes = std::max(refreshes, other.refreshes);
    recordBytes += other.recordBytes;
    fieldBytes += other.fieldBytes;
    return *this;
}

SphGhostStatistics sphGhostsStatistics() {
    Vector<hpx::future<SphGhostStatistics>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsStatisticsAction>(child));
    }
    SphGhostStatistics result;
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        result = detail::counters;
        result.leaves = detail::store.nodes.size();
        result.particles = detail::store.size;
    }
    std::exception_ptr failure;
    for (auto& child : children) {
        try {
            result += child.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return result;
}

void sphGhostsResetStatistics() {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsResetStatisticsAction>(child));
    }
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        detail::counters = SphGhostStatistics();
    }
    detail::joinGhostFutures(children);
}
