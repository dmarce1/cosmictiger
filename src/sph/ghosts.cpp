/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <map>
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

    /// Report whether this interest packet contains any inner boxes.
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

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & rank;
        archive & innerUnion;
        archive & outerUnion;
        archive & innerBoxes;
        archive & outerBoxes;
    }
};

// Field groups, in the bit order of the kSphGhost* masks. Each group is a fixed-size record
// per particle, compared and copied as bytes.
enum GhostGroup : int {
    kGroupPositions,
    kGroupInternalEnergy,
    kGroupSmoothingLength,
    kGroupRung,
    kGroupVelocity,
    kGroupDensity,
    kGroupRungCandidate,
    kGroupCount
};

struct GhostPosition {
    Position x;
    Position y;
    Position z;
};

struct GhostVelocity {
    float vx;
    float vy;
    float vz;
};

struct GhostDensity {
    float density;
    float balsaraFactor;
    float bulkViscosityCoefficient;
    float gradHCorrection;
};

static_assert(sizeof(GhostPosition) == 3 * sizeof(Position));
static_assert(sizeof(GhostVelocity) == 3 * sizeof(float));
static_assert(sizeof(GhostDensity) == 4 * sizeof(float));
static_assert(kSphGhostAllFields == (1u << kGroupCount) - 1u);
static_assert(std::tuple_size_v<decltype(SphGhostStatistics::groupFieldBytes)> == kGroupCount);

inline constexpr std::array<size_t, kGroupCount> kGroupBytes = {sizeof(GhostPosition),
    sizeof(float), sizeof(float), sizeof(char), sizeof(GhostVelocity), sizeof(GhostDensity),
    sizeof(char)};

inline constexpr std::array<char const*, kGroupCount> kGroupNames = {"positions",
    "internal energies", "smoothing lengths", "rungs", "velocities", "densities",
    "rung candidates"};

/// Test whether a field-group bit is present in the requested ghost mask.
constexpr bool hasGroup(unsigned mask, int group) {
    return (mask >> group) & 1u;
}

// Owner side: what this locality exports to one importer, and what it last sent there. The
// importer holds exactly `sent` for every valid group, slot for slot in export order, so a
// refresh sends only the slots whose current bytes differ.
struct GhostExport {
    std::vector<int> leaves;
    std::vector<Pair<ParticleInteger>> ranges;
    ParticleInteger size = 0;
    std::array<std::vector<char>, kGroupCount> sent;
    std::array<bool, kGroupCount> valid{};
};

// Importer side: one owner's ghost leaves and the fields of their particles. A block survives
// a sync whose export from its owner is unchanged, keeping its fields, and is dropped by the
// first walk after a sync its owner did not confirm.
struct GhostBlock {
    std::int64_t syncEpoch = 0;
    std::vector<SphTreeNode> nodes;
    std::vector<int> indices;
    std::vector<ParticleInteger> offsets;
    ParticleInteger size = 0;
    std::array<std::vector<char>, kGroupCount> fields;
    // The refresh that last brought each group; a read requires the current refresh.
    std::array<std::int64_t, kGroupCount> generations{};
};

struct GhostLocation {
    GhostBlock const* block = nullptr;
    int node = -1;
};

// One group of a refresh: either every slot in order, or the listed slots.
struct GhostGroupUpdate {
    int group = -1;
    bool dense = false;
    Vector<std::uint32_t> slots;
    Vector<char> bytes;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & group;
        archive & dense;
        archive & slots;
        archive & bytes;
    }

    /// Return the byte count of field payloads and their changed-slot indices.
    std::int64_t payloadBytes() const {
        return std::int64_t(bytes.size()) + std::int64_t(slots.size()) * sizeof(std::uint32_t);
    }
};

struct GhostFieldsMessage {
    int owner = -1;
    std::int64_t generation = 0;
    unsigned mask = 0;
    Vector<GhostGroupUpdate> updates;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & owner;
        archive & generation;
        archive & mask;
        archive & updates;
    }
};

static MutexType storeMutex;
static std::unordered_map<int, GhostExport> exports;
static std::map<int, GhostBlock> blocks;
static std::unordered_map<std::uint64_t, GhostLocation> nodeById;
static std::unordered_map<std::uint64_t, GhostLocation> nodeByRangeStart;
static Vector<TreeId> leafIds;
static std::int64_t syncEpoch = 0;
static std::int64_t finalizedEpoch = 0;
static SphGhostStatistics counters;
static std::atomic<int> stagingDepth{0};
// The current refresh on this locality, set when its part of the refresh collective starts.
// Refreshes count from 1 and blocks start at 0, so before the first refresh nothing reads.
static std::atomic<std::int64_t> refreshGeneration{-1};

/// Pack a remote rank and node index into one 64-bit ghost lookup key.
std::uint64_t ghostKey(int rank, std::int64_t index) {
    return (std::uint64_t(std::uint32_t(rank)) << 32) | std::uint64_t(std::uint32_t(index));
}

/// Clear export plans, staged ghost blocks, and lookup indices under the ghost-store lock.
void clearLocalGhosts() {
    std::lock_guard<MutexType> lock(storeMutex);
    exports.clear();
    blocks.clear();
    nodeById.clear();
    nodeByRangeStart.clear();
    leafIds.clear();
    finalizedEpoch = syncEpoch;
}

// The ghost leaf holding a whole remote range, and that range's first slot in its block. Every
// group in `groups` must come from the current refresh.
std::pair<GhostBlock const*, ParticleInteger> ghostSlot(
    ParticleGlobalRange const& range, unsigned groups) {
    if (finalizedEpoch != syncEpoch) {
        throw std::logic_error(std::format(
            "SPH ghosts: rank {} reads ghosts before the last sync was finalized", hpxRank()));
    }
    auto const found = nodeByRangeStart.find(ghostKey(range.proc, range.range.first));
    if (found == nodeByRangeStart.end()) {
        throw std::logic_error(std::format(
            "SPH ghosts: rank {} holds no ghost leaf for rank {} gas range [{}, {})", hpxRank(),
            range.proc, (long long) range.range.first, (long long) range.range.second));
    }
    auto const& block = *found->second.block;
    int const node = found->second.node;
    if (range.range.second > block.nodes[node].particleRange.second) {
        throw std::logic_error("SPH ghosts: remote range outruns its ghost leaf");
    }
    for (int group = 0; group < kGroupCount; ++group) {
        if (hasGroup(groups, group) && block.generations[group] != refreshGeneration) {
            throw std::logic_error(
                std::format("SPH ghosts: rank {} reads rank {} {} that this pass did not refresh",
                    hpxRank(), range.proc, kGroupNames[group]));
        }
    }
    return {&block, block.offsets[node]};
}

/// Interpret one staged ghost field group's byte storage as the requested element type.
template <class T>
T const* groupData(GhostBlock const& block, int group) {
    return reinterpret_cast<T const*>(block.fields[group].data());
}

/// Expand a box component by component to include another box.
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

// Append the current bytes of one group for local gas range [begin, end) as the readers stage
// them; unused axes of a reduced-dimension build carry 0.5 for positions and zero for
// velocities.
void packGroup(std::vector<char>& out, int group, ParticleInteger begin, ParticleInteger end) {
    size_t const first = out.size();
    out.resize(first + size_t(end - begin) * kGroupBytes[group]);
    char* destination = out.data() + first;
    auto const put = [&destination](auto const& value) {
        std::memcpy(destination, &value, sizeof(value));
        destination += sizeof(value);
    };
    if (group == kGroupRungCandidate) {
        auto const candidates = sphFetchRungCandidates(Pair<ParticleInteger>(begin, end));
        for (char const candidate : candidates) {
            put(candidate);
        }
        return;
    }
    for (ParticleInteger i = begin; i < end; i++) {
        switch (group) {
        case kGroupPositions: {
            GhostPosition value;
            value.x = sphParticlesPositionValue(kXi, i);
            value.y = kSpatialDimensions > 1 ? sphParticlesPositionValue(kYi, i) : Position(0.5);
            value.z = kSpatialDimensions > 2 ? sphParticlesPositionValue(kZi, i) : Position(0.5);
            put(value);
        } break;
        case kGroupInternalEnergy:
            put(float(sphParticlesInternalEnergy(i)));
            break;
        case kGroupSmoothingLength:
            put(float(sphParticlesSmoothingLength(i)));
            break;
        case kGroupRung:
            put(char(sphParticlesRung(i)));
            break;
        case kGroupVelocity: {
            GhostVelocity value;
            value.vx = sphParticlesVelocityValue(kXi, i);
            value.vy = kSpatialDimensions > 1 ? sphParticlesVelocityValue(kYi, i) : 0.0f;
            value.vz = kSpatialDimensions > 2 ? sphParticlesVelocityValue(kZi, i) : 0.0f;
            put(value);
        } break;
        case kGroupDensity: {
            GhostDensity value;
            value.density = sphParticlesDensity(i);
            value.balsaraFactor = sphParticlesBalsaraFactors ?
                balsara::decodeFactor(sphParticlesBalsaraFactors[i]) :
                1.0f;
            value.bulkViscosityCoefficient = float(sphParticlesBulkViscosityCoefficients[i]);
            value.gradHCorrection = sphParticlesGradHCorrections[i];
            put(value);
        } break;
        default:
            ALWAYS_ASSERT(false);
        }
    }
}

// Install or update one owner's ghost leaves. `keep` means the owner exports the same leaves
// with the same particle ranges as when the block was built, so its fields stay valid. Every
// sync follows a bounds refresh, so the records themselves are always resent.
void receiveGhostLeaves(
    int owner, std::int64_t epoch, Vector<SphTreeNode> records, Vector<int> indices, bool keep) {
    std::lock_guard<MutexType> lock(storeMutex);
    ALWAYS_ASSERT(records.size() == indices.size());
    auto& block = blocks[owner];
    if (block.syncEpoch == epoch) {
        throw std::logic_error(
            std::format("SPH ghosts: rank {} received two syncs from rank {}", hpxRank(), owner));
    }
    if (keep) {
        bool same = block.indices.size() == indices.size();
        for (size_t leaf = 0; same && leaf < indices.size(); ++leaf) {
            same = block.indices[leaf] == indices[leaf] &&
                block.nodes[leaf].particleRange == records[leaf].particleRange;
        }
        if (!same) {
            throw std::logic_error(
                std::format("SPH ghosts: rank {} keeps a rank {} block whose leaves changed",
                    hpxRank(), owner));
        }
        block.nodes.assign(records.begin(), records.end());
    } else {
        block = GhostBlock();
        block.nodes.assign(records.begin(), records.end());
        block.indices.assign(indices.begin(), indices.end());
        for (auto const& record : block.nodes) {
            block.offsets.push_back(block.size);
            block.size += record.numberOfParticles();
        }
        for (int group = 0; group < kGroupCount; ++group) {
            block.fields[group].assign(size_t(block.size) * kGroupBytes[group], 0);
        }
    }
    block.syncEpoch = epoch;
}

/// Install full or sparse field updates into the synchronized owner's block and publish
/// field generations.
void receiveGhostFields(GhostFieldsMessage message) {
    // Owners write disjoint blocks; the lock covers only the lookup. Blocks are erased only by
    // the first walk after a sync, never while a refresh runs.
    GhostBlock* block;
    {
        std::lock_guard<MutexType> lock(storeMutex);
        auto const found = blocks.find(message.owner);
        if (found == blocks.end()) {
            throw std::logic_error(
                std::format("SPH ghosts: rank {} received fields from rank {} without a sync",
                    hpxRank(), message.owner));
        }
        block = &found->second;
    }
    for (auto const& update : message.updates) {
        size_t const recordBytes = kGroupBytes[update.group];
        auto& field = block->fields[update.group];
        if (update.dense) {
            ALWAYS_ASSERT(update.bytes.size() == field.size());
            std::copy(update.bytes.begin(), update.bytes.end(), field.begin());
        } else {
            ALWAYS_ASSERT(update.bytes.size() == update.slots.size() * recordBytes);
            for (size_t k = 0; k < update.slots.size(); ++k) {
                ALWAYS_ASSERT(update.slots[k] < std::uint32_t(block->size));
                std::memcpy(field.data() + size_t(update.slots[k]) * recordBytes,
                    update.bytes.data() + k * recordBytes, recordBytes);
            }
        }
    }
    for (int group = 0; group < kGroupCount; ++group) {
        if (hasGroup(message.mask, group)) {
            block->generations[group] = message.generation;
        }
    }
}

}    // namespace detail

void sphGhostsClear();
Vector<detail::SphGhostInterest> sphGhostsPublish(std::int64_t epoch);
void sphGhostsExport(std::int64_t epoch, Vector<detail::SphGhostInterest> interests);
void sphGhostsRefreshLocal(std::int64_t generation, unsigned fields);
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

/// Drain all ghost-exchange futures and propagate the first failure afterward.
template <class T>
void joinGhostFutures(Vector<hpx::future<T>>& futures) {
    if (auto const failure = drainGhostFutures(futures, nullptr)) {
        std::rethrow_exception(failure);
    }
}

}    // namespace detail

/// Report whether multiple localities require remote gas ghost exchange.
bool sphGhostsActive() {
    return hpxSize() > 1;
}

/// Clear staged ghost state throughout the SPH collective topology.
void sphGhostsClear() {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsClearAction>(child));
    }
    detail::clearLocalGhosts();
    detail::joinGhostFutures(children);
}

// Collective: start sync `epoch` and gather every locality's interest. Blocks are kept: the
// exports decide which survive.
Vector<detail::SphGhostInterest> sphGhostsPublish(std::int64_t epoch) {
    Vector<hpx::future<Vector<detail::SphGhostInterest>>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsPublishAction>(child, epoch));
    }
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        detail::syncEpoch = epoch;
    }
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
// An export identical to the previous one to that importer keeps the importer's fields.
void sphGhostsExport(std::int64_t epoch, Vector<detail::SphGhostInterest> interests) {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsExportAction>(child, epoch, interests));
    }
    Vector<hpx::future<void>> pushes;
    std::int64_t exportedLeaves = 0;
    std::int64_t recordBytes = 0;
    std::exception_ptr failure;
    try {
        for (auto const& interest : interests) {
            if (interest.rank == hpxRank()) {
                continue;
            }
            Vector<SphTreeNode> records;
            Vector<int> indices;
            std::vector<Pair<ParticleInteger>> ranges;
            if (!interest.empty()) {
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
                        ranges.push_back(leaf.particleRange);
                    }
                }
            }
            bool keep = false;
            {
                std::lock_guard<MutexType> lock(detail::storeMutex);
                if (records.empty()) {
                    // The importer drops this owner's block at its next walk.
                    detail::exports.erase(interest.rank);
                    continue;
                }
                std::vector<int> leaves(indices.begin(), indices.end());
                auto const previous = detail::exports.find(interest.rank);
                keep = previous != detail::exports.end() && previous->second.leaves == leaves &&
                    previous->second.ranges == ranges;
                if (!keep) {
                    detail::GhostExport target;
                    target.leaves = std::move(leaves);
                    target.ranges = std::move(ranges);
                    for (auto const& range : target.ranges) {
                        target.size += range.second - range.first;
                    }
                    detail::exports[interest.rank] = std::move(target);
                }
            }
            exportedLeaves += records.size();
            recordBytes += std::int64_t(records.size()) * (sizeof(SphTreeNode) + sizeof(int));
            pushes.push_back(
                hpx::async<SphGhostsReceiveLeavesAction>(hpxLocalities()[interest.rank], hpxRank(),
                    epoch, std::move(records), std::move(indices), keep));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    failure = detail::drainGhostFutures(pushes, failure);
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

/// Start a new synchronization epoch and exchange interested remote leaf records from rank
/// zero.
void sphGhostsSync() {
    ALWAYS_ASSERT(hpxRank() == 0);
    static std::int64_t epoch = 0;
    ++epoch;
    auto interests = sphGhostsPublish(epoch);
    sphGhostsExport(epoch, std::move(interests));
}

/// Discard stale ghost blocks and rebuild deterministic node/range lookup indices for the
/// latest epoch.
void sphGhostsFinalizeSync() {
    std::lock_guard<MutexType> lock(detail::storeMutex);
    if (detail::finalizedEpoch == detail::syncEpoch) {
        return;
    }
    for (auto block = detail::blocks.begin(); block != detail::blocks.end();) {
        if (block->second.syncEpoch != detail::syncEpoch) {
            block = detail::blocks.erase(block);
        } else {
            ++block;
        }
    }
    detail::nodeById.clear();
    detail::nodeByRangeStart.clear();
    detail::leafIds.clear();
    for (auto const& [owner, block] : detail::blocks) {
        for (int node = 0; node < int(block.nodes.size()); ++node) {
            detail::GhostLocation location;
            location.block = &block;
            location.node = node;
            detail::nodeById[detail::ghostKey(owner, block.indices[node])] = location;
            detail::nodeByRangeStart[detail::ghostKey(
                owner, block.nodes[node].particleRange.first)] = location;
            TreeId id;
            id.proc = owner;
            id.index = block.indices[node];
            detail::leafIds.push_back(id);
        }
    }
    // Neighbor lists follow this order; keep it independent of arrival order.
    std::sort(detail::leafIds.begin(), detail::leafIds.end());
    detail::finalizedEpoch = detail::syncEpoch;
}

// Collective: push the requested groups of every exported particle to its importer, only the
// slots whose bytes differ from what that importer already holds.
void sphGhostsRefreshLocal(std::int64_t generation, unsigned fields) {
    Vector<hpx::future<void>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsRefreshLocalAction>(child, generation, fields));
    }
    detail::refreshGeneration = generation;
    // The all-resent baseline is what the M5 refresh sent: every group, candidates only while
    // the rung balance runs.
    bool const candidatesExist = sphRungCandidatesActive();
    Vector<hpx::future<void>> pushes;
    std::int64_t fieldBytes = 0;
    std::int64_t neededFieldBytes = 0;
    std::int64_t allFieldBytes = 0;
    std::array<std::int64_t, detail::kGroupCount> groupFieldBytes{};
    std::exception_ptr failure;
    try {
        // Exports change only in the export collective, never during a refresh, so packing
        // runs unlocked and other owners' deliveries to this locality are not held up.
        for (auto& [importer, target] : detail::exports) {
            // The export must still describe the current tree: a rebuild or migration without
            // a sync would pack particles the importer's records do not name.
            for (size_t leaf = 0; leaf < target.leaves.size(); ++leaf) {
                TreeId id;
                id.proc = hpxRank();
                id.index = target.leaves[leaf];
                if (target.ranges[leaf].second > sphParticlesSize() ||
                    id.index >= sphTreeNodesSize() ||
                    sphTreeGetNode(id)->particleRange != target.ranges[leaf]) {
                    throw std::logic_error(
                        std::format("SPH ghosts: rank {} refreshes an export to rank {} that "
                                    "predates the current tree; a sync must follow every rebuild",
                            hpxRank(), importer));
                }
            }
            detail::GhostFieldsMessage message;
            message.owner = hpxRank();
            message.generation = generation;
            message.mask = fields;
            for (int group = 0; group < detail::kGroupCount; ++group) {
                size_t const recordBytes = detail::kGroupBytes[group];
                if (group != detail::kGroupRungCandidate || candidatesExist) {
                    allFieldBytes += target.size * std::int64_t(recordBytes);
                }
                if (!detail::hasGroup(fields, group)) {
                    continue;
                }
                neededFieldBytes += target.size * std::int64_t(recordBytes);
                std::vector<char> current;
                current.reserve(size_t(target.size) * recordBytes);
                for (auto const& range : target.ranges) {
                    detail::packGroup(current, group, range.first, range.second);
                }
                detail::GhostGroupUpdate update;
                update.group = group;
                if (!target.valid[group]) {
                    update.dense = true;
                    update.bytes.assign(current.begin(), current.end());
                } else {
                    auto const& sent = target.sent[group];
                    for (ParticleInteger slot = 0; slot < target.size; ++slot) {
                        size_t const at = size_t(slot) * recordBytes;
                        if (std::memcmp(current.data() + at, sent.data() + at, recordBytes) != 0) {
                            update.slots.push_back(std::uint32_t(slot));
                            update.bytes.insert(update.bytes.end(), current.begin() + at,
                                current.begin() + at + recordBytes);
                        }
                    }
                    // A dense update is smaller once most slots changed.
                    if (update.payloadBytes() >= std::int64_t(current.size())) {
                        update.dense = true;
                        update.slots.clear();
                        update.bytes.assign(current.begin(), current.end());
                    }
                }
                fieldBytes += update.payloadBytes();
                groupFieldBytes[group] += update.payloadBytes();
                target.sent[group] = std::move(current);
                target.valid[group] = true;
                if (update.dense || !update.slots.empty()) {
                    message.updates.push_back(std::move(update));
                }
            }
            // Sent even when nothing changed: it moves the importer's groups to this refresh.
            pushes.push_back(hpx::async<SphGhostsReceiveFieldsAction>(
                hpxLocalities()[importer], std::move(message)));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    failure = detail::drainGhostFutures(pushes, failure);
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        detail::counters.refreshes++;
        detail::counters.fieldBytes += fieldBytes;
        detail::counters.neededFieldBytes += neededFieldBytes;
        detail::counters.allFieldBytes += allFieldBytes;
        for (int group = 0; group < detail::kGroupCount; ++group) {
            detail::counters.groupFieldBytes[group] += groupFieldBytes[group];
        }
    }
    failure = detail::drainGhostFutures(children, failure);
    if (failure) {
        std::rethrow_exception(failure);
    }
}

/// Push requested current fields for a new generation, omitting rung candidates outside
/// balancing.
void sphGhostsRefresh(unsigned fields) {
    ALWAYS_ASSERT(hpxRank() == 0);
    if (!sphRungCandidatesActive()) {
        // Candidates exist only while the rung balance runs.
        fields &= ~kSphGhostRungCandidates;
    }
    static std::int64_t generation = 0;
    sphGhostsRefreshLocal(++generation, fields);
}

/// Enter a scope in which remote gas reads may consume refreshed ghost fields.
void sphGhostsBeginStaging() {
    detail::stagingDepth++;
}

/// Leave the current remote-gas staging scope.
void sphGhostsEndStaging() {
    detail::stagingDepth--;
}

/// Reject a remote gas read made outside an active staging scope.
void sphGhostsRequireStaging(ParticleGlobalRange const& range) {
    // Ghost fields are pushed for, and current during, the staging of a pass. No other code
    // may read remote gas: there is no fetch path to serve it.
    if (detail::stagingDepth <= 0) {
        throw std::logic_error(
            std::format("SPH ghosts: rank {} reads rank {} gas outside the staging of a pass",
                hpxRank(), range.proc));
    }
}

/// Find a staged remote leaf by TreeId, requiring synchronization to have been finalized.
SphTreeNode const* sphGhostsFindNode(TreeId id) {
    if (detail::finalizedEpoch != detail::syncEpoch) {
        throw std::logic_error(std::format(
            "SPH ghosts: rank {} looks up a ghost before the last sync was finalized", hpxRank()));
    }
    auto const found = detail::nodeById.find(detail::ghostKey(id.proc, id.index));
    return found == detail::nodeById.end() ? nullptr :
                                             &found->second.block->nodes[found->second.node];
}

/// Return the deterministically ordered list of staged remote leaf IDs.
Vector<TreeId> const& sphGhostsLeafIds() {
    return detail::leafIds;
}

/// Copy staged remote positions into the caller's component arrays.
void sphGhostsReadPositions(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    auto const [block, source] = detail::ghostSlot(range, 1u << detail::kGroupPositions);
    auto const* values = detail::groupData<detail::GhostPosition>(*block, detail::kGroupPositions);
    ParticleInteger const count = range.range.second - range.range.first;
    for (ParticleInteger i = 0; i < count; i++) {
        x[offset + i] = values[source + i].x;
        y[offset + i] = values[source + i].y;
        z[offset + i] = values[source + i].z;
    }
}

/// Copy the requested staged thermal and support-length fields into caller buffers.
void sphGhostsReadInternalEnergyAndSmoothingLength(ParticleGlobalRange range, float* internalEnergy,
    float* smoothingLength, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    unsigned const groups = (internalEnergy ? 1u << detail::kGroupInternalEnergy : 0u) |
        (smoothingLength ? 1u << detail::kGroupSmoothingLength : 0u);
    auto const [block, source] = detail::ghostSlot(range, groups);
    ParticleInteger const count = range.range.second - range.range.first;
    if (internalEnergy) {
        auto const* values = detail::groupData<float>(*block, detail::kGroupInternalEnergy);
        std::copy(values + source, values + source + count, internalEnergy + offset);
    }
    if (smoothingLength) {
        auto const* values = detail::groupData<float>(*block, detail::kGroupSmoothingLength);
        std::copy(values + source, values + source + count, smoothingLength + offset);
    }
}

/// Copy staged remote gas rung values into the caller's buffer.
void sphGhostsReadRungs(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    auto const [block, source] = detail::ghostSlot(range, 1u << detail::kGroupRung);
    auto const* values = detail::groupData<char>(*block, detail::kGroupRung);
    ParticleInteger const count = range.range.second - range.range.first;
    std::copy(values + source, values + source + count, rungs + offset);
}

/// Copy staged remote velocity components into the caller's arrays.
void sphGhostsReadVelocities(
    ParticleGlobalRange range, float* vx, float* vy, float* vz, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    auto const [block, source] = detail::ghostSlot(range, 1u << detail::kGroupVelocity);
    auto const* values = detail::groupData<detail::GhostVelocity>(*block, detail::kGroupVelocity);
    ParticleInteger const count = range.range.second - range.range.first;
    for (ParticleInteger i = 0; i < count; i++) {
        vx[offset + i] = values[source + i].vx;
        vy[offset + i] = values[source + i].vy;
        vz[offset + i] = values[source + i].vz;
    }
}

/// Copy requested staged density, viscosity, and grad-h fields into caller buffers.
void sphGhostsReadDensity(ParticleGlobalRange range, float* density, ParticleInteger offset,
    float* balsaraFactor, float* bulkViscosityCoefficient, float* gradHCorrection) {
    if (range.range.first == range.range.second) {
        return;
    }
    auto const [block, source] = detail::ghostSlot(range, 1u << detail::kGroupDensity);
    auto const* values = detail::groupData<detail::GhostDensity>(*block, detail::kGroupDensity);
    ParticleInteger const count = range.range.second - range.range.first;
    for (ParticleInteger i = 0; i < count; i++) {
        auto const& value = values[source + i];
        if (density) {
            density[offset + i] = value.density;
        }
        if (balsaraFactor) {
            balsaraFactor[offset + i] = value.balsaraFactor;
        }
        if (bulkViscosityCoefficient) {
            bulkViscosityCoefficient[offset + i] = value.bulkViscosityCoefficient;
        }
        if (gradHCorrection) {
            gradHCorrection[offset + i] = value.gradHCorrection;
        }
    }
}

/// Copy staged remote candidate-rung values for the current balance pass.
void sphGhostsReadRungCandidates(ParticleGlobalRange range, char* rungs, ParticleInteger offset) {
    if (range.range.first == range.range.second) {
        return;
    }
    auto const [block, source] = detail::ghostSlot(range, 1u << detail::kGroupRungCandidate);
    auto const* values = detail::groupData<char>(*block, detail::kGroupRungCandidate);
    ParticleInteger const count = range.range.second - range.range.first;
    std::copy(values + source, values + source + count, rungs + offset);
}

/// Merge ghost counts and byte totals, retaining maximum synchronization and refresh
/// generations.
SphGhostStatistics& SphGhostStatistics::operator+=(SphGhostStatistics const& other) {
    leaves += other.leaves;
    particles += other.particles;
    exportedLeaves += other.exportedLeaves;
    syncs = std::max(syncs, other.syncs);
    refreshes = std::max(refreshes, other.refreshes);
    recordBytes += other.recordBytes;
    fieldBytes += other.fieldBytes;
    neededFieldBytes += other.neededFieldBytes;
    allFieldBytes += other.allFieldBytes;
    for (size_t group = 0; group < groupFieldBytes.size(); ++group) {
        groupFieldBytes[group] += other.groupFieldBytes[group];
    }
    return *this;
}

/// Snapshot local ghost occupancy and reduce exchange statistics across the SPH topology.
SphGhostStatistics sphGhostsStatistics() {
    Vector<hpx::future<SphGhostStatistics>> children;
    for (auto const& child : sphCollectiveChildren()) {
        children.push_back(hpx::async<SphGhostsStatisticsAction>(child));
    }
    SphGhostStatistics result;
    {
        std::lock_guard<MutexType> lock(detail::storeMutex);
        result = detail::counters;
        result.leaves = 0;
        result.particles = 0;
        for (auto const& [owner, block] : detail::blocks) {
            result.leaves += block.nodes.size();
            result.particles += block.size;
        }
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

/// Clear accumulated ghost-exchange counters throughout the SPH topology.
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
