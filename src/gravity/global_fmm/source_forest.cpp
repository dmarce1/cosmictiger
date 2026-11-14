/* CosmicTiger - geometric FMM source-forest exchange; BSL-1.0. */

// Certify, exchange and validate source forests; serve imported data and record measurements.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <sys/resource.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/global_fmm/measurement.hpp>
#include <cosmictiger/gravity/global_fmm/remote_cache.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree_build.hpp>

struct SourceIdHash {
    size_t operator()(TreeId id) const {
        return std::hash<uint64_t>{}((uint64_t(uint32_t(id.proc)) << 32) | uint32_t(id.index));
    }
};

struct ImportedSourceNode {
    GeometricTreeCell cell;            // Exact dyadic identity of the owner's source cell.
    TreeNode node;                     // Imported metadata with multipole storage on this locality.
    unsigned char measuredUsed = 0;    // Atomic first-use marker for each measured walk.
};
struct ImportedSourceParticle {
    std::array<Position, kDimensions> position;    // Fixed-point coordinates in the unit box.
    char type;    // Gravity mass class: dark matter or gas (including stars).
};
using ImportedNodes = std::unordered_map<TreeId, ImportedSourceNode, SourceIdHash>;
using ImportedParticles = std::unordered_map<uint64_t, ImportedSourceParticle>;

ImportedNodes importedNodes;
ImportedParticles importedParticles;
std::unordered_set<TreeId, SourceIdHash> materializedNodes;
Vector<SourceRootDescriptor> expectedRoots;
Vector<unsigned char> receivedOwners;
SourceRootDescriptor ownedRoot;
bool ownedHydroEnabled = false;
double ownedDarkMatterMass = 0;
double ownedGasMass = 0;
bool ownerReady = false;
bool importsReady = false;
MutexType importMutex;
std::atomic<bool> importsEnabled{false};
std::atomic<uint64_t> fallbackCount{0};

std::atomic<bool> fmmMeasurementEnabled{false};
FmmMeasurement measurement;
MutexType measurementMutex;

void fmmMeasureSeconds(FmmMeasurementPhase phase, double seconds) {
    std::lock_guard<MutexType> lock(measurementMutex);
    measurement.seconds[size_t(phase)] += seconds;
}

void fmmMeasureDemand(
    int peer, uint64_t requestBytes, uint64_t responseBytes, uint64_t cacheBytes) {
    if (!fmmMeasurementEnabled || peer == hpxRank()) {
        return;
    }
    std::lock_guard<MutexType> lock(measurementMutex);
    auto& entry = measurement.peers[peer];
    measurement.demandCacheValueBytes += cacheBytes;
    entry.demandSentBytes += requestBytes;
    entry.demandReceivedBytes += responseBytes;
    ++entry.demandSentMessages;
    ++entry.demandReceivedMessages;
}

void measurementControlLocal(int command) {
    fmmMeasurementEnabled = false;
    if (command == 0) {
        measurement = {};
        measurement.rank = hpxRank();
        measurement.peers.resize(hpxSize());
    } else if (command == 1) {
        // Keep the forest and frozen source generation, discard demand caches.
        treeClearRemoteCache();
        materializedNodes.clear();
        for (auto& [id, record] : importedNodes) {
            record.measuredUsed = 0;
        }
        fallbackCount = 0;
        measurement.fallbackRequests = 0;
        measurement.demandCacheValueBytes = 0;
        measurement.seconds[size_t(FmmMeasurementPhase::Traversal)] = 0;
        measurement.seconds[size_t(FmmMeasurementPhase::GpuStaging)] = 0;
        for (auto& peer : measurement.peers) {
            peer.usedNodes = 0;
            peer.demandSentBytes = peer.demandReceivedBytes = 0;
            peer.demandSentMessages = peer.demandReceivedMessages = 0;
        }
    }
    fmmMeasurementEnabled = command != 2;
}
HPX_PLAIN_ACTION(measurementControlLocal, FmmMeasurementControlAction);

void measurementControl(int command) {
    auto pending = detail::geometricLaunchAndDrain<void>(hpxSize(), [&](size_t rank) {
        return hpx::async<FmmMeasurementControlAction>(hpxLocalities()[rank], command);
    });
    for (auto& future : pending) {
        future.get();
    }
}
void fmmMeasurementBegin() {
    measurementControl(0);
}
void fmmMeasurementPrepareWalk() {
    particlesCacheFree();
    measurementControl(1);
}
void fmmMeasurementEnd() {
    measurementControl(2);
}

FmmMeasurement measurementReadLocal() {
    std::lock_guard<MutexType> lock(measurementMutex);
    measurement.fallbackRequests = fallbackCount.load();
    measurement.ownedArenaBytes = uint64_t(treeNodesSize()) *
        (sizeof(TreeNode) + fmmPositionedMultipoleStride(kFmmOrder) + sizeof(unsigned char));
    measurement.remoteMultipoleBytes = treeRemoteMultipoleBytes();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        measurement.processPeakRssBytes = uint64_t(usage.ru_maxrss) * 1024;
    }
    measurement.importedStorageBytes =
        importedNodes.size() * (sizeof(ImportedNodes::value_type) + 2 * sizeof(void*)) +
        importedNodes.bucket_count() * sizeof(void*) +
        importedParticles.size() * (sizeof(ImportedParticles::value_type) + 2 * sizeof(void*)) +
        importedParticles.bucket_count() * sizeof(void*);
    return measurement;
}
HPX_PLAIN_ACTION(measurementReadLocal, FmmMeasurementReadAction);
Vector<FmmMeasurement> fmmMeasurementRead() {
    auto pending = detail::geometricLaunchAndDrain<FmmMeasurement>(hpxSize(),
        [&](size_t rank) { return hpx::async<FmmMeasurementReadAction>(hpxLocalities()[rank]); });
    Vector<FmmMeasurement> result;
    result.reserve(pending.size());
    for (auto& future : pending) {
        result.push_back(future.get());
    }
    // Demand counters are collected at the requester. Mirror them here to
    // expose both peer directions without sending measurement-only RPCs.
    auto const original = result;
    for (size_t requester = 0; requester < result.size(); ++requester) {
        for (size_t owner = 0; owner < result.size(); ++owner) {
            if (owner != requester) {
                auto const& request = original[requester].peers[owner];
                auto& response = result[owner].peers[requester];
                response.demandSentBytes += request.demandReceivedBytes;
                response.demandReceivedBytes += request.demandSentBytes;
                response.demandSentMessages += request.demandReceivedMessages;
                response.demandReceivedMessages += request.demandSentMessages;
            }
        }
    }
    return result;
}

bool sameSourceRoot(SourceRootDescriptor const& left, SourceRootDescriptor const& right) {
    return left.root == right.root && left.generation == right.generation &&
        left.geometryVersion == right.geometryVersion && left.sourceCount == right.sourceCount &&
        left.nodeCount == right.nodeCount && left.leafCount == right.leafCount &&
        left.ewaldDepth == right.ewaldDepth && left.theta == right.theta &&
        left.smoothingLength == right.smoothingLength && left.gpu == right.gpu &&
        left.cover.begin == right.cover.begin && left.cover.end == right.cover.end;
}

void validateOwnedRoot(Vector<SourceRootDescriptor> const& roots) {
    if (!ownerReady || roots.size() != size_t(hpxSize()) ||
        !sameSourceRoot(roots[hpxRank()], ownedRoot)) {
        throw std::runtime_error("Geometric owned source generation is not ready");
    }
    for (size_t rank = 0; rank < roots.size(); ++rank) {
        auto const& root = roots[rank];
        if (root.root.proc != int(rank) || root.root.index < 0 || root.generation == 0 ||
            root.generation != ownedRoot.generation ||
            root.geometryVersion != kGeometricSourceForestVersion ||
            root.theta != ownedRoot.theta || root.smoothingLength != ownedRoot.smoothingLength ||
            root.gpu != ownedRoot.gpu) {
            throw std::runtime_error("Inconsistent geometric source-root descriptors");
        }
    }
}

uint64_t particleKey(int owner, ParticleInteger index) {
    return (uint64_t(uint32_t(owner)) << 32) | uint32_t(index);
}

long double lowerSquare(long double value) {
    return std::nextafter(value * value, -std::numeric_limits<long double>::infinity());
}

bool terminalCertified(TreeNode const& source, SourceRootDescriptor const& destination) {
    if (destination.sourceCount == 0 || source.numberOfParticles() == 0) {
        return true;
    }
    auto const& cover = destination.cover;
    constexpr std::array<int, 3> kNearestImageOffsets{-1, 0, 1};
    long double squaredDistance = 0;
    for (int axis = 0; axis < kDimensions; ++axis) {
        long double const point =
            std::ldexp(static_cast<long double>(source.pos[axis].raw()), -kPositionBits);
        long double const begin = cover.begin[axis];
        long double const end = cover.end[axis];
        long double axisDistance = 1;
        for (int const image : kNearestImageOffsets) {
            long double const shifted = point + image;
            long double const distance =
                shifted < begin ? begin - shifted : (shifted > end ? shifted - end : 0);
            long double const lower = std::max(0.0L,
                std::nextafter(distance, -std::numeric_limits<long double>::infinity()) -
                    32 * std::numeric_limits<long double>::epsilon());
            axisDistance = std::min(axisDistance, lower);
        }
        squaredDistance = std::nextafter(squaredDistance + lowerSquare(axisDistance),
            -std::numeric_limits<long double>::infinity());
    }
    long double const infinity = std::numeric_limits<long double>::infinity();
    long double const radii =
        std::nextafter(static_cast<long double>(geometricTreeRadiusLimit(source.depth)) +
                static_cast<long double>(geometricTreeRadiusLimit(destination.ewaldDepth)),
            infinity);
    long double const opening = std::nextafter(
        radii * static_cast<long double>(fmmInverseOpeningAngle(destination.theta)), infinity);
    long double const support =
        std::nextafter(radii + static_cast<long double>(destination.smoothingLength), infinity);
    long double const threshold = std::nextafter(std::max(opening, support), infinity);
    long double const thresholdSquared = std::nextafter(threshold * threshold, infinity);
    return squaredDistance > thresholdSquared;
}

void exportNode(SourceExportView& view, TreeId id, GeometricTreeCell const& cell,
    SourceRootDescriptor const& destination, FmmSourceView const& sources) {
    TreeNode node = *treeLocalNode(id);
    bool const terminal = !node.leaf && terminalCertified(node, destination);
    if (terminal) {
        node.sourceKind = kTreeSourceMultipoleTerminal;
        node.leaf = true;
        for (auto& child : node.children) {
            child = TreeId{-1, -1};
        }
    } else if (node.numberOfParticles() == 0) {
        node.sourceKind = kTreeSourceEmpty;
    } else {
        node.sourceKind = node.leaf ? kTreeSourceParticleLeaf : kTreeSourceInternal;
    }
    view.nodes.push_back(SourceExportNode{id, cell, node});
    if (node.numberOfParticles() == 0 || (terminal && !destination.gpu)) {
        return;
    }
    if (node.leaf) {
        for (ParticleInteger index = node.particleRange.first; index < node.particleRange.second;
            ++index) {
            SourceExportParticle particle;
            for (int axis = 0; axis < kDimensions; ++axis) {
                particle.position[axis] = sources.positions[axis][index];
            }
            view.appendParticle(
                particle, sources.types == nullptr ? kDarkMatterType : sources.types[index]);
        }
        return;
    }
    for (int child = 0; child < kChildCount; ++child) {
        TreeId const childId = node.children[child];
        exportNode(
            view, childId, geometricTreeChild(cell, child == kRightChild), destination, sources);
    }
}

void beginImports(Vector<SourceRootDescriptor> roots) {
    std::lock_guard<MutexType> lock(importMutex);
    validateOwnedRoot(roots);
    importsEnabled = false;
    importsReady = false;
    importedNodes.clear();
    importedParticles.clear();
    materializedNodes.clear();
    expectedRoots = std::move(roots);
    receivedOwners = Vector<unsigned char>(expectedRoots.size(), 0);
}
HPX_PLAIN_ACTION(beginImports, BeginSourceImportsAction);

void installExport(SourceExportView view) {
    FmmMeasurementScope timer(FmmMeasurementPhase::Import);
    std::lock_guard<MutexType> lock(importMutex);
    if (view.destination != hpxRank() || view.owner < 0 ||
        size_t(view.owner) >= expectedRoots.size() || view.owner == hpxRank() ||
        receivedOwners[view.owner]) {
        throw std::runtime_error("Unexpected or duplicate geometric source export");
    }
    auto const& owner = expectedRoots[view.owner];
    auto const& destination = expectedRoots[hpxRank()];
    if (view.generation != owner.generation ||
        view.geometryVersion != kGeometricSourceForestVersion ||
        owner.geometryVersion != kGeometricSourceForestVersion ||
        destination.geometryVersion != kGeometricSourceForestVersion ||
        view.positionBits != kPositionBits || view.maximumDepth != kGeometricTreeMaxDepth ||
        view.order != kFmmOrder || view.destinationEwaldDepth != destination.ewaldDepth ||
        view.theta != destination.theta || view.smoothingLength != destination.smoothingLength ||
        view.destinationGpu != destination.gpu ||
        view.destinationCover.begin != destination.cover.begin ||
        view.destinationCover.end != destination.cover.end) {
        throw std::runtime_error("Stale or incompatible geometric source export");
    }
    if (destination.sourceCount != 0 && owner.sourceCount != 0 && view.nodes.empty()) {
        throw std::runtime_error("Missing nonempty source export root");
    }
    std::unordered_set<TreeId, SourceIdHash> present;
    for (auto const& record : view.nodes) {
        if (record.id.proc != view.owner || !present.insert(record.id).second ||
            record.node.processRange.first != view.owner ||
            record.node.processRange.second != view.owner + 1 ||
            record.node.depth != record.cell.depth || record.node.particleRange.first < 0 ||
            record.node.particleRange.second < record.node.particleRange.first) {
            throw std::runtime_error("Invalid geometric source export node");
        }
        for (int axis = 0; axis < kDimensions; ++axis) {
            if (record.node.pos[axis].raw() != geometricTreeCenter(record.cell, axis).raw()) {
                throw std::runtime_error("Geometric source center does not match its identity");
            }
        }
    }
    if (destination.sourceCount != 0 && owner.sourceCount != 0 && !present.contains(owner.root)) {
        throw std::runtime_error("Geometric source export omits its root");
    }
    std::unordered_set<uint64_t> particles;
    if ((view.uniformParticleType != kDarkMatterType && view.uniformParticleType != kSphType) ||
        (!view.particleTypes.empty() && view.particleTypes.size() != view.particles.size())) {
        throw std::runtime_error("Invalid geometric source species payload");
    }
    for (char const type : view.particleTypes) {
        if (type != kDarkMatterType && type != kSphType) {
            throw std::runtime_error("Unknown geometric source gravity mass class");
        }
    }
    if (fmmMeasurementEnabled) {
        std::lock_guard<MutexType> measureLock(measurementMutex);
        auto& peer = measurement.peers[view.owner];
        peer.receivedBytes += fmmSerializedBytes(view);
        ++peer.receivedMessages;
        peer.importedNodes += view.nodes.size();
        peer.importedParticles += view.particles.size();
        for (auto const& record : view.nodes) {
            if (record.node.leaf) {
                peer.coveredSources += record.node.numberOfParticles();
                if (record.node.sourceKind == kTreeSourceMultipoleTerminal) {
                    peer.coarseSources += record.node.numberOfParticles();
                }
            }
        }
        peer.fineDetail = !view.particles.empty();
    }
    size_t particleOffset = 0;
    for (auto const& record : view.nodes) {
        auto const& node = record.node;
        if (node.sourceKind == kTreeSourceMultipoleTerminal) {
            if (!node.leaf) {
                throw std::runtime_error("Multipole terminal is not terminal");
            }
        } else if (node.sourceKind == kTreeSourceEmpty) {
            if (!node.leaf || node.numberOfParticles() != 0) {
                throw std::runtime_error("Invalid empty geometric source node");
            }
        } else if (node.sourceKind == kTreeSourceParticleLeaf) {
            if (!node.leaf || node.numberOfParticles() == 0) {
                throw std::runtime_error("Invalid geometric source particle leaf");
            }
        } else if (node.sourceKind == kTreeSourceInternal) {
            if (node.leaf) {
                throw std::runtime_error("Invalid internal geometric source node");
            }
            for (auto const& child : node.children) {
                if (child.index < 0 || !present.contains(child)) {
                    throw std::runtime_error("Geometric source export is missing a child");
                }
            }
        } else {
            throw std::runtime_error("Unknown geometric source node kind");
        }
        if (node.sourceKind == kTreeSourceParticleLeaf ||
            (node.sourceKind == kTreeSourceMultipoleTerminal && destination.gpu)) {
            size_t const count = node.numberOfParticles();
            if (count > view.particles.size() - particleOffset) {
                throw std::runtime_error("Geometric source leaf is missing particles");
            }
            for (ParticleInteger index = node.particleRange.first;
                index < node.particleRange.second; ++index) {
                if (!particles.insert(particleKey(view.owner, index)).second) {
                    throw std::runtime_error("Overlapping geometric source particle ranges");
                }
            }
            particleOffset += count;
        }
    }
    if (particleOffset != view.particles.size()) {
        throw std::runtime_error("Geometric source export has unreferenced particles");
    }
    particleOffset = 0;
    for (auto& record : view.nodes) {
        auto const& node = record.node;
        if (node.sourceKind == kTreeSourceParticleLeaf ||
            (node.sourceKind == kTreeSourceMultipoleTerminal && destination.gpu)) {
            for (ParticleInteger index = node.particleRange.first;
                index < node.particleRange.second; ++index, ++particleOffset) {
                char const type = view.particleTypes.empty() ? view.uniformParticleType :
                                                               view.particleTypes[particleOffset];
                importedParticles.emplace(particleKey(view.owner, index),
                    ImportedSourceParticle{view.particles[particleOffset].position, type});
            }
        }
        importedNodes.emplace(record.id, ImportedSourceNode{record.cell, std::move(record.node)});
    }
    receivedOwners[view.owner] = true;
}
HPX_PLAIN_ACTION(installExport, InstallSourceExportAction);

void exchangeOwner(Vector<SourceRootDescriptor> roots) {
    FmmMeasurementScope ownerTimer(FmmMeasurementPhase::OwnerExchange);
    uint64_t inFlightBytes = 0;
    auto waitStart = std::chrono::steady_clock::time_point{};
    int const owner = hpxRank();
    auto const sources = fmmSourceView();
    if (sources.masses != nullptr) {
        throw std::logic_error("Geometric source exchange requires configured species masses");
    }
    auto transfers = detail::geometricLaunchAndDrain<void>(roots.size() - 1, [&](size_t index) {
        auto const& destination = roots[index < size_t(owner) ? index : index + 1];
        FmmMeasurementScope packTimer(FmmMeasurementPhase::ExportPack);
        SourceExportView view;
        view.owner = owner;
        view.destination = destination.root.proc;
        view.generation = roots[owner].generation;
        view.destinationEwaldDepth = destination.ewaldDepth;
        view.theta = destination.theta;
        view.smoothingLength = destination.smoothingLength;
        view.destinationGpu = destination.gpu;
        view.destinationCover = destination.cover;
        if (destination.sourceCount != 0) {
            exportNode(view, roots[owner].root, GeometricTreeCell{}, destination, sources);
        }
        if (fmmMeasurementEnabled) {
            std::lock_guard<MutexType> lock(measurementMutex);
            auto& peer = measurement.peers[destination.root.proc];
            uint64_t const bytes = fmmSerializedBytes(view);
            peer.sentBytes += bytes;
            ++peer.sentMessages;
            peer.exportNodes += view.nodes.size();
            peer.exportParticles += view.particles.size();
            for (auto const& record : view.nodes) {
                peer.coarseTerminals += record.node.sourceKind == kTreeSourceMultipoleTerminal;
            }
            measurement.peakExportVectorBytes = std::max(measurement.peakExportVectorBytes,
                uint64_t(view.nodes.capacity() * sizeof(SourceExportNode) +
                    view.particles.capacity() * sizeof(SourceExportParticle) +
                    view.particleTypes.capacity()));
            inFlightBytes += bytes;
            // Conservative high-water bound: retain all dispatched payloads until
            // the owner's join, even if some acknowledgements arrive earlier.
            measurement.peakInFlightPayloadBytes =
                std::max(measurement.peakInFlightPayloadBytes, inFlightBytes);
        }
        packTimer.stop();
        auto transfer = hpx::async<InstallSourceExportAction>(
            hpxLocalities()[destination.root.proc], std::move(view));
        if (fmmMeasurementEnabled && index + 1 == roots.size() - 1) {
            waitStart = std::chrono::steady_clock::now();
        }
        return transfer;
    });
    if (fmmMeasurementEnabled && roots.size() > 1) {
        fmmMeasureSeconds(FmmMeasurementPhase::ExposedWait,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count());
    }
    for (auto& transfer : transfers) {
        transfer.get();
    }
}
HPX_PLAIN_ACTION(exchangeOwner, ExchangeSourceOwnerAction);

void finishImports() {
    std::lock_guard<MutexType> lock(importMutex);
    validateOwnedRoot(expectedRoots);
    for (size_t owner = 0; owner < expectedRoots.size(); ++owner) {
        if (int(owner) != hpxRank() && !receivedOwners[owner]) {
            throw std::runtime_error("Incomplete geometric source exchange");
        }
    }
    importsReady = true;
    importsEnabled = true;
}
HPX_PLAIN_ACTION(finishImports, FinishSourceImportsAction);

void treeSourceForestReset() {
    std::lock_guard<MutexType> lock(importMutex);
    importsEnabled = false;
    importsReady = false;
    ownerReady = false;
    importedNodes.clear();
    importedParticles.clear();
    materializedNodes.clear();
    expectedRoots.clear();
    receivedOwners.clear();
    fallbackCount = 0;
}

void treeSourceForestRegisterOwner(SourceRootDescriptor const& owner) {
    std::lock_guard<MutexType> lock(importMutex);
    if (owner.root.proc != hpxRank() || owner.root.index < 0 || owner.generation == 0 ||
        owner.geometryVersion != kGeometricSourceForestVersion) {
        throw std::runtime_error("Invalid geometric owned source descriptor");
    }
    ownedRoot = owner;
    auto const& options = getOptions();
    ownedHydroEnabled = options.hydro.enabled;
    ownedDarkMatterMass = options.initial.darkMatterMass;
    ownedGasMass = options.initial.gasMass;
    ownerReady = true;
    importsReady = false;
    importsEnabled = false;
}

void treeSourceForestValidate(Vector<SourceRootDescriptor> const& roots, float theta,
    float smoothingLength, bool gpu, bool requireImports) {
    std::lock_guard<MutexType> lock(importMutex);
    validateOwnedRoot(roots);
    if (requireImports && (!importsReady || expectedRoots.size() != roots.size())) {
        throw std::runtime_error("Geometric source exchange is not ready");
    }
    for (size_t rank = 0; requireImports && rank < roots.size(); ++rank) {
        if (!sameSourceRoot(roots[rank], expectedRoots[rank])) {
            throw std::runtime_error("Stale geometric source-root checklist");
        }
    }
    if (theta != ownedRoot.theta || smoothingLength != ownedRoot.smoothingLength ||
        gpu != ownedRoot.gpu) {
        throw std::invalid_argument("Geometric kick parameters changed; rebuild the source forest");
    }
    auto const& options = getOptions();
    if (options.hydro.enabled != ownedHydroEnabled ||
        (ownedHydroEnabled &&
            (options.initial.darkMatterMass != ownedDarkMatterMass ||
                options.initial.gasMass != ownedGasMass))) {
        throw std::invalid_argument("Geometric source masses changed; rebuild the source forest");
    }
}

void treeSourceForestUseImports(bool enabled) {
    importsEnabled = enabled;
}

bool treeSourceForestImportsEnabled() {
    return importsEnabled.load(std::memory_order_acquire);
}

TreeNode const* treeSourceForestImportedNode(TreeId id) {
    if (!importsEnabled) {
        return nullptr;
    }
    {
        std::lock_guard<MutexType> lock(importMutex);
        if (materializedNodes.contains(id)) {
            return nullptr;
        }
    }
    auto const found = importedNodes.find(id);
    if (found != importedNodes.end() && fmmMeasurementEnabled) {
        if (std::atomic_ref<unsigned char>(found->second.measuredUsed)
                .exchange(1, std::memory_order_relaxed) == 0) {
            std::lock_guard<MutexType> lock(measurementMutex);
            ++measurement.peers[id.proc].usedNodes;
        }
    }
    return found == importedNodes.end() ? nullptr : &found->second.node;
}

void treeSourceForestMaterialize(TreeId id) {
    std::lock_guard<MutexType> lock(importMutex);
    if (materializedNodes.insert(id).second) {
        fallbackCount.fetch_add(1, std::memory_order_relaxed);
    }
}

bool treeSourceForestReadParticleRange(TreeNode const& source, Position* x, Position* y,
    Position* z, char* types, float* masses, ParticleInteger offset) {
    if (!importsEnabled) {
        return false;
    }
    int const owner = source.processRange.first;
    auto const range = source.particleRange;
    auto const& options = getOptions();
    for (ParticleInteger index = range.first; index < range.second; ++index) {
        if (!importedParticles.contains(particleKey(owner, index))) {
            return false;
        }
    }
    for (ParticleInteger index = range.first; index < range.second; ++index) {
        auto const& particle = importedParticles.at(particleKey(owner, index));
        ParticleInteger const output = offset + index - range.first;
        x[output] = particle.position[kXi];
        y[output] = particle.position[kYi];
        z[output] = particle.position[kZi];
        if (types != nullptr) {
            types[output] = particle.type;
        }
        if (masses != nullptr) {
            masses[output] = !options.hydro.enabled ? 1.0f :
                particle.type == kDarkMatterType    ? float(options.initial.darkMatterMass) :
                                                      float(options.initial.gasMass);
        }
    }
    return true;
}

void treeSourceForestCountFallback() {
    fallbackCount.fetch_add(1, std::memory_order_relaxed);
}

uint64_t treeSourceForestFallbackCount() {
    return fallbackCount.load(std::memory_order_relaxed);
}

void treeExchangeSourceForest(Vector<SourceRootDescriptor> const& roots) {
    if (roots.size() != size_t(hpxSize())) {
        throw std::invalid_argument("Source root descriptors do not cover every locality");
    }
    auto pending = detail::geometricLaunchAndDrain<void>(roots.size(), [&](size_t rank) {
        return hpx::async<BeginSourceImportsAction>(hpxLocalities()[rank], roots);
    });
    for (auto& future : pending) {
        future.get();
    }
    pending = detail::geometricLaunchAndDrain<void>(roots.size(), [&](size_t rank) {
        return hpx::async<ExchangeSourceOwnerAction>(hpxLocalities()[rank], roots);
    });
    for (auto& future : pending) {
        future.get();
    }
    pending = detail::geometricLaunchAndDrain<void>(roots.size(),
        [&](size_t rank) { return hpx::async<FinishSourceImportsAction>(hpxLocalities()[rank]); });
    for (auto& future : pending) {
        future.get();
    }
}
