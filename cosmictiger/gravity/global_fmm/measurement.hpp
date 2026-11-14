/* CosmicTiger - opt-in geometric FMM measurements; BSL-1.0. */

// FMM phase timers and per-locality communication, source-use and memory measurements.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

#include <hpx/serialization/detail/preprocess_container.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/core/hpx.hpp>

// Intervals are elapsed task times, potentially overlapping, never additive CPU
// times. OwnerExchange includes packing and waits; ExposedWait excludes packing.
enum class FmmMeasurementPhase {
    LocalBuild,       // Build the owned tree, describe its source cover and register its root.
    ExportPack,       // Construct and measure destination-specific source payloads before dispatch.
    OwnerExchange,    // Pack, dispatch and wait for all exports from this owner.
    ExposedWait,      // Wait for exports after dispatching the final destination payload.
    Import,           // Wait for the import lock, validate a received payload and install it.
    Traversal,        // Execute the local source-forest kick, including any GPU work.
    GpuStaging,       // Prepare GPU trees and particles before CUDA execution.
    Count             // Number of measured phases; sizes the phase-time array.
};
inline constexpr size_t kFmmMeasurementPhases = size_t(FmmMeasurementPhase::Count);

// Serialized byte counts exclude HPX action/parcel envelopes and network framing.
struct FmmPeerMeasurement {
    uint64_t sentBytes = 0;            // Serialized source-export bytes sent to this peer.
    uint64_t receivedBytes = 0;        // Serialized source-export bytes received from this peer.
    uint64_t sentMessages = 0;         // Source-export payloads sent to this peer.
    uint64_t receivedMessages = 0;     // Source-export payloads received from this peer.
    uint64_t exportNodes = 0;          // Node records exported to this peer.
    uint64_t exportParticles = 0;      // Particle-position records exported to this peer.
    uint64_t coarseTerminals = 0;      // Exported multipole terminals without exported descendants.
    uint64_t importedNodes = 0;        // Node records imported from this peer.
    uint64_t importedParticles = 0;    // Particle-position records imported from this peer.
    uint64_t usedNodes = 0;            // Distinct imported nodes accessed during the measured walk.
    uint64_t coveredSources = 0;       // Source particles represented by imported terminal nodes.
    uint64_t coarseSources = 0;    // Source particles represented by imported multipole terminals.
    uint64_t demandSentBytes = 0;    // Serialized cache-line request/response bytes sent to peer.
    uint64_t demandReceivedBytes = 0;    // Serialized cache-line request/response bytes from peer.
    uint64_t demandSentMessages = 0;     // Measured cache-line requests/responses sent to peer.
    uint64_t demandReceivedMessages = 0;    // Measured cache-line requests/responses from peer.
    bool fineDetail = false;    // Whether this peer's imported payload includes particle positions.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & sentBytes & receivedBytes & sentMessages & receivedMessages;
        archive & exportNodes & exportParticles & coarseTerminals;
        archive & importedNodes & importedParticles & usedNodes & coveredSources & coarseSources;
        archive & demandSentBytes & demandReceivedBytes;
        archive & demandSentMessages & demandReceivedMessages & fineDetail;
    }
};

struct FmmMeasurement {
    int rank = 0;    // Locality whose measurements this record contains.
    std::array<double, kFmmMeasurementPhases> seconds{};    // Accumulated seconds indexed by phase.
    Vector<FmmPeerMeasurement> peers;    // Communication and source-use counters by peer rank.
    uint64_t fallbackRequests = 0;    // Fallback events for missing imports or terminal expansion.
    // Arena bytes include allocation capacity. Container estimates include live
    // value storage and buckets, excluding allocator metadata and HPX parcels.
    uint64_t ownedArenaBytes = 0;          // Allocated node, multipole and populated-slot storage.
    uint64_t importedStorageBytes = 0;     // Estimated imported-node/particle maps, with buckets.
    uint64_t peakExportVectorBytes = 0;    // Largest combined vector capacity of one export.
    uint64_t peakInFlightPayloadBytes = 0;    // Serialized-byte bound before the owner join.
    uint64_t remoteMultipoleBytes = 0;     // Allocated capacity of remote multipole-storage chunks.
    uint64_t demandCacheValueBytes = 0;    // Total fetched cache-line value capacity in bytes.
    uint64_t processPeakRssBytes = 0;      // Process lifetime peak resident memory, in bytes.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & rank & seconds & peers & fallbackRequests & ownedArenaBytes;
        archive & importedStorageBytes & peakExportVectorBytes & peakInFlightPayloadBytes;
        archive & remoteMultipoleBytes & demandCacheValueBytes & processPeakRssBytes;
    }
};

extern std::atomic<bool> fmmMeasurementEnabled;
void fmmMeasureSeconds(FmmMeasurementPhase phase, double seconds);
void fmmMeasureDemand(int peer, uint64_t requestBytes, uint64_t responseBytes, uint64_t cacheBytes);
// Collective calls, only between joined builds/exchanges/walks.
void fmmMeasurementBegin();
Vector<FmmMeasurement> fmmMeasurementRead();
void fmmMeasurementEnd();
void fmmMeasurementPrepareWalk();

// Size the serialized application object without a second payload allocation.
// Excludes HPX action/parcel envelopes, acknowledgements and network framing.
template <class T>
uint64_t fmmSerializedBytes(T const& value) {
    hpx::serialization::detail::preprocess_container buffer;
    hpx::serialization::output_archive archive(buffer);
    archive << value;
    return archive.bytes_written();
}

class FmmMeasurementScope {
    FmmMeasurementPhase phase_;    // Measurement phase to which this interval contributes.
    std::chrono::steady_clock::time_point start_{};    // Start of this elapsed-time interval.
    bool enabled_;    // Whether stop() still needs to record this interval.

public:
    explicit FmmMeasurementScope(FmmMeasurementPhase phase)
      : phase_(phase)
      , enabled_(fmmMeasurementEnabled.load(std::memory_order_relaxed)) {
        if (enabled_) {
            start_ = std::chrono::steady_clock::now();
        }
    }
    ~FmmMeasurementScope() {
        stop();
    }
    void stop() {
        if (enabled_) {
            fmmMeasureSeconds(phase_,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
            enabled_ = false;
        }
    }
    FmmMeasurementScope(FmmMeasurementScope const&) = delete;
    FmmMeasurementScope& operator=(FmmMeasurementScope const&) = delete;
};
