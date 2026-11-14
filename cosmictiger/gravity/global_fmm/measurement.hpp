/* CosmicTiger - opt-in geometric FMM measurements; BSL-1.0. */
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
    LocalBuild,
    ExportPack,
    OwnerExchange,
    ExposedWait,
    Import,
    Traversal,
    GpuStaging,
    Count
};
inline constexpr size_t kFmmMeasurementPhases = size_t(FmmMeasurementPhase::Count);

struct FmmPeerMeasurement {
    uint64_t sentBytes = 0;
    uint64_t receivedBytes = 0;
    uint64_t sentMessages = 0;
    uint64_t receivedMessages = 0;
    uint64_t exportNodes = 0;
    uint64_t exportParticles = 0;
    uint64_t coarseTerminals = 0;
    uint64_t importedNodes = 0;
    uint64_t importedParticles = 0;
    uint64_t usedNodes = 0;
    uint64_t coveredSources = 0;
    uint64_t coarseSources = 0;
    uint64_t demandSentBytes = 0;
    uint64_t demandReceivedBytes = 0;
    uint64_t demandSentMessages = 0;
    uint64_t demandReceivedMessages = 0;
    bool fineDetail = false;
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
    int rank = 0;
    std::array<double, kFmmMeasurementPhases> seconds{};
    Vector<FmmPeerMeasurement> peers;
    uint64_t fallbackRequests = 0;
    // Arena bytes include allocation capacity. Container estimates include live
    // value storage and buckets, excluding allocator metadata and HPX parcels.
    uint64_t ownedArenaBytes = 0;
    uint64_t importedStorageBytes = 0;
    uint64_t peakExportVectorBytes = 0;
    uint64_t peakInFlightPayloadBytes = 0;
    uint64_t remoteMultipoleBytes = 0;
    uint64_t demandCacheValueBytes = 0;
    uint64_t processPeakRssBytes = 0;
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
    FmmMeasurementPhase phase_;
    std::chrono::steady_clock::time_point start_{};
    bool enabled_;

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
