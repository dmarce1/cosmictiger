/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Host side of the order-templated CUDA kick: device resources, shared-memory
// arena layout selection and the kernel launch. Included only by
// kick_cuda_instances.hpp; the kernel itself is in kick_cuda_kernel.hpp.

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/gravity/kick_cuda_kernel.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#endif

namespace detail {

#if COSMICTIGER_PROFILE
// Four stream boundaries separate launch-packet transfer, the fused gravity
// kernel, and result transfer. The caller collects only after its existing
// stream wait, so profiling does not add a synchronization point.
class KickCudaProfileEvents {
public:
    /// Create optional timing events on the current device, cleaning up partial
    /// allocation on failure.
    explicit KickCudaProfileEvents(bool enabled) {
        if (!enabled) {
            return;
        }
        CUDA_CHECK(cudaGetDevice(&device_));
        try {
            for (auto& event : events_) {
                auto const status = cudaEventCreate(&event);
                if (status != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kick profiling event failed: ") +
                        cudaGetErrorString(status));
                }
            }
        } catch (...) {
            release();
            throw;
        }
    }

    /// Release any profiling events owned by this launch helper.
    ~KickCudaProfileEvents() noexcept {
        release();
    }

    KickCudaProfileEvents(KickCudaProfileEvents const&) = delete;
    KickCudaProfileEvents& operator=(KickCudaProfileEvents const&) = delete;

    /// Enqueue a timing marker at the requested transfer/kernel boundary when profiling
    /// is active.
    void record(int boundary, cudaStream_t stream) {
        if (device_ >= 0) {
            CUDA_CHECK(cudaEventRecord(events_[boundary], stream));
        }
    }

    /// Convert completed event intervals into transfer/kernel work times and restore
    /// the caller's active device.
    void collect() const {
        if (device_ < 0) {
            return;
        }
        int previousDevice;
        CUDA_CHECK(cudaGetDevice(&previousDevice));
        CUDA_CHECK(cudaSetDevice(device_));
        constexpr profile::WorkPhase kPhases[] = {profile::WorkPhase::KickGpuHostToDevice,
            profile::WorkPhase::KickGpuKernel, profile::WorkPhase::KickGpuDeviceToHost};
        for (int boundary = 0; boundary < 3; ++boundary) {
            float milliseconds;
            CUDA_CHECK(
                cudaEventElapsedTime(&milliseconds, events_[boundary], events_[boundary + 1]));
            profile::recordWork(kPhases[boundary], double(milliseconds) * 1.0e-3);
        }
        CUDA_CHECK(cudaSetDevice(previousDevice));
    }

private:
    /// Destroy owned events on their original device and restore the prior device
    /// without throwing.
    void release() noexcept {
        if (device_ < 0) {
            return;
        }
        int previousDevice = device_;
        bool const restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
        CUDA_WARN(cudaSetDevice(device_));
        for (auto& event : events_) {
            if (event != nullptr) {
                CUDA_WARN(cudaEventDestroy(event));
                event = nullptr;
            }
        }
        if (restoreDevice) {
            CUDA_WARN(cudaSetDevice(previousDevice));
        }
        device_ = -1;
    }

    int device_ = -1;
    cudaEvent_t events_[4]{};
};
#endif

/// Allocate device-only launch storage and throw a descriptive exception on allocation
/// failure.
inline char* kickAllocateLaunchStorage(size_t bytes) {
    char* pointer = nullptr;
    auto const status = cudaMalloc(&pointer, bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kick allocation failed: ") + cudaGetErrorString(status));
    }
    return pointer;
}

constexpr int kKickSharedAllocationGranularity = 128;
constexpr int kKickArenaAlignment = 16;
constexpr int kKickDefaultSinkTile = 128;
constexpr int kKickDefaultParticleTile = 512;
constexpr int kKickDefaultMultipoleTile = 32;
constexpr int kKickMaximumMultipoleTile = 64;
constexpr int kKickMinimumParticleTile = kWarpSize;

struct KickDeviceResources {
    int numberOfRegisters;
    int staticSharedBytes;
    int sharedPerMultiprocessor;
    int reservedSharedPerBlock;
    int maximumSharedPerBlock;
    int registersPerMultiprocessor;
    int maximumBlocksPerMultiprocessor;
    int maximumWarpsPerMultiprocessor;
};

/// Query kernel register use and device shared-memory/block limits for the selected FMM
/// order.
template <int order>
KickDeviceResources kickDeviceResources() {
    KickDeviceResources resources;
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) cudaKickKernel<order>));
    resources.numberOfRegisters = attr.numRegs;
    resources.staticSharedBytes = int(attr.sharedSizeBytes);
    int const device = cudaGetDevice();
    int threads;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.sharedPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.reservedSharedPerBlock, cudaDevAttrReservedSharedMemoryPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.maximumSharedPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.registersPerMultiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &resources.maximumBlocksPerMultiprocessor, cudaDevAttrMaxBlocksPerMultiprocessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    resources.maximumWarpsPerMultiprocessor = threads / kWarpSize;
    return resources;
}

/// Round an integer byte/count value upward to a multiple.
inline int kickRoundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/// Round an integer byte/count value downward to a multiple.
inline int kickRoundDown(int value, int multiple) {
    return (value / multiple) * multiple;
}

// Warps allocate registers in units of 256, with per-thread counts rounded to 8.
inline int kickRegisterResidencyLimit(KickDeviceResources const& resources) {
    int const registersPerWarp =
        kickRoundUp(kickRoundUp(resources.numberOfRegisters, 8) * kWarpSize, 256);
    int limit = 4 * ((resources.registersPerMultiprocessor / 4) / std::max(1, registersPerWarp));
    limit = std::min(limit, resources.maximumBlocksPerMultiprocessor);
    limit = std::min(limit, resources.maximumWarpsPerMultiprocessor);
    return std::max(1, limit);
}

// Dynamic shared bytes one block may use when `residency` blocks share a
// multiprocessor, after the driver reservation, static shared memory and the
// allocation granularity.
inline int kickSharedBudget(KickDeviceResources const& resources, int residency) {
    int const perBlock = resources.sharedPerMultiprocessor / residency;
    int budget = kickRoundDown(perBlock, kKickSharedAllocationGranularity);
    budget -= resources.reservedSharedPerBlock;
    budget -= kickRoundUp(resources.staticSharedBytes, kKickArenaAlignment);
    budget = kickRoundDown(budget, kKickSharedAllocationGranularity);
    return std::min(budget, resources.maximumSharedPerBlock - resources.staticSharedBytes);
}

// Fill the arena for a budget; returns false when the minimum tiles do not fit.
template <int order>
bool kickBuildLayout(Gravity const& gravity, bool massPlane, int residency, int budget,
    CudaKickArenaLayout& layout) {
    layout = CudaKickArenaLayout{};
    layout.order = order;
    layout.residency = residency;
    layout.listReserve = gravity.cudaKickListReserve;
    layout.budgetBytes = budget;
    layout.massPlane = massPlane;
    int const controlBytes = kickRoundUp(int(sizeof(CudaKickControl<order>)), kKickArenaAlignment);
    layout.controlOffset = 0;
    int offset = controlBytes;
    if (gravity.cudaKickSharedExpansion) {
        layout.expansionOffset = offset;
        offset += kickRoundUp(
            kickRoundUp(FmmOrder<order>::kLocalExpansionBytes_, int(alignof(Position))) +
                kDimensions * int(sizeof(Position)),
            kKickArenaAlignment);
    } else {
        layout.expansionOffset = -1;
    }
    layout.arenaOffset = offset;
    int const scratchBytes =
        kickRoundUp(FmmOrder<order>::kLocalExpansionBytes_, kKickArenaAlignment);
    int const arenaBudget = budget - offset;
    if (arenaBudget < scratchBytes) {
        return false;
    }
    // The runtime bucket bounds a leaf; the shared budget bounds a sink tile.
    // Clamp before rounding or computing bytes so a large bucket cannot
    // overflow layout arithmetic. Source staging can reduce the tile further.
    int const bucket = getOptions().compute.bucketSize;
    int const maximumSinkCapacity =
        kickRoundDown(arenaBudget / (4 * int(sizeof(float))), kWarpSize);
    bool const sinkAuto = gravity.cudaKickSinkTile == 0;
    int sinkCapacity = 0;
    if (gravity.cudaKickSinkTile > 0) {
        if (gravity.cudaKickSinkTile > maximumSinkCapacity) {
            return false;
        }
        sinkCapacity = kickRoundUp(gravity.cudaKickSinkTile, kWarpSize);
    } else if (sinkAuto) {
        if (maximumSinkCapacity == 0) {
            return false;
        }
        sinkCapacity = kickRoundUp(std::min(bucket, maximumSinkCapacity), kWarpSize);
    }
    int const particleBytesPerSource =
        int(sizeof(Position)) * kDimensions + (massPlane ? int(sizeof(float)) : 0);
    // Staged records keep the tree arena's stride, so cudaGravityPcDirect
    // stages each record with one aligned asynchronous copy.
    static_assert(FmmOrder<order>::kPositionedMultipoleStride_ % kKickArenaAlignment == 0,
        "Staged multipole records must keep the arena alignment");
    layout.multipoleStride = FmmOrder<order>::kPositionedMultipoleStride_;
    int const multipoleTarget = kKickDefaultMultipoleTile;
    bool const particleAuto = gravity.cudaKickParticleTile <= 0;
    bool const multipoleAuto = gravity.cudaKickMultipoleTile <= 0;
    int particleCapacity = particleAuto ? kKickDefaultParticleTile :
                                          kickRoundUp(gravity.cudaKickParticleTile, kWarpSize);
    int multipoleCapacity = multipoleAuto ? multipoleTarget : gravity.cudaKickMultipoleTile;
    while (true) {
        int const forceBytes = 4 * int(sizeof(float)) * sinkCapacity;
        int const particleBytes =
            kickRoundUp(particleBytesPerSource * particleCapacity, kKickArenaAlignment);
        int const multipoleBytes = multipoleCapacity * layout.multipoleStride;
        int const sourceBytes = std::max(particleBytes, multipoleBytes);
        int const arenaBytes = std::max(forceBytes + sourceBytes, scratchBytes);
        if (arenaBytes <= arenaBudget) {
            if (multipoleAuto) {
                // The staged multipoles share the particle planes; fill them.
                multipoleCapacity = std::min(kKickMaximumMultipoleTile,
                    std::max(multipoleCapacity, sourceBytes / layout.multipoleStride));
            }
            layout.sinkCapacity = sinkCapacity;
            layout.particleCapacity = particleCapacity;
            layout.multipoleCapacity = multipoleCapacity;
            layout.forceOffset = sinkCapacity > 0 ? 0 : -1;
            layout.sourceOffset = forceBytes;
            layout.scratchOffset = 0;
            layout.arenaBytes = kickRoundUp(arenaBytes, kKickArenaAlignment);
            layout.totalBytes = layout.arenaOffset + layout.arenaBytes;
            // Global list storage: one slab region per block at the reserved
            // capacities (whole-bucket force backing only without shared forces).
            layout.forceBackingCapacity = sinkCapacity > 0 ? 0 : 4 * bucket;
            layout.listSlabBytes = 0;
            if (layout.listReserve > 0) {
                CudaKickListStorage<order> storage;
                size_t const bytes = kickCarveListStorage<order>(layout, nullptr, storage);
                if (bytes > size_t(std::numeric_limits<int>::max())) {
                    throw std::length_error(
                        "CUDA kick per-block list storage exceeds its byte-offset range");
                }
                layout.listSlabBytes = int(bytes);
            }
            return layout.totalBytes <= budget;
        }
        // Trim: particles first, then multipoles, then an automatic sink tile.
        if (particleAuto && particleCapacity > kKickMinimumParticleTile &&
            particleBytes >= multipoleBytes) {
            particleCapacity -= kWarpSize;
        } else if (multipoleAuto && multipoleCapacity > 1 && multipoleBytes > particleBytes) {
            multipoleCapacity--;
        } else if (particleAuto && particleCapacity > kKickMinimumParticleTile) {
            particleCapacity -= kWarpSize;
        } else if (multipoleAuto && multipoleCapacity > 1) {
            multipoleCapacity--;
        } else if (sinkAuto && sinkCapacity > kKickDefaultSinkTile) {
            // A smaller sink tile frees force bytes; restart the source tiles
            // from their defaults instead of keeping them at the trimmed minimum.
            sinkCapacity = kKickDefaultSinkTile;
            particleCapacity = particleAuto ? kKickDefaultParticleTile : particleCapacity;
            multipoleCapacity = multipoleAuto ? multipoleTarget : multipoleCapacity;
        } else if (sinkAuto && sinkCapacity > kWarpSize) {
            sinkCapacity -= kWarpSize;
            particleCapacity = particleAuto ? kKickDefaultParticleTile : particleCapacity;
            multipoleCapacity = multipoleAuto ? multipoleTarget : multipoleCapacity;
        } else {
            return false;
        }
    }
}

/// Find a shared-memory tile layout that satisfies configured residency and CUDA's actual
/// occupancy limits.
template <int order>
CudaKickArenaLayout kickSelectLayout() {
    auto const& gravity = getOptions().gravity;
    bool const massPlane = getOptions().hydro.enabled;
    auto const resources = kickDeviceResources<order>();
    int const registerLimit = kickRegisterResidencyLimit(resources);
    int residency = gravity.cudaKickResidency > 0 ?
        std::min(gravity.cudaKickResidency, registerLimit) :
        registerLimit;
    CudaKickArenaLayout layout{};
    while (residency >= 1) {
        int budget = kickSharedBudget(resources, residency);
        while (
            budget > 0 && kickBuildLayout<order>(gravity, massPlane, residency, budget, layout)) {
            if (layout.totalBytes > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute((void const*) cudaKickKernel<order>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, layout.totalBytes));
            }
            int achieved = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &achieved, (void const*) cudaKickKernel<order>, kWarpSize, layout.totalBytes));
            if (achieved >= residency) {
                // The launch uses residency blocks per multiprocessor (and the
                // CPU/GPU handoff divides by that count), so a requested
                // residency below the achievable occupancy must be kept, not
                // replaced by what the calculator says would fit.
                layout.residency = std::min(residency, registerLimit);
                terminal::detailPrint(
                    "CUDA kick arena (FMM order %i): %i registers, %i static shared bytes, %i "
                    "blocks/SM "
                    "(register limit %i), budget %i, dynamic shared %i bytes: control %i, "
                    "expansion %s, forces %s, particle tile %i (%s mass plane), multipole tile "
                    "%i x %i bytes; list slab %i bytes per block (%i entries per list)\n",
                    order, resources.numberOfRegisters, resources.staticSharedBytes,
                    layout.residency, registerLimit, layout.budgetBytes, layout.totalBytes,
                    int(sizeof(CudaKickControl<order>)),
                    layout.expansionOffset >= 0 ? "shared" : "global",
                    layout.sinkCapacity > 0 ?
                        (std::string("shared tile ") + std::to_string(layout.sinkCapacity))
                            .c_str() :
                        "global whole bucket",
                    layout.particleCapacity, layout.massPlane ? "with" : "without",
                    layout.multipoleCapacity, layout.multipoleStride, layout.listSlabBytes,
                    layout.listReserve);
                return layout;
            }
            // The occupancy calculator disagrees with the arithmetic budget; shrink.
            budget -= kKickSharedAllocationGranularity;
        }
        if (gravity.cudaKickResidency > 0) {
            break;
        }
        residency--;
    }
    throw std::runtime_error(
        "No CUDA kick shared-memory layout fits the requested residency and tiles");
}

}    // namespace detail

/// Lazily cache the selected CUDA arena layout for this FMM order.
template <int order>
CudaKickArenaLayout const& kickArenaLayoutImpl() {
    static CudaKickArenaLayout const layout = detail::kickSelectLayout<order>();
    return layout;
}

/// Prepare reusable launch buffers, execute a nonempty CUDA kick batch, await transfers,
/// and return reduced results.
template <int order>
KickReturn cudaExecuteKicksImpl(KickParameters kparams, Position* devX, Position* devY,
    Position* devZ, TreeNode* devTreeNodes, Vector<KickWorkitem> workitems,
    detail::FmmForceOutput output, float const* sourceMasses) {
    using Parameters = CudaKickParameters<order>;
#if COSMICTIGER_PROFILE
    bool const profileEnabled = hpxRank() == 0;
#else
    bool const profileEnabled = false;
#endif
    profile::ScopedWork launchPhase(profile::WorkPhase::KickGpuPrepareLaunch, profileEnabled);
    auto const& layout = kickArenaLayoutImpl<order>();
    int numberOfBlocks = layout.residency * cudaSmpCount();
    ALWAYS_ASSERT(workitems.size());
    numberOfBlocks = std::min(numberOfBlocks, (int) workitems.size());
    static char* dataPtr = nullptr;
    static char* devDataPtr = nullptr;
    static size_t dataSize = 0;
    // Per-launch list slab: one region per launched block, kept across launches
    // (the layout is fixed per order and the launch never exceeds the residency).
    static char* devListSlab = nullptr;
    static size_t listSlabSize = 0;
    int dcount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dcount += workitems[i].dchecklist.size() + workitems[i].echecklist.size();
    }
    size_t const packetSize = sizeof(int) + sizeof(Parameters) * workitems.size() +
        sizeof(KickReturn) + sizeof(int) * dcount;
    size_t allocSize = packetSize;
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    // Reuse the launch-owned packet storage. The launch ticket keeps it alive
    // and exclusive through completion, without per-launch profiling allocations.
    constexpr size_t kProfileAlignment = alignof(profile::DeviceBlockRecord);
    size_t const profileOffset =
        (packetSize + kProfileAlignment - 1) / kProfileAlignment * kProfileAlignment;
    size_t const profileBytes =
        profileEnabled ? sizeof(profile::DeviceBlockRecord) * size_t(numberOfBlocks) : 0;
    if (profileEnabled) {
        allocSize = profileOffset + profileBytes;
    }
#endif
    if (dataSize < allocSize) {
        std::unique_ptr<char, decltype(&std::free)> replacement(
            static_cast<char*>(std::malloc(allocSize)), &std::free);
        if (!replacement) {
            throw std::bad_alloc();
        }
        // Keep the previous launch packet valid until both new allocations
        // succeed. A failed device allocation frees only the temporary host buffer.
        char* const deviceReplacement = detail::kickAllocateLaunchStorage(allocSize);
        if (devDataPtr) {
            CUDA_CHECK(cudaFree(devDataPtr));
        }
        std::free(dataPtr);
        dataPtr = replacement.release();
        devDataPtr = deviceReplacement;
        dataSize = allocSize;
    }
    size_t const slabSize =
        size_t(layout.listSlabBytes) * size_t(layout.residency) * size_t(cudaSmpCount());
    if (listSlabSize < slabSize) {
        char* const replacement = detail::kickAllocateLaunchStorage(slabSize);
        if (devListSlab) {
            CUDA_CHECK(cudaFree(devListSlab));
        }
        devListSlab = replacement;
        listSlabSize = slabSize;
    }
    size_t offset = 0;
    KickReturn* result = (KickReturn*) (dataPtr + offset);
    offset += sizeof(KickReturn);
    Parameters* ikickParams = (Parameters*) (dataPtr + offset);
    offset += sizeof(Parameters) * workitems.size();
    int* dchecks = (int*) (dataPtr + offset);
    offset += sizeof(int) * dcount;
    int* currentIndex = (int*) (dataPtr + offset);
    offset += sizeof(int);
    offset = 0;
    KickReturn* devReturnArg = (KickReturn*) (devDataPtr + offset);
    offset += sizeof(KickReturn);
    Parameters* devIkickParams = (Parameters*) (devDataPtr + offset);
    offset += sizeof(Parameters) * workitems.size();
    int* devDchecks = (int*) (devDataPtr + offset);
    offset += sizeof(int) * dcount;
    int* devCurrentIndex = (int*) (devDataPtr + offset);
    offset += sizeof(int);
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    auto* const profileRecords = profileEnabled ?
        reinterpret_cast<profile::DeviceBlockRecord*>(dataPtr + profileOffset) :
        nullptr;
    auto* const devProfileRecords = profileEnabled ?
        reinterpret_cast<profile::DeviceBlockRecord*>(devDataPtr + profileOffset) :
        nullptr;
#endif
    *currentIndex = 0;
    *result = KickReturn();
    Vector<int> dindices(workitems.size() + 1);
    Vector<int> eindices(workitems.size());
    dcount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dindices[i] = dcount;
        for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
            dchecks[dcount] = workitems[i].dchecklist[j].index;
            dcount++;
        }
        eindices[i] = dcount;
        for (auto const id : workitems[i].echecklist) {
            dchecks[dcount++] = id.index;
        }
    }

    dindices[workitems.size()] = dcount;
    CudaKickData data;
#if COSMICTIGER_WITH_SPH
    data.midpointHydro = !kparams.forceOnly && sphMidpointEnabled();
#endif
    data.deferredGasOpening = kparams.deferHydroOpening ? gravityDeferredGasOpeningData() : nullptr;
    for (int dim = 0; dim < kDimensions; ++dim) {
        data.globalAcceleration[dim] = (kparams.globalTimestep || kparams.deferGasOpening) ?
            gravityGlobalAcceleration(dim) :
            nullptr;
    }
    data.x = devX;
    data.y = devY;
    data.z = devZ;
    data.treeNodes = devTreeNodes;
    data.vel = kparams.forceOnly ? nullptr : particlesVelData();
    data.rungs = kparams.forceOnly ? nullptr : &particlesRung(0);
    data.gravitySampleIndices = kparams.forceOnly ? nullptr : particlesGravitySampleIndices;
    data.gravitySampleAccelerations =
        kparams.forceOnly ? nullptr : particlesGravitySampleAccelerations;
    data.sph = COSMICTIGER_WITH_SPH && getOptions().hydro.enabled;
    data.darkMatterMass = getOptions().initial.darkMatterMass;
    data.sphMass = getOptions().initial.gasMass;
    data.sourceMasses = sourceMasses;
    data.type = data.sph ? &particlesType(0) : nullptr;
#if COSMICTIGER_WITH_SPH
    bool const gasKick = data.sph && !kparams.forceOnly;
    data.catIndex = gasKick ? &particlesCatIndex(0) : nullptr;
    data.sphHydroFlags = gasKick && data.midpointHydro ? sphParticlesFlags : nullptr;
#else
    data.catIndex = nullptr;
#endif
    if (kparams.forceOnly) {
        ALWAYS_ASSERT(output.potential != nullptr && output.acceleration[kXi] != nullptr);
        data.gx = output.acceleration[kXi];
        data.gy = output.acceleration[kYi];
        data.gz = output.acceleration[kZi];
        data.pot = output.potential;
    } else {
        bool const saveAcceleration =
            kparams.saveForce || saveParticleAcceleration(getOptions().analysis);
        bool const savePotential =
            kparams.saveForce || saveParticlePotential(getOptions().analysis);
        data.gx = saveAcceleration ? &particlesGravityAcceleration(kXi, 0) : nullptr;
        data.gy = saveAcceleration ? &particlesGravityAcceleration(kYi, 0) : nullptr;
        data.gz = saveAcceleration ? &particlesGravityAcceleration(kZi, 0) : nullptr;
        data.pot = savePotential && kparams.doPhi ? &particlesPot(0) : nullptr;
    }
    ALWAYS_ASSERT(layout.massPlane == data.sph);

    for (int i = 0; i < workitems.size(); i++) {
        Parameters params;
        params.outputOffset = workitems[i].outputOffset;
        params.lpos = workitems[i].pos;
        params.lValue = workitems[i].l.expansion();
        params.self = workitems[i].self.index;
        params.dchecks = devDchecks + dindices[i];
        params.dcount = eindices[i] - dindices[i];
        params.echecks = devDchecks + eindices[i];
        params.ecount = dindices[i + 1] - eindices[i];
        ikickParams[i] = std::move(params);
    }
    cudaSetDevice();
#if COSMICTIGER_PROFILE
    // Keep events alive until the stream guard has drained on exception paths.
    detail::KickCudaProfileEvents deviceEvents(profileEnabled);
#endif
    CudaStream stream(cudaGetDevice());
    launchPhase.next(profile::WorkPhase::KickGpuSubmitWait);
#if COSMICTIGER_PROFILE
    auto const streamHandle = profileEnabled ? stream.nativeHandle() : nullptr;
    deviceEvents.record(0, streamHandle);
#endif
    // Device counters are fully overwritten by each block, so copy only inputs.
    stream.copyAsync(devDataPtr, dataPtr, packetSize, cudaMemcpyHostToDevice);
#if COSMICTIGER_PROFILE
    deviceEvents.record(1, streamHandle);
#endif
    cudaKickKernel<order><<<numberOfBlocks, kWarpSize, layout.totalBytes, stream.nativeHandle()>>>(
        devReturnArg, kparams, data, devIkickParams, workitems.size(), devCurrentIndex, layout,
        devListSlab
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
        ,
        devProfileRecords
#endif
    );
    stream.checkLaunch();
#if COSMICTIGER_PROFILE
    deviceEvents.record(2, streamHandle);
#endif
    stream.copyAsync(result, devReturnArg, sizeof(KickReturn), cudaMemcpyDeviceToHost);
#if COSMICTIGER_PROFILE
    deviceEvents.record(3, streamHandle);
#endif
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    if (profileEnabled) {
        stream.copyAsync(profileRecords, devProfileRecords, profileBytes, cudaMemcpyDeviceToHost);
    }
#endif
    stream.wait();
    launchPhase.stop();
#if COSMICTIGER_PROFILE
    deviceEvents.collect();
#endif
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    if (profileEnabled) {
        profile::recordDeviceBlocks(profileRecords, size_t(numberOfBlocks));
    }
#endif
    return *result;
}
