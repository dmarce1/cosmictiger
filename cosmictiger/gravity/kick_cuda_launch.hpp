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
#include <stdexcept>
#include <string>

#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/gravity/kick_cuda_kernel.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

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

inline int kickRoundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

inline int kickRoundDown(int value, int multiple) {
    return (value / multiple) * multiple;
}

// Warps allocate registers in units of 256, with per-thread counts rounded to 8.
inline int kickRegisterResidencyLimit(KickDeviceResources const& resources) {
    int const registersPerWarp =
        kickRoundUp(kickRoundUp(resources.numberOfRegisters, 8) * kWarpSize, 256);
    int limit = resources.registersPerMultiprocessor / std::max(1, registersPerWarp);
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
bool kickBuildLayout(Compute const& compute, bool massPlane, int residency, int budget,
    CudaKickArenaLayout& layout) {
    layout = CudaKickArenaLayout{};
    layout.order = order;
    layout.residency = residency;
    layout.listReserve = compute.cudaKickListReserve;
    layout.budgetBytes = budget;
    layout.massPlane = massPlane;
    int const controlBytes = kickRoundUp(int(sizeof(CudaKickControl<order>)), kKickArenaAlignment);
    layout.controlOffset = 0;
    int offset = controlBytes;
    if (compute.cudaKickSharedExpansion) {
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
    // Leaves never exceed the runtime bucket, which is capped by kMaxBucketSize,
    // so an automatic tile first tries to hold the whole bucket and only then
    // falls back to bounded sink tiles.
    int const bucket = std::min(compute.bucketSize, kMaxBucketSize);
    bool const sinkAuto = compute.cudaKickSinkTile == 0;
    int sinkCapacity = 0;
    if (compute.cudaKickSinkTile > 0) {
        sinkCapacity = kickRoundUp(compute.cudaKickSinkTile, kWarpSize);
    } else if (sinkAuto) {
        sinkCapacity = kickRoundUp(bucket, kWarpSize);
    }
    int const particleBytesPerSource =
        int(sizeof(Position)) * kDimensions + (massPlane ? int(sizeof(float)) : 0);
    // Staged records keep the tree arena's stride, so cudaGravityPcDirect
    // stages each record with one aligned asynchronous copy.
    static_assert(FmmOrder<order>::kPositionedMultipoleStride_ % kKickArenaAlignment == 0,
        "Staged multipole records must keep the arena alignment");
    layout.multipoleStride = FmmOrder<order>::kPositionedMultipoleStride_;
    int const multipoleTarget = kKickDefaultMultipoleTile;
    bool const particleAuto = compute.cudaKickParticleTile <= 0;
    bool const multipoleAuto = compute.cudaKickMultipoleTile <= 0;
    int particleCapacity = particleAuto ? kKickDefaultParticleTile :
                                          kickRoundUp(compute.cudaKickParticleTile, kWarpSize);
    int multipoleCapacity = multipoleAuto ? multipoleTarget : compute.cudaKickMultipoleTile;
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
                layout.listSlabBytes = kickCarveListStorage<order>(layout, nullptr, storage);
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

template <int order>
CudaKickArenaLayout kickSelectLayout() {
    auto const& compute = getOptions().compute;
    bool const massPlane = getOptions().hydro.enabled;
    auto const resources = kickDeviceResources<order>();
    int const registerLimit = kickRegisterResidencyLimit(resources);
    int residency = compute.cudaKickResidency > 0 ?
        std::min(compute.cudaKickResidency, registerLimit) :
        registerLimit;
    CudaKickArenaLayout layout{};
    while (residency >= 1) {
        int budget = kickSharedBudget(resources, residency);
        while (
            budget > 0 && kickBuildLayout<order>(compute, massPlane, residency, budget, layout)) {
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
        if (compute.cudaKickResidency > 0) {
            break;
        }
        residency--;
    }
    throw std::runtime_error(
        "No CUDA kick shared-memory layout fits the requested residency and tiles");
}

}    // namespace detail

template <int order>
CudaKickArenaLayout const& kickArenaLayoutImpl() {
    static CudaKickArenaLayout const layout = detail::kickSelectLayout<order>();
    return layout;
}

template <int order>
KickReturn cudaExecuteKicksImpl(KickParameters kparams, Position* devX, Position* devY,
    Position* devZ, TreeNode* devTreeNodes, Vector<KickWorkitem> workitems,
    detail::FmmForceOutput output) {
    using Parameters = CudaKickParameters<order>;
    Timer tm;
    tm.start();
    auto const& layout = kickArenaLayoutImpl<order>();
    int numberOfBlocks = layout.residency * cudaSmpCount();
    ALWAYS_ASSERT(workitems.size());
    numberOfBlocks = std::min(numberOfBlocks, (int) workitems.size());
    static char* dataPtr = nullptr;
    static char* devDataPtr = nullptr;
    static int dataSize = 0;
    // Per-launch list slab: one region per launched block, kept across launches
    // (the layout is fixed per order and the launch never exceeds the residency).
    static char* devListSlab = nullptr;
    static size_t listSlabSize = 0;
    int dcount = 0;
    int ecount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dcount += workitems[i].dchecklist.size();
        ecount += workitems[i].echecklist.size();
    }
    int const allocSize = sizeof(int) + sizeof(Parameters) * workitems.size() + sizeof(KickReturn) +
        sizeof(int) * dcount + sizeof(int) * ecount;
    if (dataSize < allocSize) {
        if (dataPtr) {
            CUDA_CHECK(cudaFree(devDataPtr));
            free(dataPtr);
        }
        dataSize = allocSize;
        dataPtr = (char*) malloc(allocSize);
        CUDA_CHECK(cudaMalloc(&devDataPtr, allocSize));
    }
    size_t const slabSize =
        size_t(layout.listSlabBytes) * size_t(layout.residency) * size_t(cudaSmpCount());
    if (listSlabSize < slabSize) {
        if (devListSlab) {
            CUDA_CHECK(cudaFree(devListSlab));
        }
        CUDA_CHECK(cudaMalloc(&devListSlab, slabSize));
        listSlabSize = slabSize;
    }
    int offset = 0;
    KickReturn* result = (KickReturn*) (dataPtr + offset);
    offset += sizeof(KickReturn);
    Parameters* ikickParams = (Parameters*) (dataPtr + offset);
    offset += sizeof(Parameters) * workitems.size();
    int* dchecks = (int*) (dataPtr + offset);
    offset += sizeof(int) * dcount;
    int* echecks = (int*) (dataPtr + offset);
    offset += sizeof(int) * ecount;
    int* currentIndex = (int*) (dataPtr + offset);
    offset += sizeof(int);
    offset = 0;
    KickReturn* devReturnArg = (KickReturn*) (devDataPtr + offset);
    offset += sizeof(KickReturn);
    Parameters* devIkickParams = (Parameters*) (devDataPtr + offset);
    offset += sizeof(Parameters) * workitems.size();
    int* devDchecks = (int*) (devDataPtr + offset);
    offset += sizeof(int) * dcount;
    int* devEchecks = (int*) (devDataPtr + offset);
    offset += sizeof(int) * ecount;
    int* devCurrentIndex = (int*) (devDataPtr + offset);
    offset += sizeof(int);
    *currentIndex = 0;
    *result = KickReturn();
    Vector<int> dindices(workitems.size() + 1);
    Vector<int> eindices(workitems.size() + 1);
    dcount = 0;
    ecount = 0;
    for (int i = 0; i < workitems.size(); i++) {
        dindices[i] = dcount;
        eindices[i] = ecount;
        for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
            dchecks[dcount] = workitems[i].dchecklist[j].index;
            dcount++;
        }
        for (int j = 0; j < workitems[i].echecklist.size(); j++) {
            echecks[ecount] = workitems[i].echecklist[j].index;
            ecount++;
        }
    }

    dindices[workitems.size()] = dcount;
    eindices[workitems.size()] = ecount;
    CudaKickData data;
    data.x = devX;
    data.y = devY;
    data.z = devZ;
    data.ghostSources = fmmSourceView(FmmParticleStore::Ghost);
    data.treeNodes = devTreeNodes;
    data.vel = kparams.forceOnly ? nullptr : particlesVelData();
    data.rungs = kparams.forceOnly ? nullptr : &particlesRung(0);
    data.sph = getOptions().hydro.enabled;
    data.darkMatterMass = getOptions().initial.darkMatterMass;
    data.sphMass = getOptions().initial.gasMass;
    data.type = data.sph ? &particlesType(0) : nullptr;
    bool const gasKick = data.sph && !kparams.forceOnly;
    data.catIndex = gasKick ? &particlesCatIndex(0) : nullptr;
    data.sphGravityAccelerationX =
        gasKick ? &sphParticlesGravityAcceleration(kXDimension, 0) : nullptr;
    data.sphGravityAccelerationY =
        gasKick ? &sphParticlesGravityAcceleration(kYDimension, 0) : nullptr;
    data.sphGravityAccelerationZ =
        gasKick ? &sphParticlesGravityAcceleration(kZDimension, 0) : nullptr;
    if (kparams.forceOnly) {
        ALWAYS_ASSERT(output.potential != nullptr && output.acceleration[kXDimension] != nullptr);
        data.gx = output.acceleration[kXDimension];
        data.gy = output.acceleration[kYDimension];
        data.gz = output.acceleration[kZDimension];
        data.pot = output.potential;
    } else {
        bool const saveAcceleration =
            kparams.saveForce || saveParticleAcceleration(getOptions().analysis);
        bool const savePotential =
            kparams.saveForce || saveParticlePotential(getOptions().analysis);
        data.gx = saveAcceleration ? &particlesGravityAcceleration(kXDimension, 0) : nullptr;
        data.gy = saveAcceleration ? &particlesGravityAcceleration(kYDimension, 0) : nullptr;
        data.gz = saveAcceleration ? &particlesGravityAcceleration(kZDimension, 0) : nullptr;
        data.pot = savePotential && kparams.doPhi ? &particlesPot(0) : nullptr;
    }
    ALWAYS_ASSERT(layout.massPlane == data.sph);

    for (int i = 0; i < workitems.size(); i++) {
        Parameters params;
        params.geometry = workitems[i].geometry;
        params.outputOffset = workitems[i].outputOffset;
        params.lpos = workitems[i].pos;
        params.lValue = workitems[i].l.template expansion<order>();
        params.self = workitems[i].self.index;
        params.dchecks = devDchecks + dindices[i];
        params.echecks = devEchecks + eindices[i];
        params.dcount = dindices[i + 1] - dindices[i];
        params.ecount = eindices[i + 1] - eindices[i];
        ikickParams[i] = std::move(params);
    }
    cudaSetDevice();
    CudaStream stream(cudaGetDevice());
    stream.copyAsync(devDataPtr, dataPtr, sizeof(char) * allocSize, cudaMemcpyHostToDevice);
    cudaKickKernel<order>
        <<<numberOfBlocks, kWarpSize, layout.totalBytes, stream.nativeHandle()>>>(devReturnArg,
            kparams, data, devIkickParams, workitems.size(), devCurrentIndex, layout, devListSlab);
    stream.checkLaunch();
    stream.copyAsync(result, devReturnArg, sizeof(KickReturn), cudaMemcpyDeviceToHost);
    stream.wait();
    return *result;
}
