/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// CUDA kick: order-templated declarations. The kernel, its device operators and
// the host launch/layout code are templates defined in kick_cuda_kernel.hpp and
// kick_cuda_launch.hpp and
// instantiated once per FMM order by the CMake-generated instance translation
// units (kick_cuda_instances.hpp). kick.cu dispatches on the runtime order.

#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/stack_vector.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kick.hpp>

template <int order>
struct CudaExpansionType {
    Array<Position, kDimensions> pos;
    cosmictiger::gravity::cartesian::Expansion<float, order> expansion;
};

struct SearchParameters {
    int self;
    int phase;
};

struct CudaKickForceTile {
    float* gx;
    float* gy;
    float* gz;
    float* phi;
};

// One block's global storage of the interaction lists, checklist stacks and
// expansion stacks. With a list reservation the host carves it from a
// per-launch slab (kickCarveListStorage), one region per block, so the kernel
// touches the pool allocator only when a list outgrows its region.
template <int order>
struct CudaKickListStorage {
    int* dchecks;         // checklist stack entries
    int* dcheckBounds;    // stack level bounds, depth + 2 entries
    int* echecks;
    int* echeckBounds;
    int* leaflist;
    int* cplist;
    int* cclist;
    int* pclist;
    int* nextlist;
    int* bothlist;
    CudaExpansionType<order>* ancestors;
    CudaExpansionType<order>* current;
    float* forceBacking;
    SearchParameters* params;
    int listEntries;       // entries per interaction list and checklist stack
    int depth;             // stack levels (ancestors, search parameters, bounds)
    int forceEntries;      // whole-bucket force backing floats; 0 with shared forces
    int currentEntries;    // 1 when the active expansion stays in global backing
};

// Each region starts on a 16-byte boundary; each block's slab region on a
// 128-byte one.
inline constexpr int kKickListRegionAlignment = 16;
inline constexpr int kKickListBlockAlignment = 128;

namespace detail {

struct KickListCarver {
    char* base;    // nullptr: measure only
    int offset;
    template <class T>
    CUDA_EXPORT inline T* take(int count) {
        offset = ((offset + kKickListRegionAlignment - 1) / kKickListRegionAlignment) *
            kKickListRegionAlignment;
        T* pointer = base ? reinterpret_cast<T*>(base + offset) : nullptr;
        offset += count * int(sizeof(T));
        return pointer;
    }
};

}    // namespace detail

// Carve one block's list storage from `base` in a fixed order, or only measure
// it when base is null. Returns the bytes per block. The host sizes the slab
// and the kernel resolves its block's regions with this same function.
template <int order>
CUDA_EXPORT inline int kickCarveListStorage(
    CudaKickArenaLayout const& layout, char* base, CudaKickListStorage<order>& storage) {
    detail::KickListCarver carver{base, 0};
    storage.listEntries = layout.listReserve;
    storage.depth = kCudaMaxDepth;
    storage.forceEntries = layout.forceBackingCapacity;
    storage.currentEntries = layout.expansionOffset >= 0 ? 0 : 1;
    storage.dchecks = carver.take<int>(storage.listEntries);
    storage.dcheckBounds = carver.take<int>(storage.depth + 2);
    storage.echecks = carver.take<int>(storage.listEntries);
    storage.echeckBounds = carver.take<int>(storage.depth + 2);
    storage.leaflist = carver.take<int>(storage.listEntries);
    storage.cplist = carver.take<int>(storage.listEntries);
    storage.cclist = carver.take<int>(storage.listEntries);
    storage.pclist = carver.take<int>(storage.listEntries);
    storage.nextlist = carver.take<int>(storage.listEntries);
    storage.bothlist = carver.take<int>(storage.listEntries);
    storage.ancestors = carver.take<CudaExpansionType<order>>(storage.depth);
    storage.params = carver.take<SearchParameters>(storage.depth);
    storage.current = carver.take<CudaExpansionType<order>>(storage.currentEntries);
    storage.forceBacking = carver.take<float>(storage.forceEntries);
    return ((carver.offset + kKickListBlockAlignment - 1) / kKickListBlockAlignment) *
        kKickListBlockAlignment;
}

// Persistent per-block control state. Only these objects are in shared memory;
// the elements of the vectors and stacks live in the block's list storage
// (or, without a reservation, in the cudaAllocate pool).
template <int order>
struct CudaKickControl {
    StackVector<int> echecks;
    StackVector<int> dchecks;
    DeviceVector<int> leaflist;
    DeviceVector<int> cplist;
    DeviceVector<int> cclist;
    DeviceVector<int> pclist;
    DeviceVector<int> nextlist;
    DeviceVector<int>
        bothlist;    // equal radii: opened sources whose children go to the sink's children
    DeviceVector<CudaExpansionType<order>> ancestors;    // expansions awaiting their right child
    DeviceVector<CudaExpansionType<order>> current;    // active expansion when not in shared memory
    DeviceVector<float> forceBacking;    // whole-bucket forces when not in shared memory
    DeviceVector<SearchParameters> params;
    BarrierType barrier;
    // Resolved arena pointers.
    Position* sourceX;
    Position* sourceY;
    Position* sourceZ;
    float* sourceMass;
    char* multipoles;
    float* scratchExpansion;
    cosmictiger::gravity::cartesian::Expansion<float, order>* expansion;
    Array<Position, kDimensions>* expansionPos;
    CudaKickForceTile forces;
    CudaKickArenaLayout layout;

    // Lists on pool blocks grown on demand.
    CudaKickControl() = default;
    // Collective: lists on the block's slab regions (every lane constructs).
    __device__ inline CudaKickControl(CudaKickListStorage<order> const& storage)
      : echecks(storage.echecks, storage.listEntries, storage.echeckBounds, storage.depth)
      , dchecks(storage.dchecks, storage.listEntries, storage.dcheckBounds, storage.depth)
      , leaflist(storage.leaflist, storage.listEntries)
      , cplist(storage.cplist, storage.listEntries)
      , cclist(storage.cclist, storage.listEntries)
      , pclist(storage.pclist, storage.listEntries)
      , nextlist(storage.nextlist, storage.listEntries)
      , bothlist(storage.bothlist, storage.listEntries)
      , ancestors(storage.ancestors, storage.depth)
      , current(storage.current, storage.currentEntries)
      , forceBacking(storage.forceBacking, storage.forceEntries)
      , params(storage.params, storage.depth) {}
};

template <int order>
struct CudaKickParameters {
    FmmGeometry geometry;
    ParticleInteger outputOffset;
    Array<Position, kDimensions> lpos;
    cosmictiger::gravity::cartesian::Expansion<float, order> lValue;
    int self;
    int* dchecks;
    int* echecks;
    int dcount;
    int ecount;
};

// listSlab: the per-launch list slab (layout.listSlabBytes per block), or null
// when the lists grow from the pool.
template <int order>
__global__ void cudaKickKernel(KickReturn* rc, KickParameters globalParams, CudaKickData data,
    CudaKickParameters<order>* params, int itemCount, int* nextItem, CudaKickArenaLayout layout,
    char* listSlab);

// Per-order host entry points, defined in kick_cuda_launch.hpp.
template <int order>
KickReturn cudaExecuteKicksImpl(KickParameters params, Position* devX, Position* devY,
    Position* devZ, TreeNode* devTreeNodes, Vector<KickWorkitem> workitems,
    detail::FmmForceOutput output);
template <int order>
CudaKickArenaLayout const& kickArenaLayoutImpl();
