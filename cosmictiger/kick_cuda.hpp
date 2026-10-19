/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

// CUDA kick: order-templated declarations. The kernel, its device operators and
// the host launch/layout code are templates defined in kick_cuda_impl.hpp and
// instantiated once per FMM order by the CMake-generated instance translation
// units (kick_cuda_instances.hpp). kick.cu dispatches on the runtime order.

#include <cooperative_groups.h>
#include <cuda/barrier>

#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/fmm_types.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/stack_vector.hpp>

template <int order>
struct CudaExpansionType {
    Array<Fixed32, NDIM> pos;
    FmmExpansion<float, order> expansion;
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

// Persistent per-block control state. Element storage of the vectors and
// stacks lives in the cudaAllocate pool; only these objects are in shared memory.
template <int order>
struct CudaKickControl {
    StackVector<int> echecks;
    StackVector<int> dchecks;
    DeviceVector<int> leaflist;
    DeviceVector<int> cplist;
    DeviceVector<int> cclist;
    DeviceVector<int> pclist;
    DeviceVector<int> nextlist;
    DeviceVector<CudaExpansionType<order>> ancestors;    // expansions awaiting their right child
    DeviceVector<CudaExpansionType<order>> current;    // active expansion when not in shared memory
    DeviceVector<float> forceBacking;    // whole-bucket forces when not in shared memory
    DeviceVector<SearchParameters> params;
    BarrierType barrier;
    // Resolved arena pointers.
    Fixed32* sourceX;
    Fixed32* sourceY;
    Fixed32* sourceZ;
    float* sourceMass;
    char* multipoles;
    float* scratchExpansion;
    FmmExpansion<float, order>* expansion;
    Array<Fixed32, NDIM>* expansionPos;
    CudaKickForceTile forces;
    CudaKickArenaLayout layout;
};

template <int order>
struct CudaKickParameters {
    Array<Fixed32, NDIM> lpos;
    FmmExpansion<float, order> lValue;
    int self;
    int* dchecks;
    int* echecks;
    int dcount;
    int ecount;
};

template <int order>
__global__ void cudaKickKernel(KickReturn* rc, KickParameters globalParams, CudaKickData data,
    CudaKickParameters<order>* params, int itemCount, int* nextItem, CudaKickArenaLayout layout);

// Per-order host entry points, defined in kick_cuda_impl.hpp.
template <int order>
KickReturn cudaExecuteKicksImpl(KickParameters params, Fixed32* devX, Fixed32* devY, Fixed32* devZ,
    TreeNode* devTreeNodes, Vector<KickWorkitem> workitems);
template <int order>
CudaKickArenaLayout const& kickArenaLayoutImpl();
