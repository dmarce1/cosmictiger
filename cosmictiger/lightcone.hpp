/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#include <cuda_fp16.h>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/cuda_unordered_map.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/range.hpp>

using LcReal = Fixed<int, 31>;

#define LC_NO_GROUP (0xFFFFFFFFFFFFFFULL)
#define LC_EDGE_GROUP (0x0ULL)

struct LcTreeId {
    int pix;
    int index;

    CUDA_EXPORT
    bool operator!=(LcTreeId const& other) const {
        return pix != other.pix || index != other.index;
    }
};

struct LcTreeNode {
    Range<double> box;
    Array<LcTreeId, NCHILD> children;
    Pair<int> partRange;
    int active;
    int lastActive;
    DeviceVector<LcTreeId> neighbors;
    int pix;
};

struct LcEntry {
    LcReal x, y, z;
    float vx, vy, vz;
    int pix;
};

using LcVelreal = __half;
constexpr double kLcVelrealRange = 100.0;

inline LcVelreal float2velreal(float v, float vmax) {
    return __double2half(v / vmax * 10000.0);
}

inline float velreal2float(LcVelreal i, float vmax) {
    return __half2float(i) / 10000.0 * vmax;
}

using LcGroup = unsigned long long;

struct LcParticle {
    Array<LcReal, NDIM> pos;
    Array<float, NDIM> vel;
    LcGroup group;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & pos;
        arc & vel;
        arc & group;
    }
};

struct LcParticles {
    Array<DeviceVector<LcReal>, NDIM> pos;
    Array<DeviceVector<float>, NDIM> vel;
    DeviceVector<LcGroup> group;
};

using LcPartMapType = CudaUnorderedMap<LcParticles>;
using LcTreeMapType = CudaUnorderedMap<DeviceVector<LcTreeNode>>;

int lcNside();
void lcInit(double, double);
int lcAddParticle(double x0, double y0, double z0, double x1, double y1, double z1, float vx,
    float vy, float vz, float t, float dt, Vector<LcParticle>& thisPartBuffer);
void lcAddParts(Vector<LcParticle>&&);
size_t lcAddParts(DeviceVector<DeviceVector<LcEntry>> const& entries, double scale, double tau);
void lcBuffer2homes();
size_t lcTimeToFlush(double, double);
void lcParticleBoundaries1();
void lcParticleBoundaries2();
void lcFormTrees(double tmax, double linkLen);
size_t lcFindGroups();
size_t lcFindNeighbors();
void lcGroups2homes();
void lcParts2groups(double a, double linkLen, int index);
void lcSave(FILE* fp);
void lcLoad(FILE* fp);
size_t cudaLightcone(DeviceVector<LcTreeId> const& leaves, LcPartMapType* partMapPtr,
    LcTreeMapType* treeMapPtr, LcGroup* nextGroup);
