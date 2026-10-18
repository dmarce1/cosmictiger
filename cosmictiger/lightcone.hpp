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
#include <cosmictiger/lightcone_entry.hpp>
#include <cosmictiger/range.hpp>

#define LC_NO_GROUP (0xFFFFFFFFFFFFFFULL)
#define LC_EDGE_GROUP (0x0ULL)

struct LightconeTreeId {
    int pix;
    int index;

    CUDA_EXPORT
    bool operator!=(LightconeTreeId const& other) const {
        return pix != other.pix || index != other.index;
    }
};

struct LightconeTreeNode {
    Range<double> box;
    Array<LightconeTreeId, NCHILD> children;
    Pair<int> particleRange;
    int active;
    int lastActive;
    DeviceVector<LightconeTreeId> neighbors;
    int pix;
};

using LightconeVelocityReal = __half;
constexpr double kLcVelrealRange = 100.0;

inline LightconeVelocityReal float2velreal(float v, float vmax) {
    return __double2half(v / vmax * 10000.0);
}

inline float velreal2float(LightconeVelocityReal i, float vmax) {
    return __half2float(i) / 10000.0 * vmax;
}

using LightconeGroup = unsigned long long;

struct LightconeParticle {
    Array<LightconeReal, NDIM> pos;
    Array<float, NDIM> vel;
    LightconeGroup group;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & pos;
        arc & vel;
        arc & group;
    }
};

struct LightconeParticles {
    Array<DeviceVector<LightconeReal>, NDIM> pos;
    Array<DeviceVector<float>, NDIM> vel;
    DeviceVector<LightconeGroup> group;
};

using LightconeParticleMapType = CudaUnorderedMap<LightconeParticles>;
using LightconeTreeMapType = CudaUnorderedMap<DeviceVector<LightconeTreeNode>>;

int lcNside();
void lcInit(double, double);
int lcAddParticle(double x0, double y0, double z0, double x1, double y1, double z1, float vx,
    float vy, float vz, float t, float dt, Vector<LightconeParticle>& thisParticleBuffer);
void lcAddParticles(Vector<LightconeParticle>&&);
size_t lcAddParticles(
    DeviceVector<DeviceVector<LightconeEntry>> const& entries, double scale, double tau);
void lcBuffer2homes();
size_t lcTimeToFlush(double, double);
void lcParticleBoundaries1();
void lcParticleBoundaries2();
void lcFormTrees(double tmax, double linkingLength);
size_t lcFindGroups();
size_t lcFindNeighbors();
void lcGroups2homes();
void lcParts2groups(double a, double linkingLength, int index);
void lcSave(FILE* fp);
void lcLoad(FILE* fp);
size_t cudaLightcone(DeviceVector<LightconeTreeId> const& leaves,
    LightconeParticleMapType* particleMapPtr, LightconeTreeMapType* treeMapPtr,
    LightconeGroup* nextGroup);
