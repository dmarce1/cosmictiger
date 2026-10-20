/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

#include <cosmictiger/analysis/lightcone_entry.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/cuda/cuda_unordered_map.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>

// HEALPix nested-scheme pixel lookup on the device (healpix.cu).
#ifdef USE_CUDA
void healpixInit();
#endif
#ifdef __CUDACC__
__device__ void vec2pixNest(long const nside, double* vec, long* ipix);
#endif

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

#ifdef USE_CUDA
using LightconeVelocityReal = __half;
constexpr double kLcVelrealRange = 100.0;

inline LightconeVelocityReal float2velreal(float v, float vmax) {
    return __double2half(v / vmax * 10000.0);
}

inline float velreal2float(LightconeVelocityReal i, float vmax) {
    return __half2float(i) / 10000.0 * vmax;
}

#endif

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
void prepareLightconeGroupIteration(DeviceVector<LightconeTreeId> const& leaves,
    LightconeTreeMapType& treeMap, Pair<int> ownedPixels);
size_t cpuLightcone(DeviceVector<LightconeTreeId> const& leaves,
    LightconeParticleMapType& particleMap, LightconeTreeMapType& treeMap, LightconeGroup& nextGroup,
    double linkingLength, int rank, int ranks);
size_t lcFindNeighbors();
void lcGroups2homes();
void lcParts2groups(double a, double linkingLength, int index);
void lcSave(FILE* fp);
void lcLoad(FILE* fp);
size_t cudaLightcone(DeviceVector<LightconeTreeId> const& leaves,
    LightconeParticleMapType* particleMapPtr, LightconeTreeMapType* treeMapPtr,
    LightconeGroup* nextGroup, double linkingLength, int rank, int ranks, int maximumBlocks = 0);
