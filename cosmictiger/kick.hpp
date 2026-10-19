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

#ifndef KICK_HPP_
#define KICK_HPP_

#include <atomic>

#include <cosmictiger/fmm_types.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/tree.hpp>

// Runtime compute.bucketSize is validated against this power-of-two cap. Tree
// leaves hold between half the runtime bucket plus one and the runtime bucket
// particles, and the CUDA kick arena is tuned for a runtime bucket equal to the
// cap: its automatic sink-force tile covers the whole bucket when it fits.
inline constexpr int kMaxBucketSize = MAX_BUCKET_SIZE;
static_assert(
    kMaxBucketSize >= 32 && kMaxBucketSize <= 1024 && (kMaxBucketSize & (kMaxBucketSize - 1)) == 0,
    "MAX_BUCKET_SIZE must be a power of two from 32 to 1024");

struct CudaKickData {
    TreeNode* treeNodes;
    Fixed32* x;
    Fixed32* y;
    Fixed32* z;
    Array<float, NDIM>* vel;
    float* sphGravityAccelerationX;
    float* sphGravityAccelerationY;
    float* sphGravityAccelerationZ;
    ParticleInteger* catIndex;
    char* type;
    bool sph;
    float darkMatterMass;
    float sphMass;
    char* rungs;
    float* gx;
    float* gy;
    float* gz;
    float* pot;
};

// Host-computed byte layout of one block's dynamic shared memory.
//
// Persistent region: the control block (list/stack control objects, the
// staging barrier and the resolved region pointers) and, when enabled, the
// active local expansion with its position. Reusable phase arena: the L2L
// output scratch, then the sink-force planes followed by the particle-source
// planes or the staged positioned multipoles, which overlap each other. Forces
// are initialized only after the cell interactions, so the scratch expansion
// aliases them. Interaction lists, ancestor expansions and checklist stacks
// stay in growable global backing owned by the control block.
// Plain aggregate: the device control block is constructed by every lane, and a
// value-initialized member would race with lane zero's later assignment.
struct CudaKickArenaLayout {
    int order;    // FMM order the layout was selected for
    int totalBytes;
    int controlOffset;
    int expansionOffset;    // -1: the active expansion stays in global backing
    int arenaOffset;
    int arenaBytes;
    int forceOffset;          // arena-relative; -1: forces use whole-bucket global backing
    int sourceOffset;         // arena-relative
    int scratchOffset;        // arena-relative L2L output expansion
    int sinkCapacity;         // sink-force tile; 0 with global force backing
    int particleCapacity;     // staged particle sources, a multiple of WARP_SIZE
    int multipoleCapacity;    // staged positioned multipoles
    int multipoleStride;      // bytes per staged multipole, a multiple of 16
    bool massPlane;           // mixed-mass sources stage a mass plane
    int residency;            // resident blocks per multiprocessor the budget targets
    int budgetBytes;          // dynamic shared bytes allowed per block at that residency
    int listReserve;          // entries reserved per interaction list and checklist stack; 0: none
};

struct KickReturn;

struct KickReturn {
    int maxRung;
    double pot;
    double kin;
    double dkin;
    double xmom;
    double ymom;
    double zmom;
    double nmom;
    double darkMatterKineticCorrection;
    double darkMatterXMomentumCorrection;
    double darkMatterYMomentumCorrection;
    double darkMatterZMomentumCorrection;
    double darkMatterMomentumNormCorrection;

    CUDA_EXPORT
    KickReturn() {
        maxRung = 0;
        pot = 0.0;
        kin = 0.0;
        xmom = ymom = zmom = nmom = 0.0;
        dkin = 0.0;
        darkMatterKineticCorrection = 0.0;
        darkMatterXMomentumCorrection = 0.0;
        darkMatterYMomentumCorrection = 0.0;
        darkMatterZMomentumCorrection = 0.0;
        darkMatterMomentumNormCorrection = 0.0;
    }
    CUDA_EXPORT
    KickReturn& operator+=(KickReturn const& other) {
        if (other.maxRung > maxRung) {
            maxRung = other.maxRung;
        }
        pot += other.pot;
        kin += other.kin;
        xmom += other.xmom;
        ymom += other.ymom;
        zmom += other.zmom;
        nmom += other.nmom;
        dkin += other.dkin;
        darkMatterKineticCorrection += other.darkMatterKineticCorrection;
        darkMatterXMomentumCorrection += other.darkMatterXMomentumCorrection;
        darkMatterYMomentumCorrection += other.darkMatterYMomentumCorrection;
        darkMatterZMomentumCorrection += other.darkMatterZMomentumCorrection;
        darkMatterMomentumNormCorrection += other.darkMatterMomentumNormCorrection;

        return *this;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & maxRung;
        arc & pot;
        arc & kin;
        arc & xmom;
        arc & ymom;
        arc & zmom;
        arc & nmom;
        arc & dkin;
        arc & darkMatterKineticCorrection;
        arc & darkMatterXMomentumCorrection;
        arc & darkMatterYMomentumCorrection;
        arc & darkMatterZMomentumCorrection;
        arc & darkMatterMomentumNormCorrection;
    }
};

struct KickParameters {
    bool htime;
    float sign;
    int minLevel;
    int minRung;
    int fixedRung{};    // Nonzero only for synchronized primordial source splitting.
    float a;
    float t0;
    float theta;
    float gravitySupportRadius;
    float eta;
    float gm;
    bool velInit;
    bool saveForce;
    bool firstCall;
    bool gpu;
    float nodeLoad;
    float maxDt;
    bool ascending;
    bool descending;
    bool top;
    bool doPhi;
    // Opening rule, stamped by kick() from the options so that the CPU walk,
    // the remote kick action and the CUDA kernel apply the same rule.
    bool openLargerCell;
    KickParameters() {
        htime = true;
        openLargerCell = false;
        maxDt = 1e30;
        doPhi = true;
        velInit = false;
        sign = 1.0;
    }
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & htime;
        arc & ascending;
        arc & descending;
        arc & top;
        arc & maxDt;
        arc & minRung;
        arc & fixedRung;
        arc & gpu;
        arc & a;
        arc & t0;
        arc & theta;
        arc & firstCall;
        arc & gravitySupportRadius;
        arc & eta;
        arc & gm;
        arc & sign;
        arc & saveForce;
        arc & nodeLoad;
        arc & minLevel;
        arc & doPhi;
        arc & velInit;
        arc & openLargerCell;
    }
};

struct KickWorkitem {
    ExpansionBuffer l;
    Array<Fixed32, NDIM> pos;
    TreeId self;
    Vector<TreeId> dchecklist;
    Vector<TreeId> echecklist;
};

struct KickWorkspace;

#ifndef __CUDACC__
// The expansion buffer carries the local expansion of the runtime FMM order;
// kick() dispatches to the per-order walk instantiation.
KickReturn kick(KickParameters, ExpansionBuffer l, Array<Fixed32, NDIM> pos, TreeId self,
    Vector<TreeId> dchecklist, Vector<TreeId> echecklist, std::shared_ptr<KickWorkspace>);
#endif
void kickShowTimings();
void kickSetRc(KickReturn kr);
#ifdef USE_CUDA
// Dispatches to the per-order CUDA kick instantiation for fmmOrder().
KickReturn cudaExecuteKicks(
    KickParameters params, Fixed32*, Fixed32*, Fixed32*, TreeNode*, Vector<KickWorkitem> workitems);
// Shared-memory policy of the runtime order's kick kernel (selected once per order).
CudaKickArenaLayout const& kickArenaLayout();
#endif
// Resident blocks per multiprocessor whose particle share sets the CPU/GPU
// handoff threshold (see kickBlockCount in kick.cu); measured, not derived.
constexpr int kKickHandoffBlocksPerMultiprocessor = 2;
int kickBlockCount();
size_t kickEstimateCudaMemoryUsage(double theta, int numberOfParticles, int checkCount);

#endif /* KICK_HPP_ */
