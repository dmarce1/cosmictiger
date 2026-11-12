/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef KICK_HPP_
#define KICK_HPP_

#include <functional>
#include <memory>

#include <cosmictiger/core/force_step.hpp>
#include <cosmictiger/gravity/fmm_geometry.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/kick_interaction_counts.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/vector3.hpp>

// Runtime compute.bucketSize is validated against this power-of-two cap. Tree
// leaves hold between half the runtime bucket plus one and the runtime bucket
// particles, and the CUDA kick arena is tuned for a runtime bucket equal to the
// cap: its automatic sink-force tile covers the whole bucket when it fits.
inline constexpr int kMaxBucketSize = MAX_BUCKET_SIZE;
static_assert(
    kMaxBucketSize >= 32 && kMaxBucketSize <= 1024 && (kMaxBucketSize & (kMaxBucketSize - 1)) == 0,
    "MAX_BUCKET_SIZE must be a power of two from 32 to 1024");

struct SphParticleFlags;

struct CudaKickData {
    TreeNode* treeNodes;
    Position* x;
    Position* y;
    Position* z;
    FmmSourceView ghostSources;
    Vector3<float>* vel;
    bool midpointHydro = false;
    SphParticleFlags const* sphHydroFlags = nullptr;
    ParticleInteger* catIndex;
    char* type;
    bool sph;
    float darkMatterMass;
    float sphMass;
    float const* sourceMasses;
    char* rungs;
    float* gx;
    float* gy;
    float* gz;
    float* pot;
    Vector3<float>* deferredGasOpening = nullptr;
    ParticleInteger const* gravitySampleIndices;
    Vector3<float>* gravitySampleAccelerations;
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
// live in global memory owned by the control block: with a list reservation,
// in the block's region of a per-launch slab (listSlabBytes per block, carved
// by kickCarveListStorage in kick_cuda.hpp, pool growth only on overflow),
// otherwise in pool blocks grown on demand.
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
    int multipoleStride;      // bytes per staged multipole: the tree arena's record stride
    bool massPlane;           // mixed-mass sources stage a mass plane
    int residency;            // resident blocks per multiprocessor the budget targets
    int budgetBytes;          // dynamic shared bytes allowed per block at that residency
    int listReserve;          // entries reserved per interaction list and checklist stack; 0: none
    int forceBackingCapacity;    // floats of whole-bucket force backing; 0 with shared forces
    int listSlabBytes;           // per-block bytes of the per-launch list slab; 0: pool growth
};

using KickReturn = ForceStepResult;

struct KickParameters {
    FmmGeometry geometry{};
    ParticleInteger outputOffset = 0;
    bool forceOnly = false;
    bool profileCoordinator = true;
    // Reserved HPX message fields from the retired global tree-kick path.
    bool globalTimestep = false;
    bool deferGasOpening = false;
    // At a coarse hydro event, collect gas opening impulses until hydro finishes.
    bool deferHydroOpening = false;
    // At all-particle common boundaries, gas uses the full top force to choose
    // a rung. Neighbor closure follows before any descending subset is built.
    bool fullGasRungSelection = false;
    // Hold those balanced gas rungs through correction walks and intervening
    // boundaries. Dark-matter rung selection remains unchanged.
    bool freezeGasRungs = false;
    double previousTimestep = 0.0;    // Reserved HPX message field from the retired path.
    bool htime;
    float sign;
    int minLevel{};
    int minRung{};
    int fixedRung{};    // Optional fixed-rung override used by kick regressions.
    float a = 1.0f;
    float t0{};
    float theta{};
    float gravitySmoothingLength{};
    float eta{};
    float gm{};
    bool velInit;
    bool saveForce{};
    bool firstCall{};
    bool gpu{};
    float nodeLoad{};
    float maxDt;
    bool ascending{};
    bool descending{};
    bool top{};
    bool doPhi;
    // Opening rule, stamped by kick() from the options so that the CPU walk,
    // the remote kick action and the CUDA kernel apply the same rule.
    bool openLargerCell;
    // Mesh solvers (pm, p3m): base timestep of the substep being closed when a
    // new outer interval changed t0, and the long half-kick durations applied
    // from the per-particle long acceleration at an outer boundary (zero: none).
    float closingBaseTimestep = 0.0f;
    float longCloseTimestep = 0.0f;
    float longOpenTimestep = 0.0f;
    // PMMM outer interval with frozen near membership (docs/pmmm.md):
    // frozenNear evaluates the near field on the stored cell membership without
    // a mesh solve; refreshMembership re-sorts by position and stores the new
    // keys first (boundary pass); frozenNearClose applies only the old-membership
    // closing correction -g_near (closingBase/2 - dt_rung/2) at a new boundary;
    // closeShortOverInterval makes the traditional short closing half-kick span
    // closingBase/2 for every rung, the remainder having been applied by the
    // old-membership pass.
    bool frozenNear = false;
    bool refreshMembership = false;
    bool frozenNearClose = false;
    bool closeShortOverInterval = false;
    /// Initialize common kick-policy defaults; callers populate the remaining
    /// invocation-specific state.
    KickParameters() {
        htime = true;
        openLargerCell = false;
        maxDt = 1e30;
        doPhi = true;
        velInit = false;
        sign = 1.0;
    }
    /// Archive `htime`, `ascending`, `descending`, `top`, `maxDt`, `minRung` and the
    /// remaining fields in transport order.
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
        arc & gravitySmoothingLength;
        arc & eta;
        arc & gm;
        arc & sign;
        arc & saveForce;
        arc & nodeLoad;
        arc & minLevel;
        arc & doPhi;
        arc & velInit;
        arc & openLargerCell;
        arc & geometry;
        arc & forceOnly;
        arc & profileCoordinator;
        arc & globalTimestep;
        arc & deferGasOpening;
        arc & deferHydroOpening;
        arc & fullGasRungSelection;
        arc & freezeGasRungs;
        arc & previousTimestep;
        arc & outputOffset;
        arc & closingBaseTimestep;
        arc & longCloseTimestep;
        arc & longOpenTimestep;
        arc & frozenNear;
        arc & refreshMembership;
        arc & frozenNearClose;
        arc & closeShortOverInterval;
    }
};

// Collective noncommon-boundary protocol: begin after migration, collect during
// gravity walks, seal before any gas sort, and finish after hydro. Base-particle
// ordering must remain fixed between seal and finish. No global/shared step or
// common-boundary rung balance may occur while collecting.
void gravityBeginDeferredGasOpening();
void gravitySealDeferredGasOpening();
KickReturn gravityFinishDeferredGasOpening();
void gravityDiscardDeferredGasOpening();
bool gravityGasOpeningDeferred();
Vector3<float>* gravityDeferredGasOpeningData();

// Rungs to add to a stored hydro rung proposal at an outer boundary. The proposal r₀ was
// quantized against the closing interval H_old, Δτ ≥ H_old 2^−r₀; under the opening
// interval H_new it needs r ≥ r₀ + log₂(H_new/H_old) (derived). A shorter new interval
// keeps the proposal, which is then finer than needed.
CUDA_EXPORT inline int hydroRungIntervalShift(KickParameters const& params) {
    if (params.closingBaseTimestep > 0.0f && params.t0 > params.closingBaseTimestep) {
        return int(ceilf(log2f(params.t0 / params.closingBaseTimestep)));
    }
    return 0;
}

struct KickWorkitem {
    FmmGeometry geometry{};
    ParticleInteger outputOffset = 0;
    ExpansionBuffer l;
    Vector3<Position> pos;
    TreeId self;
    Vector<TreeId> dchecklist;
    Vector<TreeId> echecklist;
};

struct KickWorkspace;

struct FmmForceResult {
    // Compact fields use the same index as particleIndices, whose entries are
    // owned particle indices. A complete forest has the identity mapping.
    Vector<ParticleInteger> particleIndices;
    Vector<float> potential;
    Array<Vector<float>, kDimensions> acceleration;
    KickInteractionCounts interactionCounts{};
};

namespace detail {

// The batch owns this scratch until its CPU walks and CUDA launch both finish.
struct FmmForceOutput {
    float* potential = nullptr;
    Array<float*, kDimensions> acceleration{};
};

}    // namespace detail

#ifdef USE_CUDA
// Host-side FIFO executor; its HPX state is kept out of CUDA translation units.
KickReturn kickRunCudaLaunch(std::function<KickReturn()> const& launch);
// Dispatches to the per-order CUDA kick instantiation for kFmmOrder.
KickReturn cudaExecuteKicks(KickParameters params, Position*, Position*, Position*, TreeNode*,
    Vector<KickWorkitem> workitems, detail::FmmForceOutput output = {},
    float const* sourceMasses = nullptr);
// Shared-memory policy of the build-order kick kernel.
CudaKickArenaLayout const& kickArenaLayout();
#endif
// CPU/GPU handoff divisor (see kickBlockCount in kick.cu); measured, not
// derived: kKickHandoffBlocksPerMultiprocessor resident blocks' worth of
// subtrees at kKickHandoffReferenceParticles particles, growing as the cube
// root of the particle count (one more CPU level per eightfold increase) up
// to kKickHandoffMaxScale times that many blocks (reached at 27 times the
// reference count, 884,736 particles).
constexpr int kKickHandoffBlocksPerMultiprocessor = 2;
constexpr double kKickHandoffReferenceParticles = 32768.0;
constexpr double kKickHandoffMaxScale = 3.0;
int kickBlockCount(ParticleInteger particles);
// Optional finer divisor for the near-cell walks (see kickFineBlockCount in
// kick.cu); never below kickBlockCount.
int kickFineBlockCount(ParticleInteger particles);

#endif /* KICK_HPP_ */
