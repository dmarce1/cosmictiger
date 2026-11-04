/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Device side of the order-templated CUDA kick: the tree-walk kernel and its
// warp helpers. Included through kick_cuda_launch.hpp by kick_cuda_instances.hpp.

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/cuda/device_profiler.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/gravity_cuda.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

template <int order>
__device__ inline void cudaCopyExpansion(float* destination, float const* source) {
    for (int i = threadIdx.x; i < FmmOrder<order>::kLocalCoefficients_; i += kWarpSize) {
        destination[i] = source[i];
    }
}

}    // namespace detail

template <int order>
__device__ FlopCounts doKick(KickReturn& result, KickParameters params, CudaKickData const& data,
    cosmictiger::gravity::cartesian::tracefree::Expansion<float, order> const& lValue2,
    TreeNode const& self, CudaKickForceTile const& force, int sinkBegin, int sinkCount,
    ParticleInteger outputOffset) {
    int const& tid = threadIdx.x;
    auto& allPhi = data.pot;
    auto& allGx = data.gx;
    auto& allGy = data.gy;
    auto& allGz = data.gz;
    auto* __restrict__ rungs = data.rungs;
    auto* __restrict__ vels = data.vel;
    ParticleInteger const sinkFirst = self.particleRange.first + sinkBegin;
    auto const* sinkX = data.x + sinkFirst;
    auto const* sinkY = data.y + sinkFirst;
    auto const* sinkZ = data.z + sinkFirst;
    int maxRung = 0;
    float minimumTimestep = std::numeric_limits<float>::max();
    float phiTot = 0.0f;
    float kinTot = 0.f;
    float xmomTot = 0.0f;
    float ymomTot = 0.0f;
    float zmomTot = 0.0f;
    float nmomTot = 0.0f;
    float dkinTot = 0.f;
    float darkMatterKineticCorrection = 0.0f;
    float darkMatterXMomentumCorrection = 0.0f;
    float darkMatterYMomentumCorrection = 0.0f;
    float darkMatterZMomentumCorrection = 0.0f;
    float darkMatterMomentumNormCorrection = 0.0f;

    float const gravityEffectiveRadius =
        params.gravitySupportRadius / selectedGravityKernel().supportRadiusPerEffectiveRadius;
    FlopCounter<FlopCount> flops = 0;
    // Widen support radius, divide by the double kernel radius, then narrow.
    FlopCounter<FlopCount> doubleFlops = 3;
    for (int i = tid; i < sinkCount; i += kWarpSize) {
        cosmictiger::gravity::Field<float> l2{};
        Vector3<float> dx;
        ParticleInteger sinkIndex;
        float vx;
        float vy;
        float vz;
        float dt;
        float g2;
        int rung;
        sinkIndex = sinkFirst + i;
        if (!params.forceOnly && !params.htime && rungs[sinkIndex] < params.minRung) {
            continue;
        }
        ASSERT(sinkIndex >= 0);
        dx[kXDimension] = unwrappedDistance(sinkX[i], self.pos[kXDimension]);
        dx[kYDimension] = unwrappedDistance(sinkY[i], self.pos[kYDimension]);
        dx[kZDimension] = unwrappedDistance(sinkZ[i], self.pos[kZDimension]);
        cosmictiger::gravity::cartesian::tracefree::L2P(
            l2, lValue2, fmmScaleDisplacement(dx), params.doPhi);
        float gx = force.gx[i];
        float gy = force.gy[i];
        float gz = force.gz[i];
        float phi = force.phi[i];
        phi += kScaleFactor1 * l2.potential();
        gx += kScaleFactor2 * l2.accelerationX();
        gy += kScaleFactor2 * l2.accelerationY();
        gz += kScaleFactor2 * l2.accelerationZ();
        if (params.forceOnly) {
            ParticleInteger const outputIndex = sinkIndex + outputOffset;
            allPhi[outputIndex] = params.doPhi ? phi : 0.0f;
            allGx[outputIndex] = gx;
            allGy[outputIndex] = gy;
            allGz[outputIndex] = gz;
            flops += cosmictiger::gravity::cartesian::tracefree::detail::L2P<fmmLocalDegree(order),
                         true>::kFlops_ +
                14;
            continue;
        }
        gz *= params.gm;
        gy *= params.gm;
        gx *= params.gm;
        phi *= params.gm;
        if (data.gravitySampleAccelerations != nullptr) {
            ParticleInteger const slot = data.gravitySampleIndices[sinkIndex];
            if (slot >= 0) {
                data.gravitySampleAccelerations[slot][kXDimension] = gx;
                data.gravitySampleAccelerations[slot][kYDimension] = gy;
                data.gravitySampleAccelerations[slot][kZDimension] = gz;
            }
        }
        vx = vels[sinkIndex][kXDimension];
        vy = vels[sinkIndex][kYDimension];
        vz = vels[sinkIndex][kZDimension];
        float const mass = data.sourceMasses != nullptr ?
            data.sourceMasses[sinkIndex] :
            data.sph ?
            (data.type[sinkIndex] == kDarkMatterType ? data.darkMatterMass : data.sphMass) :
            1.0f;
        float const kin0 = 0.5f * mass * sqr(vx, vy, vz);
        if (params.globalTimestep) {
            float const accelerationSquared = sqr(gx, gy, gz) + 1e-35f;
            float const timestep = fminf(params.maxDt,
                params.eta * sqrtf(params.a) *
                    sqrtf(gravityEffectiveRadius / sqrtf(accelerationSquared)));
            ALWAYS_ASSERT(isfinite(timestep) && timestep > 0.0f);
            minimumTimestep = fminf(minimumTimestep, timestep);
#if COSMICTIGER_WITH_SPH
            bool const gas = data.sph && data.type[sinkIndex] == kSphType;
            if (gas && !data.midpointHydro) {
                ParticleInteger const k = data.catIndex[sinkIndex];
                data.sphGravityAccelerationX[k] = gx;
                data.sphGravityAccelerationY[k] = gy;
                data.sphGravityAccelerationZ[k] = gz;
            }
#else
            bool const gas = false;
#endif
            if (!gas || data.midpointHydro) {
                data.globalAcceleration[kXDimension][sinkIndex] = gx;
                data.globalAcceleration[kYDimension][sinkIndex] = gy;
                data.globalAcceleration[kZDimension][sinkIndex] = gz;
                double const halfStep = params.sign * 0.5 * params.previousTimestep;
                vx = fma(double(gx), halfStep, double(vx));
                vy = fma(double(gy), halfStep, double(vy));
                vz = fma(double(gz), halfStep, double(vz));
                doubleFlops += 18;    // Double half-step and three mixed-precision FMAs.
            }
            if (!gas || data.midpointHydro) {
                rungs[sinkIndex] = 0;
                // Midpoint gas receives the same gravitational kick as dark matter.
                kinTot += 0.5f * mass * sqr(vx, vy, vz);
                xmomTot += mass * vx;
                ymomTot += mass * vy;
                zmomTot += mass * vz;
                nmomTot += mass * sqrtf(sqr(vx, vy, vz));
            }
            if (allGx != nullptr) {
                allGx[sinkIndex] = gx;
                allGy[sinkIndex] = gy;
                allGz[sinkIndex] = gz;
            }
            if (allPhi != nullptr) {
                allPhi[sinkIndex] = phi;
            }
            flops += 12;
        } else if (params.htime) {
            // Signed gravity-only ladder, including gas on correction levels.
            float const sgn = params.sign * (params.top ? 1.f : -1.f);
            flops += 1;    // Kick direction; exclude energy and momentum diagnostics.
            auto const applyKick = [&](float halfStep) {
                vx = fmaf(gx, halfStep, vx);
                vy = fmaf(gy, halfStep, vy);
                vz = fmaf(gz, halfStep, vz);
                flops += 7;
            };
            if (params.ascending) {
                dt = 0.5f * rungDt[params.minRung] * params.t0;
                flops += 2;
                if (!params.firstCall) {
                    applyKick(sgn * dt);
                }
            }
            kinTot += 0.5f * mass * sqr(vx, vy, vz);
            xmomTot += mass * vx;
            ymomTot += mass * vy;
            zmomTot += mass * vz;
            nmomTot += mass * sqrtf(sqr(vx, vy, vz));
            if (allGx != nullptr) {
                allGx[sinkIndex] = gx;
                allGy[sinkIndex] = gy;
                allGz[sinkIndex] = gz;
            }
            if (allPhi != nullptr) {
                allPhi[sinkIndex] = phi;
            }

            if (params.descending) {
                g2 = sqr(gx, gy, gz);
                dt = fminf(params.eta * sqrt(params.a * gravityEffectiveRadius * rsqrtf(g2)),
                    params.t0);    // 12
                dt = fminf(params.maxDt, dt);
                rung = rungs[sinkIndex];
                int nextRung =
                    max(params.minRung + int((int) ceilf(log2f(params.t0 / dt)) > params.minRung),
                        rung - 1);
#if COSMICTIGER_WITH_SPH
                if (data.midpointHydro && data.sph && data.type[sinkIndex] == kSphType) {
                    nextRung = max(nextRung, int(data.sphHydroRungs[data.catIndex[sinkIndex]]));
                }
#endif
                maxRung = max(nextRung, maxRung);
                ALWAYS_ASSERT(nextRung >= 0 && nextRung < kMaxRung);
                flops += 13;
                rungs[sinkIndex] = nextRung;
                dt = 0.5f * rungDt[params.minRung] * params.t0;
                applyKick(sgn * dt);
                flops += 2;
            }
        } else {
#if COSMICTIGER_WITH_SPH
            bool const gas = data.sph && data.type[sinkIndex] == kSphType;
#else
            bool const gas = false;
#endif
#if COSMICTIGER_WITH_SPH
            if (gas && !data.midpointHydro) {
                ParticleInteger const k = data.catIndex[sinkIndex];
                data.sphGravityAccelerationX[k] = gx;
                data.sphGravityAccelerationY[k] = gy;
                data.sphGravityAccelerationZ[k] = gz;
            }
#endif
            if ((!gas || data.midpointHydro) && !params.firstCall) {
                float const dt = params.sign * 0.5f * rungDt[rungs[sinkIndex]] * params.t0;
                vx = fmaf(gx, dt, vx);
                vy = fmaf(gy, dt, vy);
                vz = fmaf(gz, dt, vz);
                flops += 9;
            }
            if (!gas || data.midpointHydro) {
                // Legacy hydro owns gas momentum; midpoint hydro leaves gravity here.
                kinTot += 0.5f * mass * sqr(vx, vy, vz);
                xmomTot += mass * vx;
                ymomTot += mass * vy;
                zmomTot += mass * vz;
                nmomTot += mass * sqrtf(sqr(vx, vy, vz));
            }
            if (allGx != nullptr) {
                allGx[sinkIndex] = gx;
                allGy[sinkIndex] = gy;
                allGz[sinkIndex] = gz;
            }
            if (allPhi != nullptr) {
                allPhi[sinkIndex] = phi;
            }
            if (!gas || data.midpointHydro) {
                float const syncVx = vx, syncVy = vy, syncVz = vz;
                float const g2 = sqr(gx, gy, gz) + 1e-35f;
                float dt =
                    fminf(params.eta * sqrtf(params.a) * sqrtf(gravityEffectiveRadius / sqrtf(g2)),
                        params.t0);
                dt = fminf(dt, params.maxDt);
                rung = max(max((int) ceilf(log2f(params.t0 / dt)),
                               max((int) rungs[sinkIndex] - 1, params.minRung)),
                    1);
#if COSMICTIGER_WITH_SPH
                if (gas && data.midpointHydro) {
                    rung = max(rung, int(data.sphHydroRungs[data.catIndex[sinkIndex]]));
                }
#endif
                if (params.fixedRung != 0) {
                    maxRung = max(maxRung, int(rung));
                    rung = params.fixedRung;
                }
                ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
                rungs[sinkIndex] = rung;
                dt = params.sign * 0.5f * rungDt[rung] * params.t0;
                vx = fmaf(gx, dt, vx);
                vy = fmaf(gy, dt, vy);
                vz = fmaf(gz, dt, vz);
                darkMatterKineticCorrection +=
                    0.5f * mass * (sqr(syncVx, syncVy, syncVz) - sqr(vx, vy, vz));
                darkMatterXMomentumCorrection += mass * (syncVx - vx);
                darkMatterYMomentumCorrection += mass * (syncVy - vy);
                darkMatterZMomentumCorrection += mass * (syncVz - vz);
                darkMatterMomentumNormCorrection +=
                    mass * (sqrtf(sqr(syncVx, syncVy, syncVz)) - sqrtf(sqr(vx, vy, vz)));
                flops += 24;    // Timestep/rung selection and signed velocity kick.
            }
            maxRung = max(maxRung, (int) rungs[sinkIndex]);
        }
        float kin1 = 0.5 * mass * sqr(vx, vy, vz);
        // Only legacy hydro retains ownership of the gas gravitational kick.
        if (params.htime || data.midpointHydro || !data.sph || data.type[sinkIndex] != kSphType) {
            vels[sinkIndex][kXDimension] = vx;
            vels[sinkIndex][kYDimension] = vy;
            vels[sinkIndex][kZDimension] = vz;
        }
        phiTot += 0.5f * mass * phi;
        dkinTot += kin1 - kin0;
        // Displacement, L2P coordinate scaling and force accumulation/scaling.
        // Energy and momentum diagnostics, including their reductions, are excluded.
        flops += cosmictiger::gravity::cartesian::tracefree::detail::L2P<fmmLocalDegree(order),
                     true>::kFlops_ +
            18;
    }
    sharedReduceAdd(phiTot);
    sharedReduceAdd(xmomTot);
    sharedReduceAdd(ymomTot);
    sharedReduceAdd(zmomTot);
    sharedReduceAdd(nmomTot);
    sharedReduceAdd(kinTot);
    sharedReduceAdd(dkinTot);
    sharedReduceAdd(darkMatterKineticCorrection);
    sharedReduceAdd(darkMatterXMomentumCorrection);
    sharedReduceAdd(darkMatterYMomentumCorrection);
    sharedReduceAdd(darkMatterZMomentumCorrection);
    sharedReduceAdd(darkMatterMomentumNormCorrection);

    sharedReduceMin(minimumTimestep);
    sharedReduceMax(maxRung);
    if (tid == 0) {
        result.minimumTimestep = fmin(result.minimumTimestep, double(minimumTimestep));
        doubleFlops += 1;    // Promote the selected timestep into the shared result.
        result.maxRung = max(result.maxRung, maxRung);
        result.pot += phiTot;
        result.xmom += xmomTot;
        result.ymom += ymomTot;
        result.zmom += zmomTot;
        result.nmom += nmomTot;
        result.kin += kinTot;
        result.dkin += dkinTot;
        result.darkMatterKineticCorrection += darkMatterKineticCorrection;
        result.darkMatterXMomentumCorrection += darkMatterXMomentumCorrection;
        result.darkMatterYMomentumCorrection += darkMatterYMomentumCorrection;
        result.darkMatterZMomentumCorrection += darkMatterZMomentumCorrection;
        result.darkMatterMomentumNormCorrection += darkMatterMomentumNormCorrection;
    }
    return {flops, doubleFlops, 0};
}

// Device twin of detail::kickOpenOrDefer in kick_walk.hpp: a rejected source
// that is not accepted by any interaction is opened and retested at this
// level (retry), handed down unopened to the sink's children (defer), or
// opened with its children handed down (both: lockstep, or equal radii).
__device__ inline void cudaOpenOrDefer(bool openLargerCell, TreeNode const& sink,
    TreeNode const& source, bool& retry, bool& defer, bool& both) {
    if (source.leaf) {
        defer = true;
    } else if (!openLargerCell || sink.leaf || source.radius > sink.radius) {
        retry = true;
    } else if (source.radius < sink.radius) {
        defer = true;
    } else {
        both = true;
    }
}

// The per-order composition unit supplies this node policy before explicit
// instantiation. It is an inline device call, with no function-pointer dispatch.
template <int order>
__device__ __forceinline__ FlopCount cudaApplyPeriodicPolicy(CudaKickData const&,
    KickParameters const&, cosmictiger::gravity::cartesian::tracefree::Expansion<float, order>&,
    TreeNode const&, StackVector<int>&, DeviceVector<int>&, DeviceVector<int>&, DeviceVector<int>&,
    DeviceVector<int>&);

// One 32-thread warp per block. The dynamic shared memory follows the host
// computed CudaKickArenaLayout; see kick.hpp for the region lifetimes.
template <int order>
__global__ void cudaKickKernel(KickReturn* rc, KickParameters globalParams, CudaKickData data,
    CudaKickParameters<order>* params, int itemCount, int* nextItem, CudaKickArenaLayout layout,
    char* listSlab
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    ,
    profile::DeviceBlockRecord* profileRecords
#endif
) {
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    profile::DeviceProfiler work(profileRecords != nullptr ? profileRecords + blockIdx.x : nullptr,
        profile::DevicePhase::KickPrepare);
#else
    profile::DeviceProfiler work(nullptr, profile::DevicePhase::KickPrepare);
#endif
    int const& tid = threadIdx.x;
    extern __shared__ __align__(16) char shmemPtr[];
    using Control = CudaKickControl<order>;
    using Expansion = cosmictiger::gravity::cartesian::tracefree::Expansion<float, order>;
    constexpr int kLocalExpansionBytes = FmmOrder<order>::kLocalExpansionBytes_;
    Control& control = *reinterpret_cast<Control*>(shmemPtr + layout.controlOffset);
    if (layout.listSlabBytes > 0) {
        // The lists live in this block's region of the per-launch slab; the
        // pool is touched only if a list outgrows its reservation.
        CudaKickListStorage<order> storage;
        kickCarveListStorage<order>(
            layout, listSlab + size_t(blockIdx.x) * size_t(layout.listSlabBytes), storage);
        new (&control) Control(storage);
    } else {
        new (&control) Control;
    }
    char* arena = shmemPtr + layout.arenaOffset;
    bool const sharedExpansion = layout.expansionOffset >= 0;
    if (!sharedExpansion) {
        control.current.resize(1);
    }
    // Every lane ran the constructor; only lane zero fills the resolved layout.
    __syncwarp();
    if (tid == 0) {
        control.layout = layout;
        control.sourceX = reinterpret_cast<Position*>(arena + layout.sourceOffset);
        control.sourceY = control.sourceX + layout.particleCapacity;
        control.sourceZ = control.sourceY + layout.particleCapacity;
        control.sourceMass = layout.massPlane ?
            reinterpret_cast<float*>(control.sourceZ + layout.particleCapacity) :
            nullptr;
        control.multipoles = arena + layout.sourceOffset;
        control.scratchExpansion = reinterpret_cast<float*>(arena + layout.scratchOffset);
        if (sharedExpansion) {
            char* expansionBytes = shmemPtr + layout.expansionOffset;
            control.expansion = reinterpret_cast<Expansion*>(expansionBytes);
            control.expansionPos = reinterpret_cast<Vector3<Position>*>(expansionBytes +
                (kLocalExpansionBytes + alignof(Position) - 1) / alignof(Position) *
                    alignof(Position));
        } else {
            control.expansion = &control.current[0].expansion;
            control.expansionPos = &control.current[0].pos;
        }
        if (layout.forceOffset >= 0) {
            control.forces.gx = reinterpret_cast<float*>(arena + layout.forceOffset);
            control.forces.gy = control.forces.gx + layout.sinkCapacity;
            control.forces.gz = control.forces.gy + layout.sinkCapacity;
            control.forces.phi = control.forces.gz + layout.sinkCapacity;
        }
    }
    __syncwarp();
    auto& sparams = control.params;
    auto& ancestors = control.ancestors;
    auto& dchecks = control.dchecks;
    auto& echecks = control.echecks;
    auto& nextlist = control.nextlist;
    auto& bothlist = control.bothlist;
    auto& cclist = control.cclist;
    auto& cplist = control.cplist;
    auto& pclist = control.pclist;
    auto& leaflist = control.leaflist;
    auto& barrier = control.barrier;
    Expansion& expansion = *control.expansion;
    Vector3<Position>& expansionPos = *control.expansionPos;
    float* scratch = control.scratchExpansion;
    auto& treeNodes = data.treeNodes;
    bool const equalMass = !COSMICTIGER_WITH_SPH || !data.sph;
    bool const sharedForces = layout.forceOffset >= 0;
    float const& supportRadius = globalParams.gravitySupportRadius;
    float const inverseSupportRadius = 1.f / supportRadius;
    int index;
    auto group = cooperative_groups::this_thread_block();
    if (group.thread_rank() == 0) {
        init(&barrier, group.size());
    }
    group.sync();
    work.next(profile::DevicePhase::KickSchedule);
    if (tid == 0) {
        index = atomicAdd(nextItem, 1);
    }
    index = __shfl_sync(0xFFFFFFFF, index, 0);
    __syncthreads();
    KickReturn kr;
    // Keep the count in each lane across every item assigned to this block.
    // Submit once after all walks, avoiding reductions and atomics per helper.
    FlopCounter<FlopCount> flops = 1;
    FlopCounter<FlopCounts> interactionFlops;
    FlopCounts geometryFlops{};
    while (index < itemCount) {
        work.next(profile::DevicePhase::KickPrepare);
        FmmGeometry const geometry = params[index].geometry;
        ancestors.resize(0);
        dchecks.resize(0);
        echecks.resize(0);
        sparams.resize(0);
        detail::cudaCopyExpansion<order>(&expansion[0], &params[index].lValue[0]);
        if (tid < kDimensions) {
            expansionPos[tid] = params[index].lpos[tid];
        }
        dchecks.resize(params[index].dcount);
        echecks.resize(params[index].ecount);
        for (int i = tid; i < params[index].dcount; i += kWarpSize) {
            dchecks[i] = params[index].dchecks[i];
        }
        for (int i = tid; i < params[index].ecount; i += kWarpSize) {
            echecks[i] = params[index].echecks[i];
        }
        {
            SearchParameters sparam;
            sparam.phase = 0;
            sparam.self = params[index].self;
            sparams.pushBack(sparam);
        }
        __syncwarp();
        int depth = 0;
        while (depth >= 0) {
            auto const& self = treeNodes[sparams.back().self];
            switch (sparams.back().phase) {
            case 0: {
                // Match the CPU setup interval, including the L2L translation.
                work.next(profile::DevicePhase::KickPrepare);
                Vector3<float> dx;
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = unwrappedDistance(self.pos[dim], expansionPos[dim]);
                }
                // Displacement conversion and the three coordinate scalings
                // in l2LCudaInto are both performed by every lane.
                flops += 6;
                {
                    // Cooperative translation into the scratch expansion, then
                    // publish it as the active expansion at this node's center.
                    cosmictiger::gravity::cartesian::tracefree::L2LWarpInto(
                        scratch, expansion, fmmScaleDisplacement(dx), globalParams.doPhi);
                    detail::cudaCopyExpansion<order>(&expansion[0], scratch);
                    if (tid < kDimensions) {
                        expansionPos[tid] = self.pos[tid];
                    }
                    __syncwarp();
                    if (tid == 0) {
                        flops += globalParams.doPhi ?
                            cosmictiger::gravity::cartesian::tracefree::detail::detail::L2LWarp<
                                fmmLocalDegree(order), true>::kFlops_ :
                            cosmictiger::gravity::cartesian::tracefree::detail::detail::L2LWarp<
                                fmmLocalDegree(order), true>::kFlopsNoPotential_;
                    }
                }
                ParticleInteger const numberOfSinks =
                    self.particleRange.second - self.particleRange.first;
#if COSMICTIGER_ENABLE_FMM
                if (numberOfSinks && geometry.meshSize == 0) {
                    work.next(profile::DevicePhase::KickPeriodic);
                    flops += cudaApplyPeriodicPolicy<order>(data, globalParams, expansion, self,
                        echecks, nextlist, cclist, leaflist, bothlist);
                }
#endif
                if (numberOfSinks) {
                    work.next(profile::DevicePhase::KickTraversal);
                    nextlist.resize(0);
                    leaflist.resize(0);
                    cclist.resize(0);
                    cplist.resize(0);
                    pclist.resize(0);
                    bothlist.resize(0);
                    constexpr int kCc = 0;
                    constexpr int kLeaf = 1;
                    constexpr int kCp = 2;
                    constexpr int kPc = 3;
                    constexpr int kNext = 4;
                    constexpr int kBoth = 5;
                    constexpr int kNlists = 6;
                    Array<bool, kNlists> sw;
                    DeviceVector<int>* lists[kNlists] = {
                        &cclist, &leaflist, &cplist, &pclist, &nextlist, &bothlist};
                    auto& checks = dchecks;
                    float const thetainv = fmmInverseOpeningAngle(globalParams.theta);
                    flops += 1;
                    do {
                        int const maxi = roundUp(checks.size(), kWarpSize);
                        for (int i = tid; i < maxi; i += kWarpSize) {
                            for (int n = 0; n < kNlists; n++) {
                                sw[n] = false;
                            }
                            if (i < checks.size()) {
                                TreeNode const& other = treeNodes[checks[i]];
                                if (other.numberOfParticles()) {
                                    for (int dim = 0; dim < kDimensions; dim++) {
                                        dx[dim] = fmmPositionDifference(self.pos[dim],
                                            other.pos[dim], dim, geometry, &geometryFlops);    // 3
                                    }
                                    float const r2 = fmaf(dx[kXDimension], dx[kXDimension],
                                        fmaf(dx[kYDimension], dx[kYDimension],
                                            dx[kZDimension] * dx[kZDimension]));
                                    float const pairSupportRadius = fmmPairSupportRadius(
                                        supportRadius, self.maxGravitySupportRadius,
                                        other.maxGravitySupportRadius);
                                    FlopCount acceptanceFlops;
                                    auto acceptance =
                                        computeFmmAcceptance(r2, self.radius, other.radius, 0.0f,
                                            0.0f, pairSupportRadius, thetainv, &acceptanceFlops);
                                    flops += 5 + acceptanceFlops;    // Float norm; geometry counted
                                                                     // separately.
                                    if (!acceptance.cellCell && (other.leaf || self.leaf)) {
                                        acceptance =
                                            computeFmmAcceptance(r2, self.radius, other.radius,
                                                fmmSinkBoxDistanceSquared(
                                                    self.box, other.pos, geometry, &geometryFlops),
                                                fmmSourceBoxDistanceSquared(other.box, self.pos,
                                                    other.pos, geometry, &geometryFlops),
                                                pairSupportRadius, thetainv, &acceptanceFlops);
                                        flops += acceptanceFlops;    // Box arithmetic is counted by
                                                                     // precision at its source.
                                    }
                                    if (acceptance.cellCell && self.leaf && other.leaf &&
                                        geometry.meshSize > 0) {
                                        flops += 2;
                                    } else if (!acceptance.cellCell && self.leaf && other.leaf) {
                                        flops += 2 + int(acceptance.particleCell) +
                                            int(acceptance.cellParticle);
                                    }
                                    auto const interaction = selectFmmInteraction(
                                        bool(acceptance.cellCell), bool(acceptance.particleCell),
                                        bool(acceptance.cellParticle), self.leaf, other.leaf,
                                        self.radius, other.radius, int(self.numberOfParticles()),
                                        int(other.numberOfParticles()),
                                        FmmHybridCost<order>::kHybridPerParticle_,
                                        FmmHybridCost<order>::kParticlePair_,
                                        geometry.meshSize > 0 ? FmmHybridCost<order>::kCellCell_ :
                                                                0.0f);
                                    sw[kCc] = interaction == FmmInteraction::CellCell;
                                    sw[kPc] = interaction == FmmInteraction::ParticleCell;
                                    sw[kCp] = interaction == FmmInteraction::CellParticle;
                                    if (!sw[kCc] && !sw[kCp] && !sw[kPc]) {
                                        cudaOpenOrDefer(globalParams.openLargerCell, self, other,
                                            sw[kNext], sw[kLeaf], sw[kBoth]);
                                    }
                                }
                            }
                            {
                                Array<int, kNlists> l, total;
                                for (int n = 0; n < kNlists; n++) {
                                    l[n] = sw[n];
                                }
                                computeIndicesArray(l, total);
                                for (int n = 0; n < kNlists; n++) {
                                    int const start = lists[n]->size();
                                    __syncwarp();
                                    lists[n]->resize(start + total[n]);
                                    if (sw[n]) {
                                        (*lists[n])[l[n] + start] = checks[i];
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        checks.resize(kChildCount * nextlist.size());
                        for (int i = tid; i < nextlist.size(); i += kWarpSize) {
                            auto const& node = treeNodes[nextlist[i]];
                            auto const& children = node.children;
                            checks[kChildCount * i + kLeftChild] = children[kLeftChild].index;
                            checks[kChildCount * i + kRightChild] = children[kRightChild].index;
                        }
                        nextlist.resize(0);
                        __syncwarp();

                    } while (checks.size() && (self.leaf || globalParams.openLargerCell));
                    if (self.leaf) {
                        __syncwarp();
                        work.next(profile::DevicePhase::KickM2L);
                        interactionFlops += cudaGravityCcDirect<order>(
                            data, expansion, self, cclist, globalParams.doPhi, geometry);
                        work.next(profile::DevicePhase::KickP2L);
                        if (equalMass) {
                            interactionFlops += cudaGravityCpDirect<order, true>(data, control,
                                expansion, self, cplist, globalParams.doPhi, geometry);
                        } else {
                            interactionFlops += cudaGravityCpDirect<order, false>(data, control,
                                expansion, self, cplist, globalParams.doPhi, geometry);
                        }
                        // The cell interactions are complete, so the arena scratch is
                        // free: initialize the sink forces and stream the sink tiles.
                        work.next(profile::DevicePhase::KickPrepare);
                        if (!sharedForces) {
                            control.forceBacking.resize(4 * numberOfSinks);
                            if (tid == 0) {
                                control.forces.gx = control.forceBacking.data();
                                control.forces.gy = control.forces.gx + numberOfSinks;
                                control.forces.gz = control.forces.gy + numberOfSinks;
                                control.forces.phi = control.forces.gz + numberOfSinks;
                            }
                            __syncwarp();
                        }
                        int const tile = sharedForces ? layout.sinkCapacity : numberOfSinks;
                        for (int sinkBegin = 0; sinkBegin < numberOfSinks; sinkBegin += tile) {
                            work.next(profile::DevicePhase::KickPrepare);
                            int const sinkCount = min(tile, int(numberOfSinks - sinkBegin));
                            auto const& force = control.forces;
                            for (int l = tid; l < sinkCount; l += kWarpSize) {
                                ParticleInteger const sinkIndex =
                                    self.particleRange.first + sinkBegin + l;
                                float const mass = data.sourceMasses != nullptr ?
                                    data.sourceMasses[sinkIndex] :
                                    data.sph ?
                                    (data.type[sinkIndex] == kDarkMatterType ? data.darkMatterMass :
                                                                               data.sphMass) :
                                    1.0f;
                                force.gx[l] = force.gy[l] = force.gz[l] = 0.f;
                                force.phi[l] = mass * selfPhi() * inverseSupportRadius;
                                flops += 2;
                            }
                            __syncwarp();
                            work.next(profile::DevicePhase::KickM2P);
                            interactionFlops += cudaGravityPcDirect<order>(data, control, self,
                                sinkBegin, sinkCount, pclist, globalParams.doPhi, geometry);
                            work.next(profile::DevicePhase::KickP2P);
                            if (equalMass) {
                                interactionFlops += cudaGravityPpDirect<order, true>(data, control,
                                    self, sinkBegin, sinkCount, leaflist, supportRadius,
                                    globalParams.doPhi, geometry);
                            } else {
                                interactionFlops += cudaGravityPpDirect<order, false>(data, control,
                                    self, sinkBegin, sinkCount, leaflist, supportRadius,
                                    globalParams.doPhi, geometry);
                            }
                            __syncwarp();
                            work.next(profile::DevicePhase::KickLeafKick);
                            interactionFlops += doKick<order>(kr, globalParams, data, expansion,
                                self, force, sinkBegin, sinkCount, params[index].outputOffset);
                            // The next tile reinitializes the force planes.
                            __syncwarp();
                        }
                    } else {
                        int const start = checks.size();
                        int const bothStart = start + leaflist.size();
                        checks.resize(bothStart + kChildCount * bothlist.size());
                        cuda::memcpy_async(group, checks.data() + start, leaflist.data(),
                            leaflist.size() * sizeof(int), barrier);
                        for (int i = tid; i < bothlist.size(); i += kWarpSize) {
                            auto const& children = treeNodes[bothlist[i]].children;
                            checks[bothStart + kChildCount * i + kLeftChild] =
                                children[kLeftChild].index;
                            checks[bothStart + kChildCount * i + kRightChild] =
                                children[kRightChild].index;
                        }
                        __syncwarp();
                        work.next(profile::DevicePhase::KickM2L);
                        interactionFlops += cudaGravityCcDirect<order>(
                            data, expansion, self, cclist, globalParams.doPhi, geometry);
                        // Keep the checklist copy overlapping M2L; charge only
                        // its remaining wait here, before P2L reuses the barrier.
                        work.next(profile::DevicePhase::KickCopyWait);
                        barrier.arrive_and_wait();
                        // CP reuses the barrier, so finish the checklist copy before calling it.
                        work.next(profile::DevicePhase::KickP2L);
                        if (equalMass) {
                            interactionFlops += cudaGravityCpDirect<order, true>(data, control,
                                expansion, self, cplist, globalParams.doPhi, geometry);
                        } else {
                            interactionFlops += cudaGravityCpDirect<order, false>(data, control,
                                expansion, self, cplist, globalParams.doPhi, geometry);
                        }
                    }
                }
                work.next(profile::DevicePhase::KickSchedule);
                if (self.leaf) {
                    __syncwarp();
                    sparams.popBack();
                    depth--;
                } else {
                    // Keep this node's expansion for the right child while the
                    // left child translates the active copy.
                    TreeId const child = self.children[kLeftChild];
                    int const top = ancestors.size();
                    ancestors.resize(top + 1);
                    detail::cudaCopyExpansion<order>(&ancestors[top].expansion[0], &expansion[0]);
                    if (tid < kDimensions) {
                        ancestors[top].pos[tid] = expansionPos[tid];
                    }
                    __syncwarp();
                    dchecks.pushTop();
                    echecks.pushTop();
                    if (tid == 0) {
                        sparams.back().phase += 1;
                    }
                    __syncwarp();
                    {
                        SearchParameters sparam;
                        sparam.phase = 0;
                        sparam.self = child.index;
                        sparams.pushBack(sparam);
                    }
                    depth++;
                }

            } break;
            case 1: {
                // Restore this node's expansion for the right child.
                int const top = ancestors.size() - 1;
                detail::cudaCopyExpansion<order>(&expansion[0], &ancestors[top].expansion[0]);
                if (tid < kDimensions) {
                    expansionPos[tid] = ancestors[top].pos[tid];
                }
                __syncwarp();
                ancestors.popBack();
                dchecks.popTop();
                echecks.popTop();
                if (tid == 0) {
                    sparams.back().phase += 1;
                }
                __syncwarp();
                {
                    SearchParameters sparam;
                    sparam.phase = 0;
                    sparam.self = self.children[kRightChild].index;
                    sparams.pushBack(sparam);
                }
                depth++;
            } break;
            case 2: {
                sparams.popBack();
                depth--;
            } break;
            }
        }

        work.next(profile::DevicePhase::KickSchedule);
        if (tid == 0) {
            index = atomicAdd(nextItem, 1);
        }
        index = __shfl_sync(0xFFFFFFFF, index, 0);
        ASSERT(ancestors.size() == 0);
    }
    work.next(profile::DevicePhase::KickCleanup);
    if (tid == 0) {
        atomicAdd(&rc->dkin, kr.dkin);
        atomicAdd(&rc->kin, kr.kin);
        atomicAdd(&rc->pot, kr.pot);
        atomicAdd(&rc->xmom, kr.xmom);
        atomicAdd(&rc->ymom, kr.ymom);
        atomicAdd(&rc->zmom, kr.zmom);
        atomicAdd(&rc->nmom, kr.nmom);
        atomicAdd(&rc->darkMatterKineticCorrection, kr.darkMatterKineticCorrection);
        atomicAdd(&rc->darkMatterXMomentumCorrection, kr.darkMatterXMomentumCorrection);
        atomicAdd(&rc->darkMatterYMomentumCorrection, kr.darkMatterYMomentumCorrection);
        atomicAdd(&rc->darkMatterZMomentumCorrection, kr.darkMatterZMomentumCorrection);
        atomicAdd(&rc->darkMatterMomentumNormCorrection, kr.darkMatterMomentumNormCorrection);

        // Positive IEEE doubles have the same ordering as their unsigned bit patterns.
        atomicMin(reinterpret_cast<unsigned long long*>(&rc->minimumTimestep),
            static_cast<unsigned long long>(__double_as_longlong(kr.minimumTimestep)));
        atomicMax(&rc->maxRung, kr.maxRung);
    }
    addGpuFlops(flops, FlopPrecision::Single);
    addGpuFlops(static_cast<FlopCounts>(interactionFlops));
    addGpuFlops(geometryFlops);
    control.~Control();
    work.finish();
}
